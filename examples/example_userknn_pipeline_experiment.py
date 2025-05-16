import os
import pandas as pd
from cornac.datasets import mind as mind
from cornac.models import UserKNN
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, Precision, Activation, Calibration
from cornac.experiment.experiment import Experiment
from cornac.experiment.pipelineExperiment import PipelineExperiment
from cornac.rerankers.dynamic_attribute_penalization import DynamicAttrReRanker
from cornac.rerankers import GreedyKLReranker, PM2Reranker
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


news_files_dir = os.path.join(current_dir, 'example_news_files')

config_files_dir = os.path.join(current_dir, 'example_config_files')
sys.path.insert(0, news_files_dir)
sys.path.insert(0, config_files_dir)

input_path = news_files_dir

# === Load Feedback Data ===

uir_path = os.path.join(input_path, 'example_impression_all_uir.csv')
feedback = mind.load_feedback(fpath = uir_path)



# === Create Evaluation Split ===
eval_method = RatioSplit(
    data=feedback,
    test_size=0.2,
    exclude_unknowns=False,
    verbose=True,
    seed=123,
    rating_threshold=0.5
)



# === Prepare Features ===
sentiment_file_path  = os.path.join(input_path, 'example_sentiment.json')
sentiment = mind.load_sentiment(fpath=sentiment_file_path)
category_file_path  = os.path.join(input_path, 'example_category.json')
category = mind.load_category(fpath= category_file_path)
party_file_path = os.path.join(input_path, 'example_party.json')
entities_keep_no_party = mind.load_entities(fpath=party_file_path, keep_empty = True)


Item_sentiment = mind.build(data=sentiment, id_map=eval_method.train_set.iid_map)
Item_category = mind.build(data=category, id_map=eval_method.train_set.iid_map)
Item_entities = mind.build(data=entities_keep_no_party, id_map=eval_method.train_set.iid_map, keep_empty = True)


article_feature_df = (
    pd.Series(Item_category).to_frame("category")
    .join(pd.Series(Item_entities).to_frame("entities"), how="outer")
    .join(pd.Series(Item_sentiment).to_frame("sentiment"), how="outer")
)

# === Setup Target Distribution ===
target_distribution =   {
            "sentiment": {"type": "continuous", "distr": [
                {"min": -1, "max": -0.5, "prob": 0.25},
                {"min": -0.5, "max": 0, "prob": 25},
                {"min": 0, "max": 0.5, "prob": 0.25},
                {"min": 0.5, "max": 1.01, "prob": 0.25}
            ]},
            # // gov_party = ["party1","party4"]
            # // opp_party = ["party3", "party6"]
            "entities": {"type": "parties", "distr": [
                {"description": "only mention", "contain": ["party1","party4"], "prob": 0.2},
                {"description": "only mention", "contain":["party3", "party6"], "prob": 0.2},
                {"description": "composition", "contain": [["party3", "party6"], ["party1","party4"]], "prob": 0.2},
                {"description": "minority but can also mention", "contain": 
                   ["party1","party4", "party3", "party6"], "prob": 0.2},
                {"description": "no parties", "contain": [], "prob": 0.2}
            ]}
        }

# === Set Metrics ===
k = 10
metrics = [
    Recall(k=k),
    Precision(k=k),
    Activation(item_sentiment=Item_sentiment, divergence_type="JS", k=k),
    Calibration(item_feature=Item_category, data_type="category", divergence_type="JS", k=k)
]

# === Train and Save Recommendations with UserKNN ===
output_dir = "./results/UserKNN_output"
os.makedirs(output_dir, exist_ok=True)

userknn_model = UserKNN(name ="UserKNN", k=3, similarity="cosine")
exp = Experiment(
    eval_method=eval_method,
    models=[userknn_model],
    metrics=metrics,
    save_dir=output_dir
)
exp.run()


# === Setup Rerankers ===
greedy_reranker = GreedyKLReranker(
    item_dataframe=article_feature_df,
    diversity_dimension=["sentiment", "entities"],
    top_k=k,
    target_distributions=target_distribution,
    diversity_dimension_weight=[0.5, 0.5],
)

pm2_reranker = PM2Reranker(
    item_dataframe=article_feature_df,
    diversity_dimension=["sentiment", "entities"],
    top_k=k,
    target_distributions=target_distribution,
    diversity_dimension_weight=[0.5, 0.5],
)

party_category_json_path = os.path.join(config_files_dir, 'party_category.json')
user_simulator_path = os.path.join(config_files_dir, "user_simulator_config.ini")

dynamic_reranker = DynamicAttrReRanker(
    name="Dynamic_Pref",
    item_dataframe=article_feature_df,
    diversity_dimension=["sentiment", "entities"],
    top_k=k,
    feedback_window_size=3,
    bin_edges={"sentiment": [-1, -0.5, 0, 0.5, 1]},
    user_choice_model="preference_based_bias",
    user_simulator_config_path=user_simulator_path,
    party_category_json_path=party_category_json_path,
)

# === Run Pipeline Experiment Using Saved UserKNN Recs ===
pipeline_config_file =  os.path.join(config_files_dir, 'pipeline_experiment.ini')

pipe_exp = PipelineExperiment(
    model=[userknn_model],
    eval_method = eval_method,
    metrics=metrics,
    rerankers={
        'static': [greedy_reranker, pm2_reranker],
        'dynamic': [dynamic_reranker]
    },
    user_based=True,
    show_validation=False,
    verbose=True,
    pipeline_config_file=pipeline_config_file
)
pipe_exp.run()
