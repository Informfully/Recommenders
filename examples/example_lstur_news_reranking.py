###### Example Script: LSTUR Model ######
# This script demonstrates the standard experiment of the LSTUR model.
#
# 1. Data Preparation:
#    - Loads user-item interaction data, user history, and impression logs.
#    - Load the pre-split data into training and testing sets using `BaseMethod`.
#    - Processes item attributes (e.g., sentiment, category, complexity) 
#      for diversity metrics evaluation.
#
# 2. LSTUR Model Configuration:
#    - The LSTUR model utilizes user history and impression logs for training.
#    - Requires pre-trained word embeddings and mappings for processing news titles.
#
#
# 3. Evaluation:
#    - Defines metrics for accuracy (e.g., NDCG, Recall) and diversity (e.g., 
#      ILD, Calibration, Activation).
#    - Evaluates recommendations outputs.
#
# 4. Experiment Execution:
#    - Combines the LSTUR model, metrics, and rerankers in a Cornac `Experiment`.
#    - Results are saved for further analysis in the specified output directory.
# ============================================================================

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import logging
tf.get_logger().setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import os
# Set environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import json
import numpy as np
import pandas as pd
import pickle
import random
import sys

from cornac.data import Reader
from cornac.eval_methods import BaseMethod
from cornac.metrics import MAE, RMSE, Recall, FMeasure
from cornac.experiment.experiment import Experiment
from cornac.metrics import NDCG, AUC, MRR
from cornac.metrics import GiniCoeff, ILD, EILD, Precision, Activation, Calibration, Fragmentation, Representation, AlternativeVoices, Alpha_NDCG, Binomial
from cornac.datasets import mind as mind

from cornac.models import LSTUR
from cornac.rerankers import GreedyKLReranker, PM2Reranker, MMR_ReRanker, DynamicAttrReRanker

#  Load data and set up environment
def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)


    news_files_dir = os.path.join(current_dir, 'example_news_files')

    config_files_dir = os.path.join(current_dir, 'example_config_files')
    sys.path.insert(0, news_files_dir)
    sys.path.insert(0, config_files_dir)

    input_path = news_files_dir

    train_uir_path = os.path.join(input_path, 'example_impression_train_uir.csv')
    feedback_train = mind.load_feedback(fpath = train_uir_path)
    
    test_uir_path = os.path.join(input_path, 'example_impression_test_uir.csv')
    feedback_test = mind.load_feedback(fpath = test_uir_path)

    article_pool_path = os.path.join(input_path, 'example_article_pool.csv')
    impression_items_df = pd.read_csv(article_pool_path, dtype ={'iid': str})
    impression_iid_list = impression_items_df['iid'].tolist()

    user_history_path = os.path.join(input_path, 'example_user_history.json')
    with open(user_history_path, 'r') as file:
        user_item_history = json.load(file)
    print(f"len user_item_history:{len(user_item_history)}")

    # Paths to article text, embedding resources
    title_dict_path = os.path.join(input_path, 'example_title.json')
    word_dict_path = os.path.join(input_path, 'word_index_dict.json')
    word_embedding_path =  os.path.join(input_path, 'embedding_matrix.npy')
    
    # Split data
    rs = BaseMethod.from_splits(
        train_data=feedback_train,
        test_data=feedback_test,
        exclude_unknowns=False,
        verbose=True,
        rating_threshold=0.5
    )
    


    # Set up the LSTUR model
    model = LSTUR(
        wordEmb_file = word_embedding_path,
        wordDict_file = word_dict_path,
        newsTitle_file = title_dict_path,
        userHistory = user_item_history,
         epochs = 10,
        word_emb_dim = 300, 
        head_num = 20,
        history_size = 50, 
        title_size = 30, 
        window_size = 3 , 
        filter_num = 300, 
        gru_unit = 300,
        npratio = 6,
        dropout = 0.2, 
        learning_rate = 0.001, 
        batch_size = 64,
        seed = 42
        # article_pool=  impression_iid_list
    )

    # Load article attributes
    sentiment_file_path  = os.path.join(input_path, 'example_sentiment.json')
    sentiment = mind.load_sentiment(fpath=sentiment_file_path)
    category_file_path  = os.path.join(input_path, 'example_category.json')
    category = mind.load_category(fpath= category_file_path)
    party_file_path = os.path.join(input_path, 'example_party.json')
    entities = mind.load_entities(fpath=party_file_path)
    entities_keep_no_party = mind.load_entities(fpath=party_file_path, keep_empty = True)


    # for loading category, identfy the unqiue categories, and build category one-hot vectors
    category_vector =  mind.load_category_multi(fpath= category_file_path)
   
    Item_sentiment = mind.build(data=sentiment, id_map=rs.global_iid_map)
    Item_category = mind.build(data=category, id_map=rs.global_iid_map)
    Item_category_vec =  mind.build(data=category_vector, id_map=rs.global_iid_map)

    Item_entities = mind.build(data=entities, id_map=rs.global_iid_map)
    Item_entities_keep_no_party =  mind.build(data=entities_keep_no_party, id_map=rs.global_iid_map)

    
    ### generating one-hot encoding vectors for sentiment and party
    ### Adjust based on your needs

    def sentiment_to_one_hot(score):
        if -1 <= score < -0.5:
            return [1, 0, 0, 0]
        elif -0.5 <= score < 0:
            return [0, 1, 0, 0]
        elif 0 <= score < 0.5:
            return [0, 0, 1, 0]
        elif 0.5 <= score <= 1:
            return [0, 0, 0, 1]

    # Apply the function to each sentiment value
    sentiment_one_hot_encoded_vectors = {key: sentiment_to_one_hot(value) for key, value in sentiment.items()}

    # Save the result to a new JSON file
    with open(f"{input_path}/sentiment_one_hot_vector.json", "w", encoding="utf-8") as f:
        json.dump(sentiment_one_hot_encoded_vectors, f, indent=4)

    
    # Normalize related party names to lowercase
    gov_party = ["party1","party4"]
    opp_party = ["party3", "party6"]
    ## any party not belong to gov_party, opp_party is considered as indepent/foreign party.
    # 
    # Function to generate one-hot encoding based on mentioned parties
    def party_to_one_hot(mentioned_parties):
        if not mentioned_parties:  # Handles None, empty list, or malformed data
            return [0, 0, 0, 0, 1]

        # Convert to lowercase for case-insensitive matching
        mentions_lower = [p.lower() for p in mentioned_parties]
        gov_party_lower = [p.lower() for p in gov_party]
        opp_party_lower = [p.lower() for p in opp_party]

        has_gov = any(p in mentions_lower for p in gov_party_lower)
        has_opp = any(p in mentions_lower for p in opp_party_lower)
        has_other = any(
            p not in gov_party_lower and p not in opp_party_lower
            for p in mentions_lower
        )

        if has_gov and not has_opp and not has_other:
            return [1, 0, 0, 0, 0]  # Only government party
        elif has_opp and not has_gov and not has_other:
            return [0, 1, 0, 0, 0]  # Only opposition party
        elif has_gov and has_opp and not has_other:
            return [0, 0, 1, 0, 0]  # Both gov and opp
        elif has_other:
            return [0, 0, 0, 1, 0]  # Minority or third parties
        return [0, 0, 0, 0, 1]  # No party or unrecognized


    # Convert the party mentions to one-hot encoding
    party_one_one_hot_encoded_vectors = {key: party_to_one_hot(value) for key, value in entities_keep_no_party.items()}

    # Save the result to a new JSON file
    with open(f"{input_path}/party_one_hot_vector.json", "w", encoding="utf-8") as f:
        json.dump(party_one_one_hot_encoded_vectors, f, indent=4)
    


    Item_senti_vec = mind.build(data = sentiment_one_hot_encoded_vectors, id_map = rs.global_iid_map)
    Item_party_vec = mind.build(data = party_one_one_hot_encoded_vectors, id_map = rs.global_iid_map)

    # Combine sentiment and party one-hot encoded vectors
    senti_party_one_hot_vectors = {}

    for article_id in sentiment_one_hot_encoded_vectors:
        if article_id not in party_one_one_hot_encoded_vectors:
            raise ValueError(f"Missing party vector for article: {article_id}")
        
        sentiment_vector = sentiment_one_hot_encoded_vectors[article_id]
        party_vector = party_one_one_hot_encoded_vectors[article_id]
        
        senti_party_one_hot_vectors[article_id] = sentiment_vector + party_vector

   
    senti_party_vec_path =  os.path.join(input_path, 'combined_senti_party_vectors.json')
    
    with open(senti_party_vec_path, "w", encoding="utf-8") as f:
        json.dump(senti_party_one_hot_vectors, f, indent=4)
    
    # If you already saved the senti_party_vec, you can also load them.
    # sentiment_party_combined_vectors = mind.load_encoding_vectors(fpath=senti_party_vec_path)

    Item_party_senti_vec = mind.build(data=senti_party_one_hot_vectors, id_map=rs.global_iid_map)

    Target_Mind_distribution = {
            "sentiment": {"type": "continuous", "distr": [
                {"min": -1, "max": -0.5, "prob": 0.25},
                {"min": -0.5, "max": 0, "prob": 0.25},
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



    article_feature_dataframe_keep_no_party = (
            pd.Series(Item_category).to_frame('category')
            .join(pd.Series(Item_entities_keep_no_party).to_frame('entities'), how='outer')
            .join(pd.Series(Item_sentiment).to_frame('sentiment'), how='outer')
        )


    # Define metrics
    targetSize = 10

    sentiment_party_greedy_reranker = GreedyKLReranker(
        item_dataframe = article_feature_dataframe_keep_no_party,
        diversity_dimension = ["sentiment", "entities"],
        top_k=targetSize,
        target_distributions = Target_Mind_distribution,
        diversity_dimension_weight=[0.5, 0.5],
        user_item_history = user_item_history, 
        rerankers_item_pool = impression_iid_list 

    )

    sentiment_party_pm2_reranker = PM2Reranker(
        item_dataframe= article_feature_dataframe_keep_no_party,
        diversity_dimension=["sentiment", "entities"],
        top_k=targetSize,
        target_distributions = Target_Mind_distribution,
        diversity_dimension_weight = [0.5, 0.5],
    user_item_history = user_item_history, 
        rerankers_item_pool = impression_iid_list 

    )
    sentiment_party_mmr_reranker = MMR_ReRanker(top_k=targetSize,
                                                item_feature_vectors = Item_party_senti_vec,
                                                user_item_history = user_item_history, 
                                                rerankers_item_pool = impression_iid_list, 
                                                    lamda = 0.1)

    bin_edges = {'sentiment': [-1, -0.5, 0, 0.5, 1]}


    party_category_json_path = os.path.join(config_files_dir, 'party_category.json')
    user_simulator_path = os.path.join(config_files_dir, "user_simulator_config.ini")
    dynamic_reranker_rank_bias = DynamicAttrReRanker(name="DYN_reranker_probByposition",
                                                        item_dataframe=article_feature_dataframe_keep_no_party, diversity_dimension=["sentiment", "entities"],  top_k=targetSize,
                                                        feedback_window_size=3, bin_edges=bin_edges, user_choice_model="logarithmic_rank_bias", user_simulator_config_path=user_simulator_path,
                                                            party_category_json_path = party_category_json_path,
                                                    user_item_history = user_item_history, 
                                                    rerankers_item_pool = impression_iid_list
                                            
                                                        )

    dynamic_reranker_preference_bias = DynamicAttrReRanker(name="DYN_reranker_probByPreference",
                                                            item_dataframe=article_feature_dataframe_keep_no_party, diversity_dimension=["sentiment", "entities"],  top_k=targetSize,
                                                            feedback_window_size=3, bin_edges=bin_edges, user_choice_model="preference_based_bias", user_simulator_config_path=user_simulator_path,
                                                            party_category_json_path = party_category_json_path,
                                                            user_item_history = user_item_history, 
                                                            rerankers_item_pool = impression_iid_list)


    ild_cat = ILD(name="cat_ILD", item_feature=Item_category_vec, k=targetSize)
    ild_senti = ILD(name="senti_ILD", item_feature=Item_senti_vec, k=targetSize)
    ild_party = ILD(name="party_ILD", item_feature=Item_party_vec, k=targetSize)
    gini_cat = GiniCoeff(name="cat_gini", item_genre=Item_category_vec, k=targetSize)
    gini_senti = GiniCoeff(name="senti_gini", item_genre=Item_senti_vec, k=targetSize)
    gini_party = GiniCoeff(name="party_gini", item_genre=Item_party_vec, k=targetSize)
    metrics = [Recall(k=targetSize),
                ild_cat,
                ild_senti, 
                ild_party,
                gini_cat,
                gini_senti,
                gini_party]


    output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.', 'output_lstur_reranking_result')
    # Set up the experiment
    Experiment(
        eval_method=rs,
        models=[model],
        metrics=metrics,
        rerankers= {'static': [sentiment_party_greedy_reranker,
                            sentiment_party_pm2_reranker,
                            sentiment_party_mmr_reranker],
    'dynamic': [dynamic_reranker_rank_bias, dynamic_reranker_preference_bias]},
        save_dir=output_file_path
    ).run()


if __name__ == "__main__":
    main()
