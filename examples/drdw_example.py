# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Example for Diversity-Driven Random Walk Model"""

import cornac
from cornac.eval_methods import BaseMethod
from cornac.metrics import Precision,Recall,NDCG
from cornac.models import D_RDW
from cornac.datasets import mind
import pandas as pd
import os
import sys
# Load user-item feedback
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


news_files_dir = os.path.join(current_dir, 'example_news_files')

config_files_dir = os.path.join(current_dir, 'example_config_files')
sys.path.insert(0, news_files_dir)
sys.path.insert(0, config_files_dir)

input_path = news_files_dir

train_uir_path = os.path.join(input_path, 'example_training_graph_uir_top3.csv')
feedback_train = mind.load_feedback(fpath = train_uir_path)

test_uir_path = os.path.join(input_path, 'example_impression_test_uir.csv')
feedback_test = mind.load_feedback(fpath = test_uir_path)


# (1) load RAW user-item interaction (2) Randomly split (3) Create mapping from RAW user-item to Cornac interal user ID and item ID
rs = BaseMethod.from_splits(
    train_data=feedback_train,
    test_data=feedback_test,
    exclude_unknowns=False,
    verbose=True,
    rating_threshold=0.5
)

# Load RAW item features (correspond to RAW item ID)
sentiment_file_path  = os.path.join(input_path, 'example_sentiment.json')
sentiment = mind.load_sentiment(fpath=sentiment_file_path)
category_file_path  = os.path.join(input_path, 'example_category.json')
category = mind.load_category(fpath= category_file_path)
party_file_path = os.path.join(input_path, 'example_party.json')
entities = mind.load_entities(fpath=party_file_path)

# When using party data, set `keep_empty` based on the intended purpose:
#
# - For target distribution (used during D-RDW/re-ranking), set `keep_empty = True`.
#   This includes articles in the dictionary even if they don't mention any parties.
#
# - For computing representation (diversity metrics), set `keep_empty = False`.
#   This excludes articles that don't mention any parties.
#
# The `load_entities` function converts data from this format:
#   {"article_id": {"party1": 2, "party2": 3, ...}}
#
# To a format suitable for D-RDW, re-ranking and Representation metric computation:
#   {"article_id": ["party1", "party1", "party2", "party2", "party2", ...]}
# where each party appears in the list as many times as it is mentioned in the article.

entities_keep_no_party = mind.load_entities(fpath=party_file_path, keep_empty = True)


# mind.load_category_multi: for loading category, identfy the unqiue categories, and build category one-hot vectors.
category_vector =  mind.load_category_multi(fpath= category_file_path)

Item_sentiment = mind.build(data=sentiment, id_map=rs.global_iid_map)
Item_category = mind.build(data=category, id_map=rs.global_iid_map)
Item_category_vec =  mind.build(data=category_vector, id_map=rs.global_iid_map)

Item_entities = mind.build(data=entities, id_map=rs.global_iid_map)
Item_entities_keep_no_party =  mind.build(data=entities_keep_no_party, id_map=rs.global_iid_map)

print(f"Item_entities_keep_no_party:{Item_entities_keep_no_party}")
article_feature_dataframe = (
        pd.Series(Item_sentiment).to_frame('sentiment')
        .join(pd.Series(Item_entities_keep_no_party).to_frame('party'), how='outer')
        .join(pd.Series(Item_category).to_frame('category'), how='outer')
    )

Target_distribution = {
            "sentiment": {"type": "continuous", "distr": [
                 {"min": -1, "max": -0.5, "prob": 0},
                {"min": -0.5, "max": 0, "prob": 0},
                {"min": 0, "max": 0.5, "prob": 0.5},

                {"min": 0.5, "max": 1.01, "prob": 0.5}
            ]},


            "party": {"type": "parties", "distr": [
                {"description": "only mention", "contain": ["party6","party7"], "prob": 0.20},
                {"description": "only mention", "contain":["party9", "party1"], "prob": 0.20},
                {"description": "composition", "contain": [["party6","party7"],["party9","party1"]], "prob": 0.20},
                {"description": "minority but can also mention", "contain": 
                   ["party6","party7","party9","party1"], "prob": 0.20},
                {"description": "no parties", "contain": [], "prob": 0.20}
            ]}
        }

# Instantiate the D-RDW model
drdw = D_RDW(
        item_dataframe=article_feature_dataframe,
        diversity_dimension=["sentiment", "party"],
        target_distributions=Target_distribution,
        targetSize=10,
        maxHops=3,
        filteringCriteria=None,
        rankingType="graph_coloring",
        rankingObjectives="category",
        sampleObjective="rdw_score",
        verbose=True,
    )

# Instantiate evaluation measures
metrics = [Precision(k=200), Recall(k=200), NDCG(k=200)]

output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.', 'output_drdw_result')
# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method= rs,
    models=[drdw],
    metrics=metrics,
    user_based=True,
    save_dir=output_file_path
).run()


