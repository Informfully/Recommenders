###### Example Script: Random Baseline ######
# This script demonstrates the standard experiment of the Random model.
#
# 1. Data Preparation:
#    - Loads user-item interaction data, user history, and impression logs.
#    - Load the pre-split data into training and testing sets using `BaseMethod`.
#    - Processes item attributes (e.g., sentiment, category, complexity) 
#      for diversity metrics evaluation.
#
# 2. Random Model Configuration:
#    - The Random model utilizes user history and impression logs for training.
#    - Requires pre-trained word embeddings and mappings for processing news titles.
#
#
# 3. Evaluation:
#    - Defines metrics for accuracy (e.g., NDCG, Recall) and diversity (e.g., 
#      ILD, Calibration, Activation).
#    - Evaluates recommendations outputs.
#
# 4. Experiment Execution:
#    - Combines the Random model, metrics, and rerankers in a Cornac `Experiment`.
#    - Results are saved for further analysis in the specified output directory.
# ============================================================================


import logging, os
logging.disable(logging.WARNING)

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
from cornac.metrics import GiniCoeff, ILD
from cornac.datasets import mind as mind

from cornac.models import RandomModel

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

   
    
    # Split data
    rs = BaseMethod.from_splits(
        train_data=feedback_train,
        test_data=feedback_test,
        exclude_unknowns=False,
        verbose=True,
        rating_threshold=0.5
    )
    


    # Set up the random model
    model_without_article_pool =  RandomModel(name = "RandomModel") #example without article pool

    model_with_article_pool =  RandomModel(name = "RandomModel_articlePool",article_pool = impression_iid_list) #example with article pool

    # Load article attributes
    sentiment_file_path  = os.path.join(input_path, 'example_sentiment.json')
    sentiment = mind.load_sentiment(fpath=sentiment_file_path)
    category_file_path  = os.path.join(input_path, 'example_category.json')
    party_file_path = os.path.join(input_path, 'example_party.json')
    entities_keep_no_party = mind.load_entities(fpath=party_file_path, keep_empty = True)

    # for loading category, identfy the unqiue categories, and build category one-hot vectors
    category_vector =  mind.load_category_multi(fpath= category_file_path)
   

    Item_category_vec =  mind.build(data=category_vector, id_map=rs.global_iid_map)

    ### generating one-hot encoding vectors for sentiment and party
    ### Adjust based on your need
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
    one_hot_encoded = {key: sentiment_to_one_hot(value) for key, value in sentiment.items()}

    # Save the result to a new JSON file
    with open(f"{input_path}/combined_sentiment_one_hot.json", "w", encoding="utf-8") as f:
        json.dump(one_hot_encoded, f, indent=4)

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


    # Define metrics
    targetSize = 10

    # Diversity metrics
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


    output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.', 'output_random_baseline_result')
    # Set up the experiment
    Experiment(
        eval_method=rs,
        models=[model_without_article_pool, model_with_article_pool],
        metrics=metrics,
        save_dir=output_file_path
    ).run()


if __name__ == "__main__":
    main()
