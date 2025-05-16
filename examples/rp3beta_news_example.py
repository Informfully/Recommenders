#############################################################################################################
# VERSION:      1.0 (STANDALONE)                                                                            #
# DESCRIPTION:  Example for random walk(rp3beta) model experiment in an example dataset                               #
#############################################################################################################

import cornac
from cornac.datasets import mind as mind
from cornac.eval_methods import BaseMethod
from cornac.models import RP3_Beta
from cornac.metrics import  NDCG,  Recall
import pandas as pd
import sys
import os
import json
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

article_pool_path = os.path.join(input_path, 'example_article_pool.csv')
impression_items_df = pd.read_csv(article_pool_path, dtype ={'iid': str})
impression_iid_list = impression_items_df['iid'].tolist()

user_history_path = os.path.join(input_path, 'example_user_history.json')



# Split data
rs = BaseMethod.from_splits(
    train_data=feedback_train,
    test_data=feedback_test,
    exclude_unknowns=False,
    verbose=True,
    rating_threshold=0.5
)
    

# initialize models
model_without_article_pool =  RP3_Beta(name = "RP3beta") #example without article pool

model_with_article_pool =  RP3_Beta(name = "RP3beta_articlePool",article_pool = impression_iid_list) #example with article pool

# define metrics to evaluate the models
metrics = [ NDCG(k=20), Recall(k=20)]


output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.', 'output_rp3beta_result')

# put it together in an experiment, voilà!
cornac.Experiment(eval_method = rs, 
                  models = [ model_without_article_pool, model_with_article_pool ], 
                  metrics = metrics, 
                  save_dir = output_file_path
                   ).run()

