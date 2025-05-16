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
import unittest
from cornac.eval_methods import RatioSplit
from cornac.experiment.experiment import Experiment
from cornac.metrics import NDCG, Recall
from cornac.rerankers.pm2 import PM2Reranker
from cornac.metrics import Activation
from cornac.metrics import Calibration
from cornac.metrics import Representation
from cornac.models import UserKNN, ItemKNN
from cornac.datasets import mind as mind
from cornac.rerankers import GreedyKLReranker, LeastPopReranker, DynamicAttrReRanker
import pandas as pd


class TestExperiment(unittest.TestCase):
    def setUp(self):
    
        feedback = mind.load_feedback(
            fpath="./examples/example_news_files/example_impression_all_uir.csv")
        # Mind dataset min rating = 1
        sentiment = mind.load_sentiment(
            fpath="./examples/example_news_files/example_sentiment.json")
        category = mind.load_category(
            fpath="./examples/example_news_files/example_category.json")
        # Important! For dynamic re-ranking,  `keep_empty` need to be set as True
        entities = mind.load_entities(fpath="./examples/example_news_files/example_party.json", keep_empty = True)

        self.party_category_json_path =  "./tests/configs/reranker_configs/party_category.json"
        self.rs = RatioSplit(
            data=feedback,
            test_size=0.1,
            rating_threshold=1,
            seed=123,
            exclude_unknowns=True,
            verbose=True,
        )
        self.Item_sentiment = mind.build(
            data=sentiment, id_map=self.rs.train_set.iid_map)

        self.Item_category = mind.build(
            data=category, id_map=self.rs.train_set.iid_map)

        self.Item_entities = mind.build(
            data=entities, id_map=self.rs.train_set.iid_map)

        out_pd = (pd.Series(self.Item_category).to_frame('category')
                  .join(pd.Series(self.Item_sentiment).to_frame('sentiment'), how='outer'))
        out_pd = out_pd.join(
            pd.Series(self.Item_entities).to_frame('entities'), how='outer')
        self.item_data = out_pd
        self.top_k = 10
        self.pool_size = 600
        self.user_simulator_config_path =  "./tests/configs/reranker_configs/user_simulator_config.ini"

    def test_without_reranker(self):
        act = Activation(item_sentiment=self.Item_sentiment, k=self.top_k)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=self.top_k)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN")],
                   metrics=[NDCG(k=self.top_k), Recall(k=self.top_k), act, cal]
                   ).run()

    def test_model_and_dynamic_reranking(self):
        dyn = DynamicAttrReRanker(name="dyn_reranker", item_dataframe=self.item_data, diversity_dimension=[
                                  'sentiment', 'category', 'entities'], top_k=self.top_k, pool_size=self.pool_size, feedback_window_size=2, bin_edges={'sentiment': [-0.5, 0, 0.5, 1.0]}, user_choice_model='logarithmic_rank_bias', 
                                  user_simulator_config_path = self.user_simulator_config_path,
                                  party_category_json_path =  self.party_category_json_path)
        act = Activation(item_sentiment=self.Item_sentiment, k=self.top_k)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=self.top_k)
        rep = Representation(item_entities=self.Item_entities, k=self.top_k)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN")],
                   metrics=[NDCG(k=self.top_k), Recall(
                       k=self.top_k), act, cal, rep],
                   rerankers={'dynamic': [dyn]
                              }
                   ).run()

    def test_configFile_input(self):
        config_file = "./tests/configs/reranker_configs/test_reranker.ini"
        dyn = DynamicAttrReRanker(name="DYN_reranker_probByposition", item_dataframe=self.item_data, 
                                  config_file = config_file,
                                  party_category_json_path =  self.party_category_json_path)
        dyn_preference = DynamicAttrReRanker(name="DYN_reranker_probByPreference", item_dataframe=self.item_data,
                                             config_file = config_file,
                                             party_category_json_path =  self.party_category_json_path)
  
        self.assertEqual(dyn.name, "DYN_reranker_probByposition")
        self.assertEqual(dyn_preference.name, "DYN_reranker_probByPreference")
        # Verify configuration parameters are loaded correctly
        self.assertEqual(dyn.top_k, 20) 
        self.assertEqual(dyn.feedback_window_size, 3)  
        self.assertEqual(dyn.diversity_dimension, ["sentiment", "entities"])
        self.assertEqual(dyn.user_choice_model, "logarithmic_rank_bias")

        self.assertEqual(dyn_preference.user_choice_model, "preference_based_bias")
        self.assertEqual(dyn_preference.diversity_dimension, ["sentiment", "entities"])

        # Check item dataframe is set correctly
        self.assertTrue(dyn.item_dataframe.equals(self.item_data))
        self.assertTrue(dyn_preference.item_dataframe.equals(self.item_data))
        self.assertIn("sentiment", dyn.bin_edges)
        self.assertEqual(dyn_preference.bin_edges["sentiment"], [-1, -0.5, 0, 0.5, 1])

        act = Activation(item_sentiment=self.Item_sentiment, k=self.top_k)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=self.top_k)
        rep = Representation(item_entities=self.Item_entities, k=self.top_k)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN")],
                   metrics=[NDCG(k=self.top_k), Recall(
                       k=self.top_k), act, cal, rep],
                   rerankers={'dynamic': [dyn,dyn_preference ]
                              }
                   ).run()

    def test_model_static_and_dynamic_reranking(self):
        category_100_reranker = GreedyKLReranker(
            item_dataframe=self.item_data, diversity_dimension=["category"],  top_k=self.top_k, pool_size=self.pool_size,

            target_distributions={"category": {"type": "discrete", "distr": {
                "a": 0.2, "b": 0.2, "c": 0.2, "d": 0.1, "e": 0.1, "f": 0.2}}},
            diversity_dimension_weight=[1])
    
        category_reranker_pm2= PM2Reranker(
            item_dataframe=self.item_data, diversity_dimension=["category"],  top_k=self.top_k, pool_size=self.pool_size,

            target_distributions={"category": {"type": "discrete", "distr": {
                 "a": 0.2, "b": 0.2, "c": 0.2, "d": 0.1, "e": 0.1, "f": 0.2}}},
            diversity_dimension_weight=[1])
        dyn = DynamicAttrReRanker(name="dyn_reranker", item_dataframe=self.item_data, diversity_dimension=[
                                  'sentiment', 'category'], top_k=self.top_k, pool_size=self.pool_size, bin_edges={'sentiment': [-0.5, 0, 0.5, 1.0]}, user_choice_model='preference_based_bias', 
                                  user_simulator_config_path = self.user_simulator_config_path)
        act = Activation(item_sentiment=self.Item_sentiment, k=self.top_k)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=self.top_k)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN"),
                                    UserKNN( k=3, similarity="cosine",
                                   name="UserKNN_cosine")],
                   metrics=[NDCG(k=self.top_k), Recall(
                       k=self.top_k), act, cal],
                    rerankers={'static': [category_100_reranker,category_reranker_pm2],
                    'dynamic': [dyn]
                              }
                   ).run()

    def test_multiple_model_static_and_dynamic_reranking(self):
        category_100_reranker = GreedyKLReranker(
            item_dataframe=self.item_data, diversity_dimension=["category"],  top_k=self.top_k, pool_size=self.pool_size,

            target_distributions={"category": {"type": "discrete", "distr": {
                "weather": 0.2, "news": 0.2, "finance": 0.2, "entertainment": 0.1, "sport": 0.1, "autos": 0.1, "travel": 0.1}}},
            diversity_dimension_weight=[1])
        LeastPop_reranker = LeastPopReranker(
            top_k=self.top_k, pool_size=self.pool_size)
        sentiment_category_dyn_reranker = DynamicAttrReRanker(name="dyn_reranker", item_dataframe=self.item_data, diversity_dimension=[
            'sentiment', 'category'], top_k=self.top_k, pool_size=self.pool_size, bin_edges={'sentiment': [-0.5, 0, 0.5, 1.0]}, user_choice_model='preference_based_bias',
            user_simulator_config_path = self.user_simulator_config_path)
        
        party_dyn_reranker = DynamicAttrReRanker(name="dyn_reranker2", item_dataframe=self.item_data, diversity_dimension=[
            'entities'], top_k=self.top_k, pool_size=self.pool_size, user_choice_model='logarithmic_rank_bias', 
            user_simulator_config_path = self.user_simulator_config_path,
            party_category_json_path =self.party_category_json_path)
        act = Activation(item_sentiment=self.Item_sentiment, k=self.top_k)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=self.top_k)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN"), ItemKNN(k=3, similarity="pearson",
                                                            name="ItemKNN")],
                   metrics=[NDCG(k=self.top_k), Recall(
                       k=self.top_k), act, cal],
                   rerankers={'static': [category_100_reranker, LeastPop_reranker],
                              'dynamic': [sentiment_category_dyn_reranker, party_dyn_reranker]},
                   ).run()

        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN"), ItemKNN(k=3, similarity="pearson",
                                                            name="ItemKNN")],
                   metrics=[NDCG(k=self.top_k), Recall(
                       k=self.top_k), act, cal],
                   rerankers={
                       'dynamic': [sentiment_category_dyn_reranker, party_dyn_reranker]},
                   ).run()


if __name__ == "__main__":
    unittest.main()
