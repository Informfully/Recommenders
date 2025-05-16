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
import numpy as np

from cornac.eval_methods import RatioSplit
from cornac.experiment.experiment import Experiment
from cornac.metrics import NDCG, Recall

from cornac.metrics import Activation
from cornac.metrics import Calibration

from cornac.models import UserKNN, ItemKNN
from cornac.datasets import mind as mind
from cornac.rerankers import GreedyKLReranker, PM2Reranker, LeastPopReranker
import pandas as pd


class TestExperiment(unittest.TestCase):
    def setUp(self):
        feedback = mind.load_feedback(
            fpath = "./examples/example_news_files/example_impression_all_uir.csv")
        # Mind dataset min rating = 1
        sentiment = mind.load_sentiment(
            fpath = "./examples/example_news_files/example_sentiment.json")
        category = mind.load_category(
            fpath = "./examples/example_news_files/example_category.json")
        
        complexity = mind.load_complexity(
            fpath="./examples/example_news_files/example_readability.json")
        story = mind.load_story(fpath="./examples/example_news_files/example_story.json")
        # Important! For dynamic re-ranking,  `keep_empty` need to be set as True
        entities = mind.load_entities(fpath="./examples/example_news_files/example_party.json", keep_empty = True)

        genre = mind.load_category_multi(
            fpath = "./examples/example_news_files/example_category.json")
        min_maj = mind.load_min_maj(fpath="./examples/example_news_files/example_minor_major_ratio.json")

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
        self.Item_complexity = mind.build(
            data=complexity, id_map=self.rs.train_set.iid_map)
        self.Item_stories = mind.build(
            data=story, id_map=self.rs.train_set.iid_map)
        self.Item_entities = mind.build(
            data=entities, id_map=self.rs.train_set.iid_map, keep_empty = True)
        self.Item_min_major = mind.build(
            data=min_maj, id_map=self.rs.train_set.iid_map)
        self.Item_genre = mind.build(
            data=genre, id_map=self.rs.train_set.iid_map)
   
        self.Item_feature = self.Item_genre
        out_pd = (pd.Series(self.Item_category).to_frame('category')
                  .join(pd.Series(self.Item_entities).to_frame('entities'), how='outer'))
       
        out_pd = out_pd.join(
            pd.Series(self.Item_complexity).to_frame('complexity'), how='outer')
        out_pd = out_pd.join(
            pd.Series(self.Item_sentiment).to_frame('sentiment'), how='outer')
        out_pd = out_pd.join(
            pd.Series(self.Item_stories).to_frame('story'), how='outer')
        self.item_data = out_pd

    def test_without_reranker(self):
        act = Activation(item_sentiment=self.Item_sentiment, k=10)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=10)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN")],
                   metrics=[NDCG(k=10), Recall(k=10), act, cal]
                   ).run()

    def test_configFile_input(self):
        config_file = "./tests/configs/reranker_configs/test_reranker.ini"
        reranker_greedy_KL = GreedyKLReranker(
            item_dataframe=self.item_data, diversity_dimension=["sentiment", "category"], pool_size=600, top_k=10, config_file=config_file)
        reranker_pm2 = PM2Reranker(
            item_dataframe=self.item_data,  config_file=config_file)
        # Verify the re-ranker attributes are set from the config file
        self.assertEqual(reranker_greedy_KL.name, "GreedyKL")
        self.assertEqual(reranker_greedy_KL.top_k, 20)
        self.assertEqual(reranker_greedy_KL.pool_size, -1)
        self.assertEqual(reranker_greedy_KL.diversity_dimension, ["sentiment", "entities"])
        self.assertEqual(reranker_greedy_KL.diversity_dimension_weight, [0.5, 0.5])



        
        # Check target distributions for reranker_greedy_KL
        target_distributions = reranker_greedy_KL.target_distributions
        # Ensure "sentiment" and "entities" distributions are present in the list
        sentiment_distr = next(
            (distr for distr in target_distributions if distr["type"] == "continuous"), None)
        entities_distr = next(
            (distr for distr in target_distributions if distr["type"] == "parties"), None)

        # Verify sentiment distribution structure
        self.assertEqual(len(sentiment_distr["distr"]), 4)  # Should have 4 ranges
        self.assertAlmostEqual(sentiment_distr["distr"][0]["prob"], 0.2)

        # Verify entities distribution structure
        self.assertEqual(len(entities_distr["distr"]), 5)  # Should have 5 categories
        self.assertAlmostEqual(entities_distr["distr"][-1]["prob"], 0.4)


        

        # Verify attributes for reranker_greedy_KL
        self.assertEqual(reranker_greedy_KL.name, "GreedyKL")
        self.assertEqual(reranker_greedy_KL.top_k, 20)
        self.assertEqual(reranker_greedy_KL.pool_size, -1)
        self.assertEqual(reranker_greedy_KL.diversity_dimension, ["sentiment", "entities"])
        self.assertEqual(reranker_greedy_KL.diversity_dimension_weight, [0.5, 0.5])

    

        # Verify attributes for reranker_pm2
        self.assertEqual(reranker_pm2.name, "PM2")
        self.assertEqual(reranker_pm2.top_k, 20)
        self.assertEqual(reranker_pm2.pool_size, -1)
        self.assertEqual(reranker_pm2.diversity_dimension, ["sentiment", "entities"])
        self.assertEqual(reranker_pm2.diversity_dimension_weight, [0.5, 0.5])

        # Check target distributions for reranker_pm2
        target_distributions_pm2 = reranker_pm2.target_distributions
        sentiment_distr_pm2 = next(
            (distr for distr in target_distributions_pm2 if distr["type"] == "continuous"), None)
        entities_distr_pm2 = next(
            (distr for distr in target_distributions_pm2 if distr["type"] == "parties"), None)


        self.assertEqual(len(sentiment_distr_pm2["distr"]), 4)
        self.assertAlmostEqual(sentiment_distr_pm2["distr"][0]["prob"], 0.2)
        self.assertEqual(len(entities_distr_pm2["distr"]), 5)
        self.assertAlmostEqual(entities_distr_pm2["distr"][-1]["prob"], 0.4)


        
        act = Activation(item_sentiment=self.Item_sentiment, k=10)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=10)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN")],
                   metrics=[NDCG(k=10), Recall(k=10), act, cal],
                   rerankers={'static': [reranker_greedy_KL,reranker_pm2]}
                   ).run()

    def test_parameter_input(self):
        category_100_reranker = GreedyKLReranker(
            item_dataframe=self.item_data, diversity_dimension=["category"],  top_k=10, pool_size=600,

            target_distributions={"category": {"type": "discrete", "distr": {
                "weather": 0.2, "news": 0.2, "finance": 0.2, "entertainment": 0.1, "sport": 0.1, "autos": 0.1, "travel": 0.1}}},
            diversity_dimension_weight=[1])
        act = Activation(item_sentiment=self.Item_sentiment, k=10)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=10)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN")],
                   metrics=[NDCG(k=10), Recall(k=10), act, cal],
                   rerankers={'static': [category_100_reranker]}
                   ).run()

    def test_multi_models(self):
        category_100_reranker = GreedyKLReranker(
            item_dataframe=self.item_data, diversity_dimension=["category"],  top_k=10, pool_size=600,

            target_distributions={"category": {"type": "discrete", "distr": {
                "weather": 0.2, "news": 0.2, "finance": 0.2, "entertainment": 0.1, "sport": 0.1, "autos": 0.1, "travel": 0.1}}},
            diversity_dimension_weight=[1])
        act = Activation(item_sentiment=self.Item_sentiment, k=10)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=10)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN"), ItemKNN(k=3, similarity="pearson",
                                                            name="ItemKNN")],
                   metrics=[NDCG(k=10), Recall(k=10), act, cal],
                   rerankers={'static': [category_100_reranker]}
                   ).run()

    def test_multi_rerankers(self):
        # config_file = "./experiments/configs/reranker_configs/reranker.ini"
        config_file = "./tests/configs/reranker_configs/test_reranker.ini"
        category_80_sentiment_20_reranker = GreedyKLReranker(
            item_dataframe=self.item_data, diversity_dimension=["sentiment", "category"], pool_size=600, top_k=10, config_file=config_file)

        LeastPop_reranker = LeastPopReranker(top_k=10, pool_size=600)
        act = Activation(item_sentiment=self.Item_sentiment, k=10)
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=10)
        Experiment(eval_method=self.rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN")],
                   metrics=[NDCG(k=10), Recall(k=10), act, cal],
                   rerankers={'static': [
                       category_80_sentiment_20_reranker, LeastPop_reranker]}

                   ).run()


if __name__ == "__main__":
    unittest.main()
