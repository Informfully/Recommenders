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
from cornac.models import MostPop
from cornac.metrics import Recall
from cornac.experiment.experiment import Experiment
from cornac.experiment.pipelineExperiment import PipelineExperiment
from cornac.metrics import Precision
from cornac.metrics import Activation
from cornac.metrics import Calibration
from cornac.datasets import mind as mind
from cornac.rerankers.dynamic_attribute_penalization import DynamicAttrReRanker
from cornac.rerankers import GreedyKLReranker
from cornac.rerankers.pm2 import PM2Reranker
import pandas as pd


class TestExperiment(unittest.TestCase):
    def setUp(self):
        feedback = mind.load_feedback(
            fpath="./examples/example_news_files/example_impression_all_uir.csv")


        mind_ratio_split = RatioSplit(
            data=feedback,
            test_size=0.2,
            exclude_unknowns=True,
            verbose=True,
            seed=123,
            rating_threshold=0.5,
        )
        self.mind_ratio_split = mind_ratio_split
        self.dataset_save_path = "./tests/MostPop_trained"
        self.mind_ratio_split.save(
            self.dataset_save_path)

        # metrics
        sentiment = mind.load_sentiment(
            fpath="./examples/example_news_files/example_sentiment.json")
        category = mind.load_category(
            fpath="./examples/example_news_files/example_category.json")
        # Important! For dynamic re-ranking,  `keep_empty` need to be set as True
        entities = mind.load_entities(fpath="./examples/example_news_files/example_party.json", keep_empty = True)

        complexity = mind.load_complexity(
            fpath="./examples/example_news_files/example_readability.json")
        story = mind.load_story(fpath="./examples/example_news_files/example_story.json")


        genre = mind.load_category_multi(
            fpath = "./examples/example_news_files/example_category.json")
        min_maj = mind.load_min_maj(fpath="./examples/example_news_files/example_minor_major_ratio.json")


        Item_sentiment = mind.build(
            data=sentiment, id_map=mind_ratio_split.train_set.iid_map)

        Item_category = mind.build(
            data=category, id_map=mind_ratio_split.train_set.iid_map)
        Item_complexity = mind.build(
            data=complexity, id_map=mind_ratio_split.train_set.iid_map)
        Item_stories = mind.build(
            data=story, id_map=mind_ratio_split.train_set.iid_map)
        Item_entities = mind.build(
            data=entities, id_map=mind_ratio_split.train_set.iid_map)
        Item_min_major = mind.build(
            data=min_maj, id_map=mind_ratio_split.train_set.iid_map)
        Item_genre = mind.build(
            data=genre, id_map=mind_ratio_split.train_set.iid_map)
      
        Item_feature = Item_genre
        # prepare dataframe file for d_drw model
        article_feature_dataframe = (pd.Series(Item_category).to_frame('category')
                                     .join(pd.Series(Item_entities).to_frame('entities'), how='outer'))

        article_feature_dataframe = article_feature_dataframe.join(
            pd.Series(Item_complexity).to_frame('complexity'), how='outer')
        article_feature_dataframe = article_feature_dataframe.join(
            pd.Series(Item_sentiment).to_frame('sentiment'), how='outer')
        article_feature_dataframe = article_feature_dataframe.join(
            pd.Series(Item_stories).to_frame('story'), how='outer')
        self.article_feature_dataframe = article_feature_dataframe
        self.most_pop_model = MostPop()

        targetSize = 20

        self.metrics = [
            Recall(k=targetSize),
            Precision(k=targetSize),
            Activation(item_sentiment=Item_sentiment,
                       divergence_type='JS', k=targetSize),
            Calibration(item_feature=Item_category,
                        data_type="category", divergence_type='JS', k=targetSize),
            Calibration(item_feature=Item_complexity,
                        data_type="complexity", divergence_type='JS', k=targetSize),

        ]
        # Setup Target Distribution
        Target_Mind_distribution = {
        "category": {
            "type": "discrete",
            "distr": {
            "a": 0.16666,
            "b": 0.16666,
            "c": 0.16666,
            "d": 0.16666,
            "e": 0.16666,
            "f": 0.1667
            }
        },
        "sentiment": {
            "type": "continuous",
            "distr": [
            {
                "min": -1,
                "max": -0.5,
                "prob": 0.25
            },
            {
                "min": -0.5,
                "max": 0,
                "prob": 0.25
            },
            {
                "min": 0,
                "max": 0.5,
                "prob": 0.25
            },
            {
                "min": 0.5,
                "max": 1.01,
                "prob": 0.25
            }
            ]
        },
        "entities": {
            "type": "parties",
            "distr": [
            {
                "description": "only mention",
                "contain": ["party1"],
                "prob": 0.2
            },
            {
                "description": "only mention",
                "contain": ["party2","party3"],
                "prob": 0.2
            },
            {
                "description": "composition",
                "contain":[["party1"], ["party2","party3"]],
                "prob": 0.2
            },
            {
                "description": "minority but can also mention",
                "contain": ["party1", "party2","party3"],
                "prob": 0.2
            },
            {"description": "no parties", "contain": [], "prob": 0.2}
            ]
        }
        }

        # Set up rerankers
        self.sentiment_party_greedy_reranker = GreedyKLReranker(
            item_dataframe=article_feature_dataframe,
            diversity_dimension=["sentiment", "entities"],
            top_k=targetSize,
            target_distributions=Target_Mind_distribution,
            diversity_dimension_weight=[0.5, 0.5]
        )
     

        self.sentiment_party_pm2_reranker = PM2Reranker(
            item_dataframe=article_feature_dataframe,
            diversity_dimension=["sentiment", "entities"],
            top_k=targetSize,
            target_distributions=Target_Mind_distribution,
            diversity_dimension_weight=[0.5, 0.5]
        )

        self.party_category_json_path =  "./tests/configs/reranker_configs/party_category.json"
        bin_edges = {'sentiment': [-1, -0.5, 0, 0.5, 1]}
        self.dynamic_reranker_rank_bias = DynamicAttrReRanker(name="DYN_reranker_probByposition",
                                                              item_dataframe=article_feature_dataframe, diversity_dimension=["sentiment", "entities"],  top_k=targetSize,
                                                              feedback_window_size=3, bin_edges=bin_edges, user_choice_model="logarithmic_rank_bias",
                                                                user_simulator_config_path='./tests/configs/reranker_configs/user_simulator_config.ini',
                                                                party_category_json_path = self.party_category_json_path)

        self.dynamic_reranker_preference_bias = DynamicAttrReRanker(name="DYN_reranker_probByPreference",
                                                                    item_dataframe=article_feature_dataframe, diversity_dimension=["sentiment", "entities"],  top_k=targetSize,
                                                                    feedback_window_size=5, bin_edges=bin_edges, user_choice_model="preference_based_bias",
                                                                      user_simulator_config_path='./tests/configs/reranker_configs/user_simulator_config.ini',
                                                                      party_category_json_path = self.party_category_json_path)
        # Define reranking pipeline

    def test_with_mostpop(self):

        Experiment(eval_method=self.mind_ratio_split, models=[self.most_pop_model],
                   metrics=self.metrics,
                   save_dir=self.dataset_save_path
                   ).run()

    def test_pipeline_experiment(self):
        experiment_config_file = './tests/configs/experiment_configs/demo_experiment_pipeline.ini'

        pipelineExp = PipelineExperiment(model=[self.most_pop_model],
                                         metrics=self.metrics,
                                         rerankers={'static': [self.sentiment_party_greedy_reranker,
                                                                self.sentiment_party_pm2_reranker],
                                                    'dynamic': [self.dynamic_reranker_rank_bias, self.dynamic_reranker_preference_bias]},
                                         user_based=True,
                                         show_validation=True,
                                         verbose=False,
                                         pipeline_config_file=experiment_config_file
                                         )

        # Test get_mode_and_paths
        self.assertFalse(pipelineExp.mode_and_paths['model']['mode'])
        self.assertTrue(pipelineExp.mode_and_paths['static_reranker']['mode'])
        self.assertTrue(pipelineExp.mode_and_paths['dynamic_reranker']['mode'])

        self.assertFalse(pipelineExp.mode_and_paths['model']['eval_mode'])
        self.assertTrue(
            pipelineExp.mode_and_paths['static_reranker']['eval_mode'])
        self.assertTrue(
            pipelineExp.mode_and_paths['dynamic_reranker']['eval_mode'])

        expected_recommendation_path = self.dataset_save_path+"/MostPop"
        # Test paths
        self.assertEqual(
            pipelineExp.mode_and_paths['model']['path'], expected_recommendation_path)
        self.assertEqual(
            pipelineExp.mode_and_paths['model']['save_eval_path'], expected_recommendation_path)

        expected_static_eval_path = self.dataset_save_path+"/static_reranked"
        self.assertEqual(
            pipelineExp.mode_and_paths['static_reranker']['path'],  expected_recommendation_path)
        self.assertEqual(
            pipelineExp.mode_and_paths['static_reranker']['save_eval_path'], expected_static_eval_path)
        expected_dynamic_eval_path = self.dataset_save_path+"/dynamic_reranked"
        self.assertEqual(
            pipelineExp.mode_and_paths['dynamic_reranker']['path'], expected_recommendation_path)
        self.assertEqual(
            pipelineExp.mode_and_paths['dynamic_reranker']['save_eval_path'], expected_dynamic_eval_path)

        # Test load dataset
        self.assertEqual(self.mind_ratio_split.train_set.num_items,
                         pipelineExp.eval_method.train_set.num_items)
        self.assertEqual(self.mind_ratio_split.test_set.num_items,
                         pipelineExp.eval_method.test_set.num_items)
        self.assertEqual(self.mind_ratio_split.train_set.iid_map,
                         pipelineExp.eval_method.train_set.iid_map)

        # Test _validate_models
        self.assertEqual(self.most_pop_model.name, pipelineExp.model.name)
        pipelineExp.run()


if __name__ == "__main__":
    unittest.main()
