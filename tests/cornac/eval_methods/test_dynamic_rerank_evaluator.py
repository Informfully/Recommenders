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

from cornac.eval_methods import BaseMethod, RatioSplit,DynamicReRankEval
from cornac.data import Dataset, Reader
from cornac.metrics import Precision, Recall
from cornac.models import MostPop
from cornac.models import Recommender
from cornac.metrics import Calibration
from cornac.rerankers import LeastPopReranker, DynamicAttrReRanker
from unittest.mock import MagicMock, patch
from cornac.eval_methods.dynamic_rerank_evaluator import ranking_eval_on_dyn_rerankers, diversity_eval_on_dyn_rerankers
from cornac.rerankers import DynamicReRanker


class TestDynamicRerankEval(unittest.TestCase):
    def setUp(self):
        self.uir = [('u0', 'item0', 3.0),  # 0:popularity 1
                    ('u1', 'item1', 3.0),  # 1: popularity 1
                    ('u1', 'item2', 3.0),
                    ('u2', 'item2', 3.0),  # 2: popularity 2
                    ('u1', 'item3', 3.0),  # 3: popularity 2
                    ('u2', 'item3', 3.0),
                    ('u2', 'item4', 3.0),  # 4: popularity 2
                    ('u3', 'item4', 3.0),
                    ('u0', 'item3', 2.0),
                    ('u0', 'item2', 3.0),
                    ('u0', 'item5', 3.0),
                    ('u1', 'item5', 3.0)]
        self.item_category = {0: 'Politics', 1: 'Sports', 2:  'Economy', 3: 'Technology', 4: 'Health',
                              5: 'Politics', 6: 'Sports'}

    def test_init(self):
        bm = BaseMethod(None, verbose=True)
        self.assertTrue(bm.exclude_unknowns)
        self.assertEqual(bm.rating_threshold, 1.0)
        dyn_eval = DynamicReRankEval(bm)
        self.assertTrue(dyn_eval.BaseEvaluator.exclude_unknowns)
        self.assertEqual(dyn_eval.BaseEvaluator.rating_threshold, 1.0)

    def test_trainset_none(self):
        bm = BaseMethod(None, verbose=True)
        initial_recommender_config = "Test"
        dyn_eval = DynamicReRankEval(bm)
        try:
            dyn_eval.evaluate(initial_recommender_config, [], [], {}, False)
        except ValueError:
            assert True

    def test_testset_none(self):
        bm = BaseMethod(None, verbose=True)
        bm.train_set = Dataset.from_uir(data=Reader().read("./tests/data.txt"))
        initial_recommender_config = "Test"
        dyn_eval = DynamicReRankEval(bm)
        try:
            dyn_eval.evaluate(initial_recommender_config, [], [], {}, False)
        except ValueError:
            assert True

    

    @patch('cornac.metrics.ranking.Precision')
    @patch('cornac.models.recommender.Recommender')
    @patch('cornac.rerankers.dynamic_attribute_penalization.dynamic_attribute_rerank.DynamicAttrReRanker')
    def test_ranking_eval(self, mock_precision,mock_Recommender,mock_DynamicAttrReRanker):
        bm = BaseMethod.from_splits(data=self.uir,
                                    train_data=self.uir[:-4], test_data=self.uir[-4:])
        dyn_eval = DynamicReRankEval(bm)
        mock_model = MagicMock(spec=Recommender)
        mock_model.rank.return_value = ( [0, 1, 2, 3, 4, 5], [1,2,3,4,5,6])
        mock_model.is_fitted = True
  
        mock_dyn_reranker = MagicMock(spec=DynamicReRanker)
        mock_dyn_reranker.name = "mock_dyn_reranker"
        # iteration 0: recommendation [0,1,2]
        # iteration 1: recommendation [3,4,5]
        mock_dyn_reranker.rerank.return_value = {0: [0, 1, 2], 1: [3, 4, 5]}

        mock_metric = MagicMock()
        mock_metric.name = "mock_precision"
        # iteration 0: recommendation [0,1,2]
        # iteration 1: recommendation [3,4,5]
        mock_metric.compute.return_value = 0.6
        mock_precision.return_value = mock_metric
        # dyn_reranked_results = [
        #     {0: [0, 1, 2], 1: [3, 4, 5]}, {0: [3, 4, 5], 1: [0, 1, 2]}]
        avg_results, user_results = ranking_eval_on_dyn_rerankers(model=mock_model, metrics=[mock_metric], rerankers = [mock_dyn_reranker], train_set=dyn_eval.BaseEvaluator.train_set,
                                                 test_set=dyn_eval.BaseEvaluator.test_set, rating_threshold = dyn_eval.BaseEvaluator.rating_threshold,exclude_unknowns=dyn_eval.BaseEvaluator.exclude_unknowns,
            verbose=dyn_eval.BaseEvaluator.verbose
 )
        expected_metric_result = mock_metric.compute.return_value

        # re-ranker 0,  metric id 0, user 0, iteration 0
        self.assertEqual(user_results[0][0][0][0], expected_metric_result)
        # re-ranker 0, metric id 0, user 1, iteration 1
        self.assertEqual(user_results[0][0][1][1], expected_metric_result)
        # re-ranker 0, metric id 0
        self.assertEqual(avg_results[0][0], expected_metric_result)

    @patch('cornac.metrics.diversity.Activation')
    @patch('cornac.models.recommender.Recommender')
    @patch('cornac.rerankers.dynamic_attribute_penalization.dynamic_attribute_rerank.DynamicAttrReRanker')
    def test_diversity_eval(self, mock_Activation,mock_Recommender,mock_DynamicAttrReRanker):
        bm = BaseMethod.from_splits(data=self.uir,
                                    train_data=self.uir[:-4], test_data=self.uir[-4:])
        dyn_eval = DynamicReRankEval(bm)

        mock_model = MagicMock(spec=Recommender)
        mock_model.rank.return_value = ( [0, 1, 2, 3, 4, 5], [1,2,3,4,5,6])
        mock_model.is_fitted = True
  
        mock_dyn_reranker = MagicMock(spec=DynamicReRanker)
        mock_dyn_reranker.name = "mock_dyn_reranker"
        mock_dyn_reranker.rerank.return_value = {0: [0, 1, 2], 1: [3, 4, 5]}

        mock_metric = MagicMock()
        mock_metric.name = "mock_activation"
        # iteration 0: recommendation [0,1,2]
        # iteration 1: recommendation [3,4,5]
        mock_metric.compute.return_value = 0.8
        mock_Activation.return_value = mock_metric
        # dyn_reranked_results = [
        #     {0: [0, 1, 2], 1: [3, 4, 5]}, {0: [3, 4, 5], 1: [0, 1, 2]}]
  
        
        avg_results, user_results = diversity_eval_on_dyn_rerankers(model=mock_model, metrics=[mock_metric], rerankers = [mock_dyn_reranker], train_set=dyn_eval.BaseEvaluator.train_set,
                                                 test_set=dyn_eval.BaseEvaluator.test_set, rating_threshold = dyn_eval.BaseEvaluator.rating_threshold,exclude_unknowns=dyn_eval.BaseEvaluator.exclude_unknowns,
            verbose=dyn_eval.BaseEvaluator.verbose
 )
        expected_metric_result = mock_metric.compute.return_value

        # re-ranker 0, metric id 0, user 0, iteration 0
        self.assertEqual(user_results[0][0][0][0], expected_metric_result)
        # re-ranker 0, metric id 0, user 1, iteration 1
        self.assertEqual(user_results[0][0][1][1], expected_metric_result)
        # re-ranker 0, metric id 0,
        self.assertEqual(avg_results[0][0], expected_metric_result)

    @patch('cornac.models.recommender.Recommender')
    @patch('cornac.rerankers.dynamic_attribute_penalization.dynamic_attribute_rerank.DynamicAttrReRanker')
    def test_evaluate(self, mock_Recommender, mock_DynamicAttrReRanker):
        mock_model = MagicMock(spec=Recommender)
        mock_model.name = "MockModel"
        mock_model.ranked_items = {
             0: [0, 1, 2, 3, 4, 5], 1: [3, 4, 5, 0, 2, 1]}
        mock_model.item_scores = {
             0: [0, 1, 2, 3, 4, 5], 1: [3, 4, 5, 0, 2, 1]}
        mock_dyn_reranker = MagicMock(spec=DynamicReRanker)
        mock_dyn_reranker.name = "mock_dyn_reranker"
        # iteration 0: recommendation [0,1,2]
        # iteration 1: recommendation [3,4,5]
        mock_dyn_reranker.rerank.return_value = {0: [0, 1, 2], 1: [3, 4, 5]}
        bm = BaseMethod.from_splits(data=self.uir,
                                    train_data=self.uir[:-4], test_data=self.uir[-4:])
        dyn_eval = DynamicReRankEval(bm)


        metrics = [Precision(k=3),  Calibration(k=3, item_feature=self.item_category,
                                                data_type="category", divergence_type="kl")]
     
        # initial_recommendations = {
        #     0: [1, 2, 3], 1: [0, 2, 3]}  # dummy data
        # initial_item_rank = {
        #     0: [0, 1, 2, 3, 4, 5], 1: [3, 4, 5, 0, 2, 1]}
        result_by_model_dynamic_reranker, val_result_by_model_dynamic_reranker = dyn_eval.evaluate(
            model=mock_model, metrics=metrics,  user_based = False, rerankers=[
                mock_dyn_reranker], show_validation=False)
        self.assertEqual(
            result_by_model_dynamic_reranker.model_name, mock_model.name)

        # expected_result_key = mock_model.name+"_"+mock_dyn_reranker.name+"_"+"Precision@3"
        expected_result_key = mock_dyn_reranker.name+"_"+"Precision@3"
        self.assertIn(expected_result_key, list(
            result_by_model_dynamic_reranker.metric_avg_results.keys()))

        self.assertEqual(
            result_by_model_dynamic_reranker.metric_user_results[expected_result_key][0][0], 1/3)  # for user 0
        self.assertEqual(
            result_by_model_dynamic_reranker.metric_user_results[expected_result_key][1][0], 0)  # for user 1

        self.assertEqual(
            result_by_model_dynamic_reranker.metric_user_results[expected_result_key][0][1], 2/3)  # for user 0
        self.assertEqual(
            result_by_model_dynamic_reranker.metric_user_results[expected_result_key][1][1], 1/3)  # for user 1
        expected_avg_result = ((1/3 + 2/3) /
                               2 + (0 + 1/3)/2)/2

        self.assertEqual(
            result_by_model_dynamic_reranker.metric_avg_results[expected_result_key], expected_avg_result)

        # expected_result_key = mock_model.name+"_" + \
        #     mock_dyn_reranker.name+"_"+"Calibration_category@3"
        expected_result_key =  mock_dyn_reranker.name+"_"+"Calibration_category@3"
        self.assertIn(expected_result_key, list(
            result_by_model_dynamic_reranker.metric_avg_results.keys()))

        # for user0, history items = 0
        # for user1, history items = 1,2,3
        self.assertAlmostEqual(
            result_by_model_dynamic_reranker.metric_user_results[expected_result_key][0][0],  1.5734209022576215, places=6)  # for user 0
        self.assertAlmostEqual(
            result_by_model_dynamic_reranker.metric_user_results[expected_result_key][1][0], 3.3148040620189443,  places=6)  # for user 1
        self.assertAlmostEqual(
            result_by_model_dynamic_reranker.metric_user_results[expected_result_key][0][0], 1.5734209022576215, places=6)  # for user 0

        expected_avg_result = ((1.5734209022576215+1.5734209022576215) /
                               2 + (3.3148040620189443+6.629608124037889)/2)/2
        self.assertAlmostEqual(
            result_by_model_dynamic_reranker.metric_avg_results[expected_result_key], expected_avg_result, places=6)

    @patch('cornac.rerankers.dynamic_attribute_penalization.dynamic_attribute_rerank.DynamicAttrReRanker')
    def test_evaluate_dyn_reranker(self, mock_DynamicAttrReRanker):
        uir = [('u0', 'item0', 3.0),  # 0:popularity 1
               ('u1', 'item1', 3.0),  # 1: popularity 1
               ('u1', 'item2', 3.0),
               ('u2', 'item2', 3.0),  # 2: popularity 2
               ('u1', 'item3', 3.0),  # 3: popularity 2
               ('u2', 'item3', 3.0),
               ('u2', 'item4', 3.0),  # 4: popularity 2
               ('u3', 'item4', 3.0),
               ('u0', 'item3', 2.0),
               ('u0', 'item2', 3.0),
               ('u0', 'item5', 3.0),
               ('u1', 'item5', 3.0)]
        # dummy DataFrame with item features
        ratio_split = RatioSplit.from_splits(data=uir,
                                             train_data=uir[:-4], test_data=uir[-4:])
        dyn_eval = DynamicReRankEval(ratio_split)
        mock_dyn_reranker = MagicMock(spec=DynamicAttrReRanker)
        mock_dyn_reranker.name = "mock_dyn_reranker"
        # iteration 0: recommendation [0,1,2]
        # iteration 1: recommendation [3,4,5]
        mock_dyn_reranker.rerank.return_value = {0: [0, 1, 2], 1: [3, 4, 5]}
        model = MostPop()
        model.fit(train_set =ratio_split.train_set)
        test_result, val_result = dyn_eval.evaluate(
            model, [Recall(k=3)],  rerankers=[mock_dyn_reranker], user_based=False)

        self.assertEqual(test_result.model_name, model.name)
        # most pop will recommend 2, 3 ,4 .
        # for user 0, item 2 and 3 in the test set are clicked
        # print(f"{model.ranked_items[1]}")
        # print(f"{model.ranked_items[0]}")
        # print(f"{model.score(1)}")
        # print(f"real{item_pop}")
        # print(f"{ratio_split.train_set.num_items}")
        # print(f"{test_result.metric_user_results['Recall@3']}")
        # print(f"test set num items{ratio_split.test_set.num_items}")
        # print(f"{model.rank(1,[0,4,5], -1)}")


        # self.assertEqual(test_result.metric_user_results["Recall@3"][0], 2/3)
        # # for user 1, 0 item in the test set are clicked
        # self.assertEqual(test_result.metric_user_results["Recall@3"][1], 1)
        # self.assertAlmostEqual(test_result.metric_avg_results["Recall@3"], 5/6)
        # for user 0, the mock dynamic reranking  in the iteration 0 will recommend item 0,1,2    
        self.assertEqual(
            test_result.metric_user_results["mock_dyn_reranker_Recall@3"][0][0], 1/3)
        # for user 0, the mock dynamic reranking in the iteration 1 will recommend item 3,4,5
        self.assertEqual(
            test_result.metric_user_results["mock_dyn_reranker_Recall@3"][0][1], 2/3)

        self.assertEqual(
            test_result.metric_user_results["mock_dyn_reranker_Recall@3"][1][0], 0)

        self.assertEqual(
            test_result.metric_user_results["mock_dyn_reranker_Recall@3"][1][1], 1)
        self.assertEqual(
            test_result.metric_avg_results["mock_dyn_reranker_Recall@3"], 1/2)


if __name__ == "__main__":
    unittest.main()
