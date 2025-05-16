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

from cornac.eval_methods import BaseMethod, RatioSplit, StaticReRankEval
from cornac.data import FeatureModality, TextModality, ReviewModality, ImageModality, SentimentModality
from cornac.data import Dataset, Reader
from cornac.metrics import MAE, AUC, Recall
from cornac.models import MF
from cornac.metrics import NDCG
from cornac.metrics import GiniCoeff
from cornac.metrics import ILD
from cornac.metrics import EILD
from cornac.metrics import Activation
from cornac.metrics import Calibration
from cornac.metrics import Fragmentation
from cornac.metrics import Representation
from cornac.metrics import AlternativeVoices
from cornac.metrics import Binomial
from cornac.models import UserKNN
import numpy as np
from cornac.rerankers import LeastPopReranker


class TestRecomRerankEval(unittest.TestCase):
    def test_init(self):
        bm = BaseMethod(None, verbose=True)
        self.assertTrue(bm.exclude_unknowns)
        self.assertEqual(bm.rating_threshold, 1.0)
        rre = StaticReRankEval(bm)
        self.assertTrue(rre.BaseEvaluator.exclude_unknowns)
        self.assertEqual(rre.BaseEvaluator.rating_threshold, 1.0)

    def test_trainset_none(self):
        bm = BaseMethod(None, verbose=True)
        rre = StaticReRankEval(bm)
        try:
            rre.evaluate(None, {}, False, None)
        except ValueError:
            assert True

    def test_testset_none(self):
        bm = BaseMethod(None, verbose=True)
        bm.train_set = Dataset.from_uir(data=Reader().read("./tests/data.txt"))
        rre = StaticReRankEval(bm)
        try:
            rre.evaluate(None, {},  False, None)
        except ValueError:
            assert True

    def test_evaluate_none_reranker(self):
        data = Reader().read("./tests/data.txt")
        bm = BaseMethod.from_splits(train_data=data[:-1], test_data=data[-1:])
        model = MF(k=1, max_iter=0)
        result, val_result = bm.evaluate(
            model, metrics=[MAE()], user_based=False)
        rre = StaticReRankEval(bm)
        with self.assertRaises(ValueError):
            rre.evaluate(
                model, metrics=[MAE()], user_based=False, rerankers=None)

    def test_reranker(self):
        data = [('u0', 'i0', 3.0),
                ('u1', 'i1', 1.0),
                ('u1', 'i0', 3.0),
                ('u2', 'i1', 2.0),
                ('u2', 'i2', 2.0),
                ('u2', 'i0', 3.0),
                ('u3', 'i3', 4.0),
                ('u3', 'i1', 4.0),
                ('u3', 'i0', 4.0),
                ('u4', 'i1', 2.0),
                ('u4', 'i2', 2.0),
                ('u4', 'i3', 4.0),
                ('u5', 'i4', 5.0),
                ('u5', 'i0', 3.0),
                ('u5', 'i2', 3.0),
                ('u3', 'i5', 3.0),
                ('u0', 'i5', 3.0),
                ('u1', 'i4', 1.0)]
        bm = BaseMethod.from_splits(train_data=data[:-3], test_data=data[-3:])
        model = UserKNN(k=1, similarity="pearson", name="UserKNN-Pearson")
        ndcg = NDCG(k=2)

        Item_sentiment = {0: 0.5, 1: -0.2, 2: 0, 3: 0.8,
                          4: 1, 5: -0.7}

        act = Activation(item_sentiment=Item_sentiment, k=2)

        LeastPop_reranker = LeastPopReranker(top_k=4, pool_size=6)

        rre = StaticReRankEval(bm)
        model.fit(rre.BaseEvaluator.train_set, rre.BaseEvaluator.val_set)
        # test_item_ids_list = list(bm.test_set.item_indices)
        test_item_ids_list = list(bm.test_set.uir_tuple[1])
        item_idx2id = {v: k for k, v in bm.train_set.iid_map.items()}
        print("train set item_idx2id:{}".format(item_idx2id))
        item_idx2id = {v: k for k, v in bm.test_set.iid_map.items()}
        print("item_idx2id:{}".format(item_idx2id))
        user_idx2id = {v: k for k, v in bm.test_set.uid_map.items()}
        print("user_idx2id:{}".format(user_idx2id))
        
        result_only_diversity_metric, val_result_only_diversity_metric = rre.evaluate(
            model, metrics=[
                act], user_based=True, rerankers=[LeastPop_reranker])

        result_only_ranking_metric, val_result_only_ranking_metric = rre.evaluate(
            model, metrics=[
                ndcg], user_based=True, rerankers=[LeastPop_reranker])

        result_ranking_diversity_metric, val_result_ranking_diversity_metric = rre.evaluate(
            model, metrics=[act,
                            ndcg], user_based=True, rerankers=[LeastPop_reranker])

        self.assertEqual(
            result_only_diversity_metric.metric_avg_results['LeastPop_Activation@2'],  result_ranking_diversity_metric.metric_avg_results['LeastPop_Activation@2'])
        self.assertEqual(
            result_only_ranking_metric.metric_avg_results['LeastPop_NDCG@2'],  result_ranking_diversity_metric.metric_avg_results['LeastPop_NDCG@2'])

    def test_evaluate_reranker(self):
        data = [('u0', 'i0', 3.0),
                ('u1', 'i1', 1.0),
                ('u1', 'i0', 3.0),
                ('u2', 'i1', 2.0),
                ('u2', 'i2', 2.0),
                ('u2', 'i0', 3.0),
                ('u3', 'i3', 4.0),
                ('u3', 'i1', 4.0),
                ('u3', 'i0', 4.0),
                ('u4', 'i1', 2.0),
                ('u4', 'i2', 2.0),
                ('u4', 'i3', 4.0),
                ('u5', 'i4', 5.0),
                ('u5', 'i0', 3.0),
                ('u5', 'i2', 3.0),
                ('u3', 'i5', 3.0),
                ('u0', 'i5', 3.0),
                ('u1', 'i4', 1.0)]
        # popularity
        # i5 = 0
        # i4 =1
        # i3 = 2
        # i2 = 3
        # i1 = 4
        # i0 = 5
        ratio_split = RatioSplit.from_splits(data=data,
                                             train_data=data[:-3], test_data=data[-3:])
        rre = StaticReRankEval(ratio_split)
        LeastPop_reranker = LeastPopReranker(top_k=1, pool_size=6)
        model = MF()
        model.fit(ratio_split.train_set)
        test_result, val_result = rre.evaluate(
            model, [Recall()], rerankers=[LeastPop_reranker], user_based=True)
        item_popularity = LeastPop_reranker.popularityScores(
            [0, 1, 2, 3, 4, 5])
        expected_item_popularity = [5, 4, 3, 2, 1, 0]
        self.assertListEqual(item_popularity.tolist(),
                             expected_item_popularity)

        self.assertEqual(
            test_result.metric_user_results['LeastPop_Recall@-1'][3], 1)  # for user 3
        self.assertEqual(
            test_result.metric_user_results['LeastPop_Recall@-1'][0], 1)  # for user 0
        self.assertEqual(
            test_result.metric_user_results['LeastPop_Recall@-1'][1], 0)  # for user 1


if __name__ == "__main__":
    unittest.main()
