import unittest
import numpy as np
from cornac.eval_methods import RatioSplit

import pandas as pd
from cornac.experiment.experiment import Experiment
from cornac.models import UserKNN
from cornac.rerankers import DynamicReRanker
from cornac.datasets import mind as mind
import os


class TestDynamicRerank(unittest.TestCase):
    def setUp(self):
        self.uir = [('u0', 'item0', 3.0),
                    ('u1', 'item1', 1.0),
                    ('u2', 'item0', 2.0),
                    ('u3', 'item1', 4.0),
                    ('u4', 'item1', 2.0),
                    ('u5', 'item2', 5.0),
                    ('u2', 'item3', 2.0),
                    ('u5', 'item3', 3.0),
                    ('u3', 'item2', 3.0),
                    ('u0', 'item4', 3.0),
                    ('u1', 'item4', 1.0),
                    ('u2', 'item2', 3.0),
                    ('u4', 'item3', 3.0),
                    ('u5', 'item2', 3.0),
                    ('u3', 'item5', 3.0),
                    ('u4', 'item5', 3.0),
                    ('u5', 'item6', 2.0),
                    ('u0', 'item6', 1.0)]
        # dummy DataFrame with item features
        self.item_data = pd.DataFrame({
            'sentiment': [0.1, 0.8, 0.6, 0.4, -0.2, -0.7, -0.3],
            'category': ['Politics', 'Sports', 'Economy', 'Technology',
                         'Health', 'Economy', 'Politics'],
        }, index=[0, 1, 2, 3, 4, 5, 6])

        self.user_simulator_config_path = os.path.join(os.path.dirname(
            __file__), 'test_user_simulator_config.ini')
        self.reranker = DynamicReRanker(
            name='test_reranker',
            item_dataframe=self.item_data,
            diversity_dimension=['sentiment', 'category'],
            top_k=5,
            pool_size=-1,
            feedback_window_size=2,
            user_simulator_config_path = self.user_simulator_config_path 

        )

    def test__init__(self):
        self.assertEqual(self.reranker.name, "test_reranker")
        self.assertEqual(self.reranker.feedback_window_size, 2)
        self.assertEqual(self.reranker.top_k, 5)
        self.assertEqual(self.reranker.pool_size, -1)
        self.assertEqual(self.reranker.feedback_window_size, 2)
        self.assertEqual(self.reranker.diversity_dimension,
                         ['sentiment', 'category'])
        pd.testing.assert_frame_equal(
            self.reranker.item_dataframe, self.item_data)

    def test_execute_filters(self):
        rs = RatioSplit.from_splits(data=self.uir,
                                    train_data=self.uir[:-5], test_data=self.uir[-5:])
        user_idx = 0
        candidate_items = np.arange(7)
        prediction_scores = np.arange(7)

        self.reranker.rerank(user_idx, rs.train_set,
                             candidate_items, prediction_scores)
        self.assertEqual(self.reranker.candidate_items[user_idx], [
                         0, 1, 2, 3, 4, 5, 6])
        self.reranker.execute_filters(user_idx=user_idx)
        expected_items_dict = {0: [1, 2, 3, 5, 6]}
        expected_scores_dict = {0: [1, 2, 3, 5, 6]}

        self.assertDictEqual(
            self.reranker.candidate_items, expected_items_dict)
        # self.assertDictEqual(
        #     list(self.reranker.candidate_scores), expected_scores_dict)
        user_idx = 4
        self.reranker.rerank(user_idx, rs.train_set,
                             candidate_items, prediction_scores)
        self.reranker.execute_filters(user_idx=user_idx)
        expected_items_dict = {0: [1, 2, 3, 5, 6], 4: [0,  2, 4, 5, 6]}
        expected_scores_dict = {0: [1, 2, 3, 5, 6], 4: [0, 2, 4, 5, 6]}
        self.assertDictEqual(
            self.reranker.candidate_items, expected_items_dict)
        # self.assertDictEqual(
        #     self.reranker.candidate_scores, expected_scores_dict)

    def test_add_user(self):

        user_idx = 1
        self.reranker.user_history = {1: [1, 4]}
        self.reranker.add_user(user_idx)
        self.assertIn(user_idx, self.reranker.users)
        self.assertEqual(
            self.reranker.users[user_idx].choice_model, "logarithmic_rank_bias")
        self.assertListEqual(
            self.reranker.users[user_idx].history, [1, 4])

    def test_filter_seen_items(self):
        self.reranker.candidate_items = {1: [1, 2, 3, 4]}
        self.reranker.user_history = {1: [1, 4]}
        self.reranker.add_user(1)
        self.reranker.users[1].seen_items = [1, 2]
        self.reranker.filter_seen_items(1)
        self.assertListEqual(self.reranker.candidate_items[1], [3, 4])

    def test_rerank(self):

        rs = RatioSplit.from_splits(data=self.uir,
                                    train_data=self.uir[:-
                                                        5], test_data=self.uir[-5:],
                                    exclude_unknowns=True, verbose=True, seed=123, rating_threshold=1)

        user_idx = 5
        candidate_items = np.arange(7)
        prediction_scores = np.arange(7)
        self.reranker.rerank(user_idx, rs.train_set,
                             candidate_items, prediction_scores)

        self.assertEqual(set(self.reranker.candidate_items.keys()), {5})
        self.assertEqual(set(self.reranker.candidate_scores.keys()), {5})

        # assert that the arrays within the dictionaries are equal

        self.assertListEqual(self.reranker.candidate_items[5],  [
                             0, 1, 2, 3, 4, 5, 6])
        self.assertListEqual(list(self.reranker.candidate_scores[5]), [
                             0, 1, 2, 3, 4, 5, 6])
        rr_reduced_pool_size = DynamicReRanker(
            name="test_reranker", top_k=3, pool_size=4)
        self.assertEqual(rr_reduced_pool_size.pool_size, 4)
        rr_reduced_pool_size.rerank(
            user_idx, rs.train_set, candidate_items, prediction_scores)
        self.assertListEqual(
            list(rr_reduced_pool_size.candidate_scores[5]), [0, 1, 2, 3, 4, 5, 6])
        self.assertListEqual(
            list(rr_reduced_pool_size.candidate_items[5]),  [0, 1, 2, 3])

        rr_pool_size_negative = DynamicReRanker(
            name="test_reranker", top_k=3, pool_size=-1)
        self.assertEqual(rr_pool_size_negative.pool_size, -1)
        rr_pool_size_negative.rerank(
            user_idx, rs.train_set, candidate_items, prediction_scores)
        self.assertListEqual(
            list(rr_pool_size_negative.candidate_scores[5]), [0, 1, 2, 3, 4, 5, 6])
        self.assertListEqual(
            list(rr_pool_size_negative.candidate_items[5]),  [0, 1, 2, 3, 4, 5, 6])


if __name__ == '__main__':
    unittest.main()
