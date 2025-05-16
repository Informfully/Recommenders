import unittest
import numpy as np
from cornac.datasets import movielens
from cornac.data import Reader
from cornac.eval_methods import RatioSplit

import pandas as pd
import json
import cornac
from cornac.experiment.experiment import Experiment
from cornac.models import UserKNN
from cornac.rerankers import ReRanker
from cornac.datasets import mind as mind
import os


class TestRerank(unittest.TestCase):
    def setUp(self):
        self.data = [('u0', 'item0', 3.0),
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

    def test__init__(self):
        rr = ReRanker(name="test_reranker")
        self.assertEqual(rr.name, "test_reranker")

    def test_execute_filters(self):
        rr = ReRanker(name="test_reranker")

        rs = RatioSplit.from_splits(data=self.data,
                                    train_data=self.data[:-5], test_data=self.data[-5:])
        user_idx = 0
        candidate_items = np.arange(7)
        prediction_scores = np.arange(7)
        rr.rerank(user_idx, rs.train_set, candidate_items, prediction_scores)
        rr.execute_filters(user_idx=user_idx)
        expected_items_dict = {0: [1, 2, 3, 5, 6]}
        expected_scores_dict = {0: [1, 2, 3, 5, 6]}

        self.assertDictEqual(rr.candidate_items, expected_items_dict)

        user_idx = 4
        rr.rerank(user_idx, rs.train_set, candidate_items, prediction_scores)
        rr.execute_filters(user_idx=user_idx)
        expected_items_dict = {0: [1, 2, 3, 5, 6], 4: [0,  2, 4, 5, 6]}

        self.assertDictEqual(rr.candidate_items, expected_items_dict)

    def test_default_score(self):
        rr = ReRanker(name="test_reranker")
        rs = RatioSplit.from_splits(data=self.data,
                                    train_data=self.data[:-5], test_data=self.data[-5:])
        user_idx = 0
        candidate_items = np.arange(7)
        prediction_scores = np.arange(7)
        rr.rerank(user_idx, rs.train_set, candidate_items, prediction_scores)
        expected_mean = rs.train_set.global_mean
        self.assertEqual(rr.default_score(), expected_mean)

    def test_save_load(self):
        rr = ReRanker(name="test_reranker")
        rs = RatioSplit.from_splits(data=self.data,
                                    train_data=self.data[:-5], test_data=self.data[-5:])
        user_idx = 0
        candidate_items = np.arange(7)
        prediction_scores = np.arange(7)
        rr.rerank(user_idx, rs.train_set, candidate_items, prediction_scores)
        rr.execute_filters(user_idx=user_idx)
        current_folder_path = os.getcwd()
        saved_file = rr.save(current_folder_path)
        loaded_reranker = ReRanker.load(saved_file)
        self.assertEqual(loaded_reranker.load_from, saved_file)
        expected_items_dict = {0: [1, 2, 3, 5, 6]}

        self.assertDictEqual(
            loaded_reranker.candidate_items, expected_items_dict)

    def test_rerank(self):

        rs = RatioSplit.from_splits(data=self.data,
                                    train_data=self.data[:-
                                                         5], test_data=self.data[-5:],
                                    exclude_unknowns=True, verbose=True, seed=123, rating_threshold=1)

        rr = ReRanker(name="test_reranker", top_k=5, pool_size=10)
        user_idx = 5
        candidate_items = np.arange(7)
        prediction_scores = np.arange(7)
        rr.rerank(user_idx, rs.train_set, candidate_items, prediction_scores)
        self.assertEqual(set(rr.candidate_items.keys()), {5})
        self.assertEqual(set(rr.candidate_scores.keys()), {5})

        # assert that the arrays within the dictionaries are equal

        self.assertListEqual(list(rr.candidate_items[5]),  [
                             0, 1, 2, 3, 4, 5, 6])
        self.assertListEqual(list(rr.candidate_scores[5]), [
                             0, 1, 2, 3, 4, 5, 6])
        rr_reduced_pool_size = ReRanker(
            name="test_reranker", top_k=5, pool_size=4)
        self.assertEqual(rr_reduced_pool_size.pool_size, 4)
        rr_reduced_pool_size.rerank(
            user_idx, rs.train_set, candidate_items, prediction_scores)
        self.assertListEqual(
            list(rr_reduced_pool_size.candidate_scores[5]), [0, 1, 2, 3, 4, 5, 6])
        self.assertListEqual(
            list(rr_reduced_pool_size.candidate_items[5]),  [0, 1, 2, 3])

        rr_negative_pool_size = ReRanker(
            name="test_reranker", top_k=5, pool_size=-1)
        self.assertEqual(rr_negative_pool_size.pool_size, -1)
        rr_negative_pool_size.rerank(
            user_idx, rs.train_set, candidate_items, prediction_scores)
        self.assertListEqual(
            list(rr_negative_pool_size.candidate_scores[5]), [0, 1, 2, 3, 4, 5, 6])
        self.assertListEqual(
            list(rr_negative_pool_size.candidate_items[5]),  [0, 1, 2, 3, 4, 5, 6])

    def test_filter(self):

        rr = ReRanker(name="test_reranker")

        rs = RatioSplit.from_splits(data=self.data,
                                    train_data=self.data[:-5], test_data=self.data[-5:])
        user_idx = 0
        candidate_items = np.arange(7)
        prediction_scores = np.arange(7)
        rr.rerank(user_idx, rs.train_set, candidate_items, prediction_scores)
        rr.execute_filters(user_idx=user_idx)
        expected_items_dict = {0: [1, 2, 3, 5, 6]}
        expected_scores_dict = {0: [1, 2, 3, 5, 6]}

        self.assertDictEqual(rr.candidate_items, expected_items_dict)
        user_idx = 4
        rr.rerank(user_idx, rs.train_set, candidate_items, prediction_scores)
        rr.execute_filters(user_idx=user_idx)
        expected_items_dict = {0: [1, 2, 3, 5, 6], 4: [0,2,4,5,6]}
        expected_scores_dict = {0: [1, 2, 3, 5, 6], 4: [0,2,4,5,6]}

        self.assertDictEqual(rr.candidate_items, expected_items_dict)


if __name__ == '__main__':
    unittest.main()
