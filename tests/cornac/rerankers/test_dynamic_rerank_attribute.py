import unittest
import numpy as np
from cornac.eval_methods import RatioSplit

import pandas as pd
import cornac
from cornac.models import UserKNN
from cornac.rerankers.dynamic_attribute_penalization import DynamicAttrReRanker
from cornac.rerankers.user_simulator import UserSimulator
from cornac.datasets import mind as mind
from pandas.testing import assert_frame_equal
import os
from unittest.mock import patch, MagicMock
from itertools import repeat

class TestDynamicAttrReRanker(unittest.TestCase):
    def setUp(self):
        self.dummy_diversity_dimension = [
            "sentiment", "category", "party", "outlet"]
        self.dummy_item_features = pd.DataFrame({
            'sentiment': [0.5, 0.7, -0.1, 0.6, -0.8, 0.4, -0.3, 0.9, -0.5, 0.2,
                          0.3, -0.6, 0.0, 0.8, -0.7, 0.1, 0.75, -0.2, 0.6, -0.4],
            'category': [
                "Politics", "Technology", "Sports", "Health", "Entertainment", "Finance", "Travel",
                "Politics", "Technology", "Science",
                "Environment", "Education", "Sports", "Politics", "Technology", "Health",
                "Entertainment", "Finance", "Travel", "Science"],
            'party': [
                ['Democratic'], ['Republican'], [
                    'Democratic', 'Republican'], ['Green'],
                ['Green', 'Republican'], ['Republican'], [
                    'Democratic', 'Republican'],
                ['Democratic'], ['Republican'], [
                    None],
                ['Democratic'], ['Green'], [None], [
                    'Republican'], ['Green'], [None],
                ['Democratic', 'Republican'], ['Republican'], None, ['Green']],
            'outlet': [
                "CNN", "Fox", "BBC", "Reuters", "NYTimes", "Fox", "BBC",
                "CNN", "Fox", "Reuters",
                "NYTimes", "BBC", "CNN", "Fox", "Reuters", "CNN",
                "BBC", "NYTimes", "Fox", "Reuters"
            ]
        }, index=[i for i in range(20)])

        self.party_category_path = os.path.join(os.path.dirname(
            __file__), 'test_party_category.json')
        self.bin_edges = {'sentiment': [-1, -0.5, 0, 0.5, 1]}
        self.dummy_data = [('u0', 'item0', 1.0),
                           ('u1', 'item1', 1.0),
                           ('u2', 'item0', 1.0),
                           ('u3', 'item2', 1.0),
                           ('u2', 'item3', 1.0),
                           ('u0', 'item4', 1.0),
                           ('u3', 'item5', 1.0),
                           ('u4', 'item6', 1.0),
                           ('u2', 'item7', 1.0),
                           ('u3', 'item8', 1.0),
                           ('u4', 'item9', 1.0),
                           ('u0', 'item9', 1.0),
                           ('u1', 'item10', 1.0),
                           ('u1', 'item11', 1.0),
                           ('u3', 'item12', 1.0),
                           ('u4', 'item13', 1.0),
                           ('u0', 'item14', 1.0),
                           ('u5', 'item15', 1.0),
                           ('u5', 'item16', 1.0),
                           ('u5', 'item17', 1.0),
                           ('u5', 'item18', 1.0),
                           ('u6', 'item19', 1.0),
                           ('u0', 'item6', 1.0),
                           ('u2', 'item3', 1.0),
                           ('u1', 'item2', 1.0),
                           ('u6', 'item1', 1.0),
                           ('u7', 'item19', 1.0),
                           ]
        self.config_file_path = os.path.join(os.path.dirname(
            __file__), 'test_user_simulator_config.ini')
        self.reranker = DynamicAttrReRanker(name="test_reranker",
                                            item_dataframe=self.dummy_item_features, diversity_dimension=self.dummy_diversity_dimension, pool_size=20, top_k=3,
                                            feedback_window_size=2, bin_edges=self.bin_edges, user_choice_model="logarithmic_rank_bias", 
                                            user_simulator_config_path=self.config_file_path, party_category_json_path= self.party_category_path)

        self.ratio_split = RatioSplit.from_splits(data=self.dummy_data,
                                                  train_data=self.dummy_data[:-
                                                                             4], test_data=self.dummy_data[-4:],
                                                  exclude_unknowns=True, verbose=True, seed=123, rating_threshold=1)
        self.expected_user_history = {0: [0, 4, 9, 14], 1: [1, 10, 11], 2: [0, 3, 7], 3: [2, 5, 8, 12],
                                      4: [6, 9, 13], 5: [15, 16, 17, 18], 6: [19], 7: []}

    def test__init__(self):

        self.assertEqual(self.reranker.feedback_window_size, 2)
        self.assertEqual(self.reranker.bin_edges, self.bin_edges)
        self.assertEqual(self.reranker.top_k, 3)
        self.assertEqual(self.reranker.pool_size, 20)
        assert_frame_equal(self.reranker.item_dataframe,
                           self.dummy_item_features)
        self.assertEqual(self.reranker.name, "test_reranker")
        self.assertEqual(self.reranker.diversity_dimension,
                         self.dummy_diversity_dimension)
        self.assertEqual(self.reranker.bin_edges,
                         self.bin_edges)

    def test_initialize_attribute_frequencies(self):
        attribute_frequencies = self.reranker.initialize_attribute_frequencies()
        print("attribute_frequencies:{}".format(attribute_frequencies))
        expected_frequencies = [
            {'sentiment_-1': 0,
                'sentiment_-0.5': 0, 'sentiment_0': 0, 'sentiment_0.5': 0},
            {'category_Politics': 0, 'category_Technology': 0, 'category_Sports': 0, 'category_Health': 0,
             'category_Entertainment': 0, 'category_Finance': 0, 'category_Travel': 0, 'category_Science': 0,
             'category_Environment': 0, 'category_Education': 0},
            {'party_Democratic': 0, 'party_Republican': 0,
                'party_Both': 0, 'party_Minority': 0, 'party_None': 0},
            {'outlet_CNN': 0, 'outlet_Fox': 0, 'outlet_BBC': 0,
                'outlet_Reuters': 0, 'outlet_NYTimes': 0}
        ]

        self.assertEqual(attribute_frequencies, expected_frequencies)

    def test_categorize_party(self):
        self.assertEqual(self.reranker.categorize_party(
            ['Democratic']), 'party_Democratic')
        self.assertEqual(self.reranker.categorize_party(
            ['Republican']), 'party_Republican')
        self.assertEqual(self.reranker.categorize_party(
            ['Democratic', 'Republican']), 'party_Both')
        self.assertEqual(self.reranker.categorize_party(
            ['Green']), 'party_Minority')
        self.assertEqual(self.reranker.categorize_party(
            ['Green', 'Democratic']), 'party_Minority')
        self.assertEqual(self.reranker.categorize_party(
            ['Green', 'Democratic', 'Republican']), 'party_Minority')
        self.assertEqual(self.reranker.categorize_party([]), 'party_None')
        self.assertEqual(self.reranker.categorize_party([None]), 'party_None')

    def test_get_items_by_category(self):
        # diversity_dimension = [
        #     "sentiment", "category", "party", "outlet"]

        categorized_items = self.reranker.get_items_by_category()
        print("categorized_items[0]:{}".format(categorized_items[0]))
        expected_sentiment = {
            'sentiment_-1': [4, 11, 14], 'sentiment_-0.5': [2, 6, 8, 17, 19],
            'sentiment_0': [5, 9, 10, 12, 15], 'sentiment_0.5': [0, 1, 3, 7, 13, 16, 18],
        }

        expected_category = {
            'category_Politics': [0, 7, 13], 'category_Technology': [1, 8, 14],
            'category_Sports': [2, 12], 'category_Health': [3, 15],
            'category_Entertainment': [4, 16], 'category_Finance': [5, 17],
            'category_Travel': [6, 18], 'category_Science': [9, 19],
            'category_Environment': [10], 'category_Education': [11]
        }
        print("categorized_items[2]:{}".format(categorized_items[2]))
        expected_party = {
            'party_Democratic': [0, 7, 10], 'party_Republican': [1, 5, 8, 13, 17],
            'party_Both': [2, 6, 16], 'party_Minority': [3, 4, 11, 14, 19],
            'party_None': [9, 12, 15, 18]
        }
        expected_outlet = {
            'outlet_CNN': [0, 7, 12, 15], 'outlet_Fox': [1, 5, 8, 13, 18],
            'outlet_BBC': [2, 6, 11, 16], 'outlet_Reuters': [3, 9, 14, 19], 'outlet_NYTimes': [4, 10, 17]
        }

        self.assertEqual(categorized_items[0], expected_sentiment)
        self.assertEqual(categorized_items[1], expected_category)
        self.assertEqual(categorized_items[2], expected_party)
        self.assertEqual(categorized_items[3], expected_outlet)

    def assertDictListAlmostEqual(self, list1, list2, delta=1e-6):
        """
        Assert two lists of dictionaries are almost equal, accounting for floating point inaccuracies.
        """
        self.assertEqual(len(list1), len(list2),
                         "Lists do not have the same length")

        for d1, d2 in zip(list1, list2):
            self.assertDictAlmostEqual(d1, d2, delta)

    def assertDictAlmostEqual(self, dict1, dict2, delta):
        """
        Asserts that two dictionaries are almost equal, allowing for a delta in floating point comparisons.
        """
        self.assertEqual(set(dict1.keys()), set(dict2.keys()),
                         "Dictionaries do not have the same keys")
        for key in dict1:
            self.assertAlmostEqual(dict1[key], dict2[key], delta=delta,
                                   msg=f"Values for key '{key}' differ by more than {delta}")

    def test_infer_user_preference(self):
        user_idx = 0
        # self.expected_user_history = {0: [0, 4, 9, 14], 1: [1, 10, 11], 2: [0, 3, 7], 3: [2, 5, 8, 12],
        #   4: [6, 9, 13], 5: [15, 16, 17, 18], 6: [19], 7: []}
        self.reranker.user_history = self.expected_user_history
        preference_user_0 = self.reranker.infer_user_preference(user_idx)
        expected_preference_user_0 = [{'sentiment_-1': 0.5, 'sentiment_-0.5': 0.0, 'sentiment_0': 0.25, 'sentiment_0.5': 0.25},
                                      {'category_Politics': 0.25, 'category_Technology': 0.25, 'category_Sports': 0.0, 'category_Health': 0.0, 'category_Entertainment': 0.25,
                                          'category_Finance': 0.0, 'category_Travel': 0.0, 'category_Science': 0.25, 'category_Environment': 0.0, 'category_Education': 0.0},
                                      {'party_Democratic': 0.25, 'party_Republican': 0.0,
                                          'party_Both': 0.0, 'party_Minority': 0.5, 'party_None': 0.25},
                                      {'outlet_CNN': 0.25, 'outlet_Fox': 0.0, 'outlet_BBC': 0.0, 'outlet_Reuters': 0.5, 'outlet_NYTimes': 0.25}]
        self.assertDictListAlmostEqual(
            preference_user_0, expected_preference_user_0)
        user_idx = 1
        preference_user_1 = self.reranker.infer_user_preference(user_idx)
        expected_preference_user_1 = [{'sentiment_-1': 0.3333333333333333, 'sentiment_-0.5': 0.0, 'sentiment_0': 0.3333333333333333, 'sentiment_0.5': 0.3333333333333333},
                                      {'category_Politics': 0.0, 'category_Technology': 0.3333333333333333, 'category_Sports': 0.0, 'category_Health': 0.0, 'category_Entertainment': 0.0,
                                          'category_Finance': 0.0, 'category_Travel': 0.0, 'category_Science': 0.0, 'category_Environment': 0.3333333333333333, 'category_Education': 0.3333333333333333},
                                      {'party_Democratic': 0.3333333333333333, 'party_Republican': 0.3333333333333333,
                                          'party_Both': 0.0, 'party_Minority': 0.3333333333333333, 'party_None': 0.0},
                                      {'outlet_CNN': 0.0, 'outlet_Fox': 0.3333333333333333, 'outlet_BBC': 0.3333333333333333, 'outlet_Reuters': 0.0, 'outlet_NYTimes': 0.3333333333333333}]
        self.assertDictListAlmostEqual(
            preference_user_1, expected_preference_user_1)
        user_idx = 2
        preference_user_2 = self.reranker.infer_user_preference(user_idx)
        expected_preference_user_2 = [{'sentiment_-1': 0.0, 'sentiment_-0.5': 0.0, 'sentiment_0': 0.0, 'sentiment_0.5': 1.0},
                                      {'category_Politics': 0.6666666666666666, 'category_Technology': 0.0, 'category_Sports': 0.0, 'category_Health': 0.3333333333333333,
                                          'category_Entertainment': 0.0, 'category_Finance': 0.0, 'category_Travel': 0.0, 'category_Science': 0.0, 'category_Environment': 0.0, 'category_Education': 0.0},
                                      {'party_Democratic': 0.6666666666666666, 'party_Republican': 0.0,
                                          'party_Both': 0.0, 'party_Minority': 0.3333333333333333, 'party_None': 0.0},
                                      {'outlet_CNN': 0.6666666666666666, 'outlet_Fox': 0.0, 'outlet_BBC': 0.0, 'outlet_Reuters': 0.3333333333333333, 'outlet_NYTimes': 0.0}]
        self.assertDictListAlmostEqual(
            preference_user_2, expected_preference_user_2)

        user_idx = 3
        preference_user_3 = self.reranker.infer_user_preference(user_idx)

        expected_preference_user_3 = [{'sentiment_-1': 0.0, 'sentiment_-0.5': 0.5, 'sentiment_0': 0.5, 'sentiment_0.5': 0.0},
                                      {'category_Politics': 0.0, 'category_Technology': 0.25, 'category_Sports': 0.5, 'category_Health': 0.0, 'category_Entertainment': 0.0, 'category_Finance': 0.25,
                                       'category_Travel': 0.0, 'category_Science': 0.0, 'category_Environment': 0.0, 'category_Education': 0.0},
                                      {'party_Democratic': 0.0, 'party_Republican': 0.5,
                                       'party_Both': 0.25, 'party_Minority': 0.0, 'party_None': 0.25},
                                      {'outlet_CNN': 0.25, 'outlet_Fox': 0.5, 'outlet_BBC': 0.25, 'outlet_Reuters': 0.0, 'outlet_NYTimes': 0.0}]
        self.assertDictListAlmostEqual(
            preference_user_3, expected_preference_user_3)

        user_idx = 4
        preference_user_4 = self.reranker.infer_user_preference(user_idx)
        expected_preference_user_4 = [{'sentiment_-1': 0.0, 'sentiment_-0.5': 0.3333333333333333, 'sentiment_0': 0.3333333333333333, 'sentiment_0.5': 0.3333333333333333},
                                      {'category_Politics': 0.3333333333333333, 'category_Technology': 0.0, 'category_Sports': 0.0, 'category_Health': 0.0, 'category_Entertainment': 0.0, 'category_Finance': 0.0, 'category_Travel': 0.3333333333333333,
                                       'category_Science': 0.3333333333333333, 'category_Environment': 0.0, 'category_Education': 0.0},
                                      {'party_Democratic': 0.0, 'party_Republican': 0.3333333333333333,
                                          'party_Both': 0.3333333333333333, 'party_Minority': 0.0, 'party_None': 0.3333333333333333},
                                      {'outlet_CNN': 0.0, 'outlet_Fox': 0.3333333333333333, 'outlet_BBC': 0.3333333333333333, 'outlet_Reuters': 0.3333333333333333, 'outlet_NYTimes': 0.0}]
        self.assertDictListAlmostEqual(
            preference_user_4, expected_preference_user_4)

        user_idx = 5
        preference_user_5 = self.reranker.infer_user_preference(user_idx)
        expected_preference_user_5 = [{'sentiment_-1': 0.0, 'sentiment_-0.5': 0.25, 'sentiment_0': 0.25, 'sentiment_0.5': 0.5},  {'category_Politics': 0.0, 'category_Technology': 0.0, 'category_Sports': 0.0, 'category_Health': 0.25, 'category_Entertainment': 0.25, 'category_Finance': 0.25, 'category_Travel': 0.25,
                                                                                                                                  'category_Science': 0.0, 'category_Environment': 0.0, 'category_Education': 0.0},
                                      {'party_Democratic': 0.0, 'party_Republican': 0.25,
                                          'party_Both': 0.25, 'party_Minority': 0.0, 'party_None': 0.5},
                                      {'outlet_CNN': 0.25, 'outlet_Fox': 0.25, 'outlet_BBC': 0.25, 'outlet_Reuters': 0.0, 'outlet_NYTimes': 0.25}]
        self.assertDictListAlmostEqual(
            preference_user_5, expected_preference_user_5)

        user_idx = 6
        preference_user_6 = self.reranker.infer_user_preference(user_idx)
        expected_preference_user_6 = [{'sentiment_-1': 0.0, 'sentiment_-0.5': 1.0, 'sentiment_0': 0.0, 'sentiment_0.5': 0.0},
                                      {'category_Politics': 0.0, 'category_Technology': 0.0, 'category_Sports': 0.0, 'category_Health': 0.0, 'category_Entertainment': 0.0, 'category_Finance': 0.0, 'category_Travel': 0.0,
                                       'category_Science': 1.0, 'category_Environment': 0.0, 'category_Education': 0.0},
                                      {'party_Democratic': 0.0, 'party_Republican': 0.0,
                                       'party_Both': 0.0, 'party_Minority': 1.0, 'party_None': 0.0},
                                      {'outlet_CNN': 0.0, 'outlet_Fox': 0.0, 'outlet_BBC': 0.0, 'outlet_Reuters': 1.0, 'outlet_NYTimes': 0.0}]
        self.assertDictListAlmostEqual(
            preference_user_6, expected_preference_user_6)

        user_idx = 7
        preference_user_7 = self.reranker.infer_user_preference(user_idx)
        expected_preference_user_7 = [{'sentiment_-1': 0.0, 'sentiment_-0.5':  0.0, 'sentiment_0': 0.0, 'sentiment_0.5': 0.0},
                                      {'category_Politics': 0.0, 'category_Technology': 0.0, 'category_Sports': 0.0, 'category_Health': 0.0, 'category_Entertainment': 0.0, 'category_Finance': 0.0, 'category_Travel': 0.0,
                                       'category_Science':  0.0, 'category_Environment': 0.0, 'category_Education': 0.0},
                                      {'party_Democratic': 0.0, 'party_Republican': 0.0,
                                       'party_Both': 0.0, 'party_Minority':  0.0, 'party_None': 0.0},
                                      {'outlet_CNN': 0.0, 'outlet_Fox': 0.0, 'outlet_BBC': 0.0, 'outlet_Reuters':  0.0, 'outlet_NYTimes': 0.0}]
        self.assertDictListAlmostEqual(
            preference_user_7, expected_preference_user_7)

    def test_update_frequencies(self):
        clicked_items = [0, 9, 19]  # Simulating clicked items
        self.reranker.update_frequencies(clicked_items)

        expected_frequencies = [
            {'sentiment_-1': 0, 'sentiment_-0.5': 1,
                'sentiment_0': 1, 'sentiment_0.5': 1},
            {'category_Politics': 1, 'category_Technology': 0, 'category_Sports': 0, 'category_Health': 0,
             'category_Entertainment': 0, 'category_Finance': 0, 'category_Travel': 0, 'category_Science': 2,
             'category_Environment': 0, 'category_Education': 0},
            {'party_Democratic': 1, 'party_Republican': 0,
                'party_Both': 0, 'party_Minority': 1, 'party_None': 1},
            {'outlet_CNN': 1, 'outlet_Fox': 0, 'outlet_BBC': 0,
                'outlet_Reuters': 2, 'outlet_NYTimes': 0}
        ]

        self.assertEqual(self.reranker.attribute_frequencies,
                         expected_frequencies)

    def test_calculate_penalization(self):
        self.reranker.attribute_frequencies = [
            {'category_Politics': 5, 'category_Technology': 3,
                'category_Sports': 2, 'category_Education': 0}
        ]

        penalizations = self.reranker.calculate_penalization()

        expected_penalization = [
            {'category_Politics': 1.0, 'category_Technology': 0.6,
                'category_Sports': 0.4, 'category_Education': 0}
        ]

        self.assertEqual(penalizations, expected_penalization)
        self.reranker.attribute_frequencies = [
            {'category_Politics': 5, 'category_Technology': 3,
                'category_Sports': 2, 'category_Education': 0},
            {'outlet_CNN': 1, 'outlet_Fox': 0, 'outlet_BBC': 0,
                'outlet_Reuters': 2, 'outlet_NYTimes': 0}
        ]

        penalizations = self.reranker.calculate_penalization()

        expected_penalization = [
            {'category_Politics': 1.0, 'category_Technology': 0.6,
                'category_Sports': 0.4, 'category_Education': 0},
            {'outlet_CNN': 1/2, 'outlet_Fox': 0, 'outlet_BBC': 0,
                'outlet_Reuters': 1.0, 'outlet_NYTimes': 0
             }
        ]

        self.assertEqual(penalizations, expected_penalization)

        # zero frequencies
        self.reranker.attribute_frequencies = [
            {'category_Politics': 0, 'category_Technology': 0,
             'category_Sports': 0, 'category_Education': 0},
            {'outlet_CNN': 0, 'outlet_Fox': 0, 'outlet_BBC': 0,
             'outlet_Reuters': 0, 'outlet_NYTimes': 0}
        ]

        penalizations = self.reranker.calculate_penalization()

        expected_penalization = [
            {'category_Politics': 0, 'category_Technology': 0,
                'category_Sports': 0, 'category_Education': 0},
            {'outlet_CNN': 0, 'outlet_Fox': 0, 'outlet_BBC': 0,
                'outlet_Reuters': 0, 'outlet_NYTimes': 0
             }
        ]

        self.assertEqual(penalizations, expected_penalization)

    def assertListAlmostEqual(self, list1, list2, places=5):
        """
        Custom assertion method to check that two lists are almost equal.

        Parameters:
            list1 (list of float): The first list to compare.
            list2 (list of float): The second list to compare.
            places (int): Number of decimal places to consider.
        """
        self.assertEqual(len(list1), len(list2),
                         "The lists do not have the same length.")
        for a, b in zip(list1, list2):
            print(" expected:{}, actual:{}".format(a, b))
            self.assertAlmostEqual(a, b, places=places)

    def test_diversity_score(self):
        category_dimension = ["category"]
        category_reranker = DynamicAttrReRanker(name="test_reranker",
                                                item_dataframe=self.dummy_item_features, diversity_dimension=category_dimension, pool_size=20, top_k=3,
                                                feedback_window_size=2, bin_edges=self.bin_edges, user_choice_model="exponential", user_simulator_config_path=self.config_file_path)
        candidate_items = [0, 1, 2,]
        penalizations = [{'category_Politics': 1.0,
                          'category_Technology': 0.6, 'category_Sports': 0.4}]

        category_reranker.categorized_items = [
            {'category_Politics': [0], 'category_Technology': [
                1], 'category_Sports': [2]}
        ]

        expected_scores = np.array([[0.0], [0.4], [0.6]])
        
        scores = category_reranker.diversityScore(
            candidate_items, penalizations)
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-5)

        # self.assertListAlmostEqual(scores, [0.0, 0.4, 0.6])

        category_outlet_dimension = ["category", "outlet"]
        category_outlet_reranker = DynamicAttrReRanker(name="test_reranker",
                                                       item_dataframe=self.dummy_item_features, diversity_dimension=category_outlet_dimension, pool_size=20, top_k=3,
                                                       feedback_window_size=2, bin_edges=self.bin_edges, user_choice_model="exponential", user_simulator_config_path=self.config_file_path)
        candidate_items = [8, 6, 15, 1]
        penalizations = [{'category_Politics': 1.0,
                          'category_Technology': 0.7, 'category_Sports': 0}, {'outlet_BBC': 0.3, 'outlet_CNN': 1.0}]

        category_outlet_reranker.categorized_items = [
            {'category_Politics': [8], 'category_Technology': [
                6], 'category_Sports': [15, 1]}, {'outlet_BBC': [8, 15], 'outlet_CNN': [
                    6, 1]}
        ]

        expected_scores = np.array([
        [0.0, 0.7],   # item 8: Politics (1.0 → 0), BBC (0.3 → 0.7)
        [0.3, 0.0],   # item 6: Technology (0.7 → 0.3), CNN (1.0 → 0)
        [1.0, 0.7],   # item 15: Sports (0.0 → 1.0), BBC (0.3 → 0.7)
        [1.0, 0.0],   # item 1: Sports (0.0 → 1.0), CNN (1.0 → 0)
        ])
        scores = category_outlet_reranker.diversityScore(
            candidate_items, penalizations)

        np.testing.assert_allclose(scores, expected_scores, rtol=1e-5)
        # self.assertListAlmostEqual(scores, [0.7, 0.3, 1.7, 1])

    @patch('cornac.rerankers.user_simulator.UserSimulator')
    def test_add_user(self, mock_user_simulator):
        mock_user = MagicMock()
        mock_user.history = []
        test_user_id = 1
        mock_user.user_id = test_user_id
        mock_user_simulator.return_value = mock_user
        self.reranker.user_choice_model = 'preference_based_bias'
        self.reranker.user_history = {test_user_id: []}
        self.reranker.add_user(test_user_id)
        self.assertEqual(self.reranker.users[test_user_id].history, [])
        self.assertEqual(
            self.reranker.users[test_user_id].choice_model, 'preference_based_bias')

    @patch('cornac.rerankers.user_simulator.UserSimulator')
    @patch('numpy.random.rand') # because the dynamic re-ranking introduces random logic
    def test_filter_and_update_recommendations(self,mock_rand,  mock_user_simulator):
        mock_user = MagicMock()
        mock_user.seen_items = [0, 9, 4, 5, 2]
        mock_user.interacted_items = [[9], [4, 5]]
        mock_user.history = []
        mock_user_simulator.return_value = mock_user
        user_idx = 1
        self.reranker.users = {user_idx: mock_user}
        self.reranker.candidate_items = {user_idx: [0, 2, 8, 10, 19, 7]}

        clicked_items = [4, 5]
        expected_candidate_items_after_filtering = [8, 10, 19, 7]
        self.reranker.filter_seen_items(user_idx)
        self.assertEqual(
            self.reranker.candidate_items[user_idx], expected_candidate_items_after_filtering)

        # mock_rand.side_effect = [0.01, 0.01, 0.01, 0.01]
        mock_rand.side_effect = repeat(0.01) 
        top_k_items = self.reranker.update_recommendations(
            user_idx, clicked_items)
        expected_frequency = [
            {'sentiment_-1': 1, 'sentiment_-0.5': 0,
                'sentiment_0': 1, 'sentiment_0.5': 0},
            {'category_Politics': 0, 'category_Technology': 0, 'category_Sports': 0, 'category_Health': 0,
             'category_Entertainment': 1, 'category_Finance': 1, 'category_Travel': 0, 'category_Science': 0,
             'category_Environment': 0, 'category_Education': 0},
            {'party_Democratic': 0, 'party_Republican': 1,
                'party_Both': 0, 'party_Minority': 1, 'party_None': 0},
            {'outlet_CNN': 0, 'outlet_Fox': 1, 'outlet_BBC': 0,
                'outlet_Reuters': 0, 'outlet_NYTimes': 1}
        ]
        self.assertEqual(self.reranker.attribute_frequencies,
                         expected_frequency)
        
        # possible_lists = [[7, 19, 8], [7, 19, 10]]
        # self.assertIn(top_k_items, possible_lists)
        expected_items = {7, 19, 8, 10}
        top_k_set = set(top_k_items)

        # Ensure all top_k_items are from expected_items
        self.assertTrue(top_k_set.issubset(expected_items))

        # Optionally also check that we got exactly top_k items
        self.assertEqual(len(top_k_items), self.reranker.top_k)

        interacted_items_flat = [
            item for sublist in self.reranker.users[user_idx].interacted_items for item in sublist]
        top_k_items = self.reranker.update_recommendations(
            user_idx, interacted_items_flat)
        possible_lists = [[7, 8, 10], [7, 10, 8]]
        self.assertIn(top_k_items, possible_lists)

    @patch('cornac.rerankers.user_simulator.UserSimulator')
    @patch('cornac.rerankers.dynamic_attribute_penalization.DynamicAttrReRanker.add_user')
    def test_rerank(self, mock_add_user, mock_user_simulator):
        mock_user = MagicMock()
        mock_user.simulate_interaction.side_effect = [[0, 1], [3, 4]]
        mock_user.max_iteration = 2  # The simulation will run for 2 iterations
        mock_user.history = [2, 5, 8, 12]
        mock_user.clicked_each_iteration = 2
        mock_user.seen_items = [0, 1, 6, 9, 10, 11, 13, 14, 15, 7, 3, 4, 19]
        mock_user.interacted_items = [[0, 1], [3, 4]]
        mock_user_simulator.return_value = mock_user
        user_idx = 3
        mock_add_user.side_effect = lambda user_id: self.reranker.users.update({
            user_id: mock_user
        })
        self.reranker.candidate_items = {user_idx: [i for i in range(20)]}

        recommendation_each_iteration = self.reranker.rerank(
            user_idx=user_idx,
            interaction_history=self.ratio_split.train_set,
            candidate_items=self.reranker.candidate_items[user_idx],
            recommendation_list=self.reranker.candidate_items[user_idx]
        )
        self.assertEqual(list(recommendation_each_iteration.keys()), [0, 1])
        self.assertEqual(
            self.reranker.users[user_idx].interacted_items, [[0, 1], [3, 4]])
        # seen: 0, 1, 9, 10, 7, 3, 4, 19; history:2, 5, 8, 12
        self.assertEqual(self.reranker.candidate_items[user_idx], [
                         16, 17, 18])
        # self.assertIn(recommendation_each_iteration[0], [
        #               [17, 16, 18], [17, 18, 16]])
        # the feedback window size is 2

    @patch('cornac.rerankers.user_simulator.UserSimulator')
    @patch('cornac.rerankers.dynamic_attribute_penalization.DynamicAttrReRanker.add_user')
    def test_rerank_candidate_not_enough(self, mock_add_user, mock_user_simulator):
        mock_user = MagicMock()
        mock_user.simulate_interaction.side_effect = [[0, 1], [3, 4]]
        mock_user.max_iteration = 2
        mock_user.history = [2, 5, 8, 12]
        mock_user.clicked_each_iteration = 2
        mock_user.seen_items = [0, 1, 6, 9, 10, 11, 13, 14, 15, 7, 3, 4, 19]
        mock_user.interacted_items = [[0, 1], [3, 4]]
        mock_user_simulator.return_value = mock_user
        user_idx = 3
        mock_add_user.side_effect = lambda user_id: self.reranker.users.update({
            user_id: mock_user
        })
        # candidate_items: only item 3 are not in the user history, and will not be filtered.
        # only item3 a valid item, candidate  is less than mock_user.clicked_each_iteration
        self.reranker.candidate_items = {user_idx: [2, 3, 5]}

        recommendation_each_iteration = self.reranker.rerank(
            user_idx=user_idx,
            interaction_history=self.ratio_split.train_set,
            candidate_items=self.reranker.candidate_items[user_idx],
            recommendation_list=self.reranker.candidate_items[user_idx]
        )
        self.assertEqual(recommendation_each_iteration, {})

        # 0,1 are in the user.seen_items
        self.reranker.candidate_items = {user_idx: [0, 1, 2, 3, 5]}
        recommendation_each_iteration = self.reranker.rerank(
            user_idx=user_idx,
            interaction_history=self.ratio_split.train_set,
            candidate_items=self.reranker.candidate_items[user_idx],
            recommendation_list=self.reranker.candidate_items[user_idx]
        )
        self.assertEqual(list(recommendation_each_iteration.keys()), [0])


if __name__ == '__main__':
    unittest.main()
