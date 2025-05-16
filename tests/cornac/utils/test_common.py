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
import numpy.testing as npt
import scipy.sparse as sp

from cornac.utils.common import sigmoid
from cornac.utils.common import safe_indexing
from cornac.utils.common import validate_format
from cornac.utils.common import scale
from cornac.utils.common import clip
from cornac.utils.common import excepts
from cornac.utils.common import intersects
from cornac.utils.common import estimate_batches
from cornac.utils.common import get_rng
from cornac.utils.common import normalize
from cornac.utils.common import TargetDistributionMatcher
from cornac.utils.common import find_items_in_range
from cornac.utils.common import find_items_match
from cornac.utils.common import safe_kl_divergence
from cornac.utils.common import count_selected_in_aspects
from cornac.utils.common import get_max_keys
import pandas as pd
import math
from cornac.datasets import mind
import json
import ast


class TestCommon(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(0, sigmoid(-np.inf))
        self.assertEqual(0.5, sigmoid(0))
        self.assertEqual(1, sigmoid(np.inf))

        self.assertGreater(0.5, sigmoid(-0.1))
        self.assertGreater(sigmoid(0.1), 0.5)

    def test_scale(self):
        self.assertEqual(1, scale(0, 1, 5, 0, 1))
        self.assertEqual(3, scale(0.5, 1, 5, 0, 1))
        self.assertEqual(5, scale(1, 1, 5, 0, 1))
        self.assertEqual(1, scale(5, 0, 1, 5, 5))

        npt.assert_array_equal(scale(np.asarray([0, 0.25, 0.5, 0.75, 1]), 1, 5),
                               np.asarray([1, 2, 3, 4, 5]))

    def test_clip(self):
        self.assertEqual(1, clip(0, 1, 5))
        self.assertEqual(3, clip(3, 1, 5))
        self.assertEqual(5, clip(6, 1, 5))

        npt.assert_array_equal(clip(np.asarray([0, 3, 6]), 1, 5),
                               np.asarray([1, 3, 5]))

    def test_intersects(self):
        self.assertEqual(0, len(intersects(np.asarray([1]), np.asarray(2))))
        self.assertEqual(1, len(intersects(np.asarray([2]), np.asarray(2))))

        npt.assert_array_equal(intersects(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                               np.asarray([2, 1]))

    def test_excepts(self):
        self.assertEqual(1, len(excepts(np.asarray([1]), np.asarray(2))))
        self.assertEqual(0, len(excepts(np.asarray([2]), np.asarray(2))))

        npt.assert_array_equal(excepts(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                               np.asarray([3]))

    def test_safe_indexing(self):
        npt.assert_array_equal(safe_indexing(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                               np.asarray([2, 1]))
        npt.assert_array_equal(safe_indexing(np.asarray([3, 2, 1]), [1, 2]),
                               np.asarray([2, 1]))
        self.assertListEqual(safe_indexing([3, 2, 1], [1, 2]), [2, 1])

    def test_validate_format(self):
        self.assertEqual('UIR', validate_format('UIR', ['UIR']))
        self.assertEqual('UIRT', validate_format('UIRT', ['UIRT']))

        try:
            validate_format('iur', ['UIR'])
        except ValueError:
            assert True

    def test_estimate_batches(self):
        self.assertEqual(estimate_batches(3, 2), 2)
        self.assertEqual(estimate_batches(4, 2), 2)
        self.assertEqual(estimate_batches(1, 2), 1)

    def test_get_rng(self):
        try:
            get_rng('a')
        except ValueError:
            assert True

    def test_normalize(self):
        """
        X = array([[1., 0., 2.],
                   [0., 0., 3.],
                   [4., 5., 6.]])
        """
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1., 2., 3., 4., 5., 6.], dtype=np.float64)
        X = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
        # XA = X.A
        XA = X.toarray() 

        # normalizing rows (axis=1)
        X_l1 = XA / (np.abs(XA).sum(1).reshape(-1, 1))
        X_l2 = XA / (np.sqrt((XA ** 2).sum(1)).reshape(-1, 1))
        X_max = XA / (np.max(XA, axis=1).reshape(-1, 1))
        # sparse input
        # npt.assert_array_equal(X_l1, normalize(X, 'l1', axis=1, copy=True).A)
        # npt.assert_array_equal(X_l2, normalize(X, 'l2', axis=1, copy=True).A)
        # npt.assert_array_equal(X_max, normalize(X, 'max', axis=1, copy=True).A)
        npt.assert_array_equal(X_l1, normalize(X, 'l1', axis=1, copy=True).toarray() )
        npt.assert_array_equal(X_l2, normalize(X, 'l2', axis=1, copy=True).toarray())
        npt.assert_array_equal(X_max, normalize(X, 'max', axis=1, copy=True).toarray())
        # dense input
        npt.assert_array_equal(X_l1, normalize(XA, 'l1', axis=1, copy=True))
        npt.assert_array_equal(X_l2, normalize(XA, 'l2', axis=1, copy=True))
        npt.assert_array_equal(X_max, normalize(XA, 'max', axis=1, copy=True))

        # normalizing columns (axis=0)
        X_l1 = XA / (np.abs(XA).sum(0).reshape(1, -1))
        X_l2 = XA / (np.sqrt((XA ** 2).sum(0)).reshape(1, -1))
        X_max = XA / (np.max(XA, axis=0).reshape(1, -1))
        # sparse input
        # npt.assert_array_equal(X_l1, normalize(X, 'l1', axis=0, copy=True).A)
        # npt.assert_array_equal(X_l2, normalize(X, 'l2', axis=0, copy=True).A)
        # npt.assert_array_equal(X_max, normalize(X, 'max', axis=0, copy=True).A)
        npt.assert_array_equal(X_l1, normalize(X, 'l1', axis=0, copy=True).toarray())
        npt.assert_array_equal(X_l2, normalize(X, 'l2', axis=0, copy=True).toarray())
        npt.assert_array_equal(X_max, normalize(X, 'max', axis=0, copy=True).toarray())
        # dense input
        npt.assert_array_equal(X_l1, normalize(XA, 'l1', axis=0, copy=True))
        npt.assert_array_equal(X_l2, normalize(XA, 'l2', axis=0, copy=True))
        npt.assert_array_equal(X_max, normalize(XA, 'max', axis=0, copy=True))

        # check valid norm type
        try:
            normalize(X, norm='bla bla')
        except ValueError:
            assert True

        # check valid input shape
        try:
            normalize(XA[:, np.newaxis])
        except ValueError:
            assert True

        # copy=True, sparse
        normalized_X = normalize(X, copy=True)
        self.assertFalse(np.allclose(X.data, normalized_X.data))

        # copy=True, dense
        normalized_XA = normalize(XA, copy=True)
        self.assertFalse(np.allclose(XA, normalized_XA))

        # copy=False, sparse
        original = X.data.copy()
        normalized_X = normalize(X, copy=False)
        self.assertFalse(np.allclose(original, X.data))
        npt.assert_array_equal(normalized_X.data, X.data)

        # copy=False, dense
        original = XA.copy()
        normalized_XA = normalize(XA, copy=False)
        self.assertFalse(np.allclose(original, XA))
        npt.assert_array_equal(normalized_XA, XA)

    #######################################
    # def test_normalize_distribution(self):
    #     # Test normalization on a regular dictionary with non-zero sum.
    #     input_dict = {'a': 10, 'b': 20, 'c': 30}
    #     expected_output = {'a': 0.16666666666666666,
    #                        'b': 0.3333333333333333, 'c': 0.5}
    #     self.assertEqual(normalize_distribution(input_dict), expected_output)

    #     # Test the function with a dictionary where all values sum to zero.
    #     input_dict = {'a': 0, 'b': 0, 'c': 0}
    #     expected_output = {'a': 0, 'b': 0, 'c': 0}
    #     self.assertEqual(normalize_distribution(input_dict), expected_output)

    #     # Test the function with an empty dictionary.
    #     input_dict = {}
    #     expected_output = {}
    #     self.assertEqual(normalize_distribution(input_dict), expected_output)

    #     # Test the function with negative values.
    #     input_dict = {'a': -10, 'b': -20, 'c': -30}
    #     with self.assertRaises(ValueError):
    #         normalize_distribution(input_dict)

    #     # Ensure non-numeric values raise an error.
    #     input_dict = {'a': 'string', 'b': None, 'c': [1, 2, 3]}
    #     with self.assertRaises(ValueError):
    #         normalize_distribution(input_dict)

    def test_find_items(self):
        df = pd.DataFrame({

            'category': ['Politics', 'Technology', 'Economics', 'Politics'],
            'age': [
                10, 24, 49, 18
            ],
            'sentiment': [0.5, -0.3, 0.8, -0.6],
            'popularity': [15, 18, 20, 21]
        })
        # find_items_in_range
        result = find_items_in_range(df, 'sentiment', -0.5, 1.0)
        self.assertEqual(result, [0, 1, 2])
        result = find_items_in_range(df, 'popularity', 18, 22)
        self.assertEqual(result, [1, 2, 3])
        result = find_items_in_range(df, 'age', 60, 80)
        self.assertEqual(result, [])

        # find_items_match
        result = find_items_match(df, 'category', 'Politics')
        self.assertEqual(result, [0, 3])
        result = find_items_match(df, 'sentiment', 0.5)
        self.assertEqual(result, [0])
        result = find_items_match(df, 'category', 'Sports')
        self.assertEqual(result, [])

    def test_get_max_keys(self):
        # Test with a single key having the maximum value.
        test_dict = {'a': 1, 'b': 2, 'c': 3}
        self.assertEqual(get_max_keys(test_dict), ['c'])
        # Test with multiple keys having the same maximum value.
        test_dict = {'a': 3, 'b': 2, 'c': 3}
        result = get_max_keys(test_dict)
        # assertCountEqual ignores order
        self.assertCountEqual(result, ['a', 'c'])
        # Test with all keys having the same value.
        test_dict = {'a': 1, 'b': 1, 'c': 1}
        result = get_max_keys(test_dict)
        self.assertCountEqual(result, ['a', 'b', 'c'])
        # Test with an empty dictionary.
        test_dict = {}
        self.assertEqual(get_max_keys(test_dict), [])
        # string input
        test_dict = {'a': 'apple', 'b': 'banana', 'c': 'apple'}
        with self.assertRaises(ValueError):
            get_max_keys(test_dict)

    def test_safe_kl_divergence(self):
        # Case 1: Identical distributions
        p = {'a': 0.5, 'b': 0.5}
        q = {'a': 0.5, 'b': 0.5}
        alpha = 0.01
        result = safe_kl_divergence(
            np.array(list(p.values())),
            np.array(list(q.values())),
            alpha
        )
        self.assertAlmostEqual(result, 0.0, places=5)

        # Case 2: Mismatched distributions with zero values in q
        p = {'a': 0.7, 'b': 0.3}
        q = {'a': 0.0, 'b': 1.0}
        alpha = 0.1
        result = safe_kl_divergence(
            np.array(list(p.values())),
            np.array(list(q.values())),
            alpha
        )

        # Manually calculate expected value
        expected_q_a = alpha * 0.7
        expected_q_b = 0.9 * 1 + 0.1 * 0.3
        expected_kl_div = 0.7 * math.log(0.7 / expected_q_a) + \
            0.3 * math.log(0.3 / expected_q_b)
        self.assertAlmostEqual(result, expected_kl_div, places=5)

        # Case 3: Distribution with one element zero in p
        p = {'a': 1.0, 'b': 0.0}
        q = {'a': 0.9, 'b': 0.1}
        alpha = 0.05
        result = safe_kl_divergence(
            np.array(list(p.values())),
            np.array(list(q.values())),
            alpha
        )

        # Expect KL divergence to only include non-zero terms from p
        adjusted_q_a = 0.95 * 0.9 + 0.05 * 1.0
        expected_kl_div = 1.0 * math.log(1.0 / adjusted_q_a)
        self.assertAlmostEqual(result, expected_kl_div, places=5)

    def test_target_distribution_matcher(self):
        dummy_item_features = pd.DataFrame({
            'feature1': [0.1, 0.3, 0.2, 0.5, 0.8, 0.6, 0.7],
            'feature2': [0.6, 0.8, 0.7, 0.2, 0.2, 0.1, 0.05],
            'feature3': [0.5, 0.2, 0.9, 0.4, 0.1, 0.4, 0.3]
        }, index=[0, 1, 2, 3, 4, 5, 6])

        target_distr_dummy = {
            "feature1": {
                "type": "continuous",
                "distr": [
                    {
                        "min": 0,
                        "max": 0.3,
                        "prob": 0.2
                    },
                    {
                        "min": 0.3,
                        "max": 0.6,
                        "prob": 0.3
                    },
                    {
                        "min": 0.6,
                        "max": 1.0,
                        "prob": 0.5
                    }
                ]
            },
            "feature2": {
                "type": "continuous",
                "distr": [
                    {
                        "min": 0,
                        "max": 0.3,
                        "prob": 0.3
                    },
                    {
                        "min": 0.3,
                        "max": 0.6,
                        "prob": 0.4
                    },
                    {
                        "min": 0.6,
                        "max": 1.0,
                        "prob": 0.3
                    }
                ]
            }
        }
        targetDimension = ["feature1", "feature2"]
        selected_distribution = [target_distr_dummy[i]
                                 for i in targetDimension]
        candidate_items = [0, 1, 2, 3, 4, 5, 6]
        proportions, matched_items = TargetDistributionMatcher(
            selected_distribution, targetDimension, dummy_item_features,  candidate_items)
        expected_matched_items_feature1 = {'feature1,0,0.3': [0, 2],
                                           # Items at indices 1 and 3 have feature1 values in the range [0.3, 0.6)
                                           'feature1,0.3,0.6': [1, 3],
                                           # Items at indices 4, 5, and 6 have feature1 values in the range [0.6, 1.0)
                                           'feature1,0.6,1.0': [4, 5, 6]}

        actual_dict = matched_items[0]
        for key, expected_value in expected_matched_items_feature1.items():
            # Check if the key exists in the actual dictionary
            if key in actual_dict:
                # Assert that the value associated with the key is the same in both dictionaries
                self.assertEqual(
                    actual_dict[key], expected_value, f"Value mismatch for key: {key}")
            else:
                # Assert an error if the key is not found
                self.fail(
                    f"Key {key} expected but not found in actual dictionary.")

        expected_matched_items_feature2 = {
            'feature2,0,0.3': [3, 4, 5, 6],  # feature2 values
            'feature2,0.3,0.6': [],
            'feature2,0.6,1.0': [0, 1, 2]}
        actual_items = matched_items[1]
        for key, expected_value in expected_matched_items_feature2.items():
            # Check if the key exists in the actual dictionary
            if key in actual_items:
                # Assert that the value associated with the key is the same in both dictionaries
                self.assertEqual(
                    actual_items[key], expected_value, f"Value mismatch for key: {key}")
            else:
                # Assert an error if the key is not found
                self.fail(
                    f"Key {key} expected but not found in actual dictionary.")
        expected_proportions = {
            'feature2,0,0.3': 0.3,  # feature2 values
            'feature2,0.3,0.6': 0.4,
            'feature2,0.6,1.0': 0.3}
        for key, expected_value in expected_proportions.items():
            # Check if the key exists in the actual dictionary
            if key in proportions[1]:
                # Assert that the value associated with the key is the same in both dictionaries
                self.assertEqual(
                    proportions[1][key], expected_value, f"Value mismatch for key: {key}")
            else:
                # Assert an error if the key is not found
                self.fail(
                    f"Key {key} expected but not found in actual dictionary.")

    def test_count_selected_in_aspects(self):
        # Test case where all selected items are in the aspects.
        selected_items = [1, 2, 3, 4]
        aspect_dictionaries = [
            {'sports': [1, 2, 3], 'tech': [3, 4]},
            {'entertainment': [1, 4], 'education': [2]}
        ]
        expected = [
            {'sports': 3, 'tech': 2},
            {'entertainment': 2, 'education': 1}
        ]
        result = count_selected_in_aspects(selected_items, aspect_dictionaries)
        self.assertEqual(result, expected)
        # Test case where none of the selected items are in the aspects.
        selected_items = [5, 6, 7]
        aspect_dictionaries = [
            {'sports': [1, 2, 3], 'tech': [3, 4]},
            {'entertainment': [1, 4], 'education': [2]}
        ]
        expected = [
            {'sports': 0, 'tech': 0},
            {'entertainment': 0, 'education': 0}
        ]
        result = count_selected_in_aspects(selected_items, aspect_dictionaries)
        self.assertEqual(result, expected)
        # Test case where some selected items are in the aspects.
        selected_items = [2, 3]
        aspect_dictionaries = [
            {'sports': [1, 2, 3], 'tech': [3, 4]},
            {'entertainment': [1, 4], 'education': [2]}
        ]
        expected = [
            {'sports': 2, 'tech': 1},
            {'entertainment': 0, 'education': 1}
        ]
        result = count_selected_in_aspects(selected_items, aspect_dictionaries)
        self.assertEqual(result, expected)
        # Test case where the selected items list is empty.
        selected_items = []
        aspect_dictionaries = [
            {'sports': [1, 2, 3], 'tech': [3, 4]},
            {'entertainment': [1, 4], 'education': [2]}
        ]
        expected = [
            {'sports': 0, 'tech': 0},
            {'entertainment': 0, 'education': 0}
        ]
        result = count_selected_in_aspects(selected_items, aspect_dictionaries)
        self.assertEqual(result, expected)

    def test_processPartyData(self):
        # entities = mind.load_entities(fpath="./tests/enriched_data/party.json")
        extracted_entities = {"1":['a','b','c'],
                              "2": ['e','a','b'],
                              "3":[],
                              "4":['a','a','b','b'],
                              "5":[]}


        out_pd = pd.Series(extracted_entities).to_frame('entities')
        out_pd.reset_index(drop=True, inplace=True)
        print(out_pd)
        out_pd.to_csv('output.csv')

        # read csv file
        input_pd = pd.read_csv('output.csv')
        if 'entities' in input_pd.columns:
            input_pd['entities'] = input_pd['entities'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        for col in out_pd.columns:
            if not out_pd[col].equals(input_pd[col]):
                raise AssertionError(
                    f"Mismatch found in column: {col}\n"
                    f"Output DataFrame (first few rows):\n{out_pd[col].head()}\n"
                    f"Input DataFrame (first few rows):\n{input_pd[col].head()}\n"
                )

    def test_target_distribution_matcher_entities(self):
        entities = {
            "N38895": ["Republican Party"],  # 0
            "N35671": ["Democratic Party", "Republican Party"],  # 1
            "N28682": [],  # 2
            "N51741": ["Republican Party"],  # 3
            "N6916":  ["Democratic Party"],  # 4
            "N61186": ["Other Party", "Democratic Party"],  # 5
            "N34775": None  # 6
        }

        out_pd = pd.Series(entities).to_frame('entities')
        out_pd.reset_index(drop=True, inplace=True)

        target_distr_dummy = {
            "entities": {"type": "parties", "distr": [{"description": "only mention", "contain": ["Republican Party"], "prob": 0.2}, {"description": "only mention", "contain": ["Democratic Party"], "prob": 0.2}, {"description": "composition", "contain": [["Republican Party"], ["Democratic Party"]], "prob": 0.4}, {"description": "minority but can also mention", "contain": ["Republican Party", "Democratic Party"], "prob": 0.2}]}}
        targetDimension = ["entities"]
        selected_distribution = [target_distr_dummy[i]
                                 for i in targetDimension]
        candidate_items = [0, 1, 2, 3, 4, 5, 6]
        proportions, matched_items = TargetDistributionMatcher(
            selected_distribution, targetDimension, out_pd,  candidate_items)
        actual_items = matched_items[0]
        expected_matched_items = {
            'entities,only mention:Republican Party': [0, 3],

            'entities,only mention:Democratic Party': [4],

            "entities,composition:['Republican Party'],['Democratic Party']": [1],
            'entities,minority but can also mention:Republican Party,Democratic Party': [5],
        }
        # for key in actual_items:
        #     print(f"actual key:{key}")

        for key, expected_value in expected_matched_items.items():
            # Check if the key exists in the actual dictionary
            if key in actual_items:
                # Assert that the value associated with the key is the same in both dictionaries
                self.assertEqual(
                    actual_items[key], expected_value, f"Value mismatch for key: {key}")
            else:
                # Assert an error if the key is not found
                self.fail(
                    f"Key {key} expected but not found in actual dictionary.")

        expected_proportions = {
            'entities,only mention:Republican Party': 0.2,

            'entities,only mention:Democratic Party': 0.2,

            # 'entities,only mention:Republican Party,Democratic Party': 0.4,
            "entities,composition:['Republican Party'],['Democratic Party']": 0.4,
            'entities,minority but can also mention:Republican Party,Democratic Party': 0.2,
        }
        for key, expected_value in expected_proportions.items():
            # Check if the key exists in the actual dictionary
            if key in proportions[0]:
                # Assert that the value associated with the key is the same in both dictionaries
                self.assertEqual(
                    proportions[0][key], expected_value, f"Value mismatch for key: {key}")
            else:
                # Assert an error if the key is not found
                self.fail(
                    f"Key {key} expected but not found in actual dictionary.")


if __name__ == '__main__':
    unittest.main()
