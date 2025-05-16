import unittest
import pandas as pd
import numpy as np


from cornac.metrics.user import UserMetric
from cornac.metrics.user import UserActivation
from cornac.metrics.user import UserGiniCoeff
from cornac.metrics.user import UserAlternativeVoices
from cornac.metrics.user import UserRepresentation
from cornac.metrics.user import UserCalibration
from cornac.metrics.user import UserFragmentation
from cornac.metrics.user import UserILD


class TestDataset(unittest.TestCase):
    def test_user_metric(self):
        metric = UserMetric()

        self.assertEqual(metric.type, "user")
        self.assertIsNone(metric.name)
        try:
            metric.compute_user()
        except NotImplementedError:
            assert True

    def test_activation(self):
        # Scenario 1: Basic test case
        item_sentiments_scenario1 = {1: 0.5, 2: -0.3, 3: 0.2}
        user_seen_item_scenario1 = pd.DataFrame({'user id': [1, 2],
                                                 'item seen': [[1], [3]]})
        user_exposed_df_scenario1 = pd.DataFrame({'user id': [1, 2],
                                                 'item exposed': [[1, 2], [2, 3]]})
        activation_metric_scenario1 = UserActivation(item_sentiments=item_sentiments_scenario1,
                                                     user_seen_item=user_seen_item_scenario1)

        self.assertEqual(activation_metric_scenario1.type, 'user')
        self.assertEqual(activation_metric_scenario1.name, 'Activation')
        # Test initialization of Activation class
        self.assertEqual(activation_metric_scenario1.divergence_type, "KL")
        self.assertFalse(activation_metric_scenario1.discount)
        self.assertEqual(activation_metric_scenario1.n_bins, 5)  # default value

        activation_scores_scenario1 = activation_metric_scenario1.compute_user(user_exposed_df_scenario1)
        expected_scores_scenario1 = {1: 4.477771096259041, 2: 4.477771096259041}
        for user_id, expected_score in expected_scores_scenario1.items():
            self.assertAlmostEqual(activation_scores_scenario1[user_id], expected_score, places=6)

        # Scenario 2: Empty user exposure data
        item_sentiments_scenario2 = {1: 0.5, 2: -0.3, 3: 0.2}
        user_seen_item_scenario2 = pd.DataFrame({'user id': [],
                                                 'item seen': []})
        activation_metric_scenario2 = UserActivation(item_sentiments=item_sentiments_scenario2,
                                                     user_seen_item=user_seen_item_scenario2)

        user_exposed_df_empty = pd.DataFrame({'user id': [], 'item exposed': []})
        activation_scores_empty = activation_metric_scenario2.compute_user(user_exposed_df_empty)
        expected_scores_empty = {}
        self.assertEqual(activation_scores_empty, expected_scores_empty)

        # Scenario 3: User with no seen items
        item_sentiments_scenario3 = {1: 0.5, 2: -0.3, 3: 0.2}
        user_seen_item_empty = pd.DataFrame({'user id': [], 'item seen': []})
        activation_metric_no_seen_items = UserActivation(item_sentiments=item_sentiments_scenario3,
                                                         user_seen_item=user_seen_item_empty)
        user_exposed_df_scenario3 = pd.DataFrame({'user id': [1, 2],
                                                 'item exposed': [[1, 2], [2, 3]]})
        activation_scores_scenario3 = activation_metric_no_seen_items.compute_user(user_exposed_df_scenario3)
        expected_scores_scenario3 = {}
        self.assertEqual(activation_scores_scenario3, expected_scores_scenario3)

    def test_gini(self):
        # Sample item genres data
        item_genres = {
            1: np.array([0, 1, 0]),
            2: np.array([1, 0, 0]),
            3: np.array([0, 0, 1]),
        }

        # Sample user seen items data
        user_seen_item = pd.DataFrame({
            'user id': [1, 2, 3],  # User 1 sees item 1 and 2, User 2 sees item 1, 2 and 3, User 3 sees item 2 and 3
            'item seen': [[1, 2], [1, 2, 3], [2, 3]]
        })
        # Scenario 1: Basic test case
        # Initialize UserGiniCoeff object
        user_gini_coeff = UserGiniCoeff(item_genres, user_seen_item)
        self.assertEqual(user_gini_coeff.type, 'user')
        self.assertEqual(user_gini_coeff.name, 'GiniCoeff')
        # Expected Gini coefficients
        expected_gini_coefficients = {1: 0.5, 2: 0, 3: 0.5}
        # Compute Gini coefficients
        computed_gini_coefficients = user_gini_coeff.compute_user()
        # Check if computed Gini coefficients match the expected values
        self.assertDictEqual(computed_gini_coefficients, expected_gini_coefficients)

        # Scenario 2: User with no seen items
        user_seen_item = pd.DataFrame(columns=['user id', 'item seen'])
        user_gini_coeff = UserGiniCoeff(item_genres, user_seen_item)
        computed_gini_coefficients = user_gini_coeff.compute_user()
        self.assertEqual(computed_gini_coefficients, {})

    def test_alternative(self):
        # Sample item minority-majority data
        item_minor_major = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.9545, 0.0455]),
            3: np.array([0.0, 1.0]),
        }
        user_seen_item = pd.DataFrame({
            'user id': [1],
            'item seen': [[1, 2, 3]]
        })
        # Scenario 1: Basic test case
        user_alternative_voices = UserAlternativeVoices(item_minor_major, user_seen_item)
        self.assertEqual(user_alternative_voices.type, 'user')
        self.assertEqual(user_alternative_voices.name, 'AltVoices_mainstream')
        computed_scores = user_alternative_voices.compute_user(
            pd.DataFrame({'user id': [1], 'item exposed': [[1, 2, 3]]}))
        self.assertEqual(computed_scores[1], 0.0)

        # Multiple users scenario
        user_seen_item = pd.DataFrame({
            'user id': [1, 2, 3],
            'item seen': [[1, 2], [2], [3]]
        })
        user_alternative_voices = UserAlternativeVoices(item_minor_major, user_seen_item)
        computed_scores = user_alternative_voices.compute_user(
            pd.DataFrame({'user id': [1, 2, 3], 'item exposed': [[1, 2], [1, 2, 3], [1, 2, 3]]}))
        self.assertAlmostEqual(computed_scores[1], 0)
        self.assertAlmostEqual(computed_scores[2], 0.6605655053187146, places=6)
        self.assertAlmostEqual(computed_scores[3], 5.955566450379878, places=6)

        # Scenario 2: User with no seen items
        user_seen_item_empty = pd.DataFrame(columns=['user id', 'item seen'])
        user_alternative_voices = UserAlternativeVoices(item_minor_major, user_seen_item_empty)
        computed_scores = user_alternative_voices.compute_user(pd.DataFrame({'user id': [1, 2, 3], 'item exposed': [[1, 2], [1, 2, 3], [1, 2, 3]]}))
        self.assertEqual(computed_scores, {})

        # Scenario 3: User with no exposed items
        user_alternative_voices = UserAlternativeVoices(item_minor_major, user_seen_item)
        computed_scores = user_alternative_voices.compute_user(pd.DataFrame(columns=['user id', 'item exposed']))
        self.assertEqual(computed_scores, {})

    def test_representation(self):
        # Sample data
        item_entities = {1: ['Democratic Party'],
                         2: ['Democratic Party', 'Republican Party'],
                         3: ["Labour Party"],
                         4: ['Democratic Party', 'Democratic Party', 'Democratic Party', 'Democratic Party'],
                         5: ['Republican Party', 'Republican Party']}

        user_seen_item = pd.DataFrame({'user id': [1, 2], 'item seen': [[1, 2, 4], [5]]})
        user_exposed_df = pd.DataFrame({'user id': [1, 2], 'item exposed': [[1, 2, 4], [1, 2, 3, 4, 5]]})

        # Scenario 1: Basic test case
        user_representation = UserRepresentation(item_entities, user_seen_item)
        self.assertEqual(user_representation.type, 'user')
        self.assertEqual(user_representation.name, 'Representation')
        representation_scores = user_representation.compute_user(user_exposed_df)
        expected_scores = {1: 0.0, 2: 6.447072857710013}  # 0 means no divergence
        # self.assertEqual(representation_scores, expected_scores)
        for key in expected_scores:
            self.assertAlmostEqual(representation_scores[key], expected_scores[key], places=6)


        # Scenario 2: User with no seen items
        user_seen_item = pd.DataFrame({'user id': [], 'item seen': []})
        user_exposed_df = pd.DataFrame({'user id': [1, 2], 'item exposed': [[1, 2, 4], [1, 2, 3, 4, 5]]})
        user_representation = UserRepresentation(item_entities, user_seen_item)
        representation_scores = user_representation.compute_user(user_exposed_df)
        expected_scores = {}
        self.assertEqual(representation_scores, expected_scores)

        # Scenario 3: User with no exposed items
        user_seen_item = pd.DataFrame({'user id': [], 'item seen': []})
        user_exposed_df = pd.DataFrame({'user id': [], 'item exposed': []})
        user_representation = UserRepresentation(item_entities, user_seen_item)
        representation_scores = user_representation.compute_user(user_exposed_df)
        expected_scores = {}
        self.assertEqual(representation_scores, expected_scores)

    def test_calibration(self):
        item_feature_category = {
            1: 'action',
            2: 'comedy',
            3: 'horror',
            4: 'drama'
        }
        # Test for Calibration_category
        user_seen_item_df = pd.DataFrame({'user id': [1, 2], 'item seen': [[1, 2, 3], [1, 2]]})
        user_exposed_df = pd.DataFrame({'user id': [1, 2], 'item exposed': [[1, 2, 3], [1, 2, 3, 4]]})
        user_calibration = UserCalibration(item_features=item_feature_category,
                                           user_seen_item=user_seen_item_df,
                                           data_type="category")
        self.assertEqual(user_calibration.type, 'user')
        self.assertEqual(user_calibration.name, 'Calibration_category')

        # Scenario 1: Basic test case
        calibration_scores = user_calibration.compute_user(user_exposed_df)
        expected_scores = {1: 0, 2: 4.477771096259041}
        # self.assertEqual(calibration_scores, expected_scores)
        for key in expected_scores:
            self.assertAlmostEqual(calibration_scores[key], expected_scores[key], places=6)

        # Scenario 2: User with no seen items
        user_seen_item_df = pd.DataFrame({'user id': [], 'item seen': []})
        user_exposed_df = pd.DataFrame({'user id': [], 'item exposed': []})
        user_calibration = UserCalibration(item_features=item_feature_category,
                                           user_seen_item=user_seen_item_df,
                                           data_type="category")
        calibration_scores = user_calibration.compute_user(user_exposed_df)
        expected_scores = {}
        self.assertEqual(calibration_scores, expected_scores)

        # Test for Calibration_complexity
        item_feature_complexity = {1: 17, 2: 31, 3: 15, 4: 2}
        user_seen_item_df = pd.DataFrame({'user id': [1, 2], 'item seen': [[1, 2, 3], [4]]})
        user_exposed_df = pd.DataFrame({'user id': [1, 2], 'item exposed': [[1, 2, 3], [1, 2, 3, 4]]})
        user_calibration = UserCalibration(item_features=item_feature_complexity,
                                           user_seen_item=user_seen_item_df,
                                           data_type="complexity")
        self.assertEqual(user_calibration.type, 'user')
        self.assertEqual(user_calibration.name, 'Calibration_complexity')
        # Scenario 1: Basic test case
        calibration_scores = user_calibration.compute_user(user_exposed_df)
        expected_scores = {1: 0, 2: 6.965637456516297}
        # self.assertEqual(calibration_scores, expected_scores)
        for key in expected_scores:
            self.assertAlmostEqual(calibration_scores[key], expected_scores[key], places=6)


        # Scenario 2: User with no seen items
        user_seen_item_df = pd.DataFrame({'user id': [], 'item seen': []})
        user_exposed_df = pd.DataFrame({'user id': [], 'item exposed': []})
        user_calibration = UserCalibration(item_features=item_feature_complexity,
                                           user_seen_item=user_seen_item_df,
                                           data_type="complexity")
        calibration_scores = user_calibration.compute_user(user_exposed_df)
        expected_scores = {}
        self.assertEqual(calibration_scores, expected_scores)

    def test_fragmentation(self):
        item_stories = {1: 0, 2: 2817, 3: 0, 4: 1692}
        user_seen_item_df = pd.DataFrame({'user id': [1, 2], 'item exposed': [[2, 4], [3]]})
        user_fragmentation = UserFragmentation(item_stories=item_stories,
                                               user_exposed_item=user_seen_item_df)
        # Scenario 1: Basic test case
        self.assertEqual(user_fragmentation.divergence_type, "KL")
        self.assertFalse(user_fragmentation.discount)
        self.assertEqual(user_fragmentation.n_samples, 1)  # default value
        fragmentation_scores = user_fragmentation.compute_user()
        self.assertIsInstance(fragmentation_scores, dict)
        expected_scores = {1: 9.944412186056836, 2: 9.944412186056834}
        # self.assertEqual(fragmentation_scores, expected_scores)
        for key in expected_scores:
            self.assertAlmostEqual(fragmentation_scores[key], expected_scores[key], places=6)

        # Scenario 2: User with no seen items
        user_seen_item_df = pd.DataFrame({'user id': [], 'item exposed': []})
        user_fragmentation = UserFragmentation(item_stories=item_stories,
                                               user_exposed_item=user_seen_item_df)
        fragmentation_scores = user_fragmentation.compute_user()
        self.assertEqual(fragmentation_scores, {})

    def test_ild(self):
        # Sample item feature data
        item_features = {
            1: np.array([0, 1, 0]),
            2: np.array([1, 0, 0]),
            3: np.array([0, 0, 1]),
        }
        user_seen_item_df = pd.DataFrame({'user id': [1, 2], 'item seen': [[1, 2], [3]]})
        user_ild = UserILD(item_features=item_features, user_seen_item=user_seen_item_df)
        self.assertEqual(user_ild.type, 'user')
        self.assertEqual(user_ild.name, 'ILD')
        # Scenario 1: Basic test case
        ild_scores = user_ild.compute_user()
        expected_scores = {1: 1.0, 2: 0}
        self.assertEqual(ild_scores, expected_scores)

        # Scenario 2: User with no seen items
        user_seen_item_df = pd.DataFrame({'user id': [], 'item seen': []})
        user_ild = UserILD(item_features=item_features, user_seen_item=user_seen_item_df)
        ild_scores = user_ild.compute_user()
        expected_scores = {}
        self.assertEqual(ild_scores, expected_scores)


if __name__ == '__main__':
    unittest.main()