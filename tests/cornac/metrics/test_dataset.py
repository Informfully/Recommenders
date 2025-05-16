import unittest

import numpy as np
import math
from cornac.metrics.dataset import DatasetMetric
from cornac.metrics.dataset import DatasetActivation
from cornac.metrics.dataset import DatasetGiniCoeff
from cornac.metrics.dataset import DatasetRepresentation
from cornac.metrics.dataset import DatasetAlternativeVoices
from cornac.metrics.dataset import DatasetILD
from cornac.metrics.dataset import DatasetCalibration


class TestDataset(unittest.TestCase):
    def test_dataset_metric(self):
        metric = DatasetMetric()

        self.assertEqual(metric.type, "dataset")
        self.assertIsNone(metric.name)
        try:
            metric.compute_dataset_itself()
        except NotImplementedError:
            assert True

    def test_calculate_spacelog(self):
        # Test case 1: num_users = 100, num_items = 50, sc = 1000
        result_1 = DatasetMetric.calculate_spacelog(100, 50, 1000)
        self.assertAlmostEqual(result_1, 0.69897, places=5)

        # Test case 2: num_users = 0, num_items = 5, sc = 1000
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_spacelog(0, 5, 1000)

        # Test case 3: num_users = -10, num_items = 20, sc = 1000
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_spacelog(-10, 5, 1000)

    def test_calculate_shapelog(self):
        # Test case for positive values of num_users and num_items
        self.assertAlmostEqual(DatasetMetric.calculate_shapelog(100, 50), math.log10(100 / 50))

        # Test case for negative values of num_users and num_items
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_shapelog(-100, -50)
        # Test case when num_items is zero
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_shapelog(100, 0)

        # Test case when num_users is zero
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_shapelog(0, 50)

        # Test case when both num_users and num_items are zero
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_shapelog(0, 0)

    def test_calculate_densitylog(self):
        # Test case with valid input
        self.assertAlmostEqual(DatasetMetric.calculate_densitylog(100, 50, 200), -1.39794, places=5)

        # Test case when num_users is zero
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_densitylog(0, 50, 200)

        # Test case when num_items is zero
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_densitylog(100, 0, 200)

        # Test case when num_interactions is zero
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_densitylog(100, 50, 0)

        # Test case when all input values are zero
        with self.assertRaises(ValueError):
            DatasetMetric.calculate_densitylog(0, 0, 0)

    def test_gini_user(self):
        # Test case with valid input
        self.assertAlmostEqual(DatasetMetric.compute_gini_user(31, 2, [15, 16]), 0.0107526881, places=6)

        # Test case when num_rating is zero
        with self.assertRaises(ValueError):
            DatasetMetric.compute_gini_user(0, 10, [10, 15, 5, 20, 8, 12, 12, 7, 7, 4])

        # Test case when num_user is zero
        with self.assertRaises(ValueError):
            DatasetMetric.compute_gini_user(100, 0, [10, 15, 5])

        # Test case when user_interaction is empty
        with self.assertRaises(ValueError):
            DatasetMetric.compute_gini_user(100, 10, [])

    def test_gini_item(self):
        # Test case with balanced distribution
        self.assertAlmostEqual(DatasetMetric.compute_gini_item(100, 10, [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]), 0.0, places=5)

        # Test case with unequal distribution
        self.assertAlmostEqual(DatasetMetric.compute_gini_item(100, 5, [5, 10, 15, 25, 45]), 0.316666666666, places=5)

        # Test case with single item
        self.assertAlmostEqual(DatasetMetric.compute_gini_item(100, 1, [100]), 0.0, places=5)

        # Test case with zero total interactions
        with self.assertRaises(ValueError):
            DatasetMetric.compute_gini_item(0, 10, [5, 10, 15, 25, 45, 5, 10, 15, 25, 45])

    def test_activation(self):

        item_sentiment = {1: 0.1, 2: 0.2, 3: -0.3, 4: 0.4, 5: -0.5}
        reference_distribution = [0.2, 0.2, 0.3, 0.2, 0.1]
        activation = DatasetActivation(item_sentiment)
        self.assertEqual(activation.type, 'dataset')
        self.assertEqual(activation.name, 'Activation')

        # Test initialization of Activation class
        self.assertEqual(activation.divergence_type, "KL")
        self.assertFalse(activation.discount)
        self.assertEqual(activation.n_bins, 5)  # default value

        # Test compute_dataset_itself with and without custom reference distribution, and with empty pool
        score_uniform_ref = activation.compute_dataset_itself()
        score_custom_ref = activation.compute_dataset_itself(reference_distribution)
        score_empty_pool = activation.compute_dataset_itself(np.array([]))

        self.assertEqual(score_uniform_ref, 0.0)
        self.assertAlmostEqual(score_custom_ref, 0.08265706890383971, places=5)
        self.assertIsNone(score_empty_pool)

    def test_gini(self):
        # Test case when item_genre_data is not empty
        item_genre_data = {1: np.array([1, 0, 1]), 2: np.array([0, 1, 0]), 3: np.array([1, 1, 0])}
        gini_coeff = DatasetGiniCoeff(item_genre_data)
        self.assertEqual(gini_coeff.type, 'dataset')
        self.assertEqual(gini_coeff.name, 'GiniCoeff')
        result = gini_coeff.compute_dataset_itself()
        expected_result = 0.2  # Expected result based on the provided item_genre_data
        self.assertAlmostEqual(result, expected_result, places=5)

        # Test case when item_genre_data is empty
        item_genre_data = {}
        gini_coeff = DatasetGiniCoeff(item_genre_data)
        result = gini_coeff.compute_dataset_itself()
        self.assertIsNone(result)

        # Test case when item_genre_data is invalid (not a dictionary)
        item_genre_data = "invalid_data"
        with self.assertRaises(ValueError):
            gini_coeff = DatasetGiniCoeff(item_genre_data)

    def test_representation(self):
        # Sample data
        item_entities = {1: ['Democratic Party'],
                         2: ['Democratic Party', 'Republican Party'],
                         3: ["Labour Party"],
                         4: ['Democratic Party', 'Democratic Party', 'Democratic Party', 'Democratic Party'],
                         5: ['Republican Party', 'Republican Party']}
        representation = DatasetRepresentation(item_entities)
        self.assertEqual(representation.type, 'dataset')
        self.assertEqual(representation.name, 'Representation')
        # Test with a uniform reference distribution
        score_custom_ref = representation.compute_dataset_itself(reference_distribution=[0.1, 0.8, 0.1])
        self.assertAlmostEqual(score_custom_ref, 0.9318228347253373, places=5)
        # Test without providing a reference distribution
        score_uniform_ref = representation.compute_dataset_itself()
        self.assertAlmostEqual(score_uniform_ref, 0.09908482780013625, places=5)
        # Test with an empty item_entities
        item_entities = {}
        representation = DatasetRepresentation(item_entities)
        score_empty_pool = representation.compute_dataset_itself()
        self.assertIsNone(score_empty_pool)

    def test_alternative(self):
        # Sample data
        item_minor_major = {1: [0.5, 0.5], 2: [1., 0.]}
        alternative_voices = DatasetAlternativeVoices(item_minor_major)
        self.assertEqual(alternative_voices.type, 'dataset')
        self.assertEqual(alternative_voices.name, 'AltVoices_mainstream')
        # Test with uniform reference distribution(without providing a reference distribution)
        score_uniform_ref = alternative_voices.compute_dataset_itself()
        #self.assertIsNotNone(score_uniform_ref)
        self.assertAlmostEqual(score_uniform_ref, 0.18796574253187304, places=5)
        # Test with providing a customer reference distribution
        score_custom_ref = alternative_voices.compute_dataset_itself(reference_distribution=[0.2, 0.8])
        #self.assertIsNotNone(score_custom_ref)
        self.assertAlmostEqual(score_custom_ref, 1.005958702918937, places=5)
        # Test with an empty reference
        score_empty_pool = alternative_voices.compute_dataset_itself(np.array([]))
        self.assertIsNone(score_empty_pool)

    def test_ild(self):
        # Test with missing features for some items
        item_feature_missing = {
            1: np.array([0., 0., 1.]),
            2: np.array([0., 0., 1.]),
            3: np.array([1., 0., 0.]),
            4: np.array([1., 0.])  # Missing feature
        }
        ild_missing = DatasetILD(item_feature_missing)
        self.assertEqual(ild_missing.type, 'dataset')
        self.assertEqual(ild_missing.name, 'ILD')
        score_missing = ild_missing.compute_dataset_itself()
        self.assertIsNone(score_missing)
        # Test with custom distance type
        item_feature_custom = {
            1: np.array([0., 0., 1.]),
            2: np.array([0., 0., 1.]),
            3: np.array([1., 0., 0.])
        }
        ild_custom = DatasetILD(item_feature_custom, distance_type="euclidean")
        score_custom = ild_custom.compute_dataset_itself()
        self.assertIsNotNone(score_custom)
        self.assertAlmostEqual(score_custom, 0.9428090415820635, places=5)
        # Test with empty item_feature dictionary
        item_feature_empty = {}
        ild_empty = DatasetILD(item_feature_empty)
        score_empty = ild_empty.compute_dataset_itself()
        self.assertIsNone(score_empty)

    def test_calibration(self):
        item_feature_category = {
            1: 'action',
            2: 'comedy',
            3: 'horror',
            4: 'drama'
        }
        calibration_category = DatasetCalibration(item_feature=item_feature_category, data_type='category')
        # Test with reference distribution
        score_uniform = calibration_category.compute_dataset_itself()
        self.assertEqual(score_uniform, 0.0)
        #Test with custom distribution
        score_custom = calibration_category.compute_dataset_itself([0.7, 0.1, 0.1, 0.1])
        self.assertAlmostEqual(score_custom, 0.6404028947498385, places=5)
        self.assertEqual(calibration_category.type, 'dataset')
        self.assertEqual(calibration_category.name, 'Calibration_category')
        # Test with empty reference
        score_empty = calibration_category.compute_dataset_itself([])
        self.assertIsNone(score_empty)

        item_feature_complexity = {1: 17, 2: 31, 3: 15, 4: 2}
        calibration_complexity = DatasetCalibration(item_feature=item_feature_complexity, data_type='complexity')
        self.assertEqual(calibration_complexity.type, 'dataset')
        self.assertEqual(calibration_complexity.name, 'Calibration_complexity')
        # Test with reference distribution
        score1 = calibration_complexity.compute_dataset_itself()
        self.assertAlmostEqual(score1, 3.5890312942211344, places=5)
        # Test with custom distribution
        score2 = calibration_complexity.compute_dataset_itself([0.1, 0.1, 0.1, 0.7])
        self.assertAlmostEqual(score2, 7.596494895212117, places=5)
        score3 = calibration_complexity.compute_dataset_itself([0.4, 0.3, 0.2, 0.1])
        self.assertAlmostEqual(score3, 3.9856485706395306, places=5)
        # Test with empty input
        dataset_empty = {}
        calibration_complexity = DatasetCalibration(item_feature=dataset_empty, data_type='complexity')
        score_empty = calibration_complexity.compute_dataset_itself()
        self.assertIsNone(score_empty)


if __name__ == '__main__':
    unittest.main()
