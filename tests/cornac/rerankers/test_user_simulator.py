import unittest
import numpy as np
from cornac.eval_methods import RatioSplit
import pandas as pd
from cornac.rerankers.user_simulator import UserSimulator
from cornac.datasets import mind as mind
import datetime
import os
import configparser
from unittest.mock import patch


class TestUserSimulator(unittest.TestCase):
    def setUp(self):
        self.user_history_items_with_access_date = [(1, datetime.datetime(
            2024, 9, 20)), (2, datetime.datetime(2024, 9, 20)), (3, datetime.datetime(2024, 9, 20))]
        self.user_history_items = [1, 2]
        self.config_file_path = os.path.join(os.path.dirname(
            __file__), 'test_user_simulator_config.ini')

        self.empty_history_user_click_by_position = UserSimulator(user_id=0, user_history=[], choice_model='logarithmic_rank_bias',
                                                                  config_path=self.config_file_path
                                                                  )

        self.user_click_by_position = UserSimulator(user_id=2, user_history=self.user_history_items_with_access_date, choice_model='logarithmic_rank_bias',
                                                    config_path=self.config_file_path
                                                    )

    def test_init(self):
        self.assertEqual(self.empty_history_user_click_by_position.history, [])

        self.assertEqual(self.user_click_by_position.history,
                         [1, 2, 3])  # because frequency_threshold doesn't exist
        self.assertEqual(
            self.user_click_by_position.choice_model, 'logarithmic_rank_bias')

    @patch('os.path.exists', return_value=False)
    def test_config_path_not_exist(self, mock_exists):
        """Test UserSimulator when the config_path does not exist."""
        with self.assertRaises(FileNotFoundError) as context:
            UserSimulator(user_id=1, user_history=[],
                          config_path='non_existent_config.ini')

        # Assert the exception message is as expected
        self.assertIn("The configuration file non_existent_config.ini does not exist.", str(
            context.exception))

    def test_load_levels(self):
        with self.assertRaises(ValueError) as e:
            levels = self.user_click_by_position.load_levels(
                self.user_click_by_position.config, 'unknown_thresholds')
        self.assertIn("An error occurred while loading levels:", str(
            e.exception))

        self.user_click_by_position.load_levels(
            self.user_click_by_position.config, 'total_reads_thresholds')

        self.assertEqual(self.user_click_by_position.levels,
                         ['active', 'less_active'])

    def test_load_thresholds(self):
        threshold = self.user_click_by_position.load_thresholds(
            self.user_click_by_position.config, 'total_reads_thresholds')
        self.assertEqual(threshold,
                         {'active': 3, 'less_active': 1})

        threshold = self.user_click_by_position.load_thresholds(
            self.user_click_by_position.config, 'frequency_thresholds')
        self.assertEqual(threshold,
                         None)

    def test_calculate_activity_level_total_reads(self):
        activity_level = self.empty_history_user_click_by_position.calculate_total_reads_level()
        self.assertEqual(activity_level, 'less_active')

        activity_level = self.user_click_by_position.calculate_total_reads_level()
        self.assertEqual(activity_level, 'active')

    def test_calculate_activity_level_frequency_based(self):

        dummy_user = self.user_click_by_position
        dummy_user.has_frequency_thresholds = True
        dummy_user.frequency_thresholds = {'active': 3, 'less_active': 1}
        dummy_user.history = self.user_history_items_with_access_date
        activity_level = dummy_user.calculate_frequency_based_level()
        self.assertEqual(activity_level, 'active')

    def test_calculate_activity_level(self):
        empty_history_user_active_level = self.empty_history_user_click_by_position.calculate_activity_level()
        self.assertEqual(
            empty_history_user_active_level, 'less_active')
        three_history_user_active_level = self.user_click_by_position.calculate_activity_level()
        self.assertEqual(
            three_history_user_active_level, 'active')
        two_items_in_user_history_user = UserSimulator(user_id=2, user_history=self.user_history_items, choice_model='logarithmic_rank_bias',
                                                       config_path=self.config_file_path
                                                       )
        two_history_user_active_level = two_items_in_user_history_user.calculate_activity_level()
        self.assertEqual(
            two_history_user_active_level, 'less_active')

    def test_determine_max_iterations(self):
        max_iterations = self.empty_history_user_click_by_position.determine_max_iterations()
        self.assertEqual(max_iterations, 3)
        max_iterations = self.user_click_by_position.determine_max_iterations()
        self.assertEqual(
            max_iterations, 4)

        self.empty_history_user_click_by_position.active_level = 'unknown_level'
        max_iterations = self.empty_history_user_click_by_position.determine_max_iterations()
        self.assertEqual(max_iterations, 3)

    def test_rho(self):
        user_simulator = UserSimulator(
            user_id=1, user_history=[], config_path=self.config_file_path)
        user_simulator.preference = [
            {'category_A': 0.5, 'category_B': 0.5, 'category_C': 0}]
        user_simulator.attribute_items_mapping = [
            {'category_A': [1, 2], 'category_B': [3, 4], 'category_C': [5, 6]}]
        item_ids = [1, 3, 4]

        result = user_simulator.rho(item_ids)
        expected_result = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected_result)

        item_ids = [5, 6, 1]
        result = user_simulator.rho(item_ids)
        expected_result = np.array([0, 0, 0.5])
        np.testing.assert_array_almost_equal(result, expected_result)

        result = user_simulator.rho([])
        expected_result = np.array([])
        np.testing.assert_array_almost_equal(result, expected_result)

        # unkown attribute item_id
        item_ids = [7, 8, 9]
        result = user_simulator.rho(item_ids)
        expected_result = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_softmax(self):
        user_simulator = UserSimulator(
            user_id=1, user_history=[], config_path=self.config_file_path)
        scores = np.array([1.0, 2.0, 3.0])
        result = user_simulator.softmax(scores)
        expected_result = np.exp(scores - np.max(scores)) / \
            np.sum(np.exp(scores - np.max(scores)))
        np.testing.assert_array_almost_equal(result, expected_result)
        scores = np.array([])
        result = user_simulator.softmax(scores)
        np.testing.assert_array_almost_equal(result,  np.array([]))

        scores = np.array([0, 0, 0, 0])
        result = user_simulator.softmax(scores)
        # numpy handling empty arrays, will return an empty array.
        np.testing.assert_array_almost_equal(
            result,  np.array([0.25, 0.25, 0.25, 0.25]))

    def test_click_probability(self):
        # Test click_probability with 'logarithmic_rank_bias' choice model.
        user_simulator = UserSimulator(
            user_id=1, user_history=[], config_path=self.config_file_path)
        user_simulator.choice_model = 'logarithmic_rank_bias'
        result = user_simulator.click_probability([1, 2, 3])
        self.assertAlmostEqual(result.sum(), 1.0)

        expected_values = np.array(
            [1/(np.log(1+1)), 1/(np.log(1+2)), 1/(np.log(1+3))])
        expected_values = expected_values / expected_values.sum()
        np.testing.assert_almost_equal(result,
                                       expected_values)
        # Test click_probability with 'logarithmic_rank_bias' choice model.
        user_simulator = UserSimulator(
            user_id=1, user_history=[], config_path=self.config_file_path)
        user_simulator.preference = [
            {'category_A': 0.5, 'category_B': 0.5, 'category_C': 0}]
        user_simulator.attribute_items_mapping = [
            {'category_A': [1, 2], 'category_B': [3, 4], 'category_C': [5, 6]}]
        user_simulator.choice_model = 'preference_based_bias'
        result = user_simulator.click_probability([1, 2, 3, 4])
        self.assertAlmostEqual(result.sum(), 1.0)
        np.testing.assert_almost_equal(result, np.array(
            [0.25, 0.25, 0.25, 0.25]))

    @patch('cornac.rerankers.user_simulator.UserSimulator.click_probability')
    def test_simulate_interaction(self, mock_click_probability):
        mock_click_probability.return_value = np.array([0.4, 0.35, 0.25])

        item_ids = [1, 2, 3]
        clicks = self.user_click_by_position.simulate_interaction(item_ids)

        self.assertEqual(
            len(clicks), self.user_click_by_position.clicked_each_iteration)
        self.assertTrue(all(click in item_ids for click in clicks))
        self.assertEqual(
            self.user_click_by_position.interacted_items[0], clicks)
        self.assertEqual(self.user_click_by_position.seen_items, item_ids)

        item_ids = []
        clicks = self.user_click_by_position.simulate_interaction(item_ids)

        self.assertEqual(
            clicks, [])


if __name__ == '__main__':
    unittest.main()
