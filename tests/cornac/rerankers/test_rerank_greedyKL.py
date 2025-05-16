import unittest
import numpy as np
from cornac.datasets import movielens
from cornac.data import Reader
from cornac.eval_methods import RatioSplit, BaseMethod

import pandas as pd
import json
import cornac
from cornac.experiment.experiment import Experiment
from cornac.models import UserKNN
from cornac.rerankers import reranker
from cornac.rerankers.greedy_kl import GreedyKLReranker
from cornac.datasets import mind as mind
from pandas.testing import assert_frame_equal
import matplotlib.pyplot as plt
from cornac.utils.common import safe_kl_divergence, safe_kl_divergence_dicts
import random


class TestRerank(unittest.TestCase):
    def setUp(self):
        self.dummy_diversity_dimension = ["feature1", "category"]
        self.dummy_item_features = pd.DataFrame({
            'feature1': [0.1, 0.3, 0.2, 0.5, 0.8, 0.6, 0.7],
            'feature2': [0.6, 0.8, 0.7, 0.2, 0.2, 0.1, 0.05],
            'feature3': [0.5, 0.2, 0.9, 0.4, 0.1, 0.4, 0.3],
            'category': [
                "Politics", "Technology", "Sports", "Health", "Entertainment", "Finance", "Travel"
            ],
            # "entities" column represents political parties
            'entities': [
                ["partyA", "partyB"],  # item 0 belongs to both partyA and partyB
                ["partyA"],            # item 1 belongs to partyA
                ["partyC"],            # item 2 belongs to partyC
                None,                 # item 3 belongs to no party
                ["partyA"],            # item 4 belongs to partyA
                ["partyB"],            # item 5 belongs to partyB
                ["partyC", "partyA"]   # item 6 belongs to both partyC and partyA
            ]

        }, index=[0, 1, 2, 3, 4, 5, 6])

        self.dummy_data = [('u0', 'item0', 3.0),
                           ('u1', 'item1', 1.0),
                           ('u2', 'item0', 2.0),
                           ('u3', 'item1', 4.0),
                           ('u4', 'item1', 2.0),
                           ('u5', 'item2', 5.0),
                           ('u3', 'item2', 3.0),
                           ('u3', 'item3', 3.0),
                           ('u0', 'item4', 3.0),
                           ('u2', 'item2', 3.0),
                           ('u4', 'item3', 3.0),
                           ('u5', 'item3', 3.0),
                           ('u3', 'item5', 3.0),
                           ('u4', 'item5', 3.0),
                           ('u5', 'item6', 2.0),
                           ('u0', 'item6', 1.0),
                           ('u2', 'item3', 2.0),
                           ('u1', 'item2', 3.0),
                           ('u6', 'item1', 2.0)
                           ]
        self.expected_user_history = {0: [0, 4], 1: [1], 2: [0, 2], 3: [1, 2, 3, 5],
                                      4: [1, 3, 5], 5: [2, 3, 6], 6: []}
        self.ratio_split = RatioSplit.from_splits(data=self.dummy_data,
                                                  train_data=self.dummy_data[:-
                                                                             4], test_data=self.dummy_data[-4:],
                                                  exclude_unknowns=True, verbose=True, seed=123, rating_threshold=1)
        self.target_distr_dummy = {
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
            },
            "category":
            {
                "type": "discrete",
                "distr": {
                    "Politics": 0.25,
                    "Travel": 0.25,
                    "Sports": 0.25,
                    "Entertainment": 0.25,
                    "Technology": 0,
                    "Health": 0,
                    "Finance": 0
                }},
            "entities": {"type": "parties",
                         "distr": [{"description": "only mention", "contain": ["partyA"], "prob": 0.2},
                                   {"description": "only mention",
                                    "contain": ["partyB"], "prob": 0.2},
                                #    {"description": "only mention", "contain": [
                                #        "partyB", "partyA"], "prob": 0.2},
                                 {"description": "composition", "contain": [
                                       ["partyB"],["partyA"]], "prob": 0.2},
                                   {"description": "minority but can also mention", "contain": [
                                       "partyA", "partyB"], "prob": 0.2},  # item 2 and item 6
                                   {"description": "no parties", "contain": [], "prob": 0.2}]}

        }

    def calculate_distribution(self, df, item_ids, column_name, normalize=True, bins=None, bin_labels=None, plot=False, plot_file=None):
        """
        Calculate the distribution of values for a specified column, limited to given item IDs in a DataFrame.
        optionally plot the distribution of values for a specified column to given item IDs in a DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            item_ids (list): List of item IDs (indices) to consider in the calculation.
            column_name (str): The column for which to calculate the distribution.
            bins (list, optional): The bin edges for continuous data if the data needs to be binned.
            bin_labels (list, optional): Labels for the bins if the data needs to be binned.

            plot (bool, optional): Whether to plot the distribution.
            plot_file (str, optional): Path to save the plot as a PNG file.


        Returns:
            pd.Series: The distribution of the column, either as raw counts or percentages.
        """
        # Filter the DataFrame to only include specified item IDs
        if len(item_ids) > 0:
            filtered_df = df.loc[item_ids]

            # Check if the column needs binning
            if bins is not None and pd.api.types.is_numeric_dtype(filtered_df[column_name]):
                # Bin the data if bins are provided and the data type is numeric
                if bin_labels is None or len(bin_labels) != len(bins) - 1:
                    raise ValueError(
                        "Bin labels must be provided and match the number of bins - 1.")
                filtered_df['binned'] = pd.cut(
                    filtered_df[column_name], bins=bins,  right=False, labels=bin_labels)
                # Calculate the distribution of the binned data
                # distribution = filtered_df['binned'].value_counts(
                #     normalize=normalize)
                distribution = filtered_df['binned'].value_counts(
                    normalize=normalize).reindex(bin_labels, fill_value=0)
            else:
                # # Calculate the distribution of the categorical data
                # distribution = filtered_df[column_name].value_counts(
                #     normalize=normalize)
                all_categories = df[column_name].unique()
                distribution = filtered_df[column_name].value_counts(
                    normalize=normalize).reindex(all_categories, fill_value=0)

            if plot:
                distribution.plot(kind='bar', color='skyblue')

                plt.xlabel(column_name)
                plt.ylabel('Proportion')
                plt.title(f'Distribution of {column_name}')
                plt.xticks(rotation=45)
                plt.tight_layout()

                if plot_file:
                    plt.savefig(plot_file)
                    print(f"Plot saved as {plot_file}")
                plt.show()

            return distribution
        else:
            return {}

    def test_category_distribution(self):
        news_data = pd.DataFrame({
            'category': ['Politics', 'Technology', 'Politics', 'Sports', 'Technology', 'Politics', 'Health'],
            'age': [10, 20, 30, 15, 25, 5, 40]  # age in days
        }, index=[1, 2, 3, 4, 5, 6, 7])

        # List of item IDs
        item_ids = [1, 2, 3, 4, 5, 6, 7]
        # Test the distribution for categorical data
        category_distribution = self.calculate_distribution(
            news_data, item_ids, 'category', normalize=False, plot_file="category_distribution.png")
        expected_counts = {'Politics': 3,
                           'Technology': 2, 'Sports': 1, 'Health': 1}
        actual_counts = category_distribution.to_dict()
        self.assertEqual(expected_counts, actual_counts,
                         "Category distributions do not match.")
        # continous value
        bins = [0, 10, 20, 30, 40, 50]
        bin_labels = ['0-9', '10-19', '20-29', '30-39', '40-49']

        # Test the distribution for continuous data
        age_distribution = self.calculate_distribution(
            news_data, item_ids, 'age', normalize=False, bins=bins, bin_labels=bin_labels, plot_file="age_distribution.png")
        expected_counts = {'0-9': 1, '10-19': 2,
                           '20-29': 2, '30-39': 1, '40-49': 1}
        actual_counts = age_distribution.to_dict()
        self.assertEqual(expected_counts, actual_counts,
                         "Age distributions do not match.")

        news_data = pd.DataFrame({
            'category': ['Politics', 'Technology', 'Politics', 'Sports', 'Technology', 'Politics', 'Health'],
            'age': [10, 20, 30, 15, 25, 5, 40]  # age in days
        }, index=[1, 2, 3, 4, 5, 6, 7])

        # List of item IDs
        # This excludes the item with category 'Health'
        item_ids = [1, 2, 3, 4, 5, 6]

        # Test the distribution for categorical data
        category_distribution = self.calculate_distribution(
            news_data, item_ids, 'category', normalize=False, plot_file="category_distribution.png")

        # 'Health' should appear with a count of 0
        expected_counts = {'Politics': 3,
                           'Technology': 2, 'Sports': 1, 'Health': 0}
        actual_counts = category_distribution.to_dict()

        self.assertEqual(expected_counts, actual_counts,
                         "Category distributions do not match.")

    def test__init__(self):
        gr = GreedyKLReranker(name="test_reranker",
                              item_dataframe=self.dummy_item_features, diversity_dimension=self.dummy_diversity_dimension, pool_size=4, top_k=3,

                              target_distributions=self.target_distr_dummy,
                              diversity_dimension_weight=[0.4, 0.6])
        self.assertEqual(gr.diversity_dimension_weight, [0.4, 0.6])
        expected_target_distributions = [{
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
        },            {
            "type": "discrete",
            "distr": {
                    "Politics": 0.25,
                    "Travel": 0.25,
                    "Sports": 0.25,
                    "Entertainment": 0.25,
                    "Technology": 0,
                    "Health": 0,
                    "Finance": 0
            }}]
        self.assertEqual(gr.target_distributions,
                         expected_target_distributions)
        self.assertEqual(gr.top_k, 3)
        self.assertEqual(gr.pool_size, 4)
        assert_frame_equal(gr.item_dataframe, self.dummy_item_features)
        self.assertEqual(gr.name, "test_reranker")

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

    def assertSelectItemByFeature1Category(self, user_idx, ranked_items, selected_size, reranker):
        test_user = user_idx
        reranked_items = reranker.rerank(
            user_idx=test_user, interaction_history=self.ratio_split.train_set, candidate_items=ranked_items, prediction_scores=None)

        selected_items = reranked_items[:selected_size]

        expected_user_history = self.expected_user_history[test_user]

        self.assertEqual(
            reranker.user_history[test_user], expected_user_history)

        # simulate filtering
        remaining_item_ids = [
            x for x in ranked_items if x not in expected_user_history and x not in selected_items]

        if len(remaining_item_ids) > 0:
            expected_distr_feature1 = {'0-0.3': 0.2, '0.3-0.6': 0.3,
                                       '0.6-1.0': 0.5}
            expected_distr_category = {'Politics': 0.25, 'Travel': 0.25,
                                       'Sports': 0.25, 'Entertainment': 0.25}
            expected_diversity_score = []
            for first_candidate in remaining_item_ids:

                p = expected_distr_feature1
                selected = selected_items+[first_candidate]

                q_selected = self.calculate_distribution(self.dummy_item_features, selected, 'feature1',  normalize=True, bins=[
                    0, 0.3, 0.6, 1.0], bin_labels=["0-0.3", "0.3-0.6", "0.6-1.0"], plot=False)
                q_selected = q_selected.to_dict()

                div_feature1_selected = safe_kl_divergence_dicts(p, q_selected)

                p = expected_distr_category

                q_selected = self.calculate_distribution(
                    self.dummy_item_features, selected, 'category',  normalize=True,  plot=False)
                q_selected = q_selected.to_dict()

                div_category_selected = safe_kl_divergence_dicts(p, q_selected)

                expected_diversity_score.append(
                    div_feature1_selected*reranker.diversity_dimension_weight[0] + div_category_selected*reranker.diversity_dimension_weight[1])
            # computed by reranker

            target_distributions = reranker.target_distributions
            aspect_importance = reranker.diversity_dimension_weight
            proportions, matched_items = cornac.utils.common.TargetDistributionMatcher(
                target_distributions, reranker.diversity_dimension, self.dummy_item_features, ranked_items)

            actual_diversity = reranker.diversityScores(remaining_item_ids, selected_items, proportions,
                                                        aspect_importance, matched_items)

            self.assertListAlmostEqual(
                expected_diversity_score, actual_diversity)

            min_index = np.argmin(expected_diversity_score)

            # Get the diversity score for the selected item
            expected_first_item_diversity_score = expected_diversity_score[min_index]

            actual_first_item_diversity_score = actual_diversity[remaining_item_ids.index(
                reranked_items[len(selected_items)])]

            # in case there are multiple values with the smallest divergence
            self.assertAlmostEqual(actual_first_item_diversity_score,
                                   expected_first_item_diversity_score, places=6)
        else:
            self.assertEqual(len(reranked_items), selected_size)

    def test_rerank_items_discrete_continous(self):
        reranker = GreedyKLReranker(
            item_dataframe=self.dummy_item_features, diversity_dimension=self.dummy_diversity_dimension, pool_size=7, top_k=3,
            target_distributions=self.target_distr_dummy,
            diversity_dimension_weight=[0.6, 0.4])
        ranked_items = [2, 3, 4, 5, 6, 0, 1]
        self.assertSelectItemByFeature1Category(
            user_idx=1, ranked_items=ranked_items, selected_size=1, reranker=reranker)

        self.assertSelectItemByFeature1Category(
            user_idx=2, ranked_items=ranked_items, selected_size=0, reranker=reranker)

        self.assertSelectItemByFeature1Category(
            user_idx=6, ranked_items=ranked_items, selected_size=1, reranker=reranker)

        self.assertSelectItemByFeature1Category(
            user_idx=3, ranked_items=[], selected_size=0, reranker=reranker)

    def test__init_select_party_dimension(self):
        gr = GreedyKLReranker(name="test_reranker",
                              item_dataframe=self.dummy_item_features, diversity_dimension=["entities"], top_k=3,

                              target_distributions=self.target_distr_dummy,
                              diversity_dimension_weight=[1.0])
        self.assertEqual(gr.diversity_dimension_weight, [1.0])
        expected_target_distributions = [{"type": "parties",
                                          "distr": [{"description": "only mention", "contain": ["partyA"], "prob": 0.2},
                                                    {"description": "only mention",
                                                     "contain": ["partyB"], "prob": 0.2},
                                                    {"description": "composition", "contain": [
                                                        ["partyB"], ["partyA"]], "prob": 0.2},
                                                    {"description": "minority but can also mention", "contain": [
                                                        "partyA", "partyB"], "prob": 0.2},  # item 2 and item 6
                                                    {"description": "no parties", "contain": [], "prob": 0.2}]}
                                         ]
        self.assertEqual(gr.target_distributions,
                         expected_target_distributions)
        self.assertEqual(gr.top_k, 3)
        self.assertEqual(gr.pool_size, -1)
        assert_frame_equal(gr.item_dataframe, self.dummy_item_features)
        self.assertEqual(gr.name, "test_reranker")

    def calculate_party_distribution(self, df, item_ids, column_name="entities", normalize=True, plot=False, plot_file=None):
        """
        Calculate the distribution of party memberships, limited to given item IDs in a DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            item_ids (list): List of item IDs (indices) to consider in the calculation.
            column_name (str): The column for which to calculate the distribution (default is 'party').
            normalize (bool, optional): Whether to return percentages or raw counts (default is True for percentages).
            plot (bool, optional): Whether to plot the distribution.
            plot_file (str, optional): Path to save the plot as a PNG file.

        Returns:
            pd.Series: The distribution of the column, either as raw counts or percentages.
        """
        # Filter the DataFrame to only include specified item IDs
        filtered_df = df.loc[item_ids]
        # Define party configurations
        all_configurations = {
            "only_partyA": 0,
            "only_partyB": 0,
            "both_partyA_partyB": 0,
            "minority": 0,  # Minority: contains partyA or partyB, but must also contain a different party
            "no_parties": 0
        }

        # Handle the "party" column, which contains sets of parties
        if column_name in df.columns:

           # Count occurrences of each party or no party in the "party" feature
            for parties in filtered_df[column_name]:
                if parties is None:
                    # Treat None as an empty set (i.e., no parties mentioned)
                    parties_set = set()
                else:
                    parties_set = set(parties)
                if len(parties_set) == 0:
                    all_configurations["no_parties"] += 1
                elif parties_set == {"partyA"}:
                    all_configurations["only_partyA"] += 1
                elif parties_set == {"partyB"}:
                    all_configurations["only_partyB"] += 1
                elif parties_set == {"partyA", "partyB"}:
                    all_configurations["both_partyA_partyB"] += 1
                elif len(parties_set - {"partyA", "partyB"}) > 0:
                    # Minority case: must contain either partyA or partyB (or both) AND at least one other party
                    all_configurations["minority"] += 1

            # Convert counts to pandas Series for further manipulation
            distribution = pd.Series(all_configurations)

            # Normalize if required (convert to percentages)
            if normalize:
                distribution = distribution / distribution.sum()

            # Optionally plot the distribution
            if plot:
                distribution.plot(kind='bar', color='skyblue')
                plt.xlabel(column_name)
                plt.ylabel('Proportion' if normalize else 'Count')
                plt.title(f'Distribution of {column_name}')
                plt.xticks(rotation=45)
                plt.tight_layout()

                if plot_file:
                    plt.savefig(plot_file)
                    print(f"Plot saved as {plot_file}")
                plt.show()

            return distribution
        else:
            raise ValueError(
                "This function is specifically for the 'party' column.")

    # Example Unit Test for Party Distribution Calculation

    def test_party_distribution(self):

        # List of item IDs (all items in this case)
        item_ids = [0, 1, 2, 3, 4, 5, 6]

        # Calculate the party distribution
        party_distribution = self.calculate_party_distribution(
            self.dummy_item_features, item_ids, column_name="entities", normalize=False)

        # Expected party counts (raw counts)
        expected_counts = {"only_partyA": 2,
                           "only_partyB": 1,
                           "both_partyA_partyB": 1,
                           "minority": 2,  # Minority: contains partyA or partyB, but must also contain a different party
                           "no_parties": 1}
        actual_counts = party_distribution.to_dict()

        # Assert that the actual counts match the expected counts
        self.assertDictEqual(expected_counts, actual_counts,
                             "Party distributions do not match.")

    def assertSelectItemByPartyCategory(self, user_idx, ranked_items, selected_size, reranker):
        test_user = user_idx
        reranked_items = reranker.rerank(
            user_idx=test_user, interaction_history=self.ratio_split.train_set, candidate_items=ranked_items, prediction_scores=None)

        selected_items = reranked_items[:selected_size]

        expected_user_history = self.expected_user_history[test_user]

        self.assertEqual(
            reranker.user_history[test_user], expected_user_history)

        # simulate filtering
        remaining_item_ids = [
            x for x in ranked_items if x not in expected_user_history and x not in selected_items]

        if len(remaining_item_ids) > 0:
            expected_distr_party = {"only_partyA": 0.2,
                                    "only_partyB": 0.2,
                                    "both_partyA_partyB": 0.2,
                                    "minority": 0.2,
                                    "no_parties": 0.2}
            expected_distr_category = {'Politics': 0.25, 'Travel': 0.25,
                                       'Sports': 0.25, 'Entertainment': 0.25}
            expected_diversity_score = []
            for first_candidate in remaining_item_ids:

                p = expected_distr_party
                selected = selected_items+[first_candidate]

                q_selected = self.calculate_party_distribution(
                    self.dummy_item_features, selected, column_name="entities", normalize=True)
                q_selected = q_selected.to_dict()

                div_feature1_selected = safe_kl_divergence_dicts(p, q_selected)

                p = expected_distr_category

                q_selected = self.calculate_distribution(
                    self.dummy_item_features, selected, 'category',  normalize=True,  plot=False)
                q_selected = q_selected.to_dict()

                div_category_selected = safe_kl_divergence_dicts(p, q_selected)

                expected_diversity_score.append(
                    div_feature1_selected*reranker.diversity_dimension_weight[0] + div_category_selected*reranker.diversity_dimension_weight[1])
            # computed by reranker

            target_distributions = reranker.target_distributions
            aspect_importance = reranker.diversity_dimension_weight
            proportions, matched_items = cornac.utils.common.TargetDistributionMatcher(
                target_distributions, reranker.diversity_dimension, self.dummy_item_features, ranked_items)

            actual_diversity = reranker.diversityScores(remaining_item_ids, selected_items, proportions,
                                                        aspect_importance, matched_items)

            self.assertListAlmostEqual(
                expected_diversity_score, actual_diversity)

            min_index = np.argmin(expected_diversity_score)

            # Get the diversity score for the selected item
            expected_first_item_diversity_score = expected_diversity_score[min_index]

            actual_first_item_diversity_score = actual_diversity[remaining_item_ids.index(
                reranked_items[len(selected_items)])]

            # in case there are multiple values with the smallest divergence
            self.assertAlmostEqual(actual_first_item_diversity_score,
                                   expected_first_item_diversity_score,  places=6)
        else:
            self.assertEqual(len(reranked_items), selected_size)

    def test_rerank_party_data(self):
        reranker = GreedyKLReranker(
            item_dataframe=self.dummy_item_features, diversity_dimension=["entities", "category"], pool_size=7, top_k=3,
            target_distributions=self.target_distr_dummy,
            diversity_dimension_weight=[0.5, 0.5])
        ranked_items = [1, 0, 4, 5, 6, 2, 3]
        self.assertSelectItemByPartyCategory(
            user_idx=0, ranked_items=ranked_items, selected_size=1, reranker=reranker)

        self.assertSelectItemByPartyCategory(
            user_idx=3, ranked_items=ranked_items, selected_size=0, reranker=reranker)

        self.assertSelectItemByPartyCategory(
            user_idx=5, ranked_items=ranked_items, selected_size=1, reranker=reranker)

        self.assertSelectItemByPartyCategory(
            user_idx=4, ranked_items=[], selected_size=0, reranker=reranker)


if __name__ == '__main__':
    unittest.main()
