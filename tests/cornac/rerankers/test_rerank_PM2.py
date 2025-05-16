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
from cornac.rerankers.pm2 import PM2Reranker
from cornac.datasets import mind as mind
from pandas.testing import assert_frame_equal
import matplotlib.pyplot as plt
from cornac.utils.common import safe_kl_divergence
import random
from cornac.utils.common import get_max_keys


class TestRerank(unittest.TestCase):
    def setUp(self):
        # feedback = mind.load_feedback(
        #     fpath="./tests/enriched_data/mind_uir_20k.csv")
        # self.mind_feedback = feedback[:5000]
        # self.mind_sentiment = mind.load_sentiment(
        #     fpath="./tests/enriched_data/sentiment.json")
        # self.mind_category = mind.load_category(
        #     fpath="./tests/enriched_data/category.json")
        self.target_distr_mind = {
            "category": {
                "type": "discrete",
                "distr": {
                    "entertainment": 0.1,
                    "sports": 0.5,
                    "finance": 0.2,
                    "health": 0.2
                }
            },
            "sentiment": {
                "type": "continuous",
                "distr": [
                    {
                        "min": -1,
                        "max": -0.5,
                        "prob": 0.2
                    },
                    {
                        "min": -0.5,
                        "max": 0,
                        "prob": 0.3
                    },
                    {
                        "min": 0,
                        "max": 0.5,
                        "prob": 0.3
                    },
                    {
                        "min": 0.5,
                        "max": 1,
                        "prob": 0.2
                    }
                ]
            },
            "complexity": {
                "type": "continuous",
                "distr": [
                    {
                        "min": 0,
                        "max": 10,
                        "prob": 0.2
                    },
                    {
                        "min": 10,
                        "max": 20,
                        "prob": 0.3
                    },
                    {
                        "min": 20,
                        "max": 30,
                        "prob": 0.3
                    },
                    {
                        "min": 30,
                        "max": 100,
                        "prob": 0.2
                    }
                ]
            },
            "minority_proportion": {
                "type": "continuous",
                "distr": [
                    {
                        "min": 0,
                        "max": 0.5,
                        "prob": 0.5
                    },
                    {
                        "min": 0.5,
                        "max": 1.01,
                        "prob": 0.5
                    }
                ]
            }
        }
        self.dummy_diversity_dimension = ["feature1", "category"]
        self.dummy_item_features = pd.DataFrame({
            'feature1': [0.1, 0.3, 0.2, 0.5, 0.8, 0.6, 0.7],
            'feature2': [0.6, 0.8, 0.7, 0.2, 0.2, 0.1, 0.05],
            'feature3': [0.5, 0.2, 0.9, 0.4, 0.1, 0.4, 0.3],
            'category': [
                "Politics", "Finance", "Sports", "Health", "Entertainment", "Finance", "Sports"
            ],
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
                                       ["partyB"], ["partyA"]], "prob": 0.2},
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
            category_mapping: the item ids that corresponds to each category in the full df. (not limited to item_ids)
        """
        # Filter the DataFrame to only include specified item IDs
        # Check if the column needs binning
        category_mapping = {}
        if bins is not None and pd.api.types.is_numeric_dtype(df[column_name]):
            # Bin the data if bins are provided and the data type is numeric
            if bin_labels is None or len(bin_labels) != len(bins) - 1:
                raise ValueError(
                    "Bin labels must be provided and match the number of bins - 1.")
            df['binned'] = pd.cut(df[column_name], bins=bins,
                                  right=False, labels=bin_labels)
            # Map item IDs to bins for all items
            for label in bin_labels:
                category_mapping[label] = df[df['binned']
                                             == label].index.tolist()
        else:
            # Map item IDs to categories for all items
            for category in df[column_name].unique():
                category_mapping[category] = df[df[column_name]
                                                == category].index.tolist()

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
                distribution = filtered_df['binned'].value_counts(
                    normalize=normalize).reindex(bin_labels, fill_value=0)

            else:
                # Calculate the distribution of the categorical data
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

            return distribution, category_mapping
        else:
            return {}, category_mapping

    def test_category_distribution(self):
        news_data = pd.DataFrame({
            'category': ['Politics', 'Technology', 'Politics', 'Sports', 'Technology', 'Politics', 'Health'],
            'age': [10, 20, 30, 15, 25, 5, 40]  # age in days
        }, index=[1, 2, 3, 4, 5, 6, 7])

        # List of item IDs
        item_ids = [1, 2, 3, 4, 5, 6, 7]
        # Test the distribution for categorical data
        category_distribution, item_mapping = self.calculate_distribution(
            news_data, item_ids, 'category', normalize=False, plot_file="category_distribution.png")
        expected_counts = {'Politics': 3,
                           'Technology': 2, 'Sports': 1, 'Health': 1}
        actual_counts = category_distribution.to_dict()
        self.assertEqual(expected_counts, actual_counts,
                         "Category distributions do not match.")
        expected_mapping = {'Politics': [1, 3, 6], 'Technology': [2, 5],
                            'Sports': [4], 'Health': [7]}
        self.assertDictEqual(item_mapping, expected_mapping)
        # continous value
        bins = [0, 10, 20, 30, 40, 50]
        bin_labels = ['0-9', '10-19', '20-29', '30-39', '40-49']

        # Test the distribution for continuous data
        age_distribution, item_mapping = self.calculate_distribution(
            news_data, item_ids, 'age', normalize=False, bins=bins, bin_labels=bin_labels)
        expected_counts = {'0-9': 1, '10-19': 2,
                           '20-29': 2, '30-39': 1, '40-49': 1}
        actual_counts = age_distribution.to_dict()
        self.assertEqual(expected_counts, actual_counts,
                         "Age distributions do not match.")
        expected_mapping = {'0-9': [6], '10-19': [1, 4],
                            '20-29': [2, 5], '30-39': [3], '40-49': [7]}
        self.assertDictEqual(item_mapping, expected_mapping)

        # This excludes the item with category 'Health'
        item_ids = [1, 2, 3, 4, 5, 6]

        # Test the distribution for categorical data
        category_distribution, item_mapping = self.calculate_distribution(
            news_data, item_ids, 'category', normalize=False)

        # 'Health' should appear with a count of 0
        expected_counts = {'Politics': 3,
                           'Technology': 2, 'Sports': 1, 'Health': 0}
        actual_counts = category_distribution.to_dict()

        self.assertEqual(expected_counts, actual_counts,
                         "Category distributions do not match.")

    def test__init__(self):
        pm2 = PM2Reranker(name="test_reranker",
                          item_dataframe=self.dummy_item_features, diversity_dimension=self.dummy_diversity_dimension, pool_size=4, top_k=3,

                          target_distributions=self.target_distr_dummy,
                          diversity_dimension_weight=[0.4, 0.6])

        self.assertEqual(pm2.diversity_dimension_weight, [0.4, 0.6])
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
        self.assertEqual(pm2.target_distributions,
                         expected_target_distributions)
        self.assertEqual(pm2.top_k, 3)
        self.assertEqual(pm2.pool_size, 4)
        assert_frame_equal(pm2.item_dataframe, self.dummy_item_features)
        self.assertEqual(pm2.name, "test_reranker")

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
                                       'Sports': 0.25, 'Entertainment': 0.25,
                                       "Technology": 0,
                                       "Health": 0,
                                       "Finance": 0}
            diversity_score = []
            p_feature1 = expected_distr_feature1
            q_feature1_selected, item_mapping_feature1 = self.calculate_distribution(self.dummy_item_features, selected_items, 'feature1',  normalize=False, bins=[
                0, 0.3, 0.6, 1.0], bin_labels=["0-0.3", "0.3-0.6", "0.6-1.0"], plot=False)

            p_category = expected_distr_category
            q_category_selected, item_mapping_category = self.calculate_distribution(
                self.dummy_item_features, selected_items, 'category',  normalize=False,  plot=False)

            quotients_feature1 = {}
            for key, props in p_feature1.items():
                if key in q_feature1_selected:
                    q = props/(2*q_feature1_selected[key]+1)
                else:
                    q = props/1
                quotients_feature1[key] = q
            quotients_category = {}
            for key, props in p_category.items():
                if key in q_category_selected:
                    q = props/(2*q_category_selected[key]+1)
                else:
                    q = props/1
                quotients_category[key] = q
            max_key_quotients_category = get_max_keys(
                quotients_category)
            max_key_quotients_feature1 = get_max_keys(
                quotients_feature1)
            lamda = 0.8
            for id in remaining_item_ids:
                div_score = 0
                for key, item_indices in item_mapping_feature1.items():

                    if id in item_indices and key in max_key_quotients_feature1:
                        div_score += lamda * \
                            quotients_feature1[key] * \
                            reranker.diversity_dimension_weight[0]
                    elif id in item_indices and key not in max_key_quotients_feature1:
                        if key in quotients_feature1:
                            div_score += (1-lamda) * \
                                quotients_feature1[key] * \
                                reranker.diversity_dimension_weight[0]
                for key, item_indices in item_mapping_category.items():
                    if id in item_indices and key in max_key_quotients_category:
                        div_score += lamda * \
                            quotients_category[key] * \
                            reranker.diversity_dimension_weight[1]
                    elif id in item_indices and key not in quotients_category:
                        if key in quotients_category:
                            div_score += (1-lamda) * \
                                quotients_category[key] * \
                                reranker.diversity_dimension_weight[1]

                diversity_score.append(div_score)

            target_distributions = reranker.target_distributions
            aspect_importance = reranker.diversity_dimension_weight
            proportions, matched_items = cornac.utils.common.TargetDistributionMatcher(
                target_distributions, reranker.diversity_dimension, self.dummy_item_features, ranked_items)

            actual_diversity = reranker.diversityScores(remaining_item_ids, selected_items, proportions,
                                                        aspect_importance, matched_items)
            self.assertListAlmostEqual(diversity_score, actual_diversity)
            first_item_id = remaining_item_ids[np.argmax(diversity_score)]
            self.assertEqual(
                reranked_items[len(selected_items)], first_item_id)
        else:
            self.assertEqual(len(reranked_items), selected_size)

    def test_pm2_rerank(self):
        reranker = PM2Reranker(
            item_dataframe=self.dummy_item_features, diversity_dimension=self.dummy_diversity_dimension, pool_size=7, top_k=3,
            target_distributions=self.target_distr_dummy,
            diversity_dimension_weight=[0.6, 0.4])

        self.assertSelectItemByFeature1Category(
            user_idx=1, ranked_items=[0, 1, 2, 4, 3, 6, 5], selected_size=2, reranker=reranker)

        reranker = PM2Reranker(
            item_dataframe=self.dummy_item_features, diversity_dimension=self.dummy_diversity_dimension, pool_size=7, top_k=3,
            target_distributions=self.target_distr_dummy)

        self.assertSelectItemByFeature1Category(
            user_idx=3, ranked_items=[0, 1, 2, 4, 3, 6, 5], selected_size=1, reranker=reranker)

        self.assertSelectItemByFeature1Category(
            user_idx=2, ranked_items=[0, 1, 2, 4, 3, 6, 5], selected_size=0, reranker=reranker)

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
            party_mapping: the item ids that correspond to each party memberships in the full df (not limited to item_ids).
        """

        filtered_df = df.loc[item_ids]
        if column_name not in df.columns:
            raise ValueError(
                f"Column '{column_name}' does not exist in the DataFrame.")
        # Define party configurations
        all_configurations = {
            "only_partyA": 0,
            "only_partyB": 0,
            "both_partyA_partyB": 0,
            "minority": 0,  # Minority: contains partyA or partyB, but must also contain a different party
            "no_parties": 0
        }
        party_mapping = {
            "only_partyA": [],
            "only_partyB": [],
            "both_partyA_partyB": [],
            "minority": [],
            "no_parties": []
        }

        for idx, parties in df[column_name].items():
            if parties is None:
                # Treat None as an empty set (i.e., no parties mentioned)
                parties_set = set()
            else:
                parties_set = set(parties)

            if len(parties_set) == 0:
                party_mapping["no_parties"].append(idx)
            elif parties_set == {"partyA"}:
                party_mapping["only_partyA"].append(idx)
            elif parties_set == {"partyB"}:
                party_mapping["only_partyB"].append(idx)
            elif parties_set == {"partyA", "partyB"}:
                party_mapping["both_partyA_partyB"].append(idx)
            elif len(parties_set - {"partyA", "partyB"}) > 0:
                party_mapping["minority"].append(idx)
        # Handle the "party" column, which contains sets of parties
        for idx, parties in filtered_df[column_name].items():
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

        return distribution, party_mapping

    def test_party_distribution(self):

        # List of item IDs (all items in this case)
        item_ids = [0, 1, 2, 3, 4, 5, 6]

        # Calculate the party distribution
        party_distribution, party_mapping = self.calculate_party_distribution(
            self.dummy_item_features, item_ids, column_name="entities", normalize=False)

        expected_counts = {"only_partyA": 2,
                           "only_partyB": 1,
                           "both_partyA_partyB": 1,
                           "minority": 2,  # Minority: contains partyA or partyB, but must also contain a different party
                           "no_parties": 1}
        actual_counts = party_distribution.to_dict()

        # Assert that the actual counts match the expected counts
        self.assertDictEqual(expected_counts, actual_counts,
                             "Party distributions do not match.")

        expected_mapping = {"only_partyA": [1, 4],
                            "only_partyB": [5],
                            "both_partyA_partyB": [0],
                            # Minority: contains partyA or partyB, but must also contain a different party
                            "minority": [2, 6],
                            "no_parties": [3]}
        self.assertDictEqual(expected_mapping, party_mapping,
                             "Party mappings do not match.")

    def assertSelectItem_ByParty_Feature2(self, user_idx, ranked_items, selected_size, reranker):
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
            expected_distr_feature2 = {'0-0.3': 0.3, '0.3-0.6': 0.4,
                                       '0.6-1.0': 0.3}
            expected_distr_party = {"only_partyA": 0.2,
                                    "only_partyB": 0.2,
                                    "both_partyA_partyB": 0.2,
                                    "minority": 0.2,
                                    "no_parties": 0.2}

            diversity_score = []
            p_feature2 = expected_distr_feature2
            q_feature2_selected, item_mapping_feature2 = self.calculate_distribution(self.dummy_item_features, selected_items, 'feature2',  normalize=False, bins=[
                0, 0.3, 0.6, 1.0], bin_labels=["0-0.3", "0.3-0.6", "0.6-1.0"], plot=False)

            p_party = expected_distr_party
            q_party_selected, item_mapping_party = self.calculate_party_distribution(
                self.dummy_item_features, selected_items, 'entities',  normalize=False,  plot=False)

            quotients_feature2 = {}
            for key, props in p_feature2.items():
                if key in q_feature2_selected:
                    q = props/(2*q_feature2_selected[key]+1)
                else:
                    q = props/1
                quotients_feature2[key] = q
            quotients_party = {}
            for key, props in p_party.items():
                if key in q_party_selected:
                    q = props/(2*q_party_selected[key]+1)
                else:
                    q = props/1
                quotients_party[key] = q
            max_key_quotients_party = get_max_keys(
                quotients_party)
            max_key_quotients_feature2 = get_max_keys(
                quotients_feature2)
            lamda = 0.8

            for id in remaining_item_ids:
                div_score = 0
                for key, item_indices in item_mapping_feature2.items():

                    if id in item_indices and key in max_key_quotients_feature2:
                        div_score += lamda * \
                            quotients_feature2[key] * \
                            reranker.diversity_dimension_weight[0]
                    elif id in item_indices and key not in max_key_quotients_feature2:
                        if key in quotients_feature2:
                            div_score += (1-lamda) * \
                                quotients_feature2[key] * \
                                reranker.diversity_dimension_weight[0]

                for key, item_indices in item_mapping_party.items():
                    if id in item_indices and key in max_key_quotients_party:
                        div_score += lamda * \
                            quotients_party[key] * \
                            reranker.diversity_dimension_weight[1]
                    elif id in item_indices and key not in max_key_quotients_party:
                        if key in quotients_party:
                            div_score += (1-lamda) * \
                                quotients_party[key] * \
                                reranker.diversity_dimension_weight[1]

                diversity_score.append(div_score)
            # computed by reranker

            target_distributions = reranker.target_distributions
            aspect_importance = reranker.diversity_dimension_weight
            proportions, matched_items = cornac.utils.common.TargetDistributionMatcher(
                target_distributions, reranker.diversity_dimension, self.dummy_item_features, ranked_items)

            actual_diversity = reranker.diversityScores(remaining_item_ids, selected_items, proportions,
                                                        aspect_importance, matched_items)

            self.assertListAlmostEqual(diversity_score, actual_diversity)
            first_item_id = remaining_item_ids[np.argmax(diversity_score)]
            self.assertEqual(
                reranked_items[len(selected_items)], first_item_id)
        else:
            self.assertEqual(len(reranked_items), selected_size)

    def test_rerank_party_feature2(self):
        reranker = PM2Reranker(
            item_dataframe=self.dummy_item_features, diversity_dimension=["feature2", "entities"], pool_size=7, top_k=5,
            target_distributions=self.target_distr_dummy,
            diversity_dimension_weight=[0.7, 0.3])
        ranked_items = [1, 2, 6, 4, 0, 5, 3]
        self.assertSelectItem_ByParty_Feature2(
            user_idx=0, ranked_items=ranked_items, selected_size=3, reranker=reranker)

        self.assertSelectItem_ByParty_Feature2(
            user_idx=3, ranked_items=ranked_items, selected_size=0, reranker=reranker)

        self.assertSelectItem_ByParty_Feature2(
            user_idx=5, ranked_items=ranked_items, selected_size=1, reranker=reranker)

    def test_rerank(self):
        item_features = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.2, 0.5, 0.8, 0.6, 0.7],
            'feature2': [0.6, 0.8, 0.7, 0.2, 0.2, 0.1, 0.05],
            'feature3': [0.5, 0.2, 0.9, 0.4, 0.1, 0.4, 0.3]
        }, index=[0, 1, 2, 3, 4, 5, 6])
        data1 = [('u0', 'item0', 3.0),
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
                 ('u1', 'item6', 1.0)]
        ratio_split = RatioSplit(
            data=data1, test_size=3, exclude_unknowns=True, verbose=True, seed=123, rating_threshold=1)
        model = UserKNN(k=3, similarity="pearson", name="UserKNN")
        Experiment(eval_method=ratio_split,
                   models=[
                       model
                   ],
                   metrics=[cornac.metrics.Recall()],
                   user_based=True,
                   ).run()
        test_item_ids_list = np.arange(
            ratio_split.test_set.num_items).tolist()
        test_user = list(ratio_split.test_set.uir_tuple[0])[0]
        # test_user = list(ratio_split.test_set.user_indices)[0]
        ranked_items, itemscores = model.rank(
            test_user, test_item_ids_list)

        item_ids = ranked_items.tolist()
        # itemscores = ranked_scores
        # Specify the feature columns to use
        feature_columns = ['feature1']

        feature1_100_reranker = PM2Reranker(
            item_dataframe=item_features, diversity_dimension=feature_columns, pool_size=4, top_k=3,
            target_distributions=self.target_distr_dummy
        )
        items = feature1_100_reranker.rerank(
            user_idx=test_user, interaction_history=ratio_split.train_set, candidate_items=item_ids, prediction_scores=itemscores)

        first_item_feature1 = item_features.loc[items[0], 'feature1']
        self.assertGreaterEqual(first_item_feature1, 0.6)
        self.assertLess(first_item_feature1, 1.01)
        first_item_feature1 = item_features.loc[items[0], 'feature1']
        second_item_feature1 = item_features.loc[items[1], 'feature1']
        self.assertGreaterEqual(second_item_feature1, 0.3)
        self.assertLess(second_item_feature1, 0.6)
        third_item_feature1 = item_features.loc[items[2], 'feature1']
        self.assertGreaterEqual(third_item_feature1, 0)
        self.assertLess(third_item_feature1, 0.3)

    # def test_with_mind(self):
    #     mind_ratio_split = RatioSplit(data=self.mind_feedback, test_size=0.2, exclude_unknowns=True, verbose=True, seed=123
    #                                   )
    #     user_knn_cosine = UserKNN(
    #         k=10, similarity="cosine", name="UserKNN-Cosine")
    #     target_dim = ["category", "sentiment"]
    #     # prepare data
    #     Item_sentiment = mind.build(
    #         data=self.mind_sentiment, id_map=mind_ratio_split.train_set.iid_map)

    #     Item_category = mind.build(
    #         data=self.mind_category, id_map=mind_ratio_split.train_set.iid_map)
    #     out_pd = (pd.Series(Item_category).to_frame('category')
    #                 .join(pd.Series(Item_sentiment).to_frame('sentiment'), how='outer'))

    #     reranker = PM2Reranker(
    #         item_dataframe=out_pd, diversity_dimension=target_dim, top_k=10, pool_size=500,
    #         target_distributions=self.target_distr_mind, diversity_dimension_weight=[0.6, 0.4])
    #     Experiment(eval_method=mind_ratio_split,
    #                models=[
    #                    user_knn_cosine,
    #                ],
    #                metrics=[cornac.metrics.Recall()],
    #                user_based=True,
    #                rerankers={"static": [reranker]}
    #                ).run()

    #     test_items = np.arange(
    #         mind_ratio_split.test_set.num_items).tolist()
    #     test_user = list(mind_ratio_split.test_set.uir_tuple[0])[0]
    #     # test_user = list(mind_ratio_split.test_set.user_indices)[0]
    #     item_ids, itemscores = user_knn_cosine.rank(
    #         test_user, test_items)
    #     item_ids = item_ids.tolist()
    #     itemscores = itemscores.tolist()

    #     items = reranker.rerank(
    #         user_idx=test_user, interaction_history=mind_ratio_split.train_set, candidate_items=item_ids,
    #         prediction_scores=itemscores)


if __name__ == '__main__':
    unittest.main()
