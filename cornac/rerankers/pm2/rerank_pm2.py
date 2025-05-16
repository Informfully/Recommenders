import numpy as np
from ..reranker import ReRanker
from ...exception import ScoreException
from ...utils.common import TargetDistributionMatcher, count_selected_in_aspects, get_max_keys
import pandas as pd


class PM2Reranker(ReRanker):
    """PM2Reranker: A diversity-aware re-ranking model based on the PM-2 algorithm.


    """

    def __init__(self, name="PM2", item_dataframe=None, diversity_dimension=None, top_k=10, pool_size=-1, config_file=None,
                 target_distributions=None,
                 diversity_dimension_weight=None,
                 user_item_history = None, rerankers_item_pool = None # for mind setup
                 ):
        """
        Initialize the PM2Reranker with the specified configurations.

        Parameters:
        - `name` (str): Name of the reranker (default: "PM2").
        - `item_dataframe` (pd.DataFrame): DataFrame containing item metadata.
        - `diversity_dimension` (list): Attributes representing diversity dimensions.
        - `top_k` (int): Number of items to recommend in the ranked list (default: 10).
        - `pool_size` (int): Size of the item pool (default -1 to include all items).
        - `config_file` (str): Path to configuration file for target distributions.
            If provided, target distributions and diversity dimension weights are loaded from the file.
        - `target_distributions` (dict): Target distributions for each diversity dimension.
            Required if no configuration file is provided.
        - `diversity_dimension_weight` (list): Weights for each diversity dimension to influence scoring.

        Initializes:
        - Target distributions and diversity dimension weights (from configuration file or parameters).
        - Validates the diversity dimensions and target distributions.
        """

        super().__init__(name=name, item_dataframe=item_dataframe, diversity_dimension=diversity_dimension,
                         top_k=top_k,  pool_size=pool_size, user_item_history = user_item_history, rerankers_item_pool = rerankers_item_pool)
        

        # self.user_item_history = user_item_history
        # self.rerankers_item_pool = rerankers_item_pool

        if config_file is not None:
            self.configReranker(config_file)
        else:
            self.target_distributions = self._setup_selected_distribution(
                target_distributions)
            self.diversity_dimension_weight = diversity_dimension_weight
        self._validate_distribution_input()

    def _setup_selected_distribution(self, target_distributions):
        """
        Sets up selected distribution from target distributions based on the diversity dimension.
        Raises an error if any diversity dimension is not covered by the target distribution.
        """
        if self.diversity_dimension is None or target_distributions is None:
            raise ValueError(
                "Diversity dimensions and target distributions must not be None.")

        selected_distr = []
        for dimension in self.diversity_dimension:
            if dimension not in target_distributions:
                raise ValueError(
                    f"Target distribution for diversity dimension '{dimension}' does not exist!")
            selected_distr.append(target_distributions[dimension])

        return selected_distr

    def diversityScores(self, remaining_items, selected_items, target_distributions, dimension_importance, dimension_aspect_item_mapping, lamda=0.8):
        """
        Compute diversity scores for candidate items using the PM2 algorithm.

        This method calculates diversity scores by computing quotients for each aspect, which are derived 
        from the target distributions and current seat counts of selected items. It then evaluates the 
        contributions from the most prominent aspects (max keys) and the non-prominent aspects. 
        The balance between these contributions is controlled by the `lamda` parameter.

        Parameters:
        - `remaining_items` (list): List of candidate items not yet selected.
        - `selected_items` (list): List of items already selected.
        - `target_distributions` (list): Target distributions for each diversity dimension.
        - `dimension_importance` (list): Weights for each diversity dimension.
        - `dimension_aspect_item_mapping` (list): Mapping of items to aspects for each diversity dimension.
        - `lamda` (float): Weight balancing the contributions of prominent and non-prominent aspects (default: 0.8).

        Returns:
        - `diversity_values` (np.ndarray): Array of diversity scores for the remaining items.

        """
        # Initialize aspect counts for selected items
        seat_counts = count_selected_in_aspects(
            selected_items, dimension_aspect_item_mapping)

        remaining_items = np.array(remaining_items)
        # Initialize diversity values with zeros
        diversity_values = np.zeros(len(remaining_items))

        # Calculate quotients for each dimension
        for i, target_distribution in enumerate(target_distributions):
            aspects = dimension_aspect_item_mapping[i]
            # Ensure keys are ordered correctly
            ordered_keys = list(aspects.keys())

            quotient_matrix = np.array(
                [target_distribution[key] / (2 * seat_counts[i][key] + 1) for key in ordered_keys])

            # Find indices of max keys (handles ties for max quotients)
            max_key_indices = np.where(
                quotient_matrix == quotient_matrix.max())[0]
            # Vectorized way to count occurrences of remaining_items in aspects
            aspect_membership_matrix = np.zeros(
                (len(remaining_items), len(aspects)))

            for aspect_idx, key in enumerate(ordered_keys):
                indices = aspects[key]
                aspect_membership_matrix[:, aspect_idx] = np.isin(
                    remaining_items, indices)

            # Calculate max key contribution (for items belonging to max key aspects)
            max_key_contribution = lamda * np.sum(
                aspect_membership_matrix[:, max_key_indices] *
                quotient_matrix[max_key_indices] * dimension_importance[i],
                axis=1)

            # Compute non-max key contributions for all other aspects

            # Exclude the max key indices to get the non-max columns
            non_max_key_indices = np.setdiff1d(
                np.arange(len(ordered_keys)), max_key_indices)
            non_max_key_contribution = (1 - lamda) * np.sum(
                aspect_membership_matrix[:, non_max_key_indices] *
                quotient_matrix[non_max_key_indices] * dimension_importance[i],
                axis=1
            )

            # Add the contribution from this dimension to the overall diversity score
            diversity_values += max_key_contribution + non_max_key_contribution

        return diversity_values

    def rerank(self, user_idx, interaction_history=None, candidate_items=None, prediction_scores=None, filtering_rules: dict = None, **kwargs):
        """Re-rank candidate items based on diversity scores computed using the PM-2 algorithm.

        Parameters
        ----------
        - `user_idx` (int): ID of the user receiving recommendations.
        - `interaction_history` (cornac.data.Dataset): User-Item preference data for filtering previously interacted items.
        - `candidate_items` (list): List of items to be re-ranked.
        - `prediction_scores` (list): Prediction scores from the base recommender model (optional).
        - `filtering_rules` (dict): Filtering rules to refine candidate items (optional).
    

        Returns
        -------
        selected: Re-ranked list of items, limited to the top-k items based on diversity scores.
        """
        super().rerank(user_idx=user_idx, interaction_history=interaction_history, candidate_items=candidate_items,
                        prediction_scores=prediction_scores,  **kwargs)
        aspects_prop, aspects_items = TargetDistributionMatcher(
            self.target_distributions, self.diversity_dimension, self.item_dataframe, candidate_items)
        self.execute_filters(user_idx, filtering_rules)
        self.filter_items_in_additional_history(user_idx)  ## for Mind setup

        candidate_items = self.candidate_items[user_idx]

        selected = []
        # Boolean mask to track remaining items
        remaining_mask = np.ones(len(candidate_items), dtype=bool)

        while len(selected) < self.top_k and np.any(remaining_mask):

            remaining_items = np.array(candidate_items)[remaining_mask]
            if len(remaining_items) == 0:
                break
            scores = self.diversityScores(
                remaining_items, selected, aspects_prop, self.diversity_dimension_weight, aspects_items, lamda=0.8)
            # Select the item with the highest diversity score
            next_item_idx = np.argmax(scores)
            next_item_id = remaining_items[next_item_idx]

            # Add the selected item to the list
            selected.append(next_item_id)

            # Update remaining mask: mark the selected item as unavailable
            remaining_mask[candidate_items.index(next_item_id)] = False

        # assert len(selected) == self.top_k, f"Expected {self.top_k} items, but got {len(selected)}" # when candidate list is small, and target size cannot be achieved
        assert len(selected) == len(set(selected)), "Duplicate items found in selected"
        # cache the re-ranked list for a user
        self.ranked_items[user_idx] = selected
        return selected
