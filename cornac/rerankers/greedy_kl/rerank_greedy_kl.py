import numpy as np
from ..reranker import ReRanker
from ...utils.common import TargetDistributionMatcher, count_selected_in_aspects
from scipy.stats import entropy


class GreedyKLReranker(ReRanker):
    """GreedyKLReranker. Item are re-ranked based on target distribution.

    This class implements a diversity-aware re-ranking mechanism using the Kullback-Leibler (KL) divergence. 
    It aims to minimize the divergence between the actual distributions of selected items and target distributions 
    defined for specific diversity dimensions.
    """

    def __init__(self, name="GreedyKL",  item_dataframe=None, diversity_dimension=None, top_k=10, pool_size=-1, config_file=None,
                 target_distributions=None,
                 diversity_dimension_weight=None,
                user_item_history = None, rerankers_item_pool = None # for news setup
                 ):
        """
        Initialize the GreedyKLReranker with the provided configurations.

        The initialization process first checks for a configuration file (`target_distribution_config_file`). 
        If the configuration file is not provided, the customization parameters (`target_distributions` 
        and `diversity_dimension_weight`) must be supplied.

        Parameters:
        - `name` (str): Name of the reranker (default: "GreedyKL").
        - `item_dataframe` (pd.DataFrame): DataFrame containing item metadata.
        - `diversity_dimension` (list): Attributes representing diversity dimensions.
        - `top_k` (int): Number of items to recommend (default: 10).
        - `pool_size` (int): Size of the item pool (-1 to include all items).
        - `config_file` (str): Path to configuration file for target distributions.
            If provided, the target distributions and diversity dimension weights are loaded from the file.
        - `target_distributions` (dict): Dictionary defining target distributions for each diversity dimension.
            Required if no configuration file is provided.
        - `diversity_dimension_weight` (list): Weights for each diversity dimension to influence scoring.
            Required if no configuration file is provided.

        Initializes:
        - Target distributions and dimension weights (from configuration file or customization parameters).
        - Validates the inputs.
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
        Sets up selected distribution from target distribution based on the diversity dimension.
        Raises an error if any diversity dimension is not covered by the target distribution.
        """
        if self.diversity_dimension is None or target_distributions is None:
            raise ValueError(
                "Diversity dimensions and target distribution must not be None.")

        selected_distr = []
        for dimension in self.diversity_dimension:
            if dimension not in target_distributions:
                raise ValueError(
                    f"Target distribution for diversity dimension '{dimension}' does not exist!")
            selected_distr.append(target_distributions[dimension])

        return selected_distr

    def diversityScores(self, remaining_items, selected_items, target_distributions, dimension_importance, dimension_aspect_item_mapping, alpha=0.01, epsilon=1e-10):
        """
        Compute diversity scores for candidate items based on KL divergence.

        This method calculates diversity scores by iteratively updating distributions for selected items 
        and comparing them to the target distributions.

        Parameters:
        - `remaining_items` (list): List of candidate items not yet selected.
        - `selected_items` (list): List of items already selected.
        - `target_distributions` (list): List of target distributions for each diversity dimension.
        - `dimension_importance` (list): Weights for each diversity dimension.
        - `dimension_aspect_item_mapping` (list): Mapping of items to aspects for each diversity dimension.
        - `alpha` (float): Calibration parameter for blending current and target distributions (default: 0.01).
        - `epsilon` (float): Small value to prevent division by zero (default: 1e-10).

        Returns:
        - `diversity_scores` (np.ndarray): Array of diversity scores for the remaining items.
        """

        # Step 1: Count the selected items in the aspects

        aspect_counts = count_selected_in_aspects(
            selected_items, dimension_aspect_item_mapping)

        # Convert selected_items and remaining_items to NumPy arrays
        remaining_items = np.array(remaining_items)

        # Prepare a list to store diversity scores for remaining items
        diversity_scores = np.zeros(len(remaining_items))

        # Step 2: Iterate over each target distribution

        for j, target_distribution in enumerate(target_distributions):

            aspects = dimension_aspect_item_mapping[j]

        # Step 3: Vectorized way to count occurrences of remaining_items in aspects

            aspect_count_matrix = np.zeros(
                (len(remaining_items), len(aspects)))

            # Get ordered keys from aspects
            ordered_keys = list(aspects.keys())

            # Build a matrix where each row corresponds to a remaining_item
            # and columns correspond to aspect membership
            for aspect_idx, key in enumerate(ordered_keys):
                indices = aspects[key]
                aspect_count_matrix[:, aspect_idx] = np.isin(
                    remaining_items, indices)

            aspect_counts_for_selected = np.array(
                [aspect_counts[j][key] for key in ordered_keys])

            # Step 4: Compute the updated aspect distributions

            updated_aspect_distributions = aspect_counts_for_selected + aspect_count_matrix

            # Step 5: Normalize the aspect distributions
            # Ensure no rows sum to zero by adding a small constant (epsilon)
            row_sums = updated_aspect_distributions.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = epsilon  # Avoid division by zero
            normalized_aspect_distributions = updated_aspect_distributions / row_sums

            # Clip small values to avoid problems with log(0) during KL divergence
            normalized_aspect_distributions = np.clip(
                normalized_aspect_distributions, epsilon, 1)

            # Step 6: Compute KL divergence

            # Extract the target_distribution in the same order as aspects
            ordered_target_distribution = np.array(
                [target_distribution[key] for key in ordered_keys])

            # Reshape ordered_target_distribution to be broadcastable
            ordered_target_distribution = np.reshape(
                ordered_target_distribution, (1, -1))

            # according to the paper:Calibrated Recommendations, Harald Steck
            normalized_aspect_distributions = (
                1 - alpha) * normalized_aspect_distributions + alpha * ordered_target_distribution

            # Vectorized KL divergence calculation for remaining items using scipy.stats.entropy
            kl_divergences = entropy(
                ordered_target_distribution, normalized_aspect_distributions, axis=1)

            # Accumulate the diversity scores
            diversity_scores += dimension_importance[j] * kl_divergences

        return diversity_scores

    def rerank(self, user_idx, interaction_history=None, candidate_items=None, prediction_scores=None, filtering_rules=None, **kwargs):
        """
        Re-rank candidate items for a user using a greedy algorithm based on KL divergence.

        This method iteratively selects items that minimize the KL divergence between the current 
         distributions and the target distributions.

        Parameters:
        - `user_idx` (int): ID of the user receiving recommendations.
        - `interaction_history` (optional): cornac.data.Dataset. User-Item preference data for filtering out previously interacted items.
        - `candidate_items` (list): List of items to be re-ranked.
        - `prediction_scores` (list): Prediction scores for the candidate items (optional).
        - `filtering_rules` (dict): Filtering rules for candidate items (optional).

        Returns:
        - `selected` (list): List of `top_k` re-ranked items for the user, ordered by diversity scores.
        """

        # Prepare and execute filters

        super().rerank(user_idx=user_idx, interaction_history=interaction_history, candidate_items=candidate_items,
                        prediction_scores=prediction_scores, **kwargs)

        aspects_prop, aspects_items = TargetDistributionMatcher(
            self.target_distributions, self.diversity_dimension, self.item_dataframe, candidate_items)
        self.execute_filters(user_idx, filtering_rules)

        self.filter_items_in_additional_history(user_idx)  ## for Mind setup

        candidate_items = self.candidate_items[user_idx]
        selected = []
        # Create a boolean mask to keep track of remaining items
        remaining_mask = np.ones(len(candidate_items), dtype=bool)
        while len(selected) < self.top_k and np.any(remaining_mask):

            remaining_ids = np.array(candidate_items)[remaining_mask]
            if len(remaining_ids) == 0:
                break
            # Compute diversity scores for all candidate items in one go
            diversity_scores = self.diversityScores(
                remaining_ids, selected, aspects_prop, self.diversity_dimension_weight, aspects_items, alpha=0.1)

            next_item_idx = np.argmin(diversity_scores)
            next_item_id = remaining_ids[next_item_idx]

            # Add the selected item to the list
            selected.append(next_item_id)
            remaining_mask[candidate_items.index(next_item_id)] = False

        # assert len(selected) == self.top_k, f"Expected {self.top_k} items, but got {len(selected)}"
        assert len(selected) == len(set(selected)), "Duplicate items found in selected"
        # Store the selected items
        self.ranked_items[user_idx] = selected

        return selected
