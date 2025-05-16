# UserSimulator for dynamic re-ranking process.
# Two user choice models are provided:logarithmic_rank_bias and  preference_based_bias.
# Based on Sirui Yao, Yoni Halpern, Nithum Thain, Xuezhi Wang, Kang Lee, Flavien Prost, Ed H.
# Chi, Jilin Chen, and Alex Beutel. 2020. Measuring Recommender System Effects with
# Simulated Users. In Proceedings of the Second Workshop on Fairness, Accountability,
# Transparency, Ethics and Society on the Web. ACM, New York, NY, USA, 10 pages.
# ============================================================================
import numpy as np
import datetime
import configparser
import os


class UserSimulator:
    """
    UserSimulator: A class to simulate user interaction behavior in recommender systems.

    """

    def __init__(self, user_id, user_history, choice_model='logarithmic_rank_bias', config_path='./experiments/configs/reranker_configs/user_simulator_config.ini', preference=None, attribute_items_mapping=None):
        """
        Initialize the UserSimulator with configuration.
        Parameters:
        - `user_id` (int): User ID for which simulate on .
        - `user_history` (list): Interaction history of the user.
            - Can include tuples `(item_id, timestamp)` for activity level division based on frequency-based thresholds.
            - Can include just `item_id` for activity level division based on total reads thresholds.
        - `choice_model` (str): Model for simulating user click behavior (default: `logarithmic_rank_bias`).
            - Supported models: 
                - `logarithmic_rank_bias`: Simulates clicks based on item rank.
                - `preference_based_bias`: Simulates clicks based on user preferences for diversity dimensions.
        - `config_path` (str): Path to the configuration `.ini` file for thresholds and parameters.
        - `preference` (dict, optional): User preferences for diversity dimensions.
        - `attribute_items_mapping` (dict, optional): Mapping of items to categories for diversity dimensions.

        Initializes:
        - User activity level, click probabilities, and interaction thresholds based on the configuration file.
        - Interaction records (`seen_items` and `interacted_items`) for tracking user behavior.

        Raises:
        - `FileNotFoundError`: If the specified configuration file does not exist.
        - `ValueError`: If no thresholds are specified in the configuration file.

        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"The configuration file {config_path} does not exist.")
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        self.user_id = user_id
        self.history = user_history
        self.choice_model = choice_model

        self.seed = self.config.getint('simulation', 'seed', fallback=42)

        np.random.seed(self.seed)
        self.preference = preference
        self.attribute_items_mapping = attribute_items_mapping

        # Check if the configuration contains frequency or total read thresholds
        self.has_frequency_thresholds = 'frequency_thresholds' in self.config
        self.has_total_reads_thresholds = 'total_reads_thresholds' in self.config

        if self.has_frequency_thresholds:
            self.levels = self.load_levels(self.config, 'frequency_thresholds')

            self.frequency_thresholds = self.load_thresholds(
                self.config, 'frequency_thresholds')
        else:
            self.frequency_thresholds = None

        if self.has_total_reads_thresholds:
            self.levels = self.load_levels(
                self.config, 'total_reads_thresholds')

            self.total_reads_thresholds = self.load_thresholds(
                self.config, 'total_reads_thresholds')
        else:
            self.total_reads_thresholds = None
         # Ensure at least one type of threshold exists
        if not self.has_frequency_thresholds and not self.has_total_reads_thresholds:
            raise ValueError(
                "At least one of 'frequency_thresholds' or 'total_reads_thresholds' must be specified in the configuration file.")

        # group users into two active levels
        self.active_level = self.calculate_activity_level()

        # Load max iterations configuration according to the user's active level
        self.max_iterations_config = {level: self.config.getint(
            'max_iterations', level, fallback=3) for level in self.levels}

        # Set clicked each iteration based on activity level
        self.clicked_each_iteration = self.config.getint(
            'clicked_each_iteration', self.active_level, fallback=4)
        # Determine the maximum number of iterations
        self.max_iteration = self.determine_max_iterations()
        # Initialize interaction record
        self.seen_items = []
        self.interacted_items = []

    def load_levels(self, config, section):
        """
        Load user activity levels from the configuration file.

        Parameters:
        - `config` (ConfigParser): The configuration parser object.
        - `section` (str): The section in the configuration file containing activity levels.

        Returns:
        - `levels` (list): A list of user activity levels defined in the configuration file.

        Raises:
        - `ValueError`: If an error occurs while loading levels.


        """
        try:
            levels = list(config[section].keys())
            return levels
        except Exception as e:
            raise ValueError(
                f"An error occurred while loading levels: {str(e)}")

    def load_thresholds(self, config, section):
        """
        Load thresholds for activity levels from the configuration file.

        Parameters:
        - `config` (ConfigParser): The configuration parser object.
        - `section` (str): The section in the configuration file containing thresholds.

        Returns:
        - `thresholds` (dict): A dictionary mapping activity levels to their respective thresholds.

        """

        try:
            return {level: config.getfloat(section, level) for level in config[section].keys()}
        except Exception as e:
            print(
                f"Section '{section}' not found in the configuration while loading thresholds.")
            return None

    def calculate_activity_level(self):
        """
        Determine the user's activity level based on interaction history.

        The method uses either frequency-based thresholds or total reads thresholds 
        to categorize the user's activity level.

        Returns:
        - `active_level` (str): The activity level of the user (e.g., "very_active", "active", "less_active").

        Raises:
        - `ValueError`: If the user history format does not match the available thresholds.
        """

        if len(self.history) == 0:
            return self.levels[-1]  # Return the lowest level if no history

        # Check if history contains datetime information (for frequency calculation)

        if all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], datetime.datetime) for x in self.history):
            if self.has_frequency_thresholds:
                return self.calculate_frequency_based_level()
            else:
                # Update self.history to only keep the first element of each tuple (x[0])
                self.history = [x[0] for x in self.history]

        if self.has_total_reads_thresholds and all(isinstance(x, (int, np.integer)) for x in self.history):

            return self.calculate_total_reads_level()

        # If neither condition is met, raise an error
        raise ValueError(
            "User history format does not match available thresholds (either frequency-based or total-reads-based).")

    def calculate_frequency_based_level(self):
        """
        Calculate the user's activity level based on interaction frequency.

        Uses the dates in the user's interaction history to calculate the frequency of interactions.

        Returns:
        - str. The activity level determined by interaction frequency.

        """
        dates = [x[1] for x in self.history]
        recent_date = max(dates)
        first_date = min(dates)
        total_days = (recent_date - first_date).days + 1
        frequency = len(self.history) / total_days

        # Sort the levels based on the frequency thresholds in descending order
        sorted_levels = sorted(
            self.levels, key=lambda x: self.frequency_thresholds[x], reverse=True)

        # Dynamically determine the user's level based on frequency thresholds
        for level in sorted_levels:
            # for level in self.levels:
            if frequency >= self.frequency_thresholds[level]:
                return level

        return sorted_levels[-1]

    def calculate_total_reads_level(self):
        """
        Calculate the user's activity level based on the total number of interactions.

        Returns:
        - str. The activity level determined by the total number of reads.
        """

        number_of_reads = len(self.history)

        # Sort the levels based on the total reads thresholds in descending order
        sorted_levels = sorted(
            self.levels, key=lambda x: self.total_reads_thresholds[x], reverse=True)

        # Determine the user's level by comparing the number of reads with the thresholds
        for level in sorted_levels:
            if number_of_reads >= self.total_reads_thresholds[level]:
                return level

        # Default to the lowest level if no thresholds are met
        return sorted_levels[-1]  # the lowest level

    def determine_max_iterations(self):
        """
        Determine the maximum number of iterations for user interactions.

        Returns:
        - `max_iterations` (int): The maximum number of iterations allowed for the user.
        """

        # Get max iterations based on the user's activity level, with a default of 3 if not set.
        max_iterations_for_level = self.max_iterations_config.get(
            self.active_level, 3)

        return max_iterations_for_level

    def rho(self, item_ids):
        """
        Compute preference scores for items based on the user's preferences across diversity dimensions.

        This implementation is inspired by the $\alpha$-preference model described in:
        Sirui Yao, Yoni Halpern, Nithum Thain, Xuezhi Wang, Kang Lee, Flavien Prost, Ed H. Chi, 
        Jilin Chen, and Alex Beutel. 2020. "Measuring Recommender System Effects with Simulated Users." 
        In Proceedings of the Second Workshop on Fairness, Accountability, Transparency, Ethics and 
        Society on the Web. ACM, New York, NY, USA, 10 pages.

        The $\alpha$-preference model simulates users selecting items based on a specific attribute $\rho(v)$ 
        using a softmax distribution. The parameter $\alpha$ determines the user's preference for items with 
        that attribute:
            - Positive $\alpha$ values indicate preference for items with the attribute.
            - Negative $\alpha$ values indicate a dislike for items with the attribute.


        Parameters:
        - `item_ids` (list): A list of item IDs for which preference scores need to be calculated.

        Returns:
        - `scores` (np.ndarray): An array containing the preference scores for the provided items.
        """

        scores = np.zeros(len(item_ids))

        # Iterate over each dimension and their preferences
        for dimension_idx, preference_dict in enumerate(self.preference):
            # Go through each category and its preference value
            for category, preference_value in preference_dict.items():
                items = self.attribute_items_mapping[dimension_idx][category]
                # Create an array of True/False values indicating if item_ids are in items
                item_mask = np.isin(item_ids, items)
                # Add the preference value to the scores where item_mask is True
                scores[item_mask] += preference_value
        return scores

    def softmax(self, scores):
        """
        Compute the softmax of an array of scores.

        Parameters:
        - `scores` (np.ndarray): Array of scores.

        Returns:
        - `probabilities` (np.ndarray): Softmax-transformed probabilities.
        """

        if scores.size == 0:
            return np.array([])
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        return probabilities

    def click_probability(self, item_ids):
        """
        Calculate click probabilities for items based on the choice model.

        Parameters:
        - `item_ids` (list): List of item IDs.

        Returns:
        - `probabilities` (np.ndarray): Array of click probabilities for the items.
        """

        if self.choice_model == 'preference_based_bias':
            scores = self.rho(item_ids)
            probabilities = self.softmax(scores)
        elif self.choice_model == 'logarithmic_rank_bias':
            # Rank positions (1-indexed)
            ranks = np.arange(1, len(item_ids) + 1)
            # Calculate probabilities using the rank-based formula
            probabilities = 1 / np.log1p(ranks)

        probabilities /= probabilities.sum()
        return probabilities

    def simulate_interaction(self, item_ids):
        """
        Simulate user interaction with a list of recommended items.

        Parameters:
        - `item_ids` (list): List of items to simulate interactions for.

        Returns:
        - `clicks` (list): List of item IDs the user clicked on.

        Effect:
        - Updates `seen_items` and `interacted_items` with the results of the interaction.
        """

        clicks = []
        self.lastIterationClicked = []
        # Check if item_ids is empty
        if len(item_ids) == 0:
            print("No items to interact with when simulate user interaction.")
            return clicks

        # Ensure clicked_each_iteration doesn't exceed the number of available items
        click_count = min(self.clicked_each_iteration, len(item_ids))

        probabilities = self.click_probability(item_ids)

        # Handle random choice safely with the correct size
        try:
            chosen_index = np.random.choice(
                item_ids, size=click_count, replace=False, p=probabilities)
        except ValueError as e:
            print(f"Error during user interaction simulation: {e}")
            return clicks  # Return empty if any error occur
        clicks = chosen_index.tolist()
        self.lastIterationClicked = clicks

        self.interacted_items.append(clicks)
        self.seen_items.extend(item_ids)
        return clicks
