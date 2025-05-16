import pandas as pd
from typing import Optional, Dict, List

from .reranker import ReRanker
from .user_simulator import UserSimulator
import os
import configparser
import json


class DynamicReRanker(ReRanker):
    """DynamicReRanker: A generic base class for dynamic re-ranking. All dynamic re-ranking methods should inherit from this class.
    Subclasses are expected to implement specific re-ranking logic and customize behavior as needed.

    """

    def __init__(self, name: str,  item_dataframe: Optional[pd.DataFrame] = None, diversity_dimension: Optional[List[str]] = None, top_k: int = 20,  pool_size: int = -1, feedback_window_size: int = 1,  config_file: Optional[str] = None,
                   user_simulator_config_path: str = "./reranker_configs/user_simulator_config.ini", **kwargs):
        """
        Parameters
        ----------
        name : string. Name of the re-ranking method.
        item_dataframe : Optional[pd.DataFrame]
             - Indexed by Cornac IDs.
            - Can be `None` if no item features are used.
        diversity_dimension : Optional[List[str]].
            A list of strings specifying the dimensions along which diversity should be considered
            during re-ranking. If None, no diversity dimensions are applied. These diversity dimensions
            must correspond to the columns in `item_dataframe`. For example, if `item_dataframe`
            has columns 'sentiment' and 'category', then `diversity_dimension` could be ['sentiment', 'category'].
        top_k : int (optional). Number of items to recommend (default: 20).
        pool_size : int (optional). The number of candidate items to consider for re-ranking (default: -1 for all items).
        feedback_window_size (int): Number of feedbacks in previous iterations for recomemndation update.
        config_file (Optional[str]): Path to the configuration file for the re-ranker.

        Attributes Initialized
        ----------------------
        - Attributes for managing candidate items, prediction scores, user history, and feedback:
        `candidate_items`, `candidate_scores`, `user_history`, `shownHistory`, 
      `feedbackFromLastIteration`, and `users`.

        """
        super().__init__(name=name, item_dataframe=item_dataframe,
                         diversity_dimension=diversity_dimension,  top_k=top_k, pool_size=pool_size, **kwargs)

        self.feedback_window_size = feedback_window_size

        # items already shown to the user. The key is an user id, the value is a list
        self.shownHistory: Dict[int, List[int]] = {}
        # item clicked from last iteration. The key is an user id, the value is a list.
        self.feedbackFromLastIteration: Dict[int, List[int]] = {}
        self.users = {}
        self.bin_edges = kwargs.get('bin_edges', {})
        self.user_choice_model = kwargs.get(
            'user_choice_model', 'logarithmic_rank_bias')

        self.user_simulator_config_path = user_simulator_config_path

        # Load configuration if a config file is provided
        if config_file:
            self.configReranker(config_file)

    def configReranker(self, fpath="./reranker_configs/reranker.ini"):
        """Configure the dynamic re-ranker with parameters from a configuration file."""
        if os.path.exists(fpath):
            # Parse the configuration file for the dynamic re-ranker
            (
                top_k,
                pool_size,
                diversity_dimension,
                feedback_window_size,
                bin_edges,
                user_choice_model,
                user_simulator_config_path,
            ) = self.read_config(fpath)

            # Set the parsed values to instance variables
            self.top_k = top_k
            self.pool_size = pool_size
            self.feedback_window_size = feedback_window_size
            self.diversity_dimension = diversity_dimension
            self.bin_edges = bin_edges
            self.user_choice_model = user_choice_model
            self.user_simulator_config_path = user_simulator_config_path
        else:
            raise FileNotFoundError(f"Config file '{fpath}' does not exist.")

    def read_config(self, fpath="./reranker_configs/reranker.ini"):
        """Parse the configuration file for dynamic re-ranker-specific parameters."""
        config = configparser.ConfigParser()
        try:
            config.read(fpath)
        except configparser.Error as e:
            raise ValueError(
                f"Error parsing configuration file '{fpath}': {e}")

        if self.name not in config.sections():
            raise ValueError(
                f"Dynamic re-ranker '{self.name}' not found in the configuration file '{fpath}'.")

        # Read the section corresponding to the dynamic re-ranker name
        section = config[self.name]

        try:
            top_k = int(section.get("top_k", self.top_k))
        except ValueError as e:
            raise ValueError(
                f"Error parsing 'top_k' in section '{self.name}': {e}")

        try:
            pool_size = int(section.get("pool_size", self.pool_size))
        except ValueError as e:
            raise ValueError(
                f"Error parsing 'pool_size' in section '{self.name}': {e}")

        try:
            feedback_window_size = int(section.get("feedback_window_size", 1))
        except ValueError as e:
            raise ValueError(
                f"Error parsing 'feedback_window_size' in section '{self.name}': {e}")

        try:
            diversity_dimension = json.loads(section["diversity_dimension"])
            if not isinstance(diversity_dimension, list):
                raise ValueError("The 'diversity_dimension' must be a list.")
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Error parsing 'diversity_dimension' in section '{self.name}': {e}")

        try:
            bin_edges = json.loads(section["bin_edges"])
            if not isinstance(bin_edges, dict):
                raise ValueError("The 'bin_edges' must be a dictionary.")
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Error parsing 'bin_edges' in section '{self.name}': {e}")

        try:
            user_choice_model = section.get(
                "user_choice_model", "logarithmic_rank_bias")
        except KeyError as e:
            raise ValueError(
                f"Error parsing 'user_choice_model' in section '{self.name}': {e}")

        try:
            user_simulator_config_path = section.get(
                "user_simulator_config_path", "./reranker_configs/user_simulator_config.ini")
        except KeyError as e:
            raise ValueError(
                f"Error parsing 'user_simulator_config_path' in section '{self.name}': {e}")

        return (
            top_k,
            pool_size,
            diversity_dimension,
            feedback_window_size,
            bin_edges,
            user_choice_model,
            user_simulator_config_path,
        )

    def filter_seen_items(self, user_idx):
        """
        Filter out items that a user has already seen from the candidate items.

        Parameters:
        - `user_idx` (int): The ID of the user for whom filtering is applied.

        Effect:
        - Updates `self.candidate_items[user_idx]` to exclude items in the user's `seen_items` list.
        """
        user_seen_items = self.users[user_idx].seen_items

        candidates = self.candidate_items[user_idx]

        filtered_candidates = [
            item for item in candidates if item not in user_seen_items]
        self.candidate_items[user_idx] = filtered_candidates

    def add_user(self, user_id):
        """
        Add a `UserSimulator` instance for a specific user.

        This method initializes a `UserSimulator` for the given user based on their interaction history.

        Parameters:
        - `user_id` (int): The ID of the user to add.

        Effect:
        - Adds the `UserSimulator` instance to the `self.users` dictionary.

        Notes:
        - This method is intended to be overridden by subclasses.
        """
        simulated_user = UserSimulator(user_id, self.user_history[user_id],  config_path=self.user_simulator_config_path)
        self.users[user_id] = simulated_user

    def update_recommendations(self, user_id):
        """
        Update recommendations for a specific user.

        This method updates the candidate items for a user by applying filtering logic 
        (e.g., removing seen items). Subclasses are expected to implement additional 
        logic to refine recommendations.

        Parameters:
        - `user_id` (int): The ID of the user for whom recommendations are updated.

        Notes:
        - This method is intended to be overridden by subclasses for specific re-ranking logic.
        """
        self.filter_seen_items(user_id)
        # implementation
