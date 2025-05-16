import numpy as np

from ..reranker import ReRanker
from ...exception import ScoreException


class LeastPopReranker(ReRanker):
    """LeastPopReranker: A reranker that prioritizes items with the lowest popularity.

    This class reorders items in the recommendation list based on their popularity, favoring 
    less popular items (those with fewer interactions) over more popular ones. It uses the 
    interaction history to compute popularity scores for candidate items.
    """

    def __init__(self, name="LeastPop",  item_dataframe=None, diversity_dimension=None, top_k=10, pool_size=-1, user_item_history = None, rerankers_item_pool = None ):
        """
        Initialize the LeastPopReranker with the provided configurations.

        Parameters:
        - `name` (str): Name of the reranker (default: "LeastPop").
        - `item_dataframe` (pd.DataFrame): DataFrame containing item metadata (optional, not utilized by this reranker).
        - `diversity_dimension` (list): Attributes representing diversity dimensions (optional, not utilized by this reranker).
        - `top_k` (int): Number of items to recommend in the ranked list (default: 10).
        - `pool_size` (int): Size of the item pool for candidate selection (-1 to include all items).

        Inherits:
        - Initializes base attributes from the `ReRanker` superclass.
        """
        super().__init__(name=name, item_dataframe=item_dataframe, diversity_dimension=diversity_dimension,
                         top_k=top_k,  pool_size=pool_size,  user_item_history = user_item_history, rerankers_item_pool = rerankers_item_pool)

    def popularityScores(self, candidate_items):
        """
        Compute the popularity scores for candidate items.

        This method calculates the popularity of each candidate item based on the number of user 
        interactions recorded in the interaction history. Items with fewer interactions are assigned 
        lower popularity scores, making them more likely to appear at the top of the re-ranked list.

        Parameters:
        - `candidate_items` (list): List of item IDs for which popularity scores need to be computed.

        Returns:
        - `candidate_item_popularity` (np.ndarray): Array of popularity scores for the candidate items.
            - Items with no recorded popularity in the interaction history are assigned a default score.
        """

        item_pop = np.ediff1d(self.interaction_history.csc_matrix.indptr)
        candidate_item_popularity = np.zeros(len(candidate_items))
        default_popularity = 0  # This makes unknown items most likely to be recommended
        candidate_item_popularity = np.array([item_pop[item] if item < len(
            item_pop) else default_popularity for item in candidate_items])
        return candidate_item_popularity

    def rerank(self, user_idx, interaction_history=None, candidate_items=None, prediction_scores=None, filtering_rules: dict = None, **kwargs):
        """Re-rank candidate items based on their popularity (ascending order).

        This method sorts candidate items by their popularity scores. The least popular 
        items appear at the top of the recommendation list. It also integrates interaction history, 
        filters, and re-ranking logic inherited from the `ReRanker` superclass.


        Parameters
        ----------

        - `user_idx` (int): ID of the user receiving recommendations.
        - `interaction_history` (cornac.data.Dataset): User-Item preference data for filtering previously interacted items.
        - `candidate_items` (list): List of items to be re-ranked.
        - `prediction_scores` (list): Prediction scores from the base recommender model (optional).
        - `filtering_rules` (dict): Filtering rules to refine candidate items (optional).
    

        Returns:
        - `selected_items` (list): Re-ranked list of items, i.e., the top-k least popular items.
        """

        super().rerank(user_idx=user_idx, interaction_history=interaction_history, candidate_items=candidate_items,
                        prediction_scores=prediction_scores, **kwargs)
        self.execute_filters(
            user_idx=user_idx, filtering_rules=filtering_rules)
        candidate_items = self.candidate_items[user_idx]

        candidate_item_popularity = self.popularityScores(candidate_items)

        # Sorting the items by popularity
        # for ascending order
        sorted_indices = np.argsort(candidate_item_popularity)

        sorted_item_ids = np.array(candidate_items)[sorted_indices]
        sorted_popularity = candidate_item_popularity[sorted_indices]
        item_ids = sorted_item_ids.tolist()
        if self.top_k <= len(item_ids):
            selected_items = item_ids[:self.top_k]
        else:
            selected_items = item_ids
        self.ranked_items[user_idx] = selected_items
        return selected_items
