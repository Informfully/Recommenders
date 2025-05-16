import numpy as np
from ..reranker import ReRanker
from ...exception import ScoreException
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MMR_ReRanker(ReRanker):
    """MMR_ReRanker: A diversity-aware re-ranking model based on the MMR algorithm.

    """
    def __init__(self,  name="MMR", top_k=10, pool_size=-1, item_feature_vectors=None, user_item_history = None, rerankers_item_pool = None,lamda = 0):
        """
        Initialize the PM2Reranker with the specified configurations.

        Parameters:
        - `name` (str): Name of the reranker (default: "MMR").
        - `top_k` (int): Number of items to recommend in the ranked list (default: 10).
        - `pool_size` (int): Size of the item pool (default -1 to include all items).
        - `config_file` (str): Path to configuration file for target distributions.
            If provided, target distributions and diversity dimension weights are loaded from the file.
        
        Initializes:
        - Target distributions and diversity dimension weights (from configuration file or parameters).
        - Validates the diversity dimensions and target distributions.
        """
        if item_feature_vectors is None:
            raise ValueError("item_feature_vectors cannot be None. Please provide valid item feature vectors.")
        # if user_item_history is None:
        #     user_item_history = {}

        super().__init__(name=name, 
                         top_k=top_k,  pool_size=pool_size, user_item_history = user_item_history, rerankers_item_pool = rerankers_item_pool)
        self.item_feature_vectors = item_feature_vectors

        # self.user_item_history = user_item_history
        # self.rerankers_item_pool = rerankers_item_pool
        self.lamda = lamda
        

        

    def diversityScores(self, remaining_items, selected_items, item_feature_vectors, prediction_scores, lamda=0):
        """
        Compute diversity scores for candidate items using the MMR algorithm.

        Parameters:
        - remaining_items (list): Candidate items.
        - selected_items (list): Already selected items.
        - item_feature_vectors (dict): Mapping of items to their feature vectors.
        - prediction_scores (array-like): Relevance scores for the remaining items.
        - lamda (float): Balance factor between relevance and diversity (default: 0).

        Returns:
        - np.ndarray: MMR scores for the remaining items.
        """
        # Get feature vectors of remaining and selected items
        remaining_vectors = np.array([item_feature_vectors[item] for item in remaining_items])
        
        if selected_items:
            selected_vectors = np.array([item_feature_vectors[item] for item in selected_items])
        else:
            selected_vectors = None
        
        # Compute relevance scores (Sim_1)
        relevance_scores = np.array(prediction_scores) if lamda > 0 else np.zeros(len(remaining_items))
        # if prediction_scores is None:
        #     relevance_scores = np.zeros(len(remaining_items))
        # else:
        #     relevance_scores = np.array([prediction_scores[item] for item in remaining_items]) if lamda > 0 else np.zeros(len(remaining_items))


        # Compute diversity scores (Sim_2)
        if selected_items:
            # each pair of vectors (one from remaining_vectors and one from selected_vectors

            # similarity_matrix[i, j] represents the cosine similarity between the i-th remaining item and the j-th selected item.
            similarity_matrix = cosine_similarity(remaining_vectors, selected_vectors)

            #  computes the maximum value along each row of similarity_matrix
            max_diversity_scores = np.max(similarity_matrix, axis=1)  # max similarity with selected items
        else:
            max_diversity_scores = np.zeros(len(remaining_items))  # No selected items yet

        # Compute final MMR scores
        mmr_scores = lamda * relevance_scores - (1 - lamda) * max_diversity_scores

        return mmr_scores

 
        

    def rerank(self, user_idx, interaction_history=None, candidate_items=None, prediction_scores=None, filtering_rules: dict = None, **kwargs):
        """Re-rank candidate items based on diversity scores computed using the MMR algorithm.

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
        if not candidate_items:
            raise ValueError(f"Candidate items cannot be empty for user {user_idx}. Please provide a valid list of candidate items.")
        super().rerank( user_idx=user_idx, interaction_history=interaction_history, candidate_items=candidate_items,
                        prediction_scores=prediction_scores, **kwargs)
       
        self.execute_filters(user_idx, filtering_rules)
        self.filter_items_in_additional_history(user_idx)
        self.retrieve_prediction_scores(user_idx)

        candidate_items = self.candidate_items[user_idx]
        # candidate_prediction_scores = self.candidate_scores[user_idx]
        candidate_prediction_scores = np.array(self.candidate_scores[user_idx])  # Convert to NumPy array for indexing

        selected = []
        # Boolean mask to track remaining items
        remaining_mask = np.ones(len(candidate_items), dtype=bool)

        while len(selected) < self.top_k and np.any(remaining_mask):

            remaining_items = np.array(candidate_items)[remaining_mask]
            remaining_scores = candidate_prediction_scores[remaining_mask]  # Filter scores accordingly
            if len(remaining_items) == 0:
                break
            scores = self.diversityScores(
                remaining_items, selected,  self.item_feature_vectors, remaining_scores, lamda=self.lamda)
            # Select the item with the highest diversity score
            next_item_idx = np.argmax(scores)
            next_item_id = remaining_items[next_item_idx]

            # Add the selected item to the list
            selected.append(next_item_id)

            # Update remaining mask: mark the selected item as unavailable
            # remaining_mask[candidate_items.index(next_item_id)] = False
            remaining_mask[np.where(candidate_items == next_item_id)[0][0]] = False  # Finds the first occurrence safely

        # cache the re-ranked list for a user
        # assert len(selected) == self.top_k, f"Expected {self.top_k} items, but got {len(selected)}"
        assert len(selected) == len(set(selected)), "Duplicate items found in selected"
        self.ranked_items[user_idx] = selected
        return selected
