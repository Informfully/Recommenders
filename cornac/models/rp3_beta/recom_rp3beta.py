#############################################################################################################
# NAME:         recom_rp3beta.py                                                                            #                                                                     
# VERSION:      1.0 (STANDALONE)                                                                            #
# DESCRIPTION:  Random walk algorithm for Cornac                                                            #
# AUTHORS:      Paudel, B., Christoffel, F., Newell, C., & Bernstein, A. (2016).                            #
# PAPER:        Updatable, accurate, diverse, and scalable recommendations for interactive applications.    #
#               ACM Transactions on Interactive Intelligent Systems (TiiS), 7(1), 1-34.                     #
#############################################################################################################

import numpy 

from ..recommender import Recommender
from cornac.utils import common
from ...exception import ScoreException
import scipy.sparse as sp
import numpy as np
from .graph_recommender import GraphRec


class RP3_Beta(Recommender):
    """Random Walk Algorithm.

    Parameters
    ----------
    name: string, default: 'RP3_beta'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already
        pre-trained (U and V are not None).

    verbose: boolean, optional, default: False
        When True, running logs are displayed.    

    beta: a parameter to adjust the influence of item degree in the re-ranking formula. default: 0.7.
    For details, refer to the paper.
    
    """
    def __init__(
        self,
        beta = 0.7,
        name="RP3_beta",
        trainable=True,
        verbose=False,
     
        **kwargs
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose,  **kwargs)

        self.beta = beta
        # self.article_pool = article_pool # already set this during Recommender.__init__


    def fit(self, train_set, val_set=None):

        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object

        train_set:(userId, userHistory, userScore)
        """
        Recommender.fit(self, train_set)

        def interacted_items(csr_row):
            return [
                item_idx
                for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                if rating > 0
            ]

        gt_mat = train_set.csr_matrix
        self.train_set = train_set
        self.train_set_dict = {}
        train_user_indices = set(train_set.uir_tuple[0])
        for user_idx in train_user_indices:
            train_pos_items = interacted_items(gt_mat.getrow(user_idx))
            self.train_set_dict[user_idx] = train_pos_items
        
        train_item_indices = set(train_set.uir_tuple[1])
        self.train_set_rating = np.zeros((len(train_user_indices),
                                          len(train_item_indices)))
        for user_idx, item_idxs in self.train_set_dict.items():
            self.train_set_rating[user_idx][item_idxs] = 1
        
        self.Model_RDW = GraphRec(self.train_set_rating)
        self.Model_RDW.performInitialHop()


        return self
    
    def score(self, user_idx, item_idx=None, **kwargs):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if self.is_unknown_user(user_idx):
            print(f'unknown user : {user_idx}\n')
            raise ScoreException(
                "Can't make score prediction for (user_id=%d)" % user_idx
            )
        
        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        score = self.Model_RDW.predict_reranked_scores(user_idx, beta =  self.beta )

        if item_idx == None:
            return score
        
        return score[item_idx]

    def rank(self, user_idx, item_indices=None, k=-1, **kwargs):
        """
        Generate an ordered recommendation list for a given user.

        Parameters
        ----------
        user_idx : int
            The Cornac ID of the user for whom recommendations are being generated.

        item_indices : list of int, optional
            A list of candidate item indices to be ranked for the user. If `None`, all available items 
            are considered for ranking. Default is `None`.

        Returns
        -------
        ranked_items : list of int 
            A list of `targetSize` item indices representing the recommended items in ranked order.

        item_scores : list of float
            The random walk scores corresponding to the `item_indices`.

        """
        if  self.is_unknown_user(user_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d)" % user_idx
            )
   
        if self.article_pool is not None:

            # item_idx2id = kwargs.get("item_idx2id")
            # user_idx2id = kwargs.get("user_idx2id")
            # item_id2idx = kwargs.get("item_id2idx")
            item_idx2id = {v: k for k, v in self.iid_map.items()} # cornac item ID : raw item ID
            user_idx2id = {v: k for k, v in self.uid_map.items()} # cornac user ID : raw user ID
            item_id2idx = {k: v for k, v in self.iid_map.items()} # raw item ID : cornac item ID

            assert isinstance(item_idx2id, dict), "item_idx2id must be a dictionary"
            assert isinstance(user_idx2id, dict), "user_idx2id must be a dictionary"
            assert isinstance(item_idx2id, dict), "item_idx2id must be a dictionary"

            impression_items_list = []
            for iid in self.article_pool:
                if iid in item_id2idx:
                    idx = item_id2idx[iid]
                    impression_items_list.append(idx)

            ranked_items, item_scores = self.rank_partial( user_idx=user_idx, item_indices = impression_items_list, item_idx2id = item_idx2id, user_idx2id = user_idx2id)

            self.ranked_items[user_idx] = ranked_items
            self.item_scores[user_idx] = item_scores
            self.item_scores_mapped_indices[user_idx] = impression_items_list

            return ranked_items, item_scores

        # original cornac code: rank known items
        # obtain item scores from the model
        try:
            known_item_scores = self.score(user_idx, **kwargs)
        except ScoreException:
            known_item_scores = np.ones(self.total_items) * self.default_score()

        # check if the returned scores also cover unknown items
        # if not, all unknown items will be given the MIN score
        if len(known_item_scores) == self.total_items:
            all_item_scores = known_item_scores
        else:
            all_item_scores = np.ones(self.total_items) * np.min(known_item_scores)
            all_item_scores[: self.num_items] = known_item_scores

        # rank items based on their scores
        item_indices = (
            np.arange(self.num_items)
            if item_indices is None
            else np.asarray(item_indices)
        )
        item_scores = all_item_scores[item_indices]

        if k != -1:  # O(n + k log k), faster for small k which is usually the case
            partitioned_idx = np.argpartition(item_scores, -k)
            top_k_idx = partitioned_idx[-k:]
            sorted_top_k_idx = top_k_idx[np.argsort(item_scores[top_k_idx])]
            partitioned_idx[-k:] = sorted_top_k_idx
            ranked_items = item_indices[partitioned_idx[::-1]]
        else:  # O(n log n)
            ranked_items = item_indices[item_scores.argsort()[::-1]]

        self.ranked_items[user_idx] = ranked_items
        self.item_scores[user_idx] = item_scores
        self.item_scores_mapped_indices[user_idx]= item_indices

        return ranked_items, item_scores

    def rank_partial(self, user_idx, item_indices=None, **kwargs):
        """
        Generate a ranked recommendation list for a user, restricted to a subset of available items.

        This method is specifically designed for scenarios where the ranking is limited to a partial set 
        of available items (`item_indices`). Our experiment requires this method.

        Parameters
        ----------
        user_idx : int
            The Cornac ID of the user for whom recommendations are being generated.

        item_indices : list of int, optional
            A list of candidate item indices to be ranked for the user. If `None`, the method will 
            return a ranked list of known item indices for the user and their corresponding scores.

        Returns
        -------
        ranked_items : list of int
            A list of ranked item indices representing the recommended items, limited to the given subset.

        item_scores : list of float
            The scores corresponding to the `item_indices`.
        """
        try:
            known_item_scores = self.score(user_idx, **kwargs)
        except ScoreException:
            known_item_scores = np.ones(self.total_items) * self.default_score()

        # check if the returned scores also cover unknown items
        # if not, all unknown items will be given the MIN score
        if len(known_item_scores) == self.total_items:
            all_item_scores = known_item_scores
        else:
            all_item_scores = np.ones(self.total_items) * np.min(known_item_scores)
            all_item_scores[: self.num_items] = known_item_scores

        # Extract scores only for items in the article pool
        pool_scores = [(article_id, all_item_scores[article_id]) for article_id in item_indices]
        # Sort by score in descending order
        ranked_scores = sorted(pool_scores, key=lambda x: x[1], reverse=True)
        
        # Return just the ranked list of item indices
        ranked_items = np.asarray([item for item, _ in ranked_scores])

        random_walk_prob = np.asarray([all_item_scores[item] for item in item_indices])

        return ranked_items, random_walk_prob
