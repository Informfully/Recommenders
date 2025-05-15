import scipy.sparse as sp
import numpy as np
from cornac.utils import common
from .graph_recommender import GraphRec
from ...exception import ScoreException
from ..recommender import Recommender


class RP3_Beta(Recommender):
    """Random Walk Algorithm.

    Parameters
    ----------
    name: string, default: 'RDW'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already
        pre-trained (U and V are not None).

    verbose: boolean, optional, default: False
        When True, running logs are displayed.    

    num_users: int, default: 0
        The number of users in dataset
    
    num_items: int, default: 0
        The number of item in dataset
    
    scoring_iteration: int, default: 100
        Iteration of positions.IdeologyFromTweets training.

    scoring_loss: int, default: 10
        positions.IdeologyFromTweets loss

    b: int, default: 2
        Value "b" is the amount by which you want the algorithm to diversify recommendations:
        b > 1   will add more diverse items, e.g., b = 2, 5, 10 etc.
        b < 1   will add comparatively less (but still more than normal) items: e.g., b = 0.9, 0.1, etc.
        Important: Do not use negative values for "b"!

    iterations: int, default: 2
        The value of "iteration" determines for how long should the random walks on the graph continue:
        1       value of the usual recommender where there is no diversification
        > 1     starts to diversify
        5-10    good value based on Bibek's experiene with Twitter and hashtags

    type: str, default: "broadening"
        Vectors "user_positions" and "item_positions" containing user and article scores. And their respective 
        length needs to match the number of users and items in the training dataset UxI with the dimensions 
        len(user_positions) x len(item_position).
    
    """
    def __init__(
        self,
        beta = 0.7,
        name="RP3_beta",
        trainable=True,
        verbose=False,
     
        article_pool = []
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)

        self.beta = beta
        self.article_pool = article_pool

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

    def score(self, user_idx, item_idx=None):
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
        print(f"completed:prediction:{score}")
        
        # Map article_id to its index in iid_map
        boost_factor = 100
        for article_id in self.article_pool:
            # Map article_id to its index in iid_map
            score[article_id] = score[article_id]+boost_factor

        if item_idx == None:
            return score
        
        return score[item_idx]
