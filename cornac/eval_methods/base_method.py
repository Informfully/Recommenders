# Copyright 2018 The Cornac Authors. All Rights Reserved.
############################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from collections import OrderedDict
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from ..data import FeatureModality
from ..data import TextModality, ReviewModality
from ..data import ImageModality
from ..data import GraphModality
from ..data import SentimentModality
from ..data import Dataset
from ..metrics import RatingMetric
from ..metrics import RankingMetric
from ..metrics import DiversityMetric
from ..experiment.result import Result
from ..utils import get_rng
from ..utils.common import MyConfigParser
import os
import pickle
import json



def save_model_parameter(fpath="../experiments/configs/model_configs/parameters.ini"):
    """Extracts model parameters from a configuration file.

    Parameters
    ----------
    fpath : str, optional
        Path to the configuration file, by default "parameters.ini".

    Returns
    -------
    dict
        A dictionary containing parameter names and their values.
    """
    config = MyConfigParser()
    file_path = fpath
    config.read(file_path)
    para_info = {}
    for each_section in config.sections():
        # Check if 'tune_parameters' exists in the section
        if 'tune_parameters' in config.options(each_section):
            para_info[each_section] = config.getlist(
                each_section, 'tune_parameters')
    return para_info


def rating_eval(model, metrics, test_set, user_based=False, verbose=False):
    """Evaluate model on provided rating metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RatingMetric`.

    test_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for evaluation.

    user_based: bool, optional, default: False
        Evaluation mode. Whether results are averaging based on number of users or number of ratings.

    verbose: bool, optional, default: False
        Output evaluation progress.

    Returns
    -------
    res: (List, List)
        Tuple of two lists:
         - average result for each of the metrics
         - average result per user for each of the metrics

    """

    if len(metrics) == 0:
        return [], []

    avg_results = []
    user_results = []

    (u_indices, i_indices, r_values) = test_set.uir_tuple
    r_preds = np.fromiter(
        tqdm(
            (
                model.rate(user_idx, item_idx).item()
                for user_idx, item_idx in zip(u_indices, i_indices)
            ),
            desc="Rating",
            disable=not verbose,
            miniters=100,
            total=len(u_indices),
        ),
        dtype="float",
    )

    gt_mat = test_set.csr_matrix
    pd_mat = csr_matrix((r_preds, (u_indices, i_indices)), shape=gt_mat.shape)

    test_user_indices = set(u_indices)
    for mt in metrics:
        if user_based:  # averaging over users
            user_results.append(
                {
                    user_idx: mt.compute(
                        gt_ratings=gt_mat.getrow(user_idx).data,
                        pd_ratings=pd_mat.getrow(user_idx).data,
                    ).item()
                    for user_idx in test_user_indices
                }
            )
            avg_results.append(
                sum(user_results[-1].values()) / len(user_results[-1]))
        else:  # averaging over ratings
            user_results.append({})
            avg_results.append(mt.compute(
                gt_ratings=r_values, pd_ratings=r_preds))

    return avg_results, user_results


# Helper functions

def cache_rankings(model, user_idx, item_indices,  k = -1):
    """Caches ranked items and scores for a user to avoid redundant computations.
    Parameters
    ----------
    model : object
        The recommendation model.
    user_idx : int
        User index.
    item_indices : list or None
        Indices of items to rank.
    k (int, required) – Cut-off length for recommendations, k=-1 will return ranked list of all items. This is more important for ANN to know the limit to avoid exhaustive ranking.

    Returns
    -------
    tuple
        Ranked items and scores for the user.
    """
    # with lock:  # Locking the cache access to avoid race conditions
    if not hasattr(model, 'ranked_items'):
        model.ranked_items = {}
    if not hasattr(model, 'item_scores'):
        model.item_scores = {}

    if user_idx in model.ranked_items and user_idx in model.item_scores:
        return model.ranked_items[user_idx], model.item_scores[user_idx]


    # item_idx2id = {v: k for k, v in test_set.iid_map.items()} # cornac item ID : raw item ID
    # user_idx2id = {v: k for k, v in test_set.uid_map.items()} # cornac user ID : raw user ID
    # item_id2idx = {k: v for k, v in test_set.iid_map.items()} # raw item ID : cornac item ID
    if not getattr(model, 'is_fitted', False):
        raise RuntimeError("Model is not fitted. Please call `model.fit()` before ranking.")
    
    item_rank, item_scores = model.rank( user_idx=user_idx, item_indices=item_indices, k=k) 
    # item_rank, item_scores = model.rank( user_idx=user_idx, item_indices=item_indices, k=k,item_idx2id = item_idx2id, user_idx2id = user_idx2id, item_id2idx =  item_id2idx) 


    # Cache the results for future use
    model.ranked_items[user_idx] = item_rank
    model.item_scores[user_idx] = item_scores
    

    return item_rank, item_scores


def ranking_eval(
    model,
    metrics,
    train_set,
    test_set,
    val_set=None,
    rating_threshold=1.0,
    exclude_unknowns=True,
    verbose=False,
):
    """Evaluate model on provided ranking metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RankingMetric`.

    train_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for model training. This will be used to exclude
        observations already appeared during training.

    test_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for evaluation.

    val_set: :obj:`cornac.data.Dataset`, optional, default: None
        Dataset to be used for model selection. This will be used to exclude
        observations already appeared during validation.

    rating_threshold: float, optional, default: 1.0
        The threshold to convert ratings into positive or negative feedback.

    exclude_unknowns: bool, optional, default: True
        Ignore unknown users and items during evaluation.

    verbose: bool, optional, default: False
        Output evaluation progress.

    Returns
    -------
    res: (List, List)
        Tuple of two lists:
         - average result for each of the metrics
         - average result per user for each of the metrics

    """

    if len(metrics) == 0:
        return [], []

    max_k = max(m.k for m in metrics)

    avg_results = []
    user_results = [{} for _ in enumerate(metrics)]

    test_mat = test_set.csr_matrix
    train_mat = train_set.csr_matrix
    val_mat = None if val_set is None else val_set.csr_matrix

    def pos_items(csr_row):
        return [
            item_idx
            for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
            if rating >= rating_threshold
        ]

    test_user_indices = set(test_set.uir_tuple[0])
    for user_idx in tqdm(
        test_user_indices, desc="Ranking", disable=not verbose, miniters=100
    ):
        test_pos_items = pos_items(test_mat.getrow(user_idx))
        if len(test_pos_items) == 0:
            continue

        # binary mask for ground-truth positive items
        u_gt_pos_mask = np.zeros(test_set.num_items, dtype="int")
        u_gt_pos_mask[test_pos_items] = 1

        val_pos_items = [] if val_mat is None else pos_items(val_mat.getrow(user_idx))
        train_pos_items = (
            pos_items(train_mat.getrow(user_idx))
            if user_idx < train_mat.shape[0]
            else []
        )

        # binary mask for ground-truth negative items, removing all positive items
        u_gt_neg_mask = np.ones(test_set.num_items, dtype="int")
        u_gt_neg_mask[test_pos_items + val_pos_items + train_pos_items] = 0

        # filter items being considered for evaluation
        if exclude_unknowns:
            u_gt_pos_mask = u_gt_pos_mask[: train_set.num_items]
            u_gt_neg_mask = u_gt_neg_mask[: train_set.num_items]

        item_indices = np.nonzero(u_gt_pos_mask + u_gt_neg_mask)[0]
        u_gt_pos_items = np.nonzero(u_gt_pos_mask)[0]
        u_gt_neg_items = np.nonzero(u_gt_neg_mask)[0]


        # Cache rankings and scores

        item_rank, item_scores = cache_rankings(
        model,   user_idx=user_idx, item_indices=item_indices,  k=-1)

        for i, mt in enumerate(metrics):
            mt_score = mt.compute(
                gt_pos=u_gt_pos_items,
                gt_neg=u_gt_neg_items,
                pd_rank=item_rank,
                pd_scores=item_scores,
                item_indices=item_indices,
            )
            user_results[i][user_idx] = mt_score

    # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        avg_results.append(sum(user_results[i].values()) / len(user_results[i]))

    return avg_results, user_results



def preprocess_data_for_Fragmentation(user_idx, test_set, model, metrics, item_indices):
    """
    Prepares data for the Fragmentation metrics. It requires recommendations for other users.

    Parameters
    ----------
    user_idx : int
        User index.
    test_set : Dataset
        The test dataset.
    model : object
        The recommender model.
    metrics : list
        List of fragmentation metrics.
    item_indices : list
        Indices of items to rank.

    Returns
    -------
    list
        Processed data for fragmentation metrics.
    """

    pd_other_users = []
    max_k = max(m.k for m in metrics)
    for i, mt in enumerate(metrics):
        if "Fragmentation" in mt.name:
            if len(model.ranked_items) > mt.n_samples:
                other_users = [key for key,
                               value in model.ranked_items.items()]

                # Exclude the current user (user_idx) from the candidate list
                other_users.remove(user_idx)
            else:
                test_user_indices = set(test_set.uir_tuple[0])
                other_users = list(test_user_indices)
                other_users.remove(user_idx)  # Exclude the current user

            # Sample from other users
            sampled_users = np.random.choice(
                other_users, size=mt.n_samples, replace=False)

            sample_rank = []

            # Separate cached and uncached samples
            for x in sampled_users:
                # model_ranked_items, _ = cache_rankings(
                #     model, x, item_indices, k=-1)
                # model_ranked_items, _ = cache_rankings(
                #     model, x, item_indices, k=-1)
                model_ranked_items, _ = cache_rankings(
        model,   user_idx=x, item_indices=item_indices, k=-1)

                if len(model_ranked_items) >= mt.k and mt.k > 0:
                    sample_rank.append(model_ranked_items[:mt.k])
                else:
                    sample_rank.append(model_ranked_items)

            pd_other_users.append(sample_rank)
        else:
            pd_other_users.append([])
    return pd_other_users


def diversity_eval( model,metrics, train_set,test_set, val_set= None, rating_threshold=1, exclude_unknowns = True, verbose=False):
    """Evaluates diversity metrics for a single user.

    Parameters
    ----------
    user_idx : int
        User index.
    metrics : list
        List of diversity metrics.
    model : object
        The recommender model.
    test_set : Dataset
        The test dataset.
    train_set : Dataset
        The training dataset.
    val_set : Dataset
        The validation dataset.
    exclude_unknowns : bool
        If True, excludes unknown items from evaluation.
    rating_threshold : float
        Threshold for positive ratings.

    Returns
    -------
    tuple
        User-specific metric results, user index, and computation times.
    """

    if len(metrics) == 0:
        return [], []
    max_k = max(m.k for m in metrics)
    avg_results = []
    user_results = [{} for _ in enumerate(metrics)]

    test_mat = test_set.csr_matrix
    train_mat = train_set.csr_matrix
    val_mat = None if val_set is None else val_set.csr_matrix

    def pos_items(csr_row):
        return [
            item_idx
            for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
            if rating >= rating_threshold
        ]

    test_user_indices = set(test_set.uir_tuple[0])
    
    # Creating user history dictionary
    user_history_dict = OrderedDict()
    for user_idx in test_user_indices:
        pos_item_idx = (
            pos_items(train_mat.getrow(user_idx))
            if user_idx < train_mat.shape[0]
            else []
        )
        user_history_dict[user_idx] = pos_item_idx
    
    globalProbs = []

    for i, mt in enumerate(metrics):
        if "Binomial" in mt.name:
            global_prob = mt.globalFeatureProbs(user_history_dict)
            globalProbs.append(global_prob)
        else:
            globalProbs.append([])
    pd_other_users = preprocess_data_for_Fragmentation(
        user_idx, test_set, model, metrics, item_indices=None)


    for user_idx in tqdm(
        test_user_indices, desc="Diversity", disable=not verbose, miniters=100
    ):
        test_pos_items = pos_items(test_mat.getrow(user_idx))
        if len(test_pos_items) == 0:
            continue

        # binary mask for ground-truth positive items
        u_gt_pos_mask = np.zeros(test_set.num_items, dtype="int")
        u_gt_pos_mask[test_pos_items] = 1

        val_pos_items = [] if val_mat is None else pos_items(
            val_mat.getrow(user_idx))
        train_pos_items = (
            pos_items(train_mat.getrow(user_idx))
            if user_idx < train_mat.shape[0]
            else []
        )

        # binary mask for ground-truth negative items, removing all positive items
        u_gt_neg_mask = np.ones(test_set.num_items, dtype="int")
        u_gt_neg_mask[test_pos_items + val_pos_items + train_pos_items] = 0

        # filter items being considered for evaluation
        if exclude_unknowns:
            u_gt_pos_mask = u_gt_pos_mask[: train_set.num_items]
            u_gt_neg_mask = u_gt_neg_mask[: train_set.num_items]

        item_indices = np.nonzero(u_gt_pos_mask + u_gt_neg_mask)[0]

        item_rank, item_scores = cache_rankings(
        model,   user_idx=user_idx, item_indices=item_indices,  k=-1)

        # item_rank, item_scores = cache_rankings(
        # model,   user_idx=user_idx, item_indices=item_indices, k=-1)
        pool_ids = np.arange(test_set.num_items)
        u_gt_rating = np.zeros(test_set.num_items)
        gt_mat = test_set.csr_matrix
        gd_row = gt_mat.getrow(user_idx)
        u_gt_rating[gd_row.indices] = gd_row.data


        for i, mt in enumerate(metrics):
            mt_score = mt.compute(
            pd_rank=item_rank,
            pd_scores=item_scores,
            rating_threshold=rating_threshold,
            gt_ratings=u_gt_rating,
            globalProb=globalProbs[i],
            user_history= user_history_dict[user_idx],
            pool=pool_ids,
            pd_other_users=pd_other_users[i]
        )
            if mt_score is None:
                pass
            else:
                user_results[i][user_idx] = mt_score


            # user_results[i][user_idx] = mt_score

     # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        values = user_results[i].values()
        avg_results.append(sum(values) / len(values) if values else 0)
        # avg_results.append(
        #     sum(user_results[i].values()) / len(user_results[i]))

    return avg_results, user_results

class BaseMethod:
    """Base Evaluation Method

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value)].

    fmt: str, default: 'UIR'
        Format of the input data. Currently, we are supporting:

        'UIR': User, Item, Rating
        'UIRT': User, Item, Rating, Timestamp

    rating_threshold: float, optional, default: 1.0
        Threshold used to binarize rating values into positive or negative feedback for
        model evaluation using ranking metrics (rating metrics are not affected).

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown users and items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.

    """

    def __init__(
            self,
            data=None,
            fmt="UIR",
            rating_threshold=1.0,
            seed=None,
            exclude_unknowns=True,
            verbose=False,
            **kwargs
    ):
        self.data = data
        self.fmt = fmt
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.rating_threshold = rating_threshold
        self.exclude_unknowns = exclude_unknowns
        self.verbose = verbose
        self.seed = seed
        self.rng = get_rng(seed)
        self.global_uid_map = kwargs.get("global_uid_map", OrderedDict())
        self.global_iid_map = kwargs.get("global_iid_map", OrderedDict())

        self.user_feature = kwargs.get("user_feature", None)
        self.user_text = kwargs.get("user_text", None)
        self.user_image = kwargs.get("user_image", None)
        self.user_graph = kwargs.get("user_graph", None)
        self.item_feature = kwargs.get("item_feature", None)
        self.item_text = kwargs.get("item_text", None)
        self.item_image = kwargs.get("item_image", None)
        self.item_graph = kwargs.get("item_graph", None)
        self.sentiment = kwargs.get("sentiment", None)
        self.review_text = kwargs.get("review_text", None)

        if verbose:
            print("rating_threshold = {:.1f}".format(rating_threshold))
            print("exclude_unknowns = {}".format(exclude_unknowns))

    @property
    def total_users(self):
        return len(self.global_uid_map)

    @property
    def total_items(self):
        return len(self.global_iid_map)

    @property
    def user_feature(self):
        return self.__user_feature

    @property
    def user_text(self):
        return self.__user_text

    @user_feature.setter
    def user_feature(self, input_modality):
        if input_modality is not None and not isinstance(
                input_modality, FeatureModality
        ):
            raise ValueError(
                "input_modality has to be instance of FeatureModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_feature = input_modality

    @user_text.setter
    def user_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError(
                "input_modality has to be instance of TextModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_text = input_modality

    @property
    def user_image(self):
        return self.__user_image

    @user_image.setter
    def user_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError(
                "input_modality has to be instance of ImageModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_image = input_modality

    @property
    def user_graph(self):
        return self.__user_graph

    @user_graph.setter
    def user_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError(
                "input_modality has to be instance of GraphModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_graph = input_modality

    @property
    def item_feature(self):
        return self.__item_feature

    @property
    def item_text(self):
        return self.__item_text

    @item_feature.setter
    def item_feature(self, input_modality):
        if input_modality is not None and not isinstance(
                input_modality, FeatureModality
        ):
            raise ValueError(
                "input_modality has to be instance of FeatureModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_feature = input_modality

    @item_text.setter
    def item_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError(
                "input_modality has to be instance of TextModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_text = input_modality

    @property
    def item_image(self):
        return self.__item_image

    @item_image.setter
    def item_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError(
                "input_modality has to be instance of ImageModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_image = input_modality

    @property
    def item_graph(self):
        return self.__item_graph

    @item_graph.setter
    def item_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError(
                "input_modality has to be instance of GraphModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_graph = input_modality

    @property
    def sentiment(self):
        return self.__sentiment

    @sentiment.setter
    def sentiment(self, input_modality):
        if input_modality is not None and not isinstance(
                input_modality, SentimentModality
        ):
            raise ValueError(
                "input_modality has to be instance of SentimentModality but {}".format(
                    type(input_modality)
                )
            )
        self.__sentiment = input_modality

    @property
    def review_text(self):
        return self.__review_text

    @review_text.setter
    def review_text(self, input_modality):
        if input_modality is not None and not isinstance(
                input_modality, ReviewModality
        ):
            raise ValueError(
                "input_modality has to be instance of ReviewModality but {}".format(
                    type(input_modality)
                )
            )
        self.__review_text = input_modality

    def _reset(self):
        """Reset the random number generator for reproducibility"""
        self.rng = get_rng(self.seed)
        self.test_set = self.test_set.reset()

    @staticmethod
    def organize_metrics(metrics):
        """Organize metrics according to their types (rating or raking)

        Parameters
        ----------
        metrics: :obj:`iterable`
            List of metrics.

        """
        if isinstance(metrics, dict):
            rating_metrics = metrics.get("rating", [])
            ranking_metrics = metrics.get("ranking", [])
            diversity_metrics = metrics.get("diversity", [])
        elif isinstance(metrics, list):
            rating_metrics = []
            ranking_metrics = []
            diversity_metrics = []
            for mt in metrics:
                if isinstance(mt, RatingMetric):
                    rating_metrics.append(mt)
                elif isinstance(mt, RankingMetric) and hasattr(mt.k, "__len__"):
                    ranking_metrics.extend(
                        [mt.__class__(k=_k) for _k in sorted(set(mt.k))]
                    )
                elif isinstance(mt, DiversityMetric):
                    diversity_metrics.append(mt)
                else:
                    ranking_metrics.append(mt)
        else:
            raise ValueError("Type of metrics has to be either dict or list!")

        # sort metrics by name
        rating_metrics = sorted(rating_metrics, key=lambda mt: mt.name)
        ranking_metrics = sorted(ranking_metrics, key=lambda mt: mt.name)
        diversity_metrics = sorted(
            diversity_metrics, key=lambda mt: mt.name)

        return rating_metrics, ranking_metrics,diversity_metrics
        # Automatically decide whether to include diversity_metrics
        # if diversity_metrics:
        #     return rating_metrics, ranking_metrics, diversity_metrics
        # return rating_metrics, ranking_metrics  # Keep old behavior

    def _build_datasets(self, train_data, test_data, val_data=None):
        self.train_set = Dataset.build(
            data=train_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=False,
        )
        if self.verbose:
            print("---")
            print("Training data:")
            print("Number of users = {}".format(self.train_set.num_users))
            print("Number of items = {}".format(self.train_set.num_items))
            print("Number of ratings = {}".format(self.train_set.num_ratings))
            print("Max rating = {:.1f}".format(self.train_set.max_rating))
            print("Min rating = {:.1f}".format(self.train_set.min_rating))
            print("Global mean = {:.1f}".format(self.train_set.global_mean))

        self.test_set = Dataset.build(
            data=test_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=self.exclude_unknowns,
        )
        if self.verbose:
            print("---")
            print("Test data:")
            print("Number of users = {}".format(len(self.test_set.uid_map)))
            print("Number of items = {}".format(len(self.test_set.iid_map)))
            print("Number of ratings = {}".format(self.test_set.num_ratings))
            print(
                "Number of unknown users = {}".format(
                    self.test_set.num_users - self.train_set.num_users
                )
            )
            print(
                "Number of unknown items = {}".format(
                    self.test_set.num_items - self.train_set.num_items
                )
            )

        if val_data is not None and len(val_data) > 0:
            self.val_set = Dataset.build(
                data=val_data,
                fmt=self.fmt,
                global_uid_map=self.global_uid_map,
                global_iid_map=self.global_iid_map,
                seed=self.seed,
                exclude_unknowns=self.exclude_unknowns,
            )
            if self.verbose:
                print("---")
                print("Validation data:")
                print("Number of users = {}".format(len(self.val_set.uid_map)))
                print("Number of items = {}".format(len(self.val_set.iid_map)))
                print("Number of ratings = {}".format(self.val_set.num_ratings))

        if self.verbose:
            print("---")
            print("Total users = {}".format(self.total_users))
            print("Total items = {}".format(self.total_items))

        self.train_set.total_users = self.total_users
        self.train_set.total_items = self.total_items

    def save(self, directory_path):
        """Save the train, test, and optionally validation sets to separate files in the same directory.
        Save all attributes of the instance.
        """
        # Ensure the directory exists
        os.makedirs(directory_path, exist_ok=True)

        # Save the train, test, and validation datasets
        train_path = os.path.join(directory_path, 'train_set.pkl')
        test_path = os.path.join(directory_path, 'test_set.pkl')
        val_path = os.path.join(
            directory_path, 'val_set.pkl') if self.val_set is not None else None

        with open(train_path, 'wb') as f:
            pickle.dump(self.train_set, f)

        with open(test_path, 'wb') as f:
            pickle.dump(self.test_set, f)

        if self.val_set is not None:
            with open(val_path, 'wb') as f:
                pickle.dump(self.val_set, f)

        # Save all other attributes of the instance (besides the datasets)
        attributes_path = os.path.join(
            directory_path, 'dataset_attributes.pkl')
        attributes = {
            'data': self.data,
            'fmt': self.fmt,
            'rating_threshold': self.rating_threshold,
            'exclude_unknowns': self.exclude_unknowns,
            'verbose': self.verbose,
            'seed': self.seed,

            'global_uid_map': self.global_uid_map,
            'global_iid_map': self.global_iid_map,
            'user_feature': self.user_feature,
            'user_text': self.user_text,
            'user_image': self.user_image,
            'user_graph': self.user_graph,
            'item_feature': self.item_feature,
            'item_text': self.item_text,
            'item_image': self.item_image,
            'item_graph': self.item_graph,
            'sentiment': self.sentiment,
            'review_text': self.review_text
        }

        with open(attributes_path, 'wb') as f:
            pickle.dump(attributes, f)

        print(f"Attributes and datasets saved to {directory_path}")

    @classmethod
    def load(cls, directory_path):
        """Load all attributes and datasets of the BaseMethod instance."""

        # Load datasets
        train_path = os.path.join(directory_path, 'train_set.pkl')
        test_path = os.path.join(directory_path, 'test_set.pkl')
        val_path = os.path.join(directory_path, 'val_set.pkl') if os.path.exists(
            os.path.join(directory_path, 'val_set.pkl')) else None

        attributes_path = os.path.join(
            directory_path, 'dataset_attributes.pkl')

        # Check if the necessary files exist, otherwise raise an error
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Train set file not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test set file not found at {test_path}")
        if not os.path.exists(attributes_path):
            raise FileNotFoundError(
                f"Attributes file not found at {attributes_path}")
        if val_path is not None and not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Validation set file not found at {val_path}")

        with open(train_path, 'rb') as f:
            train_set = pickle.load(f)

        with open(test_path, 'rb') as f:
            test_set = pickle.load(f)

        val_set = None
        if val_path is not None:
            with open(val_path, 'rb') as f:
                val_set = pickle.load(f)

        # Load other attributes

        with open(attributes_path, 'rb') as f:
            attributes = pickle.load(f)

        # Recreate the instance
        instance = cls(
            data=attributes['data'],
            fmt=attributes["fmt"],
            rating_threshold=attributes['rating_threshold'],
            seed=attributes['seed'],
            exclude_unknowns=attributes['exclude_unknowns'],
            verbose=attributes['verbose']
        )

        # Set the datasets
        instance.train_set = train_set
        instance.test_set = test_set
        instance.val_set = val_set

        # Set the remaining attributes
        instance.global_uid_map = attributes['global_uid_map']
        instance.global_iid_map = attributes['global_iid_map']
        instance.user_feature = attributes['user_feature']
        instance.user_text = attributes['user_text']
        instance.user_image = attributes['user_image']
        instance.user_graph = attributes['user_graph']
        instance.item_feature = attributes['item_feature']
        instance.item_text = attributes['item_text']
        instance.item_image = attributes['item_image']
        instance.item_graph = attributes['item_graph']
        instance.sentiment = attributes['sentiment']
        instance.review_text = attributes['review_text']

        print(f"Attributes and datasets loaded from {directory_path}")

        return instance

    def _build_modalities(self):
        for user_modality in [
            self.user_feature,
            self.user_text,
            self.user_image,
            self.user_graph,
        ]:
            if user_modality is None:
                continue
            user_modality.build(
                id_map=self.global_uid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for item_modality in [
            self.item_feature,
            self.item_text,
            self.item_image,
            self.item_graph,
        ]:
            if item_modality is None:
                continue
            item_modality.build(
                id_map=self.global_iid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for modality in [self.sentiment, self.review_text]:
            if modality is None:
                continue
            modality.build(
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        self.add_modalities(
            user_feature=self.user_feature,
            user_text=self.user_text,
            user_image=self.user_image,
            user_graph=self.user_graph,
            item_feature=self.item_feature,
            item_text=self.item_text,
            item_image=self.item_image,
            item_graph=self.item_graph,
            sentiment=self.sentiment,
            review_text=self.review_text,
        )

    def add_modalities(self, **kwargs):
        """
        Add successfully built modalities to all datasets. This is handy for
        seperately built modalities that are not invoked in the build method.
        """
        self.user_feature = kwargs.get("user_feature", None)
        self.user_text = kwargs.get("user_text", None)
        self.user_image = kwargs.get("user_image", None)
        self.user_graph = kwargs.get("user_graph", None)
        self.item_feature = kwargs.get("item_feature", None)
        self.item_text = kwargs.get("item_text", None)
        self.item_image = kwargs.get("item_image", None)
        self.item_graph = kwargs.get("item_graph", None)
        self.sentiment = kwargs.get("sentiment", None)
        self.review_text = kwargs.get("review_text", None)

        for data_set in [self.train_set, self.test_set, self.val_set]:
            if data_set is None:
                continue
            data_set.add_modalities(
                user_feature=self.user_feature,
                user_text=self.user_text,
                user_image=self.user_image,
                user_graph=self.user_graph,
                item_feature=self.item_feature,
                item_text=self.item_text,
                item_image=self.item_image,
                item_graph=self.item_graph,
                sentiment=self.sentiment,
                review_text=self.review_text,
            )

    def build(self, train_data, test_data, val_data=None):
        if train_data is None or len(train_data) == 0:
            raise ValueError("train_data is required but None or empty!")
        if test_data is None or len(test_data) == 0:
            raise ValueError("test_data is required but None or empty!")

        self.global_uid_map.clear()
        self.global_iid_map.clear()

        self._build_datasets(train_data, test_data, val_data)
        self._build_modalities()

        return self

    @staticmethod
    def eval(
        model,
        train_set,
        test_set,
        val_set,
        rating_threshold,
        exclude_unknowns,
        user_based,
        rating_metrics,
        ranking_metrics,
        diversity_metrics,
        verbose,
    ):
        """Running evaluation for rating and ranking metrics respectively."""
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()
        metric_avg_time = OrderedDict()
        user_info = OrderedDict()
        model_parameter = OrderedDict()
        avg_results, user_results = rating_eval(
            model=model,
            metrics=rating_metrics,
            test_set=test_set,
            user_based=user_based,
            verbose=verbose,
        )
        for i, mt in enumerate(rating_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

    
        avg_results, user_results = ranking_eval(
            model=model,
            metrics=ranking_metrics,
            train_set=train_set,
            test_set=test_set,
            val_set=val_set,
            rating_threshold=rating_threshold,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            
        )
        for i, mt in enumerate(ranking_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]
          
        avg_results, user_results = diversity_eval(
            model=model,
            metrics= diversity_metrics,
            train_set=train_set,
            test_set=test_set,
            val_set=val_set,
            rating_threshold=rating_threshold,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
        )
        for i, mt in enumerate(diversity_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]
    

        return Result(model.name, metric_avg_results, metric_user_results,
                      user_info=user_info, model_parameter=model_parameter)

    def evaluate(self, model, metrics, user_based, show_validation=True,  **kwargs):
        """Evaluate given models according to given metrics

        Parameters
        ----------
        model: :obj:`cornac.models.Recommender`
            Recommender model to be evaluated.

        metrics: :obj:`iterable`
            List of metrics.

        user_based: bool, required
            Evaluation strategy for the rating metrics. Whether results
            are averaging based on number of users or number of ratings.

        show_validation: bool, optional, default: True
            Whether to show the results on validation set (if exists).

        kwargs:
        - `train_mode` (optional): A boolean flag indicating whether the model is in training mode. Defaults to True.

        Returns
        -------
        res: :obj:`cornac.experiment.Result`
        """
        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()
        self.organize_metrics(metrics)

        # Extract `train_mode` from kwargs; defaults to True if not provided.
        # This parameter indicates whether to execute the training phase.
        # Set to False to skip training and directly evaluate the pre-trained model.
        train_mode = kwargs.get('train_mode', True)

        ###########
        # FITTING #
        ###########
        train_time = 0
        save_path = model.name+"_trained"
        if train_mode:
            if self.verbose:
                print("\n[{}] Training started!".format(model.name))

            start = time.time()
            model.fit(self.train_set, self.val_set)
            train_time = time.time() - start
            # model.save(save_path)

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        rating_metrics, ranking_metrics, diversity_metrics = self.organize_metrics(metrics)

        start = time.time()
        model.transform(self.test_set)
        test_result = self.eval(
            model=model,
            train_set=self.train_set,
            test_set=self.test_set,
            val_set=self.val_set,
            rating_threshold=self.rating_threshold,
            exclude_unknowns=self.exclude_unknowns,
            rating_metrics=rating_metrics,
            ranking_metrics=ranking_metrics,
            diversity_metrics= diversity_metrics,
            user_based=user_based,
            verbose=self.verbose,
        )
        test_time = time.time() - start
        test_result.metric_avg_results["Train (s)"] = train_time
        test_result.metric_avg_results["Test (s)"] = test_time
        # model.save_recommendations(save_path)
        # test_result.save(save_path)

        all_para = save_model_parameter()
        parameter_values = {}

        if model.name in all_para:
            para = all_para[model.name]
            for att in dir(model):
                if att in para:
                    res = getattr(model, att)
                    if isinstance(res, list):
                        res = ','.join(str(e) for e in res)
                    parameter_values[att] = res
        test_result.model_parameter = parameter_values

        val_result = None
        if show_validation and self.val_set is not None:
            start = time.time()
            model.transform(self.val_set)
            val_result = self.eval(
                model=model,
                train_set=self.train_set,
                test_set=self.val_set,
                val_set=None,
                rating_threshold=self.rating_threshold,
                exclude_unknowns=self.exclude_unknowns,
                rating_metrics=rating_metrics,
                ranking_metrics=ranking_metrics,
                diversity_metrics= diversity_metrics,
                user_based=user_based,
                verbose=self.verbose,
            )
            val_time = time.time() - start
            val_result.metric_avg_results["Time (s)"] = val_time

        return test_result, val_result

    @classmethod
    def from_splits(
            cls,
            train_data,
            test_data,
            val_data=None,
            fmt="UIR",
            rating_threshold=1.0,
            exclude_unknowns=False,
            seed=None,
            verbose=False,
            **kwargs
    ):
        """Constructing evaluation method given data.

        Parameters
        ----------
        train_data: array-like
            Training data

        test_data: array-like
            Test data

        val_data: array-like, optional, default: None
            Validation data

        fmt: str, default: 'UIR'
            Format of the input data. Currently, we are supporting:

            'UIR': User, Item, Rating
            'UIRT': User, Item, Rating, Timestamp

        rating_threshold: float, default: 1.0
            Threshold to decide positive or negative preferences.

        exclude_unknowns: bool, default: False
            Whether to exclude unknown users/items in evaluation.

        seed: int, optional, default: None
            Random seed for reproduce the splitting.

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        method: :obj:`<cornac.eval_methods.BaseMethod>`
            Evaluation method object.

        """
        method = cls(
            fmt=fmt,
            rating_threshold=rating_threshold,
            exclude_unknowns=exclude_unknowns,
            seed=seed,
            verbose=verbose,
            **kwargs
        )

        return method.build(
            train_data=train_data, test_data=test_data, val_data=val_data
        )
