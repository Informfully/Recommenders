import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
from scipy.stats import binom
import math
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
from numpy.linalg import norm

# requirement: scikit-learn >= 1.1


class DiversityMetric:
    """Diversity Metric.

    Parameters
    ----------
    type: string, value: 'diversity'
        Type of the metric, e.g., "ranking", "rating","diversity".

    name: string, default: None
        Name of the measure.

    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    """
    def __init__(self, name=None, k=-1, higher_better=False):
        assert hasattr(k, "__len__") or k == -1 or k > 0

        self.type = "diversity"
        self.name = name
        self.k = k
        self.higher_better = higher_better

    def compute(self, **kwargs):
        raise NotImplementedError()


class NDCG_score(DiversityMetric):
    """Standard Normalized Discount Cumulative Gain by using ratings.

    Parameters
    ----------
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    References
    ----------
    C. Clarke, M. Kolla, G. Cormack, O. Vechtomova, A. Ashkan, S. Büttcher, I. MacKinnon
    Novelty and diversity in information retrieval evaluation
    Proc. 31st Int. ACM SIGIR Conf. Res. Dev. Infor. Retrieval (SIGIR ’08), Singapore, Singapore (2008), pp. 659-666

    """
    def __init__(self,  k=-1):
        DiversityMetric.__init__(self, name="NDCG_score@{}".format(k), k=k)

    @staticmethod
    def dcg_score(relevance_score):
        """Compute Discounted Cumulative Gain.

        Parameters
        ----------
        relevance_score: numpy array,
            ground_truth item relevance 
            of a recommendation list.

        Returns
        -------
        discounted_cumulative_gain: float
            The DCG score.

        """
        discounts = np.log2(np.arange(2, relevance_score.size + 2))
        dcg = np.sum(relevance_score / discounts)
        return dcg

    def compute(self, pd_rank, gt_ratings, **kwargs):
        """Compute Normalized Discounted Cumulative Gain score.

        Parameters
        ----------
        pd_rank: Numpy array
            Ranked array of item indices.(Item ranking prediction). 
        gt_ratings: Numpy array
            Item relevance of all items in the test set for a given user.
            Need to fetch item relevance for each
            item in the pd_rank.

        **kwargs: For compatibility

        Returns
        -------
        ndcg: A scalar
            Normalized Discounted Cumulative Gain score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        truncated_pd_score = []
        for x in truncated_pd_rank:
            if x < gt_ratings.size:
                truncated_pd_score.append(gt_ratings[x])
        try:
            actual_dcg = self.dcg_score(np.asarray(truncated_pd_score))
            sorted_score = np.sort(np.asarray(truncated_pd_score))[::-1]
            best_dcg = self.dcg_score(sorted_score)
            ndcg = 0
            if (best_dcg > 0):
                ndcg = actual_dcg / best_dcg
            return ndcg
        except ValueError:
            return None


class Alpha_NDCG(DiversityMetric):
    """alpha-nDCG by using genres from the user's historical data.

    Parameters
    ----------

    item_genre: dictionary.
        A dictionary that maps item indices to the item genres array(numpy array).
        The item genres array must be an ordered list,
        where a value of 0 indicates that the item does not belong to a certain category at the corresponding position
        in the genre list, and 1 indicates that the item belong to a certain category.
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.
    alpha: A parameter to reward novelty. The default value is 0.5.
        When the parameter "alpha" is set to 0, the alpha-nDCG metric aligns with the standard nDCG.

    References
    ----------
    C. Clarke, M. Kolla, G. Cormack, O. Vechtomova, A. Ashkan, S. Büttcher, I. MacKinnon
    Novelty and diversity in information retrieval evaluation
    Proc. 31st Int. ACM SIGIR Conf. Res. Dev. Infor. Retrieval (SIGIR ’08), Singapore, Singapore (2008), pp. 659-666

    """
    def __init__(self, item_genre, alpha=0.5, k=-1):
        DiversityMetric.__init__(self, name="Alpha_NDCG@{}".format(k), k=k)
        if isinstance(item_genre, dict):
            self.item_genre = item_genre
        else:
            raise ValueError(
                "alpha_NDCG received an invalid item_genre. item_genre "
                "must be a dictionary."
            )
        if (isinstance(alpha, (int, float))) and alpha >= 0 and alpha <= 1:
            self.alpha = alpha
        else:
            self.alpha = 0.5

    def alpha_Gain(self, J, alpha):
        """alpha-gain

        Parameters
        ----------
        J: 2d nd.array, relevant categories of all recommended items.
        alpha: a parameter to reward novelty.

        Returns
        -------
        g: list of item gains

        """
        g = np.zeros(J.shape[0])
        r = np.zeros((J.shape[0], J.shape[1]))
        for i in range(J.shape[1]):
            g[0] += J[0][i]
            r[0][i] += J[0][i]
        for k in range(1, J.shape[0]):
            for i in range(J.shape[1]):
                g[k] += np.sum(J[k][i]*(1-alpha)**(r[k-1][i]))
                r[k][i] = r[k-1][i]+J[k][i]
        return g

    def alpha_cumulative_gain(self, g):
        """alpha-cumulative gain (alpha-cg)

        Parameters
        ----------
        g: list of gains

        Returns
        -------
        cumulative gain of recommendation

        """
        return np.array([np.sum(g[np.arange(i+1)]) for i in range(g.shape[0])])

    def alpha_dcg(self, gain):
        """computes alpha-discounted cumulative gain (alpha-dcg)

        Parameters
        ----------
        gain: cumulative gain

        Returns
        -------
        dcg: list

        """
        discounts = np.log2(np.arange(2, gain.size + 2))
        dcg = [np.sum(gain[np.arange(i+1)] / discounts[np.arange(i+1)])
               for i in range(gain.size)]
        return dcg

    def alpha_ndcg(self, J, alpha):
        """alpha-normalized discounted cumulative gain (alpha-nDCG)

        Parameters
        ----------
        J: 2d nd.array, relevant categories of all recommended items.
        alpha: a parameter to reward novelty.

        Returns
        -------
        alpha_ndcg

        """
        score = self.alpha_Gain(J, alpha)
        actual_dcg = self.alpha_dcg(score)
        sorted_score = np.sort(score)[::-1]
        ideal_dcg = self.alpha_dcg(sorted_score)
        if all(v == 0 for v in ideal_dcg):
            alpha_ndcg = [0 for i in range(score.size)]
            return alpha_ndcg
        else:
            alpha_ndcg = [actual_dcg[i] / ideal_dcg[i]
                          for i in range(score.size)]
            return alpha_ndcg

    def compute(self, pd_rank, user_history, **kwargs):
        """Compute alpha-Normalized Discounted Cumulative Gain score.

        Parameters
        ----------
        pd_rank: Numpy array
            Ranked array of item indices.(Item ranking prediction). 
        user_history: Numpy array
            Item read/used by a user in the history.
        **kwargs: For compatibility

        Returns
        -------
        alpha_ndcg: A scalar
            alpha-NDCG score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        user_interacted_item = user_history
        rec = [self.item_genre[i] if i in self.item_genre.keys()
               else None for i in truncated_pd_rank]
        recommendation = np.array([x for x in rec if x is not None])
        his = [self.item_genre[i] if i in self.item_genre.keys(
        ) else None for i in user_interacted_item]
        history = np.array([x for x in his if x is not None])
        col = np.nonzero(np.any(history != 0, axis=0))[0]
        if len(col) == 0 or len(recommendation) == 0:
            return None
        try:
            relevant_categories = recommendation[:, col]
            result = self.alpha_ndcg(relevant_categories, self.alpha)
            return result[-1]
        except ValueError:
            return None


class GiniCoeff(DiversityMetric):
    """Gini coefficient.

    Parameters
    ----------
    item_genre: dictionary.
        A dictionary that maps item indices to the item genres array(numpy array).
        The item genres array must be an ordered list,
        where a value of 0 indicates that the item does not belong to a certain category at the corresponding position
        in the genre list, and 1 indicates that the item belong to a certain category.

    k: int, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    References
    ----------
    Ricci, F., Rokach, L. and Shapira, B., 2015. Recommender systems: introduction and challenges. Recommender systems handbook, pp.1-34.
    """
    def __init__(self, item_genre,name= None,  k=-1):
        # DiversityMetric.__init__(self, name="GiniCoeff@{}".format(k), k=k)
        if name is None:
            DiversityMetric.__init__(self, name="GiniCoeff@{}".format(k), k=k)
        else:
            DiversityMetric.__init__(self, name="{}@{}".format(name, k), k=k)
        if isinstance(item_genre, dict):
            self.item_genre = item_genre
        else:
            raise ValueError(
                "GiniCoeff received an invalid item_genre. item_genre "
                "must be a dictionary."
            )

    # Original compute function
    '''
    def compute(self, pd_rank, **kwargs):
        """Compute Gini coefficient metric.

        Parameters
        ----------
        pd_rank: Numpy array
            Ranked array of item indices.(Item ranking prediction). 

        **kwargs: For compatibility

        Returns
        -------
        A scalar
            Gini coefficient score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        recommend_items_genre = []
        for x in truncated_pd_rank:
            if x in self.item_genre.keys():
                recommend_items_genre.append(list(self.item_genre[x]))
        proportion = []
        if len(recommend_items_genre) > 0:
            for i in range(len(recommend_items_genre[0])):
                column = [row[i] for row in recommend_items_genre]
                count = Counter(column)
                proportion.append(count[1]/len(column))
        else:
            return None
        proportion_standard = []
        if sum(proportion) > 0:
            proportion_standard = [i/sum(proportion) for i in proportion]
        try:
            sort_p = sorted(proportion_standard)
            n = len(sort_p)
            G = 0
            for index in range(len(sort_p)):
                j = index + 1
                G += (2 * j - n - 1) * sort_p[index]
            return G / (n - 1)
        except ValueError:
            return None
    '''
    # New compute function, add the case when pd_rank is None and calculate the diversity for the entire dataset.

    def compute(self, pd_rank, **kwargs):
        """Compute Gini coefficient metric.

        Parameters
        ----------
        pd_rank: Numpy array or None
            Ranked array of item indices.(Item ranking prediction).
            If None, compute the diversity for the entire dataset.

        **kwargs: For compatibility

        Returns
        -------
        A scalar
            Gini coefficient score.

        """
        if pd_rank is None:
            item_genre_data = self.item_genre
        else:
            if self.k > 0:
                truncated_pd_rank = pd_rank[:self.k]
            else:
                truncated_pd_rank = pd_rank

            recommend_items_genre = []
            for x in truncated_pd_rank:
                if x in self.item_genre.keys():
                    recommend_items_genre.append(list(self.item_genre[x]))

            item_genre_data = {}
            for idx, genre_vector in enumerate(recommend_items_genre):
                item_genre_data[idx] = genre_vector

        # Calculate diversity based on item_genre_data
        recommend_items_genre = []
        for item_index, genre_vector in item_genre_data.items():
            recommend_items_genre.append(genre_vector)
        proportion = []
        if len(recommend_items_genre) > 0:
            for i in range(len(recommend_items_genre[0])):
                column = [row[i] for row in recommend_items_genre]
                count = Counter(column)
                proportion.append(count[1] / len(column))
        else:
            return None

        proportion_standard = []
        if sum(proportion) > 0:
            proportion_standard = [i / sum(proportion) for i in proportion]

        try:
            sort_p = sorted(proportion_standard)
            n = len(sort_p)
            G = 0
            for index in range(len(sort_p)):
                j = index + 1
                G += (2 * j - n - 1) * sort_p[index]
            return G / (n - 1)
        except ValueError:
            return None


def relevance(rel, rating_threshold=1):
    """For EILD metric. Compute the probability of items being liked by a heuristic mapping using rating values.
    The paper adopted [(2**max(0, i - rating_threshold) -1)/g_max for i in rel],
    but the paper used a modified formula in the experiment. We implemented the modified version.

    Parameters
    ----------
    rel: Numpy array
        The relevance/rating of each recommended item in a recommendation list. 
    rating_threshold: (float, optional, default: 1.0)
        Threshold used to separate rating values into positive or negative feedback.

    Returns
    -------
    A list of
        probability of relevance.

    References
    ----------
    S. Vargas: New approaches to diversity and novelty in recommender systems. 
    Proc. 4th BCS-IRSG Symp. Future Directions Inf. Access (FDIA 2011), Koblenz, Germany (2011), pp. 8-13.

    """
    if len(rel) == 0:
        return []
    g_max = 2 ** max(0, max(rel) - rating_threshold)
    return [(2**max(0, i - rating_threshold))/g_max for i in rel]


def discount(x, disc_type="exponential", base=0.9):
    """For EILD metric. Compute the probability of recommended items been discovered in a list.

    Parameters
    ----------
    x: int.
        Discount for an item at position l knowing that position k has been reached.
        x = max(l - k,0)

    disc_type:(Optional) String. Type of the discount method used. 
        "exponential" or "logarithmic" or "reciprocal" or "nodiscount".
        Default value is "exponential".

    base: (Optional) float between 0 and 1.
        A probability to represent at each position in the ranked recommendation list, 
        the user makes a decision whether or not to continue. 

    Returns
    -------
    A scalar
        The probability of items been discovered.

    References
    ----------
    S. Vargas: New approaches to diversity and novelty in recommender systems.
    Proc. 4th BCS-IRSG Symp. Future Directions Inf. Access (FDIA 2011), Koblenz, Germany (2011), pp. 8-13.

    """
    if disc_type == "logarithmic":
        return 1/np.log2(x + 2.0)
    elif disc_type == "exponential":
        return base**x
    elif disc_type == "reciprocal":
        return 1 / (x + 1.0)
    elif disc_type == "nodiscount":
        return 1.0


class EILD(DiversityMetric):
    """Expected intra-list diversity.

    Parameters
    ----------
    item_feature: A dictionary that maps item indices to their feature vectors. 
        The feature vector must be provided using numpy array.
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.
    disc_type: (Optional) String. Type of the discount method used. 
        Available choices are "exponential" or "logarithmic" or "reciprocal" or "nodiscount". 
        Default value is "exponential".
    base: (Optional) Float between 0 and 1. A probability to represent at each position in the ranked recommendation list, 
        the user makes a decision whether or not to continue. Default base =0.9.

    References
    ----------
    S. Vargas: New approaches to diversity and novelty in recommender systems. 
    Proc. 4th BCS-IRSG Symp. Future Directions Inf. Access (FDIA 2011), Koblenz, Germany (2011), pp. 8-13.

    """
    def __init__(self, item_feature,  name = None, k=-1, disc_type="exponential", base=0.9):
        # DiversityMetric.__init__(self, name="EILD@{}".format(k), k=k)
        if name is None:
            DiversityMetric.__init__(self, name="EILD@{}".format(k), k=k)
        else:
            DiversityMetric.__init__(self, name="{}@{}".format(name, k), k=k)
        if isinstance(item_feature, dict):
            self.item_feature = item_feature
        else:
            raise ValueError(
                "EILD received an invalid item_feature. item_feature "
                "must be a dictionary."
            )
        if isinstance(disc_type, str) and disc_type.lower() in ["exponential", "logarithmic",
                                                                "reciprocal", "nodiscount"]:
            self.disc_type = disc_type
        else:
            self.disc_type = "exponential"
        if (isinstance(base, (int, float))) and base > 0 and base <= 1:
            self.base = base
        else:
            self.base = 0.9

    def compute(self, pd_rank, gt_ratings, rating_threshold=1.0, **kwargs):
        """Compute Expected intra-list diversity score.

        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.
        gt_ratings: Numpy array
            Item relevance of all items in the test set for a given user.
            Need to fetch item relevance for each
            item in the pd_rank.
        **kwargs: For compatibility

        Returns
        -------
        eild: A scalar
            Expected intra-list diversity score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        norm = 0
        eild = 0
        rel_vec = [gt_ratings[i] for i in truncated_pd_rank]
        rel = relevance(rel_vec,  rating_threshold)
        pd_vec0 = [list(self.item_feature[i]) if i in self.item_feature.keys() else [
        ] for i in truncated_pd_rank]
        pd_vec = [x for x in pd_vec0 if x]
        try:
            for i in range(len(pd_vec)):
                ieild = 0
                inorm = 0
                for j in range(len(pd_vec)):
                    if i == j:
                        continue
                    else:
                        dist = cosine(np.array(pd_vec[i]), np.array(pd_vec[j]))
                        w = discount(
                            max(0, j - i), disc_type=self.disc_type, base=self.base) * rel[j]
                        ieild += w * dist
                        inorm += w
                if inorm > 0:
                    eild += discount(i, disc_type=self.disc_type,
                                     base=self.base) * rel[i] * ieild/inorm
                norm += discount(i, disc_type=self.disc_type, base=self.base)
            if (norm > 0):
                eild = eild/norm
            return eild
        except ValueError:
            return None


class ILD(DiversityMetric):
    """Intra-list diversity.

    Parameters
    ----------
    item_feature: A dictionary that maps item indices to their feature vectors. 
        The feature vector must be provided using numpy array.
    distance_type: (Optional) String for configuring distance measure. 
        ‘correlation’, ‘cosine’,‘euclidean’ or ‘jaccard’. 
        By default, use cosine distance metric.
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    References
    ----------
    https://www.academia.edu/2655896/Improving_recommendation_diversity

    
    """
    def __init__(self,  item_feature, name= None, k=-1, distance_type="cosine"):
        if name is None:
            DiversityMetric.__init__(self, name="ILD@{}".format(k), k=k)
        else:
            DiversityMetric.__init__(self, name="{}@{}".format(name, k), k=k)
        if isinstance(item_feature, dict):
            self.item_feature = item_feature
        else:
            raise ValueError(
                "ILD received an invalid item_feature. item_feature "
                "must be a dictionary."
            )
        if isinstance(distance_type, str) and distance_type.lower() in ["jaccard", "correlation",
                                                                        "cosine", "euclidean"]:
            self.distance_type = distance_type
        else:
            self.distance_type = "cosine"

    def compute(self, pd_rank, **kwargs):
        """Compute Intra-list diversity score.

        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.

        **kwargs: For compatibility

        Returns
        -------
        ild: A scalar
            Intra-list diversity score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        pd_vec0 = [list(self.item_feature[i]) if i in self.item_feature.keys() else [
        ] for i in truncated_pd_rank]
        pd_vec = [x for x in pd_vec0 if x]
        try:
            distance = cdist(pd_vec, pd_vec, metric=self.distance_type)
            upper_right = np.triu_indices(distance.shape[0], k=1)
            if distance[upper_right].size == 0:
                ild = 0
            else:
                ild = np.mean(distance[upper_right])
            return ild
        except ValueError:
            return None


class Binomial(DiversityMetric):
    """Binomial metric.

    Parameters
    ----------
    item_genre: dictionary.
        A dictionary that maps item indices to the item genres array(numpy array).
        The item genres array must be an ordered list,
        where a value of 0 indicates that the item does not belong to a certain genre at the corresponding position
        in the genre list, and 1 indicates that the item belong to a certain genre.
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.
    alpha: (Optional) float between 0 and 1. 
        A parameter to specify the weight of local probability when estimating probability.
        The default value is 0.9.

    References
    ----------
    S. Vargas, L. Baltrunas, A. Karatzoglou, P. Castells
    Coverage, redundancy and size-awareness in genre diversity for recommender systems
    Proc. RecSys 2014, 8th ACM Conf. Recomm. Syst., Foster City, Silicon Valley, USA (2014), pp. 209-216

    """
    def __init__(self, item_genre, name = None, k=-1, alpha=0.9):
        # DiversityMetric.__init__(self, name="Binomial@{}".format(k), k=k)
        if name is None:
            DiversityMetric.__init__(self, name="Binomial@{}".format(k), k=k)
        else:
            DiversityMetric.__init__(self, name="{}@{}".format(name, k), k=k)
        if isinstance(item_genre, dict):
            self.item_genre = item_genre
        else:
            raise ValueError(
                "Binomial received an invalid item_genre. item_genre "
                "must be a dictionary."
            )
        if (isinstance(alpha, (int, float))) and alpha >= 0 and alpha <= 1:
            self.alpha = alpha
        else:
            self.alpha = 0.9

    def globalFeatureProbs(self, user_history):
        """Compute the generality of each genre by counting the global proportion of items in the user preferences covering it.

        Parameters
        ----------
        user_history: Dictionary.
            {user Idx: list of item indices that the user liked in the history}.

        Returns
        -------
        A list
            The generality of each genre.

        """
        # Step 1: Efficiently count occurrences of items in user history without flattening
        counter = Counter()
        for items in user_history.values():
            counter.update(items)

        # Step 2: Precompute the length of genres for later use
        genre_len = [len(value) for key, value in self.item_genre.items()]
        n = max(genre_len)  # Max length of any genre

        # Step 3: Initialize result array
        result = np.zeros(n)

        # Step 4: Efficiently accumulate genre counts
        for key, value in self.item_genre.items():
            if key in counter:
                # Use vectorized addition instead of a loop
                result += value * counter[key]

        # Step 5: Compute global proportions
        total_items = sum(counter.values())

        if total_items > 0:
            return result / total_items
        else:
            return result

    # def globalFeatureProbs(self, user_history):
    #     """Compute the generality of each genre by counting the global proportion of items in the user preferences covering it.

    #     Parameters
    #     ----------
    #     user_history: Dictionary.
    #         {user Idx: list of item indices that the user liked in the history}.

    #     Returns
    #     -------
    #     A list
    #         the generality of each genre.

    #     """
    #     all_items = [value for key, value in user_history.items()]
    #     all_items_flatten = sum(all_items, [])
    #     counter = Counter(all_items_flatten)
    #     genre_len = [len(value) for key, value in self.item_genre.items()]
    #     n = max(list(set(genre_len)))
    #     result = np.zeros(n)
    #     for key, value in self.item_genre.items():
    #         if key in counter.keys():
    #             result += value * counter[key]
    #     return result/len(all_items_flatten)

    def userFeatureProbs(self, globalProb, user_history):
        """Compute the local probability by counting the proportion of items in a given user's user preference.

        Parameters
        ----------
        user_history: list of item indices that the user liked in the history.

        Returns
        -------
        A list
            the local probability of each genre.

        """
        userProbs = []
        if (self.alpha == 0):
            userProbs = globalProb
            return userProbs
        elif self.alpha > 0 and self.alpha <= 1:
            all_genre = [list(self.item_genre[x]) if x in self.item_genre else [
            ] for x in user_history]
            try:
                sum_genre = [sum(x) for x in zip(*all_genre)]
                numPreference = len(all_genre)
                weighted = [self.alpha * sum_genre[x]/numPreference +
                            (1-self.alpha)*globalProb[x] for x in range(len(sum_genre))]
                userProbs = weighted
                return userProbs
            except ValueError:
                return userProbs

    def binomialCoverage(self, pd_rank, prob):
        """Compute the product of the probabilities of the genres not represented in the recommendation list, according to the distribution.

        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.

        prob: list.  Probability of each genre.

        Returns
        -------
        user_coverage: A scalar
            Coverage value.

        """
        genre_len = [len(value) for key, value in self.item_genre.items()]
        n = max(list(set(genre_len)))
        user_coverage = {}
        all_genre = [list(self.item_genre[x]) if x in self.item_genre else []
                     for x in pd_rank]
        try:
            sum_genre = [sum(x) for x in zip(*all_genre)]
            coverage = 1
            for i in range(len(sum_genre)):
                if sum_genre[i] == 0:
                    coverage *= (1-prob[i])**len(pd_rank)
            user_coverage = coverage**(1/n)
            return user_coverage
        except ValueError:
            return None

    def binomialNonRedundancy(self, pd_rank, prob):
        """Compute the product of the "remaining tolerance" scores for each genre covered in the recommendation list.
        
        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.

        prob: list. Probability of each genre.

        Returns
        -------
        A scalar
            NonRedundancy value.

        """
        genre_len = [len(value) for key, value in self.item_genre.items()]
        n = max(list(set(genre_len)))
        all_genre = [list(self.item_genre[x]) if x in self.item_genre else []
                     for x in pd_rank]
        try:
            sum_genre = [sum(x) for x in zip(*all_genre)]
            nonRedundancy = 1
            for i in range(len(sum_genre)):
                if sum_genre[i] > 0:
                    k = sum_genre[i]-1
                    p = prob[i]
                    length = len(all_genre)
                    p0 = (1-p)**length
                    p1 = binom.cdf(k, length, p)
                    if (p0 < 1):
                        nonRedundancy *= 1 - (p1 - p0)/(1-p0)
            return nonRedundancy**(1/n)
        except ValueError:
            return None

    def compute(self, pd_rank, globalProb, user_history, **kwargs):
        """Compute the Binomial diversity score.

        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.
        globalProb: List.
            The generality of each genre.
        user_history: List.
            The item indices in the user preference for a given user.
        **kwargs: For compatibility

        Returns
        -------
        A scalar
            The Binomial diversity score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        try:
            userProb = self.userFeatureProbs(globalProb, user_history)
            if len(userProb) == 0:
                return None
            NonRedundacy = self.binomialNonRedundancy(
                list(truncated_pd_rank), userProb)
            Coverage = self.binomialCoverage(
                list(truncated_pd_rank), userProb)
            return NonRedundacy*Coverage
        except ValueError:
            return None


def harmonic_number(n):
    """Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)


def compute_divergence(distr_recommendation, distr_pool, alpha=0.001, divergence_type="kl"):
    """Compute KL divergence or JS divergence.
      
      Parameters
      ----------
      distr_pool: Dictionary. distribution of pool.

      distr_recommendation: Dictionary. distribution of recommendation.

      alpha: Is not really a tuning parameter, it's just there to make the
          computation more numerically stable(denominator cannot be 0). 

      divergence_type: (Optional) A string determining the employed method of divergence. 
          Options include "JS" or "KL", with the default value being "KL".

    """
    try:
        assert 0.99 <= sum(distr_pool.values()) <= 1.01
        assert 0.99 <= sum(distr_recommendation.values()) <= 1.01
    except AssertionError:
        return None
    kl_div = 0.
    pool_scores = []
    recom_scores = []
    merged_dic = opt_merge_max_mappings(distr_pool, distr_recommendation)
    for key in sorted(merged_dic.keys()):
        recom_score = distr_recommendation.get(key, 0.)
        pool_score = distr_pool.get(key, 0.)
        pool_scores.append((1 - alpha) * pool_score + alpha * recom_score)
        recom_scores.append((1 - alpha) * recom_score + alpha * pool_score)
    if divergence_type.lower() == "js":
        divergence = JSD(pool_scores, recom_scores)
    elif divergence_type.lower() == "kl":
        divergence = entropy(pool_scores, recom_scores, base=2)
    # return divergence
    return float(format(divergence, '.6f'))


def opt_merge_max_mappings(distr_pool, distr_recommendation):
    """ Merges two dictionaries based on the largest value in a given mapping.
    
    Parameters
    ----------
    distr_pool: Dict[Any, Comparable]
    distr_recommendation: Dict[Any, Comparable]

    Returns
    -------
    Dict[Any, Comparable]
        The merged dictionary

    """
    merged, other = (distr_pool, distr_recommendation) if len(distr_pool) > len(
        distr_recommendation) else (distr_recommendation, distr_pool)
    merged = dict(merged)
    for key in other:
        if key not in merged or other[key] > merged[key]:
            merged[key] = other[key]
    return merged


def JSD(P, Q):
    """ Compute J-S divergence.

    Parameters
    ----------
    P: Dictionary for distribution of pool.
    Q: Dictionary for distribution of recommendation.
    
    Returns
    -------
    JS divergence value.

    """
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    try:
        jsd_root = math.sqrt(
            abs(0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2))))
    except ZeroDivisionError:
        print(P)
        print(Q)
        jsd_root = None
    return jsd_root


def compute_distr_continous(arr, bins_discretizer, discount=False):
    """ Compute distribution for continous data.

    Parameters
    ----------
    arr: numpy array. Pool_affect, or recommendation_affect
        which are the continous data for pool or recommendation.
    discount: boolean. false if no discount, true if discounted

    Returns
    -------
    Dictionary. distribution of pool or recommendation. 

    """
    n = len(arr)
    sum_one_over_ranks = harmonic_number(n)
    arr_binned = bins_discretizer.transform(arr)
    distr = {}
    if discount:
        for bin in list(range(bins_discretizer.n_bins)):
            for indx, ele in enumerate(arr_binned[:, 0]):
                if ele == bin:
                    rank = indx + 1
                    bin_freq = distr.get(bin, 0.)
                    distr[bin] = round(bin_freq + 1 * 1 /
                                       rank / sum_one_over_ranks, 4)
    else:
        for bin in list(range(bins_discretizer.n_bins)):
            distr[bin] = round(np.count_nonzero(
                arr_binned == bin) / arr_binned.shape[0], 4)
    return distr


def compute_distr_category(arr, discount=False):
    """ Compute distribution for categorical data. 
    Parameters
    ----------
    arr: numpy array. Pool_affect, or recommendation_affect
        which are the categorical data for pool or recommendation.

    discount: boolean. false if no discount, true if discounted
    
    Returns
    -------
    Dictionary. distribution of pool or recommendation. 

    """
    n = len(arr)
    sum_one_over_ranks = harmonic_number(n)
    count = 0
    distr = {}
    for indx, item in enumerate(arr):
        rank = indx + 1
        story_freq = distr.get(item, 0.)
        distr[item] = round(story_freq + 1 * 1 / rank /
                            sum_one_over_ranks if discount else story_freq + 1 * 1 / n, 4)
        count += 1
    return distr


def compute_distr_listData(arr, discount=False):
    """Compute the entity distribution for a given list of Items.
    
    Parameters
    ----------
    arr: numpy array of list of entities. 
    discount: boolean. false if no discount, true if discounted
    
    Returns
    -------
    Dictionary. distribution of pool or recommendation. 

    """
    n = len(arr)
    sum_one_over_ranks = harmonic_number(n)
    rank = 0
    distr = {}
    for idx, value in arr.items():
        total = len(value)
        rank += 1
        d = Counter(value)
        for party, mentions in d.items():
            party_freq = distr.get(party, 0.)
            distr[party] = party_freq + mentions / total * 1 / rank / \
                sum_one_over_ranks if discount else party_freq + mentions / total

    if sum(distr.values()) > 0:
        factor = 1.0 / sum(distr.values())
        for key, value in distr.items():
            distr[key] = round(value * factor, 4)
    return distr


def compute_distr_majority_minority(arr,  discount=False):
    """ Compute distribution for alternative voices metric. 
    
    Parameters
    ----------
    arr: numpy array of list [number_of_minority_scores, number_of_majority_scores].
    discount: boolean. false if no discount, true if discounted
    
    Returns
    -------
    Dictionary. distribution of pool or recommendation. 

    """
    n = len(arr)
    sum_one_over_ranks = harmonic_number(n)
    count = 0
    distr = {0: 0, 1: 0}
    majority = 0
    minority = 0
    for idx, value in arr.items():
        rank = count + 1
        article_majority = value[1]
        article_minority = value[0]
        if article_minority > 0 or article_majority > 0:
            if discount:
                prob_majority = article_majority / \
                    (article_majority+article_minority) * \
                    1/rank/sum_one_over_ranks
                prob_minority = article_minority / \
                    (article_majority+article_minority) * \
                    1/rank/sum_one_over_ranks
            else:
                prob_majority = article_majority / \
                    (article_majority+article_minority)
                prob_minority = article_minority / \
                    (article_majority+article_minority)
            majority += prob_majority
            minority += prob_minority
        count += 1
    r = minority + majority
    if r > 0:
        distr[0] = round(minority / r, 4)
        distr[1] = round(majority / r, 4)
    return distr


class Activation(DiversityMetric):
    """Activation metric.

    Parameters
    ----------
    item_sentiment: Dictionary. {item idx: item sentiments}. 
        Contains item indices mapped to their respective sentiments.
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.
    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.
    divergence_type: (Optional) string that determines the divergence method employed. 
        The choices are "JS" or "KL". The default value is "KL".    

    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). 
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations.
    In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_sentiment, k=-1, divergence_type="KL", discount=False, n_bins=5):
        DiversityMetric.__init__(self, name="Activation@{}".format(k), k=k)
        if isinstance(item_sentiment, dict):
            self.item_sentiment = item_sentiment
        else:
            raise ValueError(
                "Activation received an invalid item_sentiment. item_sentiment "
                "must be a dictionary."
            )
        if isinstance(divergence_type, str) and divergence_type.lower() == "kl" or divergence_type.lower() == "js":
            self.divergence_type = divergence_type
        else:
            raise ValueError(
                "Activation received an invalid divergence_type. divergence_type "
                "must be either kl or js, and must be a string."
            )
        if isinstance(discount, bool):
            self.discount = discount
        else:
            raise ValueError(
                "Activation received an invalid discount. discount "
                "must be either True or False, and must be a boolean."
            )
        if isinstance(n_bins, int) and n_bins > 1:
            self.n_bins = n_bins
        else:
            raise ValueError(
                "Activation received an invalid number "
                "of bins. Number of bins "
                "must be at least 2, and must be an int."
            )

    def compute(self, pd_rank, pool, **kwargs):
        """Compute Activation metric score.

        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.
        pool: Numpy array
            Overall pool of items.
        **kwargs: For compatibility

        Returns
        -------
        act: A scalar
            Activation metric score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        rec = [self.item_sentiment[i] if i in self.item_sentiment.keys()
               else None for i in truncated_pd_rank]
        recommendation = np.array([x for x in rec if x is not None])
        recommendation = recommendation[~np.isnan(recommendation)]
        po = [self.item_sentiment[i] if i in self.item_sentiment.keys()
              else None for i in pool]
        pool_sen = np.array([x for x in po if x is not None])
        
        if np.isnan(recommendation).any() or np.isinf(recommendation).any():
            # print("recommendation contains NaN or infinite values.")
            recommendation = recommendation[~np.isnan(recommendation) & ~np.isinf(recommendation)]
        if np.isnan(pool_sen).any() or np.isinf(pool_sen).any():
            #  Filter out NaN and infinite values
            pool_sen = pool_sen[~np.isnan(pool_sen) & ~np.isinf(pool_sen)]

        if len(recommendation) == 0 or len(pool_sen) == 0:
            return None
        try:
            recommend_sentiment = np.abs(recommendation).reshape(-1, 1)
            pool_sentiment = np.abs(pool_sen).reshape(-1, 1)
            bins_discretizer = KBinsDiscretizer(
                encode='ordinal', n_bins=self.n_bins, strategy='uniform', subsample=None)
            bins_discretizer.fit(pool_sentiment)
            distr_pool = compute_distr_continous(
                pool_sentiment, bins_discretizer, False)
            distr_recommendation = compute_distr_continous(
                recommend_sentiment, bins_discretizer, self.discount)
     
            act = compute_divergence(
                distr_recommendation, distr_pool, divergence_type=self.divergence_type)

            return act
        except ValueError as e:
            print("ValueError encountered:", e)
            return None


class Calibration(DiversityMetric):
    """Calibration metric.

    Parameters
    ----------
    item_feature: Dictionary. {item idx: item feature vectors}. 
         A dictionary that maps item indices to their respective categories or complexities.
        Categories are discrete values, where each item has a single categorical value selected 
        from options such as {"action", "comedy", ...}. For example, {1: "action", 2:"comedy",... }. 
        Complexities are continuous values. For example, {1: 17, 2 : 22,... }.
    data_type: String.
        indicating the type of data, either "category" or "complexity".
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.
    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.
    divergence_type: (Optional) string that determines the divergence method employed. 
        The choices are "JS" or "KL". The default value is "KL".  
    n_bins: (Optional) Int.
        Determines the number of bins used for discretizing continuous data into intervals. 
        The default value is 5.  

    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). 
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations.
    In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_feature, data_type, k=-1, divergence_type="KL", discount=False, n_bins=5):
        if data_type.lower() == "complexity":
            DiversityMetric.__init__(
                self, name="Calibration_complexity@{}".format(k), k=k)
        else:
            DiversityMetric.__init__(
                self, name="Calibration_category@{}".format(k), k=k)
        if isinstance(item_feature, dict):
            self.item_feature = item_feature
        else:
            raise ValueError(
                "Calibration received an invalid item_feature. item_feature "
                "must be a dictionary."
            )
        if isinstance(data_type, str) and (data_type.lower() == "complexity" or data_type.lower() == "category"):
            self.data_type = data_type
        else:
            raise ValueError(
                "Calibration received an invalid data_type. data_type "
                "must be either category or complexity, and must be a string."
            )
        if isinstance(divergence_type, str) and divergence_type.lower() == "kl" or divergence_type.lower() == "js":
            self.divergence_type = divergence_type
        else:
            raise ValueError(
                "Calibration received an invalid divergence_type. divergence_type "
                "must be either kl or js, and must be a string."
            )
        if isinstance(discount, bool):
            self.discount = discount
        else:
            raise ValueError(
                "Calibration received an invalid discount. discount "
                "must be either True or False, and must be a boolean."
            )
        if isinstance(n_bins, int) and n_bins > 1:
            self.n_bins = n_bins
        else:
            raise ValueError(
                "Calibration received an invalid number "
                "of bins. Number of bins "
                "must be at least 2, and must be an int."
            )

    def compute(self, pd_rank, user_history, **kwargs):
        """Compute Calibration metric score.

        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.
        user_history: Numpy array
            Positive items in the user history.
        **kwargs: For compatibility

        Returns
        -------
        act: A scalar
            Calibration metric score.
        
        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        user_interacted_item = user_history
        rec = [self.item_feature[i] if i in self.item_feature.keys()
               else None for i in truncated_pd_rank]
        recommendation = np.array([x for x in rec if x is not None])
        his = [self.item_feature[i] if i in self.item_feature.keys(
        ) else None for i in user_interacted_item]
        history = np.array([x for x in his if x is not None])
        if len(recommendation) == 0 or len(history) == 0:
            return None
        if self.data_type == "category":
            try:
                freq_rec = compute_distr_category(
                    recommendation, discount=self.discount)
                freq_history = compute_distr_category(
                    history, discount=self.discount)

                cal = compute_divergence(
                    freq_rec, freq_history,  divergence_type=self.divergence_type)
                return cal
            except ValueError:
                return None
        elif self.data_type == "complexity":
            try:
                recommendation_complexity = recommendation.reshape(-1, 1)
                reading_history_complexity = history.reshape(-1, 1)
                if reading_history_complexity.shape[0] > 1:
                    bins_discretizer = KBinsDiscretizer(
                        encode='ordinal', n_bins=self.n_bins, strategy='uniform', subsample=None)
                    bins_discretizer.fit(reading_history_complexity)
                    distr_reading_history = compute_distr_continous(
                        reading_history_complexity, bins_discretizer, self.discount)
                    distr_recommendation = compute_distr_continous(
                        recommendation_complexity, bins_discretizer, self.discount)
                
                    cal = compute_divergence(
                        distr_recommendation, distr_reading_history, divergence_type=self.divergence_type)

                    return cal
                else:
                    return None
            except ValueError:
                return None


class Fragmentation(DiversityMetric):
    """Fragmentation metric.

    Parameters
    ----------
    item_story: Dictionary. {item idx: item feature vectors}. 
        A dictionary that maps item indices to  to their respective story chains. 
        The stories are categorical values.
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.
    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.
    divergence_type: (Optional) string that determines the divergence method employed. 
        The choices are "JS" or "KL". The default value is "KL".  
    n_samples: (Optional) Int.
        The number of users to compare (minimum 1). 
        The default value is 1.  

    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). 
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations.
    In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """
    def __init__(self, item_story, k=-1, divergence_type="KL", discount=False, n_samples=1):
        DiversityMetric.__init__(self, name="Fragmentation@{}".format(k), k=k)
        if isinstance(item_story, dict):
            self.item_story = item_story
        else:
            raise ValueError(
                "Fragmentation received an invalid item_story. item_story "
                "must be a dictionary."
            )
        if isinstance(divergence_type, str) and divergence_type.lower() == "kl" or divergence_type.lower() == "js":
            self.divergence_type = divergence_type
        else:
            raise ValueError(
                "Fragmentation received an invalid divergence_type. divergence_type "
                "must be either kl or js, and must be a string."
            )
        if isinstance(discount, bool):
            self.discount = discount
        else:
            raise ValueError(
                "Fragmentation received an invalid discount. discount "
                "must be either True or False, and must be a boolean."
            )
        if isinstance(n_samples, int) and n_samples >= 1:
            self.n_samples = n_samples
        else:
            raise ValueError(
                "Fragmentation received an invalid number "
                "of samples. Number of samples "
                "must be at least 1, and must be an int."
            )

    def compute(self, pd_rank, pd_other_users, **kwargs):
        """Compute Fragmentation metric score.

        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.
        pd_other_users: List.
            The recommendation items received by other n_samples users.
        **kwargs: For compatibility

        Returns
        -------
        act: A scalar
            Fragmentation metric score.

        """
        truncated_pd_other_users = []
        if self.k > 0:
            for pd_other in pd_other_users:
                truncated_pd_other_users.append(pd_other[:self.k])
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
            truncated_pd_other_users = pd_other_users
        rec = [self.item_story[i] if i in self.item_story.keys(
        ) else None for i in truncated_pd_rank]
        recommendation = np.array([x for x in rec if x is not None])
        if len(recommendation) == 0:
            return None
        divergence = []
        try:
            for other_rec in truncated_pd_other_users:
                story_other_rec = [
                    self.item_story[i] if i in self.item_story.keys() else None for i in other_rec]
                story_other_rec = np.array(
                    [x for x in story_other_rec if x is not None])
                freq_x = compute_distr_category(
                    recommendation, discount=self.discount)
                freq_y = compute_distr_category(
                    story_other_rec, discount=self.discount)

                div_recom_sample = compute_divergence(
                    freq_x, freq_y, alpha=0.001, divergence_type=self.divergence_type)
                if div_recom_sample is not None:
                    divergence.append(div_recom_sample)

            if len(divergence) > 0:
                single_frag = np.mean(divergence)
            else:
                single_frag = None
            return single_frag
        except ValueError:
            return None


class Representation(DiversityMetric):
    """Representation metric.

    Parameters
    ----------
    item_entities: Dictionary. {item idx: [item entities]}. 
        A dictionary that maps item indices to  to their respective list of entities. 

    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.

    divergence_type: (Optional) string that determines the divergence method employed. 
        The choices are "JS" or "KL". The default value is "KL".  

    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). 
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations.
    In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_entities, k=-1, divergence_type="KL", discount=False, n_samples=5):
        DiversityMetric.__init__(self, name="Representation@{}".format(k), k=k)
        if isinstance(item_entities, dict):
            self.item_entities = item_entities
        else:
            raise ValueError(
                "Representation received an invalid item_entities. item_entities "
                "must be a dictionary."
            )
        if isinstance(divergence_type, str) and divergence_type.lower() == "kl" or divergence_type.lower() == "js":
            self.divergence_type = divergence_type
        else:
            raise ValueError(
                "Representation received an invalid divergence_type. divergence_type "
                "must be either kl or js, and must be a string."
            )
        if isinstance(discount, bool):
            self.discount = discount
        else:
            raise ValueError(
                "Representation received an invalid discount. discount "
                "must be either True or False, and must be a boolean."
            )

    def compute(self, pd_rank, pool, **kwargs):
        """Compute Representation metric score.

        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.
        pool: Numpy array
            Overall pool of items.
        **kwargs: For compatibility

        Returns
        -------
        act: A scalar
            Representation metric score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        recommendation_entities = {}
        truncated_pd_rank = np.unique(truncated_pd_rank)
        pool = np.unique(pool)
        for i in truncated_pd_rank:
            if i in self.item_entities.keys():
                recommendation_entities[i] = self.item_entities[i]
        pool_entities = {}
        for i in pool:
            if i in self.item_entities.keys():
                pool_entities[i] = self.item_entities[i]
        if len(recommendation_entities) == 0 or len(pool_entities) == 0:
            return None
        try:
            distr_rec = compute_distr_listData(
                recommendation_entities, discount=self.discount)
            distr_pool = compute_distr_listData(pool_entities, discount=False)

            rep = compute_divergence(
                distr_rec, distr_pool, divergence_type=self.divergence_type)
            return rep
        # else:
        except ValueError:
            return None


class AlternativeVoices(DiversityMetric):
    """Alternative Voices metric.

    Parameters
    ----------
    item_minor_major: Dictionary. {item idx: [number of minority mentions, number of majority mentions]}. 
        A dictionary that maps item indices to  to their respective number of minority mentions and majority mentions. 

    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.

    divergence_type: (Optional) string that determines the divergence method employed. 
        The choices are "JS" or "KL". The default value is "KL".  

    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). 
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations.
    In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_minor_major, data_type='mainstream',  k=-1, divergence_type="KL", discount=False):
        if data_type.lower() == "gender":
            DiversityMetric.__init__(
                self, name="AltVoices_gender@{}".format(k), k=k)
        elif data_type.lower() == "ethnicity":
            DiversityMetric.__init__(
                self, name="AltVoices_ethnicity@{}".format(k), k=k)
        else:
            DiversityMetric.__init__(
                self, name="AltVoices_mainstream@{}".format(k), k=k)
        if isinstance(item_minor_major, dict):
            self.item_minor_major = item_minor_major
        else:
            raise ValueError(
                "AlternativeVoices received an invalid item_minor_major. item_minor_major "
                "must be a dictionary."
            )
        if isinstance(divergence_type, str) and divergence_type.lower() == "kl" or divergence_type.lower() == "js":
            self.divergence_type = divergence_type
        else:
            raise ValueError(
                "AlternativeVoices received an invalid divergence_type. divergence_type "
                "must be either kl or js, and must be a string."
            )
        if isinstance(discount, bool):
            self.discount = discount
        else:
            raise ValueError(
                "AlternativeVoices received an invalid discount. discount "
                "must be either True or False, and must be a boolean."
            )

    def compute(self, pd_rank, pool, **kwargs):
        """Compute Alternative Voices metric score.

        Parameters
        ----------
        pd_rank: Numpy array
            Item ranking prediction.
        pool: Numpy array
            Overall pool of items.
        **kwargs: For compatibility

        Returns
        -------
        act: A scalar
            Alternative Voices metric score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[:self.k]
        else:
            truncated_pd_rank = pd_rank
        recommendation_minor_major = {}
        truncated_pd_rank = np.unique(truncated_pd_rank)
        pool = np.unique(pool)
        for i in truncated_pd_rank:
            if i in self.item_minor_major.keys():
                recommendation_minor_major[i] = self.item_minor_major[i]
        pool_minor_major = {}
        for i in pool:
            if i in self.item_minor_major.keys():
                pool_minor_major[i] = self.item_minor_major[i]
        if len(pool_minor_major) == 0 or len(recommendation_minor_major) == 0:
            return None
        try:
            pool_distr = compute_distr_majority_minority(
                pool_minor_major, False)
            recommendation_distr = compute_distr_majority_minority(
                recommendation_minor_major,  discount=self.discount)
            divergence = None

            if not (recommendation_distr[0] == 0 and recommendation_distr[1] == 0):
                divergence = compute_divergence(
                    recommendation_distr, pool_distr,  divergence_type=self.divergence_type)
            return divergence
        except ValueError:
            return None
