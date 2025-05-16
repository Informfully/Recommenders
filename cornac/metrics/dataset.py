'''
This file is mainly based on the cornac.metrics.diversity.py
we make some changes to these functions in order to measure the diversity of datasets
'''


import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import cdist
import math
from sklearn.preprocessing import KBinsDiscretizer
from cornac.metrics.diversity import compute_divergence
from cornac.metrics.diversity import compute_distr_continous
from cornac.metrics.diversity import compute_distr_listData
from cornac.metrics.diversity import compute_distr_majority_minority
from cornac.metrics.diversity import compute_distr_category


class DatasetMetric:
    """Diversity Metric for the entire dataset.

    Attributes
    ----------
    type: string, value: 'dataset'
        Type of the metric, e.g., "ranking", "rating","diversity".

    name: string, default: None
        Name of the measure.
    """

    def __init__(self, name=None, higher_better=False):

        self.type = "dataset"
        self.name = name
        self.higher_better = higher_better

    def compute_dataset_itself(self, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def calculate_spacelog(num_users, num_items, sc=1000):
        """
        Calculate Spacelog.

        Parameters
        ----------
        num_users : int
            Number of users.

        num_items : int
            Number of items.

        sc : int
            Scaling factor to constrain the possible values of |num_users| x |num_items|. Set 1000 as default.

        Returns
        -------
        spacelog : float
            Spacelog value.

        Notes
        -----
        Reflects the size (and even the scale) of the user-item interaction space
        """
        # Check if num_users, num_items, and sc are positive integers
        if not isinstance(num_users, int) or not isinstance(num_items, int) or not isinstance(sc, int):
            raise ValueError("num_users, num_items, and sc must be integers")
        if num_users <= 0 or num_items <= 0 or sc <= 0:
            raise ValueError("num_users, num_items, and sc must be positive")

        spacelog = math.log10(num_users * num_items / sc)
        return spacelog

    @staticmethod
    def calculate_shapelog(num_users, num_items):
        """
        Calculate the Shapelog value.

        Parameters
        ----------
        num_users : int
            Number of users in the dataset.
        num_items : int
            Number of items in the dataset.

        Returns
        -------
        float
            The Shapelog value.

        Notes
        -----
            Measures the ratio of users to items
        """
        # Check if num_users, num_items are positive integers
        if not isinstance(num_users, int) or not isinstance(num_items, int):
            raise ValueError("num_users, num_items must be integers")
        if num_users <= 0 or num_items <= 0:
            raise ValueError("num_users, num_items must be positive")

        else:
            return math.log10(num_users / num_items)

    @staticmethod
    def calculate_densitylog(num_users, num_items, num_interactions):
        """
        Calculate the Densitylog value.

        Parameters
        ----------
        num_users : int
            Number of users in the dataset.
        num_items : int
            Number of items in the dataset.
        num_interactions : int
            Number of interactions in the dataset.

        Returns
        -------
        float
            The Densitylog value.

        Notes
        -----
            Represents the proportion of observed interactions among all the possible user-item interactions.
        """
        # Check if num_users, num_items, and num_interactions are positive integers
        if not isinstance(num_users, int) or not isinstance(num_items, int) or not isinstance(num_interactions, int):
            raise ValueError("num_users, num_items, and num_interactions must be integers")
        if num_users <= 0 or num_items <= 0 or num_interactions <= 0:
            raise ValueError("num_users, num_items, and num_interactions must be positive")
        else:
            return math.log10(num_interactions / (num_users * num_items))

    @staticmethod
    def compute_gini_user(num_interactions, num_user, user_interaction):
        """
        Compute Gini user metric.

        Parameters
         ----------
        num_interactions : int
            Total number of interactions in the dataset.
        num_user : int
            Total number of users.
        user_interaction : list
            List containing the number of interactions belonging to each user.

         Returns
        -------
        float
            Gini user metric value.

        Notes
        -----
        Gini user metric measures the inequality in the distribution of interactions among users.
        """

        # Check if num_rating and num_user are positive integers
        if not isinstance(num_interactions, int) or not isinstance(num_user, int) or num_interactions <= 0 or num_user <= 0:
            raise ValueError("num_rating and num_user must be positive integers")

        # Check if user_interaction is a non-empty list
        if not isinstance(user_interaction, list) or len(user_interaction) == 0:
            raise ValueError("user_interaction must be a non-empty list")

        # Check if user_interaction is a list of positive integers
        if not all(isinstance(count, int) and count > 0 for count in user_interaction):
            raise ValueError("user_interaction must be a list of positive integers")

        user_interaction = sorted(user_interaction)
        gini = 0
        for u in range(1, num_user + 1):
            gini += ((num_user + 1 - u) / (num_user + 1)) * (user_interaction[u - 1] / num_interactions) #0.721140332253045
        gini = 1 - 2 * gini
        return gini

    @staticmethod
    def compute_gini_item(num_interactions, num_item, item_interaction):
        """
        Compute Gini item metric.

        Parameters
        ----------
        num_interactions : int
            Total number of interactions in the dataset.
        num_item : int
            Total number of items.
        item_interaction : list
            List containing the number of interactions belonging to each item.

        Returns
        -------
        float
            Gini item metric value.

        Notes
        -----
        Gini item metric measures the inequality in the distribution of interactions among items.
        """
        # Check if num_rating and num_item are positive integers
        if not isinstance(num_interactions, int) or not isinstance(num_item, int) or num_interactions <= 0 or num_item <= 0:
            raise ValueError("num_rating and num_item must be positive integers")

        # Check if user_interaction is a non-empty list
        if not isinstance(item_interaction, list) or len(item_interaction) == 0:
            raise ValueError("item_interaction must be a non-empty list")

        # Check if user_interaction is a list of positive integers
        if not all(isinstance(count, int) and count > 0 for count in item_interaction):
            raise ValueError("item_interaction must be a list of positive integers")

        item_interaction = sorted(item_interaction)
        gini = 0
        for u in range(1, num_item + 1):
            gini += ((num_item + 1 - u) / (num_item + 1)) * (item_interaction[u - 1] / num_interactions) # 0.9642933991516575
        gini = 1 - 2 * gini
        return gini


class DatasetActivation(DatasetMetric):
    """
    Activation metric.

    Parameters
    ----------
    item_sentiment: Dictionary. {item idx: item sentiments}.
        Contains item indices mapped to their respective sentiments.

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
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_sentiment, divergence_type="KL", discount=False, n_bins=5):
        DatasetMetric.__init__(self, name="Activation")
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

    def compute_dataset_itself(self, reference_distribution=None, **kwargs):
        """
        Compute Activation metric score for the entire dataset.

        Parameters
        ----------
        reference_distribution: Numpy array or None, optional
            Reference distribution for comparison. If None, a uniform distribution is used.
        **kwargs: For compatibility

        Returns
        -------
        act: A scalar
            Activation metric score.
        """
        # Get sentiment values for all items in self.item_sentiment
        pool = [self.item_sentiment[i] if i in self.item_sentiment.keys()
                else None for i in self.item_sentiment.keys()]
        pool = [value for value in pool if value is not None and not math.isnan(value)]
        if len(pool) == 0:
            return None
        try:
            # Compute sentiment distribution for the entire dataset
            pool_sentiment = np.abs(pool).reshape(-1, 1)
            bins_discretizer = KBinsDiscretizer(
                encode='ordinal', n_bins=self.n_bins, strategy='uniform', subsample=None)
            bins_discretizer.fit(pool_sentiment)
            distr_pool = compute_distr_continous(
                pool_sentiment, bins_discretizer, False)

            # Use the provided reference distribution or create a uniform distribution if None
            # Convert reference_distribution to a dictionary if it's in array format
            if reference_distribution is None:
                reference_distribution = {idx: 1 / len(distr_pool) for idx in
                                          range(len(distr_pool))}  # Uniform distribution
            else:
                reference_distribution = {idx: prob for idx, prob in enumerate(reference_distribution)}

            act = compute_divergence(
                reference_distribution, distr_pool, divergence_type=self.divergence_type)
            return act
        except ValueError:
            return None


class DatasetGiniCoeff(DatasetMetric):
    """
    Gini coefficient.
    item_genre: dictionary.
        A dictionary that maps item indices to the item genres array(numpy array).
        The item genres array must be an ordered list,
        where a value of 0 indicates that the item does not belong to a certain category at the corresponding position
        in the genre list, and 1 indicates that the item belong to a certain category.

    References
    ----------
    - Ricci, F., Rokach, L. and Shapira, B., 2015. Recommender systems: introduction and challenges. Recommender systems handbook, pp.1-34.
    """

    def __init__(self, item_genre):
        DatasetMetric.__init__(self, name="GiniCoeff")
        if isinstance(item_genre, dict):
            self.item_genre = item_genre
        else:
            raise ValueError(
                "GiniCoeff received an invalid item_genre. item_genre "
                "must be a dictionary."
            )

    def compute_dataset_itself(self, **kwargs):
        """
        Compute Gini coefficient metric for the entire dataset.

        Parameters
        ----------

        **kwargs: For compatibility

        Returns
        -------
        A scalar
            Gini coefficient score.
        """
        pool_items_genre = []
        for item_index, genre_vector in self.item_genre.items():
            pool_items_genre.append(genre_vector.tolist())

        proportion = []
        if len(pool_items_genre) > 0:
            for i in range(len(pool_items_genre[0])):
                column = [row[i] for row in pool_items_genre]
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


class DatasetRepresentation(DatasetMetric):
    """
    Representation metric.

    Parameters
    ----------
    item_entities: Dictionary. {item idx: [item entities]}.
        A dictionary that maps item indices to their respective list of entities.

    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.

    divergence_type: (Optional) string that determines the divergence method employed.
        The choices are "JS" or "KL". The default value is "KL".

    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September).
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_entities, divergence_type="KL", discount=False):
        DatasetMetric.__init__(self, name="Representation")
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

    def compute_dataset_itself(self, reference_distribution=None, **kwargs):
        """
        Compute Representation metric score for the entire dataset.

        Parameters
        ----------

        reference_distribution: Numpy array or None, optional
            Reference distribution for comparison. If None, a uniform distribution is used.
        **kwargs: For compatibility

        Returns
        -------
        act: A scalar
            Representation metric score.
        """

        if not self.item_entities:
            return None

        try:
            # Compute sentiment distribution for the entire dataset
            distr_pool = compute_distr_listData(self.item_entities, discount=False)

            # Use the provided reference distribution or create a uniform distribution if None
            if reference_distribution is None:
                num_categories = len(distr_pool)
                uniform_prob = 1 / num_categories
                reference_distribution = {key: uniform_prob for key in distr_pool.keys()}
            else:
                reference_distribution = {key: prob for key, prob in zip(distr_pool.keys(), reference_distribution)}

            # Compute the Representation metric score using compute_divergence function
            rep = compute_divergence(
                reference_distribution, distr_pool, divergence_type=self.divergence_type)

            return rep
        except ValueError:
            return None


class DatasetAlternativeVoices(DatasetMetric):
    """Alternative Voices metric.

    Parameters
    ----------
    item_minor_major: Dictionary. {item idx: [number of minority mentions, number of majority mentions]}.
        A dictionary that maps item indices to their respective number of minority mentions and majority mentions.


    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.

    divergence_type: (Optional) string that determines the divergence method employed.
        The choices are "JS" or "KL". The default value is "KL".


    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September).
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_minor_major, data_type='mainstream', divergence_type="KL", discount=False):
        if data_type.lower() == "gender":
            DatasetMetric.__init__(
                self, name="AltVoices_gender")
        elif data_type.lower() == "ethnicity":
            DatasetMetric.__init__(
                self, name="AltVoices_ethnicity")
        else:
            DatasetMetric.__init__(
                self, name="AltVoices_mainstream")
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

    def compute_dataset_itself(self, reference_distribution=None, **kwargs):
        """
        Compute Alternative Voices metric score for the entire dataset.

        Parameters
        ----------

        reference_distribution: Numpy array or None, optional
            Reference distribution for comparison. If None, a uniform distribution is used.
        **kwargs: For compatibility

        Returns
        -------
        act: A scalar
            Alternative Voices metric score.
        """

        pool_minor_major = {}

        # Iterate over each item in the pool
        for i in self.item_minor_major:
            # Check if the item has minority-majority information
            if i in self.item_minor_major.keys():
                pool_minor_major[i] = self.item_minor_major[i]

        # If no items in the pool have minority-majority information, return None
        if len(pool_minor_major) == 0:
            return None

        try:
            # Compute the distribution of majority and minority items in the pool
            pool_distr = compute_distr_majority_minority(pool_minor_major, False)

            # Use the provided reference distribution or create a uniform distribution if None
            if reference_distribution is None:
                reference_distribution = {idx: 1 / len(pool_distr) for idx in
                                          range(len(pool_distr))}  # Uniform distribution
            else:
                reference_distribution = {idx: prob for idx, prob in enumerate(reference_distribution)}

            # Compute the divergence score using the reference distribution
            divergence = compute_divergence(reference_distribution, pool_distr, divergence_type=self.divergence_type)
            return divergence
        except ValueError:
            return None


class DatasetILD(DatasetMetric):
    """Intra-list diversity.

    Parameters
    ----------
    item_feature: A dictionary that maps item indices to their feature vectors.
        The feature vector must be provided using numpy array.
    distance_type: (Optional) String for configuring distance measure.
        ‘correlation’, ‘cosine’,‘euclidean’ or ‘jaccard’.
        By default, use cosine distance metric.

    References
    ----------
    https://www.academia.edu/2655896/Improving_recommendation_diversity
    """

    def __init__(self, item_feature, distance_type="cosine"):
        DatasetMetric.__init__(self, name="ILD")
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

    def compute_dataset_itself(self, **kwargs):
        """Compute Intra-list diversity score for the entire dataset or a given ranking.

        Parameters
        ----------

        **kwargs: For compatibility.

        Returns
        -------
        ild_dataset: A scalar
            Intra-list diversity score for the entire dataset or a given ranking.
        """

        all_items = list(self.item_feature.keys())
        all_items_array = np.array(all_items)

        pd_vec0 = [self.item_feature[i].tolist() for i in all_items_array if i in self.item_feature]
        pd_vec = [x for x in pd_vec0 if x and any(x)]
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


class DatasetCalibration(DatasetMetric):
    """Calibration metric.

    Parameters
    ----------
    item_feature: Dictionary. {item idx: item feature vectors}.
         A dictionary that maps item indices to their respective categories or complexities.
        Categories are discrete values, where each item has a single categorical value selected from options such as {"action", "comedy", ...}. For example, {1: "action", 2:"comedy",... }.
        Complexities are continuous values. For example, {1: 17, 2 : 22,... }.
    data_type: String.
        indicating the type of data, either "category" or "complexity".
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
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_feature, data_type, divergence_type="KL", discount=False, n_bins=5):
        if data_type.lower() == "complexity":
            DatasetMetric.__init__(
                self, name="Calibration_complexity")
        else:
            DatasetMetric.__init__(
                self, name="Calibration_category")
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
                "must be at least 2, and must be an int.")

    def compute_dataset_itself(self, reference_distribution=None, **kwargs):

        """
        Compute Calibration metric score for the dataset itself.

        Parameters
        ----------
        reference_distribution: Numpy array or None, optional
            Reference distribution for comparison. If None, a uniform distribution is used.

        **kwargs: For compatibility

        Returns
        -------
        cal: A scalar
            Calibration metric score for the dataset.
        """

        # Check if the self.item_feature is empty
        if not self.item_feature:
            return None

        if self.data_type == "category":
            try:
                # Compute frequency distribution of categories in the dataset
                freq_dataset = compute_distr_category(self.item_feature.values(), discount=self.discount)
                freq_dataset = {k: v for k, v in freq_dataset.items() if k is not None and not (isinstance(k, float) and math.isnan(k))}

                # Use the provided reference distribution or assume uniform distribution
                if reference_distribution is None:
                    reference_distribution = {category: 1 / len(freq_dataset) for category in freq_dataset}
                else:
                    reference_distribution = {key: prob for key, prob in zip(freq_dataset.keys(), reference_distribution)}
                # Compute divergence between the frequency distribution of the dataset and the reference distribution
                cal = compute_divergence(freq_dataset, reference_distribution, divergence_type=self.divergence_type)

                return cal
            except ValueError:
                return None
        elif self.data_type == "complexity":
            try:
                dataset_complexity = np.array(
                    [self.item_feature[item] for item in self.item_feature])
                reshape_dataset_complexity = dataset_complexity.reshape(-1, 1)
                # Discretize complexity values into bins
                bins_discretizer = KBinsDiscretizer(encode='ordinal', n_bins=self.n_bins, strategy='uniform',
                                                    subsample=None)
                bins_discretizer.fit(reshape_dataset_complexity)
                # Compute distribution of complexity levels in the dataset
                distr_dataset = compute_distr_continous(reshape_dataset_complexity, bins_discretizer, self.discount)
                # Use the provided reference distribution or assume uniform distribution
                if reference_distribution is None:
                    reference_distribution = {level: 1 / len(distr_dataset) for level in distr_dataset}
                else:
                    reference_distribution = {idx: prob for idx, prob in enumerate(reference_distribution)}
                # Compute divergence between the distribution of complexity levels in the dataset and the reference distribution
                cal = compute_divergence(distr_dataset, reference_distribution, divergence_type=self.divergence_type)

                return cal
            except ValueError:
                return None


def get_number_of_interactions(df):
    """
    Returns the total number of ratings in the DataFrame.

    Parameters:
    df (DataFrame): User-item interaction DataFrame.

    Returns:
    int: Total number of ratings.
    """
    return len(df)


def get_number_of_users(df):
    """
    Returns the number of unique users in the DataFrame.

    Parameters:
    df (DataFrame): User-item interaction DataFrame.

    Returns:
    int: Number of unique users.
    """
    return df['uid'].nunique()


def get_number_of_items(df):
    """
    Returns the number of unique items in the DataFrame.

    Parameters:
    df (DataFrame): User-item interaction DataFrame.

    Returns:
    int: Number of unique items.
    """
    return df['iid'].nunique()


def calculate_sparsity(n_users, n_items, n_ratings):
    """
    Calculate the sparsity of the user-item interaction matrix.

    Parameters:
    n_users (int): The number of unique users.
    n_items (int): The number of unique items.
    n_ratings (int): The number of available ratings (non-zero entries in the matrix).

    Returns:
    float: The sparsity of the matrix.
    """
    if not isinstance(n_users, int) or not isinstance(n_items, int):
        raise ValueError("num_users, num_items must be integers")
    if n_users <= 0 or n_items <= 0:
        raise ValueError("num_users, num_items must be positive")
    rating_matrix_size = n_users * n_items
    sparsity = 1 - (n_ratings / rating_matrix_size)

    return sparsity


def get_user_interaction_list(df):
    """
    Returns the list of user interaction counts.

    Parameters:
    df (DataFrame): DataFrame containing 'uid' column.

    Returns:
    list: A list of interaction counts for each user, sorted in descending order.
    """
    return df['uid'].value_counts().tolist()


def get_item_interaction_list(df):
    """
    Returns the list of item interaction counts.

    Parameters:
    df (DataFrame): DataFrame containing 'iid' column.

    Returns:
    list: A list of interaction counts for each item, sorted in descending order.
    """
    return df['iid'].value_counts().tolist()


def load_uir_dataset(fpath):
    '''
    Returns the dataframe of the user-item-rating file

    Parameters
    ----------
    fpath: the path of the uir file

    Returns:
    Dataframe: A datafram has user, item, rating information, named in 'uid', 'iid', 'rating'.
    -------

    '''
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        if len(df.columns) == 3:
            df.columns = ['uid', 'iid', 'rating']
        if len(df.columns) > 3:
            df.drop(columns=df.columns[0], axis=1, inplace=True)
            df.columns = ['uid', 'iid', 'rating']
        return df

