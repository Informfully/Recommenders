'''
This file is mainly based on the cornac.metrics.diversity.py
'''


import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import cdist
import math
from sklearn.preprocessing import KBinsDiscretizer
from cornac.metrics.diversity import compute_distr_continous
from cornac.metrics.diversity import compute_divergence
from cornac.metrics.diversity import compute_distr_listData
from cornac.metrics.diversity import compute_distr_majority_minority
from cornac.metrics.diversity import compute_distr_category


class UserMetric:
    """Diversity Metric.

    Attributes
    ----------
    type: string, value: 'user'
        Type of the metric, e.g., "ranking", "rating","diversity".

    name: string, default: None
        Name of the measure.
    """

    def __init__(self, name=None, higher_better=False):

        self.type = "user"
        self.name = name
        self.higher_better = higher_better

    def compute_user(self, **kwargs):
        raise NotImplementedError()


class UserActivation(UserMetric):
    """
    Activation metric.

    Parameters
    ----------
    item_sentiments: Dictionary. {item id: item sentiments}.
        Contains item id mapped to their respective sentiments.

    user_seen_item: Dataframe.
        DataFrame summarizing the items seen by each user based on user-item rating data.

    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.

    divergence_type: (Optional) string that determines the divergence method employed.
        The choices are "JS" or "KL". The default value is "KL".

    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September).
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_sentiments, user_seen_item, divergence_type="KL", discount=False, n_bins=5):
        UserMetric.__init__(self, name="Activation")
        if isinstance(item_sentiments, dict):
            self.item_sentiments = item_sentiments
        else:
            raise ValueError(
                "Activation received an invalid item_sentiments. item_sentiments "
                "must be a dictionary."
            )
        if isinstance(user_seen_item, pd.DataFrame):
            self.user_seen_item = user_seen_item
        else:
            raise ValueError(
                "Activation received an invalid user_seen_item. user_seen_item "
                "must be a dataframe."
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

    def compute_user(self, user_exposed_df, **kwargs):
        """
        Computes activation scores for users based on their seen items' sentiment compared to sentiment distributions of exposed items.

        Parameters:
            user_exposed_df (DataFrame): DataFrame containing information about items exposed to users.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary where keys are user IDs and values are corresponding activation scores.
            Activation scores quantify how much the sentiment of items seen by a user diverges from the sentiment distribution
            of items exposed to that user.
        """
        if user_exposed_df.empty or self.user_seen_item.empty:
            return {}
        self.user_seen_item['sentiment'] = self.user_seen_item['item seen'].apply(map_to_feature, item_feature_mapping=self.item_sentiments)
        user_sentiment_dict = self.user_seen_item.set_index('user id')['sentiment'].to_dict()
        user_exposed_df['sentiment'] = user_exposed_df['item exposed'].apply(map_to_feature, item_feature_mapping=self.item_sentiments)
        user_exposed_sentiment = user_exposed_df.set_index('user id')['sentiment'].to_dict()
        activation_scores = {}

        for user_id, item_sentiment in user_sentiment_dict.items():
            item_sentiment = [x for x in item_sentiment if x is not None]
            if not item_sentiment:
                activation_scores[user_id] = None
                continue

            try:
                item_sentiment = pd.to_numeric(item_sentiment, errors='coerce')
                item_sentiment = np.abs(np.array(item_sentiment)).reshape(-1, 1)
                pool_sentiment = np.abs(np.array(user_exposed_sentiment[user_id])).reshape(-1, 1)
                bins_discretizer = KBinsDiscretizer(
                    encode='ordinal', n_bins=self.n_bins, strategy='uniform', subsample=None)
                bins_discretizer.fit(pool_sentiment)
                distr_pool = compute_distr_continous(
                    pool_sentiment, bins_discretizer, False)
                distr_item_sentiments = compute_distr_continous(
                    item_sentiment, bins_discretizer, False)
                act = compute_divergence(
                    distr_item_sentiments, distr_pool, divergence_type=self.divergence_type)
                activation_scores[user_id] = act
            except ValueError:
                activation_scores[user_id] = None
        return activation_scores

    def compute_user_during_time(self, user_exposed_df, **kwargs):
        """
        Computes activation scores for the user up to that timestamp.

        Parameters:
            user_exposed_df (DataFrame): DataFrame containing information about items exposed to users.
            **kwargs: Additional keyword arguments.

        Returns:
            DataFrame: A DataFrame where each row contains time, and corresponding activation score.
        """
        if user_exposed_df.empty or self.user_seen_item.empty:
            return pd.DataFrame()

        self.user_seen_item['sentiment'] = self.user_seen_item['item seen'].apply(map_to_feature,
                                                                                  item_feature_mapping=self.item_sentiments)
        user_sentiment_dict = self.user_seen_item.set_index('Time')['sentiment'].to_dict()
        user_exposed_df['sentiment'] = user_exposed_df['item exposed'].apply(map_to_feature,
                                                                             item_feature_mapping=self.item_sentiments)
        user_exposed_sentiment = user_exposed_df.set_index('Time')['sentiment'].to_dict()
        activation_scores = {}


        for timestamp, item_sentiment in user_sentiment_dict.items():
            item_sentiment = [x for x in item_sentiment if x is not None]
            if not item_sentiment:
                activation_scores[timestamp] = None
                continue

            try:
                item_sentiment = pd.to_numeric(item_sentiment, errors='coerce')
                item_sentiment = np.abs(np.array(item_sentiment)).reshape(-1, 1)
                pool_sentiment = np.abs(np.array(user_exposed_sentiment[timestamp])).reshape(-1, 1)
                bins_discretizer = KBinsDiscretizer(
                    encode='ordinal', n_bins=self.n_bins, strategy='uniform', subsample=None)
                bins_discretizer.fit(pool_sentiment)
                distr_pool = compute_distr_continous(
                    pool_sentiment, bins_discretizer, False)
                distr_item_sentiments = compute_distr_continous(
                    item_sentiment, bins_discretizer, False)
                act = compute_divergence(
                    distr_item_sentiments, distr_pool, divergence_type=self.divergence_type)
                activation_scores[timestamp] = act
            except ValueError:
                activation_scores[timestamp] = None

        result_df = pd.DataFrame(list(activation_scores.items()), columns=['Time', 'activation score'])
        return result_df


class UserGiniCoeff(UserMetric):
    """
    Gini coefficient.
    item_genres: dictionary.
        A dictionary that maps item id to the list of item genres array(numpy array).
        The item genres array must be an ordered list,
        where a value of 0 indicates that the item does not belong to a certain category at the corresponding position
        in the genre list, and 1 indicates that the item belong to a certain category.

    user_seen_item: Dataframe.
        DataFrame summarizing the items seen by each user based on user-item rating data.

    References
    ----------
    - Ricci, F., Rokach, L. and Shapira, B., 2015. Recommender systems: introduction and challenges. Recommender systems handbook, pp.1-34.
    """

    def __init__(self, item_genres, user_seen_item):
        UserMetric.__init__(self, name="GiniCoeff")
        if isinstance(item_genres, dict):
            self.item_genres = item_genres
        else:
            raise ValueError(
                "GiniCoeff received an invalid item_genres. item_genres "
                "must be a dictionary."
            )
        if isinstance(user_seen_item, pd.DataFrame):
            self.user_seen_item = user_seen_item
        else:
            raise ValueError(
                "GiniCoeff received an invalid user_seen_item. user_seen_item "
                "must be a dataframe."
            )

    def compute_user(self, **kwargs):
        """
        Computes Gini coefficients for users based on the distribution of item categories seen by each user.

        Parameters:
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary where keys are user IDs and values are corresponding Gini coefficients.
            Gini coefficients measure the inequality of the distribution of item categories seen by each user.
            Higher Gini coefficients indicate more unequal distributions, while lower values indicate more even distributions.

        Notes:
            - It calculates the Gini coefficient based on the proportion of each category in the user's seen items.
        """
        if self.user_seen_item.empty:
            return {}

        self.user_seen_item['category'] = self.user_seen_item['item seen'].apply(map_to_feature, item_feature_mapping=self.item_genres)
        user_category_dict = self.user_seen_item.set_index('user id')['category'].to_dict()
        #print(user_category_dict)
        gini_coefficients = {}
        for user_id, item_genre_vectors in user_category_dict.items():
            item_genre_vectors = [i for i in item_genre_vectors if i is not None]
            if not item_genre_vectors:
                gini_coefficients[user_id] = None
                continue

            pool_items_genre = []
            for genre_vector in item_genre_vectors:
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
                gini_coefficients[user_id] = G / (n - 1)
            except ValueError:
                gini_coefficients[user_id] = None
        return gini_coefficients

    def compute_user_during_time(self, **kwargs):
        """
        Computes Gini coefficients scores for the user up to that timestamp.

        Parameters:
            **kwargs: Additional keyword arguments.

        Returns:
            DataFrame: A DataFrame where each row contains time, and corresponding Gini coefficients score.
        """
        if self.user_seen_item.empty:
            return pd.DataFrame()

        self.user_seen_item['category'] = self.user_seen_item['item seen'].apply(map_to_feature,
                                                                                 item_feature_mapping=self.item_genres)
        user_category_dict = self.user_seen_item.set_index('Time')['category'].to_dict()
        gini_coefficients = {}
        for timestamp, item_genre_vectors in user_category_dict.items():
            item_genre_vectors = [i for i in item_genre_vectors if i is not None]
            if not item_genre_vectors:
                gini_coefficients[timestamp] = None
                continue

            pool_items_genre = []
            for genre_vector in item_genre_vectors:
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
                gini_coefficients[timestamp] = G / (n - 1)
            except ValueError:
                gini_coefficients[timestamp] = None

        result_df = pd.DataFrame(list(gini_coefficients.items()), columns=['Time', 'gini score'])
        return result_df


class UserAlternativeVoices(UserMetric):
    """Alternative Voices metric.

    Parameters
    ----------
    item_minor_major: Dictionary. {item id: [[number of minority mentions, number of majority mentions], [number of minority mentions, number of majority mentions]]}.
        A dictionary that maps item id to the list of item min_maj array(numpy array).

    user_seen_item: Dataframe.
        DataFrame summarizing the items seen by each user based on user-item rating data.

    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.

    divergence_type: (Optional) string that determines the divergence method employed.
        The choices are "JS" or "KL". The default value is "KL".


    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September).
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_minor_major, user_seen_item, data_type='mainstream', divergence_type="KL", discount=False):
        if data_type.lower() == "gender":
            UserMetric.__init__(
                self, name="AltVoices_gender")
        elif data_type.lower() == "ethnicity":
            UserMetric.__init__(
                self, name="AltVoices_ethnicity")
        else:
            UserMetric.__init__(
                self, name="AltVoices_mainstream")
        if isinstance(item_minor_major, dict):
            self.item_minor_major = item_minor_major
        else:
            raise ValueError(
                "AlternativeVoices received an invalid item_minor_major. item_minor_major "
                "must be a dictionary."
            )
        if isinstance(user_seen_item, pd.DataFrame):
            self.user_seen_item = user_seen_item
        else:
            raise ValueError(
                "AlternativeVoice received an invalid user_seen_item. user_seen_item "
                "must be a dataframe."
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

    def compute_user(self, user_exposed_df, **kwargs):
        """
        Computes alternative voice scores for users based on the minority-majority information of items seen by each user
        compared to the distribution of minority-majority information in items exposed to users.

        Parameters:
            user_exposed_df (DataFrame): DataFrame containing information about items exposed to users.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary where keys are user IDs and values are corresponding alternative voice scores.
            Alternative voice scores quantify how much the minority-majority information of items seen by a user differs
            from the distribution of minority-majority information in items exposed to that user.
            Higher scores indicate greater divergence, while None values indicate issues during computation.
        """
        if user_exposed_df.empty or self.user_seen_item.empty:
            return {}
        self.user_seen_item['min_maj'] = self.user_seen_item['item seen'].apply(map_to_feature, item_feature_mapping=self.item_minor_major)
        user_min_maj_dict = self.user_seen_item.set_index('user id')['min_maj'].to_dict()
        user_exposed_df['min_maj'] = user_exposed_df['item exposed'].apply(map_to_feature, item_feature_mapping=self.item_minor_major)
        user_exposed_sentiment = user_exposed_df.set_index('user id')['min_maj'].to_dict()
        alternative_voice_score = {}
        for user_id, item_min_maj in user_min_maj_dict.items():
            item_min_maj = [i for i in item_min_maj if i is not None]
            if not item_min_maj:
                alternative_voice_score[user_id] = None
                continue
            pool_minor_major = {}
            # Iterate over each item in the pool
            for idx, i in enumerate(user_exposed_sentiment[user_id]):
                pool_minor_major[idx] = i
            # If no items in the pool have minority-majority information, return None
            if len(pool_minor_major) == 0:
                return None

            user_minor_major = {}
            for idx, j in enumerate(item_min_maj):
                user_minor_major[idx] = j

            try:
                pool_distr = compute_distr_majority_minority(pool_minor_major, False)
                user_distr = compute_distr_majority_minority(user_minor_major, discount=self.discount)
                divergence = None
                if not (user_distr[0] == 0 and user_distr[1] == 0):
                    divergence = compute_divergence(user_distr, pool_distr, divergence_type=self.divergence_type)
                alternative_voice_score[user_id] = divergence
            except ValueError:
                alternative_voice_score[user_id] = None
        return alternative_voice_score

    def compute_user_during_time(self, user_exposed_df, **kwargs):
        """
        Computes alternative voice scores for the user up to that timestamp.
        Parameters:
            user_exposed_df (DataFrame): DataFrame containing information about items exposed to users.
            **kwargs: Additional keyword arguments.

        Returns:
            DataFrame: A DataFrame where each row contains time, and corresponding alternative voice scores.
        """
        if user_exposed_df.empty or self.user_seen_item.empty:
            return pd.DataFrame()

        self.user_seen_item['min_maj'] = self.user_seen_item['item seen'].apply(map_to_feature,
                                                                                item_feature_mapping=self.item_minor_major)
        user_min_maj_dict = self.user_seen_item.set_index('Time')['min_maj'].to_dict()
        user_exposed_df['min_maj'] = user_exposed_df['item exposed'].apply(map_to_feature,
                                                                           item_feature_mapping=self.item_minor_major)
        user_exposed_sentiment = user_exposed_df.set_index('Time')['min_maj'].to_dict()
        alternative_voice_score = {}
        for timestamp, item_min_maj in user_min_maj_dict.items():
            item_min_maj = [i for i in item_min_maj if i is not None]
            if not item_min_maj:
                alternative_voice_score[timestamp] = None
                continue
            pool_minor_major = {}
            # Iterate over each item in the pool
            for idx, i in enumerate(user_exposed_sentiment[timestamp]):
                pool_minor_major[idx] = i
            # If no items in the pool have minority-majority information, return None
            if len(pool_minor_major) == 0:
                return None

            user_minor_major = {}
            for idx, j in enumerate(item_min_maj):
                user_minor_major[idx] = j

            try:
                pool_distr = compute_distr_majority_minority(pool_minor_major, False)
                user_distr = compute_distr_majority_minority(user_minor_major, discount=self.discount)
                divergence = None
                if not (user_distr[0] == 0 and user_distr[1] == 0):
                    divergence = compute_divergence(user_distr, pool_distr, divergence_type=self.divergence_type)
                alternative_voice_score[timestamp] = divergence
            except ValueError:
                alternative_voice_score[timestamp] = None
        result_df = pd.DataFrame(list(alternative_voice_score.items()), columns=['Time', 'alternative voice score'])
        return result_df


class UserRepresentation(UserMetric):
    """
    Representation metric.

    Parameters
    ----------
    item_entities: Dictionary. {user id: [[item entities], [item entities]]}.
        A dictionary that maps 'User ID' to the respective list of item entities array(numpy array).

    user_seen_item: Dataframe.
        DataFrame summarizing the items seen by each user based on user-item rating data.

    discount: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness.
        If True, rank-awareness is evaluated.

    divergence_type: (Optional) string that determines the divergence method employed.
        The choices are "JS" or "KL". The default value is "KL".

    References
    ----------
    Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September).
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_entities, user_seen_item, divergence_type="KL", discount=False):
        UserMetric.__init__(self, name="Representation")
        if isinstance(item_entities, dict):
            self.item_entities = item_entities
        else:
            raise ValueError(
                "Representation received an invalid user_item_entities. user_item_entities "
                "must be a dictionary."
            )
        if isinstance(user_seen_item, pd.DataFrame):
            self.user_seen_item = user_seen_item
        else:
            raise ValueError(
                "Representation received an invalid user_seen_item. user_seen_item "
                "must be a dataframe."
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

    def compute_user(self, user_exposed_df, **kwargs):
        """
        Computes representation scores for users based on the entities associated with items seen by each user
        compared to the distribution of entities in items exposed to users.

        Parameters:
            user_exposed_df (DataFrame): DataFrame containing information about items exposed to users.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary where keys are user IDs and values are corresponding representation scores.
              Representation scores quantify how much the entities associated with items seen by a user differ
              from the distribution of entities in items exposed to that user.
              Higher scores indicate greater divergence, while None values indicate issues during computation.
        """
        if user_exposed_df.empty or self.user_seen_item.empty:
            return {}
        self.user_seen_item['entities'] = self.user_seen_item['item seen'].apply(map_to_feature, item_feature_mapping=self.item_entities)
        user_representation_dict = self.user_seen_item.set_index('user id')['entities'].to_dict()
        user_exposed_df['representation'] = user_exposed_df['item exposed'].apply(map_to_feature, item_feature_mapping=self.item_entities)
        user_exposed_sentiment = user_exposed_df.set_index('user id')['representation'].to_dict()
        representation_score = {}
        for user_id, item_entity in user_representation_dict.items():
            item_entity = [i for i in item_entity if i is not None]
            if not item_entity:
                representation_score[user_id] = None
                continue

            pool_entities = {}
            # Iterate over each item in the pool
            for idx, i in enumerate(user_exposed_sentiment[user_id]):
                pool_entities[idx] = i
            # If no items in the pool have information, return None
            if len(pool_entities) == 0:
                return None

            user_entities = {}
            for idx, j in enumerate(item_entity):
                user_entities[idx] = j
            if len(user_entities) == 0 or len(pool_entities) == 0:
                return None

            try:
                pool_distr = compute_distr_listData(pool_entities, False)
                user_distr = compute_distr_listData(user_entities, discount=self.discount)
                rep = compute_divergence(user_distr, pool_distr, divergence_type=self.divergence_type)
                representation_score[user_id] = rep
            except ValueError:
                representation_score[user_id] = None
        return representation_score

    def compute_user_during_time(self, user_exposed_df, **kwargs):
        """
        Computes  representation scores for the user up to that timestamp.
        Parameters:
            user_exposed_df (DataFrame): DataFrame containing information about items exposed to users.
            **kwargs: Additional keyword arguments.

        Returns:
            DataFrame: A DataFrame where each row contains time, and corresponding representation scores.
        """
        if user_exposed_df.empty or self.user_seen_item.empty:
            return pd.DataFrame()

        self.user_seen_item['entities'] = self.user_seen_item['item seen'].apply(map_to_feature,
                                                                                 item_feature_mapping=self.item_entities)
        user_representation_dict = self.user_seen_item.set_index('Time')['entities'].to_dict()
        user_exposed_df['representation'] = user_exposed_df['item exposed'].apply(map_to_feature,
                                                                                  item_feature_mapping=self.item_entities)
        user_exposed_sentiment = user_exposed_df.set_index('Time')['representation'].to_dict()
        representation_score = {}
        for timestamp, item_entity in user_representation_dict.items():
            item_entity = [i for i in item_entity if i is not None]
            if not item_entity:
                representation_score[timestamp] = None
                continue

            pool_entities = {}
            # Iterate over each item in the pool
            for idx, i in enumerate(user_exposed_sentiment[timestamp]):
                pool_entities[idx] = i
            # If no items in the pool have information, return None
            if len(pool_entities) == 0:
                return None

            user_entities = {}
            for idx, j in enumerate(item_entity):
                user_entities[idx] = j
            if len(user_entities) == 0 or len(pool_entities) == 0:
                return None

            try:
                pool_distr = compute_distr_listData(pool_entities, False)
                user_distr = compute_distr_listData(user_entities, discount=self.discount)
                rep = compute_divergence(user_distr, pool_distr, divergence_type=self.divergence_type)
                representation_score[timestamp] = rep
            except ValueError:
                representation_score[timestamp] = None

        result_df = pd.DataFrame(list(representation_score.items()), columns=['Time', 'representation score'])
        result_df['representation score'] = result_df['representation score'].apply(
                                            lambda x: max(x, 0) if x is not None else None)

        return result_df


class UserCalibration(UserMetric):
    """Calibration metric.

    Parameters
    ----------
    item_features: Dictionary. {item id: item feature vectors}.
         A dictionary that maps item id to their respective categories or complexities.
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

    def __init__(self, item_features, user_seen_item, data_type, divergence_type="KL", discount=False, n_bins=5):
        if data_type.lower() == "complexity":
            UserMetric.__init__(
                self, name="Calibration_complexity")
        else:
            UserMetric.__init__(
                self, name="Calibration_category")
        if isinstance(item_features, dict):
            self.item_features = item_features
        else:
            raise ValueError(
                "Calibration received an invalid item_features. item_features "
                "must be a dictionary."
            )
        if isinstance(user_seen_item, pd.DataFrame):
            self.user_seen_item = user_seen_item
        else:
            raise ValueError(
                "Calibration received an invalid user_seen_item. user_seen_item "
                "must be a dataframe."
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

    def compute_user(self, user_exposed_df, **kwargs):
        """
        Computes calibration scores for users based on the similarity of item features (categories or complexities)
        between items seen by each user and items exposed to them.

        Parameters:
            user_exposed_df (DataFrame): DataFrame containing information about items exposed to users.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary where keys are user IDs and values are corresponding calibration scores.
                  Calibration scores quantify how similar the item features (categories or complexities) of items seen by a user
                  are to the distribution of item features in items exposed to that user.
                  Higher scores indicate greater divergence, while None values indicate issues during computation.

        Notes:
            - This function supports two data types for item features: "category" or "complexity".
            - It calculates calibration scores based on the divergence between the feature distributions
            of items seen by a user and those exposed to the user.
        """
        if user_exposed_df.empty or self.user_seen_item.empty:
            return {}
        if self.data_type == "category":
            self.user_seen_item['category_cali'] = self.user_seen_item['item seen'].apply(map_to_feature, item_feature_mapping=self.item_features)
            user_category_dict = self.user_seen_item.set_index('user id')['category_cali'].to_dict()
            user_exposed_df['category_cali'] = user_exposed_df['item exposed'].apply(map_to_feature, item_feature_mapping=self.item_features)
            user_exposed_complexity = user_exposed_df.set_index('user id')['category_cali'].to_dict()
            cali_category_scores = {}
            for user_id, item_category in user_category_dict.items():
                item_category = [x for x in item_category if x is not None]
                if not item_category:
                    user_category_dict[user_id] = None
                    continue
                try:
                    freq_seen = compute_distr_category(user_category_dict[user_id], discount=self.discount)
                    freq_exposed = compute_distr_category(user_exposed_complexity[user_id], discount=self.discount)
                    cal = compute_divergence(freq_seen, freq_exposed, divergence_type=self.divergence_type)
                    cali_category_scores[user_id] = cal
                except ValueError:
                    cali_category_scores[user_id] = None
            return cali_category_scores

        elif self.data_type == "complexity":
            self.user_seen_item['complexity'] = self.user_seen_item['item seen'].apply(map_to_feature, item_feature_mapping=self.item_features)
            user_complexity_dict = self.user_seen_item.set_index('user id')['complexity'].to_dict()
            user_exposed_df['complexity'] = user_exposed_df['item exposed'].apply(map_to_feature, item_feature_mapping=self.item_features)
            user_exposed_complexity = user_exposed_df.set_index('user id')['complexity'].to_dict()
            cali_complexity_scores = {}
            for user_id, item_complexity in user_complexity_dict.items():
                item_complexity = [x for x in item_complexity if x is not None]
                if not item_complexity:
                    cali_complexity_scores[user_id] = None
                    continue
                try:
                    user_complexity = np.array(item_complexity).reshape(-1, 1)
                    pool_complexity = np.array(user_exposed_complexity[user_id]).reshape(-1, 1)
                    if pool_complexity.shape[0] > 1:
                        bins_discretizer = KBinsDiscretizer(
                            encode='ordinal', n_bins=self.n_bins, strategy='uniform', subsample=None)
                        bins_discretizer.fit(pool_complexity)
                        distr_pool = compute_distr_continous(pool_complexity, bins_discretizer, self.discount)
                        distr_user = compute_distr_continous(user_complexity, bins_discretizer, self.discount)
                        cal = compute_divergence(distr_user, distr_pool, divergence_type=self.divergence_type)
                        cali_complexity_scores[user_id] = cal
                    else:
                        cali_complexity_scores[user_id] = None
                except ValueError:
                    cali_complexity_scores[user_id] = None
            return cali_complexity_scores

    def compute_user_during_time(self, user_exposed_df, **kwargs):
        """
        Computes  calibration scores for the user up to that timestamp.
        Parameters:
            user_exposed_df (DataFrame): DataFrame containing information about items exposed to users.
            **kwargs: Additional keyword arguments.

        Returns:
            DataFrame: A DataFrame where each row contains time, and corresponding calibration scores.
        """
        if user_exposed_df.empty or self.user_seen_item.empty:
            return pd.DataFrame()

        if self.data_type == "category":
            self.user_seen_item['category_cali'] = self.user_seen_item['item seen'].apply(map_to_feature,
                                                                                              item_feature_mapping=self.item_features)
            user_category_dict = self.user_seen_item.set_index('Time')['category_cali'].to_dict()
            user_exposed_df['category_cali'] = user_exposed_df['item exposed'].apply(map_to_feature,
                                                                                         item_feature_mapping=self.item_features)
            user_exposed_complexity = user_exposed_df.set_index('Time')['category_cali'].to_dict()
            cali_category_scores = {}
            for timestamp, item_category in user_category_dict.items():
                item_category = [x for x in item_category if x is not None]
                if not item_category:
                    user_category_dict[timestamp] = None
                    continue
                try:
                    freq_seen = compute_distr_category(user_category_dict[timestamp], discount=self.discount)
                    freq_exposed = compute_distr_category(user_exposed_complexity[timestamp], discount=self.discount)
                    cal = compute_divergence(freq_seen, freq_exposed, divergence_type=self.divergence_type)
                    cali_category_scores[timestamp] = cal
                except ValueError:
                    cali_category_scores[timestamp] = None
            result_df = pd.DataFrame(list(cali_category_scores.items()), columns=['Time', 'cali_category score'])
            return result_df

        elif self.data_type == "complexity":
            self.user_seen_item['complexity'] = self.user_seen_item['item seen'].apply(map_to_feature,
                                                                                           item_feature_mapping=self.item_features)
            user_complexity_dict = self.user_seen_item.set_index('Time')['complexity'].to_dict()
            user_exposed_df['complexity'] = user_exposed_df['item exposed'].apply(map_to_feature,
                                                                                      item_feature_mapping=self.item_features)
            user_exposed_complexity = user_exposed_df.set_index('Time')['complexity'].to_dict()
            cali_complexity_scores = {}
            for timestamp, item_complexity in user_complexity_dict.items():
                item_complexity = [x for x in item_complexity if x is not None]
                if not item_complexity:
                    cali_complexity_scores[timestamp] = None
                    continue
                try:
                    user_complexity = np.array(item_complexity).reshape(-1, 1)
                    pool_complexity = np.array(user_exposed_complexity[timestamp]).reshape(-1, 1)
                    if pool_complexity.shape[0] > 1:
                        bins_discretizer = KBinsDiscretizer(
                                encode='ordinal', n_bins=self.n_bins, strategy='uniform', subsample=None)
                        bins_discretizer.fit(pool_complexity)
                        distr_pool = compute_distr_continous(pool_complexity, bins_discretizer, self.discount)
                        distr_user = compute_distr_continous(user_complexity, bins_discretizer, self.discount)
                        cal = compute_divergence(distr_user, distr_pool, divergence_type=self.divergence_type)
                        cali_complexity_scores[timestamp] = cal
                    else:
                        cali_complexity_scores[timestamp] = None
                except ValueError:
                    cali_complexity_scores[timestamp] = None

            result_df = pd.DataFrame(list(cali_complexity_scores.items()), columns=['Time', 'cali_complexity score'])
            return result_df


class UserFragmentation(UserMetric):
    """Fragmentation metric.

    Parameters
    ----------
    item_stories: Dictionary. {item id: item feature vectors}.
        A dictionary that maps item id to their respective story chains.
        The stories are categorical values.
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
    RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).

    """

    def __init__(self, item_stories, user_exposed_item, divergence_type="KL", discount=False, n_samples=1):
        UserMetric.__init__(self, name="Fragmentation")
        if isinstance(item_stories, dict):
            self.item_stories = item_stories
        else:
            raise ValueError(
                "Fragmentation received an invalid item_stories. item_stories "
                "must be a dictionary."
            )
        if isinstance(user_exposed_item, pd.DataFrame):
            self.user_exposed_item = user_exposed_item
        else:
            raise ValueError(
                "Fragmentation received an invalid user_exposed_item. user_exposed_item "
                "must be a dataframe."
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

    def compute_user(self, **kwargs):
        """
        Computes fragmentation scores for users based on the divergence between the stories of items seen by each user
        and randomly selected stories of items seen by other users.

        Parameters:
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary where keys are user IDs and values are corresponding fragmentation scores.
                  Fragmentation scores quantify the extent to which the stories of items seen by a user differ from
                  randomly selected stories of items seen by other users.
                  Higher scores indicate greater divergence, while None values indicate issues during computation.

        Notes:
            - It generates random comparisons with other users' item stories to assess the divergence.
        """
        if self.user_exposed_item.empty:
            return {}
        self.user_exposed_item['story'] = self.user_exposed_item['item exposed'].apply(map_to_feature, item_feature_mapping=self.item_stories)
        user_story_dict = self.user_exposed_item.set_index('user id')['story'].to_dict()
        fragmentation_score = {}
        unique_user_ids = self.user_exposed_item['user id'].unique()
        for user_id, item_story in user_story_dict.items():
            item_story = [x for x in item_story if x is not None]
            if not item_story:
                fragmentation_score[user_id] = None
                continue
            # Generate random comparisons for each user
            selected_user_ids = np.random.choice(unique_user_ids[unique_user_ids != user_id], size=self.n_samples, replace=False)
            other_users = []
            for i in selected_user_ids:
                other_users.append(user_story_dict[i])
            divergence = []
            try:
                for other_seen in other_users:
                    story_other_seen = np.array([x for x in other_seen if x is not None])
                    freq_x = compute_distr_category(item_story, discount=self.discount)
                    freq_y = compute_distr_category(story_other_seen, discount=self.discount)
                    div_recom_sample = compute_divergence(freq_x, freq_y, alpha=0.001, divergence_type=self.divergence_type)
                    if div_recom_sample is not None:
                        divergence.append(div_recom_sample)
                if len(divergence) > 0:
                    single_frag = np.mean(divergence)
                    fragmentation_score[user_id] = single_frag
                else:
                    fragmentation_score[user_id] = None
                fragmentation_score[user_id] = single_frag
            except ValueError:
                fragmentation_score[user_id] = None
        return fragmentation_score


class UserILD(UserMetric):
    """Intra-list diversity.

    Parameters
    ----------
    item_features: A dictionary that maps item indices to their feature vectors.
        The feature vector must be provided using numpy array.
    distance_type: (Optional) String for configuring distance measure.
        ‘correlation’, ‘cosine’,‘euclidean’ or ‘jaccard’.
        By default, use cosine distance metric.

    References
    ----------
    https://www.academia.edu/2655896/Improving_recommendation_diversity
    """

    def __init__(self, item_features, user_seen_item, distance_type="cosine"):
        UserMetric.__init__(self, name="ILD")
        if isinstance(item_features, dict):
            self.item_features = item_features
        else:
            raise ValueError(
                "ILD received an invalid item_features. item_features "
                "must be a dictionary."
            )
        if isinstance(user_seen_item, pd.DataFrame):
            self.user_seen_item = user_seen_item
        else:
            raise ValueError(
                "ILD received an invalid user_seen_item. user_seen_item "
                "must be a dataframe."
            )
        if isinstance(distance_type, str) and distance_type.lower() in ["jaccard", "correlation",
                                                                        "cosine", "euclidean"]:
            self.distance_type = distance_type
        else:
            self.distance_type = "cosine"

    def compute_user(self, **kwargs):
        """
        Computes Intra-List Distance (ILD) scores for users based on the similarity of item genres within each user's seen items.

        Parameters:
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary where keys are user IDs and values are corresponding ILD scores.
                  ILD scores quantify the average distance between item features within each user's seen items.
                  Lower scores indicate greater similarity, while None values indicate issues during computation.
        """
        if self.user_seen_item.empty:
            return {}
        self.user_seen_item['genre'] = self.user_seen_item['item seen'].apply(map_to_feature, item_feature_mapping=self.item_features)
        user_genre_dict = self.user_seen_item.set_index('user id')['genre'].to_dict()
        ild_score = {}
        for user_id, item_genre in user_genre_dict.items():
            genre_vec0 = [i.tolist() for i in item_genre if i is not None]
            genre_vec = [x for x in genre_vec0 if x]
            if not genre_vec:
                ild_score[user_id] = None
                continue
            try:
                distance = cdist(genre_vec, genre_vec, metric=self.distance_type)
                upper_right = np.triu_indices(distance.shape[0], k=1)
                if distance[upper_right].size == 0:
                    ild = 0
                else:
                    ild = np.mean(distance[upper_right])
                ild_score[user_id] = ild
            except ValueError:
                ild_score[user_id] = None
        return ild_score

    def compute_user_during_time(self, **kwargs):
        """
        Computes ild scores for the user up to that timestamp.

        Parameters:
            **kwargs: Additional keyword arguments.

        Returns:
            DataFrame: A DataFrame where each row contains time, and corresponding ild score.
        """
        if self.user_seen_item.empty:
            return pd.DataFrame()

        self.user_seen_item['genre'] = self.user_seen_item['item seen'].apply(map_to_feature,
                                                                              item_feature_mapping=self.item_features)
        user_genre_dict = self.user_seen_item.set_index('Time')['genre'].to_dict()
        ild_score = {}
        for timestamp, item_genre in user_genre_dict.items():
            genre_vec0 = [i.tolist() for i in item_genre if i is not None]
            genre_vec = [x for x in genre_vec0 if x]
            if not genre_vec:
                ild_score[timestamp] = None
                continue
            try:
                distance = cdist(genre_vec, genre_vec, metric=self.distance_type)
                upper_right = np.triu_indices(distance.shape[0], k=1)
                if distance[upper_right].size == 0:
                    ild = 0
                else:
                    ild = np.mean(distance[upper_right])
                ild_score[timestamp] = ild
            except ValueError:
                ild_score[timestamp] = None

        result_df = pd.DataFrame(list(ild_score.items()), columns=['Time', 'ild score'])
        return result_df


def create_user_seen_item_df(df):
    """
    Create a DataFrame summarizing the items seen by each user based on user-item rating data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing user-item rating information.

    Returns
    -------
    DataFrame
        A new DataFrame with two columns: 'uid' and 'item seen', where 'uid' represents
        the user identifier and 'item seen' contains a list of items that the user has seen.

    """
    # Filter the DataFrame based on the condition 'rating == 1'
    filtered_df = df[df['rating'] == 1]
    user_item_df = filtered_df.groupby('uid')['iid'].apply(list).reset_index()
    user_item_df.columns = ['user id', 'item seen']

    return user_item_df


def map_to_feature(item_list, item_feature_mapping):
    """
    Maps item IDs to features based on the provided item category mapping.

    Parameters
    ----------
    item_list (list): List of item IDs.
    item_feature_mapping (dict): Dictionary mapping item IDs to features.

    Returns
    ----------
    list: List of features corresponding to the input item IDs.
    """
    features = []
    for item in item_list:
        if pd.notna(item) and item in item_feature_mapping:
            features.append(item_feature_mapping[item])
    return features


def create_score_df(score, column_name):
    """
    Create a DataFrame from a dictionary containing user IDs and activation scores.

    Parameters
    ----------
    score : dict
        Dictionary containing user IDs as keys and scores as values.
    column_name : str, optional
        Name of the column for scores.

    Returns
    -------
    pd.DataFrame
        DataFrame containing user IDs and activation scores.
    """
    df = pd.DataFrame.from_dict(score, orient='index', columns=[column_name])
    # Reset index to make user ID a column instead of index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'User_ID'}, inplace=True)
    return df


def save_dataframe_to_csv(df, file_path):
    """
    Save a DataFrame to a CSV file.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be saved.
    file_path : str
        The path to the CSV file where the DataFrame will be saved.

    Returns
    -------
    None
    """
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")


def create_user_exposed_df(df):
    """
    Aggregate user-item interaction data by user ID to create a DataFrame
    representing the items exposed to each user.

    Parameters
    ----------
     df : pandas.DataFrame
        Input DataFrame containing user-item rating information.

    Returns
    -------
    pandas DataFrame
        DataFrame where each row represents a unique user,
        and the 'item exposed' column contains lists of items exposed to each user.
    """
    user_exposed_df = df.groupby('uid')['iid'].apply(list).reset_index()
    user_exposed_df.columns = ['user id', 'item exposed']

    return user_exposed_df

