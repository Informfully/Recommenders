# Pipeline Overview
# Step 1: Perform n-hop random walks to identify candidate items based on user-item graph traversal.
# Step 2: Apply sampling strategies to select items meeting specific distribution and dimensional criteria.
# Step 3: Rank the sampled items using various ranking methods, including RDW(random walk) score, multi-objective ranking, and graph coloring.
from .graph_recommender import GraphRec
import numpy as np
import random
from .rank_core import ItemRanker
import pandas as pd
from scipy.sparse import csr_matrix
from .sample_core import DistributionSampler
import time


class Sample_And_Rank(object):

    """ A class to perform recommendation pipeline comprising sampling and ranking.
    """

    def __init__(self, train_set_rating, articlesDataframe):
        """
        Initializes the pipeline with a user-item interaction matrix and article attributes.

        Parameters:
        -----------
        train_set_rating: numpy.ndarray. A user-item matrix where:
                Rows (Users): Each row corresponds to a unique user.
                Columns (Items): Each column corresponds to a unique item.

        articlesDataframe: a pandas DataFrame designed to store and manage the features of various articles.
            Rows: Each row represents an article, indexed by its Cornac ID.
            Columns: Each column represents a specific feature of the articles, e.g.,'category'.

        """
        self.itemPool = np.array([])
        self.Model_RDW = GraphRec(train_set_rating)
        self.articlesDataframe = articlesDataframe
        self.articleRdwScore = np.array([])
        self.train_set_rating = train_set_rating
        # Total number of articles
        self.articleNum = train_set_rating.shape[1]
        self.CANDIDATESOLD = []

    def filterHeuristics(self, user_idx, itemPool, filteringCriteria, given_item_pool=None):
        """Filters items based on specified criteria to refine the candidate pool.

        Parameters
        ----------
        user_idx : int
                The user ID for whom recommendations are being generated.

        itemPool: A list of item indices. This list contains the indices of the items that are available for recommendation.

        filteringCriteria: A dictionary containing the filtering criteria that filter items before the candidate generation step.
                By default, filteringCriteria is None, meaning that if users do not specify any filtering criteria, the filtering step will be skipped.
                filterDim:  Specifies the item attribute to be used for filtering (e.g., "articleAge", "rating").
                filterThreshold: Specifies the threshold value for the specified dimension.
                comparison: Comparison type ('larger', 'less', etc.).
        given_item_pool : list or numpy.ndarray, optional
                A predefined pool of items to restrict filtering.

        Returns:
        --------
        numpy.ndarray
            A refined pool of item indices after applying the filtering criteria.
        """
        if itemPool is None or len(itemPool) == 0:
            print("itemPool is None or empty, skipping filtering.")
            return np.array([])  # Return an empty numpy array
        filteredItems = np.asarray(itemPool)

       # Apply heuristics to filter out unwanted items to speed up the sampling process. Opions:
       # 1)Based on attributes. E.g., getting rid of articles older than X hours.
        if filteringCriteria is not None:
            filterDim = filteringCriteria['filterDim']
            threshold = filteringCriteria['filterThreshold']
            comparison = filteringCriteria['comparison']
            filtered_rows = self.articlesDataframe.loc[itemPool]
            if filterDim in self.articlesDataframe.columns:
                filtered_df = filtered_rows
                if comparison == "larger":
                    # greater than
                    filtered_df = filtered_rows[filtered_rows[filterDim] > threshold]
                elif comparison == "larger_equal":
                    # Greater than or equal to
                    filtered_df = filtered_rows[filtered_rows[filterDim]
                                                >= threshold]
                elif comparison == "less":
                    # Less than
                    filtered_df = filtered_rows[filtered_rows[filterDim] < threshold]
                elif comparison == "less_equal":
                    # Less than or equal to
                    filtered_df = filtered_rows[filtered_rows[filterDim]
                                                <= threshold]
                elif comparison == "equal":
                    # Equal to
                    filtered_df = filtered_rows[filtered_rows[filterDim]
                                                == threshold]
                elif comparison == "not_equal":
                    # Not equal to
                    filtered_df = filtered_rows[filtered_rows[filterDim]
                                                != threshold]
                else:
                    raise ValueError(f"Unknown comparison type: {comparison}")

            filteredItems = filtered_df.index.to_numpy()

        # check if only perform sampling and ranking in a given item pool
        if given_item_pool is not None and len(given_item_pool) > 0:
            given_item_pool_set = set(given_item_pool)
            mask = np.isin(filteredItems, list(given_item_pool_set))
            # Ensure `mask` and `filteredItems` have the same shape before applying
            if mask.shape == filteredItems.shape:
                filteredItems = filteredItems[mask]
            else:

                # Raise a warning and return an empty array if there's a shape mismatch
                warning_message = (
                    f"Shape mismatch for user {user_idx}: "
                    f"filteredItems shape {filteredItems.shape}, mask shape {mask.shape}. "
                    "Skipping filtering for this user."
                )
                # Log the warning (could use logging module in production)
                print(warning_message)
                return np.array([])

        #  2) news articles already read by the user,
        historyArticles = np.where(self.train_set_rating[user_idx] == 1)[0]
        filteredItems = np.setdiff1d(filteredItems, historyArticles)

        # 3) ...
        return (filteredItems)

    def sampleArticles(self, targetDimensions, targetDistributions, targetSize, linear_program_coefficient):
        """
            Samples items that meet specific target distributions across multiple dimensions.

            Parameters:
            -----------
            targetDimensions : list
                List of dimensions or attributes (e.g., 'category', 'popularity') to use for sampling.

            targetDistributions : list of dict
                Desired distributions for each dimension.

            targetSize : int
                Number of items to sample.

            linear_program_coefficient : str or None
                Attribute to use for optimizing the sampling process (e.g., minimize 'articleAge').

            Returns:
            --------
            tuple:
                - target_num_items (dict): Number of items sampled for each target dimension.
                - candidateItems (list): List of sampled item indices.
        """
        candidateItems = []
        if len(self.itemPool) == 0:
            return {}, []
        # If linear_program_coefficient values corresponds to a column in self.articlesDataframe.
        if linear_program_coefficient is not None and linear_program_coefficient in self.articlesDataframe.columns and not self.articlesDataframe[linear_program_coefficient].isna().any():
            C = np.ones(self.itemPool.shape[0])
            subset = self.articlesDataframe.loc[self.itemPool,
                                                linear_program_coefficient].values
            numeric_subset = pd.to_numeric(subset, errors='coerce')

            # Check if all values are numeric (i.e., no NaN after conversion)
            all_numeric = not np.isnan(numeric_subset).any()
            # C must be valid numerical values.
            if all_numeric:
                C = subset

        # If using random walk score as the linear_program_coefficient
        elif linear_program_coefficient == "rdw_score":
            C = np.asarray(self.articleRdwScore[self.itemPool])
            # because the scipy.optimize.linprog minimizes C, we want to maximize random walk score.
            C = C * -1

        else:
            C = np.ones(self.itemPool.shape[0])

        sampler = DistributionSampler(self.articlesDataframe)

        target_num_items, candidateItems = sampler.sample_by_multi_distributions(
            self.itemPool, targetDimensions, targetDistributions, targetSize, C)
        return (target_num_items, candidateItems)

    def rankArticles(self, candidateItems, targetSize, rankingType, rankingObjectives=None, mappingList=None, ascending=None):
        """
        Ranks sampled items based on specified objectives and ranking type.

        Parameters:
        -----------
        candidateItems : list
            List of item indices to be ranked.

        targetSize : int
            Number of items to include in the final ranked list.

        rankingType : str
            Type of ranking to apply (e.g., 'rdw_score', 'multi_objectives', 'graph_coloring').

        rankingObjectives : list of str, optional
            List of objectives for multi-objective ranking (e.g., ['popularity', 'sentiment']).

        mappingList : list of dict, optional
            Mapping rules for categorical features (e.g., {'outlet': {'CNN': 1, 'BBC': 2}}).

        ascending : list of bool, optional
            Specifies sort order for each ranking objective.

        Returns:
        --------
        tuple:
            - rankedArticles (list): List of ranked item indices.
            - scores (numpy.ndarray): Corresponding scores for ranked items.
        """

        rankedArticles = []

        if rankingType == "rdw_score":
            rdwScore = self.articleRdwScore[candidateItems]
            # Sort descending and take top targetSize
            indices = np.argsort(rdwScore)[::-1][:targetSize]
            rankedArticles = candidateItems[indices].tolist()
            scores = rdwScore[indices]

        elif rankingType == "multi_objectives":
         # Check that rankingObjectives is a list and every element is a valid string column in the DataFrame
            if not isinstance(rankingObjectives, list):
                raise ValueError(
                    "rankingObjectives must be a list of attribute names.")

            for obj in rankingObjectives:
                if not isinstance(obj, str):
                    raise ValueError(
                        f"Each objective in rankingObjectives must be a string. Found: {type(obj)}")
                if obj not in self.articlesDataframe.columns:
                    raise ValueError(
                        f"Objective '{obj}' is not a valid column in the articlesDataframe.")

            # Extract relevant data for candidateItems and rankingObjectives
            candidate_data = self.articlesDataframe.loc[candidateItems,
                                                        rankingObjectives]

            # Apply mappingList if provided
            if mappingList is not None:
                if not isinstance(mappingList, list):
                    raise ValueError(
                        "mappingList must be a list of dictionaries.")

                for mappingDict in mappingList:
                    if mappingDict:
                        # Apply mapping to corresponding column
                        for key, value_map in mappingDict.items():
                            if key in candidate_data.columns:
                                candidate_data[key] = candidate_data[key].map(
                                    value_map)
                            else:
                                raise ValueError(
                                    f"Key '{key}' in mappingList is not a valid column in the articlesDataframe.")

            # Sort by the rankingObjectives with the provided ascending order
            df_sorted = candidate_data.sort_values(
                by=rankingObjectives, ascending=ascending)

            # Keep only the top targetSize articles
            rankedArticles = df_sorted.index[:targetSize].tolist()
            scores = self.articleRdwScore[rankedArticles]

        elif rankingType == "graph_coloring":

            # Ranking based on graph coloring
            if isinstance(rankingObjectives, list) and len(rankingObjectives) > 0:
                if not isinstance(rankingObjectives[0], str) or rankingObjectives[0] not in self.articlesDataframe.columns:
                    raise ValueError(
                        "The graph_coloring need to take the first element in rankingObjectives to perform ranking. It must be a valid column name in the articlesDataframe.")
                ranking_dim = rankingObjectives[0]

            elif isinstance(rankingObjectives, str) and rankingObjectives in self.articlesDataframe.columns:
                ranking_dim = rankingObjectives

            else:
                raise ValueError(
                    "For graph_coloring ranking, rankingObjectives must be a valid column in articlesDataframe.")

            gc_solver = ItemRanker(
                candidateItems, self.articlesDataframe, ranking_dim)

            rankedArticles = gc_solver.rank()

            rankedArticles = rankedArticles[:targetSize]

            scores = self.articleRdwScore[rankedArticles]
        else:
            # Default fallback (return candidate items as-is)
            rankedArticles = candidateItems[:targetSize].tolist()
            scores = self.articleRdwScore[rankedArticles]

        return (rankedArticles, scores)

    def newHop(self, user_id, targetDimensions, targetDistributions, targetSize, sampleObjective, currentHop, filteringCriteria, given_item_pool=None):
        """
        Performs an n-hop random walk to generate a pool of candidate items.

        Parameters:
        -----------
        user_id : int
            The user ID for whom recommendations are being generated.

        targetDimensions : list
            Dimensions or attributes to guide sampling (e.g., 'category').

        targetDistributions : list of dict
            Desired distributions for each dimension.

        targetSize : int
            Number of items to include in the recommendation list.

        sampleObjective : str
            Objective for sampling (e.g., minimize 'articleAge').

        currentHop : int
            Current hop number in the random walk process.

        filteringCriteria : dict or None
            Criteria for filtering items before sampling.

        given_item_pool : list or numpy.ndarray, optional
            Predefined pool of items to restrict sampling.

        Returns:
        --------
        numpy.ndarray
            Pool of candidate item indices generated from the random walk.
        """
        candidateItems = []
        # check if this user didn't have any interacted items in the history
        isEmptyHistory = np.all(self.train_set_rating[user_id, :] == 0)
        if isEmptyHistory:
            # Randomly determing itemPool.
            tarSize = targetSize * random.randint(10, 20)

            poolSize = tarSize if tarSize <= self.articleNum else int(
                self.articleNum)
            self.itemPool = random.sample(
                range(0, self.articleNum), poolSize)
            # Adjusting the random values. Multiplying by (1 - 1e-6) ensures the values don't reach exactly 1 but stay just below it.
            self.articleRdwScore = np.round(np.random.random(
                size=self.articleNum) * (1 - 1e-6) + 1e-6, 3)
        # Perfom random walk logic to generate item pool
        else:
            prob = self.Model_RDW.performMultiHop(currentHop)
            # Convert prob to CSR format if not already, this might be redundant if always CSR is guaranteed
            if not isinstance(prob, csr_matrix):
                prob = csr_matrix(prob)

            start_col = self.train_set_rating.shape[0]

            recs = prob[user_id, start_col:]

            # flatten because it will be 2D array (1, num_items)
            recs_dense = recs.toarray().flatten()
            self.articleRdwScore = recs_dense
            self.itemPool = np.nonzero(recs_dense)[0]

        self.itemPool = self.filterHeuristics(
            user_id, self.itemPool, filteringCriteria, given_item_pool=given_item_pool)
        # Sample data that satisfy multiple distributions
        # Number of dimensions and distributions must be the same
        target_num_items, candidateItems = self.sampleArticles(
            targetDimensions, targetDistributions, targetSize, sampleObjective)

        return (candidateItems)

    def addRandomArticles(self, targetDimensions, targetDistributions, targetSize, sampleObjective, given_item_pool=None):
        """
        Adds random articles to fulfill the target size based on sampling.

        This function attempts to sample articles that meet the specified target dimensions and distributions,
        and fills the remaining number of articles by randomly selecting items from the available pool.
        It is useful when the sampling step cannot meet the full target size.

        Parameters:
        -----------
        targetDimensions : list
            A list of dimensions or attributes (e.g., 'category', 'sentiment') for sampling.

        targetDistributions : list of dict
            Desired distributions for sampling.

        targetSize : int
            The total number of items to be recommended or sampled.

        sampleObjective : str
            The objective for sampling, which determines the priority for selecting items based on features
            (e.g., minimizing or maximizing a specific feature like 'sentiment' or 'popularity').

        Returns:
        --------
        sampledItems : list
            A list of candidate items sampled according to the specified target dimensions and distributions.
            If the exact number of sampled items is insufficient, the remaining slots are filled with randomly
            selected articles to reach the target size.
        """

        sampledItems = []

        for j in range(targetSize-1, 0, -1):
            # reduce target size iteratively
            target_num_items, sampledItems = self.sampleArticles(
                targetDimensions, targetDistributions, j, sampleObjective)
            if len(sampledItems) == j:
                # Exit the loop if the desired number of (reduced target size) items are found
                break
        # Calculate the number of articles still needed to meet the target size
        num_articles_to_add = targetSize - len(sampledItems)
        # Check if given_item_pool
        if not isinstance(given_item_pool, (list, np.ndarray)) or len(given_item_pool) == 0:
            all_articles = range(0, self.articleNum)
        else:
            all_articles = list(given_item_pool)
        remaining_articles = list(set(all_articles) - set(sampledItems))
        # Randomly select additional articles to fill the remaining spots
        additional_articles = np.random.choice(
            remaining_articles, num_articles_to_add, replace=False).tolist()
        sampledItems.extend(additional_articles)
        return (sampledItems)

    def checkListParity(self, candidatesOld, candidatesNew):
        """
        Checks whether two candidate lists contain the same items.
        Parameters
            ----------
            candidatesOld : list
                Original list of candidate items.

            candidatesNew : list
                New list of candidate items.

            Returns
            -------
            bool
                True if the lists contain the same items, False otherwise.
        """

        if set(candidatesOld) == set(candidatesNew):
            return (True)
        else:
            return (False)

    def performSampling(self, user_id, listSize, targetDimensions, targetDistribution,   maxHops, filteringCriteria,  sampleObjective,  rankingType, rankingObjectives, mappingList, ascending, given_item_pool=None):
        """
        Executes the complete sampling and ranking pipeline.

        Parameters:
        -----------
        user_id: int
            The user ID for whom recommendations are generated.

        listSize: int
            Target size of the recommendation list.

        targetDimensions: list
            Dimensions or attributes to guide sampling.

        targetDistribution: list of dict
            Desired distributions for sampling.

        maxHops: int
            Maximum number of hops for random walks.

        filteringCriteria: dict or None
            Criteria for filtering items before sampling.

        sampleObjective: str
            Sampling objective to optimize.

        rankingType: str
            Type of ranking to apply(e.g., 'rdw_score', 'graph_coloring').

        rankingObjectives: list or None
            Objectives for multi-objective ranking.

        mappingList: list of dict, optional
            Mapping rules for categorical features.

        ascending: list of bool, optional
            Sort order for ranking objectives.

        given_item_pool: list or numpy.ndarray, optional
            Predefined pool of items for sampling and ranking.

        Returns:
        --------
        tuple:
            - candidateItems(list): List of ranked item indices.
            - scores(numpy.ndarray): Corresponding random walk scores for ranked items.
        """
        if listSize > self.articleNum:
            listSize = self.articleNum  # Adjust sample size if it exceeds the number of items
        candidateItems = []
        self.itemPool = np.array([])
        self.articleRdwScore = np.array([])
        self.CANDIDATESOLD = []
        initialHop = 3
        currentHop = initialHop
        terminateHop = maxHops

        # Generate candidate items until the list is full and/or there is no longer any change
        while (currentHop <= terminateHop):
            candidateItems = self.newHop(user_id,
                                         targetDimensions, targetDistribution, listSize, sampleObjective, currentHop, filteringCriteria, given_item_pool=given_item_pool)

            isIdentical = self.checkListParity(
                candidateItems, self.CANDIDATESOLD)

            # If the list is full, exit the lop
            if (len(candidateItems) >= listSize):
                break
            # If the old and new candidate items are the same, exit the loop
            elif (len(self.CANDIDATESOLD) > 0 and isIdentical == True):
                break

            currentHop = currentHop + 2

            self.CANDIDATESOLD = candidateItems

        # Special case: Unable to find candidate items of the required list size
        if len(candidateItems) == 0:
            candidateItems = self.addRandomArticles(
                targetDimensions, targetDistribution, listSize, sampleObjective, given_item_pool=given_item_pool)

        candidateItems, scores = self.rankArticles(
            candidateItems, listSize, rankingType, rankingObjectives, mappingList, ascending)

        return (candidateItems, scores)
