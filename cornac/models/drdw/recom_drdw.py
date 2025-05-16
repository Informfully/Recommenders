from ..recommender import Recommender
from ...exception import ScoreException
import numpy as np
from .sample_and_rank import Sample_And_Rank
import configparser
import json
import ast
import re


class D_RDW(Recommender):

    """Diversity-driven Random walk model.
    A diversity-driven recommender model leveraging random walks on a user-item bipartite graph to generate diverse and personalized recommendation lists.
    The model uses specified target distributions and attributes to ensure diversity across multiple dimensions, such as categories, sentiment, or other user-defined features.
    """

    def __init__(self,
                 item_dataframe,
                 diversity_dimension=None,
                 target_distributions=None,
                 targetSize=24,
                 maxHops=15,
                 filteringCriteria=None,
                 rankingType="rdw_score",
                 rankingObjectives=None,
                 mappingList=None,
                 ascending=None,
                 sampleObjective="rdw_score",
                 name="D_RDW",
                 config_file=None,
                 trainable=True,
                 verbose=False,
                 **kwargs
                 ):
        
        """
        Initializes the D_RDW class.

        Parameters
        ----------
        name : str, default: 'D_RDW'
            The name of the recommender model.

        item_dataframe : pandas.DataFrame
            A DataFrame containing article attributes.
            Rows: Represent articles, indexed by Cornac IDs.
            Columns: Represent attributes (e.g., categories).

        diversity_dimension : list
            A list of feature dimensions the model will diversify.
            Example: ["category", "sentiment"].    
            Each entry in this list corresponds to a feature present in the `item_dataframe`.


        target_distributions : dict
            Defines target distributions.
            Example:
            {"category":{"type":"discrete","distr":{"weather":0.2,"news":0.2,"finance":0.2,"entertainment":0.2,"sport":0.2}},
        "sentiment":{"type":"continuous","distr":[{"min":-1,"max":-0.5,"prob":0.1},{"min":-0.5,"max":0,"prob":0.2},{"min":0,"max":0.5,"prob":0.3},{"min":0.5,"max":1,"prob":0.4}]},
        "entities":{"type":"parties","distr":[{"description":"only mention","contain":["Republican Party"],"prob":0.2},{"description":"only mention","contain":["Democratic Party"],"prob":0.2},
        {"description":"composition","contain":[
            ["Republican Party"],["Democratic Party"]],"prob":0.4},
        {"description":"minority but can also mention","contain":["Republican Party","Democratic Party"],"prob":0.2}]}}

        targetSize : int
            The length of the recommendation list.

        maxHops : int
            The maximum number of hops allowed in the random walk process.
            It must be greater than or equal to 3.

        filteringCriteria : dict, optional
            Criteria for filtering items before candidate generation. Default is None.
            Example:
            {
                "filterDim": "age",
                "filterThreshold": 24,
                "comparison": "less"
            }

        sampleObjective : str
            The feature dimension to minimize during sampling. Defaults to "rdw_score"(In this special case, maxmizing random walk score).

        rankingType : str
            Defines the ranking method.
            - 'rdw_score': Default, ranks by random walk scores.
            - 'multi_objectives': Ranks based on multiple objectives.
            - 'graph_coloring': Ensures diverse attribute distribution within the recommendation list.

        rankingObjectives : list, optional
            Specifies ranking priorities for `multi_objectives` or `graph_coloring`.
            Example: ["popularity", "sentiment"].

        mappingList : list of dict, optional. 
        When rankingType = `multi_objectives`, users may need to provide a `mappingList` to map categorical values to numerical scores.
            Example: [{"outlet": {"CNN": 1, "BBC": 2}}].

        ascending : list of boolean, optional
            Specifies the sorting order for ranking features. Default is None.
            Each element in the ascending list is a boolean value: True indicates that the feature should be sorted in ascending order,
        while False indicates descending order.

        config_file: configuration file path. Optional.

        trainable : bool, default: True
            Indicates whether the model is trainable. This is a variable that the base `Recommender` class has.

        """
        super().__init__(name, trainable, verbose,  **kwargs)
        self.item_dataframe = item_dataframe
        self.diversity_dimension = diversity_dimension
        self.targetSize = targetSize

        if maxHops < 3:
            raise ValueError(
                f"maxHops must be greater than or equal to 3, but got {maxHops}.")
        self.maxHops = maxHops
        self.targetDistribution = target_distributions
        self.rankingType = rankingType
        self.rankingObjectives = rankingObjectives
        self.mappingList = mappingList
        self.ascending = ascending
        self.sampleObjective = sampleObjective
        self.filteringCriteria = filteringCriteria
        self.configFile = config_file

        if config_file:
            self.readConfigFile(config_file)

    def readConfigFile(self, configFile):
        """
        Reads a configuration file and populates the model attributes, such as filteringCriteria.
        """
        config = configparser.ConfigParser()
        config.read(configFile)

        def get_json_config(name, key, default=None):
            """Helper function to safely load JSON data from the config."""
            try:
                if config.has_option(name, key):
                    return json.loads(config[name][key])
            except (ValueError, json.JSONDecodeError):
                pass
            return default

        def get_string_list_config(name, key, default=None):
            """Helper function to load string list from the config."""
            try:
                if config.has_option(name, key):
                    string_list = json.loads(config[name][key])
                    return [s.lower() == 'true' for s in string_list]
            except (ValueError, json.JSONDecodeError):
                pass
            return default
        section = self.name
        # Load configuration options using helper functions
        self.filteringCriteria = get_json_config(
            section, "filteringCriteria", self.filteringCriteria)
        self.diversity_dimension = get_json_config(
            section, "diversity_dimension", self.diversity_dimension)
        self.targetDistribution = get_json_config(
            section, "target_distributions", self.targetDistribution)
        self.targetSize = get_json_config(
            section, "targetSize", self.targetSize)
        self.maxHops = get_json_config(section, "maxHops", self.maxHops)
        self.rankingType = get_json_config(
            section, "rankingType", self.rankingType)
        self.rankingObjectives = get_json_config(
            section, "rankingObjectives", self.rankingObjectives)

        # Handle mapping list with special parsing (regex for nested dicts)
        if config.has_option(section, "mappingList"):
            try:
                input_str = config[section]["mappingList"]
                pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
                dict_strs = re.findall(pattern, input_str)
                self.mappingList = [ast.literal_eval(d) for d in dict_strs]
            except (ValueError, SyntaxError):
                self.mappingList = None

        # Load ascending as a list of booleans
        self.ascending = get_string_list_config(
            section, "ascending", self.ascending)

        self.sampleObjective = get_json_config(
            section, "sampleObjective", self.sampleObjective)

    def fit(self, train_set, val_set=None):
        """
          Prepares the user-item interaction data and prepare the bipartite graph for the ranking process.

          Parameters:
          ----------
            -train_set: Training data for user-item interactions.
            -val_set (optional): Validation set for model evaluation.


       """
        def interacted_items(csr_row):
            return [
                item_idx
                for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                if rating > 0
            ]

        Recommender.fit(self, train_set)
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

        self.sampleRank = Sample_And_Rank(
            self.train_set_rating, self.item_dataframe)
        return self

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

        Notes
        -----
        This method determines a ranked recommendation list of size `targetSize` for a user.
        """

        if self.article_pool is not None:

            # item_idx2id = kwargs.get("item_idx2id")
            # user_idx2id = kwargs.get("user_idx2id")
            # item_id2idx = kwargs.get("item_id2idx")
            item_idx2id = {v: k for k, v in self.iid_map.items()} # cornac item ID : raw item ID
            user_idx2id = {v: k for k, v in self.uid_map.items()} # cornac user ID : raw user ID
            item_id2idx = {k: v for k, v in self.iid_map.items()} # raw item ID : cornac item ID



            assert isinstance(item_idx2id, dict), "item_idx2id must be a dictionary"
            assert isinstance(user_idx2id, dict), "user_idx2id must be a dictionary"
            assert isinstance(item_id2idx, dict), "item_id2idx must be a dictionary"

            impression_items_list = []
            for iid in self.article_pool:
                if iid in item_id2idx:
                    idx = item_id2idx[iid]
                    impression_items_list.append(idx)


            ranked_items, item_scores = self.rank_partial( user_idx=user_idx, item_indices = impression_items_list, item_idx2id = item_idx2id, user_idx2id = user_idx2id)

            self.ranked_items[user_idx] = ranked_items
            self.item_scores[user_idx] = item_scores
            # save item_score's corresponding item indices
            self.item_scores_mapped_indices[user_idx] = impression_items_list
          

            return ranked_items, item_scores

            
        if  self.is_unknown_user(user_idx):
        # if user_idx >=  self.train_set.csr_matrix.shape[0]:
        # if self.train_set.is_unk_user(user_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d)" % user_idx
            )
        item_indices = (
        list(np.arange(self.num_items))
        if item_indices is None
        else list(item_indices)
        )

        # if item_indices is None:
        selectedTarget = []
        for i in self.diversity_dimension:
            selectedTarget.append(self.targetDistribution[i])

        ranked_items, ranked_item_scores = self.sampleRank.performSampling(
            user_idx, self.targetSize,  self.diversity_dimension, selectedTarget, self.maxHops,  self.filteringCriteria, self.sampleObjective, self.rankingType,  self.rankingObjectives, self.mappingList, self.ascending, given_item_pool=item_indices)

        random_walk_prob = self.sampleRank.articleRdwScore[item_indices]
        self.ranked_items[user_idx] = ranked_items
        self.item_scores[user_idx] = random_walk_prob

        self.item_scores_mapped_indices[user_idx] = item_indices

        return ranked_items, random_walk_prob

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
            The scores corresponding to the `ranked_items`.
        """

        if self.is_unknown_user(user_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d)" % user_idx
            )

        selectedTarget = []
        for i in self.diversity_dimension:
            selectedTarget.append(self.targetDistribution[i])

        ranked_items, ranked_item_scores = self.sampleRank.performSampling(
            user_idx, self.targetSize,  self.diversity_dimension, selectedTarget, self.maxHops,  self.filteringCriteria, self.sampleObjective, self.rankingType,  self.rankingObjectives, self.mappingList, self.ascending, given_item_pool=item_indices)


        random_walk_prob = self.sampleRank.articleRdwScore[item_indices]
        return ranked_items, random_walk_prob
    
