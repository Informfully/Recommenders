#############################################################################################################
# VERSION:          4.3 (STANDALONE)                                                                        #
# DESCRIPTION:      PLD - Political Diversity Model for Cornac                                              #
# DOCUMENTATION:    https://informfully.readthedocs.io/en/latest/participatory.html                         #
#############################################################################################################

from .article_recommender_diversity import Predict as diversity_predict
from .score_calculator import calculateArticleScore, calculatePoliticalScore
from ..recommender import Recommender
from ...exception import ScoreException
from cornac.utils import common

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import configparser
import sys


class PLD(Recommender):

    """Political diversity algorithm

    Parameters
    ----------
    name: string, default: 'PLD'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already is pre-trained.
        (U and V are not 'None').
    
    verbose: boolean, optional, default: False
        When True, running logs are displayed.
    
    num_users: int, default: 0
        The number of users in dataset.
    
    num_items: int, default: 0
        The number of items in dataset.

    party_dict: dict
        A dictionary whose keys are article ids and values are references of this articles.

    distribution: Nested Lists
        Every elment in outer layer is a list which includes user group type and a list of the article distribution for this user type. 
        Example is as the '{project_path}/examples/pld_mind.py'.

    update_score: boolean, default: True
        When 'False', use existed score files under the folder './cornac/models/pld'.

    configure_path: str, default: './parameters.ini'.
        Configure file which includes parties to be calculated.
    
    """

    def __init__(
        self,
        num_users,
        num_items,
        party_dict, 
        distribution,
        configure_path,
        user_score_path,
        item_score_path, 
        group_granularity = 0.2, # for article group granularity
        update_score = True,
        name="PLD",
        trainable=True,
        verbose=False,
        **kwargs):

        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose, **kwargs)
        
        
        self.party_dict = self._normalize_party_dict(party_dict)
        self.articles = list(party_dict.keys())

        # check the format of the distribution, make sure every user type has same article types, which means for every row there are same columns at the second element.
        articlesTypesNum = len(distribution[0][1])
        for i in range(len(distribution)):
            if len(distribution[i][1]) != articlesTypesNum:
                print("Init failed: different articles type {}.len: {} != {}".format(i, len(distribution[i][1]), articlesTypesNum))
                return None

        self.distribution = distribution
        self.group_granularity = group_granularity # for article
        self.user_group_granularity = np.abs(distribution[0][0][0] - distribution[1][0][0])
        self.num_users = num_users
        self.num_items = num_items
        self.update_score = update_score
        self.configure_path = configure_path
        self.user_score_path = user_score_path
        self.item_score_path = item_score_path

        self.group_recommendations_generated = False

    def _find_config_section(self, config, model_name):
        """Find configuration section case-insensitively."""
        model_name_lower = model_name.lower()
        
        # Look for exact match (case-insensitive)
        for section_name in config.sections():
            if section_name.lower() == model_name_lower:
                return section_name
        
        return None

    def _get_config_value(self, section, primary_key, fallback_keys=None):
        """Get configuration value with case-insensitive key matching."""
        fallback_keys = fallback_keys or []
        all_keys = [primary_key] + fallback_keys
        
        # Try each key (case-insensitive)
        for key in all_keys:
            # Try exact key first
            if key in section:
                return section[key].strip()
            
            # Try case-insensitive match
            for actual_key in section.keys():
                if actual_key.lower() == key.lower():
                    return section[actual_key].strip()
        
        # If not found, raise error with helpful message
        available_keys = list(section.keys())
        raise ValueError(
            f"Required configuration key not found. Tried: {all_keys}\n"
            f"Available keys: {available_keys}"
        )
    
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

        """

        Recommender.fit(self, train_set)

        existUserScore = os.path.isfile(self.user_score_path)
        existArticleScore = os.path.isfile(self.item_score_path)
        existScores = existUserScore and existArticleScore

        if self.update_score or not existScores:

            config = configparser.ConfigParser()
            config.read(self.configure_path)

            section_name = self._find_config_section(config, self.name)

            if section_name:
                raw_parties =  self._get_config_value(
                    config[section_name], 
                    'parties',
                    ['party_list', 'party_names', 'political_parties']
                )
                self.party_list = [party.strip() for party in raw_parties.split(",") if party.strip()]
                
                # Case-insensitive key lookup with fallbacks
                self.positive_score_party = self._get_config_value(
                    config[section_name], 
                    'positive_score_party_name',
                    ['positive_party', 'pos_party']
                )
                self.negative_score_party = self._get_config_value(
                    config[section_name],
                    'negative_score_party_name', 
                    ['negative_party', 'neg_party']
                )
                
                if self.verbose:
                    print(f"Using configuration section: [{section_name}]")
                    print(f"Loaded parties: {self.party_list}")
                    
            else:
                available_sections = list(config.sections())
                raise ValueError(
                    f"Configuration Error: No section found for model '{self.name}'.\n"
                    f"Available sections: {available_sections}.\n"
                )

            # section_name = self.name


            # if section_name in config:
            #     raw_parties = config[section_name].get('parties', '')
            #     self.party_list = raw_parties.split(",")
            #     # print(f"self.party_list: {self.party_list}")
            # else:
            #     raise ValueError(
            #         f"Configuration Error: Section '{section_name}' not found in '{self.configure_path}'.\n"
            #         f"Please check your configuration file and ensure the section [{section_name}] exists."
            #     )

            # self.positive_score_party = config[section_name]['positive_score_party_name']
            # self.negative_score_party = config[section_name]['negative_score_party_name']

            train_uir = list(zip(*train_set.uir_tuple))

            self.history_dict = common.build_history(train_uir)

            # calculate user scores
            self.userScores = calculatePoliticalScore(self.history_dict, self.party_dict, self.party_list, self.num_users)

        else:

            df = pd.read_csv(self.user_score_path)
            df = df.iloc[:, 1:]
            self.userScores = df.to_numpy()

            df = pd.read_csv(self.item_score_path)
            df = df.iloc[:, 1:]
            self.articleScores = df.to_numpy()
  
        return self

    def _normalize_party_dict(self, party_dict):
        """Normalize party dictionary for case-insensitive lookups."""
        if not isinstance(party_dict, dict):
            raise ValueError("party_dict must be a dictionary")
        
        normalized_dict = {}
        for key, value in party_dict.items():
            # Convert key to lowercase for consistent lookup
            normalized_key = str(key).lower()
            normalized_dict[normalized_key] = value
        return normalized_dict
    
    def rank(self, user_idx, item_indices = None, k = -1, **kwargs):
        if not self.group_recommendations_generated:
            self.generate_group_recommendation(item_indices = item_indices, **kwargs )
            self.group_recommendations_generated = True
        
        predictions = self.group_prediction_dict[tuple(self.userScores[user_idx])]
        prediction_idx = [self.article_pool_idx[item] for item in predictions]
        
        scores = np.zeros(len( self.article_pool_idx))
        max_score = len(predictions)

        for item in predictions:
            scores[item] = max_score  
            max_score -= 1
        
        scores = common.roundRowScore(scores, 1, 0)
        
        return prediction_idx, scores


    def generate_group_recommendation(self, item_indices = None, **kwargs):
        existUserScore = os.path.isfile(self.user_score_path)
        existArticleScore = os.path.isfile(self.item_score_path)
        existScores = existUserScore and existArticleScore
        impression_items_list = [] 
        
        if self.article_pool is not None:

            item_idx2id = {v: k for k, v in self.iid_map.items()} # cornac item ID : raw item ID
            user_idx2id = {v: k for k, v in self.uid_map.items()} # cornac user ID : raw user ID
            item_id2idx = {k: v for k, v in self.iid_map.items()} # raw item ID : cornac item ID

            assert isinstance(item_idx2id, dict), "item_idx2id must be a dictionary"
            assert isinstance(user_idx2id, dict), "user_idx2id must be a dictionary"
            assert isinstance(item_id2idx, dict), "item_id2idx must be a dictionary"
            
            for iid in self.article_pool:
                if iid in item_id2idx:
                    idx = item_id2idx[iid]
                    impression_items_list.append(idx)

        elif item_indices is None:
            impression_items_list = list(np.arange(self.total_items))
    
        else:
            impression_items_list = list(item_indices)

        self.article_pool_idx = impression_items_list

        if self.update_score or not existArticleScore:

            # calculate article scores
            self.articleScores = calculateArticleScore(
                self.history_dict,
                self.userScores,
                self.num_users,
                self.num_items,
                self.party_dict,
                self.party_list,
                self.article_pool_idx,
                self.positive_score_party,
                self.negative_score_party)

            for i in range(len(self.articleScores)):
                self.articleScores[i] = roundScore4Predict(self.articleScores[i], self.distribution, self.group_granularity)
       
            for i in range(len(self.userScores)):
                self.userScores[i] = roundScore4Predict(self.userScores[i], self.distribution,  self.user_group_granularity)

        # write score into a csv
        df = pd.DataFrame(self.articleScores, columns=[f'Score {i+1}' for i in range(len(self.articleScores[0]))])
        if existArticleScore:
            os.remove(self.item_score_path)
        df.insert(0, 'Article ID', range(len(self.articleScores)))
        df.to_csv(self.item_score_path, index=False)

        # write score into a csv
        df = pd.DataFrame(self.userScores, columns=[f'Score {i+1}' for i in range(len(self.userScores[0]))])
        if existUserScore:
            os.remove(self.user_score_path)
        df.insert(0, 'User ID', range(len(self.userScores)))
        df.to_csv(self.user_score_path, index=False)

        userScoreRange = [row[0][0] for row in self.distribution]
        # print(f"userScoreRange:{userScoreRange}")
        rowDistribution = []
        for i in range(len(self.userScores[0])):
            rowDistribution.append(userScoreRange)
        # print(f"rowDistribution:{rowDistribution}")

        userGroups = list(itertools.product(*rowDistribution))
        # print(f"userGroups:{userGroups}")

        self.group_prediction_dict = {}
        
        for userGroup in tqdm(userGroups, total=len(userGroups), desc="Computing results for every user group: "):
            predictions = diversity_predict(np.array(userGroup), self.articleScores, self.distribution, self.group_granularity)
            self.group_prediction_dict[tuple(userGroup)] = predictions
        
        # print(f"self.group_prediction_dict = {self.group_prediction_dict}")


# Function to round either user or article score to the level of granularity of score groups
def roundScore4Predict(score, distribution, group_granularity):

    # Rounding is always done with the available group scores in mind
    for i in range(len(score)): 

        for group in range(0, len(distribution)):
            
            if (abs(distribution[group][0][0] - score[i]) <= 0.5 * group_granularity):
                score[i] = distribution[group][0][0]

    return score
