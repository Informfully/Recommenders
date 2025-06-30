#############################################################################################################
# VERSION:      1.0 (STANDALONE)                                                                            #
# DESCRIPTION:  EPD - Exposure Diversity Model for Cornac                                                   #
#############################################################################################################

from ..recommender import Recommender
from .epd_core import EPD_CORE as EPD_CORE
from ...exception import ScoreException
import numpy as np
from cornac.utils import common
import os
import json
import configparser


class EPD(Recommender):

    """Explosure Diversity Algorithm.

    Parameters
    ----------
    name: string, default: 'EPD'
        The name of the recommender model.
    
    articles_collection: list
        The list of whole articles, with article political reference.

    political_type_dict: dict
        The dictionary of political classes, keys are indices, values are political type of articles.

    num_items: int
        The number of items(here are articles) in one experiment.

    k: int, optional, default: 3
        The number of political and non-political articles each time added into recommendation collection.

    pageWidth: int, optional, default: 24
        The maximum number of articles added for each user group.
    
    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already pre-trained.
        (U and V are not 'None'.)
    
    verbose: boolean, optional, default: False
        When 'True', running logs are displayed.
    
    """

    def __init__(
        self,
        party_path,
        political_type_dict,
        num_items,
        configure_path,
        k = 3,
        pageWidth = 24,
        name = "EPD",
        trainable = True,
        verbose = False,
        userGroupDict = {},
        dataset_name = "mind",
        political_ref_path = "./",
        **kwargs):
        
        super().__init__( name=name, trainable=trainable, verbose=verbose, **kwargs)
      
        self.party_path =  party_path
        self.political_ref_path = political_ref_path

        self.k = k
        self.pageWidth = pageWidth
        self.political_type_dict = political_type_dict
        self.configure_path = configure_path
        self.num_items = num_items

        self.dataset_name = dataset_name

        # Assert that dataset_name is valid
        allowed_datasets = {"mind", "ebnerd", "nemig"}
        assert self.dataset_name.lower() in allowed_datasets, (
            f"Invalid dataset_name: {self.dataset_name}. "
            f"Must be one of {allowed_datasets}"
        )

        self.userGroupDict = userGroupDict

        self.recommendation_scores = {}
        self.recommendation_groups = {}

        self.article_collection = None
        self.article_collection_index = []
        self.recommendations_id_dict =  None
    
    def load_article_collection(self, political_ref_path, party_path, configure_path, iid_map):

        dataset_lower = self.dataset_name.lower()
  
        if dataset_lower == "mind":
            articles = self.article_political_epd(party_path, configure_path, iid_map)
        elif dataset_lower == "ebnerd" or dataset_lower == "nemig":
            articles = self.article_political_epd_ebnerd(party_path, configure_path, iid_map)

        with open(political_ref_path, 'w', encoding="utf-8") as json_file:
            json.dump(articles, json_file, indent=4)

        return articles


    def article_political_epd_ebnerd(self, ppath, configure_path, iid_map):
        config = configparser.ConfigParser()
        try:
            config.read(configure_path)
            section_name = self.name

            if section_name in config:
                print(f"Model name {section_name} found in the configuration file.")
            else:
                raise ValueError(
                    f"Configuration Error: Section '{section_name}' not found in '{self.configure_path}'.\n"
                    f"Please check your configuration file and ensure the section [{section_name}] exists."
                )

            majority = config[section_name]['majority'].replace('_', ' ').split(',')
            
        except (configparser.Error, KeyError) as e:
            raise configparser.Error(f"Error reading config file {configure_path}: {e}")

        # Ensure majority is properly parsed
        if not majority or majority == {''}:
            raise ValueError("No valid majority parties found in config file.")

        if not ppath.endswith('.json'):
            raise ValueError(f"Invalid file format: {ppath}. Expected a JSON file.")
        
        try:
            with open(ppath, encoding="utf-8") as json_file:
                party_dict = json.load(json_file)
                if not isinstance(party_dict, dict):
                    raise ValueError("Invalid JSON format: Expected a dictionary at the top level.")
        except (json.JSONDecodeError, ValueError, FileNotFoundError, IOError) as e:
            raise ValueError(f"Error loading JSON file {ppath}: {e}")
        
        article_list = []
        self.article_collection_index = []

        for article, political_references in party_dict.items():

            political_references_count = 0
            minority_count = 0

            article_prop = {}
            
            if article in list(iid_map.keys()):
                # article_prop["article_id"] = iid_map[article]
                self.article_collection_index.append(iid_map[article])
                article_prop["article_id"] = article
                article_prop["political_references"] = political_references

                political_references_count = sum(political_references.get(party, 0) for party in ["GOV_PARTIES", "OPP_PARTIES", "INDEP_FOREIGN_PARTIES"])
                article_prop["political_references_count"] = political_references_count

                minority_count = political_references.get("OPP_PARTIES", 0)
                article_prop["minority_count"] = minority_count
                
                parties_in_article = set(political_references.keys())
                if parties_in_article == {"GOV_PARTIES"}:
                    majority_count = political_references.get("GOV_PARTIES", 0)
                else:
                    majority_count = 0
                article_prop["majority_count"] = majority_count

                article_list.append(article_prop)
        
        return article_list
    
    def article_political_epd(self, ppath, configure_path, iid_map):
        config = configparser.ConfigParser()
        try:
            config.read(configure_path)
            section_name = self.name

            if section_name in config:
                print(f"Model name {section_name} found in the configuration file.")
            else:
                raise ValueError(
                    f"Configuration Error: Section '{section_name}' not found in '{self.configure_path}'.\n"
                    f"Please check your configuration file and ensure the section [{section_name}] exists."
                )

            majority = config[section_name]['majority'].replace('_', ' ').split(',')
        except (configparser.Error, KeyError) as e:
            raise configparser.Error(f"Error reading config file {configure_path}: {e}")

        # Ensure majority is properly parsed
        if not majority or majority == {''}:
            raise ValueError("No valid majority parties found in config file.")

        if not ppath.endswith('.json'):
            raise ValueError(f"Invalid file format: {ppath}. Expected a JSON file.")
        
        try:
            with open(ppath, encoding="utf-8") as json_file:
                party_dict = json.load(json_file)
                if not isinstance(party_dict, dict):
                    raise ValueError("Invalid JSON format: Expected a dictionary at the top level.")
        except (json.JSONDecodeError, ValueError, FileNotFoundError, IOError) as e:
            raise ValueError(f"Error loading JSON file {ppath}: {e}")
        
        article_list = []
        self.article_collection_index = []

        for article, political_references in party_dict.items():

            political_references_count = 0
            minority_count = 0

            article_prop = {}
            
            if article in list(iid_map.keys()):
                # article_prop["article_id"] = iid_map[article]
                self.article_collection_index.append(iid_map[article])
                article_prop["article_id"] = article
                article_prop["political_references"] = political_references

                for party, count in political_references.items():
                    political_references_count += count
                    if party not in majority:
                        minority_count += count
            
                article_prop["political_references_count"] = political_references_count
                article_prop["minority_count"] = minority_count
                article_list.append(article_prop)
        
        return article_list


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

        # train_uir = list(zip(*train_set.uir_tuple))
        # users_collection = common.load_user_group_type(train_uir)
        # self.userGroupDict = {user['userId']:user['userGroup'] for user in users_collection}
        # train_articles = np.array([i for u,i,r in train_uir])

        # articles_collection = [article for article in self.article_collection if article['article_id'] in train_articles]
        # articles_collection = [article for article in self.article_collection]
        # print(f"len articles_collection:{len(articles_collection)}")
        
        self.epd_core = EPD_CORE(self.k, self.pageWidth, name  = self.name)

        return self
    

    def rank(self, user_idx, item_indices=None, k=-1, **kwargs):

        # item_idx2id = kwargs.get("item_idx2id")
        # user_idx2id = kwargs.get("user_idx2id")
        # item_id2idx = kwargs.get("item_id2idx")
        item_idx2id = {v: k for k, v in self.iid_map.items()} # cornac item ID : raw item ID
        user_idx2id = {v: k for k, v in self.uid_map.items()} # cornac user ID : raw user ID
        item_id2idx = {k: v for k, v in self.iid_map.items()} # raw item ID : cornac item ID

        assert isinstance(item_idx2id, dict), "item_idx2id must be a dictionary"
        assert isinstance(user_idx2id, dict), "user_idx2id must be a dictionary"
        assert isinstance(item_id2idx, dict), "item_id2idx must be a dictionary"

        if self.article_collection is None: 

            article_collection = self.load_article_collection(
                political_ref_path = self.political_ref_path,
                party_path = self.party_path,
                configure_path = self.configure_path,
                iid_map = item_id2idx)
            if self.article_pool is None or len(self.article_pool)==0:
                self.article_collection = article_collection
            else:
                self.article_collection = []
                for id in self.article_pool:
                # Find the article with the matching article_id and append it to article_collection
                    for article in article_collection:
                        if article['article_id'] == id:
                            self.article_collection.append(article)
                            break  
                    
                # self.article_collection = [article for article in article_collection if article['article_id'] in article_pool]
                # print(f"len EPD.article_collection:{len(self.article_collection)}")
        
        articles_collection = self.article_collection
        if articles_collection is None or len(articles_collection) == 0:
            raise ValueError("articles_collection is None or empty!")

        group = self.userGroupDict[user_idx]
        if group in self.recommendation_scores and group in self.recommendation_groups:

            return self.recommendation_groups[group], self.recommendation_scores[group] # already computed for this group

        if self.recommendations_id_dict is None:
            self.recommendations_id_dict = self.epd_core.prepare_recommendations(
                articles_collection,
                self.political_type_dict, 
                self.configure_path,
                self.dataset_name)
            # print(f"self.recommendations_id_dict:{self.recommendations_id_dict}")
         
        recommendation_rawID_list = self.recommendations_id_dict[group]

        recommendation_list = []
        for iid in recommendation_rawID_list:
            if iid in item_id2idx:
                idx = item_id2idx[iid]
                recommendation_list.append(idx)
        
        if  self.article_pool is not None:
            scores = np.zeros(len(self.article_pool))

            max_score = len(recommendation_rawID_list )
            for item in recommendation_rawID_list :
                
                article_index = self.article_pool.index(item)  # Get the index in article_collection
                scores[article_index] = max_score  # Assign the score
                max_score -= 1
            
            impression_items_list = []
            for iid in self.article_pool:
                if iid in item_id2idx:
                    idx = item_id2idx[iid]
                    impression_items_list.append(idx)
            self.item_scores_mapped_indices[user_idx] = impression_items_list

        else:
            scores = np.zeros(len( self.article_collection_index))
            max_score = len(recommendation_list)
            for item in recommendation_list:
                article_index =  self.article_collection_index.index(item)  # Get the index in article_collection
                scores[article_index] = max_score  # Assign the score
                max_score -= 1
            self.item_scores_mapped_indices[user_idx] = self.article_collection_index
            
        scores = common.roundRowScore(scores, 1, 0)

        self.recommendation_scores[group] = scores
        self.recommendation_groups[group] = recommendation_list
        # print(f"group added: {group}")
        # print(f"group's recommendation:{recommendation_list}")
        # print(f"len scores:{len(scores)}")

        return recommendation_list, scores
