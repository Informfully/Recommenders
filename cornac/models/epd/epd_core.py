#############################################################################################################
# VERSION:      1.0 (STANDALONE)                                                                            #
# DESCRIPTION:  EPD - Exposure Diversity Model for Cornac                                                   #
#############################################################################################################

from operator import itemgetter as i
from functools import cmp_to_key
import random
import configparser
import copy

class EPD_CORE():

    """  
    Parameters
    ----------
    
    k: int, optional
        The number of political and non-political articles each time added into recommendation collection

    pageWidth: int, optional
        The maximum number of articles added for each user group

    """

    def __init__(self, k, pageWidth, name = "EPD"):
        self.k = k
        self.pageWidth = pageWidth
        self.name = name


    def prepare_recommendations(self, articles_collection, political_type_dict, configure_path, dataset_name):

        non_political_articles = self.load_articles_in_list(articles_collection = articles_collection, type='non-political', dataset_name = dataset_name )

        config = configparser.ConfigParser()

        try:
            config.read(configure_path)
            if self.name not in config or "USERGROUPID" not in config[self.name]:
                raise KeyError("Missing model's section or 'USERGROUPID' key in config file.")

            # Convert user group IDs to integers safely
            try:
                user_group_id_list = [int(i.strip()) for i in config[self.name]["USERGROUPID"].split(",") if i.strip().isdigit()]
            except ValueError:
                raise ValueError(f"Invalid USERGROUPID values in {configure_path}. Expected integers separated by commas.")
        
        except (configparser.Error, KeyError, ValueError) as e:
            raise configparser.Error(f"Error reading config file {configure_path}: {e}")

        # config.read(configure_path)
        # user_group_id_list = [int(i) for i in config['EPD']["USERGROUPID"].split(',')]
        recommendations_collection_dict = {}

        # for i in list(political_type_dict.keys()):
        for i in political_type_dict.keys():
            recommendations_collection = []
            political_articles = self.load_articles_in_list(articles_collection = articles_collection, type='political', political=political_type_dict[i], dataset_name = dataset_name)
            recommendation_temp = self.create_recommendations(i, political_articles, non_political_articles)
            seen_article_ids = set()
            deduplicated_recommendations = []
            for recommendation in recommendation_temp:
                article_id = recommendation['article_id']
                if article_id not in seen_article_ids:
                    deduplicated_recommendations.append(recommendation)
                    seen_article_ids.add(article_id)  # Add the article_id to the set to mark it as see
            for recommendation in deduplicated_recommendations:
  
                recommendations_collection.append(recommendation)
            recommendations_collection_dict[i] = recommendations_collection
        user_recommendation_id_dict = {}

        for id in user_group_id_list:
            recommendations_collection = recommendations_collection_dict[id]
            user_recommendation_id_dict[id] = self.generate_user_recommendation_list(recommendations_collection, id, self.pageWidth)

        return user_recommendation_id_dict
 

    def load_articles_in_list(self, articles_collection, type, political='neutral', dataset_name = "mind"):

        articles = [] 
        dataset_lower = dataset_name.lower()

        if type == 'political':  
            if  dataset_lower == "mind":     
                for article in articles_collection:
                    if political == 'neutral':
                        if article['political_references_count'] > 0:
                            articles.append(article)
                    elif political == 'minor':
                        if article['minority_count'] > 0:
                            articles.append(article)
                    elif political == 'major':
                        if article['political_references_count'] > 0 and article['minority_count'] == 0:
                            articles.append(article)
                # # other political type
                # elif political == 'OTHER TYPE TO BE DEFINED':
                #   CUSTOMIZE YOUR CLASSIFICATION METHOD   
            elif dataset_lower == "ebnerd" or dataset_lower == "nemig":
                for article in articles_collection:
                    if political == 'neutral':
                        if article['political_references_count'] > 0:
                            articles.append(article)
                    elif political == 'minor':
                        if article['minority_count'] > 0:
                            articles.append(article)
                    elif political == 'major':
                        if article['political_references_count'] > 0 and article['majority_count'] > 0 :
                            articles.append(article)    

        elif type == 'non-political':
            for article in articles_collection:
                if article['political_references_count'] == 0:
                    articles.append(article)   

        # print(f"type:{type}, political:{political}, article len:{len(articles)}")
        return articles


    def create_recommendations(self, group, political_articles, non_political_articles):
    
        recommendations_count = 0
        recommendations_collection = []

        # Copy non political articles into another list and use the new list in subsequent steps
        _non_political_articles = non_political_articles[:]

        while (len(political_articles) + len(_non_political_articles) > 0):

            # First get the articles where local candidate is referenced
            random.shuffle(political_articles) #shuffling because not all objects arrived at the same time from the data sources
            random.shuffle(_non_political_articles)
        
            # Look at political articles + creates slices of political articles (here of size 3) if there are political articles left
            for _ in range(self.k):
                # 3 political articles
                if (len(political_articles) > 0):
                    political_article = copy.deepcopy(political_articles[0])
                    political_article['group'] = group
                    political_article['is_political'] = True
                # Insert political article in the recommendations...
                    recommendations_collection.append(political_article)
                # ...then remove this article from the list
                    political_articles.pop(0)
                    recommendations_count += 1
        
            # After a slice of political articles we add a slice of non-political
            for _ in range(self.k):
                # 3 non political articles
                if (len(_non_political_articles) > 0):
                    non_political_article = copy.deepcopy(_non_political_articles[0])
                    non_political_article['group'] = group
                    non_political_article['is_political'] = False
                # Insert political article in the recommendations...
                    recommendations_collection.append(non_political_article)
                # ...then remove this article from the list
                    _non_political_articles.pop(0)
                    recommendations_count += 1

        return recommendations_collection


    def generate_user_recommendation_list(self, recommendations_collection, user_group, pageWidth):
        recommendation_lists = []
        processed_article_ids = set()

        # Collect all recommendations that match the user group
        recommendations_cursor = [recom for recom in recommendations_collection if recom['group'] == user_group]

        # Collect all unique article IDs
        processed_article_ids = set()

        for recommendation in recommendations_cursor:
            article_id = recommendation['article_id']
            
            # Only add unique article_id
            if article_id not in processed_article_ids:
                recommendation_lists.append(article_id)
                processed_article_ids.add(article_id)

        return recommendation_lists
