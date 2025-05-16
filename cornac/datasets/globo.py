# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
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
"""
This data is built based on the Globo datasets
provided by: https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom
"""
import pandas as pd
import json
import numpy as np
import configparser
import random


def load_feedback(fpath):
    """Load the user-item ratings, scale: 0,or 1

    Parameters
    ----------
    fpath: file path where the excel file containing user-item-rating information is stored.
    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).
    """
    if fpath.endswith('.csv'):
        df_globo = pd.read_csv(fpath)
    elif fpath.endswith('.parquet'):
        df_globo = pd.read_parquet(fpath)
    else:
        raise ValueError("Unsupported file type. Only CSV and Parquet are supported.")
    if len(df_globo.columns) > 3:
        df_globo.drop(columns=df_globo.columns[0], axis=1, inplace=True)
    uirs = list(df_globo.itertuples(index=False, name=None))
    return uirs

def load_feedbackc(fpath):
    """Load the user-item ratings, scale: 0,or 1

    Parameters
    ----------
    fpath: file path where the excel file containing user-item-rating information is stored.
    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).
    """
    if fpath.endswith('.csv'):
        df_globo = pd.read_csv(fpath)
        if len(df_globo.columns) > 4:
            df_globo.drop(columns=df_globo.columns[0], axis=1, inplace=True)
        uirs = list(df_globo.itertuples(index=False, name=None))
        return uirs


def load_sentiment(fpath):
    """Load the item sentiments per item into dictionary.
    Parameters
    ----------
    fpath: file path where the file containing item sentiment information is stored.
        format can be json or csv. If the format is csv, the first column should
        be item and second column should be sentiment.

    Returns
    -------
    dictionary
        Data in the form of a dictionary (item: sentiment).

    """
    sentiment_dict = {}
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        df1 = df.dropna()
        if len(df1.columns) >= 2 and any(pd.isna(pd.to_numeric(df1[df1.columns[1]], errors='coerce'))) != True:
            sentiment_dict = dict(
                zip(df1[df1.columns[0]], df1[df1.columns[1]]))
        else:
            raise ValueError(
                "when loading sentiment, received an invalid value. sentiment "
                "must be a numerical value."
            )

    elif fpath.endswith('.json'):
        with open(fpath) as json_file:
            dictionary = json.load(json_file)
        sentiment_dict = {k: v for k, v in dictionary.items() if v is not None}
    return sentiment_dict


def load_category(fpath):
    """Load item category per item into dictionary.

    Returns
    -------
    dictionary
    """
    category_dict = {}
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        df1 = df.dropna()
        if len(df1.columns) >= 2:
            category_dict = dict(zip(df1[df1.columns[0]], df1[df1.columns[1]]))
        else:
            raise ValueError(
                "Error when loading category."
            )
    elif fpath.endswith('.json'):
        with open(fpath) as json_file:
            dictionary = json.load(json_file)
        category_dict = {k: v for k, v in dictionary.items() if v is not None}
    return category_dict


def load_category_multi(fpath):
    """Load item categories per item into dictionary.

    Returns
    -------
    dictionary
    """
    category_dict = {}
    all_category = {}
    cur_id = 0
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        df1 = df.dropna()
        if len(df1.columns) >= 2:
            item_name = df1[df1.columns[0]]
            categories = df1[df1.columns[1]]
            X = zip(item_name, categories)
            for it, cat in X:
                temp = cat.split(",")
                for item0 in temp:
                    if item0 not in all_category and item0 is not None:
                        all_category[item0] = cur_id
                        cur_id = cur_id + 1
            X1 = zip(item_name, categories)
            for it, cat in X1:
                temp = cat.split(",")
                v = np.zeros(len(all_category.keys()))
                for item0 in temp:
                    if item0 is not None:
                        v[all_category[item0]] = 1

                category_dict[it] = v
        else:
            raise ValueError(
                "Error when loading (multi) category."
            )
    elif fpath.endswith('.json'):
        with open(fpath) as json_file:
            dictionary = json.load(json_file)
        for item_name, categories in dictionary.items():
            if isinstance(categories, list):
                for i in categories:
                    if i not in all_category and i is not None:
                        all_category[i] = cur_id
                        cur_id = cur_id + 1
            else:
                if categories not in all_category and categories is not None:
                    all_category[categories] = cur_id
                    cur_id = cur_id + 1
        for item_name, categories in dictionary.items():
            v = np.zeros(len(all_category.keys()))
            if isinstance(categories, list):
                for i in categories:
                    if i is not None:
                        v[all_category[i]] = 1
            elif categories is not None:
                v[all_category[categories]] = 1
            category_dict[item_name] = v
    return category_dict


def convert_to_array(dictionary):
    '''Converts lists of values in a dictionary to numpy arrays'''
    return {k: np.array(v) for k, v in dictionary.items()}


def load_complexity(fpath):
    """Load item complexity per item into dictionary.

    Returns
    -------
    dictionary
    """
    complexity_dict = {}
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        df1 = df.dropna()
        if len(df1.columns) >= 2 and any(pd.isna(pd.to_numeric(df1[df1.columns[1]], errors='coerce'))) != True:
            complexity_dict = dict(
                zip(df1[df1.columns[0]], df1[df1.columns[1]]))
        else:
            raise ValueError(
                "when loading complexity, received an invalid value."
            )
    elif fpath.endswith('.json'):
        with open(fpath) as json_file:
            dictionary = json.load(json_file)
        complexity_dict = {k: v for k,
                           v in dictionary.items() if pd.isnull(v) == False}
    return complexity_dict


def load_story(fpath):
    """Load item story per item into dictionary where the story is an integer.
    Returns a dictionary with item identifiers as keys and story as values.
    """
    story_dict = {}
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        df1 = df.dropna()
        # Ensure the second column can be converted to int
        if len(df1.columns) >= 2 and not pd.to_numeric(df1[df1.columns[1]], errors='coerce').isna().any():
            story_dict = dict(
                zip(df1[df1.columns[0]], df1[df1.columns[1]].astype('int')))
        else:
            raise ValueError("when loading story, received an invalid value.")
    elif fpath.endswith('.json'):
        with open(fpath, 'r') as json_file:
            story_data = json.load(json_file)
        # Ensure the 'story' key exists and its value is not None before converting
        #previously
        #story_dict = {key: int(val['story']) for key, val in story_data.items() if 'story' in val and val['story'] is not None}
            story_dict = {key: int(val) for key, val in story_data.items() if val is not None}
            #story_dict = {key: int(val["story"]) 
             #             for key, val in story_data.items() 
             #             if val is not None and "story" in val and val["story"] is not None}

    else:
        raise ValueError("Unsupported file type. Only .csv and .json files are supported.")

    return story_dict

def load_entities(fpath):
    """Load item entities per item into dictionary.
    Item entities can be array.
    Returns
    -------
    dictionary
    """
    entities_dict = {}
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        df1 = df.dropna()
        if len(df1.columns) >= 2:
            item_name = df1[df1.columns[0]]
            entities = df1[df1.columns[1]]
            X = zip(item_name, entities)
            for it, ent in X:
                temp = ent.split(",")
                entities_dict[it] = temp
        else:
            raise ValueError(
                "Error when when loading entities."
            )
    elif fpath.endswith('.json'):
        with open(fpath) as json_file:
            entity = json.load(json_file)
        entities_dict = {}
        for key, value in entity.items():
            new_value = []
            if isinstance(value, dict):
                if bool(value):
                    new_value = []
                    for k, v in value.items():
                        try:
                            v1 = int(v)
                            new_value.extend([k]*v1)
                        except ValueError:
                            print(
                                "input invalid json, the frequency of entity should be an integer")
                    entities_dict[key] = new_value
            else:
                raise ValueError(
                    "Error when when loading entities."
                )
    return entities_dict


def load_min_maj(fpath, data_type="mainstream"):
    data_type = data_type
    min_maj_dict = {}
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        df1 = df.dropna()
        if len(df1.columns) >= 2:
            item_name = df1[df1.columns[0]]
            min_values = df1[df1.columns[1]]
            maj_values = df1[df1.columns[2]]
            X = zip(item_name, min_values, maj_values)
            for it, min_value, maj_value in X:
                v = np.zeros(2)
                try:
                    v[0] = float(min_value)
                    v[1] = float(maj_value)
                    min_maj_dict[it] = v
                except ValueError:
                    print(
                        "input invalid json for item {}. The minority score and majority score should be converted to float".format(it))
    elif fpath.endswith('.json'):
        with open(fpath) as json_file:
            dictionary = json.load(json_file)
        for item_name, item_data in dictionary.items():
            if data_type in item_data:
                min_maj_val = item_data[data_type]
                v = np.zeros(2)
                try:
                    v[0] = float(min_maj_val[0])
                    v[1] = float(min_maj_val[1])
                    min_maj_dict[item_name] = v
                except ValueError:
                    print("input invalid json for item {}. The minority score and majority score should be converted to float".format(
                        item_name))
            else:
                continue
    return min_maj_dict


def load_text(fpath):
    """Load text per item into dictionary.

    Returns
    -------
    dictionary
    """
    text_dict = {}
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        df1 = df.dropna()
        if len(df1.columns) >= 2:
            text_dict = dict(zip(df1[df1.columns[0]], df1[df1.columns[1]]))
        else:
            raise ValueError(
                "Error when loading text."
            )
    elif fpath.endswith('.json'):
        with open(fpath, 'r', encoding='utf-8') as json_file:
            dictionary = json.load(json_file)
        text_dict = {
            k: (v['text'] if 'text' in v and v['text'] is not None else 'default_value')
            for k, v in dictionary.items()
        }
    return text_dict

def load_diversity(fpath):
    """Load data from a CSV file into a dictionary if it contains exactly two columns.

    Parameters
    ----------
    fpath : str
        Path to the CSV file.

    Returns
    -------
    dictionary
        A dictionary where keys are values from the first column and values are from the second column.
    """
    # Read the CSV file
    df = pd.read_csv(fpath)
    # Check that the DataFrame contains exactly two columns
    if df.shape[1] == 2:
        # Create a dictionary from the DataFrame
        diversity_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    else:
        raise ValueError("The file must contain exactly two columns.")
    
    return diversity_dict

def build(data, id_map, **kwargs):
    item_id2idx = {k: v for k, v in id_map.items()}
    feature_map = {}
    for key, value in data.items():
        if key in item_id2idx:
            idx = item_id2idx[key]
            feature_map[idx] = value

    return feature_map

def load_history(hpath):

    history_dict = {}
    if hpath.endswith('.parquet'):
        df = pd.read_parquet(hpath, columns=['impressionId', 'userId', 'impressionTimestamp', 'history'])
        df1 = df.dropna()
        df_history = df1.sort_values(by='impressionTimestamp').groupby('userId').last().reset_index()[["userId", "history"]]
    else:
        raise SystemError(
                "Unsupported file type."
            )
        

    if len(df_history.columns) == 2:
        history_dict = dict(
            zip(df_history[df_history.columns[0]], df_history[df_history.columns[1]]))
    else:
        raise ValueError(
            "when loading history, received an invalid value."
        )
    
    res = list()

    n = 0

    for k,v in history_dict.items():
         
        if history_dict[k] is not None: 

            history_dict[k] = ' '.join(history_dict[k])

            res.append((k, history_dict[k], 1))
    
    return res

def load_party(ppath):

    party_dict = {}

    if ppath.endswith('.json'):
        with open(ppath) as json_file:
            party_dict = json.load(json_file)
        return party_dict


# For exposure diversity model
def article_political_epd(ppath):

    config = configparser.ConfigParser()
    config.read('parameters.ini')
    majority = config['EPD']['majority'].replace('_', ' ').split(',')

    party_dict_temp = {}

    if ppath.endswith('.json'):
        with open(ppath, encoding="utf-8") as json_file:
            party_dict_temp = json.load(json_file)
    

    party_dict = {}
    for k,v in party_dict_temp.items():
        party_dict.setdefault(k, v['party'])
    print(party_dict)
    article_list = []

    for article, political_references in party_dict.items():

        political_references_count = 0
        minority_count = 0

        article_prop = {}
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

def load_user_impression(path):
    """
    path: behaviors.tsv

    load user recommended list and calculate the rating
    
    return: [(uid, iid, rating, *)]
    """
    if path.endswith('.parquet'):

        behaviors = pd.read_parquet(path, columns=['impressionId', 'impressionsWithClick', 'userId', 'impressionTimestamp', 'history'])
        behaviors = behaviors.dropna()
        return list(zip(behaviors['userId'], behaviors['impressionsWithClick'].apply(user_all_impression), behaviors['impressionsWithClick'].apply(impression_score)))

    
def load_user_group_type(path, uid_map):
    """
    path: behaviors.tsv

    load user recommended list and calculate the rating
    
    return: [{UserId:123, userGroup}]

    TODO: specify user's userGroup and participatesIn
    """
    if path.endswith('.parquet'):
        behaviors = pd.read_parquet(path, columns=['impressionId', 'impressions', 'userId', 'impressionTimestamp', 'history'])
        behaviors = behaviors.dropna()

        uid_list = behaviors['userId'].to_list()
        userlist = []
        for uid in uid_list:
            if uid in list(uid_map.keys()):
                user = {}
                user['userId'] = uid_map[uid]
                user['userGroup'] = random.choice([1,2,3])
                userlist.append(user)
        
        return userlist

def user_all_impression(impression):
    return " ".join([token[:-2] for token in impression])

def impression_score(impression):
    pos_impression_count = float(len([token for token in impression if "-1" in token]))
    neg_impression_count = float(len([token for token in impression if "-0" in token]))
    return float(format(pos_impression_count/(pos_impression_count + neg_impression_count), ".4f"))