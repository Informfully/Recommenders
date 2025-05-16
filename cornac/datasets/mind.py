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
This data is built based on the Mind datasets
provided by: https://msnews.github.io/
"""
import pandas as pd
import json
import numpy as np
import configparser
import random
import os


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
    if not fpath.endswith('.csv'):
        raise ValueError("The file must be a CSV format.")

    try:
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(fpath)
    except Exception as e:
        raise ValueError(f"Error reading the input csv file: {e}")
    # If the DataFrame has more than 3 columns, drop all but the last 3
    if len(df.columns) > 3: 
        df = df[df.columns[-3:]] 
    # The DataFrame has at least 3 columns (after dropp)
    if len(df.columns) < 3:
        raise ValueError("The file must contain at least 3 columns: UserId, ItemId, and rating.")
    user_column = df.columns[0]
    item_column = df.columns[1]
    rating_column = df.columns[2]
    df[user_column] = df[user_column].astype(str)
    df[item_column] = df[item_column].astype(str)
    df[rating_column] = pd.to_numeric(df[rating_column], errors='coerce')  # In case of non-numeric values
    
    # Handle missing or invalid ratings (e.g., NaNs after coercion)
    df = df.dropna(subset=[rating_column])
    uirs = list(df[[user_column, item_column, rating_column]].itertuples(index=False, name=None))
    
    return uirs
    # if fpath.endswith('.csv'):
    #     # df_mind = pd.read_csv(fpath)
    #     df_mind = pd.read_csv(fpath, dtype={'UserID': str, 'ItemID': str})
    #     if len(df_mind.columns) > 3:
    #         df_mind.drop(columns=df_mind.columns[0], axis=1, inplace=True)
    #     uirs = list(df_mind.itertuples(index=False, name=None))
    #     return uirs

## feedback with category
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
        df_mind = pd.read_csv(fpath)
        if len(df_mind.columns) > 4:
            df_mind.drop(columns=df_mind.columns[0], axis=1, inplace=True)
        uirs = list(df_mind.itertuples(index=False, name=None))
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
    """Load item story per item into dictionary.
    story: int.
    Returns
    -------
    dictionary
    """
    story_dict = {}
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        df1 = df.dropna()
        if len(df1.columns) >= 2 and any(pd.isna(pd.to_numeric(df1[df1.columns[1]], errors='coerce'))) != True:
            story_dict = dict(
                zip(df1[df1.columns[0]], df1[df1.columns[1]].astype('int')))
        else:
            raise ValueError(
                "when loading story, received an invalid value."
            )
    elif fpath.endswith('.json'):
        with open(fpath) as json_file:
            story_dict0 = json.load(json_file)
        dictionary = {k: v for k, v in story_dict0.items() if v is not None}
        story_dict = dict([a, int(x)] for a, x in dictionary.items())

    return story_dict


def load_encoding_vectors(fpath):
    """
    Load  encoding vectors/embeddings from a JSON file and convert them into a dictionary of NumPy arrays.

    Args:
        fpath (str): Path to the JSON file containing the encoding vectors.

    Returns:
        dict: A dictionary with each key as an ID and the corresponding one-hot encoding as a NumPy array.

    Raises:
        ValueError: If the JSON file does not contain lists as values.
        FileNotFoundError: If the file path does not exist.
        json.JSONDecodeError: If the file is not a valid JSON format.
    """
    try:
        with open(fpath, 'r') as json_file:
            # Load JSON data
            vectors = json.load(json_file)

        # Use convert_to_array to convert all lists to NumPy arrays
        vectors_np = convert_to_array(vectors)

        print("Encoding vectors loaded and converted to NumPy arrays successfully.")
        return vectors_np

    except FileNotFoundError:
        print(f"Error: File '{fpath}' not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: File '{fpath}' is not a valid JSON format.")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise


def load_entities(fpath, keep_empty=False):
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
                # entities_dict[it] = temp
                # If keep_empty is True, allow empty lists; otherwise, skip empty ones
                if keep_empty or temp:  # Only add if temp is not empty, or allowing empty lists
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
                    for k, v in value.items():
                        try:
                            v1 = int(v)
                            new_value.extend([k]*v1)
                        except ValueError:
                            print(
                                "input invalid json, the frequency of entity should be an integer")
                    
                if keep_empty or new_value:
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


def load_min(fpath, data_type="mainstream"):
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
                    # min_maj_dict[it] = v
                    min_maj_dict[it] = v[0]
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
                    # min_maj_dict[item_name] = v
                    min_maj_dict[item_name] = v[0]
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
        with open(fpath) as json_file:
            dictionary = json.load(json_file)
        text_dict = {k: v for k, v in dictionary.items() if v is not None}
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


def load_category_party(fpath):
    """Load item party categories per item into dictionary.

    Returns
    -------
    dictionary
    """
    categories = set()
    dict = {}
    if fpath.endswith('.json'):
        with open(fpath) as json_file:
            data = json.load(json_file)
        for category_dict in data.values():
            categories.update(category_dict.keys())
        categories = sorted(categories)
        for item_id, category_dict in data.items():
            vector = np.zeros(len(categories))
            for category in category_dict.keys():
                index = categories.index(category)
                vector[index] = 1
            dict[item_id] = vector
    return dict



def load_user_group_type(path, uid_map):
    """
    path: behaviors.tsv

    load user recommended list and calculate the rating
    
    return: [{UserId:123, userGroup, participatesIn}]

    TODO: specify user's userGroup and participatesIn
    """
    if path.endswith('.tsv'):
        behaviors = pd.read_csv(path, header=None, sep='\t')
        behaviors.columns = ['ImpressionId', 'UserId', 'Time', 'History', 'Impressions']
        behaviors = behaviors.dropna()

        uid_list = behaviors['UserId'].to_list()
        userlist = []
        for uid in uid_list:
            if uid in list(uid_map.keys()):
                user = {}
                user['userId'] = uid_map[uid]
                user['userGroup'] = random.choice([1,2,3])
                userlist.append(user)
        
        return userlist



def load_party(ppath, iid_map):

    party_dict_name = {}
    party_dict = {}

    if ppath.endswith('.json'):
        with open(ppath) as json_file:
            party_dict_name = json.load(json_file)
        
        # for a, p in party_dict_name.items():
        #     if a in list(iid_map.keys()):
        #         party_dict[iid_map[a]] = p

        return party_dict_name



def load_user_political_score(users, history, party_dict):

    user_score_matrix = np.zeros((len(users), 2), dtype=float)

    for k, v in party_dict.items():
        if len(list(v)) == 0:
            party_dict[k] = 0
        else:
            democratic = 0
            republican = 0
            for i,j in v.items():
                if i == "Democratic Party":
                    democratic = j
                elif i == "Republican Party":
                    republican = j

            party_dict[k] = republican - democratic
        
    # party_dict : { article: republican_score}
    # if republican_score > 0, this article is considered as republican article, else as democratic

    for u, articles in enumerate(history):

        article_list = articles.split(' ')
        history_score_matrix = np.zeros((len(article_list), 2), dtype=float)

        for i, article in enumerate(article_list):
            if article not in party_dict.keys():
                history_score_matrix[i] = [0,0]
            else :
                if party_dict[article] > 0:
                    history_score_matrix[i] = [0,1]
                elif party_dict[article] == 0:
                    history_score_matrix[i] = [0,0]
                else:
                    history_score_matrix[i] = [1,0]
        
        # print("u: ")
        # print(history_score_matrix[:50])
        # print(len(history_score_matrix))
        # print("---------------------")

        republican_score = np.sum(history_score_matrix[:,1]) / len(history_score_matrix)
        democratic_score = np.sum(history_score_matrix[:,0]) / len(history_score_matrix)

        #scale the score
        user_score_matrix[u][1] = round(2 * republican_score - 1, 1)
        user_score_matrix[u][0] = round(2 * democratic_score - 1, 1)
    
    return user_score_matrix

# For diversity split
def load_clusters(fpath):
    """
    Load cluster information per user into a dictionary.

    Parameters
    ----------
    fpath: str
    Returns
    -------
    dict
        A dictionary with user IDs as keys and cluster information as values.
    
    Raises
    ------
    ValueError
        If the file format or contents are invalid (e.g., cluster information is not numeric).
    """
    
    cluster_dict = {}
    
    # Ensure the file is a CSV file
    if fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        
        df_cleaned = df.dropna(subset=[df.columns[6], df.columns[7]])
        
        # Check that the cluster column is numeric
        if len(df_cleaned.columns) >= 8 and not any(pd.isna(pd.to_numeric(df_cleaned[df_cleaned.columns[7]], errors='coerce'))):
            # Create the dictionary mapping user ID (7th column) to cluster information (8th column)
            cluster_dict = dict(zip(df_cleaned[df_cleaned.columns[6]], df_cleaned[df_cleaned.columns[7]]))
        else:
            raise ValueError("Cluster information must be a numerical value.")
    else:
        raise ValueError("Invalid file format. Expected a CSV file.")
    
    return cluster_dict

        

    



    





        
        
        


