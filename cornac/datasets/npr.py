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
This data is built based on the NPR datasets
provided by: https://kaggle.com/datasets/joelpl/news-portal-recommendations-npr-by-globo/
"""
import pandas as pd
import json
import numpy as np
import ast

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
        df_npr = pd.read_csv(fpath)
        if len(df_npr.columns) > 3:
            df_npr.drop(columns=df_npr.columns[0], axis=1, inplace=True)
        uirs = list(df_npr.itertuples(index=False, name=None))
        return uirs

    
# convert strings to lists
def str_to_list(s):
    return ast.literal_eval(s)    

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
            # Use apply() to convert strings in column 'topics' into lists
            df1[df1.columns[1]] = df1[df1.columns[1]].apply(str_to_list)
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
            # Use apply() to convert strings in column 'topics' into lists
            df1[df1.columns[1]] = df1[df1.columns[1]].apply(str_to_list)
            item_name = df1[df1.columns[0]]
            categories = df1[df1.columns[1]]
            X = zip(item_name, categories)
            for it, cat in X:
                for item0 in cat:
                    if item0 not in all_category and item0 is not None:
                        all_category[item0] = cur_id
                        cur_id = cur_id + 1
            X = zip(item_name, categories)
            for it, cat in X:
                v = np.zeros(len(all_category.keys()))
                for item0 in cat:
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


def build(data, id_map, **kwargs):
    item_id2idx = {k: v for k, v in id_map.items()}
    feature_map = {}
    for key, value in data.items():
        if key in item_id2idx:
            idx = item_id2idx[key]
            feature_map[idx] = value

    return feature_map
