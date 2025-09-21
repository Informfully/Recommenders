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

from configparser import ConfigParser
import numbers
import ast
import numpy as np
import scipy.sparse as sp
import pandas as pd
import random
import math
from .fast_sparse_funcs import (
    inplace_csr_row_normalize_l1,
    inplace_csr_row_normalize_l2
)

FLOAT_DTYPES = (np.float64, np.float32, np.float16)


def sigmoid(x):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-x))


def scale(values, target_min, target_max, source_min=None, source_max=None):
    """Scale the value of a numpy array "values"
    from source_min, source_max into a range [target_min, target_max]

    Parameters
    ----------
    values : Numpy array, required
        Values to be scaled.

    target_min : scalar, required
        Target minimum value.

    target_max : scalar, required
        Target maximum value.

    source_min : scalar, required, default: None
        Source minimum value if desired. If None, it will be the minimum of values.

    source_max : scalar, required, default: None
        Source minimum value if desired. If None, it will be the maximum of values.

    Returns
    -------
    res: Numpy array
        Output values mapped into range [target_min, target_max]
    """
    if source_min is None:
        source_min = np.min(values)
    if source_max is None:
        source_max = np.max(values)
    if source_min == source_max:  # improve this scenario
        source_min = 0.0

    values = (values - source_min) / (source_max - source_min)
    values = values * (target_max - target_min) + target_min
    return values


def clip(values, lower_bound, upper_bound):
    """Perform clipping to enforce values to lie
    in a specific range [lower_bound, upper_bound]

    Parameters
    ----------
    values : Numpy array, required
        Values to be clipped.

    lower_bound : scalar, required
        Lower bound of the output.

    upper_bound : scalar, required
        Upper bound of the output.

    Returns
    -------
    res: Numpy array
        Clipped values in range [lower_bound, upper_bound]
    """
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values


def intersects(x, y, assume_unique=False):
    """Return the intersection of given two arrays
    """
    mask = np.isin(x, y, assume_unique=assume_unique)
    x_intersects_y = x[mask]

    return x_intersects_y


def excepts(x, y, assume_unique=False):
    """Removing elements in array y from array x
    """
    mask = np.isin(x, y, assume_unique=assume_unique, invert=True)
    x_excepts_y = x[mask]

    return x_excepts_y


def safe_indexing(X, indices):
    """Return items or rows from X using indices.
    Allows simple indexing of lists or arrays.
    Borrowed from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis
    """
    if hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]


def validate_format(input_format, valid_formats):
    """Check the input format is in list of valid formats
    :raise ValueError if not supported
    """
    if input_format not in valid_formats:
        raise ValueError('{} data format is not in valid formats ({})'.format(input_format, valid_formats))

    return input_format


def estimate_batches(input_size, batch_size):
    """
    Estimate number of batches give `input_size` and `batch_size`
    """
    return int(np.ceil(input_size / batch_size))

# find all items in the candidate that satisify the aspect


def find_items_match(df, column, match_value):
    """
    Find indices of all rows in the DataFrame where the value in the specified column exactly matches the given value.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to look in.
        match_value (any): The value to match against.

    Returns:
        list: A list of indices where the condition is met.
    """
    # Use a boolean mask to find rows where the value matches exactly
    mask = df[column] == match_value
    return df.index[mask].tolist()


def find_items_in_range(df, column, min_value, max_value):
    """
    Find indices of all rows in the DataFrame where the value in the specified column is within the range [min_value, max_value].

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to look in.
        min_value (any): The minimum value of the range.
        max_value (any): The maximum value of the range.

    Returns:
        list: A list of indices where the condition is met.
    """
    # Use a boolean mask to find rows where the value is within the specified range
    # greater than or equal to min_value but strictly less than max_value
    mask = (df[column] >= min_value) & (df[column] < max_value)
    return df.index[mask].tolist()


# def processPartyData(input_string):
#     if isinstance(input_string, list):
#         return input_string
#     elif (pd.isna(input_string)):
#         pass
#     else:
#         input_string = input_string[1:-1]
#         # this part handles the case when te list of parties is loaded from csv file.
#         # because test_str will be: "["republican",...]", should remove first character " and last character "
#         parties = input_string.split(', ')
#         parties = [party[1:-1] for party in parties]
#         return parties  # return list of parties

def processPartyData(test_str):
    """
    Processes a party-related string or list to extract a list of party names.

    This function is designed to handle data in multiple formats, such as:
    - Directly returning a list of parties if the input is already a list.
    - Parsing and cleaning a string representation of a list (e.g., '["party1", "party2"]').

    Parameters
    ----------
    test_str : str or list
        Input data representing party affiliations or classifications.
        - If a list is provided, it is returned as-is.
        - If a string is provided, it is expected to be in a CSV-like format, e.g., '["party1", "party2"]'.

    Returns
    -------
        A list of party names after processing.
    Notes
    -----
    - If the input is a string in the format '["party1", "party2"]', it removes the enclosing square brackets 
      and quotes around party names, returning a clean list of strings.

    Examples
    --------
    >>> processPartyData(['Democratic', 'Republican'])
    ['Democratic', 'Republican']

    >>> processPartyData('["Democratic", "Republican"]')
    ['Democratic', 'Republican']

    """
    if isinstance(test_str, list):
        return test_str
    if (pd.isna(test_str)):
        # pass
        return []
    if isinstance(test_str, str):
        # evaluate the string if it's in list format
        try:
            parties = ast.literal_eval(test_str)  # directly evaluate the string as a list
            
            # Ensure the evaluated result is actually a list
            if not isinstance(parties, list):
                return []
        except:
            # If eval fails, return an empty list or handle error as needed
            return []
        return parties
        # test_str = test_str[1:-1]
        # # this part handles the case when te list of parties is loaded from csv file.
        # # because test_str will be: "["republican",...]", should remove first character " and last character "
        # parties = test_str.split(', ')
        # parties = [party[1:-1] for party in parties]
        # return parties  # return list of parties

    # If the input format is not recognized, return an empty list
    return []

def is_valid_party_list(x):
    # Acceptable: None, NaN, empty list, or list of strings
    if x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, list) and len(x) == 0):
        return True
    if isinstance(x, list):
        return all(isinstance(i, str) for i in x)
    return False

def count_selected_in_aspects(selected_items, aspect_dictionaries):
    """
    Counts how many selected items fall into each aspect across multiple dictionaries.

    Optimized to use numpy's vectorized operations for large datasets.

    Parameters:
        selected_items (list): List of selected item IDs.
        aspect_dictionaries (list of dicts): Each dictionary contains aspects as keys and lists of item IDs as values.

    Returns:
        list of dicts: Similar structure to aspect_dictionaries but with counts instead of lists.
    """
    # Convert selected_items to a NumPy array for vectorized operations
    selected_items = np.array(selected_items)

    # Initialize the output count structure with zeros
    aspect_counts = [{key: 0 for key in aspect_dict}
                     for aspect_dict in aspect_dictionaries]

    # Iterate through each dictionary and key to count occurrences
    for i, aspect_dict in enumerate(aspect_dictionaries):
        for aspect, items in aspect_dict.items():
            # Convert items to NumPy array and use np.isin for fast membership counting
            items = np.array(items)
            aspect_counts[i][aspect] = np.isin(items, selected_items).sum()

    return aspect_counts


def TargetDistributionMatcher(targetDistributions, targetDimension, item_feature_dataframe, candidate_items):
    """
    Matches items from candidate_items to a target distribution across specified dimensions.
    Optimized for speed using vectorized operations.

    Parameters:
    - targetDistributions (list of dict): Specifies the desired distribution for each dimension (e.g., category).
    - targetDimension (list): A list of dimensions to consider for matching (e.g., ['category']).
    - item_feature_dataframe (pd.DataFrame): DataFrame with item features; index is item IDs, columns are dimensions.
    - candidate_items (list): List of item IDs that are candidates for re-ranking.

    Returns:
    - target_aspect_proportions (list of dict): Target proportion for each dimension's aspect in the candidate item list.
    - matched_items (list of dict): Dictionary where keys are dimension_requirement and values are lists of item IDs.
    """

    # Check if candidate_items is None or empty
    if candidate_items is None or len(candidate_items) == 0:
        print("Warning: candidate_items is None or empty.")
        return [], []  # Return empty lists if no candidate items

    # Ensure candidate_items is a list (in case it's a numpy array or other array-like structure)
    if not isinstance(candidate_items, list):
        candidate_items = list(candidate_items)

     # Check if candidate_items exist in the item_feature_dataframe
    missing_items = [
        item for item in candidate_items if item not in item_feature_dataframe.index]
    if missing_items:
        print(
            f"Warning: The following items from candidate_items are missing in the item_feature_dataframe: {missing_items}")

    # Filter out only the relevant rows (candidate_items) from the item_feature_dataframe
    # data = item_feature_dataframe.loc[candidate_items]
    valid_items = item_feature_dataframe.index.intersection(candidate_items)
    data = item_feature_dataframe.loc[valid_items]

    target_aspect_proportions = []
    matched_items = []

    valid_party_type_words = ['only', 'minority', 'composition', 'no_party','no party','no parties','no_parties']


    # Iterate over target distributions
    for i, targetDistribution in enumerate(targetDistributions):
        description = targetDimension[i]
        tar = targetDistribution["distr"]

        if targetDistribution["type"] == "discrete":
            # Get indices for each unique value in the description column
            # Directly filter for discrete values using .isin()
            temp_dict_items = {}
            temp_dict_proportion = {}

            for aspect_value in tar.keys():
                # Filter for rows in `data` where the column `description` matches `aspect_value`
                mask = data[description].isin([aspect_value])
                temp_dict_items[f"{description},{aspect_value}"] = data.index[mask].tolist(
                )
                temp_dict_proportion[f"{description},{aspect_value}"] = tar[aspect_value]

        elif targetDistribution["type"] == "continuous":
            # Vectorized filtering for continuous ranges
            column_data = data[description].to_numpy()
            temp_dict_items = {}
            temp_dict_proportion = {}

            for item in tar:
                min_val, max_val = item['min'], item['max']
                mask = (column_data >= min_val) & (
                    column_data < max_val)  # Vectorized range filtering
                aspect = f"{description},{min_val},{max_val}"
                # Convert mask to item indices
                temp_dict_items[aspect] = data.index[mask].tolist()
                temp_dict_proportion[aspect] = item['prob']

        elif targetDistribution["type"] in  ["parties", "party", "entities", "entity"]:
            # Optimizing the handling of party-based filtering by vectorizing the process
            # Vectorized function to clean data
            cleanedData = data[description].apply(processPartyData)
            # Validate entries
            invalid_entries = cleanedData[~cleanedData.apply(is_valid_party_list)]
            if not invalid_entries.empty:
                raise ValueError(f"Invalid entries in '{targetDimension}': all non-empty lists must contain only strings.\n Unexpected entries:\n{invalid_entries}")

            # Step 3: Normalize to lowercase
            cleanedData = cleanedData.apply(
                lambda x: [s.lower() for s in x] if isinstance(x, list) and len(x) > 0 else x
            )
            temp_dict_items = {}
            temp_dict_proportion = {}

            for item in tar:
                proportion = item['prob']
                # relevant_parties = set(item['contain'])
                relevant_parties = item['contain']

                descriptor = str(item['description']).lower()
                aspect = f"{description},{item['description']}:{','.join(map(str, relevant_parties))}"
                
             
               # Validate if the descriptor contains at least one of the valid type words
                if not any(word in descriptor for word in valid_party_type_words):
                    raise ValueError(f"Invalid {descriptor}: must contain at least one of the following words: {', '.join(valid_party_type_words)}")
                if 'composition' in descriptor:
                    # Check if relevant_parties is a list of lists
                    if isinstance(relevant_parties, list):
                        if not all(isinstance(sublist, list) for sublist in relevant_parties):
                            raise ValueError(f"For 'composition' descriptor, 'contain' must be a list of lists. Received: {relevant_parties}")
                        # print(f"relevant_parties:{relevant_parties}")
                        target_sets_composition = [set(kw.lower() for kw in sublist) for sublist in relevant_parties]
                        # print(f"target_sets_composition:{target_sets_composition}")
                        
                        # Flatten the lists to create a set of all allowed parties
                        all_allowed_parties = set([party.lower() for sublist in relevant_parties for party in sublist])
                    else:
                        raise ValueError(f"For 'composition' descriptor, 'contain' must be a list of lists. Received: {relevant_parties}")
                else:
                    # For other cases (e.g., "only mention", "minority", etc.)
                    relevant_parties = set(party.lower() for party in relevant_parties)

                # if 'composition' in descriptor:
               
                #     # Flatten the lists to create a set of all allowed parties
                #     all_allowed_parties = set([party for sublist in relevant_parties for party in sublist])
                # else:
                #     # For other cases
                #     relevant_parties = set(relevant_parties)

                # if "only" in descriptor or "minority" in descriptor:
                #     # Ensure relevant_parties is a non-empty list; this is required for 'only' or 'minority' classifications

                if ("only" in descriptor or "minority" in descriptor):
                    if not isinstance(relevant_parties, (list, set)) or len(relevant_parties) == 0:
                        raise ValueError(
                            f"For descriptor '{descriptor}', 'contain' must be a non-empty list or set of parties. "
                            f"Received: {type(relevant_parties)} with value: {relevant_parties}"
                        )
                if "only" in descriptor:
                
                    # Vectorized check for exact party match
                    # mask = cleanedData.apply(lambda x: set(
                    #     x) == relevant_parties if x is not None else False)
                            # x must be a non-empty list and subset of relevant_parties
                    mask = cleanedData.apply(
                        lambda x: isinstance(x, list)
                        and len(x) > 0
                        and set(x).issubset(relevant_parties)
                    )

                elif "minority" in descriptor:
                    
                    # Vectorized check for minority parties
                    # mask = cleanedData.apply(lambda x: len(set(x).difference(
                    #     relevant_parties)) > 0 if x is not None else False)
                    mask = cleanedData.apply(
                        lambda x: isinstance(x, list)
                        and len(x) > 0
                        and len(set(x).difference(relevant_parties)) > 0
                    )
                elif "no parties" in descriptor or "no party" in descriptor:
                    # Check for rows with no parties
                    # mask = cleanedData.isna()
                    # Check for rows with no parties (empty string, empty list, or NaN)
                    # mask = cleanedData.apply(lambda x: (
                    #     pd.isna(x) or 
                    #     (isinstance(x, str) and x.strip() == "") or 
                    #     (isinstance(x, list) and len(x) == 0)
                    # ))
           

                    mask = cleanedData.apply(lambda value: (
                        value is None
                        or (isinstance(value, float) and math.isnan(value))
                        or (isinstance(value, str) and value.strip() == "")
                        or (isinstance(value, list) and len(value) == 0)
                    ))
                # Handle "composition" type
                elif "composition" in descriptor:
                    mask = cleanedData.apply(
                        lambda x: isinstance(x, list) and len(x) > 0
                                and all(len(set(x).intersection(set(sublist))) > 0 for sublist in target_sets_composition)
                                and  set(x).issubset(all_allowed_parties)
                    )

                # Convert mask to item indices
                temp_dict_items[aspect] = data.index[mask].tolist()
                temp_dict_proportion[aspect] = proportion

        target_aspect_proportions.append(temp_dict_proportion)
        matched_items.append(temp_dict_items)

    return target_aspect_proportions, matched_items


def get_max_keys(dictionary):
    """ Return all keys from the dictionary that have the maximum value. """
    if not dictionary:
        return []
    if any(not isinstance(value, (int, float)) for value in dictionary.values()):
        raise ValueError("All values in the dictionary must be numeric.")

    max_value = max(dictionary.values())  # Find the maximum value
    max_keys = [key for key, value in dictionary.items() if value == max_value]
    return max_keys

# Updated safe_kl_divergence function to work with NumPy arrays


def safe_kl_divergence(p, q, alpha=0.1, epsilon=1e-10):
    """
    Computes the KL divergence between two probability distributions p and q,
    adjusting q by a small alpha towards p to handle zero values safely.

    Parameters:

        p (np.array): First probability distribution (1D array).
        q (np.array): Second probability distribution (1D array).
        alpha (float): Small number to adjust q towards p.
        epsilon (float): A small constant to avoid division by zero or log of zero.

    Returns:
        float: The calculated KL divergence.
    """
    # Adjust q according to alpha and p
    adjusted_q = (1 - alpha) * q + alpha * p
    # Ensure adjusted_q is never less than epsilon to avoid division by zero or log(0)
    adjusted_q = np.clip(adjusted_q, epsilon, None)
    p = np.clip(p, epsilon, None)

    # Ensure no division by zero and apply the KL formula only where p > 0
    kl_div = np.sum(np.where(p > 0, p * np.log(p / adjusted_q), 0))

    return kl_div


def safe_kl_divergence_dicts(p, q, alpha=0.01):
    """
    Computes the KL divergence between two probability distributions p and q,
    adjusting q by a small alpha towards p to handle zero values safely.

    Parameters:
        p (dict): First probability distribution, as a dictionary.
        q (dict): Second probability distribution, as a dictionary.
        alpha (float): Small number to adjust q towards p.

    Returns:
        float: The calculated KL divergence.
    """
    kl_div = 0.0
    adjusted_q = {}
    # First adjust q according to alpha and p
    for key in p:
        p_value = p[key]
        q_value = q.get(key, 0)

        adjusted_q[key] = (1 - alpha) * q_value + alpha * p_value

    # Now calculate the KL divergence
    for key in p:
        p_value = p[key]
        q_value = adjusted_q[key]
        if p_value > 0 and q_value > 0:  # Only calculate where p_value is non-zero to avoid math domain errors
            kl_div += p_value * math.log(p_value / q_value)

    return kl_div


def get_rng(seed):
    '''Return a RandomState of Numpy.
    If seed is None, use RandomState singleton from numpy.
    If seed is an integer, create a RandomState from that seed.
    If seed is already a RandomState, just return it.
    '''
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} can not be used to create a numpy.random.RandomState'.format(seed))


def normalize(X, norm='l2', axis=1, copy=True):
    """Scale input vectors individually to unit norm (vector length).

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    Reference
    ---------
    https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/preprocessing/data.py#L1553

    """
    if norm not in ('l1', 'l2', 'max'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if len(X.shape) != 2:
        raise ValueError("input X must be 2D but shape={}".format(X.shape))

    X_out = X.copy() if copy else X
    X_out = X_out if X_out.dtype in FLOAT_DTYPES else X_out.astype(np.float64)

    if axis == 0:
        X_out = X_out.T

    if sp.issparse(X_out):
        X_out = X_out.tocsr()

        if norm == 'l1':
            inplace_csr_row_normalize_l1(X_out)
        elif norm == 'l2':
            inplace_csr_row_normalize_l2(X_out)
        elif norm == 'max':
            norms = X_out.max(axis=1).toarray()
            norms_elementwise = norms.repeat(np.diff(X_out.indptr))
            mask = norms_elementwise != 0
            X_out.data[mask] /= norms_elementwise[mask]
    else:
        if norm == 'l1':
            norms = np.abs(X_out).sum(axis=1)
        elif norm == 'l2':
            norms = np.sqrt((X_out ** 2).sum(axis=1))
        elif norm == 'max':
            norms = np.max(X_out, axis=1)
        norms[norms == 0] = 1.
        X_out /= norms.reshape(-1, 1)

    if axis == 0:
        X_out = X_out.T

    return X_out


# For model: RDW, EPD and PLD
def build_history(uir):
    df = pd.DataFrame(uir, columns=['uid', 'iid', 'rating'])
    filter_records = df[df['rating'] == 1]
    history = filter_records.groupby('uid')['iid'].apply(list).reset_index()
    history = history.set_index('uid')['iid'].to_dict()
    return history

def load_user_group_type(uir):
    uid_list = [uid for uid,_,_ in uir]
    userlist = []
    for uid in uid_list:
        user = {}
        user['userId'] = uid
        user['userGroup'] = random.choice([1,2,3])
        userlist.append(user)
    return userlist

def roundRowScore(row, sup=1, inf=-1):
    max = np.max(row)
    min = np.min(row)
    denominator = max - min
    if denominator == 0:
        row[:] = 0
        return row
    else:
        return (sup-inf)*(row - min)/denominator + inf
    
from configparser import ConfigParser
class MyConfigParser(ConfigParser):
    def getlist(self, section, option):
        value = self.get(section, option)
        return list(filter(None, (x.strip() for x in value.replace(" ", "").split(','))))

    def getlistint(self, section, option):
        return [int(x) for x in self.getlist(section, option)]

    def getlistfloat(self, section, option):

        return [float(x) for x in self.getlist(section,option)]
