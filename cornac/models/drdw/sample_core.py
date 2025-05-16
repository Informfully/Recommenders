import numpy as np
import pandas as pd
from scipy.optimize import linprog
import math
from scipy.sparse import csc_matrix
import ast

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
        # # this part handles the case when te list of parties is loaded from csv file.
        # # because test_str will be: "["republican",...]", should remove first character " and last character "
        try:
            parties = ast.literal_eval(test_str)  # directly evaluate the string as a list
            # Ensure the evaluated result is actually a list
            if not isinstance(parties, list):
                return []
        except:
            # If eval fails, return an empty list or handle error as needed
            return []
        return parties



    # If the input format is not recognized, return an empty list
    return []

def is_valid_party_list(x):
    # Acceptable: None, NaN, empty list, or list of strings
    if x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, list) and len(x) == 0):
        return True
    if isinstance(x, list):
        return all(isinstance(i, str) for i in x)
    return False

class DistributionSampler:
    """
    A utility class for sampling items from a dataset based on target distributions.

    This class supports discrete, continuous, and categorical (e.g., party-based) attributes,
    enabling efficient selection of items that match specific distributions.
    """

    def __init__(self, item_dataframe):
        self.item_dataframe = item_dataframe
        # store computed items per category
        self.target_num_items_per_category = {}

    def _generate_cache_key(self, key_type, feature_dim, target_proportion):
        """
        Generate a cache key based on the key type, description, and the structure of target_proportion.
        This ensures unique caching for continuous attributes, discrete attributes, and party classifications.

        Parameters:
        -----------
        key_type : str
            The type of computation ('discrete', 'continuous', or 'party').
        feature_dim : str
            The feature dimension or attribute (e.g., 'category', 'age_range').
        target_proportion : dict or list of dict
            The target attribute distribution (for discrete) or list of dictionaries for continuous or party classifications.

        Returns:
        --------
        str
            A unique cache key string.
        """
        if key_type == 'discrete':
            # For discrete attributes, use the attribute name and its probability as part of the key
            tar_key = ','.join(
                [f"{k}:{v}" for k, v in sorted(target_proportion.items())])
            return f"{key_type}:{feature_dim}:{tar_key}"

        elif key_type == 'continuous':
            # For continuous attributes, include both the range and the probability
            ranges_key = ','.join(
                [f"{item['min']}-{item['max']}:{item['prob']}" for item in target_proportion])
            return f"{key_type}:{feature_dim}:{ranges_key}"


        elif key_type == 'party':
            def flatten_if_needed(lst):
                flat = []
                for elem in lst:
                    if isinstance(elem, list):
                        flat.extend(elem)  # flatten if element is a list
                    else:
                        flat.append(elem)  # keep as is
                return flat

            party_key = ','.join(
                [f"{item['description']}:{','.join(map(str, flatten_if_needed(item['contain'])))}:{item['prob']}"
                for item in target_proportion]
            )
            return f"{key_type}:{feature_dim}:{party_key}"


        return f"{key_type}:{feature_dim}"

    def items_per_discrete_attribute(self, target_proportion, targetSize, feature_dim):
        """
        Transforms a target attribute distribution (in percentage) into the corresponding
        target number of items for each attribute in the recommended items.

        Parameters:
        -----------
        target_proportion : dict
            A dictionary where keys represent attributes, e.g., for category, "sports", "music", "politics",
            and values represent their target proportions (in percentages) of the total recommendations.
            Example: {"sports": 0.3, "music": 0.1, "politics": 0.6}.
        targetSize : int
            The total number of items to be recommended.
            Example: 100 (meaning 100 items will be recommended).
        feature_dim : str
            The feature dimension or attribute being evaluated, such as "category" or "topic".
            It is used to label each category in the returned dictionary.
            Example: "category".

        Returns:
        --------
        items_per_category : dict
            A dictionary mapping each category (in the format "description,category") to the corresponding
            number of items that should be included in the recommendations, based on the target percentages.
            If rounding issues occur, the function adjusts the category with the largest target number to ensure
            the total matches the targetSize.
            Example: {"category,sports": 30, "category,music": 10, "category,politics": 60}.
        """
        # Generate a unique cache key based on target_proportion, target_size, and description
        cache_key = self._generate_cache_key(
            'discrete', feature_dim, target_proportion)

        # Check if the result is already stored
        if cache_key in self.target_num_items_per_category:
            return self.target_num_items_per_category[cache_key]

        # Check that all distribution values are between 0 and 1
        for key, value in target_proportion.items():
            if not (0 <= value <= 1):
                raise ValueError(
                    f"Distribution value for '{key}' is not between 0 and 1.")
        # Check that the sum of the distribution equals 1
        if not np.isclose(sum(target_proportion.values()), 1.0, atol=1e-8):
            raise ValueError("Sum of the distribution values must equal 1.")

        items_per_category = {}
        totalSize = 0
        fractional_remainders = []
        for x, y in target_proportion.items():
            fractional_items = y * targetSize
            itemNum = np.floor(fractional_items).astype(int)
            remainder = fractional_items - itemNum  # Calculate the remainder

            # Create the category key
            new_x = feature_dim + ','+x
            items_per_category[new_x] = itemNum

            totalSize += itemNum
            # Store the remainder for later adjustment
            fractional_remainders.append((new_x, remainder))

        # Calculate how many items still need to be allocated
        remainder_items_needed = targetSize - totalSize
        # Second pass: distribute the remaining items based on the largest remainders
        if remainder_items_needed > 0:
            # Sort by the largest fractional remainders
            # If all the remainders are tied (i.e., they all have the same value),  will maintain the original relative order of items in the list.
            fractional_remainders.sort(key=lambda x: x[1], reverse=True)
            for i in range(remainder_items_needed):
                # Allocate one more item to the category with the largest remainder
                items_per_category[fractional_remainders[i][0]] += 1

        # Cache the result
        self.target_num_items_per_category[cache_key] = items_per_category
        return (items_per_category)

    def items_per_continous_attribute(self, tarList, targetSize, feature_dim):
        """
        Computes the target number of items for continuous attributes based on a given target distribution.

        This function processes a list of continuous attribute ranges (e.g., age ranges, sentiment ranges) 
        and their associated probabilities, calculating the corresponding number of items that should be 
        included in the recommendations based on the total target size.

        Parameters:
        -----------
        tarList : list of dict
            A list of dictionaries where each dictionary represents a continuous range of an attribute 
            and its associated probability. Each dictionary should have the following keys:
            - 'prob': The probability associated with the continuous range.
            - 'min': The minimum value of the range.
            - 'max': The maximum value of the range.
            Example: [{'prob': 0.2, 'min': 0, 'max': 10}, {'prob': 0.3, 'min': 10, 'max': 20}, ...]

        targetSize : int
            The total number of items to be computed for recommendation. 
            Example: 100 (meaning 100 items will be distributed across the continuous ranges).

        feature_dim : str
            The feature or attribute being evaluated (e.g., "age_range"). 
            This string will be used as a prefix for the keys in the returned dictionary to label 
            the continuous range of the attribute.

        Returns:
        --------
        items_per_category : dict
            A dictionary mapping each continuous attribute range (in the format "description,min,max") 
            to the corresponding number of items that should be computed based on the target percentages. 

            Example: {"age range,0,10": 20, "age range,10,20": 30}

        Notes:
        ------
        - The function ensures that the total number of items assigned across all continuous ranges 
        matches the targetSize. 
        """
        # Generate a unique cache key
        cache_key = self._generate_cache_key(
            'continuous', feature_dim, tarList)
        # Check if the result is already stored
        if cache_key in self.target_num_items_per_category:
            return self.target_num_items_per_category[cache_key]

        # Check that all distribution values are between 0 and 1
        for item in tarList:
            if not (0 <= item['prob'] <= 1):
                raise ValueError(
                    f"Distribution value for range {item['min']}-{item['max']} is not between 0 and 1.")

        # Check that the sum of the distribution equals 1
        if not np.isclose(sum(item['prob'] for item in tarList), 1.0, atol=1e-8):
            raise ValueError("Sum of the distribution values must equal 1.")

        items_per_category = {}
        totalSize = 0
        fractional_remainders = []
        for item in tarList:
            y = item['prob']
            fractional_items = y * targetSize
            x = feature_dim + ','+str(item['min'])+','+str(item['max'])
            itemNum = np.floor(fractional_items).astype(int)
            remainder = fractional_items - itemNum  # Calculate the remainder

            items_per_category[x] = itemNum
            totalSize += itemNum
            # Store the remainder for later adjustment
            fractional_remainders.append((x, remainder))

        # Calculate how many items still need to be allocated
        remainder_items_needed = targetSize - totalSize
        # Second pass: distribute the remaining items based on the largest remainders
        if remainder_items_needed > 0:
            # Sort by the largest fractional remainders
            fractional_remainders.sort(key=lambda x: x[1], reverse=True)
            for i in range(remainder_items_needed):
                # Allocate one more item to the category with the largest remainder
                items_per_category[fractional_remainders[i][0]] += 1

        # Cache the result
        self.target_num_items_per_category[cache_key] = items_per_category
        return (items_per_category)

    def items_per_party_classification(self, tarList, targetSize, feature_dim):
        """
        Computes the target number of items for each party category based on the `tarList`.

        This function processes a list of party-based categories (e.g., "Republican", "Democratic",
        "Republican and Democratic", "Minority party", "No party mentioned") and their associated
        target proportions (in percentages) and calculates the corresponding number of items
        for each party category.
        Parameters:
        -----------
        tarList : list of dict
            A list of dictionaries where each dictionary represents a party-based category
            and its associated probability. Each dictionary should have the following keys:
            - 'prob': The probability associated with the party category.
            - 'description': string, "only mention", "minority but can also mention", or "no parties".
            - 'contain': the relevant parties for 'description'.
            Example:   tarList = [
                {'prob': 0.24, 'description': 'only mention',
                    'contain': ['Republican']},
                {'prob': 0.25, 'description': 'only mention',
                    'contain': ['Democratic']},
                {'prob': 0.30, 'description': 'only mention',
                    'contain': ['Republican', 'Democratic']},
                {'prob': 0.11, 'description': 'minority but can also mention',
                    'contain': ['Republican', 'Democratic']},
                {'prob': 0.1, 'description': 'No parties',
                    'contain': []}
            ]

        targetSize : int
            The total number of items to be computed for recommendation.
            Example: 100 (meaning the total number of items is 100).

        feature_dim : str
            The feature or attribute being evaluated. This string will be used as a prefix
            for the keys in the returned dictionary to label the party category.
            Example: "entities".
        Returns:
        --------
        items_per_category : dict
            A dictionary mapping each party category (in the format "description,party_category")
            to the corresponding number of items that should be computed based on the target
            percentages. 
        Example:  items_per_category = {
                'party,only mention:Republican': 24,
                'party,only mention:Democratic': 25,
                'party,only mention:Republican,Democratic': 30,
                'party,minority but can also mention:Republican,Democratic': 11,
                'party,No parties:': 10
            }
        """
        # Generate a unique cache key
        cache_key = self._generate_cache_key('party', feature_dim, tarList)

        # Check if the result is already stored
        if cache_key in self.target_num_items_per_category:

            return self.target_num_items_per_category[cache_key]
        # Check that all distribution values are between 0 and 1
        for item in tarList:
            if not (0 <= item['prob'] <= 1):
                raise ValueError(
                    f"Distribution value for party {item['description']} is not between 0 and 1.")

        # Check that the sum of the distribution equals 1
        if not np.isclose(sum(item['prob'] for item in tarList), 1.0, atol=1e-4):
            raise ValueError("Sum of the distribution values must equal 1.")

        items_per_category = {}
        totalSize = 0
        fractional_remainders = []
        for item in tarList:
            y = item['prob']
            fractional_items = y * targetSize
            relevant_parties = ",".join(str(x) for x in item['contain'])
            # x = feature_dim + ',' + \
            #     str(item['description'])+':'+relevant_parties

                    # Handle "composition" type (i.e., multiple party groups)
            if item['description'] == "composition":
                # Retain the list format for composition
                x = feature_dim + ',' + str(item['description']) + ':' + str(item['contain'])
            else:
                x = feature_dim + ',' + str(item['description']) + ':' + relevant_parties

            itemNum = np.floor(fractional_items).astype(int)
            items_per_category[x] = itemNum
            totalSize += itemNum
            # Store the remainder for later adjustment
            remainder = fractional_items - itemNum
            fractional_remainders.append((x, remainder))

        # Calculate how many items still need to be allocated
        remainder_items_needed = targetSize - totalSize

        # Second pass: distribute the remaining items based on the largest remainders
        if remainder_items_needed > 0:
            # Sort by the largest fractional remainders
            # when equal: preserves the original order of elements with equal keys.
            fractional_remainders.sort(key=lambda x: x[1], reverse=True)
            for i in range(remainder_items_needed):
                # Allocate one more item to the category with the largest remainder
                items_per_category[fractional_remainders[i][0]] += 1
         # Cache the result
        self.target_num_items_per_category[cache_key] = items_per_category
        return (items_per_category)

    def generateMaskedMatrixDiscrete(self, data, itemPool, targetDimension, items_per_category,  cornacId_to_newId):
        """
            Generate a masked matrix for discrete attributes (e.g., categories) based on the item pool and target dimension.

            Parameters:
            -----------
            data : pd.DataFrame
                DataFrame containing item attributes, filtered for the items in the itemPool.
            itemPool : list or numpy array
                List or array of item IDs (cornac IDs) that are being considered.
            targetDimension : str
                The attribute or feature dimension (e.g., 'category', 'outlet') used for masking.
            items_per_category : dict
                Dictionary mapping each category to the number of items that should be included in the recommendations.
            cornacId_to_newId : dict
                Mapping from original item IDs (cornac IDs) to their positions in the matrix.

            Returns:
            --------
            maskedMatrix : dict
                Dictionary where keys are categories and values are numpy arrays (masked matrix) with 1s and 0s indicating
                whether an item belongs to that category or not.
        """
        # Ensure the column exists
        if targetDimension not in data.columns:
            raise ValueError(f"Column '{targetDimension}' not found in data.")
        
        lowered_column = data[targetDimension].astype(str).str.strip().str.lower()
        maskedMatrix = {}
        for category_key, target_count in items_per_category.items():
            mMatrix = np.zeros(itemPool.shape,  dtype=int)
            # category_name = category_key.split(",")[1]
            try:
                category_name = category_key.split(",")[1].strip().lower()
            except IndexError:
                raise ValueError(f"Invalid category_key format: '{category_key}'")

            ids = data.index[lowered_column == category_name].tolist()

            new_ids = [cornacId_to_newId[item_id]
                       for item_id in ids if item_id in cornacId_to_newId]
            mMatrix[new_ids] = 1
            maskedMatrix[category_key] = mMatrix
        return maskedMatrix

    def generateMaskedMatrixContinous(self, data, itemPool, targetDimension, items_per_category, cornacId_to_newId):
        """
        Generate a masked matrix for continuous attributes (e.g., numerical ranges) based on the item pool and target dimension.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing item attributes, filtered for the items in the itemPool.
        itemPool : list or numpy array
            List or array of item IDs (cornac IDs) that are being considered.
        targetDimension : str
            The attribute or feature dimension (e.g., 'sentiment', 'price') used for masking.
        items_per_category : dict
            Dictionary mapping each continuous range (in the format "description,min,max") to the number of items.
        cornacId_to_newId : dict
            Mapping from original item IDs (cornac IDs) to their positions in the matrix.

        Returns:
        --------
        maskedMatrix : dict
            Dictionary where keys are range descriptions and values are numpy arrays (masked matrix) with 1s and 0s 
            indicating whether an item belongs to that continuous range or not.
        """
        maskedMatrix = {}
        for range_key, target_count in items_per_category.items():
            mMatrix = np.zeros(itemPool.shape, dtype=int)
            str_key = range_key.split(",")
            # Extract the min and max values from the range_key ('description,min,max')
            min_value = float(str_key[1])
            max_value = float(str_key[2])
            ids = data.index[(data[targetDimension] >= min_value) & (
                data[targetDimension] < max_value)].tolist()

            # Convert original item IDs (cornac IDs) to matrix indices using 'cornacId_to_newId'
            new_ids = [cornacId_to_newId[item_id]
                       for item_id in ids if item_id in cornacId_to_newId]
            mMatrix[new_ids] = 1
            maskedMatrix[range_key] = mMatrix
        return maskedMatrix

    def generateMaskedMatrixParties(self, data, itemPool, targetDimension, items_per_category, cornacId_to_newId):
        """
        Generate a masked matrix for party-based classifications based on the item pool and target categories.

        This function creates a dictionary of binary matrices (masks) for each party classification, such as
        "only mention", "minority but can also mention", or "no parties". The masks indicate which items 
        in the pool match the classification criteria.

        Parameters:
        -----------
        data : pd.DataFrame
            A DataFrame containing item attributes, including a column with party-related data ('entities').
        itemPool : list or numpy.ndarray
            A list or array of item IDs (cornac IDs) representing the pool of items being considered.
        targetDimension : str
            The attribute or feature dimension (e.g., 'sentiment', 'price') used for masking.
        items_per_category : dict
            A dictionary mapping each party classification (formatted as "description,party_category") to the target count.
        cornacId_to_newId : dict
            A mapping from original item IDs (cornac IDs) to matrix positions.

        Returns:
        --------
        maskedMatrix : dict
            A dictionary where:
            - Keys are category descriptions (e.g., 'party,only mention:Republican').
            - Values are numpy arrays (binary masks) with 1s indicating items matching the classification and 0s otherwise.

        Notes:
        ------
        - If "only mention" is specified, the mask includes items that only mention the relevant parties.
        - If "minority" is specified, the mask includes items that mention additional parties besides the target set.
        - If "no parties" or "no party" is specified, the mask includes items with no party mentions.
        """
        maskedMatrix = {}
        cleanedData = data[targetDimension].apply(processPartyData)
        # Validate entries
        invalid_entries = cleanedData[~cleanedData.apply(is_valid_party_list)]
        if not invalid_entries.empty:
            raise ValueError(f"Invalid entries in '{targetDimension}': all non-empty lists must contain only strings.\n Unexpected entries:\n{invalid_entries}")

        # Step 3: Normalize to lowercase
        cleanedData = cleanedData.apply(
            lambda x: [s.lower() for s in x] if isinstance(x, list) and len(x) > 0 else x
        )
        valid_party_type_words = ['only', 'minority', 'composition', 'no_party','no party','no parties','no_parties']
        for category_key, target_count in items_per_category.items():
            mMatrix = np.zeros(itemPool.shape)
            # Parse the category_key (e.g., 'party,only mention:Republican,Democratic')
            description_part, party_info = category_key.split(":")
            # 'only mention' or 'minority' or 'no parties'
            descriptor = description_part.split(",")[1].lower()
            if not any(word in descriptor for word in valid_party_type_words):
                raise ValueError(f"Invalid {descriptor},{party_info}: must contain at least one of the following words: {', '.join(valid_party_type_words)}")
            # relevant_parties = party_info.split(
            #     ",")  # List of relevant parties
            if 'composition' in descriptor:
                # Extract the parties list from the string (e.g., [['party6'], ['party9']])
                # sublists = eval(party_info)
                sublists = ast.literal_eval(party_info)
                if isinstance(sublists, list):
                    if not all(isinstance(sublist, list) for sublist in sublists):
                        raise ValueError(f"For 'composition' descriptor, 'contain' must be a list of lists. Received: {sublists}")
                    target_sets_composition = [set(kw.lower() for kw in sublist) for sublist in sublists]
                    # Flatten the lists to create a set of all allowed parties
                    all_allowed_parties = set([party.lower() for sublist in sublists for party in sublist])  # Normalize to lowercase
                
                else:
                    raise ValueError(f"For 'composition' descriptor, 'contain' must be a list of lists. Received: {sublists}")
                
               
            else:
                # For other cases, split the parties normally
                relevant_parties = party_info.split(",")  # List of relevant parties
                # relevant_parties = set(relevant_parties)
                relevant_parties = set(party.lower() for party in relevant_parties)

            
            # Validation: relevant_parties must be non-empty for 'only' and 'minority'
            if ("only" in descriptor or "minority" in descriptor) and len(relevant_parties) == 0:
                raise ValueError(f"For category '{category_key}', 'only' or 'minority' descriptor must have at least one relevant party in 'contain'.")
                    
             # Loop over the cleaned data to apply the filtering logic based on the descriptor
            if "composition" in descriptor:
                # Ensure at least one party from each sublist is mentioned and no other parties outside allowed
                ids = [
                    index for index, value in enumerate(cleanedData)
                    if value is not None
                    and not (isinstance(value, float) and math.isnan(value))
                    and all(len(set(value).intersection(set(sublist))) > 0 for sublist in target_sets_composition)  # At least one from each sublist
                    and set(value).issubset(all_allowed_parties)  # No party outside the allowed sublists
                ]
                

            if "only" in descriptor:

                ids = [
                        index for index, value in enumerate(cleanedData)
                        if value is not None
                        and not (isinstance(value, float) and math.isnan(value))
                        and set(value).issubset(relevant_parties)
                        and len(set(value)) > 0  # Ensure at least one party mentioned
                    ]
           
            elif "minority" in descriptor:

                ids = [index for index, value in enumerate(
                    cleanedData) 
                    if value is not None 
                    and not (isinstance(value, float) and math.isnan(value)) 
                    and len(set(value)) > 0
                    and len(set(value).difference(relevant_parties)) > 0]

    
            elif "no parties" in descriptor or "no party" in descriptor  or "no_party" in descriptor or  "no_parties" in descriptor:
                
                ids = [
                        index for index, value in enumerate(cleanedData)
                        if value is None
                        or (isinstance(value, float) and math.isnan(value))
                        or (isinstance(value, str) and value.strip() == "")
                        or (isinstance(value, list) and len(value) == 0)
                    ]
            new_ids = [cornacId_to_newId[item_id]
                       for item_id in data.index[ids].tolist() if item_id in cornacId_to_newId]
            mMatrix[new_ids] = 1
            maskedMatrix[category_key] = mMatrix

        return maskedMatrix

    def prepareLinearProgramming(self, df, itemPool,  targetDimension, targetDistributions, targetSize):
        """
        Prepare matrices for linear programming based on target dimensions
        and distributions, and setting up the index mappings between original items and their matrix positions.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing item attributes.
        itemPool : list or numpy array
            List or array of item IDs (cornac IDs) that are being considered.
        targetDimension : list
            List of target dimensions or attributes (e.g., 'category', 'sentiment') for filtering the item pool.
        targetDistributions : list of dict
            List of target distributions to be applied to the item pool.
        targetSize : int
            The total number of items to be selected.

        Returns:
        --------
        super_dict_matrix : dict
            A dictionary containing masked matrices for each target distribution.
        super_dict_number : dict
            A dictionary with the number of items per category.
        newId_to_cornacId : dict
            Mapping from matrix index to original item ID.
        cornacId_to_newId : dict
            Mapping from original item ID to matrix index.
        """

        # Ensure that itemPool is a numpy array for efficient processing
        originalIndex = np.asarray(itemPool)

        # Filter data to only include items in itemPool
        data = df.loc[originalIndex]

        # Map the index in itemPool (0, 1, ..., len(itemPool) - 1) to original item IDs
        newIndex = np.arange(len(originalIndex))
        newId_to_cornacId = dict(enumerate(originalIndex))
        cornacId_to_newId = dict(zip(originalIndex, newIndex))
        super_dict_matrix = {}
        super_dict_number = {}

        for i in range(len(targetDistributions)):
            targetDistribution = targetDistributions[i]

            if targetDistribution["type"] == "discrete":
                tar = targetDistribution["distr"]
                items_per_category = self.items_per_discrete_attribute(
                    tar, targetSize, targetDimension[i])
                masked_matrix_dict = self.generateMaskedMatrixDiscrete(
                    data, itemPool, targetDimension[i], items_per_category, cornacId_to_newId)  # masked matrix
                super_dict_matrix.update(masked_matrix_dict)
                super_dict_number.update(items_per_category)
            elif targetDistribution["type"] == "continuous":

                # a list of distributions
                tarList = targetDistribution["distr"]
                items_per_category = self.items_per_continous_attribute(
                    tarList, targetSize, targetDimension[i])
                masked_matrix_dict = self.generateMaskedMatrixContinous(
                    data, itemPool, targetDimension[i], items_per_category, cornacId_to_newId)  # masked matrix
                super_dict_matrix.update(masked_matrix_dict)
                super_dict_number.update(items_per_category)
            elif targetDistribution["type"] == "parties" or targetDistribution["type"] == "party" or targetDistribution["type"] == "entities" or targetDistribution["type"] == "entity":
                # a list of distributions
                tarList = targetDistribution["distr"]
                items_per_category = self.items_per_party_classification(
                    tarList, targetSize, targetDimension[i])
                masked_matrix_dict = self.generateMaskedMatrixParties(
                    data, itemPool, targetDimension[i], items_per_category, cornacId_to_newId)  # masked matrix
                super_dict_matrix.update(masked_matrix_dict)
                super_dict_number.update(items_per_category)
        
        # print("completed prepareLinearProgramming")
        return (super_dict_matrix, super_dict_number, newId_to_cornacId, cornacId_to_newId)

    def sample_by_multi_distributions(self, itemPool,  targetDimension, targetDistributions, targetSize, Objective_to_be_minimized):
        """
        Samples items from the pool based on multiple target distributions using linear programming.

        Parameters
        ----------
        itemPool : list or numpy.ndarray
            Pool of candidate items.

        targetDimension : list
            Dimensions to evaluate.

        targetDistributions : list of dict
            Target distributions for sampling.

        targetSize : int
            Total number of items to select.

        Objective_to_be_minimized : numpy.ndarray
            Objective function coefficients for optimization.

        Returns
        -------
        tuple
            - dict : Item counts per category.
            - list : Selected item indices.
        """

        if not isinstance(Objective_to_be_minimized, np.ndarray):
            print(f"Invalid type for 'c': {type(Objective_to_be_minimized)}")
            return {}, []

        if np.ndim(Objective_to_be_minimized) != 1:
            print(f"Invalid input: 'c' must be a 1-D array.")
            print(
                f"Type of 'c': {type(Objective_to_be_minimized)}, Shape of 'c': {np.shape(Objective_to_be_minimized)}")
            return {}, []  # Skip computation for this invalid input

        # Objective_to_be_minimized: the objective function, e.g., aim to minimize CX according to a vector C.
        totalData = self.item_dataframe
        super_dict1, super_dict2, newId_to_cornacId, cornacId_to_newId = self.prepareLinearProgramming(
            totalData, itemPool,  targetDimension, targetDistributions, targetSize)

        all_constraints = []
        all_b_value = []
        for key, value in super_dict1.items():
            constraints = value
            b_value = super_dict2[key]
            all_constraints.append(constraints)
            all_b_value.append(b_value)

        all_constraints.append(np.ones(itemPool.shape[0]))
        all_b_value.append(targetSize)

        all_constraints = np.concatenate([all_constraints], axis=0)

        bound = (0, 1)
        # print("complete linear programming prepare")
        # print(f"Objective_to_be_minimized:{Objective_to_be_minimized}")
        # print(f"all_constraints:{all_constraints}")
        # print(f"all_b_value:{all_b_value}")
        # print(f"bound:{bound}")
        # Convert constraints to sparse matrix
        A_eq_sparse = csc_matrix(all_constraints)
        try:
            # res = linprog(c=Objective_to_be_minimized, A_ub=None,  b_ub=None,
            #               A_eq=all_constraints, b_eq=all_b_value, bounds=bound, integrality=3)
            res = linprog(c=Objective_to_be_minimized, A_ub=None, b_ub=None,
              A_eq=A_eq_sparse, b_eq=all_b_value, bounds=bound,
              method="highs-ipm")  # Use "highs" solver
            if  res.success  and res.x is not None:
                # print("solved interger programming")
                indices = np.where(res.x == 1)[0]
                cornac_index = [newId_to_cornacId[k] for k in indices.tolist()]
            else:
                cornac_index = []
            # print("complete integer programming computation")
            return (super_dict2, cornac_index)
        except ValueError as ve:
            print(f"Exception caught: {ve}")
            print(f"Invalid input 'c': {Objective_to_be_minimized}")
            return {}, []
