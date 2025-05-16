import numpy as np

from ..dynamic_reranker import DynamicReRanker
import pandas as pd

from ..user_simulator import UserSimulator
import json


class DynamicAttrReRanker(DynamicReRanker):
    """
    Dynamic Attribute Penalization Re-ranking (Dyn-AP).
    """

    def __init__(
        self,
        name="DynamicAttrReRanker",
        item_dataframe=None,
        config_file=None,
        party_category_json_path=None,
        user_item_history = None,
        rerankers_item_pool = None,
        **kwargs,
    ):
        """
        Initialize the DynamicAttrReRanker with item metadata and diversity-related configurations.

        Parameters:
        - `name` (str): Name of the reranker (default: "DynamicAttrReRanker").
        - `item_dataframe` (pd.DataFrame): DataFrame containing item metadata for re-ranking.
        - `diversity_dimension` (list): List of attributes to consider for diversity-aware categorization.
        - `top_k` (int): Number of items to recommend in the re-ranked list (default: 10).
        - `pool_size` (int): Size of the item pool for candidate selection (-1 to include all items).
        - `feedback_window_size` (int): Number of iterations to consider for feedback-based updates.
        - `bin_edges` (dict): Dictionary specifying bin edges for continuous attributes.
            Example:
                bin_edges = {'sentiment': [-1, -0.5, 0, 0.5, 1]}
        - `user_choice_model` (str): User choice simulation model (default: "logarithmic_rank_bias").
            Available options:
                - `"logarithmic_rank_bias"`
                - `"preference_based_bias"`
        - `user_simulator_config_path` (str): Path to the configuration file (INI format) for the user simulator.
            An example configuration file includes the following sections and parameters:

            [simulation]
            seed = 42

            [clicked_each_iteration]
            very_active = 5
            active = 4
            less_active = 3

            [frequency_thresholds]
            very_active = 4
            active = 2
            less_active = 0

            [total_reads_thresholds]
            very_active = 30
            active = 15
            less_active = 0

            [max_iterations]
            very_active = 5
            active = 3
            less_active = 1

        - `config_file` (str): Path to the configuration file.
        Initializes:
        - `attribute_frequencies` (list): Tracks the frequency of categorized attributes in the item data.
        - `categorized_items` (list): Organizes items into categories based on diversity dimensions.
        - `ranked_items` (dict): Stores the dynamically re-ranked items for each user.
        

        - party_category:
   [
    {
        "name": "GOV_ONLY",
        "type": "single",
        "keywords": ["gov_party"]
    },
    {
        "name": "OPP_ONLY",
        "type": "single",
        "keywords": ["opp_party"]
    },
    {
        "name": "GOV_OPP_COMPOSITION",
        "type": "composition",
        "keywords": [["gov_party"], ["opp_party"]]
    },
    {
        "name": "MINORITY",
        "type": "minority",
        "keywords": ["gov_party", "opp_party"]
    },
    {
        "name": "NO_PARTY",
        "type": "no_party",
        "keywords": []
    }
]

        
        
        """

        super().__init__(name=name, item_dataframe=item_dataframe,
                         config_file=config_file, user_item_history = user_item_history, rerankers_item_pool = rerankers_item_pool,  **kwargs)
        # Initialize additional attributes
        # self.user_item_history = user_item_history
        # self.rerankers_item_pool = rerankers_item_pool

        self.party_category_json_path = party_category_json_path
        party_related_keywords = ['parties', 'party',  'entities', 'entity']

        for attr in self.diversity_dimension:
            if attr.lower() in party_related_keywords:  
                self.load_party_categories()
        
        self.attribute_frequencies = self.initialize_attribute_frequencies()
        # print(f"self.attribute_frequencies :{self.attribute_frequencies}")
        self.categorized_items = self.get_items_by_category()
        # print(f"self.categorized_items :{self.categorized_items}")
        self.ranked_items = {}


    
    def initialize_attribute_frequencies(self):
        """
        Initializes attribute frequencies for categorical attributes based on unique values, bins, or party mentions in the DataFrame.

        Returns:
        - `attribute_frequencies` (list): A list of dictionaries, each tracking frequencies 
        for categories or bins of a specific attribute in `diversity_dimension`.
        """
        attribute_frequencies = []
        
        # Initialize party-related attributes based on party_categories
        party_related_keywords = ['parties', 'party',  'entities', 'entity']

        for attr in self.diversity_dimension:
            if attr in party_related_keywords:  
                party_freq_dict = {}  # <-- party related dict 
        
                for party_category in self.party_categories:
                    key = f"party_{party_category['name']}"
                    party_freq_dict[key] = 0  # Add each key

                attribute_frequencies.append(party_freq_dict)  # party dimension added

                # for party_category in self.party_categories:
                #     if party_category['type'] == 'single':
                #         # For single party case (e.g., 'GOV_ONLY', 'OPP_ONLY')
                #         attribute_frequencies.append({
                #             f"party_{party_category['name']}": 0
                #         })
                #     elif party_category['type'] == 'composition':
                #         # For composition case (e.g., 'GOV_OPP_BOTH')
                #         attribute_frequencies.append({
                #             f"party_{party_category['name']}": 0
                #         })
                #     elif party_category['type'] == 'minority':
                #         # For minority case (e.g., 'MINORITY')
                #         # in this case, at least one MINORITY party is mentioned, regardless of whether majority party is mentioned
                #         attribute_frequencies.append({
                #             f"party_{party_category['name']}": 0
                #         })
                #     elif party_category['type'] == 'no_party':
                #         # For no party case (e.g., 'NO_PARTY')
                #         attribute_frequencies.append({
                #             # "party_None": 0
                #             f"party_{party_category['name']}": 0
                #         })
            
       
            elif attr in self.item_dataframe:
                attr_dict = {}
                if attr in self.bin_edges:
                    bin_labels = [
                        f"{attr}_{bin_edge}" for bin_edge in self.bin_edges[attr][:-1]]
                    for label in bin_labels:
                        attr_dict[label] = 0
                else:
                    unique_values = self.item_dataframe[attr].dropna().unique()
                    for value in unique_values:
                        attr_dict[f"{attr}_{value}"] = 0
                attribute_frequencies.append(attr_dict)

        return attribute_frequencies

    def load_party_categories(self):
        """Load party categories from a JSON file path."""
        if self.party_category_json_path is not None:
            try:
                with open(self.party_category_json_path, 'r') as f:
                    self.party_categories = json.load(f)
                print("Party categories loaded from the file.")
            except Exception as e:
                # print(f"Error loading party categories from file: {e}")
                # self.party_categories = {}
                raise RuntimeError(f"Failed to load party categories from '{self.party_category_json_path}': {e}")
        else:
            raise ValueError("Error: 'party_category_json_path' is required.")

    

    def categorize_party(self, party_list):
        """
        Categorize based on party_list.

        :param party_list: list of party mentions.
        :return: party category string.
        """
        # Normalize input party list
        # party_set = set(p.lower() for p in party_list) if party_list else set()
        party_set = set(
            str(p).strip().lower()
            for p in party_list
            if p is not None and str(p).strip()
        ) if party_list else set()


        # List of valid types (keywords for matching in 'cat['type']')
        valid_type_words = ['only', 'minority', 'composition', 'no_party','no party','no parties','no_parties']


        # Validate 'party_categories' structure 
        for cat in self.party_categories:
            # Validate 'type' field: It must be a string
            if not isinstance(cat.get('type'), str):
                raise ValueError(f"Category '{cat['name']}' has an invalid 'type'. It should be a string.")

            if not any(word in cat.get('type', '').lower() for word in valid_type_words):
                raise ValueError(f"Invalid 'type' in category {cat['name']}: must contain at least one of the following words: {', '.join(valid_type_words)}")
            # Validate 'keywords' field: It must be a list (or list of lists for 'composition' types)
            if not isinstance(cat.get('keywords'), list):
                raise ValueError(f"Category '{cat['name']}' has an invalid 'keywords'. It should be a list.")
            
            # If 'composition' type, 'keywords' must be a list of lists
            if 'composition' in cat['type'] and not all(isinstance(sublist, list) for sublist in cat['keywords']):
                raise ValueError(f"Category '{cat['name']}' has 'keywords' that are not a list of lists for 'composition' type.")

        # Handle no_party first if input is empty
        if not party_set:
            for cat in self.party_categories:
                if any(phrase in cat['type'].lower() for phrase in ['no_party', 'no_parties', 'no party','no parties']):
                    return f"party_{cat['name']}" 

        # Match 'only' types first
        for cat in self.party_categories:
            if  'only' in cat['type']:
                target_set = set(kw.lower() for kw in cat['keywords'])
                # if party_set == target_set:
                #     return f"party_{cat['name']}"
                # Update: Check if party_set is a subset of the target set, and contains no extra parties
                if party_set.issubset(target_set) and len(party_set) > 0:
                    return f"party_{cat['name']}"

        # Match 'minority' type next
        for cat in self.party_categories:
            if 'minority' in cat['type']:
                target_set = set(kw.lower() for kw in cat['keywords'])
                if party_set and any(p not in target_set for p in party_set):
                    return f"party_{cat['name']}"

            # Handle 'composition' type
        for cat in self.party_categories:
            if 'composition' in cat['type']:
                target_sets = [set(kw.lower() for kw in sublist) for sublist in cat['keywords']]
                all_allowed_parties = set([party.lower() for sublist in cat['keywords'] for party in sublist])
                
                # Check that the party_set contains at least one party from each sublist in the composition
                # Also check that no party in party_set belongs to any party outside the allowed target_sets
                if all(any(p in sublist for p in party_set) for sublist in target_sets) and party_set.issubset(all_allowed_parties):
                    return f"party_{cat['name']}"

        # If none matched and party_set is still not empty, classify as 'NO_PARTY'
        for cat in self.party_categories:
            if any(phrase in cat['type'].lower() for phrase in ['no_party', 'no_parties', 'no party','no parties']):
                return f"party_{cat['name']}"
        
        


    def get_items_by_category(self):
        """
        Categorizes items in the item_dataframe based on the definitions in attribute_frequencies.

        Returns:
        list of dicts: Each dictionary corresponds to an attribute, with keys for each category
                    and values as lists of item IDs that fall into each category.
        """
        # Prepare a list of dicts mirroring the structure of attribute_frequencies with empty lists for collecting item IDs
        categorized_items = [{key: [] for key in attr_freq.keys()}
                             for attr_freq in self.attribute_frequencies]

        # Iterate over each item in the dataframe
        for idx, row in self.item_dataframe.iterrows():
            # Iterate over each attribute frequency dictionary
            for attr_index, attr_freq in enumerate(self.attribute_frequencies):
                attr_name = self.diversity_dimension[attr_index]
                item_feature = row[attr_name]

                # Handle categorization for parties differently
                if (attr_name in ['party', 'entities', 'entity', 'parties']) and isinstance(item_feature, list):
                    category = self.categorize_party(item_feature)
                    if category in attr_freq:
                        categorized_items[attr_index][category].append(idx)
                    continue  # Skip further processing for party attributes

                # Determine the appropriate category key
                if attr_name in self.bin_edges and pd.notna(item_feature):
                    # Binning for continuous attributes
                    bins = self.bin_edges[attr_name]
                    bin_index = np.digitize(
                        [item_feature], bins, right=False)[0] - 1
                    # Ensure the index is within the range
                    bin_index = max(0, min(bin_index, len(bins) - 1))
                    category_key = f"{attr_name}_{bins[bin_index]}"
                else:
                    # Direct category assignment for categorical attributes
                    category_key = f"{attr_name}_{item_feature}"

                if category_key in attr_freq:
                    categorized_items[attr_index][category_key].append(idx)
        # print(f"categorized_items:{categorized_items}")
        return categorized_items

    def add_user(self, user_id):
        """
        Add a UserSimulator to the recommender system.

        Parameters:
        - `user_id` (int): The ID of the user to add.
        """

        # For the 'preference_based_bias' choice model, simulate click behavior influenced by user preferences.
        # User preferences are inferred using the 'infer_user_preference' method.
        # The UserSimulator is initialized with user-specific preferences and item-category mappings.
        if self.user_choice_model == 'preference_based_bias':
            user_preference = self.infer_user_preference(user_id)

            simulated_user = UserSimulator(
                user_id=user_id, user_history=self.user_history[user_id], choice_model='preference_based_bias', config_path=self.user_simulator_config_path,  preference=user_preference, attribute_items_mapping=self.categorized_items)

         # Simulate user click probability based on item position as default behavior.
        else:
            simulated_user = UserSimulator(
                user_id=user_id, user_history=self.user_history[user_id], config_path=self.user_simulator_config_path)
        self.users[user_id] = simulated_user

    def infer_user_preference(self, user_id):
        """
        Infer user preferences based on their interaction history and attribute frequencies.

        Parameters:
        - `user_id` (int): The ID of the user.

        Returns:
        - `preferences` (list): A list of dictionaries where each dictionary represents 
        the user's preference for categories within an attribute.
        """
        user_attribute_count = self.initialize_attribute_frequencies()
        history_item_ids = self.user_history[user_id]

        for attr_idx, attr_freq_dict in enumerate(user_attribute_count):
            for category, item_ids in self.categorized_items[attr_idx].items():

                # Check how many clicked_items are in the list of item IDs for this category
                count = len(set(history_item_ids) & set(item_ids))
                # Update the frequency for this category
                user_attribute_count[attr_idx][category] += count

        preferences = []

        for attr_dict in user_attribute_count:
            if attr_dict:
                all_freq = sum(attr_dict.values())
                preferences_dict = {key: (freq / all_freq if all_freq > 0 else 0)for key,
                                    freq in attr_dict.items()}
            else:
                # when no data exists
                preferences_dict = {key: 0 for key in attr_dict}
            preferences.append(preferences_dict)
        return preferences

    def update_frequencies(self, clicked_items):
        """
        Update attribute frequencies in `self.attribute_frequencies` based on items clicked 
        by the user during the last iteration.

        This method updates the frequency counts for each attribute category in 
        `self.attribute_frequencies` using precomputed categories that map item IDs 
        to their corresponding attribute values.

        Parameters:
        - clicked_items (list): List of item IDs clicked by the user.

        Effect:
        - Modifies `self.attribute_frequencies` to reflect the updated frequencies for each 
        category based on the clicked items.
        """
        # Iterate through each attribute's frequency dictionary in the attribute_frequencies list
        for attr_idx, attr_freq_dict in enumerate(self.attribute_frequencies):
            for category, item_ids in self.categorized_items[attr_idx].items():
                # Check how many clicked_items are in the list of item IDs for this category
                count = len(set(clicked_items) & set(item_ids))
                # Update the frequency for this category
                self.attribute_frequencies[attr_idx][category] += count

    def calculate_penalization(self):
        """
        Calculate penalization factors for each attribute based on their frequencies.

        Returns:
        list of dicts: A list of dictionaries, each corresponding to an attribute in diversity_dimension, 
                    with penalization factors for each category.
        """
        penalizations = []
        for attr_dict in self.attribute_frequencies:
            if attr_dict:
                # Find the max frequency for the current attribute
                max_freq = max(attr_dict.values(), default=1)
                penalization_dict = {
                    key: (freq / max_freq) if max_freq > 0 else 0 for key, freq in attr_dict.items() if isinstance(freq, (int, float))}
            else:
                penalization_dict = {}  # Handle cases where there might be no data for an attribute
            penalizations.append(penalization_dict)

        return penalizations

    def diversityScore(self, candidate_items, penalizations):
        """
        Compute new diversity scores for items based on penalizations.

        Parameters:
        candidate_items (list):  List of item IDs.
        penalizations (list of dicts): A list of dictionaries, corresponding to the penalization to an attribute in diversity_dimension.

        Returns:
        list: List of diversity scores for the candidate item IDs.
        """

        # scores = np.zeros(len(candidate_items), len(self.diversity_dimension))
        scores = np.zeros((len(candidate_items), len(self.diversity_dimension)))

        candidate_set = set(candidate_items)

        # Iterate through each attribute dimension and apply penalization
        for attr_idx, attr_name in enumerate(self.diversity_dimension):
            penalization_dict = penalizations[attr_idx]

            for category, penalization_value in penalization_dict.items():
                category_items = set(
                    self.categorized_items[attr_idx].get(category, []))
                # Find the intersection between candidate items and the category
                matched_items = category_items & candidate_set

                if matched_items:
                    # Vectorized way to update scores based on penalization
                    matched_indices = [candidate_items.index(
                        item) for item in matched_items]
                    scores[matched_indices, attr_idx] += (1 - penalization_value)
        # print(f"scores are:{scores}")
        return scores

    def update_recommendations(self, user_id, clicked_items):
        """
        Update recommendations dynamically based on user feedback and diversity scores.

        Parameters:
        - `user_id` (int): The ID of the user.
        - `clicked_items` (list): List of items clicked by the user.

        Returns:
        - `top_k_items` (list): List of top-k recommended items after updating recommendations.
        """
        super().update_recommendations(user_id)
        self.attribute_frequencies = self.initialize_attribute_frequencies()
        self.update_frequencies(clicked_items)

        penalizations = self.calculate_penalization()

        candidate_items = self.candidate_items[user_id]
        scores = self.diversityScore(candidate_items, penalizations)

        recommended_items = []
        recommended_items_set = set()

        while len(recommended_items) < self.top_k:
            added_in_this_round = False
            for idx, item in enumerate(candidate_items):
                if item in recommended_items_set:
                    continue  

                # Multiply scores across dimensions
                item_score = np.prod(scores[idx])
                # print(f"scores[idx]:{scores[idx]}")
                # print(f"item_score:{item_score}")

                # Roll a random chance
                if np.random.rand() < item_score:
                    # print(f"add item :{item}")
                    recommended_items.append(item)
                    recommended_items_set.add(item)
                    added_in_this_round = True

                # Break if we reached desired size
                if len(recommended_items) >= self.top_k:
                    break

            #  to avoid infinite loop if scores are all too low

            if not added_in_this_round:
                # Get remaining candidates, sorted by highest score
                remaining_candidates = [
                    (idx, item) for idx, item in enumerate(candidate_items)
                    if item not in recommended_items_set
                ]
                score_values = np.prod(scores, axis=1)

                # Sort by descending score
                remaining_candidates.sort(
                key=lambda x: score_values[x[0]],
                reverse=True
                )


                for idx, item in remaining_candidates:
                    # print(f"fill item by score :{item}")
                    recommended_items.append(item)
                    recommended_items_set.add(item)

                    if len(recommended_items) >= self.top_k:
                        break
                
                break

            
    
        return recommended_items



    def rerank(self, user_idx, interaction_history=None, candidate_items=None, prediction_scores=None, filtering_rules: dict = None, recommendation_list=[],
              **kwargs ):
        """
        Perform dynamic re-ranking for a user by iteratively updating recommendations based on feedback.

        Parameters:
        - `user_idx` (int): ID of the user receiving recommendations.
        - `interaction_history` (cornac.data.Dataset): User-Item preference data for filtering items.
        - `candidate_items` (list): List of candidate items to rank.
        - `prediction_scores` (list): Scores of the candidate items (optional).
        - `filtering_rules` (dict): Rules for filtering items (optional).
        - `recommendation_list` (list): Existing recommendation list to perform initial user click simulation.
    

        Returns:
        - `result` (dict): Iteration-wise updated ranked lists for the user.
        """
        super().rerank(user_idx=user_idx, interaction_history=interaction_history, candidate_items=candidate_items,
                       prediction_scores=prediction_scores, **kwargs)
        self.execute_filters(
            user_idx=user_idx, filtering_rules=filtering_rules)
        self.filter_items_in_additional_history(user_idx)  ## for Mind setup
        self.add_user(user_idx)
        user = self.users[user_idx]
        result = {}
        num_iterations = self.users[user_idx].max_iteration
        recommendation_list = recommendation_list[:self.top_k] if len(
            recommendation_list) > self.top_k else recommendation_list
        for iteration in range(num_iterations):
            if len(self.candidate_items[user_idx]) < user.clicked_each_iteration:
                break
            # Simulate user interactions (clicks)
            clicks = user.simulate_interaction(
                recommendation_list)
            if len(clicks) == 0:
                break
            else:
                # Get feedback from the last N iterations (if available)
                N = self.feedback_window_size
                interacted_items = user.interacted_items[-N:] if len(
                    user.interacted_items) >= N else user.interacted_items

                # Flatten the list of interacted items from the last N iterations
                interacted_items_flat = [
                    item for sublist in interacted_items for item in sublist]

                # Update the recommender model with feedback

                recommendation_list = self.update_recommendations(
                    user_idx, interacted_items_flat)
                result[iteration] = recommendation_list
                
        self.ranked_items[user_idx] = result
        # print(f"dynamic:{self.name} recommended for user:{user_idx}:{result}")
        return result
