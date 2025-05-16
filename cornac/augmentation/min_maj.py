def get_min_maj_ratio(ne_list, **kwargs):
    """
    Enhance the dataset with a minority-majority score using named entity tags extended by Wikidata.

    Parameters
    ----------
    ne_list: list
        A list of dictionaries, where each dictionary's key is the name of a person appearing in the text, and values are another dictionary extended by Wikidata.

    Returns
    -------
    ratio: dictionary
        A dictionary where keys are score types and values are a list of minority and majority ratios.
    """
    # Extract the majority groups from keyword arguments
    major_genders = kwargs['major_gender']
    major_citizens = kwargs['major_citizen']
    major_ethnicities = kwargs['major_ethnicity']
    major_place_of_births = kwargs['major_place_of_birth']

    # Initialize count dictionary to track minority and majority counts for different categories
    count = {'gender': [0, 0], 'ethnicity': [0, 0], 'mainstream': [0, 0]}
    ratio = {}

    # Check if ne_list is a valid iterable
    if not isinstance(ne_list, list):
        raise TypeError(f"Invalid input: Expected a list for 'ne_list', but received {type(ne_list).__name__}.")
        # print("Error: ne_list is not a list. Received:", type(ne_list))
        # return {}  # Return an empty dictionary if ne_list is not valid

    # Iterate through each entity in the named entity list
    for entity in ne_list:
        if not isinstance(entity, dict):
            continue  # Skip if entity is not a dictionary

        for entity_name, entity_dict in entity.items():
            if not isinstance(entity_dict, dict) or 'key' not in entity_dict:
                continue  # Skip if entity_dict is not a dictionary or doesn't contain 'key'

            # Calculate gender score (male as majority, others as minority)
            if 'gender' in entity_dict and len(entity_dict['gender']) == 1:
                if entity_dict['gender'][0] in major_genders:
                    count['gender'][1] += entity_dict.get('frequency', 1)
                else:
                    count['gender'][0] += entity_dict.get('frequency', 1)

            # Calculate Ethnicity score (people with a 'United States' ethnicity or place of birth are majority)
            if 'citizen' in entity_dict:
                loop_break = False
                ethnicity_match = False
                place_of_birth_match = False

                for major_citizen in major_citizens:
                    if major_citizen in entity_dict['citizen']:
                        loop_break = True

                        for major_ethnicity in major_ethnicities:
                            if (major_ethnicity in entity_dict.get('ethnicity', [])) or not entity_dict.get('ethnicity'):
                                ethnicity_match = True

                        for major_place_of_birth in major_place_of_births:
                            if (major_place_of_birth in entity_dict.get('place_of_birth', [])) or not entity_dict.get('place_of_birth'):
                                place_of_birth_match = True

                        if ethnicity_match and place_of_birth_match:
                            count['ethnicity'][1] += entity_dict.get('frequency', 1)
                            break

                        count['ethnicity'][0] += entity_dict.get('frequency', 1)
                        break

                if not loop_break:
                    count['ethnicity'][0] += entity_dict.get('frequency', 1)

            # Calculate mainstream score by checking given name
            if 'givenname' in entity_dict:
                count['mainstream'][1] += entity_dict.get('frequency', 1)
            else:
                count['mainstream'][0] += entity_dict.get('frequency', 1)

    # Calculate ratios based on counts
    for k, v in count.items():
        total = v[0] + v[1]
        if total != 0:
            ratio[k] = [round(v[0] / total, 4), round(v[1] / total, 4)]

    return ratio
