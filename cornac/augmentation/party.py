import requests
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException


@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 attempts
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff from 2s to 10s
    retry=retry_if_exception_type(RequestException)  # Retry on any request-related exception
)
def get_english_label(search_string, language):
    """
    Get the English label for a given search string from Wikidata.

    Parameters:
        search_string (str): The string to search for.
        language (str): The language code in which the search should be performed.

    Returns:
        str: The English label or None if not found.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": search_string,
        "language": language
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        data = response.json()
        return data['search'][0]['label'] if data.get('search') else None
    except RequestException as e:
        raise ValueError(f"Request error occurred for '{search_string}' in language '{language}': {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred while processing '{search_string}' in language '{language}': {e}") 


def get_party(ne_list, lang, lookup_parties):
    """Enhance the dataset with political parties using named entity tags and an external lookup table.

    Parameters
    ----------
    ne_list (list): List of dictionaries with enriched named entities.
    lang (str): Language code for translation.
    lookup_parties (dict): Pre-existing party translations.

    Returns
    -------
    parties : dict
        A dictionary where keys are political parties appeared in the text and values are their frequencies.
    """
    parties = {}
    try:
        if not isinstance(ne_list, list):
            raise ValueError(f"Error: when extraing party, expected ne_list to be a list, but got {type(ne_list)} instead.")


        for entity in ne_list:
            if isinstance(entity, dict):
                for entity_name, entity_dict in entity.items():
                    if isinstance(entity_dict, dict):
                        if 'party' in entity_dict:
                            if entity_dict['party']:
                                for party in entity_dict['party']:
                                    parties[party] = parties.get(party, 0) + entity_dict.get('frequency', 1)
                        if 'ideology' in entity_dict:
                            if entity_dict['ideology']:
                                parties[entity_name] = parties.get(entity_name, 0) + entity_dict.get('frequency', 1)

        to_translate = {party: freq for party, freq in parties.items() if party not in lookup_parties}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(get_english_label, party, lang): party for party in to_translate}
            for future in futures:
                original_party = futures[future]
                try:
                    translated_party = future.result() or original_party
                except Exception:
                    translated_party = original_party  # Fallback on failure
                lookup_parties[original_party] = translated_party
                lookup_parties[translated_party] = translated_party

        for party in list(parties.keys()):
            translated_party = lookup_parties[party]
            if translated_party != party:
                parties[translated_party] = parties.get(translated_party, 0) + parties.pop(party)

    except ValueError as ve:
        raise ve  
    except Exception as e:
        raise RuntimeError(f"Error in get_party function: {e}")

    return parties, lookup_parties
