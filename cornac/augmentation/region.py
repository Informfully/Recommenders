import requests
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Set, List
from collections import OrderedDict
from requests.exceptions import JSONDecodeError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def make_request_with_retries(url, retries=3, backoff_factor=1.0):
    """
    Makes a request with retries and exponential backoff.

    Returns:
        response (requests.Response): The response object if successful.
        None: If all retries fail or response is invalid.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                try:
                    # Attempt to parse JSON to ensure it's valid
                    response.json()
                    return response
                except JSONDecodeError:
                    logging.warning(f"Invalid JSON response for URL: {url}")
            elif response.status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get('Retry-After', backoff_factor))
                logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
            else:
                response.raise_for_status()
        except (requests.RequestException, JSONDecodeError) as e:
            logging.warning(f"Request failed for URL {url}: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    logging.error(f"All retries failed for URL: {url}")
    return None


def is_valid_string(string):
    """
    Checks if a string exists in Wikidata and has a Geonames ID.
    """
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={string}&language=en&format=json"
    response = make_request_with_retries(url)
    if response is None:
        return False

    try:
        data = response.json()
    except JSONDecodeError:
        logging.error(f"Failed to decode JSON response for string: {string}")
        return False

    if 'search' not in data:
        logging.debug(f"No search results found for string: {string}")
        return False

    for result in data['search']:
        if result.get('label', '').lower() == string.lower():
            wikidata_id = result.get('id')
            if not wikidata_id:
                continue
            url_claim = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={wikidata_id}&property=P1566&format=json"
            response_claim = make_request_with_retries(url_claim)
            if response_claim is None:
                continue
            try:
                data_claim = response_claim.json()
            except JSONDecodeError:
                logging.error(f"Failed to decode JSON response for claims of {wikidata_id}")
                continue
            if 'claims' in data_claim and 'P1566' in data_claim['claims']:
                return True

    return False


def get_english_label(search_string, language):
    """
    Get the English label for a given search string from Wikidata.

    Parameters:
        search_string (str): The string to search for.
        language (str): The language code in which the search should be performed.

    Returns:
        str or None: The English label corresponding to the search string, or None if not found.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": search_string,
        "language": language
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('search'):
            # Return the English label of the first search result
            return data['search'][0].get('label')
    except (requests.RequestException, JSONDecodeError) as e:
        logging.error(f"Error fetching English label for {search_string}: {e}")
    return None


def get_region_data(entity: Dict, lookup_dict: Dict, language_tag: Optional[str] = None) -> Optional[str]:
    """
    Check if the region name is existing in Wikidata (if it is the valid name)

    Parameters:
        entity: A dictionary containing entity information, including 'text'
               and 'alternative' keys representing the region name and its
               alternatives.
        language_tag: An optional language tag (e.g., 'de') to retrieve the
                      English label if available.

    Returns:
        A valid region name in lowercase if found, otherwise None.
    """
    # Sort the list of alternative names and the official name by length in descending order
    all_alternatives = [entity['text']] + sorted(entity.get('alternative', []), key=len, reverse=True)
    # Remove duplicates while preserving order
    sorted_alternatives = list(OrderedDict.fromkeys(all_alternatives))

    # Iterate over the sorted list of names and check if any of them are valid
    for alternative in sorted_alternatives:
        alt_lower = alternative.lower()
        if alt_lower in lookup_dict:
            if lookup_dict[alt_lower]:
                for region_key in sorted_alternatives:
                    lookup_dict[region_key.lower()] = True
                return alt_lower
            else:
                continue

        if is_valid_string(alternative):
            for region_key in sorted_alternatives:
                lookup_dict[region_key.lower()] = True
            return alt_lower
        else:
            lookup_dict[alt_lower] = False

    # If a language tag was provided, try to get the English name of the region in that language
    if language_tag and sorted_alternatives:
        name_en = get_english_label(sorted_alternatives[0], language_tag)
        if name_en and (name_en.lower() != sorted_alternatives[0].lower()) and is_valid_string(name_en):
            return name_en.lower()

    return None


def get_region(ne_list: List[Dict], lookup_dict: Dict, language_tag: Optional[str] = None) -> Set:
    """ Enhance the dataset with its region (e.g. city, country and so on)

    Parameters
    ----------
    ne_list: list, list of dictionaries with named entities extracted from original text
    language_tag: An optional language tag (e.g., 'de') to retrieve the English label if available.
    lookup_dict: A dictionary for queried regions.

    Returns
    -------
    regions: set, all geographical name such as city and country
    """
    regions = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_region_data, entity, lookup_dict, language_tag)
                   for entity in ne_list if entity.get('label') in ['GPE', 'LOC']]

        for future in futures:
            result = future.result()
            if result:
                regions.append(result)

    return list(set(regions))
