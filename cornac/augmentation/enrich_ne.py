import requests
from requests.adapters import HTTPAdapter
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import time
import random
import hashlib
from threading import Lock


class WikidataQuery:
    """
    A class for querying data from Wikidata.
    """

    def __init__(self, url: str = 'https://query.wikidata.org/sparql', pool_maxsize: int = 50, pool_block: bool = True):
        self.url = url
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, pool_block=pool_block)
        self.session.mount('https://', adapter)

    def execute_query(self, query: str) -> Optional[requests.Response]:
        """
        Sends a SPARQL query to Wikidata.
        """

        retries = 3
        backoff_factor = 1

        while retries > 0:
            try:
                response = self.session.get(self.url, params={'format': 'json', 'query': query})
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', 1))
                    logging.warning(f"Rate limit exceeded. Retrying after {retry_after * backoff_factor} seconds.")
                    time.sleep(retry_after * backoff_factor)
                    backoff_factor *= 2  # Exponential backoff
                elif response.status_code == 403:
                    logging.warning(f"Error executing query: 403 Forbidden - {response.text}")
                    retry_after = int(response.headers.get('Retry-After', 1))
                    if retry_after:
                        logging.warning(f"Retrying after {retry_after} seconds.")
                        time.sleep(retry_after)
                    else:
                        break
                else:
                    response.raise_for_status()
            except requests.HTTPError as http_err:
                logging.warning(f"HTTP error occurred: {http_err}")
                break
            except requests.RequestException as err:
                logging.warning(f"An error occurred: {err}")
                break

            # Add random delay between requests
            time.sleep(random.uniform(0.5, 2))

            retries -= 1
        return None

    @staticmethod
    def read_response(data: Dict, label: str) -> List:
        """
        Returns all unique values for a particular label.
        """
        try:
            return list({x[label]['value'].lower() if label == 'occupations' else x[label]['value']
                         for x in data['results']['bindings'] if label in x})
        except (KeyError, IndexError):
            return []

    def read_person_response(self, data: Dict) -> Dict:
        """
        Retrieves values for each of the value types relevant for person data.
        """
        return {
            'givenname': self.read_response(data, 'givenname'),
            'familyname': self.read_response(data, 'familyname'),
            'gender': self.read_response(data, 'gender'),
            'occupations': self.read_response(data, 'occupations'),
            'party': self.read_response(data, 'party'),
            'citizen': self.read_response(data, 'citizen'),
            'ethnicity': self.read_response(data, 'ethnicity'),
            'place_of_birth': self.read_response(data, 'place_of_birth')
        }

    def read_org_response(self, data: Dict) -> Dict:
        """
        Retrieves values for the 'ideology' value type relevant for organization data.
        """
        return {'ideology': self.read_response(data, 'ideology')}

    def org_data_query(self, label: str, language_tags: List[str] = None) -> Dict:
        """
        Query organization data from Wikidata for multiple language tags.
        """
        if language_tags is None:
            language_tags = ['en']

        language_filter = ' || '.join(f'LANGMATCHES(LANG(?ideology), "{lang}")' for lang in language_tags)
        label_queries = ' UNION '.join(f'{{ ?s rdfs:label "{label}"@{lang} }}' for lang in language_tags)
        query = f"""
        SELECT DISTINCT ?ideology WHERE {{
            {label_queries}

            OPTIONAL {{
              ?s wdt:P1142 ?a .
              ?a rdfs:label ?ideology .
              FILTER(LANG(?ideology) = "" || ({language_filter}))
            }}
        }}
        LIMIT 3
        """

        response = self.execute_query(query)
        if response:
            try:
                return self.read_org_response(response.json())
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON: {e}")
                return {}
        return {}

    def person_data_query(self, label: str, language_tag: str = 'en') -> Dict:
        """
        Create SPARQL query for person data.
        """
        optional_clauses = []
        for field, property_id in [
            ('givenname', 'P735'),
            ('familyname', 'P734'),
            ('occupations', 'P106'),
            ('party', 'P102'),
            ('gender', 'P21'),
            ('ethnicity', 'P172'),
            ('place_of_birth', 'P19'),
            ('citizen', 'P27')
        ]:
            if field == 'place_of_birth':
                optional_clause = f"""
                OPTIONAL {{
                  ?s wdt:{property_id} ?{field}_item .
                  ?{field}_item wdt:P17 ?{field}_country .
                  ?{field}_country rdfs:label ?{field} .
                  FILTER(LANG(?{field}) = "" || LANGMATCHES(LANG(?{field}), "{language_tag}"))
                }}
                """
            else:
                optional_clause = f"""
                OPTIONAL {{
                  ?s wdt:{property_id} ?{field}_item .
                  ?{field}_item rdfs:label ?{field} .
                  FILTER(LANG(?{field}) = "" || LANGMATCHES(LANG(?{field}), "{language_tag}"))
                }}
                """
            optional_clauses.append(optional_clause)

        query = f"""
        SELECT DISTINCT ?s ?givenname ?familyname ?occupations ?party ?position ?gender ?citizen ?ethnicity ?place_of_birth ?sexuality WHERE {{
            ?s ?label '{label}'@{language_tag} .
            {''.join(optional_clauses)}
        }}
        """
        response = self.execute_query(query)
        if response:
            try:
                return self.read_person_response(response.json())
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON: {e}")
                return {}
        return {}


class EfficientDict:
    def __init__(self):
        self.hash_table = {}
        self.main_dict = {}
        self.lock = Lock()

    def _hash_value(self, value):
        # Use hashlib to generate hashes
        return hashlib.md5(str(value).encode()).hexdigest()

    def add(self, key, value):
        with self.lock:
            if key not in self.main_dict or self.hash_table[self.main_dict.get(key)] == '':
                hash_value = self._hash_value(value)
                if hash_value not in self.hash_table:
                    self.hash_table[hash_value] = value
                self.main_dict[key] = hash_value

    def get(self, key):
        with self.lock:
            hash_value = self.main_dict.get(key)
            return self.hash_table.get(hash_value)

    def __getstate__(self):
        # Copy the object's state and remove the lock object before pickling
        state = self.__dict__.copy()
        state.pop('lock', None)
        return state

    def __setstate__(self, state):
        # Restore the object's state and add the lock object after unpickling
        self.__dict__.update(state)
        self.lock = Lock()


def lookup_and_update(lookup_dict: EfficientDict, alternative: str, all_alternatives: List[str],
                      wikidata: WikidataQuery, language_tags: List[str] = None):
    # Check if alternative is in lookup_dict
    lookup_result = lookup_dict.get(alternative.lower())
    # If earlier query get nothing, directly return None
    if lookup_result == '':
        return None
    # If already enriched, update all alternatives and return stored value
    elif lookup_result:
        for dict_key in all_alternatives:
            lookup_dict.add(dict_key.lower(), lookup_result)
        return lookup_result

    # If not queried before, query Wikidata
    if language_tags:
        response = wikidata.org_data_query(alternative, language_tags)
    else:
        response = wikidata.person_data_query(alternative)
    # If Wikidata returns result, update all alternatives and return the result
    if response and any(response.values()):
        for dict_key in all_alternatives:
            lookup_dict.add(dict_key.lower(), response)
        return response
    # If Wikidata returns nothing, set alternative to 'queried'
    lookup_dict.add(alternative.lower(), '')
    return None


def get_person_data(wikidata: WikidataQuery, entity: Dict, lookup_person: EfficientDict) -> Dict:
    """
    Get person data from Wikidata.
    """
    info = {
        'key': entity['text'],
        'frequency': entity['frequency'],
        'alternative': entity['alternative']
    }
    # Prepare the sorted alternatives
    sorted_alternatives = entity['alternative'] + [entity['text']] if entity['text'] not in entity['alternative'] else \
        entity['alternative']
    sorted_alternatives = sorted(sorted_alternatives, key=len, reverse=True)
    full_names = [s for s in sorted_alternatives if ' ' in s]
    for alternative in full_names:
        lookup_result = lookup_and_update(lookup_person, alternative, full_names, wikidata)
        if lookup_result:
            return {alternative: {**lookup_result, **info}}

    # Fallback to the full name if no matches found
    full_name = full_names[0] if full_names else sorted_alternatives[0]
    return {full_name: info}


def get_org_data(wikidata: WikidataQuery, entity: Dict, lookup_org: EfficientDict, language_tags: List[str]) -> Dict:
    """
    Get organization data from Wikidata.
    """
    info = {
        'frequency': entity['frequency'],
        'alternative': entity['alternative']
    }

    # sort the official text and alternative names of entity by length
    sorted_alternatives = entity['alternative'] + [entity['text']] if entity['text'] not in entity['alternative'] else \
        entity['alternative']
    sorted_alternatives = sorted(sorted_alternatives, key=len, reverse=True)

    # check if any alternative names get results from the query
    for alternative in sorted_alternatives:
        lookup_result = lookup_and_update(lookup_org, alternative, sorted_alternatives, wikidata, language_tags)
        if lookup_result:
            return {alternative: {**lookup_result, **info}}

    return {sorted_alternatives[0]: info}


def get_enriched_ne(ne_list: List[Dict], lookup_person: EfficientDict, lookup_org: EfficientDict,
                    language_tags: List[str] = None,
                    wikidata_url: str = 'https://query.wikidata.org/sparql') -> List[Dict]:
    """
    Enhance the named entity tags by retrieving information from Wikidata.

    Parameters:
        ne_list (List[Dict]): List of dictionaries with named entities extracted from original text.
        lookup_person (EfficientDict): Dictionary for queried people.
        lookup_org (EfficientDict): Dictionary for queried organization.
        language_tags (List[str]): List of language tags to match. Default is ['en'].
        wikidata_url (str): URL of the Wikidata SPARQL endpoint. Default is 'https://query.wikidata.org/sparql'.

    Returns:
        List[Dict]: List of dictionaries with person and organization as key and its extended information as value.
    """
    wikidata = WikidataQuery(wikidata_url)
    enriched_ne = []
    if language_tags is None:
        language_tags = ['en']
    elif 'en' not in language_tags:
        language_tags = ['en'] + language_tags

    with ThreadPoolExecutor() as executor:
        futures = []
        for entity in ne_list:
            if entity['label'] in ['PERSON', 'PER']:
                futures.append(executor.submit(get_person_data, wikidata, entity, lookup_person))
            elif entity['label'] == 'ORG':
                futures.append(executor.submit(get_org_data, wikidata, entity, lookup_org, language_tags))

        for future in futures:
            result = future.result()
            if result:
                enriched_ne.append(result)

    return enriched_ne
