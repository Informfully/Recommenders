import pandas as pd
import spacy
import itertools
from difflib import SequenceMatcher
from collections import defaultdict
import networkx as nx
import community.community_louvain as community_louvain
import subprocess
import sys

def is_abbreviation(phrase1: tuple, phrase2: tuple) -> bool:
    """
    Checks if one input is an abbreviation of the other.

    Parameters:
        phrase1 (Doc): The Doc object representing the first phrase.
        phrase2 (Doc): The Doc object representing the second phrase.

    Return:
        bool: True if one input is an abbreviation of the other, False otherwise.
    """

    # Determine which is the shorter abbreviation and which is the longer full phrase
    abbreviation, full_phrase = (phrase1[0], phrase2) if len(phrase1[0]) < len(phrase2[0]) else (phrase2[0], phrase1)

    # Split the original string into a list of words
    words = full_phrase[0].split()

    # If 'abbreviation' is longer than words count, consider it as not a abbreviation
    # Can be further optimized to deal with some exceptions: Television -> TV, Corporation -> Corp (Corp.)
    if len(abbreviation) > len(words):
        return False

    # Function to check if 'abbr' matches the first letters of the words in 'words'
    def matches_abbreviation(abbr: str, words_list) -> bool:
        if len(abbr) != len(words_list):
            return False
        return all(word.lower().startswith(char.lower()) for char, word in zip(abbr, words_list))

    if matches_abbreviation(abbreviation, words):
        return True

    # Consider only the uppercase words
    upper_words = tuple(word for word in words if word[0].isupper())
    if matches_abbreviation(abbreviation, upper_words):
        return True

    # Consider deleting the conjunctions
    # Define stopword tags
    words_nstop = full_phrase[1]
    if matches_abbreviation(abbreviation, words_nstop):
        return True

    return False

# Function to install a missing SpaCy model
def install_spacy_model(model_name):
    """ Attempts to install the required SpaCy language model. """
    print(f"Attempting to install the SpaCy model: {model_name}...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])


# Define model names for different languages
LANG_MODELS = {
    'en': "en_core_web_sm",
    'pt': "pt_core_news_sm",
    'de': "de_core_news_sm",
    'fr': "fr_core_news_sm",
    'es': "es_core_news_sm",
    'zh': "zh_core_web_sm",
    'ca': "ca_core_news_sm",
    'hr': "hr_core_news_sm",
    'da': "da_core_news_sm",
    'nl': "nl_core_news_sm",
    'fi': "fi_core_news_sm",
    'el': "el_core_news_sm",
    'it': "it_core_news_sm",
    'ja': "ja_core_news_sm",
    'ko': "ko_core_news_sm",
    'lt': "lt_core_news_sm",
    'mk': "mk_core_news_sm",
    'xx': "xx_ent_wiki_sm",
    'mul': "xx_ent_wiki_sm",
    'nb': "nb_core_news_sm",
    'pl': "pl_core_news_sm",
    'ro': "ro_core_news_sm",
    'ru': "ru_core_news_sm",
    'sl': "sl_core_news_sm",
    'sv': "sv_core_news_sm",
    'uk': "uk_core_news_sm",
}


# Initialize spacy for different languages
def set_ner_lang(lang='en'):
    """  Load the appropriate SpaCy named entity recognition model based on the specified language.

    Parameters
    ----------
    lang: string, optional, default : 'en'
        The language of the input dataset.

    Returns
    -------
    NER: Initialized spacy model

    """
    
    # Get the name of the model for the specified language
    model_name = LANG_MODELS.get(lang)
    print(f"model_name:{model_name}")
    if not model_name:
        raise ValueError(f"Language '{lang}' is not supported. Available options: {list(LANG_MODELS.keys())}")
    
    try:
        try:
            ner = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' is not installed. Attempting to install...")
            install_spacy_model(model_name)
            ner = spacy.load(model_name)        
        return ner
        
    except Exception as e:
        # print(f"An error occurred while loading the SpaCy model: {e}")
        # return None
        raise RuntimeError(f"An unexpected error occurred while loading the SpaCy model '{model_name}': {e}") from e


def get_ner(text, ner_model=set_ner_lang(), **kwargs):
    """ Enhancing the dataset with its named entity tags using Spacy and clustering using community louvain algorithm.

    Parameters
    ----------
    text: string, each row of news article text in dataframe

    ner_model: Spacy model, optional, default : spacy.load("en_core_web_sm")
        Initialized spacy model for the specified language.

    Returns
    -------
    ner: dictionary, key is named entity and value is a list of tuples with frequency of its named entities

    """

    try:
        ner = []
        ne_list = []
        text = ner_model(text)
        entities = kwargs['entities']
        stopword_tags = ["DET", "PRON", "CCONJ", "SCONJ", "ADP"]

        # start named entity recognition
        for word in text.ents:
            if word.label_ in entities:
                words_nstop = tuple(token.text for token in word if token.pos_ not in stopword_tags)
                ner.append({'text': word.text, 'alternative': [],  'label': word.label_, 'start_char': word.start_char, 'end_char': word.end_char,
                           'no_stop': words_nstop})
            else:
                continue

        # calculate the distances between all the entity names which has same label
        types = list(set([x['label'] for x in ner]))
        for entity_type in types:
            of_type = [entity for entity in ner if entity['label'] == entity_type]
            names = [(entity['text'], entity['no_stop']) for entity in of_type]
            distances = []
            for phrase1, phrase2 in itertools.combinations(names, 2):
                a, b = phrase1[0], phrase2[0]
                if is_abbreviation(phrase1, phrase2):
                    similarity = 1
                elif len(a) < 3 or len(b) < 3:  # If the label has less than three characters, skip
                    similarity = 0
                elif a in b or b in a:  # If a is contained in b or the other way around (as in "Barack Obama", "Obama"), full match
                    similarity = 1
                else:
                    similarity = SequenceMatcher(None, a, b).ratio()  # Otherwise calculate the SequenceMatcher ratio to account for slight spelling errors
                distances.append({'a': a, 'b': b, 'metric': similarity})

            # Cluster the names based on the distances found
            if distances:
                df = pd.DataFrame(distances)
                thr = df[df.metric > 0.9]
                g_cap = nx.from_pandas_edgelist(thr, 'a', 'b', edge_attr='metric')
                clusters = community_louvain.best_partition(g_cap)
                if clusters:
                    v, k = max((v, k) for k, v in clusters.items())
                    length = v + 1
                else:
                    clusters = {}
                    length = 0
            else:
                clusters = {}
                length = 0

            temp = {}
            for name in names:
                name = name[0]
                if name in clusters:
                    temp[name] = clusters[name]
                else:
                    temp[name] = length
                    length += 1

            d = defaultdict(list)
            for k, v in temp.items():
                d[v].append(k)
            processed = dict(d)

            for _, v in processed.items():
                # find all entries of this cluster
                with_name = [entity for entity in of_type if entity['text'] in v]
                # find all unique occurrences of the cluster
                all_names = [entity['text'] for entity in with_name]
                label = with_name[0]['label']
                # find the name that was most often used to refer to this cluster
                most_frequent_name = max(set(all_names), key=all_names.count)
                # if the cluster is about people names, favor names that contain a space
                # eg: Barack Obama over just Obama
                if label == 'PERSON':
                    with_space = [name for name in all_names if len(name.split(" ")) > 1]
                    if len(with_space) > 0:
                        most_frequent_name = max(set(with_space), key=with_space.count)
                alternative_names = v
                spans = [(entity['start_char'], entity['end_char']) for entity in with_name]
                ne_list.append(dict({'text': most_frequent_name,
                                     'alternative': alternative_names,
                                     'spans': spans,
                                     'frequency': len(with_name),
                                     'label': label}))
    except Exception as e:
        # print(f"An error occurred while getting Named Entities: {e}")
        # ne_list = None
        raise RuntimeError(f"An error occurred while getting Named Entities: {e}")


    return ne_list
