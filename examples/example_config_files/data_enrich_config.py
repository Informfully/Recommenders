# Configuration settings for data enhancement pipeline
import os

# Get the current directory of this config file (example_config_files/)
config_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the article file relative to the config file
input_file_path = os.path.join(config_dir, '..', 'example_news_files', 'example_article.csv')
# input_file_path = './example_article.csv'  # input path must be a csv file

output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output_enriched_files')
# output_file_path = './output_enriched_files'  # output path must be a directory to put all json files
lang = 'en'  # Example: "en", "de", "es", "fr"

# Define attributes for enriching features
attributes = {
    'category': {
        'enrich': True,  # True or False
        'method': 'metadata',  # Either 'metadata' or 'zero-shot'
        'cat_file_path': input_file_path,  # If not exists, leave as empty string (i.e, '')

## If 'method' is 'zero-shot', must define 'candidate_labels'.
#         'candidate_labels':  ["politics", "public health", "economics", "business", "sports"]
    },
    'readability': {'enrich': True},
    'sentiment': {'enrich': True},
    'named_entities': {
        'enrich': True,
        # 'entities': ['PER', 'LOC', 'ORG', 'MISC'],  #  For other datasets, entities can include: 'LOC', 'MISC', 'ORG', and 'PER'.
         'entities':[ 'EVENT', 'LOC',  'NORP',  'ORG', 'PERSON'],  # For English and Chinese datasets, entities can include: 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 
        'ner_file_path': '',
    },
    'enriched_named_entities': {
        'enrich': True,
        'ener_file_path': '',
    },
    'region': {'enrich': True, 'lang': False},
    'political_party': {'enrich': True},
    'min_maj_ratio': {
        'enrich': True,
        'major_gender': ['male'],
        'major_citizen': ['United States of America'],
        'major_ethnicity': ['white people'],
        'major_place_of_birth': ['United States of America']
    },
    'story': {'enrich': True}
}
