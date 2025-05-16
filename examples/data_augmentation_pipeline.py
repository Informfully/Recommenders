import sys
import pandas as pd
import json
from cornac.augmentation import category, sentiment, readability, ner, min_maj, story, enrich_ne, region, party
import os

# Add the directory of the current script (i.e., ./examples) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Path to the subfolder inside 'examples/'
news_files_dir = os.path.join(current_dir, 'example_news_files')

config_files_dir = os.path.join(current_dir, 'example_config_files')
sys.path.insert(0, news_files_dir)
sys.path.insert(0, config_files_dir)

import data_enrich_config as config


VALID_LANGUAGE_CODES = [
    'af', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa',
    'fi', 'fr', 'ga', 'gl', 'gu', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn',
    'ko', 'ku', 'ky', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'pa', 'pl',
    'pt', 'ro', 'ru', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'uz', 'vi',
    'xh', 'zh', 'zu', 'mul', 'xx'
]

NER_LANGUAGE_CODES = [
    'pt', 'de', 'fr', 'es', 'ca', 'hr', 'da', 'nl', 'fi', 'el', 'it', 'ja', 'ko', 'lt', 'mk', 'xx', 'mul', 'nb', 'pl',
    'ro', 'ru', 'sl', 'sv', 'uk','en'
]



def is_list_of_strings(input_list):
    """Determine if the input is a non-empty list containing strings"""
    return isinstance(input_list, list) and input_list and all(isinstance(item, str) for item in input_list)


def is_language_code(code):
    """Determine if the input is a valid language code"""
    return code in VALID_LANGUAGE_CODES


def is_subset(subset, superset):
    """Determine whether list_A is a subset of list_B."""
    return set(subset).issubset(set(superset))


def check_entity_types(types, language):
    """Check and generate prompts for named entity types"""
    lang_specific_types = {
        # 'en_zh': ['PER', 'LOC', 'ORG', 'MISC'],
        'en_zh':['PERSON', 'GPE', 'ORG', 'NORP', 'FAC', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
                   'DATE', 'TIME','MISC'],
        'other': ['PER', 'LOC', 'ORG', 'MISC']
    }

    if language not in NER_LANGUAGE_CODES:
        raise ValueError(f"Named entity enrichment is not supported for {language}.")

    print(f"language:{language}")
    valid_types = lang_specific_types['en_zh'] if language in ['en', 'zh'] else lang_specific_types['other']
    print(f"valid_types:{valid_types}")
    print(f"types:{types}")
    if types and is_subset(types, valid_types):
        return f"Named entity types to enrich: {', '.join(types)}.\n"
    else:
        raise ValueError(f"Invalid entity types for {language}. Please refer to config.py.")


def generate_message():
    """Generate a message based on the configuration options."""
    options = []
    need_columns = set()
    exclude_columns = set()
    message = ""

    if not is_language_code(config.lang):
        raise ValueError("Please provide a valid two-letter language code of the input data, e.g., 'en', 'de'.")

    message += f"Language of the input data: {config.lang}.\n"

    for attr, settings in config.attributes.items():
        if settings.get('enrich'):
            options.append(f"{attr.replace('_', ' ')} enrichment")

            if attr == 'category':
                method = settings.get('method')
                if method == 'metadata':
                    need_columns.update(['id', 'category'])
                    cat_file_path = settings.get('cat_file_path')
                    message += f"Category enrichment method: metadata. Meta file: {cat_file_path or 'included in input file'}.\n"
                elif method == 'zero-shot':
                    need_columns.update(['id', 'text'])
                    exclude_columns.add('category')
                    candidate_labels = settings.get('candidate_labels', [])
                    if not is_list_of_strings(candidate_labels):
                        raise ValueError("Zero-shot category enrichment requires valid candidate labels.")
                    message += f"Category enrichment method: zero-shot. Candidate labels: {', '.join(candidate_labels)}.\n"
                else:
                    raise ValueError("Invalid category enrichment method. Choose 'metadata' or 'zero-shot'.")

            elif attr == 'named_entities':
                need_columns.update(['id', 'text'])
                exclude_columns.add('named entities')
                message += check_entity_types(settings.get('entities'), config.lang)

            elif attr == 'enriched_named_entities':
                need_columns.update(['id', 'named entities'])
                exclude_columns.add('enriched named entities')

            elif attr == 'region':
                need_columns.update(['id', 'named entities'])
                if settings.get('lang'):
                    message += "Advanced language option for region enrichment is selected.\n"

            elif attr == 'political_party':
                need_columns.update(['id', 'enriched named entities'])

            elif attr == 'min_maj_ratio':
                need_columns.update(['id', 'enriched named entities'])
                for key in ['major_gender', 'major_citizen', 'major_ethnicity', 'major_place_of_birth']:
                    if not is_list_of_strings(settings.get(key, [])):
                        raise ValueError(f"Invalid format for {key} in majority/minority ratio enrichment.")

            elif attr in ['readability', 'sentiment']:
                need_columns.update(['id', 'text'])

            elif attr == 'story':
                need_columns.update(['id', 'text', 'date', 'category'])

    need_columns.difference_update(exclude_columns)

    if options:
        message = f"Enabled options: {', '.join(options)}.\n" + message
        message += f"Required columns in input data: {', '.join(need_columns)}.\n"
        message += "Do you want to continue? (yes/no)"
    else:
        message = "No enrichment options are enabled. Do you want to continue? (yes/no)"

    return message


def process_with_progress(index, total, function, *args, **kwargs):
    """Executes a function and reports progress."""
    progress_interval = 100
    if (index + 1) % progress_interval == 0:
        progress = (index + 1) / total * 100
        print(f"Processing progress: {progress:.2f}% ({index + 1}/{total})")
    result = function(*args, **kwargs)
    return result[0] if isinstance(result, tuple) else result


def handle_exception(exception_message):
    """Handles exceptions and allows user to continue or stop."""
    print(f"Error: {exception_message}")

    for attempts in range(3):
        user_response = input(
            "Do you want to stop the process? Type 'stop' to stop or 'continue' to skip this enrichment and continue: ").strip().lower()
        if user_response in ['stop', 's']:
            print("Process stopped by user.")
            sys.exit()
        elif user_response in ['continue', 'c']:
            print("Skipping this enrichment and continuing.")
            return
        else:
            print(f"Invalid input. Please try again. Attempts left: {2 - attempts}")
    print("No valid input received after 3 attempts. Skipping this enrichment and continuing.")


def enrich_column(df, column_name, func, main_input, *args, **kwargs):
    """Enrich a DataFrame column using a specified function."""
    total_rows = len(df)
    try:
        print(f"{column_name.capitalize()} extraction in progress...")
        df[column_name] = df.apply(
            lambda row: process_with_progress(row.name, total_rows, func, row[main_input], *args, **kwargs), axis=1)
        print(f"{column_name.capitalize()} successfully extracted!")
        return True
    except Exception as e:
        # handle_exception(str(e))
        return False


def load_csv_file(df, column_name):
    """Load metadata from a file if the path is provided."""
    path = ''
    if column_name == 'category':
        path = config.attributes['category']['cat_file_path']
    if column_name in df.columns:
        return None
    if path:
        return pd.read_csv(path)
    raise ValueError(f"Data for {column_name} is not provided.")


def load_json_file(df, column_name):
    """Load column_name from a file if not already in the DataFrame."""
    path = ''
    if column_name == 'named_entities':
        path = config.attributes['named_entities']['ner_file_path']
    elif column_name == 'enriched_named_entities':
        path = config.attributes['enriched_named_entities']['ener_file_path']

    if column_name not in df.columns and path:
        with open(path, 'r') as file:
            load_data = json.load(file)
        load_df = pd.DataFrame([(key, value[column_name]) for key, value in load_data.items()],
                               columns=['id', column_name])
        df = pd.merge(df, load_df, on='id', how='inner')
    return df


def enhance_data(input_file_path, output_file_path):
    """Enhance dataset based on configuration and save the enriched data."""
    df = pd.read_csv(input_file_path)
    columns_to_save = ['id']

    # Add Category
    if config.attributes['category']['enrich']:
        enrich_method = config.attributes['category']['method']
        if enrich_method == 'zero-shot' and config.attributes['category']['candidate_labels']:
            enriched = enrich_column(df, 'category', category.get_category, 'text',
                                     candidate_labels=config.attributes['category']['candidate_labels'])
        elif enrich_method == 'metadata':
            meta_data = load_csv_file(df, 'category')
            meta_data = meta_data[:5] if meta_data else None
            if meta_data:
                enriched = enrich_column(df, 'category', category.get_category, 'id', meta_data=meta_data)
            else:
                print("Category information already included in the input file.")
                enriched = True
        else:
            enriched = 'category' in df.columns
        if not enriched:
            # handle_exception("Category enrichment failed: metadata or candidate labels required.")
            print("error:category enrichment")
        else:
            columns_to_save.append('category')

    # Add Readability
    if config.attributes['readability']['enrich']:
        enriched = enrich_column(df, 'readability', readability.get_readability, 'text', lang=config.lang)
        if enriched:
            columns_to_save.append('readability')

    # Add Sentiment
    if config.attributes['sentiment']['enrich']:
        enriched = enrich_column(df, 'sentiment', sentiment.get_sentiment, 'text')
        if enriched:
            columns_to_save.append('sentiment')

    # Add Named Entity
    if config.attributes['named_entities']['enrich']:
        enriched = enrich_column(df, 'named_entities', ner.get_ner, 'text', ner_model=ner.set_ner_lang(config.lang),
                                 entities=config.attributes['named_entities']['entities'])
        if enriched:
            columns_to_save.append('named_entities')

    # Add Enriched Named Entity
    if config.attributes['enriched_named_entities']['enrich']:
        try:
            df = load_json_file(df, 'named_entities')
            if 'named_entities' not in df.columns:
                raise ValueError("Named entities required for enrichment are missing.")
            if (config.attributes['named_entities']['enrich'] and
                    not any(entity in config.attributes['named_entities']['entities'] for entity in
                            ['PER', 'PERSON', 'ORG'])):
                raise ValueError("To enrich with Wikidata, include 'PER'/'PERSON' or 'ORG' in entities config.")

            lookup_person = enrich_ne.EfficientDict()
            lookup_org = enrich_ne.EfficientDict()
            enriched = enrich_column(df, 'enriched_named_entities', enrich_ne.get_enriched_ne, 'named_entities',
                                     lookup_person=lookup_person, lookup_org=lookup_org, language_tags=[config.lang])
            if enriched:
                columns_to_save.append('enriched_named_entities')
        except Exception as e:
            # handle_exception(str(e))
            print(f"error: {str(e)}")

    # Add Region
    if config.attributes['region']['enrich']:
        try:
            df = load_json_file(df, 'named_entities')
            if 'named_entities' not in df.columns:
                raise ValueError("Named entities required for region extraction are missing.")
            if (config.attributes['named_entities']['enrich'] and
                    not any(entity in config.attributes['named_entities']['entities'] for entity in ['GPE', 'LOC'])):
                raise ValueError("For region extraction, include 'GPE' or 'LOC' in entities config.")

            lookup_dict = {}
            enriched = enrich_column(df, 'region', region.get_region, 'named_entities', lookup_dict=lookup_dict,
                                     language_tag=config.lang if config.attributes['region']['lang'] else None)
            if enriched:
                columns_to_save.append('region')
        except Exception as e:
            # handle_exception(str(e))
            print(f"error:{str(e)}")

    # Add Political Party
    if config.attributes['political_party']['enrich']:
        try:
            df = load_json_file(df, 'enriched_named_entities')
            if 'enriched_named_entities' not in df.columns:
                raise ValueError("Enriched named entities required for political party extraction are missing.")

            lookup_parties = {}
            enriched = enrich_column(df, 'party', party.get_party, 'enriched_named_entities', config.lang, lookup_parties)
            if enriched:
                columns_to_save.append('party')
        except Exception as e:
            # handle_exception(str(e))
            print(f"error:{str(e)}")

    # Add Minority-Majority
    if config.attributes['min_maj_ratio']['enrich']:
        try:
            df = load_json_file(df, 'enriched_named_entities')
            if 'enriched_named_entities' not in df.columns:
                raise ValueError("Enriched named entities data required for minority-majority ratio extraction are missing.")

            enriched = enrich_column(df, 'min_maj_ratio', min_maj.get_min_maj_ratio, 'enriched_named_entities',
                                     major_gender=config.attributes['min_maj_ratio']['major_gender'],
                                     major_citizen=config.attributes['min_maj_ratio']['major_citizen'],
                                     major_ethnicity=config.attributes['min_maj_ratio']['major_ethnicity'],
                                     major_place_of_birth=config.attributes['min_maj_ratio']['major_place_of_birth'])
            if enriched:
                columns_to_save.append('min_maj_ratio')
        except Exception as e:
            # handle_exception(str(e))
            print(f"error:{str(e)}")

    # Add Story
    if config.attributes['story']['enrich']:
        try:
            meta_data = load_csv_file(df, 'category')
            if meta_data:
                enrich_column(df, 'category', category.get_category, 'id', meta_data=meta_data)

            print("Story extraction in progress...")
            df = story.get_story(df)
            print('Stories successfully extracted!')
            columns_to_save.append('story')
        except Exception as e:
            # handle_exception(str(e))
            print(f"error:{str(e)}")

    # Save enriched data to JSON file
    df_selected = df[columns_to_save]
    df_selected.set_index('id', inplace=True)
    df_selected.to_json(os.path.join(output_file_path, 'total.json'), orient='index', indent=4)


    # Save to CSV
    df.to_csv(os.path.join(output_file_path, 'enriched.csv'), index=False)

    print("CSV file saved successfully!")
    # Iterate over the specified keys and extract their data
    extracted_data = {}
    for item_id, item_value in df.set_index('id').to_dict(orient='index').items():
        extracted_data[item_id] = {}
        for key in columns_to_save:
            if key in item_value:
                extracted_data[item_id][key] = item_value[key]
            else:
                extracted_data[item_id][key] = None


    # Write the extracted data to individual JSON files.
    for key in columns_to_save:
        if key != 'id':
            key_data = {item_id: item[key] for item_id, item in extracted_data.items() if key in item}
            output_file = os.path.join(output_file_path, f"{key}.json")
            with open(output_file, 'w') as json_file:
                json.dump(key_data, json_file, indent=4)


if __name__ == "__main__":
    # prompt_message = generate_message()
    # for attempt in range(3):
    output_dir = config.output_file_path
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)  # Creates folder safely
            print(f"Created directory: {output_dir}")
        except PermissionError:
            print(f"Permission denied: Cannot create {output_dir}")

    enhance_data(config.input_file_path, config.output_file_path)
    sys.exit(f"Dataset successfully enhanced! Check {config.output_file_path}")
    #     elif response in ['no', 'n']:
    #         sys.exit("Process stopped by user.")
    #     else:
    #         print(f"Invalid input. Please try again. Attempts left: {2 - attempt}")

    # print("No valid input received after 3 attempts. Continuing the process.")
    # enhance_data(config.input_file_path, config.output_file_path)
    # sys.exit(f"Dataset successfully enhanced! Check {config.output_file_path}")
