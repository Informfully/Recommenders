import textstat
import unicodedata

textstat_langs = ["en", "de", "es", "fr", "it", "nl", "ru", "hu"]
new_langs = {
    "en": {  # Default config
        "fre_base": 206.835,
        "fre_sentence_length": 1.015,
        "fre_syll_per_word": 84.6,
        "syllable_threshold": 3,
    },
    "af": {
        # RE-McDermid Heyns
        ''' Richards, Rose & Jansen, C.J.M. & Zyl, Liezl. (2017). Evaluating Four Readability Formulas For Afrikaans.
        Stellenbosch Papers in Linguistics Plus. 53. 10.5842/53-0-739.
        '''
        "fre_base": 138.8989,
        "fre_sentence_length": 1.0052,
        "fre_syll_per_word": 35.4562
    },
    "bg": {
        # South Slavic languages
        "fre_base": 206.835,
        "fre_sentence_length": 1.2,
        "fre_syll_per_word": 60,
    },
    "cs": {
        # Bendová, Klára & Cinková, Silvie. (2021). Adaptation of Classic Readability Metrics to Czech. 10.1007/978-3-030-83527-9_14.
        "fre_base": 206.935,
        "fre_sentence_length": 1.672,
        "fre_syll_per_word": 62.18,
    },
    "da": {
        "fre_base": 206.835,
        "fre_sentence_length": 1.1,
        "fre_syll_per_word": 67.5,
    },
    "el": {
        ''' Dimitrios Tzimokas and Marina Matthaioudaki. 2014. Deiktes anagnosimotitas: zitimata efarmogis kai axiopistias [in Greek].
        In Major Trends in Theoretical and Applied Linguistics 3, pages 367–384. De Gruyter Open Poland.
        '''
        "fre_base": 206.835,
        "fre_sentence_length": 1.015,
        "fre_syll_per_word": 59,
    },
    "et": {
        "fre_base": 206.84,
        "fre_sentence_length": 1.1,
        "fre_syll_per_word": 65,
    },
    "hr": {
        # South Slavic languages
        "fre_base": 206.84,
        "fre_sentence_length": 1.2,
        "fre_syll_per_word": 64,
    },
    "lt": {
        "fre_base": 206.84,
        "fre_sentence_length": 1.1,
        "fre_syll_per_word": 63.5,
    },
    "lv": {
        "fre_base": 206.84,
        "fre_sentence_length": 1.1,
        "fre_syll_per_word": 64,
    },
    "nb": {
        "fre_base": 206.835,
        "fre_sentence_length": 1.05,
        "fre_syll_per_word": 65.5,
    },
    "nn": {
        "fre_base": 206.835,
        "fre_sentence_length": 0.95,
        "fre_syll_per_word": 66.5,
    },
    "pl": {
        "fre_base": 180,
        "fre_sentence_length": 1.5,
        "fre_syll_per_word": 85,
    },
    "pt": {
        # Moreno, Gleice & Moreno, Marco Polo & Hein, Nelson & Hein, Adriana. (2022). ALT: A software for readability analysis of Portuguese-language texts.
        "fre_base": 227,
        "fre_sentence_length": 1.04,
        "fre_syll_per_word": 72,
    },
    "ro": {
        "fre_base": 206.835,
        "fre_sentence_length": 1.1,
        "fre_syll_per_word": 67.5,
    },
    "sk": {
        "fre_base": 206.84,
        "fre_sentence_length": 1.2,
        "fre_syll_per_word": 66,
    },
    "sl": {
        # South Slavic languages
        "fre_base": 206.835,
        "fre_sentence_length": 1.1,
        "fre_syll_per_word": 65,
    },
    "te": {
        "fre_base": 206.835,
        "fre_sentence_length": 0.8,
        "fre_syll_per_word": 80,
    },
    "uk": {
        "fre_base": 206.835,
        "fre_sentence_length": 1.25,
        "fre_syll_per_word": 60,
    },
    "zu": {
        "fre_base": 206.835,
        "fre_sentence_length": 1.1,
        "fre_syll_per_word": 61,
    },
    "ca": {
        # Similar language to Spanish
        "fre_base": 206.84,
        "fre_sentence_length": 1.02,
        "fre_syll_per_word": 60,
    },
    "gl": {
        # Similar language to Portuguese
        "fre_base": 227,
        "fre_sentence_length": 1.04,
        "fre_syll_per_word": 72,
    },
    "sr": {
        # South Slavic languages
        "fre_base": 206.835,
        "fre_sentence_length": 1.2,
        "fre_syll_per_word": 60,
    },
    "sv": {
        "fre_base": 180,
        "fre_sentence_length": 1,
        "fre_syll_per_word": 58.5,
    }
}


def get_lang_cfg(lang, key: str) -> float:
    """ Read as get lang config """
    default = new_langs.get("en")
    config = new_langs.get(lang, default)
    return config.get(key, default.get(key))


def contains_meaningful_characters(text):
    """ Check if the text contains any meaningful characters (letters or numbers) from any language. """
    for char in text:
        # Check if the character is a letter (from any script)
        if unicodedata.category(char).startswith('L'):  # 'L' = Letter, 'N' = Number
            return True
    return False


def get_readability(text, lang='en'):
    """ Enhance the dataset with its readability using python textstat library (https://pypi.org/project/textstat/).

    Parameters
    ----------
    text: string, required
        Each row of news article text in dataframe.

    lang: string, optional, default: 'en'
        Language of the text, use the abbreviation of language following the ISO 639-1 standard
        (https://en.wikipedia.org/wiki/ISO_639-1).

    Returns
    -------
    readability: int
        Flesch Reading Ease Score. The higher the score, the easier to read, the lower the complexity
        (https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease).

    """
    # print(f"Computing readability for language:{lang}")
    try:
        textstat.set_lang(lang)
    except KeyError:  # Handle invalid language codes
        if lang in new_langs.keys():
            lang = 'en'  # Default to English
            textstat.set_lang(lang)  # Set language to English
        else:
            # print(f"Language code '{lang}' not supported.")
            # return None
            raise ValueError(f"Invalid language code '{lang}' provided. Supported language codes are: {', '.join(new_langs.keys())}")
    
    if not isinstance(text, str):
        raise TypeError(f"Invalid input: Expected a string for 'text', but received {type(text).__name__}.")
    try:
        if not text:
            return None  # Empty text
        # Check if the text contains any meaningful characters
        if not contains_meaningful_characters(text):
            return None
        lang_root = lang.split("_")[0]
        if lang_root in textstat_langs:
            readability = textstat.flesch_reading_ease(text)
        else:
            flesch = (
                    get_lang_cfg(lang_root, "fre_base")
                    - float(
                        get_lang_cfg(lang_root, "fre_sentence_length")
                        * textstat.avg_sentence_length(text)
                    )
                    - float(
                        get_lang_cfg(lang_root, "fre_syll_per_word")
                        * textstat.avg_syllables_per_word(text)
                    )
            )
            readability = round(flesch, 2)
    except Exception as e:
        # print(f"An error occurred while getting readability score: {e}")
        # readability = None
        raise RuntimeError(f"An error occurred while calculating the readability score: {e}")

    return readability
