import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os


def load_model(model_name='facebook/bart-large-mnli', cache_dir= None):
    """Load a pre-trained model from Hugging Face or use a locally cached version.

    Parameters
    ----------
    model_name: str, optional
        The name of the pre-trained model to load (defaults to `facebook/bart-large-mnli`).
    cache_dir: str, optional
        Path to the cache directory to store the downloaded model.

    Returns
    -------
    model: HuggingFace model
        Loaded pre-trained model.
    tokenizer: HuggingFace tokenizer
        Tokenizer corresponding to the model.
    """
    # Check if cache_dir is provided, otherwise use default cache
    if cache_dir is None:
        cache_dir = os.getenv('TRANSFORMERS_CACHE', os.path.join(os.path.expanduser("~"), '.cache', 'huggingface', 'transformers'))

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)  # Create cache directory if it doesn't exist.

    # Use the Hugging Face pipeline to load the model and tokenizer, caching them locally.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    return model, tokenizer

_classifier = None

def get_category(row, **kwargs):
    """ Enhance the dataset with its category (e.g. news, sports, life)
    This function requires category list defined by user

    Parameters
    ----------
    row: string, news id or article text in input file dataframe
    kwargs : list or dataframe, candidate labels or metadata dataframe

    Returns
    -------
    cat: string, corresponding category name for each news id row
    """
    global _classifier
    candidate_labels = kwargs.get('candidate_labels')
    meta_data = kwargs.get('meta_data')
    threshold = kwargs.get('threshold', 0.5)
    if candidate_labels and _classifier is None:
        model, tokenizer = load_model()
        _classifier = pipeline("zero-shot-classification", model=model,tokenizer=tokenizer)

    if candidate_labels:
         # Ensure row is a string (text)
        if not isinstance(row, str):
            raise TypeError(f"Expected row to be str (text), but got {type(row).__name__}")
        try:
            # run classifier
            res = _classifier(row, candidate_labels, multi_label=True)

            categories = res['labels']
            scores = res['scores']

            # get the highest score with a threshold
            if max(scores) > threshold:
                i = np.argmax(scores)
                return categories[i]
            else:
                return -1
        except KeyError:
            raise RuntimeError("Unexpected response format from classifier.")
        except Exception as e:
            raise RuntimeError(f"Classification failed: {e}")
        
    elif meta_data is not None:
        cat = meta_data[meta_data['id'] == row]['category']
        if not cat.empty:
            return cat.values[0]
        else:
            return -1

    # If no candidate labels and no metadata, return -1 (indicating no category found)
    return -1 
