from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import os


def load_model(model_name="cardiffnlp/xlm-roberta-base-sentiment-multilingual", cache_dir=None):
    """Load the model and tokenizer from Hugging Face or use a local cache if available.

    Parameters
    ----------
    model_name: str, optional (default="xlm-roberta-base-sentiment")
        The name of the pre-trained model on Hugging Face Hub.
    
    cache_dir: str, optional (default=None)
        The directory to cache the model. If not provided, it will use the default Hugging Face cache.

    Returns
    -------
    tokenizer: HuggingFace tokenizer
        The tokenizer corresponding to the loaded model.
    
    model: HuggingFace model
        The pre-trained model.
    """
    # Check if cache_dir is provided, otherwise use default cache
    if cache_dir is None:
        cache_dir = os.getenv('TRANSFORMERS_CACHE', os.path.join(os.path.expanduser("~"), '.cache', 'huggingface', 'transformers'))

    # Create the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # Load the model and tokenizer from Hugging Face or local cache
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    
    return model, tokenizer


# Add global variables for lazy loading
_model = None
_tokenizer = None
_sentiment_analyzer = None
def get_sentiment_analyzer():
    """Lazy load the sentiment analyzer only when needed."""
    global _model, _tokenizer, _sentiment_analyzer
    
    if _sentiment_analyzer is None:
        _model, _tokenizer = load_model()
        _sentiment_analyzer = pipeline("sentiment-analysis", model=_model, tokenizer=_tokenizer, top_k=None)
    
    return _sentiment_analyzer

def get_sentiment(text):
    """ Enhance the dataset with its sentiment (-1.0, 1.0) by analyzing sentiment on a sentence-by-sentence basis,
    and averaging the scores.

    Parameters
    ----------
    text: str
        Each row of news article text in dataframe.

    sentiment_analyzer: HuggingFace pipeline
        The sentiment analysis pipeline object.
        
    Returns
    -------
    sentiment: float
        Average sentiment score (positive score - negative score)
    """
    if not isinstance(text, str):
        return None

    try:
        sentiment_analyzer = get_sentiment_analyzer()
        # Split text into manageable chunks
        if len(text) <= 512:
            merged_sentences = [text]
        else:
            # Tokenize text to handle splitting accurately
            encoded_input = sentiment_analyzer.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            sentences = sentiment_analyzer.tokenizer.convert_ids_to_tokens(encoded_input["input_ids"].tolist()[0])

            # Split sentences using punctuation as separators
            split_sentences = " ".join(sentences).split(". ")
            merged_sentences = []
            current_chunk = ''

            for sentence in split_sentences:
                if len(current_chunk) + len(sentence) + 1 <= 512:  # +1 for the punctuation
                    current_chunk += sentence + ". "
                else:
                    merged_sentences.append(current_chunk.strip())
                    current_chunk = sentence + ". "

            if current_chunk.strip():
                merged_sentences.append(current_chunk.strip())

        # Calculate sentiment scores for each chunk
        sentiment_scores = []
        for chunk in merged_sentences:
            if chunk.strip():
                scores = sentiment_analyzer(chunk[:512])[0]
                score_dict = {item['label']: item['score'] for item in scores}
                if 'positive' not in score_dict or 'negative' not in score_dict:
                    return None  # No valid sentiment labels found
                chunk_sentiment = score_dict['positive'] - score_dict['negative']
                sentiment_scores.append(chunk_sentiment)

        if not sentiment_scores:
            print(f"Warning: No sentiment scores calculated for text: '{text[:50]}...'")
            return None  # Explicitly return None if no scores

        sentiment = round(np.mean(sentiment_scores), 2)
        return sentiment

    except Exception as e:

        raise RuntimeError(f"Error calculating sentiment for text: '{text[:50]}...'. Error: {e}")
