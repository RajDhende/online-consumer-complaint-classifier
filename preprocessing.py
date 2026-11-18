import re
import string
import pandas as pd
import nltk

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Lowercase, remove URLs, special characters, digits, and extra whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove digits
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation / special characters
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """
    Remove English stopwords from tokenised text.
    """
    if not isinstance(text, str):
        return ""

    tokens = text.split()
    filtered = [word for word in tokens if word not in STOPWORDS and len(word) > 1]
    return ' '.join(filtered)


def preprocess_pipeline(text: str) -> str:
    """
    Full preprocessing pipeline: clean_text -> remove_stopwords.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


def get_text_stats(df: pd.DataFrame, text_col: str) -> dict:
    """
    Compute basic text statistics for a DataFrame column.

    Returns a dict with:
        - avg_length  : average character length of non-empty entries
        - vocab_size  : number of unique tokens across all documents
        - total_words : total word count across all documents
    """
    if text_col not in df.columns:
        return {'avg_length': 0, 'vocab_size': 0, 'total_words': 0}

    series = df[text_col].dropna().astype(str)

    avg_length = series.str.len().mean() if len(series) > 0 else 0.0

    all_tokens = []
    for doc in series:
        all_tokens.extend(doc.split())

    total_words = len(all_tokens)
    vocab_size = len(set(all_tokens))

    return {
        'avg_length': round(float(avg_length), 2),
        'vocab_size': int(vocab_size),
        'total_words': int(total_words),
    }
