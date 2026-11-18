import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Local module
from preprocessing import preprocess_pipeline


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'complaints.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')


def load_data(path: str) -> pd.DataFrame:
    """Load CSV, auto-detect the text column, and keep relevant columns."""
    df = pd.read_csv(path, low_memory=False)

    # Try preferred text column first; fall back to Issue + Sub-issue
    if 'Consumer complaint narrative' in df.columns:
        text_col = 'Consumer complaint narrative'
    elif 'Issue' in df.columns:
        # Combine Issue and Sub-issue for richer text when narrative is absent
        df['text_combined'] = (
            df['Issue'].fillna('') + ' ' + df.get('Sub-issue', pd.Series([''] * len(df))).fillna('')
        ).str.strip()
        text_col = 'text_combined'
        print(f"[INFO] 'Consumer complaint narrative' not found. Using '{text_col}' (Issue + Sub-issue).")
    else:
        raise ValueError("No usable text column found. Need 'Consumer complaint narrative' or 'Issue'.")

    if 'Product' not in df.columns:
        raise ValueError(f"Required column 'Product' not found in {path}")

    # Rename to a consistent internal name
    if text_col != 'Consumer complaint narrative':
        df = df.rename(columns={text_col: 'Consumer complaint narrative'})

    df = df[['Consumer complaint narrative', 'Product',
             'Date received', 'Company', 'State',
             'Company response to consumer']].copy() if all(
        c in df.columns for c in ['Date received', 'Company', 'State', 'Company response to consumer']
    ) else df[['Consumer complaint narrative', 'Product']].copy()

    df = df.dropna(subset=['Consumer complaint narrative', 'Product'])
    return df


def filter_top_categories(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Keep only the top-N product categories by complaint count."""
    top_cats = df['Product'].value_counts().nlargest(n).index.tolist()
    return df[df['Product'].isin(top_cats)].copy()


def sample_per_category(df: pd.DataFrame, max_per_cat: int = 2000,
                         random_state: int = 42) -> pd.DataFrame:
    """Sample up to max_per_cat rows per product category."""
    sampled = (
        df.groupby('Product', group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max_per_cat), random_state=random_state))
    )
    return sampled.reset_index(drop=True)


def compute_metrics(y_true, y_pred, label_names) -> dict:
    """Return a dict of evaluation metrics."""
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'f1': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
        ),
    }


def print_metrics(model_name: str, metrics: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {model_name} Results")
    print(f"{'='*60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"{'='*60}\n")


def main():
    # ------------------------------------------------------------------
    # 1. Load & prepare data
    # ------------------------------------------------------------------
    print(f"Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"Rows after dropping NaN: {len(df)}")

    df = filter_top_categories(df, n=8)
    print(f"Rows after keeping top 8 categories: {len(df)}")
    print("Categories:", df['Product'].unique().tolist())

    df = sample_per_category(df, max_per_cat=2000, random_state=42)
    print(f"Rows after sampling (max 2000/cat): {len(df)}")

    # ------------------------------------------------------------------
    # 2. Preprocess text
    # ------------------------------------------------------------------
    print("\nPreprocessing text (this may take a moment) …")
    df['processed_text'] = df['Consumer complaint narrative'].apply(preprocess_pipeline)

    # ------------------------------------------------------------------
    # 3. Encode labels
    # ------------------------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(df['Product'])
    label_names = le.classes_.tolist()
    print(f"\nLabel classes: {label_names}")

    # ------------------------------------------------------------------
    # 4. TF-IDF vectorisation
    # ------------------------------------------------------------------
    print("\nFitting TF-IDF vectorizer …")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = tfidf.fit_transform(df['processed_text'])

    # ------------------------------------------------------------------
    # 5. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 6. Train models
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression …")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=5)
    lr.fit(X_train, y_train)

    print("Training Multinomial Naive Bayes …")
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    lr_pred = lr.predict(X_test)
    nb_pred = nb.predict(X_test)

    lr_metrics = compute_metrics(y_test, lr_pred, label_names)
    nb_metrics = compute_metrics(y_test, nb_pred, label_names)

    print_metrics("Logistic Regression", lr_metrics)
    print_metrics("Multinomial Naive Bayes", nb_metrics)

    # ------------------------------------------------------------------
    # 8. Select best model
    # ------------------------------------------------------------------
    if lr_metrics['f1'] >= nb_metrics['f1']:
        best_model = lr
        best_model_name = "Logistic Regression"
    else:
        best_model = nb
        best_model_name = "Multinomial Naive Bayes"

    print(f"Best model by F1-score: {best_model_name}  (F1={max(lr_metrics['f1'], nb_metrics['f1']):.4f})")

    # ------------------------------------------------------------------
    # 9. Save artefacts
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, 'trained_model.joblib')
    tfidf_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    meta_path  = os.path.join(MODEL_DIR, 'model_metadata.joblib')

    joblib.dump(best_model, model_path)
    joblib.dump(tfidf, tfidf_path)

    metadata = {
        'model_name': best_model_name,
        'lr_metrics': lr_metrics,
        'nb_metrics': nb_metrics,
        'categories': label_names,
        'label_encoder_classes': le.classes_.tolist(),
    }
    joblib.dump(metadata, meta_path)

    print(f"\nSaved model      -> {model_path}")
    print(f"Saved vectorizer -> {tfidf_path}")
    print(f"Saved metadata   -> {meta_path}")
    print("\nTraining complete.")


if __name__ == '__main__':
    main()
