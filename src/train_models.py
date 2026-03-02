"""Train sentiment classification models on Amazon product reviews."""

import os
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Reviews.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

RANDOM_STATE = 42


def load_data(sample_frac: float = 0.1, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Load Amazon reviews and prepare sentiment labels."""
    df = pd.read_csv(DATA_PATH)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    df["sentiment"] = df["Score"].apply(
        lambda s: "negative" if s <= 2 else ("neutral" if s == 3 else "positive")
    )
    df["review"] = df["Text"].fillna(df["Summary"]).fillna("")

    return df[["review", "sentiment"]]


def build_tfidf_pipeline(
    model,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> Pipeline:
    """Build a TF-IDF vectorizer pipeline with the given classifier."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", model)])


def get_models() -> dict:
    """Return model names and pipeline instances."""
    return {
        "LogisticRegression": build_tfidf_pipeline(
            LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
        ),
        "RandomForest": build_tfidf_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        ),
        "MLPClassifier": build_tfidf_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=100,
                random_state=RANDOM_STATE,
            )
        ),
    }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict:
    """Compute F1 scores and classification report."""
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    return {
        "model": model_name,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": classification_report(y_true, y_pred),
    }


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    model_name: str,
    output_dir: str,
) -> str:
    """Generate confusion matrix plot and save to reports/."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {model_name}")

    os.makedirs(output_dir, exist_ok=True)
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    return filepath


def train_and_evaluate(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    models: dict,
) -> list:
    """Train all models, evaluate with F1, save confusion matrices and models."""
    results = []
    labels = sorted(y_train.unique())

    for model_name, pipeline in models.items():
        print(f"\nTraining {model_name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_model(y_test.values, y_pred, model_name)
        results.append(metrics)

        print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"\n  Classification Report:\n{metrics['report']}")

        plot_path = plot_and_save_confusion_matrix(
            y_test.values, y_pred, labels, model_name, REPORTS_DIR
        )
        print(f"  Confusion matrix saved: {plot_path}")

        model_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_pipeline.joblib")
        joblib.dump(pipeline, model_path)
        print(f"  Model saved: {model_path}")

    return results


def main():
    """Run full training and evaluation pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data(sample_frac=0.1)
    print(f"Dataset size: {len(df)}")
    print(df["sentiment"].value_counts())

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = get_models()
    results = train_and_evaluate(X_train, X_test, y_train, y_test, models)

    print("\n" + "=" * 50)
    print("SUMMARY (F1 weighted)")
    print("=" * 50)
    for r in sorted(results, key=lambda x: x["f1_weighted"], reverse=True):
        print(f"  {r['model']}: {r['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
