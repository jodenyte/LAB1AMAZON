"""Evaluate trained sentiment models on held-out data."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Reviews.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RANDOM_STATE = 42
MODEL_NAMES = ["logisticregression", "randomforest", "mlpclassifier"]


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


def main():
    """Load models, evaluate on test split, and print metrics."""
    print("Loading data...")
    df = load_data(sample_frac=0.1)
    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Test set size: {len(X_test)}")

    results = []
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}_pipeline.joblib")
        if not os.path.exists(path):
            print(f"\nSkipping {name}: model not found at {path}")
            print("Run 'python src/train_models.py' first.")
            continue

        print(f"\nEvaluating {name}...")
        pipeline = joblib.load(path)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        })

        print(f"  Accuracy:     {acc:.4f}")
        print(f"  F1 (macro):   {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    if results:
        print("\n" + "=" * 50)
        print("SUMMARY (F1 weighted)")
        print("=" * 50)
        for r in sorted(results, key=lambda x: x["f1_weighted"], reverse=True):
            print(f"  {r['model']}: {r['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
