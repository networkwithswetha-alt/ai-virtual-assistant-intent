"""Train the TF-IDF + logistic regression intent classifier.

Usage:
    python train.py     # train, evaluate, save confusion matrix + model
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

from data import generate_dataset, TEMPLATES

MODEL_PATH = "intent_model.joblib"
INTENTS = sorted(TEMPLATES.keys())


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def main():
    os.makedirs("images", exist_ok=True)
    df = generate_dataset(n_per_intent=60, seed=11)
    X_train, X_test, y_train, y_test = train_test_split(
        df["utterance"], df["intent"], test_size=0.2,
        random_state=11, stratify=df["intent"],
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    print(f"Intents: {len(INTENTS)}  Test size: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}\n")
    print(classification_report(y_test, pred))

    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred, labels=INTENTS), display_labels=INTENTS)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title("Intent classification — confusion matrix")
    fig.tight_layout()
    fig.savefig("images/confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Confusion matrix saved to images/confusion_matrix.png")

    joblib.dump({"pipeline": pipe, "intents": INTENTS}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
