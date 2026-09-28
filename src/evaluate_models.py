"""
Evaluate the four trained classifiers on the held-out test split.

Mirrors notebooks/phishing_detection_analysis.ipynb (Sections 9 and 10):
  - accuracy, precision, recall, F1-score (weighted average) for each model
  - ROC curves and ROC-AUC for each model (SVM uses decision_function,
    min-max normalised to a [0, 1] range, since the default SVC
    configuration used in this study does not expose predict_proba)
  - confusion matrices for each model
  - results are written to results/model_comparison.csv

Reported figures (see figures/) were generated interactively in the
notebook; this script reproduces the same numeric results but does not
regenerate every plot.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
)

from data_preprocessing import load_dataset, split_and_scale
from train_models import train_all

RESULTS_DIR = "results"


def evaluate(models, X_test_scaled, y_test):
    """Compute accuracy/precision/recall/F1 for each model."""
    rows = []
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
            "Recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
            "F1-score": round(f1_score(y_test, y_pred, average="weighted"), 4),
        })
    return pd.DataFrame(rows), predictions


def compute_roc_auc(models, X_test_scaled, y_test):
    """Compute ROC-AUC for each model (SVM via normalised decision_function)."""
    auc_scores = {}
    for name, model in models.items():
        if name == "SVM":
            scores = model.decision_function(X_test_scaled)
            probs = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            probs = model.predict_proba(X_test_scaled)[:, 1]
        auc_scores[name] = roc_auc_score(y_test, probs)
    return auc_scores


def compute_confusion_matrices(predictions, y_test):
    return {name: confusion_matrix(y_test, y_pred) for name, y_pred in predictions.items()}


if __name__ == "__main__":
    df = load_dataset("data/uci-ml-phishing-dataset.csv")
    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, scaler) = split_and_scale(df)

    models = train_all(X_train_scaled, y_train)
    comparison_df, predictions = evaluate(models, X_test_scaled, y_test)
    auc_scores = compute_roc_auc(models, X_test_scaled, y_test)

    print("Model Performance Comparison")
    print(comparison_df.to_string(index=False))
    print("\nROC-AUC Scores")
    for name, auc in auc_scores.items():
        print(f"{name}: {auc:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
