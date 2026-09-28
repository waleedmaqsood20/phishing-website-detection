"""
Model interpretability: feature importance and SHAP analysis.

Mirrors notebooks/phishing_detection_analysis.ipynb (Sections 11 and 12):
  - Random Forest and Decision Tree feature importances
  - Logistic Regression coefficient magnitudes (normalised to sum to 1, for
    rough comparability with the tree-based importances)
  - SHAP TreeExplainer values for the Random Forest model, computed on a
    sample of the test set (500 samples, matching the notebook), aggregated
    as mean absolute SHAP value per feature

Results are written to:
  - results/feature_importance.csv (RF / DT / LR importances)
  - results/shap_feature_importance.csv
"""
import os

import numpy as np
import pandas as pd
import shap

from data_preprocessing import load_dataset, split_and_scale
from train_models import train_all

RESULTS_DIR = "results"
SHAP_SAMPLE_SIZE = 500


def feature_importance_table(models, feature_names):
    rf_importances = models["Random Forest"].feature_importances_
    dt_importances = models["Decision Tree"].feature_importances_

    lr_coefficients = np.abs(models["Logistic Regression"].coef_[0])
    lr_importance_normalized = lr_coefficients / lr_coefficients.sum()

    df = pd.DataFrame({
        "Feature": feature_names,
        "RF_Importance": rf_importances,
        "DT_Importance": dt_importances,
        "LR_Importance": lr_importance_normalized,
    })
    return df.sort_values("RF_Importance", ascending=False)


def shap_importance_table(rf_model, X_test_scaled, feature_names,
                           sample_size: int = SHAP_SAMPLE_SIZE):
    explainer = shap.TreeExplainer(rf_model)
    sample_size = min(sample_size, len(X_test_scaled))
    X_test_sample = X_test_scaled[:sample_size]

    shap_values = explainer.shap_values(X_test_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class ("Legitimate")
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "SHAP_Importance": mean_abs_shap})
    return df.sort_values("SHAP_Importance", ascending=False)


if __name__ == "__main__":
    df = load_dataset("data/uci-ml-phishing-dataset.csv")
    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, scaler) = split_and_scale(df)

    models = train_all(X_train_scaled, y_train)
    feature_names = X_train.columns.tolist()

    importance_df = feature_importance_table(models, feature_names)
    shap_df = shap_importance_table(models["Random Forest"], X_test_scaled, feature_names)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    importance_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
    shap_df.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"), index=False)

    print("Top 10 features by Random Forest importance:")
    print(importance_df.head(10).to_string(index=False))
    print("\nTop 10 features by SHAP importance:")
    print(shap_df.head(10).to_string(index=False))
