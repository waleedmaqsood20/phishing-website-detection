"""
Train the four classifiers used in this study on the phishing dataset.

Mirrors notebooks/phishing_detection_analysis.ipynb (Sections 6, 7, 8):
  - Random Forest (default hyperparameters, random_state=42)
  - Decision Tree (default hyperparameters, random_state=42)
  - Logistic Regression (max_iter=1000, random_state=42)
  - Support Vector Machine (default hyperparameters, random_state=42)

No cross-validation or hyperparameter search is performed here, consistent
with the original analysis. The trained Random Forest model is saved with
joblib for reuse (e.g. by explainability.py); model files are not committed
to the repository (see .gitignore) since they can be regenerated in seconds
from this script.
"""
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import RANDOM_STATE, load_dataset, split_and_scale

MODELS_DIR = "models"


def train_all(X_train_scaled, y_train):
    """Train all four models and return them in a dict keyed by name."""
    models = {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": SVC(random_state=RANDOM_STATE),
    }
    for model in models.values():
        model.fit(X_train_scaled, y_train)
    return models


if __name__ == "__main__":
    df = load_dataset("data/uci-ml-phishing-dataset.csv")
    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, scaler) = split_and_scale(df)

    models = train_all(X_train_scaled, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    rf_path = os.path.join(MODELS_DIR, "random_forest_phishing_model.pkl")
    joblib.dump(models["Random Forest"], rf_path)
    print(f"Trained Random Forest model saved to: {rf_path}")
    print(f"Trained {len(models)} models: {list(models.keys())}")
