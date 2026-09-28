"""
Data loading and preprocessing for the phishing website detection dataset.

Mirrors the preprocessing performed in
notebooks/phishing_detection_analysis.ipynb (Sections 2, 3 and 5):
  - load the UCI phishing websites CSV
  - basic data-quality checks (missing values, duplicate rows, class balance)
  - separate features (X) from the target (y), dropping the non-predictive
    'id' column
  - stratified 80/20 train-test split with a fixed random seed
  - feature scaling with StandardScaler, fit on the training split only
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Result"
ID_COLUMN = "id"


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the raw phishing dataset CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def run_data_quality_checks(df: pd.DataFrame) -> dict:
    """Return basic data-quality statistics used in the analysis."""
    return {
        "shape": df.shape,
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
    }


def split_and_scale(df: pd.DataFrame, test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE):
    """
    Split the dataset into stratified train/test sets and scale features.

    Returns
    -------
    X_train, X_test : unscaled feature DataFrames (kept for feature-name
        lookups used by the explainability step)
    X_train_scaled, X_test_scaled : numpy arrays, StandardScaler-transformed
    y_train, y_test : target Series
    scaler : the fitted StandardScaler (fit on the training split only)
    """
    X = df.drop([TARGET_COLUMN, ID_COLUMN], axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_dataset("data/uci-ml-phishing-dataset.csv")
    stats = run_data_quality_checks(df)
    print("Dataset shape:", stats["shape"])
    print("Missing values (total):", stats["missing_values_total"])
    print("Duplicate rows:", stats["duplicate_rows"])
    print("Class distribution:", stats["class_distribution"])
