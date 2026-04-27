from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_features(features_path):
    """
    Load extracted feature table.
    """
    features_path = Path(features_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    return pd.read_csv(features_path)


def prepare_xy(df, label_column="species"):
    """
    Prepare X features and y labels.
    """
    ignore_columns = {
        label_column,
        "filename",
        "full_path",
        "relative_path",
        "quality_rating",
    }

    feature_columns = [col for col in df.columns if col not in ignore_columns]

    X = df[feature_columns].copy()
    X = X.select_dtypes(include=["number"])

    y = df[label_column].copy()

    return X, y


def encode_labels(y):
    """
    Encode string species labels into numeric labels.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def make_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create stratified train-test split.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )