from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from avisense.config import load_config, ensure_directories
from avisense.dataset import encode_labels


def prepare_segment_xy(df, label_column="species"):
    """
    Prepare X and y for segment-level classification.

    Excludes metadata columns to prevent leakage.
    """
    ignore_columns = {
        label_column,
        "filename",
        "full_path",
        "relative_path",
        "quality_rating",
        "recording_id",
        "segment_id",
        "segment_start_sec",
        "segment_end_sec",
        "segment_rms",
    }

    feature_columns = [col for col in df.columns if col not in ignore_columns]

    X = df[feature_columns].copy()
    X = X.select_dtypes(include=["number"])
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().sum().sum() > 0:
        X = X.fillna(X.median(numeric_only=True))

    y = df[label_column].copy()

    return X, y


def make_recording_level_split(df, test_size=0.2, random_state=42):
    """
    Split by original recording_id, not by segment row.

    This prevents data leakage.
    """
    recording_df = (
        df[["recording_id", "species"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    train_recordings, test_recordings = train_test_split(
        recording_df,
        test_size=test_size,
        random_state=random_state,
        stratify=recording_df["species"],
    )

    train_recording_ids = set(train_recordings["recording_id"])
    test_recording_ids = set(test_recordings["recording_id"])

    train_mask = df["recording_id"].isin(train_recording_ids)
    test_mask = df["recording_id"].isin(test_recording_ids)

    return train_mask, test_mask, train_recordings, test_recordings


def make_search(estimator, param_distributions, cv, n_iter=30, random_state=42):
    """
    Create RandomizedSearchCV object.
    """
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=random_state,
        refit=True,
        return_train_score=False,
    )


def tune_xgboost(X_train, y_train, groups_train, cv, random_state=42):
    """
    Tune XGBoost if available.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=1,
    )

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1, 5, 10],
    }

    search = make_search(
        estimator=model,
        param_distributions=param_distributions,
        cv=cv,
        n_iter=30,
        random_state=random_state,
    )

    search.fit(X_train, y_train, groups=groups_train)

    return search


def tune_random_forest(X_train, y_train, groups_train, cv, random_state=42):
    """
    Tune Random Forest.
    """
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )

    param_distributions = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    search = make_search(
        estimator=model,
        param_distributions=param_distributions,
        cv=cv,
        n_iter=30,
        random_state=random_state,
    )

    search.fit(X_train, y_train, groups=groups_train)

    return search


def tune_extra_trees(X_train, y_train, groups_train, cv, random_state=42):
    """
    Tune Extra Trees.
    """
    model = ExtraTreesClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )

    param_distributions = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    search = make_search(
        estimator=model,
        param_distributions=param_distributions,
        cv=cv,
        n_iter=30,
        random_state=random_state,
    )

    search.fit(X_train, y_train, groups=groups_train)

    return search


def tune_svm(X_train, y_train, groups_train, cv, random_state=42):
    """
    Tune SVM with RBF kernel.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    param_distributions = {
        "model__C": [0.1, 1, 10, 50, 100],
        "model__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1],
    }

    search = make_search(
        estimator=pipeline,
        param_distributions=param_distributions,
        cv=cv,
        n_iter=20,
        random_state=random_state,
    )

    search.fit(X_train, y_train, groups=groups_train)

    return search


def main():
    config = load_config()
    ensure_directories(config)

    random_state = config["training"]["random_state"]

    features_path = Path(config["paths"]["features_dir"]) / "segment_features.csv"
    reports_dir = Path(config["paths"]["reports_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    splits_dir = Path(config["paths"]["splits_dir"])

    if not features_path.exists():
        raise FileNotFoundError(
            f"Segment feature file not found: {features_path}. "
            "Run python scripts/03_extract_segment_features.py first."
        )

    print("Loading segment features...")
    df = pd.read_csv(features_path)

    if df.empty:
        raise ValueError("segment_features.csv is empty.")

    required_columns = {"species", "recording_id"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Segment feature table shape: {df.shape}")
    print("\nSegments per species:")
    print(df["species"].value_counts())

    X, y = prepare_segment_xy(df, label_column="species")
    y_encoded, label_encoder = encode_labels(y)

    train_mask, test_mask, train_recordings, test_recordings = make_recording_level_split(
        df=df,
        test_size=config["training"]["test_size"],
        random_state=random_state,
    )

    train_mask_values = train_mask.values
    test_mask_values = test_mask.values

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y_encoded[train_mask_values]
    y_test = y_encoded[test_mask_values]

    train_metadata = df.loc[train_mask].copy()
    test_metadata = df.loc[test_mask].copy()

    groups_train = train_metadata["recording_id"].values

    print(f"\nTraining segments: {X_train.shape[0]}")
    print(f"Testing segments: {X_test.shape[0]}")
    print(f"Training recordings: {train_recordings.shape[0]}")
    print(f"Testing recordings: {test_recordings.shape[0]}")

    train_recordings.to_csv(
        splits_dir / "tuned_segment_train_recordings.csv",
        index=False,
    )

    test_recordings.to_csv(
        splits_dir / "tuned_segment_test_recordings.csv",
        index=False,
    )

    cv = StratifiedGroupKFold(
        n_splits=config["training"]["cv_folds"],
        shuffle=True,
        random_state=random_state,
    )

    searches = {}

    print("\nTuning XGBoost...")
    try:
        searches["xgboost_segment_tuned"] = tune_xgboost(
            X_train,
            y_train,
            groups_train,
            cv,
            random_state,
        )
    except Exception as error:
        print(f"Skipping XGBoost. Reason: {error}")

    print("\nTuning Random Forest...")
    searches["random_forest_segment_tuned"] = tune_random_forest(
        X_train,
        y_train,
        groups_train,
        cv,
        random_state,
    )

    print("\nTuning Extra Trees...")
    searches["extra_trees_segment_tuned"] = tune_extra_trees(
        X_train,
        y_train,
        groups_train,
        cv,
        random_state,
    )

    print("\nTuning SVM RBF...")
    searches["svm_rbf_segment_tuned"] = tune_svm(
        X_train,
        y_train,
        groups_train,
        cv,
        random_state,
    )

    rows = []

    for name, search in searches.items():
        rows.append(
            {
                "model": name,
                "best_cv_f1_macro": search.best_score_,
                "best_params": search.best_params_,
            }
        )

        joblib.dump(search.best_estimator_, models_dir / f"{name}.joblib")

    if not rows:
        raise RuntimeError("No tuned segment models were successfully trained.")

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("best_cv_f1_macro", ascending=False)

    results_path = reports_dir / "segment_tuned_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = searches[best_name].best_estimator_

    joblib.dump(best_model, models_dir / "best_tuned_segment_model.joblib")
    joblib.dump(label_encoder, models_dir / "tuned_segment_label_encoder.joblib")

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_metadata": train_metadata,
            "test_metadata": test_metadata,
            "train_recordings": train_recordings,
            "test_recordings": test_recordings,
            "feature_columns": X.columns.tolist(),
            "class_names": label_encoder.classes_.tolist(),
            "best_model_name": best_name,
        },
        models_dir / "tuned_segment_train_test_split.joblib",
    )

    print("\nTuned segment model comparison:")
    print(results_df)

    print(f"\nBest tuned segment model: {best_name}")
    print(f"Saved tuned segment comparison to: {results_path}")
    print(f"Saved best tuned segment model to: {models_dir / 'best_tuned_segment_model.joblib'}")


if __name__ == "__main__":
    main()