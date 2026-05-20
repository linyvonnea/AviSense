from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_validate

from avisense.config import load_config, ensure_directories
from avisense.dataset import encode_labels
from avisense.train import get_models, make_pipeline, fit_best_model, save_model
from avisense.plots import plot_model_comparison


def prepare_segment_xy(df, label_column="species"):
    """
    Prepare X and y for segment-level classification.

    Important:
    Metadata columns are excluded so the model only learns from audio features.
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
    Split by original recording_id, not by segment.

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


def train_and_evaluate_segment_models(
    X_train,
    y_train,
    groups_train,
    cv_folds=5,
    random_state=42,
):
    """
    Train and cross-validate segment-level models using grouped CV.

    StratifiedGroupKFold keeps all segments from the same original recording
    in the same fold.
    """
    models = get_models(random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    cv = StratifiedGroupKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    rows = []

    for model_name, model in models.items():
        print(f"Training segment model: {model_name}")

        pipeline = make_pipeline(model_name, model)

        try:
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                groups=groups_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False,
            )

            rows.append(
                {
                    "model": model_name,
                    "accuracy_mean": scores["test_accuracy"].mean(),
                    "accuracy_std": scores["test_accuracy"].std(),
                    "precision_macro_mean": scores["test_precision_macro"].mean(),
                    "precision_macro_std": scores["test_precision_macro"].std(),
                    "recall_macro_mean": scores["test_recall_macro"].mean(),
                    "recall_macro_std": scores["test_recall_macro"].std(),
                    "f1_macro_mean": scores["test_f1_macro"].mean(),
                    "f1_macro_std": scores["test_f1_macro"].std(),
                }
            )

        except Exception as error:
            print(f"Model failed: {model_name}")
            print(f"Reason: {error}")

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        raise RuntimeError("No segment models were successfully trained.")

    results_df = results_df.sort_values("f1_macro_mean", ascending=False)

    return results_df


def main():
    config = load_config()
    ensure_directories(config)

    features_path = Path(config["paths"]["features_dir"]) / "segment_features.csv"
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    splits_dir = Path(config["paths"]["splits_dir"])

    random_state = config["training"]["random_state"]

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
        raise ValueError(f"Missing required columns in segment_features.csv: {missing_columns}")

    print(f"Segment feature table shape: {df.shape}")

    print("\nSegments per species:")
    print(df["species"].value_counts())

    print("\nSegments per recording:")
    print(df.groupby("recording_id").size().describe())

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
        splits_dir / "segment_train_recordings.csv",
        index=False,
    )

    test_recordings.to_csv(
        splits_dir / "segment_test_recordings.csv",
        index=False,
    )

    print("\nTraining and cross-validating segment models...")

    results_df = train_and_evaluate_segment_models(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        cv_folds=config["training"]["cv_folds"],
        random_state=random_state,
    )

    results_path = reports_dir / "segment_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    plot_model_comparison(
        results_df=results_df,
        output_path=figures_dir / "segment_model_comparison.png",
    )

    best_model_name = results_df.iloc[0]["model"]

    print(f"\nBest segment model: {best_model_name}")
    print("Fitting best segment model on full segment training set...")

    best_model = fit_best_model(
        model_name=best_model_name,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
    )

    save_model(best_model, models_dir / "best_segment_model.joblib")
    joblib.dump(label_encoder, models_dir / "segment_label_encoder.joblib")

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
            "best_model_name": best_model_name,
        },
        models_dir / "segment_train_test_split.joblib",
    )

    print("\nSegment model comparison:")
    print(results_df)

    print(f"\nSaved segment model comparison to: {results_path}")
    print(f"Saved segment model comparison plot to: {figures_dir / 'segment_model_comparison.png'}")
    print(f"Saved best segment model to: {models_dir / 'best_segment_model.joblib'}")


if __name__ == "__main__":
    main()