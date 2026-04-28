from pathlib import Path
import joblib
import pandas as pd

from avisense.config import load_config, ensure_directories
from avisense.dataset import (
    load_features,
    prepare_xy,
    encode_labels,
    make_train_test_split,
)
from avisense.train import (
    train_and_evaluate_models,
    fit_best_model,
    save_model,
)
from avisense.plots import plot_model_comparison


def save_split_indices(X_train, X_test, output_dir):
    """
    Save train and test indices for documentation and reproducibility.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"train_index": X_train.index}).to_csv(
        output_dir / "train_indices.csv",
        index=False,
    )

    pd.DataFrame({"test_index": X_test.index}).to_csv(
        output_dir / "test_indices.csv",
        index=False,
    )


def main():
    config = load_config()
    ensure_directories(config)

    features_path = Path(config["paths"]["features_dir"]) / "recording_features.csv"
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    splits_dir = Path(config["paths"]["splits_dir"])

    print("Loading extracted features...")
    df = load_features(features_path)

    print(f"Feature table shape: {df.shape}")
    print("Species distribution:")
    print(df["species"].value_counts())

    X, y = prepare_xy(df, label_column="species")
    y_encoded, label_encoder = encode_labels(y)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print("Classes:")
    for class_name in label_encoder.classes_:
        print(f"- {class_name}")

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y_encoded,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    save_split_indices(
        X_train=X_train,
        X_test=X_test,
        output_dir=splits_dir,
    )

    print("Training and cross-validating baseline models...")

    results_df = train_and_evaluate_models(
        X_train=X_train,
        y_train=y_train,
        cv_folds=config["training"]["cv_folds"],
        random_state=config["training"]["random_state"],
    )

    results_path = reports_dir / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    plot_model_comparison(
        results_df=results_df,
        output_path=figures_dir / "model_comparison.png",
    )

    best_model_name = results_df.iloc[0]["model"]

    print("Fitting best model on full training set...")
    best_model = fit_best_model(
        model_name=best_model_name,
        X_train=X_train,
        y_train=y_train,
        random_state=config["training"]["random_state"],
    )

    save_model(best_model, models_dir / "best_model.joblib")
    joblib.dump(label_encoder, models_dir / "label_encoder.joblib")

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_indices": X_test.index.tolist(),
            "train_indices": X_train.index.tolist(),
            "feature_columns": X.columns.tolist(),
            "class_names": label_encoder.classes_.tolist(),
            "best_model_name": best_model_name,
        },
        models_dir / "train_test_split.joblib",
    )

    print("\nModel comparison:")
    print(results_df)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {models_dir / 'best_model.joblib'}")
    print(f"Saved label encoder to: {models_dir / 'label_encoder.joblib'}")
    print(f"Saved model comparison to: {results_path}")
    print(f"Saved model comparison plot to: {figures_dir / 'model_comparison.png'}")


if __name__ == "__main__":
    main()