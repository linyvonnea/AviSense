from pathlib import Path
import joblib

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


def main():
    config = load_config()
    ensure_directories(config)

    features_path = Path(config["paths"]["features_dir"]) / "recording_features.csv"
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    models_dir = Path(config["paths"]["models_dir"])

    df = load_features(features_path)

    X, y = prepare_xy(df, label_column="species")
    y_encoded, label_encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y_encoded,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )

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
        },
        models_dir / "train_test_split.joblib",
    )

    print("Model comparison:")
    print(results_df)
    print(f"Best model: {best_model_name}")
    print(f"Saved model to: {models_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    main()