from pathlib import Path
import joblib
import pandas as pd

from avisense.config import load_config, ensure_directories
from avisense.dataset import load_features
from avisense.evaluate import (
    save_classification_report,
    save_confusion_matrix,
    save_predictions,
)


def main():
    config = load_config()
    ensure_directories(config)

    features_path = Path(config["paths"]["features_dir"]) / "recording_features.csv"
    models_dir = Path(config["paths"]["models_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    predictions_dir = Path(config["paths"]["predictions_dir"])

    model = joblib.load(models_dir / "best_model.joblib")
    label_encoder = joblib.load(models_dir / "label_encoder.joblib")
    split = joblib.load(models_dir / "train_test_split.joblib")

    df = load_features(features_path)

    X_test = split["X_test"]
    y_test = split["y_test"]
    test_indices = split["test_indices"]

    y_pred = model.predict(X_test)

    save_classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=label_encoder.classes_,
        output_path=reports_dir / "classification_report.txt",
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        target_names=label_encoder.classes_,
        output_path=figures_dir / "confusion_matrix.png",
    )

    test_filenames = df.loc[test_indices, "filename"].tolist()

    predictions_df = save_predictions(
        filenames=test_filenames,
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        output_path=predictions_dir / "test_predictions.csv",
    )

    metrics = {
        "total_test_samples": len(y_test),
        "correct_predictions": int((y_test == y_pred).sum()),
        "incorrect_predictions": int((y_test != y_pred).sum()),
        "accuracy": float((y_test == y_pred).mean()),
    }

    pd.DataFrame([metrics]).to_csv(
        reports_dir / "final_metrics.csv",
        index=False,
    )

    print("Evaluation complete.")
    print(f"Saved classification report to: {reports_dir / 'classification_report.txt'}")
    print(f"Saved confusion matrix to: {figures_dir / 'confusion_matrix.png'}")
    print(f"Saved predictions to: {predictions_dir / 'test_predictions.csv'}")
    print(predictions_df.head())


if __name__ == "__main__":
    main()