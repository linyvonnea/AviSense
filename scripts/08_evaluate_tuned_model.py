from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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

    tuned_model_path = models_dir / "best_tuned_model.joblib"
    label_encoder_path = models_dir / "label_encoder.joblib"
    split_path = models_dir / "tuned_train_test_split.joblib"

    if not tuned_model_path.exists():
        raise FileNotFoundError(f"Tuned model not found: {tuned_model_path}")

    if not split_path.exists():
        raise FileNotFoundError(f"Tuned split file not found: {split_path}")

    print("Loading tuned model...")
    model = joblib.load(tuned_model_path)
    label_encoder = joblib.load(label_encoder_path)
    split = joblib.load(split_path)

    print("Loading feature table...")
    df = load_features(features_path)

    X_test = split["X_test"]
    y_test = split["y_test"]
    test_indices = split["test_indices"]

    best_model_name = split.get("best_model_name", "best_tuned_model")

    print(f"Evaluating tuned model: {best_model_name}")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    save_classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=label_encoder.classes_,
        output_path=reports_dir / "tuned_classification_report.txt",
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        target_names=label_encoder.classes_,
        output_path=figures_dir / "tuned_confusion_matrix.png",
    )

    test_filenames = df.loc[test_indices, "filename"].tolist()

    predictions_df = save_predictions(
        filenames=test_filenames,
        y_true=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        output_path=predictions_dir / "tuned_test_predictions.csv",
    )

    metrics = {
        "model": best_model_name,
        "total_test_samples": len(y_test),
        "correct_predictions": int((y_test == y_pred).sum()),
        "incorrect_predictions": int((y_test != y_pred).sum()),
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(reports_dir / "tuned_final_metrics.csv", index=False)

    print("\nTuned model evaluation complete.")
    print(metrics_df)

    print(f"\nSaved tuned classification report to: {reports_dir / 'tuned_classification_report.txt'}")
    print(f"Saved tuned confusion matrix to: {figures_dir / 'tuned_confusion_matrix.png'}")
    print(f"Saved tuned predictions to: {predictions_dir / 'tuned_test_predictions.csv'}")
    print(f"Saved tuned final metrics to: {reports_dir / 'tuned_final_metrics.csv'}")

    print("\nPrediction preview:")
    print(predictions_df.head())


if __name__ == "__main__":
    main()