from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from avisense.config import load_config, ensure_directories
from avisense.evaluate import save_classification_report, save_confusion_matrix


def calculate_metrics(y_true, y_pred):
    """
    Calculate standard classification metrics.
    """
    return {
        "total_samples": len(y_true),
        "correct_predictions": int((y_true == y_pred).sum()),
        "incorrect_predictions": int((y_true != y_pred).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def aggregate_recording_predictions(model, X_test, y_test, test_metadata, label_encoder):
    """
    Aggregate segment predictions into one prediction per original recording.

    If the model supports predict_proba, average probabilities across segments.
    Otherwise, use majority voting.
    """
    metadata = test_metadata.reset_index(drop=True).copy()

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)

        prob_cols = [f"prob_{i}" for i in range(probs.shape[1])]
        prob_df = pd.DataFrame(probs, columns=prob_cols)

        prob_df["recording_id"] = metadata["recording_id"].values
        prob_df["true_encoded"] = y_test

        avg_probs = prob_df.groupby("recording_id")[prob_cols].mean()
        recording_pred_encoded = avg_probs.values.argmax(axis=1)

        true_encoded = (
            prob_df.groupby("recording_id")["true_encoded"]
            .first()
            .loc[avg_probs.index]
            .values
        )

        recording_ids = avg_probs.index.tolist()

    else:
        segment_pred = model.predict(X_test)

        vote_df = pd.DataFrame(
            {
                "recording_id": metadata["recording_id"].values,
                "true_encoded": y_test,
                "pred_encoded": segment_pred,
            }
        )

        recording_ids = []
        true_encoded = []
        recording_pred_encoded = []

        for recording_id, group in vote_df.groupby("recording_id"):
            recording_ids.append(recording_id)
            true_encoded.append(group["true_encoded"].iloc[0])
            recording_pred_encoded.append(group["pred_encoded"].mode().iloc[0])

        true_encoded = np.array(true_encoded)
        recording_pred_encoded = np.array(recording_pred_encoded)

    output_df = pd.DataFrame(
        {
            "recording_id": recording_ids,
            "true_label": label_encoder.inverse_transform(true_encoded),
            "predicted_label": label_encoder.inverse_transform(recording_pred_encoded),
            "correct": true_encoded == recording_pred_encoded,
        }
    )

    return true_encoded, recording_pred_encoded, output_df


def main():
    config = load_config()
    ensure_directories(config)

    models_dir = Path(config["paths"]["models_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    predictions_dir = Path(config["paths"]["predictions_dir"])

    model_path = models_dir / "best_segment_model.joblib"
    label_encoder_path = models_dir / "segment_label_encoder.joblib"
    split_path = models_dir / "segment_train_test_split.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Segment model not found: {model_path}. "
            "Run python scripts/09_train_segment_models.py first."
        )

    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Segment label encoder not found: {label_encoder_path}")

    if not split_path.exists():
        raise FileNotFoundError(f"Segment split file not found: {split_path}")

    print("Loading segment model...")
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    split = joblib.load(split_path)

    X_test = split["X_test"]
    y_test = split["y_test"]
    test_metadata = split["test_metadata"]
    best_model_name = split.get("best_model_name", "best_segment_model")

    print(f"Evaluating segment model: {best_model_name}")

    # -------------------------------------------------------
    # Segment-level evaluation
    # -------------------------------------------------------
    y_pred_segment = model.predict(X_test)

    save_classification_report(
        y_true=y_test,
        y_pred=y_pred_segment,
        target_names=label_encoder.classes_,
        output_path=reports_dir / "segment_classification_report.txt",
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred_segment,
        target_names=label_encoder.classes_,
        output_path=figures_dir / "segment_confusion_matrix.png",
    )

    segment_metrics = calculate_metrics(y_test, y_pred_segment)
    segment_metrics["model"] = best_model_name
    segment_metrics["evaluation_level"] = "segment"

    pd.DataFrame([segment_metrics]).to_csv(
        reports_dir / "segment_final_metrics.csv",
        index=False,
    )

    segment_predictions = test_metadata[
        ["recording_id", "segment_id", "filename", "relative_path", "species"]
    ].copy()

    segment_predictions["true_label"] = label_encoder.inverse_transform(y_test)
    segment_predictions["predicted_label"] = label_encoder.inverse_transform(y_pred_segment)
    segment_predictions["correct"] = y_test == y_pred_segment

    segment_predictions.to_csv(
        predictions_dir / "segment_test_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------
    # Recording-level evaluation aggregated from segments
    # -------------------------------------------------------
    (
        y_true_recording,
        y_pred_recording,
        recording_predictions,
    ) = aggregate_recording_predictions(
        model=model,
        X_test=X_test,
        y_test=y_test,
        test_metadata=test_metadata,
        label_encoder=label_encoder,
    )

    recording_report = classification_report(
        y_true_recording,
        y_pred_recording,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    (reports_dir / "segment_recording_level_classification_report.txt").write_text(
        recording_report,
        encoding="utf-8",
    )

    save_confusion_matrix(
        y_true=y_true_recording,
        y_pred=y_pred_recording,
        target_names=label_encoder.classes_,
        output_path=figures_dir / "segment_recording_level_confusion_matrix.png",
    )

    recording_metrics = calculate_metrics(y_true_recording, y_pred_recording)
    recording_metrics["model"] = best_model_name
    recording_metrics["evaluation_level"] = "recording_aggregated_from_segments"

    pd.DataFrame([recording_metrics]).to_csv(
        reports_dir / "segment_recording_level_final_metrics.csv",
        index=False,
    )

    recording_predictions.to_csv(
        predictions_dir / "segment_recording_level_predictions.csv",
        index=False,
    )

    print("\nSegment-level metrics:")
    print(pd.DataFrame([segment_metrics]))

    print("\nRecording-level metrics aggregated from segments:")
    print(pd.DataFrame([recording_metrics]))

    print("\nSaved files:")
    print(f"- {reports_dir / 'segment_classification_report.txt'}")
    print(f"- {reports_dir / 'segment_final_metrics.csv'}")
    print(f"- {figures_dir / 'segment_confusion_matrix.png'}")
    print(f"- {predictions_dir / 'segment_test_predictions.csv'}")
    print(f"- {reports_dir / 'segment_recording_level_classification_report.txt'}")
    print(f"- {reports_dir / 'segment_recording_level_final_metrics.csv'}")
    print(f"- {figures_dir / 'segment_recording_level_confusion_matrix.png'}")
    print(f"- {predictions_dir / 'segment_recording_level_predictions.csv'}")


if __name__ == "__main__":
    main()