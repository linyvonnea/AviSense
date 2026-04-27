from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def save_classification_report(y_true, y_pred, target_names, output_path):
    """
    Save classification report as a text file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    output_path.write_text(report, encoding="utf-8")


def save_confusion_matrix(y_true, y_pred, target_names, output_path):
    """
    Save confusion matrix as an image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d",
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_predictions(filenames, y_true, y_pred, label_encoder, output_path):
    """
    Save test predictions to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "filename": filenames,
            "true_label": label_encoder.inverse_transform(y_true),
            "predicted_label": label_encoder.inverse_transform(y_pred),
            "correct": y_true == y_pred,
        }
    )

    df.to_csv(output_path, index=False)

    return df