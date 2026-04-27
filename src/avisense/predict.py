from pathlib import Path
import joblib
import pandas as pd

from avisense.features import extract_audio_features


def predict_audio_file(
    audio_path,
    model_path,
    label_encoder_path,
    target_sr=22050,
    n_mfcc=20,
):
    """
    Predict the species of a single audio file.
    """
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    features = extract_audio_features(
        audio_path=audio_path,
        species=None,
        target_sr=target_sr,
        n_mfcc=n_mfcc,
    )

    df = pd.DataFrame([features])

    ignore_columns = {
        "species",
        "filename",
        "full_path",
        "relative_path",
        "quality_rating",
    }

    X = df[[col for col in df.columns if col not in ignore_columns]]
    X = X.select_dtypes(include=["number"])

    pred_encoded = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    result = {
        "audio_path": str(audio_path),
        "predicted_species": pred_label,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = label_encoder.inverse_transform(range(len(probabilities)))

        result["probabilities"] = {
            species: float(prob)
            for species, prob in zip(classes, probabilities)
        }

    return result