from pathlib import Path
import numpy as np
import pandas as pd
import librosa


def summarize_feature(values, prefix):
    """
    Summarize time-varying features into mean, std, min, and max.
    """
    values = np.asarray(values)

    if values.ndim == 1:
        return {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_max": float(np.max(values)),
        }

    features = {}

    for i in range(values.shape[0]):
        features[f"{prefix}_{i + 1}_mean"] = float(np.mean(values[i]))
        features[f"{prefix}_{i + 1}_std"] = float(np.std(values[i]))
        features[f"{prefix}_{i + 1}_min"] = float(np.min(values[i]))
        features[f"{prefix}_{i + 1}_max"] = float(np.max(values[i]))

    return features


def extract_audio_features(audio_path, species=None, target_sr=22050, n_mfcc=20):
    """
    Extract acoustic features from one audio file.
    """
    audio_path = Path(audio_path)

    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    if len(y) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    row = {
        "filename": audio_path.name,
        "full_path": str(audio_path),
        "species": species,
        "duration": float(librosa.get_duration(y=y, sr=sr)),
        "sample_rate": sr,
    }

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    row.update(summarize_feature(mfcc, "mfcc"))

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)
    row.update(summarize_feature(delta, "delta_mfcc"))

    # Delta-delta MFCC
    delta2 = librosa.feature.delta(mfcc, order=2)
    row.update(summarize_feature(delta2, "delta2_mfcc"))

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    row.update(summarize_feature(spectral_centroid, "spectral_centroid"))
    row.update(summarize_feature(spectral_bandwidth, "spectral_bandwidth"))
    row.update(summarize_feature(spectral_rolloff, "spectral_rolloff"))
    row.update(summarize_feature(spectral_contrast, "spectral_contrast"))

    # Temporal and energy features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    row.update(summarize_feature(zero_crossing_rate, "zero_crossing_rate"))
    row.update(summarize_feature(rms, "rms"))

    return row


def extract_features_from_inventory(inventory_df, target_sr=22050, n_mfcc=20):
    """
    Extract features for every audio file listed in the inventory dataframe.
    """
    rows = []

    for _, item in inventory_df.iterrows():
        audio_path = item["full_path"]
        species = item["species"]

        try:
            features = extract_audio_features(
                audio_path=audio_path,
                species=species,
                target_sr=target_sr,
                n_mfcc=n_mfcc,
            )

            features["relative_path"] = item.get("relative_path", None)
            features["quality_rating"] = item.get("quality_rating", None)

            rows.append(features)

        except Exception as error:
            print(f"Failed to process {audio_path}: {error}")

    return pd.DataFrame(rows)