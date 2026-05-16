from pathlib import Path
import numpy as np
import pandas as pd
import librosa


def summarize_feature(values: np.ndarray, prefix: str) -> dict:
    """
    Summarize a time-varying feature into mean, std, min, and max.
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


def extract_time_domain_features(y, prefix="time"):
    """
    Extract simple time-domain features from waveform amplitude.

    These describe waveform amplitude, silence, and energy distribution.
    """
    y = np.asarray(y)

    if len(y) == 0:
        return {
            f"{prefix}_amplitude_mean": 0.0,
            f"{prefix}_amplitude_std": 0.0,
            f"{prefix}_amplitude_min": 0.0,
            f"{prefix}_amplitude_max": 0.0,
            f"{prefix}_amplitude_range": 0.0,
            f"{prefix}_abs_amplitude_mean": 0.0,
            f"{prefix}_abs_amplitude_std": 0.0,
            f"{prefix}_peak_amplitude": 0.0,
            f"{prefix}_silence_ratio": 0.0,
            f"{prefix}_energy_mean": 0.0,
            f"{prefix}_energy_std": 0.0,
            f"{prefix}_energy_entropy": 0.0,
        }

    abs_y = np.abs(y)
    energy = y ** 2

    silence_threshold = 0.01
    silence_ratio = float(np.mean(abs_y < silence_threshold))

    energy_sum = np.sum(energy)

    if energy_sum == 0:
        energy_entropy = 0.0
    else:
        energy_prob = energy / energy_sum
        energy_entropy = float(
            -np.sum(energy_prob * np.log2(energy_prob + 1e-12))
        )

    return {
        f"{prefix}_amplitude_mean": float(np.mean(y)),
        f"{prefix}_amplitude_std": float(np.std(y)),
        f"{prefix}_amplitude_min": float(np.min(y)),
        f"{prefix}_amplitude_max": float(np.max(y)),
        f"{prefix}_amplitude_range": float(np.max(y) - np.min(y)),
        f"{prefix}_abs_amplitude_mean": float(np.mean(abs_y)),
        f"{prefix}_abs_amplitude_std": float(np.std(abs_y)),
        f"{prefix}_peak_amplitude": float(np.max(abs_y)),
        f"{prefix}_silence_ratio": silence_ratio,
        f"{prefix}_energy_mean": float(np.mean(energy)),
        f"{prefix}_energy_std": float(np.std(energy)),
        f"{prefix}_energy_entropy": energy_entropy,
    }


def extract_audio_features(
    audio_path,
    species=None,
    target_sr=22050,
    n_mfcc=20,
):
    """
    Extract recording-level acoustic features from one audio file.
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

    row.update(extract_time_domain_features(y, prefix="time"))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    row.update(summarize_feature(mfcc, "mfcc"))

    delta = librosa.feature.delta(mfcc)
    row.update(summarize_feature(delta, "delta_mfcc"))

    delta2 = librosa.feature.delta(mfcc, order=2)
    row.update(summarize_feature(delta2, "delta2_mfcc"))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    row.update(summarize_feature(spectral_centroid, "spectral_centroid"))
    row.update(summarize_feature(spectral_bandwidth, "spectral_bandwidth"))
    row.update(summarize_feature(spectral_rolloff, "spectral_rolloff"))
    row.update(summarize_feature(spectral_contrast, "spectral_contrast"))

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    row.update(summarize_feature(zero_crossing_rate, "zero_crossing_rate"))
    row.update(summarize_feature(rms, "rms"))

    return row


def extract_features_from_inventory(
    inventory_df: pd.DataFrame,
    target_sr=22050,
    n_mfcc=20,
):
    """
    Extract recording-level features for all audio files listed in an inventory dataframe.
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