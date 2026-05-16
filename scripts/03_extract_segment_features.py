from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from avisense.config import load_config, ensure_directories
from avisense.data_inventory import build_inventory
from avisense.features import summarize_feature, extract_time_domain_features


def normalize_audio(y):
    """
    Normalize audio amplitude safely.
    """
    if len(y) == 0:
        return y

    max_val = np.max(np.abs(y))

    if max_val == 0:
        return y

    return y / max_val


def trim_silence(y, top_db=30):
    """
    Trim leading and trailing silence.
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def create_segments(
    y,
    sr,
    segment_duration_seconds=5,
    segment_overlap_seconds=2.5,
    max_segments_per_recording=3,
):
    """
    Create fixed-length audio segments and keep the highest-energy segments.

    Returns a list of:
    segment_id, start_sec, end_sec, segment_audio, segment_rms
    """
    segment_length = int(sr * segment_duration_seconds)
    overlap_length = int(sr * segment_overlap_seconds)
    hop_length = segment_length - overlap_length

    if hop_length <= 0:
        raise ValueError(
            "segment_overlap_seconds must be smaller than segment_duration_seconds"
        )

    if len(y) == 0:
        return []

    if len(y) <= segment_length:
        y_fixed = librosa.util.fix_length(y, size=segment_length)
        rms = float(np.sqrt(np.mean(y_fixed ** 2)))
        return [(0, 0.0, segment_duration_seconds, y_fixed, rms)]

    candidates = []

    for start in range(0, len(y) - segment_length + 1, hop_length):
        end = start + segment_length
        segment = y[start:end]
        rms = float(np.sqrt(np.mean(segment ** 2)))

        candidates.append(
            {
                "start": start,
                "end": end,
                "segment": segment,
                "rms": rms,
            }
        )

    if not candidates:
        y_fixed = librosa.util.fix_length(y, size=segment_length)
        rms = float(np.sqrt(np.mean(y_fixed ** 2)))
        return [(0, 0.0, segment_duration_seconds, y_fixed, rms)]

    candidates = sorted(candidates, key=lambda item: item["rms"], reverse=True)
    selected = candidates[:max_segments_per_recording]
    selected = sorted(selected, key=lambda item: item["start"])

    output = []

    for segment_id, item in enumerate(selected):
        start_sec = item["start"] / sr
        end_sec = item["end"] / sr

        output.append(
            (
                segment_id,
                start_sec,
                end_sec,
                item["segment"],
                item["rms"],
            )
        )

    return output


def extract_features_from_segment(
    y,
    sr,
    filename,
    full_path,
    species,
    n_mfcc=20,
):
    """
    Extract acoustic features from one audio segment.
    """
    if len(y) == 0:
        raise ValueError(f"Empty segment from: {full_path}")

    row = {
        "filename": filename,
        "full_path": str(full_path),
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


def main():
    config = load_config()
    ensure_directories(config)

    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    wav_data_dir = Path(config["paths"]["interim_wav_dir"])
    features_dir = Path(config["paths"]["features_dir"])

    target_sr = config["audio"]["target_sample_rate"]
    n_mfcc = config["features"]["n_mfcc"]

    segment_config = config.get("segments", {})

    segment_duration_seconds = segment_config.get("segment_duration_seconds", 5)
    segment_overlap_seconds = segment_config.get("segment_overlap_seconds", 2.5)
    max_segments_per_recording = segment_config.get("max_segments_per_recording", 3)

    inventory_df = build_inventory(raw_data_dir)
    source_name = "raw"

    if inventory_df.empty:
        print(f"No audio files found in {raw_data_dir}.")
        print(f"Trying WAV directory instead: {wav_data_dir}")

        inventory_df = build_inventory(wav_data_dir)
        source_name = "interim_wav"

    if inventory_df.empty:
        print("No audio files found in data/raw or data/interim/wav.")
        print("For raw data, expected:")
        print("data/raw/Brachypteryx montana/A/audio.mp3")
        print("For WAV data, expected:")
        print("data/interim/wav/Brachypteryx montana/audio.wav")
        return

    print(f"Using audio source: {source_name}")
    print(f"Found {len(inventory_df)} audio files.")
    print("Starting segment-level feature extraction...")
    
    rows = []

    for _, item in tqdm(inventory_df.iterrows(), total=len(inventory_df)):
        audio_path = Path(item["full_path"])
        species = item["species"]

        try:
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

            y = trim_silence(y)
            y = normalize_audio(y)

            segments = create_segments(
                y=y,
                sr=sr,
                segment_duration_seconds=segment_duration_seconds,
                segment_overlap_seconds=segment_overlap_seconds,
                max_segments_per_recording=max_segments_per_recording,
            )

            relative_path = item["relative_path"]

            recording_id = str(Path(relative_path).with_suffix(""))

            for segment_id, start_sec, end_sec, segment_y, segment_rms in segments:
                features = extract_features_from_segment(
                    y=segment_y,
                    sr=sr,
                    filename=audio_path.name,
                    full_path=audio_path,
                    species=species,
                    n_mfcc=n_mfcc,
                )

                features["recording_id"] = recording_id
                features["segment_id"] = segment_id
                features["segment_start_sec"] = start_sec
                features["segment_end_sec"] = end_sec
                features["segment_rms"] = segment_rms
                features["relative_path"] = relative_path
                features["quality_rating"] = item.get("quality_rating", None)

                rows.append(features)

        except Exception as error:
            print(f"Failed to process {audio_path}: {error}")

    segment_features_df = pd.DataFrame(rows)

    output_path = features_dir / "segment_features.csv"
    segment_features_df.to_csv(output_path, index=False)

    print(f"\nSaved segment features to: {output_path}")
    print(f"Shape: {segment_features_df.shape}")

    if segment_features_df.empty:
        print("No segment features were created.")
        print("Check whether the audio files can be loaded.")
        return

    print("\nSegments per species:")
    print(segment_features_df["species"].value_counts())

    print("\nQuality rating counts:")
    print(segment_features_df["quality_rating"].value_counts(dropna=False))

    print("\nSegments per recording summary:")
    print(segment_features_df.groupby("recording_id").size().describe())

    time_cols = [col for col in segment_features_df.columns if col.startswith("time_")]
    print("\nTime-domain columns:")
    print(time_cols)
    print(f"Number of time-domain columns: {len(time_cols)}")


if __name__ == "__main__":
    main()