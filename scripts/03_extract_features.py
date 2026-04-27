from pathlib import Path

from avisense.config import load_config, ensure_directories
from avisense.data_inventory import build_inventory
from avisense.features import extract_features_from_inventory


def main():
    config = load_config()
    ensure_directories(config)

    audio_dir = Path(config["paths"]["interim_wav_dir"])
    features_dir = Path(config["paths"]["features_dir"])

    target_sr = config["audio"]["target_sample_rate"]
    n_mfcc = config["features"]["n_mfcc"]

    inventory_df = build_inventory(audio_dir)

    features_df = extract_features_from_inventory(
        inventory_df=inventory_df,
        target_sr=target_sr,
        n_mfcc=n_mfcc,
    )

    output_path = features_dir / "recording_features.csv"
    features_df.to_csv(output_path, index=False)

    print(f"Saved features to: {output_path}")
    print(f"Feature table shape: {features_df.shape}")


if __name__ == "__main__":
    main()