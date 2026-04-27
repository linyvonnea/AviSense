import argparse
from pathlib import Path

from avisense.config import load_config
from avisense.predict import predict_audio_file


def main():
    parser = argparse.ArgumentParser(
        description="Predict bird species from one audio file."
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the audio file.",
    )

    args = parser.parse_args()

    config = load_config()

    models_dir = Path(config["paths"]["models_dir"])

    result = predict_audio_file(
        audio_path=args.audio,
        model_path=models_dir / "best_model.joblib",
        label_encoder_path=models_dir / "label_encoder.joblib",
        target_sr=config["audio"]["target_sample_rate"],
        n_mfcc=config["features"]["n_mfcc"],
    )

    print(result)


if __name__ == "__main__":
    main()