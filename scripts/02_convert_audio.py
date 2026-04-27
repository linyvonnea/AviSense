from pathlib import Path
from tqdm import tqdm

from avisense.config import load_config, ensure_directories
from avisense.data_inventory import build_inventory
from avisense.preprocessing import convert_to_wav, make_wav_output_path


def main():
    config = load_config()
    ensure_directories(config)

    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    wav_dir = Path(config["paths"]["interim_wav_dir"])

    target_sr = config["audio"]["target_sample_rate"]
    duration_seconds = config["audio"]["duration_seconds"]

    inventory_df = build_inventory(raw_data_dir)

    for _, row in tqdm(inventory_df.iterrows(), total=len(inventory_df)):
        input_path = Path(row["full_path"])

        output_path = make_wav_output_path(
            raw_audio_path=input_path,
            raw_data_dir=raw_data_dir,
            output_dir=wav_dir,
        )

        convert_to_wav(
            input_path=input_path,
            output_path=output_path,
            target_sr=target_sr,
            duration_seconds=duration_seconds,
            trim=True,
            normalize=True,
        )

    print(f"Converted audio files saved to: {wav_dir}")


if __name__ == "__main__":
    main()