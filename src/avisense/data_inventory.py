from pathlib import Path
import pandas as pd

from avisense.utils import find_audio_files, get_species_dirs


def build_inventory(raw_data_dir):
    """
    Build an inventory of all audio files.

    The species label is inferred from the folder name.
    """
    raw_data_dir = Path(raw_data_dir)
    rows = []

    for species_dir in get_species_dirs(raw_data_dir):
        species_name = species_dir.name
        audio_files = find_audio_files(species_dir)

        for audio_path in audio_files:
            relative_path = audio_path.relative_to(raw_data_dir)
            parts = relative_path.parts

            quality_rating = None

            # Example:
            # Brachypteryx montana/A/audio.mp3
            if len(parts) >= 3 and parts[1] in ["A", "B", "C", "D", "E"]:
                quality_rating = parts[1]

            rows.append(
                {
                    "species": species_name,
                    "filename": audio_path.name,
                    "relative_path": str(relative_path),
                    "full_path": str(audio_path),
                    "quality_rating": quality_rating,
                    "file_extension": audio_path.suffix.lower(),
                }
            )

    return pd.DataFrame(rows)


def load_species_metadata(species_dir):
    """
    Load metadata CSV from a species folder.

    Returns None if no CSV exists.
    """
    species_dir = Path(species_dir)
    csv_files = list(species_dir.glob("*.csv"))

    if not csv_files:
        return None

    metadata_path = csv_files[0]

    df = pd.read_csv(metadata_path)
    df["metadata_file"] = metadata_path.name
    df["species_folder"] = species_dir.name

    return df


def combine_metadata(raw_data_dir):
    """
    Combine all species metadata CSV files into one dataframe.
    """
    raw_data_dir = Path(raw_data_dir)
    frames = []

    for species_dir in get_species_dirs(raw_data_dir):
        df = load_species_metadata(species_dir)

        if df is not None:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)