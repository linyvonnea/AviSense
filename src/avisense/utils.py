from pathlib import Path
import re


AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]


def safe_filename(name):
    """
    Convert string to a safer filename.
    """
    name = name.strip()
    name = re.sub(r"[^\w\s.-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def find_audio_files(directory):
    """
    Recursively find audio files.

    This supports files directly inside species folders and files inside A/B/C/D/E folders.
    """
    directory = Path(directory)
    files = []

    for ext in AUDIO_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))

    return sorted(files)


def get_species_dirs(raw_data_dir):
    """
    Return species folders inside data/raw or another root directory.
    """
    raw_data_dir = Path(raw_data_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_data_dir}")

    return sorted([p for p in raw_data_dir.iterdir() if p.is_dir()])