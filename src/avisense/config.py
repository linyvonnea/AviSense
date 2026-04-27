from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    """
    Load project configuration from config.yaml.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def ensure_directories(config):
    """
    Create required directories if they do not exist.
    """
    path_keys = [
        "interim_wav_dir",
        "processed_dir",
        "metadata_dir",
        "features_dir",
        "splits_dir",
        "models_dir",
        "reports_dir",
        "figures_dir",
        "predictions_dir",
    ]

    for key in path_keys:
        path = Path(config["paths"][key])
        path.mkdir(parents=True, exist_ok=True)