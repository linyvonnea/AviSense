from pathlib import Path

from avisense.config import load_config, ensure_directories
from avisense.data_inventory import build_inventory, combine_metadata
from avisense.plots import plot_species_distribution


def main():
    config = load_config()
    ensure_directories(config)

    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    metadata_dir = Path(config["paths"]["metadata_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])

    inventory_df = build_inventory(raw_data_dir)
    metadata_df = combine_metadata(raw_data_dir)

    inventory_path = metadata_dir / "audio_inventory.csv"
    metadata_path = metadata_dir / "all_metadata.csv"
    summary_path = metadata_dir / "dataset_summary.csv"

    inventory_df.to_csv(inventory_path, index=False)

    if not metadata_df.empty:
        metadata_df.to_csv(metadata_path, index=False)

    summary = (
        inventory_df.groupby("species")
        .size()
        .reset_index(name="recording_count")
        .sort_values("recording_count", ascending=False)
    )

    summary.to_csv(summary_path, index=False)

    plot_species_distribution(
        df=inventory_df,
        output_path=figures_dir / "species_distribution.png",
    )

    print(f"Saved inventory to: {inventory_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved summary to: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()