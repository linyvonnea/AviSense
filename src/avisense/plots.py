from pathlib import Path
import matplotlib.pyplot as plt


def plot_species_distribution(df, output_path):
    """
    Plot number of recordings per species.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df["species"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    counts.plot(kind="bar", ax=ax)

    ax.set_title("Species Distribution")
    ax.set_xlabel("Species")
    ax.set_ylabel("Number of Recordings")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_model_comparison(results_df, output_path):
    """
    Plot model comparison based on macro F1-score.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = results_df.sort_values("f1_macro_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(df["model"], df["f1_macro_mean"])

    ax.set_title("Model Comparison by Macro F1-score")
    ax.set_xlabel("Macro F1-score")
    ax.set_ylabel("Model")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)