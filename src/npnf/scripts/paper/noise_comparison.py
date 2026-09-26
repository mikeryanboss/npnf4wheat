"""Compare learned homoscedastic noise against ground truth."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
from loguru import logger
from safetensors.torch import load_file

from npnf.scripts.paper.style import COLORS, get_figure, save_figure, setup_style
from npnf.scripts.utils.prediction import resolve_checkpoint

# Ground truth noise from synthetic dataset config
TRUE_STD = 0.05
TRUE_VARIANCE = TRUE_STD**2


def extract_variance_from_checkpoint(checkpoint_path: Path) -> float:
    """Extract learned variance from model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory or parent checkpoints folder.

    Returns:
        Learned variance value (after softplus transformation).
    """
    resolved_path = resolve_checkpoint(checkpoint_path)
    model_path = resolved_path / "model.safetensors"

    if not model_path.exists():
        model_path = resolved_path / "pytorch_model.bin"
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    else:
        state_dict = load_file(str(model_path))

    var_raw = state_dict["var"]
    variance = torch.nn.functional.softplus(var_raw) + 1e-8
    return variance.item()


def compute_noise_metrics(learned_variance: float) -> dict:
    """Compute comparison metrics between learned and true noise.

    Args:
        learned_variance: Model's learned variance value.

    Returns:
        Dict with comparison metrics.
    """
    learned_std = learned_variance**0.5

    return {
        "learned_std": learned_std,
        "learned_variance": learned_variance,
        "true_std": TRUE_STD,
        "true_variance": TRUE_VARIANCE,
        "std_ratio": learned_std / TRUE_STD,
        "error_percent": abs(learned_std - TRUE_STD) / TRUE_STD * 100,
    }


def build_comparison_table(
    checkpoint_folders: list[str], model_names: list[str]
) -> pl.DataFrame:
    """Build comparison table for all models.

    Args:
        checkpoint_folders: List of paths to checkpoint directories.
        model_names: Display names for each model.

    Returns:
        Polars DataFrame with comparison data.
    """
    records = []

    for folder, name in zip(checkpoint_folders, model_names, strict=True):
        variance = extract_variance_from_checkpoint(Path(folder))
        metrics = compute_noise_metrics(variance)

        records.append(
            {
                "Model": name,
                "Learned_Std": metrics["learned_std"],
                "True_Std": metrics["true_std"],
                "Std_Ratio": metrics["std_ratio"],
                "Error_Percent": metrics["error_percent"],
            }
        )

        logger.info(
            f"[{name}] variance={variance:.6f}, std={metrics['learned_std']:.4f}, "
            f"ratio={metrics['std_ratio']:.3f}, error={metrics['error_percent']:.2f}%"
        )

    return pl.DataFrame(records)


def plot_noise_comparison(df: pl.DataFrame, output_path: Path) -> None:
    """Create bar chart comparing learned vs true noise.

    Args:
        df: DataFrame with noise comparison data.
        output_path: Path to save figure (without extension).
    """
    fig, ax = get_figure("wide")

    models = df["Model"].to_list()
    learned_stds = df["Learned_Std"].to_numpy()

    x = range(len(models))

    bars = ax.bar(x, learned_stds, label="Learned Std", color=COLORS["blue"])

    ax.axhline(
        y=TRUE_STD,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=2,
        label=f"True Std ({TRUE_STD})",
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Learned vs Ground Truth Noise")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    for bar, std in zip(bars, learned_stds, strict=True):
        ax.annotate(
            f"{std:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(fig, output_path)
    plt.close(fig)
    logger.info(f"Saved comparison plot to {output_path}")


def plot_table_figure(df: pl.DataFrame, output_path: Path) -> None:
    """Create a matplotlib table figure.

    Args:
        df: DataFrame with noise comparison data.
        output_path: Path to save figure.
    """
    display_df = df.select(
        [
            "Model",
            pl.col("Learned_Std").round(4),
            pl.col("True_Std").round(4),
            pl.col("Std_Ratio").round(3),
            pl.col("Error_Percent").round(2),
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * len(display_df)))
    ax.axis("off")

    table_data = display_df.to_numpy()
    col_labels = ["Model", "Learned Std", "True Std", "Ratio", "Error (%)"]

    table = ax.table(
        cellText=table_data,  # ty: ignore[invalid-argument-type]
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(COLORS["gray"])

    save_figure(fig, output_path)
    plt.close(fig)
    logger.info(f"Saved table figure to {output_path}")


def compare_noise(
    checkpoint_folders: list[str],
    model_names: list[str],
    output_folder: str = "paper/noise_comparison",
) -> None:
    """Generate noise comparison table and figures.

    Args:
        checkpoint_folders: List of checkpoint directory paths.
        model_names: Display names for each model.
        output_folder: Output directory for results.
    """
    setup_style()

    if len(checkpoint_folders) != len(model_names):
        msg = (
            f"Number of checkpoint folders ({len(checkpoint_folders)}) must match "
            f"number of model names ({len(model_names)})"
        )
        raise ValueError(msg)

    logger.info(f"""
    Comparing noise parameters:
        - checkpoint_folders: {checkpoint_folders}
        - model_names: {model_names}
        - output_folder: {output_folder}
        - true_std: {TRUE_STD}
        - true_variance: {TRUE_VARIANCE}
    """)

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    df = build_comparison_table(checkpoint_folders, model_names)

    csv_path = output_path / "noise_comparison.csv"
    df.write_csv(csv_path)
    logger.info(f"Saved comparison data to {csv_path}")

    plot_noise_comparison(df, output_path / "noise_comparison_bar")
    plot_table_figure(df, output_path / "noise_comparison_table")

    logger.info("Done generating noise comparison outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare learned homoscedastic noise against ground truth"
    )
    parser.add_argument(
        "--checkpoint-folders",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model checkpoint folders",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        required=True,
        help="Display names for each model (must match number of checkpoint-folders)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="paper/noise_comparison",
        help="Output folder for results (default: paper/noise_comparison)",
    )
    args = parser.parse_args()

    compare_noise(
        checkpoint_folders=args.checkpoint_folders,
        model_names=args.model_names,
        output_folder=args.output_folder,
    )
