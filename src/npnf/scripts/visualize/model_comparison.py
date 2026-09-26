"""Generate publication-ready comparison plots for ANP vs ANP_NF models."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

# Publication-ready style
STYLE_CONFIG = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLORS = {
    "model1": "#1f77b4",  # Blue
    "model2": "#ff7f0e",  # Orange
}


def find_per_context_data(comparison_path: Path) -> tuple[Path | None, Path | None]:
    """Find per-context and summary CSV files, checking subfolders if needed."""
    per_context_path = comparison_path / "comparison_per_context.csv"
    summary_path = comparison_path / "comparison_summary.csv"

    if per_context_path.exists():
        return per_context_path, summary_path

    # Search conditioning subfolders
    conditioning_names = ["env_geno", "env_nogeno", "noenv_geno", "noenv_nogeno"]
    for cond in conditioning_names:
        cond_per_context = comparison_path / cond / "comparison_per_context.csv"
        if cond_per_context.exists():
            logger.info(f"Using per-context data from {cond}/ subfolder")
            return cond_per_context, comparison_path / cond / "comparison_summary.csv"

    return None, None


def load_config(comparison_folder: Path) -> dict:
    """Load comparison config to get model names."""
    config_path = comparison_folder / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    return {"model1_name": "ANP", "model2_name": "ANP_NF"}


def plot_performance_curves(
    df: pl.DataFrame,
    output_path: Path,
    model1_name: str = "ANP",
    model2_name: str = "ANP_NF",
) -> None:
    """Plot MAE vs context for model comparison."""
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(figsize=(6, 4))

    num_context = df["num_context"].to_numpy()

    ax.plot(
        num_context,
        df[f"{model1_name}_mae"].to_numpy(),
        label=model1_name,
        color=COLORS["model1"],
        linewidth=2,
    )
    ax.plot(
        num_context,
        df[f"{model2_name}_mae"].to_numpy(),
        label=model2_name,
        color=COLORS["model2"],
        linewidth=2,
        linestyle="--",
    )
    ax.set_xlabel("Number of context points")
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved performance curves to {output_path}")


def plot_summary_bars(
    summary_df: pl.DataFrame,
    output_path: Path,
    model1_name: str = "ANP",
    model2_name: str = "ANP_NF",
) -> None:
    """Figure 4: Summary bar chart of key metrics."""
    plt.rcParams.update(STYLE_CONFIG)

    metrics_to_plot = ["mae_clean", "final_mae"]
    metric_labels = ["Avg MAE", "Final MAE"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    model1_vals = []
    model2_vals = []

    for metric in metrics_to_plot:
        row = summary_df.filter(pl.col("metric") == metric)
        if len(row) > 0:
            model1_vals.append(row["model1_mean"][0])
            model2_vals.append(row["model2_mean"][0])
        else:
            model1_vals.append(0)
            model2_vals.append(0)

    bars1 = ax.bar(
        x - width / 2,
        model1_vals,
        width,
        label=model1_name,
        color=COLORS["model1"],
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        model2_vals,
        width,
        label=model2_name,
        color=COLORS["model2"],
        edgecolor="white",
    )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Model Comparison Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved summary bars to {output_path}")


def plot_epistemic_std_comparison(
    df: pl.DataFrame,
    output_path: Path,
    model1_name: str = "ANP",
    model2_name: str = "ANP_NF",
) -> None:
    """Additional: Epistemic uncertainty comparison over context."""
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(figsize=(8, 5))

    num_context = df["num_context"].to_numpy()

    ax.plot(
        num_context,
        df[f"{model1_name}_std"].to_numpy(),
        label=model1_name,
        color=COLORS["model1"],
        linewidth=2,
    )
    ax.plot(
        num_context,
        df[f"{model2_name}_std"].to_numpy(),
        label=model2_name,
        color=COLORS["model2"],
        linewidth=2,
        linestyle="--",
    )

    ax.set_xlabel("Number of context points")
    ax.set_ylabel("Epistemic Uncertainty (std)")
    ax.set_title("Epistemic Uncertainty vs Context Size")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved epistemic std comparison to {output_path}")


def plot_model_comparison(comparison_folder: str, output_format: str = "png") -> None:
    """Generate all comparison plots.

    Args:
        comparison_folder: Path to folder containing comparison CSVs
        output_format: Output format (png, pdf, or both)
    """
    comparison_path = Path(comparison_folder)
    plots_path = comparison_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    config = load_config(comparison_path)
    model1_name = config.get("model1_name", "ANP")
    model2_name = config.get("model2_name", "ANP_NF")

    logger.info(f"Generating plots for {model1_name} vs {model2_name}")

    # Load per-context data (check root first, then conditioning subfolders)
    per_context_path, summary_path = find_per_context_data(comparison_path)

    if output_format == "png":
        formats = ["png"]
    elif output_format == "pdf":
        formats = ["pdf"]
    else:
        formats = ["png", "pdf"]

    if per_context_path is not None:
        df = pl.read_csv(per_context_path)

        for fmt in formats:
            plot_performance_curves(
                df,
                plots_path / f"fig1_performance_curves.{fmt}",
                model1_name,
                model2_name,
            )
            plot_epistemic_std_comparison(
                df, plots_path / f"fig_epistemic_std.{fmt}", model1_name, model2_name
            )
    else:
        logger.warning("No per-context data found")

    if summary_path is not None and summary_path.exists():
        summary_df = pl.read_csv(summary_path)
        for fmt in formats:
            plot_summary_bars(
                summary_df,
                plots_path / f"fig4_summary_bars.{fmt}",
                model1_name,
                model2_name,
            )
    else:
        logger.warning(f"No summary data found at {summary_path}")

    logger.info(f"All plots saved to {plots_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication-ready model comparison plots"
    )
    parser.add_argument(
        "--comparison-folder",
        type=str,
        required=True,
        help="Path to folder containing comparison CSVs from model_comparison.py",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "both"],
        help="Output format for plots",
    )
    args = parser.parse_args()

    plot_model_comparison(args.comparison_folder, args.format)
