"""Visualize temperature sensitivity analysis results."""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from matplotlib.patches import Patch

TRAIN_SITES = [
    "Assens",
    "Changins",
    "Delley",
    "Grangeneuve",
    "Lindau",
    "SulzKunten",
    "Vouvry",
]
VAL_SITES = ["Zollikofen"]
TEST_SITES = ["Moudon", "Ellighausen"]

SITE_COLORS = {
    **dict.fromkeys(TRAIN_SITES, "#1f77b4"),
    **dict.fromkeys(VAL_SITES, "#2ca02c"),
    **dict.fromkeys(TEST_SITES, "#ff7f0e"),
}


def plot_site_heights(df: pl.DataFrame, output_path: Path) -> None:
    """Bar chart of mean height by site with error bars."""
    fig, ax = plt.subplots(figsize=(12, 6))

    df_sorted = df.sort("height_mean", descending=True)
    sites = df_sorted["site"].to_list()
    heights = df_sorted["height_mean"].to_numpy()
    stds = df_sorted["height_std"].to_numpy()

    colors = [SITE_COLORS.get(s, "#999999") for s in sites]

    ax.bar(sites, heights, yerr=stds, color=colors, capsize=3, alpha=0.8)

    ax.set_xlabel("Site", fontsize=12)
    ax.set_ylabel("Mean Final Height (m)", fontsize=12)
    ax.set_title(
        "Mean Plant Height by Site (Same Seeds, Different Temperatures)", fontsize=14
    )
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels(sites, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor="#1f77b4", label="Train"),
        Patch(facecolor="#2ca02c", label="Val"),
        Patch(facecolor="#ff7f0e", label="Test"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved site heights plot to {output_path}")


def plot_year_trends(df: pl.DataFrame, output_path: Path) -> None:
    """Line plot of mean height by year for each site."""
    df_with_year = df.with_columns(
        [
            pl.col("environment").str.extract(r"^(.+)_(\d{4})$", 1).alias("site"),
            pl.col("environment")
            .str.extract(r"^(.+)_(\d{4})$", 2)
            .cast(pl.Int32)
            .alias("year"),
        ]
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    sites = sorted(df_with_year["site"].unique().to_list())
    for site in sites:
        site_data = df_with_year.filter(pl.col("site") == site).sort("year")
        years = site_data["year"].to_numpy()
        heights = site_data["height_mean"].to_numpy()
        color = SITE_COLORS.get(site, "#999999")
        ax.plot(
            years,
            heights,
            marker="o",
            label=site,
            color=color,
            alpha=0.8,
            linewidth=1.5,
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mean Final Height (m)", fontsize=12)
    ax.set_title("Plant Height Variation Across Years by Site", fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved year trends plot to {output_path}")


def plot_sensitivity_distribution(df: pl.DataFrame, output_path: Path) -> None:
    """Histogram of height range across environments per seed."""
    fig, ax = plt.subplots(figsize=(10, 6))

    height_ranges = df["height_range"].to_numpy()

    ax.hist(height_ranges, bins=30, color="#1f77b4", alpha=0.7, edgecolor="black")
    ax.axvline(
        height_ranges.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {height_ranges.mean():.3f}m",
    )

    ax.set_xlabel("Height Range Across Environments (m)", fontsize=12)
    ax.set_ylabel("Number of Seeds", fontsize=12)
    ax.set_title("Temperature Sensitivity: Height Variation per Seed", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved sensitivity distribution plot to {output_path}")


def plot_height_distribution(df: pl.DataFrame, output_path: Path) -> None:
    """Box plot of height distribution by site group."""
    df_with_group = df.with_columns(
        [
            pl.when(pl.col("site").is_in(TRAIN_SITES))
            .then(pl.lit("Train"))
            .when(pl.col("site").is_in(VAL_SITES))
            .then(pl.lit("Val"))
            .otherwise(pl.lit("Test"))
            .alias("group")
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = ["Train", "Val", "Test"]
    data_by_group = []
    for group in groups:
        heights = df_with_group.filter(pl.col("group") == group)[
            "height_mean"
        ].to_numpy()
        data_by_group.append(heights)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    bp = ax.boxplot(data_by_group, tick_labels=groups, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Site Group", fontsize=12)
    ax.set_ylabel("Mean Final Height (m)", fontsize=12)
    ax.set_title("Height Distribution by Site Group", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved height distribution plot to {output_path}")


def plot_temperature_sensitivity(results_folder: str) -> None:
    """Generate all temperature sensitivity visualizations."""
    results_path = Path(results_folder)
    plots_path = results_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    # Site stats
    site_stats_path = results_path / "site_stats.csv"
    if site_stats_path.exists():
        df_sites = pl.read_csv(site_stats_path)
        plot_site_heights(df_sites, plots_path / "site_heights.png")
        plot_height_distribution(
            df_sites, plots_path / "height_distribution_boxplot.png"
        )

    # Environment stats (for year trends)
    env_stats_path = results_path / "environment_stats.csv"
    if env_stats_path.exists():
        df_env = pl.read_csv(env_stats_path)
        plot_year_trends(df_env, plots_path / "year_height_trends.png")

    # Sensitivity summary
    sensitivity_path = results_path / "sensitivity_summary.csv"
    if sensitivity_path.exists():
        df_sens = pl.read_csv(sensitivity_path)
        plot_sensitivity_distribution(
            df_sens, plots_path / "sensitivity_distribution.png"
        )

    logger.info(f"All plots saved to {plots_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize temperature sensitivity analysis results"
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        required=True,
        help="Path to folder containing temperature sensitivity CSVs",
    )
    args = parser.parse_args()
    plot_temperature_sensitivity(args.results_folder)
