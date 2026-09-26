"""Compare varying-context Sig-MMD metrics across models and sampling strategies."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib.typing import ColorType

from npnf.scripts.paper.style import COLOR_LIST, COLORS, setup_style

SAMPLING_METHODS = {"random": "Random", "uncertainty": "Uncertainty"}
OUTPUT_DIR = Path("paper/sampling_comparison")
PLOT_COLUMN = "Sig-MMD² × 10³"
MODEL_COLOR_OVERRIDES = {"LNP": COLORS["blue"], "ANP": COLORS["orange"]}
SAMPLING_STYLES = {
    "Random": {"linestyle": "-", "marker": "o"},
    "Uncertainty": {"linestyle": "--", "marker": "s"},
}
FIGURE_SIZE = (7.2, 4.2)
X_LABEL_PAD = 3.0
LEGEND_BOTTOM = 0.14
LEGEND_ANCHOR_Y = 0.165


def _ordered_unique(values: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _load_metrics(folder: Path) -> pl.DataFrame:
    """Load varying Sig-MMD context summary metrics."""
    metrics_path = folder / "sig_mmd_by_context.csv"
    if not metrics_path.exists():
        msg = f"No sig_mmd_by_context.csv found under {folder}"
        raise FileNotFoundError(msg)
    logger.info(f"Using Sig-MMD context summary from {metrics_path}")
    return pl.read_csv(metrics_path).select(["num_context", "mean"])


def _default_model_name(results_folder: Path) -> str:
    if (
        results_folder.parent.name.startswith("test_")
        and results_folder.parent.parent.name.startswith("checkpoint-")
        and len(results_folder.parents) > 3
    ):
        return results_folder.parents[3].name
    return results_folder.name


def _resolve_model_names(
    results_folders: list[Path], model_names: list[str] | None
) -> list[str]:
    if model_names is None:
        return [_default_model_name(path) for path in results_folders]
    if len(model_names) != len(results_folders):
        msg = (
            f"--model-names count ({len(model_names)}) must match "
            f"--results-folder count ({len(results_folders)})"
        )
        raise ValueError(msg)
    return model_names


def _model_palette(model_order: list[str]) -> dict[str, ColorType]:
    palette: dict[str, ColorType] = {}
    for index, model in enumerate(model_order):
        color = None
        for prefix, candidate in MODEL_COLOR_OVERRIDES.items():
            if model == prefix or model.startswith(f"{prefix}-"):
                color = candidate
                break
        palette[model] = (
            color if color is not None else COLOR_LIST[index % len(COLOR_LIST)]
        )
    return palette


def _sampling_order(values: Iterable[object]) -> list[str]:
    unique_values = _ordered_unique(values)
    present = set(unique_values)
    ordered = [label for label in SAMPLING_METHODS.values() if label in present]
    ordered.extend(value for value in unique_values if value not in ordered)
    return ordered


def load_comparison_data(
    results_folders: list[Path], model_names: list[str] | None = None
) -> pl.DataFrame:
    """Load Sig-MMD summaries for every requested model and sampling method."""
    labels = _resolve_model_names(results_folders, model_names)
    frames = []
    for results_folder, model_name in zip(results_folders, labels, strict=True):
        for method_dir_name, sampling_name in SAMPLING_METHODS.items():
            frame = (
                _load_metrics(results_folder / method_dir_name)
                .rename({"num_context": "Context Points", "mean": "Sig-MMD"})
                .with_columns(
                    (pl.col("Sig-MMD") * 1000.0).alias(PLOT_COLUMN),
                    pl.lit(sampling_name).alias("Sampling"),
                    pl.lit(model_name).alias("Model"),
                )
            )
            frames.append(frame)

    return pl.concat(frames).select(
        ["Context Points", "Sig-MMD", PLOT_COLUMN, "Sampling", "Model"]
    )


def _draw_lines(ax: plt.Axes, plot_df: pd.DataFrame) -> list[mlines.Line2D]:
    model_order = _ordered_unique(plot_df["Model"])
    sampling_order = _sampling_order(plot_df["Sampling"])
    palette = _model_palette(model_order)

    for model in model_order:
        for sampling in sampling_order:
            subset = plot_df[
                (plot_df["Model"].astype(str) == model)
                & (plot_df["Sampling"].astype(str) == sampling)
            ].sort_values("Context Points")
            if subset.empty:
                continue
            style = SAMPLING_STYLES.get(sampling, {"linestyle": "-", "marker": "o"})
            ax.plot(
                subset["Context Points"],
                subset[PLOT_COLUMN],
                color=palette[model],
                linewidth=2.0,
                markersize=4.8,
                markeredgewidth=0.8,
                markeredgecolor=palette[model],
                markerfacecolor=palette[model],
                zorder=3,
                linestyle=style["linestyle"],
                marker=style["marker"],
            )

    model_handles = [
        mlines.Line2D(
            [], [], color=palette[model], linewidth=2.2, linestyle="-", label=model
        )
        for model in model_order
    ]
    sampling_handles = []
    for sampling in sampling_order:
        style = SAMPLING_STYLES.get(sampling, {"linestyle": "-", "marker": "o"})
        sampling_handles.append(
            mlines.Line2D(
                [],
                [],
                color="#333333",
                linewidth=1.8,
                markersize=5,
                markeredgewidth=0.8,
                markeredgecolor="#333333",
                markerfacecolor="#333333",
                label=sampling,
                linestyle=style["linestyle"],
                marker=style["marker"],
            )
        )
    return [*model_handles, *sampling_handles]


def _single_folder_title(results_folder: Path) -> str:
    parent_name = results_folder.parent.name
    current_name = results_folder.name
    if parent_name and parent_name != "." and parent_name.lower() != "results":
        return f"Sig-MMD vs Context Points ({parent_name} - {current_name})"
    return f"Sig-MMD vs Context Points ({current_name})"


def plot_sig_mmd_comparison(
    results_folders: list[Path], df_combined: pl.DataFrame
) -> None:
    """Plot Sig-MMD comparison across models and sampling strategies."""
    logger.info("Plotting Sig-MMD comparison for {}", results_folders)

    plot_df = df_combined.to_pandas()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    legend_handles = _draw_lines(ax, plot_df)

    if len(results_folders) == 1 and results_folders[0].name:
        ax.set_title(_single_folder_title(results_folders[0]))
    ax.set_xlabel("Number of context points", labelpad=X_LABEL_PAD)
    ax.set_ylabel(PLOT_COLUMN)
    ax.grid(axis="x", visible=False)

    if not plot_df.empty:
        max_context_points = plot_df["Context Points"].max()
        if not pd.isna(max_context_points) and max_context_points > 30:
            plt.xticks(rotation=45)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(rect=(0.0, LEGEND_BOTTOM, 1.0, 1.0))
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
        ncol=len(legend_handles),
        fontsize=8,
        frameon=False,
        handlelength=1.8,
    )

    output_path = OUTPUT_DIR / "sig_mmd_comparison_plot"
    for fmt in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{fmt}"))
    plt.close(fig)
    logger.info(f"Saved Sig-MMD comparison plot to {output_path}")

    csv_path = OUTPUT_DIR / "sig_mmd_comparison.csv"
    df_combined.write_csv(csv_path)
    logger.info(f"Saved combined metrics to {csv_path}")


def comparison(
    results_folders: list[str], model_names: list[str] | None = None
) -> None:
    """Compare varying-context Sig-MMD metrics across models and samplers."""
    setup_style()

    results_folder_paths = [Path(path) for path in results_folders]

    logger.info(
        """
    Sampling comparison with:
        - results_folders: {}
        - model_names: {}
    """,
        results_folder_paths,
        model_names,
    )

    df_combined = load_comparison_data(results_folder_paths, model_names)
    plot_sig_mmd_comparison(results_folder_paths, df_combined)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder",
        type=str,
        nargs="+",
        required=True,
        help="One or more varying parent result folders",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional model labels matching --results-folder order",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        comparison(args.results_folder, args.model_names)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
