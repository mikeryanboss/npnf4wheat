"""Combined lodging absolute-drop visualization for prior and context predictions.

Creates a dual-panel or semantic faceted figure comparing mean absolute height
loss among lodged plants versus maximum height.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

from npnf.data.batch_loader import BatchLoader
from npnf.scripts.paper.data_types import PredictionStats
from npnf.scripts.paper.histogram_utils import calculate_height_bins
from npnf.scripts.paper.lodging_plot_style import (
    ROW_ORDER,
    SEMANTIC_GRID_BOTTOM,
    SEMANTIC_GRID_LEGEND_Y,
    SEMANTIC_GRID_SUPYLABEL_X,
    SEMANTIC_GRID_XLABEL_Y,
    VARIANT_COLORS,
    VARIANT_ORDER,
    bottom_legend_layout,
    semantic_model_groups,
    variant_legend_handles,
)
from npnf.scripts.paper.lodging_probability import (
    ModelConfig,
    _load_predictions,
    discover_models,
)
from npnf.scripts.paper.style import get_figure, setup_style
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    SyntheticBatchReconstructionCache,
    reconstruct_synthetic_batch,
)
from npnf.utils import add_panel_labels


@dataclass
class DropGroundTruth:
    """Ground truth lodging labels, max heights, and absolute drops."""

    labels: np.ndarray
    max_heights: np.ndarray
    drop_abs: np.ndarray


@dataclass
class BinnedDropMeans:
    """Binned mean absolute drops and counts for one model or ground truth."""

    name: str
    means: np.ndarray
    counts: np.ndarray
    lodged_counts: np.ndarray


@dataclass
class DropPanelData:
    """Data for a single absolute-drop panel."""

    title: str
    model_means: list[BinnedDropMeans]
    ground_truth_means: BinnedDropMeans


def _select_draw_zero_heights(y_original) -> np.ndarray:
    """Select draw 0 from synthetic multi-draw heights and squeeze to (B, T)."""
    heights = y_original.detach().cpu().float()
    if heights.ndim == 4:
        heights = heights[:, 0]
    while heights.ndim > 2 and heights.shape[-1] == 1:
        heights = heights.squeeze(-1)
    return heights.numpy()


def _load_ground_truth_drops(method_dir: Path) -> DropGroundTruth:
    """Load ground truth labels, max heights, and absolute drops."""
    loader = BatchLoader(method_dir, load_predictions=False)
    cache = SyntheticBatchReconstructionCache()

    all_labels = []
    all_max_heights = []
    all_drop_abs = []
    for batch_dict in loader:
        batch_dict = reconstruct_synthetic_batch(method_dir, batch_dict, cache=cache)
        labels = batch_dict["data"]["has_lodged"]
        if labels.ndim == 2:
            labels = labels[:, 0]
        all_labels.append(labels.detach().cpu().numpy().astype(bool))

        heights = _select_draw_zero_heights(batch_dict["data"]["height"]["Y_original"])
        max_heights = heights.max(axis=-1)
        final_heights = heights[:, -1]
        drop_abs = np.maximum(max_heights - final_heights, 0.0)
        all_max_heights.append(max_heights)
        all_drop_abs.append(drop_abs)

    return DropGroundTruth(
        labels=np.concatenate(all_labels),
        max_heights=np.concatenate(all_max_heights),
        drop_abs=np.concatenate(all_drop_abs),
    )


def calculate_binned_drop_means(
    values: np.ndarray,
    drop_abs: np.ndarray,
    lodged_mask: np.ndarray,
    bins: np.ndarray,
    min_lodged_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate mean absolute drop among lodged samples per bin."""
    values = np.asarray(values)
    drop_abs = np.asarray(drop_abs)
    lodged_mask = np.asarray(lodged_mask, dtype=bool)

    total_counts, _ = np.histogram(values, bins=bins)
    lodged_counts, _ = np.histogram(values[lodged_mask], bins=bins)
    drop_sums, _ = np.histogram(
        values[lodged_mask], bins=bins, weights=drop_abs[lodged_mask]
    )
    means = np.divide(
        drop_sums,
        lodged_counts,
        out=np.full_like(drop_sums, np.nan, dtype=float),
        where=lodged_counts > 0,
    )
    means = np.where(lodged_counts >= min_lodged_count, means, np.nan)
    return means, total_counts, lodged_counts


def _load_and_bin_model_drops(
    model: ModelConfig,
    conditioning: str,
    ground_truth: DropGroundTruth,
    bins: np.ndarray,
    relative_threshold: float,
    absolute_threshold: float,
    min_lodged_count: int,
) -> BinnedDropMeans | None:
    """Load predictions for one model/conditioning and compute binned drops."""
    path = (
        model.no_context_path if conditioning == "no_context" else model.max_height_path
    )
    if path is None:
        return None

    predictions: PredictionStats = _load_predictions(
        path, relative_threshold, absolute_threshold
    )
    logger.info(
        f"[{model.name}] {conditioning}: Loaded {len(predictions.max_height)} samples"
    )

    if conditioning == "no_context":
        values = predictions.max_height
    else:
        values = ground_truth.max_heights

    means, counts, lodged_counts = calculate_binned_drop_means(
        values=values,
        drop_abs=predictions.drop_abs,
        lodged_mask=predictions.is_lodged,
        bins=bins,
        min_lodged_count=min_lodged_count,
    )
    return BinnedDropMeans(
        name=model.name, means=means, counts=counts, lodged_counts=lodged_counts
    )


def _safe_name(name: str) -> str:
    """Sanitize a model name for CSV column labels."""
    return name.replace(" ", "_").replace("/", "_")


def _means_by_name(means: list[BinnedDropMeans]) -> dict[str, BinnedDropMeans]:
    """Return binned means keyed by model name."""
    return {model.name: model for model in means}


def _round_out_to_grid(
    limits: tuple[float, float], step: float = 0.1
) -> tuple[float, float]:
    """Round axis limits outward to multiples of the grid step."""
    lower, upper = limits
    return float(np.floor(lower / step) * step), float(np.ceil(upper / step) * step)


def _finite_x_limits(
    bin_centers: np.ndarray,
    series: list[np.ndarray],
    default: tuple[float, float] = (0.6, 2.0),
) -> tuple[float, float]:
    """Calculate x-axis limits from finite plotted values."""
    finite_x = []
    for values in series:
        mask = np.isfinite(values)
        if mask.any():
            finite_x.append(bin_centers[mask])

    if not finite_x:
        return default

    x_values = np.concatenate(finite_x)
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    margin = max((x_max - x_min) * 0.04, 0.01)
    return x_min - margin, x_max + margin


def _finite_y_limits(
    series: list[np.ndarray], default: tuple[float, float] = (0.0, 1.0)
) -> tuple[float, float]:
    """Calculate y-axis limits from finite plotted values."""
    finite_y = []
    for values in series:
        mask = np.isfinite(values)
        if mask.any():
            finite_y.append(values[mask])

    if not finite_y:
        return default

    y_values = np.concatenate(finite_y)
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    margin = max((y_max - y_min) * 0.06, 0.015)
    return y_min - margin, y_max + margin


def _save_drop_csv(
    output_path: Path,
    bins: np.ndarray,
    centers: np.ndarray,
    prior_panel: DropPanelData,
    context_panel: DropPanelData,
) -> None:
    """Save combined absolute-drop histogram data to CSV."""
    drop_data: dict = {
        "bin_left": bins[:-1],
        "bin_right": bins[1:],
        "bin_center": centers,
        "ground_truth_lodged_count": prior_panel.ground_truth_means.lodged_counts,
        "ground_truth_total_count": prior_panel.ground_truth_means.counts,
        "ground_truth_mean_abs_drop_m": prior_panel.ground_truth_means.means,
    }

    for model in prior_panel.model_means:
        safe_name = _safe_name(model.name)
        drop_data[f"prior_{safe_name}_lodged_count"] = model.lodged_counts
        drop_data[f"prior_{safe_name}_total_count"] = model.counts
        drop_data[f"prior_{safe_name}_mean_abs_drop_m"] = model.means

    for model in context_panel.model_means:
        safe_name = _safe_name(model.name)
        drop_data[f"context_{safe_name}_lodged_count"] = model.lodged_counts
        drop_data[f"context_{safe_name}_total_count"] = model.counts
        drop_data[f"context_{safe_name}_mean_abs_drop_m"] = model.means

    csv_path = output_path / "combined_lodging_absolute_drop.csv"
    pl.DataFrame(drop_data).write_csv(csv_path)


def _save_plot(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure as PNG and PDF."""
    for fmt in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{fmt}"))
    plt.close(fig)


def _plot_semantic_drop_means(
    bin_centers: np.ndarray,
    prior_panel: DropPanelData,
    context_panel: DropPanelData,
    output_path: Path,
    model_order: list[str],
) -> None:
    """Plot known paper models in semantic model-type facets."""
    semantic_groups = semantic_model_groups(model_order)
    if semantic_groups is None:
        msg = "semantic plot requested with unknown model names"
        raise ValueError(msg)

    prior_by_name = _means_by_name(prior_panel.model_means)
    context_by_name = _means_by_name(context_panel.model_means)
    ground_truth_means = prior_panel.ground_truth_means.means

    plotted_series = [ground_truth_means]
    for means_by_name in (prior_by_name, context_by_name):
        for row in ROW_ORDER:
            for variant in VARIANT_ORDER:
                model_name = semantic_groups[row].get(variant)
                if model_name is None or model_name not in means_by_name:
                    continue
                plotted_series.append(means_by_name[model_name].means)

    # Round the limits out to the 0.1 grid so the outer grid lines lie on the axes.
    xlim = _round_out_to_grid(_finite_x_limits(bin_centers, plotted_series))
    ylim = _round_out_to_grid(_finite_y_limits(plotted_series))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)
    panel_means = (prior_by_name, context_by_name)
    panel_titles = (prior_panel.title, context_panel.title)

    for row_index, row in enumerate(ROW_ORDER):
        for col_index, means_by_name in enumerate(panel_means):
            ax = axes[row_index, col_index]
            if row_index == 0:
                ax.set_title(panel_titles[col_index])

            for variant in VARIANT_ORDER:
                model_name = semantic_groups[row].get(variant)
                if model_name is None or model_name not in means_by_name:
                    continue
                ax.plot(
                    bin_centers,
                    means_by_name[model_name].means,
                    color=VARIANT_COLORS[variant],
                    linewidth=2.2,
                    alpha=0.95,
                    zorder=5,
                )

            ax.plot(
                bin_centers, ground_truth_means, color="black", linewidth=2.6, zorder=4
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if col_index == 0:
                ax.set_ylabel(row, fontweight="bold", labelpad=12)

    legend_handles = variant_legend_handles(include_ground_truth=True)
    fig.supylabel("Mean absolute drop (m)", x=SEMANTIC_GRID_SUPYLABEL_X)
    add_panel_labels(axes.flat)
    fig.tight_layout(rect=(0.07, SEMANTIC_GRID_BOTTOM, 1.0, 1.0))
    fig.text(0.5, SEMANTIC_GRID_XLABEL_Y, "Maximum height", ha="center", va="center")
    fig.legend(
        legend_handles,
        [str(handle.get_label()) for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, SEMANTIC_GRID_LEGEND_Y),
        ncol=len(legend_handles),
        frameon=False,
    )
    _save_plot(fig, output_path)


def _plot_generic_drop_means(
    bin_centers: np.ndarray,
    prior_panel: DropPanelData,
    context_panel: DropPanelData,
    output_path: Path,
    model_colors: dict[str, tuple],
    model_order: list[str],
) -> None:
    """Plot generic dual-panel absolute-drop figure."""
    fig, (ax_prior, ax_context) = get_figure("dual", nrows=1, ncols=2)
    model_handles: dict[str, plt.Line2D] = {}

    for model in prior_panel.model_means:
        line = ax_prior.plot(
            bin_centers,
            model.means,
            label=model.name,
            color=model_colors[model.name],
            linewidth=2,
            zorder=5,
        )[0]
        model_handles[model.name] = line

    gt_line = ax_prior.plot(
        bin_centers,
        prior_panel.ground_truth_means.means,
        label="Ground truth",
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=4,
    )[0]
    ax_prior.set_ylabel("Mean absolute drop (m)")
    ax_prior.set_title(prior_panel.title)

    for model in context_panel.model_means:
        line = ax_context.plot(
            bin_centers,
            model.means,
            label=model.name,
            color=model_colors[model.name],
            linewidth=2,
            zorder=5,
        )[0]
        if model.name not in model_handles:
            model_handles[model.name] = line

    ax_context.plot(
        bin_centers,
        context_panel.ground_truth_means.means,
        label="Ground truth",
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=4,
    )
    ax_context.set_title(context_panel.title)

    plotted_series = [prior_panel.ground_truth_means.means]
    plotted_series.extend(model.means for model in prior_panel.model_means)
    plotted_series.extend(model.means for model in context_panel.model_means)
    xlim = _finite_x_limits(bin_centers, plotted_series)
    ylim = _finite_y_limits(plotted_series)
    for ax in (ax_prior, ax_context):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    handles = [model_handles[name] for name in model_order if name in model_handles]
    labels = [name for name in model_order if name in model_handles]
    handles.append(gt_line)
    labels.append("Ground truth")

    model_label_count = len(labels) - 1
    legend_layout = bottom_legend_layout(
        handle_count=len(handles), model_label_count=model_label_count
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=legend_layout.legend_bottom)
    fig.text(0.5, legend_layout.x_label_y, "Maximum height", ha="center", va="top")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_layout.legend_y),
        ncol=legend_layout.ncol,
        frameon=False,
    )
    _save_plot(fig, output_path)


def plot_combined_absolute_drops(
    bin_centers: np.ndarray,
    prior_panel: DropPanelData,
    context_panel: DropPanelData,
    output_path: Path,
    model_colors: dict[str, tuple],
    model_order: list[str],
) -> None:
    """Plot absolute-drop figure, using semantic facets for known paper models."""
    if semantic_model_groups(model_order) is not None:
        _plot_semantic_drop_means(
            bin_centers=bin_centers,
            prior_panel=prior_panel,
            context_panel=context_panel,
            output_path=output_path,
            model_order=model_order,
        )
        return

    _plot_generic_drop_means(
        bin_centers=bin_centers,
        prior_panel=prior_panel,
        context_panel=context_panel,
        output_path=output_path,
        model_colors=model_colors,
        model_order=model_order,
    )


def calculate_combined_lodging_absolute_drop(
    base_folders: list[str],
    model_names: list[str],
    output_folder: str | None = None,
    bin_width: float = 0.05,
    relative_threshold: float = 0.2,
    absolute_threshold: float = 0.1,
    min_lodged_count: int = 25,
) -> None:
    """Calculate and visualize absolute lodging drops for both conditionings."""
    setup_style()

    if len(base_folders) != len(model_names):
        msg = (
            f"Number of base folders ({len(base_folders)}) must match "
            f"number of model names ({len(model_names)})."
        )
        raise ValueError(msg)

    logger.info(
        f"""
        Calculating combined lodging absolute drops with:
            - base_folders: {base_folders}
            - model_names: {model_names}
            - output_folder: {output_folder}
            - bin_width: {bin_width}
            - relative_threshold: {relative_threshold}
            - absolute_threshold: {absolute_threshold}
            - min_lodged_count: {min_lodged_count}
        """
    )

    models = discover_models(base_folders, model_names)
    if not models:
        msg = "No valid model folders found"
        raise ValueError(msg)

    first_path = next(
        (
            model.no_context_path or model.max_height_path
            for model in models
            if model.no_context_path or model.max_height_path
        ),
        None,
    )
    if first_path is None:
        msg = "No valid paths found for loading ground truth"
        raise ValueError(msg)

    ground_truth = _load_ground_truth_drops(first_path)
    logger.info(f"Loaded ground truth: {len(ground_truth.labels)} samples")
    logger.info(
        "Ground-truth: lodged={}, non_lodged={}",
        int(ground_truth.labels.sum()),
        ground_truth.labels.size - int(ground_truth.labels.sum()),
    )

    output_path = (
        Path(output_folder)
        if output_folder
        else Path("paper/combined_lodging_absolute_drop")
    )
    output_path.mkdir(parents=True, exist_ok=True)

    bins, centers = calculate_height_bins(ground_truth.max_heights, bin_width)
    gt_means, gt_counts, gt_lodged = calculate_binned_drop_means(
        values=ground_truth.max_heights,
        drop_abs=ground_truth.drop_abs,
        lodged_mask=ground_truth.labels,
        bins=bins,
        min_lodged_count=min_lodged_count,
    )
    ground_truth_means = BinnedDropMeans(
        name="Ground truth", means=gt_means, counts=gt_counts, lodged_counts=gt_lodged
    )

    prior_means: list[BinnedDropMeans] = []
    context_means: list[BinnedDropMeans] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_info: dict = {}
        for model in models:
            for cond in ("no_context", "max_height"):
                future = executor.submit(
                    _load_and_bin_model_drops,
                    model,
                    cond,
                    ground_truth,
                    bins,
                    relative_threshold,
                    absolute_threshold,
                    min_lodged_count,
                )
                future_to_info[future] = (model.name, cond)

        for future in as_completed(future_to_info):
            _name, cond = future_to_info[future]
            result = future.result()
            if result is not None:
                if cond == "no_context":
                    prior_means.append(result)
                else:
                    context_means.append(result)

    prior_panel = DropPanelData("no context", prior_means, ground_truth_means)
    context_panel = DropPanelData(
        "max-height context", context_means, ground_truth_means
    )

    _save_drop_csv(output_path, bins, centers, prior_panel, context_panel)

    model_colors = {model.name: model.color for model in models}
    plot_combined_absolute_drops(
        bin_centers=centers,
        prior_panel=prior_panel,
        context_panel=context_panel,
        output_path=output_path / "combined_lodging_absolute_drop",
        model_colors=model_colors,
        model_order=model_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined lodging absolute-drop visualization (prior + context)"
    )
    parser.add_argument(
        "--base-folders",
        type=str,
        nargs="+",
        required=True,
        help="Base paths to model results (without /max_height or /no_context suffix)",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        required=True,
        help="Display names for each model (must match number of base-folders)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (defaults to paper/combined_lodging_absolute_drop)",
    )
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--relative-threshold", type=float, default=0.2)
    parser.add_argument("--absolute-threshold", type=float, default=0.1)
    parser.add_argument("--min-lodged-count", type=int, default=25)
    args = parser.parse_args()

    calculate_combined_lodging_absolute_drop(
        base_folders=args.base_folders,
        model_names=args.model_names,
        output_folder=args.output_folder,
        bin_width=args.bin_width,
        relative_threshold=args.relative_threshold,
        absolute_threshold=args.absolute_threshold,
        min_lodged_count=args.min_lodged_count,
    )
