"""Combined lodging probability visualization for prior and context predictions.

Creates a dual-panel figure comparing:
- Left: Prior predictions (no_context) - binned by predicted max height
- Right: Context predictions (max_height) - binned by ground truth max height
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
from npnf.lodging import detect_lodging_grid
from npnf.scripts.paper.data_types import BinnedRates, GroundTruth, PredictionStats
from npnf.scripts.paper.histogram_utils import (
    calculate_binned_rates,
    calculate_height_bins,
    mask_low_sample_rates,
    reshape_grid_predictions,
)
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
from npnf.scripts.paper.style import COLOR_LIST, get_figure, setup_style
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    SyntheticBatchReconstructionCache,
    reconstruct_synthetic_batch,
)
from npnf.utils import add_panel_labels


@dataclass
class ModelConfig:
    """Configuration for a single model across conditioning types."""

    name: str
    no_context_path: Path | None
    max_height_path: Path | None
    color: tuple


@dataclass
class PanelData:
    """Data for a single panel."""

    title: str
    model_rates: list[BinnedRates]
    ground_truth_rates: BinnedRates


def assign_model_colors(model_names: list[str]) -> dict[str, tuple]:
    """Assign colors to models for cross-panel consistency."""
    model_colors = list(COLOR_LIST)
    return {
        name: model_colors[i % len(model_colors)] for i, name in enumerate(model_names)
    }


def discover_models(
    base_folders: list[str], model_names: list[str]
) -> list[ModelConfig]:
    """Discover available model folders and assign consistent colors.

    For each base_folder, checks for existence of:
      - {base_folder}/no_context/
      - {base_folder}/max_height/

    Returns list of ModelConfig with color assignments.
    Models with no valid folders are excluded with a warning.
    """
    colors = assign_model_colors(model_names)
    valid_models = []

    for base_folder, name in zip(base_folders, model_names, strict=True):
        base_path = Path(base_folder)
        no_context = base_path / "no_context"
        max_height = base_path / "max_height"

        nc_exists = no_context.is_dir()
        max_height_exists = max_height.is_dir()

        if not nc_exists and not max_height_exists:
            logger.warning(f"No valid folders for {name} at {base_folder}, skipping")
            continue

        if not nc_exists:
            logger.info(f"{name}: no_context not found, context panel only")
        if not max_height_exists:
            logger.info(f"{name}: max_height not found, prior panel only")

        valid_models.append(
            ModelConfig(
                name=name,
                no_context_path=no_context if nc_exists else None,
                max_height_path=max_height if max_height_exists else None,
                color=colors[name],
            )
        )

    return valid_models


def _load_ground_truth(method_dir: Path) -> GroundTruth:
    """Load ground truth data from batched format."""
    loader = BatchLoader(method_dir, load_predictions=False)
    cache = SyntheticBatchReconstructionCache()

    all_labels = []
    all_max_heights = []
    for batch_dict in loader:
        batch_dict = reconstruct_synthetic_batch(method_dir, batch_dict, cache=cache)
        labels = batch_dict["data"]["has_lodged"]
        if labels.ndim == 2:
            labels = labels[:, 0]
        all_labels.append(labels.numpy())

        y_original = batch_dict["data"]["height"]["Y_original"]
        if y_original.ndim == 4:
            y_original = y_original[:, 0]
        all_max_heights.append(y_original.flatten(1).max(dim=-1).values.numpy())

    return GroundTruth(
        labels=np.concatenate(all_labels), max_heights=np.concatenate(all_max_heights)
    )


def _load_predictions(
    method_dir: Path,
    relative_threshold: float,
    absolute_threshold: float,
    eps: float = 1e-8,
) -> PredictionStats:
    """Load and process predictions from batched format."""
    loader = BatchLoader(method_dir, load_batch_data=False)

    all_max, all_final, all_drop_abs, all_drop_rel, all_lodged = [], [], [], [], []

    for predictions in loader:
        grid_tensor = predictions["grid"]
        grid_tensor = grid_tensor[:, :, 0]
        grid_matrix, _ = reshape_grid_predictions(grid_tensor)

        stats = detect_lodging_grid(
            grid_matrix,
            relative_threshold=relative_threshold,
            absolute_threshold=absolute_threshold,
            eps=eps,
        )

        all_max.append(stats.max_height.numpy())
        all_final.append(stats.final_height.numpy())
        all_drop_abs.append(stats.drop_abs.numpy())
        all_drop_rel.append(stats.drop_rel.numpy())
        all_lodged.append(stats.is_lodged.numpy().astype(bool))

    return PredictionStats(
        max_height=np.concatenate(all_max),
        final_height=np.concatenate(all_final),
        drop_abs=np.concatenate(all_drop_abs),
        drop_rel=np.concatenate(all_drop_rel),
        is_lodged=np.concatenate(all_lodged),
    )


def _load_and_bin_model(
    model: ModelConfig,
    conditioning: str,
    ground_truth: GroundTruth,
    bins: np.ndarray,
    relative_threshold: float,
    absolute_threshold: float,
) -> BinnedRates | None:
    """Load predictions for one model/conditioning and compute binned rates.

    Args:
        model: Model configuration with paths.
        conditioning: "no_context" or "max_height".
        ground_truth: Shared ground truth data.
        bins: Height bin edges.
        relative_threshold: Lodging detection threshold.
        absolute_threshold: Lodging detection threshold.

    Returns:
        BinnedRates for this model/conditioning pair, or None if path missing.
    """
    path = (
        model.no_context_path if conditioning == "no_context" else model.max_height_path
    )
    if path is None:
        return None

    predictions = _load_predictions(path, relative_threshold, absolute_threshold)
    logger.info(
        f"[{model.name}] {conditioning}: Loaded {len(predictions.max_height)} samples"
    )

    single_height = None
    if conditioning == "no_context":
        # Bin by predicted max height
        rates, counts, lodged_counts = calculate_binned_rates(
            predictions.max_height, predictions.is_lodged, bins
        )
        heights = np.unique(predictions.max_height)
        if heights.size == 1:
            single_height = float(heights[0])
    else:
        # Bin by ground truth max height
        rates, counts, lodged_counts = calculate_binned_rates(
            ground_truth.max_heights, predictions.is_lodged, bins
        )

    return BinnedRates(
        name=model.name,
        rates=rates,
        counts=counts,
        lodged_counts=lodged_counts,
        single_height=single_height,
    )


def _compute_panel_data(
    title: str, model_rates: list[BinnedRates], ground_truth_rates: BinnedRates
) -> PanelData:
    """Assemble PanelData from pre-computed rates."""
    return PanelData(
        title=title, model_rates=model_rates, ground_truth_rates=ground_truth_rates
    )


def _save_histogram_csv(
    output_path: Path,
    bins: np.ndarray,
    centers: np.ndarray,
    prior_panel: PanelData,
    context_panel: PanelData,
) -> None:
    """Save combined histogram data to CSV."""
    histogram_data: dict = {
        "bin_left": bins[:-1],
        "bin_right": bins[1:],
        "bin_center": centers,
        # Ground truth (same for both panels)
        "ground_truth_lodged_count": prior_panel.ground_truth_rates.lodged_counts,
        "ground_truth_total_count": prior_panel.ground_truth_rates.counts,
        "ground_truth_lodging_rate": prior_panel.ground_truth_rates.rates,
    }

    # Prior panel models
    for model in prior_panel.model_rates:
        safe_name = model.name.replace(" ", "_").replace("/", "_")
        histogram_data[f"prior_{safe_name}_lodged_count"] = model.lodged_counts
        histogram_data[f"prior_{safe_name}_total_count"] = model.counts
        histogram_data[f"prior_{safe_name}_lodging_rate"] = model.rates

    # Context panel models
    for model in context_panel.model_rates:
        safe_name = model.name.replace(" ", "_").replace("/", "_")
        histogram_data[f"context_{safe_name}_lodged_count"] = model.lodged_counts
        histogram_data[f"context_{safe_name}_total_count"] = model.counts
        histogram_data[f"context_{safe_name}_lodging_rate"] = model.rates

    histogram_df = pl.DataFrame(histogram_data)
    histogram_df.write_csv(output_path / "combined_lodging_histogram.csv")


def _rates_by_name(rates: list[BinnedRates]) -> dict[str, BinnedRates]:
    """Return binned rates keyed by model name."""
    return {model.name: model for model in rates}


def _masked_percent_rates(rates: BinnedRates, min_sample_fraction: float) -> np.ndarray:
    """Mask low-sample rates and convert to percentages."""
    masked_rates = mask_low_sample_rates(rates.rates, rates.counts, min_sample_fraction)
    return masked_rates * 100.0


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


def _save_plot(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure as PNG and PDF."""
    for fmt in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{fmt}"))
    plt.close(fig)


def _plot_semantic_lodging_rates(
    bin_centers: np.ndarray,
    prior_panel: PanelData,
    context_panel: PanelData,
    output_path: Path,
    model_order: list[str],
    min_sample_fraction: float,
) -> None:
    """Plot known paper models in semantic model-type facets."""
    semantic_groups = semantic_model_groups(model_order)
    if semantic_groups is None:
        msg = "semantic plot requested with unknown model names"
        raise ValueError(msg)

    prior_by_name = _rates_by_name(prior_panel.model_rates)
    context_by_name = _rates_by_name(context_panel.model_rates)
    ground_truth_percent = _masked_percent_rates(
        prior_panel.ground_truth_rates, min_sample_fraction
    )

    plotted_series = [ground_truth_percent]
    for rates_by_name in (prior_by_name, context_by_name):
        for row in ROW_ORDER:
            for variant in VARIANT_ORDER:
                model_name = semantic_groups[row].get(variant)
                if model_name is None or model_name not in rates_by_name:
                    continue
                model_percent = _masked_percent_rates(
                    rates_by_name[model_name], min_sample_fraction
                )
                plotted_series.append(model_percent)

    xlim = _finite_x_limits(bin_centers, plotted_series)
    finite_maxima = [
        float(np.nanmax(series))
        for series in plotted_series
        if np.isfinite(series).any()
    ]
    y_max = max(finite_maxima) if finite_maxima else 105.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)
    panel_rates = (prior_by_name, context_by_name)
    panel_titles = (prior_panel.title, context_panel.title)

    for row_index, row in enumerate(ROW_ORDER):
        for col_index, rates_by_name in enumerate(panel_rates):
            ax = axes[row_index, col_index]
            if row_index == 0:
                ax.set_title(panel_titles[col_index])

            for variant in VARIANT_ORDER:
                model_name = semantic_groups[row].get(variant)
                if model_name is None or model_name not in rates_by_name:
                    continue
                model_rates = rates_by_name[model_name]
                if model_rates.single_height is not None:
                    # One predicted height for every sample: a line through one
                    # bin is invisible, so draw the model as one point.
                    ax.plot(
                        [model_rates.single_height],
                        [
                            100.0
                            * model_rates.lodged_counts.sum()
                            / model_rates.counts.sum()
                        ],
                        color=VARIANT_COLORS[variant],
                        marker="o",
                        markersize=7,
                        markerfacecolor="none",
                        markeredgewidth=2.0,
                        linestyle="none",
                        clip_on=False,
                        zorder=6,
                    )
                    continue
                model_percent = _masked_percent_rates(model_rates, min_sample_fraction)
                ax.plot(
                    bin_centers,
                    model_percent,
                    color=VARIANT_COLORS[variant],
                    linewidth=2.2,
                    alpha=0.95,
                    zorder=5,
                )

            ax.plot(
                bin_centers,
                ground_truth_percent,
                color="black",
                linewidth=2.6,
                zorder=4,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(0.0, y_max)
            if col_index == 0:
                ax.set_ylabel(row, fontweight="bold", labelpad=12)

    legend_handles = variant_legend_handles(include_ground_truth=True)
    fig.supylabel("Lodging %", x=SEMANTIC_GRID_SUPYLABEL_X)
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


def _plot_generic_combined_lodging_rates(
    bin_centers: np.ndarray,
    prior_panel: PanelData,
    context_panel: PanelData,
    output_path: Path,
    model_colors: dict[str, tuple],
    model_order: list[str],
    min_sample_fraction: float,
) -> None:
    """Plot generic dual-panel figure with one color per model."""
    fig, (ax_prior, ax_context) = get_figure("dual", nrows=1, ncols=2)

    # Store handles by model name to preserve user-specified order
    model_handles: dict[str, plt.Line2D] = {}

    # Plot prior panel (left)
    for model in prior_panel.model_rates:
        model_percent = _masked_percent_rates(model, min_sample_fraction)
        color = model_colors[model.name]
        line = ax_prior.plot(
            bin_centers,
            model_percent,
            label=model.name,
            color=color,
            linewidth=2,
            zorder=5,
        )[0]
        model_handles[model.name] = line

    # Ground truth for prior panel
    gt_percent = _masked_percent_rates(
        prior_panel.ground_truth_rates, min_sample_fraction
    )
    gt_line = ax_prior.plot(
        bin_centers,
        gt_percent,
        label="Ground truth",
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=4,
    )[0]

    ax_prior.set_ylabel("Lodging %")
    ax_prior.set_title(prior_panel.title)

    # Plot context panel (right)
    for model in context_panel.model_rates:
        model_percent = _masked_percent_rates(model, min_sample_fraction)
        color = model_colors[model.name]
        line = ax_context.plot(
            bin_centers,
            model_percent,
            label=model.name,
            color=color,
            linewidth=2,
            zorder=5,
        )[0]
        # Only store if not already stored from prior panel
        if model.name not in model_handles:
            model_handles[model.name] = line

    # Ground truth for context panel
    ax_context.plot(
        bin_centers,
        gt_percent,
        label="Ground truth",
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=4,
    )

    ax_context.set_title(context_panel.title)

    # Build legend in user-specified order
    handles = [model_handles[name] for name in model_order if name in model_handles]
    labels = [name for name in model_order if name in model_handles]
    handles.append(gt_line)
    labels.append("Ground truth")

    # Set consistent axis limits based on ground truth
    valid_mask = ~np.isnan(gt_percent)
    if valid_mask.any():
        x_max = float(bin_centers[valid_mask].max())
        y_max = float(np.nanmax(gt_percent))
        for ax in (ax_prior, ax_context):
            ax.set_xlim(0.6, x_max * 1.05)
            ax.set_ylim(0.0, y_max * 1.1)
    else:
        for ax in (ax_prior, ax_context):
            ax.set_xlim(0.6, 2.0)
            ax.set_ylim(0.0, 105.0)

    model_label_count = len(labels) - 1
    legend_layout = bottom_legend_layout(
        handle_count=len(handles), model_label_count=model_label_count
    )

    # Layout adjustment
    fig.tight_layout()
    fig.subplots_adjust(bottom=legend_layout.legend_bottom)

    # Shared x-axis label (positioned between plots and legend)
    fig.text(0.5, legend_layout.x_label_y, "Maximum height", ha="center", va="top")

    # Shared legend below x-label
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_layout.legend_y),
        ncol=legend_layout.ncol,
        frameon=False,
    )

    _save_plot(fig, output_path)


def plot_combined_lodging_rates(
    bin_centers: np.ndarray,
    prior_panel: PanelData,
    context_panel: PanelData,
    output_path: Path,
    model_colors: dict[str, tuple],
    model_order: list[str],
    min_sample_fraction: float = 0.01,
) -> None:
    """Plot lodging-rate figure, using semantic facets for known paper models."""
    if semantic_model_groups(model_order) is not None:
        _plot_semantic_lodging_rates(
            bin_centers=bin_centers,
            prior_panel=prior_panel,
            context_panel=context_panel,
            output_path=output_path,
            model_order=model_order,
            min_sample_fraction=min_sample_fraction,
        )
        return

    _plot_generic_combined_lodging_rates(
        bin_centers=bin_centers,
        prior_panel=prior_panel,
        context_panel=context_panel,
        output_path=output_path,
        model_colors=model_colors,
        model_order=model_order,
        min_sample_fraction=min_sample_fraction,
    )


def calculate_combined_lodging_probability(
    base_folders: list[str],
    model_names: list[str],
    output_folder: str | None = None,
    bin_width: float = 0.05,
    relative_threshold: float = 0.2,
    absolute_threshold: float = 0.1,
    min_sample_fraction: float = 0.001,
) -> None:
    """Calculate and visualize lodging probability for both conditioning types.

    Creates a dual-panel figure with no_context (prior) on left and max_height
    (context) on right.
    """
    setup_style()

    if len(base_folders) != len(model_names):
        msg = (
            f"Number of base folders ({len(base_folders)}) must match "
            f"number of model names ({len(model_names)})."
        )
        raise ValueError(msg)

    logger.info(
        f"""
        Calculating combined lodging probabilities with:
            - base_folders: {base_folders}
            - model_names: {model_names}
            - output_folder: {output_folder}
            - bin_width: {bin_width}
            - relative_threshold: {relative_threshold}
            - absolute_threshold: {absolute_threshold}
            - min_sample_fraction: {min_sample_fraction}
        """
    )

    # Discover available models
    models = discover_models(base_folders, model_names)
    if not models:
        msg = "No valid model folders found"
        raise ValueError(msg)

    # Find first valid path to load ground truth
    first_path = next(
        (
            m.no_context_path or m.max_height_path
            for m in models
            if m.no_context_path or m.max_height_path
        ),
        None,
    )
    if first_path is None:
        msg = "No valid paths found for loading ground truth"
        raise ValueError(msg)

    # Load ground truth once
    ground_truth = _load_ground_truth(first_path)
    logger.info(f"Loaded ground truth: {len(ground_truth.labels)} samples")
    lodged_ct = int(ground_truth.labels.sum())
    non_lodged_ct = ground_truth.labels.size - lodged_ct
    logger.info(f"Ground-truth: lodged={lodged_ct}, non_lodged={non_lodged_ct}")

    # Set up output path
    output_path = (
        Path(output_folder)
        if output_folder
        else Path("paper/combined_lodging_probability")
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute bins from ground truth
    bins, centers = calculate_height_bins(ground_truth.max_heights, bin_width)

    # Compute ground truth binned rates (shared across panels)
    gt_rates, gt_counts, gt_lodged = calculate_binned_rates(
        ground_truth.max_heights, ground_truth.labels, bins
    )
    ground_truth_rates = BinnedRates(
        name="Ground truth", rates=gt_rates, counts=gt_counts, lodged_counts=gt_lodged
    )

    # Load and bin all model predictions in parallel
    prior_rates: list[BinnedRates] = []
    context_rates: list[BinnedRates] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_info: dict = {}
        for model in models:
            for cond in ("no_context", "max_height"):
                future = executor.submit(
                    _load_and_bin_model,
                    model,
                    cond,
                    ground_truth,
                    bins,
                    relative_threshold,
                    absolute_threshold,
                )
                future_to_info[future] = (model.name, cond)

        for future in as_completed(future_to_info):
            _name, cond = future_to_info[future]
            result = future.result()
            if result is not None:
                if cond == "no_context":
                    prior_rates.append(result)
                else:
                    context_rates.append(result)

    prior_panel = _compute_panel_data("no context", prior_rates, ground_truth_rates)
    context_panel = _compute_panel_data(
        "max-height context", context_rates, ground_truth_rates
    )

    # Save histogram data
    _save_histogram_csv(output_path, bins, centers, prior_panel, context_panel)

    # Build color mapping
    model_colors = {m.name: m.color for m in models}

    # Plot
    plot_combined_lodging_rates(
        bin_centers=centers,
        prior_panel=prior_panel,
        context_panel=context_panel,
        output_path=output_path / "combined_lodging_probability",
        model_colors=model_colors,
        model_order=model_names,
        min_sample_fraction=min_sample_fraction,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined lodging probability visualization (prior + context)"
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
        help="Output folder (defaults to paper/combined_lodging_probability)",
    )
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--relative-threshold", type=float, default=0.2)
    parser.add_argument("--absolute-threshold", type=float, default=0.1)
    parser.add_argument("--min-sample-fraction", type=float, default=0.01)
    args = parser.parse_args()

    calculate_combined_lodging_probability(
        base_folders=args.base_folders,
        model_names=args.model_names,
        output_folder=args.output_folder,
        bin_width=args.bin_width,
        relative_threshold=args.relative_threshold,
        absolute_threshold=args.absolute_threshold,
        min_sample_fraction=args.min_sample_fraction,
    )
