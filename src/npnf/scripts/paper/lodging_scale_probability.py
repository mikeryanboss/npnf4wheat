"""Lodging probability visualization for lodging-scale sweeps.

Creates a dual-panel figure comparing:
- Left: Prior predictions (no_context), binned by predicted max height
- Right: Context predictions (max_height), binned by ground-truth max height

Optionally creates a two-row figure for comparing two model families across the
same lodging-scale labels.

Unlike lodging_probability.py, this script uses one GT simulator lodging
curve per model scale instead of one shared empirical ground-truth curve.
"""

import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from loguru import logger

from npnf.data.synthetic.lodging import lodging_probability
from npnf.scripts.paper.data_types import BinnedRates, GroundTruth, PredictionStats
from npnf.scripts.paper.histogram_utils import (
    calculate_binned_rates,
    calculate_height_bins,
    mask_low_sample_rates,
)
from npnf.scripts.paper.lodging_probability import (
    ModelConfig,
    _load_ground_truth,
    _load_predictions,
    discover_models,
)
from npnf.scripts.paper.style import get_figure, setup_style
from npnf.utils import add_panel_labels


@dataclass
class ScaleModelConfig:
    """Model configuration paired with its training lodging scale."""

    model: ModelConfig
    lodging_weibull_scale: float


@dataclass
class ScalePanelData:
    """Binned model and GT rates for one panel."""

    title: str
    model_rates: list[BinnedRates]
    gt_rates: list[BinnedRates]


@dataclass
class EmpiricalOverlay:
    """Empirical lodging curve loaded from saved batch labels."""

    name: str
    rates: BinnedRates
    color: tuple


@dataclass
class ScaleRowPanels:
    """Prior/context panels and empirical overlays for one figure row."""

    label: str
    prior_panel: ScalePanelData
    context_panel: ScalePanelData
    empirical_overlays: dict[str, EmpiricalOverlay]


@dataclass(frozen=True)
class ScaleLegendLayout:
    """Legend and shared-label placement for lodging-scale figures."""

    bottom: float
    x_label_y: float
    legend_y: float


def validate_input_lengths(
    base_folders: list[str], model_names: list[str], lodging_weibull_scales: list[float]
) -> None:
    """Validate model CLI list lengths before folder discovery filters models."""
    if len(base_folders) != len(model_names) or len(base_folders) != len(
        lodging_weibull_scales
    ):
        msg = (
            "Number of base folders, model names, and lodging Weibull scales must "
            "match: "
            f"base_folders={len(base_folders)}, "
            f"model_names={len(model_names)}, "
            f"lodging_weibull_scales={len(lodging_weibull_scales)}."
        )
        raise ValueError(msg)


def validate_two_row_input_lengths(
    base_folders: list[str],
    base_folders_row2: list[str],
    model_names: list[str],
    lodging_weibull_scales: list[float],
) -> None:
    """Validate two-row CLI list lengths before folder discovery filters models."""
    if (
        len(base_folders) != len(base_folders_row2)
        or len(base_folders) != len(model_names)
        or len(base_folders) != len(lodging_weibull_scales)
    ):
        msg = (
            "Number of row 1 base folders, row 2 base folders, model names, "
            "and lodging Weibull scales must match: "
            f"base_folders={len(base_folders)}, "
            f"base_folders_row2={len(base_folders_row2)}, "
            f"model_names={len(model_names)}, "
            f"lodging_weibull_scales={len(lodging_weibull_scales)}."
        )
        raise ValueError(msg)


def analytic_lodging_probability(
    heights: np.ndarray,
    lodging_weibull_scale: float,
    lodging_weibull_shape: float = 7.0,
    lodging_weibull_offset: float = 0.0,
    lodging_height_clamp: float = 2.0,
) -> np.ndarray:
    """Compute the synthetic simulator lodging probability curve."""
    return lodging_probability(
        torch.as_tensor(heights),
        lodging_height_clamp=lodging_height_clamp,
        lodging_weibull_shape=lodging_weibull_shape,
        lodging_weibull_scale=lodging_weibull_scale,
        lodging_weibull_offset=lodging_weibull_offset,
    ).numpy()


def _safe_name(name: str) -> str:
    """Match lodging_probability.py CSV label sanitization."""
    return name.replace(" ", "_").replace("/", "_")


def _available_model_path(model: ModelConfig) -> Path | None:
    """Return any existing method path for loading batch data."""
    return model.no_context_path or model.max_height_path


def _load_shared_ground_truth(scale_models: list[ScaleModelConfig]) -> GroundTruth:
    """Load shared test-set ground truth from the first valid model path."""
    for scale_model in scale_models:
        path = _available_model_path(scale_model.model)
        if path is None:
            continue

        ground_truth = _load_ground_truth(path)
        logger.info(
            "[{}] shared GT: Loaded {} samples",
            scale_model.model.name,
            len(ground_truth.labels),
        )
        return ground_truth

    msg = "No valid model path found for loading shared ground truth"
    raise ValueError(msg)


def _build_scale_models(
    base_folders: list[str], model_names: list[str], lodging_weibull_scales: list[float]
) -> list[ScaleModelConfig]:
    """Discover model folders and attach the corresponding lodging scale."""
    validate_input_lengths(base_folders, model_names, lodging_weibull_scales)

    scale_by_name = dict(zip(model_names, lodging_weibull_scales, strict=True))
    models = discover_models(base_folders, model_names)
    scale_models = [
        ScaleModelConfig(
            model=model, lodging_weibull_scale=float(scale_by_name[model.name])
        )
        for model in models
    ]

    if not scale_models:
        msg = "No valid model folders found"
        raise ValueError(msg)
    return scale_models


def _resolve_requested_empirical_names(
    scale_models: list[ScaleModelConfig], requested_names: set[str]
) -> set[str]:
    """Validate requested empirical overlay names for one row."""
    if not requested_names:
        return set()

    available_names = {scale_model.model.name for scale_model in scale_models}
    missing_names = requested_names - available_names
    if missing_names:
        missing = ", ".join(sorted(missing_names))
        msg = f"Cannot load empirical ground truth for unknown model name(s): {missing}"
        raise ValueError(msg)
    return set(requested_names)


def _resolve_requested_empirical_name_rows(
    scale_model_rows: list[list[ScaleModelConfig]], requested_names: set[str]
) -> list[set[str]]:
    """Validate requested empirical overlay names and return row-local names."""
    if not requested_names:
        return [set() for _ in scale_model_rows]

    found_names = set()
    requested_name_rows = []
    for scale_models in scale_model_rows:
        available_names = {scale_model.model.name for scale_model in scale_models}
        row_requested = requested_names & available_names
        found_names.update(row_requested)
        requested_name_rows.append(row_requested)

    missing_names = requested_names - found_names
    if missing_names:
        missing = ", ".join(sorted(missing_names))
        msg = f"Cannot load empirical ground truth for unknown model name(s): {missing}"
        raise ValueError(msg)

    return requested_name_rows


def _validate_prediction_length(
    model_name: str,
    conditioning: str,
    path: Path,
    predictions: PredictionStats,
    ground_truth: GroundTruth,
    ground_truth_label: str,
) -> None:
    """Ensure loaded context predictions align with ground-truth samples."""
    prediction_count = len(predictions.is_lodged)
    ground_truth_count = len(ground_truth.max_heights)
    if prediction_count != ground_truth_count:
        msg = (
            f"{model_name} {conditioning} predictions at {path} have "
            f"{prediction_count} samples, but {ground_truth_label} ground truth "
            f"has {ground_truth_count}. All lodging-scale context inputs must use "
            "the same test-set sample order as their batch labels."
        )
        raise ValueError(msg)


def _resolve_context_ground_truth(
    model_name: str,
    path: Path,
    predictions: PredictionStats,
    shared_ground_truth: GroundTruth | None,
) -> GroundTruth:
    """Use shared context ground truth, falling back for incomplete artifacts."""
    if shared_ground_truth is None:
        msg = "Context predictions require shared ground truth"
        raise ValueError(msg)

    if len(predictions.is_lodged) == len(shared_ground_truth.max_heights):
        return shared_ground_truth

    logger.warning(
        "[{}] max_height: prediction count ({}) differs from shared GT ({}); "
        "loading method-local ground truth",
        model_name,
        len(predictions.is_lodged),
        len(shared_ground_truth.max_heights),
    )
    local_ground_truth = _load_ground_truth(path)
    _validate_prediction_length(
        model_name, "max_height", path, predictions, local_ground_truth, "method-local"
    )
    return local_ground_truth


def _load_prediction_rows_parallel(
    scale_model_rows: list[list[ScaleModelConfig]],
    relative_threshold: float,
    absolute_threshold: float,
    shared_ground_truth: GroundTruth | None,
) -> tuple[
    list[dict[str, PredictionStats]],
    list[dict[str, PredictionStats]],
    list[dict[str, GroundTruth]],
]:
    """Load available prior/context predictions for all rows concurrently."""
    prior_prediction_rows: list[dict[str, PredictionStats]] = [
        {} for _ in scale_model_rows
    ]
    context_prediction_rows: list[dict[str, PredictionStats]] = [
        {} for _ in scale_model_rows
    ]
    context_ground_truth_rows: list[dict[str, GroundTruth]] = [
        {} for _ in scale_model_rows
    ]
    load_tasks = []
    for row_index, scale_models in enumerate(scale_model_rows):
        for scale_model in scale_models:
            model = scale_model.model
            if model.no_context_path is not None:
                load_tasks.append(
                    (row_index, model.name, "no_context", model.no_context_path)
                )
            if model.max_height_path is not None:
                load_tasks.append(
                    (row_index, model.name, "max_height", model.max_height_path)
                )

    if not load_tasks:
        return prior_prediction_rows, context_prediction_rows, context_ground_truth_rows

    max_workers = min(4, len(load_tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _load_predictions, path, relative_threshold, absolute_threshold
            ): (row_index, model_name, conditioning, path)
            for row_index, model_name, conditioning, path in load_tasks
        }
        for future in as_completed(future_to_task):
            row_index, model_name, conditioning, path = future_to_task[future]
            predictions = future.result()
            if conditioning == "no_context":
                prior_prediction_rows[row_index][model_name] = predictions
            else:
                context_prediction_rows[row_index][model_name] = predictions
                context_ground_truth_rows[row_index][model_name] = (
                    _resolve_context_ground_truth(
                        model_name, path, predictions, shared_ground_truth
                    )
                )
            logger.info(
                f"[{model_name}] {conditioning}: Loaded "
                f"{len(predictions.max_height)} samples"
            )

    return prior_prediction_rows, context_prediction_rows, context_ground_truth_rows


def _compute_prior_panel_data(
    scale_models: list[ScaleModelConfig],
    predictions_by_name: dict[str, PredictionStats],
    bins: np.ndarray,
    bin_centers: np.ndarray,
    lodging_weibull_shape: float,
    lodging_weibull_offset: float,
    lodging_height_clamp: float,
) -> ScalePanelData:
    """Compute binned prior prediction rates and GT references."""
    model_rates = []
    gt_rates = []

    for scale_model in scale_models:
        model = scale_model.model
        predictions = predictions_by_name.get(model.name)
        if predictions is None:
            continue

        rates, counts, lodged_counts = calculate_binned_rates(
            predictions.max_height, predictions.is_lodged, bins
        )
        model_rates.append(
            BinnedRates(
                name=model.name, rates=rates, counts=counts, lodged_counts=lodged_counts
            )
        )

        gt_rate = analytic_lodging_probability(
            bin_centers,
            lodging_weibull_scale=scale_model.lodging_weibull_scale,
            lodging_weibull_shape=lodging_weibull_shape,
            lodging_weibull_offset=lodging_weibull_offset,
            lodging_height_clamp=lodging_height_clamp,
        )
        gt_rates.append(
            BinnedRates(
                name=model.name,
                rates=gt_rate,
                counts=np.ones_like(gt_rate, dtype=int),
                lodged_counts=np.zeros_like(gt_rate, dtype=int),
            )
        )

    return ScalePanelData(
        title="no context", model_rates=model_rates, gt_rates=gt_rates
    )


def _compute_context_panel_data(
    scale_models: list[ScaleModelConfig],
    predictions_by_name: dict[str, PredictionStats],
    ground_truth_by_name: dict[str, GroundTruth],
    bins: np.ndarray,
    bin_centers: np.ndarray,
    lodging_weibull_shape: float,
    lodging_weibull_offset: float,
    lodging_height_clamp: float,
) -> ScalePanelData:
    """Compute binned context prediction rates and GT references."""
    model_rates = []
    gt_rates = []

    for scale_model in scale_models:
        model = scale_model.model
        predictions = predictions_by_name.get(model.name)
        ground_truth = ground_truth_by_name.get(model.name)
        if predictions is None or ground_truth is None:
            continue

        rates, counts, lodged_counts = calculate_binned_rates(
            ground_truth.max_heights, predictions.is_lodged, bins
        )
        model_rates.append(
            BinnedRates(
                name=model.name, rates=rates, counts=counts, lodged_counts=lodged_counts
            )
        )

        gt_rate = analytic_lodging_probability(
            bin_centers,
            lodging_weibull_scale=scale_model.lodging_weibull_scale,
            lodging_weibull_shape=lodging_weibull_shape,
            lodging_weibull_offset=lodging_weibull_offset,
            lodging_height_clamp=lodging_height_clamp,
        )
        gt_rates.append(
            BinnedRates(
                name=model.name,
                rates=gt_rate,
                counts=np.ones_like(gt_rate, dtype=int),
                lodged_counts=np.zeros_like(gt_rate, dtype=int),
            )
        )

    return ScalePanelData(
        title="max-height context", model_rates=model_rates, gt_rates=gt_rates
    )


def _compute_empirical_overlays(
    ground_truth: GroundTruth | None,
    requested_names: set[str],
    scale_models: list[ScaleModelConfig],
    bins: np.ndarray,
) -> dict[str, EmpiricalOverlay]:
    """Bin shared empirical ground truth once for requested scale labels."""
    if not requested_names:
        return {}
    if ground_truth is None:
        msg = "Empirical ground-truth overlays require shared ground truth"
        raise ValueError(msg)

    colors_by_name = {
        scale_model.model.name: scale_model.model.color for scale_model in scale_models
    }
    overlays = {}
    for name in sorted(requested_names):
        rates, counts, lodged_counts = calculate_binned_rates(
            ground_truth.max_heights, ground_truth.labels, bins
        )
        overlays[name] = EmpiricalOverlay(
            name=name,
            rates=BinnedRates(
                name=name, rates=rates, counts=counts, lodged_counts=lodged_counts
            ),
            color=colors_by_name[name],
        )
    return overlays


def _save_scale_histogram_csv(
    output_path: Path,
    bins: np.ndarray,
    centers: np.ndarray,
    prior_panel: ScalePanelData,
    context_panel: ScalePanelData,
    empirical_overlays: dict[str, EmpiricalOverlay],
) -> None:
    """Save binned model, GT, and empirical rates to CSV."""
    histogram_data: dict[str, np.ndarray] = {
        "bin_left": bins[:-1],
        "bin_right": bins[1:],
        "bin_center": centers,
    }

    for prefix, panel in (("prior", prior_panel), ("context", context_panel)):
        for model in panel.model_rates:
            safe_name = _safe_name(model.name)
            histogram_data[f"{prefix}_model_{safe_name}_lodged_count"] = (
                model.lodged_counts
            )
            histogram_data[f"{prefix}_model_{safe_name}_total_count"] = model.counts
            histogram_data[f"{prefix}_model_{safe_name}_lodging_rate"] = model.rates

        for gt_rate in panel.gt_rates:
            safe_name = _safe_name(gt_rate.name)
            histogram_data[f"{prefix}_gt_{safe_name}_lodging_rate"] = gt_rate.rates

    for name in sorted(empirical_overlays):
        overlay = empirical_overlays[name].rates
        safe_name = _safe_name(name)
        histogram_data[f"empirical_gt_{safe_name}_lodged_count"] = overlay.lodged_counts
        histogram_data[f"empirical_gt_{safe_name}_total_count"] = overlay.counts
        histogram_data[f"empirical_gt_{safe_name}_lodging_rate"] = overlay.rates

    csv_path = output_path / "lodging_scale_probability.csv"
    pl.DataFrame(histogram_data).write_csv(csv_path)


def _save_two_row_scale_histogram_csv(
    output_path: Path,
    bins: np.ndarray,
    centers: np.ndarray,
    row_panels: list[ScaleRowPanels],
) -> None:
    """Save two-row binned model, GT, and empirical rates to CSV."""
    histogram_data: dict[str, np.ndarray] = {
        "bin_left": bins[:-1],
        "bin_right": bins[1:],
        "bin_center": centers,
    }

    for row in row_panels:
        safe_row = _safe_name(row.label)
        panels = (("prior", row.prior_panel), ("context", row.context_panel))
        for prefix, panel in panels:
            for model in panel.model_rates:
                safe_name = _safe_name(model.name)
                column_prefix = f"{prefix}_model_{safe_row}_{safe_name}"
                histogram_data[f"{column_prefix}_lodged_count"] = model.lodged_counts
                histogram_data[f"{column_prefix}_total_count"] = model.counts
                histogram_data[f"{column_prefix}_lodging_rate"] = model.rates

            for gt_rate in panel.gt_rates:
                safe_name = _safe_name(gt_rate.name)
                histogram_data[f"{prefix}_gt_{safe_row}_{safe_name}_lodging_rate"] = (
                    gt_rate.rates
                )

        for name in sorted(row.empirical_overlays):
            overlay = row.empirical_overlays[name].rates
            safe_name = _safe_name(name)
            column_prefix = f"empirical_gt_{safe_row}_{safe_name}"
            histogram_data[f"{column_prefix}_lodged_count"] = overlay.lodged_counts
            histogram_data[f"{column_prefix}_total_count"] = overlay.counts
            histogram_data[f"{column_prefix}_lodging_rate"] = overlay.rates

    csv_path = output_path / "lodging_scale_probability.csv"
    pl.DataFrame(histogram_data).write_csv(csv_path)


def _plot_panel(
    ax: plt.Axes,
    bin_centers: np.ndarray,
    panel: ScalePanelData,
    empirical_overlays: dict[str, EmpiricalOverlay],
    model_colors: dict[str, tuple],
    min_sample_fraction: float,
    show_title: bool = True,
) -> tuple[list[plt.Line2D], list[str]]:
    """Plot one panel and return legend handles and labels."""
    handles = []
    labels = []
    gt_by_name = {gt_rate.name: gt_rate for gt_rate in panel.gt_rates}

    for model in panel.model_rates:
        gt_rate = gt_by_name[model.name]
        masked_rates = mask_low_sample_rates(
            model.rates, model.counts, min_sample_fraction
        )
        color = model_colors[model.name]
        line = ax.plot(
            bin_centers,
            masked_rates * 100.0,
            label=model.name,
            color=color,
            linewidth=2.0,
            zorder=5,
        )[0]

        gt_line = ax.plot(
            bin_centers,
            gt_rate.rates * 100.0,
            label=f"{model.name} GT",
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            zorder=4,
        )[0]
        handles.extend([line, gt_line])
        labels.extend([model.name, f"{model.name} GT"])

    for name in sorted(empirical_overlays):
        overlay = empirical_overlays[name]
        masked_rates = mask_low_sample_rates(
            overlay.rates.rates, overlay.rates.counts, min_sample_fraction
        )
        empirical_line = ax.plot(
            bin_centers,
            masked_rates * 100.0,
            label=f"{name} empirical GT",
            color=overlay.color,
            linestyle=":",
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            zorder=6,
        )[0]
        handles.append(empirical_line)
        labels.append(f"{name} empirical GT")

    if show_title:
        ax.set_title(panel.title)
    return handles, labels


def _resolve_axis_limits(
    bin_centers: np.ndarray, x_limit: tuple[float, float] | None
) -> tuple[float, float]:
    """Resolve x-axis limits for plotting."""
    if x_limit is not None:
        return x_limit
    return 0.6, float(bin_centers.max()) if bin_centers.size else 2.0


def _collect_visible_rate_arrays(
    panels: tuple[ScalePanelData, ScalePanelData],
    empirical_overlays: dict[str, EmpiricalOverlay],
    visible: np.ndarray,
    min_sample_fraction: float,
) -> list[np.ndarray]:
    """Collect plotted rate arrays for shared y-axis scaling."""
    y_arrays = []
    for panel in panels:
        y_arrays.extend(
            mask_low_sample_rates(model.rates, model.counts, min_sample_fraction)[
                visible
            ]
            for model in panel.model_rates
        )
        y_arrays.extend(gt_rate.rates[visible] for gt_rate in panel.gt_rates)

    y_arrays.extend(
        mask_low_sample_rates(
            overlay.rates.rates, overlay.rates.counts, min_sample_fraction
        )[visible]
        for overlay in empirical_overlays.values()
    )
    return y_arrays


def _calculate_y_axis_max(y_arrays: list[np.ndarray]) -> float:
    """Calculate a shared y-axis maximum from plotted rates."""
    finite_maxima = [
        float(np.nanmax(array)) for array in y_arrays if np.isfinite(array).any()
    ]
    if not finite_maxima:
        return 1.0
    return max(finite_maxima)


def _build_ordered_legend(
    handles: list[plt.Line2D], labels: list[str], model_order: list[str]
) -> tuple[list[plt.Line2D], list[str], int]:
    """Order legend entries so each model's entries occupy one legend column."""
    handle_by_label = {}
    for handle, label in zip(handles, labels, strict=True):
        handle_by_label.setdefault(label, handle)

    ordered_labels = []
    column_count = 0
    for name in model_order:
        model_labels = [
            label
            for label in (name, f"{name} GT", f"{name} empirical GT")
            if label in handle_by_label
        ]
        if model_labels:
            column_count += 1
            ordered_labels.extend(model_labels)
    ordered_labels.extend(
        label for label in handle_by_label if label not in ordered_labels
    )
    if column_count == 0:
        column_count = max(len(ordered_labels), 1)
    return (
        [handle_by_label[label] for label in ordered_labels],
        ordered_labels,
        column_count,
    )


def _legend_row_count(label_count: int, column_count: int) -> int:
    return math.ceil(label_count / max(column_count, 1))


def _single_row_scale_legend_layout(
    *, label_count: int, column_count: int
) -> ScaleLegendLayout:
    if _legend_row_count(label_count, column_count) > 2:
        return ScaleLegendLayout(bottom=0.240, x_label_y=0.140, legend_y=0.110)
    return ScaleLegendLayout(bottom=0.205, x_label_y=0.130, legend_y=0.100)


def _two_row_scale_legend_layout(
    *, label_count: int, column_count: int
) -> ScaleLegendLayout:
    if _legend_row_count(label_count, column_count) > 2:
        return ScaleLegendLayout(bottom=0.220, x_label_y=0.155, legend_y=0.035)
    return ScaleLegendLayout(bottom=0.190, x_label_y=0.145, legend_y=0.055)


def _plot_scale_lodging_rates(
    bin_centers: np.ndarray,
    prior_panel: ScalePanelData,
    context_panel: ScalePanelData,
    empirical_overlays: dict[str, EmpiricalOverlay],
    model_colors: dict[str, tuple],
    output_path: Path,
    model_order: list[str],
    min_sample_fraction: float,
    x_limit: tuple[float, float] | None,
) -> None:
    """Plot dual-panel scale-sweep lodging probability figure."""
    fig, (ax_prior, ax_context) = get_figure("dual", nrows=1, ncols=2)

    prior_handles, prior_labels = _plot_panel(
        ax_prior,
        bin_centers,
        prior_panel,
        empirical_overlays,
        model_colors,
        min_sample_fraction,
    )
    context_handles, context_labels = _plot_panel(
        ax_context,
        bin_centers,
        context_panel,
        empirical_overlays,
        model_colors,
        min_sample_fraction,
    )

    ax_prior.set_ylabel("Lodging %")

    x_min, x_max = _resolve_axis_limits(bin_centers, x_limit)
    for ax in (ax_prior, ax_context):
        ax.set_xlim(x_min, x_max)

    visible = (bin_centers >= x_min) & (bin_centers <= x_max)
    y_arrays = _collect_visible_rate_arrays(
        (prior_panel, context_panel), empirical_overlays, visible, min_sample_fraction
    )
    y_max = _calculate_y_axis_max(y_arrays)
    for ax in (ax_prior, ax_context):
        ax.set_ylim(0.0, min(100.0, max(65.0, y_max * 110.0)))

    handles, ordered_labels, ncol = _build_ordered_legend(
        [*prior_handles, *context_handles],
        [*prior_labels, *context_labels],
        model_order,
    )
    legend_layout = _single_row_scale_legend_layout(
        label_count=len(ordered_labels), column_count=ncol
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=legend_layout.bottom)
    fig.text(0.5, legend_layout.x_label_y, "Maximum height", ha="center", va="top")

    fig.legend(
        handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_layout.legend_y),
        ncol=ncol,
        frameon=False,
    )

    for fmt in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{fmt}"))
    plt.close(fig)


def _plot_two_row_scale_lodging_rates(
    bin_centers: np.ndarray,
    row_panels: list[ScaleRowPanels],
    model_colors: dict[str, tuple],
    output_path: Path,
    model_order: list[str],
    min_sample_fraction: float,
    x_limit: tuple[float, float] | None,
) -> None:
    """Plot two-row scale-sweep lodging probability figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)

    all_handles: list[plt.Line2D] = []
    all_labels: list[str] = []
    for row_index, row in enumerate(row_panels):
        prior_handles, prior_labels = _plot_panel(
            axes[row_index, 0],
            bin_centers,
            row.prior_panel,
            row.empirical_overlays,
            model_colors,
            min_sample_fraction,
            show_title=row_index == 0,
        )
        context_handles, context_labels = _plot_panel(
            axes[row_index, 1],
            bin_centers,
            row.context_panel,
            row.empirical_overlays,
            model_colors,
            min_sample_fraction,
            show_title=row_index == 0,
        )
        axes[row_index, 0].set_ylabel(row.label, fontweight="bold", labelpad=12)
        all_handles.extend([*prior_handles, *context_handles])
        all_labels.extend([*prior_labels, *context_labels])

    x_min, x_max = _resolve_axis_limits(bin_centers, x_limit)
    for ax in axes.ravel():
        ax.set_xlim(x_min, x_max)

    visible = (bin_centers >= x_min) & (bin_centers <= x_max)
    y_arrays = []
    for row in row_panels:
        y_arrays.extend(
            _collect_visible_rate_arrays(
                (row.prior_panel, row.context_panel),
                row.empirical_overlays,
                visible,
                min_sample_fraction,
            )
        )
    y_max = _calculate_y_axis_max(y_arrays)
    for ax in axes.ravel():
        ax.set_ylim(0.0, min(100.0, max(65.0, y_max * 110.0)))

    handles, ordered_labels, ncol = _build_ordered_legend(
        all_handles, all_labels, model_order
    )
    legend_layout = _two_row_scale_legend_layout(
        label_count=len(ordered_labels), column_count=ncol
    )

    fig.supylabel("Lodging %", x=0.055)
    # Keep the labels above the top grid line.
    add_panel_labels(axes.flat, offset_points=(6, 2))
    fig.tight_layout(rect=(0.07, legend_layout.bottom, 1.0, 1.0))
    fig.text(0.5, legend_layout.x_label_y, "Maximum height", ha="center", va="center")

    fig.legend(
        handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_layout.legend_y),
        ncol=ncol,
        frameon=False,
    )

    for fmt in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{fmt}"))
    plt.close(fig)


def calculate_ground_truth_x_limit(
    ground_truth: GroundTruth | None,
    bins: np.ndarray,
    bin_centers: np.ndarray,
    min_sample_fraction: float,
) -> tuple[float, float] | None:
    """Calculate x-axis limits from ground-truth max-height support."""
    if ground_truth is None:
        return None

    _, counts, _ = calculate_binned_rates(
        ground_truth.max_heights, ground_truth.labels, bins
    )
    masked_rates = mask_low_sample_rates(
        np.zeros_like(counts, dtype=float), counts, min_sample_fraction
    )
    valid = ~np.isnan(masked_rates)
    if not valid.any():
        msg = "No ground-truth bins pass --min-sample-fraction"
        raise ValueError(msg)

    x_limit = (0.6, float(bin_centers[valid].max()))
    logger.info(f"Using x-axis limit from ground-truth support: {x_limit}")
    return x_limit


def calculate_lodging_scale_probability(
    base_folders: list[str],
    model_names: list[str],
    lodging_weibull_scales: list[float],
    output_folder: str | None = None,
    bin_width: float = 0.05,
    relative_threshold: float = 0.2,
    absolute_threshold: float = 0.1,
    min_sample_fraction: float = 0.01,
    lodging_weibull_shape: float = 7.0,
    lodging_weibull_offset: float = 0.0,
    lodging_height_clamp: float = 2.0,
    empirical_gt_scales: list[str] | None = None,
    limit_x_to_ground_truth: bool = True,
    base_folders_row2: list[str] | None = None,
    row_labels: list[str] | None = None,
) -> None:
    """Calculate and visualize lodging probability for a lodging-scale sweep."""
    setup_style()

    if base_folders_row2 is None:
        if row_labels is not None:
            msg = "--row-labels requires --base-folders-row2"
            raise ValueError(msg)

        scale_models = _build_scale_models(
            base_folders, model_names, lodging_weibull_scales
        )
        requested_empirical = set(empirical_gt_scales or [])
        requested_empirical_names = _resolve_requested_empirical_names(
            scale_models, requested_empirical
        )
        shared_ground_truth = _load_shared_ground_truth(scale_models)
        (prior_prediction_rows, context_prediction_rows, context_ground_truth_rows) = (
            _load_prediction_rows_parallel(
                [scale_models],
                relative_threshold,
                absolute_threshold,
                shared_ground_truth,
            )
        )
        prior_predictions = prior_prediction_rows[0]
        context_predictions = context_prediction_rows[0]
        context_ground_truth = context_ground_truth_rows[0]

        bins, centers = calculate_height_bins(
            shared_ground_truth.max_heights, bin_width
        )

        prior_panel = _compute_prior_panel_data(
            scale_models=scale_models,
            predictions_by_name=prior_predictions,
            bins=bins,
            bin_centers=centers,
            lodging_weibull_shape=lodging_weibull_shape,
            lodging_weibull_offset=lodging_weibull_offset,
            lodging_height_clamp=lodging_height_clamp,
        )
        context_panel = _compute_context_panel_data(
            scale_models=scale_models,
            predictions_by_name=context_predictions,
            ground_truth_by_name=context_ground_truth,
            bins=bins,
            bin_centers=centers,
            lodging_weibull_shape=lodging_weibull_shape,
            lodging_weibull_offset=lodging_weibull_offset,
            lodging_height_clamp=lodging_height_clamp,
        )
        empirical_overlays = _compute_empirical_overlays(
            shared_ground_truth, requested_empirical_names, scale_models, bins
        )
        axis_limit = calculate_ground_truth_x_limit(
            shared_ground_truth if limit_x_to_ground_truth else None,
            bins,
            centers,
            min_sample_fraction,
        )

        output_path = (
            Path(output_folder)
            if output_folder
            else Path("paper/lodging_scale_probability")
        )
        output_path.mkdir(parents=True, exist_ok=True)
        _save_scale_histogram_csv(
            output_path, bins, centers, prior_panel, context_panel, empirical_overlays
        )
        _plot_scale_lodging_rates(
            bin_centers=centers,
            prior_panel=prior_panel,
            context_panel=context_panel,
            empirical_overlays=empirical_overlays,
            model_colors={
                scale_model.model.name: scale_model.model.color
                for scale_model in scale_models
            },
            output_path=output_path / "lodging_scale_probability",
            model_order=model_names,
            min_sample_fraction=min_sample_fraction,
            x_limit=axis_limit,
        )
        return

    if row_labels is None or len(row_labels) != 2:
        msg = (
            "--row-labels must be provided with exactly 2 values when using "
            "--base-folders-row2"
        )
        raise ValueError(msg)

    validate_two_row_input_lengths(
        base_folders, base_folders_row2, model_names, lodging_weibull_scales
    )
    requested_empirical = set(empirical_gt_scales or [])
    scale_model_rows = [
        _build_scale_models(base_folders, model_names, lodging_weibull_scales),
        _build_scale_models(base_folders_row2, model_names, lodging_weibull_scales),
    ]
    all_scale_models = [scale_model for row in scale_model_rows for scale_model in row]
    requested_empirical_name_rows = _resolve_requested_empirical_name_rows(
        scale_model_rows, requested_empirical
    )
    shared_ground_truth = _load_shared_ground_truth(all_scale_models)
    (prior_prediction_rows, context_prediction_rows, context_ground_truth_rows) = (
        _load_prediction_rows_parallel(
            scale_model_rows,
            relative_threshold,
            absolute_threshold,
            shared_ground_truth,
        )
    )

    bins, centers = calculate_height_bins(shared_ground_truth.max_heights, bin_width)

    row_panels = []
    for (
        label,
        scale_models,
        prior_predictions,
        context_predictions,
        context_ground_truth,
        requested_empirical_names,
    ) in zip(
        row_labels,
        scale_model_rows,
        prior_prediction_rows,
        context_prediction_rows,
        context_ground_truth_rows,
        requested_empirical_name_rows,
        strict=True,
    ):
        prior_panel = _compute_prior_panel_data(
            scale_models=scale_models,
            predictions_by_name=prior_predictions,
            bins=bins,
            bin_centers=centers,
            lodging_weibull_shape=lodging_weibull_shape,
            lodging_weibull_offset=lodging_weibull_offset,
            lodging_height_clamp=lodging_height_clamp,
        )
        context_panel = _compute_context_panel_data(
            scale_models=scale_models,
            predictions_by_name=context_predictions,
            ground_truth_by_name=context_ground_truth,
            bins=bins,
            bin_centers=centers,
            lodging_weibull_shape=lodging_weibull_shape,
            lodging_weibull_offset=lodging_weibull_offset,
            lodging_height_clamp=lodging_height_clamp,
        )
        empirical_overlays = _compute_empirical_overlays(
            shared_ground_truth, requested_empirical_names, scale_models, bins
        )
        row_panels.append(
            ScaleRowPanels(
                label=label,
                prior_panel=prior_panel,
                context_panel=context_panel,
                empirical_overlays=empirical_overlays,
            )
        )

    axis_limit = calculate_ground_truth_x_limit(
        shared_ground_truth if limit_x_to_ground_truth else None,
        bins,
        centers,
        min_sample_fraction,
    )
    output_path = (
        Path(output_folder)
        if output_folder
        else Path("paper/lodging_scale_probability")
    )
    output_path.mkdir(parents=True, exist_ok=True)
    _save_two_row_scale_histogram_csv(output_path, bins, centers, row_panels)
    _plot_two_row_scale_lodging_rates(
        bin_centers=centers,
        row_panels=row_panels,
        model_colors={
            scale_model.model.name: scale_model.model.color
            for scale_model in all_scale_models
        },
        output_path=output_path / "lodging_scale_probability",
        model_order=model_names,
        min_sample_fraction=min_sample_fraction,
        x_limit=axis_limit,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lodging scale sweep probability visualization"
    )
    parser.add_argument(
        "--base-folders",
        type=str,
        nargs="+",
        required=True,
        help="Base paths to model results (without /max_height or /no_context suffix)",
    )
    parser.add_argument(
        "--base-folders-row2",
        type=str,
        nargs="+",
        default=None,
        help="Optional base paths for row 2 in a two-row LNP/ANP layout",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Display labels for each lodging scale (must match number of base-folders)"
        ),
    )
    parser.add_argument(
        "--row-labels",
        type=str,
        nargs=2,
        default=None,
        help="Row labels for two-row mode, e.g. LNP ANP",
    )
    parser.add_argument(
        "--lodging-weibull-scales",
        type=float,
        nargs="+",
        required=True,
        help="Training lodging Weibull scale for each model",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (defaults to paper/lodging_scale_probability)",
    )
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--relative-threshold", type=float, default=0.2)
    parser.add_argument("--absolute-threshold", type=float, default=0.1)
    parser.add_argument("--min-sample-fraction", type=float, default=0.01)
    parser.add_argument("--lodging-weibull-shape", type=float, default=7.0)
    parser.add_argument("--lodging-weibull-offset", type=float, default=0.0)
    parser.add_argument("--lodging-height-clamp", type=float, default=2.0)
    parser.add_argument(
        "--empirical-gt-scales",
        type=str,
        nargs="*",
        default=None,
        help="Model labels for optional empirical ground-truth overlays",
    )
    parser.add_argument(
        "--no-limit-x-to-ground-truth",
        action="store_true",
        help="Do not limit the x-axis to supported ground-truth max-height bins",
    )
    args = parser.parse_args()

    calculate_lodging_scale_probability(
        base_folders=args.base_folders,
        model_names=args.model_names,
        lodging_weibull_scales=args.lodging_weibull_scales,
        output_folder=args.output_folder,
        bin_width=args.bin_width,
        relative_threshold=args.relative_threshold,
        absolute_threshold=args.absolute_threshold,
        min_sample_fraction=args.min_sample_fraction,
        lodging_weibull_shape=args.lodging_weibull_shape,
        lodging_weibull_offset=args.lodging_weibull_offset,
        lodging_height_clamp=args.lodging_height_clamp,
        empirical_gt_scales=args.empirical_gt_scales,
        limit_x_to_ground_truth=not args.no_limit_x_to_ground_truth,
        base_folders_row2=args.base_folders_row2,
        row_labels=args.row_labels,
    )
