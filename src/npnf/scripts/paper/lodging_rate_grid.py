"""Lodging rate and severity against maximum height for all models at several rates.

Rows are the model variants and columns the two model families, each without context
and with max-height context. Each training lodging rate has one colour: solid lines
are the models, dashed lines the ground truth of that rate (the simulator lodging
curve for the rate figure, the test set's mean drop of lodged plants for the severity
figure) and dotted lines the lodging rate measured in the test set.

Every rate has its own test set, so the ground truth is loaded per rate. No-context
panels bin by predicted max height, context panels by ground-truth max height. A model
that predicts one max height for every sample (a deterministic model without context)
is drawn as one point at that height, not at the center of its bin.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from matplotlib.lines import Line2D

from npnf.scripts.paper.data_types import PredictionStats
from npnf.scripts.paper.histogram_utils import (
    calculate_binned_rates,
    calculate_height_bins,
    mask_low_sample_rates,
)
from npnf.scripts.paper.lodging_absolute_drop import (
    DropGroundTruth,
    _load_ground_truth_drops,
    calculate_binned_drop_means,
)
from npnf.scripts.paper.lodging_plot_style import (
    ROW_ORDER,
    VARIANT_ORDER,
    semantic_model_groups,
)
from npnf.scripts.paper.lodging_probability import _load_predictions
from npnf.scripts.paper.lodging_scale_probability import analytic_lodging_probability
from npnf.scripts.paper.style import COLOR_LIST, setup_style


@dataclass(frozen=True)
class LodgingRate:
    """Model folders trained and tested at one lodging rate."""

    label: str
    lodging_weibull_scale: float
    folders: dict[str, Path]


@dataclass(frozen=True)
class Curve:
    """Values of one line against max height."""

    heights: np.ndarray
    values: np.ndarray


def group_folders(
    base_folders: list[str],
    model_names: list[str],
    lodging_weibull_scales: list[float],
    rate_labels: list[str],
) -> list[LodgingRate]:
    """Split the rate-major folder list into one folder per model and rate."""
    if len(rate_labels) != len(lodging_weibull_scales) or len(base_folders) != len(
        model_names
    ) * len(lodging_weibull_scales):
        msg = (
            "Expected one folder per model and rate in rate-major order and one label "
            f"per rate, got base_folders={len(base_folders)}, "
            f"model_names={len(model_names)}, "
            f"lodging_weibull_scales={len(lodging_weibull_scales)}, "
            f"rate_labels={len(rate_labels)}"
        )
        raise ValueError(msg)
    if semantic_model_groups(model_names) is None:
        msg = f"Unknown paper model names in {model_names}"
        raise ValueError(msg)
    model_count = len(model_names)
    return [
        LodgingRate(
            label=label,
            lodging_weibull_scale=scale,
            folders={
                name: Path(folder)
                for name, folder in zip(
                    model_names,
                    base_folders[index * model_count : (index + 1) * model_count],
                    strict=True,
                )
            },
        )
        for index, (label, scale) in enumerate(
            zip(rate_labels, lodging_weibull_scales, strict=True)
        )
    ]


def binned_curves(
    heights: np.ndarray,
    is_lodged: np.ndarray,
    drop_abs: np.ndarray,
    bins: np.ndarray,
    min_sample_fraction: float,
    min_lodged_count: int,
) -> tuple[Curve, Curve]:
    """Lodging percentage and mean drop of lodged samples against max height."""
    is_lodged = np.asarray(is_lodged, dtype=bool)
    if np.unique(heights).size == 1:
        severity = drop_abs[is_lodged].mean() if is_lodged.any() else np.nan
        return (
            Curve(heights[:1], np.array([100.0 * is_lodged.mean()])),
            Curve(heights[:1], np.array([severity])),
        )
    centers = (bins[:-1] + bins[1:]) / 2.0
    rates, counts, _ = calculate_binned_rates(heights, is_lodged, bins)
    percent = 100.0 * mask_low_sample_rates(rates, counts, min_sample_fraction)
    means, _, _ = calculate_binned_drop_means(
        heights, drop_abs, is_lodged, bins, min_lodged_count
    )
    return Curve(centers, percent), Curve(centers, means)


def load_rate(
    rate: LodgingRate, relative_threshold: float, absolute_threshold: float
) -> tuple[DropGroundTruth, dict[tuple[str, str], PredictionStats]]:
    """Load the rate's ground truth and the predictions of every model and method."""
    first_folder = next(iter(rate.folders.values()))
    tasks = {
        (name, method): folder / method
        for name, folder in rate.folders.items()
        for method in ("no_context", "max_height")
    }
    with ThreadPoolExecutor(max_workers=4) as executor:
        ground_truth_future = executor.submit(
            _load_ground_truth_drops, first_folder / "no_context"
        )
        futures = {
            key: executor.submit(
                _load_predictions, path, relative_threshold, absolute_threshold
            )
            for key, path in tasks.items()
        }
        ground_truth = ground_truth_future.result()
        predictions = {key: future.result() for key, future in futures.items()}
    for (name, method), stats in predictions.items():
        if stats.is_lodged.size != ground_truth.labels.size:
            msg = (
                f"{rate.label} {name} {method}: {stats.is_lodged.size} predictions "
                f"but {ground_truth.labels.size} ground-truth samples"
            )
            raise ValueError(msg)
    logger.info(
        "{}: {} samples, ground-truth lodging {:.2f} %",
        rate.label,
        ground_truth.labels.size,
        100.0 * ground_truth.labels.mean(),
    )
    return ground_truth, predictions


def plot_grid(
    curves: dict[tuple[str, str, str, str], Curve],
    model_names: list[str],
    rate_labels: list[str],
    quantity: str,
    y_label: str,
    x_limit: tuple[float, float],
    output_path: Path,
) -> None:
    """Plot one quantity for every model: variants in rows, family x method columns.

    `curves` is keyed by (quantity, line, method, rate label), where line is a model
    name, "ground truth" or "test set" and method is empty for ground-truth lines.
    """
    groups = semantic_model_groups(model_names)
    if groups is None:
        msg = f"Unknown paper model names in {model_names}"
        raise ValueError(msg)
    colors = dict(zip(rate_labels, COLOR_LIST, strict=False))
    methods = {"no_context": "no context", "max_height": "max-height context"}
    variants = [
        variant
        for variant in VARIANT_ORDER
        if any(variant in groups[family] for family in ROW_ORDER)
    ]
    figure, axes = plt.subplots(
        len(variants),
        2 * len(ROW_ORDER),
        figsize=(13, 2.8 * len(variants)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, variant in enumerate(variants):
        axes[row, 0].set_ylabel(variant, fontweight="bold")
        for family_index, family in enumerate(ROW_ORDER):
            for method_index, (method, method_label) in enumerate(methods.items()):
                axis = axes[row, 2 * family_index + method_index]
                if row == 0:
                    axis.set_title(f"{family}\n{method_label}")
                model = groups[family].get(variant)
                if model is None:
                    axis.set_visible(False)
                    continue
                axis.text(0.03, 0.93, model, transform=axis.transAxes, va="top")
                model_values = []
                for label in rate_labels:
                    for line, style in (("ground truth", "--"), ("test set", ":")):
                        reference = curves.get((quantity, line, "", label))
                        if reference is not None:
                            axis.plot(
                                reference.heights,
                                reference.values,
                                color=colors[label],
                                linestyle=style,
                                linewidth=1.2,
                            )
                    curve = curves[quantity, model, method, label]
                    single = curve.heights.size == 1
                    axis.plot(
                        curve.heights,
                        curve.values,
                        color=colors[label],
                        linewidth=1.6,
                        marker="o",
                        markersize=5 if single else 2.5,
                        markerfacecolor="none" if single else colors[label],
                    )
                    model_values.append(curve.values)
                if np.isnan(np.concatenate(model_values)).all():
                    axis.text(
                        0.5,
                        0.15,
                        "no lodged draws",
                        transform=axis.transAxes,
                        ha="center",
                        color="0.4",
                    )
                axis.set_xlim(*x_limit)
    handles = [
        Line2D([], [], color=colors[label], label=label) for label in rate_labels
    ]
    handles += [
        Line2D([], [], color="black", label="Model"),
        Line2D([], [], color="black", linestyle="--", label="Ground truth"),
    ]
    if any(key[0] == quantity and key[1] == "test set" for key in curves):
        handles.append(Line2D([], [], color="black", linestyle=":", label="Test set"))
    figure.supylabel(y_label, x=0.005)
    figure.supxlabel("Maximum height", y=0.035)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles),
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.045, 1.0, 1.0))
    for suffix in ("png", "pdf"):
        figure.savefig(output_path.with_suffix(f".{suffix}"))
    plt.close(figure)


def calculate_lodging_rate_grid(
    base_folders: list[str],
    model_names: list[str],
    lodging_weibull_scales: list[float],
    rate_labels: list[str],
    output_folder: str = "paper/lodging_rate_grid",
    bin_width: float = 0.05,
    relative_threshold: float = 0.2,
    absolute_threshold: float = 0.1,
    min_sample_fraction: float = 0.01,
    min_lodged_count: int = 25,
    lodging_weibull_shape: float = 7.0,
    lodging_weibull_offset: float = 0.0,
    lodging_height_clamp: float = 2.0,
) -> None:
    """Load every rate, bin all curves, and write both figures and their CSVs."""
    rates = group_folders(
        base_folders, model_names, lodging_weibull_scales, rate_labels
    )
    loaded = {
        rate.label: load_rate(rate, relative_threshold, absolute_threshold)
        for rate in rates
    }
    bins, centers = calculate_height_bins(
        np.concatenate([truth.max_heights for truth, _ in loaded.values()]), bin_width
    )

    curves: dict[tuple[str, str, str, str], Curve] = {}
    summary_rows = []
    for rate in rates:
        ground_truth, predictions = loaded[rate.label]
        curves["rate", "ground truth", "", rate.label] = Curve(
            centers,
            100.0
            * analytic_lodging_probability(
                centers,
                lodging_weibull_scale=rate.lodging_weibull_scale,
                lodging_weibull_shape=lodging_weibull_shape,
                lodging_weibull_offset=lodging_weibull_offset,
                lodging_height_clamp=lodging_height_clamp,
            ),
        )
        test_set_rate, test_set_severity = binned_curves(
            ground_truth.max_heights,
            ground_truth.labels,
            ground_truth.drop_abs,
            bins,
            min_sample_fraction,
            min_lodged_count,
        )
        curves["rate", "test set", "", rate.label] = test_set_rate
        curves["severity", "ground truth", "", rate.label] = test_set_severity
        sources = {("ground truth", ""): ground_truth.labels}
        drops = {("ground truth", ""): ground_truth.drop_abs}
        for (name, method), stats in predictions.items():
            heights = (
                stats.max_height if method == "no_context" else ground_truth.max_heights
            )
            rate_curve, severity_curve = binned_curves(
                heights,
                stats.is_lodged,
                stats.drop_abs,
                bins,
                min_sample_fraction,
                min_lodged_count,
            )
            curves["rate", name, method, rate.label] = rate_curve
            curves["severity", name, method, rate.label] = severity_curve
            sources[name, method] = np.asarray(stats.is_lodged, dtype=bool)
            drops[name, method] = stats.drop_abs
        for (line, method), is_lodged in sources.items():
            summary_rows.append(
                {
                    "rate": rate.label,
                    "lodging_weibull_scale": rate.lodging_weibull_scale,
                    "line": line,
                    "method": method,
                    "lodged_percent": 100.0 * float(is_lodged.mean()),
                    "mean_drop_lodged": float(drops[line, method][is_lodged].mean())
                    if is_lodged.any()
                    else None,
                }
            )

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary_rows).write_csv(output_path / "lodging_rate_grid_summary.csv")
    pl.DataFrame(
        [
            {
                "quantity": quantity,
                "rate": label,
                "line": line,
                "method": method,
                "max_height": float(height),
                "value": float(value),
            }
            for (quantity, line, method, label), curve in curves.items()
            for height, value in zip(curve.heights, curve.values, strict=True)
        ]
    ).write_csv(output_path / "lodging_rate_grid_curves.csv")

    support = np.zeros(centers.size, dtype=bool)
    for ground_truth, _ in loaded.values():
        counts, _ = np.histogram(ground_truth.max_heights, bins=bins)
        support |= counts >= min_sample_fraction * ground_truth.max_heights.size
    x_limit = (0.6, float(centers[support].max()) + bin_width)

    setup_style()
    for quantity, y_label in (
        ("rate", "Lodging %"),
        ("severity", "Mean absolute drop (m)"),
    ):
        plot_grid(
            curves,
            model_names,
            rate_labels,
            quantity,
            y_label,
            x_limit,
            output_path / f"lodging_{quantity}_grid",
        )
    logger.info("Wrote figures and CSVs to {}", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-folders",
        nargs="+",
        required=True,
        help="One folder per model and rate (without /no_context or /max_height), "
        "rate-major: all models of the first rate, then of the second, and so on.",
    )
    parser.add_argument("--model-names", nargs="+", required=True)
    parser.add_argument(
        "--lodging-weibull-scales", type=float, nargs="+", required=True
    )
    parser.add_argument("--rate-labels", nargs="+", required=True)
    parser.add_argument("--output-folder", default="paper/lodging_rate_grid")
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--relative-threshold", type=float, default=0.2)
    parser.add_argument("--absolute-threshold", type=float, default=0.1)
    parser.add_argument("--min-sample-fraction", type=float, default=0.01)
    parser.add_argument("--min-lodged-count", type=int, default=25)
    parser.add_argument("--lodging-weibull-shape", type=float, default=7.0)
    parser.add_argument("--lodging-weibull-offset", type=float, default=0.0)
    parser.add_argument("--lodging-height-clamp", type=float, default=2.0)
    args = parser.parse_args()

    calculate_lodging_rate_grid(
        base_folders=args.base_folders,
        model_names=args.model_names,
        lodging_weibull_scales=args.lodging_weibull_scales,
        rate_labels=args.rate_labels,
        output_folder=args.output_folder,
        bin_width=args.bin_width,
        relative_threshold=args.relative_threshold,
        absolute_threshold=args.absolute_threshold,
        min_sample_fraction=args.min_sample_fraction,
        min_lodged_count=args.min_lodged_count,
        lodging_weibull_shape=args.lodging_weibull_shape,
        lodging_weibull_offset=args.lodging_weibull_offset,
        lodging_height_clamp=args.lodging_height_clamp,
    )


if __name__ == "__main__":
    main()
