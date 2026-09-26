"""ANP without context: a jagged day marginal on FIP1 and a wrong lodging trend.

FIP1 panels
    The covariate-only prediction without context, averaged over the plots and
    draws of one split, one panel per ``--fip1-panels`` entry (comma-separated
    models). The references are two means of the measured plots on the model's day
    axis: the per-day mean over the plots measured on that day, which jumps because
    the measured site-years change from day to day, and the per-plot interpolated
    mean, which is smooth. A model that follows the per-day mean has learned the
    day marginal of the measurement schedule, not a growth curve.
Lodging panel
    Lodging rate against the predicted maximum height of the synthetic models
    without context, with the simulator's lodging probability and the test set's
    lodging rate.

``anp_prior_failure_fip1_summary.csv`` gives, per model, the roughness of the mean
(mean absolute second difference) and the correlation of its deviation from the
interpolated mean with the per-day mean's deviation from it.

Example:
    uv run python src/npnf/scripts/paper/anp_prior_failure.py \
        --fip1-results-dir $NPNF_RESULTS_DIR \
        --lodging-folders $NPNF_RESULTS_DIR/synth_test_plot_dataloaders/\
{LNP,ANP}-512k-training3m_set_mode_nested_noprior/6/checkpoint-3000000/\
test_plot/noenv_nogeno \
        --lodging-model-names LNP ANP --output-folder paper/anp_prior_failure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from matplotlib.lines import Line2D

from npnf.data.batch_loader import BatchLoader
from npnf.data.fip1_day_grid import METRIC_WINDOW
from npnf.metrics.blocked_scoring import load_prediction_day_axis
from npnf.scripts.metrics.fip1_observed_plots import (
    SPLIT_DATALOADERS,
    load_observed_plots,
    load_records,
)
from npnf.scripts.paper.fip1_prediction_figures import method_dir
from npnf.scripts.paper.histogram_utils import calculate_height_bins
from npnf.scripts.paper.lodging_absolute_drop import _load_ground_truth_drops
from npnf.scripts.paper.lodging_probability import _load_predictions
from npnf.scripts.paper.lodging_rate_grid import binned_curves
from npnf.scripts.paper.lodging_scale_probability import analytic_lodging_probability
from npnf.scripts.paper.style import COLOR_LIST, save_figure, setup_style
from npnf.utils import add_panel_labels


def observed_means(
    plots: list[tuple[np.ndarray, np.ndarray]], days: np.ndarray, min_plots: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Per-day and per-plot interpolated mean of the measured plots on ``days``.

    The per-day mean averages the plots measured on each day; the interpolated
    mean first interpolates every plot over its own measured span. A day covered
    by fewer than ``min_plots`` plots is NaN.
    """
    per_day_sum = np.zeros(days.size)
    per_day_count = np.zeros(days.size)
    interp_sum = np.zeros(days.size)
    interp_count = np.zeros(days.size)
    for plot_days, heights in plots:
        if plot_days.size < 2:
            continue
        measured = np.isin(days, plot_days)
        per_day_sum[measured] += heights[np.searchsorted(plot_days, days[measured])]
        per_day_count[measured] += 1
        inside = (days >= plot_days.min()) & (days <= plot_days.max())
        interp_sum[inside] += np.interp(days[inside], plot_days, heights)
        interp_count[inside] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        per_day = np.where(
            per_day_count >= min_plots, per_day_sum / per_day_count, np.nan
        )
        interp = np.where(interp_count >= min_plots, interp_sum / interp_count, np.nan)
    return per_day, interp


def roughness(values: np.ndarray) -> float:
    """Mean absolute second difference of a curve on a daily grid."""
    return float(np.nanmean(np.abs(np.diff(values, 2))))


def schedule_correlation(
    model_mean: np.ndarray, per_day: np.ndarray, interp: np.ndarray
) -> float:
    """Correlation of the model's and the per-day mean's deviation from ``interp``.

    Only the days where both observed means exist count.
    """
    both = ~np.isnan(per_day) & ~np.isnan(interp)
    return float(
        np.corrcoef(per_day[both] - interp[both], model_mean[both] - interp[both])[0, 1]
    )


def fip1_curves(
    results_dir: Path,
    split: str,
    conditioning: str,
    models: list[str],
    init: str,
    seed: int,
    datasets_offline_path: str | None,
) -> pl.DataFrame:
    """Observed means and every model's mean prediction without context, per day."""
    observed = load_observed_plots(split, datasets_offline_path)
    columns: dict[str, np.ndarray] = {}
    for model in models:
        loader = BatchLoader(
            method_dir(
                results_dir, split, conditioning, "no_context", model, init, seed
            )
        )
        axis = load_prediction_day_axis(loader).numpy().astype(int)
        records, _ = load_records(loader, "no_context", observed)
        window = (axis >= METRIC_WINDOW[0]) & (axis < METRIC_WINDOW[1])
        days = axis[window]
        if "day" in columns and not np.array_equal(columns["day"], days):
            msg = f"{model}: day axis differs from {models[0]}"
            raise ValueError(msg)
        columns["day"] = days
        columns[model] = (
            np.stack([record["grid"].numpy()[:, window] for record in records])
            .reshape(-1, window.sum())
            .mean(axis=0)
        )
    plots = []
    for plot in observed.values():
        inside = (plot["days"] >= METRIC_WINDOW[0]) & (plot["days"] < METRIC_WINDOW[1])
        plots.append((plot["days"][inside], plot["heights"][inside]))
    columns["observed per-day mean"], columns["observed interpolated mean"] = (
        observed_means(plots, columns["day"])
    )
    return pl.DataFrame(columns)


def fip1_summary(curves: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    per_day = curves["observed per-day mean"].to_numpy()
    interp = curves["observed interpolated mean"].to_numpy()
    rows: list[dict[str, str | float | None]] = [
        {
            "curve": name,
            "roughness": roughness(curves[name].to_numpy()),
            "schedule_correlation": schedule_correlation(
                curves[name].to_numpy(), per_day, interp
            ),
        }
        for name in models
    ]
    rows += [
        {"curve": name, "roughness": roughness(values), "schedule_correlation": None}
        for name, values in (
            ("observed per-day mean", per_day[~np.isnan(per_day)]),
            ("observed interpolated mean", interp[~np.isnan(interp)]),
        )
    ]
    return pl.DataFrame(rows)


def lodging_curves(
    folders: dict[str, Path],
    lodging_weibull_scale: float,
    bin_width: float = 0.05,
    relative_threshold: float = 0.2,
    absolute_threshold: float = 0.1,
    min_sample_fraction: float = 0.01,
    min_lodged_count: int = 25,
    lodging_weibull_shape: float = 7.0,
    lodging_weibull_offset: float = 0.0,
    lodging_height_clamp: float = 2.0,
) -> pl.DataFrame:
    """Lodging rate (%) against max height: models without context, ground truth."""
    ground_truth = _load_ground_truth_drops(next(iter(folders.values())) / "no_context")
    bins, centers = calculate_height_bins(ground_truth.max_heights, bin_width)
    rows = [
        {"line": "ground truth", "height": float(height), "rate": float(rate)}
        for height, rate in zip(
            centers,
            100.0
            * analytic_lodging_probability(
                centers,
                lodging_weibull_scale=lodging_weibull_scale,
                lodging_weibull_shape=lodging_weibull_shape,
                lodging_weibull_offset=lodging_weibull_offset,
                lodging_height_clamp=lodging_height_clamp,
            ),
            strict=True,
        )
    ]
    sources = [
        (
            "test set",
            ground_truth.max_heights,
            ground_truth.labels,
            ground_truth.drop_abs,
        )
    ]
    for name, folder in folders.items():
        stats = _load_predictions(
            folder / "no_context", relative_threshold, absolute_threshold
        )
        sources.append((name, stats.max_height, stats.is_lodged, stats.drop_abs))
    for name, heights, labels, drops in sources:
        curve, _ = binned_curves(
            heights, labels, drops, bins, min_sample_fraction, min_lodged_count
        )
        logger.info("{}: {:.2f} % lodged", name, 100.0 * np.mean(labels))
        rows += [
            {"line": name, "height": float(height), "rate": float(rate)}
            for height, rate in zip(curve.heights, curve.values, strict=True)
        ]
    return pl.DataFrame(rows)


def anp_prior_failure(
    fip1_curves_frame: pl.DataFrame,
    fip1_panels: list[list[str]],
    lodging: pl.DataFrame,
    lodging_model_names: list[str],
    lodging_label: str,
    output_folder: Path,
) -> None:
    setup_style()
    models = list(
        dict.fromkeys(
            lodging_model_names + [model for panel in fip1_panels for model in panel]
        )
    )
    colors = dict(zip(models, COLOR_LIST, strict=False))
    fig, axes = plt.subplots(
        1, len(fip1_panels) + 1, figsize=(4.2 * (len(fip1_panels) + 1), 3.8)
    )
    days = fip1_curves_frame["day"].to_numpy()
    for ax, panel in zip(axes[:-1], fip1_panels, strict=True):
        ax.plot(
            days,
            fip1_curves_frame["observed per-day mean"].to_numpy(),
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )
        ax.plot(
            days,
            fip1_curves_frame["observed interpolated mean"].to_numpy(),
            color="gray",
            linewidth=2.5,
            alpha=0.6,
        )
        for model in panel:
            ax.plot(
                days,
                fip1_curves_frame[model].to_numpy(),
                color=colors[model],
                linewidth=1.6,
                label=model,
            )
        ax.set_title(f"FIP1, {', '.join(panel)}")
        ax.set_xlabel("Day of year")
        ax.legend(loc="lower right", frameon=False)
    for ax in axes[1:-1]:
        ax.sharey(axes[0])
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Height")

    ax = axes[-1]
    for line, style, label in (
        ("ground truth", "--", "Simulator"),
        ("test set", ":", "Test set"),
    ):
        values = lodging.filter(pl.col("line") == line)
        ax.plot(
            values["height"],
            values["rate"],
            color="black",
            linestyle=style,
            label=label,
        )
    for model in lodging_model_names:
        values = lodging.filter(pl.col("line") == model)
        ax.plot(
            values["height"],
            values["rate"],
            color=colors[model],
            marker="o",
            markersize=2.5,
            label=model,
        )
    ax.set_title(f"Synthetic, {lodging_label} lodging")
    ax.set_xlabel("Maximum height")
    ax.set_ylabel("Lodging rate (%)")
    ax.legend(loc="center left", frameon=False)
    add_panel_labels(axes)

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="black",
                linewidth=0.8,
                alpha=0.6,
                label="Observed per-day mean",
            ),
            Line2D(
                [],
                [],
                color="gray",
                linewidth=2.5,
                alpha=0.6,
                label="Observed interpolated mean",
            ),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.35, -0.06),
        frameon=False,
    )
    save_figure(fig, output_folder / "anp_prior_failure")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fip1-results-dir", type=Path, required=True)
    parser.add_argument("--fip1-split", default="test_plot", choices=SPLIT_DATALOADERS)
    parser.add_argument("--fip1-conditioning", default="noenv_nogeno")
    parser.add_argument(
        "--fip1-panels",
        nargs="+",
        default=["ANP,ACNP", "LNP,CNP"],
        help="Comma-separated models of each FIP1 panel",
    )
    parser.add_argument(
        "--fip1-init", default="pretrained", choices=("pretrained", "scratch")
    )
    parser.add_argument("--fip1-seed", type=int, default=2)
    parser.add_argument("--datasets-offline-path", default=None)
    parser.add_argument(
        "--lodging-folders",
        type=Path,
        nargs="+",
        required=True,
        help="Conditioning folders of the synthetic runs, holding no_context",
    )
    parser.add_argument("--lodging-model-names", nargs="+", required=True)
    parser.add_argument("--lodging-weibull-scale", type=float, default=1.1)
    parser.add_argument("--lodging-label", default="19.7 %")
    parser.add_argument("--output-folder", type=Path, required=True)
    args = parser.parse_args()

    if len(args.lodging_folders) != len(args.lodging_model_names):
        msg = (
            f"Expected one lodging folder per model, got {len(args.lodging_folders)} "
            f"folders and {len(args.lodging_model_names)} names"
        )
        raise ValueError(msg)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    fip1_panels = [panel.split(",") for panel in args.fip1_panels]
    fip1_models = [model for panel in fip1_panels for model in panel]
    curves = fip1_curves(
        args.fip1_results_dir,
        args.fip1_split,
        args.fip1_conditioning,
        fip1_models,
        args.fip1_init,
        args.fip1_seed,
        args.datasets_offline_path,
    )
    curves.write_csv(args.output_folder / "anp_prior_failure_fip1_curves.csv")
    summary = fip1_summary(curves, fip1_models)
    summary.write_csv(args.output_folder / "anp_prior_failure_fip1_summary.csv")
    logger.info("FIP1 summary:\n{}", summary)
    lodging = lodging_curves(
        dict(zip(args.lodging_model_names, args.lodging_folders, strict=True)),
        args.lodging_weibull_scale,
    )
    lodging.write_csv(args.output_folder / "anp_prior_failure_lodging.csv")
    anp_prior_failure(
        curves,
        fip1_panels,
        lodging,
        args.lodging_model_names,
        args.lodging_label,
        args.output_folder,
    )


if __name__ == "__main__":
    main()
