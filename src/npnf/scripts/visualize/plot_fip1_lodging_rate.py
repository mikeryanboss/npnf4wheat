"""Compute lodging rates directly from the raw FIP1 height measurements.

This utility mirrors the probability analysis scripts but relies solely on the
recorded heights. Lodging is defined as a relative drop of at least 20 % between
the global maximum height and the minimum of the last `N` recorded measurements.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from npnf.data.datasets.fip1 import Fip1Facts, get_heights_dataset
from npnf.lodging import detect_lodging
from npnf.scripts.utils.outputs import write_csv

SPLIT_ALIASES: dict[str, list[str]] = {
    "test_all": list(Fip1Facts().test_splits),
    "fip_all": list(Fip1Facts().splits),
}


@dataclass
class SplitSummary:
    split: str
    total_samples: int
    lodged_samples: int
    lodging_rate: float
    mean_max_height: float
    median_max_height: float
    mean_drop_fraction: float
    median_drop_fraction: float
    skipped_samples: int


@dataclass
class LodgedExample:
    split: str
    index: int
    drop_rel: float
    drop_abs: float
    max_height: float
    tail_min: float
    heights: np.ndarray
    height_days: np.ndarray | None
    plot_uid: str | None
    genotype_id: str | None
    harvest_year: int | None


def _prepare_height_series(height_values: Any) -> np.ndarray | None:
    """Convert a dataset entry into a clean NumPy array."""
    if height_values is None:
        return None

    if isinstance(height_values, list):
        clean_values = [float(v) for v in height_values if v is not None]
        if not clean_values:
            return None
        values_np = np.asarray(clean_values, dtype=np.float32)
    elif isinstance(height_values, np.ndarray):
        values_np = height_values.astype(np.float32, copy=False)
    elif isinstance(height_values, torch.Tensor):
        values_np = height_values.numpy(force=True)
    else:
        values_np = np.asarray(height_values, dtype=np.float32)

    values_np = values_np[np.isfinite(values_np)]
    return values_np if values_np.size > 0 else None


def _prepare_day_series(height_days: Any) -> np.ndarray | None:
    if height_days is None:
        return None

    if isinstance(height_days, list):
        cleaned = [(float(day) if day is not None else np.nan) for day in height_days]
        values_np = np.asarray(cleaned, dtype=np.float32)
    elif isinstance(height_days, np.ndarray):
        values_np = height_days.astype(np.float32, copy=False)
    elif isinstance(height_days, torch.Tensor):
        values_np = height_days.numpy(force=True).astype(np.float32, copy=False)
    else:
        values_np = np.asarray(height_days, dtype=np.float32)

    return values_np if values_np.size > 0 else None


def _prepare_height_and_day_series(
    height_values: Any, height_days: Any
) -> tuple[np.ndarray | None, np.ndarray | None]:
    heights = _prepare_height_series(height_values)
    if heights is None:
        return None, None

    days = _prepare_day_series(height_days)
    if days is not None:
        min_len = min(len(heights), len(days))
        heights = heights[:min_len]
        days = days[:min_len]

    return heights, days


def _compute_sample_stats(
    height_series: np.ndarray,
    last_n: int,
    relative_threshold: float,
    absolute_threshold: float,
    eps: float,
) -> dict[str, float | bool]:
    if height_series.size == 0:
        return {}

    stats = detect_lodging(
        height_series,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
        tail_window=last_n,
        eps=eps,
    )

    return {
        "max_height": stats.max_height,
        "tail_min": stats.final_height,
        "drop_abs": stats.drop_abs,
        "drop_rel": stats.drop_rel,
        "is_lodged": stats.is_lodged,
    }


def _compute_histogram(
    heights: list[float], lodged_mask: list[bool], bin_width: float
) -> dict[str, np.ndarray] | None:
    if not heights or bin_width <= 0:
        return None

    heights_np = np.asarray(heights, dtype=np.float32)
    lodged_np = np.asarray(lodged_mask, dtype=bool)

    min_height = float(np.min(heights_np))
    max_height = float(np.max(heights_np))
    if np.isclose(max_height, min_height):
        bins = np.array([min_height, max_height + bin_width], dtype=np.float32)
    else:
        num_bins = int(np.ceil((max_height - min_height) / bin_width))
        bins = np.linspace(min_height, min_height + bin_width * num_bins, num_bins + 1)

    lodged_counts, _ = np.histogram(heights_np[lodged_np], bins=bins)
    total_counts, _ = np.histogram(heights_np, bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        lodging_rates = np.divide(
            lodged_counts,
            total_counts,
            out=np.zeros_like(lodged_counts, dtype=float),
            where=total_counts > 0,
        )

    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    return {
        "bins_left": bins[:-1],
        "bins_right": bins[1:],
        "bin_centers": bin_centers,
        "lodged_counts": lodged_counts,
        "total_counts": total_counts,
        "lodging_rates": lodging_rates,
    }


def _resolve_dataset_split(split: str | list[str]) -> str | list[str]:
    if isinstance(split, str):
        return SPLIT_ALIASES.get(split, split)
    return split


def _analyze_split(
    split: str,
    datasets_offline_path: str | None,
    last_n: int,
    relative_threshold: float,
    absolute_threshold: float,
    eps: float,
    bin_width: float,
) -> tuple[SplitSummary, dict[str, np.ndarray] | None, list[LodgedExample]]:
    dataset = get_heights_dataset(
        split=_resolve_dataset_split(split),
        standardize=False,
        datasets_offline_path=datasets_offline_path,
    )

    max_heights: list[float] = []
    drop_fracs: list[float] = []
    lodged: list[bool] = []
    lodged_examples: list[LodgedExample] = []
    skipped = 0

    iterator = range(len(dataset))
    for index in tqdm(iterator, desc=f"Analyzing {split}", leave=False):
        sample = dataset[index]
        heights, height_days = _prepare_height_and_day_series(
            sample.get("height_values"), sample.get("height_days")
        )
        if heights is None:
            skipped += 1
            continue
        stats = _compute_sample_stats(
            heights,
            last_n=last_n,
            relative_threshold=relative_threshold,
            absolute_threshold=absolute_threshold,
            eps=eps,
        )
        if not stats:
            skipped += 1
            continue
        max_heights.append(float(stats["max_height"]))
        drop_fracs.append(float(stats["drop_rel"]))
        lodged.append(bool(stats["is_lodged"]))
        if stats["is_lodged"]:
            example = LodgedExample(
                split=split,
                index=index,
                drop_rel=float(stats["drop_rel"]),
                drop_abs=float(stats["drop_abs"]),
                max_height=float(stats["max_height"]),
                tail_min=float(stats["tail_min"]),
                heights=heights.copy(),
                height_days=None if height_days is None else height_days.copy(),
                plot_uid=(
                    str(sample["plot_uid"])
                    if sample.get("plot_uid") is not None
                    else None
                ),
                genotype_id=(
                    str(sample["genotype_id"])
                    if sample.get("genotype_id") is not None
                    else None
                ),
                harvest_year=(
                    int(sample["harvest_year"])
                    if sample.get("harvest_year") is not None
                    else None
                ),
            )
            lodged_examples.append(example)

    total = len(max_heights)
    lodged_count = int(np.sum(lodged)) if lodged else 0
    lodging_rate = lodged_count / total if total else 0.0

    summary = SplitSummary(
        split=split,
        total_samples=total,
        lodged_samples=lodged_count,
        lodging_rate=lodging_rate,
        mean_max_height=float(np.mean(max_heights)) if total else float("nan"),
        median_max_height=float(np.median(max_heights)) if total else float("nan"),
        mean_drop_fraction=float(np.mean(drop_fracs)) if drop_fracs else float("nan"),
        median_drop_fraction=float(np.median(drop_fracs))
        if drop_fracs
        else float("nan"),
        skipped_samples=skipped,
    )

    histogram = _compute_histogram(max_heights, lodged, bin_width=bin_width)
    return summary, histogram, lodged_examples


def _write_summary_csv(path: Path, summaries: list[SplitSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "total_samples",
        "lodged_samples",
        "lodging_rate",
        "mean_max_height",
        "median_max_height",
        "mean_drop_fraction",
        "median_drop_fraction",
        "skipped_samples",
    ]
    write_csv(path, (summary.__dict__ for summary in summaries), fieldnames)


def _write_hist_csv(path: Path, histogram: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bin_left",
                "bin_right",
                "bin_center",
                "total_count",
                "lodged_count",
                "lodging_rate",
            ]
        )
        for i in range(len(histogram["bin_centers"])):
            writer.writerow(
                [
                    float(histogram["bins_left"][i]),
                    float(histogram["bins_right"][i]),
                    float(histogram["bin_centers"][i]),
                    int(histogram["total_counts"][i]),
                    int(histogram["lodged_counts"][i]),
                    float(histogram["lodging_rates"][i]),
                ]
            )


def _plot_histogram(
    split: str, histogram: dict[str, np.ndarray], output_path: Path
) -> None:
    widths = histogram["bins_right"] - histogram["bins_left"]
    fig, ax_rate = plt.subplots(figsize=(8, 4))
    rate_bars = ax_rate.bar(
        histogram["bin_centers"],
        histogram["lodging_rates"],
        width=widths * 0.95,
        color="#1f77b4",
        alpha=0.85,
        align="center",
        label="Lodging rate",
    )

    ax_rate.set_xlabel("Maximum height (m)")
    ax_rate.set_ylabel("Lodging rate")
    ax_rate.set_title(f"Lodging rate by height bin ({split})")
    if len(histogram["lodging_rates"]) > 0:
        max_rate = float(np.nanmax(histogram["lodging_rates"]))
    else:
        max_rate = 0.0
    ylim_max = max(0.05, min(1.0, max_rate * 1.2 if max_rate > 0 else 0.05))
    ax_rate.set_ylim(0.0, ylim_max)
    ax_rate.grid(axis="y", alpha=0.3, linestyle="--")

    ax_counts = ax_rate.twinx()
    lodged_counts = histogram["lodged_counts"]
    non_lodged_counts = np.maximum(
        histogram["total_counts"] - histogram["lodged_counts"], 0
    )
    count_width = widths * 0.45
    counts_nonlodged = ax_counts.bar(
        histogram["bin_centers"],
        non_lodged_counts,
        width=count_width,
        color="#c7d3eb",
        alpha=0.6,
        align="center",
        label="Non-lodged count",
    )
    counts_lodged = ax_counts.bar(
        histogram["bin_centers"],
        lodged_counts,
        width=count_width,
        color="#d62728",
        alpha=0.65,
        align="center",
        bottom=non_lodged_counts,
        label="Lodged count",
    )
    max_count = (
        float(np.max(histogram["total_counts"]))
        if histogram["total_counts"].size > 0
        else 0.0
    )
    ax_counts.set_ylabel("Samples per height bin")
    ax_counts.set_ylim(0.0, max(max_count * 1.2, 1.0))

    handles = [rate_bars, counts_nonlodged, counts_lodged]
    labels = [str(h.get_label()) for h in handles]
    ax_rate.legend(handles, labels, loc="upper right", fontsize="small")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_lodged_timeseries(
    split: str,
    examples: list[LodgedExample],
    output_dir: Path,
    count: int,
    relative_threshold: float,
    sort_mode: str,
) -> None:
    if count <= 0 or not examples:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if sort_mode == "height":
        sorted_examples = sorted(examples, key=lambda ex: ex.max_height)
    else:
        sorted_examples = sorted(examples, key=lambda ex: ex.drop_rel)
    num_examples = min(count, len(sorted_examples))

    for rank, example in enumerate(sorted_examples[:num_examples], start=1):
        heights = example.heights
        days = example.height_days
        x_label = "Measurement index"
        if days is not None:
            mask = ~np.isnan(days)
            if mask.any():
                x_values = days[mask]
                y_values = heights[mask]
                x_label = "Days since Sep 1 (previous year)"
            else:
                x_values = np.arange(len(heights))
                y_values = heights
        else:
            x_values = np.arange(len(heights))
            y_values = heights

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x_values, y_values, marker="o", linewidth=1.5, color="#1f77b4")
        ax.axhline(
            example.max_height,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.0,
            label=f"Peak {example.max_height:.2f} m",
        )
        ax.axhline(
            example.tail_min,
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label=f"Tail min {example.tail_min:.2f} m",
        )
        ax.axhline(
            example.max_height * (1 - relative_threshold),
            color="#ff7f0e",
            linestyle="-.",
            linewidth=1.0,
            label=f"Threshold {(1 - relative_threshold):.0%} of peak",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Height (m)")
        metadata = []
        if example.plot_uid:
            metadata.append(f"plot={example.plot_uid}")
        if example.genotype_id:
            metadata.append(f"geno={example.genotype_id}")
        if example.harvest_year is not None:
            metadata.append(f"year={example.harvest_year}")
        meta_str = ", ".join(metadata)
        title = (
            f"{split} index={example.index} "
            f"drop={example.drop_rel:.2%} (abs {example.drop_abs:.2f} m)"
        )
        if meta_str:
            title = f"{title}\n{meta_str}"
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize="small", frameon=False)
        ax.grid(alpha=0.2, linestyle="--")
        fig.tight_layout()
        filename = f"{rank:02d}_index{example.index}_drop_{example.drop_rel:.3f}.png"
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate lodging rate on FIP1 heights using trailing-window definition."
        )
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(Fip1Facts().splits),
        help="Dataset splits to analyze.",
    )
    parser.add_argument(
        "--datasets-offline-path",
        type=str,
        default=None,
        help="Optional offline datasets directory.",
    )
    parser.add_argument(
        "--relative-drop-threshold",
        type=float,
        default=0.2,
        help="Minimum relative drop to mark a sample as lodged.",
    )
    parser.add_argument(
        "--absolute-drop-threshold",
        type=float,
        default=0.1,
        help="Minimum absolute drop (in meters) to mark a sample as lodged.",
    )
    parser.add_argument(
        "--last-n-measurements",
        type=int,
        default=5,
        help=("Use the minimum over the last N measurements for the drop computation."),
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.05,
        help="Bin width (in meters) for the optional histogram output.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Numerical stability term when computing relative drops.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, summaries and histograms are written to this directory.",
    )
    parser.add_argument(
        "--plot-split",
        type=str,
        default=None,
        help="If provided, save a lodging-rate plot for this split.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Path to store the lodging-rate plot (defaults to output_dir).",
    )
    parser.add_argument(
        "--lodged-timeseries-count",
        type=int,
        default=0,
        help=(
            "Number of lodged samples (closest to threshold) to plot as time-series."
        ),
    )
    parser.add_argument(
        "--lodged-timeseries-split",
        type=str,
        default=None,
        help=(
            "Split to sample lodged time-series from "
            "(defaults to plot split or the first requested split)."
        ),
    )
    parser.add_argument(
        "--lodged-timeseries-output",
        type=Path,
        default=None,
        help=("Directory to save lodged time-series plots (defaults to output_dir)."),
    )
    parser.add_argument(
        "--lodged-timeseries-sort",
        choices=("threshold", "height"),
        default="threshold",
        help=(
            "Selection rule for lodged time-series: 'threshold' keeps samples "
            "closest to the drop threshold, while 'height' picks the smallest "
            "max-height lodged samples."
        ),
    )
    return parser.parse_args()


def main() -> None:  # noqa: PLR0912
    args = parse_args()
    summaries: list[SplitSummary] = []
    histograms: dict[str, dict[str, np.ndarray]] = {}
    lodged_examples_by_split: dict[str, list[LodgedExample]] = {}

    splits_to_run = list(dict.fromkeys(args.splits))
    if args.plot_split is not None and args.plot_split not in splits_to_run:
        splits_to_run.append(args.plot_split)

    for split in splits_to_run:
        summary, histogram, lodged_examples = _analyze_split(
            split=split,
            datasets_offline_path=args.datasets_offline_path,
            last_n=args.last_n_measurements,
            relative_threshold=args.relative_drop_threshold,
            absolute_threshold=args.absolute_drop_threshold,
            eps=args.eps,
            bin_width=args.bin_width,
        )
        summaries.append(summary)
        if histogram is not None:
            histograms[summary.split] = histogram
        lodged_examples_by_split[summary.split] = lodged_examples

        logger.info(
            f"[{summary.split}] total={summary.total_samples} "
            f"lodged={summary.lodged_samples} "
            f"rate={summary.lodging_rate:.2%} "
            f"mean_max={summary.mean_max_height:.3f} "
            f"mean_drop={summary.mean_drop_fraction:.2%} "
            f"skipped={summary.skipped_samples}"
        )

    if summaries:
        overall_total = sum(s.total_samples for s in summaries)
        overall_lodged = sum(s.lodged_samples for s in summaries)
        overall_rate = overall_lodged / overall_total if overall_total else 0.0
        logger.info(
            f"[overall] total={overall_total} lodged={overall_lodged} "
            f"rate={overall_rate:.2%}"
        )

    if args.output_dir is not None:
        # Create subfolders for organized output
        data_path = args.output_dir / "data"
        plots_path = args.output_dir / "plots"
        data_path.mkdir(parents=True, exist_ok=True)
        plots_path.mkdir(parents=True, exist_ok=True)

        summary_path = data_path / "lodging_summary.csv"
        _write_summary_csv(summary_path, summaries)
        for split, histogram in histograms.items():
            hist_path = data_path / f"lodging_hist_{split}.csv"
            _write_hist_csv(hist_path, histogram)
        logger.info("Saved summary artifacts to {}", args.output_dir.resolve())

    if args.plot_split is not None:
        histogram = histograms.get(args.plot_split)
        if histogram is None:
            logger.warning(
                "No histogram available for plot split '{}'. Skipping plot.",
                args.plot_split,
            )
        else:
            if args.plot_output is not None:
                plot_path = args.plot_output
            elif args.output_dir is not None:
                plot_path = (
                    args.output_dir / "plots" / f"lodging_rate_{args.plot_split}.png"
                )
            else:
                plot_path = Path(f"lodging_rate_{args.plot_split}.png")
            _plot_histogram(args.plot_split, histogram, plot_path)
            logger.info(
                "Saved lodging-rate plot for {} to {}",
                args.plot_split,
                plot_path.resolve(),
            )

    if args.lodged_timeseries_count > 0:
        target_split = (
            args.lodged_timeseries_split
            or args.plot_split
            or (args.splits[0] if args.splits else None)
        )
        if target_split is None:
            logger.warning(
                "Unable to plot lodged time-series because no split was specified."
            )
        else:
            examples = lodged_examples_by_split.get(target_split)
            if not examples:
                logger.warning(
                    "No lodged samples recorded for split '{}' to plot.", target_split
                )
            else:
                if args.lodged_timeseries_output is not None:
                    output_dir = args.lodged_timeseries_output
                elif args.output_dir is not None:
                    output_dir = args.output_dir / "plots" / "timeseries" / target_split
                else:
                    output_dir = Path(f"lodged_timeseries_{target_split}")
                _plot_lodged_timeseries(
                    target_split,
                    examples,
                    output_dir,
                    args.lodged_timeseries_count,
                    args.relative_drop_threshold,
                    args.lodged_timeseries_sort,
                )
                logger.info(
                    "Saved lodged time-series plots for {} to {}",
                    target_split,
                    output_dir.resolve(),
                )


if __name__ == "__main__":
    main()
