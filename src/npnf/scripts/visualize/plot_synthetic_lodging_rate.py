"""Summarize lodging behaviour on the synthetic dataset."""

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
from hydra_zen import instantiate
from loguru import logger
from tqdm.auto import tqdm

from npnf.data.configs.datasets.synthetic import TrainConfig_64k, ValConfig
from npnf.data.synthetic.lodging import lodging_probability
from npnf.scripts.utils.outputs import write_csv

DEFAULT_SPLITS = ["validation"]

SPLIT_CONFIGS: dict[str, Any] = {"train": TrainConfig_64k, "validation": ValConfig}

SPLIT_ALIASES = {
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "dev": "validation",
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
    year_site_uid: str | None
    harvest_year: int | None


def _resolve_split(split: str) -> str:
    canonical = split.lower()
    return SPLIT_ALIASES.get(canonical, canonical)


def _extract_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return int(flat[0]) if flat.size > 0 else None
    return int(value)


def _prepare_height_series(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        arr = values.numpy(force=True)
    else:
        arr = np.asarray(values)
    arr = arr.astype(np.float32, copy=False)
    arr = arr[np.isfinite(arr)]
    return arr if arr.size > 0 else None


def _prepare_day_series(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        arr = values.numpy(force=True)
    else:
        arr = np.asarray(values)
    arr = arr.astype(np.float32, copy=False)
    return arr if arr.size > 0 else None


def _prepare_mask_series(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        arr = values.numpy(force=True).astype(bool, copy=False)
    else:
        arr = np.asarray(values).astype(bool, copy=False)
    return arr if arr.size > 0 else None


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


def _instantiate_dataset(
    split: str,
    weather_data_path: str | None,
    num_samples: int | None,
    num_points: int | None,
):
    canonical = _resolve_split(split)
    if canonical not in SPLIT_CONFIGS:
        msg = f"Unknown split '{split}'"
        raise ValueError(msg)
    dataset_cfg = SPLIT_CONFIGS[canonical]
    dataset_kwargs: dict[str, Any] = {}
    if weather_data_path is not None:
        dataset_kwargs["dataset_temperature"] = {"path": weather_data_path}
    if num_samples is not None:
        dataset_kwargs["num_samples"] = num_samples
    if num_points is not None:
        dataset_kwargs["num_points"] = num_points
    return instantiate(dataset_cfg, **dataset_kwargs)


def _analyze_split(
    split: str, dataset, bin_width: float, eps: float
) -> tuple[SplitSummary, dict[str, np.ndarray] | None, list[LodgedExample]]:
    max_heights: list[float] = []
    drop_fracs: list[float] = []
    lodged: list[bool] = []
    lodged_examples: list[LodgedExample] = []
    skipped = 0

    iterator = range(len(dataset))
    for index in tqdm(iterator, desc=f"Analyzing {split}", leave=False):
        sample = dataset[index]
        heights_full = _prepare_height_series(sample.get("height_values_all_nonoise"))
        heights_obs, height_days = _prepare_height_and_day_series(
            sample.get("height_values"), sample.get("height_days")
        )
        heights = heights_full if heights_full is not None else heights_obs
        if heights is None:
            skipped += 1
            continue
        max_height = float(np.max(heights))
        lodged_mask = _prepare_mask_series(sample.get("height_lodged_mask_all"))
        if lodged_mask is not None and heights_full is not None and lodged_mask.any():
            tail_min = float(np.min(heights_full[lodged_mask]))
        else:
            tail_min = float(heights[-1])
        drop_abs = max_height - tail_min
        drop_rel = drop_abs / max(max_height, eps)
        has_lodged = sample.get("has_lodged")
        if isinstance(has_lodged, torch.Tensor):
            is_lodged = bool(has_lodged.item())
        else:
            is_lodged = bool(has_lodged)
        max_heights.append(max_height)
        drop_fracs.append(drop_rel)
        lodged.append(is_lodged)
        if is_lodged and heights is not None:
            heights_for_plot = heights_obs if heights_obs is not None else heights
            example = LodgedExample(
                split=split,
                index=index,
                drop_rel=drop_rel,
                drop_abs=drop_abs,
                max_height=max_height,
                tail_min=tail_min,
                heights=heights_for_plot.copy(),
                height_days=None if height_days is None else height_days.copy(),
                year_site_uid=(
                    str(sample.get("yearsite_uid"))
                    if sample.get("yearsite_uid") is not None
                    else None
                ),
                harvest_year=_extract_int(sample.get("harvest_year")),
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
        if example.year_site_uid:
            metadata.append(f"site={example.year_site_uid}")
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


def _plot_logistic_curve(
    height_clamp: float, scale: float, inflection: float, output_path: Path, split: str
) -> None:
    clamp = max(height_clamp, 1e-6)
    heights = np.linspace(0.0, clamp, 512)
    proportions = heights / clamp
    logits = scale * (proportions - inflection)
    probs = 1.0 / (1.0 + np.exp(-logits))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        heights, probs, label="Logistic probability", color="#9467bd", linewidth=2.0
    )
    ax.axvline(inflection * clamp, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Maximum height (m)")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Logistic lodging curve ({split})\n"
        f"scale={scale:.2f}, inflection={inflection:.2f}, clamp={clamp:.2f} m"
    )
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_weibull_curve(
    height_clamp: float,
    shape: float,
    scale_m: float,
    offset_m: float,
    output_path: Path,
    split: str,
) -> None:
    clamp = max(height_clamp, 1e-6)
    heights = np.linspace(0.0, clamp, 512)
    offset = np.clip(offset_m, 0.0, clamp)
    probs = lodging_probability(
        torch.as_tensor(heights),
        lodging_height_clamp=clamp,
        lodging_weibull_shape=max(shape, 1e-6),
        lodging_weibull_scale=scale_m,
        lodging_weibull_offset=offset,
    ).numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        heights,
        probs,
        label="Weibull (shifted) probability",
        color="#2ca02c",
        linewidth=2.0,
    )
    ax.axvline(offset, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Maximum height (m)")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Weibull lodging curve ({split})\n"
        f"shape={shape:.2f}, scale={scale_m:.2f} m, offset={offset:.2f} m "
        f"(clamp={clamp:.2f} m)"
    )
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate lodging statistics on the synthetic dataset "
            "using the dataset-provided lodging labels."
        )
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Dataset splits to analyze (train, validation, test).",
    )
    parser.add_argument(
        "--weather-data-path",
        type=str,
        default=None,
        help="Optional override for the weather CSV used to build the dataset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override the number of samples generated per split.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Override the number of height observations per sample.",
    )
    parser.add_argument(
        "--relative-drop-threshold",
        type=float,
        default=0.2,
        help="Reference threshold shown on lodged time-series plots.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.05,
        help="Bin width (in meters) for histogram summaries.",
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
        help="Number of lodged samples to plot as time-series.",
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
        help="Directory to save lodged time-series plots (defaults to output_dir).",
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
    parser.add_argument(
        "--plot-logistic",
        action="store_true",
        help="If set, save a standalone logistic probability curve.",
    )
    parser.add_argument(
        "--logistic-split",
        type=str,
        default=None,
        help="Split providing the height clamp for the logistic curve.",
    )
    parser.add_argument(
        "--logistic-scale",
        type=float,
        default=10.0,
        help="Slope parameter for the logistic curve (higher = sharper transition).",
    )
    parser.add_argument(
        "--logistic-inflection",
        type=float,
        default=0.5,
        help="Height proportion (0-1) where logistic probability is 0.5.",
    )
    parser.add_argument(
        "--logistic-height-clamp",
        type=float,
        default=None,
        help=(
            "Override the maximum height (m) used for the logistic axis "
            "(defaults to dataset clamp)."
        ),
    )
    parser.add_argument(
        "--plot-logistic-output",
        type=Path,
        default=None,
        help="Path for the logistic plot (defaults to output_dir).",
    )
    parser.add_argument(
        "--plot-weibull",
        action="store_true",
        help="If set, save a shifted Weibull lodging probability curve.",
    )
    parser.add_argument(
        "--weibull-split",
        type=str,
        default=None,
        help="Split providing the height clamp for the Weibull curve.",
    )
    parser.add_argument(
        "--weibull-shape",
        type=float,
        default=None,
        help=(
            "Shape parameter (>1 accelerates with height). Defaults to dataset value."
        ),
    )
    parser.add_argument(
        "--weibull-scale",
        type=float,
        default=None,
        help=(
            "Scale parameter (meters) controlling the width "
            "(defaults to dataset value)."
        ),
    )
    parser.add_argument(
        "--weibull-offset",
        type=float,
        default=None,
        help=(
            "Offset in meters before the Weibull rise begins "
            "(defaults to dataset value)."
        ),
    )
    parser.add_argument(
        "--weibull-height-clamp",
        type=float,
        default=None,
        help=(
            "Override the maximum height (m) used for the Weibull axis "
            "(defaults to dataset clamp)."
        ),
    )
    parser.add_argument(
        "--plot-weibull-output",
        type=Path,
        default=None,
        help="Path for the Weibull plot (defaults to output_dir).",
    )
    return parser.parse_args()


def main() -> None:  # noqa: PLR0912
    args = parse_args()
    summaries: list[SplitSummary] = []
    histograms: dict[str, dict[str, np.ndarray]] = {}
    lodged_examples_by_split: dict[str, list[LodgedExample]] = {}
    datasets_by_split: dict[str, Any] = {}

    splits_to_run: list[str] = []
    for split in args.splits:
        canonical = _resolve_split(split)
        if canonical not in SPLIT_CONFIGS:
            msg = f"Unsupported split '{split}'"
            raise ValueError(msg)
        if canonical not in splits_to_run:
            splits_to_run.append(canonical)
    plot_split = None
    if args.plot_split is not None:
        canonical_plot = _resolve_split(args.plot_split)
        if canonical_plot not in SPLIT_CONFIGS:
            msg = f"Unsupported plot split '{args.plot_split}'"
            raise ValueError(msg)
        if canonical_plot not in splits_to_run:
            splits_to_run.append(canonical_plot)
        plot_split = canonical_plot

    for split in splits_to_run:
        dataset = _instantiate_dataset(
            split,
            weather_data_path=args.weather_data_path,
            num_samples=args.num_samples,
            num_points=args.num_points,
        )
        datasets_by_split[split] = dataset
        summary, histogram, lodged_examples = _analyze_split(
            split=split, dataset=dataset, bin_width=args.bin_width, eps=args.eps
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
            f"[overall] total={overall_total} "
            f"lodged={overall_lodged} rate={overall_rate:.2%}"
        )

    if args.output_dir is not None:
        # Create subfolders for organized output
        data_path = args.output_dir / "data"
        plots_path = args.output_dir / "plots"
        data_path.mkdir(parents=True, exist_ok=True)
        plots_path.mkdir(parents=True, exist_ok=True)

        summary_path = data_path / "synthetic_lodging_summary.csv"
        _write_summary_csv(summary_path, summaries)
        for split, histogram in histograms.items():
            hist_path = data_path / f"synthetic_lodging_hist_{split}.csv"
            _write_hist_csv(hist_path, histogram)
        logger.info("Saved summary artifacts to {}", args.output_dir.resolve())

    if plot_split is not None:
        histogram = histograms.get(plot_split)
        if histogram is None:
            logger.warning(
                "No histogram available for plot split '{}'. Skipping plot.", plot_split
            )
        else:
            if args.plot_output is not None:
                plot_path = args.plot_output
            elif args.output_dir is not None:
                plots_dir = args.output_dir / "plots"
                plot_path = plots_dir / f"synthetic_lodging_rate_{plot_split}.png"
            else:
                plot_path = Path(f"synthetic_lodging_rate_{plot_split}.png")
            _plot_histogram(plot_split, histogram, plot_path)
            logger.info(
                "Saved lodging-rate plot for {} to {}", plot_split, plot_path.resolve()
            )

    if args.lodged_timeseries_count > 0:
        target_split = (
            _resolve_split(args.lodged_timeseries_split)
            if args.lodged_timeseries_split
            else (
                plot_split
                if plot_split is not None
                else (splits_to_run[0] if splits_to_run else None)
            )
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

    if args.plot_logistic:
        target_split = (
            _resolve_split(args.logistic_split)
            if args.logistic_split is not None
            else (splits_to_run[0] if splits_to_run else None)
        )
        if target_split is None:
            logger.warning(
                "Unable to plot logistic curve because no split was specified."
            )
        else:
            dataset = datasets_by_split.get(target_split)
            if dataset is None:
                logger.warning(
                    (
                        "Dataset for split '{}' was not analyzed; "
                        "cannot plot logistic curve."
                    ),
                    target_split,
                )
            else:
                clamp = (
                    float(args.logistic_height_clamp)
                    if args.logistic_height_clamp is not None
                    else float(dataset.lodging_height_clamp)
                )
                if args.plot_logistic_output is not None:
                    plot_path = args.plot_logistic_output
                elif args.output_dir is not None:
                    plots_dir = args.output_dir / "plots"
                    plot_path = plots_dir / f"logistic_curve_{target_split}.png"
                else:
                    plot_path = Path(f"logistic_curve_{target_split}.png")
                _plot_logistic_curve(
                    clamp,
                    args.logistic_scale,
                    args.logistic_inflection,
                    plot_path,
                    target_split,
                )
                logger.info(
                    "Saved logistic curve plot for {} to {}",
                    target_split,
                    plot_path.resolve(),
                )

    if args.plot_weibull:
        target_split = (
            _resolve_split(args.weibull_split)
            if args.weibull_split is not None
            else (splits_to_run[0] if splits_to_run else None)
        )
        if target_split is None:
            logger.warning(
                "Unable to plot Weibull curve because no split was specified."
            )
        else:
            dataset = datasets_by_split.get(target_split)
            if dataset is None:
                logger.warning(
                    (
                        "Dataset for split '{}' was not analyzed; "
                        "cannot plot Weibull curve."
                    ),
                    target_split,
                )
            else:
                clamp = (
                    float(args.weibull_height_clamp)
                    if args.weibull_height_clamp is not None
                    else float(dataset.lodging_height_clamp)
                )
                shape = (
                    float(args.weibull_shape)
                    if args.weibull_shape is not None
                    else float(dataset.lodging_weibull_shape)
                )
                scale = (
                    float(args.weibull_scale)
                    if args.weibull_scale is not None
                    else float(dataset.lodging_weibull_scale)
                )
                offset = (
                    float(args.weibull_offset)
                    if args.weibull_offset is not None
                    else float(dataset.lodging_weibull_offset)
                )
                if args.plot_weibull_output is not None:
                    plot_path = args.plot_weibull_output
                elif args.output_dir is not None:
                    plots_dir = args.output_dir / "plots"
                    plot_path = plots_dir / f"weibull_curve_{target_split}.png"
                else:
                    plot_path = Path(f"weibull_curve_{target_split}.png")
                _plot_weibull_curve(
                    clamp, shape, scale, offset, plot_path, target_split
                )
                logger.info(
                    "Saved Weibull curve plot for {} to {}",
                    target_split,
                    plot_path.resolve(),
                )


if __name__ == "__main__":
    main()
