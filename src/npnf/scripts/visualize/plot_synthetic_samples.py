"""Utility for visualizing simulated synthetic crop height samples.

This script instantiates the dataset through the Hydra configs living under
``npnf.data.configs.datasets.synthetic`` and produces a grid of traces that
show the latent growth curve together with the noisy measurement points for a
handful of randomly chosen samples.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra_zen import instantiate
from loguru import logger

from npnf.data.configs.datasets.synthetic import TrainConfig_64k, ValConfig
from npnf.data.datasets.synthetic import SyntheticDataset

DATASET_CONFIGS = {"train": TrainConfig_64k, "val": ValConfig}


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor to numpy for plotting."""

    return tensor.numpy(force=True)


def load_dataset(split: str, *, num_samples: int, num_points: int) -> SyntheticDataset:
    """Instantiate the requested split with a smaller sample pool for plotting."""

    config = DATASET_CONFIGS[split]
    logger.info(
        "Loading {} dataset with {} samples per environment and {} target points",
        split,
        num_samples,
        num_points,
    )
    return instantiate(config, num_samples=num_samples, num_points=num_points)


def pick_indices(total: int, count: int, seed: int | None = None) -> Iterable[int]:
    generator = torch.Generator()
    if seed is not None:
        generator = generator.manual_seed(seed)
    perm = torch.randperm(total, generator=generator)
    return perm[:count].tolist()


def plot_samples(
    dataset: torch.utils.data.Dataset,
    indices: Iterable[int],
    output_path: Path,
    *,
    split: str,
) -> None:
    indices = list(indices)
    if not indices:
        msg = "No indices provided for plotting."
        raise ValueError(msg)

    num_plots = len(indices)
    num_cols = min(3, num_plots)
    num_rows = math.ceil(num_plots / num_cols)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharex=True
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, index in zip(axes, indices, strict=False):
        sample = dataset[index]

        days_full = _to_numpy(sample["height_days_all"])
        heights_full = _to_numpy(sample["height_values_all_nonoise"])
        days_sampled = _to_numpy(sample["height_days"])
        heights_noisy = _to_numpy(sample["height_values"])
        mask_lodged = _to_numpy(sample["height_lodged_mask"]).astype(bool)
        mask_lodged_full = _to_numpy(sample["height_lodged_mask_all"]).astype(bool)

        ax.plot(days_full, heights_full, color="#4c72b0", linewidth=2.0, label="latent")
        if len(days_sampled) > 0:
            ax.scatter(
                days_sampled[~mask_lodged],
                heights_noisy[~mask_lodged],
                s=15,
                color="#2c7fb8",
                alpha=0.9,
                label="observed",
            )
            if mask_lodged.any():
                ax.scatter(
                    days_sampled[mask_lodged],
                    heights_noisy[mask_lodged],
                    s=20,
                    color="#d95f02",
                    marker="x",
                    label="lodged obs",
                )

        lodged_day = None
        if mask_lodged_full.any():
            lodged_day = days_full[np.argmax(mask_lodged_full)]
            ax.axvline(lodged_day, color="#d95f02", linestyle="--", linewidth=1.0)

        year = int(sample["harvest_year"].item())
        site = sample["yearsite_uid"]
        lodged_flag = bool(sample["has_lodged"].item())
        subtitle = f"{site} {year}"
        if lodged_flag:
            subtitle += " • lodged"
        ax.set_title(subtitle)
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Height (m)")
        ax.grid(alpha=0.3)

    for ax in axes[num_plots:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"Synthetic samples ({split})", fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.98, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote sample plot to {}", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=sorted(DATASET_CONFIGS),
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--num-plots",
        type=int,
        default=6,
        help="Number of random samples to visualize.",
    )
    parser.add_argument(
        "--samples-per-environment",
        type=int,
        default=8,
        help="How many simulations to draw per weather environment.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=150,
        help="Target number of measurement points per sample.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to store the resulting figure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed that controls which samples are drawn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path
    if output_path is None:
        output_path = Path("plots") / f"synthetic_samples_{args.split}.png"

    dataset = load_dataset(
        args.split,
        num_samples=max(1, args.samples_per_environment),
        num_points=args.num_points,
    )
    num_plots = min(args.num_plots, len(dataset))
    if num_plots <= 0:
        msg = "Cannot plot zero samples."
        raise ValueError(msg)

    indices = pick_indices(len(dataset), num_plots, args.seed)
    logger.info("Plotting indices: {}", indices)
    plot_samples(dataset, indices, output_path, split=args.split)


if __name__ == "__main__":
    main()
