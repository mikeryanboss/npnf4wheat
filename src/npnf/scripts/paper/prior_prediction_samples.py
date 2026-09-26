"""Paper figure: prediction samples for a selected method."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")

import matplotlib.pyplot as plt
from loguru import logger

from npnf.data.batch_loader import BatchLoader
from npnf.scripts.paper.style import COLOR_LIST, COLORS, save_figure, setup_style
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    SyntheticBatchReconstructionCache,
    reconstruct_synthetic_batch,
)

SAMPLE_ALPHA = 0.22
SAMPLE_LINEWIDTH = 1.0
MEAN_COLOR = "#111111"
MEAN_LINEWIDTH = 2.8


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one plotted model."""

    name: str
    method_path: Path
    color: tuple[float, float, float]


@dataclass(frozen=True)
class PanelData:
    label: str
    x_values: np.ndarray
    sample_trajectories: np.ndarray
    mean_trajectory: np.ndarray
    sample_color: tuple[float, float, float] | str


@dataclass(frozen=True)
class TrajectoryData:
    x_values: np.ndarray
    sample_trajectories: np.ndarray
    mean_trajectory: np.ndarray


def assign_model_colors(
    model_names: list[str],
) -> dict[str, tuple[float, float, float]]:
    """Assign colors to models in the same order as other paper scripts."""
    model_colors = list(COLOR_LIST)
    return {
        name: model_colors[i % len(model_colors)] for i, name in enumerate(model_names)
    }


def discover_models(
    base_folders: list[str], model_names: list[str], method: str
) -> list[ModelConfig]:
    """Discover prediction folders for the requested models and method."""
    if len(base_folders) != len(model_names):
        msg = (
            f"Number of base folders ({len(base_folders)}) must match "
            f"number of model names ({len(model_names)})."
        )
        raise ValueError(msg)

    colors = assign_model_colors(model_names)
    models = []
    for base_folder, name in zip(base_folders, model_names, strict=True):
        method_path = Path(base_folder) / method
        if not method_path.is_dir():
            logger.warning(f"{name}: {method} not found at {method_path}")
            continue
        models.append(
            ModelConfig(name=name, method_path=method_path, color=colors[name])
        )

    return models


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _num_batches_to_load(loader: BatchLoader, num_batches: int | None) -> int:
    return len(loader) if num_batches is None else min(num_batches, len(loader))


def _update_sample_trajectories(
    sampled_trajectories: np.ndarray | None,
    sampled_priorities: np.ndarray | None,
    batch_trajectories: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_samples <= 0:
        return (
            np.empty((0, batch_trajectories.shape[1]), dtype=batch_trajectories.dtype),
            np.empty(0),
        )

    batch_priorities = rng.random(len(batch_trajectories))
    if sampled_trajectories is None or sampled_priorities is None:
        candidate_trajectories = batch_trajectories
        candidate_priorities = batch_priorities
    else:
        candidate_trajectories = np.concatenate(
            [sampled_trajectories, batch_trajectories], axis=0
        )
        candidate_priorities = np.concatenate(
            [sampled_priorities, batch_priorities], axis=0
        )

    keep_count = min(num_samples, len(candidate_priorities))
    if keep_count == len(candidate_priorities):
        keep_indices = np.arange(keep_count)
    else:
        keep_indices = np.argpartition(candidate_priorities, keep_count - 1)[
            :keep_count
        ]

    return candidate_trajectories[keep_indices], candidate_priorities[keep_indices]


def _finalize_trajectory_data(
    *,
    x_values: torch.Tensor | None,
    sum_trajectory: np.ndarray | None,
    total_count: int,
    sample_trajectories: np.ndarray | None,
    source: Path | str,
) -> TrajectoryData:
    if x_values is None or sum_trajectory is None or sample_trajectories is None:
        msg = f"No trajectory batches found in {source}"
        raise ValueError(msg)

    return TrajectoryData(
        x_values=_as_numpy(x_values),
        sample_trajectories=sample_trajectories,
        mean_trajectory=sum_trajectory / total_count,
    )


def _load_model_trajectories(
    method_dir: Path,
    num_batches: int | None,
    num_samples: int,
    rng: np.random.Generator,
) -> TrajectoryData:
    loader = BatchLoader(method_dir)
    n_batches = _num_batches_to_load(loader, num_batches)
    sample_trajectories = None
    sample_priorities = None
    sum_trajectory = None
    total_count = 0
    x_values = None

    for batch_idx in range(n_batches):
        predictions, batch = loader[batch_idx]
        trajectories = predictions["grid"][:, 0, :, :, 0].flatten(0, 1)
        trajectories_np = _as_numpy(trajectories)
        sum_batch = trajectories_np.sum(axis=0, dtype=np.float64)
        sum_trajectory = (
            sum_batch if sum_trajectory is None else sum_trajectory + sum_batch
        )
        total_count += len(trajectories_np)
        sample_trajectories, sample_priorities = _update_sample_trajectories(
            sample_trajectories, sample_priorities, trajectories_np, rng, num_samples
        )
        if x_values is None:
            x_values = batch["grid_points"]["X"][:, 0]

    return _finalize_trajectory_data(
        x_values=x_values,
        sum_trajectory=sum_trajectory,
        total_count=total_count,
        sample_trajectories=sample_trajectories,
        source=method_dir,
    )


def _load_ground_truth_trajectories(
    method_dirs: list[Path],
    num_batches: int | None,
    num_samples: int,
    rng: np.random.Generator,
) -> TrajectoryData:
    sample_trajectories = None
    sample_priorities = None
    sum_trajectory = None
    total_count = 0
    x_values = None

    for method_dir in method_dirs:
        loader = BatchLoader(method_dir, load_predictions=False)
        n_batches = _num_batches_to_load(loader, num_batches)
        cache = SyntheticBatchReconstructionCache()

        for batch_idx in range(n_batches):
            batch = loader[batch_idx]
            batch = reconstruct_synthetic_batch(method_dir, batch, cache=cache)
            data = batch["data"]
            y_original = data["height"]["Y_original"]

            if y_original.ndim == 4:
                trajectories = y_original[..., 0].flatten(0, 1)
            else:
                trajectories = y_original[..., 0]

            trajectories_np = _as_numpy(trajectories)
            sum_batch = trajectories_np.sum(axis=0, dtype=np.float64)
            sum_trajectory = (
                sum_batch if sum_trajectory is None else sum_trajectory + sum_batch
            )
            total_count += len(trajectories_np)
            sample_trajectories, sample_priorities = _update_sample_trajectories(
                sample_trajectories,
                sample_priorities,
                trajectories_np,
                rng,
                num_samples,
            )

            if x_values is None:
                x_values = data["height"]["X"][0, :, 0]

    return _finalize_trajectory_data(
        x_values=x_values,
        sum_trajectory=sum_trajectory,
        total_count=total_count,
        sample_trajectories=sample_trajectories,
        source="ground truth",
    )


def _plot_samples(ax: plt.Axes, *, panel: PanelData) -> None:
    for trajectory in panel.sample_trajectories:
        ax.plot(
            panel.x_values,
            trajectory,
            color=panel.sample_color,
            alpha=SAMPLE_ALPHA,
            linewidth=SAMPLE_LINEWIDTH,
        )

    ax.plot(
        panel.x_values,
        panel.mean_trajectory,
        color=MEAN_COLOR,
        linewidth=MEAN_LINEWIDTH,
        zorder=5,
    )
    ax.set_title(panel.label)


def _save_layout(
    panels: list[PanelData],
    output_dir: Path,
    *,
    filename: str,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes_flat = np.asarray(axes).reshape(-1)

    for ax, panel in zip(axes_flat, panels, strict=True):
        _plot_samples(ax, panel=panel)

    for ax in axes_flat:
        ax.set_xlim(180, 320)
        ax.set_xticks(np.arange(180, 321, 40))
        ax.set_ylim(bottom=0)

    fig.supxlabel("Day of year")
    fig.supylabel("Height")

    save_figure(fig, output_dir / filename)
    plt.close(fig)


def plot_prediction_samples(
    base_folders: list[str],
    model_names: list[str],
    method: str,
    figure_name: str,
    output_folder: str | None = None,
    *,
    num_batches: int | None,
    num_samples: int,
    seed: int,
) -> None:
    setup_style()
    rng = np.random.default_rng(seed)

    models = discover_models(base_folders, model_names, method)
    if not models:
        msg = f"No valid {method} model folders found"
        raise ValueError(msg)

    method_paths = [model.method_path for model in models]
    ground_truth = _load_ground_truth_trajectories(
        method_paths, num_batches, num_samples, rng
    )

    panels = [
        PanelData(
            label="Ground Truth",
            x_values=ground_truth.x_values,
            sample_trajectories=ground_truth.sample_trajectories,
            mean_trajectory=ground_truth.mean_trajectory,
            sample_color=COLORS["gray"],
        )
    ]
    for model in models:
        model_data = _load_model_trajectories(
            model.method_path, num_batches, num_samples, rng
        )
        panels.append(
            PanelData(
                label=model.name,
                x_values=model_data.x_values,
                sample_trajectories=model_data.sample_trajectories,
                mean_trajectory=model_data.mean_trajectory,
                sample_color=model.color,
            )
        )

    output_dir = Path(output_folder) if output_folder else Path("paper") / figure_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_layout(
        panels, output_dir, filename=figure_name, nrows=1, ncols=4, figsize=(15, 4)
    )
    _save_layout(
        panels,
        output_dir,
        filename=f"{figure_name}_2x2",
        nrows=2,
        ncols=2,
        figsize=(8, 7),
    )


def plot_prior_prediction_samples(
    base_folders: list[str],
    model_names: list[str],
    method: str,
    output_folder: str | None = None,
    *,
    num_batches: int | None,
    num_samples: int,
    seed: int,
) -> None:
    plot_prediction_samples(
        base_folders=base_folders,
        model_names=model_names,
        method=method,
        figure_name="prior_prediction_samples",
        output_folder=output_folder,
        num_batches=num_batches,
        num_samples=num_samples,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper figure: prediction samples for a selected method"
    )
    parser.add_argument(
        "--base-folders",
        type=str,
        nargs="+",
        required=True,
        help="Base paths to model results (without /<method> suffix)",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        required=True,
        help="Display names for each model (must match number of base-folders)",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Prediction method subfolder to load below each base-folder",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (defaults to paper/prior_prediction_samples)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Number of prediction batches to load (defaults to all)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=140,
        help="Number of sample trajectories to draw in each panel",
    )
    parser.add_argument("--seed", type=int, default=7, help="Sample selection seed")
    args = parser.parse_args()

    plot_prior_prediction_samples(
        base_folders=args.base_folders,
        model_names=args.model_names,
        method=args.method,
        output_folder=args.output_folder,
        num_batches=args.num_batches,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
