"""Plot predicted trajectories across conditioning variants."""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from npnf.data.batch_loader import BatchLoader
from npnf.scripts.paper.style import COLORS, save_figure, setup_style
from npnf.scripts.utils.constants import CONFIG_NAMES, find_conditioning_folders
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    SyntheticBatchReconstructionCache,
    reconstruct_synthetic_batch,
)


def load_predictions(method_dir: Path, batch_indices: np.ndarray) -> np.ndarray:
    """Load predicted trajectories for given batch indices.

    Returns:
        Trajectories array with shape [N, T_pred].
    """
    loader = BatchLoader(method_dir, load_batch_data=False)

    trajectories = []
    for idx in batch_indices:
        predictions = loader[int(idx)]
        # predictions["grid"] shape: [batch, steps, global, time, value]
        traj = predictions["grid"][:, 0, 0, :, 0].numpy()
        trajectories.append(traj)

    return np.concatenate(trajectories, axis=0)


def load_ground_truth(
    method_dir: Path, batch_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ground truth and grid_x for given batch indices.

    Returns:
        Tuple of (grid_x [T_pred], gt_x [T_gt], gt_y [N, T_gt]).
    """
    loader = BatchLoader(method_dir, load_predictions=False)
    cache = SyntheticBatchReconstructionCache()

    gt_trajectories = []
    grid_x = np.empty(0)
    gt_x = np.empty(0)

    for idx in batch_indices:
        batch_dict = loader[int(idx)]
        batch_dict = reconstruct_synthetic_batch(method_dir, batch_dict, cache=cache)
        if grid_x.size == 0:
            grid_x = batch_dict["grid_points"]["X"][:, 0].numpy()
            gt_x = batch_dict["data"]["height"]["X"][0, :, 0].numpy()
        gt_y = batch_dict["data"]["height"]["Y"][:, :, 0].numpy()
        gt_trajectories.append(gt_y)

    return grid_x, gt_x, np.concatenate(gt_trajectories, axis=0)


def plot_trajectories(
    configs: dict[str, Path],
    output_folder: Path,
    num_trajectories: int = 200,
    num_batches: int = 5,
    seed: int = 42,
) -> None:
    """Create multi-panel figure with predicted trajectories."""
    setup_style()
    rng = np.random.default_rng(seed)

    nrows = 2
    ncols = 2
    fig, axes_grid = plt.subplots(
        nrows, ncols, figsize=(9, 7), sharey=True, sharex=True
    )
    axes = axes_grid.flatten()

    color = sns.color_palette("crest", n_colors=1)[0]

    # Select batch indices once (shared across all configs)
    first_dir = next(iter(configs.values()))
    total_batches = len(BatchLoader(first_dir))
    num_batches = min(num_batches, total_batches)
    batch_indices = rng.choice(total_batches, size=num_batches, replace=False)
    batch_indices.sort()

    # Load ground truth once from the first config (same data across all)
    grid_x, gt_x, gt_y = load_ground_truth(first_dir, batch_indices)

    # Select sample indices once
    total_samples = len(gt_y)
    if total_samples > num_trajectories:
        sample_idx = rng.choice(total_samples, size=num_trajectories, replace=False)
        gt_y = gt_y[sample_idx]
    else:
        sample_idx = np.arange(total_samples)

    logger.info(f"Selected {len(sample_idx)} samples from {num_batches} batches")

    for i, (config_name, method_dir) in enumerate(configs.items()):
        ax = axes[i]
        display_name = CONFIG_NAMES.get(config_name, config_name)
        logger.info(f"Loading {config_name} from {method_dir}")

        trajectories = load_predictions(method_dir, batch_indices)
        trajectories = trajectories[sample_idx]

        logger.info(
            f"  Plotting {len(trajectories)} trajectories "
            f"(max height range: "
            f"{trajectories.max(axis=1).min():.2f} - "
            f"{trajectories.max(axis=1).max():.2f})"
        )

        # Plot ground truth behind predictions
        for gt in gt_y:
            ax.plot(gt_x, gt, color=COLORS["gray"], alpha=0.15, linewidth=0.8, zorder=1)

        # Plot predicted trajectories
        for traj in trajectories:
            ax.plot(grid_x, traj, color=color, alpha=0.15, linewidth=0.8, zorder=2)

        # Plot mean ground truth
        mean_gt = gt_y.mean(axis=0)
        ax.plot(
            gt_x,
            mean_gt,
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="Mean ground truth",
            zorder=9,
        )

        # Plot mean prediction
        mean_traj = trajectories.mean(axis=0)
        ax.plot(
            grid_x,
            mean_traj,
            color="darkblue",
            linewidth=2.0,
            label="Mean prediction",
            zorder=10,
        )

        ax.set_xlim(180, 320)
        ax.set_title(display_name)
        if i >= 2:
            ax.set_xlabel("Day")
        if i % 2 == 0:
            ax.set_ylabel("Height")

    axes[-1].legend(loc="upper left", frameon=True, framealpha=0.8)
    fig.tight_layout()

    output_folder.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_folder / "trajectories")
    plt.close(fig)
    logger.info(f"Saved figure to {output_folder}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot predicted trajectories across conditioning variants"
    )
    parser.add_argument(
        "--base-results-path",
        type=str,
        required=True,
        help="Base path containing conditioning subfolders",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="no_context",
        help="Prediction method (default: no_context)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (defaults to base_path/trajectory_plots/)",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=200,
        help="Number of trajectories to plot per panel (default: 200)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of batches to load per config (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for batch/trajectory selection (default: 42)",
    )
    args = parser.parse_args()

    base_path = Path(args.base_results_path)
    configs = find_conditioning_folders(base_path, args.method)

    if not configs:
        msg = f"No {args.method} folders found under {base_path}"
        raise FileNotFoundError(msg)

    logger.info(f"Found {len(configs)} configurations: {list(configs.keys())}")

    output_folder = (
        Path(args.output_folder)
        if args.output_folder
        else base_path / "trajectory_plots" / args.method
    )

    plot_trajectories(
        configs,
        output_folder,
        num_trajectories=args.num_trajectories,
        num_batches=args.num_batches,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
