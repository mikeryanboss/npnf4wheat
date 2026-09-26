import argparse
import math
from multiprocessing import Pool
from pathlib import Path

import imageio as iio
import matplotlib as mpl
import numpy as np
from PIL import Image

mpl.use("Agg")  # Use non-interactive backend for performance
import matplotlib.pyplot as plt
import seaborn as sns
import tensordict
import torch
from loguru import logger

from npnf.data.batch_loader import BatchLoader
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    reconstruct_synthetic_batch,
)
from npnf.utils import fig2img

plt.ioff()  # Turn off interactive mode

GLOBAL_MEAN_LINEWIDTH = 2.3
GLOBAL_STD_ALPHA = 0.01
GLOBAL_STD_MIN_ALPHA = 0.005
STD_ALPHA_BASE_TRAJECTORIES = 16
MAX_CONTEXT_STEPS = 100  # Focus on useful range where model behavior is meaningful


def _compute_std_alpha(num_trajectories: int) -> float:
    """Scale the std band alpha as trajectories increase to reduce overdraw."""
    if num_trajectories <= 0:
        return GLOBAL_STD_ALPHA

    ratio = STD_ALPHA_BASE_TRAJECTORIES / num_trajectories
    scaled = GLOBAL_STD_ALPHA * min(1.0, math.sqrt(ratio))
    return max(scaled, GLOBAL_STD_MIN_ALPHA)


def process_single_sample(args):
    """Process a single sample to create and save GIF."""
    (
        i,
        context_prior_sample,
        context_posterior_sample,
        predictions_sample,
        sample_context_indices,
        grid_x,
        x_min,
        x_max,
        y_min,
        y_max,
        gifs_path,
        training_days_min,
        training_days_max,
    ) = args

    logger.info(f"Processing sample {i}")

    # Set style once per worker
    sns.set_style("whitegrid")

    predictions_list = list(predictions_sample)[:MAX_CONTEXT_STEPS]
    num_steps = len(predictions_list)

    # Pre-compute all context slices to avoid repeated TensorDict creation
    context_slices = [context_prior_sample]  # Step 0: use initial context
    if sample_context_indices is not None:
        for step_index in range(1, num_steps):
            selected_indices = sample_context_indices[:step_index]
            indices_tensor = torch.tensor(selected_indices, dtype=torch.long)
            context_slices.append(
                tensordict.TensorDict(
                    {
                        key: (
                            tensor[:, indices_tensor]
                            if tensor.ndim > 2
                            else tensor[indices_tensor]
                        )
                        for key, tensor in context_posterior_sample.items()
                    }
                )
            )
    else:
        context_slices = [context_prior_sample] * num_steps

    # Create figure once and reuse
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)

    # Stream frames directly to GIF file
    gif_path = gifs_path / f"sample_{i}.gif"
    with iio.get_writer(gif_path, mode="I", duration=1000, loop=0) as writer:
        for step_index, prediction in enumerate(predictions_list):
            context_at_step = context_slices[step_index]

            img = create_uncertainty_plot(
                context_prior=context_at_step,
                context_posterior=context_posterior_sample,
                grid_x=grid_x,
                predictions=prediction,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                training_days_min=training_days_min,
                training_days_max=training_days_max,
                fig=fig,
                ax=ax,
            )
            # Convert PIL Image to numpy array for imageio
            writer.append_data(np.asarray(img))  # ty: ignore[unresolved-attribute]

    plt.close(fig)
    logger.info(f"Saved GIF for sample {i}")

    return i


def create_uncertainty_plot(  # noqa: PLR0912
    context_prior: tensordict.TensorDict,
    context_posterior: tensordict.TensorDict,
    predictions: tensordict.TensorDict | dict[str, torch.Tensor],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_x: torch.Tensor | None = None,
    training_days_min: float = 177.0,
    training_days_max: float = 329.0,
    figsize: tuple[float, float] | None = None,
    dpi: int = 120,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    targets: tensordict.TensorDict | None = None,
) -> Image.Image:
    # Determine if we own the figure (for cleanup)
    owns_figure = fig is None

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 7), dpi=dpi)
    else:
        ax.clear()

    # Define a nice color palette with high contrast
    colors = {
        "predictions": "#2E86AB",  # Nice blue
        "context_prior": "#A23B72",  # Deep magenta/purple
        "context_posterior": "#F18F01",  # Bright orange
        "targets": "#4CAF50",  # Green
        "training_range": "#C73E1D",  # Deep red
    }

    def _maybe_get_prediction_field(container, key: str):
        if isinstance(container, dict):
            return container.get(key)
        try:
            return container[key]
        except KeyError:
            return None

    pred_grid = predictions["grid"]
    if not isinstance(pred_grid, torch.Tensor):
        msg = "Prediction grid must be a torch.Tensor for visualization."
        raise TypeError(msg)
    pred_grid = pred_grid.detach().cpu()
    if pred_grid.ndim != 3:
        msg = "Prediction grid must have shape [global, time, value]."
        raise ValueError(msg)

    grid_std = _maybe_get_prediction_field(predictions, "grid_std")
    if grid_std is not None:
        if not isinstance(grid_std, torch.Tensor):
            msg = "Prediction std grid must be a torch.Tensor."
            raise TypeError(msg)
        grid_std = grid_std.detach().cpu()
        if grid_std.ndim != 3:
            msg = "Prediction std grid must have shape [global, time, value]."
            raise ValueError(msg)
        grid_std = grid_std[: pred_grid.shape[0]]

    if grid_x is None:
        grid_x_tensor = torch.arange(pred_grid.shape[2], dtype=pred_grid.dtype)
    elif isinstance(grid_x, torch.Tensor):
        grid_x_tensor = grid_x.detach().cpu()
    else:
        grid_x_tensor = torch.as_tensor(grid_x)
    grid_x_tensor = grid_x_tensor.reshape(-1)
    grid_x_np = grid_x_tensor.numpy(force=True)

    num_global = pred_grid.shape[0]
    std_alpha = _compute_std_alpha(num_global)
    palette = sns.color_palette("crest", n_colors=max(num_global, 1))
    gaussian_band_label_used = False
    gaussian_line_label_used = False

    def _squeeze_value_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.squeeze(-1) if tensor.shape[-1] == 1 else tensor

    for global_index in range(num_global):
        color = palette[global_index % len(palette)]
        mean_trace_tensor = _squeeze_value_dim(pred_grid[global_index])
        std_trace_tensor = (
            _squeeze_value_dim(grid_std[global_index]) if grid_std is not None else None
        )

        mean_trace = mean_trace_tensor.numpy(force=True)
        if std_trace_tensor is not None:
            std_trace = std_trace_tensor.numpy(force=True)
            ax.fill_between(
                grid_x_np,
                mean_trace - std_trace,
                mean_trace + std_trace,
                color=color,
                alpha=std_alpha,
                zorder=1.2,
                label="Global sample ±1σ" if not gaussian_band_label_used else None,
            )
            gaussian_band_label_used = True

        ax.plot(
            grid_x_np,
            mean_trace,
            linewidth=GLOBAL_MEAN_LINEWIDTH,
            alpha=0.95,
            zorder=2,
            color=color,
            label="Global sample μ" if not gaussian_line_label_used else None,
        )
        gaussian_line_label_used = True

    overall_mean = _squeeze_value_dim(pred_grid.mean(dim=0)).numpy(force=True)
    ax.plot(
        grid_x_np,
        overall_mean,
        linewidth=GLOBAL_MEAN_LINEWIDTH + 0.4,
        alpha=0.9,
        zorder=2.5,
        color="#1B5E20",  # Dark green for global aggregate line
        label="Overall mean",
    )

    # Plot context_prior and context_posterior points with clean, standard styling
    ax.scatter(
        context_prior["X"],  # ty: ignore[invalid-argument-type]
        context_prior["Y"],  # ty: ignore[invalid-argument-type]
        color=colors["context_prior"],
        label="Context Prior Points",
        zorder=5,
        s=40,
        alpha=0.8,
    )
    ax.scatter(
        context_posterior["X"],  # ty: ignore[invalid-argument-type]
        context_posterior["Y"],  # ty: ignore[invalid-argument-type]
        color=colors["context_posterior"],
        label="Context Posterior Points",
        zorder=4,
        s=40,
        alpha=0.8,
    )
    if targets is not None:
        ax.scatter(
            targets["X"],  # ty: ignore[invalid-argument-type]
            targets["Y"],  # ty: ignore[invalid-argument-type]
            color=colors["targets"],
            label="Target Points",
            zorder=3,
            s=40,
            alpha=0.8,
        )

    # Add vertical lines to show training data range with better styling
    ax.axvline(
        x=training_days_min,
        color=colors["training_range"],
        linestyle="--",
        alpha=0.8,
        linewidth=2.5,
        label="Training Range",
        zorder=3,
    )
    ax.axvline(
        x=training_days_max,
        color=colors["training_range"],
        linestyle="--",
        alpha=0.8,
        linewidth=2.5,
        zorder=3,
    )

    # Improve legend with better styling
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        fontsize=11,
    )
    legend.get_frame().set_facecolor("white")

    # Set limits and improve axis styling
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Days", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax.grid(visible=True, alpha=0.3, linewidth=0.5)

    # Improve tick styling
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Tight layout for better spacing
    fig.tight_layout()

    # Convert to image
    fig.canvas.draw()
    img = fig2img(fig)

    # Only close figure if we created it
    if owns_figure:
        plt.close(fig)

    return img


def visualize(
    results_folder: str,
    batch_index: int = 0,
    x_min: float = 200.0,
    x_max: float = 350.0,
    y_min: float = -0.1,
    y_max: float = 2.0,
    training_days_min: float = 177.0,
    training_days_max: float = 329.0,
    num_images: int = 16,
) -> None:
    logger.info(f"Visualizing results from: {results_folder}, batch {batch_index}")

    method_dir = Path(results_folder)
    loader = BatchLoader(method_dir)
    gifs_path = method_dir / "gifs"
    gifs_path.mkdir(parents=True, exist_ok=True)

    predictions, batch_dict = loader[batch_index]
    batch_dict = reconstruct_synthetic_batch(method_dir, batch_dict)
    grid_x = batch_dict["grid_points"]["X"]
    ground_truth = batch_dict["data"]["height"]
    context_indices = batch_dict.get("context_indices")

    predictions_list = list(predictions)
    ground_truth_list = list(ground_truth)

    # Create empty context prior (step 0 has no context)
    sample_gt = ground_truth_list[0]
    empty_context = tensordict.TensorDict(
        {
            k: torch.empty(0, v.shape[-1], device=v.device, dtype=v.dtype)
            for k, v in sample_gt.items()
        },
        batch_size=[0],
    )

    samples_to_process = min(len(predictions_list), num_images)
    logger.info(f"Processing {samples_to_process} samples from batch {batch_index}")

    sample_args = [
        (
            i,
            empty_context,
            ground_truth_list[i],
            predictions_list[i],
            context_indices[i] if context_indices else None,
            grid_x,
            x_min,
            x_max,
            y_min,
            y_max,
            gifs_path,
            training_days_min,
            training_days_max,
        )
        for i in range(samples_to_process)
    ]

    with Pool() as pool:
        pool.map(process_single_sample, sample_args)

    logger.info(f"Saved {samples_to_process} GIFs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", type=str, required=True)
    parser.add_argument("--batch-index", type=int, default=0)
    args = parser.parse_args()
    visualize(args.results_folder, batch_index=args.batch_index)
