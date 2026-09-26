"""Paper figure: target-split overfitting prediction sample."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")

import matplotlib.pyplot as plt

from npnf.data.batch_loader import BatchLoader
from npnf.scripts.paper.style import COLOR_LIST, save_figure, setup_style
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    reconstruct_synthetic_batch,
)
from npnf.utils import add_panel_labels

DEFAULT_BATCH_INDEX = 289
DEFAULT_SAMPLE_INDEX = 18
DEFAULT_NUM_DRAWS = 28

DRAW_ALPHA = 0.42
DRAW_LINEWIDTH = 1.35
CONTEXT_ALPHA = 0.42
CONTEXT_SIZE = 9
CONTEXT_COLOR = "#333333"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    method_dir: Path
    color: tuple[float, float, float]


@dataclass(frozen=True)
class SampleData:
    grid_x: np.ndarray
    prediction_draws: np.ndarray
    full_x: np.ndarray
    ground_truth: np.ndarray
    context_x: np.ndarray
    context_y: np.ndarray


def discover_models(
    base_folders: list[str], column_names: list[str], method: str
) -> list[ModelConfig]:
    if len(base_folders) != len(column_names):
        msg = (
            f"Number of base folders ({len(base_folders)}) must match "
            f"number of column names ({len(column_names)})."
        )
        raise ValueError(msg)

    colors = [COLOR_LIST[1], COLOR_LIST[2], COLOR_LIST[0]]
    models = []
    for index, (base_folder, name) in enumerate(
        zip(base_folders, column_names, strict=True)
    ):
        method_dir = Path(base_folder) / method
        if not method_dir.is_dir():
            msg = f"{name}: {method} not found at {method_dir}"
            raise FileNotFoundError(msg)
        models.append(ModelConfig(name, method_dir, colors[index % len(colors)]))
    return models


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def load_sample(method_dir: Path, batch_index: int, sample_index: int) -> SampleData:
    predictions, batch = BatchLoader(method_dir)[batch_index]
    batch = reconstruct_synthetic_batch(method_dir, batch)
    height = batch["data"]["height"]
    y_original = height["Y_original"][sample_index]
    ground_truth = y_original[0, :, 0] if y_original.ndim == 3 else y_original[:, 0]
    noisy_y = height["Y"][sample_index, :, 0]
    full_x = height["X"][sample_index, :, 0]
    max_index = int(ground_truth.argmax().item())

    return SampleData(
        grid_x=_as_numpy(batch["grid_points"]["X"][:, 0]),
        prediction_draws=_as_numpy(predictions["grid"][sample_index, 0, :, :, 0]),
        full_x=_as_numpy(full_x),
        ground_truth=_as_numpy(ground_truth),
        context_x=_as_numpy(full_x[: max_index + 1]),
        context_y=_as_numpy(noisy_y[: max_index + 1]),
    )


def _assert_matched_sample(samples: list[SampleData]) -> None:
    reference = samples[0]
    for sample in samples[1:]:
        if not (
            np.allclose(sample.full_x, reference.full_x)
            and np.allclose(sample.ground_truth, reference.ground_truth)
            and np.allclose(sample.grid_x, reference.grid_x)
        ):
            msg = "Selected batch/sample is not matched across model folders."
            raise ValueError(msg)


def _plot_panel(
    ax: plt.Axes, model: ModelConfig, sample: SampleData, num_draws: int
) -> None:
    ax.scatter(
        sample.context_x,
        sample.context_y,
        color=CONTEXT_COLOR,
        s=CONTEXT_SIZE,
        alpha=CONTEXT_ALPHA,
        linewidths=0,
        zorder=1,
    )
    for draw in sample.prediction_draws[:num_draws]:
        ax.plot(
            sample.grid_x,
            draw,
            color=model.color,
            alpha=DRAW_ALPHA,
            linewidth=DRAW_LINEWIDTH,
            zorder=3,
        )

    ax.set_xlim(180, 320)
    ax.set_xticks(np.arange(180, 321, 40))
    ax.set_ylim(bottom=0)


def plot_target_split_overfitting(
    base_folders: list[str],
    column_names: list[str],
    *,
    method: str,
    output_folder: str | None,
    batch_index: int,
    sample_index: int,
    num_draws: int,
    base_folders_row2: list[str] | None = None,
    row_labels: list[str] | None = None,
    sample_loader: Callable[[Path, int, int], SampleData] = load_sample,
    stem: str = "target_split_overfitting",
) -> None:
    setup_style()
    models_row1 = discover_models(base_folders, column_names, method)
    models_rows: list[list[ModelConfig]] = [models_row1]

    if base_folders_row2 is not None:
        if row_labels is None or len(row_labels) != 2:
            msg = (
                "--row-labels must be provided with exactly 2 values"
                " when using --base-folders-row2"
            )
            raise ValueError(msg)
        if len(base_folders_row2) != len(column_names):
            msg = (
                f"Number of base folders for row 2 ({len(base_folders_row2)}) "
                f"must match number of column names ({len(column_names)})."
            )
            raise ValueError(msg)
        models_row2 = discover_models(base_folders_row2, column_names, method)
        models_rows.append(models_row2)
    elif row_labels is not None:
        msg = "--row-labels requires --base-folders-row2"
        raise ValueError(msg)

    samples_rows: list[list[SampleData]] = []
    for models in models_rows:
        samples = [
            sample_loader(model.method_dir, batch_index, sample_index)
            for model in models
        ]
        samples_rows.append(samples)

    all_samples = [s for row in samples_rows for s in row]
    _assert_matched_sample(all_samples)

    nrows = len(models_rows)
    ncols = len(column_names)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10, 3.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_idx, (models, samples) in enumerate(
        zip(models_rows, samples_rows, strict=True)
    ):
        for col_idx, (model, sample) in enumerate(zip(models, samples, strict=True)):
            _plot_panel(axes[row_idx, col_idx], model, sample, num_draws)

    for col_idx, name in enumerate(column_names):
        axes[0, col_idx].set_title(name)

    if row_labels is not None:
        for row_idx, label in enumerate(row_labels):
            axes[row_idx, 0].set_ylabel(label, fontweight="bold")
        fig.supylabel("Height")
    else:
        axes[0, 0].set_ylabel("Height")

    fig.supxlabel("Day of year")
    add_panel_labels(axes.flat)

    output_dir = (
        Path(output_folder)
        if output_folder is not None
        else Path("paper") / "target_split_overfitting"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir / stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper figure: target-split overfitting prediction sample"
    )
    parser.add_argument(
        "--base-folders",
        type=str,
        nargs="+",
        required=True,
        help="Base paths to model results (without /<method> suffix)",
    )
    parser.add_argument(
        "--column-names",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Column titles displayed above each column"
            " (must match number of base-folders)"
        ),
    )
    parser.add_argument(
        "--base-folders-row2",
        type=str,
        nargs="+",
        default=None,
        help="Base paths for row 2 (optional; triggers 2-row layout)",
    )
    parser.add_argument(
        "--row-labels",
        type=str,
        nargs=2,
        default=None,
        help="Row labels for 2-row layout (e.g., LNP ANP)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="max_height",
        help="Prediction method subfolder to load below each base-folder",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (defaults to paper/target_split_overfitting)",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=DEFAULT_BATCH_INDEX,
        help="Prediction batch index to visualize",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=DEFAULT_SAMPLE_INDEX,
        help="Sample index within the selected batch",
    )
    parser.add_argument(
        "--num-draws",
        type=int,
        default=DEFAULT_NUM_DRAWS,
        help="Number of prediction draws to plot per panel",
    )
    args = parser.parse_args()

    plot_target_split_overfitting(
        base_folders=args.base_folders,
        column_names=args.column_names,
        method=args.method,
        output_folder=args.output_folder,
        batch_index=args.batch_index,
        sample_index=args.sample_index,
        num_draws=args.num_draws,
        base_folders_row2=args.base_folders_row2,
        row_labels=args.row_labels,
    )


if __name__ == "__main__":
    main()
