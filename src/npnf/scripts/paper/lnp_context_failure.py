"""LNP updates little from its context when covariates are given.

The prediction draws for one target of a synthetic test set, by default the
environment split at 19.7 %, with max-height context, one panel per
``--sample-panels`` entry (model and conditioning). The dots are the context, the
dashed line the target's noise-free trajectory. By default LNP without covariates,
LNP with genotype and environment, and ANP with both.

Example:
    uv run python src/npnf/scripts/paper/lnp_context_failure.py \
        --run-folders $NPNF_RESULTS_DIR/synth_test_environment_dataloaders/\
{LNP,ANP}-512k-training3m_set_mode_nested_noprior/6/checkpoint-3000000/\
test_environment \
        --model-names LNP ANP --output-folder paper/lnp_context_failure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from npnf.data.batch_loader import BatchLoader
from npnf.scripts.paper.style import COLOR_LIST, save_figure, setup_style
from npnf.scripts.paper.training_objectives import (
    DEFAULT_NUM_DRAWS,
    ModelConfig,
    SampleData,
    _plot_panel,
)
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    reconstruct_synthetic_batch,
)
from npnf.utils import add_panel_labels


def conditioning_labels() -> dict[str, str]:
    return {
        "noenv_nogeno": "∅",
        "env_nogeno": "e",
        "noenv_geno": "g",
        "env_geno": "g+e",
    }


def pair_ids(batch: dict) -> list[tuple[str, str]]:
    return list(
        zip(batch["data"]["genotype_id"], batch["data"]["yearsite_uid"], strict=True)
    )


def locate(directory: Path, target: tuple[str, str]) -> tuple[int, int]:
    """Batch and row of ``target`` (genotype, environment) in the saved batches."""
    loader = BatchLoader(directory, load_predictions=False)
    for batch_index in range(len(loader)):
        ids = pair_ids(loader[batch_index])
        if target in ids:
            return batch_index, ids.index(target)
    msg = f"{target} is not in {directory}"
    raise ValueError(msg)


def load_target_sample(directory: Path, target: tuple[str, str]) -> SampleData:
    """Prediction draws and max-height context of one target."""
    batch_index, row = locate(directory, target)
    predictions, batch = BatchLoader(directory)[batch_index]
    height = reconstruct_synthetic_batch(directory, batch)["data"]["height"]
    days = height["X"][row, :, 0].float().numpy()
    ground_truth = height["Y_original"][row, 0, :, 0].float().numpy()
    context = slice(0, int(ground_truth.argmax()) + 1)
    return SampleData(
        grid_x=batch["grid_points"]["X"][:, 0].float().numpy(),
        prediction_draws=predictions["grid"][row, 0, :, :, 0].float().numpy(),
        full_x=days,
        ground_truth=ground_truth,
        context_x=days[context],
        context_y=height["Y"][row, context, 0].float().numpy(),
    )


def lnp_context_failure(
    run_folders: dict[str, Path],
    sample_panels: list[tuple[str, str]],
    target: tuple[str, str],
    num_draws: int,
    output_folder: Path,
) -> None:
    """Write the figure and the plotted values."""
    setup_style()
    output_folder.mkdir(parents=True, exist_ok=True)
    model_names = list(run_folders)
    labels = conditioning_labels()
    samples = [
        load_target_sample(run_folders[model] / conditioning / "max_height", target)
        for model, conditioning in sample_panels
    ]
    pl.DataFrame(
        [
            {
                "model": model,
                "conditioning": conditioning,
                "day": float(day),
                "draw_mean": float(mean),
                "draw_sd": float(sd),
                "ground_truth": float(
                    np.interp(day, sample.full_x, sample.ground_truth)
                ),
            }
            for (model, conditioning), sample in zip(
                sample_panels, samples, strict=True
            )
            for day, mean, sd in zip(
                sample.grid_x,
                sample.prediction_draws.mean(axis=0),
                sample.prediction_draws.std(axis=0),
                strict=True,
            )
        ]
    ).write_csv(output_folder / "lnp_context_failure_samples.csv")

    fig, sample_axes = plt.subplots(
        1, len(samples), figsize=(10, 3.4), sharey=True, squeeze=False
    )
    sample_axes = list(sample_axes[0])
    for ax, (model, conditioning), sample in zip(
        sample_axes, sample_panels, samples, strict=True
    ):
        color = COLOR_LIST[model_names.index(model)]
        _plot_panel(ax, ModelConfig(model, Path(), color), sample, num_draws)
        ax.plot(
            sample.full_x,
            sample.ground_truth,
            color="black",
            linestyle="--",
            linewidth=1.0,
            zorder=4,
        )
        ax.set_title(f"{model}, {labels[conditioning]}")
        ax.set_xlabel("Day of year")
    sample_axes[0].set_ylabel("Height")
    add_panel_labels(sample_axes)
    save_figure(fig, output_folder / "lnp_context_failure")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-folders",
        type=Path,
        nargs="+",
        required=True,
        help="Split folders of the runs, holding <conditioning>/<method>",
    )
    parser.add_argument("--model-names", nargs="+", required=True)
    parser.add_argument(
        "--sample-panels",
        nargs="+",
        default=["LNP:noenv_nogeno", "LNP:env_geno", "ANP:env_geno"],
        help="model:conditioning of each sample panel",
    )
    parser.add_argument("--genotype", default="G_0002")
    parser.add_argument("--environment", default="Synth06_2042")
    parser.add_argument("--num-draws", type=int, default=DEFAULT_NUM_DRAWS)
    parser.add_argument("--output-folder", type=Path, required=True)
    args = parser.parse_args()

    if len(args.run_folders) != len(args.model_names):
        msg = (
            f"Expected one run folder per model, got {len(args.run_folders)} folders "
            f"and {len(args.model_names)} names"
        )
        raise ValueError(msg)
    run_folders = dict(zip(args.model_names, args.run_folders, strict=True))
    sample_panels = [tuple(panel.split(":")) for panel in args.sample_panels]
    unknown = [model for model, _ in sample_panels if model not in run_folders]
    if unknown:
        msg = f"Sample panels name models without a run folder: {unknown}"
        raise ValueError(msg)
    lnp_context_failure(
        run_folders,
        sample_panels,
        (args.genotype, args.environment),
        args.num_draws,
        args.output_folder,
    )


if __name__ == "__main__":
    main()
