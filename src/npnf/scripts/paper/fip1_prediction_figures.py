"""FIP1 prediction figures in the style of the paper's own figures.

Reads the saved predictions that ``sig_mmd_fip1_context_matched_blocked.py``
scores, so the figures and the numbers come from the same files. Two kinds:

``samples``
    The paper's prediction-sample figure (``training_objectives.py``) on one
    measured FIP1 plot: rows LNP and ANP, one column per initialisation
    (FIP1 from scratch, FIP1 pre-trained), the context the model
    was given as dots and that plot's predicted draws.
``exclusive-days``
    The covariate-only predicted mean on the dates that belong to one site-year
    only, against that site-year's observed mean. The models that follow each
    site-year's level there identify the harvest year from the date alone.

Example:
    uv run src/npnf/scripts/paper/fip1_prediction_figures.py \
        --results-dir $NPNF_RESULTS_DIR --figure samples \
        --output-dir paper/fip1_prediction_samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from npnf.data.batch_loader import BatchLoader
from npnf.data.fip1_day_grid import METRIC_WINDOW
from npnf.metrics.blocked_scoring import load_prediction_day_axis
from npnf.metrics.blocks import UnitMember
from npnf.scripts.metrics.fip1_metric_grid import unit_yearsite_days
from npnf.scripts.metrics.fip1_observed_plots import (
    SPLIT_DATALOADERS,
    load_observed_plots,
    load_records,
    reconstruct_context,
)
from npnf.scripts.paper.style import COLOR_LIST, save_figure, setup_style
from npnf.scripts.paper.training_objectives import (
    DEFAULT_NUM_DRAWS,
    SampleData,
    plot_target_split_overfitting,
)
from npnf.utils import add_panel_labels

RUN_TEMPLATE = "{model}-FIP-2930-fip1_{init}_1m_seed{seed}"
CHECKPOINT = "checkpoint-1000000"
INITIALISATIONS = {"FIP1 from scratch": "scratch", "FIP1 pre-trained": "pretrained"}


def method_dir(
    results_dir: Path,
    split: str,
    conditioning: str,
    mode: str,
    model: str,
    init: str,
    seed: int,
) -> Path:
    """Saved predictions of one run."""
    run = f"{RUN_TEMPLATE.format(model=model, init=init, seed=seed)}/{CHECKPOINT}"
    return results_dir / SPLIT_DATALOADERS[split] / run / split / conditioning / mode


def locate_plot(directory: Path, plot_uid: str) -> tuple[int, int]:
    """Batch and row of ``plot_uid`` in the saved batches."""
    loader = BatchLoader(directory, load_predictions=False)
    for batch_index in range(len(loader)):
        uids = [str(uid) for uid in loader[batch_index]["data"]["plot_uid"]]
        if plot_uid in uids:
            return batch_index, uids.index(plot_uid)
    msg = f"{plot_uid} is not in {directory}"
    raise ValueError(msg)


def default_plot(observed: dict[str, dict]) -> str:
    """The plot whose maximum height is nearest the 90th percentile.

    The paper's sample figures select their targets by the same percentile.
    """
    heights = {uid: float(plot["heights"].max()) for uid, plot in observed.items()}
    target = float(np.percentile(list(heights.values()), 90.0))
    return min(heights, key=lambda uid: (abs(heights[uid] - target), uid))


def observed_sample_loader(observed: dict[str, dict]):
    """A ``training_objectives`` sample loader that reads the measured plot."""

    def load(directory: Path, batch_index: int, sample_index: int) -> SampleData:
        predictions, batch = BatchLoader(directory)[batch_index]
        plot = observed[str(batch["data"]["plot_uid"][sample_index])]
        context_days, context_values = reconstruct_context(
            directory.name, batch, sample_index, plot
        )
        return SampleData(
            grid_x=batch["grid_points"]["X"][:, 0].numpy(),
            prediction_draws=predictions["grid"][sample_index, 0, :, :, 0].numpy(),
            full_x=plot["days"],
            ground_truth=plot["heights"],
            context_x=context_days,
            context_y=context_values,
        )

    return load


def samples_figure(args: argparse.Namespace) -> None:
    observed = load_observed_plots(args.split, args.datasets_offline_path)
    plot_uid = args.plot_uid or default_plot(observed)
    rows = [
        [
            str(
                method_dir(
                    args.results_dir,
                    args.split,
                    args.conditioning,
                    args.mode,
                    model,
                    init,
                    args.seed,
                ).parent
            )
            for init in INITIALISATIONS.values()
        ]
        for model in ("LNP", "ANP")
    ]
    batch_index, sample_index = locate_plot(Path(rows[0][0]) / args.mode, plot_uid)
    logger.info("Plot {} (batch {}, row {})", plot_uid, batch_index, sample_index)
    plot_target_split_overfitting(
        rows[0],
        list(INITIALISATIONS),
        method=args.mode,
        output_folder=str(args.output_dir),
        batch_index=batch_index,
        sample_index=sample_index,
        num_draws=DEFAULT_NUM_DRAWS,
        base_folders_row2=rows[1],
        row_labels=["LNP", "ANP"],
        sample_loader=observed_sample_loader(observed),
        stem="fip1_prediction_samples",
    )


def exclusive_days(records: list[dict]) -> dict[str, np.ndarray]:
    """Dates that only one site-year measured, per site-year."""
    members = [
        UnitMember(
            genotype_id=record["gid"], yearsite_uid=record["ys"], condition_index=index
        )
        for index, record in enumerate(records)
    ]
    shared = unit_yearsite_days(members, records)
    counts: dict[int, int] = {}
    for days in shared.values():
        for day in days.tolist():
            counts[day] = counts.get(day, 0) + 1
    return {
        yearsite: np.array([day for day in days.tolist() if counts[day] == 1])
        for yearsite, days in shared.items()
    }


def exclusive_days_figure(args: argparse.Namespace) -> None:
    observed = load_observed_plots(args.split, args.datasets_offline_path)
    predictions = {}
    for model in args.models:
        loader = BatchLoader(
            method_dir(
                args.results_dir,
                args.split,
                args.conditioning,
                args.mode,
                model,
                args.init,
                args.seed,
            )
        )
        axis = load_prediction_day_axis(loader).numpy().astype(int)
        records, _ = load_records(loader, args.mode, observed)
        window = (axis >= METRIC_WINDOW[0]) & (axis < METRIC_WINDOW[1])
        mean = np.stack([record["grid"].numpy()[:, window] for record in records])
        predictions[model] = (axis[window], mean.reshape(-1, window.sum()).mean(0))
    exclusive = {
        yearsite: days
        for yearsite, days in sorted(exclusive_days(records).items())
        if days.size >= 2
    }

    fig, axes = plt.subplots(
        nrows=len(args.models),
        ncols=len(exclusive),
        figsize=(2.6 * len(exclusive), 3.2 * len(args.models)),
        sharey=True,
        squeeze=False,
    )
    for row, model in enumerate(args.models):
        days, mean = predictions[model]
        for column, (yearsite, own_days) in enumerate(exclusive.items()):
            ax = axes[row, column]
            observed_mean = np.stack(
                [
                    np.interp(own_days, record["days"], record["heights"])
                    for record in records
                    if record["ys"] == yearsite
                ]
            ).mean(0)
            ax.plot(
                own_days,
                mean[np.searchsorted(days, own_days)],
                color=COLOR_LIST[row],
                marker="o",
            )
            ax.plot(own_days, observed_mean, color="black", linestyle="--")
            if row == 0:
                ax.set_title(yearsite)
        axes[row, 0].set_ylabel(model, fontweight="bold")
    axes[0, 0].set_ylim(bottom=0)
    add_panel_labels(axes.flat)
    fig.supylabel("Height")
    fig.supxlabel("Day of year")
    fig.legend(
        handles=[
            plt.Line2D([], [], color="gray", marker="o", label="Predicted mean"),
            plt.Line2D([], [], color="black", linestyle="--", label="Observed mean"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output_dir / "fip1_exclusive_days")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--figure", required=True, choices=("samples", "exclusive-days")
    )
    parser.add_argument("--split", default="test_plot", choices=SPLIT_DATALOADERS)
    parser.add_argument("--conditioning", default="env_nogeno")
    parser.add_argument(
        "--mode",
        default="max_height",
        choices=("no_context", "random_context", "max_height"),
    )
    parser.add_argument("--models", nargs="+", default=["ANP", "ACNP"])
    parser.add_argument(
        "--init", default="pretrained", choices=("pretrained", "scratch")
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--plot-uid", default=None)
    parser.add_argument("--datasets-offline-path", default=None)
    args = parser.parse_args()

    setup_style()
    if args.figure == "samples":
        samples_figure(args)
    else:
        exclusive_days_figure(args)


if __name__ == "__main__":
    main()
