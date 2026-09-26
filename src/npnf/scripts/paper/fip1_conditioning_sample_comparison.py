"""The paper's conditioning sample comparison, on FIP1.

``conditioning_sample_comparison.py`` builds the paper's trajectory grid: one row
per covariate conditioning, one column per model, and the ground truth in the
first column. The only part of it that knows about the simulator is the
ground-truth source, so this script supplies a FIP1 one, which reads the measured
plots that the Sig-MMD scorer also reads, and reuses the rest unchanged.

A measured plot is one realisation, on that plot's own flight dates. The source
puts it on the saved prediction grid by linear interpolation inside the measured
range and leaves the rest of the grid empty, so the drawn observation never
extends past the dates the plot actually has.

The figure pools the four test sets. Each alone is narrow: the environment splits
hold 2019 only, and the plot split holds few plots per genotype. Together they hold
748 disjoint plots over all seven site-years, and the x range is the span of the
shared day grid, which is defined over the same seven site-years.

Example:
    uv run src/npnf/scripts/paper/fip1_conditioning_sample_comparison.py \
        --results-dir $NPNF_RESULTS_DIR \
        --models CNP ACNP LNP-NF-Prior-Posterior ANP-NF-Prior-Posterior \
        --method max_height \
        --output-folder .grapes/444/tmp/figures/max_height
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

from npnf.data.batch_loader import BatchLoader
from npnf.data.datasets.fip1 import Fip1Facts
from npnf.data.fip1_day_grid import fip1_day_grid
from npnf.scripts.metrics.fip1_observed_plots import (
    SPLIT_DATALOADERS,
    load_observed_plots,
)
from npnf.scripts.paper.conditioning_sample_comparison import (
    CONDITIONING_DIRS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_TARGET_PERCENTILE,
    SAMPLE_ALPHA,
    SAMPLE_LINEWIDTH,
    plot_conditioning_sample_comparison,
)

RUN_TEMPLATE = "{model}-FIP-2930-fip1_{init}_1m_seed{seed}"
# The row labels of the paper's own sample figure: no covariate, genotype markers,
# temperature, both.
ROW_LABELS = (r"$\boldsymbol{\emptyset}$", "g", "e", "g+e")


class ObservedTruth:
    """Measured FIP1 plots as the ground-truth column, one per batch row."""

    def __init__(self, observed: dict[str, dict]) -> None:
        self._observed = observed

    def rows(
        self, method_dir: Path, batch: dict
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        days = batch["grid_points"]["X"][:, 0].numpy().astype(np.float64)
        rows = []
        for plot_uid in batch["data"]["plot_uid"]:
            plot = self._observed[str(plot_uid)]
            curve = np.interp(
                days, plot["days"], plot["heights"], left=np.nan, right=np.nan
            )
            rows.append(curve[None, :])
        return days, rows


def model_roots(
    results_dir: Path,
    split: str,
    models: list[str],
    init: str,
    seed: int,
    checkpoint: str,
) -> list[str]:
    """The conditioning-directory parents of one FIP1 run per model."""
    return [
        str(
            results_dir
            / SPLIT_DATALOADERS[split]
            / RUN_TEMPLATE.format(model=model, init=init, seed=seed)
            / checkpoint
            / split
        )
        for model in models
    ]


def pooled_roots(
    split_roots: dict[str, list[str]], method: str, pooled_dir: Path
) -> list[str]:
    """One conditioning-directory parent per model over the batches of every split.

    ``split_roots`` holds, per split, the roots of ``model_roots`` in model order.
    The plotting reads one method directory per model and conditioning, so the
    saved batches of the splits are linked into one numbered sequence.
    """
    roots = []
    for model_roots_of_splits in zip(*split_roots.values(), strict=True):
        root = pooled_dir / Path(model_roots_of_splits[0]).parents[1].name
        for conditioning in CONDITIONING_DIRS:
            pooled = root / conditioning / method / "predictions"
            pooled.mkdir(parents=True)
            batches = [
                batch
                for split_root in model_roots_of_splits
                for batch in sorted(
                    (Path(split_root) / conditioning / method / "predictions").iterdir()
                )
            ]
            for number, batch in enumerate(batches):
                (pooled / f"{number:03d}").symlink_to(batch.resolve())
        roots.append(str(root))
    return roots


def plot_conditions(method_dir: Path) -> list[tuple[str, str, str]]:
    """Genotype, site-year and plot of every plot the saved batches hold."""
    loader = BatchLoader(method_dir, load_predictions=False)
    conditions = []
    for batch_index in range(len(loader)):
        data = loader[batch_index]["data"]
        conditions.extend(
            (str(genotype_id), str(yearsite_uid), str(plot_uid))
            for genotype_id, yearsite_uid, plot_uid in zip(
                data["genotype_id"], data["yearsite_uid"], data["plot_uid"], strict=True
            )
        )
    return conditions


def default_targets(
    conditions: list[tuple[str, str, str]], observed: dict[str, dict]
) -> tuple[str, str]:
    """Genotype and site-year of the pair nearest the 90th height percentile.

    FIP1's trial design is incomplete, so the genotype at that percentile and the
    site-year at that percentile need not occur together, and the `g+e` row needs
    a pair that does. Selecting the pair first keeps the row populated.

    Only genotypes that the plots cover in the largest number of site-years are
    candidates. The `g` row then really holds several years, which is the whole
    difference between it and the `g+e` row.
    """
    site_years: dict[str, set[str]] = {}
    for genotype_id, yearsite_uid, _ in conditions:
        site_years.setdefault(genotype_id, set()).add(yearsite_uid)
    widest = max(len(years) for years in site_years.values())

    heights: dict[tuple[str, str], list[float]] = {}
    for genotype_id, yearsite_uid, plot_uid in conditions:
        if len(site_years[genotype_id]) < widest:
            continue
        heights.setdefault((genotype_id, yearsite_uid), []).append(
            float(observed[plot_uid]["heights"].max())
        )
    means = {pair: float(np.mean(values)) for pair, values in heights.items()}
    percentile = float(
        np.percentile(list(means.values()), DEFAULT_TARGET_PERCENTILE * 100.0)
    )
    return min(means, key=lambda pair: (abs(means[pair] - percentile), pair))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--model-names", nargs="+", default=None)
    parser.add_argument(
        "--method",
        default="max_height",
        choices=("no_context", "random_context", "max_height"),
    )
    parser.add_argument(
        "--init", default="pretrained", choices=("pretrained", "scratch")
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--checkpoint", default="checkpoint-1000000")
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--figure-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-genotype", default=None)
    parser.add_argument("--target-environment", default=None)
    parser.add_argument("--datasets-offline-path", default=None)
    args = parser.parse_args()

    splits = Fip1Facts().test_splits
    observed = {}
    for split in splits:
        observed |= load_observed_plots(split, args.datasets_offline_path)
    anchors = fip1_day_grid(args.datasets_offline_path).anchors
    logger.info("Shared day grid spans {} to {}", anchors[0], anchors[-1])

    with tempfile.TemporaryDirectory() as pooled_dir:
        split_roots = {
            split: model_roots(
                args.results_dir,
                split,
                args.models,
                args.init,
                args.seed,
                args.checkpoint,
            )
            for split in splits
        }
        roots = pooled_roots(split_roots, args.method, Path(pooled_dir))
        conditions = plot_conditions(Path(roots[-1]) / "env_geno" / args.method)
        target_genotype, target_environment = default_targets(conditions, observed)
        plot_conditioning_sample_comparison(
            model_roots=roots,
            model_names=args.model_names or args.models,
            method=args.method,
            output_folder=args.output_folder,
            reference_model_index=-1,
            num_batches=args.num_batches,
            num_samples=args.num_samples,
            seed=args.figure_seed,
            target_genotype=args.target_genotype or target_genotype,
            target_environment=args.target_environment or target_environment,
            sample_alpha=SAMPLE_ALPHA,
            sample_linewidth=SAMPLE_LINEWIDTH,
            row_labels=list(ROW_LABELS),
            truth_source=ObservedTruth(observed),
            x_limits=(int(anchors[0]), int(anchors[-1])),
        )


if __name__ == "__main__":
    main()
