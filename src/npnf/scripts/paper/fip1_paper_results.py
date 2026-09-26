"""Fold the FIP1 Sig-MMD summaries into a synthetic-shaped results tree.

``mmd_table.py`` and ``mmd_plot.py`` build the paper's synthetic table and point
plot by walking
``<dataloader>/<model>/<seed>/<checkpoint>/<split>/<conditioning>/<mode>/<metric
dir>/sig_mmd_summary.csv`` and reading one metric column per cell. The FIP1
campaign writes the same tree, except that the training seed sits in the run
directory name, the metric directory carries the FIP1 scorer's name and the score
column is already scaled by 1000.

This script rewrites those summaries into the synthetic layout and column names,
folding the five training seeds of a cell into their mean, so the paper scripts
read the FIP1 scores with no change other than the FIP1 split names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

from npnf.scripts.paper.fip1_sig_mmd_table import collect

TARGET_METRIC_DIR = "sig_mmd_context_matched_blocked"
SEED_DIR = "seeds"
CELL = ("dataloader", "model", "init", "checkpoint", "split", "conditioning", "mode")


def fold_seeds(scores: pl.DataFrame) -> pl.DataFrame:
    """Mean over the training seeds of each cell, in the synthetic column names.

    ``mean``, ``median`` and ``std`` are the synthetic quantities in raw units:
    the cell's score, its median over units and its sd over units. ``sd`` is the
    spread the letter reports, the sd over training seeds.
    """
    return (
        scores.group_by(CELL)
        .agg(
            pl.col("seed").n_unique().alias("n_seeds"),
            pl.col("seed").sort().cast(pl.String).str.join(",").alias("seeds"),
            pl.col("n_units").first(),
            pl.col("n_blocks").first(),
            pl.col("n_plots").first(),
            pl.col("day_tolerance").first(),
            pl.col("n_anchor_days").first(),
            (pl.col("sig_mmd_x1000").mean() / 1000.0).alias("mean"),
            (pl.col("sig_mmd_x1000_median").mean() / 1000.0).alias("median"),
            (pl.col("sig_mmd_x1000_std").mean() / 1000.0).alias("std"),
            (pl.col("sig_mmd_x1000").std(ddof=1) / 1000.0).alias("sd"),
            pl.col("sig_mmd_x1000").mean().alias("mean_x1000"),
            pl.col("sig_mmd_x1000").std(ddof=1).alias("sd_x1000"),
        )
        .with_columns(
            pl.lit("context_matched_blocked").alias("view"),
            pl.lit("blocked").alias("estimator"),
        )
        .sort(CELL)
    )


def model_dir_name(model: str, init: str | None) -> str:
    """Model folder of the aggregated tree, the `--models` value of the table.

    A baseline has no initialisation and keeps its run name.
    """
    return model if init is None else f"{model}-fip1_{init}"


def cell_path(row: dict, output_dir: Path) -> Path:
    return (
        output_dir
        / row["dataloader"]
        / model_dir_name(row["model"], row["init"])
        / SEED_DIR
        / row["checkpoint"]
        / row["split"]
        / row["conditioning"]
        / row["mode"]
        / TARGET_METRIC_DIR
        / "sig_mmd_summary.csv"
    )


def write_tree(folded: pl.DataFrame, output_dir: Path) -> int:
    for row in folded.to_dicts():
        path = cell_path(row, output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([row]).write_csv(path)
    return folded.height


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--baseline-runs",
        nargs="*",
        default=[],
        help="Run directories of fitted baselines to fold in as one-seed models, "
        "under their run name (e.g. LodgingMixtureSpline-FIP-2930).",
    )
    args = parser.parse_args()

    scores = collect(args.source, args.baseline_runs)
    folded = fold_seeds(scores)
    written = write_tree(folded, args.output_dir)
    folded.write_csv(args.output_dir / "fip1_folded_cells.csv")
    logger.info(
        "cells={} from {} scored directories, models={}, seeds per cell {}-{}",
        written,
        scores.height,
        folded["model"].n_unique(),
        folded["n_seeds"].min(),
        folded["n_seeds"].max(),
    )


if __name__ == "__main__":
    main()
