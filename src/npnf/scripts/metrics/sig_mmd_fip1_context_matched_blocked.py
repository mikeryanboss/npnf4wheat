"""Blocked context-matched Sig-MMD for FIP1 predictions.

The real-data sibling of ``sig_mmd_context_matched_blocked.py``. It shares that
scorer's covariate units (``UnitScope``), block construction
(``build_unit_blocks``), options (``BlockedScoringOptions``) and normalised
signature kernel; the reference side and the day axis differ.

On synthetic data the reference is the simulator: for a target with context ``c``
the scorer weights the candidate pool by how well each candidate's clean curve
explains ``c``, samples a candidate proportional to those weights, and takes one of
its 64 oracle draws. FIP1 has no simulator, so the reference is the *observed*
plots (``fip1_observed_plots``): the same posterior runs over the candidates'
measured heights, and every plot of the unit enters once, weighted by the
posterior averaged over the unit's targets (the sampler's expectation, see
``reference_side``). The pool includes the target, as ``candidate_pool`` does,
which resolves to the target's own cell at condition scope.

Four FIP1 specifics:

* Replicate plots share ``(genotype_id, yearsite_uid)``, so the unit member is the
  plot (``plot_uid``); ``UnitScope.unit_id`` still groups members into the four
  covariate scopes exactly as on the synthetic side.
* The environment of the E and E&G units is the harvest year, not the trial:
  2019 has two trials (FPWW024, FPWW028) with one temperature record, so a model
  gets the same environment input for both. Each plot still enters at its own
  trial's measurement days.
* One MMD per unit, pooling every yearsite the unit spans, because the unit is the
  estimand: a model conditioned on no covariates predicts the whole test
  population's marginal, so scoring it against a single harvest year would
  penalise the behaviour that is correct.
* Every path lies on the shared FIP1 day grid (``npnf.data.fip1_day_grid``), the
  same for every test set, so scores are comparable between test sets, a model
  with no covariates contributes one path and ``self_x`` is exactly 1.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

from npnf.data.batch_loader import BatchLoader
from npnf.data.fip1_day_grid import DAY_TOLERANCE, fip1_day_grid
from npnf.metrics.blocked_scoring import (
    BlockedScoringOptions,
    PredictionMode,
    load_prediction_day_axis,
)
from npnf.metrics.blocks import UnitMember, UnitScope, build_unit_blocks
from npnf.metrics.sig_mmd import weighted_sig_mmd
from npnf.metrics.utils import std_and_se, summary_value
from npnf.scripts.metrics.fip1_metric_grid import (
    day_offsets,
    heights_at_days,
    metric_paths,
    model_curves_at_days,
)
from npnf.scripts.metrics.fip1_observed_plots import (
    context_weights,
    load_observed_plots,
    load_records,
    split_from_dataloader_name,
)


def weighted_sig_mmd_x1000(
    model_paths: torch.Tensor, reference_paths: torch.Tensor, weights: np.ndarray
) -> float:
    """``weighted_sig_mmd`` in the reported x1000 units."""
    return 1000.0 * weighted_sig_mmd(model_paths, reference_paths, weights)


def reference_side(
    members: list[UnitMember],
    records: list[dict],
    assigned: dict[str, np.ndarray],
    anchors: np.ndarray,
    sigma: float,
) -> tuple[list[UnitMember], torch.Tensor, np.ndarray, float]:
    """Observed paths of one unit, with the weight the context matching gives them.

    Every plot of the unit enters once, weighted by the posterior over candidates.
    The synthetic scorer *samples* candidates instead, which it can afford because
    a resampled candidate still contributes a distinct oracle draw. A FIP1 plot has
    one realisation, so sampling N slots from N plots would duplicate trajectories.
    The weighted form is that sampler's expectation with no sampling variance.
    """
    ordered = [
        member
        for yearsite in sorted({member.yearsite_uid for member in members})
        for member in members
        if member.yearsite_uid == yearsite
    ]
    paths = metric_paths(
        np.stack(
            [
                heights_at_days(
                    records[member.condition_index], assigned[member.yearsite_uid]
                )
                for member in ordered
            ]
        ),
        anchors,
    )
    pool_records = [records[member.condition_index] for member in ordered]
    posteriors = np.stack(
        [
            context_weights(
                pool_records,
                records[member.condition_index]["context_days"],
                records[member.condition_index]["context_values"],
                sigma,
            )
            for member in ordered
        ]
    )
    # Mixture over targets: what the per-target sampler draws from, in expectation.
    effective_size = float(np.mean(1.0 / (posteriors**2).sum(axis=1)))
    return ordered, paths, posteriors.mean(axis=0), effective_size


def score_unit(
    members: list[UnitMember],
    records: list[dict],
    day_axis: np.ndarray,
    anchors: np.ndarray,
    assigned: dict[str, np.ndarray],
    *,
    draw_count: int,
    n_blocks: int,
    block_size: int,
    seed: int,
    unit_scope: UnitScope,
    unit_id: str,
    sigma: float,
    min_reference_plots: int,
) -> list[dict]:
    """Score one covariate unit, pooling all of its yearsites into one estimate.

    The unit is the estimand: a model conditioned on no covariates is compared
    against the whole test population, one conditioned on a genotype against that
    genotype in every year it appears. Every path lies on the split's aligned grid,
    so a unit spanning several yearsites is one rectangular tensor.
    """
    members = [member for member in members if member.yearsite_uid in assigned]
    if len(members) < min_reference_plots:
        return []

    ordered, reference_paths, reference_weights, effective_size = reference_side(
        members, records, assigned, anchors, sigma
    )
    yearsite_order = sorted({member.yearsite_uid for member in members})
    offsets = day_offsets(anchors, {ys: assigned[ys] for ys in yearsite_order})

    # A unit with at most one block of draws puts all of them into every block, and
    # the reference is fixed, so every block would give the same score.
    if draw_count * len(ordered) <= block_size:
        n_blocks = 1
    model_blocks = build_unit_blocks(
        ordered,
        draws_per_member=draw_count,
        unit_scope=unit_scope,
        unit_id=unit_id,
        n_blocks=n_blocks,
        block_size=block_size,
        seed=seed,
    )

    # The model side sits on the anchors themselves, never on a yearsite's assigned
    # days: that is what keeps one deterministic curve a single path.
    rows = []
    for block_index, model_block in enumerate(model_blocks):
        model_curves = torch.stack(
            [
                model_curves_at_days(
                    records[selection.condition_index]["grid"], day_axis, anchors
                )[selection.draw_index]
                for selection in model_block
            ]
        )
        rows.append(
            {
                "unit_scope": unit_scope,
                "unit_id": unit_id,
                "yearsite_uids": "|".join(yearsite_order),
                "block_index": block_index,
                "n_yearsites": len(yearsite_order),
                "n_days": anchors.size,
                "mean_day_offset": float(offsets.mean()),
                "max_day_offset": int(offsets.max()),
                "n_members": len(members),
                "n_model": model_curves.shape[0],
                "n_reference": len(ordered),
                "context_ess": effective_size,
                "sig_mmd_x1000": weighted_sig_mmd_x1000(
                    metric_paths(model_curves, anchors),
                    reference_paths,
                    reference_weights,
                ),
            }
        )
    return rows


def split_units(
    records: list[dict], unit_scope: UnitScope
) -> dict[str, list[UnitMember]]:
    units: dict[str, list[UnitMember]] = defaultdict(list)
    for condition_index, record in enumerate(records):
        units[unit_scope.unit_id(record["gid"], record["environment"])].append(
            UnitMember(
                genotype_id=record["gid"],
                yearsite_uid=record["ys"],
                condition_index=condition_index,
            )
        )
    return units


def write_outputs(
    output_path: Path, block_rows: list[dict], summary_fields: dict
) -> None:
    blocks = pl.DataFrame(block_rows)
    blocks.write_csv(output_path / "sig_mmd_blocks.csv")

    per_unit = (
        blocks.group_by("unit_scope", "unit_id")
        .agg(
            pl.col("sig_mmd_x1000").mean().alias("sig_mmd_x1000"),
            pl.col("sig_mmd_x1000").len().alias("n_blocks"),
            pl.col("n_yearsites").first(),
            pl.col("n_days").first(),
            pl.col("mean_day_offset").first(),
            pl.col("n_members").first(),
            pl.col("n_reference").sum(),
            pl.col("context_ess").mean(),
        )
        .sort("unit_id")
    )
    per_unit.write_csv(output_path / "sig_mmd_per_unit.csv")

    std, standard_error = std_and_se(per_unit["sig_mmd_x1000"].to_list())
    summary = pl.DataFrame(
        [
            {
                **summary_fields,
                "n_units": per_unit.height,
                "n_blocks": blocks.height,
                "sig_mmd_x1000": summary_value(per_unit["sig_mmd_x1000"], "mean"),
                "sig_mmd_x1000_median": summary_value(
                    per_unit["sig_mmd_x1000"], "median"
                ),
                "sig_mmd_x1000_std": std,
                "sig_mmd_x1000_se": standard_error,
            }
        ]
    )
    summary.write_csv(output_path / "sig_mmd_summary.csv")
    logger.info(
        "Wrote {} ({} units, {} blocks, Sig-MMD x1000 = {:.3f})",
        output_path / "sig_mmd_summary.csv",
        per_unit.height,
        blocks.height,
        summary["sig_mmd_x1000"][0],
    )


def sig_mmd_fip1_context_matched_blocked_analysis(
    method_dir: str,
    output_folder: str | None = None,
    *,
    options: BlockedScoringOptions | None = None,
    context_sigma: float = 0.02,
    min_reference_plots: int = 1,
    datasets_offline_path: str | None = None,
    output_folder_name: str = "sig_mmd_fip1_context_matched_blocked",
) -> None:
    """Score one FIP1 method directory with the synthetic scorer's block options.

    ``context_sigma`` is FIP1's measured height residual scale, rounded; the
    simulator's own ``scale_noise=0.05`` is 2-3x too wide and flattens the
    posterior. One reference plot is enough, because the kernel score against a
    point mass is well defined.
    """
    if options is None:
        options = BlockedScoringOptions()
    method_path = Path(method_dir)
    mode = method_path.name
    if mode not in set(PredictionMode):
        msg = f"method directory must be named one of {sorted(PredictionMode)}"
        raise ValueError(msg)

    loader = BatchLoader(method_path)
    unit_scope = UnitScope.from_config(loader.config, method_path)
    day_axis = load_prediction_day_axis(loader).numpy().astype(int)
    split = split_from_dataloader_name(loader.config["dataloader_name"])  # ty: ignore[not-subscriptable]
    output_path = (
        Path(output_folder)
        if output_folder is not None
        else method_path / output_folder_name
    )
    output_path.mkdir(parents=True, exist_ok=True)

    observed = load_observed_plots(split, datasets_offline_path)
    records, saved_draws = load_records(loader, mode, observed)
    draw_count = min(options.n_samples, saved_draws)
    logger.info(
        "mode={}, unit_scope={}, split={}, plots={}, model draws={}, grid={} days",
        mode,
        unit_scope,
        split,
        len(records),
        draw_count,
        day_axis.size,
    )

    units = split_units(records, unit_scope)
    grid = fip1_day_grid(datasets_offline_path)
    anchors, assigned = grid.anchors, grid.assigned
    n_blocks = options.blocks_for_scope(unit_scope)

    start = time.perf_counter()
    block_rows: list[dict] = []
    for unit_id in tqdm(sorted(units), unit="unit", desc="fip1-blocked: score"):
        block_rows.extend(
            score_unit(
                units[unit_id],
                records,
                day_axis,
                anchors,
                assigned,
                draw_count=draw_count,
                n_blocks=n_blocks,
                block_size=options.block_size,
                seed=options.seed,
                unit_scope=unit_scope,
                unit_id=unit_id,
                sigma=context_sigma,
                min_reference_plots=min_reference_plots,
            )
        )
    if not block_rows:
        msg = f"No unit of {method_path} had {min_reference_plots} plots to score"
        raise ValueError(msg)

    offsets = day_offsets(anchors, assigned)
    write_outputs(
        output_path,
        block_rows,
        {
            "method_dir": str(method_path),
            "mode": mode,
            "unit_scope": unit_scope,
            "split": split,
            "environment_unit": "harvest_year",
            "n_plots": len(records),
            "model_draws": draw_count,
            "context_sigma": context_sigma,
            "day_tolerance": DAY_TOLERANCE,
            "n_anchor_days": int(anchors.size),
            "anchor_first_day": int(anchors.min()),
            "anchor_last_day": int(anchors.max()),
            "mean_day_offset": float(offsets.mean()),
            "max_day_offset": int(offsets.max()),
            "block_size": options.block_size,
            "seed": options.seed,
            "elapsed_seconds": time.perf_counter() - start,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method_dirs", nargs="+")
    parser.add_argument("--output-folder", default=None)
    BlockedScoringOptions.add_arguments(parser)
    parser.add_argument("--context-sigma", type=float, default=0.02)
    parser.add_argument("--min-reference-plots", type=int, default=1)
    parser.add_argument("--datasets-offline-path", default=None)
    parser.add_argument(
        "--output-folder-name", default="sig_mmd_fip1_context_matched_blocked"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_folder is not None and len(args.method_dirs) > 1:
        msg = "--output-folder only applies to a single method directory"
        raise ValueError(msg)
    for method_dir in args.method_dirs:
        sig_mmd_fip1_context_matched_blocked_analysis(
            method_dir,
            args.output_folder,
            options=BlockedScoringOptions.from_args(args),
            context_sigma=args.context_sigma,
            min_reference_plots=args.min_reference_plots,
            datasets_offline_path=args.datasets_offline_path,
            output_folder_name=args.output_folder_name,
        )


if __name__ == "__main__":
    main()
