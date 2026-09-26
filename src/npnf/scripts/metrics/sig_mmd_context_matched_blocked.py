"""Blocked context-matched Sig-MMD for all synthetic prediction modes.

This scorer combines a fixed block budget with a context-aware oracle
reference. Within each covariate unit it estimates the pooled model
distribution against the pooled context-matched oracle reference using a fixed
number of MMD blocks, so the number of (expensive) signature-kernel evaluations
is bounded by the configured block counts rather than the number of conditions.

All three method modes (``no_context``, ``random_context``, ``max_height``) and
all four covariate scopes (``global``, ``environment``, ``genotype``,
``condition``) flow through one unified path:
``no_context`` is the degenerate case where the reconstructed context is empty
and ``context_weights`` returns uniform weights over the candidate pool.

With ``--mahalanobis-artifact`` the scorer also writes the blocked CSig-MMD
results. Each block's Gram matrices give both scores. The oracle blocks and the
block plans do not change; the precomputed ``oracle_signature_mahalanobis/``
artifact adds the global MRCD censoring rule (context matching changes only the
oracle reference distribution, not the censoring rule).
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from loguru import logger
from tqdm import tqdm

from npnf.data.batch_loader import BatchLoader
from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.data.fip1_day_grid import align_trajectories_to_day_axis
from npnf.metrics.blocked_scoring import (
    BlockedScoringOptions,
    ContextRecord,
    ModelBank,
    PredictionMode,
    aggregate_diagnostics,
    build_id_to_index,
    candidate_pool,
    extract_metadata,
    group_units,
    load_prediction_day_axis,
    load_records_and_bank,
    pool_key,
    posterior_tile_or_uniform,
    reconstruct_context,
    selection_tensors,
    stable_seed,
)
from npnf.metrics.blocks import Selection, UnitScope, build_unit_blocks
from npnf.metrics.csig_mmd import (
    Censoring,
    CensoringParameters,
    CensoringReference,
    CensoringSweep,
    load_censoring_reference,
)
from npnf.metrics.oracle_config import (
    get_oracle_config_for_method_dir,
    get_seed_b_config_for_dataloader_name,
)
from npnf.metrics.sig_mmd import normalized_sig_mmd
from npnf.metrics.signature import (
    SYNTHETIC_SIGNATURE_HEIGHT_SCALE,
    kernel_sigma_suffix,
    to_metric_paths,
)
from npnf.metrics.utils import positive_int, quantile_value, std_and_se, summary_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blocked context-matched Sig-MMD for no_context, random_context, and "
            "max_height synthetic predictions."
        )
    )
    parser.add_argument(
        "--method-dir",
        type=str,
        required=True,
        help=(
            "Path to a no_context, random_context, or max_height method directory "
            "with predictions/."
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help=(
            "Output directory. Normal default: "
            "<method-dir>/sig_mmd_context_matched_blocked. Oracle-self default: "
            "<split-dir>/sig_mmd_context_matched_blocked_oracle_self/"
            "<conditioning>/<mode> when the method dir is under a conditioning "
            "directory, otherwise <method-parent>/"
            "sig_mmd_context_matched_blocked_oracle_self/<mode>."
        ),
    )
    parser.add_argument(
        "--oracle-load-chunk-size",
        type=positive_int,
        default=1024,
        help=(
            "Number of conditions per oracle CPU loading chunk (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--oracle-self",
        action="store_true",
        help=(
            "Estimate the blocked context-matched oracle noise floor by comparing "
            "SeedB context-matched oracle samples against SeedA context-matched "
            "oracle samples. The method directory supplies config, grid, target "
            "rows, and context metadata; saved predictions.pt files are not read."
        ),
    )
    parser.add_argument(
        "--no-lead-lag",
        action="store_true",
        help="Disable lead-lag augmentation and use raw height-plus-time paths.",
    )
    parser.add_argument(
        "--mahalanobis-artifact",
        type=str,
        default=None,
        help=(
            "oracle_signature_mahalanobis artifact directory. When given, the "
            "scorer also writes CSig-MMD results from the same Gram matrices."
        ),
    )
    parser.add_argument(
        "--csig-output-folder",
        type=str,
        default=None,
        help=(
            "CSig-MMD output directory "
            "(default: <method-dir>/csig_mmd_context_matched_blocked)."
        ),
    )
    parser.add_argument(
        "--self-kernel-batch-size",
        type=positive_int,
        default=512,
        help=(
            "Batch size for model signature self-kernel norms of CSig-MMD "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--kernel-sigma",
        type=float,
        default=1.0,
        help=(
            "Parameter sigma of the RBF static kernel exp(-||x - y||^2 / sigma) of "
            "the Sig-MMD and CSig-MMD Gram matrices. Other values than the default "
            "add _sigma=<value> to the default output folders (default: %(default)s)."
        ),
    )
    CensoringParameters.add_arguments(parser)
    CensoringSweep.add_arguments(parser)
    BlockedScoringOptions.add_arguments(parser)
    return parser


def _default_output_path(
    method_path: Path, *, oracle_self: bool, kernel_sigma: float = 1.0
) -> Path:
    suffix = kernel_sigma_suffix(kernel_sigma)
    if not oracle_self:
        return method_path / f"sig_mmd_context_matched_blocked{suffix}"
    conditioning_dir_names = {"noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"}
    if method_path.parent.name in conditioning_dir_names:
        return (
            method_path.parent.parent
            / f"sig_mmd_context_matched_blocked_oracle_self{suffix}"
            / method_path.parent.name
            / method_path.name
        )
    return (
        method_path.parent
        / f"sig_mmd_context_matched_blocked_oracle_self{suffix}"
        / method_path.name
    )


def _batch_dict_from_loader_item(batch: tuple | dict) -> dict:
    if isinstance(batch, tuple):
        _, batch_dict = batch
        return batch_dict
    return batch


def _load_batch_day_axis(loader: BatchLoader) -> torch.Tensor:
    """Read the saved evaluation grid from batch metadata.

    Unlike ``load_prediction_day_axis``, this also works when the loader was
    constructed with ``load_predictions=False`` and therefore returns only
    ``batch.pt`` dictionaries.
    """
    batch_dict = _batch_dict_from_loader_item(loader.load_batch(0))
    return batch_dict["grid_points"]["X"].squeeze(-1).float()


def _context_sampling_seed(
    seed: int, mode: str, unit_scope: str, gid: str, ys: str, rng_namespace: str | None
) -> int:
    """Stable per-target candidate-sampling seed.

    ``rng_namespace=None`` preserves the existing normal model-vs-oracle seed
    payload exactly. Oracle-self uses side-specific namespaces so SeedB/model-like
    and SeedA/reference candidate draws are independent while remaining
    deterministic for a fixed base seed.
    """
    if rng_namespace is None:
        return stable_seed(seed, mode, unit_scope, gid, ys)
    payload = f"{seed}\0{rng_namespace}\0{mode}\0{unit_scope}\0{gid}\0{ys}".encode()
    digest = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(digest, "little", signed=False)


def _validate_seed_b_dataset_alignment(
    seed_a_dataset: SyntheticDataset,
    seed_b_dataset: SyntheticDataset,
    seed_a_id_to_index: dict[tuple[str, str], int],
    seed_b_id_to_index: dict[tuple[str, str], int],
) -> None:
    """Validate SeedB can safely share SeedA flat candidate-pool indices."""
    missing_in_b = sorted(set(seed_a_id_to_index) - set(seed_b_id_to_index))
    missing_in_a = sorted(set(seed_b_id_to_index) - set(seed_a_id_to_index))
    if missing_in_b or missing_in_a:
        parts: list[str] = []
        if missing_in_b:
            gid, ys = missing_in_b[0]
            parts.append(
                "SeedB is missing SeedA condition "
                f"(genotype_id={gid!r}, yearsite_uid={ys!r})"
            )
        if missing_in_a:
            gid, ys = missing_in_a[0]
            parts.append(
                "SeedA is missing SeedB condition "
                f"(genotype_id={gid!r}, yearsite_uid={ys!r})"
            )
        raise ValueError("; ".join(parts))

    for key in sorted(seed_a_id_to_index):
        seed_a_index = seed_a_id_to_index[key]
        seed_b_index = seed_b_id_to_index[key]
        if seed_a_index != seed_b_index:
            gid, ys = key
            msg = (
                "SeedA and SeedB flat-index ordering differs for condition "
                f"(genotype_id={gid!r}, yearsite_uid={ys!r}): "
                f"SeedA index={seed_a_index}, SeedB index={seed_b_index}. "
                "Oracle-self context-matched sampling requires matching ordering "
                "because candidate pools are expressed as dataset flat indices."
            )
            raise ValueError(msg)

    if seed_a_dataset.num_genotypes != seed_b_dataset.num_genotypes:
        msg = (
            "SeedA and SeedB num_genotypes differ: "
            f"{seed_a_dataset.num_genotypes} vs {seed_b_dataset.num_genotypes}"
        )
        raise ValueError(msg)
    if seed_a_dataset.num_yearsites != seed_b_dataset.num_yearsites:
        msg = (
            "SeedA and SeedB num_yearsites differ: "
            f"{seed_a_dataset.num_yearsites} vs {seed_b_dataset.num_yearsites}"
        )
        raise ValueError(msg)
    seed_a_days = seed_a_dataset.get_height_days_all().float()
    seed_b_days = seed_b_dataset.get_height_days_all().float()
    if seed_a_days.shape != seed_b_days.shape or not bool(
        torch.equal(seed_a_days, seed_b_days)
    ):
        msg = "SeedA and SeedB height day axes differ"
        raise ValueError(msg)


def _load_oracle_self_records(
    loader: BatchLoader,
    *,
    seed_a_id_to_index: dict[tuple[str, str], int],
    seed_b_id_to_index: dict[tuple[str, str], int],
    mode: str,
    clean: np.ndarray,
    noise: np.ndarray,
    sub_full_idx: np.ndarray,
) -> tuple[tuple[tuple[str, str], ...], dict[int, ContextRecord]]:
    """Stream batch metadata/context records without reading saved predictions."""
    keys: list[tuple[str, str]] = []
    row_contexts: list[ContextRecord] = []

    for batch in tqdm(
        loader,
        total=len(loader),
        unit="batch",
        desc="context-matched-blocked: load metadata",
    ):
        batch_dict = _batch_dict_from_loader_item(batch)
        gids, yss = extract_metadata(batch_dict)

        for row_index, (gid, ys) in enumerate(zip(gids, yss, strict=True)):
            key = (gid, ys)
            try:
                u0 = seed_a_id_to_index[key]
            except KeyError as error:
                msg = (
                    "SeedA oracle dataset does not contain prediction condition "
                    f"(genotype_id={gid!r}, yearsite_uid={ys!r})"
                )
                raise ValueError(msg) from error
            try:
                seed_b_u0 = seed_b_id_to_index[key]
            except KeyError as error:
                msg = (
                    "SeedB oracle dataset does not contain prediction condition "
                    f"(genotype_id={gid!r}, yearsite_uid={ys!r})"
                )
                raise ValueError(msg) from error
            if seed_b_u0 != u0:
                msg = (
                    "SeedA and SeedB flat-index ordering differs for prediction "
                    f"condition (genotype_id={gid!r}, yearsite_uid={ys!r}): "
                    f"SeedA index={u0}, SeedB index={seed_b_u0}"
                )
                raise ValueError(msg)

            ctx_full_idx, ctx_vals = reconstruct_context(
                mode, batch_dict, row_index, clean[u0], noise[u0], sub_full_idx
            )
            keys.append(key)
            row_contexts.append(
                {
                    "u0": u0,
                    "gid": gid,
                    "ys": ys,
                    "ctx_full_idx": ctx_full_idx,
                    "ctx_vals": ctx_vals,
                }
            )

    if not keys:
        msg = f"No prediction rows found under {loader.method_dir}"
        raise ValueError(msg)

    seen: dict[tuple[str, str], int] = {}
    for index, key in enumerate(keys):
        if key in seen:
            msg = (
                "Duplicate prediction condition in method directory: "
                f"genotype_id={key[0]!r}, yearsite_uid={key[1]!r}"
            )
            raise ValueError(msg)
        seen[key] = index

    sorted_keys = tuple(sorted(keys))
    record_by_model_row = {
        condition_index: row_contexts[seen[key]]
        for condition_index, key in enumerate(sorted_keys)
    }
    return sorted_keys, record_by_model_row


def _load_oracle_draws(
    dataset: SyntheticDataset,
    needed_pairs: set[tuple[int, int]],
    *,
    oracle_day_axis: torch.Tensor,
    prediction_day_axis: torch.Tensor,
    chunk_size: int,
) -> dict[tuple[int, int], torch.Tensor]:
    """Load only the requested ``(condition, draw_index)`` trajectories, grid-aligned.

    ``SyntheticDataset.get_lodged_chunk`` takes one shared draw-index list per call, so
    conditions are grouped by the exact set of draw indices they need; each group is
    loaded in chunks of ``chunk_size`` conditions. Loading a draw subset is identical to
    loading every draw and slicing, so scores are unchanged. When every candidate needs
    every draw this degenerates to one group and loads exactly what a dense load would.
    """
    draws_by_condition: dict[int, set[int]] = defaultdict(set)
    for condition, draw_index in needed_pairs:
        draws_by_condition[condition].add(draw_index)

    conditions_by_drawset: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for condition, draw_set in draws_by_condition.items():
        conditions_by_drawset[tuple(sorted(draw_set))].append(condition)

    loaded: dict[tuple[int, int], torch.Tensor] = {}
    for draw_tuple, conditions in conditions_by_drawset.items():
        draw_list = list(draw_tuple)
        for start in range(0, len(conditions), chunk_size):
            chunk_conditions = sorted(conditions[start : start + chunk_size])
            chunk = dataset.get_lodged_chunk(chunk_conditions, draw_list)
            aligned = align_trajectories_to_day_axis(
                chunk["height_values_all_nonoise"].float(),
                oracle_day_axis,
                prediction_day_axis,
            )
            for offset, condition in enumerate(chunk_conditions):
                for axis, draw_index in enumerate(draw_list):
                    loaded[(condition, draw_index)] = aligned[offset, axis]
    return loaded


def _resolve_unit_oracle_blocks(
    oracle_plan_blocks: Sequence[Sequence[Selection]],
    record_by_model_row: dict[int, ContextRecord],
    *,
    unit_scope: UnitScope,
    mode: str,
    seed: int,
    clean: np.ndarray,
    num_genotypes: int,
    num_yearsites: int,
    sigma: float,
    days_np: np.ndarray,
    peak_days: np.ndarray,
    dataset: SyntheticDataset,
    oracle_day_axis: torch.Tensor,
    prediction_day_axis: torch.Tensor,
    chunk_size: int,
    context_tile_size: int,
    rng_namespace: str | None = None,
) -> tuple[
    list[torch.Tensor],
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[int, float, float]],
    dict[str, float],
]:
    """Build per-block oracle heights, per-target diagnostics, and timing for one unit.

    Oracle selections are grouped by target model-bank row. Context posteriors are
    computed once per target, vectorized across all targets sharing a candidate pool
    (see ``tile_posterior``); the selections targeting a target draw candidate
    conditions ``∝ w`` with a target-stable seed, then take the named lodging
    draw of the sampled candidate. Diagnostics are aggregated over the distinct target
    rows the oracle plan actually selected. The third return value is a profiling dict
    with elapsed seconds and counts for this unit. ``rng_namespace`` is only used by
    oracle-self to make the SeedB/model-like and SeedA/reference candidate samplers
    independent; ``None`` preserves the existing normal scorer seed payload. The
    second return value holds, per block, the sampled dataset condition and draw
    index of each oracle row.
    """
    refs_by_target: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for block_index, block in enumerate(oracle_plan_blocks):
        for position, selection in enumerate(block):
            refs_by_target[selection.condition_index].append(
                (block_index, position, selection.draw_index)
            )

    assignment: dict[tuple[int, int], tuple[int, int]] = {}
    diagnostics: list[tuple[int, float, float]] = []
    needed: set[int] = set()
    context_weights_seconds = 0.0
    pool_size = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Targets sharing a candidate pool (global/environment/genotype) or a singleton
    # (condition) reuse one clean[pool]/peak_days[pool] gather, and their context
    # posteriors are computed together: the per-target SSE over the pool is
    # vectorized into two matmuls (expand-the-square) and tiled over targets, on
    # GPU when available. Reordering targets by pool is safe — each target's RNG,
    # diagnostics, and assignment are independent of processing order.
    targets_by_pool: dict[int, list[int]] = defaultdict(list)
    for condition_index in sorted(refs_by_target):
        u0 = int(record_by_model_row[condition_index]["u0"])
        targets_by_pool[pool_key(unit_scope, u0, num_yearsites)].append(condition_index)

    for cond_list in targets_by_pool.values():
        u0 = int(record_by_model_row[cond_list[0]]["u0"])
        cw_start = time.perf_counter()
        pool = candidate_pool(unit_scope, u0, num_genotypes, num_yearsites)
        peak_pool = peak_days[pool]
        pool_size = len(pool)
        ctx_idx_list = [record_by_model_row[ci]["ctx_full_idx"] for ci in cond_list]
        ctx_vals_list = [record_by_model_row[ci]["ctx_vals"] for ci in cond_list]
        pool_has_context = any(np.asarray(idx).size > 0 for idx in ctx_idx_list)
        clean_pool_t: torch.Tensor | None = None
        clean2_pool_t: torch.Tensor | None = None
        if pool_has_context:
            clean_pool_t = (
                torch.as_tensor(clean[pool], dtype=torch.float64, device=device)
                .t()
                .contiguous()
            )
            clean2_pool_t = clean_pool_t * clean_pool_t
        uniform_weights: np.ndarray | None = None
        context_weights_seconds += time.perf_counter() - cw_start

        for start in range(0, len(cond_list), context_tile_size):
            stop = min(start + context_tile_size, len(cond_list))
            tile_ctx_idx = ctx_idx_list[start:stop]
            tile_ctx_vals = ctx_vals_list[start:stop]
            cw_start = time.perf_counter()
            weights_tile, ess_tile, uniform_weights = posterior_tile_or_uniform(
                clean_pool_t,
                clean2_pool_t,
                tile_ctx_idx,
                tile_ctx_vals,
                sigma,
                device,
                pool_size,
                uniform_weights,
            )
            context_weights_seconds += time.perf_counter() - cw_start

            for offset, condition_index in enumerate(cond_list[start:stop]):
                record = record_by_model_row[condition_index]
                gid = str(record["gid"])
                ys = str(record["ys"])
                ctx_full_idx = record["ctx_full_idx"]
                weights = (
                    uniform_weights if weights_tile is None else weights_tile[offset]
                )
                assert weights is not None
                ess = ess_tile[offset]
                if ctx_full_idx.size == 0:
                    post_peak_mass = 0.0
                else:
                    latest_context_day = float(days_np[ctx_full_idx].max())
                    post_peak_mass = float(
                        weights[peak_pool < latest_context_day].sum()
                    )
                diagnostics.append((int(ctx_full_idx.size), float(ess), post_peak_mass))

                refs = sorted(
                    refs_by_target[condition_index], key=lambda ref: (ref[0], ref[1])
                )
                rng = np.random.default_rng(
                    _context_sampling_seed(
                        seed, mode, unit_scope, gid, ys, rng_namespace
                    )
                )
                sampled = rng.choice(pool, size=len(refs), p=weights)
                for (block_index, position, draw_index), candidate in zip(
                    refs, sampled, strict=True
                ):
                    candidate = int(candidate)
                    assignment[(block_index, position)] = (candidate, int(draw_index))
                    needed.add(candidate)

    needed_pairs = set(assignment.values())
    oracle_load_start = time.perf_counter()
    loaded = _load_oracle_draws(
        dataset,
        needed_pairs,
        oracle_day_axis=oracle_day_axis,
        prediction_day_axis=prediction_day_axis,
        chunk_size=chunk_size,
    )
    oracle_load_seconds = time.perf_counter() - oracle_load_start

    oracle_blocks: list[torch.Tensor] = []
    block_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for block_index, block in enumerate(oracle_plan_blocks):
        pairs = [assignment[(block_index, position)] for position in range(len(block))]
        oracle_blocks.append(torch.stack([loaded[pair] for pair in pairs], dim=0))
        block_pairs.append(
            (
                np.array([pair[0] for pair in pairs], dtype=np.int64),
                np.array([pair[1] for pair in pairs], dtype=np.int64),
            )
        )

    profile = {
        "context_weights_seconds": context_weights_seconds,
        "oracle_load_seconds": oracle_load_seconds,
        "n_targets": float(len(refs_by_target)),
        "pool_size": float(pool_size),
        "n_unique_candidates": float(len(needed)),
        "n_loaded_pairs": float(len(loaded)),
    }
    return oracle_blocks, block_pairs, diagnostics, profile


def write_outputs(
    output_path: Path,
    *,
    mode: str,
    block_rows: list[dict[str, object]],
    per_unit_rows: list[dict[str, object]],
    unit_member_rows: list[dict[str, object]],
    options: BlockedScoringOptions,
    model_draw_count: int,
    oracle_draw_count: int,
    oracle_self: bool,
) -> None:
    blocks_df = pl.DataFrame(block_rows).sort(["unit_scope", "unit_id", "block_index"])
    per_unit_df = pl.DataFrame(per_unit_rows).sort(["unit_scope", "unit_id"])
    unit_members_df = pl.DataFrame(unit_member_rows).sort(
        ["unit_scope", "unit_id", "genotype_id", "yearsite_uid"]
    )

    blocks_df.write_csv(output_path / "sig_mmd_blocks.csv")
    logger.info("Wrote {}", output_path / "sig_mmd_blocks.csv")
    per_unit_df.write_csv(output_path / "sig_mmd_per_unit.csv")
    logger.info("Wrote {}", output_path / "sig_mmd_per_unit.csv")
    unit_members_df.write_csv(output_path / "sig_mmd_unit_members.csv")
    logger.info("Wrote {}", output_path / "sig_mmd_unit_members.csv")

    scores = per_unit_df["score"]
    summary_data: dict[str, list[object]] = {
        "view": ["context_matched_blocked"],
        "estimator": ["blocked_oracle_self" if oracle_self else "blocked"],
        "mode": [mode],
        "block_size": [options.block_size],
        "global_blocks": [options.global_blocks],
        "environment_blocks": [options.environment_blocks],
        "genotype_blocks": [options.genotype_blocks],
        "condition_blocks": [options.condition_blocks],
        "seed": [options.seed],
        "requested_n_samples": [options.n_samples],
        "draw_count": [model_draw_count],
        "model_draw_count": [model_draw_count],
        "oracle_draw_count": [oracle_draw_count],
        "n_units": [per_unit_df.height],
        "n_blocks": [blocks_df.height],
        "mean": [summary_value(scores, "mean")],
        "median": [summary_value(scores, "median")],
        "std": [summary_value(scores, "std")],
        "p95": [quantile_value(scores, 0.95)],
        "mean_n_context": [summary_value(per_unit_df["mean_n_context"], "mean")],
        "mean_ess": [summary_value(per_unit_df["mean_ess"], "mean")],
        "mean_post_peak_mass": [
            summary_value(per_unit_df["mean_post_peak_mass"], "mean")
        ],
    }
    if oracle_self:
        summary_data.update(
            {
                "oracle_self": [True],
                "model_source": ["SeedB_context_matched_oracle"],
                "reference_source": ["SeedA_context_matched_oracle"],
            }
        )
    summary = pl.DataFrame(summary_data)
    summary.write_csv(output_path / "sig_mmd_summary.csv")
    logger.info("Wrote {}", output_path / "sig_mmd_summary.csv")


def write_csig_outputs(
    output_path: Path,
    *,
    mode: str,
    block_rows: list[dict[str, object]],
    per_unit_rows: list[dict[str, object]],
    unit_member_rows: list[dict[str, object]],
    options: BlockedScoringOptions,
    model_draw_count: int,
    oracle_draw_count: int,
    censoring: Censoring,
) -> None:
    blocks_df = pl.DataFrame(block_rows).sort(["unit_scope", "unit_id", "block_index"])
    per_unit_df = pl.DataFrame(per_unit_rows).sort(["unit_scope", "unit_id"])
    unit_members_df = pl.DataFrame(unit_member_rows).sort(
        ["unit_scope", "unit_id", "genotype_id", "yearsite_uid"]
    )

    blocks_df.write_csv(output_path / "csig_mmd_blocks.csv")
    logger.info("Wrote {}", output_path / "csig_mmd_blocks.csv")
    per_unit_df.write_csv(output_path / "csig_mmd_per_unit.csv")
    logger.info("Wrote {}", output_path / "csig_mmd_per_unit.csv")
    unit_members_df.write_csv(output_path / "csig_mmd_unit_members.csv")
    logger.info("Wrote {}", output_path / "csig_mmd_unit_members.csv")

    scores = per_unit_df["score"]
    summary = pl.DataFrame(
        {
            "source": ["model"],
            "view": ["context_matched_blocked"],
            "estimator": ["blocked"],
            "mode": [mode],
            "block_size": [options.block_size],
            "global_blocks": [options.global_blocks],
            "environment_blocks": [options.environment_blocks],
            "genotype_blocks": [options.genotype_blocks],
            "condition_blocks": [options.condition_blocks],
            "seed": [options.seed],
            "requested_n_samples": [options.n_samples],
            "draw_count": [model_draw_count],
            "model_draw_count": [model_draw_count],
            "oracle_draw_count": [oracle_draw_count],
            "n_units": [per_unit_df.height],
            "n_blocks": [blocks_df.height],
            "alpha": [censoring.alpha],
            "beta": [censoring.beta],
            "c_squared": [censoring.c_squared],
            "mean": [summary_value(scores, "mean")],
            "median": [summary_value(scores, "median")],
            "std": [summary_value(scores, "std")],
            "p95": [quantile_value(scores, 0.95)],
            "mean_w_oracle": [summary_value(per_unit_df["mean_w_oracle"], "mean")],
            "mean_w_model": [summary_value(per_unit_df["mean_w_model"], "mean")],
            "mean_n_context": [summary_value(per_unit_df["mean_n_context"], "mean")],
            "mean_ess": [summary_value(per_unit_df["mean_ess"], "mean")],
            "mean_post_peak_mass": [
                summary_value(per_unit_df["mean_post_peak_mass"], "mean")
            ],
        }
    )
    summary.write_csv(output_path / "csig_mmd_summary.csv")
    logger.info("Wrote {}", output_path / "csig_mmd_summary.csv")


def _dataset_artifact_rows(
    dataset: SyntheticDataset, reference: CensoringReference
) -> np.ndarray:
    """Artifact condition row of every dataset condition (context candidates too)."""
    metadata = dataset.get_condition_metadata()
    return np.array(
        [
            reference.artifact_row(str(gid), str(ys))
            for gid, ys in zip(
                metadata["genotype_id"], metadata["yearsite_uid"], strict=True
            )
        ],
        dtype=np.int64,
    )


def sig_mmd_context_matched_blocked_analysis(  # noqa: PLR0912
    method_dir: str,
    output_folder: str | None = None,
    *,
    options: BlockedScoringOptions | None = None,
    oracle_load_chunk_size: int = 1024,
    lead_lag: bool = True,
    oracle_self: bool = False,
    mahalanobis_artifact: str | None = None,
    csig_output_folder: str | None = None,
    censoring: CensoringParameters | None = None,
    censoring_sweep: CensoringSweep | None = None,
    self_kernel_batch_size: int = 512,
    kernel_sigma: float = 1.0,
) -> None:
    if options is None:
        options = BlockedScoringOptions()
    if censoring is None:
        censoring = CensoringParameters()
    if censoring_sweep and mahalanobis_artifact is None:
        msg = "The CSig-MMD sweep (--sweep-*) requires --mahalanobis-artifact"
        raise ValueError(msg)
    if mahalanobis_artifact is not None and oracle_self:
        msg = "CSig-MMD (--mahalanobis-artifact) cannot be combined with oracle-self"
        raise ValueError(msg)
    if mahalanobis_artifact is not None and not lead_lag:
        msg = "CSig-MMD (--mahalanobis-artifact) requires lead-lag paths"
        raise ValueError(msg)
    if self_kernel_batch_size <= 0:
        msg = f"self_kernel_batch_size must be > 0; got {self_kernel_batch_size}"
        raise ValueError(msg)
    if kernel_sigma <= 0:
        msg = f"kernel_sigma must be > 0; got {kernel_sigma}"
        raise ValueError(msg)
    method_path = Path(method_dir)
    try:
        mode = PredictionMode(method_path.name)
    except ValueError as error:
        msg = (
            "blocked context-matched Sig-MMD requires a method directory named one "
            f"of {sorted(PredictionMode)}; got {method_path.name!r}"
        )
        raise ValueError(msg) from error

    output_path = (
        Path(output_folder)
        if output_folder is not None
        else _default_output_path(
            method_path, oracle_self=oracle_self, kernel_sigma=kernel_sigma
        )
    )
    csig_output_path = (
        Path(csig_output_folder)
        if csig_output_folder is not None
        else method_path
        / f"csig_mmd_context_matched_blocked{kernel_sigma_suffix(kernel_sigma)}"
    )

    loader = (
        BatchLoader(method_path, load_predictions=False)
        if oracle_self
        else BatchLoader(method_path)
    )
    unit_scope = UnitScope.from_config(loader.config, method_path)

    output_path.mkdir(parents=True, exist_ok=True)
    prediction_day_axis = (
        _load_batch_day_axis(loader)
        if oracle_self
        else load_prediction_day_axis(loader)
    )
    logger.info("Grid: {} points", prediction_day_axis.shape[0])

    oracle_cfg, dataloader_name = get_oracle_config_for_method_dir(method_path)
    logger.info(
        "Oracle config: {} (dataloader={})", oracle_cfg.__name__, dataloader_name
    )
    logger.info("Loading SeedA oracle dataset")
    seed_a_dataset: SyntheticDataset = instantiate(oracle_cfg)
    seed_a_id_to_index = build_id_to_index(seed_a_dataset)

    seed_b_dataset: SyntheticDataset | None = None
    seed_b_id_to_index: dict[tuple[str, str], int] | None = None
    seed_b_draw_count: int | None = None
    if oracle_self:
        seed_b_cfg = get_seed_b_config_for_dataloader_name(dataloader_name)
        logger.info("Loading SeedB oracle dataset: {}", seed_b_cfg.__name__)
        seed_b_dataset = instantiate(seed_b_cfg)
        seed_b_id_to_index = build_id_to_index(seed_b_dataset)
        _validate_seed_b_dataset_alignment(
            seed_a_dataset, seed_b_dataset, seed_a_id_to_index, seed_b_id_to_index
        )
        seed_b_draw_count = min(options.n_samples, seed_b_dataset.n_lodging_draws)
        if seed_b_draw_count <= 0:
            msg = f"No SeedB oracle draws available for n_samples={options.n_samples}"
            raise ValueError(msg)

    noise_tensor = seed_a_dataset._noise  # noqa: SLF001
    subsample_indices = seed_a_dataset._subsample_indices  # noqa: SLF001
    if (
        noise_tensor is None
        or subsample_indices is None
        or seed_a_dataset.num_genotypes is None
        or seed_a_dataset.num_yearsites is None
    ):
        msg = "Synthetic oracle dataset is not fully composed"
        raise RuntimeError(msg)
    heights_clean = cast(
        torch.Tensor,
        seed_a_dataset._intermediate["heights_clean"],  # noqa: SLF001
    )
    clean = heights_clean.detach().cpu().numpy()
    noise = noise_tensor.detach().cpu().numpy()
    sub_full_idx = subsample_indices.squeeze(0).detach().cpu().numpy()
    sigma = float(seed_a_dataset.scale_noise)
    num_genotypes = int(seed_a_dataset.num_genotypes)
    num_yearsites = int(seed_a_dataset.num_yearsites)
    oracle_day_axis = seed_a_dataset.get_height_days_all().float()
    days_np = oracle_day_axis.detach().cpu().numpy()
    peak_days = days_np[clean.argmax(axis=1)]

    metric_day_axis_norm = (prediction_day_axis - oracle_day_axis.min()) / (
        oracle_day_axis.max() - oracle_day_axis.min()
    )
    height_scale = SYNTHETIC_SIGNATURE_HEIGHT_SCALE

    reference: CensoringReference | None = None
    dataset_artifact_rows: np.ndarray | None = None
    oracle_draw_count = min(options.n_samples, seed_a_dataset.n_lodging_draws)
    if mahalanobis_artifact is not None:
        reference = load_censoring_reference(
            mahalanobis_artifact,
            censoring,
            sweep=censoring_sweep,
            prediction_day_axis=prediction_day_axis,
            metric_day_axis_norm=metric_day_axis_norm,
            height_scale=height_scale,
        )
        dataset_artifact_rows = _dataset_artifact_rows(seed_a_dataset, reference)
        oracle_draw_count = min(oracle_draw_count, reference.draw_count)
        logger.info(
            "CSig-MMD: alpha={:.4f}, beta={:.4f}, c²={:.4f}, artifact={}",
            reference.main.alpha,
            reference.main.beta,
            reference.main.c_squared,
            mahalanobis_artifact,
        )
    if oracle_draw_count <= 0:
        msg = f"No SeedA oracle draws available for n_samples={options.n_samples}"
        raise ValueError(msg)

    load_start = time.perf_counter()
    model_bank: ModelBank | None = None
    if oracle_self:
        if seed_b_draw_count is None or seed_b_id_to_index is None:
            msg = "oracle-self setup did not initialize SeedB draw metadata"
            raise RuntimeError(msg)
        keys, record_by_model_row = _load_oracle_self_records(
            loader,
            seed_a_id_to_index=seed_a_id_to_index,
            seed_b_id_to_index=seed_b_id_to_index,
            mode=mode,
            clean=clean,
            noise=noise,
            sub_full_idx=sub_full_idx,
        )
        model_draw_count = seed_b_draw_count
    else:
        model_bank, record_by_model_row = load_records_and_bank(
            loader,
            id_to_index=seed_a_id_to_index,
            mode=mode,
            clean=clean,
            noise=noise,
            sub_full_idx=sub_full_idx,
            oracle_draw_count=oracle_draw_count,
            progress_label="context-matched-blocked: load",
        )
        keys = model_bank.keys
        model_draw_count = model_bank.draw_count
    load_seconds = time.perf_counter() - load_start

    units = group_units(unit_scope, keys)
    n_blocks_per_unit = options.blocks_for_scope(unit_scope)
    total_blocks = len(units) * n_blocks_per_unit
    logger.info(
        "Params: mode={}, unit_scope={}, oracle_self={}, n_units={}, block_size={}, "
        "blocks_per_unit={}, total_blocks={}, n_samples={}, model_draw_count={}, "
        "oracle_draw_count={}, seed={}, context_tile_size={}, lead_lag={}, "
        "kernel_sigma={}",
        mode,
        unit_scope,
        oracle_self,
        len(units),
        options.block_size,
        n_blocks_per_unit,
        total_blocks,
        options.n_samples,
        model_draw_count,
        oracle_draw_count,
        options.seed,
        options.context_tile_size,
        lead_lag,
        kernel_sigma,
    )

    block_rows: list[dict[str, object]] = []
    per_unit_rows: list[dict[str, object]] = []
    unit_member_rows: list[dict[str, object]] = []
    censoring_count = 0 if reference is None else len(reference.censorings)
    csig_block_rows: list[list[dict[str, object]]] = [
        [] for _ in range(censoring_count)
    ]
    csig_per_unit_rows: list[list[dict[str, object]]] = [
        [] for _ in range(censoring_count)
    ]
    kernel_seconds = 0.0
    context_weights_seconds = 0.0
    oracle_load_seconds = 0.0
    total_targets = 0
    total_unique_candidates = 0
    total_loaded_pairs = 0
    max_pool_size = 0
    progress = tqdm(total=total_blocks, unit="block", desc="context-matched blocks")
    try:
        for unit_id in sorted(units):
            members = units[unit_id]
            model_blocks = build_unit_blocks(
                members,
                draws_per_member=model_draw_count,
                unit_scope=unit_scope,
                unit_id=unit_id,
                n_blocks=n_blocks_per_unit,
                block_size=options.block_size,
                seed=options.seed,
            )
            oracle_blocks = build_unit_blocks(
                members,
                draws_per_member=oracle_draw_count,
                unit_scope=unit_scope,
                unit_id=unit_id,
                n_blocks=n_blocks_per_unit,
                block_size=options.block_size,
                seed=options.seed,
            )
            if oracle_self:
                if seed_b_dataset is None:
                    msg = "oracle-self setup did not initialize SeedB dataset"
                    raise RuntimeError(msg)
                model_block_heights, _, _, model_profile = _resolve_unit_oracle_blocks(
                    model_blocks,
                    record_by_model_row,
                    unit_scope=unit_scope,
                    mode=mode,
                    seed=options.seed,
                    clean=clean,
                    num_genotypes=num_genotypes,
                    num_yearsites=num_yearsites,
                    sigma=sigma,
                    days_np=days_np,
                    peak_days=peak_days,
                    dataset=seed_b_dataset,
                    oracle_day_axis=oracle_day_axis,
                    prediction_day_axis=prediction_day_axis,
                    chunk_size=oracle_load_chunk_size,
                    context_tile_size=options.context_tile_size,
                    rng_namespace="oracle_self_model",
                )
                oracle_block_heights, oracle_block_pairs, diagnostics, unit_profile = (
                    _resolve_unit_oracle_blocks(
                        oracle_blocks,
                        record_by_model_row,
                        unit_scope=unit_scope,
                        mode=mode,
                        seed=options.seed,
                        clean=clean,
                        num_genotypes=num_genotypes,
                        num_yearsites=num_yearsites,
                        sigma=sigma,
                        days_np=days_np,
                        peak_days=peak_days,
                        dataset=seed_a_dataset,
                        oracle_day_axis=oracle_day_axis,
                        prediction_day_axis=prediction_day_axis,
                        chunk_size=oracle_load_chunk_size,
                        context_tile_size=options.context_tile_size,
                        rng_namespace="oracle_self_reference",
                    )
                )
                profiles = (model_profile, unit_profile)
            else:
                model_block_heights = None
                oracle_block_heights, oracle_block_pairs, diagnostics, unit_profile = (
                    _resolve_unit_oracle_blocks(
                        oracle_blocks,
                        record_by_model_row,
                        unit_scope=unit_scope,
                        mode=mode,
                        seed=options.seed,
                        clean=clean,
                        num_genotypes=num_genotypes,
                        num_yearsites=num_yearsites,
                        sigma=sigma,
                        days_np=days_np,
                        peak_days=peak_days,
                        dataset=seed_a_dataset,
                        oracle_day_axis=oracle_day_axis,
                        prediction_day_axis=prediction_day_axis,
                        chunk_size=oracle_load_chunk_size,
                        context_tile_size=options.context_tile_size,
                    )
                )
                profiles = (unit_profile,)
            for profile in profiles:
                context_weights_seconds += profile["context_weights_seconds"]
                oracle_load_seconds += profile["oracle_load_seconds"]
                total_targets += int(profile["n_targets"])
                total_unique_candidates += int(profile["n_unique_candidates"])
                total_loaded_pairs += int(profile["n_loaded_pairs"])
                max_pool_size = max(max_pool_size, int(profile["pool_size"]))

            scores: list[float] = []
            csig_fields: list[list[dict[str, float]]] = [
                [] for _ in range(censoring_count)
            ]
            for block_index, (
                model_selections,
                oracle_heights,
                (oracle_conditions, oracle_draws),
            ) in enumerate(
                zip(model_blocks, oracle_block_heights, oracle_block_pairs, strict=True)
            ):
                if oracle_self:
                    if model_block_heights is None:
                        msg = "oracle-self model blocks were not resolved"
                        raise RuntimeError(msg)
                    model_heights = model_block_heights[block_index].float()
                else:
                    if model_bank is None:
                        msg = "normal scoring requires a loaded model bank"
                        raise RuntimeError(msg)
                    model_condition_indices, model_draw_indices = selection_tensors(
                        model_selections
                    )
                    model_heights = model_bank.heights[
                        model_condition_indices, model_draw_indices
                    ].float()
                model_paths = to_metric_paths(
                    model_heights, height_scale, metric_day_axis_norm, lead_lag=lead_lag
                )
                oracle_paths = to_metric_paths(
                    oracle_heights.float(),
                    height_scale,
                    metric_day_axis_norm,
                    lead_lag=lead_lag,
                )
                kernel_start = time.perf_counter()
                if reference is None:
                    score = normalized_sig_mmd(
                        model_paths, oracle_paths, sigma=kernel_sigma
                    )
                else:
                    assert dataset_artifact_rows is not None
                    censoring_fields = reference.score(
                        oracle_paths,
                        model_paths,
                        reference.distances[
                            dataset_artifact_rows[oracle_conditions], oracle_draws
                        ],
                        height_scale,
                        metric_day_axis_norm,
                        self_kernel_batch_size,
                        label=(
                            f"unit_scope={unit_scope!r}, unit_id={unit_id!r}, "
                            f"block_index={block_index}"
                        ),
                        sigma=kernel_sigma,
                    )
                    score = censoring_fields[0]["sig_score"]
                    for index, score_fields in enumerate(censoring_fields):
                        csig_fields[index].append(score_fields)
                        csig_block_rows[index].append(
                            {
                                "source": "model",
                                "mode": mode,
                                "unit_scope": unit_scope,
                                "unit_id": unit_id,
                                "block_index": block_index,
                                "block_size": options.block_size,
                                "model_sample_size": int(model_heights.shape[0]),
                                "oracle_sample_size": int(oracle_heights.shape[0]),
                                "score": score_fields["score"],
                                "mean_w_oracle": score_fields["mean_w_oracle"],
                                "mean_w_model": score_fields["mean_w_model"],
                            }
                        )
                kernel_seconds += time.perf_counter() - kernel_start
                scores.append(score)
                block_row: dict[str, object] = {
                    "mode": mode,
                    "unit_scope": unit_scope,
                    "unit_id": unit_id,
                    "block_index": block_index,
                    "block_size": options.block_size,
                    "model_sample_size": int(model_heights.shape[0]),
                    "oracle_sample_size": int(oracle_heights.shape[0]),
                    "score": score,
                }
                if oracle_self:
                    block_row["oracle_self"] = True
                block_rows.append(block_row)
                progress.update(1)

            score_std, score_se = std_and_se(scores)
            model_selected_count = sum(len(block) for block in model_blocks)
            oracle_selected_count = sum(len(block) for block in oracle_blocks)
            full_model_pool_size = len(members) * model_draw_count
            full_oracle_pool_size = len(members) * oracle_draw_count
            mean_n_context, mean_ess, mean_post_peak_mass = aggregate_diagnostics(
                diagnostics
            )
            per_unit_row: dict[str, object] = {
                "mode": mode,
                "unit_scope": unit_scope,
                "unit_id": unit_id,
                "n_members": len(members),
                "n_blocks": n_blocks_per_unit,
                "block_size": options.block_size,
                "full_model_pool_size": full_model_pool_size,
                "full_oracle_pool_size": full_oracle_pool_size,
                "model_sample_size": model_selected_count,
                "oracle_sample_size": oracle_selected_count,
                "score": float(np.mean(scores)),
                "score_std": score_std,
                "score_se": score_se,
                "mean_n_context": mean_n_context,
                "mean_ess": mean_ess,
                "mean_post_peak_mass": mean_post_peak_mass,
            }
            if oracle_self:
                per_unit_row["oracle_self"] = True
            per_unit_rows.append(per_unit_row)
            for index, fields in enumerate(csig_fields):
                csig_scores = [block["score"] for block in fields]
                csig_std, csig_se = std_and_se(csig_scores)
                csig_per_unit_rows[index].append(
                    {
                        "source": "model",
                        "mode": mode,
                        "unit_scope": unit_scope,
                        "unit_id": unit_id,
                        "n_members": len(members),
                        "n_blocks": n_blocks_per_unit,
                        "block_size": options.block_size,
                        "full_model_pool_size": full_model_pool_size,
                        "full_oracle_pool_size": full_oracle_pool_size,
                        "model_sample_size": model_selected_count,
                        "oracle_sample_size": oracle_selected_count,
                        "score": float(np.mean(csig_scores)),
                        "score_std": csig_std,
                        "score_se": csig_se,
                        "mean_w_oracle": float(
                            np.mean([block["mean_w_oracle"] for block in fields])
                        ),
                        "mean_w_model": float(
                            np.mean([block["mean_w_model"] for block in fields])
                        ),
                        "mean_n_context": mean_n_context,
                        "mean_ess": mean_ess,
                        "mean_post_peak_mass": mean_post_peak_mass,
                    }
                )
            for member in members:
                unit_member_row: dict[str, object] = {
                    "mode": mode,
                    "unit_scope": unit_scope,
                    "unit_id": unit_id,
                    "genotype_id": member.genotype_id,
                    "yearsite_uid": member.yearsite_uid,
                }
                if oracle_self:
                    unit_member_row["oracle_self"] = True
                unit_member_rows.append(unit_member_row)
    finally:
        progress.close()

    logger.info(
        "Profile (seconds): load={:.2f}, context_weights={:.2f}, oracle_load={:.2f}, "
        "kernel={:.2f}",
        load_seconds,
        context_weights_seconds,
        oracle_load_seconds,
        kernel_seconds,
    )
    logger.info(
        "Profile (counts): selected_targets={}, pool_size={}, unique_candidates={}, "
        "loaded_pairs={}, kernel_blocks={}",
        total_targets,
        max_pool_size,
        total_unique_candidates,
        total_loaded_pairs,
        total_blocks,
    )

    write_outputs(
        output_path,
        mode=mode,
        block_rows=block_rows,
        per_unit_rows=per_unit_rows,
        unit_member_rows=unit_member_rows,
        options=options,
        model_draw_count=model_draw_count,
        oracle_draw_count=oracle_draw_count,
        oracle_self=oracle_self,
    )
    if reference is not None:
        csig_member_rows = [{"source": "model", **row} for row in unit_member_rows]
        for index, rule in enumerate(reference.censorings):
            path = (
                csig_output_path
                if index == 0
                else csig_output_path / "sweep" / rule.name
            )
            path.mkdir(parents=True, exist_ok=True)
            write_csig_outputs(
                path,
                mode=mode,
                block_rows=csig_block_rows[index],
                per_unit_rows=csig_per_unit_rows[index],
                unit_member_rows=csig_member_rows,
                options=options,
                model_draw_count=model_draw_count,
                oracle_draw_count=oracle_draw_count,
                censoring=rule,
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        sig_mmd_context_matched_blocked_analysis(
            method_dir=args.method_dir,
            output_folder=args.output_folder,
            options=BlockedScoringOptions.from_args(args),
            oracle_load_chunk_size=args.oracle_load_chunk_size,
            lead_lag=not args.no_lead_lag,
            oracle_self=args.oracle_self,
            mahalanobis_artifact=args.mahalanobis_artifact,
            csig_output_folder=args.csig_output_folder,
            censoring=CensoringParameters.from_args(args),
            censoring_sweep=CensoringSweep.from_args(args),
            self_kernel_batch_size=args.self_kernel_batch_size,
            kernel_sigma=args.kernel_sigma,
        )
    except (FileNotFoundError, NotADirectoryError, TypeError, ValueError) as error:
        msg = f"error: {error}"
        raise SystemExit(msg) from error


if __name__ == "__main__":
    main()
