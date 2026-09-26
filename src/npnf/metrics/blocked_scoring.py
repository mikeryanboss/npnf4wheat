"""Scaffolding shared by the blocked context-matched Sig-MMD and CSig-MMD scorers."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

import numpy as np
import torch
from tensordict import TensorDict
from tqdm import tqdm

from npnf.data.batch_loader import BatchLoader
from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.metrics.blocks import Selection, UnitMember, UnitScope
from npnf.metrics.utils import id_to_str, positive_int


class PredictionMode(StrEnum):
    """Method directory names the context-matched scorers accept."""

    NO_CONTEXT = "no_context"
    RANDOM_CONTEXT = "random_context"
    MAX_HEIGHT = "max_height"


@dataclass(frozen=True)
class BlockedScoringOptions:
    """Block budget and sampling knobs shared by the blocked context-matched scorers."""

    n_samples: int = 64
    block_size: int = 512
    global_blocks: int = 1000
    environment_blocks: int = 16
    genotype_blocks: int = 1
    condition_blocks: int = 1
    seed: int = 0
    context_tile_size: int = 2048

    def __post_init__(self) -> None:
        for name in (
            "n_samples",
            "block_size",
            "global_blocks",
            "environment_blocks",
            "genotype_blocks",
            "condition_blocks",
            "context_tile_size",
        ):
            value = getattr(self, name)
            if value <= 0:
                msg = f"{name} must be > 0; got {value}"
                raise ValueError(msg)

    def blocks_for_scope(self, unit_scope: UnitScope) -> int:
        return {
            UnitScope.GLOBAL: self.global_blocks,
            UnitScope.ENVIRONMENT: self.environment_blocks,
            UnitScope.GENOTYPE: self.genotype_blocks,
            UnitScope.CONDITION: self.condition_blocks,
        }[unit_scope]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register the shared scorer flags with this class's field defaults."""
        defaults = cls()
        parser.add_argument(
            "--n-samples",
            type=positive_int,
            default=defaults.n_samples,
            help="Maximum trajectory draws per condition (default: %(default)s).",
        )
        parser.add_argument(
            "--block-size",
            type=positive_int,
            default=defaults.block_size,
            help="Trajectories per side in each MMD block (default: %(default)s).",
        )
        parser.add_argument(
            "--global-blocks",
            type=positive_int,
            default=defaults.global_blocks,
            help="Blocks for P/global units (default: %(default)s).",
        )
        parser.add_argument(
            "--environment-blocks",
            type=positive_int,
            default=defaults.environment_blocks,
            help="Blocks per E/environment unit (default: %(default)s).",
        )
        parser.add_argument(
            "--genotype-blocks",
            type=positive_int,
            default=defaults.genotype_blocks,
            help="Blocks per G/genotype unit (default: %(default)s).",
        )
        parser.add_argument(
            "--condition-blocks",
            type=positive_int,
            default=defaults.condition_blocks,
            help=(
                "Blocks per E&G/condition unit, used when use_temperature and "
                "use_marker are both set (default: %(default)s)."
            ),
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=defaults.seed,
            help=(
                "Base seed for deterministic block construction and stochastic "
                "context-matched oracle sampling (default: %(default)s)."
            ),
        )
        parser.add_argument(
            "--context-tile-size",
            type=positive_int,
            default=defaults.context_tile_size,
            help=(
                "Targets per vectorized context-posterior tile (default: %(default)s)."
            ),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BlockedScoringOptions:
        return cls(
            n_samples=args.n_samples,
            block_size=args.block_size,
            global_blocks=args.global_blocks,
            environment_blocks=args.environment_blocks,
            genotype_blocks=args.genotype_blocks,
            condition_blocks=args.condition_blocks,
            seed=args.seed,
            context_tile_size=args.context_tile_size,
        )


class ContextRecord(TypedDict):
    u0: int
    gid: str
    ys: str
    ctx_full_idx: np.ndarray
    ctx_vals: np.ndarray


@dataclass(frozen=True)
class ModelBank:
    keys: tuple[tuple[str, str], ...]
    heights: torch.Tensor
    draw_count: int


def build_id_to_index(dataset: SyntheticDataset) -> dict[tuple[str, str], int]:
    metadata = dataset.get_condition_metadata()
    return {
        (str(gid), str(ys)): idx
        for idx, (gid, ys) in enumerate(
            zip(metadata["genotype_id"], metadata["yearsite_uid"], strict=True)
        )
    }


def extract_grid(predictions: TensorDict) -> torch.Tensor:
    return predictions.select("grid").flatten()["grid"].squeeze(-1)


def extract_metadata(batch_dict: dict) -> tuple[list[str], list[str]]:
    metadata = batch_dict["data"].flatten()
    gids = [id_to_str(gid) for gid in metadata["genotype_id"]]
    yss = [id_to_str(ys) for ys in metadata["yearsite_uid"]]
    return gids, yss


def load_prediction_day_axis(loader: BatchLoader) -> torch.Tensor:
    first_batch = loader.load_batch(0)
    if not isinstance(first_batch, tuple):
        msg = "blocked Sig-MMD requires saved predictions and batch metadata"
        raise TypeError(msg)
    _, batch_dict = first_batch
    return batch_dict["grid_points"]["X"].squeeze(-1).float()


def group_units(
    unit_scope: UnitScope, keys: Sequence[tuple[str, str]]
) -> dict[str, list[UnitMember]]:
    units: dict[str, list[UnitMember]] = defaultdict(list)
    for condition_index, (gid, ys) in enumerate(keys):
        units[unit_scope.unit_id(gid, ys)].append(
            UnitMember(
                genotype_id=gid, yearsite_uid=ys, condition_index=condition_index
            )
        )
    return {
        unit_id: sorted(members, key=lambda m: (m.genotype_id, m.yearsite_uid))
        for unit_id, members in units.items()
    }


def selection_tensors(
    selections: Sequence[Selection],
) -> tuple[torch.Tensor, torch.Tensor]:
    condition_indices = torch.tensor(
        [selection.condition_index for selection in selections], dtype=torch.long
    )
    draw_indices = torch.tensor(
        [selection.draw_index for selection in selections], dtype=torch.long
    )
    return condition_indices, draw_indices


def candidate_pool(
    unit_scope: UnitScope, u0: int, num_genotypes: int, num_yearsites: int
) -> np.ndarray:
    if unit_scope == UnitScope.CONDITION:
        return np.array([u0], dtype=int)
    genotype_index = u0 // num_yearsites
    yearsite_index = u0 % num_yearsites
    if unit_scope == UnitScope.ENVIRONMENT:
        return np.arange(num_genotypes, dtype=int) * num_yearsites + yearsite_index
    if unit_scope == UnitScope.GENOTYPE:
        return genotype_index * num_yearsites + np.arange(num_yearsites, dtype=int)
    return np.arange(num_genotypes * num_yearsites, dtype=int)


def pool_key(unit_scope: UnitScope, u0: int, num_yearsites: int) -> int:
    """Identity of a row's candidate pool, for grouping rows that share oracle loads."""
    if unit_scope == UnitScope.CONDITION:
        return u0
    if unit_scope == UnitScope.ENVIRONMENT:
        return u0 % num_yearsites
    if unit_scope == UnitScope.GENOTYPE:
        return u0 // num_yearsites
    return 0


def stable_seed(seed: int, mode: str, unit_scope: str, gid: str, ys: str) -> int:
    payload = f"{seed}\0{mode}\0{unit_scope}\0{gid}\0{ys}".encode()
    digest = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(digest, "little", signed=False)


def context_positions(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(int).reshape(-1)
    return np.asarray(value, dtype=int).reshape(-1)


def reconstruct_context(
    mode: str,
    batch_dict: dict,
    row_index: int,
    clean_row: np.ndarray,
    noise_row: np.ndarray,
    sub_full_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == PredictionMode.NO_CONTEXT:
        return np.array([], dtype=int), np.array([], dtype=float)
    if mode == PredictionMode.RANDOM_CONTEXT:
        context_indices = batch_dict.get("context_indices")
        if context_indices is None:
            msg = (
                "random_context method directory is missing context_indices in batch.pt"
            )
            raise ValueError(msg)
        positions = context_positions(context_indices[row_index])
    elif mode == PredictionMode.MAX_HEIGHT:
        clean_sub = clean_row[sub_full_idx]
        positions = np.arange(int(clean_sub.argmax()) + 1, dtype=int)
    else:
        msg = f"Unsupported context mode {mode!r}"
        raise ValueError(msg)

    ctx_full_idx = sub_full_idx[positions]
    ctx_vals = clean_row[ctx_full_idx] + noise_row[ctx_full_idx]
    return ctx_full_idx, ctx_vals


def aggregate_diagnostics(
    diagnostics: Sequence[tuple[int, float, float]],
) -> tuple[float, float, float]:
    if not diagnostics:
        return float("nan"), float("nan"), float("nan")
    n_context = float(np.mean([entry[0] for entry in diagnostics]))
    ess = float(np.mean([entry[1] for entry in diagnostics]))
    post_peak_mass = float(np.mean([entry[2] for entry in diagnostics]))
    return n_context, ess, post_peak_mass


def load_records_and_bank(
    loader: BatchLoader,
    *,
    id_to_index: dict[tuple[str, str], int],
    mode: str,
    clean: np.ndarray,
    noise: np.ndarray,
    sub_full_idx: np.ndarray,
    oracle_draw_count: int,
    progress_label: str,
) -> tuple[ModelBank, dict[int, ContextRecord]]:
    """Stream the loader once, building both the model bank and context records.

    Reconstructing ``random_context`` needs ``batch_dict["context_indices"]``, so
    model heights and reconstructed context are captured together in a single pass.
    The returned ``ModelBank`` rejects duplicate conditions, sorts keys
    lexicographically, and clamps the draw count to the smallest saved draw count;
    the record map is keyed by the **model-bank row position** (sorted-key index) so
    oracle/model selections can recover each row's ``u0`` and context.
    """
    keys: list[tuple[str, str]] = []
    row_heights: list[torch.Tensor] = []
    row_contexts: list[ContextRecord] = []
    min_model_draws: int | None = None

    for predictions, batch_dict in tqdm(
        loader, total=len(loader), unit="batch", desc=progress_label
    ):
        grid = extract_grid(predictions).detach().cpu()
        if grid.ndim != 3:
            msg = (
                "Expected prediction grid with shape (B, S, G), got "
                f"{tuple(grid.shape)}"
            )
            raise ValueError(msg)
        gids, yss = extract_metadata(batch_dict)
        if len(gids) != grid.shape[0] or len(yss) != grid.shape[0]:
            msg = "Prediction batch metadata length does not match grid rows"
            raise ValueError(msg)

        for row_index, (gid, ys) in enumerate(zip(gids, yss, strict=True)):
            try:
                u0 = id_to_index[(gid, ys)]
            except KeyError as error:
                msg = (
                    "Oracle dataset does not contain prediction condition "
                    f"(genotype_id={gid!r}, yearsite_uid={ys!r})"
                )
                raise ValueError(msg) from error
            ctx_full_idx, ctx_vals = reconstruct_context(
                mode, batch_dict, row_index, clean[u0], noise[u0], sub_full_idx
            )
            heights = grid[row_index].float().clone()
            draws = int(heights.shape[0])
            min_model_draws = (
                draws if min_model_draws is None else min(min_model_draws, draws)
            )
            keys.append((gid, ys))
            row_heights.append(heights)
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
    if min_model_draws is None or min_model_draws <= 0:
        msg = f"No model draws found under {loader.method_dir}"
        raise ValueError(msg)

    model_draw_count = min(oracle_draw_count, min_model_draws)
    if model_draw_count <= 0:
        msg = f"No draws available for requested n_samples={oracle_draw_count}"
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
    bank = torch.stack(
        [row_heights[seen[key]][:model_draw_count] for key in sorted_keys], dim=0
    ).contiguous()
    record_by_model_row = {
        condition_index: row_contexts[seen[key]]
        for condition_index, key in enumerate(sorted_keys)
    }
    return (
        ModelBank(keys=sorted_keys, heights=bank, draw_count=model_draw_count),
        record_by_model_row,
    )


def context_weights(
    clean_np: np.ndarray,
    pool: np.ndarray | list[int],
    ctx_full_idx: np.ndarray | list[int],
    ctx_vals: np.ndarray | list[float],
    sigma: float,
) -> tuple[np.ndarray, float]:
    """Weight candidate conditions by clean-curve context likelihood."""
    pool_arr = np.asarray(pool, dtype=int).reshape(-1)
    ctx_idx_arr = np.asarray(ctx_full_idx, dtype=int).reshape(-1)
    ctx_vals_arr = np.asarray(ctx_vals, dtype=float).reshape(-1)

    if pool_arr.size == 0:
        msg = "context_weights requires a non-empty candidate pool"
        raise ValueError(msg)
    if sigma <= 0.0:
        msg = f"sigma must be positive; got {sigma}"
        raise ValueError(msg)
    if ctx_idx_arr.size != ctx_vals_arr.size:
        msg = (
            "ctx_full_idx and ctx_vals must have matching lengths; got "
            f"{ctx_idx_arr.size} and {ctx_vals_arr.size}"
        )
        raise ValueError(msg)

    if ctx_idx_arr.size == 0:
        weights = np.full(pool_arr.size, 1.0 / pool_arr.size, dtype=float)
        return weights, float(pool_arr.size)

    h = np.asarray(clean_np)[pool_arr][:, ctx_idx_arr]
    sse = ((ctx_vals_arr[None, :] - h) ** 2).sum(axis=1)
    logw = -sse / (2.0 * sigma**2)
    logw -= logw.max()
    weights = np.exp(logw)
    weights /= weights.sum()
    return weights, float(1.0 / np.sum(weights**2))


def weights_from_clean_pool(
    clean_pool: np.ndarray, ctx_full_idx: np.ndarray, ctx_vals: np.ndarray, sigma: float
) -> tuple[np.ndarray, float]:
    """``context_weights`` with the per-pool ``clean[pool]`` gather hoisted out.

    ``clean_pool`` must equal ``clean[pool]`` for the target's candidate pool. Returns
    bit-identical weights and ESS to ``context_weights(clean, pool, ...)`` so targets
    that share a candidate pool reuse the gather without changing the posterior.
    """
    ctx_idx_arr = np.asarray(ctx_full_idx, dtype=int).reshape(-1)
    ctx_vals_arr = np.asarray(ctx_vals, dtype=float).reshape(-1)
    pool_size = clean_pool.shape[0]
    if ctx_idx_arr.size == 0:
        weights = np.full(pool_size, 1.0 / pool_size, dtype=float)
        return weights, float(pool_size)
    h = clean_pool[:, ctx_idx_arr]
    sse = ((ctx_vals_arr[None, :] - h) ** 2).sum(axis=1)
    logw = -sse / (2.0 * sigma**2)
    logw -= logw.max()
    weights = np.exp(logw)
    weights /= weights.sum()
    return weights, float(1.0 / np.sum(weights**2))


def tile_posterior(
    clean_pool_t: torch.Tensor,
    clean2_pool_t: torch.Tensor,
    ctx_idx_tile: Sequence[np.ndarray],
    ctx_vals_tile: Sequence[np.ndarray],
    sigma: float,
    device: torch.device,
    pool_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized context posterior for one tile of targets sharing a candidate pool.

    Computes the same weights/ESS as ``weights_from_clean_pool`` per target, but for
    the whole tile at once via the expand-the-square identity::

        sse[t, p] = a[t] - 2 (B @ Cᵀ)[t, p] + (M @ (C²)ᵀ)[t, p]
        a[t] = Σ_d M[t, d] V[t, d]² ,  B = M · V

    where ``C = clean_pool`` (``clean_pool_t`` is ``Cᵀ``), ``V``/``M`` hold each
    target's context values and 0/1 day mask. This re-associates the SSE into two
    matmuls run on ``device`` (GPU when available), so the result is numerically
    equivalent to the per-target path (summation order differs), not bit-identical.
    ``clean_pool_t`` and ``clean2_pool_t`` are float64 ``(D, pool_size)`` tensors,
    built once per pool.

    Returns ``weights`` ``(tile, pool_size)`` and ``ess`` ``(tile,)`` as numpy float64.
    Empty-context targets get uniform weights and ``ess == pool_size`` exactly.
    """
    n_days = clean_pool_t.shape[0]
    tile = len(ctx_idx_tile)
    mask = torch.zeros((tile, n_days), dtype=torch.float64, device=device)
    vals = torch.zeros((tile, n_days), dtype=torch.float64, device=device)
    empty: list[int] = []
    for i in range(tile):
        idx = np.asarray(ctx_idx_tile[i], dtype=np.int64).reshape(-1)
        if idx.size == 0:
            empty.append(i)
            continue
        col = torch.as_tensor(idx, device=device)
        mask[i, col] = 1.0
        vals[i, col] = torch.as_tensor(
            np.asarray(ctx_vals_tile[i], dtype=np.float64).reshape(-1), device=device
        )
    if len(empty) == tile:
        weights = np.full((tile, pool_size), 1.0 / pool_size, dtype=float)
        ess = np.full(tile, float(pool_size), dtype=float)
        return weights, ess

    a = (mask * vals * vals).sum(dim=1)
    b = mask * vals
    logw = b @ clean_pool_t
    logw.mul_(-2.0)
    logw.addmm_(mask, clean2_pool_t)
    logw.add_(a[:, None])
    logw.mul_(-1.0 / (2.0 * sigma**2))
    logw.sub_(logw.max(dim=1, keepdim=True).values)
    logw.exp_()
    logw.div_(logw.sum(dim=1, keepdim=True))
    ess = (1.0 / (logw * logw).sum(dim=1)).cpu().numpy()
    weights = logw.cpu().numpy()
    for i in empty:
        weights[i] = 1.0 / pool_size
        ess[i] = float(pool_size)
    return weights, ess


def posterior_tile_or_uniform(
    clean_pool_t: torch.Tensor | None,
    clean2_pool_t: torch.Tensor | None,
    ctx_idx_tile: Sequence[np.ndarray],
    ctx_vals_tile: Sequence[np.ndarray],
    sigma: float,
    device: torch.device,
    pool_size: int,
    uniform_weights: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Return tiled weights/ESS, with an exact empty-context fast path."""
    if all(np.asarray(idx).size == 0 for idx in ctx_idx_tile):
        if uniform_weights is None:
            uniform_weights = np.full(pool_size, 1.0 / pool_size, dtype=float)
        ess = np.full(len(ctx_idx_tile), float(pool_size), dtype=float)
        return None, ess, uniform_weights
    if clean_pool_t is None or clean2_pool_t is None:
        msg = "non-empty context tile is missing clean-pool tensors"
        raise RuntimeError(msg)
    weights, ess = tile_posterior(
        clean_pool_t,
        clean2_pool_t,
        ctx_idx_tile,
        ctx_vals_tile,
        sigma,
        device,
        pool_size,
    )
    return weights, ess, uniform_weights
