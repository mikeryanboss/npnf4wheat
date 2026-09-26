"""Paper figures comparing conditioning-specific prediction samples."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")

import matplotlib.pyplot as plt
from loguru import logger

from npnf.data.batch_loader import BatchLoader
from npnf.scripts.paper.style import COLOR_LIST, COLORS, setup_style
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    SyntheticBatchReconstructionCache,
    reconstruct_synthetic_batch,
)
from npnf.utils import add_panel_labels

CONDITIONING_DIRS = ("noenv_nogeno", "noenv_geno", "env_nogeno", "env_geno")
DEFAULT_TARGET_PERCENTILE = 0.90
SELECTION_RULE = (
    "nearest 90th percentile mean maximum height in env_geno ground truth "
    "over loaded batches"
)
DEFAULT_OUTPUT_FOLDER = Path("paper") / "conditioning_sample_comparison"
DEFAULT_NUM_SAMPLES = 256
DEFAULT_SEED = 42
SAMPLE_ALPHA = 0.27
SAMPLE_LINEWIDTH = 1.0
MEAN_COLOR = "#111111"
MEAN_LINEWIDTH = 2.3


@dataclass(frozen=True)
class TrajectoryPanel:
    x_values: np.ndarray
    samples: np.ndarray
    mean: np.ndarray


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class TruthSource(Protocol):
    """Ground-truth trajectories of a saved batch: the x axis and one array per row."""

    def rows(
        self, method_dir: Path, batch: dict
    ) -> tuple[np.ndarray, list[np.ndarray]]: ...


class SyntheticTruth:
    """Simulator trajectories of a saved batch, one array per batch row.

    The ground-truth column of every figure comes from a source of this shape:
    ``rows`` returns the shared x axis and, per row of the batch, the true
    trajectories of that condition. FIP1 has no simulator and supplies its own
    source, which reads the measured plots.
    """

    def __init__(self) -> None:
        self._cache = SyntheticBatchReconstructionCache()

    def rows(
        self, method_dir: Path, batch: dict
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        batch = reconstruct_synthetic_batch(method_dir, batch, cache=self._cache)
        height = batch["data"]["height"]
        y_original = _as_numpy(height["Y_original"][..., 0])
        rows = [row if row.ndim == 2 else row[None, :] for row in y_original]
        return _as_numpy(height["X"][0, :, 0]), rows


def _add_to_mean(
    total: np.ndarray | None, counts: np.ndarray | None, trajectories: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate a point-wise sum and how many trajectories cover each point."""
    batch_total = np.asarray(np.nansum(trajectories, axis=0, dtype=np.float64))
    batch_counts = np.isfinite(trajectories).sum(axis=0)
    if total is None or counts is None:
        return batch_total, batch_counts
    return total + batch_total, counts + batch_counts


def _panel_mean(total: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Point-wise mean, NaN where no trajectory covers the point."""
    return np.divide(total, counts, out=np.full(total.shape, np.nan), where=counts > 0)


def ids(batch: dict, field: str) -> np.ndarray:
    return np.array([str(value) for value in batch["data"][field]])


def sample_mask(
    batch: dict, genotype_id: str | None, yearsite_uid: str | None
) -> np.ndarray:
    data = batch["data"]
    keep = np.ones(len(data["genotype_id"]), dtype=bool)
    if genotype_id is not None:
        keep &= ids(batch, "genotype_id") == genotype_id
    if yearsite_uid is not None:
        keep &= ids(batch, "yearsite_uid") == yearsite_uid
    return keep


def _num_batches_to_load(loader: BatchLoader, num_batches: int | None) -> int:
    return len(loader) if num_batches is None else min(num_batches, len(loader))


def _prioritized_batch_indices(
    loader: BatchLoader, num_batches: int | None, batch_indices: list[int] | None
) -> list[int]:
    n_batches = _num_batches_to_load(loader, num_batches)
    if batch_indices is None:
        return list(range(n_batches))

    primary = [idx for idx in batch_indices if 0 <= idx < n_batches]
    primary_set = set(primary)
    return [*primary, *(idx for idx in range(n_batches) if idx not in primary_set)]


def _load_batch_dict(loader: BatchLoader, batch_idx: int) -> dict:
    batch = loader[batch_idx]
    if not isinstance(batch, dict):
        msg = f"Expected batch metadata only from {loader.method_dir}"
        raise TypeError(msg)
    return batch


def collect_condition_pair_batches(
    method_dir: Path, num_batches: int | None
) -> dict[tuple[str, str], list[int]]:
    """Return the loaded batch indices where each condition pair appears."""
    loader = BatchLoader(method_dir, load_predictions=False)
    pair_batches: dict[tuple[str, str], list[int]] = {}
    for batch_idx in range(_num_batches_to_load(loader, num_batches)):
        batch = _load_batch_dict(loader, batch_idx)
        for pair in _batch_pairs(batch):
            pair_batches.setdefault(pair, []).append(batch_idx)
    return pair_batches


def collect_condition_pairs(
    method_dir: Path, num_batches: int | None
) -> list[tuple[str, str]]:
    """Return sorted unique (genotype_id, yearsite_uid) pairs from data."""
    return sorted(collect_condition_pair_batches(method_dir, num_batches))


def _select_pairs(
    all_pairs: list[tuple[str, str]],
    genotype_id: str | None,
    yearsite_uid: str | None,
    rng: np.random.Generator,
    max_pairs: int,
) -> list[tuple[str, str]]:
    """Filter pairs by optional genotype/env, then randomly subset to max_pairs."""
    filtered = [
        (g, e)
        for g, e in all_pairs
        if (genotype_id is None or g == genotype_id)
        and (yearsite_uid is None or e == yearsite_uid)
    ]
    if len(filtered) <= max_pairs:
        return filtered
    indices = rng.choice(len(filtered), size=max_pairs, replace=False)
    return [filtered[i] for i in sorted(indices)]


def _derived_seed(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _rng_for(seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(_derived_seed(seed, *parts))


def _pair_sample_quotas(
    pairs: list[tuple[str, str]], num_samples: int
) -> dict[tuple[str, str], int]:
    if not pairs:
        msg = "At least one condition pair is required for sampling."
        raise ValueError(msg)
    if len(pairs) > num_samples:
        msg = (
            f"Number of selected condition pairs ({len(pairs)}) must not exceed "
            f"num_samples ({num_samples})."
        )
        raise ValueError(msg)

    base_quota, extra_count = divmod(num_samples, len(pairs))
    quotas = dict.fromkeys(pairs, base_quota)
    for pair in pairs[:extra_count]:
        quotas[pair] += 1
    return quotas


def _batch_pairs(batch: dict) -> list[tuple[str, str]]:
    genotype_ids = ids(batch, "genotype_id")
    environment_ids = ids(batch, "yearsite_uid")
    return list(zip(genotype_ids, environment_ids, strict=True))


def _pair_mask(
    batch_pairs: list[tuple[str, str]], pair_set: set[tuple[str, str]]
) -> np.ndarray:
    return np.array([pair in pair_set for pair in batch_pairs], dtype=bool)


def update_sample_trajectories(
    sampled_trajectories: np.ndarray | None,
    sampled_priorities: np.ndarray | None,
    batch_trajectories: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch_priorities = rng.random(len(batch_trajectories))
    if sampled_trajectories is None or sampled_priorities is None:
        candidate_trajectories = batch_trajectories
        candidate_priorities = batch_priorities
    else:
        candidate_trajectories = np.concatenate(
            [sampled_trajectories, batch_trajectories], axis=0
        )
        candidate_priorities = np.concatenate(
            [sampled_priorities, batch_priorities], axis=0
        )

    keep_count = min(num_samples, len(candidate_priorities))
    if keep_count == len(candidate_priorities):
        keep_indices = np.arange(keep_count)
    else:
        keep_indices = np.argpartition(candidate_priorities, keep_count - 1)[
            :keep_count
        ]
    return candidate_trajectories[keep_indices], candidate_priorities[keep_indices]


def _finalize_panel(
    *,
    x_values: torch.Tensor | np.ndarray | None,
    samples: np.ndarray | None,
    total: np.ndarray | None,
    counts: np.ndarray | None,
    source: str,
    genotype_id: str | None,
    yearsite_uid: str | None,
) -> TrajectoryPanel:
    if x_values is None or samples is None or total is None or counts is None:
        msg = (
            f"No trajectories found in {source} "
            f"for genotype_id={genotype_id!r}, yearsite_uid={yearsite_uid!r}"
        )
        raise ValueError(msg)

    return TrajectoryPanel(
        x_values=_as_numpy(x_values), samples=samples, mean=_panel_mean(total, counts)
    )


def load_model_panel(
    method_dir: Path,
    genotype_id: str | None,
    yearsite_uid: str | None,
    rng: np.random.Generator,
    num_samples: int,
    num_batches: int | None,
) -> TrajectoryPanel:
    loader = BatchLoader(method_dir)
    sample_trajectories = None
    sample_priorities = None
    total = None
    counts = None
    x_values = None

    for batch_idx in range(_num_batches_to_load(loader, num_batches)):
        predictions, batch = loader[batch_idx]
        keep = sample_mask(batch, genotype_id, yearsite_uid)
        if not keep.any():
            continue

        grid = predictions["grid"]
        keep_t = torch.as_tensor(keep, device=grid.device)
        trajectories = _as_numpy(grid[keep_t][:, 0, :, :, 0].flatten(0, 1))
        total, counts = _add_to_mean(total, counts, trajectories)
        sample_trajectories, sample_priorities = update_sample_trajectories(
            sample_trajectories, sample_priorities, trajectories, rng, num_samples
        )
        if x_values is None:
            x_values = batch["grid_points"]["X"][:, 0]

    return _finalize_panel(
        x_values=x_values,
        samples=sample_trajectories,
        total=total,
        counts=counts,
        source=str(method_dir),
        genotype_id=genotype_id,
        yearsite_uid=yearsite_uid,
    )


def load_model_stratified(
    method_dir: Path,
    pairs: list[tuple[str, str]],
    *,
    seed: int,
    num_samples: int,
    num_batches: int | None,
    batch_indices: list[int] | None = None,
) -> TrajectoryPanel:
    """Load model trajectories for a fixed set of condition pairs.

    The selected pairs define the panel population.  The mean is computed from
    all trajectories for those pairs, while displayed trajectories are sampled
    per pair so rows stay condition-diverse without collapsing fixed-pair rows
    to a single curve.
    """
    loader = BatchLoader(method_dir)
    pair_set = set(pairs)
    pair_quotas = _pair_sample_quotas(pairs, num_samples)
    pair_samples: dict[tuple[str, str], np.ndarray | None] = dict.fromkeys(pairs)
    pair_priorities: dict[tuple[str, str], np.ndarray | None] = dict.fromkeys(pairs)
    found_pairs: set[tuple[str, str]] = set()
    rng = np.random.default_rng(seed)
    total = None
    counts = None
    x_values = None

    for batch_idx in _prioritized_batch_indices(loader, num_batches, batch_indices):
        if pair_set <= found_pairs:
            break
        predictions, batch = loader[batch_idx]
        grid = predictions["grid"]
        if x_values is None:
            x_values = batch["grid_points"]["X"][:, 0]

        row_pairs = _batch_pairs(batch)
        keep = _pair_mask(row_pairs, pair_set)
        if not keep.any():
            continue

        for row_idx in np.flatnonzero(keep):
            row_idx = int(row_idx)
            pair = row_pairs[row_idx]
            found_pairs.add(pair)
            trajectories = _as_numpy(grid[row_idx, 0, :, :, 0])

            total, counts = _add_to_mean(total, counts, trajectories)
            pair_samples[pair], pair_priorities[pair] = update_sample_trajectories(
                pair_samples[pair],
                pair_priorities[pair],
                trajectories,
                rng,
                pair_quotas[pair],
            )

    missing_pairs = pair_set - found_pairs
    if missing_pairs:
        examples = sorted(missing_pairs)[:5]
        msg = (
            f"{method_dir}: {len(missing_pairs)} of {len(pair_set)} requested "
            f"condition pairs were not found; examples={examples}"
        )
        raise ValueError(msg)

    found_samples = [sample for sample in pair_samples.values() if sample is not None]
    if not found_samples or total is None or counts is None or x_values is None:
        msg = (
            f"No trajectories found in {method_dir} for any of the "
            f"{len(pairs)} requested pairs"
        )
        raise ValueError(msg)

    return TrajectoryPanel(
        x_values=_as_numpy(x_values),
        samples=np.concatenate(found_samples, axis=0),
        mean=_panel_mean(total, counts),
    )


def load_ground_truth_panel(
    method_dir: Path,
    genotype_id: str | None,
    yearsite_uid: str | None,
    rng: np.random.Generator,
    num_samples: int,
    num_batches: int | None,
    truth_source: TruthSource | None = None,
) -> TrajectoryPanel:
    loader = BatchLoader(method_dir, load_predictions=False)
    truth = truth_source if truth_source is not None else SyntheticTruth()
    sample_trajectories = None
    sample_priorities = None
    total = None
    counts = None
    x_values = None

    for batch_idx in range(_num_batches_to_load(loader, num_batches)):
        batch = _load_batch_dict(loader, batch_idx)
        keep = sample_mask(batch, genotype_id, yearsite_uid)
        if not keep.any():
            continue
        truth_x, truth_rows = truth.rows(method_dir, batch)
        trajectories = np.concatenate(
            [truth_rows[row_idx] for row_idx in np.flatnonzero(keep)], axis=0
        )

        total, counts = _add_to_mean(total, counts, trajectories)
        sample_trajectories, sample_priorities = update_sample_trajectories(
            sample_trajectories, sample_priorities, trajectories, rng, num_samples
        )
        if x_values is None:
            x_values = truth_x

    return _finalize_panel(
        x_values=x_values,
        samples=sample_trajectories,
        total=total,
        counts=counts,
        source=str(method_dir),
        genotype_id=genotype_id,
        yearsite_uid=yearsite_uid,
    )


def load_ground_truth_stratified(
    method_dir: Path,
    pairs: list[tuple[str, str]],
    *,
    seed: int,
    num_samples: int,
    num_batches: int | None,
    batch_indices: list[int] | None = None,
    truth_source: TruthSource | None = None,
) -> TrajectoryPanel:
    """Load ground-truth trajectories for a fixed set of condition pairs."""
    loader = BatchLoader(method_dir, load_predictions=False)
    truth = truth_source if truth_source is not None else SyntheticTruth()
    pair_set = set(pairs)
    pair_quotas = _pair_sample_quotas(pairs, num_samples)
    pair_samples: dict[tuple[str, str], np.ndarray | None] = dict.fromkeys(pairs)
    pair_priorities: dict[tuple[str, str], np.ndarray | None] = dict.fromkeys(pairs)
    found_pairs: set[tuple[str, str]] = set()
    rng = np.random.default_rng(seed)
    total = None
    counts = None
    x_values = None

    for batch_idx in _prioritized_batch_indices(loader, num_batches, batch_indices):
        if pair_set <= found_pairs:
            break
        batch = _load_batch_dict(loader, batch_idx)
        row_pairs = _batch_pairs(batch)
        keep = _pair_mask(row_pairs, pair_set)
        if not keep.any():
            continue

        truth_x, truth_rows = truth.rows(method_dir, batch)
        if x_values is None:
            x_values = truth_x

        for row_idx in np.flatnonzero(keep):
            row_idx = int(row_idx)
            pair = row_pairs[row_idx]
            found_pairs.add(pair)
            trajectories = truth_rows[row_idx]

            total, counts = _add_to_mean(total, counts, trajectories)
            pair_samples[pair], pair_priorities[pair] = update_sample_trajectories(
                pair_samples[pair],
                pair_priorities[pair],
                trajectories,
                rng,
                pair_quotas[pair],
            )

    missing_pairs = pair_set - found_pairs
    if missing_pairs:
        examples = sorted(missing_pairs)[:5]
        msg = (
            f"{method_dir}: {len(missing_pairs)} of {len(pair_set)} requested "
            f"condition pairs were not found; examples={examples}"
        )
        raise ValueError(msg)

    found_samples = [sample for sample in pair_samples.values() if sample is not None]
    if not found_samples or total is None or counts is None or x_values is None:
        msg = (
            f"No trajectories found in {method_dir} for any of the "
            f"{len(pairs)} requested pairs"
        )
        raise ValueError(msg)

    return TrajectoryPanel(
        x_values=_as_numpy(x_values),
        samples=np.concatenate(found_samples, axis=0),
        mean=_panel_mean(total, counts),
    )


def _select_nearest_percentile(
    grouped_values: dict[str, list[float]], percentile: float
) -> str:
    means = {
        group_id: float(np.mean(values)) for group_id, values in grouped_values.items()
    }
    percentile_value = float(np.percentile(list(means.values()), percentile * 100.0))
    selected = min(
        means, key=lambda group_id: (abs(means[group_id] - percentile_value), group_id)
    )
    return str(selected)


def select_default_ground_truth_targets(
    reference_method_dir: Path,
    num_batches: int | None,
    truth_source: TruthSource | None = None,
) -> tuple[str, str]:
    loader = BatchLoader(reference_method_dir, load_predictions=False)
    truth = truth_source if truth_source is not None else SyntheticTruth()
    genotype_max_heights: dict[str, list[float]] = {}
    environment_max_heights: dict[str, list[float]] = {}

    for batch_idx in range(_num_batches_to_load(loader, num_batches)):
        batch = _load_batch_dict(loader, batch_idx)
        genotype_ids = ids(batch, "genotype_id")
        environment_ids = ids(batch, "yearsite_uid")
        _, truth_rows = truth.rows(reference_method_dir, batch)

        for row_idx, (genotype_id, environment_id) in enumerate(
            zip(genotype_ids, environment_ids, strict=True)
        ):
            values = np.nanmax(truth_rows[row_idx], axis=-1).reshape(-1).tolist()
            genotype_max_heights.setdefault(genotype_id, []).extend(values)
            environment_max_heights.setdefault(environment_id, []).extend(values)

    if not genotype_max_heights or not environment_max_heights:
        msg = f"No ground-truth trajectories found in {reference_method_dir}"
        raise ValueError(msg)

    target_genotype = _select_nearest_percentile(
        genotype_max_heights, DEFAULT_TARGET_PERCENTILE
    )
    target_environment = _select_nearest_percentile(
        environment_max_heights, DEFAULT_TARGET_PERCENTILE
    )
    return target_genotype, target_environment


def _method_dir(model_root: Path, conditioning_dir: str, method: str) -> Path:
    return model_root / conditioning_dir / method


def _validate_inputs(
    model_roots: list[Path],
    model_names: list[str],
    method: str,
    reference_model_index: int,
    num_batches: int | None,
    num_samples: int,
) -> int:
    if not model_roots:
        msg = "At least one --model-root is required."
        raise ValueError(msg)
    if len(model_roots) != len(model_names):
        msg = (
            f"Number of model roots ({len(model_roots)}) must match "
            f"number of model names ({len(model_names)})."
        )
        raise ValueError(msg)
    if not -len(model_roots) <= reference_model_index < len(model_roots):
        msg = (
            f"--reference-model-index must be between 0 and {len(model_roots) - 1} "
            f"or a valid negative index."
        )
        raise ValueError(msg)
    if num_batches is not None and num_batches <= 0:
        msg = "--num-batches must be positive when supplied."
        raise ValueError(msg)
    if num_samples <= 0:
        msg = "--num-samples must be positive."
        raise ValueError(msg)

    for root, name in zip(model_roots, model_names, strict=True):
        for conditioning_dir in CONDITIONING_DIRS:
            path = _method_dir(root, conditioning_dir, method)
            if not path.is_dir():
                msg = f"{name}: required method directory not found: {path}"
                raise ValueError(msg)

    return reference_model_index % len(model_roots)


def _resolve_targets(
    *,
    reference_method_dir: Path,
    num_batches: int | None,
    target_genotype: str | None,
    target_environment: str | None,
    truth_source: TruthSource | None = None,
) -> tuple[str, str, str, str]:
    default_genotype, default_environment = select_default_ground_truth_targets(
        reference_method_dir, num_batches, truth_source
    )
    genotype_id = target_genotype if target_genotype is not None else default_genotype
    yearsite_uid = (
        target_environment if target_environment is not None else default_environment
    )
    default_source = f"default percentile {DEFAULT_TARGET_PERCENTILE:.2f}"
    genotype_source = "CLI override" if target_genotype is not None else default_source
    environment_source = (
        "CLI override" if target_environment is not None else default_source
    )
    return genotype_id, yearsite_uid, genotype_source, environment_source


def _write_selected_targets(
    output_dir: Path,
    *,
    genotype_id: str,
    yearsite_uid: str,
    genotype_source: str,
    environment_source: str,
    reference_method_dir: Path,
    num_batches: int | None,
) -> None:
    num_batches_text = "all available" if num_batches is None else str(num_batches)
    text = (
        f"genotype_id={genotype_id}\n"
        f"genotype_source={genotype_source}\n"
        f"yearsite_uid={yearsite_uid}\n"
        f"yearsite_source={environment_source}\n"
        f"selection_rule={SELECTION_RULE}\n"
        f"default_target_percentile={DEFAULT_TARGET_PERCENTILE:.2f}\n"
        f"reference_method_dir={reference_method_dir}\n"
        f"num_batches={num_batches_text}\n"
    )
    (output_dir / "selected_targets.txt").write_text(text)


def _make_row_condition_sets(
    *,
    all_pairs: list[tuple[str, str]] | None,
    reference_method_dir: Path,
    genotype_id: str,
    yearsite_uid: str,
    num_samples: int,
    seed: int,
    num_batches: int | None,
) -> dict[str, list[tuple[str, str]]]:
    """Select the fixed condition-pair set used by each figure row."""
    if all_pairs is None:
        all_pairs = collect_condition_pairs(reference_method_dir, num_batches)

    row_filters: list[tuple[str, str | None, str | None]] = [
        ("Prior", None, None),
        ("Geno", genotype_id, None),
        ("Env", None, yearsite_uid),
        ("Geno + Env", genotype_id, yearsite_uid),
    ]
    row_pairs: dict[str, list[tuple[str, str]]] = {}
    for label, row_genotype_id, row_yearsite_uid in row_filters:
        pairs = _select_pairs(
            all_pairs,
            row_genotype_id,
            row_yearsite_uid,
            _rng_for(seed, "condition-pairs", label),
            num_samples,
        )
        if not pairs:
            msg = (
                f"No condition pairs selected for row {label!r} "
                f"with genotype_id={row_genotype_id!r}, "
                f"yearsite_uid={row_yearsite_uid!r}"
            )
            raise ValueError(msg)
        row_pairs[label] = pairs
    return row_pairs


def _plot_panel(
    ax: plt.Axes,
    panel: TrajectoryPanel,
    *,
    color: tuple[float, float, float] | str,
    sample_alpha: float,
    sample_linewidth: float,
) -> None:
    for trajectory in panel.samples:
        ax.plot(
            panel.x_values,
            trajectory,
            color=color,
            alpha=sample_alpha,
            linewidth=sample_linewidth,
        )
    ax.plot(
        panel.x_values, panel.mean, color=MEAN_COLOR, linewidth=MEAN_LINEWIDTH, zorder=5
    )


def _format_axes(axes: np.ndarray, x_limits: tuple[float, float] | None = None) -> None:
    for ax in axes.flat:
        if x_limits is None:
            ax.set_xlim(180, 320)
            ax.set_xticks(np.arange(180, 321, 40))
        else:
            # A caller with its own scored day range sets it here and keeps the
            # automatic ticks, which the fixed 40-day steps would not fit.
            ax.set_xlim(*x_limits)
        ax.set_ylim(bottom=0)
    if x_limits is not None:
        # Autoscaling sees the curves outside the window too, so fit the shared
        # y axis to the drawn days only.
        top = max(
            np.nanmax(y[(x >= x_limits[0]) & (x <= x_limits[1])], initial=0.0)
            for ax in axes.flat
            for x, y in (line.get_data() for line in ax.lines)
        )
        axes.flat[0].set_ylim(0, 1.05 * top)


def _save_grid_figure(
    fig: plt.Figure,
    axes: np.ndarray,
    output_dir: Path,
    stem: str,
    row_labels: list[str],
) -> None:
    # Keep the labels above the top grid line.
    add_panel_labels(axes.flat, offset_points=(6, 2))
    fig.tight_layout(rect=(0.10, 0.085, 0.995, 0.98))

    grid_left = axes[-1, 0].get_position().x0
    grid_right = axes[-1, -1].get_position().x1
    fig.text(
        (grid_left + grid_right) / 2, 0.052, "Day of year", ha="center", va="center"
    )

    for row_idx, label in enumerate(row_labels):
        box = axes[row_idx, 0].get_position()
        fig.text(
            box.x0 - 0.045,
            (box.y0 + box.y1) / 2,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
        )

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


def _make_conditioning_model_comparison(
    *,
    model_roots: list[Path],
    model_names: list[str],
    model_colors: list[tuple[float, float, float]],
    reference_root: Path,
    method: str,
    output_dir: Path,
    row_condition_sets: dict[str, list[tuple[str, str]]],
    row_batch_indices: dict[str, list[int]],
    seed: int,
    num_batches: int | None,
    num_samples: int,
    sample_alpha: float,
    sample_linewidth: float,
    row_labels: list[str] | None = None,
    truth_source: TruthSource | None = None,
    x_limits: tuple[float, float] | None = None,
) -> None:
    rows: list[tuple[str, str, list[tuple[str, str]]]] = [
        ("Prior", "noenv_nogeno", row_condition_sets["Prior"]),
        ("Geno", "noenv_geno", row_condition_sets["Geno"]),
        ("Env", "env_nogeno", row_condition_sets["Env"]),
        ("Geno + Env", "env_geno", row_condition_sets["Geno + Env"]),
    ]
    columns = ["Ground Truth", *model_names]
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(3.75 * len(columns), 11.6),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(rows), len(columns))

    for col_idx, column in enumerate(columns):
        axes[0, col_idx].set_title(column)

    stem = "conditioning_model_comparison"
    for row_idx, (row_label, conditioning_dir, pairs) in enumerate(rows):
        row_seed = _derived_seed(seed, stem, row_label, "trajectories")
        ground_truth = load_ground_truth_stratified(
            _method_dir(reference_root, conditioning_dir, method),
            pairs,
            seed=row_seed,
            num_samples=num_samples,
            num_batches=num_batches,
            batch_indices=row_batch_indices[row_label],
            truth_source=truth_source,
        )
        _plot_panel(
            axes[row_idx, 0],
            ground_truth,
            color=COLORS["gray"],
            sample_alpha=sample_alpha,
            sample_linewidth=sample_linewidth,
        )

        for col_idx, (model_root, color) in enumerate(
            zip(model_roots, model_colors, strict=True), start=1
        ):
            panel = load_model_stratified(
                _method_dir(model_root, conditioning_dir, method),
                pairs,
                seed=row_seed,
                num_samples=num_samples,
                num_batches=num_batches,
                batch_indices=row_batch_indices[row_label],
            )
            _plot_panel(
                axes[row_idx, col_idx],
                panel,
                color=color,
                sample_alpha=sample_alpha,
                sample_linewidth=sample_linewidth,
            )

    _format_axes(axes, x_limits)
    display_row_labels = (
        row_labels if row_labels is not None else [row[0] for row in rows]
    )
    _save_grid_figure(fig, axes, output_dir, stem, display_row_labels)


def _make_env_geno_aggregation_comparison(
    *,
    model_roots: list[Path],
    model_names: list[str],
    model_colors: list[tuple[float, float, float]],
    reference_root: Path,
    method: str,
    output_dir: Path,
    row_condition_sets: dict[str, list[tuple[str, str]]],
    row_batch_indices: dict[str, list[int]],
    seed: int,
    num_batches: int | None,
    num_samples: int,
    sample_alpha: float,
    sample_linewidth: float,
    row_labels: list[str] | None = None,
    truth_source: TruthSource | None = None,
    x_limits: tuple[float, float] | None = None,
) -> None:
    rows: list[tuple[str, list[tuple[str, str]]]] = [
        ("Prior", row_condition_sets["Prior"]),
        ("Geno", row_condition_sets["Geno"]),
        ("Env", row_condition_sets["Env"]),
        ("Geno + Env", row_condition_sets["Geno + Env"]),
    ]
    columns = ["Ground Truth", *model_names]
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(3.75 * len(columns), 11.6),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(rows), len(columns))

    for col_idx, column in enumerate(columns):
        axes[0, col_idx].set_title(column)

    stem = "env_geno_aggregation_comparison"
    for row_idx, (row_label, pairs) in enumerate(rows):
        row_seed = _derived_seed(seed, stem, row_label, "trajectories")
        ground_truth = load_ground_truth_stratified(
            _method_dir(reference_root, "env_geno", method),
            pairs,
            seed=row_seed,
            num_samples=num_samples,
            num_batches=num_batches,
            batch_indices=row_batch_indices[row_label],
            truth_source=truth_source,
        )
        _plot_panel(
            axes[row_idx, 0],
            ground_truth,
            color=COLORS["gray"],
            sample_alpha=sample_alpha,
            sample_linewidth=sample_linewidth,
        )

        for col_idx, (model_root, color) in enumerate(
            zip(model_roots, model_colors, strict=True), start=1
        ):
            panel = load_model_stratified(
                _method_dir(model_root, "env_geno", method),
                pairs,
                seed=row_seed,
                num_samples=num_samples,
                num_batches=num_batches,
                batch_indices=row_batch_indices[row_label],
            )
            _plot_panel(
                axes[row_idx, col_idx],
                panel,
                color=color,
                sample_alpha=sample_alpha,
                sample_linewidth=sample_linewidth,
            )

    _format_axes(axes, x_limits)
    display_row_labels = (
        row_labels if row_labels is not None else [row[0] for row in rows]
    )
    _save_grid_figure(fig, axes, output_dir, stem, display_row_labels)


def plot_conditioning_sample_comparison(
    *,
    model_roots: list[str],
    model_names: list[str],
    method: str,
    output_folder: str | None,
    reference_model_index: int,
    num_batches: int | None,
    num_samples: int,
    seed: int,
    target_genotype: str | None,
    target_environment: str | None,
    sample_alpha: float,
    sample_linewidth: float,
    row_labels: list[str] | None = None,
    truth_source: TruthSource | None = None,
    x_limits: tuple[float, float] | None = None,
) -> None:
    setup_style()
    roots = [Path(root) for root in model_roots]
    reference_idx = _validate_inputs(
        roots, model_names, method, reference_model_index, num_batches, num_samples
    )
    reference_root = roots[reference_idx]
    reference_method_dir = _method_dir(reference_root, "env_geno", method)
    output_dir = Path(output_folder) if output_folder else DEFAULT_OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    genotype_id, yearsite_uid, genotype_source, environment_source = _resolve_targets(
        reference_method_dir=reference_method_dir,
        num_batches=num_batches,
        target_genotype=target_genotype,
        target_environment=target_environment,
        truth_source=truth_source,
    )
    _write_selected_targets(
        output_dir,
        genotype_id=genotype_id,
        yearsite_uid=yearsite_uid,
        genotype_source=genotype_source,
        environment_source=environment_source,
        reference_method_dir=reference_method_dir,
        num_batches=num_batches,
    )
    logger.info(
        "Using genotype_id={} ({}) and yearsite_uid={} ({})",
        genotype_id,
        genotype_source,
        yearsite_uid,
        environment_source,
    )

    model_colors = [
        COLOR_LIST[idx % len(COLOR_LIST)] for idx in range(len(model_names))
    ]
    pair_batches = collect_condition_pair_batches(reference_method_dir, num_batches)
    row_condition_sets = _make_row_condition_sets(
        all_pairs=sorted(pair_batches),
        reference_method_dir=reference_method_dir,
        genotype_id=genotype_id,
        yearsite_uid=yearsite_uid,
        num_samples=num_samples,
        seed=seed,
        num_batches=num_batches,
    )
    row_batch_indices = {
        label: sorted({idx for pair in pairs for idx in pair_batches[pair]})
        for label, pairs in row_condition_sets.items()
    }
    logger.info(
        "Selected condition-pair counts by row: {}",
        {label: len(pairs) for label, pairs in row_condition_sets.items()},
    )
    _make_conditioning_model_comparison(
        model_roots=roots,
        model_names=model_names,
        model_colors=model_colors,
        reference_root=reference_root,
        method=method,
        output_dir=output_dir,
        row_condition_sets=row_condition_sets,
        row_batch_indices=row_batch_indices,
        seed=seed,
        num_batches=num_batches,
        num_samples=num_samples,
        sample_alpha=sample_alpha,
        sample_linewidth=sample_linewidth,
        row_labels=row_labels,
        truth_source=truth_source,
        x_limits=x_limits,
    )
    _make_env_geno_aggregation_comparison(
        model_roots=roots,
        model_names=model_names,
        model_colors=model_colors,
        reference_root=reference_root,
        method=method,
        output_dir=output_dir,
        row_condition_sets=row_condition_sets,
        row_batch_indices=row_batch_indices,
        seed=seed,
        num_batches=num_batches,
        num_samples=num_samples,
        sample_alpha=sample_alpha,
        sample_linewidth=sample_linewidth,
        row_labels=row_labels,
        truth_source=truth_source,
        x_limits=x_limits,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper figures comparing conditioning-specific samples and "
            "env_geno aggregation."
        )
    )
    parser.add_argument(
        "--model-roots",
        type=str,
        nargs="+",
        required=True,
        help="Model result roots containing conditioning directories.",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        required=True,
        help="Display names for each model root, in the same order.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Prediction method subfolder below each conditioning directory.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help=f"Output folder (defaults to {DEFAULT_OUTPUT_FOLDER}).",
    )
    parser.add_argument(
        "--reference-model-index",
        type=int,
        default=-1,
        help=(
            "Model root index used for ground truth and target selection. "
            "Default -1 uses the last model root, matching the reviewed prototype "
            "and avoiding assumptions that batch data is identical across roots."
        ),
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Number of prediction batches to load (defaults to all).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=(
            "Maximum number of displayed trajectories per panel. Also caps the "
            "number of condition pairs selected per row; trajectories are sampled "
            "from all draws/prediction samples for those pairs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for displayed trajectory sampling.",
    )
    parser.add_argument(
        "--target-genotype",
        type=str,
        default=None,
        help=(
            "Optional genotype_id override. Defaults to the genotype nearest the "
            f"{DEFAULT_TARGET_PERCENTILE * 100:.0f}%% mean-maximum-height percentile."
        ),
    )
    parser.add_argument(
        "--target-environment",
        type=str,
        default=None,
        help=(
            "Optional yearsite_uid override. Defaults to the environment nearest "
            f"the {DEFAULT_TARGET_PERCENTILE * 100:.0f}%% "
            "mean-maximum-height percentile."
        ),
    )
    parser.add_argument(
        "--sample-alpha",
        type=float,
        default=SAMPLE_ALPHA,
        help="Alpha for sampled trajectories.",
    )
    parser.add_argument(
        "--sample-linewidth",
        type=float,
        default=SAMPLE_LINEWIDTH,
        help="Line width for sampled trajectories.",
    )
    parser.add_argument(
        "--row-labels",
        type=str,
        nargs=4,
        default=None,
        help=(
            "Optional display labels for rows in order: Prior, Geno, Env, Geno + Env."
        ),
    )
    args = parser.parse_args()

    plot_conditioning_sample_comparison(
        model_roots=args.model_roots,
        model_names=args.model_names,
        method=args.method,
        output_folder=args.output_folder,
        reference_model_index=args.reference_model_index,
        num_batches=args.num_batches,
        num_samples=args.num_samples,
        seed=args.seed,
        target_genotype=args.target_genotype,
        target_environment=args.target_environment,
        sample_alpha=args.sample_alpha,
        sample_linewidth=args.sample_linewidth,
        row_labels=args.row_labels,
    )


if __name__ == "__main__":
    main()
