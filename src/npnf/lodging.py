"""Unified lodging detection utilities.

Lodging is detected when a plant's height drops significantly from its peak.
This module provides consistent detection logic for both FIP1 real data and
model predictions.

Algorithm (FIP1 robust method):
- Max height: Median of top 3 heights (robust to outliers)
- Final height: Minimum of last N measurements after peak
- Classification: is_lodged = (drop_rel >= 0.2) AND (drop_abs >=0.1)
"""

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_RELATIVE_THRESHOLD = 0.2
DEFAULT_ABSOLUTE_THRESHOLD = 0.1
DEFAULT_TAIL_WINDOW = 5
DEFAULT_EPS = 1e-6


@dataclass
class LodgingStats:
    """Statistics from lodging detection on a single height trajectory."""

    max_height: float
    final_height: float
    drop_abs: float
    drop_rel: float
    is_lodged: bool


@dataclass
class BatchLodgingStats:
    """Batched lodging detection results for multiple trajectories."""

    max_height: torch.Tensor  # (batch,)
    final_height: torch.Tensor  # (batch,)
    drop_abs: torch.Tensor  # (batch,)
    drop_rel: torch.Tensor  # (batch,)
    is_lodged: torch.Tensor  # (batch,) bool


def _robust_max_height_np(heights: np.ndarray) -> tuple[float, int]:
    """Compute robust max height using median of top 3 values.

    Returns the max height and the index of that height in the array.
    """
    if heights.size == 0:
        return 0.0, 0

    top_k = min(3, heights.size)
    partition_index = np.argpartition(heights, heights.size - top_k)[-top_k:]
    top_indices = sorted(partition_index, key=lambda index: (-heights[index], index))

    max_index = int(top_indices[1]) if top_k == 3 else int(top_indices[0])

    return float(heights[max_index]), max_index


def _tail_min_np(heights: np.ndarray, peak_index: int, tail_window: int) -> float:
    """Compute minimum height in tail window after peak."""
    post_peak = heights[peak_index + 1 :]
    if post_peak.size == 0:
        return float(heights[peak_index])

    tail = post_peak[-tail_window:] if post_peak.size > tail_window else post_peak

    return float(np.min(tail))


def robust_max_height(heights: np.ndarray | torch.Tensor) -> tuple[float, int]:
    """Robust peak height and its index, using the FIP1 median-of-top-3 rule.

    Public wrapper over the estimator ``detect_lodging`` uses internally, for
    callers that need the peak of a real (noisy) trajectory rather than the full
    lodging statistics.
    """
    if isinstance(heights, torch.Tensor):
        heights = heights.numpy(force=True)
    return _robust_max_height_np(np.asarray(heights, dtype=np.float32).ravel())


def detect_lodging(
    heights: np.ndarray | torch.Tensor,
    relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD,
    absolute_threshold: float = DEFAULT_ABSOLUTE_THRESHOLD,
    tail_window: int = DEFAULT_TAIL_WINDOW,
    eps: float = DEFAULT_EPS,
) -> LodgingStats:
    """Detect lodging in a single height trajectory.

    Uses the FIP1 robust method:
    - Max height is the median of the top 3 height measurements
    - Final height is the minimum of the last `tail_window` measurements after peak
    - Lodging requires BOTH relative and absolute thresholds to be exceeded

    Args:
        heights: 1D array of height measurements over time.
        relative_threshold: Minimum relative drop (fraction of max height).
        absolute_threshold: Minimum absolute drop in meters.
        tail_window: Number of measurements at end to consider for final height.
        eps: Small value for numerical stability.

    Returns:
        LodgingStats with detection results.
    """
    if isinstance(heights, torch.Tensor):
        heights = heights.numpy(force=True)

    heights = np.asarray(heights, dtype=np.float32).ravel()
    heights = heights[np.isfinite(heights)]

    if heights.size == 0:
        return LodgingStats(
            max_height=0.0,
            final_height=0.0,
            drop_abs=0.0,
            drop_rel=0.0,
            is_lodged=False,
        )

    max_height, peak_index = _robust_max_height_np(heights)

    if max_height <= eps:
        return LodgingStats(
            max_height=max_height,
            final_height=max_height,
            drop_abs=0.0,
            drop_rel=0.0,
            is_lodged=False,
        )

    final_height = _tail_min_np(heights, peak_index, tail_window)
    drop_abs = max_height - final_height
    drop_rel = drop_abs / max(max_height, eps)

    is_lodged = (drop_rel >= relative_threshold) and (drop_abs >= absolute_threshold)

    return LodgingStats(
        max_height=max_height,
        final_height=final_height,
        drop_abs=drop_abs,
        drop_rel=drop_rel,
        is_lodged=is_lodged,
    )


def detect_lodging_grid(
    grid: torch.Tensor,
    relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD,
    absolute_threshold: float = DEFAULT_ABSOLUTE_THRESHOLD,
    eps: float = DEFAULT_EPS,
) -> BatchLodgingStats:
    """Detect lodging from a 2D prediction grid (batch, time).

    This is a simplified version for prediction grids where we use:
    - Max height: Global maximum per sample
    - Final height: Last value per sample
    - No tail window (assumes grid represents full trajectory)

    This matches the behavior needed for evaluating model predictions where
    we want to compare the predicted maximum to the predicted final value.

    Args:
        grid: (batch, time) tensor of predicted heights.
        relative_threshold: Minimum relative drop.
        absolute_threshold: Minimum absolute drop in meters.
        eps: Small value for numerical stability.

    Returns:
        BatchLodgingStats with detection results for each sample.
    """
    max_heights = grid.max(dim=-1).values
    final_heights = grid[..., -1]

    drop_abs = max_heights - final_heights
    drop_rel = drop_abs / torch.clamp(max_heights.abs(), min=eps)

    is_lodged = (drop_rel >= relative_threshold) & (drop_abs >= absolute_threshold)

    return BatchLodgingStats(
        max_height=max_heights,
        final_height=final_heights,
        drop_abs=drop_abs,
        drop_rel=drop_rel,
        is_lodged=is_lodged,
    )
