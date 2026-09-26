"""In-memory observation residual and noisy-path calculations."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from npnf.metrics.sig_mmd import normalized_sig_mmd
from npnf.metrics.signature import to_metric_paths


def normalized_triplet_residuals(
    days: ArrayLike, heights: ArrayLike, max_gap: float = 7
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Return variance-normalized residuals, neighbor predictions and interior indices.

    Calculations use float64. With independent, homoscedastic errors and a
    linear signal, each residual has variance sigma squared, even on unequal grids.
    Adjacent residuals overlap: they must not be treated as independent replicates.
    """
    days = np.asarray(days, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    left, right = np.diff(days)[:-1], np.diff(days)[1:]
    indices = np.flatnonzero((left <= max_gap) & (right <= max_gap)) + 1
    a = right[indices - 1] / (left[indices - 1] + right[indices - 1])
    b = 1 - a
    prediction = a * heights[:, indices - 1] + b * heights[:, indices + 1]
    z = (heights[:, indices] - prediction) / np.sqrt(1 + a**2 + b**2)
    return z, prediction, indices


def sum_squared_triplet_residuals(
    days: ArrayLike, heights: ArrayLike, max_gap: float
) -> tuple[float, int]:
    """Return residual sum of squares and triplet count for pooled RMS variation."""
    residuals, _, _ = normalized_triplet_residuals(days, heights, max_gap)
    squared = float(np.sum(residuals**2))
    count = residuals.size
    return squared, count


def score_noisy_trajectories(
    real_paths: torch.Tensor,
    heights: torch.Tensor,
    noise: torch.Tensor,
    sigma_m: float,
    time: torch.Tensor,
    height_scale: float,
    *,
    lead_lag: bool,
    max_batch: int,
) -> float:
    """Score one supplied observation-noise draw against real metric paths."""
    synthetic_paths = to_metric_paths(
        heights + sigma_m * noise, height_scale, time, lead_lag=lead_lag
    )
    return normalized_sig_mmd(real_paths, synthetic_paths, max_batch=max_batch)
