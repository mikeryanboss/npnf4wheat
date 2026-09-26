"""Shared histogram and rate calculation utilities for paper analysis scripts."""

import math

import numpy as np
import torch


def calculate_height_bins(
    heights: np.ndarray, bin_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate height bins and bin centers from data range.

    Args:
        heights: Array of height values to determine bin range.
        bin_width: Width of each bin.

    Returns:
        Tuple of (bin_edges, bin_centers).
    """
    height_min = float(heights.min())
    height_max = float(heights.max())

    num_bins = max(1, math.ceil((height_max - height_min) / bin_width))
    bins = np.linspace(height_min, height_min + bin_width * num_bins, num_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    return bins, centers


def calculate_binned_rates(
    values: np.ndarray, mask: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate binned rates where mask=True vs total.

    Args:
        values: Array of values to bin.
        mask: Boolean mask indicating which values to count in numerator.
        bins: Bin edges from np.histogram.

    Returns:
        Tuple of (rates, total_counts, masked_counts).
    """
    masked_counts, _ = np.histogram(values[mask], bins=bins)
    total_counts, _ = np.histogram(values, bins=bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.divide(
            masked_counts,
            total_counts,
            out=np.zeros_like(masked_counts, dtype=float),
            where=total_counts > 0,
        )
    return rates, total_counts, masked_counts


def mask_low_sample_rates(
    rates: np.ndarray,
    counts: np.ndarray,
    min_fraction: float,
    total_samples: float | None = None,
) -> np.ndarray:
    """Mask rates with insufficient samples as NaN.

    Args:
        rates: Rate values (numerator/denominator).
        counts: Sample counts per bin.
        min_fraction: Minimum fraction of total samples required.
        total_samples: Total sample count. If None, computed from counts.sum().

    Returns:
        Rates with low-sample bins set to NaN.
    """
    if min_fraction <= 0.0:
        return rates

    total = total_samples if total_samples is not None else float(np.nansum(counts))
    if total <= 0.0:
        return rates

    mask = counts >= (total * min_fraction)
    return np.where(mask, rates, np.nan)


def reshape_grid_predictions(grid_tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Reshape grid predictions to (num_samples, num_points).

    Handles multi-dimensional prediction tensors with trailing singleton
    dimensions by squeezing them out, then reshaping to 2D.

    Args:
        grid_tensor: Prediction tensor with shape (..., num_points) or
            (..., num_points, 1, ..., 1).

    Returns:
        Tuple of (reshaped_grid, num_samples) where reshaped_grid has
        shape (num_samples, num_points).

    Raises:
        ValueError: If tensor doesn't have at least 2 dimensions after
            squeezing trailing singletons.
    """
    grid = grid_tensor.detach().cpu().float()
    while grid.ndim > 1 and grid.shape[-1] == 1:
        grid = grid.squeeze(-1)
    if grid.ndim < 2:
        msg = "Grid predictions tensor must contain at least one grid dimension."
        raise ValueError(msg)

    num_points = int(grid.shape[-1])
    leading_shape = tuple(int(dim) for dim in grid.shape[:-1])
    num_samples = int(np.prod(leading_shape)) if leading_shape else 1
    reshaped = grid.reshape(num_samples, num_points)
    return reshaped, num_samples
