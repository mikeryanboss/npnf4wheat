"""Utilities for batch data processing and feature extraction."""

from typing import Any, Literal

import torch

Metric = Literal["max_height", "augc", "time_to_pct_max"]


def compute_all_batch_max_heights(batch: Any) -> torch.Tensor:
    """Compute max heights for all samples in a batch.

    Args:
        batch: Flattened batch data (batch_dict["data"].flatten())

    Returns:
        Tensor of max heights with shape (num_samples,)
    """
    max_heights = []
    for i in range(batch.batch_size[0]):
        sample_heights = batch[i]["height"]["Y_original"]
        if sample_heights.is_nested:
            sample_heights = sample_heights.values()
        max_heights.append(sample_heights.flatten().max())
    return torch.stack(max_heights)


def compute_all_batch_augc(batch: Any) -> torch.Tensor:
    """Compute area under growth curve for all samples.

    Restricts to grid day range (150-349) for fair comparison with predictions.

    Args:
        batch: Flattened batch data (batch_dict["data"].flatten())

    Returns:
        Tensor of AUGC values with shape (num_samples,)
    """
    augcs = []
    for i in range(batch.batch_size[0]):
        height = batch[i]["height"]
        x = height["X"].flatten()
        y = height["Y_original"]
        if y.is_nested:
            y = y.values()
        y = y.flatten()

        mask = (x >= 150) & (x <= 349)
        x_masked = x[mask]
        y_masked = y[mask]

        if len(x_masked) < 2:
            augcs.append(torch.tensor(0.0))
            continue

        augcs.append(torch.trapezoid(y_masked, x_masked))

    return torch.stack(augcs)


def compute_all_batch_time_to_pct_max(batch: Any, threshold_pct: float) -> torch.Tensor:
    """Compute day each sample first reaches threshold_pct of max height.

    Args:
        batch: Flattened batch data (batch_dict["data"].flatten())
        threshold_pct: Fraction of max height (e.g. 0.9 for 90%)

    Returns:
        Tensor of day values with shape (num_samples,)
    """
    days = []
    for i in range(batch.batch_size[0]):
        height = batch[i]["height"]
        x = height["X"].flatten()
        y = height["Y_original"]
        if y.is_nested:
            y = y.values()
        y = y.flatten()

        max_h = y.max()
        threshold = threshold_pct * max_h
        crossed = y >= threshold
        first_index = crossed.float().argmax().long()
        days.append(x[first_index])

    return torch.stack(days)


def extract_predicted_feature(
    predictions_grid: torch.Tensor, metric: Metric, threshold_pct: float
) -> torch.Tensor:
    """Extract a feature from grid predictions.

    Args:
        predictions_grid: Grid predictions with shape (..., 200, 1)
        metric: Feature to extract
        threshold_pct: Fraction of max height for time_to_pct_max

    Returns:
        Flattened tensor of feature values
    """
    heights = predictions_grid[..., 0]

    if metric == "max_height":
        return heights.max(dim=-1).values.flatten()

    if metric == "augc":
        return torch.trapezoid(heights, dx=1.0, dim=-1).flatten()

    if metric == "time_to_pct_max":
        max_h = heights.max(dim=-1).values
        threshold = threshold_pct * max_h.unsqueeze(-1)
        crossed = heights >= threshold
        first_index = crossed.float().argmax(dim=-1)
        day = 150 + first_index
        return day.float().flatten()

    msg = f"Unknown metric: {metric}"
    raise ValueError(msg)


def extract_actual_feature(
    batch: Any, metric: Metric, threshold_pct: float
) -> torch.Tensor:
    """Extract a feature from actual observations.

    Args:
        batch: Flattened batch data (batch_dict["data"].flatten())
        metric: Feature to extract
        threshold_pct: Fraction of max height for time_to_pct_max

    Returns:
        Tensor of feature values with shape (num_samples,)
    """
    if metric == "max_height":
        return compute_all_batch_max_heights(batch)
    if metric == "augc":
        return compute_all_batch_augc(batch)
    if metric == "time_to_pct_max":
        return compute_all_batch_time_to_pct_max(batch, threshold_pct)

    msg = f"Unknown metric: {metric}"
    raise ValueError(msg)
