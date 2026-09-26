"""Single-shot synthetic prediction metrics and their CSV output."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
from loguru import logger


def compute_mae(
    mc_predictions: torch.Tensor, ground_truth: torch.Tensor
) -> torch.Tensor:
    """Compute MAE between prediction mean and ground truth.

    Args:
        mc_predictions: Monte Carlo samples with shape (num_samples, ...)
        ground_truth: Ground truth values with shape (...)

    Returns:
        Scalar tensor with MAE value (for accumulation).
    """
    pred_mean = mc_predictions.mean(dim=0)
    return (pred_mean - ground_truth).abs().mean()


def compute_epistemic_std(mc_samples: torch.Tensor) -> torch.Tensor:
    """Compute mean epistemic standard deviation across MC samples.

    Args:
        mc_samples: Monte Carlo samples with shape (num_samples, ...)

    Returns:
        Scalar tensor with mean std value (for accumulation).
    """
    return mc_samples.std(dim=0).mean()


def save_synthetic_metrics_csv(
    output_dir: Path,
    metrics: dict[str, list[float] | float],
    filename: str = "synthetic_metrics.csv",
    include_context_index: bool = False,
) -> None:
    """Save synthetic metrics to CSV with optional num_context index.

    Handles both scalar and list inputs automatically.

    Args:
        output_dir: Directory to save the CSV file
        metrics: Dictionary with metric names as keys and values as lists or scalars.
        filename: Name of the output CSV file
        include_context_index: If True, add "num_context" as row index column
    """
    if not metrics:
        logger.warning("No metrics to save")
        return

    # Normalize scalars to lists
    data = {}
    for key, value in metrics.items():
        data[key] = value if isinstance(value, list) else [value]

    df = pl.DataFrame(data)
    if include_context_index:
        df = df.with_row_index("num_context")

    output_path = output_dir / filename
    df.write_csv(output_path)
    logger.info(f"Saved metrics to {output_path}")
