"""Core synthetic metric calculations for single-shot predictions."""

import argparse
from pathlib import Path

import torch
from loguru import logger

from npnf.data.batch_loader import BatchLoader
from npnf.metrics.single_shot import (
    compute_epistemic_std,
    compute_mae,
    save_synthetic_metrics_csv,
)
from npnf.scripts.utils.prediction import unnest_tensor
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    SyntheticBatchReconstructionCache,
    reconstruct_synthetic_batch,
)


def _process_sample(
    pred: torch.Tensor, y_orig: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process a single sample, returning MC samples and clean ground truth."""
    pred = unnest_tensor(pred)
    y_orig = unnest_tensor(y_orig)

    # Get midpoint to split Y_original
    num_total = len(y_orig)
    midpoint = num_total // 2

    # Second half of Y_original
    y_clean_second_half = y_orig[midpoint:]

    # Squeeze singleton dims: (1, 128, 73, 1) -> (128, 73) or (1, 1, 73, 1) -> (73,)
    pred = pred.squeeze()
    if pred.dim() == 1:  # Single MC sample case
        pred = pred.unsqueeze(0)
    mc_flat = pred.reshape(pred.shape[0], -1)
    clean_flat = y_clean_second_half.reshape(-1)

    return mc_flat, clean_flat


def calculate_single_shot_metrics(method_dir: Path) -> dict:
    """Calculate metrics for single-shot predictions (predict_half format).

    Args:
        method_dir: Path to method directory containing batch subdirectories

    Returns:
        Dictionary with mae_clean, mean_epistemic_std
    """
    loader = BatchLoader(method_dir)
    cache = SyntheticBatchReconstructionCache()

    all_mc_samples = []
    all_clean = []

    for predictions, batch_dict in loader:
        try:
            pred_context_posteriors = predictions["targets"]
        except KeyError:
            msg = (
                f"{method_dir}: calculate_synthetic.py requires "
                "predictions['targets']; grid-only one-step predictions are "
                "unsupported by this legacy metric"
            )
            raise ValueError(msg) from None

        batch_dict = reconstruct_synthetic_batch(method_dir, batch_dict, cache=cache)
        height_data = batch_dict["data"].get("height")
        if height_data is None or "Y_original" not in height_data:
            msg = f"{method_dir}: batch data must contain data/height/Y_original"
            raise ValueError(msg)

        Y_original = height_data["Y_original"]

        # Flatten if needed (within-batch structure)
        if len(pred_context_posteriors.shape) > 4:
            pred_context_posteriors = pred_context_posteriors.reshape(
                -1, *pred_context_posteriors.shape[2:]
            )

        # Process each sample in this batch
        y_orig_list = list(Y_original)
        pred_list = list(pred_context_posteriors)

        for pred, y_orig in zip(pred_list, y_orig_list, strict=True):
            mc_flat, clean_flat = _process_sample(pred, y_orig)
            all_mc_samples.append(mc_flat)
            all_clean.append(clean_flat)

    if not all_mc_samples:
        logger.warning("No valid samples found - skipping synthetic metrics")
        return {}

    # Concatenate across all samples from all batches
    all_mc = torch.cat(all_mc_samples, dim=1)
    all_truth = torch.cat(all_clean, dim=0)

    return {
        "mae_clean": compute_mae(all_mc, all_truth).item(),
        "mean_epistemic_std": compute_epistemic_std(all_mc).item(),
    }


def save_single_shot_metrics(output_dir: Path, metrics: dict) -> None:
    """Save single-shot metrics to CSV."""
    save_synthetic_metrics_csv(output_dir, metrics)


def calculate_synthetic_metric(results_folder: str) -> None:
    """Main entry point for single-shot metric calculation."""
    method_dir = Path(results_folder)

    logger.info(f"Calculating single-shot synthetic metrics for: {method_dir}")

    metrics = calculate_single_shot_metrics(method_dir)
    save_single_shot_metrics(method_dir, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate synthetic metrics for single-shot predictions"
    )
    parser.add_argument("--results-folder", type=str, required=True)
    args = parser.parse_args()
    calculate_synthetic_metric(args.results_folder)
