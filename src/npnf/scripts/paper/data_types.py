"""Shared type definitions for paper analysis scripts."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionStats:
    """Statistics from lodging detection on model predictions.

    These are numpy arrays (post-.numpy() conversion from torch tensors).
    """

    max_height: np.ndarray
    final_height: np.ndarray
    drop_abs: np.ndarray
    drop_rel: np.ndarray
    is_lodged: np.ndarray


@dataclass
class BinnedRates:
    """Binned lodging rates and counts for a single model or ground truth."""

    name: str
    rates: np.ndarray
    counts: np.ndarray
    lodged_counts: np.ndarray
    single_height: float | None = None


@dataclass
class GroundTruth:
    """Ground truth lodging labels and heights."""

    labels: np.ndarray  # bool array indicating lodging status
    max_heights: np.ndarray  # float array of actual max heights from Y_original
