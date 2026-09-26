"""In-memory empirical lodging detection and parameter fitting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from torch import Tensor

from npnf.lodging import detect_lodging


class DetectorOptions(TypedDict):
    relative_threshold: float
    absolute_threshold: float
    tail_window: int


def fit_weibull_scale(
    observed_max_height: ArrayLike,
    target: int,
    *,
    shape: float,
    height_clamp: float,
    offset: float,
) -> float:
    """Match the mean Weibull event probability to the observed lodged fraction.

    Each input is a retained plot's maximum over its full observed schedule.
    Apply the fixed clamp and offset: h = max(min(max_height, clamp) - offset, 0).
    Solve sum(1 - exp(-(h / scale)**shape)) = target without conditioning on
    simulated trajectories, event detectability, or realized event counts.
    """
    height = np.maximum(
        np.minimum(np.asarray(observed_max_height, dtype=np.float64), height_clamp)
        - offset,
        0,
    )
    positive = height[height > 0]
    if not 0 < target < len(positive):
        msg = "A finite positive scale requires 0 < target < positive-height count"
        raise ValueError(msg)
    normalized_height = (-np.log1p(-target / len(positive))) ** (1 / shape)
    lower = positive.min() / normalized_height / 2
    upper = positive.max() / normalized_height * 2

    def residual(scale: float) -> float:
        probabilities = -np.expm1(-((height / scale) ** shape))
        return float(probabilities.sum() - target)

    return float(brentq(residual, lower, upper))


def detect_lodged_plots(
    observations: Iterable[tuple[Tensor, Tensor]], detector: DetectorOptions
) -> NDArray[np.bool_]:
    return np.asarray(
        [detect_lodging(values, **detector).is_lodged for _, values in observations]
    )
