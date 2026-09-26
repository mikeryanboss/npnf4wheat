"""Variance calibration package."""

from .extraction import extract_variance_params
from .objectives import (
    DEFAULT_SEED_LIST,
    compute_all_metrics,
    compute_swiss_calibration_targets,
    compute_variance_components,
    generate_synthetic_for_calibration,
    robust_variance_objective,
)
from .optimization import optimize_variance_params
from .visualize import (
    plot_temperature_validation,
    plot_variance_optimization,
    plot_variance_validation,
)

__all__ = [
    "DEFAULT_SEED_LIST",
    "compute_all_metrics",
    "compute_swiss_calibration_targets",
    "compute_variance_components",
    "extract_variance_params",
    "generate_synthetic_for_calibration",
    "optimize_variance_params",
    "plot_temperature_validation",
    "plot_variance_optimization",
    "plot_variance_validation",
    "robust_variance_objective",
]
