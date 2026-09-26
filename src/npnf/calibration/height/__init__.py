"""Height calibration package."""

from .data import (
    FIP1YearData,
    HeightOptimizationConfig,
    HeightOptimizationResults,
    load_fip1_year_data,
)
from .evaluation.validation import (
    validate_fip1_correlation,
    validate_height,
    validate_warm_short_correlation,
    validate_year_pattern,
)
from .optimization import EvalContext, optimize_pool_params

__all__ = [
    "EvalContext",
    "FIP1YearData",
    "HeightOptimizationConfig",
    "HeightOptimizationResults",
    "load_fip1_year_data",
    "optimize_pool_params",
    "plot_height_distribution_comparison",
    "plot_trajectory_comparison_by_year",
    "validate_fip1_correlation",
    "validate_height",
    "validate_warm_short_correlation",
    "validate_year_pattern",
]


def __getattr__(name: str):
    """Lazy imports for visualization functions to avoid circular imports."""
    _viz_names = {
        "plot_height_distribution_comparison",
        "plot_trajectory_comparison_by_year",
    }
    if name in _viz_names:
        from .evaluation import visualize

        return getattr(visualize, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
