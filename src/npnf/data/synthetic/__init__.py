"""Synthetic data generation for growth datasets.

This module provides generators for synthetic temperature profiles and
genotype parameters used in training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npnf.data.synthetic.height.genotype import (
        GenotypeParamBounds,
        GenotypeParams,
        SyntheticGenotypeGenerator,
    )
    from npnf.data.synthetic.height.response_surface import (
        DEFAULT_DEGREE,
        DEFAULT_T_KNOTS,
        DEFAULT_TAU_KNOTS,
        evaluate_surface,
    )
    from npnf.data.synthetic.temperature import SyntheticTemperatureGenerator

__all__ = [
    "DEFAULT_DEGREE",
    "DEFAULT_TAU_KNOTS",
    "DEFAULT_T_KNOTS",
    "GenotypeParamBounds",
    "GenotypeParams",
    "SyntheticGenotypeGenerator",
    "SyntheticTemperatureGenerator",
    "evaluate_surface",
]

_SYMBOL_TO_IMPORT: dict[str, tuple[str, str]] = {
    "GenotypeParamBounds": (
        "npnf.data.synthetic.height.genotype",
        "GenotypeParamBounds",
    ),
    "GenotypeParams": ("npnf.data.synthetic.height.genotype", "GenotypeParams"),
    "SyntheticGenotypeGenerator": (
        "npnf.data.synthetic.height.genotype",
        "SyntheticGenotypeGenerator",
    ),
    "SyntheticTemperatureGenerator": (
        "npnf.data.synthetic.temperature",
        "SyntheticTemperatureGenerator",
    ),
    "evaluate_surface": (
        "npnf.data.synthetic.height.response_surface",
        "evaluate_surface",
    ),
    "DEFAULT_T_KNOTS": (
        "npnf.data.synthetic.height.response_surface",
        "DEFAULT_T_KNOTS",
    ),
    "DEFAULT_TAU_KNOTS": (
        "npnf.data.synthetic.height.response_surface",
        "DEFAULT_TAU_KNOTS",
    ),
    "DEFAULT_DEGREE": ("npnf.data.synthetic.height.response_surface", "DEFAULT_DEGREE"),
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy synthetic submodules on first symbol access."""
    if name not in _SYMBOL_TO_IMPORT:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_name, symbol_name = _SYMBOL_TO_IMPORT[name]
    module = __import__(module_name, fromlist=[symbol_name])
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
