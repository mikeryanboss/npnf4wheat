"""Synthetic temperature generator with calibrated parameters.

This module provides a generator for synthetic hourly temperature courses that
match the statistical properties of real Swiss weather data. The generator creates
realistic site-year combinations with proper correlation structure.

Computation Hierarchy
---------------------
The temperature generation has two levels of computation:

1. **Year-level** (same across all sites for a given year): Stored in `YearParams`.
   - Shared weather series (AR(1) process).

2. **Site-year level** (unique to each combination): Computed in `_generate_year_batch`.
   - All RNG-dependent values: local weather deviation, diurnal factors,
     measurement noise.

"""

from .generation import generate_sites, generate_years
from .generator import SyntheticTemperatureGenerator
from .harmonic import harmonic_components, harmonic_model
from .params import (
    CALIBRATION_PARAMS_ENV_VAR,
    DiurnalAmplitudeParams,
    DiurnalParams,
    DiurnalTimingParams,
    HarmonicParams,
    OutputParams,
    SiteParams,
    SiteVariationParams,
    TemperatureParams,
    WaveformShapeParams,
    WeatherParams,
    YearParams,
    YearVariationParams,
    load_temperature_params,
    temperature_params_from_mapping,
)

__all__ = [
    "CALIBRATION_PARAMS_ENV_VAR",
    "DiurnalAmplitudeParams",
    "DiurnalParams",
    "DiurnalTimingParams",
    "HarmonicParams",
    "OutputParams",
    "SiteParams",
    "SiteVariationParams",
    "SyntheticTemperatureGenerator",
    "TemperatureParams",
    "WaveformShapeParams",
    "WeatherParams",
    "YearParams",
    "YearVariationParams",
    "generate_sites",
    "generate_years",
    "harmonic_components",
    "harmonic_model",
    "load_temperature_params",
    "temperature_params_from_mapping",
]
