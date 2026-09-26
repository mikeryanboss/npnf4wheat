"""Parameter dataclasses and I/O for temperature generation.

This module contains:
- Parameter dataclasses: `SiteParams`, `YearParams`, `TemperatureParams`
  and sub-struct dataclasses composing `TemperatureParams`
- Parameter loading: `temperature_params_from_mapping()`, `load_temperature_params()`
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CALIBRATION_PARAMS_ENV_VAR = "NPNF_TEMPERATURE_PARAMS_PATH"


@dataclass(frozen=True)
class HarmonicParams:
    """Five scalar parameters for a 2-harmonic seasonal model.

    Used as the generation-side type for both the daily-mean seasonal
    cycle and the diurnal amplitude seasonal cycle.
    """

    base: float
    amplitude: float
    phase: float
    amplitude_2: float
    phase_2: float


@dataclass
class SiteParams:
    """Fixed characteristics of a synthetic site."""

    name: str
    temp_offset: float  # °C, fixed site effect
    peak_hour_offset: float  # Hours, fixed site peak hour offset
    trough_hour_offset: float  # Hours, fixed site trough hour offset
    seasonal_amplitude_factor: float
    seasonal_phase_shift: float  # days
    diurnal_amplitude_winter: float  # °C
    diurnal_amplitude_summer: float  # °C
    weather_correlation: float  # How much site follows shared weather
    weather_diurnal_coupling: float  # Per-site weather→diurnal coupling β
    peak_hour_weather_coupling: float  # Per-site weather→peak hour coupling β
    trough_hour_weather_coupling: float  # Per-site weather→trough hour coupling β


@dataclass
class YearParams:
    """Characteristics of a synthetic year."""

    year_id: int
    temp_anomaly: float  # °C, shared year effect
    seasonal_amplitude_anomaly: float
    diurnal_amplitude_factor: float  # Year effect on diurnal cycle
    peak_hour_shift: float  # Year effect on diurnal timing (hours from peak)
    trough_hour_shift: float  # Hours, year effect on trough timing
    weather_series: np.ndarray = field(default_factory=lambda: np.array([]))


# === Sub-struct dataclasses for TemperatureParams ===


@dataclass(frozen=True)
class WeatherParams:
    """AR(1) weather anomaly process."""

    ar_coefficient: float
    marginal_std: float
    anomaly_soft_cap: float


@dataclass(frozen=True)
class SiteVariationParams:
    """Per-site sampling distributions."""

    offset_std: float
    peak_hour_std: float
    trough_hour_std: float
    seasonal_amp_std: float
    phase_shift_std: float
    diurnal_winter_loc: float
    diurnal_winter_std: float
    diurnal_summer_loc: float
    diurnal_summer_std: float
    weather_correlation_min: float
    weather_correlation_max: float
    weather_diurnal_coupling_min: float
    weather_diurnal_coupling_max: float
    peak_hour_weather_coupling_min: float
    peak_hour_weather_coupling_max: float
    trough_hour_weather_coupling_min: float
    trough_hour_weather_coupling_max: float


@dataclass(frozen=True)
class YearVariationParams:
    """Per-year sampling distributions."""

    anomaly_std: float
    seasonal_amp_std: float
    diurnal_amplitude_factor_std: float
    peak_hour_shift_std: float
    trough_hour_shift_std: float


@dataclass(frozen=True)
class DiurnalAmplitudeParams:
    """Diurnal amplitude modulation."""

    siteyear_factor_std: float
    siteyear_factor_min: float
    siteyear_factor_max: float
    daily_noise_std: float
    daily_noise_min: float
    daily_noise_max: float


@dataclass(frozen=True)
class DiurnalTimingParams:
    """Fixed peak/trough hours, noise clips, and calibrated noise stds."""

    peak_hour: float
    daily_peak_hour_noise_std: float
    daily_peak_hour_noise_min: float
    daily_peak_hour_noise_max: float
    trough_hour: float
    daily_trough_hour_noise_std: float
    daily_trough_hour_noise_min: float
    daily_trough_hour_noise_max: float


@dataclass(frozen=True)
class WaveformShapeParams:
    """Seasonally varying waveform shape exponents (derived analytically)."""

    rise_exponent_winter: float
    rise_exponent_summer: float
    fall_exponent_winter: float
    fall_exponent_summer: float


@dataclass(frozen=True)
class DiurnalParams:
    """Container for all diurnal cycle parameters."""

    seasonal: HarmonicParams
    amplitude: DiurnalAmplitudeParams
    timing: DiurnalTimingParams
    shape: WaveformShapeParams


@dataclass(frozen=True)
class OutputParams:
    """Measurement noise parameters."""

    measurement_noise_std: float


@dataclass(frozen=True)
class TemperatureParams:
    """Top-level parameters for synthetic temperature generation.

    Composed of 6 frozen sub-structs grouped by physical process.
    """

    seasonal: HarmonicParams
    weather: WeatherParams
    site: SiteVariationParams
    year: YearVariationParams
    diurnal: DiurnalParams
    output: OutputParams


def temperature_params_from_mapping(values: Mapping[str, Any]) -> TemperatureParams:
    """Build TemperatureParams from a nested mapping with strict key checking."""
    return TemperatureParams(
        seasonal=HarmonicParams(**values["seasonal"]),
        weather=WeatherParams(**values["weather"]),
        site=SiteVariationParams(**values["site"]),
        year=YearVariationParams(**values["year"]),
        diurnal=DiurnalParams(
            seasonal=HarmonicParams(**values["diurnal"]["seasonal"]),
            amplitude=DiurnalAmplitudeParams(**values["diurnal"]["amplitude"]),
            timing=DiurnalTimingParams(**values["diurnal"]["timing"]),
            shape=WaveformShapeParams(**values["diurnal"]["shape"]),
        ),
        output=OutputParams(**values["output"]),
    )


def load_temperature_params(path: str | Path) -> TemperatureParams:
    """Load TemperatureParams from a calibration params JSON file."""
    with Path(path).open() as f:
        data = json.load(f)
    payload = data.get("parameters", data)
    return temperature_params_from_mapping(payload)
