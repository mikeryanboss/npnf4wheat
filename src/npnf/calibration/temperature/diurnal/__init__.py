"""Diurnal pattern fitting: timing, shape, coupling."""

from .coupling import (
    fit_peak_hour_weather_coupling,
    fit_trough_hour_weather_coupling,
    fit_weather_diurnal_coupling,
)
from .fitting import TimingResult, fit_timing_params
from .visualize import plot_extracted_params

__all__ = [
    "TimingResult",
    "fit_peak_hour_weather_coupling",
    "fit_timing_params",
    "fit_trough_hour_weather_coupling",
    "fit_weather_diurnal_coupling",
    "plot_extracted_params",
]
