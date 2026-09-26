"""Weather-diurnal coupling extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import zscore

from npnf.calibration.temperature.data import HarmonicFit, SiteRange

if TYPE_CHECKING:
    from npnf.calibration.temperature.data import ObservedData


def _compute_weather_z(data: ObservedData, seasonal: HarmonicFit) -> np.ndarray:
    """Z-score weather anomalies per site-year.

    Returns (N, 274) array of standardized anomalies.
    """
    weather_anomaly = data.daily_means - seasonal.values[None, :]
    return zscore(weather_anomaly, axis=1)


def fit_weather_diurnal_coupling(
    data: ObservedData, seasonal: HarmonicFit, diurnal: HarmonicFit
) -> SiteRange:
    """Fit per-site linear weather->diurnal amplitude coupling."""
    weather_z = _compute_weather_z(data, seasonal)
    diurnal_ratio = data.diurnal_amplitude / diurnal.values[None, :]

    slopes = []
    for idx in data.site_indices.values():
        site_z = weather_z[idx].flatten()
        site_ratio = diurnal_ratio[idx].flatten()
        slope = np.polyfit(site_z, site_ratio, 1)[0]
        slopes.append(slope)

    return SiteRange(min=float(min(slopes)), max=float(max(slopes)))


def fit_peak_hour_weather_coupling(
    data: ObservedData, seasonal: HarmonicFit
) -> SiteRange:
    """Fit per-site amplitude-weighted weather->peak hour coupling."""
    weather_z = _compute_weather_z(data, seasonal)

    slopes = []
    for idx in data.site_indices.values():
        site_z = weather_z[idx].flatten()
        site_peak = data.peak_hours[idx].flatten()
        w = data.diurnal_amplitude[idx].flatten()
        slope = np.polyfit(site_z, site_peak, 1, w=np.sqrt(w))[0]
        slopes.append(slope)

    return SiteRange(min=float(min(slopes)), max=float(max(slopes)))


def fit_trough_hour_weather_coupling(
    data: ObservedData, seasonal: HarmonicFit
) -> SiteRange:
    """Fit per-site amplitude-weighted weather->trough hour coupling."""
    weather_z = _compute_weather_z(data, seasonal)

    slopes = []
    for idx in data.site_indices.values():
        site_z = weather_z[idx].flatten()
        site_trough = data.trough_hours[idx].flatten()
        w = data.diurnal_amplitude[idx].flatten()
        slope = np.polyfit(site_z, site_trough, 1, w=np.sqrt(w))[0]
        slopes.append(slope)

    return SiteRange(min=float(min(slopes)), max=float(max(slopes)))
