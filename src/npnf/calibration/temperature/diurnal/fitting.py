"""Diurnal pattern fitting: timing and shape parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from npnf.calibration.temperature.data import TimingResult
from npnf.calibration.temperature.variance.extraction import _fit_shape_exponent
from npnf.data.synthetic.temperature import WaveformShapeParams

if TYPE_CHECKING:
    from npnf.calibration.temperature.data import HarmonicFit, ObservedData


def fit_timing_params(
    data: ObservedData, diurnal_harmonic: HarmonicFit
) -> tuple[TimingResult, TimingResult]:
    """Fit peak and trough hour timing from observed data.

    Computes the median observed peak/trough hour from the mean deviation
    profile per day (averaged across all site-years). This cancels weather
    AR(1) and noise, giving timing consistent with the pure diurnal signal
    used in generation.

    Returns (peak, trough) timing results.
    """
    # Mean deviation profile per day: cancels weather and noise
    mean_dev = (data.temps - data.temps.mean(axis=2, keepdims=True)).mean(
        axis=0
    )  # (274, 24)

    # Peak: argmax of mean deviation profile, filtered to daytime [10, 18]
    peak_raw = mean_dev.argmax(axis=1).astype(np.float64)
    mean_peak = np.where((peak_raw >= 10) & (peak_raw <= 18), peak_raw, np.nan)
    median_peak = float(np.nanmedian(mean_peak))

    peak_deviation_mask = ~np.isnan(data.daytime_peak_hours)
    peak_deviations = data.peak_hours - median_peak
    masked_peak_dev = peak_deviations[peak_deviation_mask]

    peak = TimingResult(
        base_lag=median_peak,
        seasonal_amp=0.0,
        noise_std=float(np.std(masked_peak_dev)),
        noise_min=float(np.percentile(masked_peak_dev, 5)),
        noise_max=float(np.percentile(masked_peak_dev, 95)),
        deviations=peak_deviations,
        deviation_mask=peak_deviation_mask,
    )

    # Trough: argmin of mean deviation profile, filtered to ≤12h
    trough_raw = mean_dev.argmin(axis=1).astype(np.float64)
    mean_trough = np.where(trough_raw <= 12, trough_raw, np.nan)
    median_trough = float(np.nanmedian(mean_trough))

    trough_deviation_mask = ~np.isnan(data.sunrise_trough_hours)
    trough_deviations = data.trough_hours - median_trough
    masked_trough_dev = trough_deviations[trough_deviation_mask]

    trough = TimingResult(
        base_lag=median_trough,
        seasonal_amp=0.0,
        noise_std=float(np.std(masked_trough_dev)),
        noise_min=float(np.percentile(masked_trough_dev, 5)),
        noise_max=float(np.percentile(masked_trough_dev, 95)),
        deviations=trough_deviations,
        deviation_mask=trough_deviation_mask,
    )

    return peak, trough


def _mean_deviation_profile(data: ObservedData, season_mask: np.ndarray) -> np.ndarray:
    """Mean diurnal deviation profile for days matching season_mask.

    Averages over all sites, years, and masked days to cancel weather
    noise, returning a clean (24,) profile.
    """
    temps = data.temps[:, season_mask, :]  # (N, masked_days, 24)
    profile = temps.mean(axis=(0, 1))  # (24,)
    return profile - profile.mean()


def fit_shape_params(
    data: ObservedData, diurnal_harmonic: HarmonicFit
) -> WaveformShapeParams:
    """Fit winter and summer waveform shape exponents from observed data.

    Computes mean diurnal profiles for winter (seasonal_factor < 0.1) and
    summer (seasonal_factor > 0.9) windows, then fits power-cosine
    exponents to each.
    """
    seasonal_factor = diurnal_harmonic.normalized
    winter_mask = seasonal_factor < 0.1
    summer_mask = seasonal_factor > 0.9

    winter_profile = _mean_deviation_profile(data, winter_mask)
    summer_profile = _mean_deviation_profile(data, summer_mask)

    winter_trough = int(np.argmin(winter_profile))
    winter_peak = int(np.argmax(winter_profile))
    summer_trough = int(np.argmin(summer_profile))
    summer_peak = int(np.argmax(summer_profile))

    return WaveformShapeParams(
        rise_exponent_winter=_fit_shape_exponent(
            winter_profile, winter_trough, winter_peak, "rise"
        ),
        rise_exponent_summer=_fit_shape_exponent(
            summer_profile, summer_trough, summer_peak, "rise"
        ),
        fall_exponent_winter=_fit_shape_exponent(
            winter_profile, winter_trough, winter_peak, "fall"
        ),
        fall_exponent_summer=_fit_shape_exponent(
            summer_profile, summer_trough, summer_peak, "fall"
        ),
    )
