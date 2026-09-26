"""Variance parameter extraction from Swiss weather data."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import zscore

from npnf.calibration.temperature.data import FixedVarianceParams
from npnf.calibration.temperature.helpers import fit_harmonic

if TYPE_CHECKING:
    from npnf.calibration.temperature.data import CalibrationData, ObservedData


@dataclasses.dataclass(frozen=True)
class YearDetrendingResult:
    """Intermediate values from year-mean detrending."""

    year_means: np.ndarray
    trend_line: np.ndarray
    trend_coeffs: np.ndarray
    detrended: np.ndarray
    detrended_std: float
    raw_std: float


def detrend_year_means(observed: ObservedData) -> YearDetrendingResult:
    """Compute year means and remove a linear trend.

    Returns a result object with all intermediate values needed by both
    parameter extraction and visualization.
    """
    years = observed.unique_years.astype(float)
    yr_means = year_means(observed.daily_means, observed.year_indices)
    trend_coeffs = np.polyfit(years, yr_means, 1)
    trend_line = np.polyval(trend_coeffs, years)
    detrended = yr_means - trend_line
    return YearDetrendingResult(
        year_means=yr_means,
        trend_line=trend_line,
        trend_coeffs=trend_coeffs,
        detrended=detrended,
        detrended_std=float(np.std(detrended)),
        raw_std=float(np.std(yr_means)),
    )


def _compute_subdaily_variance(
    observed: ObservedData, measurement_noise_std: float
) -> float:
    """Estimate subdaily weather variance via consecutive-day differencing."""
    hourly_dev = observed.temps - observed.temps.mean(axis=2, keepdims=True)
    for site in observed.unique_sites:
        idx = observed.site_indices[site]
        site_profile = hourly_dev[idx].mean(axis=0)
        hourly_dev[idx] -= site_profile[None, :, :]

    diff_within_day_variance = float(np.mean(np.diff(hourly_dev, axis=1).var(axis=2)))
    noise_variance = measurement_noise_std**2
    return max(0.0, diff_within_day_variance / 2.0 - noise_variance)


def _fit_site_seasonal(observed: ObservedData) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-site 2-harmonic seasonal models, return (phases, amplitudes)."""
    phases: list[float] = []
    amplitudes: list[float] = []
    for site in observed.unique_sites:
        idx = observed.site_indices[site]
        site_seasonal = observed.daily_means[idx].mean(axis=0)
        harmonic, _rmse = fit_harmonic(site_seasonal)
        phases.append(harmonic.params.phase)
        amplitudes.append(harmonic.params.amplitude)
    return np.array(phases), np.array(amplitudes)


def _weather_proxy_correlations(
    observed: ObservedData, anomalies: np.ndarray
) -> np.ndarray:
    """Per-site correlation with shared year-mean weather proxy."""
    shared_proxy = np.zeros_like(anomalies)
    for year in observed.unique_years:
        idx = observed.year_indices[int(year)]
        shared_proxy[idx] = anomalies[idx].mean(axis=0)

    correlations = []
    for site in observed.unique_sites:
        idx = observed.site_indices[site]
        correlation = np.corrcoef(
            anomalies[idx].flatten(), shared_proxy[idx].flatten()
        )[0, 1]
        correlations.append(correlation)
    return np.array(correlations)


def _site_diurnal_amplitude(
    observed: ObservedData, season_mask: np.ndarray
) -> np.ndarray:
    """Per-site pure diurnal amplitude for days matching season_mask.

    Uses the range of the mean hourly profile (averaged over days and years)
    rather than the mean of per-day ranges. Averaging first cancels out
    weather AR(1) and noise, isolating the diurnal component.
    """
    amplitudes: list[float] = []
    for site in observed.unique_sites:
        idx = observed.site_indices[site]
        temps = observed.temps[idx][:, season_mask, :]  # (n_years, masked_days, 24)
        mean_profile = temps.mean(axis=(0, 1))  # (24,)
        amplitudes.append(float(mean_profile.max() - mean_profile.min()))
    return np.array(amplitudes)


def site_means(values: np.ndarray, indices: dict) -> np.ndarray:
    """Per-site mean of a per-sample quantity."""
    return np.array([values[idx].mean() for idx in indices.values()])


def year_means(values: np.ndarray, indices: dict) -> np.ndarray:
    """Per-year mean of a per-sample quantity."""
    return np.array([values[idx].mean() for idx in indices.values()])


def _fixed_temperature_params(observed: ObservedData) -> dict[str, float]:
    """Site and year mean temperature spread."""
    result = detrend_year_means(observed)
    return {
        "site_offset_std": float(
            np.std(site_means(observed.daily_means, observed.site_indices))
        ),
        "year_anomaly_std": result.detrended_std,
    }


def _fixed_timing_params(observed: ObservedData) -> dict[str, float]:
    """Peak and trough hour spread across sites and years."""
    sunrise = observed.sunrise_trough_hours
    site_trough_means = np.array(
        [
            np.nanmean(sunrise[observed.site_indices[site]])
            for site in observed.unique_sites
        ]
    )
    year_trough_means = np.array(
        [
            np.nanmean(sunrise[observed.year_indices[year]])
            for year in observed.unique_years
        ]
    )
    return {
        "site_peak_hour_std": float(
            np.std(site_means(observed.peak_hours, observed.site_indices))
        ),
        "peak_hour_shift_std": float(
            np.std(
                year_means(observed.peak_hours, observed.year_indices)
                - observed.peak_hours.mean()
            )
        ),
        "site_trough_hour_std": float(np.nanstd(site_trough_means)),
        "trough_hour_shift_std": float(
            np.nanstd(year_trough_means - np.nanmean(sunrise))
        ),
    }


def _fixed_seasonal_params(data: CalibrationData) -> dict[str, float]:
    """Site-level seasonal phase and amplitude variation."""
    assert data.seasonal is not None
    site_phases, site_amplitudes = _fit_site_seasonal(data.observed)
    amplitude_factors = site_amplitudes / data.seasonal.harmonic.params.amplitude
    return {
        "site_phase_shift_std": float(
            np.std(site_phases - data.seasonal.harmonic.params.phase)
        ),
        "site_seasonal_amp_std": float(np.std(amplitude_factors)),
    }


def _fixed_diurnal_params(data: CalibrationData) -> dict[str, float]:
    """Per-site diurnal amplitude std in winter and summer (loc is optimized)."""
    assert data.diurnal is not None
    seasonal_factor = data.diurnal.seasonal_harmonic.normalized
    winter_amps = _site_diurnal_amplitude(data.observed, seasonal_factor < 0.1)
    summer_amps = _site_diurnal_amplitude(data.observed, seasonal_factor > 0.9)
    return {
        "site_diurnal_winter_std": float(np.std(winter_amps)),
        "site_diurnal_summer_std": float(np.std(summer_amps)),
    }


def _fixed_weather_params(
    data: CalibrationData, anomalies: np.ndarray
) -> dict[str, float]:
    """Weather correlation range across sites."""
    correlations = _weather_proxy_correlations(data.observed, anomalies)
    return {
        "weather_correlation_min": float(correlations.min()),
        "weather_correlation_max": float(correlations.max()),
    }


def _fixed_daily_diurnal_noise_clips(
    data: CalibrationData, anomalies: np.ndarray
) -> dict[str, float]:
    """Daily diurnal noise clip bounds from noise-only residuals."""
    assert data.diurnal is not None
    num_sites = len(data.observed.unique_sites)
    expected_amplitude = data.diurnal.seasonal_harmonic.values
    diurnal_ratio = data.observed.diurnal_amplitude / expected_amplitude[None, :]

    weather_z = zscore(anomalies, axis=1)

    site_coupling = dict(
        zip(
            data.observed.unique_sites,
            np.linspace(
                data.diurnal.site_weather_diurnal_coupling_min,
                data.diurnal.site_weather_diurnal_coupling_max,
                num_sites,
            ),
            strict=False,
        )
    )
    sample_couplings = np.array([site_coupling[site] for site in data.observed.sites])
    coupling_effect = 1.0 + sample_couplings[:, None] * weather_z
    adjusted_ratio = diurnal_ratio / coupling_effect
    amplitude_mask = data.observed.diurnal_amplitude >= 3.0
    filtered_mean = float(adjusted_ratio[amplitude_mask].mean())
    daily_noise = adjusted_ratio / filtered_mean

    masked_noise = daily_noise[amplitude_mask]
    return {
        "daily_diurnal_noise_min": float(np.percentile(masked_noise, 5)),
        "daily_diurnal_noise_max": float(np.percentile(masked_noise, 95)),
    }


def _fixed_siteyear_factor_bounds(data: CalibrationData) -> dict[str, float]:
    """Min/max of site-year diurnal factors from two-way residuals."""
    assert data.diurnal is not None
    expected_amplitude = data.diurnal.seasonal_harmonic.values
    sample_ratios = (
        data.observed.diurnal_amplitude / expected_amplitude[None, :]
    ).mean(axis=1)
    grand_mean = sample_ratios.mean()
    site_mean = dict(
        zip(
            data.observed.unique_sites,
            site_means(sample_ratios, data.observed.site_indices),
            strict=False,
        )
    )
    year_mean = {
        int(year): mean
        for year, mean in zip(
            data.observed.unique_years,
            year_means(sample_ratios, data.observed.year_indices),
            strict=False,
        )
    }
    residuals = (
        sample_ratios
        - np.array([site_mean[site] for site in data.observed.sites])
        - np.array([year_mean[int(year)] for year in data.observed.years])
        + grand_mean
    )
    factors = 1.0 + residuals
    return {
        "siteyear_factor_min": float(factors.min()),
        "siteyear_factor_max": float(factors.max()),
    }


def _extract_fixed_params(
    data: CalibrationData, anomalies: np.ndarray
) -> FixedVarianceParams:
    """Extract 18 fixed point-value parameters from Swiss data."""
    return FixedVarianceParams(
        **_fixed_temperature_params(data.observed),
        **_fixed_timing_params(data.observed),
        **_fixed_seasonal_params(data),
        **_fixed_diurnal_params(data),
        **_fixed_weather_params(data, anomalies),
        **_fixed_daily_diurnal_noise_clips(data, anomalies),
        **_fixed_siteyear_factor_bounds(data),
    )


def _bound_seasonal_params(data: CalibrationData) -> dict[str, tuple[float, float]]:
    """Year-to-year seasonal amplitude variation."""
    assert data.seasonal is not None
    year_amplitudes: list[float] = []
    for year in data.observed.unique_years:
        idx = data.observed.year_indices[int(year)]
        seasonal = data.observed.daily_means[idx].mean(axis=0)
        harmonic, _rmse = fit_harmonic(seasonal)
        year_amplitudes.append(harmonic.params.amplitude)
    amplitude_factors = (
        np.array(year_amplitudes) / data.seasonal.harmonic.params.amplitude
    )
    return {"year_seasonal_amplitude_std": (0.0, float(np.std(amplitude_factors)))}


def _max_per_year_std(
    deviations: np.ndarray, mask: np.ndarray, observed: ObservedData
) -> float:
    """Max of per-year stds of masked deviations."""
    per_year_stds = []
    for year in observed.unique_years:
        idx = observed.year_indices[int(year)]
        year_devs = deviations[idx][mask[idx]]
        if len(year_devs) > 0:
            per_year_stds.append(float(np.std(year_devs)))
    return float(np.max(per_year_stds))


def _bound_timing_params(data: CalibrationData) -> dict[str, tuple[float, float]]:
    """Daily timing noise upper bounds as max per-year std."""
    assert data.diurnal is not None
    peak = data.diurnal.peak_timing
    trough = data.diurnal.trough_timing
    assert peak is not None
    assert trough is not None
    return {
        "daily_peak_hour_noise_std": (
            0.0,
            _max_per_year_std(peak.deviations, peak.deviation_mask, data.observed),
        ),
        "daily_trough_hour_noise_std": (
            0.0,
            _max_per_year_std(trough.deviations, trough.deviation_mask, data.observed),
        ),
    }


def _bound_weather_params(
    data: CalibrationData, anomalies: np.ndarray, subdaily_variance: float
) -> dict[str, tuple[float, float]]:
    """Per-site weather marginal std range."""
    per_site_stds = [
        float(
            np.sqrt(
                np.std(anomalies[data.observed.site_indices[site]]) ** 2
                + subdaily_variance
            )
        )
        for site in data.observed.unique_sites
    ]
    return {
        "weather_marginal_std": (
            float(np.min(per_site_stds)),
            float(np.max(per_site_stds)),
        )
    }


def _bound_year_diurnal_variation(
    data: CalibrationData,
) -> dict[str, tuple[float, float]]:
    """Year-to-year diurnal amplitude variation."""
    yr_means = year_means(data.observed.diurnal_amplitude, data.observed.year_indices)
    year_factors = yr_means / data.observed.diurnal_amplitude.mean()
    return {"diurnal_amplitude_factor_std": (0.0, float(np.std(year_factors)))}


def _bound_siteyear_diurnal_residuals(
    data: CalibrationData,
) -> dict[str, tuple[float, float]]:
    """Site-year diurnal factor residuals via two-way decomposition."""
    assert data.diurnal is not None
    expected_amplitude = data.diurnal.seasonal_harmonic.values
    sample_ratios = (
        data.observed.diurnal_amplitude / expected_amplitude[None, :]
    ).mean(axis=1)
    grand_mean = sample_ratios.mean()
    site_mean = dict(
        zip(
            data.observed.unique_sites,
            site_means(sample_ratios, data.observed.site_indices),
            strict=False,
        )
    )
    year_mean = {
        int(year): mean
        for year, mean in zip(
            data.observed.unique_years,
            year_means(sample_ratios, data.observed.year_indices),
            strict=False,
        )
    }
    residuals = (
        sample_ratios
        - np.array([site_mean[site] for site in data.observed.sites])
        - np.array([year_mean[int(year)] for year in data.observed.years])
        + grand_mean
    )
    per_year_stds = [
        float(np.std(residuals[data.observed.year_indices[int(year)]]))
        for year in data.observed.unique_years
    ]
    return {"siteyear_diurnal_factor_std": (0.0, float(np.max(per_year_stds)))}


def _bound_daily_diurnal_noise(
    data: CalibrationData, anomalies: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Daily diurnal noise after removing weather-coupling effect."""
    assert data.diurnal is not None
    num_sites = len(data.observed.unique_sites)
    expected_amplitude = data.diurnal.seasonal_harmonic.values
    diurnal_ratio = data.observed.diurnal_amplitude / expected_amplitude[None, :]

    weather_z = zscore(anomalies, axis=1)

    site_coupling = dict(
        zip(
            data.observed.unique_sites,
            np.linspace(
                data.diurnal.site_weather_diurnal_coupling_min,
                data.diurnal.site_weather_diurnal_coupling_max,
                num_sites,
            ),
            strict=False,
        )
    )
    sample_couplings = np.array([site_coupling[site] for site in data.observed.sites])
    coupling_effect = 1.0 + sample_couplings[:, None] * weather_z
    adjusted_ratio = diurnal_ratio / coupling_effect
    amplitude_mask = data.observed.diurnal_amplitude >= 3.0
    filtered_mean = float(adjusted_ratio[amplitude_mask].mean())
    daily_noise = adjusted_ratio / filtered_mean  # (N, 274)

    per_year_stds = []
    for year in data.observed.unique_years:
        idx = data.observed.year_indices[int(year)]
        year_values = daily_noise[idx][amplitude_mask[idx]]
        if len(year_values) > 0:
            per_year_stds.append(float(np.std(year_values)))
    return {"daily_diurnal_noise_std": (0.0, float(np.max(per_year_stds)))}


def _fit_shape_exponent(
    profile: np.ndarray, trough_idx: int, peak_idx: int, half: str
) -> float:
    """Fit a power-cosine exponent to one half of a diurnal profile.

    Args:
        profile: (24,) mean diurnal deviation profile.
        trough_idx: Hour index of the trough.
        peak_idx: Hour index of the peak.
        half: "rise" (trough→peak) or "fall" (peak→trough).

    Returns:
        Fitted exponent alpha minimizing RMSE of the power-cosine model.
    """
    from scipy.optimize import minimize_scalar

    if half == "rise":
        duration = (peak_idx - trough_idx) % 24
        hours = np.arange(duration + 1)
        segment = np.array(
            [profile[(trough_idx + h) % 24] for h in range(duration + 1)]
        )
    else:
        duration = (trough_idx - peak_idx) % 24
        hours = np.arange(duration + 1)
        segment = np.array([profile[(peak_idx + h) % 24] for h in range(duration + 1)])

    if duration < 4:
        return 1.0

    phase = hours / duration  # [0, 1]
    seg_min, seg_max = segment.min(), segment.max()
    seg_range = seg_max - seg_min
    if seg_range < 1e-6:
        return 1.0

    # Normalize to [-1, 1]
    normalized = 2.0 * (segment - seg_min) / seg_range - 1.0

    def rmse(alpha: float) -> float:
        if half == "rise":
            model = -np.cos(np.pi * phase**alpha)
        else:
            model = np.cos(np.pi * phase**alpha)
        return float(np.sqrt(np.mean((normalized - model) ** 2)))

    result = minimize_scalar(rmse, bounds=(0.5, 2.0), method="bounded")
    return float(result.x)


def _bound_diurnal_loc(data: CalibrationData) -> dict[str, tuple[float, float]]:
    """Diurnal amplitude loc bounds for winter and summer.

    Upper bound: pure diurnal amplitude from mean profile extraction.
    Lower bound: 50% of upper, leaving room for weather AR(1) contribution.
    """
    assert data.diurnal is not None
    seasonal_factor = data.diurnal.seasonal_harmonic.normalized
    winter_amps = _site_diurnal_amplitude(data.observed, seasonal_factor < 0.1)
    summer_amps = _site_diurnal_amplitude(data.observed, seasonal_factor > 0.9)
    winter_loc = float(np.mean(winter_amps))
    summer_loc = float(np.mean(summer_amps))
    return {
        "site_diurnal_winter_loc": (winter_loc * 0.5, winter_loc),
        "site_diurnal_summer_loc": (summer_loc * 0.5, summer_loc),
    }


def _extract_bound_params(
    data: CalibrationData, anomalies: np.ndarray, subdaily_variance: float
) -> dict[str, tuple[float, float]]:
    """Extract 9 bound parameters with data-derived (lower, upper) limits."""
    return {
        **_bound_seasonal_params(data),
        **_bound_timing_params(data),
        **_bound_weather_params(data, anomalies, subdaily_variance),
        **_bound_year_diurnal_variation(data),
        **_bound_diurnal_loc(data),
        **_bound_siteyear_diurnal_residuals(data),
        **_bound_daily_diurnal_noise(data, anomalies),
    }


def extract_variance_params(
    data: CalibrationData,
) -> tuple[FixedVarianceParams, dict[str, tuple[float, float]]]:
    """Extract variance parameters from Swiss data.

    Returns:
        (fixed, bounds) where fixed is the point-value parameters,
        and bounds maps param name to (lower, upper) tuple.
    """
    assert data.seasonal is not None
    anomalies = data.observed.daily_means - data.seasonal.harmonic.values[None, :]
    subdaily_variance = _compute_subdaily_variance(
        data.observed, data.constants.measurement_noise_std
    )

    fixed = _extract_fixed_params(data, anomalies)
    bounds = _extract_bound_params(data, anomalies, subdaily_variance)
    return fixed, bounds
