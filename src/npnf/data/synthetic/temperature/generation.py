"""Generation logic and factory functions for synthetic temperature data.

This module contains:
- Vectorized year-batch generation: `_generate_year_batch()`
- Factory functions: `generate_sites()`, `generate_years()`
"""

import numpy as np
from scipy.optimize import brentq
from scipy.signal import lfilter

from .harmonic import harmonic_model
from .params import (
    DiurnalAmplitudeParams,
    DiurnalTimingParams,
    HarmonicParams,
    SiteParams,
    SiteVariationParams,
    TemperatureParams,
    WaveformShapeParams,
    WeatherParams,
    YearParams,
    YearVariationParams,
)

# 64-node Gauss-Legendre quadrature on [0, 1] for vectorized cosine integrals
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)
_GL_T = (_GL_NODES + 1.0) / 2.0
_GL_W = _GL_WEIGHTS / 2.0


def _sample_truncnorm(rng, loc, scale, size, low=None, high=None, n_sigma=3.0):
    """Sample from a truncated normal via rejection sampling.

    When low/high are not provided, bounds default to loc ± n_sigma * scale.
    Nearly zero overhead when bounds are at +/-3 sigma or wider.
    """
    if low is None:
        low = loc - n_sigma * scale
    if high is None:
        high = loc + n_sigma * scale
    samples = np.atleast_1d(rng.normal(loc, scale, size))
    mask = (samples < low) | (samples > high)
    while np.any(mask):
        samples[mask] = rng.normal(loc, scale, int(np.sum(mask)))
        mask = (samples < low) | (samples > high)
    return samples


def daily_average_acf1(rho_hour: float) -> float:
    lags = np.arange(24)
    auto_sum = np.sum(rho_hour ** np.abs(lags[:, None] - lags[None, :]))
    cross_sum = np.sum(rho_hour ** np.abs(24 + lags[None, :] - lags[:, None]))
    return cross_sum / auto_sum


def _compute_rho_hour(ar_coefficient: float) -> float:
    """Derive hourly AR(1) coefficient from daily via ACF matching."""
    return brentq(lambda r: daily_average_acf1(r) - ar_coefficient, 0.9, 0.999)


def _prestack_site_params(sites: list[SiteParams]) -> dict[str, np.ndarray]:
    """Extract site-level scalar parameters into arrays for vectorized computation."""
    return {
        "corrs": np.array([s.weather_correlation for s in sites]),
        "temp_offsets": np.array([s.temp_offset for s in sites]),
        "peak_hour_offsets": np.array([s.peak_hour_offset for s in sites]),
        "trough_hour_offsets": np.array([s.trough_hour_offset for s in sites]),
        "seasonal_amp_factors": np.array([s.seasonal_amplitude_factor for s in sites]),
        "seasonal_phase_shifts": np.array([s.seasonal_phase_shift for s in sites]),
        "diurnal_winters": np.array([s.diurnal_amplitude_winter for s in sites]),
        "diurnal_summers": np.array([s.diurnal_amplitude_summer for s in sites]),
        "weather_diurnal_coupling": np.array(
            [s.weather_diurnal_coupling for s in sites]
        ),
        "peak_hour_weather_coupling": np.array(
            [s.peak_hour_weather_coupling for s in sites]
        ),
        "trough_hour_weather_coupling": np.array(
            [s.trough_hour_weather_coupling for s in sites]
        ),
    }


def _compute_seasonal(
    site_params: dict[str, np.ndarray],
    year: YearParams,
    seasonal: HarmonicParams,
    num_days: int,
) -> np.ndarray:
    """Annual warm/cold cycle per site (S, num_days)."""
    omega = 2 * np.pi / 365
    days = np.arange(num_days, dtype=np.float64)
    effective_phases = seasonal.phase + site_params["seasonal_phase_shifts"]
    cos1_bases = np.cos(omega * (days[None, :] - effective_phases[:, None]))
    effective_amps = (
        seasonal.amplitude
        * site_params["seasonal_amp_factors"]
        * year.seasonal_amplitude_anomaly
    )
    cos2_daily = seasonal.amplitude_2 * np.cos(2 * omega * (days - seasonal.phase_2))
    return effective_amps[:, None] * cos1_bases + cos2_daily[None, :]


def _compute_site_weather(
    year: YearParams,
    site_params: dict[str, np.ndarray],
    weather: WeatherParams,
    rng: np.random.Generator,
    num_sites: int,
    num_hours: int,
) -> np.ndarray:
    """Correlated hourly weather process per site (S, num_hours).

    Mixes the shared year weather with site-local AR(1) deviations, then
    applies a soft cap to limit extreme anomalies.
    """
    rho_hour = _compute_rho_hour(weather.ar_coefficient)
    innovation_std = weather.marginal_std * np.sqrt(1 - rho_hour**2)

    local_innovations = rng.normal(0, innovation_std, (num_sites, num_hours))
    local_deviations = lfilter([1], [1, -rho_hour], local_innovations, axis=1)
    local_deviations -= local_deviations.mean(axis=1, keepdims=True)

    corrs = site_params["corrs"]
    site_weather = (
        corrs[:, None] * year.weather_series[None, :]
        + np.sqrt(1 - corrs[:, None] ** 2) * local_deviations
    )
    return weather.anomaly_soft_cap * np.tanh(site_weather / weather.anomaly_soft_cap)


def _compute_diurnal_amplitude(
    weather_z: np.ndarray,
    site_params: dict[str, np.ndarray],
    year: YearParams,
    diurnal_amp: DiurnalAmplitudeParams,
    diurnal_seasonal_factor: np.ndarray,
    rng: np.random.Generator,
    num_sites: int,
    num_days: int,
) -> np.ndarray:
    """Per-hour diurnal amplitude (S, num_hours).

    Combines seasonal base amplitude (winter/summer interpolation), a
    site-year random scaling, and daily modulation from weather coupling
    and day-to-day noise.
    """
    # Site-year random scaling: one draw per site per year
    siteyear_factors = _sample_truncnorm(
        rng,
        1.0,
        diurnal_amp.siteyear_factor_std,
        num_sites,
        low=diurnal_amp.siteyear_factor_min,
        high=diurnal_amp.siteyear_factor_max,
    )

    # Daily modulation: warm days tend to have larger diurnal swings
    weather_coupling = (
        1.0 + site_params["weather_diurnal_coupling"][:, None] * weather_z
    )
    daily_noise = _sample_truncnorm(
        rng,
        1.0,
        diurnal_amp.daily_noise_std,
        (num_sites, num_days),
        low=diurnal_amp.daily_noise_min,
        high=diurnal_amp.daily_noise_max,
    )
    daily_mod = np.repeat(weather_coupling * daily_noise, 24, axis=1)  # (S, num_hours)

    # Seasonal base: interpolate winter↔summer amplitude by time of year
    diurnal_base = (
        site_params["diurnal_winters"][:, None]
        + (site_params["diurnal_summers"] - site_params["diurnal_winters"])[:, None]
        * diurnal_seasonal_factor[None, :]
    )  # (S, num_hours)

    return (
        diurnal_base
        * year.diurnal_amplitude_factor
        * siteyear_factors[:, None]
        * daily_mod
    )


def _waveform_mean_correction(
    shape: WaveformShapeParams,
    diurnal_seasonal_factor: np.ndarray,
    base_rise_dur: float,
) -> np.ndarray:
    """Smooth waveform mean as a function of seasonal factor and base timing.

    Returns the expected waveform mean per hour (same shape as
    diurnal_seasonal_factor), computed from the deterministic exponents
    and base rise/fall durations. Varies continuously over the year.
    """
    base_fall_dur = 24.0 - base_rise_dur

    unique_sf, inverse = np.unique(diurnal_seasonal_factor, return_inverse=True)
    re = (
        shape.rise_exponent_winter
        + (shape.rise_exponent_summer - shape.rise_exponent_winter) * unique_sf
    )
    fe = (
        shape.fall_exponent_winter
        + (shape.fall_exponent_summer - shape.fall_exponent_winter) * unique_sf
    )

    # Vectorized Gauss-Legendre quadrature: ∫₀¹ cos(π·tᵖ) dt for all exponents
    all_exp = np.concatenate([re, fe])
    t_p = _GL_T[:, None] ** all_exp[None, :]
    all_integrals = _GL_W @ np.cos(np.pi * t_p)
    n = len(unique_sf)
    means = (
        -base_rise_dur * all_integrals[:n] + base_fall_dur * all_integrals[n:]
    ) / 24.0

    return means[inverse]


def _compute_diurnal_waveform(
    weather_z: np.ndarray,
    site_params: dict[str, np.ndarray],
    year: YearParams,
    timing: DiurnalTimingParams,
    shape: WaveformShapeParams,
    diurnal_seasonal_factor: np.ndarray,
    rng: np.random.Generator,
    num_sites: int,
    num_days: int,
) -> np.ndarray:
    """Zero-mean piecewise half-cosine waveform (S, num_hours).

    Computes trough→peak→trough timing per site per day (fixed base hours
    with weather-driven peak-hour shift and daily noise), then evaluates
    the asymmetric cosine shape. A smooth analytical correction is
    subtracted so the waveform has approximately zero daily mean,
    ensuring the diurnal cycle does not shift daily mean temperatures.
    """
    hours_hourly = np.tile(np.arange(24, dtype=np.float64), num_days)

    # Daily timing noise + weather-driven peak-hour shift
    peak_noise = _sample_truncnorm(
        rng,
        0,
        timing.daily_peak_hour_noise_std,
        (num_sites, num_days),
        low=timing.daily_peak_hour_noise_min,
        high=timing.daily_peak_hour_noise_max,
    )
    trough_noise = _sample_truncnorm(
        rng,
        0,
        timing.daily_trough_hour_noise_std,
        (num_sites, num_days),
        low=timing.daily_trough_hour_noise_min,
        high=timing.daily_trough_hour_noise_max,
    )
    peak_shift = site_params["peak_hour_weather_coupling"][:, None] * weather_z
    trough_shift = site_params["trough_hour_weather_coupling"][:, None] * weather_z

    effective_peak_hour = (
        timing.peak_hour
        + year.peak_hour_shift
        + site_params["peak_hour_offsets"][:, None]
        + np.repeat(peak_noise + peak_shift, 24, axis=1)
    )  # (S, num_hours)
    effective_trough_hour = (
        timing.trough_hour
        + year.trough_hour_shift
        + site_params["trough_hour_offsets"][:, None]
        + np.repeat(trough_noise + trough_shift, 24, axis=1)
    )  # (S, num_hours)

    rise_dur = np.clip((effective_peak_hour - effective_trough_hour) % 24.0, 4.0, 20.0)
    fall_dur = 24.0 - rise_dur
    h_since_trough = (hours_hourly[None, :] - effective_trough_hour) % 24.0
    in_rise = h_since_trough < rise_dur

    rise_exp = (
        shape.rise_exponent_winter
        + (shape.rise_exponent_summer - shape.rise_exponent_winter)
        * diurnal_seasonal_factor
    )
    fall_exp = (
        shape.fall_exponent_winter
        + (shape.fall_exponent_summer - shape.fall_exponent_winter)
        * diurnal_seasonal_factor
    )

    rise_t = np.clip(h_since_trough / rise_dur, 0, 1)
    fall_t = np.clip((h_since_trough - rise_dur) / fall_dur, 0, 1)
    waveform = np.where(
        in_rise, -np.cos(np.pi * rise_t**rise_exp), np.cos(np.pi * fall_t**fall_exp)
    )

    # Subtract smooth analytical mean so diurnal cycle doesn't shift daily temps
    base_rise_dur = (timing.peak_hour - timing.trough_hour) % 24.0
    correction = _waveform_mean_correction(
        shape, diurnal_seasonal_factor, base_rise_dur
    )
    return waveform - correction[None, :]


def _compute_diurnal(
    site_weather: np.ndarray,
    site_params: dict[str, np.ndarray],
    year: YearParams,
    diurnal_amp: DiurnalAmplitudeParams,
    timing: DiurnalTimingParams,
    shape: WaveformShapeParams,
    diurnal_seasonal: HarmonicParams,
    rng: np.random.Generator,
    num_sites: int,
    num_days: int,
) -> np.ndarray:
    """Full diurnal (day/night) signal (S, num_hours): amplitude x waveform / 2.

    Normalised daily weather drives both the amplitude coupling and the
    peak-hour shift, so weather_z is computed here and shared by both.
    """
    # Compute diurnal seasonal factor (shared by amplitude and waveform)
    days_hourly = np.repeat(np.arange(num_days, dtype=np.float64), 24)
    raw = harmonic_model(
        days_hourly,
        diurnal_seasonal.base,
        diurnal_seasonal.amplitude,
        diurnal_seasonal.phase,
        diurnal_seasonal.amplitude_2,
        diurnal_seasonal.phase_2,
    )
    diurnal_seasonal_factor = (raw - raw.min()) / (raw.max() - raw.min())

    daily_weather = site_weather.reshape(num_sites, num_days, 24).mean(axis=2)
    weather_z = (daily_weather - daily_weather.mean(axis=1, keepdims=True)) / (
        daily_weather.std(axis=1, keepdims=True) + 1e-6
    )  # (S, num_days) — shared by amplitude coupling and peak-hour shift

    amplitude = _compute_diurnal_amplitude(
        weather_z,
        site_params,
        year,
        diurnal_amp,
        diurnal_seasonal_factor,
        rng,
        num_sites,
        num_days,
    )
    waveform = _compute_diurnal_waveform(
        weather_z,
        site_params,
        year,
        timing,
        shape,
        diurnal_seasonal_factor,
        rng,
        num_sites,
        num_days,
    )
    return amplitude * waveform / 2.0


def _generate_year_batch(
    num_sites: int,
    year: YearParams,
    temperature_parameters: TemperatureParams,
    num_days: int,
    site_parameters: dict[str, np.ndarray],
    seed: int,
    year_index: int,
) -> np.ndarray:
    """Generate hourly temperatures for all sites in a single year.

    Vectorized generation of hourly temperatures for all sites in a year.
    Uses a single RNG per year-batch with batched draws for performance.

    Returns:
        Temperature array of shape (S, num_days, 24).
    """
    num_hours = num_days * 24
    rng = np.random.default_rng(seed * 10000 + year_index)

    seasonal = _compute_seasonal(
        site_parameters, year, temperature_parameters.seasonal, num_days
    )
    site_weather = _compute_site_weather(
        year, site_parameters, temperature_parameters.weather, rng, num_sites, num_hours
    )
    diurnal = _compute_diurnal(
        site_weather,
        site_parameters,
        year,
        temperature_parameters.diurnal.amplitude,
        temperature_parameters.diurnal.timing,
        temperature_parameters.diurnal.shape,
        temperature_parameters.diurnal.seasonal,
        rng,
        num_sites,
        num_days,
    )

    temperature = (
        temperature_parameters.seasonal.base
        + site_parameters["temp_offsets"][:, None]
        + year.temp_anomaly
        + np.repeat(seasonal, 24, axis=1)
        + site_weather
        + diurnal
        + rng.normal(
            0,
            temperature_parameters.output.measurement_noise_std,
            (num_sites, num_hours),
        )
    )

    return temperature.reshape(num_sites, num_days, 24)


def generate_sites(
    num_sites: int, site: SiteVariationParams, rng: np.random.Generator
) -> list[SiteParams]:
    """Generate fixed site characteristics.

    Args:
        num_sites: Number of sites to generate.
        site: Parameters for site variation ranges.
        rng: Random number generator.

    Returns:
        List of SiteParams for each site.
    """
    sites = []
    for i in range(num_sites):
        site_params = SiteParams(
            name=f"Site_{i:02d}",
            temp_offset=_sample_truncnorm(rng, 0, site.offset_std, 1).item(),
            peak_hour_offset=_sample_truncnorm(rng, 0, site.peak_hour_std, 1).item(),
            trough_hour_offset=_sample_truncnorm(
                rng, 0, site.trough_hour_std, 1
            ).item(),
            seasonal_amplitude_factor=_sample_truncnorm(
                rng, 1.0, site.seasonal_amp_std, 1
            ).item(),
            seasonal_phase_shift=_sample_truncnorm(
                rng, 0, site.phase_shift_std, 1
            ).item(),
            diurnal_amplitude_winter=_sample_truncnorm(
                rng, site.diurnal_winter_loc, site.diurnal_winter_std, 1
            ).item(),
            diurnal_amplitude_summer=_sample_truncnorm(
                rng, site.diurnal_summer_loc, site.diurnal_summer_std, 1
            ).item(),
            weather_correlation=rng.uniform(
                site.weather_correlation_min, site.weather_correlation_max
            ),
            weather_diurnal_coupling=rng.uniform(
                site.weather_diurnal_coupling_min, site.weather_diurnal_coupling_max
            ),
            peak_hour_weather_coupling=rng.uniform(
                site.peak_hour_weather_coupling_min, site.peak_hour_weather_coupling_max
            ),
            trough_hour_weather_coupling=rng.uniform(
                site.trough_hour_weather_coupling_min,
                site.trough_hour_weather_coupling_max,
            ),
        )
        sites.append(site_params)
    return sites


def generate_years(
    num_years: int,
    num_days: int,
    start_year: int,
    weather: WeatherParams,
    year: YearVariationParams,
    rng: np.random.Generator,
) -> list[YearParams]:
    """Generate year characteristics with shared weather.

    Args:
        num_years: Number of years to generate.
        num_days: Days per year.
        start_year: First synthetic year ID.
        weather: Weather AR(1) parameters.
        year: Year variation parameters.
        rng: Random number generator.

    Returns:
        List of YearParams for each year.
    """
    rho_hour = _compute_rho_hour(weather.ar_coefficient)
    innovation_std = weather.marginal_std * np.sqrt(1 - rho_hour**2)
    num_hours = num_days * 24

    years = []
    for year_index in range(num_years):
        # Generate shared weather (continuous hourly AR(1) process)
        innovations = rng.normal(0, innovation_std, num_hours)
        weather_series = lfilter([1], [1, -rho_hour], innovations)
        weather_series -= weather_series.mean()

        year_params = YearParams(
            year_id=start_year + year_index,
            temp_anomaly=_sample_truncnorm(rng, 0, year.anomaly_std, 1).item(),
            seasonal_amplitude_anomaly=_sample_truncnorm(
                rng, 1.0, year.seasonal_amp_std, 1
            ).item(),
            diurnal_amplitude_factor=_sample_truncnorm(
                rng, 1.0, year.diurnal_amplitude_factor_std, 1
            ).item(),
            peak_hour_shift=_sample_truncnorm(
                rng, 0, year.peak_hour_shift_std, 1
            ).item(),
            trough_hour_shift=_sample_truncnorm(
                rng, 0, year.trough_hour_shift_std, 1
            ).item(),
            weather_series=weather_series,
        )
        years.append(year_params)
    return years
