"""Temperature calibration data utilities."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence

import numpy as np
from datasets import Dataset
from scipy.ndimage import median_filter

from npnf.data.datasets.fip1 import Fip1Facts
from npnf.data.process.temperature import load_weather_data
from npnf.data.synthetic.temperature import (
    DiurnalAmplitudeParams,
    DiurnalParams,
    DiurnalTimingParams,
    HarmonicParams,
    OutputParams,
    SiteVariationParams,
    TemperatureParams,
    WaveformShapeParams,
    WeatherParams,
    YearVariationParams,
    harmonic_model,
)


@dataclasses.dataclass(frozen=True)
class HarmonicFit:
    """Result of a 2-harmonic seasonal fit.

    Wraps ``HarmonicParams`` with precomputed arrays ``values``
    (the model evaluated over 274 days) and ``normalized``
    (min-max scaled to [0, 1]).  Used by calibration code that
    needs the evaluated curve; generation code only needs the
    scalar ``params``.
    """

    NUM_DAYS = 274
    DAYS = np.arange(NUM_DAYS, dtype=np.float64)

    params: HarmonicParams
    values: np.ndarray = dataclasses.field(init=False, repr=False)
    normalized: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        vals = harmonic_model(
            self.DAYS,
            self.params.base,
            self.params.amplitude,
            self.params.phase,
            self.params.amplitude_2,
            self.params.phase_2,
        )
        object.__setattr__(self, "values", vals)
        object.__setattr__(
            self, "normalized", (vals - vals.min()) / (vals.max() - vals.min())
        )


def load_swiss_data(
    include_years: Sequence[int] | None = None, sites: Sequence[str] | None = None
) -> Dataset:
    """Load Swiss weather data for calibration.

    Excludes the FIP1 held-out year (``Fip1Facts.held_out_year``).

    Args:
        include_years: If set, only include these years.
        sites: If set, only include these sites.
    """
    weather_data_path = os.environ.get("NPNF_WEATHER_DATA_PATH")
    if weather_data_path is None:
        msg = "NPNF_WEATHER_DATA_PATH is not set"
        raise ValueError(msg)
    return load_weather_data(
        weather_data_path,
        sites=sites,
        include_years=include_years,
        exclude_year=Fip1Facts().held_out_year,
    )


@dataclasses.dataclass(frozen=True)
class ObservedData:
    """Temperature data reshaped and reduced once for the whole pipeline.

    Created via ``from_dataset()``; every downstream function receives
    this instead of the raw HuggingFace ``Dataset``.
    """

    temps: np.ndarray  # (N, 274, 24)
    daily_means: np.ndarray  # (N, 274)
    daily_max: np.ndarray  # (N, 274)
    diurnal_amplitude: np.ndarray  # (N, 274)
    mean_seasonal: np.ndarray  # (274,)
    peak_hours: np.ndarray  # (N, 274) float64
    trough_hours: np.ndarray  # (N, 274) float64
    daytime_peak_hours: np.ndarray  # (N, 274) float64, non-daytime → NaN
    sunrise_trough_hours: np.ndarray  # (N, 274) float64, non-sunrise → NaN
    years: np.ndarray  # (N,)
    sites: list[str]
    unique_sites: list[str]
    site_indices: dict[str, np.ndarray]
    unique_years: np.ndarray
    year_indices: dict[int, np.ndarray]

    @classmethod
    def from_dataset(cls, data: Dataset) -> ObservedData:
        temperatures = np.asarray(data["temperature_values"], dtype=float).reshape(
            -1, 274 * 24
        )
        temperatures = median_filter(temperatures, size=(1, 3)).reshape(-1, 274, 24)

        daily_means = temperatures.mean(axis=2)
        daily_max = temperatures.max(axis=2)
        daily_min = temperatures.min(axis=2)
        diurnal_amplitude = daily_max - daily_min

        sites = list(data["site"])
        years = np.asarray(data["harvest_year"], dtype=int)

        sites_arr = np.array(sites)
        unique_sites = sorted(set(sites))
        site_indices = {site: np.where(sites_arr == site)[0] for site in unique_sites}

        unique_years = np.unique(years)
        year_indices = {int(year): np.where(years == year)[0] for year in unique_years}

        return cls(
            temps=temperatures,
            daily_means=daily_means,
            daily_max=daily_max,
            diurnal_amplitude=diurnal_amplitude,
            mean_seasonal=daily_means.mean(axis=0),
            peak_hours=(peak := temperatures.argmax(axis=2).astype(np.float64)),
            trough_hours=(trough := temperatures.argmin(axis=2).astype(np.float64)),
            daytime_peak_hours=np.where((peak >= 10.0) & (peak <= 18.0), peak, np.nan),
            sunrise_trough_hours=np.where(trough <= 12.0, trough, np.nan),
            years=years,
            sites=sites,
            unique_sites=unique_sites,
            site_indices=site_indices,
            unique_years=unique_years,
            year_indices=year_indices,
        )

    @property
    def num_samples(self) -> int:
        return self.temps.shape[0]

    @property
    def abs_weather_anomalies(self) -> np.ndarray:
        """Absolute daily mean anomalies (flattened)."""
        return np.abs(self.daily_means - self.mean_seasonal[None, :]).flatten()

    @property
    def weather_anomaly_soft_cap(self) -> float:
        """99.5th percentile of absolute daily mean anomalies."""
        return float(np.percentile(self.abs_weather_anomalies, 99.5))


@dataclasses.dataclass(frozen=True)
class SiteRange:
    """Min/max coupling slope across sites."""

    min: float
    max: float


@dataclasses.dataclass(frozen=True)
class TimingResult:
    """Result of fitting hour = reference + base_lag + seasonal_amp * factor."""

    base_lag: float
    seasonal_amp: float
    noise_std: float
    noise_min: float
    noise_max: float
    deviations: np.ndarray  # (N, 274) raw deviations from model
    deviation_mask: np.ndarray  # (N, 274) bool — valid entries


@dataclasses.dataclass
class SeasonalCalibration:
    """Stage 1 seasonal cycle + AR(1) model parameters."""

    harmonic: HarmonicFit
    ar_coefficient: float


@dataclasses.dataclass
class DiurnalCalibration:
    """Diurnal pattern parameters: seasonal shape, timing, exponents, coupling."""

    # Seasonal shape
    seasonal_harmonic: HarmonicFit
    # Timing (direct embed of generation type — noise stds are initial estimates,
    # overridden by optimizer in final params)
    timing: DiurnalTimingParams
    # Waveform shape (analytically derived, used directly in final params)
    shape: WaveformShapeParams
    # Weather-diurnal coupling (→ SiteVariationParams at assembly)
    site_weather_diurnal_coupling_min: float
    site_weather_diurnal_coupling_max: float
    site_peak_hour_weather_coupling_min: float
    site_peak_hour_weather_coupling_max: float
    site_trough_hour_weather_coupling_min: float
    site_trough_hour_weather_coupling_max: float
    # Timing fit results (for variance bound extraction)
    peak_timing: TimingResult | None = None
    trough_timing: TimingResult | None = None


@dataclasses.dataclass(frozen=True)
class FixedVarianceParams:
    """18 point-value parameters derived directly from Swiss data."""

    site_offset_std: float
    year_anomaly_std: float
    site_peak_hour_std: float
    peak_hour_shift_std: float
    site_trough_hour_std: float
    trough_hour_shift_std: float
    site_phase_shift_std: float
    site_seasonal_amp_std: float
    weather_correlation_min: float
    weather_correlation_max: float
    site_diurnal_winter_std: float
    site_diurnal_summer_std: float
    daily_diurnal_noise_min: float
    daily_diurnal_noise_max: float
    siteyear_factor_min: float
    siteyear_factor_max: float


@dataclasses.dataclass(frozen=True)
class CalibratedVarianceParams:
    """9 parameters determined by Bayesian optimization."""

    year_seasonal_amplitude_std: float
    diurnal_amplitude_factor_std: float
    daily_peak_hour_noise_std: float
    daily_trough_hour_noise_std: float
    weather_marginal_std: float
    daily_diurnal_noise_std: float
    siteyear_diurnal_factor_std: float
    site_diurnal_winter_loc: float
    site_diurnal_summer_loc: float


@dataclasses.dataclass(frozen=True)
class VarianceParams:
    """Variance decomposition parameters (fixed + calibrated)."""

    fixed: FixedVarianceParams
    calibrated: CalibratedVarianceParams


@dataclasses.dataclass
class OptimizationContext:
    """Transient optimization state — not serialized to final params."""

    fixed: FixedVarianceParams
    bounds: dict[str, tuple[float, float]]
    swiss_targets: dict[str, dict[str, float]]
    swiss_mean_targets: dict[str, float]


@dataclasses.dataclass
class FixedConstants:
    """Physical constants and sensor parameters (not data-derived)."""

    measurement_noise_std: float = 0.3


@dataclasses.dataclass
class CalibrationData:
    """Single object threaded through the calibration pipeline."""

    observed: ObservedData
    constants: FixedConstants
    seasonal: SeasonalCalibration | None = None
    diurnal: DiurnalCalibration | None = None
    variance: VarianceParams | None = None
    optimization: OptimizationContext | None = None

    def to_temperature_params(self) -> TemperatureParams:
        """Build TemperatureParams from calibrated data.

        Requires seasonal, diurnal, and variance to be populated
        (variance must have both fixed and calibrated sub-params set).
        """
        assert self.seasonal is not None
        assert self.diurnal is not None
        assert self.variance is not None
        fixed = self.variance.fixed
        calibrated = self.variance.calibrated

        # Override timing noise stds with calibrated values
        final_timing = dataclasses.replace(
            self.diurnal.timing,
            daily_peak_hour_noise_std=calibrated.daily_peak_hour_noise_std,
            daily_trough_hour_noise_std=calibrated.daily_trough_hour_noise_std,
        )

        return TemperatureParams(
            seasonal=self.seasonal.harmonic.params,
            weather=WeatherParams(
                ar_coefficient=self.seasonal.ar_coefficient,
                marginal_std=calibrated.weather_marginal_std,
                anomaly_soft_cap=self.observed.weather_anomaly_soft_cap,
            ),
            site=SiteVariationParams(
                offset_std=fixed.site_offset_std,
                peak_hour_std=fixed.site_peak_hour_std,
                trough_hour_std=fixed.site_trough_hour_std,
                seasonal_amp_std=fixed.site_seasonal_amp_std,
                phase_shift_std=fixed.site_phase_shift_std,
                diurnal_winter_loc=calibrated.site_diurnal_winter_loc,
                diurnal_winter_std=fixed.site_diurnal_winter_std,
                diurnal_summer_loc=calibrated.site_diurnal_summer_loc,
                diurnal_summer_std=fixed.site_diurnal_summer_std,
                weather_correlation_min=fixed.weather_correlation_min,
                weather_correlation_max=fixed.weather_correlation_max,
                weather_diurnal_coupling_min=self.diurnal.site_weather_diurnal_coupling_min,
                weather_diurnal_coupling_max=self.diurnal.site_weather_diurnal_coupling_max,
                peak_hour_weather_coupling_min=self.diurnal.site_peak_hour_weather_coupling_min,
                peak_hour_weather_coupling_max=self.diurnal.site_peak_hour_weather_coupling_max,
                trough_hour_weather_coupling_min=self.diurnal.site_trough_hour_weather_coupling_min,
                trough_hour_weather_coupling_max=self.diurnal.site_trough_hour_weather_coupling_max,
            ),
            year=YearVariationParams(
                anomaly_std=fixed.year_anomaly_std,
                seasonal_amp_std=calibrated.year_seasonal_amplitude_std,
                diurnal_amplitude_factor_std=calibrated.diurnal_amplitude_factor_std,
                peak_hour_shift_std=fixed.peak_hour_shift_std,
                trough_hour_shift_std=fixed.trough_hour_shift_std,
            ),
            diurnal=DiurnalParams(
                seasonal=self.diurnal.seasonal_harmonic.params,
                amplitude=DiurnalAmplitudeParams(
                    siteyear_factor_std=calibrated.siteyear_diurnal_factor_std,
                    siteyear_factor_min=fixed.siteyear_factor_min,
                    siteyear_factor_max=fixed.siteyear_factor_max,
                    daily_noise_std=calibrated.daily_diurnal_noise_std,
                    daily_noise_min=fixed.daily_diurnal_noise_min,
                    daily_noise_max=fixed.daily_diurnal_noise_max,
                ),
                timing=final_timing,
                shape=self.diurnal.shape,
            ),
            output=OutputParams(
                measurement_noise_std=self.constants.measurement_noise_std
            ),
        )


__all__ = [
    "CalibratedVarianceParams",
    "CalibrationData",
    "DiurnalCalibration",
    "FixedConstants",
    "FixedVarianceParams",
    "HarmonicFit",
    "ObservedData",
    "OptimizationContext",
    "SeasonalCalibration",
    "SiteRange",
    "TimingResult",
    "VarianceParams",
    "load_swiss_data",
]
