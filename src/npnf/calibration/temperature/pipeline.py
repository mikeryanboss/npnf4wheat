"""Temperature calibration pipeline utilities."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from npnf.calibration.temperature.data import (
    CalibratedVarianceParams,
    CalibrationData,
    DiurnalCalibration,
    FixedVarianceParams,
    ObservedData,
    OptimizationContext,
    SeasonalCalibration,
    VarianceParams,
)
from npnf.calibration.temperature.diurnal.coupling import (
    fit_peak_hour_weather_coupling,
    fit_trough_hour_weather_coupling,
    fit_weather_diurnal_coupling,
)
from npnf.calibration.temperature.diurnal.fitting import (
    fit_shape_params,
    fit_timing_params,
)
from npnf.calibration.temperature.helpers import acf_lag1, fit_harmonic
from npnf.calibration.temperature.variance.extraction import (
    extract_variance_params as _extract_variance_params,
)
from npnf.calibration.temperature.variance.objectives import (
    compute_swiss_calibration_targets,
)
from npnf.calibration.temperature.variance.optimization import optimize_variance_params
from npnf.data.synthetic.temperature import (
    DiurnalAmplitudeParams,
    DiurnalParams,
    DiurnalTimingParams,
    OutputParams,
    SiteVariationParams,
    TemperatureParams,
    WeatherParams,
    YearVariationParams,
)

if TYPE_CHECKING:
    from optuna import Study

logger = logging.getLogger(__name__)


def fit_seasonal_model(data: CalibrationData) -> SeasonalCalibration:
    """Fit seasonal model and AR(1) coefficient."""
    harmonic, fit_rmse = fit_harmonic(data.observed.mean_seasonal)
    residuals = data.observed.daily_means - harmonic.values
    ar = float(acf_lag1(residuals).mean())
    params = SeasonalCalibration(harmonic=harmonic, ar_coefficient=ar)

    logger.info(
        "  base_temp=%.3f  amplitude=%.3f  ar=%.4f  rmse=%.4f",
        harmonic.params.base,
        harmonic.params.amplitude,
        params.ar_coefficient,
        fit_rmse,
    )

    assert fit_rmse < 3.0, f"fit_rmse={fit_rmse:.4f} >= 3.0"
    assert 0.0 < params.ar_coefficient < 1.0, (
        f"ar_coefficient={params.ar_coefficient:.4f} not in (0, 1)"
    )

    return params


def fit_diurnal_model(data: CalibrationData) -> DiurnalCalibration:
    """Fit diurnal pattern: timing, exponents, weather coupling."""
    assert data.seasonal is not None
    diurnal_harmonic, _ = fit_harmonic(data.observed.diurnal_amplitude.mean(axis=0))
    peak, trough = fit_timing_params(data.observed, diurnal_harmonic)
    amplitude_coupling = fit_weather_diurnal_coupling(
        data.observed, data.seasonal.harmonic, diurnal_harmonic
    )
    peak_hour_coupling = fit_peak_hour_weather_coupling(
        data.observed, data.seasonal.harmonic
    )
    trough_hour_coupling = fit_trough_hour_weather_coupling(
        data.observed, data.seasonal.harmonic
    )

    timing = DiurnalTimingParams(
        peak_hour=peak.base_lag,
        daily_peak_hour_noise_std=peak.noise_std,
        daily_peak_hour_noise_min=peak.noise_min,
        daily_peak_hour_noise_max=peak.noise_max,
        trough_hour=trough.base_lag,
        daily_trough_hour_noise_std=trough.noise_std,
        daily_trough_hour_noise_min=trough.noise_min,
        daily_trough_hour_noise_max=trough.noise_max,
    )

    shape = fit_shape_params(data.observed, diurnal_harmonic)

    return DiurnalCalibration(
        seasonal_harmonic=diurnal_harmonic,
        timing=timing,
        shape=shape,
        site_weather_diurnal_coupling_min=amplitude_coupling.min,
        site_weather_diurnal_coupling_max=amplitude_coupling.max,
        site_peak_hour_weather_coupling_min=peak_hour_coupling.min,
        site_peak_hour_weather_coupling_max=peak_hour_coupling.max,
        site_trough_hour_weather_coupling_min=trough_hour_coupling.min,
        site_trough_hour_weather_coupling_max=trough_hour_coupling.max,
        peak_timing=peak,
        trough_timing=trough,
    )


def _build_optimization_context(
    observed: ObservedData,
    fixed: FixedVarianceParams,
    bounds: dict[str, tuple[float, float]],
) -> OptimizationContext:
    """Compute calibration targets for the optimizer."""
    targets = compute_swiss_calibration_targets(
        observed.temps, observed.sites, observed.years
    )

    return OptimizationContext(
        fixed=fixed,
        bounds=bounds,
        swiss_targets=targets["swiss_targets"],
        swiss_mean_targets=targets["swiss_mean_targets"],
    )


def extract_variance(data: CalibrationData) -> OptimizationContext:
    """Extract fixed variance parameters and build optimization context.

    Does not set ``data.variance`` — that is deferred until
    ``optimize_variance_step`` produces real calibrated values.
    """
    fixed, bounds = _extract_variance_params(data)
    return _build_optimization_context(data.observed, fixed, bounds)


def _build_base_temperature_params(data: CalibrationData) -> TemperatureParams:
    """Build base TemperatureParams from seasonal + diurnal + fixed variance.

    Calibrated variance fields are filled with zeros as placeholders;
    the optimizer overwrites them per trial via ``dataclasses.replace``.
    """
    assert data.seasonal is not None
    assert data.diurnal is not None
    assert data.optimization is not None
    fixed = data.optimization.fixed

    # Start from diurnal timing embed, override noise stds with zero placeholders
    base_timing = dataclasses.replace(
        data.diurnal.timing,
        daily_peak_hour_noise_std=0.0,
        daily_trough_hour_noise_std=0.0,
    )

    return TemperatureParams(
        seasonal=data.seasonal.harmonic.params,
        weather=WeatherParams(
            ar_coefficient=data.seasonal.ar_coefficient,
            marginal_std=0.0,
            anomaly_soft_cap=data.observed.weather_anomaly_soft_cap,
        ),
        site=SiteVariationParams(
            offset_std=fixed.site_offset_std,
            peak_hour_std=fixed.site_peak_hour_std,
            trough_hour_std=fixed.site_trough_hour_std,
            seasonal_amp_std=fixed.site_seasonal_amp_std,
            phase_shift_std=fixed.site_phase_shift_std,
            diurnal_winter_loc=0.0,
            diurnal_winter_std=fixed.site_diurnal_winter_std,
            diurnal_summer_loc=0.0,
            diurnal_summer_std=fixed.site_diurnal_summer_std,
            weather_correlation_min=fixed.weather_correlation_min,
            weather_correlation_max=fixed.weather_correlation_max,
            weather_diurnal_coupling_min=data.diurnal.site_weather_diurnal_coupling_min,
            weather_diurnal_coupling_max=data.diurnal.site_weather_diurnal_coupling_max,
            peak_hour_weather_coupling_min=data.diurnal.site_peak_hour_weather_coupling_min,
            peak_hour_weather_coupling_max=data.diurnal.site_peak_hour_weather_coupling_max,
            trough_hour_weather_coupling_min=data.diurnal.site_trough_hour_weather_coupling_min,
            trough_hour_weather_coupling_max=data.diurnal.site_trough_hour_weather_coupling_max,
        ),
        year=YearVariationParams(
            anomaly_std=fixed.year_anomaly_std,
            seasonal_amp_std=0.0,
            diurnal_amplitude_factor_std=0.0,
            peak_hour_shift_std=fixed.peak_hour_shift_std,
            trough_hour_shift_std=fixed.trough_hour_shift_std,
        ),
        diurnal=DiurnalParams(
            seasonal=data.diurnal.seasonal_harmonic.params,
            amplitude=DiurnalAmplitudeParams(
                siteyear_factor_std=0.0,
                siteyear_factor_min=fixed.siteyear_factor_min,
                siteyear_factor_max=fixed.siteyear_factor_max,
                daily_noise_std=0.0,
                daily_noise_min=fixed.daily_diurnal_noise_min,
                daily_noise_max=fixed.daily_diurnal_noise_max,
            ),
            timing=base_timing,
            shape=data.diurnal.shape,
        ),
        output=OutputParams(measurement_noise_std=data.constants.measurement_noise_std),
    )


def optimize_variance_step(
    data: CalibrationData,
    output_dir: Path,
    *,
    num_trials: int = 200,
    num_startup_trials: int = 20,
    trim_fraction: float = 0.2,
    mean_weight: float = 0.3,
) -> tuple[VarianceParams, Study]:
    """Bayesian optimization of variance parameters."""
    assert data.optimization is not None
    base_params = _build_base_temperature_params(data)

    logger.info("Running Bayesian optimization with %d trials...", num_trials)

    best_params, study = optimize_variance_params(
        num_trials=num_trials,
        output_dir=output_dir,
        show_progress=True,
        swiss_targets=data.optimization.swiss_targets,
        swiss_mean_targets=data.optimization.swiss_mean_targets,
        num_startup_trials=num_startup_trials,
        trim_fraction=trim_fraction,
        mean_weight=mean_weight,
        base_params=base_params,
        calibrated_bounds=data.optimization.bounds,
    )

    logger.info(
        "  trial #%d  loss=%.6f  mean=%.6f  std=%.6f",
        study.best_trial.number,
        study.best_value,
        study.best_trial.user_attrs["mean_loss"],
        study.best_trial.user_attrs["std_loss"],
    )

    variance = VarianceParams(
        fixed=data.optimization.fixed,
        calibrated=CalibratedVarianceParams(**best_params),
    )

    return variance, study


def build_final_params(data: CalibrationData, output_dir: Path) -> TemperatureParams:
    """Merge all calibrated parameters and save."""
    tp = data.to_temperature_params()
    final_params = dataclasses.asdict(tp)
    params_path = output_dir / "temperature/params_pool.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w") as f:
        json.dump(final_params, f, indent=2)
    return tp
