"""Objective functions for variance parameter optimization."""

import dataclasses

import numpy as np
import optuna
from joblib import Parallel, delayed
from scipy.stats import trim_mean

from npnf.calibration.temperature.helpers import acf_lag1
from npnf.calibration.temperature.variance.extraction import site_means, year_means
from npnf.data.synthetic.temperature import (
    SyntheticTemperatureGenerator,
    TemperatureParams,
)


def compute_variance_components(
    values: np.ndarray, num_sites: int, num_years: int
) -> dict[str, float]:
    """Compute std-based variance components from a balanced site x year grid.

    Args:
        values: Metric values, shape (num_sites * num_years,)
        num_sites: Number of sites
        num_years: Number of years

    Returns:
        {"std_site": ..., "std_year": ..., "std_total": ...}
        All stds use population std (ddof=0).
    """
    grid = values.reshape(num_sites, num_years)
    std_site = float(grid.mean(axis=1).std(ddof=0))
    std_year = float(grid.mean(axis=0).std(ddof=0))
    std_total = float(values.std(ddof=0))
    return {"std_site": std_site, "std_year": std_year, "std_total": std_total}


def _suggest_calibrated_params(
    trial: optuna.Trial, bounds: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """Suggest calibrated parameters using data-derived bounds."""
    return {
        name: trial.suggest_float(name, lower, upper)
        for name, (lower, upper) in bounds.items()
    }


def generate_synthetic_for_calibration(
    params: TemperatureParams, num_sites: int = 20, num_years: int = 20, seed: int = 42
) -> np.ndarray:
    """Generate synthetic temperatures for variance calibration.

    Args:
        params: Temperature generation parameters.
        num_sites: Number of sites to generate
        num_years: Number of years to generate
        seed: Random seed

    Returns:
        temps: shape (num_sites * num_years, 274, 24)
    """
    generator = SyntheticTemperatureGenerator(
        num_sites=num_sites, num_years=num_years, seed=seed, temperature_params=params
    )
    return generator.generate_all().numpy()


def compute_all_metrics(temps: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all calibration metrics from temperature arrays.

    Args:
        temps: Temperature array, shape (num_samples, 274, 24)

    Returns:
        Dict mapping metric name -> values array, shape (num_samples,)
    """
    # Daily stats: shape (num_samples, 274)
    daily_max = temps.max(axis=2)
    daily_min = temps.min(axis=2)
    daily_mean = temps.mean(axis=2)

    # Diurnal amplitude: mean of (daily_max - daily_min)
    diurnal_amplitude = (daily_max - daily_min).mean(axis=1)

    # Per-season diurnal amplitude daily std (avoids conflating seasonal
    # ramp with day-to-day noise that the whole-window metric suffered from)
    diurnal_range = daily_max - daily_min
    winter_diurnal_daily_std = diurnal_range[:, :61].std(axis=1)
    summer_diurnal_daily_std = diurnal_range[:, 200:261].std(axis=1)

    # Day-to-day std
    day_to_day_std = daily_mean.std(axis=1)

    # Daily mean lag-1 autocorrelation
    daily_mean_acf1 = acf_lag1(daily_mean)

    # Hour of max/min (argmax/argmin along hour axis)
    hour_of_max_indices = temps.argmax(axis=2).astype(float)  # (num_samples, 274)
    hour_of_min_indices = temps.argmin(axis=2).astype(float)  # (num_samples, 274)

    # Filter to valid timing windows (match calibration masks in data.py:103-104)
    hour_of_max_filtered = np.where(
        (hour_of_max_indices >= 10) & (hour_of_max_indices <= 18),
        hour_of_max_indices,
        np.nan,
    )
    hour_of_min_filtered = np.where(
        hour_of_min_indices <= 12, hour_of_min_indices, np.nan
    )

    hour_of_max = np.nanmean(hour_of_max_filtered, axis=1)
    hour_of_min = np.nanmean(hour_of_min_filtered, axis=1)

    # Max temp reached
    max_temp_reached = daily_max.max(axis=1)

    # Hot days: count of days with max > 30C
    hot_days = (daily_max > 30).sum(axis=1).astype(float)

    # Cold days: count of days with min < 0C
    cold_days = (daily_min < 0).sum(axis=1).astype(float)

    # Hour-of-max/min std: variability of peak/trough timing across days
    hour_of_max_std = np.nanstd(hour_of_max_filtered, axis=1, ddof=0)
    hour_of_min_std = np.nanstd(hour_of_min_filtered, axis=1, ddof=0)

    # Seasonal diurnal amplitude (diurnal_range computed above at line 89)
    winter_diurnal_amplitude = diurnal_range[:, :61].mean(axis=1)
    summer_diurnal_amplitude = diurnal_range[:, 200:261].mean(axis=1)

    # Diurnal shape: mean temperature during morning and afternoon windows
    morning_mean = temps[:, :, 4:10].mean(axis=(1, 2))
    afternoon_mean = temps[:, :, 14:20].mean(axis=(1, 2))

    return {
        "diurnal_amplitude": diurnal_amplitude,
        "winter_diurnal_daily_std": winter_diurnal_daily_std,
        "summer_diurnal_daily_std": summer_diurnal_daily_std,
        "day_to_day_std": day_to_day_std,
        "daily_mean_acf1": daily_mean_acf1,
        "hour_of_max": hour_of_max,
        "hour_of_min": hour_of_min,
        "max_temp_reached": max_temp_reached,
        "hot_days": hot_days,
        "cold_days": cold_days,
        "hour_of_max_std": hour_of_max_std,
        "hour_of_min_std": hour_of_min_std,
        "winter_diurnal_amplitude": winter_diurnal_amplitude,
        "summer_diurnal_amplitude": summer_diurnal_amplitude,
        "morning_mean": morning_mean,
        "afternoon_mean": afternoon_mean,
    }


def compute_swiss_calibration_targets(
    temps: np.ndarray, sites: list[str] | np.ndarray, years: np.ndarray
) -> dict[str, dict]:
    """Compute Swiss calibration target values from real weather data.

    Uses group-by aggregation for variance decomposition (handles
    unbalanced site/year grids). Population std (ddof=0).

    Args:
        temps: Temperature array, shape (num_samples, 274, 24)
        sites: Per-sample site labels
        years: Per-sample year labels

    Returns:
        Dict with keys:
        - "swiss_targets": {metric: {"std_site", "std_year", "std_total"}}
        - "swiss_mean_targets": {metric: mean_value}
    """
    metrics = compute_all_metrics(temps)
    sites_arr = np.asarray(sites)
    years_arr = np.asarray(years)
    site_indices = {
        site: np.where(sites_arr == site)[0] for site in sorted(set(sites_arr))
    }
    year_indices = {
        year: np.where(years_arr == year)[0] for year in sorted(set(years_arr.tolist()))
    }

    swiss_targets: dict[str, dict[str, float]] = {}
    swiss_mean_targets: dict[str, float] = {}

    for metric_name, values in metrics.items():
        swiss_mean_targets[metric_name] = float(np.mean(values))
        sm = site_means(values, site_indices)
        ym = year_means(values, year_indices)
        swiss_targets[metric_name] = {
            "std_site": float(sm.std(ddof=0)),
            "std_year": float(ym.std(ddof=0)),
            "std_total": float(values.std(ddof=0)),
        }

    return {"swiss_targets": swiss_targets, "swiss_mean_targets": swiss_mean_targets}


DEFAULT_SEED_LIST = [42, 123, 456, 0, 1, 2, 3, 4, 5, 6]
SCREENING_SEED = 99


def _compute_components_for_seed(
    params: TemperatureParams,
    seed: int,
    swiss_targets: dict[str, dict[str, float]],
    num_sites: int = 30,
    num_years: int = 30,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Compute std-based variance components and metric means for a single seed.

    Args:
        params: Temperature generation parameters.
        seed: Random seed
        swiss_targets: Target dict (used only for metric names)
        num_sites: Number of sites in the generation grid
        num_years: Number of years in the generation grid

    Returns:
        Tuple of (components, metric_means) where:
        - components: {metric_name: {"std_site", "std_year", "std_total"}}
        - metric_means: {metric_name: float} for all metrics
    """
    temps = generate_synthetic_for_calibration(
        params, num_sites=num_sites, num_years=num_years, seed=seed
    )
    metrics = compute_all_metrics(temps)

    components = {
        metric_name: compute_variance_components(
            metrics[metric_name], num_sites, num_years
        )
        for metric_name in swiss_targets
    }

    metric_means = {name: float(np.mean(values)) for name, values in metrics.items()}
    return components, metric_means


def _compute_seed_loss(
    components: dict[str, dict[str, float]],
    swiss_targets: dict[str, dict[str, float]],
    *,
    mean_weight: float = 0.3,
    metric_means: dict[str, float],
    swiss_mean_targets: dict[str, float],
) -> tuple[float, dict[str, dict[str, float]]]:
    """Compute normalized squared-error loss for one seed's std components.

    Uses normalized error: (actual - target) / max(abs(target), 1e-6),
    then squares and sums across all metrics and components.

    Args:
        components: Computed std components for this seed
            {metric: {"std_site", "std_year", "std_total"}}
        swiss_targets: Target std-component values
        mean_weight: Weight for mean-target loss terms
        metric_means: Actual means per metric from this seed's generation
        swiss_mean_targets: Target means per metric from Swiss data

    Returns:
        Tuple of (total_loss, component_losses) where
        component_losses maps metric -> component -> loss
    """

    total = 0.0
    component_losses: dict[str, dict[str, float]] = {}

    for metric in swiss_targets:
        component_losses[metric] = {}
        for component in ["std_site", "std_year", "std_total"]:
            target = swiss_targets[metric][component]
            actual = components[metric][component]
            normalized_error = (actual - target) / max(abs(target), 1e-6)
            loss = normalized_error**2
            component_losses[metric][component] = loss
            total += loss

        # Mean-target loss
        target = swiss_mean_targets[metric]
        actual = metric_means[metric]
        normalized_error = (actual - target) / max(abs(target), 1e-6)
        loss = mean_weight * normalized_error**2
        component_losses[metric]["mean"] = loss
        total += loss

    return total, component_losses


def _apply_calibrated_params(
    base_params: TemperatureParams, calibrated_params: dict[str, float]
) -> TemperatureParams:
    """Apply flat calibrated param dict to nested TemperatureParams."""
    new_weather = dataclasses.replace(
        base_params.weather, marginal_std=calibrated_params["weather_marginal_std"]
    )
    new_year = dataclasses.replace(
        base_params.year,
        seasonal_amp_std=calibrated_params["year_seasonal_amplitude_std"],
        diurnal_amplitude_factor_std=calibrated_params["diurnal_amplitude_factor_std"],
    )
    new_amplitude = dataclasses.replace(
        base_params.diurnal.amplitude,
        daily_noise_std=calibrated_params["daily_diurnal_noise_std"],
        siteyear_factor_std=calibrated_params["siteyear_diurnal_factor_std"],
    )
    new_timing = dataclasses.replace(
        base_params.diurnal.timing,
        daily_peak_hour_noise_std=calibrated_params["daily_peak_hour_noise_std"],
        daily_trough_hour_noise_std=calibrated_params["daily_trough_hour_noise_std"],
    )
    new_site = dataclasses.replace(
        base_params.site,
        diurnal_winter_loc=calibrated_params["site_diurnal_winter_loc"],
        diurnal_summer_loc=calibrated_params["site_diurnal_summer_loc"],
    )
    new_diurnal = dataclasses.replace(
        base_params.diurnal, amplitude=new_amplitude, timing=new_timing
    )
    return dataclasses.replace(
        base_params,
        weather=new_weather,
        site=new_site,
        year=new_year,
        diurnal=new_diurnal,
    )


def robust_variance_objective(
    trial: optuna.Trial,
    swiss_targets: dict[str, dict[str, float]],
    seed_list: list[int],
    base_params: TemperatureParams,
    calibrated_bounds: dict[str, tuple[float, float]],
    swiss_mean_targets: dict[str, float],
    trim_fraction: float = 0.2,
    mean_weight: float = 0.3,
) -> float:
    """Robust multi-seed objective for variance parameter optimization.

    Evaluates each trial over multiple seeds and aggregates losses using
    trimmed mean aggregation. Uses normalized squared-error loss on std
    components.

    A low-fidelity screening (15x15, first seed) is performed first.
    Unpromising trials are pruned before full evaluation.

    Args:
        trial: Optuna trial for parameter suggestions
        swiss_targets: Target std-component values
        seed_list: Seeds to evaluate
        base_params: Base TemperatureParams with fixed values.
            Trial suggestions are overlaid via dataclasses.replace.
        calibrated_bounds: Data-derived bounds for the 9 optimizer params
        trim_fraction: Fraction to trim from each end for trimmed mean
            aggregation (default 0.2, i.e. 20% from each side)
        mean_weight: Weight for mean-target loss terms

    Returns:
        Aggregate loss across seeds
    """
    calibrated_params = _suggest_calibrated_params(trial, calibrated_bounds)
    params = _apply_calibrated_params(base_params, calibrated_params)

    # Stage 0: low-fidelity screening (15x15, dedicated screening seed)
    lo_components, lo_means = _compute_components_for_seed(
        params, SCREENING_SEED, swiss_targets, num_sites=15, num_years=15
    )
    lo_loss, _ = _compute_seed_loss(
        lo_components,
        swiss_targets,
        mean_weight=mean_weight,
        metric_means=lo_means,
        swiss_mean_targets=swiss_mean_targets,
    )
    trial.report(lo_loss, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned

    # Stage 1: full-fidelity evaluation (30x30, all seeds in parallel)
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_compute_components_for_seed)(params, seed, swiss_targets)
        for seed in seed_list
    )
    seed_losses = []
    for components, metric_means in results:
        loss, _ = _compute_seed_loss(
            components,
            swiss_targets,
            mean_weight=mean_weight,
            metric_means=metric_means,
            swiss_mean_targets=swiss_mean_targets,
        )
        seed_losses.append(loss)

    mean_loss = float(np.mean(seed_losses))
    std_loss = float(np.std(seed_losses))
    aggregate = float(trim_mean(seed_losses, trim_fraction))

    trial.report(aggregate, step=1)

    for seed, loss in zip(seed_list, seed_losses, strict=True):
        trial.set_user_attr(f"seed_{seed}_loss", loss)
    trial.set_user_attr("mean_loss", mean_loss)
    trial.set_user_attr("std_loss", std_loss)

    return aggregate


__all__ = [
    "DEFAULT_SEED_LIST",
    "SCREENING_SEED",
    "compute_all_metrics",
    "compute_swiss_calibration_targets",
    "compute_variance_components",
    "generate_synthetic_for_calibration",
    "robust_variance_objective",
]
