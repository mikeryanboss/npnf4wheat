"""Optuna-based optimization for height calibration."""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import optuna
import torch
from scipy.stats import trim_mean

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.data import (
    HeightOptimizationConfig,
    HeightOptimizationResults,
)
from npnf.calibration.height.forward import _precompute_t_basis
from npnf.data.synthetic.height.params import (
    HeightPoolParams,
    height_pool_params_from_trial_params,
)
from npnf.data.synthetic.height.response_surface import BSplineLUT
from npnf.data.synthetic.temperature.generator import SyntheticTemperatureGenerator
from npnf.data.synthetic.temperature.params import (
    CALIBRATION_PARAMS_ENV_VAR,
    load_temperature_params,
)

from .defaults import _compute_fip1_defaults
from .objectives import EvalContext, _evaluate_single_seed

if TYPE_CHECKING:
    from npnf.calibration.height.data import FIP1YearData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StudyContext:
    """Immutable context passed to each Optuna trial evaluation."""

    config: HeightOptimizationConfig
    param_bounds: dict[str, tuple[float, float]]
    precomputed_screening: dict
    precomputed_full: dict
    eval_ctx: EvalContext
    num_genotypes: int


def _generate_ordering_temperatures(
    calibration_params_path: str | Path | None,
    primary_seed: int,
    num_ordering_years: int,
    *,
    dates: HeightDates,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic ordering temperatures with all noise zeroed.

    Returns:
        Tuple of (ordering_temperatures, ordering_growing_season_temperatures)
        where ordering_temperatures has shape (num_ordering_years, 274, 24) and
        ordering_growing_season_temperatures has shape (num_ordering_years,).
    """
    params_path = calibration_params_path or os.environ.get(CALIBRATION_PARAMS_ENV_VAR)
    if params_path is None:
        msg = (
            "No calibration_params_path given and "
            f"{CALIBRATION_PARAMS_ENV_VAR} is not set"
        )
        raise ValueError(msg)
    base_temperature_params = load_temperature_params(Path(params_path))
    noiseless_temperature_params = dataclasses.replace(
        base_temperature_params,
        weather=dataclasses.replace(base_temperature_params.weather, marginal_std=0.0),
        year=dataclasses.replace(
            base_temperature_params.year,
            seasonal_amp_std=0.0,
            diurnal_amplitude_factor_std=0.0,
            peak_hour_shift_std=0.0,
            trough_hour_shift_std=0.0,
        ),
        diurnal=dataclasses.replace(
            base_temperature_params.diurnal,
            amplitude=dataclasses.replace(
                base_temperature_params.diurnal.amplitude, daily_noise_std=0.0
            ),
            timing=dataclasses.replace(
                base_temperature_params.diurnal.timing,
                daily_peak_hour_noise_std=0.0,
                daily_trough_hour_noise_std=0.0,
            ),
        ),
        output=dataclasses.replace(
            base_temperature_params.output, measurement_noise_std=0.0
        ),
    )
    temperature_generator = SyntheticTemperatureGenerator(
        num_sites=1,
        num_years=num_ordering_years,
        seed=primary_seed,
        temperature_params=noiseless_temperature_params,
    )
    ordering_temperatures = temperature_generator.generate_all()  # (K, 274, 24)

    ordering_growing_season_temperatures = ordering_temperatures[
        :, dates.tau_start_idx : dates.growing_period_end_idx
    ].mean(dim=(1, 2))
    return ordering_temperatures, ordering_growing_season_temperatures


def _precompute_evaluation_batches(
    year_temps: dict[int, torch.Tensor],
    years: list[int],
    num_genotypes: int,
    ordering_temperatures: torch.Tensor,
    num_ordering_genotypes: int,
    *,
    dates: HeightDates,
) -> tuple[dict, dict, BSplineLUT]:
    """Batch temperatures per evaluation pass and precompute B-spline basis.

    Builds three sets of tensors (full, screening, ordering) on the best
    available device so they can be reused across all Optuna trials.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Height calibration using device: %s", device)

    temperature_batch = torch.stack([year_temps[y] for y in years])

    hourly_temps_full = temperature_batch.repeat(num_genotypes, 1, 1)
    hourly_temps_screening = temperature_batch.repeat(200, 1, 1)
    ordering_hourly_temps = ordering_temperatures.repeat(num_ordering_genotypes, 1, 1)

    temperature_basis_full = _precompute_t_basis(hourly_temps_full, dates=dates).to(
        device
    )
    temperature_basis_screening = _precompute_t_basis(
        hourly_temps_screening, dates=dates
    ).to(device)
    temperature_basis_ordering = _precompute_t_basis(
        ordering_hourly_temps, dates=dates
    ).to(device)

    hourly_temps_full = hourly_temps_full.to(device)
    hourly_temps_screening = hourly_temps_screening.to(device)
    ordering_hourly_temps = ordering_hourly_temps.to(device)

    precomputed_full = {
        "hourly_temps": hourly_temps_full,
        "ordering_hourly_temps": ordering_hourly_temps,
        "B_T": temperature_basis_full,
        "ordering_B_T": temperature_basis_ordering,
    }
    precomputed_screening = {
        "hourly_temps": hourly_temps_screening,
        "B_T": temperature_basis_screening,
    }

    tau_lookup_table = BSplineLUT()
    tau_lookup_table.basis_grid = tau_lookup_table.basis_grid.to(device)

    return precomputed_full, precomputed_screening, tau_lookup_table


def _objective(trial: optuna.Trial, study_ctx: _StudyContext) -> float:
    """Optuna objective: suggest params, run screening + full evaluation."""
    config = study_ctx.config
    seed_list = config.seed_list

    trial_params: dict[str, float] = {}
    for name in config.optimized_param_names:
        trial_params[name] = trial.suggest_float(name, *study_ctx.param_bounds[name])

    # Screening pass (low-fidelity)
    screening_loss, _ = _evaluate_single_seed(
        study_ctx.eval_ctx,
        trial_params,
        seed=seed_list[0],
        num_genotypes=200,
        precomputed=study_ctx.precomputed_screening,
        skip_ordering=True,
    )
    trial.report(screening_loss, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned

    # Full-fidelity evaluation
    results = [
        _evaluate_single_seed(
            study_ctx.eval_ctx,
            trial_params,
            seed=seed,
            num_genotypes=study_ctx.num_genotypes,
            precomputed=study_ctx.precomputed_full,
        )
        for seed in seed_list
    ]

    seed_losses = [loss for loss, _ in results]
    aggregate = float(trim_mean(seed_losses, proportiontocut=0.15))
    trial.report(aggregate, step=1)

    trial.set_user_attr("mean_loss", float(np.mean(seed_losses)))
    trial.set_user_attr("std_loss", float(np.std(seed_losses)))

    _, first_diagnostics = results[0]
    trial.set_user_attr("height_loss", first_diagnostics["height_loss"])
    trial.set_user_attr("ordering_loss", first_diagnostics["ordering_loss"])
    trial.set_user_attr("growth_end_loss", first_diagnostics["growth_end_loss"])
    trial.set_user_attr("ordering_slope", first_diagnostics["ordering_slope"])

    return aggregate


def _extract_best_results(
    study: optuna.Study,
    study_ctx: _StudyContext,
    *,
    num_trials: int,
    num_ordering_years: int,
    exclude_height_years: set[int] | None,
) -> tuple[HeightPoolParams, HeightOptimizationResults]:
    """Extract best trial params and build result objects."""
    ctx = study_ctx.eval_ctx

    trial_history = [
        {
            "number": t.number,
            "value": t.value,
            "height_loss": t.user_attrs.get("height_loss"),
            "ordering_loss": t.user_attrs.get("ordering_loss"),
            "growth_end_loss": t.user_attrs.get("growth_end_loss"),
        }
        for t in study.trials
        if t.value is not None
    ]

    best = study.best_trial
    best_trial_params = dict(best.params)

    _, best_diagnostics = _evaluate_single_seed(
        ctx,
        best_trial_params,
        seed=study_ctx.config.seed_list[0],
        num_genotypes=study_ctx.num_genotypes,
        precomputed=study_ctx.precomputed_full,
    )

    pool_params = height_pool_params_from_trial_params(best_trial_params)

    optimization_results = HeightOptimizationResults(
        best_trial=best.number,
        optimization_loss=study.best_value,
        height_loss=best_diagnostics["height_loss"],
        ordering_loss=best_diagnostics["ordering_loss"],
        growth_end_loss=best_diagnostics["growth_end_loss"],
        ordering_slope=best_diagnostics["ordering_slope"],
        mean_height=best_diagnostics["mean_height"],
        finite_loss=best_diagnostics["finite_loss"],
        height_range_loss=best_diagnostics["height_range_loss"],
        non_finite_trajectory_count=best_diagnostics["non_finite_trajectory_count"],
        final_height_peryear_loss=best_diagnostics["final_height_peryear_loss"],
        sim_pooled_mean=best_diagnostics["sim_pooled_mean"],
        sim_pooled_std=best_diagnostics["sim_pooled_std"],
        num_ordering_genotypes=ctx.num_ordering_genotypes,
        num_ordering_years=num_ordering_years,
        num_trials=num_trials,
        num_genotypes=study_ctx.num_genotypes,
        mean_loss=best.user_attrs.get("mean_loss"),
        std_loss=best.user_attrs.get("std_loss"),
        exclude_height_years=(
            sorted(exclude_height_years) if exclude_height_years else []
        ),
        sim_height_means=best_diagnostics["sim_year_means"],
        sim_height_stds=best_diagnostics["sim_year_stds"],
        trial_history=trial_history,
        optuna_params=dict(best.params),
    )

    return pool_params, optimization_results


def optimize_pool_params(
    num_genotypes: int = 1000,
    num_trials: int = 3000,
    calibration_params_path: str | Path | None = None,
    exclude_height_years: set[int] | None = None,
    year_data: FIP1YearData | None = None,
    *,
    config: HeightOptimizationConfig,
    dates: HeightDates,
) -> tuple[HeightPoolParams, HeightOptimizationResults]:
    """Optimize pool distribution parameters using MMD (Energy Distance).

    Calibrates B-spline control point and tau_max distributions by sampling
    genotypes, running the forward model, and comparing height distributions
    against FIP1 observations.

    Args:
        num_genotypes: Number of genotypes to sample per trial.
        num_trials: Number of Optuna optimization trials.
        calibration_params_path: Path to temperature calibration params.
        exclude_height_years: Years to exclude from height losses.
        year_data: Pre-loaded FIP1 year data (required).
        config: Optimization config.

    Returns:
        Tuple of (HeightPoolParams, HeightOptimizationResults).
    """
    if year_data is None:
        msg = "year_data is required"
        raise ValueError(msg)
    num_ordering_genotypes = 200
    num_ordering_years = 20

    years = sorted(year_data.year_temps.keys())

    fip1_defaults = _compute_fip1_defaults(
        year_temps=year_data.year_temps,
        observed_heights=year_data.observed_heights,
        observed_growth_end=year_data.observed_growth_end,
        dates=dates,
    )
    param_bounds = {
        name: (fip1_defaults[name]["min"], fip1_defaults[name]["max"])
        for name in config.optimized_param_names
    }
    initial_params = {
        name: fip1_defaults[name]["default"] for name in config.optimized_param_names
    }

    ordering_temperatures, ordering_growing_season_temperatures = (
        _generate_ordering_temperatures(
            calibration_params_path,
            config.seed_list[0],
            num_ordering_years,
            dates=dates,
        )
    )

    precomputed_full, precomputed_screening, tau_lookup_table = (
        _precompute_evaluation_batches(
            year_data.year_temps,
            years,
            num_genotypes,
            ordering_temperatures,
            num_ordering_genotypes,
            dates=dates,
        )
    )

    study_ctx = _StudyContext(
        config=config,
        param_bounds=param_bounds,
        precomputed_screening=precomputed_screening,
        precomputed_full=precomputed_full,
        eval_ctx=EvalContext(
            years=years,
            year_temps=year_data.year_temps,
            observed_trajectories=year_data.observed_trajectories,
            observed_heights=year_data.observed_heights,
            observed_growth_end=year_data.observed_growth_end,
            num_ordering_genotypes=num_ordering_genotypes,
            exclude_height_years=exclude_height_years or set(),
            ordering_temps=ordering_temperatures,
            ordering_gs_temps=ordering_growing_season_temperatures,
            tau_lut=tau_lookup_table,
            dates=dates,
        ),
        num_genotypes=num_genotypes,
    )

    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="pool_params_calibration",
        direction="minimize",
        storage=None,
        sampler=optuna.samplers.TPESampler(seed=config.seed_list[0]),
        pruner=optuna.pruners.PercentilePruner(
            percentile=50.0, n_startup_trials=10, n_warmup_steps=0
        ),
    )
    study.enqueue_trial(initial_params)
    study.optimize(
        functools.partial(_objective, study_ctx=study_ctx),
        n_trials=num_trials,
        show_progress_bar=True,
    )

    return _extract_best_results(
        study,
        study_ctx,
        num_trials=num_trials,
        num_ordering_years=num_ordering_years,
        exclude_height_years=exclude_height_years,
    )
