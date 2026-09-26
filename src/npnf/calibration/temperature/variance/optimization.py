"""Bayesian optimization for calibration parameters."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import optuna
from joblib import Parallel, delayed

from npnf.calibration.temperature.variance.objectives import (
    DEFAULT_SEED_LIST,
    SCREENING_SEED,
    _apply_calibrated_params,
    _compute_components_for_seed,
    _compute_seed_loss,
    robust_variance_objective,
)
from npnf.data.synthetic.temperature.params import TemperatureParams

logger = logging.getLogger(__name__)


def optimize_variance_params(
    num_trials: int = 50,
    *,
    output_dir: Path | None = None,
    show_progress: bool = True,
    swiss_targets: dict[str, dict[str, float]],
    swiss_mean_targets: dict[str, float],
    seed_list: list[int] = DEFAULT_SEED_LIST,
    num_startup_trials: int = 20,
    trim_fraction: float = 0.2,
    mean_weight: float = 0.3,
    base_params: TemperatureParams,
    calibrated_bounds: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], optuna.Study]:
    """Optimize variance parameters using Bayesian optimization.

    Uses multi-seed robust objective with trimmed mean aggregation
    and multi-fidelity evaluation with pruning.

    Args:
        num_trials: Number of optimization trials
        output_dir: Optional directory to save study database and diagnostics
        show_progress: Whether to show progress bar
        swiss_targets: Target std-component values
        seed_list: Seeds for multi-seed evaluation
        num_startup_trials: Random trials before TPE exploitation begins
        trim_fraction: Fraction to trim from each end for trimmed mean
            aggregation (default 0.2)
        mean_weight: Weight for mean-target loss terms
        base_params: Base TemperatureParams with fixed values.
            Trial suggestions are overlaid via dataclasses.replace.
        calibrated_bounds: Data-derived bounds for the 9 optimizer params.

    Returns:
        Tuple of (best_params dict, optuna Study object)
    """
    logger.info(
        "Calibrated bounds (9D): %s",
        {k: f"[{lo:.4f}, {hi:.4f}]" for k, (lo, hi) in calibrated_bounds.items()},
    )

    storage = None
    if output_dir is not None:
        db_path = output_dir / "temperature/3_variance/optimization/variance_study.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()
        storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="variance_calibration",
        direction="minimize",
        storage=storage,
        sampler=optuna.samplers.TPESampler(
            seed=42, n_startup_trials=num_startup_trials
        ),
        pruner=optuna.pruners.PercentilePruner(
            percentile=50.0, n_startup_trials=10, n_warmup_steps=0
        ),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        lambda trial: robust_variance_objective(
            trial,
            swiss_targets,
            seed_list=seed_list,
            base_params=base_params,
            calibrated_bounds=calibrated_bounds,
            swiss_mean_targets=swiss_mean_targets,
            trim_fraction=trim_fraction,
            mean_weight=mean_weight,
        ),
        n_trials=num_trials,
        show_progress_bar=show_progress,
    )

    if output_dir is not None:
        _write_diagnostics(
            study,
            swiss_targets,
            seed_list,
            output_dir,
            base_params,
            swiss_mean_targets=swiss_mean_targets,
            num_startup_trials=num_startup_trials,
            trim_fraction=trim_fraction,
            mean_weight=mean_weight,
        )

    return study.best_params, study


def _write_diagnostics(
    study: optuna.Study,
    swiss_targets: dict[str, dict[str, float]],
    seed_list: list[int],
    output_dir: Path,
    base_params: TemperatureParams,
    *,
    swiss_mean_targets: dict[str, float],
    num_startup_trials: int = 20,
    trim_fraction: float = 0.2,
    mean_weight: float = 0.3,
) -> None:
    """Write Stage 2 diagnostics JSON with per-seed data."""
    best = study.best_trial
    best_params = best.params

    eval_params = _apply_calibrated_params(base_params, best_params)
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_compute_components_for_seed)(eval_params, seed, swiss_targets)
        for seed in seed_list
    )
    seed_losses = []
    per_seed_data = []
    for seed, (components, metric_means) in zip(seed_list, results, strict=True):
        loss, component_losses = _compute_seed_loss(
            components,
            swiss_targets,
            mean_weight=mean_weight,
            metric_means=metric_means,
            swiss_mean_targets=swiss_mean_targets,
        )
        seed_losses.append(loss)
        per_seed_data.append(
            {
                "seed": seed,
                "loss": loss,
                "components": components,
                "component_losses": component_losses,
                "metric_means": metric_means,
            }
        )

    mean_loss_val = float(np.mean(seed_losses))
    std_loss = float(np.std(seed_losses))

    num_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )

    trial_summaries = []
    for trial in study.trials:
        if trial.value is None or trial.state == optuna.trial.TrialState.PRUNED:
            continue
        trial_summaries.append(
            {
                "trial_number": trial.number,
                "aggregate_loss": trial.value,
                "mean_loss": trial.user_attrs["mean_loss"],
                "std_loss": trial.user_attrs["std_loss"],
            }
        )

    diagnostics = {
        "objective_space": "std_components_v1",
        "seed_list": seed_list,
        "trim_fraction": trim_fraction,
        "num_startup_trials": num_startup_trials,
        "mean_weight": mean_weight,
        "mean_targets": swiss_mean_targets,
        "fidelity_schedule": {
            "stages": [
                {
                    "step": 0,
                    "num_sites": 15,
                    "num_years": 15,
                    "seeds": [SCREENING_SEED],
                },
                {"step": 1, "num_sites": 30, "num_years": 30, "seeds": seed_list},
            ],
            "pruner": "PercentilePruner",
            "pruner_percentile": 50.0,
        },
        "num_pruned_trials": num_pruned,
        "best_trial_number": best.number,
        "best_aggregate_loss": best.value,
        "best_mean_loss": mean_loss_val,
        "best_std_loss": std_loss,
        "best_params": best_params,
        "targets": swiss_targets,
        "per_seed": per_seed_data,
        "base_params": dataclasses.asdict(base_params),
        "trial_summaries": trial_summaries,
        "num_trials": len(study.trials),
    }

    diag_path = (
        output_dir / "temperature/3_variance/optimization/variance_diagnostics.json"
    )
    with diag_path.open("w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Diagnostics saved to: %s", diag_path)


__all__ = ["optimize_variance_params"]
