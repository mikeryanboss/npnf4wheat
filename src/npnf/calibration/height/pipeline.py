"""Height calibration pipeline steps."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.data import HeightOptimizationConfig
from npnf.calibration.height.optimization import optimize_pool_params

if TYPE_CHECKING:
    from npnf.calibration.height.data import FIP1YearData, HeightOptimizationResults
    from npnf.data.synthetic.height.params import HeightPoolParams

logger = logging.getLogger(__name__)


def optimize_step(
    year_data: FIP1YearData,
    output_dir: Path,
    num_trials: int = 3000,
    num_genotypes: int = 1000,
    exclude_height_years: set[int] | None = None,
    *,
    dates: HeightDates,
) -> tuple[HeightPoolParams, HeightOptimizationResults]:
    """[2/3] Optimize pool parameters and save results.

    Runs Bayesian optimization of B-spline surface CP and tau_max
    distributions. Saves height/fit.json and height/params_pool.json.

    Args:
        year_data: Per-year aggregated FIP1 data.
        output_dir: Directory to save output.
        num_trials: Number of Optuna optimization trials.
        num_genotypes: Number of genotypes to sample per trial.
        exclude_height_years: Years to exclude from height losses.
            Defaults to {2016}. Pass empty set to include all years.

    Returns:
        Tuple of (pool_params, optimization_results).
    """
    if exclude_height_years is None:
        exclude_height_years = {2016}

    output_dir = Path(output_dir)
    config = HeightOptimizationConfig()

    logger.info("Num trials: %d", num_trials)
    logger.info("Num genotypes: %d", num_genotypes)
    if exclude_height_years:
        logger.info(
            "Excluding years from height losses: %s", sorted(exclude_height_years)
        )

    temp_params_path = output_dir / "temperature/params_pool.json"
    pool_params, optimization_results = optimize_pool_params(
        num_genotypes=num_genotypes,
        num_trials=num_trials,
        calibration_params_path=temp_params_path,
        exclude_height_years=exclude_height_years,
        year_data=year_data,
        config=config,
        dates=dates,
    )

    logger.info("  Best trial: #%d", optimization_results.best_trial)
    logger.info("  Total loss: %.4f", optimization_results.optimization_loss)
    logger.info("  Height loss: %.4f", optimization_results.height_loss)
    logger.info("  Ordering slope: %.6f", optimization_results.ordering_slope)
    logger.info("  Mean height: %.4fm", optimization_results.mean_height)

    # Serialize results (exclude bulky trial history from saved file)
    results_dict = dataclasses.asdict(optimization_results)
    results_dict.pop("trial_history", None)

    output_path = output_dir / "height/fit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Saved results to %s", output_path)

    pool_path = output_dir / "height/params_pool.json"
    with pool_path.open("w") as f:
        json.dump(dataclasses.asdict(pool_params), f, indent=2)
    logger.info("Saved pool params to %s", pool_path)

    return pool_params, optimization_results
