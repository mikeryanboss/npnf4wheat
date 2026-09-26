"""Height calibration orchestrator.

Runs the 3-step pipeline in order:
  [1/3] Load FIP1 data
  [2/3] Optimize pool parameters
  [3/3] Generate diagnostic visualizations

Usage:
    python -m npnf.scripts.calibration.height [OPTIONS]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.data import load_fip1_year_data
from npnf.calibration.height.evaluation.visualize import (
    _compute_height_trajectories,
    plot_height_distribution_comparison,
    plot_response_surface,
    plot_trajectory_comparison_by_year,
    plot_trajectory_diagnostic_samples,
    plot_warm_short,
)
from npnf.calibration.height.optimization.visualize import plot_height_optimization
from npnf.calibration.height.pipeline import optimize_step
from npnf.data.datasets.fip1 import Fip1Facts, get_heights_dataset
from npnf.data.pools import GenotypePool

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    output_dir: Path,
    num_trials: int = 3000,
    num_genotypes: int = 1000,
    exclude_height_years: set[int] | None = None,
) -> None:
    """Run the full height calibration pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = HeightDates()

    logger.info("[1/3] Loading FIP1 data")
    fip1_dataset = get_heights_dataset(
        split=["train", "validation", "test_plot", "test_genotype", "test_environment"],
        standardize=False,
        exclude_years=[Fip1Facts().held_out_year],
        datasets_offline_path=os.environ.get("NPNF_DATASET_PATH"),
    ).dataset
    year_data = load_fip1_year_data(fip1_data=fip1_dataset, dates=dates)

    logger.info("[2/3] Optimizing pool parameters")
    pool_params, optimization_results = optimize_step(
        year_data,
        output_dir,
        num_trials=num_trials,
        num_genotypes=num_genotypes,
        exclude_height_years=exclude_height_years,
        dates=dates,
    )

    # Visualization pass
    logger.info("[3/3] Generating visualizations")

    pool = GenotypePool.sample(
        num_genotypes=num_genotypes, seed=42, height_pool_params=pool_params
    )

    # Simulate heights per year
    year_temps = year_data.year_temps
    all_sim_heights = []
    for year in sorted(year_temps):
        year_temp = year_temps[year].unsqueeze(0).expand(len(pool), -1, -1)
        trajectories, _ = _compute_height_trajectories(
            year_temp, genotype_pool=pool, dates=dates
        )
        all_sim_heights.append(trajectories.max(dim=-1).values)

    sim_heights_by_year = {
        year: heights.detach().numpy()
        for year, heights in zip(sorted(year_temps), all_sim_heights, strict=True)
    }

    plot_height_distribution_comparison(fip1_dataset, sim_heights_by_year, output_dir)
    logger.info("Created: plots/height/distribution.png")

    temp_params_path = output_dir / "temperature/params_pool.json"
    plot_warm_short(
        pool,
        output_dir,
        calibration_params_path=temp_params_path if temp_params_path.exists() else None,
        dates=dates,
    )
    logger.info("Created: plots/height/warm_short.png")

    plot_trajectory_comparison_by_year(
        fip1_dataset, output_dir, genotype_pool=pool, dates=dates
    )
    logger.info("Created: plots/height/trajectory_by_year.png")

    trial_history = optimization_results.trial_history
    if trial_history:
        plot_height_optimization(trial_history, optimization_results, output_dir)
        logger.info("Created: plots/height/optimization.png")

    plot_response_surface(pool_params, pool, output_dir)
    logger.info("Created: plots/height/response_surface.png")

    plot_trajectory_diagnostic_samples(year_temps, pool, output_dir, dates=dates)
    logger.info("Created: plots/height/trajectory_diagnostic_1..5.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run height calibration pipeline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/calibration"),
        help="Directory to save results (default: results/calibration)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3000,
        help="Number of Optuna optimization trials (default: 3000)",
    )
    parser.add_argument(
        "--num-genotypes",
        type=int,
        default=1000,
        help="Number of genotypes to sample per trial (default: 1000)",
    )
    parser.add_argument(
        "--exclude-height-years",
        type=int,
        nargs="*",
        default=None,
        help="Years to exclude from height losses (default: {2016})",
    )
    args = parser.parse_args()

    kwargs = {
        "output_dir": args.output_dir,
        "num_trials": args.num_trials,
        "num_genotypes": args.num_genotypes,
    }
    if args.exclude_height_years is not None:
        kwargs["exclude_height_years"] = set(args.exclude_height_years)
    run_pipeline(**kwargs)


if __name__ == "__main__":
    sys.exit(main())
