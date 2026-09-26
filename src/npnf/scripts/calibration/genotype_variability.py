"""Validate the genotype variability of the simulator against FIP1.

It reads FIP1 and the calibrated prior, and it writes CSV tables, a JSON file
of the settings and a figure.

Usage:
    python -m npnf.scripts.calibration.genotype_variability [OPTIONS]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.data import load_fip1_year_data
from npnf.calibration.height.genotype_variability import (
    GenotypeVariabilitySettings,
    bootstrap_ratio_intervals,
    fip1_traits_by_year,
    implied_fip1_residual_standard_deviation,
    load_fip1_metadata,
    simulate_pool_traits,
    summary_table,
    year_pair_table,
)
from npnf.calibration.height.genotype_variability.visualize import (
    plot_genotype_variability,
)
from npnf.data.datasets.fip1 import Fip1Facts, get_heights_dataset
from npnf.data.pools import GenotypePool
from npnf.data.synthetic.height.params import load_height_pool_params
from npnf.scripts.utils.outputs import save_json, write_csv


def run(output_dir: Path, parameters_path: Path, num_genotypes: int, seed: int) -> None:
    """Compare pool genotype variability against FIP1 and write the artifacts."""
    dataset_path = os.environ["NPNF_DATASET_PATH"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = HeightDates()
    settings = GenotypeVariabilitySettings()
    facts = Fip1Facts()

    heights_dataset = get_heights_dataset(
        split=list(facts.splits), standardize=False, datasets_offline_path=dataset_path
    ).dataset
    year_data = load_fip1_year_data(fip1_data=heights_dataset, dates=dates)
    metadata = load_fip1_metadata(dataset_path, facts.splits)
    fip1 = fip1_traits_by_year(heights_dataset, metadata)

    pool = GenotypePool.sample(
        num_genotypes=num_genotypes,
        seed=seed,
        height_pool_params=load_height_pool_params(parameters_path),
    )
    max_heights, growth_end_days, elongation_rates = simulate_pool_traits(
        year_data.year_temps, pool, dates
    )

    implied_residual = implied_fip1_residual_standard_deviation(fip1)
    tables = {
        "summary": summary_table(
            fip1, max_heights, growth_end_days, elongation_rates, implied_residual
        ),
        "year_pairs": year_pair_table(fip1, max_heights, seed, implied_residual),
        "ratio_intervals": bootstrap_ratio_intervals(
            fip1,
            max_heights,
            growth_end_days,
            elongation_rates,
            seed=seed,
            settings=settings,
        ),
    }
    for name, rows in tables.items():
        write_csv(output_dir / f"{name}.csv", rows, list(rows[0]))
    save_json(
        output_dir / "settings.json",
        {
            "dataset": dataset_path,
            "calibrated_parameters": str(parameters_path),
            "num_genotypes": num_genotypes,
            "seed": seed,
            "held_out_year": facts.held_out_year,
            "rate_low_fraction": settings.rate_low_fraction,
            "rate_high_fraction": settings.rate_high_fraction,
            "bootstrap_resamples": settings.bootstrap_resamples,
            "bootstrap_level": settings.bootstrap_level,
        },
    )
    plot_genotype_variability(
        tables["summary"],
        tables["ratio_intervals"],
        output_dir / "genotype_variability",
    )
    logging.info(
        "wrote %s, settings.json and the figure to %s", ", ".join(tables), output_dir
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/calibration/genotype_variability")
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("results/calibration/height/params_pool.json"),
    )
    parser.add_argument("--num-genotypes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    run(arguments.output, arguments.parameters, arguments.num_genotypes, arguments.seed)


if __name__ == "__main__":
    main()
