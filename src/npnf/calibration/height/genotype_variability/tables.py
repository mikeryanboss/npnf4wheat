"""Comparison tables of the genotype-variability analysis."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from npnf.calibration.height.genotype_variability.traits import (
    noise_corrected_genotype_standard_deviation,
    standard_deviation,
)
from npnf.data.datasets.fip1 import Fip1Facts


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, or nan when the denominator is unusable."""
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def summary_table(
    fip1: dict,
    simulated_max_heights: dict,
    simulated_growth_end_days: dict,
    simulated_elongation_rates: dict,
    implied_residual: dict[int, float],
) -> list[dict]:
    """Per-year spread of max height, growth-cessation day and elongation rate.

    ``fip1_*_genotype_standard_deviation`` is the FIP1 genotype SD with the plot
    noise removed, and ``*_ratio`` divides the simulated SD by it. Heights are in
    metres, days are counted from 1 September and rates are in metres per day.
    """
    rows = []
    for year in sorted(fip1):
        fip1_year = fip1[year]
        blues = list(fip1_year["blue"].values())
        # H2 = varG / var(BLUE), so sqrt(H2) * SD(BLUE) is the noise-free
        # genotype SD, comparable with the noiseless pool.
        genotype_deviation = float(np.sqrt(fip1_year["heritability"])) * (
            standard_deviation(blues)
        )
        # The spread the calibration matched, for comparison with the above.
        plot_deviation = standard_deviation(fip1_year["plot_max_height"])
        simulated_deviation = standard_deviation(simulated_max_heights[year])

        # The pool has no replicate plots, so the real counterpart is the spread
        # of genotype means with the replicate error removed.
        growth_end_deviation = noise_corrected_genotype_standard_deviation(
            fip1_year["genotype_growth_end_day"],
            fip1_year["effective_replicates_growth_end_day"],
        )
        simulated_growth_end_deviation = standard_deviation(
            simulated_growth_end_days[year]
        )
        rate_deviation = noise_corrected_genotype_standard_deviation(
            fip1_year["genotype_elongation_rate"],
            fip1_year["effective_replicates_elongation_rate"],
        )
        simulated_rate = simulated_elongation_rates[year][
            np.isfinite(simulated_elongation_rates[year])
        ]
        simulated_rate_deviation = standard_deviation(simulated_rate)

        rows.append(
            {
                "year": year,
                "held_out": year == Fip1Facts().held_out_year,
                "num_genotypes": len(blues),
                "fip1_max_height_genotype_standard_deviation": genotype_deviation,
                "fip1_max_height_plot_standard_deviation": plot_deviation,
                "simulated_max_height_standard_deviation": simulated_deviation,
                "max_height_ratio": _ratio(simulated_deviation, genotype_deviation),
                "fip1_heritability": fip1_year["heritability"],
                "fip1_effective_replicates": fip1_year[
                    "effective_replicates_max_height"
                ],
                "fip1_implied_residual_standard_deviation": implied_residual[year],
                "fip1_growth_end_day_genotype_standard_deviation": (
                    growth_end_deviation
                ),
                "simulated_growth_end_day_standard_deviation": (
                    simulated_growth_end_deviation
                ),
                "growth_end_day_ratio": (
                    _ratio(simulated_growth_end_deviation, growth_end_deviation)
                ),
                "fip1_elongation_rate_mean": float(
                    np.mean(list(fip1_year["genotype_mean_elongation_rate"].values()))
                ),
                "simulated_elongation_rate_mean": float(np.mean(simulated_rate)),
                "fip1_elongation_rate_genotype_standard_deviation": (rate_deviation),
                "simulated_elongation_rate_standard_deviation": (
                    simulated_rate_deviation
                ),
                "elongation_rate_ratio": _ratio(
                    simulated_rate_deviation, rate_deviation
                ),
            }
        )
    return rows


def implied_fip1_residual_standard_deviation(fip1: dict) -> dict[int, float]:
    """Back out the FIP1 plot-level residual SD from the BLUE spread and H2.

    var(BLUE) = varG + varEps / r and H2 = varG / var(BLUE), so
    varEps = r * (1 - H2) * var(BLUE), with r the effective replication. The
    residual holds all variation the spatial model did not remove, so it is
    larger than the observation noise alone.

    The BLUEs and H2 come from two fits that differ in the genotype term, so the
    result is an approximation.
    """
    implied = {}
    for year, fip1_year in fip1.items():
        variance_blue = float(np.var(list(fip1_year["blue"].values()), ddof=1))
        implied[year] = float(
            np.sqrt(
                fip1_year["effective_replicates_max_height"]
                * (1.0 - fip1_year["heritability"])
                * variance_blue
            )
        )
    return implied


def _mean_error(fip1: dict, implied_residual: dict[int, float], year: int) -> float:
    """SD of the error of a genotype mean over the effective FIP1 plots."""
    return implied_residual[year] / np.sqrt(
        fip1[year]["effective_replicates_max_height"]
    )


def year_pair_table(
    fip1: dict,
    simulated_max_heights: dict,
    seed: int,
    implied_residual: dict[int, float],
) -> list[dict]:
    """Genotype rank correlation between year pairs, real and simulated.

    Each simulated value gets the error of a mean over the effective number of
    FIP1 plots of that year, so that it is as precise as a FIP1 BLUE.
    """
    generator = np.random.default_rng(seed)
    years = sorted(fip1)
    rows = []
    for position, year_a in enumerate(years):
        for year_b in years[position + 1 :]:
            blue_a, blue_b = fip1[year_a]["blue"], fip1[year_b]["blue"]
            shared = sorted(set(blue_a) & set(blue_b))
            simulated_a = simulated_max_heights[year_a]
            simulated_b = simulated_max_heights[year_b]
            index = generator.choice(len(simulated_a), size=len(shared), replace=False)
            clean_a, clean_b = simulated_a[index], simulated_b[index]
            mean_a = clean_a + generator.normal(
                0, _mean_error(fip1, implied_residual, year_a), size=clean_a.shape
            )
            mean_b = clean_b + generator.normal(
                0, _mean_error(fip1, implied_residual, year_b), size=clean_b.shape
            )

            rows.append(
                {
                    "year_a": year_a,
                    "year_b": year_b,
                    "num_shared_genotypes": len(shared),
                    "fip1_blue_spearman": float(
                        spearmanr(
                            [blue_a[genotype] for genotype in shared],
                            [blue_b[genotype] for genotype in shared],
                        ).statistic
                    ),
                    "simulated_spearman_noise_matched": float(
                        spearmanr(mean_a, mean_b).statistic
                    ),
                }
            )
    return rows
