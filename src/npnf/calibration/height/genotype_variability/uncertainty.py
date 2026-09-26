"""Bootstrap intervals for the simulator/real dispersion ratios."""

from __future__ import annotations

import numpy as np

from npnf.calibration.height.genotype_variability.constants import (
    GenotypeVariabilitySettings,
)
from npnf.calibration.height.genotype_variability.traits import (
    effective_replicates,
    noise_corrected_genotype_standard_deviation,
    standard_deviation,
)


def _corrected_standard_deviation(cells: list[list[float]]) -> float:
    """Noise-free genotype SD of resampled cells, with r recomputed from them."""
    by_genotype = {str(index): values for index, values in enumerate(cells)}
    return noise_corrected_genotype_standard_deviation(
        by_genotype, effective_replicates(by_genotype)
    )


def _finite_standard_deviation(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return standard_deviation(finite) if finite.size > 1 else float("nan")


def bootstrap_ratio_intervals(
    fip1: dict,
    simulated_max_heights: dict,
    simulated_growth_end_days: dict,
    simulated_elongation_rates: dict,
    seed: int = 0,
    settings: GenotypeVariabilitySettings | None = None,
) -> list[dict]:
    """Percentile intervals of the three dispersion ratios, per year.

    Resamples real genotypes (with all their plots) and simulated genotypes
    independently, with replacement, and recomputes each ratio. FIP1 H2 is held
    fixed. The intervals cover genotype sampling only, not the choice of panel.
    """
    settings = settings or GenotypeVariabilitySettings()
    num_resamples = settings.bootstrap_resamples
    generator = np.random.default_rng(seed)
    tail = 100 * (1 - settings.bootstrap_level) / 2
    rows = []
    for year in sorted(fip1):
        fip1_year = fip1[year]
        blues = np.array(list(fip1_year["blue"].values()))
        heritability_root = float(np.sqrt(fip1_year["heritability"]))
        growth_end_cells = list(fip1_year["genotype_growth_end_day"].values())
        rate_cells = list(fip1_year["genotype_elongation_rate"].values())
        simulated = {
            "max_height": np.asarray(simulated_max_heights[year], dtype=np.float64),
            "growth_end_day": np.asarray(
                simulated_growth_end_days[year], dtype=np.float64
            ),
            "elongation_rate": np.asarray(
                simulated_elongation_rates[year], dtype=np.float64
            ),
        }
        ratios: dict[str, list[float]] = {trait: [] for trait in simulated}
        for _ in range(num_resamples):
            fip1_deviation = {
                "max_height": heritability_root
                * standard_deviation(
                    blues[generator.integers(len(blues), size=len(blues))]
                ),
                "growth_end_day": _corrected_standard_deviation(
                    [
                        growth_end_cells[index]
                        for index in generator.integers(
                            len(growth_end_cells), size=len(growth_end_cells)
                        )
                    ]
                ),
                "elongation_rate": _corrected_standard_deviation(
                    [
                        rate_cells[index]
                        for index in generator.integers(
                            len(rate_cells), size=len(rate_cells)
                        )
                    ]
                ),
            }
            for trait, values in simulated.items():
                resampled = values[generator.integers(len(values), size=len(values))]
                ratios[trait].append(
                    _finite_standard_deviation(resampled) / fip1_deviation[trait]
                )
        for trait, values in ratios.items():
            finite = np.array([value for value in values if np.isfinite(value)])
            lower, upper = (
                np.percentile(finite, [tail, 100 - tail])
                if finite.size
                else (float("nan"), float("nan"))
            )
            rows.append(
                {
                    "year": year,
                    "trait": trait,
                    "ratio_lower": float(lower),
                    "ratio_upper": float(upper),
                    "num_undefined": num_resamples - int(finite.size),
                }
            )
    return rows
