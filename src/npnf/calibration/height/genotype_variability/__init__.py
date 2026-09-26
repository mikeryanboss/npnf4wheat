"""Genotype-variability validation of the calibrated height generator.

Compares the genotype-level spread of the calibrated pool with FIP1 on four
axes: maximum height, growth-cessation day, elongation rate and rank stability
across years.
"""

from .constants import GenotypeVariabilitySettings
from .data import fip1_traits_by_year, load_fip1_metadata
from .simulate import simulate_pool_traits
from .tables import (
    implied_fip1_residual_standard_deviation,
    summary_table,
    year_pair_table,
)
from .traits import (
    effective_replicates,
    elongation_rate,
    noise_corrected_genotype_standard_deviation,
    pooled_within_genotype_standard_deviation,
    standard_deviation,
)
from .uncertainty import bootstrap_ratio_intervals

__all__ = [
    "GenotypeVariabilitySettings",
    "bootstrap_ratio_intervals",
    "effective_replicates",
    "elongation_rate",
    "fip1_traits_by_year",
    "implied_fip1_residual_standard_deviation",
    "load_fip1_metadata",
    "noise_corrected_genotype_standard_deviation",
    "pooled_within_genotype_standard_deviation",
    "simulate_pool_traits",
    "standard_deviation",
    "summary_table",
    "year_pair_table",
]
