"""Settings for the genotype-variability analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenotypeVariabilitySettings:
    """Fixed choices of the genotype-variability analysis.

    ``rate_low_fraction``, ``rate_high_fraction``: elongation-rate window as
    fractions of each trajectory's maximum height. The lower bound is 0.40
    because the 2016 records start with plots at up to 37% of final height.

    ``bootstrap_resamples``, ``bootstrap_level``: number of genotype resamples
    and coverage of the percentile intervals.
    """

    rate_low_fraction: float = 0.40
    rate_high_fraction: float = 0.80
    bootstrap_resamples: int = 1000
    bootstrap_level: float = 0.95
