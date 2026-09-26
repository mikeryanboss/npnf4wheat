"""Per-trajectory and pooled statistics shared by the genotype-variability tables."""

from __future__ import annotations

import numpy as np
import scipy.stats

from npnf.calibration.height.genotype_variability.constants import (
    GenotypeVariabilitySettings,
)


def standard_deviation(values: list[float] | np.ndarray) -> float:
    """Sample SD (ddof=1) as a plain float, for the table builders."""
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def elongation_rate(
    days: np.ndarray,
    heights: np.ndarray,
    settings: GenotypeVariabilitySettings | None = None,
) -> float:
    """Mean height gain per day between two fractions of the maximum height.

    Returns metres per day, or nan when the record starts above the lower
    fraction or the trajectory never rises. The window is relative to each
    trajectory's own maximum, so rates are comparable across years.
    """
    settings = settings or GenotypeVariabilitySettings()
    low_fraction = settings.rate_low_fraction
    high_fraction = settings.rate_high_fraction
    peak = int(np.argmax(heights))
    if peak == 0:
        return float("nan")
    rise_days = days[: peak + 1].astype(np.float64)
    # Enforce monotonicity so np.interp can invert height -> day through noise.
    rise_heights = np.maximum.accumulate(heights[: peak + 1].astype(np.float64))
    max_height = rise_heights[-1]
    if not np.isfinite(max_height) or max_height <= 0:
        return float("nan")
    if rise_heights[0] > low_fraction * max_height:
        return float("nan")
    day_low, day_high = np.interp(
        [low_fraction * max_height, high_fraction * max_height], rise_heights, rise_days
    )
    if day_high <= day_low:
        return float("nan")
    return float((high_fraction - low_fraction) * max_height / (day_high - day_low))


def effective_replicates(by_genotype: dict[str, list[float]]) -> float:
    """Harmonic mean of the number of finite plot values per genotype.

    A genotype mean over k plots has noise variance varEps / k, so the spread of
    genotype means carries varEps * mean(1 / k) = varEps / r with this r.
    Genotypes without a finite value are dropped.
    """
    counts = [
        count
        for values in by_genotype.values()
        if (count := sum(1 for value in values if np.isfinite(value))) > 0
    ]
    return float(scipy.stats.hmean(counts)) if counts else float("nan")


def pooled_within_genotype_standard_deviation(
    by_genotype: dict[str, list[float]],
) -> float:
    """Pooled within-genotype SD over cells with at least two finite values.

    Each cell variance is weighted by its count minus one, which gives the
    residual mean square of a one-way ANOVA over cells. Returns nan when no cell
    has two finite values.
    """
    variances = []
    weights = []
    for values in by_genotype.values():
        finite = np.asarray(
            [value for value in values if np.isfinite(value)], dtype=np.float64
        )
        if finite.size < 2:
            continue
        variances.append(np.var(finite, ddof=1))
        weights.append(finite.size - 1)
    if not weights:
        return float("nan")
    return float(np.sqrt(np.average(variances, weights=weights)))


def noise_corrected_genotype_standard_deviation(
    by_genotype: dict[str, list[float]], effective_replicates: float
) -> float:
    """Genotype SD with the plot noise removed.

    var(genotype means) = varG + var_within / r, with r the harmonic mean of the
    plot counts, so varG = var(genotype means) - var_within / r.
    """
    means = [
        float(np.mean(finite))
        for values in by_genotype.values()
        if (finite := [value for value in values if np.isfinite(value)])
    ]
    mean_deviation = standard_deviation(means) if len(means) > 1 else float("nan")
    within_deviation = pooled_within_genotype_standard_deviation(by_genotype)
    variance = mean_deviation**2
    if np.isfinite(within_deviation):
        variance -= within_deviation**2 / effective_replicates
    return float(np.sqrt(variance)) if variance > 0 else float("nan")
