from __future__ import annotations

import numpy as np
import pytest

from npnf.calibration.height.genotype_variability import (
    GenotypeVariabilitySettings,
    bootstrap_ratio_intervals,
    effective_replicates,
    elongation_rate,
    implied_fip1_residual_standard_deviation,
    pooled_within_genotype_standard_deviation,
    standard_deviation,
    summary_table,
)


def _fip1_year(
    blues: dict[str, float], targets: dict[str, list[float]], heritability: float = 0.98
) -> dict:
    plot_values = [value for values in targets.values() for value in values]
    counts = [len(values) for values in targets.values()]
    harmonic_mean = float(len(counts) / sum(1.0 / count for count in counts))
    # One cessation day per plot, 280 + 4 * genotype index, constant within a
    # genotype: the same shape as the rates below.
    growth_end = {
        genotype: [280.0 + 4.0 * index] * len(values)
        for index, (genotype, values) in enumerate(targets.items())
    }
    return {
        "heritability": heritability,
        "effective_replicates_max_height": harmonic_mean,
        "effective_replicates_elongation_rate": harmonic_mean,
        "effective_replicates_growth_end_day": harmonic_mean,
        "plot_max_height": plot_values,
        "blue": blues,
        "genotype_growth_end_day": growth_end,
        "genotype_max_height": targets,
        # Constant within a genotype, varying across them, as real rates are.
        "genotype_elongation_rate": {
            genotype: [0.015 + 0.001 * index] * len(values)
            for index, (genotype, values) in enumerate(targets.items())
        },
        "genotype_mean_elongation_rate": {
            genotype: 0.015 + 0.001 * index for index, genotype in enumerate(targets)
        },
    }


def test_implied_residual_inverts_the_heritability_definition():
    """var(BLUE) = varG + varEps / r, so the back-out must reproduce varEps."""
    blues = {
        f"G{index}": float(value)
        for index, value in enumerate(np.linspace(0.8, 1.0, 50))
    }
    # Unequal replication: r is the harmonic mean of 1, 2 and 3, not the nominal
    # two. FIP1 has single-plot genotypes in 2022 and triples in 2019.
    targets = {"G0": [0.90], "G1": [0.90, 0.91], "G2": [0.90, 0.91, 0.92]}
    fip1 = {2017: _fip1_year(blues, targets)}

    implied = implied_fip1_residual_standard_deviation(fip1)

    effective_replicates = fip1[2017]["effective_replicates_max_height"]
    assert effective_replicates == pytest.approx(18.0 / 11.0)
    variance_blue = float(np.var(list(blues.values()), ddof=1))
    heritability = 0.98
    expected = np.sqrt(effective_replicates * (1.0 - heritability) * variance_blue)
    assert implied[2017] == pytest.approx(expected)
    # A perfectly heritable trait leaves no residual.
    assert implied[2017] < np.sqrt(variance_blue)


def test_summary_table_noise_free_genotype_spread_and_dispersion_ratios():
    """sigma_G = sqrt(H2) * SD(BLUE); the ratio columns divide by what they name."""
    blues = {"G0": 0.80, "G1": 0.90, "G2": 1.00}
    targets = {"G0": [0.78, 0.82], "G1": [0.88, 0.92], "G2": [0.98, 1.02]}
    heritability = 0.81
    fip1 = {2018: _fip1_year(blues, targets, heritability)}
    simulated = np.array([0.80, 0.90, 1.00, 1.10])
    simulated_growth_end_days = np.array([280.0, 282.0, 284.0, 286.0])

    (row,) = summary_table(
        fip1,
        {2018: simulated},
        {2018: simulated_growth_end_days},
        {2018: np.full(4, 0.012)},
        {2018: 0.02},
    )

    noise_free = np.sqrt(heritability) * standard_deviation(list(blues.values()))
    assert row["fip1_max_height_genotype_standard_deviation"] == (
        pytest.approx(noise_free)
    )
    assert noise_free == pytest.approx(0.9 * 0.1)
    assert row["max_height_ratio"] == pytest.approx(
        standard_deviation(simulated) / noise_free
    )
    plot_targets = [value for values in targets.values() for value in values]
    assert row["fip1_max_height_plot_standard_deviation"] == pytest.approx(
        standard_deviation(plot_targets)
    )


def test_growth_end_ratio_removes_replicate_mean_error_with_unequal_cells():
    """The corrected real spread is var(genotype means) - var_within / r."""
    blues = {"G0": 0.80, "G1": 0.90, "G2": 1.00}
    # Cell sizes 1, 2 and 3, so r is their harmonic mean 18/11, not 2.
    targets = {"G0": [0.80], "G1": [0.90, 0.92], "G2": [1.00, 1.02, 1.04]}
    fip1 = {2018: _fip1_year(blues, targets)}
    growth_end = {"G0": [281.0], "G1": [283.0, 285.0], "G2": [287.0, 289.0, 291.0]}
    fip1[2018]["genotype_growth_end_day"] = growth_end
    simulated_growth_end_days = np.array([280.0, 282.0, 284.0, 286.0])

    (row,) = summary_table(
        fip1,
        {2018: np.array([0.80, 0.90, 1.00])},
        {2018: simulated_growth_end_days},
        {2018: np.full(3, 0.012)},
        {2018: 0.02},
    )

    means = [281.0, 284.0, 289.0]
    # Pooled within: cell variances 2.0 (weight 1) and 4.0 (weight 2).
    within_variance = (1 * 2.0 + 2 * 4.0) / 3
    effective_replicates = 18.0 / 11.0
    corrected = np.sqrt(np.var(means, ddof=1) - within_variance / effective_replicates)
    assert row["fip1_growth_end_day_genotype_standard_deviation"] == (
        pytest.approx(corrected)
    )
    assert row["growth_end_day_ratio"] == (
        pytest.approx(standard_deviation(simulated_growth_end_days) / corrected)
    )


def test_held_out_year_is_flagged():
    blues = {"G0": 0.85, "G1": 0.95}
    fip1 = {2019: _fip1_year(blues, {"G0": [0.85], "G1": [0.95]})}
    simulated = {2019: np.array([0.85, 0.95])}

    (row,) = summary_table(
        fip1,
        simulated,
        {2019: np.array([280.0, 284.0])},
        {2019: np.full(2, 0.012)},
        {2019: 0.02},
    )

    assert row["held_out"] is True


def test_elongation_rate_recovers_a_known_slope():
    """On a linear rise the window slope is the slope, whatever the window."""
    days = np.arange(101.0)
    heights = days / 100.0  # 0.01 m per day up to 1.0 m

    assert elongation_rate(days, heights) == pytest.approx(0.01)


def test_elongation_rate_is_nan_when_the_record_starts_too_late():
    """A crossing below the first measurement would have to be extrapolated."""
    days = np.arange(60.0, 101.0)
    # Starts at 60% of max, above the 40% lower threshold.
    heights = days / 100.0

    assert np.isnan(elongation_rate(days, heights))
    # The same trajectory observed from the start is fine.
    full_days = np.arange(101.0)
    assert np.isfinite(elongation_rate(full_days, full_days / 100.0))
    settings = GenotypeVariabilitySettings()
    assert settings.rate_low_fraction < 0.6 < settings.rate_high_fraction


def test_elongation_rate_ignores_the_post_peak_decline():
    """Lodging drops height after the peak; the rate must not see it."""
    rise = np.arange(0.0, 1.01, 0.01)
    heights = np.concatenate([rise, [0.3, 0.2]])
    days = np.arange(len(heights), dtype=float)

    assert elongation_rate(days, heights) == pytest.approx(0.01)


def test_effective_replicates_is_the_harmonic_mean_of_finite_counts():
    """NaN values do not count as plots, and empty genotypes are dropped."""
    by_genotype = {
        "one": [0.9],
        "two": [0.9, 1.0],
        "three": [0.9, float("nan"), 1.0, 1.1],
        "none": [float("nan")],
    }

    assert effective_replicates(by_genotype) == pytest.approx(18.0 / 11.0)


def test_pooled_within_genotype_standard_deviation_uses_only_replicated_cells():
    by_genotype = {
        "replicated_a": [1.0, 3.0],  # SS 2.0, df 1
        "replicated_b": [2.0, 4.0],  # SS 2.0, df 1
        "singleton": [7.0],  # ignored: no within-cell information
        "has_nan": [5.0, float("nan")],  # collapses to a singleton, ignored
    }

    pooled = pooled_within_genotype_standard_deviation(by_genotype)

    assert pooled == pytest.approx(np.sqrt(4.0 / 2.0))


def test_pooled_within_genotype_standard_deviation_matches_a_one_way_anova():
    """The pooled SD is the residual mean square of a one-way ANOVA over cells."""
    pandas = pytest.importorskip("pandas")
    formula_api = pytest.importorskip("statsmodels.formula.api")
    generator = np.random.default_rng(0)
    # Unbalanced on purpose: the weighting by plot count only matters when the
    # counts differ.
    by_genotype = {
        f"G{index}": list(generator.normal(loc=index, scale=0.5, size=2 + index % 3))
        for index in range(15)
    }

    pooled = pooled_within_genotype_standard_deviation(by_genotype)

    frame = pandas.DataFrame(
        [
            (genotype, value)
            for genotype, values in by_genotype.items()
            for value in values
        ],
        columns=["genotype", "value"],
    )
    model = formula_api.ols("value ~ C(genotype)", data=frame).fit()
    assert pooled == pytest.approx(np.sqrt(model.mse_resid))


def test_bootstrap_intervals_cover_the_point_ratio():
    """Resampling genotypes gives an interval around the full-sample ratio."""
    generator = np.random.default_rng(0)
    genotypes = [f"g{index}" for index in range(60)]
    effects = generator.normal(0.0, 1.0, size=len(genotypes))
    fip1 = {
        2018: {
            "blue": {
                genotype: 0.9 + 0.1 * effect
                for genotype, effect in zip(genotypes, effects, strict=True)
            },
            "heritability": 0.96,
            "genotype_growth_end_day": {
                genotype: list(290.0 + 4.0 * effect + generator.normal(0, 1, size=2))
                for genotype, effect in zip(genotypes, effects, strict=True)
            },
            "genotype_elongation_rate": {
                genotype: list(0.015 + 0.002 * effect + generator.normal(0, 5e-4, 2))
                for genotype, effect in zip(genotypes, effects, strict=True)
            },
        }
    }
    simulated_max_heights = {2018: 0.9 + 0.1 * generator.normal(size=200)}
    simulated_growth_end_days = {2018: 290.0 + 4.0 * generator.normal(size=200)}
    simulated_elongation_rates = {2018: 0.015 + 0.002 * generator.normal(size=200)}

    rows = bootstrap_ratio_intervals(
        fip1,
        simulated_max_heights,
        simulated_growth_end_days,
        simulated_elongation_rates,
        settings=GenotypeVariabilitySettings(bootstrap_resamples=200),
    )

    assert [row["trait"] for row in rows] == [
        "max_height",
        "growth_end_day",
        "elongation_rate",
    ]
    for row in rows:
        assert row["num_undefined"] == 0
        assert row["ratio_lower"] < 1.0 < row["ratio_upper"]
