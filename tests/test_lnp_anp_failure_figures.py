from __future__ import annotations

import numpy as np
import pytest

from npnf.scripts.paper import anp_prior_failure as anp


def test_observed_means_separate_the_schedule_from_the_growth() -> None:
    # Two plots on the same line h = day / 10, measured on alternate days, one
    # of them 1 m higher: the per-day mean jumps, the interpolated mean does not.
    low = (np.arange(0, 11, 2), np.arange(0, 11, 2) / 10)
    high = (np.arange(1, 10, 2), np.arange(1, 10, 2) / 10 + 1.0)
    days = np.arange(1, 10)

    per_day, interp = anp.observed_means([low, high], days, min_plots=1)

    np.testing.assert_allclose(per_day[::2], days[::2] / 10 + 1.0)
    np.testing.assert_allclose(per_day[1::2], days[1::2] / 10)
    np.testing.assert_allclose(interp, days / 10 + 0.5)
    assert anp.roughness(interp) == pytest.approx(0.0)
    assert anp.roughness(per_day) == pytest.approx(2.0)


def test_observed_means_mask_days_with_too_few_plots() -> None:
    plot = (np.array([0, 2]), np.array([0.0, 2.0]))

    per_day, interp = anp.observed_means([plot], np.array([0, 1, 3]), min_plots=1)

    np.testing.assert_allclose(per_day, [0.0, np.nan, np.nan])
    np.testing.assert_allclose(interp, [0.0, 1.0, np.nan])


def test_schedule_correlation_is_one_for_a_model_following_the_per_day_mean() -> None:
    interp = np.linspace(0.0, 1.0, 6)
    per_day = interp + np.array([0.1, -0.1, 0.2, -0.2, 0.1, np.nan])

    assert anp.schedule_correlation(0.5 * per_day + 0.5 * interp, per_day, interp) == (
        pytest.approx(1.0)
    )
