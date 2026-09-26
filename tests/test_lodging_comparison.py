"""Scientific boundaries for empirical lodging rates."""

from __future__ import annotations

import numpy as np
import torch

from npnf.calibration.height.evaluation.lodging_comparison import (
    DetectorOptions,
    detect_lodged_plots,
    fit_weibull_scale,
)


def test_pooled_scale_matches_observed_fraction_with_clamp_and_offset() -> None:
    observed_max_height = np.array([0.2, 0.6, 0.9, 1.3, 1.8, 2.5])
    target, shape = 2, 7.0
    scale = fit_weibull_scale(
        observed_max_height, target, shape=shape, height_clamp=1.5, offset=0.4
    )
    # Mixed observed maxima include a zero-probability plot and two clamped plots.
    effective_height = np.array([0.0, 0.2, 0.5, 0.9, 1.1, 1.1])
    probabilities = -np.expm1(-((effective_height / scale) ** shape))
    np.testing.assert_allclose(
        probabilities.mean(), target / len(observed_max_height), rtol=1e-10
    )


def test_detection_uses_full_schedule_before_common_window() -> None:
    detector: DetectorOptions = {
        "relative_threshold": 0.20,
        "absolute_threshold": 0.10,
        "tail_window": 5,
    }
    sim_days = torch.arange(200, 331)
    curves = torch.ones(2, len(sim_days))
    curves[0, -1] = 0.5
    schedules = [torch.tensor([200, 210, 220, 330]), torch.tensor([200, 220, 230, 330])]
    observed = [
        (days, curve[days - sim_days[0]])
        for curve, days in zip(curves, schedules, strict=True)
    ]
    np.testing.assert_array_equal(
        detect_lodged_plots(observed, detector), [True, False]
    )
    days = torch.tensor([200, 220])
    aligned = curves[:, days - sim_days[0]]
    np.testing.assert_array_equal(aligned, np.ones((2, 2)))
    assert not detect_lodged_plots(
        [(days, values) for values in aligned], detector
    ).any()
