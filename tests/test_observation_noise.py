"""Scientific contracts for the unequal-grid effective-variation estimator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from npnf.calibration.height.evaluation.observation_noise import (
    normalized_triplet_residuals,
)
from npnf.scripts.calibration.observation_noise import estimate_pooled_noise_std


def test_unequal_grid_annihilates_linear_signal_and_preserves_noise_variance() -> None:
    days = np.array([0.0, 2.0, 7.0])
    signal = np.array([0.3, 0.7, 1.7])
    z, prediction, indices = normalized_triplet_residuals(days, signal[None])
    np.testing.assert_allclose(z, 0, atol=1e-15)
    np.testing.assert_allclose(prediction, [[0.7]])
    np.testing.assert_array_equal(indices, [1])
    # Responses to independent unit perturbations are the noise coefficients.
    responses, _, _ = normalized_triplet_residuals(days, np.eye(3))
    a, b = 5 / 7, 2 / 7
    np.testing.assert_allclose(
        responses[:, 0], np.array([-a, 1, -b]) / np.sqrt(1 + a * a + b * b)
    )
    np.testing.assert_allclose(np.sum(responses**2), 1)


def test_linear_signal_recovers_known_three_centimetre_noise() -> None:
    days = np.array([0, 2, 7, 8, 15, 18, 22], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(471))
    heights = 0.2 + 0.005 * days + 0.03 * rng.standard_normal((20000, len(days)))
    z, _, _ = normalized_triplet_residuals(days, heights)
    assert np.sqrt(np.mean(z**2)) == pytest.approx(0.03, rel=0.02)


def test_gap_limit_includes_boundary_but_excludes_long_neighbors() -> None:
    days = np.array([0, 7, 14, 22, 23])
    heights = np.array([[0.0, 0.2, 0.4, 3.0, 4.0]])
    _, _, indices = normalized_triplet_residuals(days, heights, max_gap=7)
    np.testing.assert_array_equal(indices, [1])
    _, _, unrestricted = normalized_triplet_residuals(days, heights, max_gap=np.inf)
    np.testing.assert_array_equal(unrestricted, [1, 2, 3])


def test_estimation_uses_dense_all_plot_residuals_and_triplet_weighting(
    tmp_path: Path,
) -> None:
    years = (2016, 2017, 2018, 2019, 2021, 2022)
    input_dir = tmp_path / "lodging"
    output_dir = tmp_path / "noise"
    input_dir.mkdir()
    output_dir.mkdir()
    squared_sum = 0.0
    triplets = 0
    plots = 0
    for index, year in enumerate(years, start=1):
        amplitude = 0.01 * index
        real = np.tile([0.0, amplitude, 0.0, amplitude, 0.0], (index, 1))
        np.savez(
            input_dir / f"aligned_{year}.npz",
            days=np.array([210, 220, 230]),
            real=np.zeros((index, 3)),
            dense_days=np.array([210, 215, 220, 225, 230]),
            dense_real=real,
        )
        squared_sum += index * 3 * amplitude**2 / 1.5
        triplets += index * 3
        plots += index
    sigma = estimate_pooled_noise_std(input_dir, output_dir, years)
    assert sigma == pytest.approx(np.sqrt(squared_sum / triplets))
    pooled = json.loads((output_dir / "estimation_settings.json").read_text())["pooled"]
    assert pooled["n_plots"] == plots
    assert pooled["n_triplets"] == triplets
    assert pooled["sigma_rms_m"] == sigma
