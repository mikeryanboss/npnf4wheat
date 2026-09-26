import numpy as np
import pytest

from npnf.scripts.metrics.fip1_observed_plots import (
    context_weights,
    reconstruct_context,
    split_from_dataloader_name,
)


def plot(days, heights):
    return {"days": np.array(days), "heights": np.array(heights, dtype=float)}


def test_split_from_dataloader_name_covers_every_fip1_split():
    assert split_from_dataloader_name("fip1_test_plot_dataloaders") == "test_plot"
    assert (
        split_from_dataloader_name("fip1_test_genotype_environment_dataloaders")
        == "test_genotype_environment"
    )


def test_context_weights_are_uniform_without_context():
    pool = [plot([1, 2], [0.1, 0.2]), plot([1, 2], [0.9, 1.0])]
    weights = context_weights(pool, np.array([]), np.array([]), sigma=0.05)
    assert weights.tolist() == [0.5, 0.5]


def test_context_weights_concentrate_on_the_matching_plot():
    pool = [plot([1, 2], [0.10, 0.20]), plot([1, 2], [0.90, 1.00])]
    weights = context_weights(
        pool, np.array([1, 2]), np.array([0.11, 0.21]), sigma=0.05
    )
    assert weights[0] > 0.99
    assert weights.sum() == pytest.approx(1.0)


def test_context_weights_ignore_days_a_candidate_never_measured():
    # The candidate measured only day 1, so day 5 must not contribute a residual.
    pool = [plot([1], [0.10]), plot([1], [0.90])]
    weights = context_weights(
        pool, np.array([1, 5]), np.array([0.10, 0.50]), sigma=0.05
    )
    assert weights[0] > weights[1]


def test_context_weights_stay_finite_without_shared_days():
    # A candidate from another harvest year shares no measurement day with the
    # context. Exact-day matching gave it zero log-likelihood, i.e. the highest
    # weight after normalisation; interpolation must rank it below a real match.
    matching = plot([200, 210, 220], [0.10, 0.20, 0.30])
    other_year = plot([203, 213, 223], [0.80, 0.90, 1.00])
    weights = context_weights(
        [matching, other_year], np.array([200, 210]), np.array([0.10, 0.20]), 0.02
    )
    assert weights[0] > 0.99
    assert weights.sum() == pytest.approx(1.0)


def test_reconstruct_context_no_context_is_empty():
    days, values = reconstruct_context("no_context", {}, 0, plot([1, 2], [0.1, 0.2]))
    assert days.size == 0
    assert values.size == 0


def test_reconstruct_context_random_context_reads_saved_indices():
    days, values = reconstruct_context(
        "random_context",
        {"context_indices": [[0, 2]]},
        0,
        plot([200, 210, 220], [0.1, 0.5, 0.9]),
    )
    assert days.tolist() == [200, 220]
    assert values.tolist() == [0.1, 0.9]


def test_reconstruct_context_max_height_stops_at_the_robust_peak():
    # Robust peak is the median of the top three (0.60 at index 3), not the 2.0 spike.
    target = plot([200, 210, 220, 230, 240], [0.1, 0.5, 2.0, 0.6, 0.55])
    days, values = reconstruct_context("max_height", {}, 0, target)
    assert days.tolist() == [200, 210, 220, 230]
    assert values.tolist() == [0.1, 0.5, 2.0, 0.6]
