# ruff: noqa: SLF001
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from npnf.scripts.paper import lodging_probability as lp
from npnf.scripts.paper.data_types import BinnedRates, GroundTruth, PredictionStats


def rates(name: str, values: list[float], single_height: float | None = None):
    counts = np.full(len(values), 100)
    lodged = np.asarray(values) * counts
    return BinnedRates(
        name=name,
        rates=np.asarray(values),
        counts=counts,
        lodged_counts=lodged,
        single_height=single_height,
    )


def keep_figure(monkeypatch, module) -> list[plt.Figure]:
    figures: list[plt.Figure] = []
    monkeypatch.setattr(
        module, "_save_plot", lambda figure, _path: figures.append(figure)
    )
    return figures


def test_load_and_bin_model_records_a_single_predicted_height(monkeypatch) -> None:
    heights = np.full(6, 0.84)
    monkeypatch.setattr(
        lp,
        "_load_predictions",
        lambda *_arguments: PredictionStats(
            max_height=heights,
            final_height=heights,
            drop_abs=np.zeros(6),
            drop_rel=np.zeros(6),
            is_lodged=np.zeros(6, dtype=bool),
        ),
    )
    model = lp.ModelConfig("CNP", Path("prior"), Path("context"), color=(0, 0, 0))
    bins = np.array([0.8, 0.85, 0.9])

    ground_truth = GroundTruth(labels=np.zeros(6, dtype=bool), max_heights=heights)

    prior = lp._load_and_bin_model(model, "no_context", ground_truth, bins, 0.2, 0.1)

    assert prior is not None
    assert prior.single_height == 0.84


def test_semantic_lodging_plot_draws_single_heights_and_full_y_range(
    monkeypatch,
) -> None:
    figures = keep_figure(monkeypatch, lp)
    centers = np.array([0.8, 0.9, 1.0])
    ground_truth = rates("ground truth", [0.1, 0.3, 0.6])
    prior = lp.PanelData(
        "no context",
        [
            rates("CNP", [0.0, 0.0, 0.0], single_height=0.84),
            rates("LNP", [0.1, 0.2, 0.5]),
        ],
        ground_truth,
    )
    context = lp.PanelData(
        "max-height context",
        [rates("CNP", [0.0, 0.2, 1.0]), rates("LNP", [0.1, 0.3, 0.5])],
        ground_truth,
    )

    lp._plot_semantic_lodging_rates(
        centers, prior, context, Path("unused"), ["CNP", "LNP"], min_sample_fraction=0.0
    )

    axes = figures[0].axes
    points = [line for line in axes[0].lines if line.get_marker() == "o"]
    assert len(points) == 1
    np.testing.assert_allclose(np.asarray(points[0].get_xdata(), dtype=float), [0.84])
    np.testing.assert_allclose(np.asarray(points[0].get_ydata(), dtype=float), [0.0])
    assert axes[1].get_ylim()[1] >= 100.0
