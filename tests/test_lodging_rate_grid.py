from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from npnf.scripts.paper import lodging_rate_grid as grid
from npnf.scripts.paper.data_types import PredictionStats
from npnf.scripts.paper.lodging_absolute_drop import DropGroundTruth


def paper_model_names() -> list[str]:
    return [
        "CNP",
        "ACNP",
        *(
            f"{base}{variant}"
            for base in ("LNP", "ANP")
            for variant in ("", "-NF-Prior", "-NF-Posterior", "-NF-Prior-Posterior")
        ),
    ]


def test_group_folders_splits_rate_major_folders() -> None:
    rates = grid.group_folders(
        ["a/CNP", "a/LNP", "b/CNP", "b/LNP"], ["CNP", "LNP"], [1.1, 1.5], ["hi", "lo"]
    )

    assert [rate.label for rate in rates] == ["hi", "lo"]
    assert [rate.lodging_weibull_scale for rate in rates] == [1.1, 1.5]
    assert rates[1].folders == {"CNP": Path("b/CNP"), "LNP": Path("b/LNP")}


def test_group_folders_rejects_mismatched_counts() -> None:
    with pytest.raises(ValueError, match=r"base_folders=3.*rate_labels=2"):
        grid.group_folders(["a", "b", "c"], ["CNP", "LNP"], [1.1, 1.5], ["hi", "lo"])


def test_group_folders_rejects_unknown_model_names() -> None:
    with pytest.raises(ValueError, match="Unknown paper model names"):
        grid.group_folders(["a"], ["GP"], [1.1], ["hi"])


def test_binned_curves_draws_a_single_height_at_that_height() -> None:
    rate, severity = grid.binned_curves(
        heights=np.full(4, 0.842),
        is_lodged=np.array([False, False, False, False]),
        drop_abs=np.zeros(4),
        bins=np.array([0.8, 0.85, 0.9]),
        min_sample_fraction=0.01,
        min_lodged_count=1,
    )

    np.testing.assert_allclose(rate.heights, [0.842])
    np.testing.assert_allclose(rate.values, [0.0])
    np.testing.assert_allclose(severity.heights, [0.842])
    assert np.isnan(severity.values).all()


def test_binned_curves_masks_sparse_bins() -> None:
    rate, severity = grid.binned_curves(
        heights=np.array([0.7, 0.71, 0.72, 0.73, 0.95]),
        is_lodged=np.array([True, False, True, True, True]),
        drop_abs=np.array([0.2, 0.0, 0.4, 0.6, 0.5]),
        bins=np.array([0.6, 0.8, 1.0]),
        min_sample_fraction=0.3,
        min_lodged_count=2,
    )

    np.testing.assert_allclose(rate.heights, [0.7, 0.9])
    np.testing.assert_allclose(rate.values[0], 75.0)
    assert np.isnan(rate.values[1])
    np.testing.assert_allclose(severity.values[0], 0.4)
    assert np.isnan(severity.values[1])


def test_calculate_writes_figures_and_summary(monkeypatch, tmp_path) -> None:
    generator = np.random.default_rng(0)
    sample_count = 400
    max_heights = generator.uniform(0.6, 1.2, sample_count)

    def fake_ground_truth(path: Path) -> DropGroundTruth:
        lodged = max_heights > (1.0 if "low" in str(path) else 0.9)
        return DropGroundTruth(
            labels=lodged, max_heights=max_heights, drop_abs=0.4 * lodged
        )

    def fake_predictions(path: Path, *_thresholds: float) -> PredictionStats:
        deterministic = path.parent.name in ("CNP", "ACNP")
        if deterministic and path.name == "no_context":
            heights = np.full(sample_count, 0.84)
            lodged = np.zeros(sample_count, dtype=bool)
        else:
            heights = max_heights
            lodged = max_heights > 0.95
        return PredictionStats(
            max_height=heights,
            final_height=heights,
            drop_abs=0.3 * lodged,
            drop_rel=np.zeros(sample_count),
            is_lodged=lodged,
        )

    monkeypatch.setattr(grid, "_load_ground_truth_drops", fake_ground_truth)
    monkeypatch.setattr(grid, "_load_predictions", fake_predictions)

    model_names = paper_model_names()
    output = tmp_path / "out"
    grid.calculate_lodging_rate_grid(
        base_folders=[
            f"{rate}/{name}" for rate in ("high", "low") for name in model_names
        ],
        model_names=model_names,
        lodging_weibull_scales=[1.1, 1.5],
        rate_labels=["19.7 %", "2.7 %"],
        output_folder=str(output),
    )

    for stem in ("lodging_rate_grid", "lodging_severity_grid"):
        assert (output / f"{stem}.png").is_file()
        assert (output / f"{stem}.pdf").is_file()
    summary = pl.read_csv(output / "lodging_rate_grid_summary.csv")
    assert summary.height == 2 * (1 + 2 * len(model_names))
    cnp = summary.filter((pl.col("line") == "CNP") & (pl.col("method") == "no_context"))
    assert cnp["lodged_percent"].to_list() == [0.0, 0.0]
    assert cnp["mean_drop_lodged"].null_count() == 2
    curves = pl.read_csv(output / "lodging_rate_grid_curves.csv")
    single = curves.filter(
        (pl.col("line") == "CNP")
        & (pl.col("method") == "no_context")
        & (pl.col("quantity") == "rate")
    )
    assert single["max_height"].to_list() == pytest.approx([0.84, 0.84])
