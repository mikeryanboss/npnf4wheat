from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from npnf.scripts.paper.conditioning_sample_comparison import CONDITIONING_DIRS
from npnf.scripts.paper.fip1_conditioning_sample_comparison import (
    ObservedTruth,
    default_targets,
    model_roots,
    pooled_roots,
)

OBSERVED = {
    "plot-a": {"days": np.array([210, 220, 230]), "heights": np.array([0.1, 0.3, 0.5])}
}


def _batch(days):
    return {
        "grid_points": {"X": torch.tensor(days, dtype=torch.float32).unsqueeze(-1)},
        "data": TensorDict({"plot_uid": ["plot-a"]}, batch_size=[1]),  # ty: ignore[invalid-argument-type]
    }


def test_observed_truth_interpolates_inside_the_measured_range_only():
    days, rows = ObservedTruth(OBSERVED).rows(
        Path("unused"), _batch([200.0, 215.0, 230.0, 250.0])
    )

    assert days.tolist() == [200.0, 215.0, 230.0, 250.0]
    assert len(rows) == 1
    curve = rows[0][0]
    # A plot is one realisation; days it did not measure carry no height, so the
    # panel mean over plots uses only the plots measured on that day.
    assert np.isnan(curve[0])
    assert np.isnan(curve[3])
    assert np.isclose(curve[1], 0.2)
    assert curve[2] == 0.5


def test_default_targets_take_a_genotype_that_spans_the_most_site_years():
    conditions = [
        ("g-tall", "2016", "tall-16"),
        ("g-wide", "2016", "wide-16"),
        ("g-wide", "2017", "wide-17"),
    ]
    observed = {
        "tall-16": {"heights": np.array([2.0])},
        "wide-16": {"heights": np.array([0.8])},
        "wide-17": {"heights": np.array([0.9])},
    }

    # `g-tall` is the tallest, but it grows in one site-year, so the `g` row would
    # repeat the `g+e` row.
    assert default_targets(conditions, observed) == ("g-wide", "2017")


def test_model_roots_point_at_the_conditioning_parent_of_each_run(tmp_path):
    roots = model_roots(
        tmp_path, "test_environment", ["CNP", "ANP"], "pretrained", 2, "checkpoint-1m"
    )
    assert roots == [
        str(
            tmp_path
            / "fip1_test_environment_dataloaders"
            / f"{model}-FIP-2930-fip1_pretrained_1m_seed2"
            / "checkpoint-1m"
            / "test_environment"
        )
        for model in ("CNP", "ANP")
    ]


def test_pooled_roots_number_the_batches_of_every_split_in_one_sequence(tmp_path):
    split_roots = {}
    for split, num_batches in (("test_plot", 2), ("test_genotype", 1)):
        (root,) = model_roots(
            tmp_path / "results", split, ["CNP"], "pretrained", 2, "checkpoint-1m"
        )
        split_roots[split] = [root]
        for conditioning in CONDITIONING_DIRS:
            for number in range(num_batches):
                batch = Path(root) / conditioning / "max_height" / "predictions"
                (batch / f"{number:03d}").mkdir(parents=True)
                (batch / f"{number:03d}" / "split").write_text(split)

    (root,) = pooled_roots(split_roots, "max_height", tmp_path / "pooled")

    for conditioning in CONDITIONING_DIRS:
        batches = sorted(
            (Path(root) / conditioning / "max_height" / "predictions").iterdir()
        )
        assert [(batch / "split").read_text() for batch in batches] == [
            "test_plot",
            "test_plot",
            "test_genotype",
        ]
