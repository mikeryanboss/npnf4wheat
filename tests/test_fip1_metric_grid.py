import numpy as np
import pytest
import torch

from npnf.metrics.blocks import UnitMember
from npnf.scripts.metrics.fip1_metric_grid import (
    day_offsets,
    heights_at_days,
    metric_paths,
    model_curves_at_days,
    unit_yearsite_days,
)


def plot(days, heights):
    return {"days": np.array(days), "heights": np.array(heights, dtype=float)}


def members_of(records):
    return [
        UnitMember(genotype_id="g", yearsite_uid=record["ys"], condition_index=index)
        for index, record in enumerate(records)
    ]


def test_heights_at_days_selects_the_matching_observations():
    assert heights_at_days(
        plot([200, 210, 220], [0.1, 0.5, 0.9]), np.array([210, 220])
    ).tolist() == [0.5, 0.9]


def test_model_curves_at_days_indexes_the_saved_grid():
    grid = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    day_axis = np.array([61, 62, 63, 64, 65, 66])
    sampled = model_curves_at_days(grid, day_axis, np.array([62, 65]))
    assert sampled.tolist() == [[1.0, 4.0], [7.0, 10.0]]


def test_metric_paths_appends_the_normalised_day_channel():
    days = np.array([200, 261, 321])
    paths = metric_paths(np.zeros((2, 3)), days)
    assert paths.shape[0] == 2
    # `to_metric_paths` applies the lead-lag transform and carries the day as a
    # third channel, normalised to [0, 1] over the metric window.
    assert paths.shape[-1] == 3
    assert paths[0, :, -1].min() == pytest.approx(0.0)
    assert paths[0, :, -1].max() == pytest.approx(1.0, abs=1e-2)


def test_unit_yearsite_days_drops_yearsites_without_two_shared_days():
    records = [
        {"days": np.array([210, 220, 230]), "heights": np.zeros(3), "ys": "A"},
        {"days": np.array([210, 220, 240]), "heights": np.zeros(3), "ys": "A"},
        {"days": np.array([250, 260]), "heights": np.zeros(2), "ys": "B"},
        {"days": np.array([250, 270]), "heights": np.zeros(2), "ys": "B"},
    ]
    days = unit_yearsite_days(members_of(records), records)
    assert days["A"].tolist() == [210, 220]
    assert "B" not in days  # only day 250 is shared, and one day is not a path


def test_day_offsets_measure_each_assigned_day_against_its_anchor():
    anchors = np.array([210, 220])
    assigned = {"B": np.array([211, 222]), "A": np.array([210, 219])}
    assert day_offsets(anchors, assigned).tolist() == [0, 1, 1, 2]
