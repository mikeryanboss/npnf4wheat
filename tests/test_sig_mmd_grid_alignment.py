from __future__ import annotations

import pytest
import torch

from npnf.data.fip1_day_grid import align_trajectories_to_day_axis


def test_align_trajectories_to_day_axis_gathers_exact_days():
    trajectories = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]]])
    source_day_axis = torch.tensor([61.0, 137.0, 199.0, 204.0])
    target_day_axis = torch.tensor([61.0, 199.0, 204.0])

    aligned = align_trajectories_to_day_axis(
        trajectories, source_day_axis, target_day_axis
    )

    expected = torch.tensor([[[1.0, 3.0, 4.0], [11.0, 13.0, 14.0]]])
    assert torch.equal(aligned, expected)


def test_align_trajectories_to_day_axis_raises_on_non_exact_days():
    trajectories = torch.zeros(1, 1, 4)
    source_day_axis = torch.tensor([61.0, 137.0, 199.0, 204.0])
    target_day_axis = torch.tensor([61.0, 150.0])

    with pytest.raises(ValueError, match="exact subset"):
        align_trajectories_to_day_axis(trajectories, source_day_axis, target_day_axis)
