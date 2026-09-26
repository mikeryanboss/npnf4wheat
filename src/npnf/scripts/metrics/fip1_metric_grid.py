"""Helpers of the FIP1 Sig-MMD protocol that put heights on the day grid.

The grid itself is ``npnf.data.fip1_day_grid``: one set of anchor dates for every
FIP1 test set, because the harvest years share no measurement day and scores on
different grids are not comparable. The scoring lives in
``sig_mmd_fip1_context_matched_blocked.py``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from npnf.data.fip1_day_grid import METRIC_WINDOW, shared_days
from npnf.metrics.blocks import UnitMember
from npnf.metrics.signature import SYNTHETIC_SIGNATURE_HEIGHT_SCALE, to_metric_paths


def unit_yearsite_days(
    members: list[UnitMember], records: list[dict]
) -> dict[str, np.ndarray]:
    """Metric-window days shared by this unit's plots, per yearsite.

    Plots of one harvest year share a flight schedule, plots of different years
    generally share no measurement day at all, so the day set is per yearsite. A
    yearsite contributing fewer than two days is dropped.
    """
    by_yearsite: dict[str, list[dict]] = defaultdict(list)
    for member in members:
        by_yearsite[member.yearsite_uid].append(records[member.condition_index])
    days = {}
    for yearsite, group in by_yearsite.items():
        shared = shared_days([plot["days"] for plot in group])
        if shared.size >= 2:
            days[yearsite] = shared
    return days


def day_offsets(anchors: np.ndarray, assigned: dict[str, np.ndarray]) -> np.ndarray:
    """How far each assigned measurement day sits from its anchor, in days."""
    return np.concatenate(
        [np.abs(assigned[yearsite] - anchors) for yearsite in sorted(assigned)]
    )


def heights_at_days(plot: dict, days: np.ndarray) -> np.ndarray:
    """Observed heights of one plot at ``days``, which it must have measured."""
    positions = np.searchsorted(plot["days"], days)
    return plot["heights"][positions]


def model_curves_at_days(
    grid: torch.Tensor, day_axis: np.ndarray, days: np.ndarray
) -> torch.Tensor:
    """Sample saved grid predictions ``(S, G)`` at ``days``."""
    positions = np.searchsorted(day_axis, days)
    return grid[:, torch.as_tensor(positions, dtype=torch.long)]


def metric_paths(heights: np.ndarray | torch.Tensor, days: np.ndarray) -> torch.Tensor:
    """Lead-lag metric paths of ``heights`` sampled at ``days``.

    As on the synthetic side, where the oracle day axis maps to [0, 1], the scored
    window (the half-open ``METRIC_WINDOW``) maps to [0, 1].
    """
    first, stop = METRIC_WINDOW
    t_norm = torch.tensor((days - first) / (stop - 1 - first)).float()
    return to_metric_paths(
        torch.as_tensor(heights).float(), SYNTHETIC_SIGNATURE_HEIGHT_SCALE, t_norm
    )
