"""The day grid that every FIP1 distributional comparison is scored on.

FIP1's site-years have different measurement schedules and share no measurement
date, so scores from different test sets are comparable only on one grid of dates.
The anchors are the dates that every site-year can serve; each site-year contributes
its nearest selected measurement date. The scored height stays that plot's real
measurement, with only its time coordinate moved to the anchor.

The grid is a property of the measurement schedule, not of a metric, so nothing here
imports the metric library. It currently gives 12 anchors, day 223 to 302, over the
seven test site-years, at 0.0166 m rms distortion.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

import datasets
import numpy as np
import torch
from torch import Tensor

from npnf.data.datasets.fip1 import Fip1Facts, get_heights_dataset

METRIC_WINDOW = (200, 322)
DAY_TOLERANCE = 3
MIN_ANCHOR_GAP = 3


@dataclass(frozen=True)
class SharedDayGrid:
    """Anchor dates and the measurement dates selected for each site-year."""

    anchors: np.ndarray
    assigned: dict[str, np.ndarray]


def shared_days(
    plot_days: list[np.ndarray], *, window: tuple[int, int] = METRIC_WINDOW
) -> np.ndarray:
    """Measurement dates common to all plots, inside the half-open ``window``."""
    shared = set(plot_days[0].tolist())
    for days in plot_days[1:]:
        shared &= set(days.tolist())
    days = np.array(sorted(shared), dtype=int)
    return days[(days >= window[0]) & (days < window[1])]


def align_days(
    yearsite_days: dict[str, np.ndarray],
    tolerance: int = DAY_TOLERANCE,
    min_gap: int = MIN_ANCHOR_GAP,
) -> SharedDayGrid:
    """Match the site-years onto one grid, within ``tolerance`` days.

    Works in stages of widening slack, 0 first and ``tolerance`` last, so exact
    matches are always taken before any date moves. Within a stage it walks
    candidate anchors from the earliest to the latest and keeps one when every
    site-year still has an unused measurement date within the stage's slack and no
    accepted anchor is closer than ``min_gap``. Each site-year then selects its
    closest such date, so no measurement date is used twice and no anchor is reused.
    """
    remaining = {
        yearsite: sorted(int(day) for day in days)
        for yearsite, days in yearsite_days.items()
    }
    first = min(days[0] for days in remaining.values())
    last = max(days[-1] for days in remaining.values())
    matched: dict[int, dict[str, int]] = {}
    for slack in range(tolerance + 1):
        for anchor in range(first, last + 1):
            if any(abs(anchor - taken) < min_gap for taken in matched):
                continue
            picked = {}
            for yearsite, days in remaining.items():
                near = [day for day in days if abs(day - anchor) <= slack]
                if not near:
                    picked = {}
                    break
                picked[yearsite] = min(near, key=lambda day: (abs(day - anchor), day))
            if not picked:
                continue
            for yearsite, day in picked.items():
                remaining[yearsite].remove(day)
            matched[anchor] = picked
    anchors = sorted(matched)
    return SharedDayGrid(
        anchors=np.array(anchors, dtype=int),
        assigned={
            yearsite: np.array(
                [matched[anchor][yearsite] for anchor in anchors], dtype=int
            )
            for yearsite in remaining
        },
    )


def select_year(
    dataset: datasets.Dataset, year: int, grid: SharedDayGrid
) -> datasets.Dataset:
    """Retain plots with all selected measurement dates, in original row order."""
    return dataset.filter(
        lambda harvest_year, site, days: (
            int(harvest_year) == year
            and np.isin(grid.assigned[str(site)], np.asarray(days)).all()
        ),
        input_columns=["harvest_year", "yearsite_uid", "height_days"],
    )


def align_trajectories_to_day_axis(
    trajectories: torch.Tensor,
    source_day_axis: torch.Tensor,
    target_day_axis: torch.Tensor,
) -> torch.Tensor:
    """Gather trajectories at target days, which must all lie exactly on source days."""
    source_day_axis = source_day_axis.to(dtype=torch.float32)
    target_day_axis = target_day_axis.to(dtype=torch.float32)

    indices = torch.searchsorted(source_day_axis, target_day_axis)
    in_bounds = indices < source_day_axis.numel()
    clamped = indices.clamp(max=source_day_axis.numel() - 1)
    exact = in_bounds & torch.isclose(source_day_axis[clamped], target_day_axis)
    if not bool(exact.all()):
        n_missing = int((~exact).sum())
        n_total = target_day_axis.numel()
        msg = (
            f"target_day_axis must be an exact subset of source_day_axis; "
            f"{n_missing}/{n_total} target days do not match."
        )
        raise ValueError(msg)
    return trajectories.index_select(2, clamped.to(trajectories.device))


def observed_at_grid(
    rows: Iterable[Mapping[str, object]], grid: SharedDayGrid
) -> Tensor:
    """Keep measured values at each site's selected measurement dates, in row order."""
    return torch.stack(
        [
            align_trajectories_to_day_axis(
                cast(Tensor, row["height_values"])[None, None, :],
                cast(Tensor, row["height_days"]),
                torch.from_numpy(grid.assigned[str(row["yearsite_uid"])]),
            )[0, 0]
            for row in rows
        ]
    )


def observed_on_common_days(
    rows: Iterable[Mapping[str, object]], *, window: tuple[int, int] = METRIC_WINDOW
) -> tuple[Tensor, Tensor]:
    """Return common measurement dates and heights in row order, without interpolation.

    Dates use the dataset's original time coordinates. Keep the first measurement
    on repeated dates, including when a row's dates are not sorted.
    """
    plants: list[dict[int, float]] = []
    for row in rows:
        plant: dict[int, float] = {}
        for day, height in zip(
            cast(Tensor, row["height_days"]).tolist(),
            cast(Tensor, row["height_values"]).tolist(),
            strict=False,
        ):
            if window[0] <= day < window[1]:
                plant.setdefault(int(day), float(height))
        plants.append(plant)
    days = shared_days(
        [np.fromiter(plant, dtype=int) for plant in plants], window=window
    )
    heights = torch.tensor(
        [[plant[day] for day in days] for plant in plants], dtype=torch.float32
    )
    return torch.from_numpy(days), heights


@functools.lru_cache(maxsize=1)
def fip1_day_grid(datasets_offline_path: str | None = None) -> SharedDayGrid:
    """The grid every FIP1 comparison is scored on, over all test site-years."""
    plot_days: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in Fip1Facts().test_splits:
        dataset = get_heights_dataset(
            split=split, datasets_offline_path=datasets_offline_path
        )
        for sample in dataset:
            plot_days[str(sample["yearsite_uid"])].append(
                sample["height_days"].numpy().astype(int)
            )
    return align_days(
        {yearsite: shared_days(days) for yearsite, days in sorted(plot_days.items())}
    )
