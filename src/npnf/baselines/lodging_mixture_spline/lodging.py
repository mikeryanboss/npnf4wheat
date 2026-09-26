"""Lodging drops: detection from the heights alone and the drop geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from npnf.baselines.lodging_mixture_spline.utils import running_mean
from npnf.calibration.height.constants import HeightDates


@dataclass(frozen=True)
class Detection:
    lodged: torch.Tensor  # (N,) bool
    mask: torch.Tensor  # (N, D) bool, observations on or after the lodging day
    day: torch.Tensor  # (N,) lodging day, meaningful where ``lodged``


@dataclass(frozen=True)
class DropDetector:
    """Changepoint detector: for every growth day, the mean of the ``window``
    observations before the day minus the mean of all observations from the day
    on (post-season points included). A trajectory is lodged when the largest
    contrast exceeds ``threshold``, on the day of the largest contrast."""

    threshold: float = 0.1
    window: int = 7
    min_observations_after: int = 5
    start_day: int = HeightDates.tau_start_day
    end_day: int = HeightDates.plant_growth_stop_max

    def detect(
        self,
        days: torch.Tensor,
        values: torch.Tensor,
        observed: torch.Tensor | None = None,
    ) -> Detection:
        """Detect drops in ``values`` (N, D) on ``days`` (D,), any order. Only the
        entries ``observed`` (N, D) count, so rows can have their own days."""
        order, sorted_days, count = sort_observations(days, observed, values.shape)
        ordered = values.gather(1, order)
        position = torch.arange(len(days), device=values.device)
        present = position < count[:, None]
        cumulative = F.pad(
            torch.where(present, ordered, 0.0).double().cumsum(dim=1), (1, 0)
        )
        before = (
            cumulative[:, position] - cumulative[:, (position - self.window).clamp(0)]
        ) / self.window
        count_after = count[:, None] - position
        after = (cumulative.gather(1, count[:, None]) - cumulative[:, position]) / (
            count_after.clamp(min=1)
        )
        valid = (
            (sorted_days >= self.start_day)
            & (sorted_days < self.end_day)
            & (count_after >= self.min_observations_after)
            & (position >= self.window)
        )
        contrast = torch.where(valid, (before - after).float(), -torch.inf)
        best, best_position = contrast.max(dim=1)
        lodged = best > self.threshold
        day = sorted_days.gather(1, best_position[:, None])[:, 0]
        mask = lodged[:, None] & (days.float()[None] >= day[:, None])
        if observed is not None:
            mask &= observed
        return Detection(lodged, mask, day)


@dataclass(frozen=True)
class DropGeometry:
    """A drop starts ``delay`` days after the end of growth and falls linearly
    over ``transition`` days to ``severity`` times the plateau height."""

    final_offset_days: int = 25
    transition_fraction: float = 0.1
    max_transition_days: int = 30
    min_severity: float = 0.05
    plateau_tolerance: float = 0.01

    def clip(self, triples: torch.Tensor) -> torch.Tensor:
        """Clip (delay, severity, transition) triples (M, 3) to their ranges."""
        low = triples.new_tensor([0.0, self.min_severity, 0.0])
        high = triples.new_tensor([torch.inf, 1.0, self.max_transition_days])
        return triples.clamp(low, high)

    def extract(
        self,
        days: torch.Tensor,
        values: torch.Tensor,
        lodging_day: torch.Tensor,
        plateau_height: torch.Tensor,
        growth_end: torch.Tensor,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(M, 3) triples of the lodged trajectories ``values`` (M, D), of which
        only the entries ``observed`` (M, D) count.

        The final level is the mean from ``final_offset_days`` after the lodging
        day on; trajectories without such observations are left out. The
        transition ends where the running mean first falls within
        ``transition_fraction`` of the drop above the final level.
        """
        order, sorted_days, count = sort_observations(days, observed, values.shape)
        ordered = values.gather(1, order)
        present = torch.arange(len(days), device=values.device) < count[:, None]
        late = present & (
            sorted_days >= (lodging_day + self.final_offset_days)[:, None]
        )
        final = (ordered * late).sum(dim=1) / late.sum(dim=1).clamp(min=1)
        level = final + self.transition_fraction * (plateau_height - final)
        reached = (
            present
            & (sorted_days >= lodging_day[:, None])
            & (running_mean(ordered, count=count) <= level[:, None])
        )
        first = reached.float().argmax(dim=1)
        transition = torch.where(
            reached.any(dim=1),
            sorted_days.gather(1, first[:, None])[:, 0] - lodging_day,
            self.max_transition_days,
        )
        triples = torch.stack(
            [lodging_day - growth_end, final / plateau_height, transition], dim=1
        )
        return self.clip(triples[late.any(dim=1)])

    def growth_end(
        self, daily_heights: torch.Tensor, days: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plateau of smooth curves ``daily_heights`` (..., D) on consecutive
        ``days``: the first day within ``plateau_tolerance`` of the maximum (the
        day growth stops), and the maximum. Drop delays count from this day."""
        plateau_height = daily_heights.max(dim=-1).values
        on_plateau = (
            daily_heights >= (plateau_height - self.plateau_tolerance)[..., None]
        )
        first = on_plateau.float().argmax(dim=-1)
        return days.to(daily_heights.device)[first].float(), plateau_height


def observed_peak(
    values: torch.Tensor, excluded: torch.Tensor, observed: torch.Tensor | None = None
) -> torch.Tensor:
    """Peak of the 3-observation running mean of ``values`` (N, D) over the
    entries ``observed`` (N, D) that are not ``excluded`` (for example after a
    lodging drop). Without ``observed``, the columns are observations in order."""
    if observed is None:
        peak = running_mean(values).masked_fill(excluded, -1.0).max(dim=-1).values
        return peak.clamp(min=0)
    columns = torch.arange(values.shape[1], device=values.device)
    order, _sorted, count = sort_observations(columns, observed, values.shape)
    smooth = running_mean(values.gather(1, order), count=count)
    hidden = excluded.gather(1, order) | (columns >= count[:, None])
    return smooth.masked_fill(hidden, -1.0).max(dim=-1).values.clamp(min=0)


def sort_observations(
    days: torch.Tensor, observed: torch.Tensor | None, shape: torch.Size
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row order that puts the ``observed`` entries of rows with ``shape``
    (N, D) first, sorted by ``days`` (D,). Returns the order (N, D), the sorted
    days (N, D, ``inf`` after the observations) and the counts (N,)."""
    if observed is None:
        observed = torch.ones(shape, dtype=torch.bool, device=days.device)
    key = torch.where(observed, days.float()[None].to(observed.device), torch.inf)
    sorted_days, order = key.sort(dim=1, stable=True)
    return order, sorted_days, observed.sum(dim=1)
