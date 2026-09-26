"""Height calibration data loading utilities."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import datasets
import numpy as np
import torch
from scipy.interpolate import make_splrep

from npnf.calibration.height.constants import HeightDates
from npnf.data.fip1_day_grid import observed_on_common_days


@dataclass(frozen=True)
class HeightOptimizationConfig:
    """Configuration constants for height optimization."""

    optimized_param_names: tuple[str, ...] = (
        *(f"cp_{i}" for i in range(30)),
        "cp_sigma",
        "length_scale_T",
        "length_scale_tau",
        "tau_max_loc",
        "tau_max_scale",
    )
    nonneg_param_names: tuple[str, ...] = (
        *(f"cp_{i}" for i in range(30)),
        "cp_sigma",
        "length_scale_T",
        "length_scale_tau",
        "tau_max_loc",
        "tau_max_scale",
    )
    seed_list: tuple[int, ...] = (42, 123, 456, 0, 1, 2, 3)


@dataclass(frozen=True)
class FIP1YearData:
    """Per-year aggregated FIP1 data."""

    year_temps: dict[int, torch.Tensor]
    observed_heights: dict[int, torch.Tensor]
    observed_height_means: dict[int, float]
    observed_height_stds: dict[int, float]
    observed_growth_end: dict[int, torch.Tensor]
    observed_trajectories: dict[int, dict]
    trajectory_days: list[np.ndarray]
    trajectory_heights: list[np.ndarray]
    trajectory_years: list[int]


@dataclass(frozen=True)
class HeightOptimizationResults:
    """Diagnostics from height optimization (not part of HeightPoolParams)."""

    best_trial: int
    optimization_loss: float
    height_loss: float
    ordering_loss: float
    growth_end_loss: float
    ordering_slope: float
    mean_height: float
    finite_loss: float
    height_range_loss: float
    non_finite_trajectory_count: float
    final_height_peryear_loss: float
    sim_pooled_mean: float
    sim_pooled_std: float
    num_ordering_genotypes: int
    num_ordering_years: int
    num_trials: int
    num_genotypes: int
    mean_loss: float | None
    std_loss: float | None
    exclude_height_years: list[int]
    sim_height_means: dict[int, float]
    sim_height_stds: dict[int, float]
    trial_history: list[dict[str, Any]]
    optuna_params: dict[str, float]


def load_fip1_year_data(
    fip1_data: datasets.Dataset, *, dates: HeightDates
) -> FIP1YearData:
    """Load FIP1 data organized by year for Stage 4 calibration.

    Args:
        fip1_data: HuggingFace Dataset.

    Returns:
        FIP1YearData with per-year temperature, height, trajectory, and
        growth-end data.
    """
    all_temps = torch.stack(list(fip1_data["temperature_values"]))  # (N, 274, 24)
    all_years = torch.tensor(list(fip1_data["harvest_year"]))

    # Group temperatures by year and average
    unique_years, inverse = all_years.unique(return_inverse=True)
    year_sums = torch.zeros(len(unique_years), 274, 24, dtype=all_temps.dtype)
    year_sums.index_add_(0, inverse, all_temps)
    year_counts = torch.bincount(inverse).unsqueeze(1).unsqueeze(2)
    year_temps = {
        int(y): (year_sums[i] / year_counts[i]) for i, y in enumerate(unique_years)
    }

    # Pad heights and batch top-3 median
    padded = torch.nn.utils.rnn.pad_sequence(
        list(fip1_data["height_values"]), batch_first=True, padding_value=float("-inf")
    )
    top3_medians = torch.topk(padded, 3, dim=1).values.median(dim=1).values

    heights_by_year: dict[int, list[float]] = defaultdict(list)
    for year, median in zip(all_years.tolist(), top3_medians.tolist(), strict=True):
        heights_by_year[year].append(median)

    # Trajectory data (variable-length, kept as lists)
    trajectory_days = [d.to(torch.float64).numpy() for d in fip1_data["height_days"]]
    trajectory_heights = [
        h.to(torch.float64).numpy() for h in fip1_data["height_values"]
    ]
    trajectory_years = all_years.tolist()

    observed_trajectories = _build_trajectory_matrices(fip1_data, dates=dates)
    observed_growth_end = compute_growth_end_days(fip1_data)

    observed_heights = {
        year: torch.tensor(heights) for year, heights in heights_by_year.items()
    }
    observed_height_means = {
        year: float(np.mean(heights)) for year, heights in heights_by_year.items()
    }
    observed_height_stds = {
        year: float(np.std(heights)) for year, heights in heights_by_year.items()
    }

    return FIP1YearData(
        year_temps=year_temps,
        observed_heights=observed_heights,
        observed_height_means=observed_height_means,
        observed_height_stds=observed_height_stds,
        observed_growth_end=observed_growth_end,
        observed_trajectories=observed_trajectories,
        trajectory_days=trajectory_days,
        trajectory_heights=trajectory_heights,
        trajectory_years=trajectory_years,
    )


def _build_trajectory_matrices(
    fip1_data: datasets.Dataset, *, dates: HeightDates
) -> dict[int, dict]:
    """Build per-year observed trajectory matrices preserving per-plant identity.

    For each year, intersects measurement days across all plants to get common
    dates, then builds a (K_y, T_y) matrix of observed heights.

    Returns:
        {year: {"day_indices": Tensor (T_y,), "trajectories": Tensor (K_y, T_y)}}
        where day_indices are ints relative to TAU_START_DAY.
    """
    plants_by_year: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(list)
    for height_trajectory, day_trajectory, year in zip(
        fip1_data["height_values"],
        fip1_data["height_days"],
        fip1_data["harvest_year"],
        strict=True,
    ):
        in_window = (day_trajectory >= dates.tau_start_day) & (
            day_trajectory < dates.plant_growth_stop_max
        )
        if bool(in_window.any()):
            plants_by_year[int(year)].append(
                {"height_days": day_trajectory, "height_values": height_trajectory}
            )

    result: dict[int, dict] = {}
    for year_int, plants in plants_by_year.items():
        days, trajectories = observed_on_common_days(
            plants, window=(dates.tau_start_day, dates.plant_growth_stop_max)
        )
        if not days.numel():
            continue
        result[year_int] = {
            "day_indices": days - dates.tau_start_day,
            "trajectories": trajectories,
        }
    return result


def _detect_growth_end_day(
    days: np.ndarray,
    heights: np.ndarray,
    *,
    n_boundary_avg: int = 3,
    n_pad_points: int = 15,
    pad_spacing_days: float = 7.0,
    smoothing_factor: float = 5e-4,
    n_fine_points: int = 500,
    rate_threshold_frac: float = 0.25,
    height_gate_frac: float = 0.70,
) -> float | None:
    """Detect growth-end day for a single FIP1 trajectory.

    Fits a smoothing spline, takes the analytical derivative, and finds
    the first crossing of the rate below ``rate_threshold_frac`` of the
    max rate that occurs after the spline has reached
    ``height_gate_frac`` of its max height.  This ensures we detect the
    actual deceleration phase, not a spurious early bump.

    Args:
        days: Measurement days (Sep-1 based), must be sorted ascending.
        heights: Height measurements in meters (same order as days).
        n_boundary_avg: Number of edge measurements to average for padding.
        n_pad_points: Number of padding points on each side.
        pad_spacing_days: Spacing between padding points in days.
        smoothing_factor: Spline smoothing factor (per data point).
        n_fine_points: Number of points in the fine evaluation grid.
        rate_threshold_frac: Rate threshold as fraction of max rate.
        height_gate_frac: Height gate as fraction of max height.

    Returns:
        Growth-end day (Sep-1 based), or None if spline fit fails.
    """

    days = days.astype(np.float64)
    heights = heights.astype(np.float64)

    # Pad boundaries using mean of nearest measurements (robust to noise)
    pre_days = days[0] - pad_spacing_days * np.arange(n_pad_points, 0, -1)
    pre_heights = np.full(n_pad_points, heights[:n_boundary_avg].mean())
    post_days = days[-1] + pad_spacing_days * np.arange(1, n_pad_points + 1)
    post_heights = np.full(n_pad_points, heights[-n_boundary_avg:].mean())

    days_padded = np.concatenate([pre_days, days, post_days])
    heights_padded = np.concatenate([pre_heights, heights, post_heights])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spline = make_splrep(
            days_padded, heights_padded, s=len(days_padded) * smoothing_factor
        )
        if any("maxit" in str(w.message) for w in caught):
            return None

    fine_days = np.linspace(days[0], days[-1], n_fine_points)
    fine_heights = spline(fine_days)
    fine_rate = spline(fine_days, nu=1)

    # Restrict to region before max height (avoids post-plateau artifacts)
    max_height_idx = np.argmax(fine_heights)
    rate_region = fine_rate[: max_height_idx + 1]
    days_region = fine_days[: max_height_idx + 1]
    heights_region = fine_heights[: max_height_idx + 1]

    max_rate = rate_region.max() if len(rate_region) > 0 else fine_rate.max()

    # Find first index where spline reaches the height gate
    height_gate = height_gate_frac * fine_heights.max()
    past_gate = heights_region >= height_gate
    if not past_gate.any():
        return float(fine_days[max_height_idx])
    gate_idx = np.argmax(past_gate)

    # Find first crossing of rate below threshold after the height gate
    below = rate_region[gate_idx:] < rate_threshold_frac * max_rate
    crossings = np.where(below[:-1] != below[1:])[0]

    if len(crossings) >= 1:
        return float(days_region[gate_idx + crossings[0]])
    return float(fine_days[max_height_idx])


def compute_growth_end_days(fip1_data: datasets.Dataset) -> dict[int, torch.Tensor]:
    """Compute per-year growth-end day distributions from FIP1 trajectories.

    Args:
        fip1_data: HuggingFace Dataset.

    Returns:
        {year: tensor of growth-end days (Sep-1 based)} for each year.
    """
    growth_end_by_year: dict[int, list[float]] = defaultdict(list)

    for day_trajectory, height_trajectory, year in zip(
        fip1_data["height_days"],
        fip1_data["height_values"],
        fip1_data["harvest_year"],
        strict=True,
    ):
        day_values = day_trajectory.numpy()
        height_values = height_trajectory.numpy()

        growth_end_day = _detect_growth_end_day(day_values, height_values)
        if growth_end_day is not None:
            growth_end_by_year[int(year)].append(growth_end_day)

    return {
        year: torch.tensor(days, dtype=torch.float32)
        for year, days in growth_end_by_year.items()
    }


__all__ = [
    "FIP1YearData",
    "HeightOptimizationConfig",
    "HeightOptimizationResults",
    "load_fip1_year_data",
]
