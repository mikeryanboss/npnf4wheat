"""Height calibration objective function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.forward import _run_bspline_forward_fast
from npnf.data.pools import GenotypePool
from npnf.data.synthetic.height.params import height_pool_params_from_trial_params
from npnf.data.synthetic.height.response_surface import BSplineLUT


def _negative_distance_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative distance kernel used for the calibration MMD objective."""
    return -torch.cdist(x, y, p=2)


def _mmd_squared(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute MMD^2 without depending on the deleted metrics module."""
    k_xx = _negative_distance_kernel(x, x).mean()
    k_yy = _negative_distance_kernel(y, y).mean()
    k_xy = _negative_distance_kernel(x, y).mean()
    return float(k_xx + k_yy - 2 * k_xy)


@dataclass(frozen=True)
class EvalContext:
    """Evaluation context for height calibration — constant across an Optuna study."""

    years: list[int]
    year_temps: dict[int, torch.Tensor]
    observed_trajectories: dict[int, dict]
    observed_heights: dict[int, torch.Tensor]
    observed_growth_end: dict[int, torch.Tensor]
    num_ordering_genotypes: int
    exclude_height_years: set[int]
    ordering_temps: torch.Tensor
    ordering_gs_temps: torch.Tensor
    tau_lut: BSplineLUT
    dates: HeightDates


def _simulate_heights(
    context: EvalContext,
    trial_params: dict[str, float],
    seed: int,
    num_genotypes: int,
    precomputed: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample genotypes and run the B-spline forward model.

    Returns:
        Tuple of (sim_heights, sim_trajectory_by_year, pool_params) where:
        - sim_heights: shape (num_genotypes, num_years)
        - sim_trajectory_by_year: shape (num_genotypes, num_years, growth_period)
        - pool_params: genotype parameter tensor on device
    """
    num_years = len(context.years)
    height_pool_params = height_pool_params_from_trial_params(trial_params)
    pool = GenotypePool.sample(
        num_genotypes=num_genotypes, seed=seed, height_pool_params=height_pool_params
    )

    hourly_temps = precomputed["hourly_temps"]
    temperature_basis = precomputed["B_T"]
    device = hourly_temps.device

    params = pool.params.to(device)
    params_expanded = params.repeat_interleave(num_years, dim=0)
    total_batch = num_genotypes * num_years

    trajectories = _run_bspline_forward_fast(
        hourly_temps[:total_batch],
        params_expanded,
        temperature_basis[:total_batch],
        context.tau_lut,
        dates=context.dates,
    )

    heights = trajectories.max(dim=-1).values
    sim_heights = heights.reshape(num_genotypes, num_years)

    growing_period = trajectories[
        :, context.dates.tau_start_idx : context.dates.growing_period_end_idx
    ]
    sim_trajectory_by_year = growing_period.reshape(
        num_genotypes, num_years, context.dates.maximum_growth_period
    )

    return sim_heights, sim_trajectory_by_year, params


def _compute_final_height_loss(
    context: EvalContext, sim_heights: torch.Tensor
) -> tuple[float, float, float]:
    """Compute per-year MMD² between simulated and observed final heights.

    Returns:
        Tuple of (per_year_loss, pooled_mean, pooled_std).
    """
    year_to_idx = {y: i for i, y in enumerate(context.years)}
    pooled_parts: list[torch.Tensor] = []
    per_year_terms: list[float] = []

    for year, observed_heights_year in context.observed_heights.items():
        if year not in year_to_idx:
            continue
        if context.exclude_height_years and year in context.exclude_height_years:
            continue
        idx = year_to_idx[year]
        simulated_heights_year = sim_heights[:, idx].detach().cpu()
        per_year_terms.append(
            _mmd_squared(
                simulated_heights_year.unsqueeze(-1),
                observed_heights_year.unsqueeze(-1).float(),
            )
        )
        pooled_parts.append(simulated_heights_year)

    per_year_loss = float(np.mean(per_year_terms)) if per_year_terms else 0.0
    pooled = torch.cat(pooled_parts) if pooled_parts else torch.zeros(1)
    return per_year_loss, float(pooled.mean()), float(pooled.std())


def _compute_trajectory_loss(
    context: EvalContext, sim_trajectory_by_year: torch.Tensor
) -> float:
    """Compute multivariate trajectory MMD loss per year."""
    terms: list[float] = []
    for i, year in enumerate(context.years):
        if context.exclude_height_years and year in context.exclude_height_years:
            continue
        if year in context.observed_trajectories:
            observed = context.observed_trajectories[year]
            day_indices = observed["day_indices"]
            observed_trajectory = observed["trajectories"]
            simulated_subset = sim_trajectory_by_year[:, i, day_indices].detach().cpu()
            terms.append(_mmd_squared(simulated_subset, observed_trajectory))

    return float(np.mean(terms)) if terms else 0.0


def _compute_growth_end_loss(
    context: EvalContext, sim_trajectory_by_year: torch.Tensor
) -> float:
    """Compute growth-end timing loss (MMD on growth cessation day)."""
    year_to_idx = {y: i for i, y in enumerate(context.years)}
    sim_growth_end_idx = sim_trajectory_by_year.argmax(dim=-1)
    sim_growth_end_day = (context.dates.tau_start_day + sim_growth_end_idx).float()

    terms: list[float] = []
    for year, observed_growth_end_year in context.observed_growth_end.items():
        if year not in year_to_idx:
            continue
        idx = year_to_idx[year]
        sim_growth_end_year = sim_growth_end_day[:, idx].detach().cpu()
        per_year_std = max(float(observed_growth_end_year.std()), 0.5)
        terms.append(
            _mmd_squared(
                sim_growth_end_year.unsqueeze(-1),
                observed_growth_end_year.unsqueeze(-1).float(),
            )
            / per_year_std
        )

    return float(np.mean(terms)) if terms else 0.0


def _compute_ordering_loss(
    context: EvalContext,
    pool_params: torch.Tensor,
    num_genotypes: int,
    precomputed: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute ordering loss: warmer years should produce shorter plants.

    Also computes guardrail losses (finite trajectories, height range).
    """
    num_ordering_years = context.ordering_temps.shape[0]
    num_ordering_genotypes = min(context.num_ordering_genotypes, num_genotypes)

    ordering_params = pool_params[:num_ordering_genotypes]
    ordering_params_expanded = ordering_params.repeat_interleave(
        num_ordering_years, dim=0
    )
    ordering_batch_size = num_ordering_genotypes * num_ordering_years

    ordering_hourly_temps = precomputed["ordering_hourly_temps"][:ordering_batch_size]
    ordering_temperature_basis = precomputed["ordering_B_T"][:ordering_batch_size]

    ordering_trajectories = _run_bspline_forward_fast(
        ordering_hourly_temps,
        ordering_params_expanded,
        ordering_temperature_basis,
        context.tau_lut,
        dates=context.dates,
    )

    ordering_max_heights = ordering_trajectories.max(dim=-1).values
    ordering_heights = ordering_max_heights.reshape(
        num_ordering_genotypes, num_ordering_years
    )

    # Slope of mean height vs growing-season temperature (should be negative)
    ordering_mean_height = ordering_heights.mean(dim=0).detach().cpu().numpy()
    ordering_gs_temps = context.ordering_gs_temps.cpu().numpy()
    slope = np.polyfit(ordering_gs_temps, ordering_mean_height, 1)[0]
    margin = 0.005
    ordering_loss = float(max(0, slope + margin))

    # Guardrails
    finite_mask = torch.isfinite(ordering_trajectories)
    non_finite_count = int((~finite_mask).sum().item())
    finite_loss = 1000.0 * float(non_finite_count)

    ordering_mean = float(ordering_heights.mean())
    height_range_violation = max(0.0, 0.6 - ordering_mean) + max(
        0.0, ordering_mean - 1.2
    )
    height_range_loss = 20.0 * height_range_violation

    return {
        "ordering_loss": ordering_loss,
        "ordering_slope": float(slope),
        "finite_loss": finite_loss,
        "height_range_loss": float(height_range_loss),
        "non_finite_count": float(non_finite_count),
    }


_ZERO_ORDERING = {
    "ordering_loss": 0.0,
    "ordering_slope": 0.0,
    "finite_loss": 0.0,
    "height_range_loss": 0.0,
    "non_finite_count": 0.0,
}


class SeedDiagnostics(TypedDict):
    """Per-seed loss components and simulated height statistics."""

    height_loss: float
    ordering_loss: float
    growth_end_loss: float
    ordering_slope: float
    finite_loss: float
    height_range_loss: float
    non_finite_trajectory_count: float
    mean_height: float
    final_height_peryear_loss: float
    sim_pooled_mean: float
    sim_pooled_std: float
    sim_year_means: dict[int, float]
    sim_year_stds: dict[int, float]


def _evaluate_single_seed(
    context: EvalContext,
    trial_params: dict[str, float],
    seed: int,
    num_genotypes: int,
    precomputed: dict[str, torch.Tensor],
    *,
    skip_ordering: bool = False,
) -> tuple[float, SeedDiagnostics]:
    """Evaluate height calibration objective for a single seed."""
    sim_heights, sim_trajectory_by_year, pool_params = _simulate_heights(
        context, trial_params, seed, num_genotypes, precomputed
    )

    final_height_loss, pooled_mean, pooled_std = _compute_final_height_loss(
        context, sim_heights
    )
    trajectory_loss = _compute_trajectory_loss(context, sim_trajectory_by_year)
    growth_end_loss = _compute_growth_end_loss(context, sim_trajectory_by_year)

    if skip_ordering:
        ordering = _ZERO_ORDERING
    else:
        ordering = _compute_ordering_loss(
            context, pool_params, num_genotypes, precomputed
        )

    total_loss = (
        trajectory_loss
        + ordering["ordering_loss"]
        + growth_end_loss
        + ordering["finite_loss"]
        + ordering["height_range_loss"]
        + final_height_loss
    )

    year_means = sim_heights.mean(dim=0).detach().cpu().numpy()
    year_stds = sim_heights.std(dim=0).detach().cpu().numpy()

    diagnostics: SeedDiagnostics = {
        "height_loss": trajectory_loss,
        "ordering_loss": ordering["ordering_loss"],
        "growth_end_loss": growth_end_loss,
        "ordering_slope": ordering["ordering_slope"],
        "finite_loss": ordering["finite_loss"],
        "height_range_loss": ordering["height_range_loss"],
        "non_finite_trajectory_count": ordering["non_finite_count"],
        "mean_height": float(sim_heights.mean()),
        "final_height_peryear_loss": final_height_loss,
        "sim_pooled_mean": pooled_mean,
        "sim_pooled_std": pooled_std,
        "sim_year_means": {
            year: float(year_means[i]) for i, year in enumerate(context.years)
        },
        "sim_year_stds": {
            year: float(year_stds[i]) for i, year in enumerate(context.years)
        },
    }

    return total_loss, diagnostics
