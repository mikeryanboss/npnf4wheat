"""B-spline forward model for height calibration."""

import torch

from npnf.calibration.height.constants import HeightDates
from npnf.data.synthetic.height.genotype import GenotypeParams
from npnf.data.synthetic.height.response_surface import (
    DEFAULT_DEGREE,
    DEFAULT_T_KNOTS,
    DEFAULT_TAU_KNOTS,
    BSplineLUT,
    compute_bspline_basis,
    compute_growth_from_basis,
)


def _run_bspline_forward(
    hourly_temps: torch.Tensor, params: torch.Tensor, *, dates: HeightDates
) -> torch.Tensor:
    """Run B-spline surface forward model on hourly temperatures.

    Evaluates the response surface at each of 24 hourly temperatures per day,
    then sums to get daily growth (mm/day). Thermal time (tau) uses daily means.

    Args:
        hourly_temps: Hourly temperatures, shape (batch_size, 274, 24)
        params: Genotype parameters, shape (batch_size, 37)

    Returns:
        Height trajectories in meters, shape (batch_size, 274); zero before day 200
    """
    batch_size = hourly_temps.shape[0]
    field_names = GenotypeParams.field_names()
    param_dict = {name: params[:, i] for i, name in enumerate(field_names)}

    # Slice temperatures from day 200 onwards (photoperiod/vernalization gate)
    T_hourly_active = hourly_temps[:, dates.tau_start_idx :, :]  # (batch_size, 135, 24)
    days_active = T_hourly_active.shape[1]
    T_flat = T_hourly_active.reshape(batch_size, -1)  # (batch_size, 135*24)

    # Thermal time accumulation at hourly resolution
    tau_hourly = T_flat.clamp(min=0).cumsum(dim=-1)
    tau_max = param_dict["tau_max"]
    tau_norm_hourly = (tau_hourly / tau_max.unsqueeze(-1)).clamp(max=1.0)

    # Control points; enforce R(T, τ=0) = 0 via τ=0 column fixed at zero
    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    cps = GenotypeParams.build_clamped_control_grid(params, n_T_basis, n_tau_basis)

    # Compute B_T and B_tau bases at hourly resolution
    T_clamped = T_flat.clamp(DEFAULT_T_KNOTS[0], DEFAULT_T_KNOTS[-1])
    B_T = compute_bspline_basis(T_clamped.reshape(-1), DEFAULT_T_KNOTS, DEFAULT_DEGREE)
    B_T = B_T.reshape(batch_size, days_active, 24, n_T_basis)

    tau_clamped = tau_norm_hourly.clamp(
        float(DEFAULT_TAU_KNOTS[0]), float(DEFAULT_TAU_KNOTS[-1])
    )
    B_tau = compute_bspline_basis(
        tau_clamped.reshape(-1), DEFAULT_TAU_KNOTS, DEFAULT_DEGREE
    )
    B_tau = B_tau.reshape(batch_size, days_active, 24, n_tau_basis)

    # Shared einsum + clamp + maturity gate + daily sum
    tau_norm_3d = tau_norm_hourly.reshape(batch_size, days_active, 24)
    growth = compute_growth_from_basis(B_T, B_tau, cps, tau_norm_3d)
    growth[:, dates.maximum_growth_period :] = 0.0

    heights_active = growth.cumsum(dim=-1) / 1000.0

    # Zero-pad back to full 274-day window
    heights = torch.zeros(batch_size, dates.total_days)
    heights[:, dates.tau_start_idx :] = heights_active
    return heights


@torch.inference_mode()
def _precompute_t_basis(
    hourly_temps: torch.Tensor,
    T_knots: torch.Tensor = DEFAULT_T_KNOTS,
    degree: int = DEFAULT_DEGREE,
    *,
    dates: HeightDates,
) -> torch.Tensor:
    """Precompute B-spline temperature basis for fixed hourly temperatures.

    Since hourly_temps are constant across Optuna trials, the temperature
    basis only needs to be computed once and reused.

    Args:
        hourly_temps: Hourly temperatures, shape (batch_size, 274, 24)
        T_knots: Temperature knot positions
        degree: Spline degree

    Returns:
        B_T basis matrix, shape (batch_size, days_active, 24, n_T_basis)
    """
    T_hourly_active = hourly_temps[:, dates.tau_start_idx :, :]
    batch_size, days_active = T_hourly_active.shape[0], T_hourly_active.shape[1]
    T_flat = T_hourly_active.reshape(batch_size, -1)
    T_clamped = T_flat.clamp(T_knots[0], T_knots[-1])
    flat_T = T_clamped.reshape(-1)
    n_T_basis = len(T_knots) + degree - 1
    B_T = compute_bspline_basis(flat_T, T_knots, degree)
    return B_T.reshape(batch_size, days_active, 24, n_T_basis)


@torch.inference_mode()
def _run_bspline_forward_fast(
    hourly_temps: torch.Tensor,
    params: torch.Tensor,
    precomputed_B_T: torch.Tensor,
    tau_lut: BSplineLUT | None = None,
    *,
    dates: HeightDates,
) -> torch.Tensor:
    """Fast B-spline forward model with precomputed temperature basis.

    Identical to _run_bspline_forward but skips B_T computation (precomputed)
    and uses a lookup table for B_tau evaluation when provided.

    Args:
        hourly_temps: Hourly temperatures, shape (batch_size, 274, 24)
        params: Genotype parameters, shape (batch_size, 37)
        precomputed_B_T: Precomputed T basis, shape (batch_size, 135, 24, n_T_basis)
        tau_lut: Optional precomputed lookup table for B_tau basis.

    Returns:
        Height trajectories in meters, shape (batch_size, 274); zero before day 200
    """
    batch_size = hourly_temps.shape[0]
    field_names = GenotypeParams.field_names()
    param_dict = {name: params[:, i] for i, name in enumerate(field_names)}

    # Slice temperatures from day 200 onwards
    T_hourly_active = hourly_temps[:, dates.tau_start_idx :, :]

    # Thermal time accumulation at hourly resolution
    T_flat = T_hourly_active.reshape(batch_size, -1)
    tau_hourly = T_flat.clamp(min=0).cumsum(dim=-1)
    tau_max = param_dict["tau_max"]
    tau_norm_hourly = (tau_hourly / tau_max.unsqueeze(-1)).clamp(max=1.0)

    # Control points
    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    cps = GenotypeParams.build_clamped_control_grid(params, n_T_basis, n_tau_basis)

    # B_tau at hourly resolution
    tau_clamped = tau_norm_hourly.clamp(
        float(DEFAULT_TAU_KNOTS[0]), float(DEFAULT_TAU_KNOTS[-1])
    )
    flat_tau = tau_clamped.reshape(-1)
    if tau_lut is not None:
        B_tau = tau_lut(flat_tau)
    else:
        B_tau = compute_bspline_basis(flat_tau, DEFAULT_TAU_KNOTS, DEFAULT_DEGREE)
    B_tau = B_tau.reshape(batch_size, -1, 24, n_tau_basis)

    # Shared einsum + clamp + maturity gate + daily sum
    B_T = precomputed_B_T[:batch_size]
    tau_norm_3d = tau_norm_hourly.reshape(batch_size, -1, 24)
    growth = compute_growth_from_basis(B_T, B_tau, cps, tau_norm_3d)
    growth[:, dates.maximum_growth_period :] = 0.0

    heights_active = growth.cumsum(dim=-1) / 1000.0

    # Zero-pad back to full 274-day window
    heights = torch.zeros(batch_size, dates.total_days, device=hourly_temps.device)
    heights[:, dates.tau_start_idx :] = heights_active
    return heights
