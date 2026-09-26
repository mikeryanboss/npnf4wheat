"""FIP1 initialization defaults for height calibration."""

import logging

import numpy as np
import torch

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.forward import _run_bspline_forward
from npnf.data.synthetic.height.response_surface import (
    DEFAULT_DEGREE,
    DEFAULT_T_KNOTS,
    DEFAULT_TAU_KNOTS,
    N_ASCENDING,
    N_TAU_FREE,
    PEAK_T_ROW,
    build_unimodal_cps,
    compute_greville_points,
)

logger = logging.getLogger(__name__)


def _derive_tau_max(
    year_temps: dict[int, torch.Tensor],
    observed_growth_end: dict[int, torch.Tensor],
    years: list[int],
    *,
    dates: HeightDates,
) -> tuple[float, float]:
    """Derive tau_max location and scale from FIP1 growth-end timing.

    For each year, accumulates hourly thermal time from tau_start and reads
    off the value at the mean observed growth-end day.

    Returns:
        (tau_max_loc, tau_max_scale) with scale floored at 500.
    """
    per_year_tau: list[float] = []
    for year in years:
        temps = year_temps[year]  # (274, 24)
        hourly_tau = (
            temps[dates.tau_start_idx :].reshape(-1).clamp(min=0).cumsum(dim=-1)
        )

        mean_growth_end_day = float(observed_growth_end[year].mean())
        hour_idx = min(
            round((mean_growth_end_day - dates.tau_start_day) * 24), len(hourly_tau) - 1
        )
        per_year_tau.append(float(hourly_tau[hour_idx]))

    loc = float(np.mean(per_year_tau))
    scale = max(float(np.std(per_year_tau)), 500.0)
    return loc, scale


def _build_initial_mean_cps(
    year_temps: dict[int, torch.Tensor],
    observed_heights: dict[int, torch.Tensor],
    years: list[int],
    tau_max_loc: float,
    *,
    dates: HeightDates,
) -> torch.Tensor:
    """Build an initial mean CP surface scaled to match observed FIP1 heights.

    Constructs a triangular T-shape x tau-ramp via ``build_unimodal_cps``,
    runs the forward model to measure the resulting mean height, and rescales
    the CPs so simulated heights match observed heights on average.

    Returns:
        Flat tensor of 36 initial CP values.
    """
    t_greville = compute_greville_points(
        DEFAULT_T_KNOTS, DEFAULT_DEGREE, skip_first=1, skip_last=1
    )
    tau_greville = compute_greville_points(
        DEFAULT_TAU_KNOTS, DEFAULT_DEGREE, skip_first=1, skip_last=1
    )

    # Unit-peak triangular temperature shape at Greville positions
    peak_t = float(t_greville[PEAK_T_ROW])
    t_max_boundary = float(DEFAULT_T_KNOTS[-1])  # 35.0
    shape = torch.zeros(len(t_greville))
    for i, t in enumerate(t_greville.tolist()):
        if i <= PEAK_T_ROW:
            shape[i] = t / peak_t
        else:
            shape[i] = max(0.0, (t_max_boundary - t) / (t_max_boundary - peak_t))

    # Encode as ascending increments + descending decrements for build_unimodal_cps
    encoded = torch.zeros(N_ASCENDING + (len(t_greville) - PEAK_T_ROW - 1))
    encoded[0] = shape[0]
    for i in range(1, N_ASCENDING):
        encoded[i] = shape[i] - shape[i - 1]
    for i in range(len(t_greville) - PEAK_T_ROW - 1):
        encoded[N_ASCENDING + i] = shape[PEAK_T_ROW + i] - shape[PEAK_T_ROW + i + 1]

    # Tau ramp proportional to developmental position
    tau_ramp = torch.tensor(
        [float(tau_greville[j]) / float(tau_greville[-1]) for j in range(N_TAU_FREE)]
    ).clamp(min=0.1)

    # Run forward model to measure height scale mismatch
    unscaled_cps = build_unimodal_cps(
        encoded.unsqueeze(0), tau_ramp.unsqueeze(0)
    ).squeeze(0)
    params = torch.cat([unscaled_cps, torch.tensor([tau_max_loc])]).unsqueeze(0)
    temps_batch = torch.stack([year_temps[y] for y in years])
    trajectories = _run_bspline_forward(
        temps_batch, params.expand(len(years), -1), dates=dates
    )
    ref_mean_height = float(trajectories.max(dim=-1).values.mean())

    observed_mean = float(torch.cat([observed_heights[y] for y in years]).mean())
    height_scale = observed_mean / max(ref_mean_height, 1e-6)

    # Rescale both factors by sqrt so their product scales by height_scale
    sqrt_scale = height_scale**0.5
    return build_unimodal_cps(
        (encoded * sqrt_scale).unsqueeze(0), (tau_ramp * sqrt_scale).unsqueeze(0)
    ).squeeze(0)


def _compute_fip1_defaults(
    year_temps: dict[int, torch.Tensor],
    observed_heights: dict[int, torch.Tensor],
    observed_growth_end: dict[int, torch.Tensor],
    *,
    dates: HeightDates,
) -> dict[str, dict[str, float]]:
    """Compute calibration initialization defaults from FIP1 data.

    Derives tau_max, initial mean CP values, and covariance hyperparameter
    defaults from observed FIP1 growth-end timing and height distributions.

    Returns:
        Dict mapping the 35 Optuna parameter names to ``{default, min, max}``.
    """
    years = sorted(set(year_temps) & set(observed_growth_end))

    tau_max_loc, tau_max_scale = _derive_tau_max(
        year_temps, observed_growth_end, years, dates=dates
    )
    mean_cps = _build_initial_mean_cps(
        year_temps, observed_heights, years, tau_max_loc, dates=dates
    )
    max_cp = float(mean_cps.max())

    # Assemble per-parameter {default, min, max} bounds
    defaults: dict[str, dict[str, float]] = {}
    for i in range(30):
        val = float(mean_cps[i])
        defaults[f"cp_{i}"] = {"default": val, "min": 0.0, "max": max(4.0 * val, 1e-4)}

    defaults["cp_sigma"] = {
        "default": 0.1 * max_cp,
        "min": 1e-4,
        "max": max(max_cp, 1e-3),
    }
    defaults["length_scale_T"] = {"default": 7.0, "min": 3.5, "max": 21.0}
    defaults["length_scale_tau"] = {"default": 0.25, "min": 0.125, "max": 0.75}
    defaults["tau_max_loc"] = {
        "default": tau_max_loc,
        "min": tau_max_loc - 3 * tau_max_scale,
        "max": tau_max_loc + 3 * tau_max_scale,
    }
    defaults["tau_max_scale"] = {
        "default": tau_max_scale,
        "min": tau_max_scale / 4,
        "max": tau_max_scale * 4,
    }

    logger.info("FIP1-derived defaults (Gaussian CP prior):")
    logger.info("  tau_max: loc=%.0f scale=%.0f", tau_max_loc, tau_max_scale)
    logger.info("  max_cp=%.4f, cp_sigma_default=%.4f", max_cp, 0.1 * max_cp)

    return defaults
