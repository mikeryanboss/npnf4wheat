"""Regression tests: tau_max maturity plateau semantics (ETH-883).

Verifies:
1. No positive growth increments after maturity in the dataset forward path.
2. No positive growth increments after maturity in the calibration forward path.
3. Dataset and calibration forward paths are numerically aligned for same inputs.
"""

import torch

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.forward import _run_bspline_forward
from npnf.data.datasets.synthetic import (
    _compute_growth_chunked,
    _precompute_yearsite_basis,
)
from npnf.data.synthetic.height.genotype import GenotypeParams
from npnf.data.synthetic.height.response_surface import (
    DEFAULT_DEGREE,
    DEFAULT_T_KNOTS,
    DEFAULT_TAU_KNOTS,
)

_DATES = HeightDates()


def _make_params_and_temps(B: int = 4, D: int = 274, tau_max: float = 500.0):
    """Create synthetic params and temps that ensure maturity is reached mid-season."""
    torch.manual_seed(42)
    n_params = len(GenotypeParams.field_names())
    params = torch.rand(B, n_params) * 5.0
    # Set tau_max low enough that maturity is reached well before day 274
    params[:, -1] = tau_max  # tau_max is last field

    # Warm temperatures to ensure fast thermal time accumulation
    hourly_temps = torch.full((B, D, 24), 15.0)
    return params, hourly_temps


def _run_dataset_forward(hourly_temps, params):
    """Run the SyntheticDataset growth path (without lodging/noise)."""
    B = hourly_temps.shape[0]
    B_T_unique, raw_tau_unique = _precompute_yearsite_basis(hourly_temps, dates=_DATES)
    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    cps = GenotypeParams.build_clamped_control_grid(params, n_T_basis, n_tau_basis)
    return _compute_growth_chunked(
        yearsite_indices=torch.arange(B),
        B_T_unique=B_T_unique,
        raw_tau_unique=raw_tau_unique,
        tau_max=params[:, -1],
        cps=cps,
        num_days=hourly_temps.shape[1],
        dates=_DATES,
    )


def _mature_mask(hourly_temps, params):
    """Daily-mean tau for conservative maturity detection, padded to full season."""
    T_active = hourly_temps[:, _DATES.tau_start_idx :, :].mean(dim=-1)
    tau = torch.cumsum(T_active.clamp(min=0), dim=-1)
    tau_norm = (tau / params[:, -1].unsqueeze(-1)).clamp(max=1.0)

    full_tau_norm = torch.zeros(hourly_temps.shape[0], hourly_temps.shape[1])
    full_tau_norm[:, _DATES.tau_start_idx :] = tau_norm
    return full_tau_norm >= 1.0


def _assert_no_growth_after_maturity(heights, mature_mask):
    assert mature_mask.any(), "Test setup error: no sample reached maturity"

    # mature_mask for increments: shift by 1 since diff reduces length by 1
    height_increments = torch.diff(heights, dim=-1)
    increments_after_maturity = height_increments[mature_mask[:, 1:]]
    assert torch.allclose(
        increments_after_maturity,
        torch.zeros_like(increments_after_maturity),
        atol=1e-7,
    ), (
        f"Found non-zero height increments after maturity: "
        f"max={increments_after_maturity.abs().max().item()}"
    )


def test_no_growth_after_maturity_dataset():
    """Dataset forward: no positive height increments after maturity."""
    params, hourly_temps = _make_params_and_temps()
    heights = _run_dataset_forward(hourly_temps, params)

    _assert_no_growth_after_maturity(heights, _mature_mask(hourly_temps, params))


def test_no_growth_after_maturity_calibration():
    """Calibration forward: zero growth where tau_norm >= 1.0."""
    params, hourly_temps = _make_params_and_temps()
    heights = _run_bspline_forward(hourly_temps, params, dates=_DATES)

    _assert_no_growth_after_maturity(heights, _mature_mask(hourly_temps, params))


def test_dataset_calibration_parity():
    """Dataset and calibration forward paths must produce identical heights."""
    params, hourly_temps = _make_params_and_temps()

    heights_cal = _run_bspline_forward(hourly_temps, params, dates=_DATES)
    heights_ds = _run_dataset_forward(hourly_temps, params)

    assert torch.allclose(heights_cal, heights_ds, atol=1e-6), (
        f"Dataset/calibration parity violation: "
        f"max diff={(heights_cal - heights_ds).abs().max().item()}"
    )
