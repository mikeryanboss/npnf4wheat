"""GPU correctness test for chunked growth computation (ETH-00017).

Verifies that running _compute_growth_chunked on GPU produces the same
results as on CPU, to within ~1e-5 tolerance from FP associativity.
"""

import pytest
import torch

from npnf.calibration.height.constants import HeightDates
from npnf.data.datasets.synthetic import (
    _compute_growth_chunked,
    _precompute_yearsite_basis,
)
from npnf.data.synthetic.height.genotype import GenotypeParams
from npnf.data.synthetic.height.response_surface import (
    DEFAULT_DEGREE,
    DEFAULT_T_KNOTS,
    DEFAULT_TAU_KNOTS,
    BSplineLUT,
)

_DATES = HeightDates()


def _make_test_inputs(G: int = 4, Y: int = 2, num_days: int = 274):
    """Create test inputs for growth computation."""
    torch.manual_seed(42)

    n_params = len(GenotypeParams.field_names())
    params = torch.rand(G, n_params) * 5.0
    params[:, -1] = torch.tensor([400.0, 600.0, 800.0, 1000.0][:G])
    temperatures = torch.randn(Y, 274, 24) * 5.0 + 10.0

    genotype_indices = torch.arange(G).repeat_interleave(Y)
    yearsite_indices = torch.arange(Y).repeat(G)
    params_expanded = params[genotype_indices]

    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    cps = GenotypeParams.build_clamped_control_grid(
        params_expanded, n_T_basis, n_tau_basis
    )

    field_names = GenotypeParams.field_names()
    tau_max = params_expanded[:, field_names.index("tau_max")]

    B_T_unique, raw_tau_unique = _precompute_yearsite_basis(temperatures, dates=_DATES)

    return yearsite_indices, B_T_unique, raw_tau_unique, tau_max, cps, num_days


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_growth_chunked_gpu_matches_cpu():
    """GPU growth computation matches CPU to ~1e-5."""
    yearsite_indices, B_T_unique, raw_tau_unique, tau_max, cps, num_days = (
        _make_test_inputs()
    )

    # CPU path
    tau_lut_cpu = BSplineLUT()
    heights_cpu = _compute_growth_chunked(
        yearsite_indices=yearsite_indices,
        B_T_unique=B_T_unique,
        raw_tau_unique=raw_tau_unique,
        tau_max=tau_max,
        cps=cps,
        num_days=num_days,
        tau_lut=tau_lut_cpu,
        dates=_DATES,
    )

    # GPU path
    device = torch.device("cuda")
    tau_lut_gpu = BSplineLUT()
    tau_lut_gpu.basis_grid = tau_lut_gpu.basis_grid.to(device)

    heights_gpu = _compute_growth_chunked(
        yearsite_indices=yearsite_indices.to(device),
        B_T_unique=B_T_unique.to(device),
        raw_tau_unique=raw_tau_unique.to(device),
        tau_max=tau_max.to(device),
        cps=cps.to(device),
        num_days=num_days,
        tau_lut=tau_lut_gpu,
        dates=_DATES,
    )

    heights_gpu_cpu = heights_gpu.cpu()

    assert heights_gpu_cpu.shape == heights_cpu.shape
    assert torch.allclose(heights_gpu_cpu, heights_cpu, atol=5e-5, rtol=5e-5), (
        f"Max abs diff: {(heights_gpu_cpu - heights_cpu).abs().max().item():.2e}"
    )

    # Verify non-trivial: growth actually happened
    assert heights_cpu[:, -1].max() > 0.1, "Heights should be non-trivially positive"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_growth_chunked_gpu_no_lut():
    """GPU growth computation works without LUT (compute_bspline_basis fallback)."""
    yearsite_indices, B_T_unique, raw_tau_unique, tau_max, cps, num_days = (
        _make_test_inputs()
    )

    # CPU path without LUT
    heights_cpu = _compute_growth_chunked(
        yearsite_indices=yearsite_indices,
        B_T_unique=B_T_unique,
        raw_tau_unique=raw_tau_unique,
        tau_max=tau_max,
        cps=cps,
        num_days=num_days,
        tau_lut=None,
        dates=_DATES,
    )

    # GPU path without LUT
    device = torch.device("cuda")
    heights_gpu = _compute_growth_chunked(
        yearsite_indices=yearsite_indices.to(device),
        B_T_unique=B_T_unique.to(device),
        raw_tau_unique=raw_tau_unique.to(device),
        tau_max=tau_max.to(device),
        cps=cps.to(device),
        num_days=num_days,
        tau_lut=None,
        dates=_DATES,
    )

    heights_gpu_cpu = heights_gpu.cpu()

    assert torch.allclose(heights_gpu_cpu, heights_cpu, atol=5e-5, rtol=5e-5), (
        f"Max abs diff: {(heights_gpu_cpu - heights_cpu).abs().max().item():.2e}"
    )
