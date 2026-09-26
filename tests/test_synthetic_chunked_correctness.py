"""Correctness test for chunked growth computation (ETH-00011).

Verifies that the new precomputed-yearsite + chunked-einsum path produces
the same growth heights as the original evaluate_surface path, up to
~1e-5 tolerance from FP associativity differences in the factored einsum.
"""

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
    evaluate_surface,
)

_DATES = HeightDates()


def _reference_growth(
    temperatures_expanded: torch.Tensor, params: torch.Tensor, num_days: int = 274
) -> torch.Tensor:
    """Compute growth using the original evaluate_surface path (pre-ETH-00011)."""
    B = temperatures_expanded.shape[0]
    field_names = GenotypeParams.field_names()
    param_dict = {name: params[:, i] for i, name in enumerate(field_names)}

    T_hourly_active = temperatures_expanded[:, _DATES.tau_start_idx :, :]
    T_flat = T_hourly_active.reshape(B, -1)

    tau_hourly = T_flat.clamp(min=0).cumsum(dim=-1)
    tau_max = param_dict["tau_max"]
    tau_norm_hourly = (tau_hourly / tau_max.unsqueeze(-1)).clamp(max=1.0)

    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    cps = GenotypeParams.build_clamped_control_grid(params, n_T_basis, n_tau_basis)

    growth_hourly = evaluate_surface(
        T_flat,
        tau_norm_hourly,
        cps,
        DEFAULT_T_KNOTS,
        DEFAULT_TAU_KNOTS,
        degree=DEFAULT_DEGREE,
    )
    growth_hourly = growth_hourly.masked_fill(tau_norm_hourly >= 1.0, 0.0)
    growth = growth_hourly.reshape(B, -1, 24).sum(dim=-1)

    heights_active = growth.cumsum(dim=-1) / 1000.0
    heights_grown = torch.zeros(B, num_days)
    heights_grown[:, _DATES.tau_start_idx :] = heights_active
    return heights_grown


def test_chunked_matches_evaluate_surface():
    """New chunked path matches original evaluate_surface to ~1e-5."""
    torch.manual_seed(42)

    G, Y = 4, 2
    num_days = 274
    n_params = len(GenotypeParams.field_names())

    # Synthetic inputs
    params = torch.rand(G, n_params) * 5.0
    params[:, -1] = torch.tensor([400.0, 600.0, 800.0, 1000.0])  # tau_max
    temperatures = torch.randn(Y, 274, 24) * 5.0 + 10.0  # realistic-ish

    # Factorial expansion
    genotype_indices = torch.arange(G).repeat_interleave(Y)
    yearsite_indices = torch.arange(Y).repeat(G)
    params_expanded = params[genotype_indices]

    # Reference: original evaluate_surface path
    temperatures_expanded = temperatures[yearsite_indices]
    heights_ref = _reference_growth(temperatures_expanded, params_expanded, num_days)

    # New: precompute + chunked path
    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    cps = GenotypeParams.build_clamped_control_grid(
        params_expanded, n_T_basis, n_tau_basis
    )
    field_names = GenotypeParams.field_names()
    tau_max = params_expanded[:, field_names.index("tau_max")]

    B_T_unique, raw_tau_unique = _precompute_yearsite_basis(temperatures, dates=_DATES)
    heights_new = _compute_growth_chunked(
        yearsite_indices=yearsite_indices,
        B_T_unique=B_T_unique,
        raw_tau_unique=raw_tau_unique,
        tau_max=tau_max,
        cps=cps,
        num_days=num_days,
        dates=_DATES,
    )

    # Should match to ~1e-5 (FP associativity in factored einsum)
    assert heights_new.shape == heights_ref.shape
    assert torch.allclose(heights_new, heights_ref, atol=1e-5, rtol=1e-5), (
        f"Max abs diff: {(heights_new - heights_ref).abs().max().item():.2e}"
    )

    # Verify non-trivial: growth actually happened
    assert heights_new[:, -1].max() > 0.1, "Heights should be non-trivially positive"
