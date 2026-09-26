"""Tests for B-spline response surface module."""

import torch

from npnf.data.synthetic.height.response_surface import (
    DEFAULT_DEGREE,
    DEFAULT_T_KNOTS,
    DEFAULT_TAU_KNOTS,
    compute_bspline_basis,
    evaluate_surface,
)


def test_partition_of_unity():
    """Basis functions sum to 1.0 for any x within the knot span."""
    knots = DEFAULT_T_KNOTS
    x = torch.linspace(knots[0], knots[-1], 50)
    for degree in [1, 2, 3]:
        B = compute_bspline_basis(x, knots, degree)
        sums = B.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
            f"Partition of unity failed for degree {degree}: "
            f"max deviation {(sums - 1).abs().max()}"
        )


def test_non_negativity():
    """All basis function values >= 0."""
    knots = DEFAULT_T_KNOTS
    x = torch.linspace(knots[0], knots[-1], 100)
    for degree in [1, 2, 3]:
        B = compute_bspline_basis(x, knots, degree)
        assert (B >= -1e-7).all(), f"Negative basis values for degree {degree}"


def test_boundary_clamping():
    """Zero boundary rows in control points → zero growth at T=0 and T=T_max."""
    T_knots = DEFAULT_T_KNOTS
    tau_knots = DEFAULT_TAU_KNOTS
    degree = DEFAULT_DEGREE

    n_T = len(T_knots) + degree - 1
    n_tau = len(tau_knots) + degree - 1

    cp = torch.ones(1, n_T, n_tau) * 5.0
    cp[0, 0, :] = 0.0  # First T row
    cp[0, -1, :] = 0.0  # Last T row

    T = torch.tensor([[T_knots[0].item(), T_knots[-1].item()]])  # (1, 2)
    tau = torch.tensor([[0.5, 0.5]])

    growth = evaluate_surface(T, tau, cp, T_knots, tau_knots, degree)
    assert torch.allclose(growth, torch.zeros_like(growth), atol=1e-6), (
        f"Boundary growth not zero: {growth}"
    )


def test_constant_surface():
    """All-ones control points → surface evaluates to 1.0 everywhere."""
    T_knots = DEFAULT_T_KNOTS
    tau_knots = DEFAULT_TAU_KNOTS
    degree = DEFAULT_DEGREE

    n_T = len(T_knots) + degree - 1
    n_tau = len(tau_knots) + degree - 1

    cp = torch.ones(2, n_T, n_tau)
    T = torch.rand(2, 10) * (T_knots[-1] - T_knots[0]) + T_knots[0]
    tau = torch.rand(2, 10)

    growth = evaluate_surface(T, tau, cp, T_knots, tau_knots, degree)
    assert torch.allclose(growth, torch.ones_like(growth), atol=1e-5), (
        f"Constant surface not 1.0: max deviation {(growth - 1).abs().max()}"
    )


def test_batch_consistency():
    """Batch results match sample-by-sample computation."""
    B, D = 3, 20
    T_knots = DEFAULT_T_KNOTS
    tau_knots = DEFAULT_TAU_KNOTS
    degree = DEFAULT_DEGREE

    n_T = len(T_knots) + degree - 1
    n_tau = len(tau_knots) + degree - 1

    torch.manual_seed(42)
    T = torch.rand(B, D) * 28.0
    tau = torch.rand(B, D)
    cp = torch.rand(B, n_T, n_tau) * 5.0

    # Batch
    growth_batch = evaluate_surface(T, tau, cp, T_knots, tau_knots, degree)

    # Sample-by-sample
    growth_single = torch.zeros(B, D)
    for b in range(B):
        growth_single[b] = evaluate_surface(
            T[b : b + 1], tau[b : b + 1], cp[b : b + 1], T_knots, tau_knots, degree
        )[0]

    assert torch.allclose(growth_batch, growth_single, atol=1e-5), (
        f"Batch vs single max diff: {(growth_batch - growth_single).abs().max()}"
    )
