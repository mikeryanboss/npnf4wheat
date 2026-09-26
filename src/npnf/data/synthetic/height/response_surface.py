"""B-spline response surface for synthetic height growth.

Tensor-product B-spline surface R(T, τ_norm) that maps temperature and
normalized thermal time to growth rate. Ported from scripts/research/eth_870_bspline.py.

Only depends on torch — no other npnf imports.
"""

import torch

# Default grid: 6 T knots × 5 τ knots → 8×7 cubic basis
DEFAULT_T_KNOTS = torch.linspace(0.0, 35.0, 6)  # [0.0, 7.0, 14.0, 21.0, 28.0, 35.0]
DEFAULT_TAU_KNOTS = torch.linspace(0.0, 1.0, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_DEGREE = 3


def make_clamped_knots(knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Create a clamped knot vector by repeating endpoints."""
    return torch.cat([knots[:1].expand(degree), knots, knots[-1:].expand(degree)])


def compute_bspline_basis(
    x: torch.Tensor, knots: torch.Tensor, degree: int
) -> torch.Tensor:
    """Compute B-spline basis matrix using Cox-de Boor recursion.

    Args:
        x: Evaluation points, shape (...,)
        knots: Interior knot positions (NOT clamped), shape (n_knots,)
        degree: Spline degree (1=linear, 2=quadratic, 3=cubic)

    Returns:
        Basis matrix, shape (..., n_basis) where n_basis = n_knots + degree - 1
    """
    t = make_clamped_knots(knots, degree)
    n_basis = len(t) - degree - 1
    x_shape = x.shape
    x_flat = x.reshape(-1).unsqueeze(-1)  # (N, 1)

    # Degree 0: indicator functions
    left = t[:-1].unsqueeze(0)
    right = t[1:].unsqueeze(0)
    B = ((x_flat >= left) & (x_flat < right)).float()
    # Include right endpoint: activate last non-degenerate basis function.
    # With clamped knots, the last `degree` intervals are zero-width, so the
    # relevant basis function is at index n_basis - 1.
    at_right = x_flat[:, 0] == t[-1]
    B[:, n_basis - 1] = torch.where(at_right, 1.0, B[:, n_basis - 1])

    # Cox-de Boor recursion
    for p in range(1, degree + 1):
        n = len(t) - p - 1
        B_new = torch.zeros(x_flat.shape[0], n, device=x.device, dtype=x.dtype)

        for i in range(n):
            denom_left = t[i + p] - t[i]
            if denom_left > 0:
                B_new[:, i] += (x_flat[:, 0] - t[i]) / denom_left * B[:, i]
            denom_right = t[i + p + 1] - t[i + 1]
            if denom_right > 0:
                B_new[:, i] += (t[i + p + 1] - x_flat[:, 0]) / denom_right * B[:, i + 1]

        B = B_new

    return B.reshape(*x_shape, n_basis)


class BSplineLUT:
    """Lookup table for fast B-spline basis evaluation via linear interpolation.

    Precomputes basis values on a fine uniform grid, then uses index arithmetic
    + lerp instead of Cox-de Boor recursion. ~4x faster than compute_bspline_basis
    on large tensors (>1M points) with max error ~1.8e-4 at 65K grid resolution.
    """

    def __init__(
        self,
        knots: torch.Tensor = DEFAULT_TAU_KNOTS,
        degree: int = DEFAULT_DEGREE,
        n_grid: int = 65536,
    ) -> None:
        t_min = float(knots[0])
        t_max = float(knots[-1])
        grid = torch.linspace(t_min, t_max, n_grid)
        self.basis_grid = compute_bspline_basis(
            grid, knots, degree
        )  # (n_grid, n_basis)
        self.t_min = t_min
        self.t_max = t_max
        self.n_grid = n_grid
        self.step = (t_max - t_min) / (n_grid - 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis at arbitrary points via interpolation.

        Args:
            x: Evaluation points, shape (...,). Must be within [t_min, t_max].

        Returns:
            Basis matrix, same shape as compute_bspline_basis output.
        """
        x_shape = x.shape
        x_flat = x.reshape(-1)
        idx_f = (x_flat - self.t_min) / self.step
        idx_f = idx_f.clamp(0, self.n_grid - 2)
        idx_lo = idx_f.long()
        frac = (idx_f - idx_lo.float()).unsqueeze(-1)  # (N, 1)
        b_lo = self.basis_grid[idx_lo]
        b_hi = self.basis_grid[idx_lo + 1]
        result = b_lo + frac * (b_hi - b_lo)
        n_basis = self.basis_grid.shape[1]
        return result.reshape(*x_shape, n_basis)


def compute_growth_from_basis(
    B_T: torch.Tensor, B_tau: torch.Tensor, cps: torch.Tensor, tau_norm_3d: torch.Tensor
) -> torch.Tensor:
    """Factored B-spline einsum + non-negativity clamp + maturity gate + daily sum.

    Single source of truth for the inner growth computation shared by the
    synthetic dataset and the calibration forward model. Callers are responsible
    for applying any hard post-maturity cap (e.g. MAXIMUM_GROWTH_PERIOD zeroing).

    Args:
        B_T: Temperature basis, shape (B, D, 24, n_T_basis)
        B_tau: Thermal-time basis, shape (B, D, 24, n_tau_basis)
        cps: Control point grids, shape (B, n_T_basis, n_tau_basis)
        tau_norm_3d: Normalised thermal time, shape (B, D, 24); growth is zeroed
            wherever tau_norm_3d >= 1.0 (maturity gate).

    Returns:
        Daily growth in mm/day, shape (B, D), non-negative.
    """
    partial = torch.einsum("bij,bdhj->bdhi", cps, B_tau)
    growth_hourly = torch.einsum("bdhi,bdhi->bdh", B_T, partial)
    growth_hourly = growth_hourly.clamp(min=0)
    growth_hourly = growth_hourly.masked_fill(tau_norm_3d >= 1.0, 0.0)
    return growth_hourly.sum(dim=-1)


def compute_greville_points(
    knots: torch.Tensor, degree: int, skip_first: int = 0, skip_last: int = 0
) -> torch.Tensor:
    """Compute Greville abscissae for B-spline basis functions.

    Greville points are the average of consecutive knot spans for each basis
    function. They serve as natural domain positions for control points.

    Args:
        knots: Interior knot positions (NOT clamped), shape (n_knots,).
        degree: Spline degree.
        skip_first: Number of leading basis functions to skip.
        skip_last: Number of trailing basis functions to skip.

    Returns:
        Tensor of Greville abscissae, shape (n_selected,).
    """
    clamped = make_clamped_knots(knots, degree)
    n_basis = len(knots) + degree - 1
    end = n_basis - skip_last
    return torch.tensor(
        [float(clamped[i + 1 : i + degree + 1].mean()) for i in range(skip_first, end)]
    )


# Peak row index within the 6 free T rows (Greville position = 21.0°C)
PEAK_T_ROW = 3
N_T_FREE = 6
N_TAU_FREE = 5
N_ASCENDING = PEAK_T_ROW + 1  # rows 0..3 (4 increments)
N_DESCENDING = N_T_FREE - PEAK_T_ROW - 1  # rows 4..5 (2 decrements)
N_INCREMENTS = N_ASCENDING + N_DESCENDING  # 6 total


def build_unimodal_cps(
    increments: torch.Tensor, amplitudes: torch.Tensor
) -> torch.Tensor:
    """Reconstruct 36 free CPs from unimodal T-shape increments and τ amplitudes.

    The T-response shape is built from non-negative increments (ascending to peak
    at row PEAK_T_ROW) and non-negative decrements (descending after peak). The
    shape is then scaled per τ column by the amplitude factors.

    Args:
        increments: Non-negative increments, shape (B, 6).
            [a0, a1, a2, a3, d0, d1] where a_i are ascending increments
            and d_i are descending decrements.
        amplitudes: Non-negative τ-column amplitudes, shape (B, 6).

    Returns:
        Free CPs in row-major order, shape (B, 36).
    """
    B = increments.shape[0]
    asc = increments[:, :N_ASCENDING]  # (B, 4)
    desc = increments[:, N_ASCENDING:]  # (B, 2)

    # Ascending: cumulative sum of increments
    asc_cumsum = asc.cumsum(dim=-1)  # (B, 4) → [a0, a0+a1, ...]
    peak = asc_cumsum[:, -1:]  # (B, 1)

    # Descending: peak minus cumulative decrements
    desc_cumsum = desc.cumsum(dim=-1)  # (B, 2) → [d0, d0+d1]
    desc_vals = peak - desc_cumsum  # (B, 2)

    # Full T-shape: (B, 6)
    shape = torch.cat([asc_cumsum, desc_vals], dim=-1)

    # Outer product: shape × amplitude → (B, 6, 6) → (B, 36)
    cps_free = shape.unsqueeze(-1) * amplitudes.unsqueeze(-2)
    return cps_free.reshape(B, N_T_FREE * N_TAU_FREE)


def evaluate_surface(
    T: torch.Tensor,
    tau_norm: torch.Tensor,
    control_points: torch.Tensor,
    T_knots: torch.Tensor,
    tau_knots: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    """Evaluate tensor-product B-spline surface.

    Args:
        T: Temperature values, shape (B, D)
        tau_norm: Normalized thermal time, shape (B, D)
        control_points: Per-genotype control points, shape (B, n_T_basis, n_tau_basis)
        T_knots: Temperature knot positions, shape (n_T_knots,)
        tau_knots: Development time knot positions, shape (n_tau_knots,)
        degree: Spline degree

    Returns:
        Growth rates, shape (B, D), clamped >= 0
    """
    B, D = T.shape

    # Clamp inputs to knot range
    T_clamped = T.clamp(T_knots[0], T_knots[-1])
    tau_clamped = tau_norm.clamp(tau_knots[0], tau_knots[-1])

    # Compute basis matrices
    flat_T = T_clamped.reshape(-1)
    flat_tau = tau_clamped.reshape(-1)
    B_T = compute_bspline_basis(flat_T, T_knots, degree).reshape(B, D, -1)
    B_tau = compute_bspline_basis(flat_tau, tau_knots, degree).reshape(B, D, -1)

    # Surface: R(T, τ) = Σ_ij B_T_i(T) * C_ij * B_tau_j(τ)
    growth = torch.einsum("bdi,bij,bdj->bd", B_T, control_points, B_tau)

    return growth.clamp(min=0)
