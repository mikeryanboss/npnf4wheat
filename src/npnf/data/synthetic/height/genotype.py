"""Synthetic genotype parameter generation.

This module provides the SyntheticGenotypeGenerator class for sampling
genotype parameters used in the B-spline response surface growth model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from npnf.data.synthetic.height.params import HeightPoolParams

# Number of free control points: 6 T rows x 5 tau columns
# (τ=0 and τ=1 columns are structurally fixed at zero via cps[:, 1:-1, 1:-1] = cps_free)
NUM_FREE_CPS = 30


def _sample_truncated_normal(
    loc: torch.Tensor | float,
    scale: torch.Tensor | float,
    low: torch.Tensor | float,
    high: torch.Tensor | float,
    num_samples: int,
) -> torch.Tensor:
    """Sample from a truncated normal via inverse-CDF transform."""
    device = None
    for value in (loc, scale, low, high):
        if isinstance(value, torch.Tensor):
            device = value.device
            break

    target_shape = (num_samples,)
    loc_t = torch.broadcast_to(
        torch.as_tensor(loc, dtype=torch.float32, device=device), target_shape
    )
    scale_t = torch.broadcast_to(
        torch.as_tensor(scale, dtype=torch.float32, device=device), target_shape
    )
    low_t = torch.broadcast_to(
        torch.as_tensor(low, dtype=torch.float32, device=device), target_shape
    )
    high_t = torch.broadcast_to(
        torch.as_tensor(high, dtype=torch.float32, device=device), target_shape
    )
    if torch.any(low_t >= high_t):
        msg = "Invalid truncated normal bounds: low must be < high."
        raise ValueError(msg)

    eps = 1e-6
    scale_t = scale_t.clamp_min(eps)
    standard_low = (low_t - loc_t) / scale_t
    standard_high = (high_t - loc_t) / scale_t

    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    cdf_low = 0.5 * (1.0 + torch.erf(standard_low * inv_sqrt_2))
    cdf_high = 0.5 * (1.0 + torch.erf(standard_high * inv_sqrt_2))
    cdf_low = cdf_low.clamp(eps, 1.0 - eps)
    cdf_high = cdf_high.clamp(eps, 1.0 - eps)
    cdf_span = (cdf_high - cdf_low).clamp_min(eps)

    u = cdf_low + torch.rand_like(loc_t) * cdf_span
    u = u.clamp(eps, 1.0 - eps)
    standard_samples = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    samples = loc_t + scale_t * standard_samples
    return samples.clamp(low_t, high_t)


@dataclass
class GenotypeParams:
    """Parameters for a single genotype in the B-spline growth model.

    Control points define the B-spline response surface R(T, τ_norm).
    The 30 free CPs are laid out as 6 T rows x 5 tau columns (the first
    and last T rows are fixed at zero for boundary clamping; the τ=0 and
    τ=1 columns are structurally fixed at zero via cps[:, 1:-1, 1:-1] = cps_free).
    """

    # 30 free control points (non-negative)
    cp_0: float
    cp_1: float
    cp_2: float
    cp_3: float
    cp_4: float
    cp_5: float
    cp_6: float
    cp_7: float
    cp_8: float
    cp_9: float
    cp_10: float
    cp_11: float
    cp_12: float
    cp_13: float
    cp_14: float
    cp_15: float
    cp_16: float
    cp_17: float
    cp_18: float
    cp_19: float
    cp_20: float
    cp_21: float
    cp_22: float
    cp_23: float
    cp_24: float
    cp_25: float
    cp_26: float
    cp_27: float
    cp_28: float
    cp_29: float

    # Thermal time to maturity (degree-days)
    tau_max: float

    def to_tensor(self) -> torch.Tensor:
        """Convert parameters to a 1D tensor in canonical order.

        Returns:
            Tensor of shape (31,) with parameter values.
        """
        return torch.tensor(
            [getattr(self, f.name) for f in fields(self)], dtype=torch.float32
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> GenotypeParams:
        """Create GenotypeParams from a tensor.

        Args:
            tensor: Tensor of shape (31,) with parameter values.

        Returns:
            GenotypeParams instance.
        """
        field_names = [f.name for f in fields(cls)]
        return cls(**{name: float(tensor[i]) for i, name in enumerate(field_names)})

    @classmethod
    def field_names(cls) -> list[str]:
        """Return list of parameter names in canonical order."""
        return [f.name for f in fields(cls)]

    @classmethod
    def build_clamped_control_grid(
        cls, params_tensor: torch.Tensor, n_T_basis: int, n_tau_basis: int
    ) -> torch.Tensor:
        """Build the zero-padded B-spline control point grid from a parameter tensor.

        Args:
            params_tensor: Tensor of shape (B, 31) containing genotype parameters.
            n_T_basis: Number of temperature basis functions.
            n_tau_basis: Number of thermal time basis functions.

        Returns:
            Grid of shape (B, n_T_basis, n_tau_basis) with clamped boundaries.
        """
        B = params_tensor.shape[0]
        field_names = cls.field_names()

        # Extract the free CP indices
        cp_indices = [field_names.index(f"cp_{i}") for i in range(NUM_FREE_CPS)]
        cps_free = params_tensor[:, cp_indices]  # (B, 30)

        n_T_free = n_T_basis - 2
        cps_free = cps_free.reshape(B, n_T_free, n_tau_basis - 2)

        cps = torch.zeros(
            B,
            n_T_basis,
            n_tau_basis,
            dtype=params_tensor.dtype,
            device=params_tensor.device,
        )
        cps[:, 1:-1, 1:-1] = cps_free  # τ=0 and τ=1 columns stay zero
        return cps


@dataclass
class GenotypeParamBounds:
    """Bounds for each genotype parameter, used for normalization.

    Each field is a tuple of (min, max) defining the valid range.
    """

    # Control point bounds
    cp_0: tuple[float, float]
    cp_1: tuple[float, float]
    cp_2: tuple[float, float]
    cp_3: tuple[float, float]
    cp_4: tuple[float, float]
    cp_5: tuple[float, float]
    cp_6: tuple[float, float]
    cp_7: tuple[float, float]
    cp_8: tuple[float, float]
    cp_9: tuple[float, float]
    cp_10: tuple[float, float]
    cp_11: tuple[float, float]
    cp_12: tuple[float, float]
    cp_13: tuple[float, float]
    cp_14: tuple[float, float]
    cp_15: tuple[float, float]
    cp_16: tuple[float, float]
    cp_17: tuple[float, float]
    cp_18: tuple[float, float]
    cp_19: tuple[float, float]
    cp_20: tuple[float, float]
    cp_21: tuple[float, float]
    cp_22: tuple[float, float]
    cp_23: tuple[float, float]
    cp_24: tuple[float, float]
    cp_25: tuple[float, float]
    cp_26: tuple[float, float]
    cp_27: tuple[float, float]
    cp_28: tuple[float, float]
    cp_29: tuple[float, float]

    # τ_max bounds
    tau_max: tuple[float, float]

    def normalize(self, params: GenotypeParams) -> torch.Tensor:
        """Normalize parameters to [0, 1] range using bounds.

        Args:
            params: GenotypeParams to normalize.

        Returns:
            Tensor of shape (31,) with normalized values in [0, 1].
        """
        normalized = []
        for f in fields(params):
            value = getattr(params, f.name)
            lo, hi = getattr(self, f.name)
            normalized.append((value - lo) / (hi - lo + 1e-8))
        return torch.tensor(normalized, dtype=torch.float32)

    def denormalize(self, tensor: torch.Tensor) -> GenotypeParams:
        """Convert normalized tensor back to GenotypeParams.

        Args:
            tensor: Tensor of shape (31,) with values in [0, 1].

        Returns:
            GenotypeParams with denormalized values.
        """
        field_list = fields(GenotypeParams)
        values = {}
        for i, f in enumerate(field_list):
            lo, hi = getattr(self, f.name)
            values[f.name] = float(tensor[i]) * (hi - lo) + lo
        return GenotypeParams(**values)

    @classmethod
    def from_dict(
        cls, bounds_dict: dict[str, tuple[float, float]]
    ) -> GenotypeParamBounds:
        """Create GenotypeParamBounds from a dictionary.

        Args:
            bounds_dict: Dict mapping parameter names to (min, max) tuples.

        Returns:
            GenotypeParamBounds instance.
        """
        return cls(**bounds_dict)

    def to_dict(self) -> dict[str, tuple[float, float]]:
        """Convert to dictionary mapping parameter names to bounds."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


class SyntheticGenotypeGenerator:
    """Generator for synthetic genotype parameters via Gaussian CP prior.

    Samples the 6x5 free control-point grid from a matrix-normal
    distribution: C_g = M_free + cp_sigma * L_T @ Z_g @ L_tau^T,
    where K_T and K_tau are RBF kernels on the free Greville points.
    The tau_max parameter is sampled independently from a truncated normal.

    Accepts a HeightPoolParams instance containing MeanControlPointParams,
    ControlPointCovarianceParams, and a TruncatedNormalDist for tau_max.
    """

    def __init__(self, height_pool_params: HeightPoolParams) -> None:
        self.height_pool_params = height_pool_params

    @torch.inference_mode()
    def sample(
        self, num_genotypes: int, seed: int
    ) -> tuple[torch.Tensor, GenotypeParamBounds]:
        """Sample genotype parameters.

        Args:
            num_genotypes: Number of genotypes to sample
            seed: Random seed for reproducibility

        Returns:
            Tuple of (params, bounds) where:
                - params: Tensor of shape (num_genotypes, 31)
                - bounds: GenotypeParamBounds for normalization
        """
        import gpytorch.kernels

        from npnf.data.synthetic.height.response_surface import (
            DEFAULT_DEGREE,
            DEFAULT_T_KNOTS,
            DEFAULT_TAU_KNOTS,
            compute_greville_points,
        )

        torch.manual_seed(seed)

        B = num_genotypes
        hpp = self.height_pool_params

        # Build mean free CP grid (6, 5) from stored cp_0..cp_29
        mean_values = torch.tensor(
            [getattr(hpp.mean_control_points, f"cp_{i}") for i in range(NUM_FREE_CPS)],
            dtype=torch.float32,
        )
        M_free = mean_values.reshape(6, 5)

        # Build kernel matrices on free Greville points
        t_greville = compute_greville_points(
            DEFAULT_T_KNOTS, DEFAULT_DEGREE, skip_first=1, skip_last=1
        )
        tau_greville = compute_greville_points(
            DEFAULT_TAU_KNOTS, DEFAULT_DEGREE, skip_first=1, skip_last=1
        )

        rbf_T = gpytorch.kernels.RBFKernel()
        rbf_T.lengthscale = hpp.covariance.length_scale_T  # ty: ignore[invalid-assignment]
        K_T = rbf_T(t_greville.unsqueeze(-1)).to_dense() + (
            hpp.covariance.jitter * torch.eye(6)
        )
        L_T = torch.linalg.cholesky(K_T)

        rbf_tau = gpytorch.kernels.RBFKernel()
        rbf_tau.lengthscale = hpp.covariance.length_scale_tau  # ty: ignore[invalid-assignment]
        K_tau = rbf_tau(tau_greville.unsqueeze(-1)).to_dense() + (
            hpp.covariance.jitter * torch.eye(5)
        )
        L_tau = torch.linalg.cholesky(K_tau)

        # Matrix-normal sample
        Z = torch.randn(B, 6, 5)
        cp_sigma = hpp.covariance.cp_sigma
        C_free = M_free.unsqueeze(0) + cp_sigma * torch.einsum(
            "ij,bjk,lk->bil", L_T, Z, L_tau
        )

        # Clip
        lower = (M_free - 4 * cp_sigma).clamp(min=0.0).unsqueeze(0)
        upper = (M_free + 4 * cp_sigma).unsqueeze(0)
        C_free = C_free.clamp(min=lower, max=upper)

        cps_flat = C_free.reshape(B, NUM_FREE_CPS)

        # Sample tau_max
        tau_max = _sample_truncated_normal(
            hpp.tau_max.loc, hpp.tau_max.scale, hpp.tau_max.min, hpp.tau_max.max, B
        )

        params = torch.cat([cps_flat, tau_max.unsqueeze(1)], dim=1)  # (B, 31)

        # Build bounds from clipping interval
        bounds_dict = {}
        for i in range(NUM_FREE_CPS):
            row, col = i // 5, i % 5
            lo = max(0.0, float(M_free[row, col]) - 4 * cp_sigma)
            hi = float(M_free[row, col]) + 4 * cp_sigma
            bounds_dict[f"cp_{i}"] = (lo, hi)
        bounds_dict["tau_max"] = (hpp.tau_max.min, hpp.tau_max.max)

        bounds = GenotypeParamBounds.from_dict(bounds_dict)
        return params, bounds
