"""Penalised B-spline (P-spline) growth curves."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from npnf.calibration.height.constants import HeightDates
from npnf.data.synthetic.height.response_surface import compute_bspline_basis


@dataclass(frozen=True)
class PSpline:
    """Cubic B-spline curves on the growth window with a first-difference penalty.

    A curve is constant outside the window. The penalty keeps it flat where a
    trajectory has no observations, for example after a lodging drop.
    """

    start_day: int = HeightDates.tau_start_day
    end_day: int = HeightDates.plant_growth_stop_max
    degree: int = 3
    num_interior_knots: int = 6
    penalty: float = 0.01

    @property
    def num_coefficients(self) -> int:
        return self.num_interior_knots + self.degree + 1

    @property
    def growth_days(self) -> torch.Tensor:
        """Every day of the growth window, both ends included."""
        return torch.arange(self.start_day, self.end_day + 1)

    def basis(self, days: torch.Tensor) -> torch.Tensor:
        """(D, K) basis rows of ``days``."""
        knots = torch.linspace(
            self.start_day,
            self.end_day,
            self.num_interior_knots + 2,
            dtype=torch.float64,
        )
        clamped = days.detach().cpu().double().clamp(self.start_day, self.end_day)
        basis = compute_bspline_basis(clamped, knots, self.degree)
        return basis.float().to(days.device)

    def heights(self, coefficients: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Non-negative heights (..., D) of the curves ``coefficients`` (..., K)."""
        basis = self.basis(days.to(coefficients.device))
        return (coefficients @ basis.T).clamp(min=0)

    def fit(
        self, days: torch.Tensor, values: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """(N, K) coefficients of one weighted penalised least-squares fit per row
        of ``values`` (N, D); a zero weight excludes an observation."""
        basis = self.basis(days)
        normal = _gram(basis, weights) + self.penalty_matrix
        return torch.linalg.solve(normal, (weights * values) @ basis)

    def estimation_covariance(
        self, days: torch.Tensor, weights: torch.Tensor, noise_scale: float
    ) -> torch.Tensor:
        """(K, K) covariance of the fitted coefficients from observation noise,
        ``σ² A⁻¹ G A⁻¹`` with ``A = G + λP``, averaged over the rows."""
        gram = _gram(self.basis(days), weights)
        inverse = torch.linalg.inv(gram + self.penalty_matrix)
        return noise_scale**2 * (inverse @ gram @ inverse).mean(dim=0)

    @property
    def penalty_matrix(self) -> torch.Tensor:
        """``λ P`` with ``P = DᵀD`` for the first differences ``D``."""
        difference = torch.diff(torch.eye(self.num_coefficients), dim=0)
        return self.penalty * difference.T @ difference


def _gram(basis: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """(N, K, K) weighted Gram matrices, one per row of ``weights`` (N, D)."""
    return torch.einsum("nd,di,dj->nij", weights, basis, basis)
