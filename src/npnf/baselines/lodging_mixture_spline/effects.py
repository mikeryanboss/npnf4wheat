"""Additive genotype and year-site effects on the spline coefficients, and their
prediction from covariates for genotypes and year-sites not seen in training."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch import nn

from npnf.baselines.lodging_mixture_spline.utils import covariance_root, sample_gaussian
from npnf.calibration.height.constants import HeightDates


class AdditiveEffects(nn.Module):
    """Coefficients = mean + genotype effect + year-site effect + Gaussian residual."""

    mean: torch.Tensor  # (K,)
    genotype: torch.Tensor  # (G, K)
    yearsite: torch.Tensor  # (Y, K)
    residual_covariance: torch.Tensor  # (K, K)

    def __init__(self, num_genotypes: int, num_yearsites: int, size: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("genotype", torch.zeros(num_genotypes, size))
        self.register_buffer("yearsite", torch.zeros(num_yearsites, size))
        self.register_buffer("residual_covariance", torch.zeros(size, size))

    def fit(
        self,
        coefficients: torch.Tensor,
        genotype_index: torch.Tensor,
        yearsite_index: torch.Tensor,
        estimation_covariance: torch.Tensor,
        residual_rows: torch.Tensor,
        num_rounds: int = 50,
    ) -> None:
        """Additive effects of the centred coefficients (N, K) by backfitting:
        each effect is the group mean after the other effect is removed, which
        is also correct when not every genotype is in every year-site. The
        residual covariance comes from the rows ``residual_rows`` and excludes
        ``estimation_covariance``, the noise of their per-row fits, so sampled
        curves carry only the non-additive variation of the data."""
        mean = coefficients.mean(dim=0)
        centered = coefficients - mean
        yearsite = torch.zeros_like(self.yearsite)
        for _round in range(num_rounds):
            genotype = _group_mean(
                centered - yearsite[yearsite_index], genotype_index, len(self.genotype)
            )
            yearsite = _group_mean(
                centered - genotype[genotype_index], yearsite_index, len(self.yearsite)
            )
        residual = centered - genotype[genotype_index] - yearsite[yearsite_index]
        root = covariance_root(
            torch.cov(residual[residual_rows].T) - estimation_covariance
        )
        self.mean.copy_(mean)
        self.genotype.copy_(genotype)
        self.yearsite.copy_(yearsite)
        self.residual_covariance.copy_(root @ root.T)


class EffectRegression(nn.Module):
    """Ridge regression of effects on standardized covariates.

    For genotype markers this is GBLUP: the ridge is relative to the number of
    covariates, as in the relationship matrix ``K = Z Zᵀ / p``. The ridge is chosen
    by leave-one-out error, and the covariance of the leave-one-out errors is the
    prediction error that ``sample`` adds.
    """

    covariate_mean: torch.Tensor  # (D,)
    covariate_scale: torch.Tensor  # (D,)
    coefficients: torch.Tensor  # (K, D)
    error_covariance: torch.Tensor  # (K, K)

    def __init__(self, num_covariates: int, size: int) -> None:
        super().__init__()
        self.register_buffer("covariate_mean", torch.zeros(num_covariates))
        self.register_buffer("covariate_scale", torch.ones(num_covariates))
        self.register_buffer("coefficients", torch.zeros(size, num_covariates))
        self.register_buffer("error_covariance", torch.zeros(size, size))

    def fit(
        self,
        covariates: torch.Tensor,
        effects: torch.Tensor,
        ridges: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    ) -> dict[str, float]:
        """Fit on training ``covariates`` (N, D) and ``effects`` (N, K); return
        the chosen ridge and the leave-one-out R²."""
        scaler = StandardScaler().fit(covariates.double().numpy())
        ridge_cv = RidgeCV(
            alphas=np.asarray(ridges) * covariates.shape[1],
            fit_intercept=False,
            scoring="neg_mean_squared_error",
            store_cv_results=True,
        )
        target = effects.double().numpy()
        ridge_cv.fit(scaler.transform(covariates.double().numpy()), target)
        loo_errors = ridge_cv.cv_results_ - target[..., None]  # (N, K, ridges)
        best = int(np.square(loo_errors).sum(axis=(0, 1)).argmin())  # as RidgeCV
        loo_error = loo_errors[..., best]
        self.covariate_mean.copy_(torch.as_tensor(scaler.mean_))
        self.covariate_scale.copy_(torch.as_tensor(scaler.scale_))
        self.coefficients.copy_(torch.as_tensor(ridge_cv.coef_))
        self.error_covariance.copy_(torch.as_tensor(np.cov(loo_error, rowvar=False)))
        total = np.square(target - target.mean(axis=0)).sum()
        return {
            "ridge": ridges[best],
            "loo_r2": float(1.0 - np.square(loo_error).sum() / total),
        }

    def predict(self, covariates: torch.Tensor) -> torch.Tensor:
        """(R, K) predicted effects of ``covariates`` (R, D)."""
        standardized = (covariates - self.covariate_mean) / self.covariate_scale
        return standardized @ self.coefficients.T

    def sample(
        self, covariates: torch.Tensor, num_draws: int, generator: torch.Generator
    ) -> torch.Tensor:
        """(R, num_draws, K) predicted effects plus drawn prediction errors."""
        predicted = self.predict(covariates.to(self.coefficients.device))
        shape = (covariates.shape[0], num_draws)
        errors = sample_gaussian(self.error_covariance, shape, generator)
        return predicted.unsqueeze(1).to(errors.device) + errors


def cumulative_thermal_time(temperatures: torch.Tensor) -> torch.Tensor:
    """(Y, 123) cumulative degree-hours at the end of every day of the growth
    window (days 200 to 322) from hourly ``temperatures`` (Y, 274, 24),
    accumulated as the simulator does (hourly temperature clamped at 0)."""
    dates = HeightDates()
    window = temperatures[:, dates.tau_start_idx : dates.growing_period_end_idx + 1]
    return window.float().clamp(min=0).sum(dim=2).cumsum(dim=1)


def _group_mean(
    values: torch.Tensor, index: torch.Tensor, num_groups: int
) -> torch.Tensor:
    """(num_groups, K) mean of the rows of ``values`` in each group of ``index``."""
    groups = torch.zeros(num_groups, values.shape[1], dtype=values.dtype)
    return groups.index_reduce_(0, index, values, "mean", include_self=False)
