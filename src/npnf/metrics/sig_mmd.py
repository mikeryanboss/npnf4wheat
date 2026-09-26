"""Signature MMD (Sig-MMD) estimators."""

from __future__ import annotations

import numpy as np
import torch

from npnf.metrics.signature import (
    cosine_normalized,
    signature_kernel_gram,
    to_kernel_device,
    unbiased_self_term,
)


def kernel_grams(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_batch: int | None = None,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y = to_kernel_device(x.float(), y.float())
    k_xx = signature_kernel_gram(x, x, max_batch=max_batch, sigma=sigma)  # (m, m)
    k_xy = signature_kernel_gram(x, y, max_batch=max_batch, sigma=sigma)  # (m, n)
    k_yy = signature_kernel_gram(y, y, max_batch=max_batch, sigma=sigma)  # (n, n)
    return k_xx, k_xy, k_yy


def mmd_squared(k_xx: torch.Tensor, k_xy: torch.Tensor, k_yy: torch.Tensor) -> float:
    """Unbiased MMD² from the two self Gram blocks and the cross Gram block."""
    mmd2 = unbiased_self_term(k_xx) - 2 * k_xy.mean() + unbiased_self_term(k_yy)
    return float(mmd2)


def normalized_sig_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_batch: int | None = None,
    sigma: float = 1.0,
) -> float:
    """Compute normalised Signature MMD between two batches of paths.

    Normalises the signature kernel by self-kernel norms before computing
    the unbiased MMD^2 estimator:

        K_norm(a, b) = K(a, b) / sqrt(K(a, a) * K(b, b))

    Uses the RBF static kernel exp(-||x - y||² / sigma) and dyadic order 2.
    Paths must already be lead-lag transformed with the actual time channel
    appended (as produced by ``to_metric_paths``); both ``lead_lag`` and
    ``time_aug`` are therefore off.

    Args:
        x: Pre-transformed paths from distribution P, shape ``(m, length, dim)``.
        y: Pre-transformed paths from distribution Q, shape ``(n, length, dim)``.
        max_batch: Optional kernel batch-size limit; does not subsample paths.
        sigma: Parameter of the RBF static kernel.

    Returns:
        The unbiased normalised MMD^2 (float, >= 0 in expectation).
    """
    k_xx, k_xy, k_yy = kernel_grams(x, y, max_batch=max_batch, sigma=sigma)

    # Self-kernel norms for normalisation
    diag_x = k_xx.diagonal()  # (m,)
    diag_y = k_yy.diagonal()  # (n,)

    k_xx = cosine_normalized(k_xx, diag_x, diag_x)
    k_xy = cosine_normalized(k_xy, diag_x, diag_y)
    k_yy = cosine_normalized(k_yy, diag_y, diag_y)

    return mmd_squared(k_xx, k_xy, k_yy)


def weighted_sig_mmd(x: torch.Tensor, y: torch.Tensor, a_y: np.ndarray) -> float:
    """Compute normalised Sig-MMD with weights on the reference side.

    The model side ``x`` remains uniformly weighted. The reference side ``y``
    carries non-negative per-trajectory weights ``a_y``. The kernel and
    normalisation match ``normalized_sig_mmd`` exactly; only the y self-term
    and x-y cross-term aggregations change.
    """
    if x.shape[0] == 0:
        msg = "weighted_sig_mmd requires at least one model trajectory"
        raise ValueError(msg)
    if y.shape[0] == 0:
        msg = "weighted_sig_mmd requires at least one reference trajectory"
        raise ValueError(msg)

    weights = torch.as_tensor(a_y, dtype=torch.float64)
    if weights.ndim != 1 or weights.numel() != y.shape[0]:
        msg = (
            "a_y must be a 1D weight array with one entry per reference "
            f"trajectory; got shape {tuple(weights.shape)} for {y.shape[0]} paths"
        )
        raise ValueError(msg)
    if bool((weights < 0).any()):
        msg = "a_y must contain non-negative weights"
        raise ValueError(msg)
    weight_sum = weights.sum()
    if float(weight_sum) <= 0.0:
        msg = "a_y weights must sum to a positive value"
        raise ValueError(msg)
    weights = weights / weight_sum

    k_xx, k_xy, k_yy = kernel_grams(x, y)
    diag_x = k_xx.diagonal()
    diag_y = k_yy.diagonal()

    k_xx = cosine_normalized(k_xx, diag_x, diag_x)
    k_xy = cosine_normalized(k_xy, diag_x, diag_y)
    k_yy = cosine_normalized(k_yy, diag_y, diag_y)

    a = weights.to(dtype=k_yy.dtype, device=k_yy.device)
    self_x = unbiased_self_term(k_xx)
    cross = (k_xy @ a).mean()
    a_k_a = a @ (k_yy @ a)
    sum_a2 = (a * a).sum()
    denom = 1.0 - sum_a2
    if float(denom) <= torch.finfo(k_yy.dtype).eps:
        self_y = a_k_a
    else:
        self_y = (a_k_a - sum_a2) / denom
    return float(self_x - 2.0 * cross + self_y)


def unnormalized_sig_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute unnormalised Signature MMD between two batches of paths.

    Uses the same signature-kernel configuration and final MMD^2 estimator as
    ``normalized_sig_mmd`` but does not divide Gram entries by self-kernel
    norms.

    Args:
        x: Pre-transformed paths from distribution P, shape ``(m, length, dim)``.
        y: Pre-transformed paths from distribution Q, shape ``(n, length, dim)``.

    Returns:
        The unbiased unnormalised MMD^2 (float, >= 0 in expectation).
    """
    k_xx, k_xy, k_yy = kernel_grams(x, y)
    return mmd_squared(k_xx, k_xy, k_yy)
