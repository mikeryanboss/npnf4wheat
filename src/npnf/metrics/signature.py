"""Signature paths, the signature kernel, and truncated-signature helpers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pysiglib
import torch
from loguru import logger

SYNTHETIC_SIGNATURE_HEIGHT_SCALE = 1.5
"""Fixed metre scale for synthetic signature metric height channels."""


def to_metric_paths(
    heights: torch.Tensor,
    height_scale: float,
    t_norm: torch.Tensor,
    *,
    lead_lag: bool = True,
) -> torch.Tensor:
    """Build metric paths for signature computation.

    When ``lead_lag`` is true, applies lead-lag via pysiglib then appends
    actual normalised day values as the last channel, matching pysiglib's
    (lead-lag -> time) ordering. At each stay step 2k the time is t_norm[k];
    at each transition step 2k+1 the time is the midpoint
    (t_norm[k] + t_norm[k+1]) / 2.

    When ``lead_lag`` is false, returns raw normalised height-plus-time paths
    on the original grid with channels ``[h, t]``.

    Args:
        heights: Height trajectories, shape (N, G).
        height_scale: Normalisation scale for heights.
        t_norm: Normalised day values in [0, 1], shape (G,).
        lead_lag: Whether to apply the lead-lag lift before appending time.

    Returns:
        If ``lead_lag`` is true, paths of shape ``(N, 2G-1, 3)`` =
        ``[lag_h, lead_h, t]``. Otherwise paths of shape ``(N, G, 2)`` =
        ``[h, t]``. Output dtype is float32.
    """
    N, G = heights.shape
    h = (heights.float() / height_scale).unsqueeze(-1).clone()  # (N, G, 1)
    t_float = t_norm.float()
    if not lead_lag:
        t_expanded = t_float[None, :, None].expand(N, -1, 1).contiguous()
        return torch.cat([h, t_expanded], dim=-1)  # (N, G, 2)

    ll = cast(torch.Tensor, pysiglib.transform_path(h, lead_lag=True))  # (N, 2G-1, 2)
    t_ll = torch.empty(2 * G - 1, dtype=torch.float32, device=t_float.device)
    t_ll[0::2] = t_float
    t_ll[1::2] = (t_float[:-1] + t_float[1:]) / 2
    t_expanded = t_ll[None, :, None].expand(N, -1, 1).contiguous()
    return torch.cat([ll, t_expanded], dim=-1)  # (N, 2G-1, 3)


def kernel_sigma_suffix(sigma: float) -> str:
    """Output-folder suffix of a static-kernel ``sigma``; empty for the default 1."""
    return "" if sigma == 1.0 else f"_sigma={sigma:g}"


def signature_kernel_gram(
    paths_a: torch.Tensor,
    paths_b: torch.Tensor,
    *,
    max_batch: int | None = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Signature-kernel Gram matrix shared by the Sig-MMD and CSig-MMD estimators.

    RBF static kernel exp(-||x - y||² / sigma), dyadic order 2. Paths must already
    carry their lead-lag lift and time channel (as produced by
    ``to_metric_paths``), so both ``lead_lag`` and ``time_aug`` are off.
    ``max_batch`` limits kernel batch size without subsampling paths.
    """
    return cast(
        torch.Tensor,
        pysiglib.sig_kernel_gram(
            paths_a,
            paths_b,
            dyadic_order=2,
            static_kernel=pysiglib.RBFKernel(sigma=sigma),
            time_aug=False,
            lead_lag=False,
            max_batch=-1 if max_batch is None else max_batch,
        ),
    )


def to_kernel_device(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Move path and weight tensors to the GPU when one is available."""
    if torch.cuda.is_available():
        return tuple(tensor.cuda() for tensor in tensors)
    return tensors


def cosine_normalized(
    gram: torch.Tensor, diag_rows: torch.Tensor, diag_cols: torch.Tensor
) -> torch.Tensor:
    """Divide Gram entries by the self-kernel norms of their row and column paths."""
    return gram / torch.sqrt(torch.outer(diag_rows, diag_cols))


def unbiased_self_term(k: torch.Tensor) -> torch.Tensor:
    """Off-diagonal mean of a square Gram block; the diagonal entry for a singleton."""
    sample_count = k.shape[0]
    if sample_count == 1:
        return k[0, 0]
    return (k.sum() - k.trace()) / (sample_count * (sample_count - 1))


def compute_normalized_truncated_sigs(
    paths: np.ndarray, depth: int, self_kernel_batch_size: int = 512
) -> np.ndarray:
    """Compute kernel-normalised truncated signatures (level 0 dropped).

    Removes the signature "height bias" (larger trajectories have larger
    self-norm) by dividing each truncated signature by ``sqrt(k_sig(x, x))``
    with the same sig kernel config used for the MMD. Empirically validated by
    ``score_sig_mahalanobis.py``: after this normalisation, lodging surfaces as
    the dominant outlier regardless of plant size.

    Args:
        paths: Shape ``(N, L, D)``, already lifted (e.g. via ``to_metric_paths``).
        depth: Signature truncation depth.
        self_kernel_batch_size: Batch size for the self-kernel computation
            to cap GPU memory.

    Returns:
        Normalised signatures of shape ``(N, sig_dim - 1)`` with level 0 dropped.
    """
    sigs = pysiglib.sig(paths, degree=depth, time_aug=False, lead_lag=False)[:, 1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths_torch = torch.from_numpy(paths).to(device)
    self_k = np.empty(len(paths))
    n_batches = (len(paths) + self_kernel_batch_size - 1) // self_kernel_batch_size
    for b, i in enumerate(range(0, len(paths), self_kernel_batch_size)):
        batch = paths_torch[i : i + self_kernel_batch_size]
        diag = cast(
            torch.Tensor,
            pysiglib.sig_kernel(
                batch,
                batch,
                dyadic_order=2,
                static_kernel=pysiglib.RBFKernel(sigma=1.0),
                time_aug=False,
                lead_lag=False,
            ),
        )
        self_k[i : i + self_kernel_batch_size] = diag.sqrt().cpu().numpy()
        if b == 0 or (b + 1) % 100 == 0 or b + 1 == n_batches:
            logger.info("Self-kernel: batch {}/{}", b + 1, n_batches)

    return sigs / self_k[:, None]


def mahalanobis_distance(
    sigs: np.ndarray, location: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    """Compute Mahalanobis distance ``sqrt((s - μ)ᵀ Σ⁻¹ (s - μ))`` per row.

    Args:
        sigs: Shape ``(N, D)``.
        location: Shape ``(D,)``.
        precision: Shape ``(D, D)``, the precision (inverse covariance).

    Returns:
        Distances of shape ``(N,)``.
    """
    delta = sigs - location
    return np.sqrt((delta @ precision * delta).sum(axis=1))
