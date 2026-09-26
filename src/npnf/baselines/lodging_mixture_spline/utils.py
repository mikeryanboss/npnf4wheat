"""Domain-free tensor helpers of the lodging-mixture spline baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def covariance_root(covariance: torch.Tensor) -> torch.Tensor:
    """Root ``R`` with ``R @ R.T`` the nearest positive semi-definite matrix."""
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors * eigenvalues.clamp(min=0).sqrt()


def sample_gaussian(
    covariance: torch.Tensor, shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    """Zero-mean Gaussian draws of shape ``(*shape, K)`` for a (K, K) covariance."""
    root = covariance_root(covariance.to(generator.device))
    noise = torch.randn(
        (*shape, root.shape[0]), generator=generator, device=generator.device
    )
    return noise @ root.T


def running_mean(
    values: torch.Tensor, width: int = 3, count: torch.Tensor | None = None
) -> torch.Tensor:
    """Centred running mean along the last dimension, edges replicated.

    With ``count`` (N,), row ``n`` of ``values`` (N, D) holds ``count[n]`` values
    followed by padding; the mean then replicates the last value, not the padding.
    """
    if count is None:
        flat = values.reshape(-1, 1, values.shape[-1])
        padded = F.pad(flat, (width // 2, width // 2), mode="replicate")
        return F.avg_pool1d(padded, width, stride=1).reshape(values.shape)
    num_rows, num_columns = values.shape
    offsets = torch.arange(width, device=values.device) - width // 2
    positions = torch.arange(num_columns, device=values.device)[:, None] + offsets
    last = (count.to(values.device) - 1).clamp(min=0)[:, None, None]
    index = torch.minimum(positions.clamp(min=0)[None], last)
    gathered = values.gather(1, index.reshape(num_rows, -1))
    return gathered.reshape(num_rows, num_columns, width).mean(dim=-1)
