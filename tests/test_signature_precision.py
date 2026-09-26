"""Signature-kernel metrics must not depend on the GPU's float32 matmul mode."""

import subprocess
import sys

import pytest
import torch

from npnf.metrics.sig_mmd import mmd_squared, normalized_sig_mmd
from npnf.metrics.signature import (
    cosine_normalized,
    signature_kernel_gram,
    to_metric_paths,
)

TF32_GPU = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


def test_importing_npnf_keeps_full_float32_matmuls():
    code = (
        "import torch, npnf, npnf.metrics.sig_mmd, npnf.metrics.csig_mmd, "
        "npnf.scripts.predict.max_height; "
        "print(torch.backends.fp32_precision, "
        "torch.backends.cuda.matmul.fp32_precision)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "tf32" not in result.stdout.split()


@pytest.mark.skipif(not TF32_GPU, reason="needs a GPU with TF32 support")
def test_single_reference_sig_mmd_on_gpu_matches_float64():
    # One reference path leaves no reference cross terms to cancel a kernel error.
    # With TF32 matmuls the GPU value is 39.445e-3 instead of 38.562e-3.
    generator = torch.Generator().manual_seed(0)
    days = torch.linspace(0, 1, 12)
    mean_height = torch.sigmoid((days - 0.5) * 10)
    model = to_metric_paths(
        mean_height + 0.03 * torch.randn(64, 12, generator=generator), 1.0, days
    )
    reference = to_metric_paths(
        mean_height[None] + 0.03 * torch.randn(1, 12, generator=generator), 1.0, days
    )

    x, y = model.double(), reference.double()
    k_xx = signature_kernel_gram(x, x)
    k_xy = signature_kernel_gram(x, y)
    k_yy = signature_kernel_gram(y, y)
    expected = mmd_squared(
        cosine_normalized(k_xx, k_xx.diagonal(), k_xx.diagonal()),
        cosine_normalized(k_xy, k_xx.diagonal(), k_yy.diagonal()),
        cosine_normalized(k_yy, k_yy.diagonal(), k_yy.diagonal()),
    )

    assert normalized_sig_mmd(model, reference) == pytest.approx(expected, abs=2e-5)
