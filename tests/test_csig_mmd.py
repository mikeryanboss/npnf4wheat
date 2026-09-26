from __future__ import annotations

import types
from typing import cast

import numpy as np
import pytest
import torch

from npnf.metrics.csig_mmd import (
    Censoring,
    CensoringParameters,
    CensoringSweep,
    censored_mmd_squared,
    sig_and_csig_mmd_squared,
)
from npnf.metrics.mahalanobis_artifact import (
    SignatureMahalanobisArtifact,
    censoring_threshold,
)
from npnf.metrics.sig_mmd import normalized_sig_mmd
from npnf.metrics.signature import to_metric_paths


@pytest.mark.parametrize("singleton", [False, True])
def test_one_pass_matches_sig_and_dense_csig_estimators(singleton):
    t_norm = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    if singleton:
        paths_x = to_metric_paths(torch.tensor([[1.0, 2.0, 3.0]]), 10.0, t_norm)
        paths_y = to_metric_paths(torch.tensor([[1.5, 2.5, 3.5]]), 10.0, t_norm)
        w = torch.tensor([0.2], dtype=torch.float32)
        v = torch.tensor([0.3], dtype=torch.float32)
    else:
        paths_x = to_metric_paths(
            torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.5, 4.0]]), 10.0, t_norm
        )
        paths_y = to_metric_paths(
            torch.tensor([[1.5, 2.5, 3.5], [3.0, 3.5, 4.5], [5.0, 5.5, 6.0]]),
            10.0,
            t_norm,
        )
        w = torch.tensor([0.2, 0.8], dtype=torch.float32)
        v = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float32)

    sig, csigs = sig_and_csig_mmd_squared(
        paths_x, paths_y, [(w, v), (1 - w, v / 2)], 10.0, t_norm
    )

    assert sig == normalized_sig_mmd(paths_y, paths_x)
    assert csigs[0] == pytest.approx(
        censored_mmd_squared(paths_x, paths_y, w, v, 10.0, t_norm), abs=1e-5
    )
    assert csigs[1] == pytest.approx(
        censored_mmd_squared(paths_x, paths_y, 1 - w, v / 2, 10.0, t_norm), abs=1e-5
    )
    # A rule scored alone equals the same rule scored with others.
    assert sig_and_csig_mmd_squared(
        paths_x, paths_y, [(1 - w, v / 2)], 10.0, t_norm
    ) == (sig, [csigs[1]])


def test_censoring_sweep_combines_every_threshold_with_every_beta():
    distances = np.arange(1000, dtype=np.float64)
    artifact = cast(
        SignatureMahalanobisArtifact,
        types.SimpleNamespace(
            metadata={},
            mrcd_location=np.zeros(2),
            mrcd_precision=np.eye(2),
            mahalanobis_distances=distances,
            has_lodged=distances >= 800,
        ),
    )
    main = CensoringParameters()
    main_rule = main.resolve(artifact)

    rules = CensoringSweep(
        lodged_recalls=(0.95,),
        alphas=(0.9,),
        beta_times_c_squared=(12.8,),
        betas=(1.2,),
    ).resolve(artifact, main)

    recall_alpha, recall_c_squared = censoring_threshold(artifact, None, 0.95)
    quantile_c_squared = float(np.quantile(distances, 0.9))
    assert rules == (
        Censoring(recall_alpha, 12.8 / recall_c_squared, recall_c_squared),
        Censoring(recall_alpha, 1.2, recall_c_squared),
        Censoring(0.9, 12.8 / quantile_c_squared, quantile_c_squared),
        Censoring(0.9, 1.2, quantile_c_squared),
    )
    # Without thresholds the main threshold is used; without betas, 25.6 / c².
    assert CensoringSweep(betas=(1.2,)).resolve(artifact, main) == (
        Censoring(main_rule.alpha, 1.2, main_rule.c_squared),
    )
    assert CensoringSweep(alphas=(0.9,)).resolve(artifact, main) == (
        Censoring(0.9, 25.6 / quantile_c_squared, quantile_c_squared),
    )
    with pytest.raises(ValueError, match="repeats a rule"):
        CensoringSweep(betas=(1.2, 1.2)).resolve(artifact, main)
    with pytest.raises(ValueError, match="alpha must be in"):
        CensoringSweep(alphas=(1.5,))
    assert not CensoringSweep()


def test_kernel_sigma_equals_rescaled_paths():
    # exp(-||x - y||² / sigma) on paths equals the sigma = 1 kernel on
    # paths / sqrt(sigma).
    t_norm = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    paths_x = to_metric_paths(
        torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.5, 4.0]]), 10.0, t_norm
    )
    paths_y = to_metric_paths(
        torch.tensor([[1.5, 2.5, 3.5], [3.0, 3.5, 4.5], [5.0, 5.5, 6.0]]), 10.0, t_norm
    )
    w = torch.tensor([0.2, 0.8], dtype=torch.float32)
    v = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float32)

    sig = normalized_sig_mmd(paths_y, paths_x, sigma=4.0)
    assert sig == pytest.approx(normalized_sig_mmd(paths_y / 2, paths_x / 2), abs=1e-6)
    assert sig != pytest.approx(normalized_sig_mmd(paths_y, paths_x), abs=1e-6)
    one_pass_sig, _ = sig_and_csig_mmd_squared(
        paths_x, paths_y, [(w, v)], 10.0, t_norm, sigma=4.0
    )
    assert one_pass_sig == sig
