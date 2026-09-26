"""Regression test: sampled CP matrix is not rank-1 under Gaussian prior."""

import torch

from npnf.data.pools import GenotypePool
from npnf.data.synthetic.height.genotype import NUM_FREE_CPS
from npnf.data.synthetic.height.params import (
    ControlPointCovarianceParams,
    HeightPoolParams,
    MeanControlPointParams,
    TruncatedNormalDist,
)


def _make_test_params() -> HeightPoolParams:
    """Build HeightPoolParams with a nontrivial mean surface."""
    # Triangular mean: peaks at row 3, ramps across tau
    mean_vals = {}
    for i in range(30):
        row, col = i // 5, i % 5
        t_shape = max(0.0, 1.0 - abs(row - 3) / 3.0)
        tau_ramp = (col + 1) / 5.0
        mean_vals[f"cp_{i}"] = t_shape * tau_ramp * 5.0
    return HeightPoolParams(
        mean_control_points=MeanControlPointParams(**mean_vals),
        covariance=ControlPointCovarianceParams(
            cp_sigma=0.5, length_scale_T=7.0, length_scale_tau=0.25, jitter=1e-6
        ),
        tau_max=TruncatedNormalDist(loc=2400.0, scale=300.0, min=1500.0, max=3300.0),
    )


def test_sampled_cp_matrix_is_not_rank_one():
    """With nonzero cp_sigma, sampled CPs have rank > 1."""
    hpp = _make_test_params()
    pool = GenotypePool.sample(64, seed=42, height_pool_params=hpp)

    # Take first genotype's CPs, reshape to (6, 5)
    cp_matrix = pool.params[0, :NUM_FREE_CPS].reshape(6, 5)

    # Rank test: SVD, count singular values > tolerance
    _, S, _ = torch.linalg.svd(cp_matrix)
    rank = int((S > 1e-4).sum())
    assert rank > 1, f"CP matrix has rank {rank}, expected > 1"
