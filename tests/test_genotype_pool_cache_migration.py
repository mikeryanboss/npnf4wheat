"""Regression test: stale genotype pool cache migration (ETH-882).

Verifies that _get_or_create_genotype_pool recovers from a stale cache file
with old-schema keys and regenerates a valid current-schema pool.
"""

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import torch

from npnf.data.configs.datasets.pools import _get_or_create_genotype_pool
from npnf.data.synthetic.height.genotype import GenotypeParams


def _write_stale_cache(path: Path) -> None:
    """Write a synthetic old-schema cache file that will fail to load."""
    stale_data = {
        "genotypes": [
            {"temp_min_start": 5.0, "r_max": 0.05, "old_param": 1.0} for _ in range(10)
        ],
        "bounds": {"temp_min_start": (0.0, 10.0), "r_max": (0.0, 0.1)},
        "genotype_ids": [f"G_{i:04d}" for i in range(10)],
        "seed": 42,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stale_data, path)


def _make_dist(loc=0.0, scale=1.0, lo=-3.0, hi=3.0):
    """Create a truncated-normal distribution dict."""
    return {"loc": loc, "scale": scale, "min": lo, "max": hi}


def _make_calibration_dir(base: Path) -> Path:
    """Create a minimal calibration params file for pool creation (nested format)."""
    cal_dir = base / "calibration"
    height_dir = cal_dir / "height"
    height_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "mean_control_points": {f"cp_{i}": 1.0 for i in range(30)},
        "covariance": {
            "cp_sigma": 0.1,
            "length_scale_T": 7.0,
            "length_scale_tau": 0.25,
            "jitter": 1e-6,
        },
        "tau_max": _make_dist(loc=2400.0, scale=200.0, lo=2000.0, hi=2800.0),
    }

    with (height_dir / "params_pool.json").open("w") as f:
        json.dump(params, f)

    return cal_dir


def test_stale_cache_triggers_regeneration(tmp_path):
    """Stale cache with old-schema keys is replaced with valid current-schema pool."""
    cache_path = tmp_path / "pools" / "genotype.pt"
    _write_stale_cache(cache_path)
    assert cache_path.exists()

    cal_dir = _make_calibration_dir(tmp_path)

    with (
        patch.dict("os.environ", {"NPNF_CALIBRATION_DIR": str(cal_dir)}),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        pool = _get_or_create_genotype_pool(path=cache_path, seed=42)

    # Should have warned about stale cache
    stale_warnings = [x for x in w if "Stale genotype pool cache" in str(x.message)]
    assert len(stale_warnings) == 1

    # Pool should be valid current-schema
    expected_width = len(GenotypeParams.field_names())
    assert pool.params.shape[1] == expected_width
    assert pool.markers.shape[1] == expected_width

    # Cache should be overwritten — second load should succeed without warning
    from npnf.data.pools import GenotypePool

    pool2 = GenotypePool.load(cache_path)
    assert pool2.params.shape[1] == expected_width


def test_missing_calibration_still_raises(tmp_path):
    """FileNotFoundError for missing calibration files is NOT silenced."""
    cache_path = tmp_path / "pools" / "genotype.pt"
    # No stale cache, no calibration file → should raise
    import os

    import pytest

    env = {"NPNF_CALIBRATION_DIR": str(tmp_path / "nonexistent")}
    with (
        patch.dict(os.environ, env),
        pytest.raises(FileNotFoundError, match="calibration"),
    ):
        _get_or_create_genotype_pool(path=cache_path, seed=42)
