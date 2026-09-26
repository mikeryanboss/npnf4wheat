"""Hydra-zen configs for genotype and yearsite pools."""

import os
from pathlib import Path

from npnf.configs.utils import builds, partial_builds
from npnf.data.pools import GenotypePool, YearsitePool

# =============================================================================
# Pool Parameters
# =============================================================================

# Genotype pool sizing. Distribution parameters (c_k, tau_max) are fully
# determined by calibration results in params_pool.json; only num_genotypes
# is a fixed default.
_GENOTYPE_POOL_PARAMS = {"num_genotypes": 10_000}

# Standard yearsite pool parameters (shared across all pools)
# 8 sites × 40 years = 320 yearsites
# Sites 0-3: training, Sites 4-7: test/val
# Years 0-31: training, Years 32-39: test/val
_YEARSITE_POOL_PARAMS = {
    "num_sites": 8,
    "num_years": 40,
    "num_days": 274,
    "start_year": 2010,
}


# =============================================================================
# Pool Creation Functions
# =============================================================================


def _get_or_create_genotype_pool(path: str | Path, seed: int) -> GenotypePool:
    """Load genotype pool from cache, or create and cache it.

    Reads calibrated parameters from
    ``{NPNF_CALIBRATION_DIR}/height/params_pool.json``
    and merges with fixed defaults from ``_GENOTYPE_POOL_PARAMS``.
    """
    cache_path = Path(path)
    if cache_path.exists():
        try:
            return GenotypePool.load(cache_path)
        except (KeyError, TypeError):
            import warnings

            warnings.warn(
                f"Stale genotype pool cache at {cache_path}, regenerating.",
                stacklevel=2,
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_dir = Path(
        os.environ.get("NPNF_CALIBRATION_DIR", "results/calibration")
    )
    params_path = calibration_dir / "height/params_pool.json"
    if not params_path.exists():
        msg = (
            f"Height calibration file not found at {params_path}. "
            "Run height calibration first, or set NPNF_CALIBRATION_DIR."
        )
        raise FileNotFoundError(msg)

    from npnf.data.synthetic.height.params import load_height_pool_params

    height_pool_params = load_height_pool_params(params_path)
    num_genotypes = _GENOTYPE_POOL_PARAMS["num_genotypes"]
    pool = GenotypePool.sample(
        num_genotypes=num_genotypes, seed=seed, height_pool_params=height_pool_params
    )
    pool.save(cache_path)
    return pool


def _get_or_create_yearsite_pool(path: str | Path, seed: int) -> YearsitePool:
    """Load yearsite pool from cache, or create and cache it.

    Resolves temperature calibration params via
    ``{NPNF_CALIBRATION_DIR}/temperature/params_pool.json``.
    """
    cache_path = Path(path)
    if cache_path.exists():
        return YearsitePool.load(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_dir = Path(
        os.environ.get("NPNF_CALIBRATION_DIR", "results/calibration")
    )
    params_path = calibration_dir / "temperature/params_pool.json"
    if not params_path.exists():
        msg = (
            f"Temperature calibration file not found at {params_path}. "
            "Run temperature calibration first, "
            "or set NPNF_CALIBRATION_DIR."
        )
        raise FileNotFoundError(msg)

    pool = YearsitePool.sample_synthetic(
        seed=seed, calibration_params_path=str(params_path), **_YEARSITE_POOL_PARAMS
    )
    pool.save(cache_path)
    return pool


# =============================================================================
# Synthetic Pool Configs (Cached, Single Source of Truth)
# =============================================================================

POOLS_PATH = "dataset_cache/pools"

# Fixed seeds for pool construction. These are also referenced by
# BaseSyntheticDatasetConfig so the dataset cache key can be computed
# without instantiating the pools.
GENOTYPE_POOL_SEED = 42
YEARSITE_POOL_SEED = 42
YEARSITE_POOL_SOURCE = "synthetic"

# partial_builds → Hydra-Zen resolves these to functools.partial callables
# instead of eagerly calling _get_or_create_*. SyntheticDataset invokes the
# callables only on cache miss, so training with a valid dataset cache does
# not require the pool or its calibration files.
GenotypePoolConfig = partial_builds(
    _get_or_create_genotype_pool,
    path=f"{POOLS_PATH}/genotype.pt",
    seed=GENOTYPE_POOL_SEED,
)

YearsitePoolConfig = partial_builds(
    _get_or_create_yearsite_pool,
    path=f"{POOLS_PATH}/yearsite.pt",
    seed=YEARSITE_POOL_SEED,
)


# =============================================================================
# Yearsite Grid Selection Helpers
# =============================================================================


def get_yearsite_indices(
    sites: list[int] | range, years: list[int] | range, num_years: int = 40
) -> list[int]:
    """Convert sites x years grid to flat yearsite indices."""
    return [s * num_years + y for s in sites for y in years]


# Training yearsite indices (nested subsets)
TrainYearsiteIndices_1k = builds(get_yearsite_indices, sites=range(2), years=range(8))
TrainYearsiteIndices_8k = builds(get_yearsite_indices, sites=range(2), years=range(16))
TrainYearsiteIndices_64k = builds(get_yearsite_indices, sites=range(4), years=range(16))
TrainYearsiteIndices_512k = builds(
    get_yearsite_indices, sites=range(4), years=range(32)
)

# Test yearsite indices
TestYearsiteIndices_SeenYears = builds(
    get_yearsite_indices, sites=range(4, 8), years=range(16)
)
TestYearsiteIndices_SeenSites = builds(
    get_yearsite_indices, sites=range(4), years=range(32, 40)
)
TestYearsiteIndices_Unseen = builds(
    get_yearsite_indices, sites=range(4, 8), years=range(32, 40)
)

# Validation yearsite indices
ValYearsiteIndices = builds(
    get_yearsite_indices, sites=range(4, 6), years=range(35, 40)
)
