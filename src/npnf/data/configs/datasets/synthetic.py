"""Hydra-zen configs for SyntheticDataset."""

from npnf.configs.utils import builds
from npnf.data.configs.datasets.pools import (
    GENOTYPE_POOL_SEED,
    YEARSITE_POOL_SOURCE,
    GenotypePoolConfig,
    TestYearsiteIndices_SeenSites,
    TestYearsiteIndices_SeenYears,
    TestYearsiteIndices_Unseen,
    TrainYearsiteIndices_1k,
    TrainYearsiteIndices_8k,
    TrainYearsiteIndices_64k,
    TrainYearsiteIndices_512k,
    ValYearsiteIndices,
    YearsitePoolConfig,
)
from npnf.data.datasets.synthetic import SyntheticDataset

# =============================================================================
# Index Layout (8 sites × 40 years = 320 yearsites)
# =============================================================================
# Flat index: yearsite_index = site_index * 40 + year_index
#
# Sites 0-3: training (4 sites)
# Sites 4-7: test/val (4 sites)
# Years 0-31: training (32 years)
# Years 32-39: test/val (8 years)
#
# Training dataset sizes (nested subsets, power-of-2):
# | Config | Genotypes | Sites | Years | Yearsites | Samples |
# |--------|-----------|-------|-------|-----------|---------|
# | 1k     | 64        | 0-1   | 0-7   | 16        | 1,024   |
# | 8k     | 256       | 0-1   | 0-15  | 32        | 8,192   |
# | 64k    | 1,024     | 0-3   | 0-15  | 64        | 65,536  |
# | 512k   | 4,096     | 0-3   | 0-31  | 128       | 524,288 |

# =============================================================================
# Base Dataset Config
# =============================================================================

BaseSyntheticDatasetConfig = builds(
    SyntheticDataset,
    genotype_pool=GenotypePoolConfig,
    yearsite_pool=YearsitePoolConfig,
    # Pool identity scalars. Must match the defaults baked into
    # GenotypePoolConfig / YearsitePoolConfig so the dataset cache key
    # reflects the pool these configs would produce.
    genotype_pool_seed=GENOTYPE_POOL_SEED,
    yearsite_pool_source=YEARSITE_POOL_SOURCE,
    scale_noise=0.05,
    enable_lodging=True,
    eval_mode=False,
    cache_dir="dataset_cache",
    num_pre_season_points=10,
    num_post_season_points=10,
    lodging_height_clamp=2.0,
    lodging_weibull_shape=7.0,
    # Previous: lodging_weibull_scale:
    #   1.5 (2.7%),
    #   1.4 (~4.3%),
    #   1.3 (~7.0%),
    #   1.2 (~11.7%),
    #   1.1 (~19.6%)
    #   on the 512k train split
    lodging_weibull_scale=1.1,
    lodging_weibull_offset=0.0,
    lodging_severity_min=0.25,
    lodging_severity_max=0.8,
    lodging_transition_steps_min=0,
    lodging_transition_steps_max=20,
    n_lodging_draws=1,
    lodging_seed=43,
)


# =============================================================================
# Training Dataset Size Variants
# =============================================================================

# 1k: 64 genotypes × 16 yearsites = 1,024 samples
TrainConfig_1k = builds(
    SyntheticDataset,
    genotype_indices=range(64),
    yearsite_indices=TrainYearsiteIndices_1k,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

# 8k: 256 genotypes × 32 yearsites = 8,192 samples
TrainConfig_8k = builds(
    SyntheticDataset,
    genotype_indices=range(256),
    yearsite_indices=TrainYearsiteIndices_8k,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

# 64k: 1024 genotypes × 64 yearsites = 65,536 samples
TrainConfig_64k = builds(
    SyntheticDataset,
    genotype_indices=range(1024),
    yearsite_indices=TrainYearsiteIndices_64k,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

# 512k: 4096 genotypes × 128 yearsites = 524,288 samples
TrainConfig_512k = builds(
    SyntheticDataset,
    genotype_indices=range(4096),
    yearsite_indices=TrainYearsiteIndices_512k,
    builds_bases=(BaseSyntheticDatasetConfig,),
)


# =============================================================================
# Validation Dataset Config
# =============================================================================

ValConfig = builds(
    SyntheticDataset,
    genotype_indices=range(9000, 9100),
    yearsite_indices=ValYearsiteIndices,
    eval_mode=True,
    n_lodging_draws=1,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)


# =============================================================================
# Test Dataset Configs
# =============================================================================

TestConfig_Plot = builds(
    SyntheticDataset,
    genotype_indices=range(1000),
    yearsite_indices=TrainYearsiteIndices_64k,
    eval_mode=True,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Genotype = builds(
    SyntheticDataset,
    genotype_indices=range(8000, 9000),
    yearsite_indices=TrainYearsiteIndices_64k,
    eval_mode=True,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Site = builds(
    SyntheticDataset,
    genotype_indices=range(1000),
    yearsite_indices=TestYearsiteIndices_SeenYears,
    eval_mode=True,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Year = builds(
    SyntheticDataset,
    genotype_indices=range(1000),
    yearsite_indices=TestYearsiteIndices_SeenSites,
    eval_mode=True,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Environment = builds(
    SyntheticDataset,
    genotype_indices=range(1000),
    yearsite_indices=TestYearsiteIndices_Unseen,
    eval_mode=True,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Unseen = builds(
    SyntheticDataset,
    genotype_indices=range(8000, 9000),
    yearsite_indices=TestYearsiteIndices_Unseen,
    eval_mode=True,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)


TestConfig_Plot_NoLodging = builds(
    SyntheticDataset,
    genotype_indices=range(1000),
    yearsite_indices=TrainYearsiteIndices_64k,
    eval_mode=True,
    enable_lodging=False,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Genotype_NoLodging = builds(
    SyntheticDataset,
    genotype_indices=range(8000, 9000),
    yearsite_indices=TrainYearsiteIndices_64k,
    eval_mode=True,
    enable_lodging=False,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Site_NoLodging = builds(
    SyntheticDataset,
    genotype_indices=range(1000),
    yearsite_indices=TestYearsiteIndices_SeenYears,
    eval_mode=True,
    enable_lodging=False,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Year_NoLodging = builds(
    SyntheticDataset,
    genotype_indices=range(1000),
    yearsite_indices=TestYearsiteIndices_SeenSites,
    eval_mode=True,
    enable_lodging=False,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Environment_NoLodging = builds(
    SyntheticDataset,
    genotype_indices=range(1000),
    yearsite_indices=TestYearsiteIndices_Unseen,
    eval_mode=True,
    enable_lodging=False,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)

TestConfig_Unseen_NoLodging = builds(
    SyntheticDataset,
    genotype_indices=range(8000, 9000),
    yearsite_indices=TestYearsiteIndices_Unseen,
    eval_mode=True,
    enable_lodging=False,
    n_lodging_draws=64,
    lodging_seed=42,
    builds_bases=(BaseSyntheticDatasetConfig,),
)


# =============================================================================
# Seed-B Variants (independent lodging draws for oracle-self baseline)
# =============================================================================
# Only lodging_seed differs from the primary test configs; lodging_seed is
# part of SyntheticDataset._compute_cache_key so caches don't collide. Same
# (genotype_id, yearsite_uid) pairs as the non-SeedB siblings, so id_to_index
# alignment in sig_mmd.py is preserved. No-lodging variants are intentionally
# omitted: without lodging, draws are deterministic per condition.

_SEED_B_LODGING_SEED = 142

TestConfig_Plot_SeedB = builds(
    SyntheticDataset, lodging_seed=_SEED_B_LODGING_SEED, builds_bases=(TestConfig_Plot,)
)

TestConfig_Genotype_SeedB = builds(
    SyntheticDataset,
    lodging_seed=_SEED_B_LODGING_SEED,
    builds_bases=(TestConfig_Genotype,),
)

TestConfig_Site_SeedB = builds(
    SyntheticDataset, lodging_seed=_SEED_B_LODGING_SEED, builds_bases=(TestConfig_Site,)
)

TestConfig_Year_SeedB = builds(
    SyntheticDataset, lodging_seed=_SEED_B_LODGING_SEED, builds_bases=(TestConfig_Year,)
)

TestConfig_Environment_SeedB = builds(
    SyntheticDataset,
    lodging_seed=_SEED_B_LODGING_SEED,
    builds_bases=(TestConfig_Environment,),
)

TestConfig_Unseen_SeedB = builds(
    SyntheticDataset,
    lodging_seed=_SEED_B_LODGING_SEED,
    builds_bases=(TestConfig_Unseen,),
)


TestConfig_Plot_Lodging1_3 = builds(
    SyntheticDataset, lodging_weibull_scale=1.3, builds_bases=(TestConfig_Plot,)
)

TestConfig_Plot_Lodging1_3_SeedB = builds(
    SyntheticDataset, lodging_weibull_scale=1.3, builds_bases=(TestConfig_Plot_SeedB,)
)

TestConfig_Genotype_Lodging1_3 = builds(
    SyntheticDataset, lodging_weibull_scale=1.3, builds_bases=(TestConfig_Genotype,)
)

TestConfig_Genotype_Lodging1_3_SeedB = builds(
    SyntheticDataset,
    lodging_weibull_scale=1.3,
    builds_bases=(TestConfig_Genotype_SeedB,),
)

TestConfig_Environment_Lodging1_3 = builds(
    SyntheticDataset, lodging_weibull_scale=1.3, builds_bases=(TestConfig_Environment,)
)

TestConfig_Environment_Lodging1_3_SeedB = builds(
    SyntheticDataset,
    lodging_weibull_scale=1.3,
    builds_bases=(TestConfig_Environment_SeedB,),
)

TestConfig_Unseen_Lodging1_3 = builds(
    SyntheticDataset, lodging_weibull_scale=1.3, builds_bases=(TestConfig_Unseen,)
)

TestConfig_Unseen_Lodging1_3_SeedB = builds(
    SyntheticDataset, lodging_weibull_scale=1.3, builds_bases=(TestConfig_Unseen_SeedB,)
)

TestConfig_Plot_Lodging1_5 = builds(
    SyntheticDataset, lodging_weibull_scale=1.5, builds_bases=(TestConfig_Plot,)
)

TestConfig_Plot_Lodging1_5_SeedB = builds(
    SyntheticDataset, lodging_weibull_scale=1.5, builds_bases=(TestConfig_Plot_SeedB,)
)

TestConfig_Genotype_Lodging1_5 = builds(
    SyntheticDataset, lodging_weibull_scale=1.5, builds_bases=(TestConfig_Genotype,)
)

TestConfig_Genotype_Lodging1_5_SeedB = builds(
    SyntheticDataset,
    lodging_weibull_scale=1.5,
    builds_bases=(TestConfig_Genotype_SeedB,),
)

TestConfig_Environment_Lodging1_5 = builds(
    SyntheticDataset, lodging_weibull_scale=1.5, builds_bases=(TestConfig_Environment,)
)

TestConfig_Environment_Lodging1_5_SeedB = builds(
    SyntheticDataset,
    lodging_weibull_scale=1.5,
    builds_bases=(TestConfig_Environment_SeedB,),
)

TestConfig_Unseen_Lodging1_5 = builds(
    SyntheticDataset, lodging_weibull_scale=1.5, builds_bases=(TestConfig_Unseen,)
)

TestConfig_Unseen_Lodging1_5_SeedB = builds(
    SyntheticDataset, lodging_weibull_scale=1.5, builds_bases=(TestConfig_Unseen_SeedB,)
)


TestConfig_Plot_Noise0_02 = builds(
    SyntheticDataset, scale_noise=0.02, builds_bases=(TestConfig_Plot,)
)

TestConfig_Plot_Noise0_02_SeedB = builds(
    SyntheticDataset, scale_noise=0.02, builds_bases=(TestConfig_Plot_SeedB,)
)

TestConfig_Genotype_Noise0_02 = builds(
    SyntheticDataset, scale_noise=0.02, builds_bases=(TestConfig_Genotype,)
)

TestConfig_Genotype_Noise0_02_SeedB = builds(
    SyntheticDataset, scale_noise=0.02, builds_bases=(TestConfig_Genotype_SeedB,)
)

TestConfig_Environment_Noise0_02 = builds(
    SyntheticDataset, scale_noise=0.02, builds_bases=(TestConfig_Environment,)
)

TestConfig_Environment_Noise0_02_SeedB = builds(
    SyntheticDataset, scale_noise=0.02, builds_bases=(TestConfig_Environment_SeedB,)
)

TestConfig_Unseen_Noise0_02 = builds(
    SyntheticDataset, scale_noise=0.02, builds_bases=(TestConfig_Unseen,)
)

TestConfig_Unseen_Noise0_02_SeedB = builds(
    SyntheticDataset, scale_noise=0.02, builds_bases=(TestConfig_Unseen_SeedB,)
)
