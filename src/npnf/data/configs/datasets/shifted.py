"""Hydra-zen configs for the frozen shifted synthetic test splits.

The splits are read from the frozen bundle in `dataset_cache/test_sets/shifted`,
relative to the working directory like the other dataset caches, so run from the
repository root. Create it there with
`python -m npnf.scripts.calibration.shifted_test_set run --output-dir
dataset_cache/test_sets/shifted`.
"""

from pathlib import Path

from npnf.configs.utils import builds
from npnf.data.datasets.shifted import load_dataset


def shifted_split_config(
    variant: str, output_dir: str | Path = "dataset_cache/test_sets/shifted"
):
    """Config that loads one split of a frozen bundle."""
    return builds(load_dataset, output_dir=str(output_dir), variant=variant)


ShiftedConfig_Seen = shifted_split_config("seen")
ShiftedConfig_Geno = shifted_split_config("geno")
ShiftedConfig_Env = shifted_split_config("env")
ShiftedConfig_Unseen = shifted_split_config("unseen")

ShiftedConfig_Seen_SeedB = shifted_split_config("seen_seed_b")
ShiftedConfig_Geno_SeedB = shifted_split_config("geno_seed_b")
ShiftedConfig_Env_SeedB = shifted_split_config("env_seed_b")
ShiftedConfig_Unseen_SeedB = shifted_split_config("unseen_seed_b")
