"""Registry of the synthetic test sets and their splits.

`legacy` is the default test set. It holds the old test configs under their
original dataloader names, so that existing results and saved prediction configs
keep resolving to the old oracle. `shifted` is optional. `lodging1_3` and
`lodging1_5` are the four legacy design splits at lodging scales 1.3 and 1.5,
and `noise0_02` the same splits at observation noise 0.02 m.

Every split fills a `role` of the 2x2 design where it has one: seen (training
genotypes and environments), geno (new genotypes), env (new environments) and
unseen (both new). Scripts that loop over the four design splits use
`design_splits`, so that the same code runs on either test set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Any

from npnf.data.configs.datasets import shifted, synthetic


class SyntheticTestSet(StrEnum):
    LEGACY = "legacy"
    SHIFTED = "shifted"
    LODGING1_3 = "lodging1_3"
    LODGING1_5 = "lodging1_5"
    NOISE0_02 = "noise0_02"

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser) -> None:
        """Add the standard ``--test-set`` option (default: legacy)."""
        parser.add_argument(
            "--test-set",
            choices=list(cls),
            default=cls.LEGACY,
            help=(
                "Synthetic test set (default: legacy, the old test splits and their "
                "existing results; shifted is optional)."
            ),
        )


@dataclass(frozen=True)
class SyntheticTestSplit:
    test_set: SyntheticTestSet
    alias: str
    role: str | None
    dataloader_name: str
    split_key: str
    title: str
    oracle: Any
    seed_b: Any | None = None


@cache
def _registry() -> tuple[SyntheticTestSplit, ...]:
    splits = [
        SyntheticTestSplit(
            SyntheticTestSet.SHIFTED,
            alias,
            alias,
            f"synth_shifted_{alias}_dataloaders",
            f"test_{alias}",
            title,
            oracle,
            seed_b,
        )
        for alias, title, oracle, seed_b in (
            (
                "seen",
                "Seen",
                shifted.ShiftedConfig_Seen,
                shifted.ShiftedConfig_Seen_SeedB,
            ),
            (
                "geno",
                "Genotype",
                shifted.ShiftedConfig_Geno,
                shifted.ShiftedConfig_Geno_SeedB,
            ),
            (
                "env",
                "Environment",
                shifted.ShiftedConfig_Env,
                shifted.ShiftedConfig_Env_SeedB,
            ),
            (
                "unseen",
                "Unseen",
                shifted.ShiftedConfig_Unseen,
                shifted.ShiftedConfig_Unseen_SeedB,
            ),
        )
    ]
    for alias, role in (
        ("plot", "seen"),
        ("genotype", "geno"),
        ("site", None),
        ("year", None),
        ("environment", "env"),
        ("unseen", "unseen"),
    ):
        title = alias.title()
        splits.append(
            SyntheticTestSplit(
                SyntheticTestSet.LEGACY,
                alias,
                role,
                f"synth_test_{alias}_dataloaders",
                f"test_{alias}",
                title,
                getattr(synthetic, f"TestConfig_{title}"),
                getattr(synthetic, f"TestConfig_{title}_SeedB"),
            )
        )
        splits.append(
            SyntheticTestSplit(
                SyntheticTestSet.LEGACY,
                f"{alias}_no_lodging",
                None,
                f"synth_test_{alias}_no_lodging_dataloaders",
                f"test_{alias}",
                f"{title} (no lodging)",
                getattr(synthetic, f"TestConfig_{title}_NoLodging"),
            )
        )
    for test_set in (
        SyntheticTestSet.LODGING1_3,
        SyntheticTestSet.LODGING1_5,
        SyntheticTestSet.NOISE0_02,
    ):
        suffix = test_set.value.title()
        for alias, role in (
            ("plot", "seen"),
            ("genotype", "geno"),
            ("environment", "env"),
            ("unseen", "unseen"),
        ):
            title = alias.title()
            splits.append(
                SyntheticTestSplit(
                    test_set,
                    alias,
                    role,
                    f"synth_test_{alias}_{test_set}_dataloaders",
                    f"test_{alias}",
                    title,
                    getattr(synthetic, f"TestConfig_{title}_{suffix}"),
                    getattr(synthetic, f"TestConfig_{title}_{suffix}_SeedB"),
                )
            )
    return tuple(splits)


def splits_of(
    test_set: SyntheticTestSet | str = SyntheticTestSet.LEGACY,
) -> tuple[SyntheticTestSplit, ...]:
    """All splits of a test set, in registry order."""
    return tuple(split for split in _registry() if split.test_set == test_set)


def design_splits(
    test_set: SyntheticTestSet | str = SyntheticTestSet.LEGACY,
) -> tuple[SyntheticTestSplit, ...]:
    """The four splits of the 2x2 design, ordered seen, geno, env, unseen."""
    by_role = {split.role: split for split in splits_of(test_set) if split.role}
    return tuple(by_role[role] for role in ("seen", "geno", "env", "unseen"))


def design_aliases(
    test_set: SyntheticTestSet | str = SyntheticTestSet.LEGACY,
) -> tuple[str, ...]:
    """Aliases of the four design splits, ordered seen, geno, env, unseen."""
    return tuple(split.alias for split in design_splits(test_set))


def split_named(
    alias: str, test_set: SyntheticTestSet | str = SyntheticTestSet.LEGACY
) -> SyntheticTestSplit:
    """Look up a split by its alias within a test set."""
    for split in splits_of(test_set):
        if split.alias == alias:
            return split
    known = [split.alias for split in splits_of(test_set)]
    msg = f"Unknown {test_set} test split {alias!r}. Known: {known}"
    raise ValueError(msg)


def split_for_dataloader(dataloader_name: str) -> SyntheticTestSplit:
    """Look up a split of any test set by its dataloader name."""
    for split in _registry():
        if split.dataloader_name == dataloader_name:
            return split
    known = [split.dataloader_name for split in _registry()]
    msg = f"Unknown dataloader_name {dataloader_name!r}. Known: {known}"
    raise ValueError(msg)
