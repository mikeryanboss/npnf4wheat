"""Covariate units and MMD blocks shared by the blocked Sig-MMD and CSig-MMD scorers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np


class UnitScope(StrEnum):
    """Which conditions share one MMD unit, set by the covariates a model sees."""

    GLOBAL = "global"
    ENVIRONMENT = "environment"
    GENOTYPE = "genotype"
    CONDITION = "condition"

    @classmethod
    def from_config(cls, config: dict | None, method_path: Path) -> UnitScope:
        """Scope of a method directory from its ``use_temperature``/``use_marker``."""
        if config is None or not {"use_temperature", "use_marker"} <= config.keys():
            msg = (
                "Sig-MMD scoring requires use_temperature and use_marker in "
                f"{method_path / 'config.json'}"
            )
            raise ValueError(msg)
        return {
            (False, False): cls.GLOBAL,
            (True, False): cls.ENVIRONMENT,
            (False, True): cls.GENOTYPE,
            (True, True): cls.CONDITION,
        }[(bool(config["use_temperature"]), bool(config["use_marker"]))]

    def unit_id(self, gid: str, ys: str) -> str:
        """Id of the unit that holds condition ``(gid, ys)``."""
        return {
            UnitScope.GLOBAL: str(UnitScope.GLOBAL),
            UnitScope.ENVIRONMENT: ys,
            UnitScope.GENOTYPE: gid,
            UnitScope.CONDITION: f"{gid}::{ys}",
        }[self]


@dataclass(frozen=True)
class UnitMember:
    genotype_id: str
    yearsite_uid: str
    condition_index: int


@dataclass(frozen=True)
class Selection:
    condition_index: int
    draw_index: int


def unit_seed(seed: int, unit_scope: UnitScope, unit_id: str) -> int:
    """Deterministic per-unit seed of the block sampling."""
    payload = f"{seed}\0{unit_scope}\0{unit_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def build_unit_blocks(
    members: Sequence[UnitMember],
    *,
    draws_per_member: int,
    unit_scope: UnitScope,
    unit_id: str,
    n_blocks: int,
    block_size: int,
    seed: int,
) -> tuple[tuple[Selection, ...], ...]:
    """Split the ``draws_per_member`` draws of every member of a unit into MMD blocks.

    Blocks hold ``block_size`` draws, or all draws of a smaller unit. Each pass
    draws as many complete blocks as the unit has draws for (at most the blocks
    still missing), balanced across members and without replacement, and cuts them
    apart. Passes repeat until ``n_blocks`` blocks exist. A unit with enough draws
    therefore gets disjoint blocks from one pass, and no block holds a draw twice.
    """
    rng = np.random.default_rng(unit_seed(seed, unit_scope, unit_id))
    total_draws = len(members) * draws_per_member
    size = min(block_size, total_draws)
    blocks: list[tuple[Selection, ...]] = []
    while len(blocks) < n_blocks:
        count = min(total_draws // size, n_blocks - len(blocks))
        selections = _balanced_selections(members, draws_per_member, count * size, rng)
        blocks.extend(
            tuple(selections[index * size : (index + 1) * size])
            for index in range(count)
        )
    return tuple(blocks)


def _balanced_selections(
    members: Sequence[UnitMember],
    draws_per_member: int,
    selected: int,
    rng: np.random.Generator,
) -> list[Selection]:
    """Shuffled ``selected`` distinct draws, spread evenly over the members."""
    counts = np.full(len(members), selected // len(members))
    remaining = selected - int(counts.sum())
    if remaining:
        counts[rng.permutation(len(members))[:remaining]] += 1
    selections = [
        Selection(condition_index=member.condition_index, draw_index=int(draw))
        for member, count in zip(members, counts, strict=True)
        if count
        for draw in rng.choice(draws_per_member, size=count, replace=False)
    ]
    rng.shuffle(selections)
    return selections
