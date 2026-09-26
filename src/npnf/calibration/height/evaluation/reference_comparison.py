"""In-memory whole-genotype partitions and observed reference scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import torch
from numpy.typing import NDArray

from npnf.metrics.sig_mmd import normalized_sig_mmd


def split_genotypes(
    genotype_ids: NDArray, year: int, split_seed: int
) -> NDArray[np.bool_]:
    """Mark group A; its complement is B, retaining whole replicated genotypes."""
    unique = np.unique(genotype_ids)
    shuffled = np.random.default_rng(split_seed * 10000 + year).permutation(unique)
    return np.isin(genotype_ids, shuffled[: len(shuffled) // 2])


def score_within_year_splits(
    year: int,
    paths: torch.Tensor,
    genotype_ids: NDArray,
    split_seeds: Sequence[int],
    n_shared_days: int,
) -> tuple[NDArray[np.bool_], list[dict[str, int | float]], dict[str, int | float]]:
    """Score the fixed sequence of whole-genotype observed population splits."""
    membership = np.stack(
        [split_genotypes(genotype_ids, year, seed) for seed in split_seeds]
    )
    runs = []
    for split_seed, a in zip(split_seeds, membership, strict=True):
        b = ~a
        runs.append(
            {
                "year": year,
                "split_seed": split_seed,
                "n_a": int(a.sum()),
                "n_b": int(b.sum()),
                "n_genotypes_a": len(np.unique(genotype_ids[a])),
                "n_genotypes_b": len(np.unique(genotype_ids[b])),
                "real_real": normalized_sig_mmd(paths[a], paths[b], max_batch=64),
            }
        )
    real_scores = [row["real_real"] for row in runs]
    n_a = membership.sum(axis=1)
    n_b = len(paths) - n_a
    summary = {
        "year": year,
        "n_real": len(paths),
        "n_genotypes": len(np.unique(genotype_ids)),
        "n_shared_days": n_shared_days,
        "n_splits": len(split_seeds),
        "n_a_min": int(n_a.min()),
        "n_a_max": int(n_a.max()),
        "n_b_min": int(n_b.min()),
        "n_b_max": int(n_b.max()),
        "real_mean": float(np.mean(real_scores)),
        "real_sd": float(np.std(real_scores, ddof=1)),
    }
    return membership, runs, summary


def score_between_year_pairs(
    years: Sequence[int], real_paths: Mapping[int, torch.Tensor], n_shared_days: int
) -> list[dict[str, int | float]]:
    """Score each unordered pair of full observed year populations."""
    rows = []
    for year_a, year_b in combinations(years, 2):
        x, y = real_paths[year_a], real_paths[year_b]
        rows.append(
            {
                "year_a": year_a,
                "year_b": year_b,
                "n_a": len(x),
                "n_b": len(y),
                "n_shared_days": n_shared_days,
                "sig_mmd_squared": normalized_sig_mmd(x, y, max_batch=64),
            }
        )
    return rows
