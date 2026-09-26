"""Empirical reference splits preserve replicated genotype groups."""

from __future__ import annotations

import numpy as np

from npnf.calibration.height.evaluation.reference_comparison import split_genotypes


def test_uneven_genotype_replication_stays_disjoint_and_exhaustive() -> None:
    genotype_ids = np.array([30, 10, 30, 50, 20, 30, 40, 10, 50, 50, 50])
    a = split_genotypes(genotype_ids, 2017, 617)
    b = ~a
    genotypes_a = set(genotype_ids[a])
    genotypes_b = set(genotype_ids[b])
    assert len(genotypes_a) == 2
    assert len(genotypes_b) == 3
    assert genotypes_a.isdisjoint(genotypes_b)
    assert genotypes_a | genotypes_b == set(genotype_ids)
    # Replicate order must not change membership of any genotype.
    np.testing.assert_array_equal(
        split_genotypes(genotype_ids[::-1], 2017, 617), a[::-1]
    )
