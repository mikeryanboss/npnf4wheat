"""Pools for factorial experimental designs.

This module provides GenotypePool and YearsitePool classes for creating
factorial combinations of genotypes and yearsite environments.
"""

from npnf.data.pools.genotype_pool import (
    GenotypeParamBounds,
    GenotypeParams,
    GenotypePool,
)
from npnf.data.pools.yearsite_pool import YearsitePool

__all__ = ["GenotypeParamBounds", "GenotypeParams", "GenotypePool", "YearsitePool"]
