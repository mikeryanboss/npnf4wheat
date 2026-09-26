"""Utility classes for dataset sampling.

This module provides helper classes for yearsite-based weighted sampling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class YearsiteIndex:
    """Index mapping for yearsite-based sampling with weights.

    Attributes:
        yearsite_to_indices: Dict mapping (year, site) tuples to list of dataset indices
        yearsite_weights: Dict mapping (year, site) tuples to sampling weights
        yearsites: List of all (year, site) tuples for indexing
        total_weight: Sum of all yearsite weights for normalization
    """

    yearsite_to_indices: dict[tuple[int, str], list[int]]
    yearsite_weights: dict[tuple[int, str], float]
    yearsites: list[tuple[int, str]]
    total_weight: float

    @classmethod
    def build_uniform_weights(
        cls, years: list[int], sites: list[str], indices: list[int] | None = None
    ) -> "YearsiteIndex":
        """Build yearsite index with uniform weights.

        Args:
            years: List of years for each sample
            sites: List of sites for each sample
            indices: Optional list of indices (defaults to range(len(years)))

        Returns:
            YearsiteIndex with uniform weights (1.0) for all yearsites
        """
        if indices is None:
            indices = list(range(len(years)))

        if len(years) != len(sites) or len(years) != len(indices):
            msg = "Years, sites, and indices must have same length"
            raise ValueError(msg)

        # Group indices by yearsite
        yearsite_to_indices: dict[tuple[int, str], list[int]] = {}
        for index, year, site in zip(indices, years, sites, strict=False):
            yearsite = (year, site)
            if yearsite not in yearsite_to_indices:
                yearsite_to_indices[yearsite] = []
            yearsite_to_indices[yearsite].append(index)

        # Create uniform weights
        yearsites = list(yearsite_to_indices.keys())
        yearsite_weights = dict.fromkeys(yearsites, 1.0)
        total_weight = float(len(yearsites))

        return cls(
            yearsite_to_indices=yearsite_to_indices,
            yearsite_weights=yearsite_weights,
            yearsites=yearsites,
            total_weight=total_weight,
        )

    @classmethod
    def build_inverse_frequency_weights(
        cls, years: list[int], sites: list[str], indices: list[int] | None = None
    ) -> "YearsiteIndex":
        """Build yearsite index with inverse frequency weights.

        Yearsites with fewer samples get higher weights to balance representation.

        Args:
            years: List of years for each sample
            sites: List of sites for each sample
            indices: Optional list of indices (defaults to range(len(years)))

        Returns:
            YearsiteIndex with inverse frequency weights
        """
        if indices is None:
            indices = list(range(len(years)))

        if len(years) != len(sites) or len(years) != len(indices):
            msg = "Years, sites, and indices must have same length"
            raise ValueError(msg)

        # Group indices by yearsite and count samples
        yearsite_to_indices: dict[tuple[int, str], list[int]] = {}
        for index, year, site in zip(indices, years, sites, strict=False):
            yearsite = (year, site)
            if yearsite not in yearsite_to_indices:
                yearsite_to_indices[yearsite] = []
            yearsite_to_indices[yearsite].append(index)

        # Calculate inverse frequency weights
        yearsites = list(yearsite_to_indices.keys())
        yearsite_counts = {ys: len(yearsite_to_indices[ys]) for ys in yearsites}

        # Weight = 1 / count (inverse frequency)
        yearsite_weights = {ys: 1.0 / count for ys, count in yearsite_counts.items()}
        total_weight = sum(yearsite_weights.values())

        return cls(
            yearsite_to_indices=yearsite_to_indices,
            yearsite_weights=yearsite_weights,
            yearsites=yearsites,
            total_weight=total_weight,
        )

    @classmethod
    def build_custom_weights(
        cls,
        years: list[int],
        sites: list[str],
        custom_weights: dict[tuple[int, str], float],
        indices: list[int] | None = None,
    ) -> "YearsiteIndex":
        """Build yearsite index with custom weights.

        Args:
            years: List of years for each sample
            sites: List of sites for each sample
            custom_weights: Dict mapping (year, site) to custom weight values
            indices: Optional list of indices (defaults to range(len(years)))

        Returns:
            YearsiteIndex with custom weights
        """
        if indices is None:
            indices = list(range(len(years)))

        if len(years) != len(sites) or len(years) != len(indices):
            msg = "Years, sites, and indices must have same length"
            raise ValueError(msg)

        # Group indices by yearsite
        yearsite_to_indices: dict[tuple[int, str], list[int]] = {}
        for index, year, site in zip(indices, years, sites, strict=False):
            yearsite = (year, site)
            if yearsite not in yearsite_to_indices:
                yearsite_to_indices[yearsite] = []
            yearsite_to_indices[yearsite].append(index)

        # Use provided custom weights
        yearsites = list(yearsite_to_indices.keys())

        # Validate that all yearsites have weights
        missing_weights = set(yearsites) - set(custom_weights.keys())
        if missing_weights:
            msg = f"Missing weights for yearsites: {missing_weights}"
            raise ValueError(msg)

        # Only include yearsites that exist in data
        yearsite_weights = {ys: custom_weights[ys] for ys in yearsites}
        total_weight = sum(yearsite_weights.values())

        return cls(
            yearsite_to_indices=yearsite_to_indices,
            yearsite_weights=yearsite_weights,
            yearsites=yearsites,
            total_weight=total_weight,
        )

    def sample_yearsite(self, generator: torch.Generator) -> tuple[int, str]:
        """Sample a yearsite according to weights.

        Args:
            generator: PyTorch random generator for reproducibility

        Returns:
            Sampled (year, site) tuple
        """
        # Create probability distribution from weights
        weights_array = np.array([self.yearsite_weights[ys] for ys in self.yearsites])
        probs = weights_array / self.total_weight

        # Sample using torch for consistency
        sample_index = int(
            torch.multinomial(
                torch.from_numpy(probs).float(), num_samples=1, generator=generator
            ).item()
        )

        return self.yearsites[sample_index]

    def sample_index_from_yearsite(
        self, yearsite: tuple[int, str], generator: torch.Generator
    ) -> int:
        """Sample a random index from the given yearsite.

        Args:
            yearsite: (year, site) tuple to sample from
            generator: PyTorch random generator for reproducibility

        Returns:
            Random dataset index from the yearsite
        """
        indices = self.yearsite_to_indices[yearsite]
        if not indices:
            msg = f"No indices found for yearsite {yearsite}"
            raise ValueError(msg)

        # Sample uniformly from indices within the yearsite
        sample_index = int(
            torch.randint(0, len(indices), size=(1,), generator=generator).item()
        )

        return indices[sample_index]

    def get_yearsite_info(self) -> dict[str, Any]:
        """Get summary information about the yearsite distribution.

        Returns:
            Dict with yearsite statistics
        """
        yearsite_counts = {
            ys: len(indices) for ys, indices in self.yearsite_to_indices.items()
        }

        years = sorted({ys[0] for ys in self.yearsites})
        sites = sorted({ys[1] for ys in self.yearsites})

        return {
            "num_yearsites": len(self.yearsites),
            "num_years": len(years),
            "num_sites": len(sites),
            "years": years,
            "sites": sites,
            "yearsite_counts": yearsite_counts,
            "yearsite_weights": dict(self.yearsite_weights),
            "total_samples": sum(yearsite_counts.values()),
            "total_weight": self.total_weight,
        }


def verify_dataset_cache_key(
    method_dir: Path, saved_cache_key: str | None, loaded_cache_key: str
) -> None:
    """Check a saved artifact's dataset identity against the instantiated dataset."""
    if saved_cache_key != loaded_cache_key:
        msg = (
            f"{method_dir}: dataset_identity cache_key mismatch; "
            f"artifact has {saved_cache_key!r}, resolved dataset has "
            f"{loaded_cache_key!r}"
        )
        raise ValueError(msg)
