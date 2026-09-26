"""GenotypePool: Fixed pool of synthetic genotypes with reproducible parameters.

This module provides a GenotypePool class that samples and stores genotype parameters
deterministically from a seed. The parameters are normalized to [0,1] for use as
synthetic marker inputs to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from npnf.data.synthetic.height.genotype import (
    GenotypeParamBounds,
    GenotypeParams,
    SyntheticGenotypeGenerator,
)

if TYPE_CHECKING:
    from npnf.data.synthetic.height.params import HeightPoolParams

__all__ = ["GenotypeParamBounds", "GenotypeParams", "GenotypePool"]


def _normalize_tensor(
    params: torch.Tensor, bounds: GenotypeParamBounds
) -> torch.Tensor:
    """Normalize a batch of parameters to [0, 1] using bounds.

    Args:
        params: Tensor of shape (N, num_params) with raw parameter values.
        bounds: GenotypeParamBounds with (min, max) for each parameter.

    Returns:
        Tensor of shape (N, num_params) with normalized values in [0, 1].
    """
    lo = []
    hi = []
    for f in fields(GenotypeParams):
        bound = getattr(bounds, f.name)
        lo.append(bound[0])
        hi.append(bound[1])
    lo_tensor = torch.tensor(lo, dtype=params.dtype, device=params.device)
    hi_tensor = torch.tensor(hi, dtype=params.dtype, device=params.device)
    return (params - lo_tensor) / (hi_tensor - lo_tensor + 1e-8)


@dataclass
class GenotypePool:
    """Fixed pool of synthetic genotypes with reproducible parameters.

    Stores parameters as tensors for efficient batch access. Individual genotypes
    can be retrieved as GenotypeParams dataclasses via __getitem__.

    Attributes:
        _params: Raw parameters tensor of shape (num_genotypes, num_params)
        _markers: Normalized parameters tensor (num_genotypes, num_params), [0,1]
        bounds: GenotypeParamBounds for normalization
        genotype_ids: List of genotype identifiers like ["G_0000", "G_0001", ...]
        seed: Random seed used for sampling
    """

    _params: torch.Tensor
    _markers: torch.Tensor
    bounds: GenotypeParamBounds
    genotype_ids: list[str]
    seed: int
    _field_name_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build field name to index mapping."""
        self._field_name_to_index = {
            f.name: i for i, f in enumerate(fields(GenotypeParams))
        }

    @classmethod
    def sample(
        cls, num_genotypes: int, seed: int, *, height_pool_params: HeightPoolParams
    ) -> GenotypePool:
        """Sample genotypes deterministically from seed.

        Args:
            num_genotypes: Number of genotypes to sample
            seed: Random seed for reproducibility
            height_pool_params: Typed pool sampling parameters.

        Returns:
            GenotypePool with sampled parameters
        """
        generator = SyntheticGenotypeGenerator(height_pool_params)
        params, bounds = generator.sample(num_genotypes, seed)
        markers = _normalize_tensor(params, bounds)
        genotype_ids = [f"G_{i:04d}" for i in range(num_genotypes)]

        return cls(
            _params=params,
            _markers=markers,
            bounds=bounds,
            genotype_ids=genotype_ids,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.genotype_ids)

    def __getitem__(self, index: int) -> GenotypeParams:
        """Get a single genotype by index.

        Returns:
            GenotypeParams instance for the specified index
        """
        return GenotypeParams.from_tensor(self._params[index])

    @property
    def params(self) -> torch.Tensor:
        """Raw parameters tensor of shape (num_genotypes, num_params)."""
        return self._params

    @property
    def markers(self) -> torch.Tensor:
        """Normalized parameters (num_genotypes, num_params), [0,1]."""
        return self._markers

    def get_param(self, name: str) -> torch.Tensor:
        """Get a single parameter across all genotypes.

        Args:
            name: Parameter name (e.g., 'r_max', 'temp_min_start')

        Returns:
            Tensor of shape (num_genotypes,) with the parameter values
        """
        index = self._field_name_to_index[name]
        return self._params[:, index]

    def get_param_dict(self, index: int) -> dict[str, torch.Tensor]:
        """Get parameters as a dictionary for a single genotype.

        Args:
            index: Genotype index

        Returns:
            Dictionary mapping parameter names to scalar tensors
        """
        row = self._params[index]
        return {name: row[i] for name, i in self._field_name_to_index.items()}

    def subset(self, indices: list[int] | range) -> GenotypePool:
        """Return new pool with subset of genotypes.

        Args:
            indices: Indices of genotypes to include

        Returns:
            New GenotypePool with only the specified genotypes
        """
        indices_tensor = torch.tensor(list(indices), dtype=torch.long)
        return GenotypePool(
            _params=self._params[indices_tensor],
            _markers=self._markers[indices_tensor],
            bounds=self.bounds,
            genotype_ids=[self.genotype_ids[i] for i in indices],
            seed=self.seed,
        )

    def save(self, path: Path | str) -> None:
        """Save pool to disk.

        Args:
            path: Path to save the pool
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        field_names = GenotypeParams.field_names()
        genotypes_data = [
            {name: float(self._params[i, j]) for j, name in enumerate(field_names)}
            for i in range(self._params.shape[0])
        ]
        torch.save(
            {
                "genotypes": genotypes_data,
                "bounds": self.bounds.to_dict(),
                "genotype_ids": self.genotype_ids,
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> GenotypePool:
        """Load pool from disk.

        Args:
            path: Path to load the pool from

        Returns:
            Loaded GenotypePool
        """
        data = torch.load(path, weights_only=False)
        bounds = GenotypeParamBounds.from_dict(data["bounds"])
        field_names = GenotypeParams.field_names()
        num_genotypes = len(data["genotypes"])
        params = torch.zeros(num_genotypes, len(field_names), dtype=torch.float32)
        for i, g in enumerate(data["genotypes"]):
            for j, name in enumerate(field_names):
                params[i, j] = g[name]

        markers = _normalize_tensor(params, bounds)

        return cls(
            _params=params,
            _markers=markers,
            bounds=bounds,
            genotype_ids=data["genotype_ids"],
            seed=data["seed"],
        )
