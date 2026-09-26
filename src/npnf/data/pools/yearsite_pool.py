"""YearsitePool: Fixed pool of yearsite temperature profiles.

This module provides a YearsitePool class that stores temperature profiles
from real weather data or synthetic sources. The profiles are used with
GenotypePool to create factorial experimental designs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import einops as EO
import numpy as np
import torch

from npnf.data.synthetic.temperature import SyntheticTemperatureGenerator


@dataclass
class YearsitePool:
    """Fixed pool of yearsite temperature profiles.

    Growth start and growth end are computed dynamically from Arrhenius
    accumulation in the dataset (see MODEL.md for details).

    Attributes:
        temperatures: Temperature tensor of shape (num_yearsites, 274, 24)
            where 274 is the number of days and 24 is hours per day
        yearsite_ids: List of yearsite identifiers like ["Lindau_2016", ...]
        source: Origin of the data, either "weather_data" or "synthetic"
        seed: Random seed if synthetic, None for weather data
    """

    temperatures: torch.Tensor
    yearsite_ids: list[str]
    source: str
    seed: int | None

    @classmethod
    def from_weather_data(cls, weather_data: dict[str, Any]) -> "YearsitePool":
        """Create pool from real weather data dict.

        Args:
            weather_data: Dictionary from load_weather_data() or instantiated config
                Must contain 'temperature_values' and 'yearsite_uid' keys

        Returns:
            YearsitePool with temperature profiles from weather data
        """
        # Reshape temperature values from (b, d*h) to (b, d, h)
        temperatures = EO.rearrange(
            torch.as_tensor(np.asarray(weather_data["temperature_values"])),
            "b (d h) -> b d h",
            h=24,
        )
        yearsite_ids = list(weather_data["yearsite_uid"])

        return cls(
            temperatures=temperatures,
            yearsite_ids=yearsite_ids,
            source="weather_data",
            seed=None,
        )

    @classmethod
    def sample_synthetic(
        cls,
        num_sites: int,
        num_years: int,
        seed: int,
        calibration_params_path: str | Path | None = None,
        # Optional: expose generator parameters
        num_days: int = 274,
        start_year: int = 2010,
        **kwargs: Any,
    ) -> "YearsitePool":
        """Generate synthetic temperature profiles.

        Creates num_sites x num_years yearsite combinations with
        calibrated temperature courses matching Swiss weather patterns.

        Args:
            num_sites: Number of synthetic sites to generate
            num_years: Number of synthetic years per site
            seed: Random seed for reproducibility
            calibration_params_path: Path to temperature calibration params JSON.
                If omitted, SyntheticTemperatureGenerator resolves from env var.
            num_days: Days per year (default 274, matching real data)
            start_year: First synthetic year ID (default 2010)
            **kwargs: Additional parameters passed to SyntheticTemperatureGenerator

        Returns:
            YearsitePool with synthetic temperature profiles
        """
        generator = SyntheticTemperatureGenerator(
            num_sites=num_sites,
            num_years=num_years,
            num_days=num_days,
            seed=seed,
            start_year=start_year,
            calibration_params_path=calibration_params_path,
            **kwargs,
        )

        # Generate all temperatures as torch tensor
        temperatures = generator.generate_all()

        # Generate yearsite IDs: "Synth{site:02d}_{year}"
        yearsite_ids = []
        for site_index in range(num_sites):
            for year_index in range(num_years):
                year_id = start_year + year_index
                yearsite_ids.append(f"Synth{site_index:02d}_{year_id}")

        return cls(
            temperatures=temperatures,
            yearsite_ids=yearsite_ids,
            source="synthetic",
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.yearsite_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get temperature profile for a single yearsite.

        Args:
            index: Yearsite index

        Returns:
            Temperature tensor of shape (274, 24)
        """
        return self.temperatures[index]

    def get_yearsite_id(self, index: int) -> str:
        """Get yearsite ID for an index.

        Args:
            index: Yearsite index

        Returns:
            Yearsite identifier string
        """
        return self.yearsite_ids[index]

    def subset(self, indices: list[int] | range) -> "YearsitePool":
        """Return new pool with subset of yearsites.

        Args:
            indices: Indices of yearsites to include

        Returns:
            New YearsitePool with only the specified yearsites
        """
        indices_list = list(indices)
        return YearsitePool(
            temperatures=self.temperatures[indices_list],
            yearsite_ids=[self.yearsite_ids[i] for i in indices_list],
            source=self.source,
            seed=self.seed,
        )

    def save(self, path: Path | str) -> None:
        """Save pool to disk.

        Args:
            path: Path to save the pool
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "temperatures": self.temperatures,
                "yearsite_ids": self.yearsite_ids,
                "source": self.source,
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "YearsitePool":
        """Load pool from disk.

        Args:
            path: Path to load the pool from

        Returns:
            Loaded YearsitePool
        """
        data = torch.load(path, weights_only=False)
        return cls(**data)
