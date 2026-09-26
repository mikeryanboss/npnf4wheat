"""Main generator class for synthetic temperature data.

This module contains:
- `SyntheticTemperatureGenerator`: Main class for generating synthetic data
"""

import os
from pathlib import Path

import numpy as np
import torch

from .generation import (
    _generate_year_batch,
    _prestack_site_params,
    generate_sites,
    generate_years,
)
from .params import (
    CALIBRATION_PARAMS_ENV_VAR,
    TemperatureParams,
    load_temperature_params,
)


class SyntheticTemperatureGenerator:
    """Generator for synthetic temperature data with proper site/year structure.

    Sites have fixed characteristics (offset, weather correlation) while years
    share weather patterns with site-specific deviations.

    Example:
        >>> generator = SyntheticTemperatureGenerator(
        ...     num_sites=10,
        ...     num_years=10,
        ...     seed=42,
        ...     calibration_params_path=(
        ...         "results/calibration/temperature/params_pool.json"
        ...     ),
        ... )
        >>> temps = generator.generate_all()  # (100, 274, 24)
    """

    def __init__(
        self,
        num_sites: int = 10,
        num_years: int = 10,
        num_days: int = 274,
        seed: int = 42,
        start_year: int = 2010,
        temperature_params: TemperatureParams | None = None,
        calibration_params_path: str | Path | None = None,
    ) -> None:
        """Initialize the generator with site and year parameters.

        Args:
            num_sites: Number of synthetic sites to generate
            num_years: Number of synthetic years per site
            num_days: Days per year (default 274, matching real data)
            seed: Random seed for reproducibility
            start_year: First synthetic year ID
            temperature_params: Explicit global temperature parameters. If omitted,
                parameters are loaded from *calibration_params_path* (or env var).
            calibration_params_path: Path to calibration params JSON.
                Used when *temperature_params* is not passed.
        """
        self.num_sites = num_sites
        self.num_years = num_years
        self.num_yearsites = self.num_sites * self.num_years
        self.num_days = num_days
        self.seed = seed
        self.start_year = start_year
        self.rng = np.random.default_rng(seed)

        if temperature_params is None:
            params_path = calibration_params_path or os.environ.get(
                CALIBRATION_PARAMS_ENV_VAR
            )
            if params_path is None:
                msg = (
                    "No calibration_params_path given and "
                    f"{CALIBRATION_PARAMS_ENV_VAR} is not set"
                )
                raise ValueError(msg)
            temperature_params = load_temperature_params(Path(params_path))

        self.temperature_params = temperature_params

        self.sites = generate_sites(
            self.num_sites, self.temperature_params.site, self.rng
        )
        self.years = generate_years(
            self.num_years,
            self.num_days,
            self.start_year,
            self.temperature_params.weather,
            self.temperature_params.year,
            self.rng,
        )

    def generate_all(self) -> torch.Tensor:
        """Generate all site-year combinations as torch tensor.

        Generates all sites per year in vectorized batches.

        The ordering is: all years for site 0, then all years for site 1, etc.

        Returns:
            Tensor of shape (num_sites * num_years, num_days, 24)
        """
        temperatures = np.zeros((self.num_yearsites, self.num_days, 24))

        # Pre-stack site parameters for vectorized computation
        site_parameters = _prestack_site_params(self.sites)

        # Generate year batches (all sites per year, then scatter to site-major order)
        for year_index, year in enumerate(self.years):
            batch = _generate_year_batch(
                self.num_sites,
                year,
                self.temperature_params,
                self.num_days,
                site_parameters,
                self.seed,
                year_index,
            )  # (num_sites, num_days, 24)
            for site_index in range(self.num_sites):
                flat = site_index * self.num_years + year_index
                temperatures[flat] = batch[site_index]

        return torch.from_numpy(temperatures).float()
