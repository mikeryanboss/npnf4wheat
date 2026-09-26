"""Forward-run the calibrated genotype pool under each FIP1 year's temperature."""

from __future__ import annotations

import numpy as np
import torch

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.forward import _run_bspline_forward
from npnf.calibration.height.genotype_variability.traits import elongation_rate
from npnf.data.pools import GenotypePool


def simulate_pool_traits(
    year_temperatures: dict[int, torch.Tensor], pool: GenotypePool, dates: HeightDates
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Simulate every genotype in the pool under every year's temperature.

    Returns per-year arrays of (max height in metres, growth-cessation day,
    elongation rate in metres per day).
    """
    max_heights: dict[int, np.ndarray] = {}
    growth_end: dict[int, np.ndarray] = {}
    rates: dict[int, np.ndarray] = {}
    parameters = pool.params
    for year, temperatures in sorted(year_temperatures.items()):
        batch = temperatures.unsqueeze(0).expand(len(pool), -1, -1)
        trajectories = _run_bspline_forward(batch, parameters, dates=dates)
        max_heights[year] = trajectories.max(dim=-1).values.detach().numpy()
        peak_index = trajectories.argmax(dim=-1).detach().numpy()
        growth_end[year] = peak_index + dates.day_temperature_start
        curves = trajectories.detach().numpy()
        days = np.arange(curves.shape[-1]) + dates.day_temperature_start
        rates[year] = np.array([elongation_rate(days, curve) for curve in curves])
    return max_heights, growth_end, rates
