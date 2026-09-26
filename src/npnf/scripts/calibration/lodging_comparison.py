"""Fit an empirical lodging rate and save aligned FIP1 lodging simulations."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

import datasets
import numpy as np
import torch
from torch import Tensor

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.evaluation.lodging_comparison import (
    DetectorOptions,
    detect_lodged_plots,
    fit_weibull_scale,
)
from npnf.calibration.height.evaluation.visualize import _compute_height_trajectories
from npnf.data import fip1_day_grid as grid_protocol
from npnf.data.configs.datasets.synthetic import BaseSyntheticDatasetConfig
from npnf.data.datasets.fip1 import Fip1Facts, get_heights_dataset
from npnf.data.pools import GenotypePool
from npnf.data.synthetic.height.params import load_height_pool_params
from npnf.data.synthetic.lodging import apply_lodging_from_params, sample_lodging_params
from npnf.scripts.utils.outputs import save_json


class LodgingOptions(TypedDict):
    lodging_height_clamp: float
    lodging_weibull_shape: float
    lodging_weibull_scale: float
    lodging_weibull_offset: float
    lodging_severity_min: float
    lodging_severity_max: float
    lodging_transition_steps_min: int
    lodging_transition_steps_max: int


@dataclass
class ObservedYearPopulation:
    plots: datasets.Dataset
    real: Tensor
    dense_days: Tensor
    dense_real: Tensor
    detected_count: int


@dataclass(frozen=True)
class PooledLodgingFit:
    scale: float
    target_count: int
    n_plots: int
    target_rate: float
    expected_count: float


def prepare_observed_years(
    dataset: datasets.Dataset,
    years: Sequence[int],
    shared_grid: grid_protocol.SharedDayGrid,
    detector: DetectorOptions,
) -> tuple[dict[int, ObservedYearPopulation], dict[str, object]]:
    """Select plots with all required dates, preserving order and measurements."""
    prepared = {}
    population_selection = {}
    source_years = np.asarray(dataset["harvest_year"])
    source_uids = np.asarray(dataset["plot_uid"], dtype=str)
    for year in years:
        yearly = grid_protocol.select_year(dataset, year, shared_grid)
        retained = {str(uid) for uid in yearly["plot_uid"]}
        available_uids = source_uids[source_years == year].tolist()
        population_selection[str(year)] = {
            "n_available": len(available_uids),
            "n_retained": len(yearly),
            "excluded_plot_uids": [
                uid for uid in available_uids if uid not in retained
            ],
        }
        observations = (
            (row["height_days"].long(), row["height_values"]) for row in yearly
        )
        target = int(detect_lodged_plots(observations, detector).sum())
        dense_days, dense_real = grid_protocol.observed_on_common_days(yearly)
        prepared[year] = ObservedYearPopulation(
            plots=yearly,
            real=grid_protocol.observed_at_grid(yearly, shared_grid),
            dense_days=dense_days,
            dense_real=dense_real,
            detected_count=target,
        )
    return prepared, population_selection


def fit_lodging_from_observations(
    prepared: Mapping[int, ObservedYearPopulation], defaults: LodgingOptions
) -> PooledLodgingFit:
    """Fit once on observed full-schedule maxima, before any simulation."""
    heights = np.asarray(
        [
            float(row["height_values"].max())
            for observed in prepared.values()
            for row in observed.plots
        ],
        dtype=np.float64,
    )
    target = sum(observed.detected_count for observed in prepared.values())
    shape = defaults["lodging_weibull_shape"]
    height_clamp = defaults["lodging_height_clamp"]
    offset = defaults["lodging_weibull_offset"]
    scale = fit_weibull_scale(
        heights, target, shape=shape, height_clamp=height_clamp, offset=offset
    )
    effective = np.maximum(np.minimum(heights, height_clamp) - offset, 0)
    expected = -np.expm1(-((effective / scale) ** shape))
    return PooledLodgingFit(
        scale=scale,
        target_count=target,
        n_plots=len(heights),
        target_rate=target / len(heights),
        expected_count=float(expected.sum()),
    )


def simulate_and_save_lodging_year(
    output_dir: Path,
    year: int,
    observed: ObservedYearPopulation,
    days: Tensor,
    pool: GenotypePool,
    seeds: Sequence[int],
    lodging_parameters: LodgingOptions,
) -> None:
    """Sample unchanged simulator events with the frozen empirical scale."""
    with torch.no_grad():
        clean, sim_days = _compute_height_trajectories(
            torch.stack(list(observed.plots["temperature_values"])),
            pool,
            dates=HeightDates(),
        )
        synthetic_lodging = []
        for seed in seeds:
            params = sample_lodging_params(
                clean,
                1,
                torch.Generator().manual_seed(seed * 10000 + year),
                enable_lodging=True,
                **lodging_parameters,
            )
            lodged, _, _ = apply_lodging_from_params(clean, params)
            synthetic_lodging.append(lodged[:, 0, days - sim_days[0]])
    np.savez_compressed(
        output_dir / f"aligned_{year}.npz",
        days=days.numpy(),
        real=observed.real.numpy(),
        synthetic_lodging=torch.stack(synthetic_lodging).numpy(),
        plot_uids=np.asarray([str(uid) for uid in observed.plots["plot_uid"]]),
        seeds=np.asarray(seeds),
        dense_days=observed.dense_days.numpy(),
        dense_real=observed.dense_real.numpy(),
    )
    logging.info(
        "%s: N=%s, observed lodged=%s, shared dates=%s",
        year,
        len(observed.plots),
        observed.detected_count,
        len(days),
    )


def fit_and_simulate_lodging(output_dir: Path) -> None:
    """Prepare observations, fit the pooled rate, then simulate the same years."""
    years = Fip1Facts().years
    splits = list(Fip1Facts().splits)
    params_path = Path("results/calibration/height/params_pool.json")
    detector: DetectorOptions = {
        "relative_threshold": 0.20,
        "absolute_threshold": 0.10,
        "tail_window": 5,
    }
    seeds = tuple(range(200, 205))
    lodging_fields = (
        "lodging_height_clamp",
        "lodging_weibull_shape",
        "lodging_weibull_scale",
        "lodging_weibull_offset",
        "lodging_severity_min",
        "lodging_severity_max",
        "lodging_transition_steps_min",
        "lodging_transition_steps_max",
    )
    # Hydra-zen creates these configuration fields dynamically.
    defaults = cast(
        LodgingOptions,
        {name: getattr(BaseSyntheticDatasetConfig, name) for name in lodging_fields},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    dataset_path = os.environ["NPNF_DATASET_PATH"]
    dataset = get_heights_dataset(
        split=splits, standardize=False, datasets_offline_path=dataset_path
    ).dataset
    shared_grid = grid_protocol.fip1_day_grid(dataset_path)
    prepared, population_selection = prepare_observed_years(
        dataset, years, shared_grid, detector
    )
    pooled_fit = fit_lodging_from_observations(prepared, defaults)
    pool = GenotypePool.sample(
        num_genotypes=1000,
        seed=42,
        height_pool_params=load_height_pool_params(params_path),
    )
    for year, observed in prepared.items():
        simulate_and_save_lodging_year(
            output_dir,
            year,
            observed,
            torch.from_numpy(shared_grid.anchors),
            pool,
            seeds,
            {**defaults, "lodging_weibull_scale": pooled_fit.scale},
        )
    save_json(
        output_dir / "lodging_settings.json",
        {
            "dataset_path": dataset_path,
            "splits": splits,
            "years": years,
            "population_selection": population_selection,
            "metric_grid": {
                "anchors": shared_grid.anchors.tolist(),
                "assigned_days": {
                    key: value.tolist() for key, value in shared_grid.assigned.items()
                },
            },
            "pooled_fit": asdict(pooled_fit),
        },
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fit_and_simulate_lodging(args.output_dir)


if __name__ == "__main__":
    main()
