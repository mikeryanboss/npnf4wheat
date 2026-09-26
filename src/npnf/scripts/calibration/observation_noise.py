"""Estimate FIP1 observation variation and score complete-grid lodging plus noise."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from npnf.calibration.height.evaluation.observation_noise import (
    score_noisy_trajectories,
    sum_squared_triplet_residuals,
)
from npnf.metrics.signature import to_metric_paths
from npnf.scripts.utils.outputs import save_json, write_csv


def estimate_pooled_noise_std(
    lodging_dir: Path, output_dir: Path, years: Sequence[int], max_gap: float = 7
) -> float:
    """Estimate pooled residual variation across all comparison years."""
    sum_squares = 0.0
    n_triplets = 0
    n_plots = 0
    for year in years:
        with np.load(lodging_dir / f"aligned_{year}.npz", allow_pickle=False) as data:
            days, real = data["dense_days"], data["dense_real"]
        squared, count = sum_squared_triplet_residuals(days, real, max_gap)
        sum_squares += squared
        n_triplets += count
        n_plots += len(real)
    sigma_m = float(np.sqrt(sum_squares / n_triplets))
    save_json(
        output_dir / "estimation_settings.json",
        {
            "primary_max_adjacent_gap_days": max_gap,
            "pooled_years": years,
            "pooled": {
                "n_plots": n_plots,
                "n_triplets": n_triplets,
                "sigma_rms_m": sigma_m,
            },
        },
    )
    return sigma_m


def score_lodging_with_noise(
    lodging_dir: Path, sigma_m: float, output_dir: Path
) -> None:
    """Add independent observation noise to each saved pooled-lodging realization."""
    lodging_settings = json.loads((lodging_dir / "lodging_settings.json").read_text())
    weibull_scale = lodging_settings["pooled_fit"]["scale"]
    seeds = tuple(range(100, 105))
    runs, summaries = [], []
    for year in lodging_settings["years"]:
        aligned_path = lodging_dir / f"aligned_{year}.npz"
        with np.load(aligned_path, allow_pickle=False) as data:
            days, real, synthetic_lodging, lodging_seeds = [
                data[key] for key in ("days", "real", "synthetic_lodging", "seeds")
            ]
        time = (torch.from_numpy(days).float() - 200) / 121
        x = to_metric_paths(torch.from_numpy(real).float(), 1.5, time, lead_lag=True)
        values = []
        for lodging_seed, noise_seed, synthetic in zip(
            lodging_seeds, seeds, synthetic_lodging, strict=True
        ):
            stream_seed = noise_seed + year * 1000
            rng = torch.Generator(device="cpu").manual_seed(stream_seed)
            noise = torch.randn(
                synthetic.shape, generator=rng, dtype=torch.float32, device="cpu"
            )
            value = score_noisy_trajectories(
                x,
                torch.from_numpy(synthetic).float(),
                noise,
                sigma_m,
                time,
                1.5,
                lead_lag=True,
                max_batch=64,
            )
            values.append(value)
            runs.append(
                {
                    "year": year,
                    "lodging_seed": int(lodging_seed),
                    "noise_seed": noise_seed,
                    "sigma_m": sigma_m,
                    "weibull_scale": weibull_scale,
                    "sig_mmd_squared": value,
                }
            )
        summaries.append(
            {
                "year": year,
                "n_real": len(real),
                "n_synthetic": synthetic_lodging.shape[1],
                "n_shared_days": len(days),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "n_draws": len(values),
                "sigma_m": sigma_m,
                "weibull_scale": weibull_scale,
            }
        )
        logging.info("%s: completed five combined lodging-plus-noise draws", year)
    write_csv(output_dir / "combined_runs.csv", runs, list(runs[0]))
    write_csv(output_dir / "combined_summary.csv", summaries, list(summaries[0]))
    save_json(
        output_dir / "combined_settings.json",
        {
            "sigma_m": sigma_m,
            "weibull_scale": weibull_scale,
            "population_selection": lodging_settings["population_selection"],
        },
    )


def estimate_noise_and_score_lodging(lodging_dir: Path, output_dir: Path) -> None:
    settings = json.loads((lodging_dir / "lodging_settings.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    sigma_m = estimate_pooled_noise_std(lodging_dir, output_dir, settings["years"])
    with torch.no_grad():
        score_lodging_with_noise(lodging_dir, sigma_m, output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lodging-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    estimate_noise_and_score_lodging(args.lodging_dir, args.output_dir)


if __name__ == "__main__":
    main()
