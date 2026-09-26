"""Paper figure: genotype diversity of height response surface across τ_norm slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from npnf.data.pools import GenotypePool
from npnf.data.synthetic.height.genotype import NUM_FREE_CPS
from npnf.data.synthetic.height.params import load_height_pool_params
from npnf.data.synthetic.height.response_surface import (
    DEFAULT_DEGREE,
    DEFAULT_T_KNOTS,
    DEFAULT_TAU_KNOTS,
    evaluate_surface,
)
from npnf.scripts.paper.style import COLORS, save_figure, setup_style

TAU_SLICES = {0.25: COLORS["blue"], 0.5: COLORS["green"], 0.75: COLORS["red"]}


def plot_genotype_diversity(pool_params_path: Path, output_dir: Path) -> None:
    setup_style()

    pool_params = load_height_pool_params(pool_params_path)
    genotype_pool = GenotypePool.sample(1000, 42, height_pool_params=pool_params)

    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    n_T_free = n_T_basis - 2

    cps_free_mean = torch.tensor(
        [
            getattr(pool_params.mean_control_points, f"cp_{i}")
            for i in range(NUM_FREE_CPS)
        ],
        dtype=torch.float32,
    ).reshape(n_T_free, n_tau_basis - 2)
    cps_mean = torch.zeros(1, n_T_basis, n_tau_basis)
    cps_mean[0, 1:-1, 1:-1] = cps_free_mean

    T_vals = torch.linspace(0.0, float(DEFAULT_T_KNOTS[-1]), 200)
    tau_vals = torch.linspace(0.0, 1.0, 200)
    T_grid, tau_grid = torch.meshgrid(T_vals, tau_vals, indexing="ij")

    growth_mean = (
        evaluate_surface(
            T_grid.reshape(1, -1),
            tau_grid.reshape(1, -1),
            cps_mean,
            DEFAULT_T_KNOTS,
            DEFAULT_TAU_KNOTS,
            DEFAULT_DEGREE,
        )
        .reshape(200, 200)
        .detach()
        .numpy()
    )

    n_sample = min(50, len(genotype_pool.params))
    params_batch = genotype_pool.params[:n_sample]
    cps_free_batch = params_batch[:, :NUM_FREE_CPS].reshape(
        n_sample, n_T_free, n_tau_basis - 2
    )
    cps_batch = torch.zeros(n_sample, n_T_basis, n_tau_basis)
    cps_batch[:, 1:-1, 1:-1] = cps_free_batch

    T_eval = T_grid.reshape(1, -1).expand(n_sample, -1)
    tau_eval = tau_grid.reshape(1, -1).expand(n_sample, -1)
    growth_batch = (
        evaluate_surface(
            T_eval,
            tau_eval,
            cps_batch,
            DEFAULT_T_KNOTS,
            DEFAULT_TAU_KNOTS,
            DEFAULT_DEGREE,
        )
        .reshape(n_sample, 200, 200)
        .detach()
        .numpy()
    )

    T_np = T_vals.numpy()
    T_max = float(DEFAULT_T_KNOTS[-1])

    # Band version
    fig, ax = plt.subplots(figsize=(6, 4))
    for tau_val, color in TAU_SLICES.items():
        idx = int(tau_val * (len(tau_vals) - 1))
        curves = growth_batch[:, :, idx]
        lo = np.percentile(curves, 5, axis=0)
        hi = np.percentile(curves, 95, axis=0)
        ax.fill_between(T_np, lo, hi, color=color, alpha=0.15)
        ax.plot(
            T_np, growth_mean[:, idx], color=color, lw=2, label=f"τ = {tau_val:.2f}"
        )
    ax.set_xlabel("T (°C)")
    ax.set_ylabel("Growth rate (mm/h)")
    ax.set_xlim(0, T_max)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir / "height_genotype_diversity_bands")
    plt.close(fig)

    # Lines version
    fig, ax = plt.subplots(figsize=(6, 4))
    for tau_val, color in TAU_SLICES.items():
        idx = int(tau_val * (len(tau_vals) - 1))
        for i in range(n_sample):
            ax.plot(T_np, growth_batch[i, :, idx], color=color, alpha=0.15, lw=0.6)
        ax.plot(
            T_np, growth_mean[:, idx], color=color, lw=2, label=f"τ = {tau_val:.2f}"
        )
    ax.set_xlabel("T (°C)")
    ax.set_ylabel("Growth rate (mm/h)")
    ax.set_xlim(0, T_max)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "height_genotype_diversity_lines")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper figure: genotype diversity of height response surface"
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=Path("results/calibration/height/params_pool.json"),
        help="Path to calibrated height pool params JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/height_genotype_diversity"),
        help="Output directory for figure",
    )
    args = parser.parse_args()
    plot_genotype_diversity(args.params_path, args.output_dir)


if __name__ == "__main__":
    main()
