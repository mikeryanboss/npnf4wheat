"""Paper figure: 3D synthetic height response surface with a synthetic
temperature trajectory overlaid. Both the genotype surface and the temperature
year are drawn from the calibrated generative model, keyed off a single seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from npnf.calibration.height.constants import HeightDates
from npnf.data.synthetic.height.genotype import (
    GenotypeParams,
    SyntheticGenotypeGenerator,
)
from npnf.data.synthetic.height.params import load_height_pool_params
from npnf.data.synthetic.height.response_surface import (
    DEFAULT_DEGREE,
    DEFAULT_T_KNOTS,
    DEFAULT_TAU_KNOTS,
    evaluate_surface,
)
from npnf.scripts.paper.style import setup_style


def _build_sampled_surface(
    pool_params_path: Path, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, float]:
    """Build a single sampled response surface from the CP prior.

    Draws one genotype from the matrix-normal control-point prior (the same
    path the synthetic dataset uses). Returns the grid arrays plus the
    genotype's own sampled `tau_max` so the trajectory matures consistently
    with the surface.

    Returns:
        (T_vals, tau_vals, growth_surface, cps_sample, tau_max).
    """
    pool_params = load_height_pool_params(pool_params_path)

    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1  # 8
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1  # 7

    generator = SyntheticGenotypeGenerator(pool_params)
    params, _ = generator.sample(num_genotypes=1, seed=seed)  # (1, 31)
    cps_sample = GenotypeParams.build_clamped_control_grid(
        params, n_T_basis, n_tau_basis
    )
    tau_max = float(params[0, -1])

    T_vals = torch.linspace(0.0, float(DEFAULT_T_KNOTS[-1]), 201)
    tau_vals = torch.linspace(0.0, 1.0, 201)
    T_grid, tau_grid = torch.meshgrid(T_vals, tau_vals, indexing="ij")

    growth = (
        evaluate_surface(
            T_grid.reshape(1, -1),
            tau_grid.reshape(1, -1),
            cps_sample,
            DEFAULT_T_KNOTS,
            DEFAULT_TAU_KNOTS,
            DEFAULT_DEGREE,
        )
        .reshape(201, 201)
        .detach()
        .numpy()
    )
    return T_vals.numpy(), tau_vals.numpy(), growth, cps_sample, tau_max


def _sample_synthetic_temps(
    temp_params_path: Path, dates: HeightDates, seed: int
) -> torch.Tensor:
    """Draw a single synthetic hourly-temperature year from the calibrated model.

    Returns a (total_days, 24) tensor aligned to Nov 1 at index 0 -- the same
    layout as the real FIP1 year tensors -- so it drops straight into
    `_run_trajectory`.
    """
    from npnf.data.synthetic.temperature import SyntheticTemperatureGenerator

    generator = SyntheticTemperatureGenerator(
        num_sites=1,
        num_years=1,
        num_days=dates.total_days,
        seed=seed,
        calibration_params_path=temp_params_path,
    )
    temps = generator.generate_all()  # (1, total_days, 24)
    return temps[0]


def _run_trajectory(
    hourly_temp: torch.Tensor, cps: torch.Tensor, tau_max: float, dates: HeightDates
) -> dict[str, np.ndarray]:
    """Run forward pass and return daily trajectory diagnostics."""
    T_hourly_active = hourly_temp[dates.tau_start_idx :, :]  # (135, 24)
    T_daily_active = T_hourly_active.mean(dim=-1)  # (135,)

    T_flat = T_hourly_active.reshape(-1)
    tau_hourly = T_flat.clamp(min=0).cumsum(dim=-1)
    tau_norm_hourly = (tau_hourly / tau_max).clamp(max=1.0)

    growth_hourly = evaluate_surface(
        T_flat.unsqueeze(0),
        tau_norm_hourly.unsqueeze(0),
        cps,
        DEFAULT_T_KNOTS,
        DEFAULT_TAU_KNOTS,
        DEFAULT_DEGREE,
    )[0]
    growth_hourly = growth_hourly.masked_fill(tau_norm_hourly >= 1.0, 0.0)
    growth_daily = growth_hourly.reshape(-1, 24).sum(dim=-1)

    tau_norm_daily = tau_norm_hourly.reshape(-1, 24)[:, -1]
    num_days = T_hourly_active.shape[0]
    days = torch.arange(dates.tau_start_day, dates.tau_start_day + num_days)

    return {
        "days": days.numpy(),
        "T_active": T_daily_active.detach().numpy(),
        "tau_norm": tau_norm_daily.detach().numpy(),
        "growth_daily": growth_daily.detach().numpy(),
    }


def plot_height_response_surface(
    pool_params_path: Path, output_dir: Path, seed: int, temp_params_path: Path
) -> None:
    """Create 3D response surface figure with temperature trajectory overlay.

    Both layers are synthetic draws from the calibrated generative model: a
    single genotype from the control-point prior (with its own sampled tau_max)
    and a single synthetic temperature year, both keyed off `seed`.
    """
    setup_style()
    dates = HeightDates()

    # Sampled genotype surface + matching synthetic temperature year
    T_vals, tau_vals, growth_surface, cps, tau_max = _build_sampled_surface(
        pool_params_path, seed
    )
    hourly_temp = _sample_synthetic_temps(temp_params_path, dates, seed)
    T_mesh, tau_mesh = np.meshgrid(T_vals, tau_vals, indexing="ij")

    # Run trajectory
    traj = _run_trajectory(hourly_temp, cps, tau_max, dates)
    tau_norm_traj = traj["tau_norm"]
    T_traj = traj["T_active"]
    days = traj["days"]

    # Evaluate growth rate on the surface at trajectory points
    T_traj_t = torch.tensor(T_traj, dtype=torch.float32).unsqueeze(0)
    tau_traj_t = torch.tensor(tau_norm_traj, dtype=torch.float32).unsqueeze(0)
    growth_traj = (
        evaluate_surface(
            T_traj_t,
            tau_traj_t,
            cps,
            DEFAULT_T_KNOTS,
            DEFAULT_TAU_KNOTS,
            DEFAULT_DEGREE,
        )[0]
        .detach()
        .numpy()
    )
    # Zero growth after maturity (same as forward model)
    growth_traj[tau_norm_traj >= 1.0] = 0.0

    # -- Figure --
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # Disable computed z-ordering so zorder works like 2D (trajectory on top)
    ax.computed_zorder = False

    # Surface — axes: x=τ_norm, y=T, z=growth
    ax.plot_surface(
        tau_mesh,
        T_mesh,
        growth_surface,
        cmap="viridis",
        alpha=1.0,
        rstride=10,
        cstride=10,
        edgecolor=(0, 0, 0, 0.1),
        linewidth=0.2,
        antialiased=True,
    )

    # Truncate trajectory at maturity (no post-maturity zero-growth tail)
    mature_mask = tau_norm_traj >= 1.0
    if mature_mask.any():
        end_idx = int(np.argmax(mature_mask)) + 1  # include maturity point
    else:
        end_idx = len(tau_norm_traj)

    # Trajectory (offset above surface to avoid clipping)
    z_offset = growth_surface.max() * 0.06
    traj_days = days[:end_idx]
    points = np.array(
        [tau_norm_traj[:end_idx], T_traj[:end_idx], growth_traj[:end_idx] + z_offset]
    ).T  # (N, 3)
    # Continuous line: black border + white core
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="black",
        linewidth=2.5,
        zorder=10,
    )
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="white",
        linewidth=1.2,
        zorder=11,
    )
    # Day dots
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="white",
        s=18,
        zorder=12,
        edgecolors="black",
        linewidths=0.6,
        depthshade=False,
        alpha=1.0,
    )

    # Day labels at start and end
    import matplotlib.patheffects as pe

    label_offset = max(z_offset * 2, 0.15)
    for i in [0, -1]:
        ax.text(
            points[i, 0],
            points[i, 1],
            points[i, 2] + label_offset,
            str(int(traj_days[i])),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=15,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )

    # Clean panes, keep grid
    ax.xaxis.pane.fill = False  # ty: ignore[unresolved-attribute]
    ax.yaxis.pane.fill = False  # ty: ignore[unresolved-attribute]
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")  # ty: ignore[unresolved-attribute]
    ax.yaxis.pane.set_edgecolor("lightgray")  # ty: ignore[unresolved-attribute]
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(True, alpha=0.3)

    ax.set_xlabel("\u03c4", labelpad=8)
    ax.set_ylabel("T (\u00b0C)", labelpad=8)
    ax.set_zlabel("Growth rate (mm/h)")
    ax.set_xlim(1, 0)  # τ_norm: 1 at back, 0 at front
    ax.set_ylim(0, 35)  # T: 35 at top
    # Dense minor ticks for grid lines matching the surface wireframe
    # 21 ticks = 21 wireframe lines (stride 10 on 201 points), label every 4th
    ax.set_xticks(np.linspace(0, 1, 21))
    ax.set_xticklabels(
        ["" if i % 4 else f"{v:.1f}" for i, v in enumerate(np.linspace(0, 1, 21))]
    )
    ax.set_yticks(np.linspace(0, 35, 21))
    ax.set_yticklabels(
        ["" if i % 4 else f"{v:.0f}" for i, v in enumerate(np.linspace(0, 35, 21))]
    )
    # Z ticks at clean 0.2 intervals, zlim snapped to last tick
    z_max_clean = np.ceil(growth_surface.max() / 0.2) * 0.2
    ax.set_zlim(0, z_max_clean)
    ax.set_zticks(np.arange(0, z_max_clean + 0.01, 0.2))  # ty: ignore[call-non-callable]
    ax.view_init(elev=55, azim=-50)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save directly (tight_layout doesn't work well with 3D)
    path = output_dir / f"height_response_surface_seed{seed}"
    for fmt in ("png", "pdf"):
        fig.savefig(
            path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight", pad_inches=0.5
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper figure: 3D height response surface with trajectory"
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
        default=Path("paper/height_response_surface"),
        help="Output directory for figure",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Seed(s) for the synthetic genotype surface + temperature year. "
        "Pass several to render one figure per seed.",
    )
    parser.add_argument(
        "--temp-params-path",
        type=Path,
        default=Path("results/calibration/temperature/params_pool.json"),
        help="Path to calibrated temperature params JSON",
    )
    args = parser.parse_args()
    for seed in args.seeds:
        plot_height_response_surface(
            args.params_path, args.output_dir, seed, args.temp_params_path
        )


if __name__ == "__main__":
    main()
