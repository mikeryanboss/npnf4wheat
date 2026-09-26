"""Height calibration evaluation and diagnostic visualization."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

from npnf.calibration.height.constants import HeightDates
from npnf.calibration.height.data import compute_growth_end_days
from npnf.calibration.height.forward import _run_bspline_forward
from npnf.calibration.shared.plotting import (
    COLOR_FIP1,
    COLOR_SYNTHETIC,
    save_figure,
    setup_plot_style,
)
from npnf.data.datasets.fip1 import Fip1Facts
from npnf.data.pools import YearsitePool
from npnf.data.synthetic.height.genotype import NUM_FREE_CPS
from npnf.data.synthetic.height.response_surface import (
    DEFAULT_DEGREE,
    DEFAULT_T_KNOTS,
    DEFAULT_TAU_KNOTS,
    evaluate_surface,
    make_clamped_knots,
)

if TYPE_CHECKING:
    import datasets

    from npnf.data.pools.genotype_pool import GenotypePool
    from npnf.data.synthetic.height.params import HeightPoolParams


def _compute_height_trajectories(
    temps: torch.Tensor, genotype_pool: GenotypePool, *, dates: HeightDates
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute full growth trajectories using B-spline surface model.

    Args:
        temps: Temperature tensor, shape (B, 274, 24)
        genotype_pool: GenotypePool to sample varied genotype parameters.

    Returns:
        Tuple of:
        - trajectories: Height trajectories in meters, shape (B, 274)
        - days: Day indices, shape (274,)
    """
    batch_size = temps.shape[0]

    rng = torch.Generator()
    rng.manual_seed(42)
    indices = torch.randint(0, len(genotype_pool), (batch_size,), generator=rng)
    params = genotype_pool.params[indices]

    trajectories = _run_bspline_forward(temps, params, dates=dates)

    trajectory_days = torch.arange(
        dates.day_temperature_start, dates.day_temperature_start + trajectories.shape[1]
    )

    return trajectories, trajectory_days


def plot_trajectory_comparison_by_year(
    fip1_data: datasets.Dataset,
    output_dir: Path,
    genotype_pool: GenotypePool,
    *,
    dates: HeightDates,
) -> None:
    """Create per-year trajectory comparison: FIP1 real vs synthetic.

    For each FIP1 year, shows:
    - Real FIP1 trajectories for that year (background)
    - Synthetic trajectories generated using the SAME temperatures (foreground)
    - Growth-end day distributions (IQR bands + median) for both

    Args:
        fip1_data: HuggingFace Dataset.
        output_dir: Output directory for saving figure
        genotype_pool: GenotypePool for realistic variation.

    Saves: plots/height/trajectory_by_year.png
    """
    setup_plot_style()

    # Compute FIP1 growth-end days per year
    fip1_growth_end_by_year = compute_growth_end_days(fip1_data)

    # Group FIP1 data by year
    fip1_heights = fip1_data["height_values"]
    fip1_days = fip1_data["height_days"]
    fip1_years = fip1_data["harvest_year"]
    fip1_temps = fip1_data["temperature_values"]

    by_year: dict[int, dict[str, list]] = defaultdict(
        lambda: {"heights": [], "days": [], "temps": []}
    )

    for i in range(len(fip1_heights)):
        year = int(fip1_years[i].item())
        if year == Fip1Facts().held_out_year:
            continue
        by_year[year]["heights"].append(fip1_heights[i])
        by_year[year]["days"].append(fip1_days[i])
        by_year[year]["temps"].append(fip1_temps[i])

    years = sorted(by_year.keys())
    num_years = len(years)

    if num_years == 0:
        return

    fig, axes = plt.subplots(num_years, 1, figsize=(12, 4 * num_years), sharex=True)
    if num_years == 1:
        axes = [axes]

    day_min, day_max = 175, 330
    height_max = 1.5

    for row_index, year in enumerate(years):
        ax = axes[row_index]
        year_data = by_year[year]
        num_samples = len(year_data["heights"])

        # Plot FIP1 real trajectories (background)
        for i in range(num_samples):
            days = year_data["days"][i].numpy()
            heights = year_data["heights"][i].numpy()
            mask = (days >= day_min) & (days <= day_max)
            ax.plot(
                days[mask], heights[mask], alpha=0.2, color=COLOR_FIP1, linewidth=0.5
            )

        # Compute synthetic trajectories using same temperatures
        synth_growth_end_days_np = None
        if year_data["temps"]:
            temps_batch = torch.stack(year_data["temps"]).float()
            synth_trajectory, synth_days = _compute_height_trajectories(
                temps_batch, genotype_pool=genotype_pool, dates=dates
            )
            synth_trajectory_np = synth_trajectory.detach().numpy()
            synth_days_np = synth_days.detach().numpy()

            # Synthetic growth-end: day of peak height per trajectory
            synth_growth_end_days_np = (
                dates.day_temperature_start
                + synth_trajectory.detach().argmax(dim=-1).numpy()
            ).astype(float)

            for i in range(min(len(synth_trajectory_np), 100)):
                trajectory = synth_trajectory_np[i]
                mask = (synth_days_np >= day_min) & (synth_days_np <= day_max)
                ax.plot(
                    synth_days_np[mask],
                    trajectory[mask],
                    alpha=0.3,
                    color=COLOR_SYNTHETIC,
                    linewidth=0.5,
                )

        # Growth-end markers: rug ticks on baseline + IQR band + median
        growth_end_rug_height = 0.035 * height_max  # short tick marks
        growth_end_band_alpha = 0.15
        growth_end_line_alpha = 0.8

        # FIP1 growth-end
        if year in fip1_growth_end_by_year:
            growth_end = fip1_growth_end_by_year[year].numpy()
            ax.vlines(
                growth_end,
                0,
                growth_end_rug_height,
                color=COLOR_FIP1,
                alpha=0.3,
                linewidth=0.7,
            )
            q25, q50, q75 = np.percentile(growth_end, [25, 50, 75])
            ax.axvspan(q25, q75, alpha=growth_end_band_alpha, color=COLOR_FIP1)
            ax.axvline(
                q50, color=COLOR_FIP1, linestyle="--", alpha=growth_end_line_alpha
            )

        # Synthetic growth-end
        if synth_growth_end_days_np is not None:
            ax.vlines(
                synth_growth_end_days_np,
                0,
                growth_end_rug_height,
                color=COLOR_SYNTHETIC,
                alpha=0.3,
                linewidth=0.7,
            )
            q25, q50, q75 = np.percentile(synth_growth_end_days_np, [25, 50, 75])
            ax.axvspan(q25, q75, alpha=growth_end_band_alpha, color=COLOR_SYNTHETIC)
            ax.axvline(
                q50, color=COLOR_SYNTHETIC, linestyle="--", alpha=growth_end_line_alpha
            )

        ax.set_xlim(day_min, day_max)
        ax.set_ylim(0, height_max)
        ax.set_ylabel("Height (m)")
        ax.set_title(f"{year} (n={num_samples} samples)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        if row_index == 0:
            legend_label = "Synthetic (varied genotypes)"
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color=COLOR_FIP1,
                    linewidth=2,
                    alpha=0.5,
                    label="FIP1 Real",
                ),
                Line2D(
                    [0],
                    [0],
                    color=COLOR_SYNTHETIC,
                    linewidth=2,
                    alpha=0.7,
                    label=legend_label,
                ),
                Line2D(
                    [0],
                    [0],
                    color=COLOR_FIP1,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=growth_end_line_alpha,
                    label="FIP1 growth end (median, IQR)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=COLOR_SYNTHETIC,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=growth_end_line_alpha,
                    label="Synth growth end (median, IQR)",
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper left")

    axes[-1].set_xlabel("Day of Year (Sep 1 = 0)")

    fig.suptitle(
        "Per-Year Trajectory Validation\n"
        "FIP1 Real vs Synthetic (using same temperatures)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    save_figure(fig, "height/plots/trajectory_by_year", output_dir)


def plot_response_surface(
    height_pool_params: HeightPoolParams, genotype_pool: GenotypePool, output_dir: Path
) -> None:
    """Visualize the calibrated B-spline temperature response surface.

    Creates a figure with 5 panels (3D surface spans the left column):
    - Left (tall): 3D surface with knot grid lines and control points
      colored by T basis row
    - Top-middle: Filled contour map of the mean-genotype surface
    - Top-right: Growth rate vs temperature at fixed τ_norm slices
    - Bottom-middle: Control point grid heatmap (8x5)
    - Bottom-right: Genotype diversity at τ_norm=0.5 (n≤50 genotypes)

    Saves: plots/height/response_surface.png
    """
    setup_plot_style()

    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1  # 8
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1  # 7
    n_T_free = n_T_basis - 2  # 6

    # Mean control points: read directly from params
    cps_free = torch.tensor(
        [
            getattr(height_pool_params.mean_control_points, f"cp_{i}")
            for i in range(NUM_FREE_CPS)
        ],
        dtype=torch.float32,
    ).reshape(n_T_free, n_tau_basis - 2)
    cps_mean = torch.zeros(1, n_T_basis, n_tau_basis)
    cps_mean[0, 1:-1, 1:-1] = cps_free

    # Evaluation grid
    T_max = float(DEFAULT_T_KNOTS[-1])
    T_vals = torch.linspace(0.0, T_max, 200)
    tau_vals = torch.linspace(0.0, 1.0, 200)
    T_grid, tau_grid = torch.meshgrid(T_vals, tau_vals, indexing="ij")
    growth = (
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
    T_np = T_vals.numpy()
    tau_np = tau_vals.numpy()

    # Greville abscissae — domain positions of control points
    t_T = make_clamped_knots(DEFAULT_T_KNOTS, DEFAULT_DEGREE).numpy()
    t_tau = make_clamped_knots(DEFAULT_TAU_KNOTS, DEFAULT_DEGREE).numpy()
    degree = DEFAULT_DEGREE
    T_greville = np.array(
        [t_T[i + 1 : i + degree + 1].mean() for i in range(n_T_basis)]
    )
    tau_greville = np.array(
        [t_tau[i + 1 : i + degree + 1].mean() for i in range(n_tau_basis)]
    )

    T_knots_np = DEFAULT_T_KNOTS.numpy()
    tau_knots_np = DEFAULT_TAU_KNOTS.numpy()
    cp_full_np = cps_mean[0].detach().numpy()  # (8, 5)

    # Layout: 2 rows × 3 cols; left column spans both rows for the 3D surface
    fig = plt.figure(figsize=(16, 10))
    gridspec = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    ax_3d = fig.add_subplot(gridspec[:, 0], projection="3d")
    ax_contour = fig.add_subplot(gridspec[0, 1])
    ax_slices = fig.add_subplot(gridspec[0, 2])
    ax_heatmap = fig.add_subplot(gridspec[1, 1])
    ax_diversity = fig.add_subplot(gridspec[1, 2])

    # --- 3D surface -------------------------------------------------
    T_mesh, tau_mesh = np.meshgrid(T_np, tau_np, indexing="ij")
    ax_3d.plot_surface(T_mesh, tau_mesh, growth, cmap="viridis", alpha=0.85)

    for T_k in T_knots_np:
        idx = int((T_k / T_max) * (len(T_np) - 1))
        ax_3d.plot(
            np.full_like(tau_np, T_k),
            tau_np,
            growth[idx, :],
            color="black",
            lw=1.8,
            alpha=0.9,
        )
    for tau_k in tau_knots_np:
        idx = int(tau_k * (len(tau_np) - 1))
        ax_3d.plot(
            T_np,
            np.full_like(T_np, tau_k),
            growth[:, idx],
            color="black",
            lw=1.8,
            alpha=0.9,
        )

    row_colors = plt.colormaps["tab10"](np.linspace(0, 1, n_T_basis))
    for i, (T_k, color) in enumerate(zip(T_greville, row_colors, strict=False)):
        ax_3d.scatter(
            np.full(n_tau_basis, T_k),
            tau_greville,
            cp_full_np[i],
            color=color,
            s=50,
            depthshade=False,
            label=f"T={T_k:.1f}\u00b0C",
        )

    ax_3d.set_xlabel("T (\u00b0C)", labelpad=6)
    ax_3d.set_ylabel("\u03c4_norm", labelpad=6)
    ax_3d.set_zlabel("Growth rate (mm/h)", labelpad=6)
    ax_3d.set_title("R(T, \u03c4_norm) \u2014 3D surface", fontsize=10)
    ax_3d.legend(loc="upper left", fontsize=6, title="T basis row", title_fontsize=6)

    # --- Contour map ------------------------------------------------
    contour_set = ax_contour.contourf(tau_np, T_np, growth, levels=20, cmap="viridis")
    for T_k in T_knots_np:
        ax_contour.axhline(T_k, color="white", lw=0.6, alpha=0.5, ls="--")
    for tau_k in tau_knots_np:
        ax_contour.axvline(tau_k, color="white", lw=0.6, alpha=0.5, ls="--")
    plt.colorbar(contour_set, ax=ax_contour, label="mm/h")
    ax_contour.set_xlabel("\u03c4_norm")
    ax_contour.set_ylabel("T (\u00b0C)")
    ax_contour.set_title("R(T, \u03c4_norm) \u2014 contour", fontsize=10)

    # --- T-slices ---------------------------------------------------
    tau_slices = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    slice_colors = plt.colormaps["plasma"](np.linspace(0.1, 0.9, len(tau_slices)))
    for tau_val, color in zip(tau_slices, slice_colors, strict=False):
        idx = int(tau_val * (len(tau_np) - 1))
        ax_slices.plot(
            T_np, growth[:, idx], color=color, label=f"\u03c4={tau_val:.1f}", lw=1.5
        )
    ax_slices.set_xlabel("T (\u00b0C)")
    ax_slices.set_ylabel("Growth rate (mm/h)")
    ax_slices.set_title("Growth vs T at fixed \u03c4_norm", fontsize=10)
    ax_slices.legend(title="\u03c4_norm", fontsize=7, loc="upper left")
    ax_slices.set_xlim(0, T_max)
    ax_slices.set_ylim(bottom=0)
    ax_slices.grid(True, alpha=0.3)

    # --- CP heatmap -------------------------------------------------
    vmin = cp_full_np[1:-1, :].min()
    vmax = cp_full_np[1:-1, :].max()
    im = ax_heatmap.imshow(
        cp_full_np,
        aspect="auto",
        cmap="RdYlGn",
        vmin=min(0.0, vmin),
        vmax=max(vmax, 1.0),
    )
    T_row_labels = [f"{v:.1f}" for v in np.linspace(0, T_max, n_T_basis)]
    tau_col_labels = [f"{v:.2f}" for v in np.linspace(0, 1, n_tau_basis)]
    ax_heatmap.set_xticks(range(n_tau_basis))
    ax_heatmap.set_xticklabels(tau_col_labels)
    ax_heatmap.set_yticks(range(n_T_basis))
    ax_heatmap.set_yticklabels(T_row_labels)
    ax_heatmap.set_xlabel("\u03c4 basis index")
    ax_heatmap.set_ylabel("T basis index (\u00b0C)")
    ax_heatmap.set_title(f"Control point grid (8\u00d7{n_tau_basis})", fontsize=10)
    for i in range(n_T_basis):
        for j in range(n_tau_basis):
            ax_heatmap.text(
                j, i, f"{cp_full_np[i, j]:.1f}", ha="center", va="center", fontsize=7
            )
    plt.colorbar(im, ax=ax_heatmap, label="CP value")

    # --- Genotype diversity -----------------------------------------
    tau_mid_idx = 100  # τ_norm = 0.5
    n_sample = min(50, len(genotype_pool.params))
    params_batch = genotype_pool.params[:n_sample]
    cps_free_batch = params_batch[:, :NUM_FREE_CPS].reshape(
        n_sample, n_T_free, n_tau_basis - 2
    )
    cps_batch = torch.zeros(n_sample, n_T_basis, n_tau_basis)
    cps_batch[:, 1:-1, 1:-1] = cps_free_batch  # τ=0 and τ=1 columns stay zero
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
    for i in range(n_sample):
        ax_diversity.plot(
            T_np,
            growth_batch[i, :, tau_mid_idx],
            color=COLOR_SYNTHETIC,
            alpha=0.3,
            lw=0.8,
        )
    ax_diversity.plot(
        T_np, growth[:, tau_mid_idx], color="black", lw=2, label="Mean", zorder=5
    )
    ax_diversity.set_xlabel("T (\u00b0C)")
    ax_diversity.set_ylabel("Growth rate (mm/h)")
    ax_diversity.set_title(
        f"Genotype diversity at \u03c4_norm=0.5\n(n={n_sample})", fontsize=10
    )
    ax_diversity.legend(fontsize=8)
    ax_diversity.set_xlim(0, T_max)
    ax_diversity.set_ylim(bottom=0)
    ax_diversity.grid(True, alpha=0.3)

    fig.suptitle(
        "Calibrated B-spline Response Surface R(T, \u03c4_norm)",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, "height/plots/response_surface", output_dir)


def _run_bspline_forward_with_intermediates(
    hourly_temp: torch.Tensor, params: torch.Tensor, *, dates: HeightDates
) -> dict[str, Any]:
    """Run single-sample B-spline forward pass, returning all intermediate quantities.

    Evaluates the response surface at each of 24 hourly temperatures per day,
    then sums to get daily growth (mm/day). Thermal time (tau) uses daily means.

    Args:
        hourly_temp: Hourly temperatures, shape (274, 24)
        params: Genotype parameters, shape (37,)

    Returns:
        Dict with keys: days, T_active, tau_norm, growth, height, tau_max
    """
    T_hourly_active = hourly_temp[dates.tau_start_idx :, :]  # (135, 24)
    T_daily_active = T_hourly_active.mean(dim=-1)  # (135,) — for diagnostics plot
    tau_max = params[-1]

    # Thermal time at hourly resolution
    T_flat = T_hourly_active.reshape(-1)  # (135*24,)
    tau_hourly = T_flat.clamp(min=0).cumsum(dim=-1)  # (135*24,)
    tau_norm_hourly = (tau_hourly / tau_max).clamp(max=1.0)

    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    n_T_free = n_T_basis - 2

    cps_free = params[:NUM_FREE_CPS].reshape(n_T_free, n_tau_basis - 2)
    cps = torch.zeros(1, n_T_basis, n_tau_basis)
    cps[0, 1:-1, 1:-1] = cps_free

    # Evaluate surface at hourly resolution
    growth_hourly = evaluate_surface(
        T_flat.unsqueeze(0),
        tau_norm_hourly.unsqueeze(0),
        cps,
        DEFAULT_T_KNOTS,
        DEFAULT_TAU_KNOTS,
        degree=DEFAULT_DEGREE,
    )[0]  # (135*24,)

    # Zero growth after maturity at hourly resolution, then sum to daily
    growth_hourly = growth_hourly.masked_fill(tau_norm_hourly >= 1.0, 0.0)
    growth = growth_hourly.reshape(-1, 24).sum(dim=-1)  # (135,)
    height = growth.cumsum(dim=-1) / 1000.0

    # End-of-day tau_norm for diagnostics
    tau_norm_daily = tau_norm_hourly.reshape(-1, 24)[:, -1]  # (135,)
    # Per-day Δτ_norm: sum of positive hourly temps / tau_max
    delta_tau_daily = T_hourly_active.clamp(min=0).sum(dim=-1) / tau_max  # (135,)
    num_days = T_hourly_active.shape[0]
    days = torch.arange(dates.tau_start_day, dates.tau_start_day + num_days)
    return {
        "days": days.numpy(),
        "T_active": T_daily_active.detach().numpy(),
        "tau_norm": tau_norm_daily.detach().numpy(),
        "delta_tau_daily": delta_tau_daily.detach().numpy(),
        "growth": growth.detach().numpy(),
        "height": height.detach().numpy(),
        "tau_max": float(tau_max),
    }


def _build_genotype_surface(
    params: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute R(T, tau_norm) on a 200x200 grid for a single genotype."""
    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    n_T_free = n_T_basis - 2

    cps_free = params[:NUM_FREE_CPS].reshape(n_T_free, n_tau_basis - 2)
    cps = torch.zeros(1, n_T_basis, n_tau_basis)
    cps[0, 1:-1, 1:-1] = cps_free

    T_vals = torch.linspace(0.0, float(DEFAULT_T_KNOTS[-1]), 200)
    tau_vals = torch.linspace(0.0, 1.0, 200)
    T_grid, tau_grid = torch.meshgrid(T_vals, tau_vals, indexing="ij")

    growth = evaluate_surface(
        T_grid.reshape(1, -1),
        tau_grid.reshape(1, -1),
        cps,
        DEFAULT_T_KNOTS,
        DEFAULT_TAU_KNOTS,
        DEFAULT_DEGREE,
    ).reshape(200, 200)

    return T_vals.numpy(), tau_vals.numpy(), growth.detach().numpy()


def _plot_one_trajectory_diagnostic(
    diagnostics: dict[str, Any],
    T_vals: np.ndarray,
    tau_vals: np.ndarray,
    growth_surface: np.ndarray,
    label: str,
) -> plt.Figure:
    """Create a 4-panel trajectory diagnostic figure for one (year, genotype) sample.

    Layout (3 rows x 2 cols, constrained_layout):
    - Top-left:    Temperature colored by Δτ_norm + τ_norm twin axis
    - Middle-left: Height trajectory colored by growth rate
    - Bottom-left: Phase portrait — growth rate vs height, colored by day
    - Right (tall): Response surface R(T, τ_norm) + plant path overlay
    """
    days_np = diagnostics["days"]
    T_np = diagnostics["T_active"]
    tau_norm_np = diagnostics["tau_norm"]
    growth_np = diagnostics["growth"]
    height_np = diagnostics["height"]
    delta_tau_np = diagnostics["delta_tau_daily"]

    mature_arr = tau_norm_np >= 1.0
    mature_idx = int(np.argmax(mature_arr)) if mature_arr.any() else None
    mature_day = days_np[mature_idx] if mature_idx is not None else None

    fig = plt.figure(figsize=(14, 11), layout="constrained")
    gridspec = GridSpec(3, 2, figure=fig)
    ax_temp = fig.add_subplot(gridspec[0, 0])
    ax_trajectory = fig.add_subplot(gridspec[1, 0])
    ax_phase = fig.add_subplot(gridspec[2, 0])
    ax_surface = fig.add_subplot(gridspec[:, 1])

    time_progress = np.linspace(0, 1, len(days_np))
    day_norm = plt.Normalize(vmin=days_np[0], vmax=days_np[-1])

    def _make_line_collection(
        x: np.ndarray, y: np.ndarray, colors: np.ndarray, cmap: str, norm: Any
    ) -> LineCollection:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line_collection = LineCollection(
            segments,  # ty: ignore[invalid-argument-type]
            cmap=cmap,
            norm=norm,
            linewidth=2.2,
            zorder=4,
        )
        line_collection.set_array(colors)
        return line_collection

    # -- Panel 1: Temperature + τ_norm --
    lc_temp = _make_line_collection(
        days_np, T_np, delta_tau_np, "RdYlBu_r", plt.Normalize(0, delta_tau_np.max())
    )
    ax_temp.add_collection(lc_temp)
    plt.colorbar(lc_temp, ax=ax_temp, label="\u0394\u03c4_norm / day", pad=0.02)
    ax_temp.fill_between(
        days_np, T_np, 0, where=T_np < 0, color="lightgray", alpha=0.5, zorder=1
    )
    ax_temp.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.7)
    if mature_day is not None:
        ax_temp.axvline(
            mature_day,
            color="steelblue",
            ls="--",
            lw=1.2,
            alpha=0.9,
            label=f"maturity day {mature_day}",
        )
        ax_temp.axvspan(
            mature_day,
            days_np[-1],
            color="lightgray",
            alpha=0.35,
            zorder=0,
            label="\u03c4_norm = 1 (no growth)",
        )
    ax_tau = ax_temp.twinx()
    ax_tau.plot(days_np, tau_norm_np, color="darkorange", lw=1.5, zorder=3)
    ax_tau.axhline(1.0, color="darkorange", lw=0.7, ls=":", alpha=0.7)
    ax_tau.set_ylim(0, 1.15)
    ax_tau.set_ylabel("\u03c4_norm", color="darkorange", fontsize=9)
    ax_tau.tick_params(axis="y", labelcolor="darkorange", labelsize=8)
    ax_temp.set_xlim(days_np[0], days_np[-1])
    ax_temp.set_ylabel("T (\u00b0C)")
    ax_temp.set_title(
        "Daily mean T & \u03c4_norm (color = \u0394\u03c4_norm/day)", fontsize=9
    )
    ax_temp.grid(True, alpha=0.2)
    if mature_day is not None:
        ax_temp.legend(fontsize=7, loc="upper left")

    # -- Panel 2: Height trajectory --
    norm_growth = plt.Normalize(vmin=0, vmax=max(float(growth_np.max()), 1e-3))
    lc_height = _make_line_collection(
        days_np, height_np, growth_np, "YlOrRd", norm_growth
    )
    ax_trajectory.add_collection(lc_height)
    plt.colorbar(lc_height, ax=ax_trajectory, label="Growth (mm/day)", pad=0.02)
    if mature_day is not None:
        ax_trajectory.axvline(mature_day, color="steelblue", ls="--", lw=1.2, alpha=0.8)
    ax_trajectory.set_xlim(days_np[0], days_np[-1])
    ax_trajectory.set_ylim(0, max(float(height_np.max()) * 1.1, 0.05))
    ax_trajectory.set_ylabel("Height (m)")
    ax_trajectory.set_title("Height trajectory (color = growth rate)", fontsize=9)
    ax_trajectory.grid(True, alpha=0.25)

    # -- Panel 3: Phase portrait — growth rate vs height --
    lc_phase = _make_line_collection(
        height_np, growth_np, time_progress, "plasma", plt.Normalize(0, 1)
    )
    ax_phase.add_collection(lc_phase)
    sm_day = plt.cm.ScalarMappable(cmap="plasma", norm=day_norm)
    sm_day.set_array([])
    plt.colorbar(sm_day, ax=ax_phase, label="Day (Sep 1 = 0)", pad=0.02)
    ax_phase.scatter(
        [height_np[0]],
        [growth_np[0]],
        color="green",
        s=70,
        zorder=6,
        marker="o",
        edgecolors="k",
        lw=0.8,
        label="start",
    )
    if mature_idx is not None:
        ax_phase.scatter(
            [height_np[mature_idx]],
            [growth_np[mature_idx]],
            color="steelblue",
            s=90,
            zorder=6,
            marker="*",
            edgecolors="k",
            lw=0.8,
            label="maturity",
        )
    ax_phase.scatter(
        [height_np[-1]],
        [growth_np[-1]],
        color="red",
        s=70,
        zorder=6,
        marker="X",
        edgecolors="k",
        lw=0.8,
        label="end",
    )
    ax_phase.set_xlim(0, max(float(height_np.max()) * 1.1, 0.05))
    ax_phase.set_ylim(bottom=0)
    ax_phase.set_xlabel("Height (m)")
    ax_phase.set_ylabel("Growth (mm/day)")
    ax_phase.set_title("Phase portrait: growth vs. height (color = day)", fontsize=9)
    ax_phase.grid(True, alpha=0.25)
    ax_phase.legend(fontsize=7)

    # -- Right panel: Response surface + path --
    contour_set = ax_surface.contourf(
        tau_vals, T_vals, growth_surface, levels=20, cmap="viridis", alpha=0.85
    )
    plt.colorbar(contour_set, ax=ax_surface, label="R(T, \u03c4_norm) [mm/h]", pad=0.02)
    for T_k in DEFAULT_T_KNOTS.numpy():
        ax_surface.axhline(T_k, color="white", lw=0.5, alpha=0.35, ls="--")
    for tau_k in DEFAULT_TAU_KNOTS.numpy():
        ax_surface.axvline(tau_k, color="white", lw=0.5, alpha=0.35, ls="--")
    lc_surface = _make_line_collection(
        tau_norm_np, T_np, time_progress, "plasma", plt.Normalize(0, 1)
    )
    lc_surface.set_alpha(0.95)
    ax_surface.add_collection(lc_surface)
    plt.colorbar(sm_day, ax=ax_surface, label="Day (Sep 1 = 0)", pad=0.02)
    ax_surface.scatter(
        [tau_norm_np[0]],
        [T_np[0]],
        color="green",
        s=90,
        zorder=8,
        marker="o",
        edgecolors="k",
        lw=0.8,
        label="start",
    )
    if mature_idx is not None:
        ax_surface.scatter(
            [tau_norm_np[mature_idx]],
            [T_np[mature_idx]],
            color="white",
            s=90,
            zorder=9,
            marker="*",
            edgecolors="k",
            lw=0.8,
            label=f"maturity day {mature_day}",
        )
    ax_surface.scatter(
        [tau_norm_np[-1]],
        [T_np[-1]],
        color="red",
        s=90,
        zorder=8,
        marker="X",
        edgecolors="k",
        lw=0.8,
        label="end",
    )
    ax_surface.set_xlabel("\u03c4_norm")
    ax_surface.set_ylabel("T (\u00b0C)")
    ax_surface.set_xlim(0, 1)
    ax_surface.set_ylim(0, float(DEFAULT_T_KNOTS[-1]))
    ax_surface.set_title("R(T, \u03c4_norm) surface + plant path", fontsize=10)
    ax_surface.legend(fontsize=7, loc="lower right")

    fig.suptitle(label, fontsize=11, fontweight="bold")
    return fig


def plot_trajectory_diagnostic_samples(
    year_temps: dict[int, torch.Tensor],
    genotype_pool: GenotypePool,
    output_dir: Path,
    *,
    dates: HeightDates,
) -> None:
    """Generate 5 individual trajectory diagnostic figures spanning maturity dynamics.

    Selects 5 (year, genotype) pairs ranging from "τ_norm never reaches 1" to
    "early maturity with long post-maturity tail". Each sample is saved as a
    separate figure: plots/height/trajectory_diagnostic_{1..5}.png.

    Args:
        year_temps: Per-year mean temperature tensors, shape (274, 24) each.
        genotype_pool: GenotypePool to draw genotypes from.
        output_dir: Root output directory.
    """
    setup_plot_style()

    # Compute cumulative active-season tau for each available year
    available_years = sorted(y for y in year_temps if y != Fip1Facts().held_out_year)
    cumulative_tau = {
        y: float(
            year_temps[y][dates.tau_start_idx :]
            .reshape(-1)
            .clamp(min=0)
            .cumsum(dim=-1)[-1]
        )
        for y in available_years
    }

    # Sort years by cumulative tau (cool → warm), genotypes by tau_max (high → low)
    years_sorted = sorted(available_years, key=lambda y: cumulative_tau[y])
    tau_maxes = genotype_pool.params[:, -1].detach().numpy()
    genotype_order = np.argsort(tau_maxes)[::-1]  # highest tau_max first

    num_years = len(years_sorted)
    num_pool = len(genotype_order)
    num_samples = 5

    # Build 5 pairs: t=0 → (coldest year, highest tau_max); t=1 → (warmest, lowest)
    for i in range(num_samples):
        t = i / (num_samples - 1)
        year = years_sorted[round(t * (num_years - 1))]
        genotype_idx = int(genotype_order[round(t * (num_pool - 1))])

        params = genotype_pool.params[genotype_idx]  # (37,)
        tau_max_val = float(params[-1])
        status = (
            "reaches maturity" if tau_max_val < cumulative_tau[year] else "no maturity"
        )

        diagnostics = _run_bspline_forward_with_intermediates(
            year_temps[year], params, dates=dates
        )
        T_vals, tau_vals, growth_surface = _build_genotype_surface(params)

        cum_tau = cumulative_tau[year]
        figure_label = (
            f"Sample {i + 1}/5 \u2014 Year {year} | "
            f"\u03c4_max={tau_max_val:.0f} | cum_\u03c4={cum_tau:.0f} | {status}"
        )
        fig = _plot_one_trajectory_diagnostic(
            diagnostics, T_vals, tau_vals, growth_surface, figure_label
        )
        save_figure(fig, f"height/plots/trajectory_diagnostic_{i + 1}", output_dir)
        plt.close(fig)


def plot_height_distribution_comparison(
    fip1_data: datasets.Dataset,
    simulated_heights_by_year: dict[int, np.ndarray],
    output_dir: Path,
) -> None:
    """Compare FIP1 vs synthetic final height distributions.

    Creates 1x3 grid:
    - Left: Overlaid histograms of final heights
    - Middle: Mean height by year (FIP1 vs Synthetic)
    - Right: Q-Q plot

    Saves: plots/height/distribution.png
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fip1_heights_list = fip1_data["height_values"]
    fip1_years = fip1_data["harvest_year"]

    def _top3_median_final_height(trajectory: torch.Tensor) -> float:
        values = trajectory.numpy().astype(np.float64)
        if values.size == 0:
            return float("nan")
        top_k = min(3, values.size)
        top_vals = np.partition(values, values.size - top_k)[-top_k:]
        return float(np.median(top_vals))

    fip1_final_heights = np.array(
        [_top3_median_final_height(heights) for heights in fip1_heights_list]
    )

    simulated_heights = np.concatenate(list(simulated_heights_by_year.values()))

    # Plot histograms (left panel)
    ax = axes[0]
    bins = np.linspace(0.4, 1.4, 41)
    ax.hist(
        fip1_final_heights,
        bins=bins,
        alpha=0.5,
        color=COLOR_FIP1,
        label=f"FIP1 (\u03bc={np.mean(fip1_final_heights):.3f}, "
        f"\u03c3={np.std(fip1_final_heights):.3f})",
        density=True,
    )
    ax.hist(
        simulated_heights,
        bins=bins,
        alpha=0.5,
        color=COLOR_SYNTHETIC,
        label=f"Synthetic (\u03bc={np.mean(simulated_heights):.3f}, "
        f"\u03c3={np.std(simulated_heights):.3f})",
        density=True,
    )
    ax.axvline(
        np.mean(fip1_final_heights), color=COLOR_FIP1, linestyle="--", linewidth=2
    )
    ax.axvline(
        np.mean(simulated_heights), color=COLOR_SYNTHETIC, linestyle="--", linewidth=2
    )
    ax.set_xlabel("Final Height (m)")
    ax.set_ylabel("Density")
    ax.set_title("Height Distribution Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Middle: Mean height by year
    ax = axes[1]
    fip1_by_year: dict[int, list[float]] = {}
    for i, heights in enumerate(fip1_heights_list):
        year = int(fip1_years[i].item())
        if year not in fip1_by_year:
            fip1_by_year[year] = []
        fip1_by_year[year].append(_top3_median_final_height(heights))

    years = sorted(fip1_by_year.keys())
    fip1_means = [np.mean(fip1_by_year[y]) for y in years]
    fip1_stds = [np.std(fip1_by_year[y]) for y in years]
    sim_means = [
        float(np.mean(simulated_heights_by_year[y]))
        if y in simulated_heights_by_year
        else np.nan
        for y in years
    ]
    sim_stds = [
        float(np.std(simulated_heights_by_year[y]))
        if y in simulated_heights_by_year
        else 0.0
        for y in years
    ]

    x = np.arange(len(years))
    width = 0.35
    ax.bar(
        x - width / 2,
        fip1_means,
        width,
        yerr=fip1_stds,
        color=COLOR_FIP1,
        alpha=0.7,
        label="FIP1",
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        sim_means,
        width,
        yerr=sim_stds,
        color=COLOR_SYNTHETIC,
        alpha=0.7,
        label="Synthetic",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Height (m)")
    ax.set_title("Mean Height by Year")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: Q-Q plot
    ax = axes[2]
    fip1_sorted = np.sort(fip1_final_heights)
    synth_sorted = np.sort(simulated_heights)
    num_points = min(100, len(fip1_sorted), len(synth_sorted))
    fip1_quantiles = np.percentile(fip1_sorted, np.linspace(0, 100, num_points))
    synth_quantiles = np.percentile(synth_sorted, np.linspace(0, 100, num_points))
    ax.scatter(fip1_quantiles, synth_quantiles, alpha=0.6, s=20)
    lims = [
        min(fip1_quantiles.min(), synth_quantiles.min()),
        max(fip1_quantiles.max(), synth_quantiles.max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("FIP1 Quantiles")
    ax.set_ylabel("Synthetic Quantiles")
    ax.set_title("Q-Q Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Height Distribution Validation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_figure(fig, "height/plots/distribution", output_dir)


def plot_warm_short(
    genotype_pool: GenotypePool,
    output_dir: Path,
    calibration_params_path: str | Path | None = None,
    num_sites: int = 10,
    num_years: int = 10,
    seed: int = 42,
    *,
    dates: HeightDates,
) -> None:
    """Warm→short diagnostic: warmer synthetic environments produce shorter plants.

    Generates fully synthetic temperatures via YearsitePool.sample_synthetic(),
    runs the B-spline forward model for all genotype x yearsite combinations,
    and shows the correlation between mean growing-season temperature and
    mean simulated final height per yearsite.

    Creates 1x2 figure:
    - Left: Scatter of mean temperature vs mean height per yearsite,
      with linear regression line and Pearson r / p annotation.
    - Right: Bar chart of mean height per yearsite sorted cool→warm,
      colored by temperature.

    Saves: plots/height/warm_short.png
    """
    setup_plot_style()

    # Generate synthetic temperatures
    yearsite_pool = YearsitePool.sample_synthetic(
        num_sites=num_sites,
        num_years=num_years,
        seed=seed,
        calibration_params_path=calibration_params_path,
    )

    num_genotypes = len(genotype_pool)
    num_yearsites = len(yearsite_pool)

    # Run forward model for each yearsite with all genotypes
    mean_temps = np.empty(num_yearsites)
    mean_heights = np.empty(num_yearsites)
    yearsite_ids = []

    for i in range(num_yearsites):
        temp = yearsite_pool.temperatures[i]  # (274, 24)
        temp_batch = temp.unsqueeze(0).expand(num_genotypes, -1, -1)

        trajectories, _ = _compute_height_trajectories(
            temp_batch, genotype_pool=genotype_pool, dates=dates
        )
        final_heights = trajectories.max(dim=-1).values

        mean_temps[i] = temp.mean().item()
        mean_heights[i] = final_heights.mean().item()
        yearsite_ids.append(yearsite_pool.yearsite_ids[i])

    # Sort by temperature for bar chart
    temp_order = np.argsort(mean_temps)
    sorted_ids = [yearsite_ids[i] for i in temp_order]
    sorted_temps = mean_temps[temp_order]
    sorted_heights = mean_heights[temp_order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -- Left: scatter with regression --
    ax = axes[0]
    ax.scatter(mean_temps, mean_heights, s=40, alpha=0.7, zorder=3)

    slope, intercept, r_value, p_value, _ = stats.linregress(mean_temps, mean_heights)
    x_fit = np.linspace(mean_temps.min(), mean_temps.max(), 50)
    ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Mean Growing-Season Temperature (\u00b0C)")
    ax.set_ylabel("Mean Final Height (m)")
    ax.set_title("Warm \u2192 Short: Temperature vs Height")
    ax.annotate(
        f"r = {r_value:.3f}, p = {p_value:.3f}",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "wheat", "alpha": 0.8},
    )
    ax.grid(True, alpha=0.3)

    # -- Right: bar chart sorted cool→warm --
    ax = axes[1]
    cmap = plt.colormaps["coolwarm"]
    norm = plt.Normalize(sorted_temps.min(), sorted_temps.max())
    colors = [cmap(norm(t)) for t in sorted_temps]

    x = np.arange(len(sorted_ids))
    ax.bar(x, sorted_heights, color=colors, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_ids, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Yearsite (sorted cool \u2192 warm)")
    ax.set_ylabel("Mean Final Height (m)")
    ax.set_title("Height by Yearsite (cool \u2192 warm)")
    ax.grid(True, alpha=0.3, axis="y")

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=ax, label="Mean Temperature (\u00b0C)")

    fig.suptitle(
        "Warm \u2192 Short: Fully Synthetic Verification",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_figure(fig, "height/plots/warm_short", output_dir)
