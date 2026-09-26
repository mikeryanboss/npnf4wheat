"""Height optimization diagnostic visualization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from npnf.calibration.shared.plotting import (
    COLOR_SYNTHETIC,
    save_figure,
    setup_plot_style,
)

if TYPE_CHECKING:
    from npnf.calibration.height.data import HeightOptimizationResults


def plot_height_optimization(
    trial_history: list[dict[str, Any]],
    results: HeightOptimizationResults,
    output_dir: Path,
) -> None:
    """Plot height optimization diagnostics from trial history.

    Creates 2x2 grid:
    - Top-left: Optimization history (loss vs trial)
    - Top-right: Height loss vs ordering loss
    - Bottom-left: Per-year simulated height
    - Bottom-right: Loss component breakdown

    Saves: plots/height/optimization.png
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    trial_numbers = [t["number"] for t in trial_history]
    trial_values = [t["value"] for t in trial_history]

    best_number = results.best_trial
    best = next((t for t in trial_history if t["number"] == best_number), None)

    # Top-left: Optimization history
    ax = axes[0, 0]
    if trial_values:
        ax.plot(trial_numbers, trial_values, "b-", alpha=0.4, linewidth=0.5)
        running_best = np.minimum.accumulate(trial_values)
        ax.plot(trial_numbers, running_best, "r-", linewidth=2, label="Best so far")
        ax.axhline(
            y=min(trial_values),
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Best: {min(trial_values):.4f}",
        )
        if best is not None:
            ax.scatter(
                [best["number"]], [best["value"]], c="red", s=200, marker="*", zorder=5
            )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Height loss vs ordering loss
    ax = axes[0, 1]
    if trial_history:
        height_losses = [t.get("height_loss", np.nan) for t in trial_history]
        ordering_losses = [t.get("ordering_loss", np.nan) for t in trial_history]
        ax.scatter(
            height_losses,
            ordering_losses,
            c=trial_values,
            cmap="viridis_r",
            alpha=0.6,
            s=10,
        )
        if best is not None:
            ax.scatter(
                [best.get("height_loss") or 0],
                [best.get("ordering_loss") or 0],
                c="red",
                s=200,
                marker="*",
                label="Best",
                zorder=5,
            )
            ax.legend()
    ax.set_xlabel("Height Loss")
    ax.set_ylabel("Ordering Loss")
    ax.set_title("Loss Components")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Per-year height comparison
    ax = axes[1, 0]
    years = sorted(results.sim_height_means.keys())
    sim_means = [results.sim_height_means[y] for y in years]
    x = np.arange(len(years))
    ax.bar(x, sim_means, color=COLOR_SYNTHETIC, alpha=0.7, label="Simulated")
    ax.axhline(
        results.mean_height,
        color=COLOR_SYNTHETIC,
        linestyle="--",
        alpha=0.5,
        label=f"Sim mean: {results.mean_height:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Height (m)")
    ax.set_title("Simulated Height by Year")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: empty (reserved for future diagnostics)
    ax = axes[1, 1]
    ax.set_axis_off()

    fig.suptitle(
        f"Height Calibration Optimization ({len(trial_history)} trials)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(fig, "height/plots/optimization", output_dir)
