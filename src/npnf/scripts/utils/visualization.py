"""Visualization utilities for NPNF scripts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def save_figure(fig: Figure, path: Path, dpi: int = 200, *, close: bool = True) -> None:
    """Save figure to path, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if close:
        plt.close(fig)


def plot_ecdf(ax: Axes, values: np.ndarray, label: str, color) -> None:
    """Empirical CDF of the finite values, drawn on at most 1500 ranks."""
    values = np.asarray(values).reshape(-1)
    values = np.sort(values[np.isfinite(values)])
    if values.size:
        # Plot ranks on a bounded grid, without changing the empirical quantiles.
        indices = np.unique(
            np.linspace(0, values.size - 1, min(1500, values.size)).astype(int)
        )
        ax.plot(values[indices], (indices + 1) / values.size, label=label, color=color)
    ax.set_ylabel("Cumulative fraction")
