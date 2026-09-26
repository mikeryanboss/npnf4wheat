"""Shared plotting helpers and style constants for calibration plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Standard colors
COLOR_SWISS = "#1f77b4"  # Blue
COLOR_FIP1 = "#2ca02c"  # Green
COLOR_SYNTHETIC = "#d62728"  # Red


def setup_plot_style() -> None:
    """Set up consistent plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11


def save_figure(fig: Figure, name: str, output_dir: Path) -> None:
    """Save figure to ``output_dir / <name>.png``.

    Callers provide the full relative path (e.g.
    ``"temperature/1_seasonal/seasonal"``).  Intermediate directories
    are created automatically.
    """
    path = output_dir / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
