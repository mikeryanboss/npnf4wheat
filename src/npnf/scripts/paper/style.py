"""Unified styling for paper figures."""

import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

mpl.use("Agg")

# Seaborn colorblind palette
PALETTE = sns.color_palette("colorblind")
COLORS = {
    "blue": PALETTE[0],
    "orange": PALETTE[1],
    "green": PALETTE[2],
    "red": PALETTE[3],
    "purple": PALETTE[4],
    "brown": PALETTE[5],
    "pink": PALETTE[6],
    "gray": PALETTE[7],
    "yellow": PALETTE[8],
    "cyan": PALETTE[9],
}
COLOR_LIST = list(COLORS.values())

# Standard figure sizes (width, height) in inches
FIGURE_SIZES = {"single": (5, 3.5), "wide": (7, 4), "dual": (10, 4), "triple": (15, 4)}

# Paper-ready rcParams
STYLE_CONFIG = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}


def setup_style() -> None:
    """Initialize consistent styling for all figures."""
    sns.set_theme(style="whitegrid", palette="colorblind", rc=STYLE_CONFIG)


def get_figure(
    size: str = "single", nrows: int = 1, ncols: int = 1, **kwargs
) -> tuple[plt.Figure, Any]:
    """Create figure with standard sizing."""
    figsize = FIGURE_SIZES.get(size, FIGURE_SIZES["single"])
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure in PNG and PDF formats."""
    path = Path(path)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{fmt}"))


def log_ticks(low: float, high: float) -> list[float]:
    """1-2-5 tick values between low and high (both positive)."""
    return [
        float(f"{mantissa}e{exponent}")
        for exponent in range(
            math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1
        )
        for mantissa in (1, 2, 5)
        if low <= float(f"{mantissa}e{exponent}") <= high
    ]


def set_log_y_axis(
    ax: plt.Axes, values: pd.Series, *, bar_ends: pd.Series | None = None
) -> None:
    """Log y-axis around the plotted values with 1-2-5 tick labels.

    ``bar_ends`` are the ends of error bars drawn on the values. The positive
    ends are kept inside the axis, so only a bar that reaches zero runs to the
    bottom and a cut bar is not read as one.
    """
    smallest = float(values.min())
    if smallest <= 0:
        msg = (
            f"--log-y needs positive values, the smallest plotted value is {smallest:g}"
        )
        raise ValueError(msg)
    if bar_ends is not None:
        values = pd.concat([values, bar_ends[bar_ends > 0]])
    low, high = float(values.min()) / 1.4, float(values.max()) * 1.4
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ticks = log_ticks(low, high)
    ax.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
    ax.minorticks_off()
