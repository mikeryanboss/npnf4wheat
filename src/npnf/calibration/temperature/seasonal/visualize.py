"""Seasonal model visualization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf as sm_acf

from npnf.calibration.shared.plotting import (
    COLOR_SWISS,
    COLOR_SYNTHETIC,
    save_figure,
    setup_plot_style,
)
from npnf.data.synthetic.temperature.harmonic import harmonic_components

if TYPE_CHECKING:
    from npnf.calibration.temperature.data import ObservedData, SeasonalCalibration


def plot_seasonal_fit(
    calibration_arrays: ObservedData,
    fitted_params: SeasonalCalibration,
    output_dir: Path,
) -> None:
    """Plot seasonal model fit vs Swiss data."""
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    swiss_daily = calibration_arrays.daily_means  # (N, 274)
    overall_mean = swiss_daily.mean(axis=0)

    days = np.arange(274)

    h = fitted_params.harmonic
    rho = fitted_params.ar_coefficient

    fitted = h.values

    # Component decomposition
    base = h.params.base
    amp1 = h.params.amplitude
    phase1 = h.params.phase
    amp2 = h.params.amplitude_2
    phase2 = h.params.phase_2

    # Top-left: Daily means with fitted curve
    ax1 = axes[0, 0]
    ax1.fill_between(
        days,
        swiss_daily.mean(axis=0) - swiss_daily.std(axis=0),
        swiss_daily.mean(axis=0) + swiss_daily.std(axis=0),
        alpha=0.3,
        color=COLOR_SWISS,
        label="Swiss +/- 1 std",
    )
    ax1.plot(days, overall_mean, color=COLOR_SWISS, linewidth=2, label="Swiss Mean")
    ax1.plot(
        days,
        fitted,
        color="black",
        linewidth=2,
        linestyle="--",
        label="Fitted Seasonal",
    )
    ax1.set_xlabel("Day of Season (Nov 1 = 0)")
    ax1.set_ylabel("Temperature (C)")
    ax1.set_title("Seasonal Temperature Pattern")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top-right: Residuals distribution
    ax2 = axes[0, 1]
    residuals = overall_mean - fitted
    ax2.hist(
        residuals,
        bins=30,
        density=True,
        alpha=0.7,
        color=COLOR_SWISS,
        label="Residuals",
    )
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = norm.pdf(x_norm, loc=np.mean(residuals), scale=np.std(residuals))
    ax2.plot(x_norm, y_norm, color=COLOR_SYNTHETIC, linewidth=2, label="Normal Fit")
    ax2.set_xlabel("Residual (C)")
    ax2.set_ylabel("Density")
    ax2.set_title("Residuals Distribution")
    ax2.legend()

    # Bottom-left: Harmonic components
    ax3 = axes[1, 0]
    h1, h2 = harmonic_components(days, amp1, phase1, amp2, phase2)
    ax3.plot(days, h1, label="First Harmonic", linewidth=2)
    ax3.plot(days, h2, label="Second Harmonic", linewidth=2)
    ax3.axhline(y=base, color="gray", linestyle="--")
    ax3.set_xlabel("Day of Season")
    ax3.set_ylabel("Temperature Contribution (C)")
    ax3.set_title("Harmonic Components")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom-right: ACF
    ax4 = axes[1, 1]
    acf_vals = sm_acf(overall_mean, nlags=30, fft=True)
    ax4.bar(np.arange(31), acf_vals, alpha=0.7, color=COLOR_SWISS)
    if rho:
        expected_acf = [rho**k for k in range(31)]
        ax4.plot(
            np.arange(31),
            expected_acf,
            "r--",
            linewidth=2,
            label=f"AR(1) rho={rho:.3f}",
        )
    ax4.set_xlabel("Lag (days)")
    ax4.set_ylabel("Autocorrelation")
    ax4.set_title("Autocorrelation Function")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle("Seasonal Temperature Model Fit", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(fig, "temperature/1_seasonal/seasonal", output_dir)
