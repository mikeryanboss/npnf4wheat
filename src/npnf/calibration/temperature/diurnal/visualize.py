"""Diurnal parameter visualization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from npnf.calibration.shared.plotting import (
    COLOR_SWISS,
    COLOR_SYNTHETIC,
    save_figure,
    setup_plot_style,
)

if TYPE_CHECKING:
    from npnf.calibration.temperature.data import ObservedData
    from npnf.data.synthetic.temperature import DiurnalTimingParams


def plot_extracted_params(
    calibration_arrays: ObservedData, output_dir: Path, timing: DiurnalTimingParams
) -> None:
    """Diagnostic plots for the data-derived parameters from Step 1."""
    setup_plot_style()

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    hours = np.arange(24)

    temps = calibration_arrays.temps

    # ---- (0, 0) Diurnal pattern with peak hour ----
    ax = axes[0, 0]
    mean_diurnal = temps.mean(axis=(0, 1))
    std_diurnal = temps.std(axis=(0, 1))
    ax.fill_between(
        hours,
        mean_diurnal - std_diurnal,
        mean_diurnal + std_diurnal,
        alpha=0.25,
        color=COLOR_SWISS,
        label="+/- 1 std",
    )
    ax.plot(hours, mean_diurnal, color=COLOR_SWISS, linewidth=2, label="Mean")
    empirical_peak_hr = int(np.argmax(mean_diurnal))
    ax.axvline(
        empirical_peak_hr,
        color=COLOR_SYNTHETIC,
        linestyle="--",
        linewidth=1.5,
        label=f"empirical peak = {empirical_peak_hr}h",
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Mean Diurnal Pattern")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=9)

    # ---- (0, 1) Diurnal amplitude noise ratio ----
    ax = axes[0, 1]
    diurnal_amp = calibration_arrays.diurnal_amplitude
    ratio = (diurnal_amp / diurnal_amp.mean()).flatten()
    ax.hist(
        ratio,
        bins=80,
        color=COLOR_SWISS,
        alpha=0.7,
        density=True,
        label="Diurnal amp ratio",
    )
    dmin = float(np.percentile(ratio, 5))
    dmax = float(np.percentile(ratio, 95))
    ax.axvline(
        dmin,
        color=COLOR_SYNTHETIC,
        linestyle="--",
        linewidth=1.5,
        label=f"noise_min = {dmin:.2f}",
    )
    ax.axvline(
        dmax,
        color=COLOR_SYNTHETIC,
        linestyle="-.",
        linewidth=1.5,
        label=f"noise_max = {dmax:.2f}",
    )
    ax.set_xlabel("Diurnal amplitude / mean amplitude")
    ax.set_ylabel("Density")
    ax.set_title("Diurnal Amplitude Noise")
    ax.legend(fontsize=9)

    # ---- (1, 0) Peak hour deviation ----
    ax = axes[1, 0]
    peak_dev = calibration_arrays.peak_hours - timing.peak_hour
    daytime = ~np.isnan(calibration_arrays.daytime_peak_hours)
    daytime_dev = peak_dev[daytime]
    ax.hist(
        daytime_dev,
        bins=60,
        color=COLOR_SWISS,
        alpha=0.7,
        density=True,
        label="Daytime peak deviation",
    )
    p5 = float(np.percentile(daytime_dev, 5))
    p95 = float(np.percentile(daytime_dev, 95))
    ax.axvline(
        p5, color=COLOR_SYNTHETIC, linestyle="--", linewidth=1.5, label=f"p5 = {p5:.2f}"
    )
    ax.axvline(
        p95,
        color=COLOR_SYNTHETIC,
        linestyle="-.",
        linewidth=1.5,
        label=f"p95 = {p95:.2f}",
    )
    ax.set_xlabel(f"Peak hour - {timing.peak_hour:.1f}h (hours)")
    ax.set_ylabel("Density")
    ax.set_title("Peak Hour Deviation (Daytime)")
    ax.legend(fontsize=9)

    # ---- (1, 1) Daily max temperature distribution with soft cap ----
    ax = axes[1, 1]
    flat_daily_max = calibration_arrays.daily_max.flatten()
    ax.hist(
        flat_daily_max,
        bins=100,
        color=COLOR_SWISS,
        alpha=0.7,
        density=True,
        label="Daily max temps",
    )
    ax.set_xlabel("Daily max temperature (C)")
    ax.set_ylabel("Density")
    ax.set_title("Daily Max Temperature Distribution")
    ax.legend(fontsize=9)

    # ---- (2, 0) Weather anomaly distribution with soft cap ----
    ax = axes[2, 0]
    abs_anomalies = calibration_arrays.abs_weather_anomalies
    ax.hist(
        abs_anomalies,
        bins=60,
        color=COLOR_SWISS,
        alpha=0.7,
        density=True,
        label="Abs weather anomalies",
    )
    cap = calibration_arrays.weather_anomaly_soft_cap
    ax.axvline(
        cap,
        color=COLOR_SYNTHETIC,
        linestyle="--",
        linewidth=1.5,
        label=f"anomaly_soft_cap = {cap:.1f}",
    )
    ax.set_xlabel("Absolute anomaly (C)")
    ax.set_ylabel("Density")
    ax.set_title("Weather Anomaly Distribution")
    ax.legend(fontsize=9)

    # ---- (2, 1) Trough hour deviation ----
    ax = axes[2, 1]
    trough_dev = calibration_arrays.trough_hours - timing.trough_hour
    morning = ~np.isnan(calibration_arrays.sunrise_trough_hours)
    morning_dev = trough_dev[morning]
    ax.hist(
        morning_dev,
        bins=60,
        color=COLOR_SWISS,
        alpha=0.7,
        density=True,
        label="Morning trough deviation",
    )
    p5 = float(np.percentile(morning_dev, 5))
    p95 = float(np.percentile(morning_dev, 95))
    ax.axvline(
        p5, color=COLOR_SYNTHETIC, linestyle="--", linewidth=1.5, label=f"p5 = {p5:.2f}"
    )
    ax.axvline(
        p95,
        color=COLOR_SYNTHETIC,
        linestyle="-.",
        linewidth=1.5,
        label=f"p95 = {p95:.2f}",
    )
    ax.set_xlabel(f"Trough hour - {timing.trough_hour:.1f}h (hours)")
    ax.set_ylabel("Density")
    ax.set_title("Trough Hour Deviation (Morning)")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Diurnal Parameters - Diagnostic Overview", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    save_figure(fig, "temperature/2_diurnal/extracted_params", output_dir)
