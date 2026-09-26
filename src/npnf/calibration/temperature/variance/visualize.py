"""Variance calibration visualization functions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf as sm_acf

from npnf.calibration.shared.plotting import (
    COLOR_SWISS,
    COLOR_SYNTHETIC,
    save_figure,
    setup_plot_style,
)
from npnf.calibration.temperature.variance.extraction import detrend_year_means

if TYPE_CHECKING:
    from datasets import Dataset
    from optuna import Study

    from npnf.calibration.temperature.data import CalibrationData, ObservedData
    from npnf.data.synthetic.temperature import TemperatureParams

METRIC_LABELS = {
    "diurnal_amplitude": "Diurnal Amplitude",
    "winter_diurnal_daily_std": "Winter Diurnal Daily Std",
    "summer_diurnal_daily_std": "Summer Diurnal Daily Std",
    "day_to_day_std": "Day-to-Day Std",
    "hour_of_max": "Hour of Max",
    "hour_of_min": "Hour of Min",
    "hot_days": "Hot Days",
    "cold_days": "Cold Days",
    "max_temp_reached": "Max Temp Reached",
    "hour_of_max_std": "Hour-of-Max Std",
    "hour_of_min_std": "Hour-of-Min Std",
    "morning_mean": "Morning Mean (4\u201310h)",
    "afternoon_mean": "Afternoon Mean (14\u201320h)",
}

METRIC_UNITS = {
    "diurnal_amplitude": "°C",
    "winter_diurnal_daily_std": "°C",
    "summer_diurnal_daily_std": "°C",
    "day_to_day_std": "°C",
    "hour_of_max": "hour",
    "hour_of_min": "hour",
    "hot_days": "days",
    "cold_days": "days",
    "max_temp_reached": "°C",
    "hour_of_max_std": "hour",
    "hour_of_min_std": "hour",
    "morning_mean": "°C",
    "afternoon_mean": "°C",
}


def _swiss_temps_from_dataset(data: Dataset) -> np.ndarray:
    """Reshape Swiss dataset to (N, 274, 24) in one vectorized call."""
    return np.asarray(data["temperature_values"], dtype=float).reshape(-1, 274, 24)


def plot_temperature_validation(
    swiss_data: Dataset, synthetic_data: np.ndarray, output_dir: Path
) -> None:
    """Compare synthetic temperatures with Swiss data.

    This validates the variance calibration by checking whether synthetic
    temperatures generated with calibrated variance parameters match the
    statistical properties of real Swiss weather data.
    """
    setup_plot_style()

    swiss_temps = _swiss_temps_from_dataset(swiss_data)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1, Col 1: Daily mean distribution
    ax1 = axes[0, 0]
    swiss_daily = swiss_temps.mean(axis=-1).flatten()
    synth_daily = synthetic_data.mean(axis=-1).flatten()
    ax1.hist(
        swiss_daily, bins=50, alpha=0.5, color=COLOR_SWISS, label="Swiss", density=True
    )
    ax1.hist(
        synth_daily,
        bins=50,
        alpha=0.5,
        color=COLOR_SYNTHETIC,
        label="Synthetic",
        density=True,
    )
    ax1.set_xlabel("Daily Mean Temperature (C)")
    ax1.set_ylabel("Density")
    ax1.set_title("Daily Mean Distribution")
    ax1.legend()

    # Row 1, Col 2: Diurnal cycle with +/-1 sigma variance bands
    ax2 = axes[0, 1]
    swiss_diurnal_all = swiss_temps.mean(axis=1)  # (num_samples, 24)
    swiss_diurnal_mean = swiss_diurnal_all.mean(axis=0)  # (24,)
    swiss_diurnal_std = swiss_diurnal_all.std(axis=0)  # (24,)
    synth_diurnal_all = synthetic_data.mean(axis=1)  # (num_samples, 24)
    synth_diurnal_mean = synth_diurnal_all.mean(axis=0)  # (24,)
    synth_diurnal_std = synth_diurnal_all.std(axis=0)  # (24,)

    hours = np.arange(24)
    ax2.plot(hours, swiss_diurnal_mean, color=COLOR_SWISS, linewidth=2, label="Swiss")
    ax2.fill_between(
        hours,
        swiss_diurnal_mean - swiss_diurnal_std,
        swiss_diurnal_mean + swiss_diurnal_std,
        color=COLOR_SWISS,
        alpha=0.2,
    )
    ax2.plot(
        hours, synth_diurnal_mean, color=COLOR_SYNTHETIC, linewidth=2, label="Synthetic"
    )
    ax2.fill_between(
        hours,
        synth_diurnal_mean - synth_diurnal_std,
        synth_diurnal_mean + synth_diurnal_std,
        color=COLOR_SYNTHETIC,
        alpha=0.2,
    )
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Temperature (C)")
    ax2.set_title("Diurnal Cycle (+/-1 sigma)")
    ax2.legend()

    # Row 1, Col 3: Seasonal pattern with +/-1 sigma variance bands
    ax3 = axes[0, 2]
    swiss_daily_means = swiss_temps.mean(axis=-1)  # (num_samples, 274)
    swiss_seasonal_mean = swiss_daily_means.mean(axis=0)  # (274,)
    swiss_seasonal_std = swiss_daily_means.std(axis=0)  # (274,)
    synth_daily_means = synthetic_data.mean(axis=-1)  # (num_samples, 274)
    synth_seasonal_mean = synth_daily_means.mean(axis=0)  # (274,)
    synth_seasonal_std = synth_daily_means.std(axis=0)  # (274,)

    days = np.arange(274)
    ax3.plot(days, swiss_seasonal_mean, color=COLOR_SWISS, linewidth=2, label="Swiss")
    ax3.fill_between(
        days,
        swiss_seasonal_mean - swiss_seasonal_std,
        swiss_seasonal_mean + swiss_seasonal_std,
        color=COLOR_SWISS,
        alpha=0.2,
    )
    ax3.plot(
        days, synth_seasonal_mean, color=COLOR_SYNTHETIC, linewidth=2, label="Synthetic"
    )
    ax3.fill_between(
        days,
        synth_seasonal_mean - synth_seasonal_std,
        synth_seasonal_mean + synth_seasonal_std,
        color=COLOR_SYNTHETIC,
        alpha=0.2,
    )
    ax3.set_xlabel("Day of Season (Nov 1 = 0)")
    ax3.set_ylabel("Temperature (C)")
    ax3.set_title("Seasonal Pattern (+/-1 sigma)")
    ax3.legend()

    # Row 1, Col 4: Seasonal diurnal amplitude (daily max - min over 24h)
    ax_amp = axes[0, 3]
    swiss_amplitude = swiss_temps.max(axis=-1) - swiss_temps.min(axis=-1)  # (N, 274)
    swiss_amp_mean = swiss_amplitude.mean(axis=0)  # (274,)
    swiss_amp_std = swiss_amplitude.std(axis=0)  # (274,)
    synth_amplitude = synthetic_data.max(axis=-1) - synthetic_data.min(
        axis=-1
    )  # (N, 274)
    synth_amp_mean = synth_amplitude.mean(axis=0)  # (274,)
    synth_amp_std = synth_amplitude.std(axis=0)  # (274,)

    ax_amp.plot(days, swiss_amp_mean, color=COLOR_SWISS, linewidth=2, label="Swiss")
    ax_amp.fill_between(
        days,
        swiss_amp_mean - swiss_amp_std,
        swiss_amp_mean + swiss_amp_std,
        color=COLOR_SWISS,
        alpha=0.2,
    )
    ax_amp.plot(
        days, synth_amp_mean, color=COLOR_SYNTHETIC, linewidth=2, label="Synthetic"
    )
    ax_amp.fill_between(
        days,
        synth_amp_mean - synth_amp_std,
        synth_amp_mean + synth_amp_std,
        color=COLOR_SYNTHETIC,
        alpha=0.2,
    )
    ax_amp.set_xlabel("Day of Season (Nov 1 = 0)")
    ax_amp.set_ylabel("Diurnal Range (C)")
    ax_amp.set_title("Seasonal Diurnal Amplitude (+/-1 sigma)")
    ax_amp.legend()

    # Row 2, Col 1: Day-to-day std
    ax4 = axes[1, 0]
    swiss_d2d = np.diff(swiss_temps.mean(axis=-1), axis=1).std(axis=0)
    synth_d2d = np.diff(synthetic_data.mean(axis=-1), axis=1).std(axis=0)
    ax4.plot(days[1:], swiss_d2d, color=COLOR_SWISS, linewidth=2, label="Swiss")
    ax4.plot(days[1:], synth_d2d, color=COLOR_SYNTHETIC, linewidth=2, label="Synthetic")
    ax4.set_xlabel("Day of Season")
    ax4.set_ylabel("Day-to-Day Std (C)")
    ax4.set_title("Temperature Variability")
    ax4.legend()

    # Row 2, Col 2: Hot days count
    ax5 = axes[1, 1]
    threshold = 30
    swiss_hot = (swiss_temps.max(axis=-1) > threshold).sum(axis=1)
    synth_hot = (synthetic_data.max(axis=-1) > threshold).sum(axis=1)
    ax5.hist(swiss_hot, bins=20, alpha=0.5, color=COLOR_SWISS, label="Swiss")
    ax5.hist(synth_hot, bins=20, alpha=0.5, color=COLOR_SYNTHETIC, label="Synthetic")
    ax5.set_xlabel(f"Days > {threshold}C")
    ax5.set_ylabel("Count")
    ax5.set_title("Hot Days per Season")
    ax5.legend()

    # Row 2, Col 3: ACF comparison
    ax6 = axes[1, 2]
    swiss_acf = sm_acf(swiss_temps.mean(axis=(0, 2)), nlags=30, fft=True)
    synth_acf = sm_acf(synthetic_data.mean(axis=(0, 2)), nlags=30, fft=True)
    lags = np.arange(31)
    ax6.plot(lags, swiss_acf, color=COLOR_SWISS, linewidth=2, marker="o", label="Swiss")
    ax6.plot(
        lags,
        synth_acf,
        color=COLOR_SYNTHETIC,
        linewidth=2,
        marker="s",
        label="Synthetic",
    )
    ax6.set_xlabel("Lag (days)")
    ax6.set_ylabel("Autocorrelation")
    ax6.set_title("ACF Comparison")
    ax6.legend()

    axes[1, 3].set_visible(False)

    fig.suptitle(
        "Temperature Validation - Swiss vs Synthetic", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(
        fig, "temperature/3_variance/comparison/variance_comparison", output_dir
    )


def plot_variance_optimization(study: Any, output_dir: Path) -> None:
    """Plot Bayesian optimization progress and results.

    Creates 2x2 grid:
    - Top-left: Optimization history (trial vs loss)
    - Top-right: Parameter importance (bar chart)
    - Bottom-left: Best trial parameter values (bar chart)
    - Bottom-right: Per-seed loss distribution (box + strip)

    Saves: plots/temperature/variance_optimization.png
    """
    import optuna

    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Optimization history (complete trials only, pruned as background)
    ax1 = axes[0, 0]
    complete = [
        (t.number, t.value)
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    pruned = [
        (t.number, t.value)
        for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED and t.value is not None
    ]
    if pruned:
        p_nums, p_vals = zip(*pruned, strict=False)
        ax1.scatter(
            p_nums, p_vals, color="gray", alpha=0.25, s=10, label="Pruned (lo-fi)"
        )
    if complete:
        c_nums, c_vals = zip(*complete, strict=False)
        ax1.scatter(c_nums, c_vals, color="b", alpha=0.6, s=12)
        best = min(c_vals)
        ax1.axhline(y=best, color="r", linestyle="--", label=f"Best: {best:.4f}")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Loss")
    ax1.set_title("Optimization History")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Top-right: Parameter importance
    ax2 = axes[0, 1]
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        importance_vals = list(importances.values())
        ax2.barh(params, importance_vals, color=COLOR_SWISS)
        ax2.set_xlabel("Importance")
        ax2.set_title("Parameter Importance")
    except Exception:  # noqa: BLE001
        ax2.text(
            0.5, 0.5, "Importance analysis\nnot available", ha="center", va="center"
        )
        ax2.set_title("Parameter Importance")

    # Bottom-left: Best parameters
    ax3 = axes[1, 0]
    if study.best_trial:
        params = study.best_trial.params
        param_names = list(params.keys())
        param_vals = list(params.values())
        bars = ax3.bar(param_names, param_vals, color=COLOR_SWISS)
        for bar, value in zip(bars, param_vals, strict=False):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax3.set_ylabel("Value")
    ax3.set_title("Best Trial Parameters")
    ax3.grid(True, alpha=0.3, axis="y")

    # Bottom-right: per-seed loss distribution from best trial
    ax4 = axes[1, 1]
    seed_losses = [
        v
        for k, v in study.best_trial.user_attrs.items()
        if k.startswith("seed_") and k.endswith("_loss")
    ]
    if seed_losses:
        ax4.boxplot(seed_losses, vert=True, widths=0.5)
        ax4.scatter(
            [1] * len(seed_losses), seed_losses, alpha=0.6, color=COLOR_SWISS, zorder=3
        )
        mean_val = np.mean(seed_losses)
        std_val = np.std(seed_losses)
        ax4.axhline(
            y=mean_val,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_val:.4f}",
        )
        ax4.set_title(f"Per-Seed Loss (std={std_val:.4f})")
        ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No per-seed data", ha="center", va="center")
        ax4.set_title("Per-Seed Loss Distribution")
    ax4.set_ylabel("Loss")
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Variance Parameter Optimization", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(
        fig, "temperature/3_variance/optimization/variance_optimization", output_dir
    )


def plot_variance_validation(
    swiss_targets: dict[str, dict[str, float]],
    synthetic_components: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Compare std-component decomposition per-metric: Swiss vs synthetic.

    Creates 2x3 grid showing std_site/std_year/std_total for each of 6 metrics.
    This is the validation-focused plot - shows whether synthetic data matches
    Swiss variance structure for each metric individually.

    Args:
        swiss_targets: Swiss calibration targets dict
            Format: {metric: {"std_site": v, "std_year": v, "std_total": v}}
        synthetic_components: Synthetic std components from calibrated generator.
            Same format as swiss_targets.
        output_dir: Output directory for saving figure.

    Saves: plots/temperature/variance.png
    """
    setup_plot_style()

    metrics = list(swiss_targets.keys())
    metric_labels = [METRIC_LABELS.get(m, m) for m in metrics]
    num_cols = 3
    num_rows = math.ceil(len(metrics) / num_cols)
    components = ["std_site", "std_year", "std_total"]

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten()

    x = np.arange(len(components))
    width = 0.35

    for index, (metric, label) in enumerate(zip(metrics, metric_labels, strict=True)):
        ax = axes[index]

        swiss_vals = [swiss_targets.get(metric, {}).get(c, 0) for c in components]
        synth_vals = [
            synthetic_components.get(metric, {}).get(c, 0) for c in components
        ]

        bars1 = ax.bar(
            x - width / 2,
            swiss_vals,
            width,
            label="Swiss Target",
            color=COLOR_SWISS,
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            synth_vals,
            width,
            label="Synthetic",
            color=COLOR_SYNTHETIC,
            alpha=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(["Site Std", "Year Std", "Total Std"])
        ax.set_ylabel("Std")
        ax.set_title(label, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                fontsize=8,
                color=COLOR_SWISS,
            )
        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                fontsize=8,
                color=COLOR_SYNTHETIC,
            )

        if index == 0:
            ax.legend(loc="upper right", fontsize=9)

    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Variance Validation - Std Components by Metric\n"
        "Swiss Target vs Calibrated Synthetic",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_figure(fig, "temperature/3_variance/comparison/variance", output_dir)


def plot_metric_means_comparison(
    swiss_mean_targets: dict[str, float],
    synthetic_means: dict[str, float],
    output_dir: Path,
) -> None:
    """Grouped bar chart comparing Swiss target means vs synthetic means."""
    setup_plot_style()

    metrics = list(swiss_mean_targets.keys())
    labels = [METRIC_LABELS.get(m, m) for m in metrics]
    swiss_vals = [swiss_mean_targets[m] for m in metrics]
    synth_vals = [synthetic_means.get(m, 0.0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(
        x - width / 2,
        swiss_vals,
        width,
        label="Swiss Target",
        color=COLOR_SWISS,
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        synth_vals,
        width,
        label="Synthetic",
        color=COLOR_SYNTHETIC,
        alpha=0.8,
    )

    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Value")
    ax.set_title("Metric Means: Swiss Target vs Synthetic", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    save_figure(fig, "temperature/3_variance/comparison/metric_means", output_dir)


def plot_loss_breakdown(diagnostics_path: Path, output_dir: Path) -> None:
    """Stacked bar chart of per-metric loss contributions."""
    import json

    setup_plot_style()

    with diagnostics_path.open() as f:
        diagnostics = json.load(f)

    # Average component_losses across all seeds for robustness
    per_seed = diagnostics["per_seed"]
    all_metrics_keys: list[str] = list(per_seed[0]["component_losses"].keys())
    avg_losses: dict[str, dict[str, float]] = {}
    for mk in all_metrics_keys:
        avg_losses[mk] = {}
        for comp in ["std_site", "std_year", "std_total", "mean"]:
            vals = [s["component_losses"][mk].get(comp, 0) for s in per_seed]
            avg_losses[mk][comp] = float(np.mean(vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Left: horizontal stacked bar per metric
    labels = [METRIC_LABELS.get(m, m) for m in all_metrics_keys]
    loss_components = ["std_site", "std_year", "std_total", "mean"]
    comp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    y_pos = np.arange(len(all_metrics_keys))

    left = np.zeros(len(all_metrics_keys))
    for comp, color in zip(loss_components, comp_colors, strict=True):
        vals = [avg_losses[m].get(comp, 0) for m in all_metrics_keys]
        ax1.barh(y_pos, vals, left=left, label=comp, color=color, alpha=0.8)
        left += np.array(vals)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Loss")
    ax1.set_title("Per-Metric Loss Breakdown", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="x")

    # Right: aggregate by category
    std_comp_loss = sum(
        avg_losses[m].get(c, 0)
        for m in all_metrics_keys
        for c in ["std_site", "std_year", "std_total"]
    )
    mean_loss = sum(avg_losses[m].get("mean", 0) for m in all_metrics_keys)
    categories = ["Std Components", "Mean Targets"]
    cat_vals = [std_comp_loss, mean_loss]
    bars = ax2.bar(categories, cat_vals, color=comp_colors[:2], alpha=0.8)
    for bar, v in zip(bars, cat_vals, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss by Category", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Loss Breakdown (seed-averaged)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_figure(fig, "temperature/3_variance/optimization/loss_breakdown", output_dir)


def plot_seed_stability(diagnostics_path: Path, output_dir: Path) -> None:
    """Box/strip plots showing per-seed loss variability."""
    import json

    setup_plot_style()

    with diagnostics_path.open() as f:
        diagnostics = json.load(f)

    per_seed = diagnostics["per_seed"]
    seed_losses = [s["loss"] for s in per_seed]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: per-seed total loss distribution
    ax1.boxplot(seed_losses, vert=True, widths=0.5)
    ax1.scatter(
        [1] * len(seed_losses), seed_losses, alpha=0.6, color=COLOR_SWISS, zorder=3
    )
    mean_val = np.mean(seed_losses)
    median_val = np.median(seed_losses)
    std_val = np.std(seed_losses)
    ax1.axhline(
        y=mean_val, color="r", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.4f}"
    )
    ax1.set_title(
        f"Per-Seed Total Loss\nmedian={median_val:.4f}, std={std_val:.4f}",
        fontweight="bold",
    )
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: per-seed std_total by metric
    metric_keys = list(per_seed[0]["components"].keys())
    metric_labels = [METRIC_LABELS.get(m, m) for m in metric_keys]
    data_by_metric = []
    for mk in metric_keys:
        vals = [s["components"][mk]["std_total"] for s in per_seed]
        data_by_metric.append(vals)

    positions = np.arange(1, len(metric_keys) + 1)
    ax2.boxplot(data_by_metric, positions=positions, widths=0.5)
    for i, vals in enumerate(data_by_metric):
        ax2.scatter(
            [positions[i]] * len(vals),
            vals,
            alpha=0.4,
            color=COLOR_SWISS,
            s=20,
            zorder=3,
        )
    ax2.set_xticks(positions)
    ax2.set_xticklabels(metric_labels, rotation=30, ha="right")
    ax2.set_ylabel("std_total")
    ax2.set_title("Per-Seed std_total by Metric", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Seed Stability", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_figure(fig, "temperature/3_variance/optimization/seed_stability", output_dir)


def plot_site_year_distributions(
    swiss_data: Dataset,
    synthetic_temps: np.ndarray,
    num_sites: int,
    num_years: int,
    output_dir: Path,
) -> None:
    """Overlapping histograms of site-mean and year-mean distributions."""
    from npnf.calibration.temperature.data import ObservedData
    from npnf.calibration.temperature.variance.extraction import site_means, year_means
    from npnf.calibration.temperature.variance.objectives import compute_all_metrics

    setup_plot_style()

    ca = ObservedData.from_dataset(swiss_data)
    swiss_metrics = compute_all_metrics(ca.temps)
    synth_metrics = compute_all_metrics(synthetic_temps)

    plot_metrics = list(swiss_metrics.keys())
    num_metrics = len(plot_metrics)

    fig, axes = plt.subplots(num_metrics, 2, figsize=(12, 3 * num_metrics))

    for row, metric_name in enumerate(plot_metrics):
        label = METRIC_LABELS.get(metric_name, metric_name)
        swiss_vals = swiss_metrics[metric_name]
        synth_vals = synth_metrics[metric_name]

        # Swiss site means and year means
        swiss_site_means = site_means(swiss_vals, ca.site_indices)
        swiss_year_means = year_means(swiss_vals, ca.year_indices)

        # Synthetic: reshape to (num_sites, num_years)
        synth_grid = synth_vals.reshape(num_sites, num_years)
        synth_site_means = synth_grid.mean(axis=1)
        synth_year_means = synth_grid.mean(axis=0)

        # Left: site distribution
        ax_site = axes[row, 0]
        ax_site.hist(
            swiss_site_means,
            bins=15,
            alpha=0.5,
            density=True,
            color=COLOR_SWISS,
            label="Swiss",
        )
        ax_site.hist(
            synth_site_means,
            bins=15,
            alpha=0.5,
            density=True,
            color=COLOR_SYNTHETIC,
            label="Synthetic",
        )
        ax_site.set_title(f"{label} — Site Means")
        ax_site.set_ylabel("Density")
        if row == 0:
            ax_site.legend(fontsize=8)

        # Right: year distribution
        ax_year = axes[row, 1]
        ax_year.hist(
            swiss_year_means,
            bins=15,
            alpha=0.5,
            density=True,
            color=COLOR_SWISS,
            label="Swiss",
        )
        ax_year.hist(
            synth_year_means,
            bins=15,
            alpha=0.5,
            density=True,
            color=COLOR_SYNTHETIC,
            label="Synthetic",
        )
        ax_year.set_title(f"{label} — Year Means")
        if row == 0:
            ax_year.legend(fontsize=8)

    fig.suptitle("Site & Year Mean Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    save_figure(
        fig, "temperature/3_variance/comparison/site_year_distributions", output_dir
    )


def plot_hourly_heatmap(
    swiss_data: Dataset, synthetic_temps: np.ndarray, output_dir: Path
) -> None:
    """Heatmaps of mean temperature by (day, hour): Swiss, Synthetic, Residual."""
    setup_plot_style()

    swiss_temps = _swiss_temps_from_dataset(swiss_data)
    swiss_mean = swiss_temps.mean(axis=0)  # (274, 24)
    synth_mean = synthetic_temps.mean(axis=0)  # (274, 24)
    residual = swiss_mean - synth_mean

    vmin = min(swiss_mean.min(), synth_mean.min())
    vmax = max(swiss_mean.max(), synth_mean.max())
    res_vmax = max(abs(residual.min()), abs(residual.max()))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

    ax1.pcolormesh(
        np.arange(25), np.arange(275), swiss_mean, vmin=vmin, vmax=vmax, cmap="RdYlBu_r"
    )
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Day of Season (Nov 1 = 0)")
    ax1.set_title("Swiss Mean", fontweight="bold")

    im2 = ax2.pcolormesh(
        np.arange(25), np.arange(275), synth_mean, vmin=vmin, vmax=vmax, cmap="RdYlBu_r"
    )
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Day of Season (Nov 1 = 0)")
    ax2.set_title("Synthetic Mean", fontweight="bold")

    fig.colorbar(im2, ax=[ax1, ax2], label="Temperature (°C)", shrink=0.8)

    im3 = ax3.pcolormesh(
        np.arange(25),
        np.arange(275),
        residual,
        vmin=-res_vmax,
        vmax=res_vmax,
        cmap="RdBu_r",
    )
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Day of Season (Nov 1 = 0)")
    ax3.set_title("Swiss − Synthetic (°C)", fontweight="bold")
    fig.colorbar(im3, ax=ax3, label="Residual (°C)", shrink=0.8)

    fig.suptitle(
        "Mean Temperature Heatmap (day × hour)", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_figure(fig, "temperature/3_variance/spatial/heatmap", output_dir)


def plot_qq_metrics(
    swiss_metrics: dict[str, np.ndarray],
    synthetic_metrics: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """QQ plots for each metric: Swiss vs synthetic quantiles.

    Each panel shows quantile-quantile comparison with:
    - Points colored by quantile (blue=low, red=high)
    - R² and MAE agreement statistics
    - Sample count and median values
    """
    setup_plot_style()

    metrics = list(swiss_metrics.keys())
    num_metrics = len(metrics)
    ncols = 3
    nrows = math.ceil(num_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes_flat = axes.flatten()
    quantiles = np.linspace(0, 1, 100)

    for i, metric_name in enumerate(metrics):
        ax = axes_flat[i]
        swiss_vals = swiss_metrics[metric_name]
        synth_vals = synthetic_metrics[metric_name]
        swiss_q = np.quantile(swiss_vals, quantiles)
        synth_q = np.quantile(synth_vals, quantiles)

        # Color by quantile: blue (low) -> red (high)
        colors = plt.colormaps["RdYlBu_r"](quantiles)
        ax.scatter(swiss_q, synth_q, s=18, c=colors, alpha=0.8, edgecolors="none")

        # Diagonal reference
        lo = min(swiss_q.min(), synth_q.min())
        hi = max(swiss_q.max(), synth_q.max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "k--",
            alpha=0.4,
            linewidth=1,
        )

        # Agreement statistics
        ss_res = np.sum((synth_q - swiss_q) ** 2)
        ss_tot = np.sum((swiss_q - swiss_q.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.mean(np.abs(synth_q - swiss_q))

        unit = METRIC_UNITS.get(metric_name, "")
        mae_str = f"{mae:.2f} {unit}".strip() if unit else f"{mae:.3f}"
        stats_text = f"R²={r2:.3f}\nMAE={mae_str}"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

        # Sample count and medians
        n_swiss = len(swiss_vals)
        n_synth = len(synth_vals)
        med_swiss = np.median(swiss_vals)
        med_synth = np.median(synth_vals)
        info_text = f"N: {n_swiss}/{n_synth}\nmed: {med_swiss:.1f}/{med_synth:.1f}"
        ax.text(
            0.95,
            0.05,
            info_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontfamily="monospace",
            color="0.4",
        )

        label = METRIC_LABELS.get(metric_name, metric_name)
        unit_suffix = f" ({unit})" if unit else ""
        ax.set_xlabel(f"Swiss{unit_suffix}")
        ax.set_ylabel(f"Synthetic{unit_suffix}")
        ax.set_title(label, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    for i in range(len(metrics), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        "QQ Plots: Swiss vs Synthetic (100 quantiles)", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(fig, "temperature/3_variance/comparison/qq", output_dir)


def plot_sample_traces(
    swiss_data: Dataset,
    synthetic_temps: np.ndarray,
    output_dir: Path,
    num_samples: int = 10,
    seed: int = 42,
) -> None:
    """Spaghetti plots of raw temperature traces: seasonal + diurnal.

    Top row: daily-mean seasonal curves (~num_samples each).
    Bottom row: 24h diurnal profiles for 3 representative days.
    """
    setup_plot_style()

    rng = np.random.default_rng(seed)
    swiss_temps = _swiss_temps_from_dataset(swiss_data)

    swiss_idx = rng.choice(len(swiss_temps), size=num_samples, replace=False)
    synth_idx = rng.choice(len(synthetic_temps), size=num_samples, replace=False)

    days = np.arange(274)
    hours = np.arange(24)
    # Representative days: Dec (day 30), Mar (day 120), Jun (day 210)
    sample_days = [30, 120, 210]
    day_labels = ["Day 30 (Dec)", "Day 120 (Mar)", "Day 210 (Jun)"]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # --- Top row: seasonal daily-mean spaghetti (spans all 3 columns) ---
    ax_top = fig.add_subplot(gs[0, :])
    for i in swiss_idx:
        daily_mean = swiss_temps[i].mean(axis=1)  # (274,)
        ax_top.plot(
            days,
            daily_mean,
            color=COLOR_SWISS,
            alpha=0.3,
            linewidth=0.8,
            label="Swiss" if i == swiss_idx[0] else None,
        )
    for i in synth_idx:
        daily_mean = synthetic_temps[i].mean(axis=1)  # (274,)
        ax_top.plot(
            days,
            daily_mean,
            color=COLOR_SYNTHETIC,
            alpha=0.3,
            linewidth=0.8,
            label="Synthetic" if i == synth_idx[0] else None,
        )
    ax_top.set_xlabel("Day of Season (Nov 1 = 0)")
    ax_top.set_ylabel("Daily Mean Temperature (C)")
    ax_top.set_title("Seasonal Daily-Mean Traces", fontweight="bold")
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=0.3)

    # --- Bottom row: diurnal profiles for representative days ---
    for col, (day_idx, day_label) in enumerate(
        zip(sample_days, day_labels, strict=True)
    ):
        ax = fig.add_subplot(gs[1, col])
        for i in swiss_idx:
            ax.plot(
                hours,
                swiss_temps[i, day_idx, :],
                color=COLOR_SWISS,
                alpha=0.3,
                linewidth=0.8,
                label="Swiss" if (i == swiss_idx[0] and col == 0) else None,
            )
        for i in synth_idx:
            ax.plot(
                hours,
                synthetic_temps[i, day_idx, :],
                color=COLOR_SYNTHETIC,
                alpha=0.3,
                linewidth=0.8,
                label="Synthetic" if (i == synth_idx[0] and col == 0) else None,
            )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Temperature (C)")
        ax.set_title(f"Diurnal Profile — {day_label}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Sample Temperature Traces: Swiss vs Synthetic", fontsize=14, fontweight="bold"
    )

    save_figure(fig, "temperature/3_variance/spatial/sample_traces", output_dir)


def plot_metric_tradeoffs(diagnostics_path: Path, output_dir: Path) -> None:
    """Correlation matrix of per-seed metric losses.

    Shows whether improving one metric tends to degrade another across
    seeds. Strong positive correlations suggest coupled metrics; negative
    correlations reveal trade-offs the optimizer must navigate.
    """
    import json

    setup_plot_style()

    with diagnostics_path.open() as f:
        diagnostics = json.load(f)

    per_seed = diagnostics["per_seed"]
    metric_keys = list(per_seed[0]["component_losses"].keys())

    # Sum component losses per metric per seed
    seed_metric_losses = np.zeros((len(per_seed), len(metric_keys)))
    for i, seed_data in enumerate(per_seed):
        for j, mk in enumerate(metric_keys):
            comps = seed_data["component_losses"][mk]
            seed_metric_losses[i, j] = sum(comps.values())

    labels = [METRIC_LABELS.get(m, m) for m in metric_keys]

    # Correlation matrix
    corr = np.corrcoef(seed_metric_losses.T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: correlation heatmap
    im = ax1.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax1.text(
                j,
                i,
                f"{corr[i, j]:.2f}",  # ty: ignore[not-subscriptable]
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(corr[i, j]) > 0.5 else "black",  # ty: ignore[not-subscriptable]
            )
    fig.colorbar(im, ax=ax1, label="Pearson r", shrink=0.8)
    ax1.set_title("Loss Correlation Across Seeds", fontweight="bold")

    # Right: per-seed metric loss profiles (line per seed)
    for i, seed_data in enumerate(per_seed):
        vals = seed_metric_losses[i]
        ax2.plot(
            range(len(labels)),
            vals,
            "o-",
            alpha=0.5,
            markersize=4,
            label=f"seed {seed_data['seed']}" if i < 3 else None,
        )
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Total Metric Loss")
    ax2.set_title("Per-Seed Metric Loss Profiles", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=2)

    fig.suptitle("Cross-Metric Trade-off Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(fig, "temperature/3_variance/optimization/metric_tradeoffs", output_dir)


def plot_optimization_convergence(diagnostics_path: Path, output_dir: Path) -> None:
    """Optimization convergence curve with per-metric loss attribution.

    Left: running-best total loss over completed trials.
    Right: per-seed x per-metric loss heatmap at the best trial, showing
    which metrics dominate the final loss and how stable they are.
    """
    import json

    setup_plot_style()

    with diagnostics_path.open() as f:
        diagnostics = json.load(f)

    trial_summaries = diagnostics["trial_summaries"]
    per_seed = diagnostics["per_seed"]
    metric_keys = list(per_seed[0]["component_losses"].keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: convergence curve
    trials = sorted(trial_summaries, key=lambda t: t["trial_number"])
    trial_nums = [t["trial_number"] for t in trials]
    losses = [t["aggregate_loss"] for t in trials]

    # Running best
    running_best = np.minimum.accumulate(losses)

    ax1.scatter(trial_nums, losses, s=8, alpha=0.3, color="gray", label="Trials")
    ax1.plot(trial_nums, running_best, "r-", linewidth=2, label="Running best")
    ax1.axhline(
        y=running_best[-1],
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Best: {running_best[-1]:.4f}",
    )
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("Aggregate Loss")
    ax1.set_title("Optimization Convergence", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: per-seed × per-metric loss heatmap at best trial
    labels = [METRIC_LABELS.get(m, m) for m in metric_keys]
    seed_labels = [f"seed {s['seed']}" for s in per_seed]

    loss_matrix = np.zeros((len(per_seed), len(metric_keys)))
    for i, seed_data in enumerate(per_seed):
        for j, mk in enumerate(metric_keys):
            comps = seed_data["component_losses"][mk]
            loss_matrix[i, j] = sum(comps.values())

    im = ax2.imshow(loss_matrix, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(seed_labels)))
    ax2.set_yticklabels(seed_labels, fontsize=8)
    for i in range(len(seed_labels)):
        for j in range(len(labels)):
            ax2.text(
                j,
                i,
                f"{loss_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white"
                if loss_matrix[i, j] > loss_matrix.max() * 0.6
                else "black",
            )
    fig.colorbar(im, ax=ax2, label="Loss", shrink=0.8)
    ax2.set_title("Best Trial: Seed × Metric Loss", fontweight="bold")

    fig.suptitle(
        "Optimization Convergence & Loss Attribution", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(
        fig, "temperature/3_variance/optimization/optimization_convergence", output_dir
    )


def plot_extreme_temperature_cdf(
    swiss_metrics: dict[str, np.ndarray],
    synthetic_metrics: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """CDF comparison for extreme temperature metrics.

    Shows empirical CDFs for max_temp_reached and hot_days side by side,
    validating whether synthetic data reproduces the tails of the
    temperature distribution.
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: max_temp_reached CDF
    for ax, metric_name, title in [
        (ax1, "max_temp_reached", "Max Temperature Reached (°C)"),
        (ax2, "hot_days", "Hot Days (max > 30°C)"),
    ]:
        swiss_vals = np.sort(swiss_metrics[metric_name])
        synth_vals = np.sort(synthetic_metrics[metric_name])
        swiss_cdf = np.arange(1, len(swiss_vals) + 1) / len(swiss_vals)
        synth_cdf = np.arange(1, len(synth_vals) + 1) / len(synth_vals)

        ax.step(
            swiss_vals,
            swiss_cdf,
            where="post",
            color=COLOR_SWISS,
            linewidth=2,
            label="Swiss",
        )
        ax.step(
            synth_vals,
            synth_cdf,
            where="post",
            color=COLOR_SYNTHETIC,
            linewidth=2,
            label="Synthetic",
        )

        # KS statistic
        from scipy.stats import ks_2samp

        ks_stat, ks_p = ks_2samp(swiss_vals, synth_vals)

        # Summary stats
        ax.text(
            0.05,
            0.95,
            f"Swiss: μ={swiss_vals.mean():.1f}, σ={swiss_vals.std():.1f}\n"
            f"Synth: μ={synth_vals.mean():.1f}, σ={synth_vals.std():.1f}\n"
            f"KS={ks_stat:.3f} (p={ks_p:.3f})",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel(title)
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Extreme Temperature Validation (CDF)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_figure(
        fig, "temperature/3_variance/comparison/extreme_temperature", output_dir
    )


def plot_seed_metric_means(diagnostics_path: Path, output_dir: Path) -> None:
    """Box plots of per-seed metric mean values with Swiss targets.

    Shows the distribution of each metric's mean across seeds, overlaid
    with Swiss target values. Wide boxes indicate seed-sensitive metrics.
    """
    import json

    setup_plot_style()

    with diagnostics_path.open() as f:
        diagnostics = json.load(f)

    per_seed = diagnostics["per_seed"]
    mean_targets = diagnostics.get("mean_targets", {})

    # Check if metric_means is available (added in ETH-679)
    if "metric_means" not in per_seed[0]:
        return

    metric_keys = list(per_seed[0]["metric_means"].keys())
    labels = [METRIC_LABELS.get(m, m) for m in metric_keys]

    fig, axes = plt.subplots(
        1, len(metric_keys), figsize=(3 * len(metric_keys), 5), sharey=False
    )

    for i, (mk, label) in enumerate(zip(metric_keys, labels, strict=True)):
        ax = axes[i]
        vals = [s["metric_means"][mk] for s in per_seed]

        ax.boxplot(vals, widths=0.5)
        ax.scatter(
            [1] * len(vals), vals, alpha=0.5, color=COLOR_SYNTHETIC, s=25, zorder=3
        )

        # Swiss target
        if mk in mean_targets:
            target = mean_targets[mk]
            ax.axhline(
                y=target,
                color=COLOR_SWISS,
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Swiss: {target:.2f}",
            )

        unit = METRIC_UNITS.get(mk, "")
        unit_str = f" ({unit})" if unit else ""
        ax.set_title(label, fontweight="bold", fontsize=9)
        ax.set_ylabel(f"Value{unit_str}" if i == 0 else "")
        ax.set_xticklabels([])
        ax.grid(True, alpha=0.3, axis="y")
        if mk in mean_targets:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Per-Seed Metric Means vs Swiss Targets", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_figure(fig, "temperature/3_variance/comparison/seed_metric_means", output_dir)


def plot_year_means_detrending(
    calibration_arrays: ObservedData, output_dir: Path
) -> None:
    """Diagnostic plot for the year-means detrending step."""
    setup_plot_style()

    result = detrend_year_means(calibration_arrays)
    unique_years = calibration_arrays.unique_years
    trend_per_decade = result.trend_coeffs[0] * 10

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    ax_left.scatter(unique_years, result.year_means, color=COLOR_SWISS, s=40, zorder=3)
    ax_left.plot(
        unique_years,
        result.trend_line,
        color=COLOR_SYNTHETIC,
        linewidth=2,
        linestyle="--",
        label="Linear trend",
    )
    ax_left.set_xlabel("Year")
    ax_left.set_ylabel("Mean daily temperature (\u00b0C)")
    ax_left.set_title("Per-Year Mean Temperatures")
    ax_left.text(
        0.05,
        0.95,
        f"Trend: {trend_per_decade:+.2f} °C/decade\nRaw std: {result.raw_std:.3f} °C",
        transform=ax_left.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    ax_left.legend(fontsize=9)

    ax_right.scatter(unique_years, result.detrended, color=COLOR_SWISS, s=40, zorder=3)
    ax_right.axhline(0, color="gray", linewidth=0.8)
    ax_right.axhspan(
        -result.detrended_std,
        result.detrended_std,
        color=COLOR_SWISS,
        alpha=0.15,
        label=f"\u00b11\u03c3 = \u00b1{result.detrended_std:.3f} \u00b0C",
    )
    ax_right.set_xlabel("Year")
    ax_right.set_ylabel("Detrended anomaly (\u00b0C)")
    ax_right.set_title("Detrended Year Anomalies")
    ax_right.text(
        0.05,
        0.95,
        f"year_anomaly_std: {result.detrended_std:.3f} \u00b0C",
        transform=ax_right.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    ax_right.legend(fontsize=9)

    fig.suptitle("Year-Mean Detrending (ETH-610)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_figure(fig, "temperature/3_variance/spatial/year_detrending", output_dir)


def visualize_variance_diagnostics(
    data: CalibrationData,
    output_dir: Path,
    study: Study | None = None,
    swiss_data: Dataset | None = None,
    final_params: TemperatureParams | None = None,
) -> None:
    """Generate full variance diagnostic plots after variance optimization."""
    import logging

    from npnf.calibration.temperature.data import load_swiss_data
    from npnf.calibration.temperature.variance.objectives import (
        compute_all_metrics,
        compute_variance_components,
        generate_synthetic_for_calibration,
    )

    logger = logging.getLogger(__name__)

    tp = final_params if final_params is not None else data.to_temperature_params()
    assert data.optimization is not None
    swiss_targets = data.optimization.swiss_targets
    swiss_mean_targets = data.optimization.swiss_mean_targets

    temps = generate_synthetic_for_calibration(tp, num_sites=20, num_years=20)
    metrics = compute_all_metrics(temps)

    synth_components_by_metric = {}
    for metric_name in swiss_targets:
        synth_components_by_metric[metric_name] = compute_variance_components(
            metrics[metric_name], 20, 20
        )

    plot_variance_validation(swiss_targets, synth_components_by_metric, output_dir)
    logger.info("Created: plots/temperature/variance.png")

    if swiss_data is None:
        swiss_data = load_swiss_data()
    plot_temperature_validation(swiss_data, temps, output_dir)
    logger.info("Created: plots/temperature/variance_comparison.png")

    if study is not None:
        plot_variance_optimization(study, output_dir)
        logger.info("Created: plots/temperature/variance_optimization.png")

    swiss_temps = np.asarray(swiss_data["temperature_values"], dtype=float).reshape(
        -1, 274, 24
    )
    swiss_metrics = compute_all_metrics(swiss_temps)

    synth_means = {name: float(np.mean(vals)) for name, vals in metrics.items()}

    diagnostics_path = (
        output_dir / "temperature/3_variance/optimization/variance_diagnostics.json"
    )

    plot_metric_means_comparison(swiss_mean_targets, synth_means, output_dir)
    logger.info("Created: plots/temperature/metric_means.png")

    if diagnostics_path.exists():
        plot_loss_breakdown(diagnostics_path, output_dir)
        logger.info("Created: plots/temperature/loss_breakdown.png")

        plot_seed_stability(diagnostics_path, output_dir)
        logger.info("Created: plots/temperature/seed_stability.png")

    plot_site_year_distributions(swiss_data, temps, 20, 20, output_dir)
    logger.info("Created: plots/temperature/site_year_distributions.png")

    plot_hourly_heatmap(swiss_data, temps, output_dir)
    logger.info("Created: plots/temperature/heatmap.png")

    plot_qq_metrics(swiss_metrics, metrics, output_dir)
    logger.info("Created: plots/temperature/qq.png")

    plot_sample_traces(swiss_data, temps, output_dir)
    logger.info("Created: plots/temperature/sample_traces.png")

    plot_extreme_temperature_cdf(swiss_metrics, metrics, output_dir)
    logger.info("Created: plots/temperature/extreme_temperature.png")

    if diagnostics_path.exists():
        plot_metric_tradeoffs(diagnostics_path, output_dir)
        logger.info("Created: plots/temperature/metric_tradeoffs.png")

        plot_optimization_convergence(diagnostics_path, output_dir)
        logger.info("Created: plots/temperature/optimization_convergence.png")

        plot_seed_metric_means(diagnostics_path, output_dir)
        logger.info("Created: plots/temperature/seed_metric_means.png")
