"""Distribution report for the shifted test population."""

from __future__ import annotations

import html
import itertools
import json
from enum import StrEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from npnf.data.synthetic.lodging import lodging_probability
from npnf.scripts.paper.style import COLORS, save_figure, setup_style
from npnf.scripts.utils.outputs import html_table, write_csv
from npnf.scripts.utils.visualization import plot_ecdf


class Population(StrEnum):
    """The populations of the report, in plotting order."""

    TRAIN = "train"
    ORIGINAL = "original"
    SEEN = "seen"
    GENO = "geno"
    ENV = "env"
    UNSEEN = "unseen"

    @property
    def label(self) -> str:
        return {
            "train": "Training",
            "original": "Old unseen test",
            "seen": "Seen",
            "geno": "New genotypes",
            "env": "New environments",
            "unseen": "Unseen",
        }[self]

    @property
    def color(self) -> str:
        return COLORS[
            {
                "train": "gray",
                "original": "blue",
                "seen": "orange",
                "geno": "red",
                "env": "green",
                "unseen": "purple",
            }[self]
        ]


class Figure(StrEnum):
    PARAMETERS = "parameters"
    ENVIRONMENTS = "environments"
    GROWTH_DISTRIBUTIONS = "growth_distributions"
    TRAJECTORIES = "trajectories"
    LODGING = "lodging"
    HEIGHT_LODGING_LAW = "height_lodging_law"


def load_distributions(root: Path) -> dict[Population, dict[str, np.ndarray]]:
    """The exported arrays of every population."""
    data = {}
    for population in Population:
        with np.load(
            root / "distributions" / f"{population}.npz", allow_pickle=False
        ) as archive:
            data[population] = {key: archive[key] for key in archive.files}
    return data


def _save(fig, directory: Path, name: str) -> None:
    save_figure(fig, directory / name)
    plt.close(fig)


def _summaries(data: dict[Population, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    for population, arrays in data.items():
        lodged = arrays["lodging"].astype(bool)
        measurements = {
            "maximum_clean_height_m": arrays["clean_max_height"],
            "day_reaching_95_percent_height": arrays["timing"],
            "day_reaching_maximum_clean_height": arrays["peak_day"],
            "tau_max": arrays["genotype_params"][:, -1],
            "mean_free_control_point": arrays["genotype_params"][:, :-1].mean(axis=1),
            "environment_mean_temperature_C": arrays["temperatures"].mean(axis=(1, 2)),
            "lodging_incidence": lodged.astype(float),
            "lodging_day_given_lodged": arrays["lodging_day"][lodged],
            "severity_final_height_fraction_given_lodged": arrays["severity"][lodged],
            "sampled_absolute_drop_m_given_lodged": (
                arrays["clean_max_height"][:, None] * (1 - arrays["severity"])
            )[lodged],
        }
        for measure, values in measurements.items():
            values = np.asarray(values).reshape(-1)
            values = values[np.isfinite(values)]
            quantiles = (
                np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
                if values.size
                else [None] * 5
            )
            rows.append(
                {
                    "population": population,
                    "measure": measure,
                    "count": values.size,
                    "mean": float(values.mean()) if values.size else None,
                    "std": float(values.std()) if values.size else None,
                    **dict(
                        zip(
                            ("p05", "p25", "median", "p75", "p95"),
                            quantiles,
                            strict=True,
                        )
                    ),
                }
            )
    return rows


def _plot_parameters(data, directory: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for population, arrays in data.items():
        params = arrays["genotype_params"]
        label, color = population.label, population.color
        plot_ecdf(axes[0], params[:, -1], label, color)
        plot_ecdf(axes[1], params[:, :-1].mean(axis=1), label, color)
        quantiles = np.quantile(params[:, :-1], [0.05, 0.5, 0.95], axis=0)
        indices = np.arange(30)
        axes[2].plot(indices, quantiles[1], label=label, color=color)
        axes[2].fill_between(
            indices, quantiles[0], quantiles[2], color=color, alpha=0.12
        )
    axes[0].set(xlabel="Genotype thermal-time requirement", title="Maturity parameter")
    axes[1].set(xlabel="Mean of 30 free control points", title="Growth-response level")
    axes[2].set(
        xlabel="Free control-point index",
        ylabel="Control-point value",
        title="Median and 5–95% range",
    )
    axes[0].legend()
    _save(fig, directory, "parameters")


def _plot_environments(data, directory: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for population, arrays in data.items():
        temperatures = arrays["temperatures"]
        label, color = population.label, population.color
        plot_ecdf(axes[0], temperatures.mean(axis=(1, 2)), label, color)
        daily = temperatures.mean(axis=2)
        quantiles = np.quantile(daily, [0.05, 0.5, 0.95], axis=0)
        days = np.arange(daily.shape[1]) + 61
        axes[1].plot(days, quantiles[1], label=label, color=color)
        axes[1].fill_between(days, quantiles[0], quantiles[2], color=color, alpha=0.12)
    axes[0].set(
        xlabel="Environment mean temperature (°C)", title="Environment distribution"
    )
    axes[1].set(
        xlabel="Day of year",
        ylabel="Daily mean temperature (°C)",
        title="Median and 5–95% environments",
    )
    axes[0].legend()
    _save(fig, directory, "environments")


def _plot_growth(data, directory: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for population, arrays in data.items():
        label, color = population.label, population.color
        plot_ecdf(axes[0], arrays["clean_max_height"], label, color)
        plot_ecdf(axes[1], arrays["timing"], label, color)
    axes[0].set(
        xlabel="Maximum clean height (m)", title="All genotype × environment conditions"
    )
    axes[1].set(
        xlabel="Day reaching 95% of maximum clean height", title="Growth timing"
    )
    axes[0].legend()
    _save(fig, directory, "growth_distributions")
    fig, axes = plt.subplots(
        2, len(Population), figsize=(4 * len(Population), 8), sharex=True, sharey=True
    )
    for column, population in enumerate(Population):
        arrays = data[population]
        days = arrays["days"]
        active = (days >= 200) & (days <= 321)
        for row, key in enumerate(("clean", "heights")):
            paths = arrays[key][:, active]
            ax = axes[row, column]
            for path in paths[:12]:
                ax.plot(
                    days[active],
                    path,
                    color=population.color,
                    alpha=0.12,
                    linewidth=0.7,
                )
            quantiles = np.quantile(paths, [0.05, 0.5, 0.95], axis=0)
            ax.fill_between(
                days[active],
                quantiles[0],
                quantiles[2],
                color=population.color,
                alpha=0.2,
            )
            ax.plot(days[active], quantiles[1], color=population.color, linewidth=2)
            ax.set_title(
                f"{population.label}: {'clean' if row == 0 else 'with lodging'}"
            )
            ax.set_xlabel("Day of year")
            if column == 0:
                ax.set_ylabel("Height (m)")
    fig.suptitle(
        "Fixed trajectory subsamples: median, 5–95% band and 12 sample paths; "
        "no observation noise"
    )
    _save(fig, directory, "trajectories")


def _plot_lodging(data, directory: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for index, (population, arrays) in enumerate(data.items()):
        mask = arrays["lodging"].astype(bool)
        label, color = population.label, population.color
        axes[0].bar(index, 100 * mask.mean(), color=color)
        plot_ecdf(axes[1], arrays["lodging_day"][mask], label, color)
        plot_ecdf(axes[2], arrays["severity"][mask], label, color)
        drop = arrays["clean_max_height"][:, None] * (1 - arrays["severity"])
        plot_ecdf(axes[3], drop[mask], label, color)
    axes[0].set(
        xticks=range(len(Population)),
        xticklabels=[population.label for population in Population],
        ylabel="Lodged draws (%)",
        title="Unchanged height-dependent lodging law",
    )
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[1].set(xlabel="Lodging day", title="Timing, conditional on lodging")
    axes[2].set(
        xlabel="Sampled final-height / maximum-height fraction",
        title="Severity, conditional on lodging",
    )
    axes[3].set(
        xlabel="Maximum height − sampled final height (m)",
        title="Absolute drop, conditional on lodging",
    )
    axes[1].legend()
    _save(fig, directory, "lodging")


def _plot_height_lodging_law(data, root: Path, directory: Path) -> None:
    parameters = json.loads((root / "manifest.json").read_text())["datasets"]["unseen"][
        "parameters"
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.4, 1.4, 21)
    for population, arrays in data.items():
        peaks = arrays["clean_max_height"]
        points = [
            (peaks[mask].mean(), arrays["lodging"][mask].mean())
            for lower, upper in itertools.pairwise(bins)
            if (mask := (peaks >= lower) & (peaks < upper)).sum() >= 20
        ]
        ax.plot(
            [height for height, _ in points],
            [100 * fraction for _, fraction in points],
            "o",
            color=population.color,
            label=population.label,
            markersize=4,
        )
    heights = np.linspace(0.4, 1.4, 200)
    probability = lodging_probability(
        torch.as_tensor(heights),
        **{
            key: parameters[key]
            for key in (
                "lodging_height_clamp",
                "lodging_weibull_shape",
                "lodging_weibull_scale",
                "lodging_weibull_offset",
            )
        },
    )
    ax.plot(heights, 100 * probability.numpy(), "k-", label="Unchanged simulator law")
    ax.set(
        xlabel="Maximum clean height (m)",
        ylabel="Lodging probability (%)",
        title="Population changes; height-dependent lodging relationship does not",
    )
    ax.legend()
    _save(fig, directory, "height_lodging_law")


def report(output_dir: str | Path) -> None:
    """Write inspectable PNG/PDF figures, CSV tables and a linked HTML report."""
    root = Path(output_dir)
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    setup_style()
    data = load_distributions(root)
    summary = _summaries(data)
    write_csv(root / "distribution_summary.csv", summary, list(summary[0]))
    _plot_parameters(data, directory)
    _plot_environments(data, directory)
    _plot_growth(data, directory)
    _plot_lodging(data, directory)
    _plot_height_lodging_law(data, root, directory)
    selection = json.loads((root / "manifest.json").read_text())["selection"]
    sections = [
        f"<h2>{name.replace('_', ' ').title()}</h2>"
        f'<a href="figures/{name}.pdf">PDF</a>'
        f'<img src="figures/{name}.png" alt="{name}">'
        for name in Figure
    ]
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Shifted simulator test population</title>
<style>
body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px}
img{width:100%;height:auto}
table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}
th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}
th:first-child,td:first-child{text-align:left}
pre{white-space:pre-wrap}h2{margin-top:40px}
</style>
<h1>Shifted simulator test population</h1>
<p>Unchanged generators, growth, lodging and observation mechanisms.
The test population is a selection: shorter genotypes and cooler year-sites,
by thresholds on simulator outputs. Seen = selected training genotypes ×
selected training year-sites; new genotypes and year-sites come from new pools of
the unchanged generators and pass the same thresholds.
Genotype markers retain training normalization.</p>
<p>Scalar growth distributions cover every genotype–environment condition.
Trajectory bands use a fixed subsample. Lodging distributions use the recorded
draws (one per training condition; 64 per test condition).
Factorial conditions are not independent environments;
descriptive ranges are not confidence intervals.</p>
<h2>Frozen selection</h2><pre>"""
        + html.escape(json.dumps(selection, indent=2))
        + "</pre>"
        + "".join(sections)
    )
    document += (
        "<h2>Population summaries</h2><p><a href='distribution_summary.csv'>CSV</a></p>"
        + html_table(summary)
    )
    document += (
        "<h2>Provenance</h2><a href='manifest.json'>Dataset manifest</a> | "
        "<a href='selection.json'>Frozen selection</a></html>"
    )
    (root / "report.html").write_text(document)
