"""Single-panel figure of the simulator/real dispersion ratio per trait.

Uses the paper style, not the calibration plotting helpers, because the figure
goes into the paper. The errors on the three traits have opposite signs, so the
figure shows each ratio separately, with its bootstrap interval.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from npnf.scripts.paper.style import COLORS, get_figure, save_figure, setup_style


def plot_genotype_variability(
    summary: list[dict], intervals: list[dict], output_path: Path
) -> None:
    """Simulator/FIP1 genotype SD ratio, one line per trait."""
    # Trait, label, colour, marker and horizontal offset for each series. The
    # years are separate trials, so the points are not joined by lines.
    ratio_series = (
        ("max_height", "Maximum height", "blue", "o", -0.15),
        ("growth_end_day", "Growth-cessation day", "green", "^", 0.0),
        ("elongation_rate", "Elongation rate", "orange", "s", 0.15),
    )
    setup_style()
    figure, axis = get_figure("single")

    positions = range(len(summary))
    axis.axhline(1.0, color="black", linewidth=1.0, zorder=1)
    bounds = {(row["year"], row["trait"]): row for row in intervals}
    for trait, label, color, marker, offset in ratio_series:
        ratios = [row[f"{trait}_ratio"] for row in summary]
        interval_rows = [bounds[(row["year"], trait)] for row in summary]
        axis.errorbar(
            [position + offset for position in positions],
            ratios,
            yerr=[
                [
                    ratio - row["ratio_lower"]
                    for ratio, row in zip(ratios, interval_rows, strict=True)
                ],
                [
                    row["ratio_upper"] - ratio
                    for ratio, row in zip(ratios, interval_rows, strict=True)
                ],
            ],
            fmt=marker,
            capsize=2,
            label=label,
            color=COLORS[color],
            zorder=2,
        )

    axis.set_xticks(list(positions))
    axis.set_xticklabels(
        [
            f"{row['year']}\n(held out)" if row["held_out"] else str(row["year"])
            for row in summary
        ]
    )
    axis.set_ylabel("Simulator / real across-genotype SD")
    axis.set_xlabel("Year")
    # Above the axes: every in-axes corner is occupied by one of the series.
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=2,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.6,
    )

    save_figure(figure, output_path)
    plt.close(figure)
