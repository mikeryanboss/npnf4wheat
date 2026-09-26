"""Optional point styling helpers for raw Sig-MMD paper plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.typing import ColorType, FillStyleType

from npnf.data.configs.datasets.synthetic_test_sets import (
    SyntheticTestSet,
    design_splits,
)
from npnf.scripts.paper.style import COLOR_LIST, COLORS

POINT_STYLE_CHOICES = ("gray", "testset-context", "swapped-fill")
TESTSET_CONTEXT_POINT_ALPHA = 0.65
TESTSET_CONTEXT_POINT_SIZE = 20.0
TESTSET_CONTEXT_JITTER_FRACTION = 0.25
SWAPPED_FILL_POINT_ALPHA = 0.84
SWAPPED_FILL_MARKER_SIZE = 5.6
SWAPPED_FILL_JITTER = 0.23
SUMMARY_COLOR = "#111111"

PREDICTION_METHOD_LABELS = {
    "no_context": "no context",
    "random_context": "random context",
    "max_height": "max context",
}
CONDITIONING_LABELS = {
    "noenv_nogeno": "P",
    "env_nogeno": "E",
    "noenv_geno": "G",
    "env_geno": "E&G",
}
METHOD_COLOR_ORDER = ("no context", "random context", "max context")
SPLIT_MARKER_ORDER = ("seen", "env", "geno", "unseen")
CONDITION_FILL_ORDER = (
    "noenv_nogeno",
    "env_nogeno",
    "noenv_geno",
    "env_geno",
    "P",
    "E",
    "G",
    "E&G",
)
TEST_SET_MARKERS = {"seen": "o", "env": "s", "geno": "^", "unseen": "D"}
CONDITION_FILLSTYLES: dict[str, FillStyleType] = {
    "P": "none",
    "noenv_nogeno": "none",
    "E": "left",
    "env_nogeno": "left",
    "G": "right",
    "noenv_geno": "right",
    "E&G": "full",
    "env_geno": "full",
}


def split_label(split: str) -> str:
    """Short label of a split alias of any synthetic test set: its design role."""
    roles = {
        test_split.alias: test_split.role
        for test_set in SyntheticTestSet
        for test_split in design_splits(test_set)
    }
    return roles.get(split) or split


def prediction_method_label(method: str) -> str:
    return PREDICTION_METHOD_LABELS.get(method, method.replace("_", " "))


def conditioning_label(conditioning: str) -> str:
    return CONDITIONING_LABELS.get(conditioning, conditioning)


def ordered_intersection(values: Sequence[str], preferred: Sequence[str]) -> list[str]:
    unique_values = list(dict.fromkeys(str(value) for value in values))
    present = set(unique_values)
    ordered = [value for value in preferred if value in present]
    ordered.extend(value for value in unique_values if value not in ordered)
    return ordered


def split_palette(split_order: Sequence[str]) -> dict[str, ColorType]:
    known = {
        "seen": COLORS["blue"],
        "plot": COLORS["blue"],
        "env": COLORS["orange"],
        "environment": COLORS["orange"],
        "geno": COLORS["green"],
        "genotype": COLORS["green"],
        "unseen": COLORS["purple"],
    }
    return {
        split: known.get(split.lower(), COLOR_LIST[index % len(COLOR_LIST)])
        for index, split in enumerate(split_order)
    }


def method_markers(method_order: Sequence[str]) -> dict[str, str]:
    known = {
        "no context": "o",
        "no_context": "o",
        "random context": "s",
        "random_context": "s",
        "max context": "^",
        "max_height": "^",
    }
    fallback = ("o", "s", "^", "D", "P", "X")
    return {
        method: known.get(method.lower(), fallback[index % len(fallback)])
        for index, method in enumerate(method_order)
    }


def _legend_handle(
    *,
    label: str,
    color: ColorType = "#333333",
    marker: str = "o",
    markerfacecolor: ColorType | None = None,
) -> mlines.Line2D:
    return mlines.Line2D(
        [],
        [],
        marker=marker,
        markerfacecolor=color if markerfacecolor is None else markerfacecolor,
        markeredgecolor=color,
        linestyle="",
        markersize=5,
        label=label,
    )


def testset_context_handles(
    split_order: Sequence[str], method_order: Sequence[str]
) -> list[mlines.Line2D]:
    palette = split_palette(split_order)
    markers = method_markers(method_order)
    split_handles = [
        _legend_handle(label=split, color=palette[split], marker="o")
        for split in split_order
    ]
    method_handles = [
        _legend_handle(
            label=method,
            color="#333333",
            marker=markers[method],
            markerfacecolor="#777777",
        )
        for method in method_order
    ]
    return [*split_handles, *method_handles]


def raise_boxplot_lines_above_points(ax: plt.Axes) -> None:
    """Keep box fills below points while drawing outlines/lines above points."""
    for patch in list(ax.patches):
        patch.set_zorder(1)
        outline = PathPatch(
            patch.get_path(),
            transform=patch.get_transform(),
            facecolor="none",
            edgecolor=patch.get_edgecolor(),
            linewidth=patch.get_linewidth(),
            linestyle=patch.get_linestyle(),
            zorder=4,
        )
        ax.add_patch(outline)
    for line in ax.lines:
        line.set_zorder(5)


def _hue_offsets(hue_order: Sequence[str], width: float) -> dict[str, float]:
    if len(hue_order) == 1:
        return {hue_order[0]: 0.0}
    dodge_width = width / len(hue_order)
    offsets = np.linspace(
        -width / 2 + dodge_width / 2, width / 2 - dodge_width / 2, len(hue_order)
    )
    return dict(zip(hue_order, offsets, strict=True))


def overlay_testset_context_points(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    hue_col: str,
    y_col: str,
    x_order: Sequence[str],
    hue_order: Sequence[str],
    split_order: Sequence[str],
    method_order: Sequence[str],
    split_col: str = "split_label",
    method_col: str = "prediction_method_label",
    box_width: float = 0.72,
    rng_seed: int = 0,
) -> None:
    """Overlay colored/marked source points aligned to a dodged boxplot."""
    position_by_x = {str(value): index for index, value in enumerate(x_order)}
    offsets = _hue_offsets([str(value) for value in hue_order], box_width)
    dodge_width = box_width / max(len(hue_order), 1)
    palette = split_palette(split_order)
    markers = method_markers(method_order)
    rng = np.random.default_rng(rng_seed)

    for split in split_order:
        for method in method_order:
            subset = plot_df[
                (plot_df[split_col].astype(str) == split)
                & (plot_df[method_col].astype(str) == method)
            ]
            if subset.empty:
                continue
            x = np.array([position_by_x[str(value)] for value in subset[x_col]])
            x = x + np.array([offsets[str(value)] for value in subset[hue_col]])
            x = x + rng.uniform(
                -dodge_width * TESTSET_CONTEXT_JITTER_FRACTION,
                dodge_width * TESTSET_CONTEXT_JITTER_FRACTION,
                size=len(subset),
            )
            ax.scatter(
                x,
                subset[y_col],
                s=TESTSET_CONTEXT_POINT_SIZE,
                marker=markers[method],
                color=palette[split],
                alpha=TESTSET_CONTEXT_POINT_ALPHA,
                edgecolors="none",
                linewidths=0,
                zorder=2,
            )


def method_colors(method_order: Sequence[str]) -> dict[str, ColorType]:
    known = {
        "no context": COLORS["blue"],
        "no_context": COLORS["blue"],
        "random context": COLORS["orange"],
        "random_context": COLORS["orange"],
        "max context": COLORS["green"],
        "max_height": COLORS["green"],
    }
    return {
        method: known.get(method.lower(), COLOR_LIST[index % len(COLOR_LIST)])
        for index, method in enumerate(method_order)
    }


def test_set_markers(split_order: Sequence[str]) -> dict[str, str]:
    fallback = ("o", "s", "^", "D", "P", "X")
    return {
        split: TEST_SET_MARKERS.get(
            split_label(split.lower()), fallback[index % len(fallback)]
        )
        for index, split in enumerate(split_order)
    }


def condition_fillstyles(condition_order: Sequence[str]) -> dict[str, FillStyleType]:
    fallback: tuple[FillStyleType, ...] = ("none", "left", "right", "full")
    return {
        condition: CONDITION_FILLSTYLES.get(condition, fallback[index % len(fallback)])
        for index, condition in enumerate(condition_order)
    }


def swapped_fill_orders(
    *,
    split_values: Sequence[str],
    method_values: Sequence[str],
    condition_values: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    return (
        ordered_intersection(method_values, METHOD_COLOR_ORDER),
        ordered_intersection(split_values, SPLIT_MARKER_ORDER),
        ordered_intersection(condition_values, CONDITION_FILL_ORDER),
    )


def draw_swapped_fill_points(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_order: Sequence[str],
    split_order: Sequence[str],
    method_order: Sequence[str],
    condition_order: Sequence[str],
    split_col: str = "split_label",
    method_col: str = "prediction_method_label",
    condition_col: str = "conditioning_label",
    rng_seed: int = 0,
    jitter: float = SWAPPED_FILL_JITTER,
    yerr_col: str | None = None,
    geometric: bool = False,
) -> None:
    """Draw one marker per row, optionally with a vertical error bar.

    ``yerr_col`` names a column of symmetric error magnitudes. Bars are drawn
    here rather than by the caller because the x positions are jittered with a
    seeded RNG and could not otherwise be reproduced. With ``geometric``, the
    column holds log sds and the bar runs from ``value / exp(sd)`` to
    ``value * exp(sd)``.
    """
    positions = {str(value): index for index, value in enumerate(x_order)}
    colors = method_colors(method_order)
    markers = test_set_markers(split_order)
    fillstyles = condition_fillstyles(condition_order)
    rng = np.random.default_rng(rng_seed)
    jittered_x = [positions[str(value)] for value in plot_df[x_col]]
    jittered_x = np.array(jittered_x, dtype=float)
    jittered_x += rng.uniform(-jitter, jitter, size=len(plot_df))

    for x, row in zip(jittered_x, plot_df.to_dict("records"), strict=True):
        method = str(row[method_col])
        split = str(row[split_col])
        condition = str(row[condition_col])
        color = colors[method]
        if yerr_col is not None:
            error = row.get(yerr_col)
            if error is not None and np.isfinite(error) and error > 0:
                value = float(row[y_col])
                if geometric:
                    factor = float(np.exp(error))
                    lower, upper = value - value / factor, value * factor - value
                else:
                    # Clip at zero: a replicate sd can exceed the mean, and a
                    # bar crossing into negative Sig-MMD would be meaningless.
                    lower, upper = min(float(error), value), float(error)
                ax.errorbar(
                    x,
                    row[y_col],
                    yerr=[[lower], [upper]],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=SWAPPED_FILL_POINT_ALPHA,
                    zorder=1,
                )
        ax.plot(
            x,
            row[y_col],
            linestyle="",
            marker=markers[split],
            markersize=SWAPPED_FILL_MARKER_SIZE,
            markerfacecolor=color,
            markerfacecoloralt="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            fillstyle=fillstyles[condition],
            alpha=SWAPPED_FILL_POINT_ALPHA,
            zorder=2,
        )


SUMMARY_SPLIT_OFFSET = 0.22


def draw_centered_summary_bars(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_order: Sequence[str],
    split_col: str | None = None,
    split_order: Sequence[str] | None = None,
    seed_sd_col: str | None = None,
    geometric: bool = False,
) -> None:
    """Draw the swapped-fill mean/median/IQR summary overlay.

    When ``split_col`` and ``split_order`` are given, one bar is drawn per test
    set instead of one per model. Pooling test sets makes the IQR span the gap
    between an easy and a hard split, so the bar ends up describing split
    difficulty rather than spread within a split.

    When ``seed_sd_col`` names the per-row sd over seeds, the overlay shows the
    two sources of spread instead of the median and IQR: a thin whisker of
    +/- 1 sd over the rows and a thick bar of +/- the typical sd over seeds,
    the root mean square of ``seed_sd_col`` over the rows. With ``geometric``,
    the mean and the sd over the rows are taken of the log values,
    ``seed_sd_col`` holds log sds and the bars are factors around the
    geometric mean.
    """
    if split_col is not None and split_order is not None:
        offsets = summary_split_offsets(len(split_order))
        for offset, split in zip(offsets, split_order, strict=True):
            subset = plot_df[plot_df[split_col].astype(str) == str(split)]
            if subset.empty:
                continue
            _draw_summary_bars_at(
                ax,
                subset,
                x_col=x_col,
                y_col=y_col,
                x_order=x_order,
                offset=offset,
                seed_sd_col=seed_sd_col,
                geometric=geometric,
            )
        return
    _draw_summary_bars_at(
        ax,
        plot_df,
        x_col=x_col,
        y_col=y_col,
        x_order=x_order,
        offset=0.0,
        seed_sd_col=seed_sd_col,
        geometric=geometric,
    )


def summary_split_offsets(count: int) -> list[float]:
    """Return symmetric x offsets so per-split bars do not overlap."""
    if count == 1:
        return [0.0]
    step = (2 * SUMMARY_SPLIT_OFFSET) / (count - 1)
    return [-SUMMARY_SPLIT_OFFSET + step * index for index in range(count)]


def _draw_summary_bars_at(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_order: Sequence[str],
    offset: float,
    seed_sd_col: str | None = None,
    geometric: bool = False,
) -> None:
    if seed_sd_col is not None:
        _draw_seed_spread_at(
            ax,
            plot_df,
            x_col=x_col,
            y_col=y_col,
            x_order=x_order,
            offset=offset,
            seed_sd_col=seed_sd_col,
            geometric=geometric,
        )
        return
    grouped = plot_df.groupby(x_col, sort=False, observed=True)[y_col]
    summary = grouped.agg(
        mean="mean",
        median="median",
        q25=lambda series: series.quantile(0.25),
        q75=lambda series: series.quantile(0.75),
    )
    for position, category in enumerate(x_order):
        if category not in summary.index:
            continue
        index = position + offset
        row = summary.loc[category]
        ax.vlines(index, row.q25, row.q75, color=SUMMARY_COLOR, linewidth=1.2, zorder=6)
        ax.hlines(
            [row.q25, row.q75],
            index - 0.07,
            index + 0.07,
            color=SUMMARY_COLOR,
            linewidth=1.2,
            zorder=6,
        )
        ax.hlines(
            row["median"],
            index - 0.13,
            index + 0.13,
            color=SUMMARY_COLOR,
            linewidth=2.2,
            zorder=7,
        )
        ax.scatter(
            index,
            row["mean"],
            marker="D",
            s=22,
            facecolor="white",
            edgecolor=SUMMARY_COLOR,
            linewidth=1.0,
            zorder=8,
        )


def _draw_seed_spread_at(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_order: Sequence[str],
    offset: float,
    seed_sd_col: str,
    geometric: bool,
) -> None:
    for position, category in enumerate(x_order):
        subset = plot_df[plot_df[x_col].astype(str) == str(category)]
        if subset.empty:
            continue
        index = position + offset
        cells = np.log(subset[y_col]) if geometric else subset[y_col]
        mean = float(cells.mean())
        cell_sd = float(cells.std(ddof=1)) if len(subset) > 1 else 0.0
        seed_sd = float(np.sqrt((subset[seed_sd_col] ** 2).mean()))
        cell_low, cell_high = mean - cell_sd, mean + cell_sd
        seed_low, seed_high = mean - seed_sd, mean + seed_sd
        if geometric:
            mean, cell_low, cell_high, seed_low, seed_high = np.exp(
                [mean, cell_low, cell_high, seed_low, seed_high]
            )
        ax.vlines(
            index, cell_low, cell_high, color=SUMMARY_COLOR, linewidth=1.0, zorder=6
        )
        ax.hlines(
            [cell_low, cell_high],
            index - 0.06,
            index + 0.06,
            color=SUMMARY_COLOR,
            linewidth=1.0,
            zorder=6,
        )
        ax.vlines(
            index, seed_low, seed_high, color=SUMMARY_COLOR, linewidth=4.0, zorder=7
        )
        ax.scatter(
            index,
            mean,
            marker="D",
            s=26,
            facecolor="white",
            edgecolor=SUMMARY_COLOR,
            linewidth=1.0,
            zorder=8,
        )


def _condition_legend_label(
    condition: str, condition_labels: Mapping[str, str] | None
) -> str:
    if condition_labels is None:
        return condition
    return condition_labels.get(condition, condition)


def swapped_fill_handles(
    method_order: Sequence[str],
    split_order: Sequence[str],
    condition_order: Sequence[str],
    condition_labels: Mapping[str, str] | None = None,
    *,
    seed_spread: bool = False,
    geometric: bool = False,
) -> list[mlines.Line2D]:
    colors = method_colors(method_order)
    markers = test_set_markers(split_order)
    fillstyles = condition_fillstyles(condition_order)
    method_handles = [
        _legend_handle(label=method, color=colors[method], marker="o")
        for method in method_order
    ]
    split_handles = [
        _legend_handle(
            label=split,
            color="#333333",
            marker=markers[split],
            markerfacecolor="#777777",
        )
        for split in split_order
    ]
    condition_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="#777777",
            markerfacecoloralt="white",
            markeredgecolor="#333333",
            fillstyle=fillstyles[condition],
            linestyle="",
            markersize=5,
            label=_condition_legend_label(condition, condition_labels),
        )
        for condition in condition_order
    ]
    mean_handle = mlines.Line2D(
        [],
        [],
        marker="D",
        markerfacecolor="white",
        markeredgecolor=SUMMARY_COLOR,
        linestyle="",
        markersize=5,
        label="Geometric mean" if geometric else "Mean",
    )
    if seed_spread:
        spread = "Geometric sd" if geometric else "± sd"
        summary_handles = [
            mean_handle,
            mlines.Line2D(
                [], [], color=SUMMARY_COLOR, linewidth=4.0, label=f"{spread} over seeds"
            ),
            mlines.Line2D(
                [],
                [],
                color=SUMMARY_COLOR,
                linewidth=1.0,
                label=f"{spread} over settings",
            ),
        ]
    else:
        summary_handles = [
            mean_handle,
            mlines.Line2D([], [], color=SUMMARY_COLOR, linewidth=2.2, label="Median"),
            mlines.Line2D([], [], color=SUMMARY_COLOR, linewidth=1.2, label="IQR"),
        ]
    return [*method_handles, *split_handles, *condition_handles, *summary_handles]


def add_swapped_fill_legend(
    fig: plt.Figure,
    *,
    method_order: Sequence[str],
    split_order: Sequence[str],
    condition_order: Sequence[str],
    condition_labels: Mapping[str, str] | None = None,
    anchor_y: float = 0.105,
    seed_spread: bool = False,
    geometric: bool = False,
) -> None:
    handles = swapped_fill_handles(
        method_order,
        split_order,
        condition_order,
        condition_labels,
        seed_spread=seed_spread,
        geometric=geometric,
    )
    method_count = len(method_order)
    split_count = len(split_order)
    condition_count = len(condition_order)
    grouped_handles = [
        handles[:method_count],
        handles[method_count : method_count + split_count],
        handles[
            method_count + split_count : method_count + split_count + condition_count
        ],
        handles[method_count + split_count + condition_count :],
    ]
    anchors = (0.16, 0.39, 0.61, 0.83)
    ncols = (1, 2, 2, 1)
    for group_handles, anchor_x, ncol in zip(
        grouped_handles, anchors, ncols, strict=True
    ):
        if not group_handles:
            continue
        legend = fig.legend(
            handles=group_handles,
            labels=[handle.get_label() for handle in group_handles],
            loc="upper center",
            bbox_to_anchor=(anchor_x, anchor_y),
            ncol=ncol,
            fontsize=11,
            frameon=False,
            handlelength=1.5,
            columnspacing=1.0,
        )
        fig.add_artist(legend)
