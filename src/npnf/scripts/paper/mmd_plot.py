"""Build all-model raw Sig-MMD / CSig-MMD paper boxplots from results."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from npnf.scripts.paper import mmd_table
from npnf.scripts.paper.sig_mmd_point_style import (
    SUMMARY_COLOR,
    SWAPPED_FILL_MARKER_SIZE,
    SWAPPED_FILL_POINT_ALPHA,
    add_swapped_fill_legend,
    condition_fillstyles,
    draw_centered_summary_bars,
    draw_swapped_fill_points,
    method_colors,
    overlay_testset_context_points,
    raise_boxplot_lines_above_points,
    swapped_fill_orders,
    test_set_markers,
    testset_context_handles,
)
from npnf.scripts.paper.style import COLORS, set_log_y_axis, setup_style

DEFAULT_OUTPUT_DIRS = {
    "sig_mmd": Path("paper/sig_mmd_plot"),
    "csig_mmd": Path("paper/csig_mmd_plot"),
}
DEFAULT_PREDICTION_METHODS = ["no_context", "random_context", "max_height"]
POINT_STYLE_CHOICES = ("testset-context", "gray", "swapped-fill")
POINT_COLOR = "#222222"
POINT_ALPHA = 0.30
POINT_SIZE = 2.0
BOX_WIDTH = 0.72
GRAY_POINT_JITTER = 0.18


@dataclass(frozen=True)
class Config:
    results_base: Path
    specs: tuple[mmd_table.TableSpec, ...]
    prediction_methods: tuple[str, ...]
    prediction_method_labels: tuple[str, ...]
    checkpoint: str
    metric: str
    metric_type: str
    view: str
    prediction_method_views: dict[str, str]
    output_dir: Path
    title: str
    point_style: str
    average_test_sets: bool
    average_prediction_methods: bool
    log_y: bool = False
    aggregate_replicates: bool = False
    summary_by_split: bool = False
    right_panel_models: tuple[str, ...] = ()


def default_output_dir(metric_type: str) -> Path:
    mmd_table.metric_type_config(metric_type)
    return DEFAULT_OUTPUT_DIRS[metric_type]


def averaging_suffix(
    *, average_test_sets: bool = False, average_prediction_methods: bool = False
) -> str:
    """Return a filename suffix that describes active averaging dimensions."""
    parts = []
    if average_test_sets:
        parts.append("testset")
    if average_prediction_methods:
        parts.append("prediction_method")
    return "_".join([*parts, "averaged"]) if parts else ""


def output_stem(
    metric_type: str,
    view: str = mmd_table.DEFAULT_VIEW,
    prediction_method_views: Mapping[str, str] | None = None,
    *,
    average_test_sets: bool = False,
    average_prediction_methods: bool = False,
) -> str:
    stem_prefix = (
        mmd_table.metric_stem_prefix(metric_type, view)
        if prediction_method_views is None
        else mmd_table.metric_stem_prefix_for_prediction_method_views(
            metric_type, prediction_method_views
        )
    )
    stem = f"{stem_prefix}_boxplot"
    suffix = averaging_suffix(
        average_test_sets=average_test_sets,
        average_prediction_methods=average_prediction_methods,
    )
    return f"{stem}_{suffix}" if suffix else stem


def geometric_replicates(cfg: Config) -> bool:
    """Replicate means and sds are geometric on a log axis."""
    return cfg.aggregate_replicates and cfg.log_y


def scaled_metric_column(metric_type: str) -> str:
    stem_prefix = mmd_table.metric_type_config(metric_type)["stem_prefix"]
    return f"{stem_prefix}_x1000"


def y_axis_label(metric_type: str) -> str:
    label = mmd_table.metric_type_config(metric_type)["label"]
    return f"{label}² × 10³"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an all-model raw Sig-MMD / CSig-MMD boxplot directly from "
            "results directories."
        )
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default=mmd_table.default_results_base(),
        help="Base results directory (default: $NPNF_RESULTS_DIR or results).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=mmd_table.DEFAULT_MODELS,
        help="Model result folder names, in x-axis order.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help="Model display labels. Defaults to known labels, else model names.",
    )
    mmd_table.add_test_set_arguments(parser)
    mmd_table.add_model_checkpoints_argument(parser)
    parser.add_argument(
        "--split-labels",
        nargs="+",
        default=None,
        help="Display labels for splits. Defaults to built-in labels.",
    )
    parser.add_argument(
        "--conditionings",
        nargs="+",
        default=mmd_table.DEFAULT_CONDITIONINGS,
        help="Conditioning directory names to include in plotted source points.",
    )
    parser.add_argument(
        "--conditioning-labels",
        nargs="+",
        default=None,
        help="Display labels for conditionings. Defaults to P E G 'E&G'.",
    )
    parser.add_argument(
        "--prediction-methods",
        nargs="+",
        default=DEFAULT_PREDICTION_METHODS,
        choices=sorted(mmd_table.PREDICTION_METHODS),
        help="Prediction methods to include in plotted source points.",
    )
    parser.add_argument(
        "--prediction-method-names",
        nargs="+",
        default=None,
        help="Display labels for --prediction-methods. Defaults to readable names.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=mmd_table.DEFAULT_CHECKPOINT,
        help="Common checkpoint: latest, N, or checkpoint-N (default: latest).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=mmd_table.DEFAULT_METRIC,
        help="Metric column from summary CSV (default: mean).",
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default=mmd_table.DEFAULT_METRIC_TYPE,
        choices=mmd_table.metric_type_choices(),
        help="Metric to plot: sig_mmd or csig_mmd. (default: sig_mmd)",
    )
    parser.add_argument(
        "--view",
        type=str,
        default=mmd_table.DEFAULT_VIEW,
        choices=mmd_table.VIEW_CHOICES,
        help=(
            "Metric view to plot: condition or "
            "context_matched_blocked. (default: condition)"
        ),
    )
    parser.add_argument(
        "--prediction-method-views",
        nargs="*",
        default=None,
        metavar="METHOD=VIEW",
        help=(
            "Per-prediction-method metric view overrides. Each token must be "
            "METHOD=VIEW; unlisted methods use --view."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for boxplot outputs. Defaults to paper/sig_mmd_plot for "
            "sig_mmd and paper/csig_mmd_plot for csig_mmd."
        ),
    )
    parser.add_argument(
        "--title",
        default="",
        help="Figure title. Pass an empty string to omit the title.",
    )
    parser.add_argument(
        "--point-style",
        choices=POINT_STYLE_CHOICES,
        default="testset-context",
        help="Point overlay style: testset-context colors points by test set and "
        "shapes them by prediction method; gray uses a simple gray point overlay; "
        "swapped-fill colors points by prediction method, shapes them by test set, "
        "and fills them by conditioning.",
    )
    parser.add_argument(
        "--average-test-sets",
        action="store_true",
        help=(
            "Average plotted values over requested test splits before plotting. "
            "Currently supported with --point-style swapped-fill."
        ),
    )
    parser.add_argument(
        "--average-prediction-methods",
        action="store_true",
        help=(
            "Average plotted values over requested prediction methods before "
            "plotting. Currently supported with --point-style swapped-fill."
        ),
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Logarithmic y-axis with 1-2-5 tick labels. Every plotted value must "
        "be positive.",
    )
    parser.add_argument(
        "--summary-by-split",
        action="store_true",
        help=(
            "Draw one mean/median/IQR summary bar per test set instead of one "
            "per model. Pooling test sets makes the IQR span the gap between an "
            "easy and a hard split, so it describes split difficulty rather "
            "than spread within a split. Currently supported with "
            "--point-style swapped-fill."
        ),
    )
    parser.add_argument(
        "--aggregate-replicates",
        action="store_true",
        help=(
            "Treat runs sharing a --model-names label as replicates of one "
            "model: draw one point per evaluation cell at the replicate mean, "
            "with a +/- 1 sd error bar across replicates. The summary bar then "
            "shows the mean, +/- 1 sd over the cells (thin) and +/- the typical "
            "sd over replicates (thick) instead of the median and IQR. With "
            "--log-y, means and sds are geometric and the bars are factors. Without "
            "this, each replicate is drawn as its own point and its own summary bar. "
            "Currently supported with --point-style swapped-fill."
        ),
    )
    parser.add_argument(
        "--right-panel-models",
        nargs="*",
        default=[],
        help=(
            "--model-names labels drawn in a second panel on the right with its own "
            "y range, for models whose scores are far above the others."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    prediction_methods = list(args.prediction_methods)
    prediction_method_labels = mmd_table.resolve_prediction_method_labels(
        prediction_methods, args.prediction_method_names
    )
    view = mmd_table.validate_metric_view(args.metric_type, args.view)
    prediction_method_views = mmd_table.resolve_prediction_method_views(
        prediction_methods,
        metric_type=args.metric_type,
        default_view=view,
        raw_overrides=args.prediction_method_views,
    )
    if (
        args.average_test_sets or args.average_prediction_methods
    ) and args.point_style != "swapped-fill":
        msg = (
            "--average-test-sets and --average-prediction-methods are currently "
            "supported only with --point-style swapped-fill."
        )
        raise ValueError(msg)
    if args.aggregate_replicates and args.point_style != "swapped-fill":
        msg = (
            "--aggregate-replicates is currently supported only with "
            "--point-style swapped-fill."
        )
        raise ValueError(msg)
    if args.summary_by_split and args.point_style != "swapped-fill":
        msg = (
            "--summary-by-split is currently supported only with "
            "--point-style swapped-fill."
        )
        raise ValueError(msg)
    specs = mmd_table.validate_and_build_specs(args)
    if args.right_panel_models:
        if (
            args.point_style != "swapped-fill"
            or args.average_test_sets
            or args.average_prediction_methods
        ):
            msg = (
                "--right-panel-models is currently supported only with "
                "--point-style swapped-fill and without averaging."
            )
            raise ValueError(msg)
        unknown = set(args.right_panel_models) - {spec.model_name for spec in specs}
        if unknown:
            msg = f"--right-panel-models are not --model-names: {sorted(unknown)}."
            raise ValueError(msg)
    return Config(
        results_base=Path(args.results_base),
        specs=tuple(specs),
        prediction_methods=tuple(prediction_methods),
        prediction_method_labels=tuple(prediction_method_labels),
        checkpoint=args.checkpoint,
        metric=args.metric,
        metric_type=args.metric_type,
        view=view,
        prediction_method_views=prediction_method_views,
        output_dir=args.output_dir or default_output_dir(args.metric_type),
        title=args.title,
        point_style=args.point_style,
        average_test_sets=args.average_test_sets,
        average_prediction_methods=args.average_prediction_methods,
        log_y=args.log_y,
        aggregate_replicates=args.aggregate_replicates,
        summary_by_split=args.summary_by_split,
        right_panel_models=tuple(args.right_panel_models),
    )


def read_raw_values(cfg: Config) -> pd.DataFrame:
    resolved_rows = mmd_table.load_summaries_for_methods(
        cfg.results_base,
        list(cfg.specs),
        prediction_methods=list(cfg.prediction_methods),
        metric_type=cfg.metric_type,
        checkpoint=cfg.checkpoint,
        metric=cfg.metric,
        view=cfg.view,
        prediction_method_views=cfg.prediction_method_views,
    )
    label_by_method = dict(
        zip(cfg.prediction_methods, cfg.prediction_method_labels, strict=True)
    )
    plot_column = scaled_metric_column(cfg.metric_type)

    return pd.DataFrame(
        [
            {
                "model": row.spec.model,
                "model_name": row.spec.model_name,
                "prediction_method": row.prediction_method,
                "prediction_method_label": label_by_method[row.prediction_method],
                "conditioning": row.spec.conditioning,
                "conditioning_label": row.spec.conditioning_label,
                "checkpoint": row.checkpoint,
                "view": row.view,
                "split": row.spec.split,
                "split_label": row.spec.split_label,
                cfg.metric: row.value,
                plot_column: row.value * 1000.0,
                "summary_path": str(row.summary_path),
            }
            for row in resolved_rows
        ]
    )


def _prediction_method_views_from_values(values: pd.DataFrame) -> str:
    views_by_method: dict[str, list[str]] = {}
    for method, view in values[["prediction_method", "view"]].itertuples(
        index=False, name=None
    ):
        method_views = views_by_method.setdefault(str(method), [])
        view = str(view)
        if view not in method_views:
            method_views.append(view)
    return ";".join(
        f"{method}={'|'.join(views_by_method[method])}"
        for method in sorted(views_by_method)
    )


REPLICATE_SD_COLUMN = "replicate_sd"
REPLICATE_N_COLUMN = "replicate_n"
REPLICATE_CELL_KEYS = (
    "model_name",
    "split",
    "split_label",
    "conditioning",
    "conditioning_label",
    "prediction_method",
    "prediction_method_label",
    "checkpoint",
    "view",
)


def aggregate_replicate_values(
    values: pd.DataFrame, metric_type: str, *, geometric: bool = False
) -> pd.DataFrame:
    """Collapse runs sharing a display label into one row per evaluation cell.

    Several run names may map to one ``model_name`` when a model has been
    retrained under different seeds. Without this, each replicate contributes
    its own marker and its own summary bar. Aggregating keeps one marker per
    cell and attaches the across-replicate sd, so the drawn spread is
    run-to-run variation rather than a pile of overlapping runs.

    With ``geometric``, the cell value is the geometric mean over replicates
    and the sd is that of the natural log, a factor ``exp(sd)`` around the
    geometric mean. On a log axis a +/- sd bar is stretched below the mean,
    while a factor bar is symmetric.
    """
    plot_column = scaled_metric_column(metric_type)
    keys = [key for key in REPLICATE_CELL_KEYS if key in values.columns]
    if geometric:
        smallest = float(values[plot_column].min())
        if smallest <= 0:
            msg = f"geometric means need positive values, the smallest is {smallest:g}"
            raise ValueError(msg)
        values = values.assign(**{plot_column: np.log(values[plot_column])})
    grouped = values.groupby(keys, as_index=False, sort=False).agg(
        **{
            plot_column: (plot_column, "mean"),
            REPLICATE_SD_COLUMN: (plot_column, lambda series: series.std(ddof=1)),
            REPLICATE_N_COLUMN: (plot_column, "size"),
        }
    )
    # A single-run label has an undefined sample sd; draw it bare rather than
    # implying zero spread.
    grouped[REPLICATE_SD_COLUMN] = grouped[REPLICATE_SD_COLUMN].fillna(0.0)
    if geometric:
        grouped[plot_column] = np.exp(grouped[plot_column])
    grouped["model"] = grouped["model_name"]
    return grouped


def summarize_by_model(
    values: pd.DataFrame,
    metric_type: str,
    prediction_method_views: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    plot_column = scaled_metric_column(metric_type)
    summary = values.groupby(["model", "model_name"], as_index=False, sort=False).agg(
        n=(plot_column, "size"),
        mean=(plot_column, "mean"),
        median=(plot_column, "median"),
        q05=(plot_column, lambda series: series.quantile(0.05)),
        q25=(plot_column, lambda series: series.quantile(0.25)),
        q75=(plot_column, lambda series: series.quantile(0.75)),
        q95=(plot_column, lambda series: series.quantile(0.95)),
        min=(plot_column, "min"),
        max=(plot_column, "max"),
    )
    summary["prediction_method_views"] = (
        _prediction_method_views_from_values(values)
        if prediction_method_views is None
        else mmd_table.format_prediction_method_views(prediction_method_views)
    )
    if "averaged_over" in values.columns:
        summary["averaged_over"] = _unique_join(values["averaged_over"])
        if "n_rows" in values.columns:
            source_counts = values.groupby(
                ["model", "model_name"], as_index=False, sort=False
            )["n_rows"].sum()
            summary = summary.merge(
                source_counts.rename(columns={"n_rows": "source_n"}),
                on=["model", "model_name"],
                how="left",
            )
    return summary


def _ordered_unique(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _label_by_value(
    values: pd.DataFrame, *, value_col: str, label_col: str
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for value, label in values[[value_col, label_col]].itertuples(
        index=False, name=None
    ):
        labels.setdefault(str(value), str(label))
    return labels


def _unique_join(values: pd.Series) -> str:
    return ";".join(sorted(dict.fromkeys(str(value) for value in values)))


def _single_or_mixed(values: pd.Series) -> str:
    unique_values = sorted(dict.fromkeys(str(value) for value in values))
    return unique_values[0] if len(unique_values) == 1 else "mixed"


def _averaged_over_label(
    *, average_test_sets: bool, average_prediction_methods: bool
) -> str:
    labels = []
    if average_test_sets:
        labels.append("split")
    if average_prediction_methods:
        labels.append("prediction_method")
    return ",".join(labels)


def average_plot_values(
    values: pd.DataFrame,
    *,
    metric: str,
    metric_type: str,
    average_test_sets: bool = False,
    average_prediction_methods: bool = False,
) -> pd.DataFrame:
    """Average raw source rows over requested plotting dimensions."""
    if not average_test_sets and not average_prediction_methods:
        return values

    plot_column = scaled_metric_column(metric_type)
    group_cols = [
        "model",
        "model_name",
        "conditioning",
        "conditioning_label",
        "checkpoint",
    ]
    if not average_prediction_methods:
        group_cols.extend(["prediction_method", "prediction_method_label"])
    if not average_test_sets:
        group_cols.extend(["split", "split_label"])

    averaged = (
        values.groupby(group_cols, as_index=False, sort=False)
        .agg(
            **{
                metric: (metric, "mean"),
                plot_column: (plot_column, "mean"),
                "n_rows": (plot_column, "size"),
                "view": ("view", _single_or_mixed),
                "source_views": ("view", _unique_join),
                "source_prediction_methods": ("prediction_method", _unique_join),
                "source_prediction_method_labels": (
                    "prediction_method_label",
                    _unique_join,
                ),
                "source_splits": ("split", _unique_join),
                "source_split_labels": ("split_label", _unique_join),
                "summary_paths": ("summary_path", _unique_join),
            }
        )
        .copy()
    )
    averaged["metric_type"] = metric_type
    averaged["averaged_over"] = _averaged_over_label(
        average_test_sets=average_test_sets,
        average_prediction_methods=average_prediction_methods,
    )
    return averaged


def _mean_handle() -> mlines.Line2D:
    return mlines.Line2D(
        [],
        [],
        marker="D",
        markerfacecolor="white",
        markeredgecolor="#333333",
        linestyle="",
        markersize=4,
        label="Mean",
    )


def _source_handle() -> mlines.Line2D:
    return mlines.Line2D(
        [],
        [],
        marker="o",
        markerfacecolor=POINT_COLOR,
        markeredgecolor=POINT_COLOR,
        linestyle="",
        markersize=4,
        label="Source row",
    )


def raw_ylim(values: pd.Series, errors: pd.Series | None = None) -> tuple[float, float]:
    """Return y limits covering the points and any error bars drawn on them.

    Limits are set explicitly, so an error bar taller than every point would
    otherwise be cut at the axis edge - understating the spread exactly where
    it is largest. The axis starts at 0, or below the lowest value if that is
    negative (an unbiased score of a near-perfect model).
    """
    low = min(float(values.min()), 0.0)
    high = float(values.max())
    if errors is not None:
        high = max(high, float((values + errors).max()))
    pad = max((high - low) * 0.08, 0.4)
    return (low - pad if low < 0 else 0.0), high + pad


def _set_y_axis(
    ax: plt.Axes,
    values: pd.Series,
    *,
    log_y: bool,
    errors: pd.Series | None = None,
    geometric: bool = False,
) -> None:
    """Set the y limits so that the points and their bars fit.

    With ``geometric``, ``errors`` are log sds, drawn as bars from
    ``value / exp(sd)`` to ``value * exp(sd)``.
    """
    if not log_y:
        ax.set_ylim(*raw_ylim(values, errors))
        return
    bar_ends = None
    if errors is not None:
        if geometric:
            factors = np.exp(errors)
            bar_ends = pd.concat([values / factors, values * factors])
        else:
            bar_ends = pd.concat([values - errors, values + errors])
    set_log_y_axis(ax, values, bar_ends=bar_ends)


def _centered_offsets(values: Sequence[str], width: float) -> dict[str, float]:
    values = [str(value) for value in values]
    if len(values) <= 1:
        return dict.fromkeys(values, 0.0)
    step = (2 * width) / (len(values) - 1)
    return {value: -width + index * step for index, value in enumerate(values)}


def _summary_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D([], [], color=SUMMARY_COLOR, linewidth=1.1, label="5–95%"),
        mlines.Line2D([], [], color=SUMMARY_COLOR, linewidth=4.0, label="25–75%"),
        mlines.Line2D([], [], color=SUMMARY_COLOR, linewidth=2.2, label="Median"),
        mlines.Line2D(
            [],
            [],
            marker="D",
            markerfacecolor="white",
            markeredgecolor=SUMMARY_COLOR,
            linestyle="",
            markersize=5,
            label="Mean",
        ),
    ]


def _add_grouped_bottom_legends(
    fig: plt.Figure, groups: Sequence[tuple[list[mlines.Line2D], int]]
) -> None:
    if len(groups) == 1:
        anchors = (0.5,)
    elif len(groups) == 2:
        anchors = (0.38, 0.72)
    else:
        anchors = (0.18, 0.47, 0.78)
    for (handles, ncol), anchor_x in zip(groups, anchors, strict=True):
        legend = fig.legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc="upper center",
            bbox_to_anchor=(anchor_x, 0.16),
            ncol=ncol,
            fontsize=11,
            frameon=False,
            handlelength=1.5,
            columnspacing=1.0,
        )
        fig.add_artist(legend)


def plot_averaged_swapped_fill(
    values: pd.DataFrame, cfg: Config, output_path_stem: Path
) -> None:
    setup_style()
    plot_column = scaled_metric_column(cfg.metric_type)
    model_order = _ordered_unique(values["model_name"])
    has_method = "prediction_method_label" in values.columns
    has_split = "split_label" in values.columns
    method_values = (
        _ordered_unique(values["prediction_method_label"]) if has_method else []
    )
    split_values = _ordered_unique(values["split_label"]) if has_split else []
    method_order, split_order, condition_order = swapped_fill_orders(
        split_values=split_values,
        method_values=method_values,
        condition_values=_ordered_unique(values["conditioning"]),
    )
    condition_labels = _label_by_value(
        values, value_col="conditioning", label_col="conditioning_label"
    )

    plot_df = values.copy()
    plot_df["model_name"] = pd.Categorical(
        plot_df["model_name"], categories=model_order, ordered=True
    )
    plot_df["conditioning"] = pd.Categorical(
        plot_df["conditioning"], categories=condition_order, ordered=True
    )
    if has_method:
        plot_df["prediction_method_label"] = pd.Categorical(
            plot_df["prediction_method_label"], categories=method_order, ordered=True
        )
    if has_split:
        plot_df["split_label"] = pd.Categorical(
            plot_df["split_label"], categories=split_order, ordered=True
        )

    model_positions = {model: index for index, model in enumerate(model_order)}
    condition_offsets = _centered_offsets(condition_order, 0.27)
    method_offsets = _centered_offsets(method_order, 0.025) if has_method else {}
    split_offsets = _centered_offsets(split_order, 0.025) if has_split else {}
    fillstyles = condition_fillstyles(condition_order)
    colors = method_colors(method_order) if has_method else {}
    markers = test_set_markers(split_order) if has_split else {}
    default_color = COLORS["blue"]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for row in plot_df.to_dict("records"):
        condition = str(row["conditioning"])
        method = str(row["prediction_method_label"]) if has_method else ""
        split = str(row["split_label"]) if has_split else ""
        x = model_positions[str(row["model_name"])] + condition_offsets[condition]
        if has_method:
            x += method_offsets[method]
        if has_split:
            x += split_offsets[split]
        color = colors[method] if has_method else default_color
        marker = markers[split] if has_split else "o"
        ax.plot(
            x,
            row[plot_column],
            linestyle="",
            marker=marker,
            markersize=SWAPPED_FILL_MARKER_SIZE + 1.4,
            markerfacecolor=color,
            markerfacecoloralt="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            fillstyle=fillstyles[condition],
            alpha=SWAPPED_FILL_POINT_ALPHA,
            zorder=3,
        )

    draw_centered_summary_bars(
        ax, plot_df, x_col="model_name", y_col=plot_column, x_order=model_order
    )
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    ax.set_ylabel(y_axis_label(cfg.metric_type))
    if cfg.title:
        ax.set_title(cfg.title)
    _set_y_axis(
        ax,
        plot_df[plot_column],
        log_y=cfg.log_y,
        errors=plot_df.get(REPLICATE_SD_COLUMN),
        geometric=geometric_replicates(cfg),
    )
    ax.set_xticks(range(len(model_order)), model_order)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    legend_groups: list[tuple[list[mlines.Line2D], int]] = []
    if has_method:
        legend_groups.append(
            (
                [
                    mlines.Line2D(
                        [],
                        [],
                        marker="o",
                        markerfacecolor=colors[method],
                        markeredgecolor=colors[method],
                        linestyle="",
                        markersize=5,
                        label=method,
                    )
                    for method in method_order
                ],
                1,
            )
        )
    if has_split:
        legend_groups.append(
            (
                [
                    mlines.Line2D(
                        [],
                        [],
                        marker=markers[split],
                        markerfacecolor="#777777",
                        markeredgecolor="#333333",
                        linestyle="",
                        markersize=5,
                        label=split,
                    )
                    for split in split_order
                ],
                2,
            )
        )
    legend_groups.append(
        (
            [
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
                    label=condition_labels.get(condition, condition),
                )
                for condition in condition_order
            ],
            2,
        )
    )
    legend_groups.append((_summary_handles(), 2))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.39)
    _add_grouped_bottom_legends(fig, legend_groups)
    for fmt in ("png", "pdf"):
        fig.savefig(output_path_stem.with_suffix(f".{fmt}"))
    plt.close(fig)


def _draw_swapped_fill_panel(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    cfg: Config,
    *,
    x_order: Sequence[str],
    split_order: Sequence[str],
    method_order: Sequence[str],
    condition_order: Sequence[str],
) -> None:
    """The swapped-fill points, summary bars and axes of the models of one panel."""
    plot_column = scaled_metric_column(cfg.metric_type)
    geometric = geometric_replicates(cfg)
    draw_swapped_fill_points(
        ax,
        panel_df,
        x_col="model_name",
        y_col=plot_column,
        x_order=x_order,
        split_order=split_order,
        method_order=method_order,
        condition_order=condition_order,
        condition_col="conditioning",
        rng_seed=289,
        yerr_col=(
            REPLICATE_SD_COLUMN if REPLICATE_SD_COLUMN in panel_df.columns else None
        ),
        geometric=geometric,
    )
    draw_centered_summary_bars(
        ax,
        panel_df,
        x_col="model_name",
        y_col=plot_column,
        x_order=x_order,
        split_col="split_label" if cfg.summary_by_split else None,
        split_order=split_order if cfg.summary_by_split else None,
        seed_sd_col=REPLICATE_SD_COLUMN if cfg.aggregate_replicates else None,
        geometric=geometric,
    )
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    _set_y_axis(
        ax,
        panel_df[plot_column],
        log_y=cfg.log_y,
        errors=panel_df.get(REPLICATE_SD_COLUMN),
        geometric=geometric,
    )
    ax.set_xticks(range(len(x_order)), x_order)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def plot_boxplot(values: pd.DataFrame, cfg: Config, output_path_stem: Path) -> None:
    setup_style()
    plot_column = scaled_metric_column(cfg.metric_type)
    geometric = geometric_replicates(cfg)
    model_order = _ordered_unique(values["model_name"])
    split_order = _ordered_unique(values["split_label"])
    method_order = _ordered_unique(values["prediction_method_label"])

    plot_df = values.copy()
    plot_df["model_name"] = pd.Categorical(
        plot_df["model_name"], categories=model_order, ordered=True
    )
    plot_df["split_label"] = pd.Categorical(
        plot_df["split_label"], categories=split_order, ordered=True
    )
    plot_df["prediction_method_label"] = pd.Categorical(
        plot_df["prediction_method_label"], categories=method_order, ordered=True
    )

    if cfg.point_style == "swapped-fill":
        method_order, split_order, condition_order = swapped_fill_orders(
            split_values=split_order,
            method_values=method_order,
            condition_values=_ordered_unique(plot_df["conditioning"]),
        )
        condition_labels = _label_by_value(
            plot_df, value_col="conditioning", label_col="conditioning_label"
        )
        plot_df["conditioning"] = pd.Categorical(
            plot_df["conditioning"], categories=condition_order, ordered=True
        )
        left_order = [m for m in model_order if m not in cfg.right_panel_models]
        panel_orders = [left_order]
        if cfg.right_panel_models:
            panel_orders.append(list(cfg.right_panel_models))
        fig, axes = plt.subplots(
            1,
            len(panel_orders),
            figsize=(10.5, 4.8),
            squeeze=False,
            gridspec_kw={"width_ratios": [len(order) for order in panel_orders]},
        )
        for ax, panel_order in zip(axes[0], panel_orders, strict=True):
            _draw_swapped_fill_panel(
                ax,
                plot_df[plot_df["model_name"].astype(str).isin(panel_order)],
                cfg,
                x_order=panel_order,
                split_order=split_order,
                method_order=method_order,
                condition_order=condition_order,
            )
        ax = axes[0][0]
        ax.set_ylabel(y_axis_label(cfg.metric_type))
        if cfg.title:
            ax.set_title(cfg.title)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.39)
        add_swapped_fill_legend(
            fig,
            method_order=method_order,
            split_order=split_order,
            condition_order=condition_order,
            condition_labels=condition_labels,
            anchor_y=0.16,
            seed_spread=cfg.aggregate_replicates,
            geometric=geometric,
        )
        for fmt in ("png", "pdf"):
            fig.savefig(output_path_stem.with_suffix(f".{fmt}"))
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.boxplot(
        data=plot_df,
        x="model_name",
        y=plot_column,
        order=model_order,
        color=COLORS["blue"],
        width=BOX_WIDTH,
        whis=(5, 95),
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markersize": 4,
            "zorder": 5,
        },
        linewidth=1.0,
        ax=ax,
    )

    if cfg.point_style == "testset-context":
        plot_df["box_hue"] = "all"
        overlay_testset_context_points(
            ax,
            plot_df,
            x_col="model_name",
            hue_col="box_hue",
            y_col=plot_column,
            x_order=model_order,
            hue_order=["all"],
            split_order=split_order,
            method_order=method_order,
            box_width=BOX_WIDTH,
            rng_seed=289,
        )
        raise_boxplot_lines_above_points(ax)
        legend_handles = [
            *testset_context_handles(split_order, method_order),
            _mean_handle(),
        ]
        legend_ncol = 4
        legend_bottom = 0.36
        legend_anchor = 0.04
    else:
        sns.stripplot(
            data=plot_df,
            x="model_name",
            y=plot_column,
            order=model_order,
            jitter=GRAY_POINT_JITTER,
            color=POINT_COLOR,
            size=POINT_SIZE,
            alpha=POINT_ALPHA,
            ax=ax,
        )
        raise_boxplot_lines_above_points(ax)
        legend_handles = [_source_handle(), _mean_handle()]
        legend_ncol = len(legend_handles)
        legend_bottom = 0.26
        legend_anchor = 0.04

    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    ax.set_ylabel(y_axis_label(cfg.metric_type))
    if cfg.title:
        ax.set_title(cfg.title)
    _set_y_axis(
        ax,
        plot_df[plot_column],
        log_y=cfg.log_y,
        errors=plot_df.get(REPLICATE_SD_COLUMN),
    )
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    fig.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom)
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, legend_anchor),
        ncol=legend_ncol,
        fontsize=11,
        frameon=False,
        handlelength=1.5,
    )

    for fmt in ("png", "pdf"):
        fig.savefig(output_path_stem.with_suffix(f".{fmt}"))
    plt.close(fig)


def write_outputs(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(
        cfg.metric_type,
        cfg.view,
        cfg.prediction_method_views,
        average_test_sets=cfg.average_test_sets,
        average_prediction_methods=cfg.average_prediction_methods,
    )
    values = read_raw_values(cfg)
    if cfg.aggregate_replicates:
        values = aggregate_replicate_values(
            values, cfg.metric_type, geometric=geometric_replicates(cfg)
        )
    plot_values = average_plot_values(
        values,
        metric=cfg.metric,
        metric_type=cfg.metric_type,
        average_test_sets=cfg.average_test_sets,
        average_prediction_methods=cfg.average_prediction_methods,
    )
    summary = summarize_by_model(
        plot_values, cfg.metric_type, cfg.prediction_method_views
    )

    values.to_csv(cfg.output_dir / f"{stem}_raw_values.csv", index=False)
    if cfg.average_test_sets or cfg.average_prediction_methods:
        plot_values.to_csv(cfg.output_dir / f"{stem}_averaged_values.csv", index=False)
        plot_averaged_swapped_fill(plot_values, cfg, cfg.output_dir / stem)
    else:
        plot_boxplot(values, cfg, cfg.output_dir / stem)
    summary.to_csv(cfg.output_dir / f"{stem}_model_summary.csv", index=False)

    metric_label = mmd_table.metric_type_config(cfg.metric_type)["label"]
    logger.info(f"Wrote all-model {metric_label} boxplot and CSVs to {cfg.output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    try:
        write_outputs(parse_args(argv))
    except (FileNotFoundError, ValueError) as error:
        msg = f"error: {error}"
        raise SystemExit(msg) from error


if __name__ == "__main__":
    main()
