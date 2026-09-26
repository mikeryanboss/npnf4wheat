"""Paper dataset-scaling figure from raw Sig-MMD summaries."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.artist import Artist

from npnf.data.configs.datasets.synthetic_test_sets import split_named
from npnf.scripts.paper import mmd_table
from npnf.scripts.paper.sig_mmd_point_style import (
    POINT_STYLE_CHOICES,
    add_swapped_fill_legend,
    conditioning_label as default_conditioning_label,
    draw_centered_summary_bars,
    draw_swapped_fill_points,
    overlay_testset_context_points,
    prediction_method_label,
    raise_boxplot_lines_above_points,
    split_label,
    swapped_fill_orders,
    testset_context_handles,
)
from npnf.scripts.paper.style import COLOR_LIST, set_log_y_axis, setup_style
from npnf.utils import add_panel_labels

DEFAULT_MODELS = ("ANP", "LNP")
DEFAULT_DATASET_SIZES = ("1k", "8k", "64k", "512k")
DEFAULT_TRAINING_LENGTHS = ("100k", "1m", "3m")
DEFAULT_CONDITIONINGS = ("noenv_nogeno", "noenv_geno", "env_nogeno", "env_geno")
DEFAULT_PREDICTION_METHODS = ("no_context", "max_height", "random_context")

PLOT_COLUMN = "sig_mmd_x1000"
POINT_COLOR = "#222222"
POINT_ALPHA = 0.30
POINT_SIZE = 2.0
BOX_WIDTH = 0.72
GROUP_GAP_CATEGORY_PREFIX = "__training_gap_"
DATASET_LABEL_Y = -0.070
TRAINING_LABEL_Y = -0.165
X_AXIS_ROW_LABEL_X = -0.110
GROUP_DIVIDER_COLOR = "#d9d9d9"


@dataclass(frozen=True)
class Config:
    results_base: Path
    output_dir: Path
    models: tuple[str, ...]
    test_set: str
    splits: tuple[str, ...]
    dataset_sizes: tuple[str, ...]
    dataset_size_labels: tuple[str, ...]
    training_lengths: tuple[str, ...]
    set_mode: str
    conditionings: tuple[str, ...]
    conditioning_labels: tuple[str, ...]
    prediction_methods: tuple[str, ...]
    prediction_method_labels: tuple[str, ...]
    metric: str
    view: str
    prediction_method_views: dict[str, str]
    title: str
    point_style: str
    model_rows: tuple[str, ...] | None
    row_labels: tuple[str, ...] | None
    log_y: bool = False


@dataclass(frozen=True)
class MetricCsvLookup:
    split: str
    model: str
    dataset_size: str
    training_length: str
    conditioning: str
    prediction_method: str
    view: str
    run_name: str
    run_dir: Path
    pattern: str
    matches: tuple[Path, ...]


@dataclass(frozen=True)
class TrainingFirstLayout:
    plot_order: tuple[str, ...]
    dataset_tick_positions: tuple[float, ...]
    dataset_tick_labels: tuple[str, ...]
    training_label_positions: tuple[tuple[float, str], ...]
    divider_positions: tuple[float, ...]


def _tokens(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values)


def _optional_tokens(values: Sequence[str] | None) -> tuple[str, ...] | None:
    return None if values is None else _tokens(values)


def _resolve_optional_labels(
    *,
    values: Sequence[str],
    labels: Sequence[str] | None,
    default_labels: Sequence[str],
    mismatch_message: str,
) -> tuple[str, ...]:
    if labels is None:
        return _tokens(default_labels)
    resolved_labels = _tokens(labels)
    if len(values) != len(resolved_labels):
        raise ValueError(mismatch_message)
    return resolved_labels


def _duplicate_tokens(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _label_by_value(
    values: pd.DataFrame, *, value_col: str, label_col: str
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for value, label in values[[value_col, label_col]].itertuples(
        index=False, name=None
    ):
        labels.setdefault(str(value), str(label))
    return labels


def _normalize_set_mode(value: str) -> str:
    return value.strip().replace("_", "-")


def _dataset_size_value(token: str) -> int:
    token = token.strip().lower()
    return int(token[:-1]) * 1024 if token.endswith("k") else int(token)


def _dataset_size_label(token: str) -> str:
    size = _dataset_size_value(token)
    return f"{size // 1024}k" if size >= 1024 and size % 1024 == 0 else str(size)


def _training_steps(token: str) -> int:
    token = token.strip().lower()
    if token.endswith("m"):
        return int(token[:-1]) * 1_000_000
    if token.endswith("k"):
        return int(token[:-1]) * 1_000
    return int(token)


def _training_label(token: str) -> str:
    steps = _training_steps(token)
    if steps % 1_000_000 == 0:
        return f"{steps // 1_000_000}M"
    if steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    return str(steps)


def _training_run_token(token: str) -> str:
    return _training_label(token).lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the raw Sig-MMD dataset-scaling paper figure."
    )
    parser.add_argument(
        "--results-base",
        type=Path,
        required=True,
        help="Base results directory containing split dataloader result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/dataset_scaling"),
        help="Directory for figure and CSV outputs.",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--model-rows",
        nargs="*",
        default=None,
        help=(
            "Optional model names to plot as separate stacked rows instead of "
            "pooling all --models values, e.g. --model-rows LNP ANP."
        ),
    )
    parser.add_argument(
        "--row-labels",
        nargs="*",
        default=None,
        help="Optional row labels for --model-rows; defaults to the model names.",
    )
    mmd_table.add_test_set_arguments(parser)
    parser.add_argument("--dataset-sizes", nargs="+", default=DEFAULT_DATASET_SIZES)
    parser.add_argument(
        "--dataset-size-labels",
        nargs="+",
        default=None,
        help="Dataset-size display labels. Defaults to normalized --dataset-sizes.",
    )
    parser.add_argument(
        "--training-lengths", nargs="+", default=DEFAULT_TRAINING_LENGTHS
    )
    parser.add_argument("--set-mode", default="nested-noprior")
    parser.add_argument("--conditionings", nargs="+", default=DEFAULT_CONDITIONINGS)
    parser.add_argument(
        "--conditioning-labels",
        nargs="+",
        default=None,
        help="Conditioning display labels. Defaults to known labels, else values.",
    )
    parser.add_argument(
        "--prediction-methods", nargs="+", default=DEFAULT_PREDICTION_METHODS
    )
    parser.add_argument(
        "--prediction-method-labels",
        nargs="+",
        default=None,
        help=(
            "Prediction-method display labels. Defaults to known readable labels, "
            "else values with underscores replaced by spaces."
        ),
    )
    parser.add_argument("--metric", default="mean")
    parser.add_argument(
        "--view",
        default=mmd_table.DEFAULT_VIEW,
        choices=mmd_table.VIEW_CHOICES,
        help=(
            "Sig-MMD view to read: condition or "
            "context_matched_blocked. (default: condition)"
        ),
    )
    parser.add_argument(
        "--prediction-method-views",
        nargs="*",
        default=None,
        metavar="METHOD=VIEW",
        help=(
            "Per-prediction-method Sig-MMD view overrides. Each token must be "
            "METHOD=VIEW; unlisted methods use --view."
        ),
    )
    parser.add_argument(
        "--title",
        default="Dataset scaling",
        help="Figure title. Pass an empty string to omit the title.",
    )
    parser.add_argument(
        "--point-style",
        choices=POINT_STYLE_CHOICES,
        default="gray",
        help="Point overlay style: gray preserves the default; testset-context "
        "colors points by test set and shapes them by prediction method; "
        "swapped-fill colors points by prediction method, shapes them by test "
        "set, and fills them by conditioning.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Logarithmic y-axis with 1-2-5 tick labels. Every plotted value must "
        "be positive.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    models = _tokens(args.models)
    model_rows = _optional_tokens(args.model_rows)
    row_labels = _optional_tokens(args.row_labels)
    splits = tuple(
        mmd_table.resolve_splits(args, roles=("seen", "env", "geno", "unseen"))
    )

    dataset_sizes = tuple(size.lower() for size in _tokens(args.dataset_sizes))
    dataset_size_labels = _resolve_optional_labels(
        values=dataset_sizes,
        labels=args.dataset_size_labels,
        default_labels=tuple(_dataset_size_label(size) for size in dataset_sizes),
        mismatch_message=(
            "--dataset-sizes and --dataset-size-labels must have the same length"
        ),
    )

    if model_rows is None:
        if row_labels is not None:
            msg = "--row-labels requires --model-rows"
            raise ValueError(msg)
    else:
        if not model_rows or any(model == "" for model in model_rows):
            msg = "--model-rows requires at least one non-empty model name"
            raise ValueError(msg)
        duplicates = _duplicate_tokens(model_rows)
        if duplicates:
            msg = f"--model-rows contains duplicate model names: {duplicates}"
            raise ValueError(msg)
        unknown_models = [model for model in model_rows if model not in set(models)]
        if unknown_models:
            msg = (
                "--model-rows values must also be included in --models; "
                f"missing from --models: {unknown_models}"
            )
            raise ValueError(msg)
        if row_labels is not None:
            if any(label == "" for label in row_labels):
                msg = "--row-labels requires non-empty labels"
                raise ValueError(msg)
            if len(row_labels) != len(model_rows):
                msg = (
                    "--row-labels must contain exactly one label per "
                    f"--model-rows value: row_labels={len(row_labels)}, "
                    f"model_rows={len(model_rows)}"
                )
                raise ValueError(msg)

    conditionings = _tokens(args.conditionings)
    conditioning_labels = _resolve_optional_labels(
        values=conditionings,
        labels=args.conditioning_labels,
        default_labels=tuple(
            default_conditioning_label(conditioning) for conditioning in conditionings
        ),
        mismatch_message=(
            "--conditionings and --conditioning-labels must have the same length"
        ),
    )
    prediction_methods = _tokens(args.prediction_methods)
    prediction_method_labels = _resolve_optional_labels(
        values=prediction_methods,
        labels=args.prediction_method_labels,
        default_labels=tuple(
            prediction_method_label(method) for method in prediction_methods
        ),
        mismatch_message=(
            "--prediction-methods and --prediction-method-labels must match"
        ),
    )
    view = mmd_table.validate_metric_view("sig_mmd", args.view)
    prediction_method_views = mmd_table.resolve_prediction_method_views(
        prediction_methods,
        metric_type="sig_mmd",
        default_view=view,
        raw_overrides=args.prediction_method_views,
    )

    return Config(
        results_base=args.results_base.expanduser(),
        output_dir=args.output_dir.expanduser(),
        models=models,
        test_set=args.test_set,
        splits=splits,
        dataset_sizes=dataset_sizes,
        dataset_size_labels=dataset_size_labels,
        training_lengths=tuple(
            length.lower() for length in _tokens(args.training_lengths)
        ),
        set_mode=_normalize_set_mode(args.set_mode),
        conditionings=conditionings,
        conditioning_labels=conditioning_labels,
        prediction_methods=prediction_methods,
        prediction_method_labels=prediction_method_labels,
        metric=args.metric,
        view=view,
        prediction_method_views=prediction_method_views,
        title=args.title,
        point_style=args.point_style,
        model_rows=model_rows,
        row_labels=row_labels,
        log_y=args.log_y,
    )


def _run_name(
    model: str, dataset_size: str, training_length: str, set_mode: str
) -> str:
    return (
        f"{model}-{_dataset_size_label(dataset_size)}"
        f"-training{_training_run_token(training_length)}"
        f"_set_mode_{set_mode.replace('-', '_')}"
    )


def _metric_csv_lookup(
    cfg: Config,
    *,
    split: str,
    model: str,
    dataset_size: str,
    training_length: str,
    conditioning: str,
    prediction_method: str,
) -> MetricCsvLookup:
    run_name = _run_name(model, dataset_size, training_length, cfg.set_mode)
    test_split = split_named(split, cfg.test_set)
    run_dir = cfg.results_base / test_split.dataloader_name / run_name
    view = mmd_table.prediction_method_view(
        cfg.prediction_method_views, prediction_method
    )
    metric_dir = mmd_table.metric_dir_name("sig_mmd", view)
    summary_file = mmd_table.metric_type_config("sig_mmd")["summary_file"]
    pattern = (
        f"*/checkpoint-*/{test_split.split_key}/{conditioning}/{prediction_method}/"
        f"{metric_dir}/{summary_file}"
    )
    return MetricCsvLookup(
        split=split,
        model=model,
        dataset_size=dataset_size,
        training_length=training_length,
        conditioning=conditioning,
        prediction_method=prediction_method,
        view=view,
        run_name=run_name,
        run_dir=run_dir,
        pattern=pattern,
        matches=tuple(sorted(run_dir.glob(pattern))),
    )


def _format_metric_csv_failure(lookup: MetricCsvLookup) -> str:
    lines = [
        (
            f"  split={lookup.split}, model={lookup.model}, "
            f"dataset_size={lookup.dataset_size}, "
            f"training_length={lookup.training_length}, "
            f"conditioning={lookup.conditioning}, "
            f"prediction_method={lookup.prediction_method}, "
            f"view={lookup.view}, "
            f"run_name={lookup.run_name}, found={len(lookup.matches)}"
        ),
        f"    expected: {lookup.run_dir / lookup.pattern}",
    ]
    if lookup.matches:
        lines.append("    matches:")
        lines.extend(f"      {path}" for path in lookup.matches)
    return "\n".join(lines)


def _raise_metric_csv_failures(failures: Sequence[MetricCsvLookup]) -> None:
    detail = "\n".join(_format_metric_csv_failure(failure) for failure in failures)
    msg = (
        "Expected one metric CSV per requested dataset-scaling cell, "
        f"but {len(failures)} cell(s) had missing or ambiguous matches:\n{detail}"
    )
    raise FileNotFoundError(msg)


def _metric_csv_path(
    cfg: Config,
    *,
    split: str,
    model: str,
    dataset_size: str,
    training_length: str,
    conditioning: str,
    prediction_method: str,
) -> Path:
    lookup = _metric_csv_lookup(
        cfg,
        split=split,
        model=model,
        dataset_size=dataset_size,
        training_length=training_length,
        conditioning=conditioning,
        prediction_method=prediction_method,
    )
    if len(lookup.matches) != 1:
        _raise_metric_csv_failures([lookup])
    return lookup.matches[0]


def _read_metric(csv_path: Path, metric: str) -> float:
    return float(pd.read_csv(csv_path).at[0, metric])


def collect_metrics(cfg: Config) -> pd.DataFrame:
    conditioning_label_by_conditioning = dict(
        zip(cfg.conditionings, cfg.conditioning_labels, strict=True)
    )
    dataset_size_label_by_size = dict(
        zip(cfg.dataset_sizes, cfg.dataset_size_labels, strict=True)
    )
    prediction_label_by_method = dict(
        zip(cfg.prediction_methods, cfg.prediction_method_labels, strict=True)
    )
    lookups: list[MetricCsvLookup] = []
    failures: list[MetricCsvLookup] = []
    for split, model, size, training, conditioning, method in product(
        cfg.splits,
        cfg.models,
        cfg.dataset_sizes,
        cfg.training_lengths,
        cfg.conditionings,
        cfg.prediction_methods,
    ):
        lookup = _metric_csv_lookup(
            cfg,
            split=split,
            model=model,
            dataset_size=size,
            training_length=training,
            conditioning=conditioning,
            prediction_method=method,
        )
        if len(lookup.matches) == 1:
            lookups.append(lookup)
        else:
            failures.append(lookup)

    if failures:
        _raise_metric_csv_failures(failures)

    rows = [
        {
            "split": lookup.split,
            "model": lookup.model,
            "dataset_size": _dataset_size_value(lookup.dataset_size),
            "dataset_size_label": dataset_size_label_by_size[lookup.dataset_size],
            "training_steps": _training_steps(lookup.training_length),
            "training_length_label": _training_label(lookup.training_length),
            "set_mode": cfg.set_mode,
            "conditioning": lookup.conditioning,
            "conditioning_label": conditioning_label_by_conditioning[
                lookup.conditioning
            ],
            "prediction_method": lookup.prediction_method,
            "prediction_method_label": prediction_label_by_method[
                lookup.prediction_method
            ],
            "view": lookup.view,
            cfg.metric: _read_metric(lookup.matches[0], cfg.metric),
            "summary_path": str(lookup.matches[0]),
        }
        for lookup in lookups
    ]
    metrics = pd.DataFrame(rows)
    metrics[PLOT_COLUMN] = metrics[cfg.metric] * 1000.0
    return metrics


def _prediction_method_views_from_metrics(metrics: pd.DataFrame) -> str:
    views_by_method: dict[str, list[str]] = {}
    for method, view in metrics[["prediction_method", "view"]].itertuples(
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


def summarize_metrics(
    metrics: pd.DataFrame, prediction_method_views: dict[str, str] | None = None
) -> pd.DataFrame:
    summary = metrics.groupby(
        [
            "dataset_size",
            "dataset_size_label",
            "training_steps",
            "training_length_label",
        ],
        as_index=False,
        sort=True,
    ).agg(
        n=(PLOT_COLUMN, "size"),
        mean=(PLOT_COLUMN, "mean"),
        median=(PLOT_COLUMN, "median"),
        q05=(PLOT_COLUMN, lambda values: values.quantile(0.05)),
        q25=(PLOT_COLUMN, lambda values: values.quantile(0.25)),
        q75=(PLOT_COLUMN, lambda values: values.quantile(0.75)),
        q95=(PLOT_COLUMN, lambda values: values.quantile(0.95)),
        min=(PLOT_COLUMN, "min"),
        max=(PLOT_COLUMN, "max"),
    )
    summary["prediction_method_views"] = (
        _prediction_method_views_from_metrics(metrics)
        if prediction_method_views is None
        else mmd_table.format_prediction_method_views(prediction_method_views)
    )
    return summary


def _dataset_palette(dataset_order: Sequence[str]) -> dict[str, object]:
    return {
        label: COLOR_LIST[index % len(COLOR_LIST)]
        for index, label in enumerate(dataset_order)
    }


def _plot_category(training_label: str, dataset_label: str) -> str:
    return f"{training_label}|{dataset_label}"


def _training_gap_category(index: int) -> str:
    return f"{GROUP_GAP_CATEGORY_PREFIX}{index}"


def _training_first_layout(
    dataset_order: Sequence[str], training_order: Sequence[str]
) -> TrainingFirstLayout:
    plot_order: list[str] = []
    dataset_tick_positions: list[float] = []
    dataset_tick_labels: list[str] = []
    training_label_positions: list[tuple[float, str]] = []
    divider_positions: list[float] = []

    for training_index, training in enumerate(training_order):
        if training_index > 0:
            divider_positions.append(float(len(plot_order)))
            plot_order.append(_training_gap_category(training_index))

        group_positions: list[float] = []
        for dataset in dataset_order:
            category = _plot_category(training, dataset)
            position = float(len(plot_order))
            plot_order.append(category)
            dataset_tick_positions.append(position)
            dataset_tick_labels.append(dataset)
            group_positions.append(position)

        if group_positions:
            group_center = (group_positions[0] + group_positions[-1]) / 2
            training_label_positions.append((group_center, training))

    return TrainingFirstLayout(
        plot_order=tuple(plot_order),
        dataset_tick_positions=tuple(dataset_tick_positions),
        dataset_tick_labels=tuple(dataset_tick_labels),
        training_label_positions=tuple(training_label_positions),
        divider_positions=tuple(divider_positions),
    )


def _add_training_first_category(plot_df: pd.DataFrame) -> pd.DataFrame:
    category_df = plot_df.copy()
    category_df["plot_category"] = [
        _plot_category(training, dataset)
        for training, dataset in zip(
            category_df["training_length_label"].astype(str),
            category_df["dataset_size_label"].astype(str),
            strict=True,
        )
    ]
    return category_df


def _training_first_fig_width(layout: TrainingFirstLayout) -> float:
    return max(7.2, len(layout.plot_order) * 0.68)


def _apply_training_first_xaxis(
    ax: plt.Axes, layout: TrainingFirstLayout, *, show_xticklabels: bool
) -> None:
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    ax.set_xticks(layout.dataset_tick_positions)
    ax.set_xlim(-0.95, len(layout.plot_order) - 0.5)

    for divider_position in layout.divider_positions:
        ax.axvline(divider_position, color=GROUP_DIVIDER_COLOR, linewidth=0.9, zorder=0)

    ax.tick_params(axis="x", labelbottom=False, length=0)
    if not show_xticklabels:
        return

    for position, label in zip(
        layout.dataset_tick_positions, layout.dataset_tick_labels, strict=True
    ):
        ax.text(
            position,
            DATASET_LABEL_Y,
            label,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            clip_on=False,
        )

    for position, label in layout.training_label_positions:
        ax.text(
            position,
            TRAINING_LABEL_Y,
            label,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            clip_on=False,
        )

    ax.text(
        X_AXIS_ROW_LABEL_X,
        DATASET_LABEL_Y,
        "dataset size",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        X_AXIS_ROW_LABEL_X,
        TRAINING_LABEL_Y,
        "training length",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )


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


def _raw_ylim(values: pd.Series) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    pad = max((high - low) * 0.08, 0.4)
    return max(0.0, low - pad), high + pad


def _set_y_axis(ax: plt.Axes, values: pd.Series, *, log_y: bool) -> None:
    if log_y:
        set_log_y_axis(ax, values)
    else:
        ax.set_ylim(*_raw_ylim(values))


def _save_figure_formats(fig: plt.Figure, output_path: Path) -> None:
    for fmt in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{fmt}"))


def _add_axes_centered_supylabel(
    fig: plt.Figure, axes: Iterable[plt.Axes], label: str
) -> None:
    positions = [ax.get_position() for ax in axes]
    bottom = min(position.y0 for position in positions)
    top = max(position.y1 for position in positions)
    fig.supylabel(label, x=0.060, y=(bottom + top) / 2)


def _validate_model_rows(metrics: pd.DataFrame, cfg: Config) -> None:
    if cfg.model_rows is None:
        return

    present_models = set(metrics["model"].astype(str))
    missing_models = [model for model in cfg.model_rows if model not in present_models]
    if missing_models:
        msg = (
            "--model-rows values must be present in collected metrics; "
            f"missing models: {missing_models}"
        )
        raise ValueError(msg)


def _model_row_labels(cfg: Config) -> tuple[str, ...]:
    if cfg.model_rows is None:
        return ()
    return cfg.row_labels if cfg.row_labels is not None else cfg.model_rows


def _plot_training_boxplot_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    cfg: Config,
    dataset_order: Sequence[str],
    layout: TrainingFirstLayout,
    *,
    show_title: bool,
    show_xticklabels: bool,
) -> tuple[list[Artist], list[str], int, float, float]:
    sns.boxplot(
        data=plot_df,
        x="plot_category",
        y=PLOT_COLUMN,
        order=layout.plot_order,
        hue="dataset_size_label",
        hue_order=dataset_order,
        dodge=False,
        palette=_dataset_palette(dataset_order),
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
    raw_box_handles, raw_box_labels = ax.get_legend_handles_labels()
    box_handles_by_label = {
        label: handle
        for handle, label in zip(raw_box_handles, raw_box_labels, strict=False)
    }
    box_labels = [label for label in dataset_order if label in box_handles_by_label]
    box_handles = [box_handles_by_label[label] for label in box_labels]

    if cfg.point_style == "gray":
        sns.stripplot(
            data=plot_df,
            x="plot_category",
            y=PLOT_COLUMN,
            order=layout.plot_order,
            jitter=0.18,
            color=POINT_COLOR,
            size=POINT_SIZE,
            alpha=POINT_ALPHA,
            legend=False,
            ax=ax,
        )
        legend_handles = [*box_handles, _mean_handle()]
        legend_labels = [*box_labels, "Mean"]
        legend_ncol = len(legend_handles)
        legend_bottom = 0.34
        legend_anchor = 0.16
    else:
        point_df = plot_df.copy()
        prediction_label_by_method = dict(
            zip(cfg.prediction_methods, cfg.prediction_method_labels, strict=True)
        )
        point_df["split_label"] = point_df["split"].map(split_label)
        point_df["prediction_method_label"] = point_df["prediction_method"].map(
            prediction_label_by_method
        )
        point_df["box_hue"] = "box"
        split_order = [split_label(split) for split in cfg.splits]
        method_order = list(cfg.prediction_method_labels)
        overlay_testset_context_points(
            ax,
            point_df,
            x_col="plot_category",
            hue_col="box_hue",
            y_col=PLOT_COLUMN,
            x_order=layout.plot_order,
            hue_order=("box",),
            split_order=split_order,
            method_order=method_order,
            box_width=BOX_WIDTH,
            rng_seed=285,
        )
        raise_boxplot_lines_above_points(ax)
        style_handles = testset_context_handles(split_order, method_order)
        legend_handles = [*box_handles, *style_handles, _mean_handle()]
        legend_labels = [str(handle.get_label()) for handle in legend_handles]
        legend_ncol = 4
        legend_bottom = 0.42
        legend_anchor = 0.20

    _apply_training_first_xaxis(ax, layout, show_xticklabels=show_xticklabels)
    if show_title and cfg.title:
        ax.set_title(cfg.title)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    return legend_handles, legend_labels, legend_ncol, legend_bottom, legend_anchor


def _plot_single_training_boxplot(
    plot_df: pd.DataFrame,
    cfg: Config,
    output_path: Path,
    dataset_order: Sequence[str],
    layout: TrainingFirstLayout,
) -> None:
    fig, ax = plt.subplots(figsize=(_training_first_fig_width(layout), 4.2))
    legend_handles, legend_labels, legend_ncol, legend_bottom, legend_anchor = (
        _plot_training_boxplot_panel(
            ax,
            plot_df,
            cfg,
            dataset_order,
            layout,
            show_title=True,
            show_xticklabels=True,
        )
    )
    ax.set_ylabel("Sig-MMD² × 10³")
    _set_y_axis(ax, plot_df[PLOT_COLUMN], log_y=cfg.log_y)

    fig.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_anchor),
        ncol=legend_ncol,
        fontsize=8,
        frameon=False,
        handlelength=1.5,
    )

    _save_figure_formats(fig, output_path)
    plt.close(fig)


def _plot_model_row_training_boxplots(
    plot_df: pd.DataFrame,
    cfg: Config,
    output_path: Path,
    dataset_order: Sequence[str],
    layout: TrainingFirstLayout,
) -> None:
    assert cfg.model_rows is not None
    nrows = len(cfg.model_rows)
    row_labels = _model_row_labels(cfg)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(_training_first_fig_width(layout), max(4.2, 3.1 * nrows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    legend_handles: list[Artist] = []
    legend_labels: list[str] = []
    legend_ncol = 1

    for row_index, (model, row_label) in enumerate(
        zip(cfg.model_rows, row_labels, strict=True)
    ):
        ax = axes[row_index, 0]
        row_df = plot_df[plot_df["model"].astype(str) == model].copy()
        handles, labels, ncol, _, _ = _plot_training_boxplot_panel(
            ax,
            row_df,
            cfg,
            dataset_order,
            layout,
            show_title=row_index == 0,
            show_xticklabels=row_index == nrows - 1,
        )
        if row_index == 0:
            legend_handles = handles
            legend_labels = labels
            legend_ncol = ncol
        ax.set_ylabel(row_label, fontweight="bold")
        _set_y_axis(ax, plot_df[PLOT_COLUMN], log_y=cfg.log_y)

    legend_bottom = 0.24 if cfg.point_style == "gray" else 0.32
    legend_anchor = 0.11 if cfg.point_style == "gray" else 0.16
    fig.tight_layout(rect=(0.07, legend_bottom, 1.0, 1.0))
    _add_axes_centered_supylabel(fig, axes[:, 0], "Sig-MMD² × 10³")
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_anchor),
        ncol=legend_ncol,
        fontsize=8,
        frameon=False,
        handlelength=1.5,
    )

    _save_figure_formats(fig, output_path)
    plt.close(fig)


def _swapped_fill_plot_df(plot_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    swapped_df = plot_df.copy()
    prediction_label_by_method = dict(
        zip(cfg.prediction_methods, cfg.prediction_method_labels, strict=True)
    )
    swapped_df["split_label"] = swapped_df["split"].map(split_label)
    swapped_df["prediction_method_label"] = swapped_df["prediction_method"].map(
        prediction_label_by_method
    )
    if "conditioning_label" not in swapped_df:
        swapped_df["conditioning_label"] = swapped_df["conditioning"].map(
            default_conditioning_label
        )
    return _add_training_first_category(swapped_df)


def _draw_swapped_fill_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    cfg: Config,
    layout: TrainingFirstLayout,
    split_order: Sequence[str],
    method_order: Sequence[str],
    condition_order: Sequence[str],
    *,
    show_title: bool,
    show_xticklabels: bool,
) -> None:
    draw_swapped_fill_points(
        ax,
        plot_df,
        x_col="plot_category",
        y_col=PLOT_COLUMN,
        x_order=layout.plot_order,
        split_order=split_order,
        method_order=method_order,
        condition_order=condition_order,
        condition_col="conditioning",
        rng_seed=385,
    )
    draw_centered_summary_bars(
        ax, plot_df, x_col="plot_category", y_col=PLOT_COLUMN, x_order=layout.plot_order
    )
    _apply_training_first_xaxis(ax, layout, show_xticklabels=show_xticklabels)
    if show_title and cfg.title:
        ax.set_title(cfg.title)


def _swapped_fill_style_orders(
    plot_df: pd.DataFrame, cfg: Config
) -> tuple[list[str], list[str], list[str]]:
    return swapped_fill_orders(
        split_values=[split_label(split) for split in cfg.splits],
        method_values=list(cfg.prediction_method_labels),
        condition_values=list(dict.fromkeys(plot_df["conditioning"].astype(str))),
    )


def _plot_single_swapped_fill(
    plot_df: pd.DataFrame,
    cfg: Config,
    output_path: Path,
    layout: TrainingFirstLayout,
    method_order: Sequence[str],
    split_order: Sequence[str],
    condition_order: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(_training_first_fig_width(layout), 4.2))
    _draw_swapped_fill_panel(
        ax,
        plot_df,
        cfg,
        layout,
        split_order,
        method_order,
        condition_order,
        show_title=True,
        show_xticklabels=True,
    )
    ax.set_ylabel("Sig-MMD² × 10³")
    _set_y_axis(ax, plot_df[PLOT_COLUMN], log_y=cfg.log_y)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.39)
    add_swapped_fill_legend(
        fig,
        method_order=method_order,
        split_order=split_order,
        condition_order=condition_order,
        condition_labels=_label_by_value(
            plot_df, value_col="conditioning", label_col="conditioning_label"
        ),
        anchor_y=0.25,
    )
    _save_figure_formats(fig, output_path)
    plt.close(fig)


def _plot_model_row_swapped_fill(
    plot_df: pd.DataFrame,
    cfg: Config,
    output_path: Path,
    layout: TrainingFirstLayout,
    method_order: Sequence[str],
    split_order: Sequence[str],
    condition_order: Sequence[str],
) -> None:
    assert cfg.model_rows is not None
    nrows = len(cfg.model_rows)
    row_labels = _model_row_labels(cfg)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(_training_first_fig_width(layout), max(4.2, 3.1 * nrows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, (model, row_label) in enumerate(
        zip(cfg.model_rows, row_labels, strict=True)
    ):
        ax = axes[row_index, 0]
        row_df = plot_df[plot_df["model"].astype(str) == model].copy()
        _draw_swapped_fill_panel(
            ax,
            row_df,
            cfg,
            layout,
            split_order,
            method_order,
            condition_order,
            show_title=row_index == 0,
            show_xticklabels=row_index == nrows - 1,
        )
        ax.set_ylabel(row_label, fontweight="bold")
        _set_y_axis(ax, plot_df[PLOT_COLUMN], log_y=cfg.log_y)

    # Put the labels above the panel, clear of the top grid line.
    add_panel_labels(axes.flat, offset_points=(6, 12))
    fig.tight_layout(rect=(0.07, 0.30, 1.0, 1.0))
    _add_axes_centered_supylabel(fig, axes[:, 0], "Sig-MMD² × 10³")
    add_swapped_fill_legend(
        fig,
        method_order=method_order,
        split_order=split_order,
        condition_order=condition_order,
        condition_labels=_label_by_value(
            plot_df, value_col="conditioning", label_col="conditioning_label"
        ),
        anchor_y=0.31,
    )
    _save_figure_formats(fig, output_path)
    plt.close(fig)


def plot_boxplot(metrics: pd.DataFrame, cfg: Config, output_path: Path) -> None:
    setup_style()
    dataset_order = list(cfg.dataset_size_labels)
    training_order = [_training_label(training) for training in cfg.training_lengths]

    plot_df = metrics.copy()
    plot_df["dataset_size_label"] = pd.Categorical(
        plot_df["dataset_size_label"], categories=dataset_order, ordered=True
    )
    plot_df["training_length_label"] = pd.Categorical(
        plot_df["training_length_label"], categories=training_order, ordered=True
    )
    plot_df = _add_training_first_category(plot_df)
    layout = _training_first_layout(dataset_order, training_order)
    _validate_model_rows(plot_df, cfg)

    if cfg.point_style == "swapped-fill":
        swapped_df = _swapped_fill_plot_df(plot_df, cfg)
        method_order, split_order, condition_order = _swapped_fill_style_orders(
            swapped_df, cfg
        )
        if cfg.model_rows is None:
            _plot_single_swapped_fill(
                swapped_df,
                cfg,
                output_path,
                layout,
                method_order,
                split_order,
                condition_order,
            )
        else:
            _plot_model_row_swapped_fill(
                swapped_df,
                cfg,
                output_path,
                layout,
                method_order,
                split_order,
                condition_order,
            )
        return

    if cfg.model_rows is None:
        _plot_single_training_boxplot(plot_df, cfg, output_path, dataset_order, layout)
    else:
        _plot_model_row_training_boxplots(
            plot_df, cfg, output_path, dataset_order, layout
        )


def write_outputs(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics(cfg)
    summary = summarize_metrics(metrics, cfg.prediction_method_views)

    metrics.to_csv(cfg.output_dir / "dataset_scaling_metrics.csv", index=False)
    summary.to_csv(cfg.output_dir / "dataset_scaling_summary.csv", index=False)
    plot_boxplot(metrics, cfg, cfg.output_dir / "dataset_scaling")

    logger.info(f"Wrote dataset scaling figure and CSVs to {cfg.output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    write_outputs(parse_args(argv))


if __name__ == "__main__":
    main()
