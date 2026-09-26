"""Paper context-training figure from raw Sig-MMD summaries."""

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

from npnf.scripts.paper import mmd_table
from npnf.scripts.paper.sig_mmd_point_style import (
    POINT_STYLE_CHOICES,
    add_swapped_fill_legend,
    conditioning_label as default_conditioning_label,
    draw_centered_summary_bars,
    draw_swapped_fill_points,
    overlay_testset_context_points,
    prediction_method_label as default_prediction_method_label,
    raise_boxplot_lines_above_points,
    swapped_fill_orders,
    testset_context_handles,
)
from npnf.scripts.paper.style import COLOR_LIST, COLORS, set_log_y_axis, setup_style
from npnf.scripts.utils.paths import checkpoint_step
from npnf.utils import add_panel_labels

DEFAULT_MODELS = (
    "LNP-512k-training3m_set_mode_nested_all",
    "LNP-512k-training3m_set_mode_nested_noprior",
    "LNP-512k-training3m_set_mode_nested_nested",
    "LNP-512k-training3m_set_mode_nested_noprior_holdout",
    "ANP-512k-training3m_set_mode_nested_all",
    "ANP-512k-training3m_set_mode_nested_noprior",
    "ANP-512k-training3m_set_mode_nested_nested",
    "ANP-512k-training3m_set_mode_nested_noprior_holdout",
)
DEFAULT_MODEL_NAMES = (
    "LNP-NP",
    "LNP-NP-noprior",
    "LNP-NP-nested",
    "LNP-NP-noprior_holdout",
    "ANP-NP",
    "ANP-NP-noprior",
    "ANP-NP-nested",
    "ANP-NP-noprior_holdout",
)
DEFAULT_CONDITIONINGS = ("noenv_nogeno", "env_geno")
DEFAULT_CONDITIONING_LABELS = ("P", "E&G")
DEFAULT_PREDICTION_METHODS = ("no_context", "random_context", "max_height")
DEFAULT_PREDICTION_METHOD_LABELS = ("no context", "random context", "max context")
RECIPE_LABELS = {
    "NP": "NP",
    "noprior": "no prior",
    "nested": "nested",
    "noprior_holdout": "no prior holdout",
}
ARCHITECTURE_PALETTE = {"LNP": COLORS["blue"], "ANP": COLORS["orange"]}
PLOT_COLUMN = "sig_mmd_x1000"
POINT_COLOR = "#222222"
POINT_ALPHA = 0.30
POINT_SIZE = 2.0
BOX_WIDTH = 0.72


@dataclass(frozen=True)
class Config:
    results_base: Path
    output_dir: Path
    models: tuple[str, ...]
    model_names: tuple[str, ...]
    recipe_labels: tuple[str, ...] | None
    splits: tuple[str, ...]
    conditionings: tuple[str, ...]
    conditioning_labels: tuple[str, ...]
    prediction_methods: tuple[str, ...]
    prediction_method_labels: tuple[str, ...]
    checkpoint: str
    test_set: str
    metric: str
    view: str
    prediction_method_views: dict[str, str]
    title: str
    point_style: str
    architecture_rows: tuple[str, ...] | None
    row_labels: tuple[str, ...] | None
    log_y: bool = False


def _tokens(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values)


def _optional_tokens(values: Sequence[str] | None) -> tuple[str, ...] | None:
    return None if values is None else _tokens(values)


def _default_model_names(models: Sequence[str]) -> tuple[str, ...]:
    label_by_model = dict(zip(DEFAULT_MODELS, DEFAULT_MODEL_NAMES, strict=True))
    return tuple(label_by_model.get(model, model) for model in models)


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


def _validate_recipe_labels(
    recipe_labels: tuple[str, ...] | None, model_names: Sequence[str]
) -> None:
    if recipe_labels is None:
        return
    recipe_order = recipe_order_from_model_names(model_names)
    if not recipe_labels:
        msg = "--recipe-labels requires at least one label"
        raise ValueError(msg)
    if any(label == "" for label in recipe_labels):
        msg = "--recipe-labels requires non-empty labels"
        raise ValueError(msg)
    if len(recipe_labels) != len(recipe_order):
        msg = (
            "--recipe-labels must contain exactly one label per unique "
            f"recipe: recipe_labels={len(recipe_labels)}, recipes={len(recipe_order)}"
        )
        raise ValueError(msg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the raw Sig-MMD context-training paper figure."
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
        default=Path("paper/context_training"),
        help="Directory for figure and CSV outputs.",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help="Model display labels. Defaults to known labels, else --models values.",
    )
    parser.add_argument(
        "--recipe-labels",
        nargs="*",
        default=None,
        help=(
            "Optional x-axis labels for recipes, aligned with the unique "
            "recipe order derived from --model-names."
        ),
    )
    parser.add_argument(
        "--architecture-rows",
        nargs="*",
        default=None,
        help=(
            "Optional architecture names to plot as separate stacked rows instead "
            "of one architecture-hue panel, e.g. --architecture-rows LNP ANP."
        ),
    )
    parser.add_argument(
        "--row-labels",
        nargs="*",
        default=None,
        help=(
            "Optional row labels for --architecture-rows; defaults to the "
            "architecture names."
        ),
    )
    mmd_table.add_test_set_arguments(parser)
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
    parser.add_argument("--checkpoint", default="latest")
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
        default="Context training",
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
    model_names = _resolve_optional_labels(
        values=models,
        labels=args.model_names,
        default_labels=_default_model_names(models),
        mismatch_message="--models and --model-names must have the same length",
    )
    recipe_labels = _optional_tokens(args.recipe_labels)
    architecture_rows = _optional_tokens(args.architecture_rows)
    row_labels = _optional_tokens(args.row_labels)
    splits = tuple(
        mmd_table.resolve_splits(args, roles=("seen", "env", "geno", "unseen"))
    )
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
            default_prediction_method_label(method) for method in prediction_methods
        ),
        mismatch_message=(
            "--prediction-methods and --prediction-method-labels must match"
        ),
    )

    _validate_recipe_labels(recipe_labels, model_names)
    if architecture_rows is None:
        if row_labels is not None:
            msg = "--row-labels requires --architecture-rows"
            raise ValueError(msg)
    else:
        if not architecture_rows or any(
            architecture == "" for architecture in architecture_rows
        ):
            msg = (
                "--architecture-rows requires at least one non-empty architecture name"
            )
            raise ValueError(msg)
        duplicates = _duplicate_tokens(architecture_rows)
        if duplicates:
            msg = (
                "--architecture-rows contains duplicate architecture names: "
                f"{duplicates}"
            )
            raise ValueError(msg)
        if row_labels is not None:
            if any(label == "" for label in row_labels):
                msg = "--row-labels requires non-empty labels"
                raise ValueError(msg)
            if len(row_labels) != len(architecture_rows):
                msg = (
                    "--row-labels must contain exactly one label per "
                    f"--architecture-rows value: row_labels={len(row_labels)}, "
                    f"architecture_rows={len(architecture_rows)}"
                )
                raise ValueError(msg)

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
        model_names=model_names,
        recipe_labels=recipe_labels,
        splits=splits,
        conditionings=conditionings,
        conditioning_labels=conditioning_labels,
        prediction_methods=prediction_methods,
        prediction_method_labels=prediction_method_labels,
        checkpoint=args.checkpoint,
        test_set=args.test_set,
        metric=args.metric,
        view=view,
        prediction_method_views=prediction_method_views,
        title=args.title,
        point_style=args.point_style,
        architecture_rows=architecture_rows,
        row_labels=row_labels,
        log_y=args.log_y,
    )


def normalize_checkpoint(checkpoint: str) -> str:
    checkpoint = checkpoint.strip().lower()
    if checkpoint == "latest":
        return checkpoint
    if checkpoint.startswith("checkpoint-"):
        return checkpoint
    if checkpoint.isdigit():
        return f"checkpoint-{checkpoint}"
    msg = f"Unsupported checkpoint value: {checkpoint!r}"
    raise ValueError(msg)


def summary_pattern(
    *,
    test_set: str,
    split: str,
    model: str,
    conditioning: str,
    prediction_method: str,
    checkpoint: str | None,
    view: str,
) -> str:
    dataloader_dir, split_dir, _, _ = mmd_table.split_info(split, test_set)
    checkpoint_part = "checkpoint-*" if checkpoint is None else checkpoint
    metric_dir = mmd_table.metric_dir_name("sig_mmd", view)
    summary_file = mmd_table.metric_type_config("sig_mmd")["summary_file"]
    return str(
        Path(dataloader_dir)
        / model
        / "*"
        / checkpoint_part
        / split_dir
        / conditioning
        / prediction_method
        / metric_dir
        / summary_file
    )


def find_summary_paths(
    cfg: Config,
    *,
    split: str,
    model: str,
    conditioning: str,
    prediction_method: str,
    checkpoint: str | None,
) -> list[Path]:
    view = mmd_table.prediction_method_view(
        cfg.prediction_method_views, prediction_method
    )
    pattern = summary_pattern(
        test_set=cfg.test_set,
        split=split,
        model=model,
        conditioning=conditioning,
        prediction_method=prediction_method,
        checkpoint=checkpoint,
        view=view,
    )
    return sorted(path for path in cfg.results_base.glob(pattern) if path.is_file())


def iter_cells(cfg: Config) -> Iterable[tuple[str, str, str, str]]:
    yield from product(
        cfg.splits, cfg.models, cfg.conditionings, cfg.prediction_methods
    )


def resolve_checkpoint(cfg: Config) -> str:
    checkpoint = normalize_checkpoint(cfg.checkpoint)
    if checkpoint != "latest":
        return checkpoint

    common_steps: set[int] | None = None
    missing: list[tuple[str, str, str, str]] = []
    per_cell_steps: list[tuple[tuple[str, str, str, str], set[int]]] = []
    for split, model, conditioning, prediction_method in iter_cells(cfg):
        paths = find_summary_paths(
            cfg,
            split=split,
            model=model,
            conditioning=conditioning,
            prediction_method=prediction_method,
            checkpoint=None,
        )
        steps = {
            step
            for path in paths
            if (step := checkpoint_step(path.parents[4].name)) is not None
        }
        if not steps:
            missing.append((split, model, conditioning, prediction_method))
        per_cell_steps.append(((split, model, conditioning, prediction_method), steps))
        common_steps = steps if common_steps is None else common_steps & steps

    if missing:
        detail = "\n".join(
            f"  split={split}, model={model}, conditioning={conditioning}, "
            f"prediction_method={prediction_method}"
            for split, model, conditioning, prediction_method in missing[:20]
        )
        msg = f"Missing Sig-MMD summaries for cells:\n{detail}"
        raise FileNotFoundError(msg)
    if not common_steps:
        detail = "\n".join(
            f"  {cell}: {sorted(steps)}" for cell, steps in per_cell_steps[:40]
        )
        msg = (
            "No common checkpoint exists across requested cells. "
            f"Available examples:\n{detail}"
        )
        raise ValueError(msg)
    return f"checkpoint-{max(common_steps)}"


def read_metric(summary_path: Path, metric: str) -> float:
    df = pd.read_csv(summary_path)
    if df.empty:
        msg = f"Summary CSV is empty: {summary_path}"
        raise ValueError(msg)
    if metric not in df.columns:
        msg = f"Metric column {metric!r} missing from {summary_path}"
        raise KeyError(msg)
    return float(df.at[0, metric])


def recipe_from_model_name(model_name: str) -> str:
    suffix = model_name.split("-NP", 1)[1]
    if not suffix:
        return "NP"
    return suffix.lstrip("-")


def recipe_order_from_model_names(model_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(recipe_from_model_name(name) for name in model_names))


def default_recipe_label(recipe: str) -> str:
    return RECIPE_LABELS.get(recipe, recipe.replace("_", " "))


def recipe_label_by_recipe(cfg: Config) -> dict[str, str]:
    recipe_order = recipe_order_from_model_names(cfg.model_names)
    if cfg.recipe_labels is None:
        return {recipe: default_recipe_label(recipe) for recipe in recipe_order}
    return dict(zip(recipe_order, cfg.recipe_labels, strict=True))


def collect_metrics(cfg: Config, checkpoint: str) -> pd.DataFrame:
    model_name_by_model = dict(zip(cfg.models, cfg.model_names, strict=True))
    conditioning_label_by_conditioning = dict(
        zip(cfg.conditionings, cfg.conditioning_labels, strict=True)
    )
    prediction_label_by_method = dict(
        zip(cfg.prediction_methods, cfg.prediction_method_labels, strict=True)
    )
    recipe_label_by_recipe_value = recipe_label_by_recipe(cfg)

    rows: list[dict[str, object]] = []
    for split, model, conditioning, prediction_method in iter_cells(cfg):
        view = mmd_table.prediction_method_view(
            cfg.prediction_method_views, prediction_method
        )
        paths = find_summary_paths(
            cfg,
            split=split,
            model=model,
            conditioning=conditioning,
            prediction_method=prediction_method,
            checkpoint=checkpoint,
        )
        if len(paths) != 1:
            joined = "\n  ".join(str(path) for path in paths)
            msg = (
                f"Expected exactly one summary for split={split}, model={model}, "
                f"conditioning={conditioning}, prediction_method={prediction_method}, "
                f"view={view}, checkpoint={checkpoint}; found {len(paths)}.\n  "
                f"{joined}"
            )
            raise FileNotFoundError(msg)
        model_name = model_name_by_model[model]
        architecture = model_name.split("-", 1)[0]
        recipe = recipe_from_model_name(model_name)
        rows.append(
            {
                "model": model,
                "model_name": model_name,
                "architecture": architecture,
                "recipe": recipe,
                "recipe_label": recipe_label_by_recipe_value[recipe],
                "split": split,
                "split_label": mmd_table.split_info(split, cfg.test_set)[3],
                "conditioning": conditioning,
                "conditioning_label": conditioning_label_by_conditioning[conditioning],
                "prediction_method": prediction_method,
                "prediction_method_label": prediction_label_by_method[
                    prediction_method
                ],
                "checkpoint": checkpoint,
                "view": view,
                cfg.metric: read_metric(paths[0], cfg.metric),
                "summary_path": str(paths[0]),
            }
        )
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
        ["architecture", "recipe", "recipe_label"], as_index=False, sort=False
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


def _architecture_palette(architecture_order: Sequence[str]) -> dict[str, object]:
    return {
        architecture: ARCHITECTURE_PALETTE.get(
            architecture, COLOR_LIST[index % len(COLOR_LIST)]
        )
        for index, architecture in enumerate(architecture_order)
    }


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


def _validate_architecture_rows(metrics: pd.DataFrame, cfg: Config) -> None:
    if cfg.architecture_rows is None:
        return

    present_architectures = set(metrics["architecture"].astype(str))
    missing_architectures = [
        architecture
        for architecture in cfg.architecture_rows
        if architecture not in present_architectures
    ]
    if missing_architectures:
        msg = (
            "--architecture-rows values must be present in collected metrics; "
            f"missing architectures: {missing_architectures}"
        )
        raise ValueError(msg)


def _architecture_row_labels(cfg: Config) -> tuple[str, ...]:
    if cfg.architecture_rows is None:
        return ()
    return cfg.row_labels if cfg.row_labels is not None else cfg.architecture_rows


def _draw_swapped_fill_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    cfg: Config,
    category_order: Sequence[str],
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
        x_order=category_order,
        split_order=split_order,
        method_order=method_order,
        condition_order=condition_order,
        condition_col="conditioning",
        rng_seed=386,
    )
    draw_centered_summary_bars(
        ax, plot_df, x_col="plot_category", y_col=PLOT_COLUMN, x_order=category_order
    )
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    if show_title and cfg.title:
        ax.set_title(cfg.title)
    ax.set_xticks(range(len(category_order)))
    if show_xticklabels:
        ax.set_xticklabels(category_order)
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def _plot_architecture_row_swapped_fill(
    plot_df: pd.DataFrame,
    cfg: Config,
    output_path: Path,
    category_order: Sequence[str],
    method_order: Sequence[str],
    split_order: Sequence[str],
    condition_order: Sequence[str],
    condition_labels: dict[str, str],
) -> None:
    assert cfg.architecture_rows is not None
    nrows = len(cfg.architecture_rows)
    row_labels = _architecture_row_labels(cfg)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(max(7.2, len(category_order) * 1.05), max(4.2, 3.1 * nrows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, (architecture, row_label) in enumerate(
        zip(cfg.architecture_rows, row_labels, strict=True)
    ):
        ax = axes[row_index, 0]
        row_df = plot_df[plot_df["architecture"].astype(str) == architecture].copy()
        _draw_swapped_fill_panel(
            ax,
            row_df,
            cfg,
            category_order,
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
        condition_labels=condition_labels,
        anchor_y=0.31,
    )
    _save_figure_formats(fig, output_path)
    plt.close(fig)


def _draw_architecture_row_boxplot_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    cfg: Config,
    recipe_order: Sequence[str],
    architecture: str,
    color: object,
    *,
    show_title: bool,
    show_xticklabels: bool,
) -> tuple[list[mlines.Line2D], list[str], int]:
    sns.boxplot(
        data=plot_df,
        x="recipe_label",
        y=PLOT_COLUMN,
        order=recipe_order,
        color=color,
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

    if cfg.point_style == "gray":
        sns.stripplot(
            data=plot_df,
            x="recipe_label",
            y=PLOT_COLUMN,
            order=recipe_order,
            jitter=0.18,
            color=POINT_COLOR,
            size=POINT_SIZE,
            alpha=POINT_ALPHA,
            ax=ax,
        )
        legend_handles = [_mean_handle()]
        legend_labels = ["Mean"]
        legend_ncol = 1
    else:
        point_df = plot_df.copy()
        point_df["row_hue"] = architecture
        split_order = _ordered_unique(point_df["split_label"])
        method_order = _ordered_unique(point_df["prediction_method_label"])
        overlay_testset_context_points(
            ax,
            point_df,
            x_col="recipe_label",
            hue_col="row_hue",
            y_col=PLOT_COLUMN,
            x_order=recipe_order,
            hue_order=[architecture],
            split_order=split_order,
            method_order=method_order,
            box_width=BOX_WIDTH,
            rng_seed=286,
        )
        raise_boxplot_lines_above_points(ax)
        style_handles = testset_context_handles(split_order, method_order)
        legend_handles = [*style_handles, _mean_handle()]
        legend_labels = [str(handle.get_label()) for handle in legend_handles]
        legend_ncol = 4

    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    if show_title and cfg.title:
        ax.set_title(cfg.title)
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    return legend_handles, legend_labels, legend_ncol


def _plot_architecture_row_boxplots(
    plot_df: pd.DataFrame,
    cfg: Config,
    output_path: Path,
    recipe_order: Sequence[str],
    architecture_order: Sequence[str],
) -> None:
    assert cfg.architecture_rows is not None
    nrows = len(cfg.architecture_rows)
    row_labels = _architecture_row_labels(cfg)
    architecture_palette = _architecture_palette(architecture_order)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(7.2, max(4.2, 3.1 * nrows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    legend_handles: list[mlines.Line2D] = []
    legend_labels: list[str] = []
    legend_ncol = 1

    for row_index, (architecture, row_label) in enumerate(
        zip(cfg.architecture_rows, row_labels, strict=True)
    ):
        ax = axes[row_index, 0]
        row_df = plot_df[plot_df["architecture"].astype(str) == architecture].copy()
        handles, labels, ncol = _draw_architecture_row_boxplot_panel(
            ax,
            row_df,
            cfg,
            recipe_order,
            architecture,
            architecture_palette.get(architecture, COLOR_LIST[0]),
            show_title=row_index == 0,
            show_xticklabels=row_index == nrows - 1,
        )
        if row_index == 0:
            legend_handles = handles
            legend_labels = labels
            legend_ncol = ncol
        ax.set_ylabel(row_label, fontweight="bold")
        _set_y_axis(ax, plot_df[PLOT_COLUMN], log_y=cfg.log_y)

    legend_bottom = 0.13 if cfg.point_style == "gray" else 0.22
    legend_anchor = 0.065 if cfg.point_style == "gray" else 0.105
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


def _plot_swapped_fill(
    plot_df: pd.DataFrame,
    cfg: Config,
    output_path: Path,
    recipe_order: Sequence[str],
    architecture_order: Sequence[str],
) -> None:
    if cfg.architecture_rows is None:
        plot_df["plot_category"] = (
            plot_df["recipe_label"].astype(str)
            + "\n"
            + plot_df["architecture"].astype(str)
        )
        category_order = [
            f"{recipe}\n{architecture}"
            for recipe in recipe_order
            for architecture in architecture_order
            if f"{recipe}\n{architecture}" in set(plot_df["plot_category"])
        ]
    else:
        plot_df["plot_category"] = plot_df["recipe_label"].astype(str)
        category_order = [
            recipe for recipe in recipe_order if recipe in set(plot_df["plot_category"])
        ]
    method_order, split_order, condition_order = swapped_fill_orders(
        split_values=_ordered_unique(plot_df["split_label"]),
        method_values=_ordered_unique(plot_df["prediction_method_label"]),
        condition_values=_ordered_unique(plot_df["conditioning"]),
    )
    condition_labels = _label_by_value(
        plot_df, value_col="conditioning", label_col="conditioning_label"
    )
    if cfg.architecture_rows is not None:
        _plot_architecture_row_swapped_fill(
            plot_df,
            cfg,
            output_path,
            category_order,
            method_order,
            split_order,
            condition_order,
            condition_labels,
        )
        return
    fig, ax = plt.subplots(figsize=(max(7.2, len(category_order) * 1.05), 4.2))
    _draw_swapped_fill_panel(
        ax,
        plot_df,
        cfg,
        category_order,
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
        condition_labels=condition_labels,
        anchor_y=0.22,
    )
    _save_figure_formats(fig, output_path)
    plt.close(fig)


def plot_boxplot(metrics: pd.DataFrame, cfg: Config, output_path: Path) -> None:
    setup_style()
    recipe_order = _ordered_unique(metrics["recipe_label"])
    architecture_order = _ordered_unique(metrics["architecture"])

    plot_df = metrics.copy()
    plot_df["recipe_label"] = pd.Categorical(
        plot_df["recipe_label"], categories=recipe_order, ordered=True
    )
    plot_df["architecture"] = pd.Categorical(
        plot_df["architecture"], categories=architecture_order, ordered=True
    )
    _validate_architecture_rows(plot_df, cfg)

    if cfg.point_style == "swapped-fill":
        _plot_swapped_fill(plot_df, cfg, output_path, recipe_order, architecture_order)
        return

    if cfg.architecture_rows is not None:
        _plot_architecture_row_boxplots(
            plot_df, cfg, output_path, recipe_order, architecture_order
        )
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.boxplot(
        data=plot_df,
        x="recipe_label",
        y=PLOT_COLUMN,
        hue="architecture",
        order=recipe_order,
        hue_order=architecture_order,
        palette=_architecture_palette(architecture_order),
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
    box_handles, box_labels = ax.get_legend_handles_labels()

    if cfg.point_style == "gray":
        sns.stripplot(
            data=plot_df,
            x="recipe_label",
            y=PLOT_COLUMN,
            hue="architecture",
            order=recipe_order,
            hue_order=architecture_order,
            dodge=True,
            jitter=0.18,
            palette=dict.fromkeys(architecture_order, POINT_COLOR),
            size=POINT_SIZE,
            alpha=POINT_ALPHA,
            legend=False,
            ax=ax,
        )
        legend_handles = [*box_handles, _mean_handle()]
        legend_labels = [*box_labels, "Mean"]
        legend_ncol = len(legend_handles)
        legend_bottom = 0.20
        legend_anchor = 0.105
    else:
        split_order = _ordered_unique(plot_df["split_label"])
        method_order = _ordered_unique(plot_df["prediction_method_label"])
        overlay_testset_context_points(
            ax,
            plot_df,
            x_col="recipe_label",
            hue_col="architecture",
            y_col=PLOT_COLUMN,
            x_order=recipe_order,
            hue_order=architecture_order,
            split_order=split_order,
            method_order=method_order,
            box_width=BOX_WIDTH,
            rng_seed=286,
        )
        raise_boxplot_lines_above_points(ax)
        style_handles = testset_context_handles(split_order, method_order)
        legend_handles = [*box_handles, *style_handles, _mean_handle()]
        legend_labels = [handle.get_label() for handle in legend_handles]
        legend_ncol = 4
        legend_bottom = 0.34
        legend_anchor = 0.115

    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    ax.set_ylabel("Sig-MMD² × 10³")
    if cfg.title:
        ax.set_title(cfg.title)
    _set_y_axis(ax, plot_df[PLOT_COLUMN], log_y=cfg.log_y)

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
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

    for fmt in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{fmt}"))
    plt.close(fig)


def write_outputs(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(cfg)
    logger.info(f"Using checkpoint {checkpoint}")

    metrics = collect_metrics(cfg, checkpoint)
    summary = summarize_metrics(metrics, cfg.prediction_method_views)

    metrics.to_csv(cfg.output_dir / "context_training_metrics.csv", index=False)
    summary.to_csv(cfg.output_dir / "context_training_summary.csv", index=False)
    plot_boxplot(metrics, cfg, cfg.output_dir / "context_training")

    logger.info(f"Wrote context training figure and CSVs to {cfg.output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    write_outputs(parse_args(argv))


if __name__ == "__main__":
    main()
