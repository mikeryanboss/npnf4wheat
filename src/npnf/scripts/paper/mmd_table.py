"""Build paper LaTeX tables and Markdown copies from Sig-MMD / CSig-MMD summaries."""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger

from npnf.data.configs.datasets.synthetic_test_sets import (
    SyntheticTestSet,
    design_splits,
    split_named,
)
from npnf.scripts.utils.outputs import markdown_table
from npnf.scripts.utils.paths import checkpoint_step, normalize_checkpoint

DEFAULT_MODELS = [
    "CNP-512k",
    "ANP-512k",
    "ANP-NF-Prior-512k",
    "ANP-NF-Posterior-512k",
    "ANP-NF-Prior-Posterior-512k",
]
DEFAULT_MODEL_NAMES = [
    "CNP",
    "ANP",
    "ANP-NF-Prior",
    "ANP-NF-Posterior",
    "ANP-NF-Prior-Posterior",
]
DEFAULT_CONDITIONINGS = ["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"]
DEFAULT_CONDITIONING_LABELS = ["P", "E", "G", "E&G"]
DEFAULT_PREDICTION_METHOD = "no_context"
DEFAULT_CHECKPOINT = "latest"
DEFAULT_METRIC = "mean"
DEFAULT_DECIMALS = 3
DEFAULT_METRIC_TYPE = "sig_mmd"
DEFAULT_VIEW = "condition"
VIEW_CHOICES = ("condition", "context_matched_blocked")
LATEX_VALUE_SCALE = 1000.0

_METRIC_TYPE_CONFIG = {
    "sig_mmd": {
        "dir_name": "sig_mmd",
        "summary_file": "sig_mmd_summary.csv",
        "latex_header": r"Mean Sig-MMD$^2 \times 10^3$ ($\downarrow$)",
        "default_output_folder": "paper/sig_mmd_table",
        "stem_prefix": "sig_mmd",
        "label": "Sig-MMD",
    },
    "csig_mmd": {
        "dir_name": "csig_mmd",
        "summary_file": "csig_mmd_summary.csv",
        "latex_header": r"Mean CSig-MMD$^2 \times 10^3$ ($\downarrow$)",
        "default_output_folder": "paper/csig_mmd_table",
        "stem_prefix": "csig_mmd",
        "label": "CSig-MMD",
    },
}


def metric_type_choices() -> list[str]:
    return sorted(_METRIC_TYPE_CONFIG)


def metric_type_config(metric_type: str) -> dict[str, str]:
    """Return the metric-type configuration dict."""
    if metric_type not in _METRIC_TYPE_CONFIG:
        known = metric_type_choices()
        msg = f"Unknown --metric-type {metric_type!r}. Known: {known}"
        raise ValueError(msg)
    return _METRIC_TYPE_CONFIG[metric_type]


def _mtc(metric_type: str) -> dict[str, str]:
    return metric_type_config(metric_type)


def validate_view(view: str) -> str:
    """Return a normalized view name or raise for unsupported values."""
    if view not in VIEW_CHOICES:
        msg = f"Unknown --view {view!r}. Known: {list(VIEW_CHOICES)}"
        raise ValueError(msg)
    return view


def validate_metric_view(metric_type: str, view: str) -> str:
    """Return a normalized view name or raise for unsupported combinations."""
    metric_type_config(metric_type)
    return validate_view(view)


def metric_dir_name(metric_type: str, view: str = DEFAULT_VIEW) -> str:
    """Return the metric output directory for a metric type and view."""
    c = _mtc(metric_type)
    view = validate_metric_view(metric_type, view)
    if view == "condition":
        return c["dir_name"]
    if view == "context_matched_blocked":
        return f"{c['dir_name']}_context_matched_blocked"
    msg = f"Unsupported metric view: {view!r}"
    raise ValueError(msg)


def metric_stem_prefix(metric_type: str, view: str = DEFAULT_VIEW) -> str:
    """Return the output stem prefix for a metric type and view."""
    c = _mtc(metric_type)
    view = validate_metric_view(metric_type, view)
    if view == "condition":
        return c["stem_prefix"]
    if view == "context_matched_blocked":
        return f"{c['stem_prefix']}_context_matched_blocked"
    msg = f"Unsupported metric view: {view!r}"
    raise ValueError(msg)


def parse_prediction_method_view_overrides(
    raw_overrides: Sequence[str] | None,
) -> dict[str, str]:
    """Parse --prediction-method-views METHOD=VIEW tokens."""
    overrides: dict[str, str] = {}
    for token in raw_overrides or ():
        if "=" not in token:
            msg = (
                "--prediction-method-views entries must use METHOD=VIEW syntax; "
                f"missing '=' in {token!r}."
            )
            raise ValueError(msg)
        prediction_method, view = token.split("=", 1)
        prediction_method = prediction_method.strip()
        view = view.strip()
        if prediction_method == "":
            msg = (
                "--prediction-method-views entries must name a non-empty "
                f"prediction method; got {token!r}."
            )
            raise ValueError(msg)
        if view == "":
            msg = (
                "--prediction-method-views entries must name a non-empty view; "
                f"got {token!r}."
            )
            raise ValueError(msg)
        if prediction_method in overrides:
            msg = (
                "--prediction-method-views contains duplicate overrides for "
                f"prediction method {prediction_method!r}."
            )
            raise ValueError(msg)
        overrides[prediction_method] = view
    return overrides


def resolve_prediction_method_views(
    prediction_methods: Sequence[str],
    *,
    metric_type: str,
    default_view: str,
    raw_overrides: Sequence[str] | Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return each selected prediction method's effective metric view."""
    selected_methods = list(prediction_methods)
    selected_set = set(selected_methods)
    effective = {
        prediction_method: validate_metric_view(metric_type, default_view)
        for prediction_method in selected_methods
    }
    overrides = (
        dict(raw_overrides)
        if isinstance(raw_overrides, Mapping)
        else parse_prediction_method_view_overrides(raw_overrides)
    )
    unselected = [
        prediction_method
        for prediction_method in overrides
        if prediction_method not in selected_set
    ]
    if unselected:
        msg = (
            "--prediction-method-views may only override selected prediction "
            f"methods. Unselected: {unselected}. Selected: {selected_methods}."
        )
        raise ValueError(msg)
    for prediction_method, view in overrides.items():
        effective[prediction_method] = validate_metric_view(metric_type, view)
    return effective


def prediction_method_view(
    prediction_method_views: Mapping[str, str], prediction_method: str
) -> str:
    """Return the effective view for one selected prediction method."""
    try:
        return prediction_method_views[prediction_method]
    except KeyError as error:
        msg = (
            "No effective view was resolved for prediction method "
            f"{prediction_method!r}."
        )
        raise ValueError(msg) from error


def metric_stem_prefix_for_prediction_method_views(
    metric_type: str, prediction_method_views: Mapping[str, str]
) -> str:
    """Return a homogeneous view stem or a visible mixed-view stem."""
    views = list(dict.fromkeys(prediction_method_views.values()))
    if not views:
        return metric_stem_prefix(metric_type, DEFAULT_VIEW)
    for view in views:
        validate_metric_view(metric_type, view)
    if len(set(views)) == 1:
        return metric_stem_prefix(metric_type, views[0])
    return f"{_mtc(metric_type)['stem_prefix']}_mixed_views"


def format_prediction_method_views(
    prediction_method_views: Mapping[str, str],
    prediction_methods: Sequence[str] | None = None,
) -> str:
    """Format a deterministic source trace for summary CSVs."""
    methods = (
        list(prediction_methods)
        if prediction_methods is not None
        else sorted(prediction_method_views)
    )
    return ";".join(
        f"{prediction_method}="
        f"{prediction_method_view(prediction_method_views, prediction_method)}"
        for prediction_method in methods
    )


PREDICTION_METHODS = {"no_context", "max_height", "random_context"}
PREDICTION_METHOD_LABELS = {
    "no_context": "no context",
    "random_context": "random context",
    "max_height": "max context",
}


@dataclass(frozen=True)
class TableSpec:
    model: str
    model_name: str
    split: str
    split_label: str
    conditioning: str
    conditioning_label: str
    test_set: str = SyntheticTestSet.LEGACY.value
    checkpoint: str | None = None  # the model's own checkpoint, instead of --checkpoint


@dataclass(frozen=True)
class ResolvedSummary:
    spec: TableSpec
    prediction_method: str
    checkpoint: str
    view: str
    summary_path: Path
    value: float
    standard_error: float = 0.0
    floored_value: float | None = None  # the value compared for bold, if not ``value``


@dataclass(frozen=True)
class AveragedSummary:
    model: str
    model_name: str
    prediction_method: str
    prediction_method_label: str
    conditioning: str
    conditioning_label: str
    checkpoint: str
    view: str
    value: float
    source_splits: tuple[str, ...]
    source_split_labels: tuple[str, ...]
    summary_paths: tuple[Path, ...]
    standard_error: float = 0.0
    floored_value: float | None = None


@dataclass(frozen=True)
class _SummaryLookupFailure:
    kind: str
    results_base: Path
    spec: TableSpec
    prediction_method: str
    metric_type: str
    checkpoint: str | None
    view: str
    expected_glob: str
    matched_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _ResolvedSummaryPath:
    spec: TableSpec
    prediction_method: str
    view: str
    summary_path: Path
    checkpoint: str


@dataclass(frozen=True)
class _RestartCommandGroup:
    results_base: str
    checkpoint: str
    models: tuple[str, ...]
    splits: tuple[str, ...]
    conditionings: tuple[str, ...]
    prediction_methods: tuple[str, ...]
    n_cells: int


def default_results_base() -> str:
    return os.environ["NPNF_RESULTS_DIR"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paper CSV and LaTeX tables from Sig-MMD / CSig-MMD "
        "summaries."
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default=default_results_base(),
        help="Base results directory (default: $NPNF_RESULTS_DIR or results).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model result folder names, in row order.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help="Model display labels. Defaults to known labels, else model names.",
    )
    add_test_set_arguments(parser)
    add_model_checkpoints_argument(parser)
    parser.add_argument(
        "--split-labels",
        nargs="+",
        default=None,
        help="Display labels for splits. Defaults to built-in labels.",
    )
    parser.add_argument(
        "--conditionings",
        nargs="+",
        default=DEFAULT_CONDITIONINGS,
        help="Conditioning directory names, in nested-column order.",
    )
    parser.add_argument(
        "--conditioning-labels",
        nargs="+",
        default=None,
        help="Display labels for conditionings. Defaults to P E G 'E&G'.",
    )
    parser.add_argument(
        "--prediction-method",
        type=str,
        default=DEFAULT_PREDICTION_METHOD,
        choices=sorted(PREDICTION_METHODS),
        help="Prediction method to table. Run once per method.",
    )
    parser.add_argument(
        "--prediction-methods",
        nargs="+",
        default=None,
        choices=sorted(PREDICTION_METHODS),
        help="Prediction methods for a method-grouped table. When set, "
        "the table averages over the requested splits.",
    )
    parser.add_argument(
        "--prediction-method-names",
        nargs="+",
        default=None,
        help="Display labels for --prediction-methods. Defaults to readable "
        "method names.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Common checkpoint: latest, N, or checkpoint-N (default: latest).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help="Metric column from summary CSV (default: mean).",
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default=DEFAULT_METRIC_TYPE,
        choices=metric_type_choices(),
        help=(
            f"Metric to table: sig_mmd or csig_mmd. (default: {DEFAULT_METRIC_TYPE})"
        ),
    )
    parser.add_argument(
        "--view",
        type=str,
        default=DEFAULT_VIEW,
        choices=VIEW_CHOICES,
        help=(
            "Metric view to table: condition or context_matched_blocked. "
            f"(default: {DEFAULT_VIEW})"
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
        "--decimals",
        type=int,
        default=DEFAULT_DECIMALS,
        help="Decimal places for LaTeX values (default: 3).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder for CSV and LaTeX files "
        "(default depends on --metric-type).",
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Model name to use as reference for cell coloring (e.g. LNP). "
        "Must match a --model-names value. Omit to disable coloring.",
    )
    parser.add_argument(
        "--floors",
        nargs="*",
        default=[],
        metavar="SPLIT=VALUE",
        help="Score of a perfect model per split, in table units (x 1000); a value "
        "below it counts as the floor when bold is decided. Unlisted splits use 0, "
        "the expectation of the unbiased estimator. FIP1 uses the half-split floors.",
    )
    parser.add_argument(
        "--floor-sds",
        nargs="*",
        default=[],
        metavar="SPLIT=VALUE",
        help="Standard error per split, in table units, of a cell with one covariate "
        "unit and no seed spread, in place of the spread of its blocks. FIP1 uses the "
        "sd of the half-split floor, because the blocks share the observed plots.",
    )
    return parser


def _known_model_labels(models: list[str]) -> list[str]:
    label_by_model = dict(zip(DEFAULT_MODELS, DEFAULT_MODEL_NAMES, strict=True))
    return [label_by_model.get(model, model) for model in models]


def _resolve_labels(
    *,
    values: list[str],
    labels: list[str] | None,
    default_labels: list[str] | None,
    name: str,
) -> list[str]:
    if labels is not None:
        if len(labels) != len(values):
            msg = f"--{name} length ({len(labels)}) must match values ({len(values)})."
            raise ValueError(msg)
        return labels
    if default_labels is not None and len(default_labels) == len(values):
        return default_labels
    return values


def resolve_prediction_method_labels(
    prediction_methods: list[str], prediction_method_names: list[str] | None
) -> list[str]:
    default_labels = [PREDICTION_METHOD_LABELS[method] for method in prediction_methods]
    return _resolve_labels(
        values=prediction_methods,
        labels=prediction_method_names,
        default_labels=default_labels,
        name="prediction-method-names",
    )


def add_test_set_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --test-set and --splits (default: the four design splits of the test set)."""
    SyntheticTestSet.add_argument(parser)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Design splits of --test-set (default: all four).",
    )


def add_model_checkpoints_argument(parser: argparse.ArgumentParser) -> None:
    """Add --model-checkpoints, the checkpoint of a model that has its own."""
    parser.add_argument(
        "--model-checkpoints",
        nargs="*",
        default=[],
        metavar="MODEL=CHECKPOINT",
        help="Checkpoint (N or checkpoint-N) of a --models entry that shares no "
        "training step with the others, for example a fitted baseline at "
        "checkpoint-0. The other models use --checkpoint.",
    )


def parse_model_checkpoints(
    raw: Sequence[str], models: Sequence[str]
) -> dict[str, str]:
    """``MODEL=CHECKPOINT`` pairs as checkpoint directory names by model."""
    checkpoints = {}
    for item in raw:
        model, separator, checkpoint = item.partition("=")
        step = checkpoint.removeprefix("checkpoint-")
        if not separator or not step.isdigit():
            msg = (
                "--model-checkpoints expects MODEL=N or MODEL=checkpoint-N, "
                f"got {item!r}."
            )
            raise ValueError(msg)
        if model not in models:
            msg = f"--model-checkpoints names {model!r}, which is not in --models."
            raise ValueError(msg)
        checkpoints[model] = f"checkpoint-{step}"
    return checkpoints


# FIP1 scores enter the paper scripts as a synthetic-shaped result tree under these
# split names. FIP1 has no simulator oracle, so its splits are not in the synthetic
# test-set registry. Values: dataloader directory, split directory, title, role.
FIP1_SPLITS = {
    "fip1_plot": ("fip1_test_plot_dataloaders", "test_plot", "Plot", "seen"),
    "fip1_genotype": (
        "fip1_test_genotype_dataloaders",
        "test_genotype",
        "Genotype",
        "geno",
    ),
    "fip1_environment": (
        "fip1_test_environment_dataloaders",
        "test_environment",
        "Environment 2019",
        "env",
    ),
    "fip1_unseen": (
        "fip1_test_genotype_environment_dataloaders",
        "test_genotype_environment",
        "Unseen 2019",
        "unseen",
    ),
}


def split_info(split: str, test_set: str) -> tuple[str, str, str, str | None]:
    """Dataloader directory, split directory, title and role of a split."""
    if split in FIP1_SPLITS:
        return FIP1_SPLITS[split]
    test_split = split_named(split, test_set)
    return (
        test_split.dataloader_name,
        test_split.split_key,
        test_split.title,
        test_split.role,
    )


def resolve_splits(
    args: argparse.Namespace, roles: Sequence[str] = ("seen", "geno", "env", "unseen")
) -> list[str]:
    """Requested design-split aliases, or the four of --test-set in ``roles`` order."""
    by_role = {split.role: split.alias for split in design_splits(args.test_set)}
    if args.splits is None:
        return [by_role[role] for role in roles]
    splits = [split.strip().lower() for split in args.splits]
    known = [*by_role.values(), *FIP1_SPLITS]
    unknown_splits = [split for split in splits if split not in known]
    if unknown_splits:
        msg = f"Unknown split(s): {unknown_splits}. Known: {sorted(known)}."
        raise ValueError(msg)
    return splits


def validate_and_build_specs(args: argparse.Namespace) -> list[TableSpec]:
    models = list(args.models)
    model_names = _resolve_labels(
        values=models,
        labels=args.model_names,
        default_labels=_known_model_labels(models),
        name="model-names",
    )

    splits = resolve_splits(args)
    split_default_labels = [split_info(split, args.test_set)[2] for split in splits]
    split_labels = _resolve_labels(
        values=splits,
        labels=args.split_labels,
        default_labels=split_default_labels,
        name="split-labels",
    )

    conditionings = list(args.conditionings)
    conditioning_default_labels = [
        DEFAULT_CONDITIONING_LABELS[DEFAULT_CONDITIONINGS.index(conditioning)]
        if conditioning in DEFAULT_CONDITIONINGS
        else conditioning
        for conditioning in conditionings
    ]
    conditioning_labels = _resolve_labels(
        values=conditionings,
        labels=args.conditioning_labels,
        default_labels=conditioning_default_labels,
        name="conditioning-labels",
    )

    model_checkpoints = parse_model_checkpoints(args.model_checkpoints, models)
    specs: list[TableSpec] = []
    for model, model_name in zip(models, model_names, strict=True):
        for split, split_label in zip(splits, split_labels, strict=True):
            for conditioning, conditioning_label in zip(
                conditionings, conditioning_labels, strict=True
            ):
                specs.append(
                    TableSpec(
                        model=model,
                        model_name=model_name,
                        split=split,
                        split_label=split_label,
                        conditioning=conditioning,
                        conditioning_label=conditioning_label,
                        test_set=args.test_set,
                        checkpoint=model_checkpoints.get(model),
                    )
                )
    return specs


def _summary_search_root_pattern(
    results_base: Path,
    spec: TableSpec,
    prediction_method: str,
    metric_type: str,
    checkpoint: str | None,
    view: str = DEFAULT_VIEW,
) -> tuple[Path, str]:
    dataloader_name, split_dir, _, _ = split_info(spec.split, spec.test_set)
    checkpoint_part = checkpoint if checkpoint is not None else "checkpoint-*"
    c = _mtc(metric_type)
    metric_dir = metric_dir_name(metric_type, view)
    root = results_base / dataloader_name / spec.model
    pattern = (
        f"*/{checkpoint_part}/{split_dir}/{spec.conditioning}/"
        f"{prediction_method}/{metric_dir}/{c['summary_file']}"
    )
    return root, pattern


def _find_summary_paths(
    results_base: Path,
    spec: TableSpec,
    prediction_method: str,
    metric_type: str,
    checkpoint: str | None,
    view: str = DEFAULT_VIEW,
) -> list[Path]:
    root, pattern = _summary_search_root_pattern(
        results_base, spec, prediction_method, metric_type, checkpoint, view
    )
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _summary_checkpoint_name(summary_path: Path) -> str:
    return summary_path.parents[4].name


def _cell_description(spec: TableSpec, prediction_method: str) -> str:
    return (
        f"model={spec.model}, split={spec.split}, "
        f"conditioning={spec.conditioning}, prediction_method={prediction_method}"
    )


def _summary_lookup_failure(
    kind: str,
    results_base: Path,
    spec: TableSpec,
    prediction_method: str,
    metric_type: str,
    checkpoint: str | None,
    view: str,
    paths: list[Path],
) -> _SummaryLookupFailure:
    root, pattern = _summary_search_root_pattern(
        results_base, spec, prediction_method, metric_type, checkpoint, view
    )
    return _SummaryLookupFailure(
        kind=kind,
        results_base=results_base,
        spec=spec,
        prediction_method=prediction_method,
        metric_type=metric_type,
        checkpoint=checkpoint,
        view=view,
        expected_glob=str(root / pattern),
        matched_paths=tuple(paths),
    )


def _format_summary_lookup_failure(failure: _SummaryLookupFailure) -> str:
    c = _mtc(failure.metric_type)
    metric_dir = metric_dir_name(failure.metric_type, failure.view)
    checkpoint_label = (
        failure.checkpoint if failure.checkpoint is not None else "checkpoint-*"
    )
    lines = [
        (
            f"- {failure.kind}: "
            f"{_cell_description(failure.spec, failure.prediction_method)}, "
            f"metric_type={failure.metric_type}, view={failure.view}, "
            f"checkpoint={checkpoint_label}, metric_dir={metric_dir}, "
            f"summary_file={c['summary_file']}, found={len(failure.matched_paths)}"
        ),
        f"  expected_glob={failure.expected_glob}",
    ]
    if failure.matched_paths:
        lines.append("  matched_paths:")
        lines.extend(f"    {path}" for path in failure.matched_paths)
    return "\n".join(lines)


def _restart_checkpoint(failure: _SummaryLookupFailure) -> str:
    return failure.checkpoint if failure.checkpoint is not None else "latest"


def _restart_cell(
    failure: _SummaryLookupFailure,
) -> tuple[str, str, str, str, str, str]:
    return (
        str(failure.results_base),
        _restart_checkpoint(failure),
        failure.spec.model,
        failure.spec.split,
        failure.spec.conditioning,
        failure.prediction_method,
    )


def _restart_group_from_cells(
    cells: set[tuple[str, str, str, str, str, str]],
) -> _RestartCommandGroup:
    values = [tuple(sorted({cell[index] for cell in cells})) for index in range(6)]
    return _RestartCommandGroup(
        results_base=values[0][0],
        checkpoint=values[1][0],
        models=values[2],
        splits=values[3],
        conditionings=values[4],
        prediction_methods=values[5],
        n_cells=len(cells),
    )


def _is_complete_restart_block(cells: set[tuple[str, str, str, str, str, str]]) -> bool:
    expected = 1
    for index in range(6):
        expected *= len({cell[index] for cell in cells})
    return len(cells) == expected


def _extract_restart_groups(
    remaining_cells: set[tuple[str, str, str, str, str, str]],
    fixed_indexes: tuple[int, ...],
) -> list[_RestartCommandGroup]:
    groups: list[_RestartCommandGroup] = []
    consumed: set[tuple[str, str, str, str, str, str]] = set()
    keys = sorted(
        {tuple(cell[index] for index in fixed_indexes) for cell in remaining_cells}
    )
    for key in keys:
        group_cells = {
            cell
            for cell in remaining_cells
            if tuple(cell[index] for index in fixed_indexes) == key
        }
        if len(group_cells) <= 1 or not _is_complete_restart_block(group_cells):
            continue
        groups.append(_restart_group_from_cells(group_cells))
        consumed.update(group_cells)
    remaining_cells.difference_update(consumed)
    return groups


def _restart_command_groups(
    cells: set[tuple[str, str, str, str, str, str]],
) -> list[_RestartCommandGroup]:
    remaining_cells = set(cells)
    groups: list[_RestartCommandGroup] = []
    for fixed_indexes in [
        (0, 1, 3),
        (0, 1, 2),
        (0, 1, 4),
        (0, 1, 5),
        (0, 1, 2, 3),
        (0, 1, 2, 3, 5),
        (0, 1, 2, 3, 4),
    ]:
        groups.extend(_extract_restart_groups(remaining_cells, fixed_indexes))
    groups.extend(_restart_group_from_cells({cell}) for cell in sorted(remaining_cells))
    return groups


def _join_values(values: tuple[str, ...]) -> str:
    return " ".join(values)


def _restart_results_base_arg(results_base: str) -> str:
    if Path(results_base).is_absolute():
        return "$NPNF_RESULTS_DIR"
    return results_base


def _quote_command_arg(value: str) -> str:
    if value == "$NPNF_RESULTS_DIR":
        return '"$NPNF_RESULTS_DIR"'
    return shlex.quote(value)


def _format_restart_command(group: _RestartCommandGroup) -> str:
    command = [
        "uv",
        "run",
        "python",
        "scripts/launch_sig_mmd_context_matched_blocked_jobs.py",
        "--results-base",
        _restart_results_base_arg(group.results_base),
        "--models",
        *group.models,
        "--splits",
        *group.splits,
    ]
    if set(group.conditionings) != set(DEFAULT_CONDITIONINGS):
        command.extend(["--conditionings", *group.conditionings])
    if set(group.prediction_methods) != PREDICTION_METHODS:
        command.extend(["--prediction-methods", *group.prediction_methods])
    if group.checkpoint != DEFAULT_CHECKPOINT:
        command.extend(["--checkpoint", group.checkpoint])
    return " ".join(_quote_command_arg(part) for part in command)


def _context_matched_missing_cells(
    failures: list[_SummaryLookupFailure],
) -> set[tuple[str, str, str, str, str, str]]:
    return {
        _restart_cell(failure)
        for failure in failures
        if failure.kind == "missing"
        and failure.metric_type == "sig_mmd"
        and failure.view == "context_matched_blocked"
    }


def _format_context_matched_restart_summary(
    cells: set[tuple[str, str, str, str, str, str]],
) -> str:
    if not cells:
        return ""

    lines = [
        "Restart summary for missing context-matched blocked Sig-MMD jobs:",
        f"  missing_cells={len(cells)}",
        f"  models={_join_values(tuple(sorted({cell[2] for cell in cells})))}",
        f"  splits={_join_values(tuple(sorted({cell[3] for cell in cells})))}",
        f"  conditionings={_join_values(tuple(sorted({cell[4] for cell in cells})))}",
        (
            "  prediction_methods="
            f"{_join_values(tuple(sorted({cell[5] for cell in cells})))}"
        ),
    ]
    return "\n".join(lines)


def _format_context_matched_relaunch_commands(
    cells: set[tuple[str, str, str, str, str, str]],
) -> str:
    if not cells:
        return ""

    groups = _restart_command_groups(cells)
    lines = ["Suggested relaunch commands (cover missing cells only):"]
    for group in groups:
        lines.extend(
            [
                (
                    f"  # {group.n_cells} cell(s): "
                    f"models={_join_values(group.models)}, "
                    f"splits={_join_values(group.splits)}, "
                    f"conditionings={_join_values(group.conditionings)}, "
                    f"prediction_methods={_join_values(group.prediction_methods)}"
                ),
                f"  {_format_restart_command(group)}",
            ]
        )
    return "\n".join(lines)


def _raise_summary_lookup_failures(failures: list[_SummaryLookupFailure]) -> None:
    if not failures:
        return
    c = _mtc(failures[0].metric_type)
    has_missing = any(failure.kind == "missing" for failure in failures)
    has_ambiguous = any(failure.kind == "ambiguous" for failure in failures)
    if has_ambiguous and not has_missing:
        prefix = f"Ambiguous {c['summary_file']} files"
    elif has_missing and not has_ambiguous:
        prefix = f"Missing {c['summary_file']} files"
    else:
        prefix = f"Missing or ambiguous {c['summary_file']} files"
    detail = "\n".join(_format_summary_lookup_failure(failure) for failure in failures)
    restart_cells = _context_matched_missing_cells(failures)
    restart_summary = _format_context_matched_restart_summary(restart_cells)
    relaunch_commands = _format_context_matched_relaunch_commands(restart_cells)
    if restart_summary:
        msg = (
            f"{prefix} for {len(failures)} requested cell(s):\n"
            f"{restart_summary}\n\nDetailed lookup diagnostics:\n{detail}\n\n"
            f"{relaunch_commands}"
        )
    else:
        msg = f"{prefix} for {len(failures)} requested cell(s):\n{detail}"
    if has_ambiguous:
        raise ValueError(msg)
    raise FileNotFoundError(msg)


def resolve_common_checkpoint_for_methods(
    results_base: Path,
    specs: list[TableSpec],
    prediction_methods: list[str],
    metric_type: str,
    checkpoint: str,
    view: str = DEFAULT_VIEW,
    prediction_method_views: Mapping[str, str] | None = None,
) -> str:
    c = _mtc(metric_type)
    effective_views = resolve_prediction_method_views(
        prediction_methods,
        metric_type=metric_type,
        default_view=view,
        raw_overrides=prediction_method_views,
    )
    normalized = normalize_checkpoint(checkpoint)
    if normalized != "latest":
        return normalized

    common_steps: set[int] | None = None
    per_cell_steps: list[tuple[TableSpec, str, set[int]]] = []
    lookup_failures: list[_SummaryLookupFailure] = []
    for prediction_method in prediction_methods:
        method_view = prediction_method_view(effective_views, prediction_method)
        for spec in specs:
            paths = _find_summary_paths(
                results_base, spec, prediction_method, metric_type, None, method_view
            )
            if not paths:
                lookup_failures.append(
                    _summary_lookup_failure(
                        "missing",
                        results_base,
                        spec,
                        prediction_method,
                        metric_type,
                        None,
                        method_view,
                        paths,
                    )
                )
                continue
            steps = {
                step
                for path in paths
                if (step := checkpoint_step(_summary_checkpoint_name(path))) is not None
            }
            per_cell_steps.append((spec, prediction_method, steps))
            common_steps = steps if common_steps is None else common_steps & steps

    _raise_summary_lookup_failures(lookup_failures)

    if not common_steps:
        detail = "\n".join(
            f"  {_cell_description(spec, prediction_method)}: {sorted(steps)}"
            for spec, prediction_method, steps in per_cell_steps
        )
        msg = (
            f"No common checkpoint is available for every requested {c['label']} "
            "table cell. Specify --checkpoint explicitly. Available steps:\n"
            f"{detail}"
        )
        raise ValueError(msg)

    return f"checkpoint-{max(common_steps)}"


def resolve_common_checkpoint(
    results_base: Path,
    specs: list[TableSpec],
    prediction_method: str,
    metric_type: str,
    checkpoint: str,
    view: str = DEFAULT_VIEW,
) -> str:
    return resolve_common_checkpoint_for_methods(
        results_base, specs, [prediction_method], metric_type, checkpoint, view
    )


def _read_metric(summary_path: Path, metric: str) -> float:
    df = pl.read_csv(summary_path)
    if metric not in df.columns:
        msg = f"Metric column {metric!r} missing from {summary_path}."
        raise ValueError(msg)
    if df.height == 0:
        msg = f"Summary CSV is empty: {summary_path}."
        raise ValueError(msg)
    raw_value = df[metric][0]
    if raw_value is None:
        msg = f"Metric column {metric!r} is null in {summary_path}."
        raise ValueError(msg)
    value = float(raw_value)
    if not math.isfinite(value):
        msg = f"Metric column {metric!r} is non-finite ({value!r}) in {summary_path}."
        raise ValueError(msg)
    return value


def _read_standard_error(summary_path: Path, floor_sd: float | None = None) -> float:
    """Standard error of a summary's mean: over the training seeds of a folded FIP1
    cell, over the covariate units of a cell with several, or, for a cell with one
    unit, ``floor_sd`` if given, else over its blocks (from its per-unit file). 0
    when the summary records none. A folded cell of one run has no seed spread and
    takes the standard error of its units."""
    row = pl.read_csv(summary_path).row(0, named=True)
    if row.get("n_seeds") is not None and row.get("sd") is not None:
        return float(row["sd"]) / math.sqrt(int(row["n_seeds"]))
    if row.get("n_units") is None or row.get("std") is None:
        return 0.0
    if int(row["n_units"]) > 1:
        return float(row["std"]) / math.sqrt(int(row["n_units"]))
    if floor_sd is not None:
        return floor_sd
    per_unit = summary_path.with_name(summary_path.name.replace("summary", "per_unit"))
    if not per_unit.exists():
        return 0.0
    return float(pl.read_csv(per_unit)["score_se"][0])


def parse_floors(raw: Sequence[str], option: str = "--floors") -> dict[str, float]:
    """``SPLIT=VALUE`` pairs in table units as metric values by split."""
    floors = {}
    for item in raw:
        split, separator, value = item.partition("=")
        if not separator:
            msg = f"{option} expects SPLIT=VALUE, got {item!r}."
            raise ValueError(msg)
        floors[split] = float(value) / LATEX_VALUE_SCALE
    return floors


def _resolved_summary(
    resolved: _ResolvedSummaryPath,
    metric: str,
    floors: Mapping[str, float],
    floor_sds: Mapping[str, float],
) -> ResolvedSummary:
    value = _read_metric(resolved.summary_path, metric)
    return ResolvedSummary(
        spec=resolved.spec,
        prediction_method=resolved.prediction_method,
        checkpoint=resolved.checkpoint,
        view=resolved.view,
        summary_path=resolved.summary_path,
        value=value,
        standard_error=_read_standard_error(
            resolved.summary_path, floor_sds.get(resolved.spec.split)
        ),
        floored_value=max(value, floors.get(resolved.spec.split, 0.0)),
    )


def _resolve_summary_paths_for_methods(
    results_base: Path,
    specs: list[TableSpec],
    prediction_methods: list[str],
    metric_type: str,
    checkpoint: str,
    view: str,
    prediction_method_views: Mapping[str, str] | None = None,
) -> list[_ResolvedSummaryPath]:
    effective_views = resolve_prediction_method_views(
        prediction_methods,
        metric_type=metric_type,
        default_view=view,
        raw_overrides=prediction_method_views,
    )
    common_specs = [spec for spec in specs if spec.checkpoint is None]
    common_checkpoint = (
        resolve_common_checkpoint_for_methods(
            results_base,
            common_specs,
            prediction_methods,
            metric_type,
            checkpoint,
            view,
            effective_views,
        )
        if common_specs
        else normalize_checkpoint(checkpoint)
    )
    lookup_failures: list[_SummaryLookupFailure] = []
    resolved_paths: list[_ResolvedSummaryPath] = []
    for prediction_method in prediction_methods:
        method_view = prediction_method_view(effective_views, prediction_method)
        for spec in specs:
            resolved_checkpoint = spec.checkpoint or common_checkpoint
            paths = _find_summary_paths(
                results_base,
                spec,
                prediction_method,
                metric_type,
                resolved_checkpoint,
                method_view,
            )
            if not paths:
                lookup_failures.append(
                    _summary_lookup_failure(
                        "missing",
                        results_base,
                        spec,
                        prediction_method,
                        metric_type,
                        resolved_checkpoint,
                        method_view,
                        paths,
                    )
                )
                continue
            if len(paths) > 1:
                lookup_failures.append(
                    _summary_lookup_failure(
                        "ambiguous",
                        results_base,
                        spec,
                        prediction_method,
                        metric_type,
                        resolved_checkpoint,
                        method_view,
                        paths,
                    )
                )
                continue
            resolved_paths.append(
                _ResolvedSummaryPath(
                    spec=spec,
                    prediction_method=prediction_method,
                    view=method_view,
                    summary_path=paths[0],
                    checkpoint=resolved_checkpoint,
                )
            )

    _raise_summary_lookup_failures(lookup_failures)
    return resolved_paths


def load_summaries(
    results_base: Path,
    specs: list[TableSpec],
    *,
    prediction_method: str,
    metric_type: str,
    checkpoint: str,
    metric: str,
    view: str = DEFAULT_VIEW,
    floors: Mapping[str, float] | None = None,
    floor_sds: Mapping[str, float] | None = None,
) -> list[ResolvedSummary]:
    view = validate_metric_view(metric_type, view)
    resolved_paths = _resolve_summary_paths_for_methods(
        results_base, specs, [prediction_method], metric_type, checkpoint, view
    )
    return [
        _resolved_summary(resolved, metric, floors or {}, floor_sds or {})
        for resolved in resolved_paths
    ]


def build_long_dataframe(rows: list[ResolvedSummary], metric: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "model": row.spec.model,
                "model_name": row.spec.model_name,
                "split": row.spec.split,
                "split_label": row.spec.split_label,
                "conditioning": row.spec.conditioning,
                "conditioning_label": row.spec.conditioning_label,
                "prediction_method": row.prediction_method,
                "checkpoint": row.checkpoint,
                "view": row.view,
                metric: row.value,
                "summary_path": str(row.summary_path),
            }
            for row in rows
        ]
    )


def build_wide_dataframe(rows: list[ResolvedSummary]) -> pl.DataFrame:
    model_order = []
    by_model: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.spec.model not in by_model:
            model_order.append(row.spec.model)
            by_model[row.spec.model] = {
                "model": row.spec.model,
                "model_name": row.spec.model_name,
            }
        column = f"{row.spec.split}__{row.spec.conditioning}"
        by_model[row.spec.model][column] = row.value
    return pl.DataFrame([by_model[model] for model in model_order])


def load_summaries_for_methods(
    results_base: Path,
    specs: list[TableSpec],
    *,
    prediction_methods: list[str],
    metric_type: str,
    checkpoint: str,
    metric: str,
    view: str = DEFAULT_VIEW,
    prediction_method_views: Mapping[str, str] | None = None,
    floors: Mapping[str, float] | None = None,
    floor_sds: Mapping[str, float] | None = None,
) -> list[ResolvedSummary]:
    view = validate_metric_view(metric_type, view)
    resolved_paths = _resolve_summary_paths_for_methods(
        results_base,
        specs,
        prediction_methods,
        metric_type,
        checkpoint,
        view,
        prediction_method_views,
    )
    return [
        _resolved_summary(resolved, metric, floors or {}, floor_sds or {})
        for resolved in resolved_paths
    ]


def average_over_splits(
    rows: list[ResolvedSummary],
    *,
    prediction_methods: list[str],
    prediction_method_labels: list[str],
    conditionings: list[str],
    splits: list[str],
) -> list[AveragedSummary]:
    row_by_key = {
        (
            row.spec.model,
            row.prediction_method,
            row.spec.split,
            row.spec.conditioning,
        ): row
        for row in rows
    }
    prediction_label_by_method = dict(
        zip(prediction_methods, prediction_method_labels, strict=True)
    )

    model_order = []
    model_labels: dict[str, str] = {}
    for row in rows:
        if row.spec.model not in model_labels:
            model_order.append(row.spec.model)
            model_labels[row.spec.model] = row.spec.model_name

    averaged_rows: list[AveragedSummary] = []
    for model in model_order:
        for prediction_method in prediction_methods:
            for conditioning in conditionings:
                source_rows = [
                    row_by_key[(model, prediction_method, split, conditioning)]
                    for split in splits
                ]
                first = source_rows[0]
                averaged_rows.append(
                    AveragedSummary(
                        model=model,
                        model_name=model_labels[model],
                        prediction_method=prediction_method,
                        prediction_method_label=prediction_label_by_method[
                            prediction_method
                        ],
                        conditioning=conditioning,
                        conditioning_label=first.spec.conditioning_label,
                        checkpoint=first.checkpoint,
                        view=first.view,
                        value=sum(row.value for row in source_rows) / len(source_rows),
                        source_splits=tuple(row.spec.split for row in source_rows),
                        source_split_labels=tuple(
                            row.spec.split_label for row in source_rows
                        ),
                        summary_paths=tuple(row.summary_path for row in source_rows),
                        standard_error=math.hypot(
                            *(row.standard_error for row in source_rows)
                        )
                        / len(source_rows),
                        floored_value=sum(_floored(row) for row in source_rows)
                        / len(source_rows),
                    )
                )
    return averaged_rows


def build_averaged_long_dataframe(
    rows: list[AveragedSummary], metric: str
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "model": row.model,
                "model_name": row.model_name,
                "prediction_method": row.prediction_method,
                "prediction_method_label": row.prediction_method_label,
                "conditioning": row.conditioning,
                "conditioning_label": row.conditioning_label,
                "checkpoint": row.checkpoint,
                "view": row.view,
                "source_splits": ",".join(row.source_splits),
                "source_split_labels": ",".join(row.source_split_labels),
                "n_splits": len(row.source_splits),
                metric: row.value,
                "summary_paths": ";".join(str(path) for path in row.summary_paths),
            }
            for row in rows
        ]
    )


def build_averaged_wide_dataframe(rows: list[AveragedSummary]) -> pl.DataFrame:
    model_order = []
    by_model: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.model not in by_model:
            model_order.append(row.model)
            by_model[row.model] = {"model": row.model, "model_name": row.model_name}
        column = f"{row.prediction_method}__{row.conditioning}"
        by_model[row.model][column] = row.value
    return pl.DataFrame([by_model[model] for model in model_order])


def averaged_as_latex_rows(rows: list[AveragedSummary]) -> list[ResolvedSummary]:
    return [
        ResolvedSummary(
            spec=TableSpec(
                model=row.model,
                model_name=row.model_name,
                split=row.prediction_method,
                split_label=row.prediction_method_label,
                conditioning=row.conditioning,
                conditioning_label=row.conditioning_label,
            ),
            prediction_method=row.prediction_method,
            checkpoint=row.checkpoint,
            view=row.view,
            summary_path=row.summary_paths[0],
            value=row.value,
            standard_error=row.standard_error,
            floored_value=row.floored_value,
        )
        for row in rows
    ]


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _compute_cell_color(
    value: float,
    reference_value: float,
    max_positive_delta: float,
    max_negative_delta: float,
    n_levels: int = 10,
    max_color_pct: int = 60,
) -> str:
    """Return a LaTeX cellcolor command based on deviation from reference.

    Red (worse) and green (better) are independently scaled:
    red intensity = delta / max_positive_delta
    green intensity = |delta| / max_negative_delta
    """
    delta = value - reference_value
    if delta == 0:
        return ""
    if delta > 0:
        if max_positive_delta == 0:
            return ""
        intensity = min(delta / max_positive_delta, 1.0)
    else:
        if max_negative_delta == 0:
            return ""
        intensity = min(abs(delta) / max_negative_delta, 1.0)
    level = max(1, round(intensity * n_levels)) * (max_color_pct // n_levels)
    if delta < 0:
        return rf"\cellcolor{{green!{level}}}"
    return rf"\cellcolor{{red!{level}}}"


@dataclass(frozen=True)
class _ReferenceColors:
    reference_model_key: str
    ref_values: dict[tuple[str, str], float]
    max_positive_delta: float
    max_negative_delta: float


def _resolve_reference_colors(
    rows: list[ResolvedSummary],
    row_by_key: dict[tuple[str, str, str], ResolvedSummary],
    *,
    splits: list[str],
    conditionings: list[str],
    reference_model: str,
    model_labels: dict[str, str],
) -> _ReferenceColors:
    for model, name in model_labels.items():
        if name == reference_model:
            reference_model_key = model
            break
    else:
        available = sorted(set(model_labels.values()))
        msg = (
            f"--reference-model {reference_model!r} not found in "
            f"--model-names. Available: {available}"
        )
        raise ValueError(msg)

    ref_values: dict[tuple[str, str], float] = {}
    max_positive_delta = 0.0
    max_negative_delta = 0.0
    for split in splits:
        for conditioning in conditionings:
            key = (split, conditioning)
            ref_val = row_by_key[(reference_model_key, split, conditioning)].value
            ref_values[key] = ref_val
            for row in rows:
                if row.spec.split == split and row.spec.conditioning == conditioning:
                    delta = row.value - ref_val
                    max_positive_delta = max(delta, max_positive_delta)
                    if delta < 0 and abs(delta) > max_negative_delta:
                        max_negative_delta = abs(delta)
    return _ReferenceColors(
        reference_model_key=reference_model_key,
        ref_values=ref_values,
        max_positive_delta=max_positive_delta,
        max_negative_delta=max_negative_delta,
    )


def build_latex_table(
    rows: list[ResolvedSummary],
    *,
    splits: list[str],
    conditionings: list[str],
    decimals: int,
    metric_type: str,
    reference_model: str | None = None,
) -> str:
    c = _mtc(metric_type)
    split_labels = _ordered_labels(rows, "split", splits)
    conditioning_labels = _ordered_labels(rows, "conditioning", conditionings)
    row_by_key = _row_by_key(rows)
    tied_by_column = _tied_by_column(rows, splits, conditionings)
    model_labels = _model_labels(rows)
    model_order = list(model_labels)

    ref_colors: _ReferenceColors | None = None
    if reference_model is not None:
        ref_colors = _resolve_reference_colors(
            rows,
            row_by_key,
            splits=splits,
            conditionings=conditionings,
            reference_model=reference_model,
            model_labels=model_labels,
        )

    n_metric_columns = len(splits) * len(conditionings)
    preamble_lines = []
    if ref_colors is not None:
        preamble_lines.append("% Add to your preamble: \\usepackage[table]{xcolor}")
        preamble_lines.append("")
    preamble = "\n".join(preamble_lines)

    lines = [f"\\begin{{tabular}}{{l{'r' * n_metric_columns}}}", r"\toprule"]
    lines.append(
        " & ".join(
            ["", f"\\multicolumn{{{n_metric_columns}}}{{c}}{{{c['latex_header']}}}"]
        )
        + r" \\"
    )

    grouped_headers = ["Model"]
    grouped_headers.extend(
        f"\\multicolumn{{{len(conditionings)}}}{{c}}{{{latex_escape(label)}}}"
        for label in split_labels
    )
    lines.append(" & ".join(grouped_headers) + r" \\")

    nested_headers = [""]
    for _split in splits:
        nested_headers.extend(latex_escape(label) for label in conditioning_labels)
    lines.append(" & ".join(nested_headers) + r" \\")
    lines.append(r"\midrule")

    for model in model_order:
        values = [latex_escape(model_labels[model])]
        for split in splits:
            for conditioning in conditionings:
                row = row_by_key[(model, split, conditioning)]
                value = f"{row.value * LATEX_VALUE_SCALE:.{decimals}f}"
                if model in tied_by_column[(split, conditioning)]:
                    value = rf"\textbf{{{value}}}"
                if ref_colors is not None and model != ref_colors.reference_model_key:
                    color_cmd = _compute_cell_color(
                        row.value,
                        ref_colors.ref_values[(split, conditioning)],
                        ref_colors.max_positive_delta,
                        ref_colors.max_negative_delta,
                    )
                    if color_cmd:
                        value = rf"{color_cmd} {value}"
                values.append(value)
        lines.append(" & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return preamble + "\n".join(lines) + "\n"


def build_markdown_table(
    rows: list[ResolvedSummary],
    *,
    splits: list[str],
    conditionings: list[str],
    decimals: int,
) -> str:
    """The LaTeX table's values as Markdown: the same bold, no shading."""
    row_by_key = _row_by_key(rows)
    tied_by_column = _tied_by_column(rows, splits, conditionings)
    split_labels = _ordered_labels(rows, "split", splits)
    conditioning_labels = _ordered_labels(rows, "conditioning", conditionings)
    header = ["Model"] + [
        f"{split_label} {conditioning_label}"
        for split_label in split_labels
        for conditioning_label in conditioning_labels
    ]
    body = []
    for model, model_label in _model_labels(rows).items():
        cells = [model_label]
        for split in splits:
            for conditioning in conditionings:
                row = row_by_key[(model, split, conditioning)]
                value = f"{row.value * LATEX_VALUE_SCALE:.{decimals}f}"
                if model in tied_by_column[(split, conditioning)]:
                    value = f"**{value}**"
                cells.append(value)
        body.append(cells)
    return markdown_table(header, body)


def _row_by_key(
    rows: list[ResolvedSummary],
) -> dict[tuple[str, str, str], ResolvedSummary]:
    return {
        (row.spec.model, row.spec.split, row.spec.conditioning): row for row in rows
    }


def _floored(row: ResolvedSummary) -> float:
    return row.value if row.floored_value is None else row.floored_value


def _tied_by_column(
    rows: list[ResolvedSummary], splits: list[str], conditionings: list[str]
) -> dict[tuple[str, str], set[str]]:
    """Models to bold per column: the lowest value and every value that does not
    differ from it. A value below the floor counts as the floor, and two values
    differ when the difference exceeds twice its standard error."""
    tied = {}
    for split in splits:
        for conditioning in conditionings:
            column = [
                row
                for row in rows
                if row.spec.split == split and row.spec.conditioning == conditioning
            ]
            best = min(column, key=_floored)
            tied[(split, conditioning)] = {
                row.spec.model
                for row in column
                if _floored(row) - _floored(best)
                <= 2 * math.hypot(row.standard_error, best.standard_error)
            }
    return tied


def _model_labels(rows: list[ResolvedSummary]) -> dict[str, str]:
    """Model label by model, in the order of first appearance."""
    labels: dict[str, str] = {}
    for row in rows:
        labels.setdefault(row.spec.model, row.spec.model_name)
    return labels


def _ordered_labels(
    rows: list[ResolvedSummary], field: str, values: list[str]
) -> list[str]:
    labels: dict[str, str] = {}
    for row in rows:
        if field == "split":
            labels[row.spec.split] = row.spec.split_label
        elif field == "conditioning":
            labels[row.spec.conditioning] = row.spec.conditioning_label
        else:
            msg = f"Unsupported label field: {field}"
            raise ValueError(msg)
    return [labels[value] for value in values]


def output_stem(
    prediction_method: str, metric_type: str, view: str = DEFAULT_VIEW
) -> str:
    safe_method = re.sub(r"[^A-Za-z0-9_-]+", "_", prediction_method)
    return f"{metric_stem_prefix(metric_type, view)}_{safe_method}"


def output_stem_for_methods(
    prediction_methods: list[str],
    metric_type: str,
    view: str = DEFAULT_VIEW,
    prediction_method_views: Mapping[str, str] | None = None,
) -> str:
    safe_methods = [
        re.sub(r"[^A-Za-z0-9_-]+", "_", method) for method in prediction_methods
    ]
    effective_views = resolve_prediction_method_views(
        prediction_methods,
        metric_type=metric_type,
        default_view=view,
        raw_overrides=prediction_method_views,
    )
    stem_prefix = metric_stem_prefix_for_prediction_method_views(
        metric_type, effective_views
    )
    return f"{stem_prefix}_{'_'.join(safe_methods)}_avg_splits"


def write_outputs(
    rows: list[ResolvedSummary],
    *,
    metric: str,
    metric_type: str,
    view: str,
    splits: list[str],
    conditionings: list[str],
    decimals: int,
    output_folder: Path,
    prediction_method: str,
    reference_model: str | None = None,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    stem = output_stem(prediction_method, metric_type, view)

    long_path = output_folder / f"{stem}_long.csv"
    wide_path = output_folder / f"{stem}_wide.csv"
    tex_path = output_folder / f"{stem}.tex"
    markdown_path = output_folder / f"{stem}.md"

    build_long_dataframe(rows, metric).write_csv(long_path)
    build_wide_dataframe(rows).write_csv(wide_path)
    tex_path.write_text(
        build_latex_table(
            rows,
            splits=splits,
            conditionings=conditionings,
            decimals=decimals,
            metric_type=metric_type,
            reference_model=reference_model,
        )
    )
    markdown_path.write_text(
        build_markdown_table(
            rows, splits=splits, conditionings=conditionings, decimals=decimals
        )
        + "\n"
    )

    logger.info("Wrote {}", long_path)
    logger.info("Wrote {}", wide_path)
    logger.info("Wrote {}", tex_path)
    logger.info("Wrote {}", markdown_path)


def write_averaged_outputs(
    rows: list[AveragedSummary],
    *,
    metric: str,
    metric_type: str,
    view: str,
    prediction_methods: list[str],
    conditionings: list[str],
    decimals: int,
    output_folder: Path,
    prediction_method_views: Mapping[str, str] | None = None,
    reference_model: str | None = None,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    stem = output_stem_for_methods(
        prediction_methods, metric_type, view, prediction_method_views
    )

    long_path = output_folder / f"{stem}_long.csv"
    wide_path = output_folder / f"{stem}_wide.csv"
    tex_path = output_folder / f"{stem}.tex"
    markdown_path = output_folder / f"{stem}.md"

    build_averaged_long_dataframe(rows, metric).write_csv(long_path)
    build_averaged_wide_dataframe(rows).write_csv(wide_path)
    table_rows = averaged_as_latex_rows(rows)
    tex_path.write_text(
        build_latex_table(
            table_rows,
            splits=prediction_methods,
            conditionings=conditionings,
            decimals=decimals,
            metric_type=metric_type,
            reference_model=reference_model,
        )
    )
    markdown_path.write_text(
        build_markdown_table(
            table_rows,
            splits=prediction_methods,
            conditionings=conditionings,
            decimals=decimals,
        )
        + "\n"
    )

    logger.info("Wrote {}", long_path)
    logger.info("Wrote {}", wide_path)
    logger.info("Wrote {}", tex_path)
    logger.info("Wrote {}", markdown_path)


def generate_table(args: argparse.Namespace) -> None:
    if args.decimals < 0:
        msg = "--decimals must be >= 0."
        raise ValueError(msg)

    if args.prediction_method_names is not None and args.prediction_methods is None:
        msg = "--prediction-method-names requires --prediction-methods."
        raise ValueError(msg)

    metric_type = args.metric_type
    view = validate_metric_view(metric_type, args.view)
    c = _mtc(metric_type)

    specs = validate_and_build_specs(args)
    results_base = Path(args.results_base)
    checkpoint = normalize_checkpoint(args.checkpoint)
    splits = resolve_splits(args)
    conditionings = list(args.conditionings)
    output_folder = (
        Path(args.output_folder)
        if args.output_folder is not None
        else Path(c["default_output_folder"])
    )

    if args.prediction_methods is not None:
        prediction_methods = list(args.prediction_methods)
        prediction_method_views = resolve_prediction_method_views(
            prediction_methods,
            metric_type=metric_type,
            default_view=view,
            raw_overrides=args.prediction_method_views,
        )
        prediction_method_labels = resolve_prediction_method_labels(
            prediction_methods, args.prediction_method_names
        )
        rows = load_summaries_for_methods(
            results_base,
            specs,
            prediction_methods=prediction_methods,
            metric_type=metric_type,
            checkpoint=checkpoint,
            metric=args.metric,
            view=view,
            prediction_method_views=prediction_method_views,
            floors=parse_floors(args.floors),
            floor_sds=parse_floors(args.floor_sds, "--floor-sds"),
        )
        averaged_rows = average_over_splits(
            rows,
            prediction_methods=prediction_methods,
            prediction_method_labels=prediction_method_labels,
            conditionings=conditionings,
            splits=splits,
        )
        write_averaged_outputs(
            averaged_rows,
            metric=args.metric,
            metric_type=metric_type,
            view=view,
            prediction_methods=prediction_methods,
            conditionings=conditionings,
            decimals=args.decimals,
            output_folder=output_folder,
            prediction_method_views=prediction_method_views,
            reference_model=args.reference_model,
        )

        checkpoints = sorted({row.checkpoint for row in rows})
        logger.info(
            "Built {} {} table for prediction_methods={}, "
            "prediction_method_views={}, metric={}, checkpoint={}, "
            "averaged_splits={}",
            view,
            c["label"],
            prediction_methods,
            format_prediction_method_views(prediction_method_views, prediction_methods),
            args.metric,
            checkpoints,
            splits,
        )
        return

    prediction_method_views = resolve_prediction_method_views(
        [args.prediction_method],
        metric_type=metric_type,
        default_view=view,
        raw_overrides=args.prediction_method_views,
    )
    effective_view = prediction_method_view(
        prediction_method_views, args.prediction_method
    )
    rows = load_summaries(
        results_base,
        specs,
        prediction_method=args.prediction_method,
        metric_type=metric_type,
        checkpoint=checkpoint,
        metric=args.metric,
        view=effective_view,
        floors=parse_floors(args.floors),
        floor_sds=parse_floors(args.floor_sds, "--floor-sds"),
    )
    write_outputs(
        rows,
        metric=args.metric,
        metric_type=metric_type,
        view=effective_view,
        splits=splits,
        conditionings=conditionings,
        decimals=args.decimals,
        output_folder=output_folder,
        prediction_method=args.prediction_method,
        reference_model=args.reference_model,
    )

    checkpoints = sorted({row.checkpoint for row in rows})
    logger.info(
        "Built {} {} table for prediction_method={}, metric={}, checkpoint={}",
        effective_view,
        c["label"],
        args.prediction_method,
        args.metric,
        checkpoints,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        generate_table(args)
    except (FileNotFoundError, ValueError) as error:
        msg = f"error: {error}"
        raise SystemExit(msg) from error


if __name__ == "__main__":
    main()
