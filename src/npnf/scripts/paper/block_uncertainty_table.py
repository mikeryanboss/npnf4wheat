"""Paired per-block uncertainty for Sig-MMD / CSig-MMD model differences.

Every blocked Sig-MMD / CSig-MMD score is the mean over per-block scores that the
metric scripts persist next to the summary CSV. Those block scores are the stored
sampling distribution of the estimator, so the uncertainty of a reported score is
recoverable without re-running any evaluation.

Models within a cell are scored against the same oracle draws under the same block
partition (the block RNG in `blocks.build_unit_blocks` is seeded from the
unit, not the model), so their block scores are correlated. Differences between
models are therefore paired on the blocking unit before the standard error is taken.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger

from npnf.data.configs.datasets.synthetic_test_sets import SyntheticTestSet, split_named
from npnf.scripts.paper.mmd_table import (
    DEFAULT_CONDITIONING_LABELS,
    DEFAULT_CONDITIONINGS,
    add_test_set_arguments,
    latex_escape,
    metric_dir_name,
    metric_stem_prefix,
    metric_type_choices,
    metric_type_config,
    resolve_splits,
)

BLOCK_KEYS = ["unit_scope", "unit_id", "block_index"]
BLOCK_VIEWS = ("context_matched_blocked",)
DEFAULT_VIEW = "context_matched_blocked"
DEFAULT_CHECKPOINT = "checkpoint-3000000"
DEFAULT_PREDICTION_METHODS = ["no_context", "random_context", "max_height"]
DEFAULT_MODELS = [
    f"{base}{variant}-512k-training3m_set_mode_nested_noprior"
    for base in ("LNP", "ANP")
    for variant in ("", "-NF-Prior", "-NF-Posterior", "-NF-Prior-Posterior")
]
DEFAULT_MODEL_NAMES = [
    f"{base}{variant}"
    for base in ("LNP", "ANP")
    for variant in ("", "-NF-Prior", "-NF-Posterior", "-NF-Prior-Posterior")
]
DEFAULT_BASELINE_MODEL = "CNP-512k-training3m_set_mode_nested_noprior"
DEFAULT_BASELINE_LABEL = "CNP"
DEFAULT_DECIMALS = 1
RESOLUTION_THRESHOLD = 2.0


@dataclass(frozen=True)
class PairStat:
    """Paired difference between two models in one cell."""

    model_a: str
    model_b: str
    split: str
    conditioning: str
    prediction_method: str
    n_blocks: int
    mean_a: float
    mean_b: float
    mean_diff: float
    standard_error: float
    t_value: float
    correlation: float
    relative_diff: float | None


def block_file_name(metric_type: str) -> str:
    return f"{metric_type_config(metric_type)['stem_prefix']}_blocks.csv"


def validate_block_view(view: str) -> str:
    if view not in BLOCK_VIEWS:
        msg = (
            f"Unknown --view {view!r}. "
            f"Views with stored block scores: {list(BLOCK_VIEWS)}"
        )
        raise ValueError(msg)
    return view


def find_block_path(
    results_base: Path,
    model: str,
    split: str,
    conditioning: str,
    prediction_method: str,
    metric_type: str,
    view: str,
    checkpoint: str,
    *,
    test_set: str = SyntheticTestSet.LEGACY.value,
) -> Path | None:
    """Locate a cell's block CSV, or None when the cell was never scored.

    The dataset-size path component varies between runs, so the search mirrors
    `mmd_table._find_summary_paths` and globs over it rather than reconstructing it.
    """
    test_split = split_named(split, test_set)
    root = results_base / test_split.dataloader_name / model
    pattern = (
        f"*/{checkpoint}/{test_split.split_key}/{conditioning}/{prediction_method}/"
        f"{metric_dir_name(metric_type, view)}/{block_file_name(metric_type)}"
    )
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple block files matched {}; using {}", root / pattern, matches[0]
        )
    return matches[0]


def read_block_scores(path: Path) -> pl.DataFrame:
    """Read a cell's model-side block scores keyed by their blocking unit.

    The unit scope follows the conditioning rather than being constant: unconditioned
    cells block globally, while genotype-, environment- and fully conditioned cells
    block by genotype, environment and condition respectively. Keying on the full
    (scope, unit, block) triple therefore compares like with like in every cell.

    CSig-MMD block files also carry oracle-side rows; only the model side is scored.
    """
    frame = pl.read_csv(path)
    if "source" in frame.columns:
        frame = frame.filter(pl.col("source") == "model")
    return frame.select([*BLOCK_KEYS, "score"]).sort(BLOCK_KEYS)


def read_baseline_mean(
    results_base: Path,
    baseline_model: str,
    split: str,
    conditioning: str,
    prediction_method: str,
    metric_type: str,
    view: str,
    checkpoint: str,
    *,
    test_set: str = SyntheticTestSet.LEGACY.value,
) -> float | None:
    """Mean block score of the baseline model in one cell, or None if unscored.

    Absolute Sig-MMD carries no interpretable scale, so gaps are reported relative to
    a fixed reference model. Reading it per cell keeps every pair in a cell on the
    same denominator without letting cell-to-cell scale differences leak in.
    """
    path = find_block_path(
        results_base,
        baseline_model,
        split,
        conditioning,
        prediction_method,
        metric_type,
        view,
        checkpoint,
        test_set=test_set,
    )
    if path is None:
        logger.warning(
            "No {} block file for baseline model={} split={} "
            "conditioning={} method={}; relative gaps unavailable for this cell",
            view,
            baseline_model,
            split,
            conditioning,
            prediction_method,
        )
        return None
    return float(read_block_scores(path)["score"].mean())  # ty: ignore[invalid-argument-type]


def paired_stat(
    model_a: str,
    model_b: str,
    split: str,
    conditioning: str,
    prediction_method: str,
    scores_a: pl.DataFrame,
    scores_b: pl.DataFrame,
    mean_baseline: float | None,
) -> PairStat | None:
    """Pair two models' block scores on their blocking unit and describe the gap.

    Returns None when the two cells share fewer than two blocks, which leaves the
    standard error undefined.

    `mean_baseline` is the baseline model's mean block score in this same cell and
    is shared by every pair in the cell, so the relative gaps of different pairs are
    comparable. It is None when the baseline was not scored for the cell.
    """
    joined = scores_a.join(scores_b, on=BLOCK_KEYS, how="inner", suffix="_b")
    differences = (joined["score"] - joined["score_b"]).to_numpy()
    n_blocks = len(differences)
    if n_blocks < 2:
        return None
    standard_error = float(differences.std(ddof=1) / n_blocks**0.5)
    mean_diff = float(differences.mean())
    # Equal differences in every block (e.g. two models whose draws are all censored
    # to the zero path) have no spread: no gap is unresolved, any other gap is exact.
    t_value = (
        mean_diff / standard_error
        if standard_error > 0
        else math.copysign(math.inf, mean_diff)
        if mean_diff != 0
        else 0.0
    )
    correlation = float(
        pl.DataFrame({"a": joined["score"], "b": joined["score_b"]})
        .select(pl.corr("a", "b"))
        .item()
    )
    relative_diff = (
        None
        if mean_baseline is None or mean_baseline == 0.0
        else mean_diff / mean_baseline
    )
    return PairStat(
        model_a=model_a,
        model_b=model_b,
        split=split,
        conditioning=conditioning,
        prediction_method=prediction_method,
        n_blocks=n_blocks,
        mean_a=float(joined["score"].mean()),  # ty: ignore[invalid-argument-type]
        mean_b=float(joined["score_b"].mean()),  # ty: ignore[invalid-argument-type]
        mean_diff=mean_diff,
        standard_error=standard_error,
        t_value=t_value,
        correlation=correlation,
        relative_diff=relative_diff,
    )


def collect_pair_stats(
    results_base: Path,
    models: Sequence[str],
    model_labels: dict[str, str],
    splits: Sequence[str],
    conditionings: Sequence[str],
    prediction_method: str,
    metric_type: str,
    view: str,
    checkpoint: str,
    baseline_model: str,
    *,
    test_set: str = SyntheticTestSet.LEGACY.value,
) -> tuple[list[PairStat], list[str]]:
    """Compute every model-pair statistic for one prediction method.

    Returns the statistics and a list of cells that were skipped for missing data.

    `baseline_model` is read once per cell and supplies the shared denominator for
    every pair's relative gap. It is not itself compared against the other models.
    """
    stats: list[PairStat] = []
    missing: list[str] = []
    for split in splits:
        for conditioning in conditionings:
            mean_baseline = read_baseline_mean(
                results_base,
                baseline_model,
                split,
                conditioning,
                prediction_method,
                metric_type,
                view,
                checkpoint,
                test_set=test_set,
            )
            if mean_baseline is None:
                missing.append(f"baseline {baseline_model} / {split} / {conditioning}")
            scores: dict[str, pl.DataFrame] = {}
            for model in models:
                path = find_block_path(
                    results_base,
                    model,
                    split,
                    conditioning,
                    prediction_method,
                    metric_type,
                    view,
                    checkpoint,
                    test_set=test_set,
                )
                if path is None:
                    missing.append(f"{model} / {split} / {conditioning}")
                    logger.warning(
                        "No {} block file for model={} split={} "
                        "conditioning={} method={}",
                        view,
                        model,
                        split,
                        conditioning,
                        prediction_method,
                    )
                    continue
                scores[model] = read_block_scores(path)
            for model_a, model_b in itertools.combinations(models, 2):
                if model_a not in scores or model_b not in scores:
                    continue
                stat = paired_stat(
                    model_labels[model_a],
                    model_labels[model_b],
                    split,
                    conditioning,
                    prediction_method,
                    scores[model_a],
                    scores[model_b],
                    mean_baseline,
                )
                if stat is None:
                    missing.append(f"{model_a} vs {model_b} / {split} / {conditioning}")
                    logger.warning(
                        "Fewer than two shared blocks for {} vs {} "
                        "split={} conditioning={} method={}",
                        model_a,
                        model_b,
                        split,
                        conditioning,
                        prediction_method,
                    )
                    continue
                stats.append(stat)
    return stats, missing


def build_long_dataframe(stats: Sequence[PairStat]) -> pl.DataFrame:
    return pl.DataFrame([vars(stat) for stat in stats])


def build_pair_summary(stats: Sequence[PairStat]) -> pl.DataFrame:
    """Aggregate every cell of every prediction method into one row per model pair.

    A pair whose gap sits below the resolution threshold in more than half its cells
    is unresolved by evaluation noise; `median_abs_t` says how far the typical cell
    is from that threshold and `median_abs_relative_diff` how large the gap is.
    """
    return (
        build_long_dataframe(stats)
        .group_by(["model_a", "model_b"])
        .agg(
            pl.len().alias("n_cells"),
            pl.col("t_value").abs().median().alias("median_abs_t"),
            (pl.col("t_value").abs() < RESOLUTION_THRESHOLD).sum().alias("n_weak"),
            pl.col("relative_diff").abs().median().alias("median_abs_relative_diff"),
        )
        .with_columns((pl.col("n_weak") / pl.col("n_cells")).alias("frac_weak"))
        .sort("median_abs_t")
    )


def build_summary_latex(
    summary: pl.DataFrame, metric_type: str, baseline_label: str, decimals: int
) -> str:
    """Render the per-pair summary, marking pairs unresolved in most of their cells."""
    label = metric_type_config(metric_type)["label"]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        f" & \\multicolumn{{4}}{{c}}{{{label} differences across all cells}}" + r" \\",
        (
            "Pair & Cells & Median $|t|$ & $|t| < 2$ & Median gap "
            f"(\\% of {latex_escape(baseline_label)}) \\\\"
        ),
        r"\midrule",
    ]
    for row in summary.iter_rows(named=True):
        pair = latex_escape(f"{row['model_a']} vs {row['model_b']}")
        if row["frac_weak"] > 0.5:
            pair = f"\\textbf{{{pair}}}"
        relative = row["median_abs_relative_diff"]
        gap = "--" if relative is None else f"{100 * relative:.{decimals}f}"
        lines.append(
            f"{pair} & {row['n_cells']} & {row['median_abs_t']:.{decimals}f} & "
            f"{row['n_weak']} & {gap} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_latex_table(
    stats: Sequence[PairStat],
    splits: Sequence[str],
    split_labels: Sequence[str],
    conditionings: Sequence[str],
    conditioning_labels: Sequence[str],
    metric_type: str,
    decimals: int,
) -> str:
    """Render paired t values as rows of model pairs by split x conditioning."""
    by_key = {
        (stat.model_a, stat.model_b, stat.split, stat.conditioning): stat
        for stat in stats
    }
    pair_order: list[tuple[str, str]] = []
    for stat in stats:
        if (stat.model_a, stat.model_b) not in pair_order:
            pair_order.append((stat.model_a, stat.model_b))

    n_metric_columns = len(splits) * len(conditionings)
    header = (
        f"Paired $t$ for {metric_type_config(metric_type)['label']} differences "
        "(uncorrected)"
    )
    lines = [f"\\begin{{tabular}}{{l{'r' * n_metric_columns}}}", r"\toprule"]
    lines.append(f" & \\multicolumn{{{n_metric_columns}}}{{c}}{{{header}}}" + r" \\")

    grouped = ["Pair"]
    grouped.extend(
        f"\\multicolumn{{{len(conditionings)}}}{{c}}{{{latex_escape(label)}}}"
        for label in split_labels
    )
    lines.append(" & ".join(grouped) + r" \\")

    nested = [""]
    for _split in splits:
        nested.extend(latex_escape(label) for label in conditioning_labels)
    lines.append(" & ".join(nested) + r" \\")
    lines.append(r"\midrule")

    for model_a, model_b in pair_order:
        cells = [latex_escape(f"{model_a} vs {model_b}")]
        for split in splits:
            for conditioning in conditionings:
                stat = by_key.get((model_a, model_b, split, conditioning))
                cells.append("--" if stat is None else f"{stat.t_value:.{decimals}f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def resolve_labels(models: list[str], names: list[str] | None) -> dict[str, str]:
    if names is None:
        return {model: model for model in models}
    if len(names) != len(models):
        msg = f"--model-names has {len(names)} entries but --models has {len(models)}"
        raise ValueError(msg)
    return dict(zip(models, names, strict=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired per-block uncertainty table for model score differences."
    )
    parser.add_argument("--results-base", default=os.environ["NPNF_RESULTS_DIR"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--model-names", nargs="+", default=DEFAULT_MODEL_NAMES)
    add_test_set_arguments(parser)
    parser.add_argument("--split-labels", nargs="+", default=None)
    parser.add_argument("--conditionings", nargs="+", default=DEFAULT_CONDITIONINGS)
    parser.add_argument(
        "--conditioning-labels", nargs="+", default=DEFAULT_CONDITIONING_LABELS
    )
    parser.add_argument(
        "--prediction-methods", nargs="+", default=DEFAULT_PREDICTION_METHODS
    )
    parser.add_argument(
        "--metric-type", default="sig_mmd", choices=metric_type_choices()
    )
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    parser.add_argument("--view", default=DEFAULT_VIEW, choices=list(BLOCK_VIEWS))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--decimals", type=int, default=DEFAULT_DECIMALS)
    parser.add_argument("--output-dir", default="paper/block_uncertainty")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    view = validate_block_view(args.view)
    results_base = Path(args.results_base)
    model_labels = resolve_labels(args.models, args.model_names)
    splits = resolve_splits(args)
    split_labels = args.split_labels or [
        split_named(split, args.test_set).title for split in splits
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[PairStat] = []
    for prediction_method in args.prediction_methods:
        stats, missing = collect_pair_stats(
            results_base,
            args.models,
            model_labels,
            splits,
            args.conditionings,
            prediction_method,
            args.metric_type,
            view,
            args.checkpoint,
            args.baseline_model,
            test_set=args.test_set,
        )
        if not stats:
            logger.warning(
                "No {} cells found for prediction method {}; skipping table.",
                view,
                prediction_method,
            )
            continue

        prefix = metric_stem_prefix(args.metric_type, view)
        stem = f"{prefix}_block_uncertainty_{prediction_method}"
        long_path = output_dir / f"{stem}_long.csv"
        tex_path = output_dir / f"{stem}.tex"
        build_long_dataframe(stats).write_csv(long_path)
        tex_path.write_text(
            build_latex_table(
                stats,
                splits=splits,
                split_labels=split_labels,
                conditionings=args.conditionings,
                conditioning_labels=args.conditioning_labels,
                metric_type=args.metric_type,
                decimals=args.decimals,
            )
        )
        logger.info("Wrote {}", long_path)
        logger.info("Wrote {}", tex_path)
        unresolved = sum(
            1 for stat in stats if abs(stat.t_value) < RESOLUTION_THRESHOLD
        )
        logger.info(
            "{}: {} pairs across {} cells, {} with |t| < 2, {} missing cells",
            prediction_method,
            len(stats),
            len(splits) * len(args.conditionings),
            unresolved,
            len(missing),
        )
        all_stats.extend(stats)

    if not all_stats:
        return

    prefix = metric_stem_prefix(args.metric_type, view)
    summary = build_pair_summary(all_stats)
    summary_csv = output_dir / f"{prefix}_block_uncertainty_summary.csv"
    summary_tex = output_dir / f"{prefix}_block_uncertainty_summary.tex"
    summary.write_csv(summary_csv)
    summary_tex.write_text(
        build_summary_latex(
            summary,
            metric_type=args.metric_type,
            baseline_label=args.baseline_label,
            decimals=args.decimals,
        )
    )
    logger.info("Wrote {}", summary_csv)
    logger.info("Wrote {}", summary_tex)
    majority_unresolved = summary.filter(pl.col("frac_weak") > 0.5)
    logger.info(
        "summary: {} pairs, {} majority-unresolved",
        summary.height,
        majority_unresolved.height,
    )


if __name__ == "__main__":
    main()
