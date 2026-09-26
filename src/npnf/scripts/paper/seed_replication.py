"""Seed replication of the paper models: spread over seeds and seed-level tests.

Each paper model is trained with several seeds. The seed of the paper run has no
run-name suffix, the other seeds have ``_seed<N>``. For every metric, model, seed
and evaluation cell (test split, conditioning, prediction method), the script
reads the blocked context-matched summary and writes:

- ``seed_scores.csv``: one row per metric, model, seed and cell (score x 1000).
- ``seed_cells.csv``: mean, sd and number of seeds per metric, model and cell.
- ``seed_models.csv``: per metric, model and prediction method, the mean and sd over
  seeds of the score averaged over the cells, the root mean square of the per-cell
  seed sd, and the sd of the per-cell means over the cells.
- ``seed_comparisons.csv``: per metric, model pair and prediction method, the mean
  relative gap, the cells whose Welch t-test over seeds is significant after a Holm
  correction over the cells of the prediction method, and a Welch t-test over seeds
  of the cell-averaged score.
- ``seed_models.md``: the model table as mean ± sd over seeds.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from npnf.scripts.paper import mmd_table


@dataclass(frozen=True)
class Config:
    results_base: Path
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    paper_seed: int
    run_template: str
    splits: tuple[str, ...]
    conditionings: tuple[str, ...]
    prediction_methods: tuple[str, ...]
    metric_types: tuple[str, ...]
    comparisons: tuple[tuple[str, str], ...]
    significance_level: float
    output_dir: Path


def default_models() -> list[str]:
    return [
        "CNP",
        "ACNP",
        "LNP",
        "LNP-NF-Prior",
        "LNP-NF-Posterior",
        "LNP-NF-Prior-Posterior",
        "ANP",
        "ANP-NF-Prior",
        "ANP-NF-Posterior",
        "ANP-NF-Prior-Posterior",
    ]


def default_comparisons() -> list[str]:
    """Pairs ``candidate:reference``: each flow variant against its base model."""
    pairs = [
        f"{base}-NF-{flow}:{base}"
        for base in ("LNP", "ANP")
        for flow in ("Prior", "Posterior", "Prior-Posterior")
    ]
    return [*pairs, "ANP:LNP", "ACNP:CNP"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-base", default=mmd_table.default_results_base())
    parser.add_argument("--models", nargs="+", default=default_models())
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2, 3, 13, 31])
    parser.add_argument(
        "--paper-seed",
        type=int,
        default=42,
        help="Seed of the paper runs, whose run names have no seed suffix.",
    )
    parser.add_argument(
        "--run-template",
        default="{model}-512k-training3m_set_mode_nested_noprior{suffix}",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["plot", "environment", "genotype", "unseen"]
    )
    parser.add_argument(
        "--conditionings",
        nargs="+",
        default=["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"],
    )
    parser.add_argument(
        "--prediction-methods",
        nargs="+",
        default=["no_context", "random_context", "max_height"],
    )
    parser.add_argument("--metric-types", nargs="+", default=["sig_mmd", "csig_mmd"])
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=default_comparisons(),
        help="Model pairs as candidate:reference.",
    )
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/seed_replication")
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    comparisons = tuple(
        (candidate, reference)
        for candidate, _, reference in (
            pair.partition(":") for pair in args.comparisons
        )
    )
    return Config(
        results_base=Path(args.results_base),
        models=tuple(args.models),
        seeds=tuple(args.seeds),
        paper_seed=args.paper_seed,
        run_template=args.run_template,
        splits=tuple(args.splits),
        conditionings=tuple(args.conditionings),
        prediction_methods=tuple(args.prediction_methods),
        metric_types=tuple(args.metric_types),
        comparisons=comparisons,
        significance_level=args.significance_level,
        output_dir=args.output_dir,
    )


def run_name(cfg: Config, model: str, seed: int) -> str:
    suffix = "" if seed == cfg.paper_seed else f"_seed{seed}"
    return cfg.run_template.format(model=model, suffix=suffix)


def read_seed_scores(cfg: Config) -> pd.DataFrame:
    """Read every summary; fails if one is missing."""
    rows = []
    for metric_type in cfg.metric_types:
        for model in cfg.models:
            for seed in cfg.seeds:
                specs = [
                    mmd_table.TableSpec(
                        model=run_name(cfg, model, seed),
                        model_name=model,
                        split=split,
                        split_label=split,
                        conditioning=conditioning,
                        conditioning_label=conditioning,
                    )
                    for split in cfg.splits
                    for conditioning in cfg.conditionings
                ]
                summaries = mmd_table.load_summaries_for_methods(
                    cfg.results_base,
                    specs,
                    prediction_methods=list(cfg.prediction_methods),
                    metric_type=metric_type,
                    checkpoint="checkpoint-3000000",
                    metric="mean",
                    view="context_matched_blocked",
                )
                rows.extend(
                    {
                        "metric_type": metric_type,
                        "model": model,
                        "seed": seed,
                        "split": summary.spec.split,
                        "conditioning": summary.spec.conditioning,
                        "prediction_method": summary.prediction_method,
                        "score": summary.value * 1000.0,
                        "summary_path": str(summary.summary_path),
                    }
                    for summary in summaries
                )
    return pd.DataFrame(rows)


def summarize_cells(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["metric_type", "model", "split", "conditioning", "prediction_method"]
    return scores.groupby(keys, as_index=False, sort=False).agg(
        mean=("score", "mean"),
        sd=("score", lambda series: series.std(ddof=1)),
        num_seeds=("score", "size"),
    )


def seed_averages(scores: pd.DataFrame) -> pd.DataFrame:
    """Score of each seed averaged over the cells of a prediction method."""
    keys = ["metric_type", "model", "prediction_method", "seed"]
    return scores.groupby(keys, as_index=False, sort=False)["score"].mean()


def summarize_models(scores: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["metric_type", "model", "prediction_method"]
    over_seeds = (
        seed_averages(scores)
        .groupby(keys, as_index=False, sort=False)
        .agg(
            mean=("score", "mean"),
            sd_over_seeds=("score", lambda series: series.std(ddof=1)),
            num_seeds=("score", "size"),
        )
    )
    over_cells = cells.groupby(keys, as_index=False, sort=False).agg(
        cell_seed_sd_rms=("sd", lambda series: float(np.sqrt(np.mean(series**2)))),
        sd_over_cells=("mean", lambda series: series.std(ddof=1)),
        num_cells=("mean", "size"),
    )
    return over_seeds.merge(over_cells, on=keys)


def holm_reject(p_values: np.ndarray, significance_level: float) -> np.ndarray:
    """Holm step-down: which hypotheses are rejected at the family-wise level."""
    order = np.argsort(p_values)
    reject = np.zeros(len(p_values), dtype=bool)
    for rank, index in enumerate(order):
        if p_values[index] > significance_level / (len(p_values) - rank):
            break
        reject[index] = True
    return reject


def compare_models(
    scores: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    significance_level: float,
) -> pd.DataFrame:
    cell_keys = ["split", "conditioning"]
    averages = seed_averages(scores)
    rows = []
    methods = scores[["metric_type", "prediction_method"]].drop_duplicates()
    for metric_type, prediction_method in methods.itertuples(index=False):
        group = scores[
            (scores["metric_type"] == metric_type)
            & (scores["prediction_method"] == prediction_method)
        ]
        for candidate, reference in comparisons:
            cell_results = []
            for cell, cell_scores in group.groupby(cell_keys, sort=False):
                candidate_scores = cell_scores.loc[
                    cell_scores["model"] == candidate, "score"
                ].to_numpy()
                reference_scores = cell_scores.loc[
                    cell_scores["model"] == reference, "score"
                ].to_numpy()
                test = stats.ttest_ind(
                    candidate_scores, reference_scores, equal_var=False
                )
                gap = candidate_scores.mean() - reference_scores.mean()
                cell_results.append(
                    (cell, gap, gap / reference_scores.mean(), test.pvalue)
                )
            gaps = np.array([result[1] for result in cell_results])
            relative_gaps = np.array([result[2] for result in cell_results])
            p_values = np.array([result[3] for result in cell_results])
            reject = holm_reject(p_values, significance_level)
            average_scores = averages[
                (averages["metric_type"] == metric_type)
                & (averages["prediction_method"] == prediction_method)
            ]
            average_test = stats.ttest_ind(
                average_scores.loc[average_scores["model"] == candidate, "score"],
                average_scores.loc[average_scores["model"] == reference, "score"],
                equal_var=False,
            )
            rows.append(
                {
                    "metric_type": metric_type,
                    "prediction_method": prediction_method,
                    "candidate": candidate,
                    "reference": reference,
                    "num_cells": len(cell_results),
                    "mean_relative_gap_percent": 100.0 * relative_gaps.mean(),
                    "cells_candidate_lower": int((gaps < 0).sum()),
                    "cells_significant_lower": int((reject & (gaps < 0)).sum()),
                    "cells_significant_higher": int((reject & (gaps > 0)).sum()),
                    "average_t": float(average_test.statistic),
                    "average_p": float(average_test.pvalue),
                }
            )
    return pd.DataFrame(rows)


def model_table_markdown(models: pd.DataFrame, cfg: Config) -> str:
    columns = [
        (metric_type, prediction_method)
        for metric_type in cfg.metric_types
        for prediction_method in cfg.prediction_methods
    ]
    header = "| Model | " + " | ".join(
        f"{metric_type} {prediction_method}"
        for metric_type, prediction_method in columns
    )
    lines = [header + " |", "|---|" + "---:|" * len(columns)]
    indexed = models.set_index(["model", "metric_type", "prediction_method"])
    for model in cfg.models:
        cells = [
            "{:.2f} ± {:.2f}".format(
                *indexed.loc[(model, metric_type, prediction_method)][
                    ["mean", "sd_over_seeds"]
                ]
            )
            for metric_type, prediction_method in columns
        ]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(cfg: Config) -> None:
    scores = read_seed_scores(cfg)
    cells = summarize_cells(scores)
    models = summarize_models(scores, cells)
    comparisons = compare_models(scores, cfg.comparisons, cfg.significance_level)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(cfg.output_dir / "seed_scores.csv", index=False)
    cells.to_csv(cfg.output_dir / "seed_cells.csv", index=False)
    models.to_csv(cfg.output_dir / "seed_models.csv", index=False)
    comparisons.to_csv(cfg.output_dir / "seed_comparisons.csv", index=False)
    (cfg.output_dir / "seed_models.md").write_text(model_table_markdown(models, cfg))
    logger.info(f"Wrote {len(scores)} scores to {cfg.output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    write_outputs(parse_args(argv))


if __name__ == "__main__":
    main()
