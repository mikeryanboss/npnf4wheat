"""Sensitivity of Sig-MMD and CSig-MMD model orders to the metric hyperparameters.

Reads the blocked Sig-MMD and CSig-MMD outputs of each model, split and
conditioning for one prediction method. The CSig-MMD rules are the `sweep/`
subfolders that `sig_mmd_context_matched_blocked.py --csig --sweep-...` writes; each
kernel bandwidth other than 1 has its own `*_sigma=<value>` folders
(`--kernel-sigma`). Writes to `--output-dir`:

- `paired_counts.md`: for each model pair, bandwidth and rule, the number of
  settings in which the first model is lower, the paired block test does not
  resolve the difference (|t| < 2), and the second model is lower. Sig-MMD and the
  CSig-MMD thresholds at `--beta-times-c-squared` form the columns.
- `beta_sensitivity.md`: per bandwidth and threshold, the median relative spread of
  a score over the swept sharpnesses, and the number of settings in which the order
  of all models changes with the sharpness.
- `tail_weights.md`: mean CSig-MMD weight of the reference and of each model per
  threshold at `--beta-times-c-squared`.
- `lodged_recall.md`: per threshold, c² and the fraction of the lodged draws of the
  censoring reference (`--reference-artifact`) above c².
- `paired_t.csv` and `csig_summaries.csv`: the long data behind the tables.
- `sig_mmd_summaries.csv`: the block-averaged Sig-MMD score of each model and
  setting per bandwidth.
- `csig_mmd_summaries.csv`: the same for CSig-MMD per bandwidth and threshold at
  `--beta-times-c-squared`.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from npnf.data.configs.datasets.synthetic_test_sets import split_named
from npnf.metrics.mahalanobis_artifact import load_signature_mahalanobis_artifact
from npnf.metrics.signature import kernel_sigma_suffix
from npnf.scripts.paper.block_uncertainty_table import paired_stat, read_block_scores
from npnf.scripts.paper.mmd_table import add_test_set_arguments, resolve_splits


def method_dir(
    results_base: Path,
    model: str,
    split: str,
    conditioning: str,
    prediction_method: str,
    checkpoint: str,
    test_set: str,
) -> Path:
    """The scored prediction folder of one cell; the dataset-size level is globbed."""
    test_split = split_named(split, test_set)
    root = results_base / test_split.dataloader_name / model
    pattern = (
        f"*/{checkpoint}/{test_split.split_key}/{conditioning}/{prediction_method}"
    )
    matches = sorted(path for path in root.glob(pattern) if path.is_dir())
    if not matches:
        msg = f"No prediction folder matches {root / pattern}"
        raise FileNotFoundError(msg)
    return matches[0]


def read_csig_rules(csig_dir: Path) -> pl.DataFrame:
    """One row per sweep rule: its folder, alpha, beta * c², score and tail weights."""
    rows = []
    for summary_path in sorted((csig_dir / "sweep").glob("*/csig_mmd_summary.csv")):
        summary = pl.read_csv(summary_path).row(0, named=True)
        rows.append(
            {
                "folder": summary_path.parent,
                "alpha": round(summary["alpha"], 4),
                "beta_times_c_squared": round(
                    summary["beta"] * summary["c_squared"], 1
                ),
                "c_squared": summary["c_squared"],
                "score": summary["mean"],
                "weight_reference": summary["mean_w_oracle"],
                "weight_model": summary["mean_w_model"],
            }
        )
    if not rows:
        msg = f"No CSig-MMD sweep summaries under {csig_dir / 'sweep'}"
        raise FileNotFoundError(msg)
    return pl.DataFrame(rows)


def verdict(t_value: float, resolution: float = 2.0) -> str:
    """'A' if the first model is resolved lower, 'B' if the second is, else 'tie'."""
    if t_value <= -resolution:
        return "A"
    if t_value >= resolution:
        return "B"
    return "tie"


def lodged_recall(
    distances: np.ndarray, has_lodged: np.ndarray, c_squared: float
) -> float:
    """Fraction of the lodged reference draws above the threshold ``c_squared``."""
    return float(np.mean(distances[has_lodged] > c_squared))


def count_table(paired: pl.DataFrame, rules: Sequence[str]) -> pl.DataFrame:
    """'A / tie / B' counts per pair and bandwidth, one column per rule."""
    pair_order = pl.Enum(paired["pair"].unique(maintain_order=True))
    counts = (
        paired.group_by("pair", "sigma", "rule", maintain_order=True)
        .agg(
            *(
                (pl.col("verdict") == name).sum().alias(name)
                for name in ("A", "tie", "B")
            )
        )
        .with_columns(pl.format("{} / {} / {}", "A", "tie", "B").alias("counts"))
    )
    return (
        counts.pivot(
            on="rule", index=["pair", "sigma"], values="counts", maintain_order=True
        )
        .select("pair", "sigma", *rules)
        .sort(pl.col("pair").cast(pair_order), "sigma")
    )


def beta_sensitivity(summaries: pl.DataFrame) -> pl.DataFrame:
    """Median relative score spread over beta and settings with a changed order."""
    setting = ["sigma", "alpha", "split", "conditioning"]
    spread = (
        summaries.group_by(*setting, "model")
        .agg(
            (
                (pl.col("score").max() - pl.col("score").min())
                / pl.col("score").median()
            ).alias("spread")
        )
        .group_by("sigma", "alpha")
        .agg(pl.col("spread").median().alias("median_relative_spread"))
    )
    orders = (
        summaries.sort("score")
        .group_by(*setting, "beta_times_c_squared")
        .agg(pl.col("model").str.join(" < ").alias("order"))
        .group_by(setting)
        .agg((pl.col("order").n_unique() > 1).alias("order_changes"))
        .group_by("sigma", "alpha")
        .agg(pl.col("order_changes").sum().alias("settings_with_order_change"))
    )
    return spread.join(orders, on=["sigma", "alpha"]).sort("sigma", "alpha")


def to_markdown(frame: pl.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * frame.width
    rows = [
        "| "
        + " | ".join(
            f"{value:.4g}" if isinstance(value, float) else str(value) for value in row
        )
        + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-base", default=os.environ.get("NPNF_RESULTS_DIR"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--model-names", nargs="+", default=None)
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Model-name pairs FIRST,SECOND; a negative t means FIRST is lower.",
    )
    add_test_set_arguments(parser)
    parser.add_argument(
        "--conditionings",
        nargs="+",
        default=["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"],
    )
    parser.add_argument("--prediction-method", default="no_context")
    parser.add_argument("--checkpoint", default="checkpoint-3000000")
    parser.add_argument("--kernel-sigmas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--beta-times-c-squared", type=float, default=25.6)
    parser.add_argument(
        "--reference-artifact",
        default=None,
        help="Censoring reference (default: <results-base>/sig_mahalanobis/train/"
        "oracle_signature_mahalanobis).",
    )
    parser.add_argument("--output-dir", default="paper/csig_sensitivity")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results_base = Path(args.results_base)
    names = dict(zip(args.models, args.model_names or args.models, strict=True))
    pairs = [tuple(pair.split(",")) for pair in args.pairs]
    splits = resolve_splits(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_rows = []
    summary_frames = []
    sig_rows = []
    for sigma in args.kernel_sigmas:
        suffix = kernel_sigma_suffix(sigma)
        for split in splits:
            for conditioning in args.conditionings:
                cell = {"sigma": sigma, "split": split, "conditioning": conditioning}
                block_paths = {}
                for model, name in names.items():
                    folder = method_dir(
                        results_base,
                        model,
                        split,
                        conditioning,
                        args.prediction_method,
                        args.checkpoint,
                        args.test_set,
                    )
                    csig_rules = read_csig_rules(
                        folder / f"csig_mmd_context_matched_blocked{suffix}"
                    ).with_columns(
                        **{key: pl.lit(value) for key, value in cell.items()},
                        model=pl.lit(name),
                    )
                    summary_frames.append(csig_rules)
                    sig_dir = folder / f"sig_mmd_context_matched_blocked{suffix}"
                    sig_summary = pl.read_csv(sig_dir / "sig_mmd_summary.csv")
                    sig_rows.append(
                        {**cell, "model": name, "score": sig_summary["mean"][0]}
                    )
                    block_paths[name] = {"Sig-MMD": sig_dir / "sig_mmd_blocks.csv"}
                    for rule in csig_rules.filter(
                        pl.col("beta_times_c_squared") == args.beta_times_c_squared
                    ).iter_rows(named=True):
                        block_paths[name][f"alpha = {rule['alpha']:.2f}"] = (
                            rule["folder"] / "csig_mmd_blocks.csv"
                        )
                for first, second in pairs:
                    for rule, path in block_paths[first].items():
                        stat = paired_stat(
                            first,
                            second,
                            split,
                            conditioning,
                            args.prediction_method,
                            read_block_scores(path),
                            read_block_scores(block_paths[second][rule]),
                            None,
                        )
                        if stat is None:
                            msg = f"Fewer than two shared blocks in {path}"
                            raise ValueError(msg)
                        paired_rows.append(
                            {
                                **cell,
                                "rule": rule,
                                "pair": f"{first} vs {second}",
                                "t": stat.t_value,
                                "verdict": verdict(stat.t_value),
                            }
                        )

    paired = pl.DataFrame(paired_rows)
    summaries = pl.concat(summary_frames).drop("folder")
    paired.write_csv(output_dir / "paired_t.csv")
    summaries.write_csv(output_dir / "csig_summaries.csv")
    pl.DataFrame(sig_rows).write_csv(output_dir / "sig_mmd_summaries.csv")
    summaries.filter(
        pl.col("beta_times_c_squared") == args.beta_times_c_squared
    ).select("sigma", "alpha", "split", "conditioning", "model", "score").write_csv(
        output_dir / "csig_mmd_summaries.csv"
    )
    rules = list(dict.fromkeys(paired["rule"]))
    (output_dir / "paired_counts.md").write_text(
        to_markdown(count_table(paired, rules))
    )
    (output_dir / "beta_sensitivity.md").write_text(
        to_markdown(beta_sensitivity(summaries))
    )
    tail_weights = (
        summaries.filter(
            pl.col("beta_times_c_squared") == args.beta_times_c_squared,
            pl.col("sigma") == args.kernel_sigmas[0],
        )
        .group_by("alpha")
        .agg(
            pl.col("weight_reference").mean().alias("reference"),
            *(
                pl.col("weight_model")
                .filter(pl.col("model") == name)
                .mean()
                .alias(name)
                for name in names.values()
            ),
        )
        .sort("alpha")
    )
    (output_dir / "tail_weights.md").write_text(to_markdown(tail_weights))

    reference = load_signature_mahalanobis_artifact(
        args.reference_artifact
        or results_base / "sig_mahalanobis/train/oracle_signature_mahalanobis"
    )
    distances = np.asarray(reference.mahalanobis_distances)
    has_lodged = np.asarray(reference.has_lodged, dtype=np.bool_)
    thresholds = (
        summaries.group_by("alpha").agg(pl.col("c_squared").first()).sort("alpha")
    )
    recall_table = thresholds.with_columns(
        lodged_recall=pl.col("c_squared").map_elements(
            lambda c_squared: lodged_recall(distances, has_lodged, c_squared),
            return_dtype=pl.Float64,
        ),
        lodging_rate=pl.lit(float(has_lodged.mean())),
    )
    (output_dir / "lodged_recall.md").write_text(to_markdown(recall_table))
    logger.info("Wrote the sensitivity tables to {}", output_dir)


if __name__ == "__main__":
    main()
