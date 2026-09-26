"""Collect FIP1 blocked context-matched Sig-MMD scores into paper tables.

Walks the prediction results tree, reads every
``sig_mmd_fip1_context_matched_blocked/sig_mmd_summary.csv`` written by
``npnf.scripts.metrics.sig_mmd_fip1_context_matched_blocked``, and reports mean
and sample sd over the five training seeds of each
(model, init, split, conditioning, prediction mode). Two Markdown tables compare
the initialisations: the mean per test set and regime (``fip1_initialisation.md``)
and the paired runs (``fip1_pretraining_pairs.md``).

The covariate conditioning fixes the unit scope of the scorer
(``noenv_nogeno`` -> global, ``env_nogeno`` -> environment, ``noenv_geno`` ->
genotype, ``env_geno`` -> condition), so conditioning and scope are one axis.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger

from npnf.scripts.utils.outputs import markdown_table

CONDITIONINGS = ("noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno")
MODES = ("no_context", "random_context", "max_height")
SPLITS = ("test_plot", "test_genotype", "test_environment", "test_genotype_environment")
SCORER_DIR = "sig_mmd_fip1_context_matched_blocked"

FIP1_RUN = re.compile(
    r"^(?P<model>.+)-FIP-2930-fip1_(?P<init>scratch|pretrained)_1m_seed(?P<seed>\d+)$"
)


def parse_run_name(run_name: str) -> tuple[str, str, int]:
    """(model, init, seed) for a FIP1 run directory name."""
    match = FIP1_RUN.match(run_name)
    if match is None:
        msg = f"Unrecognised run name {run_name!r}"
        raise ValueError(msg)
    return match["model"], match["init"], int(match["seed"])


def summary_paths(results_dir: Path, baseline_runs: Sequence[str] = ()) -> list[Path]:
    """Scorer summaries of the FIP1-trained runs and of ``baseline_runs``.

    The synthetic checkpoints were also predicted on the FIP1 test sets; they are
    not part of the FIP1 results.
    """
    patterns = [
        f"fip1_test_*_dataloaders/{run}/**/{SCORER_DIR}"
        for run in ("*-FIP-2930-fip1_*", *baseline_runs)
    ]
    return sorted(
        path
        for pattern in patterns
        for path in results_dir.glob(f"{pattern}/sig_mmd_summary.csv")
    )


def collect(results_dir: Path, baseline_runs: Sequence[str] = ()) -> pl.DataFrame:
    """One row per scored method directory, with its run, cell and seed.

    A baseline run is one fit: its model is the run name, with no
    initialisation and seed 0.
    """
    rows = []
    for summary_path in summary_paths(results_dir, baseline_runs):
        method_dir = summary_path.parent.parent
        dataloader, run_name, checkpoint, split, conditioning, mode = (
            method_dir.relative_to(results_dir).parts
        )
        if run_name in baseline_runs:
            model, init, seed = run_name, None, 0
        else:
            model, init, seed = parse_run_name(run_name)
        summary = pl.read_csv(summary_path)
        rows.append(
            {
                "dataloader": dataloader,
                "run": run_name,
                "model": model,
                "init": init,
                "seed": seed,
                "checkpoint": checkpoint,
                "split": split,
                "conditioning": conditioning,
                "mode": mode,
                "unit_scope": summary["unit_scope"][0],
                "n_units": summary["n_units"][0],
                "n_blocks": summary["n_blocks"][0],
                "n_plots": summary["n_plots"][0],
                "model_draws": summary["model_draws"][0],
                "context_sigma": summary["context_sigma"][0],
                "day_tolerance": summary["day_tolerance"][0],
                "n_anchor_days": summary["n_anchor_days"][0],
                "sig_mmd_x1000": summary["sig_mmd_x1000"][0],
                "sig_mmd_x1000_median": summary["sig_mmd_x1000_median"][0],
                "sig_mmd_x1000_std": summary["sig_mmd_x1000_std"][0],
            }
        )
    if not rows:
        msg = f"No {SCORER_DIR}/sig_mmd_summary.csv found under {results_dir}"
        raise ValueError(msg)
    return pl.DataFrame(rows)


def mean_sd_text(mean: float, sd: float | None, decimals: int = 3) -> str:
    if sd is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    grouped = (
        results.group_by("model", "init", "split", "conditioning", "mode", "unit_scope")
        .agg(
            pl.col("seed").n_unique().alias("n_seeds"),
            pl.col("sig_mmd_x1000").mean().alias("mean"),
            pl.col("sig_mmd_x1000").std(ddof=1).alias("sd"),
        )
        .sort("split", "conditioning", "mode", "model", "init")
    )
    return grouped.with_columns(
        pl.struct("mean", "sd")
        .map_elements(
            lambda row: mean_sd_text(row["mean"], row["sd"]), return_dtype=pl.String
        )
        .alias("text")
    )


def comparison_table(summary: pl.DataFrame, split: str, mode: str) -> str:
    table = summary.filter((pl.col("split") == split) & (pl.col("mode") == mode))
    rows = []
    for model in sorted(table["model"].unique().to_list()):
        for init in ("scratch", "pretrained"):
            cells = []
            for conditioning in CONDITIONINGS:
                cell = table.filter(
                    (pl.col("model") == model)
                    & (pl.col("init") == init)
                    & (pl.col("conditioning") == conditioning)
                )
                cells.append(cell["text"][0] if cell.height else "")
            if any(cells):
                rows.append([model, init, *cells])
    return markdown_table(["model", "init", *CONDITIONINGS], rows, label_columns=2)


@dataclass(frozen=True)
class TableLabels:
    """Labels of the initialisation tables, as (value, label) pairs in table order."""

    splits: tuple[tuple[str, str], ...] = (
        ("test_plot", "seen"),
        ("test_genotype", "geno"),
        ("test_environment", "env"),
        ("test_genotype_environment", "unseen"),
    )
    modes: tuple[tuple[str, str], ...] = (
        ("no_context", "no context"),
        ("random_context", "sparse context"),
        ("max_height", "max-height context"),
    )
    inits: tuple[tuple[str, str], ...] = (("scratch", "scr"), ("pretrained", "pre"))


def initialisation_table(summary: pl.DataFrame, labels: TableLabels) -> str:
    """Mean over configurations and covariate settings, ± sd over configurations.

    Each configuration first enters as its mean over the four covariate settings,
    so the sd is the spread between configurations.
    """
    per_model = summary.group_by("split", "mode", "init", "model").agg(
        pl.col("mean").mean()
    )
    cells = per_model.group_by("split", "mode", "init").agg(
        pl.col("mean").std(ddof=1).alias("sd"), pl.col("mean").mean()
    )
    text = {
        (row["split"], row["mode"], row["init"]): mean_sd_text(
            row["mean"], row["sd"], decimals=2
        )
        for row in cells.iter_rows(named=True)
    }
    columns = [(mode, init) for mode, _ in labels.modes for init, _ in labels.inits]
    header = ["Test set"] + [
        f"{mode_label} {init_label}"
        for _, mode_label in labels.modes
        for _, init_label in labels.inits
    ]
    rows = [
        [split_label] + [text[(split, mode, init)] for mode, init in columns]
        for split, split_label in labels.splits
    ]
    return markdown_table(header, rows)


def pairing_table(results: pl.DataFrame, labels: TableLabels) -> str:
    """Pre-trained against from-scratch runs, paired per configuration, seed, test
    set and covariate setting.

    The gain is the score from scratch minus the score after pre-training, so a
    positive gain means that pre-training is closer.
    """
    paired = results.pivot(
        on="init",
        index=["model", "seed", "split", "conditioning", "mode"],
        values="sig_mmd_x1000",
    ).with_columns((pl.col("scratch") - pl.col("pretrained")).alias("gain"))
    rows = []
    for mode, mode_label in labels.modes:
        gain = paired.filter(pl.col("mode") == mode)["gain"]
        rows.append(
            [mode_label, str(gain.len()), str((gain > 0).sum()), f"{gain.mean():.2f}"]
        )
    return markdown_table(
        ["Context", "Paired cells", "Pre-trained closer", "Mean gain"], rows
    )


def write_comparison(summary: pl.DataFrame, path: Path) -> None:
    sections = [
        "# FIP1 blocked context-matched Sig-MMD",
        "",
        "Normalised unbiased Sig-MMD^2 x 1000, mean ± sd over five training seeds.",
        "Lower is better. The conditioning column fixes the scorer's unit scope:",
        "`noenv_nogeno` = global, `env_nogeno` = environment, `noenv_geno` = genotype,",
        "`env_geno` = condition.",
        "",
    ]
    for mode in MODES:
        for split in SPLITS:
            table = comparison_table(summary, split, mode)
            if table.count("\n") < 2:
                continue
            sections += [f"## {mode} - {split}", "", table, ""]
    path.write_text("\n".join(sections))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = collect(args.results_dir)
    summary = summarize(results)
    results.write_csv(args.output_dir / "fip1_sig_mmd.csv")
    summary.write_csv(args.output_dir / "fip1_sig_mmd_summary.csv")
    write_comparison(summary, args.output_dir / "fip1_sig_mmd_comparison.md")
    labels = TableLabels()
    (args.output_dir / "fip1_initialisation.md").write_text(
        initialisation_table(summary, labels) + "\n"
    )
    (args.output_dir / "fip1_pretraining_pairs.md").write_text(
        pairing_table(results, labels) + "\n"
    )
    logger.info(
        "rows={} summary_rows={} runs={} models={}",
        results.height,
        summary.height,
        results["run"].n_unique(),
        results["model"].n_unique(),
    )


if __name__ == "__main__":
    main()
