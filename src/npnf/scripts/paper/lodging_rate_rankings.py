"""Model rankings per lodging rate from split-averaged Sig-MMD / CSig-MMD tables.

Each input is the `*_avg_splits_long.csv` that `mmd_table.py` writes for one test
set. Every column of that paper table (prediction method x conditioning) ranks the
models, 1 being the lowest score. The output gives each model's mean rank and the
number of columns it ranks first, per lodging rate, and the Kendall tau between the
mean-rank order of each rate and that of the first rate. `--kendall-models` restricts
the tau to a subset, e.g. the latent models, whose order the deterministic models at
the bottom of every rate would otherwise inflate.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from loguru import logger
from scipy.stats import kendalltau


def parse_tables(values: Sequence[str]) -> dict[str, Path]:
    tables = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            msg = f"--tables entries must be LABEL=PATH, got {value!r}"
            raise ValueError(msg)
        tables[label] = Path(path)
    return tables


def rank_models(table: pl.DataFrame) -> pl.DataFrame:
    """Mean rank and first-place count of each model over the table columns."""
    columns = ["prediction_method", "conditioning"]
    complete = table.filter(pl.len().over(columns) == table["model_name"].n_unique())
    return (
        complete.with_columns(
            pl.col("mean").rank(method="average").over(columns).alias("rank")
        )
        .group_by("model_name", maintain_order=True)
        .agg(
            pl.col("rank").mean().alias("mean_rank"),
            (pl.col("rank") == 1).sum().alias("first"),
            pl.len().alias("columns"),
        )
    )


def lodging_rate_rankings(
    tables: dict[str, Path], kendall_models: Sequence[str] | None = None
) -> tuple[pl.DataFrame, list[str]]:
    """Wide mean-rank table over all rates and the Kendall tau lines.

    Models are ranked among all models of each table; `kendall_models` only selects
    the models whose mean-rank orders the tau compares (all models when None).
    """
    ranks = {label: rank_models(pl.read_csv(path)) for label, path in tables.items()}
    labels = list(ranks)
    wide = ranks[labels[0]].select("model_name")
    for label in labels:
        wide = wide.join(
            ranks[label].select(
                "model_name",
                pl.col("mean_rank").alias(f"mean_rank_{label}"),
                pl.col("first").alias(f"first_{label}"),
            ),
            on="model_name",
            how="left",
            maintain_order="left",
        )
    compared = wide
    if kendall_models is not None:
        missing = set(kendall_models) - set(wide["model_name"])
        if missing:
            msg = f"--kendall-models not in the tables: {sorted(missing)}"
            raise ValueError(msg)
        compared = wide.filter(pl.col("model_name").is_in(list(kendall_models)))
    reference = compared[f"mean_rank_{labels[0]}"].to_numpy()
    tau_lines = []
    for label in labels[1:]:
        result = kendalltau(reference, compared[f"mean_rank_{label}"].to_numpy())
        tau_lines.append(
            f"Kendall tau of mean ranks ({compared.height} models), "
            f"{labels[0]} vs {label}: {result.statistic:.3f} (p = {result.pvalue:.3g})"
        )
    return wide, tau_lines


def latex_table(wide: pl.DataFrame, labels: Sequence[str], decimals: int) -> str:
    header = " & ".join(["Model", *labels])
    rows = [
        " & ".join(
            [
                row["model_name"],
                *(f"{row[f'mean_rank_{label}']:.{decimals}f}" for label in labels),
            ]
        )
        + r" \\"
        for row in wide.iter_rows(named=True)
    ]
    return "\n".join(
        [
            r"\begin{tabular}{l" + "r" * len(labels) + "}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tables",
        nargs="+",
        required=True,
        metavar="LABEL=PATH",
        help="Split-averaged long CSV of mmd_table.py per lodging rate; the first "
        "rate is the reference for Kendall tau.",
    )
    parser.add_argument(
        "--kendall-models",
        nargs="+",
        default=None,
        help="Models compared by Kendall tau (default: all models).",
    )
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--stem", default="lodging_rate_rankings")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args(argv)

    tables = parse_tables(args.tables)
    wide, tau_lines = lodging_rate_rankings(tables, args.kendall_models)
    output = Path(args.output_folder)
    output.mkdir(parents=True, exist_ok=True)
    wide.write_csv(output / f"{args.stem}.csv")
    (output / f"{args.stem}.tex").write_text(
        latex_table(wide, list(tables), args.decimals) + "\n"
    )
    (output / f"{args.stem}_kendall_tau.txt").write_text("\n".join(tau_lines) + "\n")
    logger.info("\n{}\n{}", wide, "\n".join(tau_lines))


if __name__ == "__main__":
    main()
