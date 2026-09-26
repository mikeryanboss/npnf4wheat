"""FIP 1.0-style final-height scores of the saved FIP1 trajectory predictions.

FIP 1.0 (Roth et al. 2025, GigaScience, doi:10.1093/gigascience/giaf051,
Tables 9 to 11) scores genomic prediction of final height on the test sets of the
FIP1 data set. Its scripts (``quality_checks/Genomic_prediction_visualize.R`` of
the FIP 1.0 traits repository) compute the Pearson correlation and the RMSE
between predicted and observed genotype-year BLUEs within each harvest year and
average them over the years of a test set. This script applies the same rule to
the covariate-only (``no_context``) predictions of the FIP1-trained runs.

The final height of a draw is the median of its three highest daily values, the
rule that ``fip1_traits_by_year`` applies to the measured plots. A plot's
prediction is the mean over its draws, and a genotype-year prediction is the mean
over its plots. The genotype level compares with ``height_final_blue``, the plot
level with ``height_final_value``. Without markers in the covariate setting, all
genotypes of a year share one predictive distribution, so the correlation is not
reported for those settings.

As a reference, a GBLUP on the training and validation BLUEs (year means plus a
ridge regression of the genotype deviations on the markers) is scored with the
same rule. As in FIP 1.0, an unseen year gets the mean of the training years.

``final_height_scores.csv`` has one row per run, split, covariate setting, level
and genotype subset; ``final_height_summary.csv`` the mean and sample sd over
seeds; ``final_height.md`` the genotype-level tables for the genotypes with
markers, next to the GBLUP reference and the FIP 1.0 values.

Example:
    uv run src/npnf/scripts/paper/fip1_final_height.py \
        --dataset $NPNF_PROJECT_DIR/../dataset_fip1_gabi_grouped \
        --results-dir $NPNF_PROJECT_DIR/../results \
        --output-dir paper/fip1_final_height
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
from loguru import logger

from npnf.baselines.lodging_mixture_spline.effects import EffectRegression
from npnf.data.batch_loader import BatchLoader
from npnf.data.datasets.fip1 import Fip1Facts
from npnf.scripts.paper.fip1_sig_mmd_table import FIP1_RUN, mean_sd_text
from npnf.scripts.utils.outputs import markdown_table


def final_height(curves: torch.Tensor) -> torch.Tensor:
    """Median of the three highest values along the last axis."""
    return curves.topk(3, dim=-1).values.median(dim=-1).values


def plot_final_heights(method_dir: Path) -> pd.DataFrame:
    """Mean predicted final height and number of draws of every plot."""
    frames = []
    for predictions, batch in BatchLoader(method_dir):
        grid = predictions.get("grid").float()[:, 0, :, :, 0]  # (plots, draws, days)
        heights = final_height(grid)
        frames.append(
            pd.DataFrame(
                {
                    "plot_uid": list(batch["data"]["plot_uid"]),
                    "predicted": heights.mean(dim=1).numpy(),
                    "num_draws": grid.shape[1],
                }
            )
        )
    plots = pd.concat(frames, ignore_index=True)
    duplicated = plots["plot_uid"].duplicated()
    if duplicated.any():
        # A stale batch of an earlier prediction run repeats plots of batch 000.
        logger.warning(f"{method_dir}: dropping {duplicated.sum()} repeated plots")
    return plots[~duplicated]


def genotype_year_means(plots: pd.DataFrame) -> pd.DataFrame:
    """Mean prediction per genotype and year, with the genotype-year BLUE."""
    return plots.groupby(["genotype_id", "harvest_year"], as_index=False).agg(
        predicted=("predicted", "mean"), observed=("height_final_blue", "first")
    )


def score(units: pd.DataFrame, *, correlation: bool) -> tuple[float, float]:
    """Pearson r and RMSE within each harvest year, averaged over the years."""
    correlations, errors = [], []
    for _, year in units.groupby("harvest_year"):
        difference = year["predicted"] - year["observed"]
        errors.append(np.sqrt(np.mean(np.square(difference))))
        if correlation:
            correlations.append(np.corrcoef(year["predicted"], year["observed"])[0, 1])
    return (float(np.mean(correlations)) if correlation else np.nan), float(
        np.mean(errors)
    )


def load_plots(dataset_path: Path) -> dict[str, pd.DataFrame]:
    """Plot metadata, final-height targets and markers of every split."""
    dataset_dict = datasets.load_from_disk(str(dataset_path))
    if not isinstance(dataset_dict, datasets.DatasetDict):
        msg = f"Expected a DatasetDict at {dataset_path}"
        raise TypeError(msg)
    columns = [
        "plot_uid",
        "genotype_id",
        "harvest_year",
        "height_final_value",
        "height_final_blue",
        "marker_biallelic_codes",
    ]
    plots = {}
    for split in Fip1Facts().splits:
        frame = dataset_dict[split].select_columns(columns).to_pandas()
        frame["has_markers"] = frame["marker_biallelic_codes"].map(
            lambda codes: codes is not None and len(codes) > 0
        )
        plots[split] = frame
    return plots


def gblup_predictions(plots: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """GBLUP reference predictions for the plots with markers of every test set."""
    training = pd.concat([plots["train"], plots["validation"]])
    training = training[training["has_markers"]].drop_duplicates(
        ["genotype_id", "harvest_year"]
    )
    year_means = training.groupby("harvest_year")["height_final_blue"].mean()
    training["deviation"] = training["height_final_blue"] - training[
        "harvest_year"
    ].map(year_means)
    genotypes = training.groupby("genotype_id").agg(
        deviation=("deviation", "mean"), markers=("marker_biallelic_codes", "first")
    )
    regression = EffectRegression(len(genotypes["markers"].iloc[0]), 1)
    fit = regression.fit(
        torch.tensor(np.stack(genotypes["markers"]), dtype=torch.float32),
        torch.tensor(genotypes["deviation"].to_numpy())[:, None],
    )
    logger.info(
        f"GBLUP on {len(genotypes)} genotypes: ridge {fit['ridge']}, "
        f"leave-one-out R² {fit['loo_r2']:.3f}"
    )
    predictions = {}
    for split in Fip1Facts().test_splits:
        frame = plots[split][plots[split]["has_markers"]]
        markers = torch.tensor(
            np.stack(frame["marker_biallelic_codes"]), dtype=torch.float32
        )
        offsets = frame["harvest_year"].map(year_means).fillna(year_means.mean())
        predictions[split] = pd.DataFrame(
            {
                "plot_uid": frame["plot_uid"].to_numpy(),
                "predicted": offsets.to_numpy()
                + regression.predict(markers)[:, 0].numpy(),
                "num_draws": 1,
            }
        )
    return predictions


def score_rows(
    predicted: pd.DataFrame, plots: pd.DataFrame, *, correlation: bool
) -> list[dict]:
    """Genotype and plot scores, for all genotypes and for those with markers."""
    joined = predicted.merge(plots, on="plot_uid", validate="one_to_one")
    if len(joined) != len(predicted):
        msg = "Predicted plots missing from the data set"
        raise ValueError(msg)
    rows = []
    for subset, frame in (
        ("all", joined),
        ("with_markers", joined[joined["has_markers"]]),
    ):
        if frame.empty:
            continue
        levels = {
            "genotype": genotype_year_means(frame),
            "plot": frame.assign(observed=frame["height_final_value"]),
        }
        for level, units in levels.items():
            r, rmse = score(units, correlation=correlation)
            rows.append(
                {
                    "level": level,
                    "subset": subset,
                    "num_units": len(units),
                    "num_draws": int(frame["num_draws"].min()),
                    "r": r,
                    "rmse": rmse,
                }
            )
    return rows


def run_identity(run: str) -> tuple[str, str, int]:
    """(model, init, seed) of a FIP1 run; the spline baseline has one fit."""
    match = FIP1_RUN.match(run)
    if match is None:
        return run, "fit", 0
    return match["model"], match["init"], int(match["seed"])


def fip1_published() -> dict[str, tuple[float, float]]:
    """FIP 1.0 ID model, GABI markers, balanced split: (r, RMSE in m) of final height.

    Tables 11 (test (G)), 10 (test (E)) and 9 (test (G and E)); FIP 1.0 has no
    genomic prediction on unseen plots.
    """
    return {
        "test_genotype": (0.83, 0.07),
        "test_environment": (0.96, 0.04),
        "test_genotype_environment": (0.80, 0.06),
    }


def markdown_report(summary: pd.DataFrame, reference: pd.DataFrame) -> str:
    """Genotype-level tables for the genotypes with markers, one per test set."""
    settings = {
        "P": "noenv_nogeno",
        "E": "env_nogeno",
        "G": "noenv_geno",
        "E&G": "env_geno",
    }
    header = ["Model", "Init", "RMSE P", "RMSE E", "r G", "RMSE G", "r E&G", "RMSE E&G"]
    selected = summary[
        (summary["level"] == "genotype") & (summary["subset"] == "with_markers")
    ]
    reference = reference[reference["level"] == "genotype"]
    published = fip1_published()
    sections = []
    for split in Fip1Facts().test_splits:
        rows = []
        for (model, init), group in selected[selected["split"] == split].groupby(
            ["model", "init"], sort=True
        ):
            cells = group.set_index("setting")
            row = [model, init]
            for column in header[2:]:
                metric, setting = column.split(" ")
                if settings[setting] not in cells.index:
                    row.append("")
                    continue
                cell = cells.loc[settings[setting]]
                sd = cell[f"{metric.lower()}_sd"]
                row.append(
                    mean_sd_text(
                        cell[f"{metric.lower()}_mean"], None if pd.isna(sd) else sd
                    )
                )
            rows.append(row)
        gblup = reference[reference["split"] == split].iloc[0]
        rows.append(
            [
                "GBLUP (this data)",
                "",
                "",
                "",
                f"{gblup['r']:.3f}",
                f"{gblup['rmse']:.3f}",
                "",
                "",
            ]
        )
        if split in published:
            r, rmse = published[split]
            rows.append(
                ["GBLUP ID (FIP 1.0)", "", "", "", f"{r:.2f}", f"{rmse:.2f}", "", ""]
            )
        sections.append(
            f"### {split}\n\n{markdown_table(header, rows, label_columns=2)}"
        )
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    plots = load_plots(args.dataset)
    method_dirs = sorted(
        args.results_dir.glob(
            "fip1_test_*_dataloaders/*-FIP-2930*/checkpoint-*/test_*/*/no_context"
        )
    )
    logger.info(f"{len(method_dirs)} method directories")
    with ProcessPoolExecutor(args.workers) as executor:
        predictions = list(executor.map(plot_final_heights, method_dirs))

    rows = []
    for method_dir, predicted in zip(method_dirs, predictions, strict=True):
        _, run, _, split, setting, _ = method_dir.relative_to(args.results_dir).parts
        model, init, seed = run_identity(run)
        identity = {
            "model": model,
            "init": init,
            "seed": seed,
            "run": run,
            "split": split,
            "setting": setting,
        }
        rows.extend(
            identity | row
            for row in score_rows(
                predicted, plots[split], correlation=setting.endswith("_geno")
            )
        )
    scores = pd.DataFrame(rows)

    reference_rows = []
    for split, predicted in gblup_predictions(plots).items():
        reference_rows.extend(
            {"split": split} | row
            for row in score_rows(predicted, plots[split], correlation=True)
        )
    # GBLUP predicts only genotypes with markers, so "all" repeats that subset.
    reference = pd.DataFrame(reference_rows).query("subset == 'with_markers'")

    summary = (
        scores.groupby(["model", "init", "split", "setting", "level", "subset"])
        .agg(
            n_seeds=("seed", "count"),
            r_mean=("r", "mean"),
            r_sd=("r", lambda values: values.std(ddof=1)),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", lambda values: values.std(ddof=1)),
        )
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "final_height_scores.csv", index=False)
    summary.to_csv(args.output_dir / "final_height_summary.csv", index=False)
    reference.to_csv(args.output_dir / "final_height_gblup_reference.csv", index=False)
    (args.output_dir / "final_height.md").write_text(
        markdown_report(summary, reference)
    )
    logger.info(f"{len(scores)} score rows written to {args.output_dir}")


if __name__ == "__main__":
    main()
