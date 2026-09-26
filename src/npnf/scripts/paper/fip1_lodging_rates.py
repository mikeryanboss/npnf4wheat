"""Observed and predicted lodging rates on the FIP1 test splits.

Reports, per (split, harvest year),

* the fraction of *observed* plots the detector flags, and
* the fraction of covariate-only prior samples of a trained run it flags, one
  sample per plot on the daily axis.

The distributional scores of the same runs come from
``npnf.scripts.metrics.sig_mmd_fip1_context_matched_blocked`` and
``fip1_sig_mmd_table.py``, which read saved predictions instead of loading
models. This script keeps its own prior pass because the lodging detector needs a
daily curve, which the scored predictions do not carry outside the metric window.

Runs that differ only in their training seed are aggregated: every row carries a
``seed`` column and ``lodging_summary.csv`` reports mean and sample sd over the
seeds of each (model, init, split, year). ``lodging.md`` holds the two tables of
the review answer (Table S7).

The checkpoints are independent, so the evaluation can be split with
``--shard i/n``; a shard writes only ``lodging.csv``. A final pass with
``--from-shards`` concatenates them and writes the summary.

Example:
    uv run src/npnf/scripts/paper/fip1_lodging_rates.py \
        --runs $NPNF_PROJECT_DIR/*-FIP-2930-fip1_{scratch,pretrained}_1m_seed* \
        --output-dir paper/fip1_lodging
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensordict
import torch
from accelerate import Accelerator
from loguru import logger
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from npnf.data.datasets.fip1 import get_heights_dataset
from npnf.data.process.collate import collate_fn_fip1_heights
from npnf.lodging import detect_lodging
from npnf.scripts.paper.fip1_sig_mmd_table import mean_sd_text, parse_run_name
from npnf.scripts.utils.outputs import markdown_table
from npnf.scripts.utils.prediction import (
    get_empty_observations,
    predict_batch_with_grid,
    resolve_checkpoint,
    resolve_model_from_checkpoint,
)

SPLITS = {
    "plot": "test_plot",
    "genotype": "test_genotype",
    "environment": "test_environment",
    "unseen": "test_genotype_environment",
}
SUMMARY_KEYS = ("model", "init", "metric", "split", "year", "n_plots")
MODEL_DAY_AXIS = np.arange(61, 365)
INPUT_DAY_OFFSET, INPUT_DAY_SCALE = 212.5, 151.5


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample sd of ``value`` over the seeds of each group.

    ``sd`` is empty where only one seed contributes, per the reporting convention
    shared with the synthetic seed-replication tables.
    """
    summary = (
        results.groupby(list(SUMMARY_KEYS), dropna=False)["value"]
        .agg(n_seeds="count", mean="mean", sd=lambda values: values.std(ddof=1))
        .reset_index()
    )
    summary["text"] = [
        mean_sd_text(row.mean, None if pd.isna(row.sd) else row.sd)
        for row in summary.itertuples(index=False)
    ]
    return summary


def lodging_table(
    summary: pd.DataFrame,
    models: list[str],
    split_labels: tuple[tuple[str, str], ...] = (
        ("plot", "seen"),
        ("genotype", "geno"),
        ("environment", "env"),
        ("unseen", "unseen"),
    ),
) -> str:
    """Observed rate per (split, year), and the mean / maximum predicted rate over
    those cells per model and initialisation. Splits appear as their labels, in
    the order of ``split_labels``."""
    observed = summary[summary["metric"] == "lodging_rate_real"].drop_duplicates(
        ["split", "year"]
    )
    observed_rows = [
        [label, str(int(row.year)), str(int(row.n_plots)), f"{row.mean:.3f}"]
        for split, label in split_labels
        for row in observed[observed["split"] == split]
        .sort_values("year")
        .itertuples(index=False)
    ]
    predicted = {
        (model, init): f"{rates['mean']:.3f} / {rates['max']:.3f}"
        for (model, init), rates in summary[summary["metric"] == "lodging_rate_pred"]
        .groupby(["model", "init"])["mean"]
        .agg(["mean", "max"])
        .iterrows()
    }
    predicted_rows = [
        [
            model,
            *(predicted.get((model, init), "-") for init in ("scratch", "pretrained")),
        ]
        for model in models
        if (model, "scratch") in predicted or (model, "pretrained") in predicted
    ]
    return (
        markdown_table(
            ["Test set", "Year", "Plots", "Observed lodging rate"],
            observed_rows,
            label_columns=2,
        )
        + "\n\n"
        + markdown_table(["Model", "scr", "pre"], predicted_rows)
    )


def select_shard(runs: list[Path], shard: str | None) -> list[Path]:
    """``i/n``: keep the runs at 1-based positions ``i, i + n, i + 2n, ...``."""
    if shard is None:
        return runs
    index, count = (int(part) for part in shard.split("/"))
    return [run for position, run in enumerate(runs) if position % count == index - 1]


def curves_at_days(curves: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Select columns of daily ``curves`` (N, len(MODEL_DAY_AXIS)) at ``days``."""
    return curves[:, np.searchsorted(MODEL_DAY_AXIS, days)]


def predicted_lodging_rate(
    curves: np.ndarray, days_per_plot: list[np.ndarray]
) -> float:
    """Fraction of daily ``curves`` detected as lodged at each plot's own dates."""
    lodged = [
        detect_lodging(curves_at_days(curve[None], days)[0]).is_lodged
        for curve, days in zip(curves, days_per_plot, strict=True)
    ]
    return float(np.mean(lodged))


def observed_lodging_rate(heights_per_plot: list[np.ndarray]) -> float:
    return float(np.mean([detect_lodging(h).is_lodged for h in heights_per_plot]))


def load_model(run_dir: Path, accelerator: Accelerator) -> torch.nn.Module:
    checkpoint = resolve_checkpoint(run_dir / "checkpoints")
    model = resolve_model_from_checkpoint(str(checkpoint))
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")))
    model = accelerator.prepare(model)
    model.eval()
    return model


def load_split(split: str, batch_size: int) -> DataLoader:
    dataset = get_heights_dataset(
        split=split, datasets_offline_path=os.environ["NPNF_DATASET_PATH"]
    )
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn_fip1_heights
    )


def grid_points(device: torch.device) -> tensordict.TensorDict:
    days = torch.tensor(MODEL_DAY_AXIS, device=device, dtype=torch.float32)[:, None]
    return tensordict.TensorDict(
        {"X": days, "X_normalized": (days - INPUT_DAY_OFFSET) / INPUT_DAY_SCALE},
        device=device,
        batch_size=(len(MODEL_DAY_AXIS),),
    )


@torch.no_grad()
def prior_curves(
    model, loaders: dict[str, DataLoader], device: torch.device
) -> dict[tuple[str, int], dict]:
    """Temperature-only prior samples on the daily axis, one per plot.

    Cells are keyed by ``(split, harvest year)``: the detector compares a curve
    with the dates of its own plot, and a year is the finest cell the tables use.
    """
    grid = grid_points(device)
    cells: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"plot_uid": [], "days": [], "heights": [], "curves": []}
    )
    for split, loader in loaders.items():
        for batch in loader:
            count = batch["temperature"].shape[0]
            empty = get_empty_observations(count, device, torch.float32)
            result = predict_batch_with_grid(
                model,
                empty,
                empty.clone(),
                grid,
                num_samples=1,
                temperatures=batch["temperature"].to(device),
                include_targets_std=False,
                include_grid_std=False,
            )
            curves = result["grid"][:, 0, :, 0].float().cpu().numpy()
            heights = batch["height"]
            for index in range(count):
                cell = cells[(split, int(batch["harvest_year"][index]))]
                cell["plot_uid"].append(str(batch["plot_uid"][index]))
                cell["days"].append(heights["X"][index][:, 0].numpy().astype(int))
                cell["heights"].append(heights["Y"][index][:, 0].numpy())
                cell["curves"].append(curves[index])
    for cell in cells.values():
        cell["curves"] = np.stack(cell["curves"])
    return cells


def evaluate_lodging(
    model, loaders: dict[str, DataLoader], seed: int, device: torch.device
) -> list[dict]:
    torch.manual_seed(seed)
    rows = []
    for (split, year), cell in sorted(prior_curves(model, loaders, device).items()):
        base = {"split": split, "year": year, "n_plots": len(cell["plot_uid"])}
        rows.append(
            {
                **base,
                "metric": "lodging_rate_pred",
                "value": predicted_lodging_rate(cell["curves"], cell["days"]),
            }
        )
        rows.append(
            {
                **base,
                "metric": "lodging_rate_real",
                "value": observed_lodging_rate(cell["heights"]),
            }
        )
    return rows


def evaluate_runs(
    args: argparse.Namespace, loaders: dict[str, DataLoader]
) -> pd.DataFrame:
    accelerator = Accelerator()
    rows: list[dict] = []
    for run in select_shard(args.runs, args.shard):
        model_name, init, seed = parse_run_name(run.name)
        logger.info("Evaluating {}", run.name)
        model = load_model(run, accelerator)
        base = {"run": run.name, "model": model_name, "init": init, "seed": seed}
        rows.extend(
            {**base, **row}
            for row in evaluate_lodging(model, loaders, args.seed, accelerator.device)
        )
        del model
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", default=None, metavar="I/N")
    parser.add_argument("--from-shards", nargs="*", type=Path, default=[])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
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
        ],
        help="Row order of the predicted-rate table in lodging.md.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = {
        split: load_split(dataset_split, args.batch_size)
        for split, dataset_split in SPLITS.items()
    }

    if args.from_shards:
        results = pd.concat(
            [pd.read_csv(shard / "lodging.csv") for shard in args.from_shards],
            ignore_index=True,
        )
    else:
        results = evaluate_runs(args, loaders)

    results.to_csv(args.output_dir / "lodging.csv", index=False)
    logger.info("Wrote {}", args.output_dir / "lodging.csv")
    if args.shard:
        return

    summary = summarize(results)
    summary.to_csv(args.output_dir / "lodging_summary.csv", index=False)
    (args.output_dir / "lodging.md").write_text(
        lodging_table(summary, args.models) + "\n"
    )
    logger.info("Wrote {} and lodging.md", args.output_dir / "lodging_summary.csv")


if __name__ == "__main__":
    main()
