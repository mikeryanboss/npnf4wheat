"""Compare observed genotype-group splits and full year populations on one grid."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from numpy.typing import NDArray

from npnf.calibration.height.evaluation.reference_comparison import (
    score_between_year_pairs,
    score_within_year_splits,
)
from npnf.data import fip1_day_grid as grid_protocol
from npnf.data.datasets.fip1 import get_heights_dataset
from npnf.data.fip1_day_grid import SharedDayGrid, observed_at_grid, select_year
from npnf.metrics.signature import to_metric_paths
from npnf.scripts.utils.outputs import write_csv


def load_reference_inputs(
    lodging_dir: Path,
) -> tuple[dict[str, Any], Dataset, SharedDayGrid]:
    """Load measured-data settings and verify their current shared day grid."""
    settings_path = lodging_dir / "lodging_settings.json"
    settings = json.loads(settings_path.read_text())
    dataset = get_heights_dataset(
        split=settings["splits"],
        standardize=False,
        datasets_offline_path=settings["dataset_path"],
    ).dataset
    grid = grid_protocol.fip1_day_grid(settings["dataset_path"])
    if settings["metric_grid"]["anchors"] != grid.anchors.tolist() or settings[
        "metric_grid"
    ]["assigned_days"] != {key: value.tolist() for key, value in grid.assigned.items()}:
        msg = "Lodging inputs do not use the current shared FIP1 day grid"
        raise ValueError(msg)
    return settings, dataset, grid


def load_observed_paths_and_identities(
    lodging_dir: Path, year: int, dataset: Dataset, grid: SharedDayGrid
) -> tuple[torch.Tensor, NDArray, NDArray]:
    """Recover genotype identities and verify every saved measured trajectory."""
    with np.load(lodging_dir / f"aligned_{year}.npz", allow_pickle=False) as data:
        days, real, plot_uids = [data[key] for key in ("days", "real", "plot_uids")]
    if not np.array_equal(days, grid.anchors):
        msg = f"{year}: shared days differ from the fixed reference anchors"
        raise ValueError(msg)
    yearly = select_year(dataset, year, grid).select_columns(
        ["plot_uid", "genotype_id", "yearsite_uid", "height_days", "height_values"]
    )
    rows = {str(row["plot_uid"]): row for row in yearly}
    if (
        len(rows) != len(yearly)
        or len(np.unique(plot_uids)) != len(plot_uids)
        or set(rows) != set(plot_uids.tolist())
    ):
        msg = f"{year}: saved plot identities do not match the original population"
        raise ValueError(msg)
    ordered = [rows[str(uid)] for uid in plot_uids]
    observed = observed_at_grid(ordered, grid).numpy()
    if not np.array_equal(real, observed):
        msg = f"{year}: saved rows differ from original aligned measurements"
        raise ValueError(msg)
    genotype_ids = np.asarray(
        [
            row["genotype_id"].item()
            if isinstance(row["genotype_id"], torch.Tensor)
            else row["genotype_id"]
            for row in ordered
        ]
    )
    time = (torch.from_numpy(days).float() - 200) / 121
    paths = to_metric_paths(torch.from_numpy(real).float(), 1.5, time, lead_lag=True)
    return paths, plot_uids, genotype_ids


def compare_real_populations(lodging_dir: Path, output_dir: Path) -> None:
    split_seeds = tuple(range(600, 650))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    settings, dataset, grid = load_reference_inputs(lodging_dir)
    runs, summaries, real_paths = [], [], {}
    for year in settings["years"]:
        paths, plot_uids, genotype_ids = load_observed_paths_and_identities(
            lodging_dir, year, dataset, grid
        )
        real_paths[year] = paths
        membership, yearly_runs, summary = score_within_year_splits(
            year, paths, genotype_ids, split_seeds, len(grid.anchors)
        )
        np.savez_compressed(
            output_dir / f"splits_{year}.npz",
            membership=membership,
            plot_uids=plot_uids,
            genotype_ids=genotype_ids,
            seeds=np.asarray(split_seeds),
        )
        runs.extend(yearly_runs)
        summaries.append(summary)
        logging.info("%s: completed 50 genotype-group splits, N=%s", year, len(paths))
    year_rows = score_between_year_pairs(
        settings["years"], real_paths, len(grid.anchors)
    )
    write_csv(output_dir / "reference_runs.csv", runs, list(runs[0]))
    write_csv(output_dir / "reference_summary.csv", summaries, list(summaries[0]))
    write_csv(output_dir / "year_reference.csv", year_rows, list(year_rows[0]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lodging-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare_real_populations(args.lodging_dir, args.output_dir)


if __name__ == "__main__":
    main()
