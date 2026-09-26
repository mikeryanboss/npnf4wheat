from pathlib import Path

import numpy as np
import pandas as pd

from npnf.scripts.paper.fip1_lodging_rates import (
    MODEL_DAY_AXIS,
    curves_at_days,
    lodging_table,
    observed_lodging_rate,
    predicted_lodging_rate,
    select_shard,
    summarize,
)


def test_select_shard_partitions_the_runs():
    runs = [Path(f"run{index}") for index in range(5)]
    shards = [select_shard(runs, f"{index}/3") for index in (1, 2, 3)]
    assert [len(shard) for shard in shards] == [2, 2, 1]
    assert sorted(run for shard in shards for run in shard) == runs
    assert select_shard(runs, None) == runs


def test_curves_at_days_uses_the_daily_axis():
    curves = np.arange(2 * MODEL_DAY_AXIS.size, dtype=float).reshape(2, -1)
    sampled = curves_at_days(curves, np.array([61, 200]))
    assert sampled[0].tolist() == [0.0, float(200 - 61)]
    assert sampled[1, 0] == float(MODEL_DAY_AXIS.size)


def test_lodging_rates_flag_a_dropping_trajectory():
    days = np.arange(200, 300, 10)
    growing = np.linspace(0.0, 0.9, days.size)
    lodged = np.concatenate([np.linspace(0.0, 0.9, 6), np.full(days.size - 6, 0.4)])
    assert observed_lodging_rate([growing, lodged]) == 0.5

    # The predicted rate reads each plot's own dates out of the daily curve.
    curves = np.zeros((2, MODEL_DAY_AXIS.size))
    positions = np.searchsorted(MODEL_DAY_AXIS, days)
    curves[0, positions] = growing
    curves[1, positions] = lodged
    assert predicted_lodging_rate(curves, [days, days]) == 0.5


def test_summarize_reports_mean_and_sd_over_seeds():
    results = pd.DataFrame(
        [
            {
                "model": "CNP",
                "init": "pretrained",
                "metric": "lodging_rate_pred",
                "split": "plot",
                "year": 2016,
                "n_plots": 79,
                "seed": seed,
                "value": value,
            }
            for seed, value in [(2, 0.0), (3, 0.1)]
        ]
    )
    summary = summarize(results)
    assert summary["n_seeds"].tolist() == [2]
    assert summary["mean"].tolist() == [0.05]
    assert summary["text"].tolist() == ["0.050 ± 0.071"]

    single = summarize(results[results["seed"] == 2])
    assert single["text"].tolist() == ["0.000"]


def test_lodging_table_lists_observed_cells_and_predicted_mean_and_maximum():
    rows = [
        ("CNP", "scratch", "lodging_rate_real", "genotype", 2016, 60, 0.017),
        ("CNP", "scratch", "lodging_rate_real", "plot", 2017, 78, 0.0),
        ("CNP", "scratch", "lodging_rate_real", "plot", 2016, 79, 0.101),
        ("CNP", "scratch", "lodging_rate_pred", "plot", 2016, 79, 0.002),
        ("CNP", "scratch", "lodging_rate_pred", "plot", 2017, 78, 0.010),
        ("CNP", "pretrained", "lodging_rate_real", "plot", 2016, 79, 0.101),
        ("CNP", "pretrained", "lodging_rate_pred", "plot", 2016, 79, 0.004),
    ]
    summary = pd.DataFrame(
        rows, columns=["model", "init", "metric", "split", "year", "n_plots", "mean"]
    )
    # Observed cells follow the split order plot, genotype, ..., then the year; a
    # model without runs is left out.
    assert lodging_table(summary, ["ACNP", "CNP"]).splitlines() == [
        "| Test set | Year | Plots | Observed lodging rate |",
        "|---|---|---:|---:|",
        "| seen | 2016 | 79 | 0.101 |",
        "| seen | 2017 | 78 | 0.000 |",
        "| geno | 2016 | 60 | 0.017 |",
        "",
        "| Model | scr | pre |",
        "|---|---:|---:|",
        "| CNP | 0.006 / 0.010 | 0.004 / 0.004 |",
    ]
