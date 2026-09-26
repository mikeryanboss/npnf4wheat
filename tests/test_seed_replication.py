from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npnf.scripts.paper import seed_replication


def _write_summary(
    root: Path, run: str, metric_type: str, conditioning: str, mean: float
) -> None:
    path = (
        root
        / "synth_test_plot_dataloaders"
        / run
        / "6"
        / "checkpoint-3000000"
        / "test_plot"
        / conditioning
        / "no_context"
        / f"{metric_type}_context_matched_blocked"
        / f"{metric_type}_summary.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "view,n_units,n_conditions,mean,median\n"
        f"context_matched_blocked,2,2,{mean},0.1\n"
    )


def test_seed_tables_and_comparison(tmp_path: Path) -> None:
    scores = {
        # model -> seed -> score x 1000 in both cells
        "LNP": {42: 2.0, 2: 2.2, 3: 1.8},
        "LNP-NF-Prior": {42: 1.0, 2: 1.1, 3: 0.9},
    }
    for model, by_seed in scores.items():
        for seed, score in by_seed.items():
            suffix = "" if seed == 42 else f"_seed{seed}"
            run = f"{model}-512k-training3m_set_mode_nested_noprior{suffix}"
            for conditioning in ("noenv_nogeno", "env_geno"):
                _write_summary(tmp_path, run, "sig_mmd", conditioning, score / 1000.0)

    output_dir = tmp_path / "out"
    seed_replication.main(
        [
            "--results-base", str(tmp_path),
            "--models", "LNP", "LNP-NF-Prior",
            "--seeds", "42", "2", "3",
            "--splits", "plot",
            "--conditionings", "noenv_nogeno", "env_geno",
            "--prediction-methods", "no_context",
            "--metric-types", "sig_mmd",
            "--comparisons", "LNP-NF-Prior:LNP",
            "--output-dir", str(output_dir),
        ]
    )  # fmt: skip

    models = pd.read_csv(output_dir / "seed_models.csv").set_index("model")
    assert models.loc["LNP", "mean"] == pytest.approx(2.0)
    assert models.loc["LNP", "sd_over_seeds"] == pytest.approx(0.2)
    assert models.loc["LNP-NF-Prior", "cell_seed_sd_rms"] == pytest.approx(0.1)

    comparison = pd.read_csv(output_dir / "seed_comparisons.csv").iloc[0]
    assert comparison["mean_relative_gap_percent"] == pytest.approx(-50.0)
    assert comparison["cells_significant_lower"] == 2
    assert comparison["cells_significant_higher"] == 0
    assert "| LNP | 2.00 ± 0.20 |" in (output_dir / "seed_models.md").read_text()


def test_holm_reject_steps_down_and_stops() -> None:
    reject = seed_replication.holm_reject(np.array([0.01, 0.04, 0.03, 0.2]), 0.05)
    # thresholds 0.0125, 0.0167, 0.025: 0.01 passes, 0.03 fails and stops
    assert reject.tolist() == [True, False, False, False]
