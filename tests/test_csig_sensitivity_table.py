from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from npnf.data.configs.datasets.synthetic_test_sets import split_named
from npnf.scripts.paper import csig_sensitivity_table as table


def _method_dir(root: Path, model: str) -> Path:
    test_split = split_named("plot", "legacy")
    return (
        root
        / test_split.dataloader_name
        / model
        / "6/checkpoint-3000000"
        / test_split.split_key
        / "noenv_nogeno/no_context"
    )


def _write_blocks(path: Path, scores: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "unit_scope": ["global"] * len(scores),
            "unit_id": ["all"] * len(scores),
            "block_index": list(range(len(scores))),
            "score": scores,
        }
    ).write_csv(path)


def _write_rule(
    csig_dir: Path, alpha: float, beta_c2: float, c_squared: float, scores: list[float]
) -> None:
    folder = csig_dir / "sweep" / f"alpha={alpha:.4f}_beta={beta_c2 / c_squared:.4f}"
    _write_blocks(folder / "csig_mmd_blocks.csv", scores)
    pl.DataFrame(
        {
            "alpha": [alpha],
            "beta": [beta_c2 / c_squared],
            "c_squared": [c_squared],
            "mean": [float(np.mean(scores))],
            "mean_w_oracle": [0.2],
            "mean_w_model": [0.1],
        }
    ).write_csv(folder / "csig_mmd_summary.csv")


def test_counts_beta_order_changes_and_lodged_recall(tmp_path, monkeypatch):
    # "first" is lower under Sig-MMD and at alpha 0.8, the two tie at alpha 0.95.
    # At alpha 0.8 the order flips at beta c² 51.2, which is not the main sharpness.
    rules = {
        "first": {
            (0.8, 25.6): [1.0, 1.1, 0.9],
            (0.8, 51.2): [3.0, 3.1, 2.9],
            (0.95, 25.6): [1.0, 2.0, 3.0],
        },
        "second": {
            (0.8, 25.6): [2.0, 2.1, 1.9],
            (0.8, 51.2): [2.0, 2.1, 1.9],
            (0.95, 25.6): [1.1, 1.9, 3.1],
        },
    }
    sig = {"first": [1.0, 1.2, 0.8], "second": [2.0, 2.2, 1.8]}
    for model, model_rules in rules.items():
        folder = _method_dir(tmp_path, model)
        sig_dir = folder / "sig_mmd_context_matched_blocked"
        _write_blocks(sig_dir / "sig_mmd_blocks.csv", sig[model])
        pl.DataFrame({"mean": [float(np.mean(sig[model]))]}).write_csv(
            sig_dir / "sig_mmd_summary.csv"
        )
        for (alpha, beta_c2), scores in model_rules.items():
            _write_rule(
                folder / "csig_mmd_context_matched_blocked",
                alpha,
                beta_c2,
                {0.8: 2.0, 0.95: 6.0}[alpha],
                scores,
            )

    distances = np.array([1.0, 3.0, 5.0, 7.0])
    has_lodged = np.array([False, True, True, True])
    monkeypatch.setattr(
        table,
        "load_signature_mahalanobis_artifact",
        lambda path: SimpleNamespace(
            mahalanobis_distances=distances, has_lodged=has_lodged
        ),
    )
    output = tmp_path / "out"
    table.main(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "first",
            "second",
            "--pairs",
            "first,second",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--output-dir",
            str(output),
        ]
    )

    counts = (output / "paired_counts.md").read_text()
    assert "| first vs second | 1 | 1 / 0 / 0 | 1 / 0 / 0 | 0 / 1 / 0 |" in counts
    assert "alpha = 0.80 | alpha = 0.95" in counts
    beta = (output / "beta_sensitivity.md").read_text()
    assert "| 1 | 0.8 | 0.5 | 1 |" in beta
    assert "| 1 | 0.95 | 0 | 0 |" in beta
    summaries = pl.read_csv(output / "csig_summaries.csv")
    assert set(summaries["beta_times_c_squared"]) == {25.6, 51.2}
    sig_summaries = pl.read_csv(output / "sig_mmd_summaries.csv")
    assert sig_summaries.columns == ["sigma", "split", "conditioning", "model", "score"]
    assert sig_summaries.rows() == [
        (1.0, "plot", "noenv_nogeno", "first", pytest.approx(1.0)),
        (1.0, "plot", "noenv_nogeno", "second", pytest.approx(2.0)),
    ]
    csig_main = pl.read_csv(output / "csig_mmd_summaries.csv")
    assert csig_main.columns == [
        "sigma",
        "alpha",
        "split",
        "conditioning",
        "model",
        "score",
    ]
    assert sorted(csig_main.rows()) == [
        (1.0, 0.8, "plot", "noenv_nogeno", "first", pytest.approx(1.0)),
        (1.0, 0.8, "plot", "noenv_nogeno", "second", pytest.approx(2.0)),
        (1.0, 0.95, "plot", "noenv_nogeno", "first", pytest.approx(2.0)),
        (1.0, 0.95, "plot", "noenv_nogeno", "second", pytest.approx(6.1 / 3)),
    ]
    assert "| 0.8 | 2 | 1 | 0.75 |" in (output / "lodged_recall.md").read_text()


def test_lodged_recall_counts_lodged_draws_above_the_threshold():
    distances = np.array([1.0, 3.0, 5.0, 7.0])
    has_lodged = np.array([False, True, True, True])
    assert table.lodged_recall(distances, has_lodged, 4.0) == pytest.approx(2 / 3)
