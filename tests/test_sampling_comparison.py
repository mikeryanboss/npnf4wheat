from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from npnf.scripts.paper import sampling_comparison


def _write_method_summary(parent: Path, method: str, means: list[float]) -> None:
    method_dir = parent / method
    method_dir.mkdir(parents=True)
    pl.DataFrame(
        {"num_context": list(range(len(means))), "n": [10] * len(means), "mean": means}
    ).write_csv(method_dir / "sig_mmd_by_context.csv")


def _write_varying_parent(parent: Path, offset: float = 0.0) -> Path:
    _write_method_summary(parent, "random", [offset + 0.1, offset + 0.2])
    _write_method_summary(parent, "uncertainty", [offset + 0.3, offset + 0.4])
    return parent


def test_sampling_comparison_single_folder_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    parent = _write_varying_parent(tmp_path / "inputs" / "model-a")

    sampling_comparison.comparison([str(parent)])

    output_dir = tmp_path / "paper" / "sampling_comparison"
    assert (output_dir / "sig_mmd_comparison_plot.png").exists()
    assert (output_dir / "sig_mmd_comparison_plot.pdf").exists()

    df = pl.read_csv(output_dir / "sig_mmd_comparison.csv")
    assert df.columns == [
        "Context Points",
        "Sig-MMD",
        sampling_comparison.PLOT_COLUMN,
        "Sampling",
        "Model",
    ]
    assert df["Model"].unique().to_list() == ["model-a"]
    assert sorted(df["Sampling"].unique().to_list()) == ["Random", "Uncertainty"]


def test_sampling_comparison_combines_two_named_model_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lnp = _write_varying_parent(tmp_path / "inputs" / "lnp", offset=0.0)
    anp = _write_varying_parent(tmp_path / "inputs" / "anp", offset=1.0)

    sampling_comparison.comparison([str(lnp), str(anp)], model_names=["LNP", "ANP"])

    df = pl.read_csv(
        tmp_path / "paper" / "sampling_comparison" / "sig_mmd_comparison.csv"
    )
    assert sorted(df["Model"].unique().to_list()) == ["ANP", "LNP"]
    assert sorted(df["Sampling"].unique().to_list()) == ["Random", "Uncertainty"]
    assert df.height == 8
    anp_random = df.filter(
        (pl.col("Model") == "ANP") & (pl.col("Sampling") == "Random")
    )
    assert anp_random["Sig-MMD"].to_list() == [1.1, 1.2]
    assert anp_random[sampling_comparison.PLOT_COLUMN].to_list() == [1100.0, 1200.0]


def test_sampling_comparison_rejects_mismatched_model_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        sampling_comparison.main(
            [
                "--results-folder",
                str(tmp_path / "lnp"),
                str(tmp_path / "anp"),
                "--model-names",
                "LNP",
            ]
        )

    assert (
        "--model-names count (1) must match --results-folder count (2)"
        in capsys.readouterr().err
    )
