import numpy as np
import polars as pl

from npnf.scripts.paper import mmd_table
from npnf.scripts.paper.fip1_paper_results import cell_path, fold_seeds, model_dir_name
from npnf.scripts.paper.fip1_sig_mmd_table import collect

SUMMARY_TEMPLATE = (
    "method_dir,mode,unit_scope,split,n_units,n_blocks,n_plots,model_draws,"
    "sig_mmd_x1000,sig_mmd_x1000_median,sig_mmd_x1000_std,sig_mmd_x1000_se,"
    "context_sigma,day_tolerance,n_anchor_days\n"
    "d,no_context,global,test_plot,1,1,250,64,{score},{score},0.5,0.5,0.02,2,13\n"
)


def _write_campaign(root, scores_by_seed):
    for seed, score in scores_by_seed.items():
        scorer_dir = (
            root
            / "fip1_test_plot_dataloaders"
            / f"CNP-FIP-2930-fip1_pretrained_1m_seed{seed}"
            / "checkpoint-1000000"
            / "test_plot"
            / "noenv_nogeno"
            / "no_context"
            / "sig_mmd_fip1_context_matched_blocked"
        )
        scorer_dir.mkdir(parents=True)
        (scorer_dir / "sig_mmd_summary.csv").write_text(
            SUMMARY_TEMPLATE.format(score=score)
        )


def test_fold_seeds_averages_the_seeds_into_raw_synthetic_columns(tmp_path):
    _write_campaign(tmp_path, {2: 4.0, 13: 6.0, 31: 8.0})
    folded = fold_seeds(collect(tmp_path))

    assert folded.height == 1
    row = folded.to_dicts()[0]
    assert row["n_seeds"] == 3
    assert row["seeds"] == "2,13,31"
    # `mean` is the synthetic column and carries raw units, so the table's own
    # x1000 scaling reproduces the campaign's reported score.
    assert row["mean"] == 6.0 / 1000.0
    assert row["mean_x1000"] == 6.0
    assert np.isclose(row["sd_x1000"], 2.0)
    assert row["view"] == "context_matched_blocked"


def test_written_cell_sits_where_mmd_table_globs_for_it(tmp_path):
    _write_campaign(tmp_path, {2: 4.0})
    row = fold_seeds(collect(tmp_path)).to_dicts()[0]
    path = cell_path(row, tmp_path / "folded")
    path.parent.mkdir(parents=True)
    pl.DataFrame([row]).write_csv(path)

    spec = mmd_table.TableSpec(
        model=model_dir_name("CNP", "pretrained"),
        model_name="CNP",
        split="fip1_plot",
        split_label="Plot",
        conditioning="noenv_nogeno",
        conditioning_label="P",
    )
    summaries = mmd_table.load_summaries(
        tmp_path / "folded",
        [spec],
        prediction_method="no_context",
        metric_type="sig_mmd",
        checkpoint="latest",
        metric="mean",
        view="context_matched_blocked",
    )
    assert [summary.value for summary in summaries] == [4.0 / 1000.0]


def test_baseline_run_folds_in_under_its_run_name(tmp_path):
    _write_campaign(tmp_path, {2: 4.0})
    scorer_dir = (
        tmp_path
        / "fip1_test_plot_dataloaders"
        / "Baseline-FIP-2930"
        / "checkpoint-0"
        / "test_plot"
        / "noenv_nogeno"
        / "no_context"
        / "sig_mmd_fip1_context_matched_blocked"
    )
    scorer_dir.mkdir(parents=True)
    (scorer_dir / "sig_mmd_summary.csv").write_text(SUMMARY_TEMPLATE.format(score=1.5))

    assert collect(tmp_path).height == 1
    folded = fold_seeds(collect(tmp_path, ["Baseline-FIP-2930"]))

    baseline = folded.filter(pl.col("model") == "Baseline-FIP-2930").to_dicts()[0]
    assert baseline["n_seeds"] == 1
    assert baseline["mean_x1000"] == 1.5
    path = cell_path(baseline, tmp_path / "folded")
    assert path.parts[-8] == "Baseline-FIP-2930"
    assert path.parts[-6] == "checkpoint-0"
