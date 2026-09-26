import polars as pl

from npnf.scripts.paper.fip1_sig_mmd_table import (
    TableLabels,
    collect,
    initialisation_table,
    mean_sd_text,
    pairing_table,
    parse_run_name,
)

SUMMARY_CSV = (
    "method_dir,mode,unit_scope,split,n_units,n_blocks,n_plots,model_draws,"
    "context_sigma,day_tolerance,n_anchor_days,sig_mmd_x1000,"
    "sig_mmd_x1000_median,sig_mmd_x1000_std\n"
    "d,no_context,global,test_plot,1,1,250,64,0.02,2,13,12.5,12.5,0.5\n"
)


def test_parse_run_name_reads_fip1_runs():
    assert parse_run_name(
        "LNP-NF-Prior-Posterior-FIP-2930-fip1_pretrained_1m_seed13"
    ) == ("LNP-NF-Prior-Posterior", "pretrained", 13)
    assert parse_run_name("CNP-FIP-2930-fip1_scratch_1m_seed2") == ("CNP", "scratch", 2)


def test_mean_sd_text_drops_the_pm_for_a_single_seed():
    assert mean_sd_text(1.5, None) == "1.500"
    assert mean_sd_text(1.5, 0.25) == "1.500 ± 0.250"


def test_collect_skips_the_synthetic_checkpoints(tmp_path):
    for relative in [
        (
            "fip1_test_plot_dataloaders/CNP-FIP-2930-fip1_scratch_1m_seed2/"
            "checkpoint-1000000/test_plot/noenv_nogeno/no_context"
        ),
        (
            "fip1_test_plot_dataloaders/"
            "CNP-512k-training3m_set_mode_nested_noprior_seed2/"
            "6/checkpoint-3000000/test_plot/noenv_nogeno/no_context"
        ),
    ]:
        scorer_dir = tmp_path / relative / "sig_mmd_fip1_context_matched_blocked"
        scorer_dir.mkdir(parents=True)
        (scorer_dir / "sig_mmd_summary.csv").write_text(SUMMARY_CSV)

    results = collect(tmp_path)
    assert results["run"].to_list() == ["CNP-FIP-2930-fip1_scratch_1m_seed2"]
    assert set(results["split"].to_list()) == {"test_plot"}
    assert set(results["day_tolerance"].to_list()) == {2}
    assert set(results["n_anchor_days"].to_list()) == {13}


def test_initialisation_table_spreads_over_configurations_not_settings():
    # Each configuration enters as its mean over the covariate settings (2 and 5),
    # so the sd is over the two configurations, not over the four cells.
    summary = pl.DataFrame(
        {
            "split": ["test_plot"] * 4,
            "mode": ["max_height"] * 4,
            "init": ["pretrained"] * 4,
            "model": ["ANP", "ANP", "LNP", "LNP"],
            "conditioning": ["noenv_nogeno", "env_geno"] * 2,
            "mean": [1.0, 3.0, 4.0, 6.0],
        }
    )
    labels = TableLabels(
        splits=(("test_plot", "seen"),),
        modes=(("max_height", "max-height context"),),
        inits=(("pretrained", "pre"),),
    )
    table = initialisation_table(summary, labels)
    assert table == (
        "| Test set | max-height context pre |\n|---|---:|\n| seen | 3.50 ± 2.12 |"
    )


def test_pairing_table_counts_the_pairs_where_pretraining_is_closer():
    results = pl.DataFrame(
        {
            "model": ["ANP"] * 4,
            "seed": [2, 2, 3, 3],
            "split": ["test_plot"] * 4,
            "conditioning": ["noenv_nogeno"] * 4,
            "mode": ["no_context"] * 4,
            "init": ["scratch", "pretrained"] * 2,
            "sig_mmd_x1000": [5.0, 4.0, 2.0, 2.5],
        }
    )
    table = pairing_table(results, TableLabels(modes=(("no_context", "no context"),)))
    assert table.splitlines()[-1] == "| no context | 2 | 1 | 0.25 |"
