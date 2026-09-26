from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from npnf.data.configs.datasets.synthetic_test_sets import split_named
from npnf.scripts.paper import block_uncertainty_table as but


def _write_blocks(
    root: Path,
    *,
    model: str,
    scores: list[float],
    split: str = "plot",
    conditioning: str = "noenv_nogeno",
    prediction_method: str = "no_context",
    metric_type: str = "sig_mmd",
    view: str = "context_matched_blocked",
    checkpoint: str = "checkpoint-3000000",
    dataset_size: str = "6",
    unit_scope: str = "global",
    extra_rows: list[dict] | None = None,
) -> Path:
    test_split = split_named(split, "legacy")
    dataloader_name, split_dir = test_split.dataloader_name, test_split.split_key
    directory = (
        root
        / dataloader_name
        / model
        / dataset_size
        / checkpoint
        / split_dir
        / conditioning
        / prediction_method
        / but.metric_dir_name(metric_type, view)
    )
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source": "model",
            "unit_scope": unit_scope,
            "unit_id": "u0",
            "block_index": index,
            "score": s,
        }
        for index, s in enumerate(scores)
    ]
    rows.extend(extra_rows or [])
    path = directory / but.block_file_name(metric_type)
    pl.DataFrame(rows).write_csv(path)
    return path


def _frame(block_indices: list[int], scores: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unit_scope": ["global"] * len(scores),
            "unit_id": ["u0"] * len(scores),
            "block_index": block_indices,
            "score": scores,
        }
    )


def test_paired_stat_matches_hand_computed_values():
    a = _frame([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    b = _frame([0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])

    stat = but.paired_stat("A", "B", "plot", "noenv_nogeno", "no_context", a, b, None)

    assert stat is not None
    # differences are [0, 1, 2, 3]: mean 1.5, sd(ddof=1) sqrt(5/3), se sd/2
    assert stat.n_blocks == 4
    assert stat.mean_diff == pytest.approx(1.5)
    assert stat.standard_error == pytest.approx((5 / 3) ** 0.5 / 2)
    assert stat.t_value == pytest.approx(1.5 / ((5 / 3) ** 0.5 / 2))


def test_paired_stat_with_equal_block_differences_has_no_division_by_zero():
    censored = _frame([0, 1, 2], [0.5, 0.5, 0.5])
    shifted = _frame([0, 1, 2], [0.25, 0.25, 0.25])

    same = but.paired_stat(
        "A", "B", "plot", "noenv_nogeno", "no_context", censored, censored, None
    )
    apart = but.paired_stat(
        "A", "B", "plot", "noenv_nogeno", "no_context", censored, shifted, None
    )

    assert same is not None
    assert apart is not None
    assert same.standard_error == 0.0
    assert same.t_value == 0.0
    assert apart.t_value == math.inf


def test_paired_stat_pairs_on_block_index_not_row_order():
    a = _frame([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    ordered = _frame([0, 1, 2, 3], [4.0, 1.0, 3.0, 2.0])
    shuffled = ordered.sort("score")

    from_ordered = but.paired_stat(
        "A", "B", "plot", "c", "no_context", a, ordered, None
    )
    from_shuffled = but.paired_stat(
        "A", "B", "plot", "c", "no_context", a, shuffled, None
    )

    assert from_ordered is not None
    assert from_shuffled is not None
    assert from_shuffled.t_value == pytest.approx(from_ordered.t_value)


def test_paired_stat_ignores_unmatched_block_indices():
    a = _frame([0, 1, 2], [1.0, 2.0, 3.0])
    b = _frame([1, 2, 9], [1.0, 1.0, 5.0])

    stat = but.paired_stat("A", "B", "plot", "c", "no_context", a, b, None)

    assert stat is not None
    assert stat.n_blocks == 2


def test_read_block_scores_drops_oracle_side_rows(tmp_path):
    path = _write_blocks(
        tmp_path,
        model="LNP",
        scores=[0.1, 0.2],
        extra_rows=[
            {
                "source": "oracle",
                "unit_scope": "global",
                "unit_id": "u0",
                "block_index": 0,
                "score": 9.9,
            }
        ],
    )

    frame = but.read_block_scores(path)

    assert frame["score"].to_list() == [0.1, 0.2]


def test_read_block_scores_keeps_non_global_scopes(tmp_path):
    """Conditioned cells block by genotype/environment/condition, not globally."""
    path = _write_blocks(
        tmp_path,
        model="LNP",
        scores=[0.1, 0.2],
        conditioning="noenv_geno",
        unit_scope="genotype",
    )

    frame = but.read_block_scores(path)

    assert frame.height == 2
    assert frame["unit_scope"].unique().to_list() == ["genotype"]


def test_find_block_path_returns_none_for_missing_cell(tmp_path):
    assert (
        but.find_block_path(
            tmp_path,
            "LNP",
            "plot",
            "noenv_nogeno",
            "no_context",
            "sig_mmd",
            "context_matched_blocked",
            "checkpoint-3000000",
        )
        is None
    )


def test_find_block_path_globs_over_dataset_size(tmp_path):
    written = _write_blocks(tmp_path, model="LNP", scores=[0.1], dataset_size="6")

    found = but.find_block_path(
        tmp_path,
        "LNP",
        "plot",
        "noenv_nogeno",
        "no_context",
        "sig_mmd",
        "context_matched_blocked",
        "checkpoint-3000000",
    )

    assert found == written


def test_collect_pair_stats_reports_missing_cells_without_failing(tmp_path):
    _write_blocks(tmp_path, model="LNP", scores=[0.1, 0.2, 0.3, 0.5])
    _write_blocks(tmp_path, model="ANP", scores=[0.2, 0.2, 0.4, 0.4])
    # ANP is missing for env_geno, so that cell yields no pair.

    stats, missing = but.collect_pair_stats(
        tmp_path,
        ["LNP", "ANP"],
        {"LNP": "LNP", "ANP": "ANP"},
        ["plot"],
        ["noenv_nogeno", "env_geno"],
        "no_context",
        "sig_mmd",
        "context_matched_blocked",
        "checkpoint-3000000",
        "LNP",
    )

    assert len(stats) == 1
    assert stats[0].conditioning == "noenv_nogeno"
    # ANP, LNP and the baseline are all unscored for env_geno
    assert len(missing) == 3
    assert all("env_geno" in entry for entry in missing)


def test_build_latex_table_has_one_row_per_pair_and_all_columns(tmp_path):
    for model, scores in (
        ("LNP", [0.1, 0.2, 0.3, 0.5]),
        ("ANP", [0.2, 0.2, 0.4, 0.4]),
        ("CNP", [0.3, 0.1, 0.5, 0.3]),
    ):
        for conditioning in ("noenv_nogeno", "env_geno"):
            _write_blocks(
                tmp_path, model=model, scores=scores, conditioning=conditioning
            )

    models = ["LNP", "ANP", "CNP"]
    stats, missing = but.collect_pair_stats(
        tmp_path,
        models,
        {model: model for model in models},
        ["plot"],
        ["noenv_nogeno", "env_geno"],
        "no_context",
        "sig_mmd",
        "context_matched_blocked",
        "checkpoint-3000000",
        "CNP",
    )
    table = but.build_latex_table(
        stats,
        splits=["plot"],
        split_labels=["Plot"],
        conditionings=["noenv_nogeno", "env_geno"],
        conditioning_labels=["P", "E&G"],
        metric_type="sig_mmd",
        decimals=1,
    )

    assert missing == []
    assert table.startswith(r"\begin{tabular}{lrr}")
    assert table.rstrip().endswith(r"\end{tabular}")
    # three pairs from three models
    assert table.count(" vs ") == 3
    assert r"E\&G" in table


def test_validate_block_view_rejects_condition_view():
    with pytest.raises(ValueError, match="condition"):
        but.validate_block_view("condition")


def test_relative_diff_divides_by_the_baseline_mean():
    a = _frame([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    b = _frame([0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])

    stat = but.paired_stat("A", "B", "plot", "c", "no_context", a, b, 0.5)

    assert stat is not None
    # mean_diff is 1.5, baseline mean is 0.5
    assert stat.relative_diff == pytest.approx(3.0)


def test_relative_diff_shares_one_denominator_across_pairs_in_a_cell(tmp_path):
    for model, scores in (
        ("LNP", [0.1, 0.2, 0.3, 0.5]),
        ("ANP", [0.2, 0.2, 0.4, 0.4]),
        ("CNP", [0.4, 0.4, 0.4, 0.4]),
    ):
        _write_blocks(tmp_path, model=model, scores=scores)

    models = ["LNP", "ANP"]
    stats, _ = but.collect_pair_stats(
        tmp_path,
        models,
        {model: model for model in models},
        ["plot"],
        ["noenv_nogeno"],
        "no_context",
        "sig_mmd",
        "context_matched_blocked",
        "checkpoint-3000000",
        "CNP",
    )

    assert len(stats) == 1
    # CNP's blocks are all 0.4, so every pair in the cell divides by 0.4
    assert stats[0].relative_diff == pytest.approx(stats[0].mean_diff / 0.4)


def test_baseline_model_is_not_compared_as_a_pair_member(tmp_path):
    for model, scores in (
        ("LNP", [0.1, 0.2, 0.3, 0.5]),
        ("ANP", [0.2, 0.2, 0.4, 0.4]),
        ("CNP", [0.3, 0.1, 0.5, 0.3]),
    ):
        _write_blocks(tmp_path, model=model, scores=scores)

    models = ["LNP", "ANP"]
    stats, _ = but.collect_pair_stats(
        tmp_path,
        models,
        {model: model for model in models},
        ["plot"],
        ["noenv_nogeno"],
        "no_context",
        "sig_mmd",
        "context_matched_blocked",
        "checkpoint-3000000",
        "CNP",
    )

    assert len(stats) == 1
    assert "CNP" not in {stats[0].model_a, stats[0].model_b}


def test_relative_diff_is_none_when_the_baseline_cell_is_missing(tmp_path):
    _write_blocks(tmp_path, model="LNP", scores=[0.1, 0.2, 0.3, 0.5])
    _write_blocks(tmp_path, model="ANP", scores=[0.2, 0.2, 0.4, 0.4])

    models = ["LNP", "ANP"]
    stats, missing = but.collect_pair_stats(
        tmp_path,
        models,
        {model: model for model in models},
        ["plot"],
        ["noenv_nogeno"],
        "no_context",
        "sig_mmd",
        "context_matched_blocked",
        "checkpoint-3000000",
        "CNP",
    )

    assert len(stats) == 1
    assert stats[0].relative_diff is None
    # the t value survives a missing baseline
    assert stats[0].t_value != 0.0
    assert any("baseline" in entry for entry in missing)


def test_relative_diff_is_none_for_a_zero_baseline():
    a = _frame([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    b = _frame([0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])

    stat = but.paired_stat("A", "B", "plot", "c", "no_context", a, b, 0.0)

    assert stat is not None
    assert stat.relative_diff is None


def test_prediction_method_reaches_the_long_frame():
    a = _frame([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    b = _frame([0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
    stat = but.paired_stat("A", "B", "plot", "c", "max_height", a, b, 1.0)

    assert stat is not None
    frame = but.build_long_dataframe([stat])

    assert frame["prediction_method"].to_list() == ["max_height"]


def _stat(model_a, model_b, t_value, relative_diff=0.1, conditioning="c"):
    return but.PairStat(
        model_a=model_a,
        model_b=model_b,
        split="plot",
        conditioning=conditioning,
        prediction_method="no_context",
        n_blocks=1000,
        mean_a=1.0,
        mean_b=1.0,
        mean_diff=0.1,
        standard_error=0.01,
        t_value=t_value,
        correlation=0.5,
        relative_diff=relative_diff,
    )


def test_build_pair_summary_aggregates_and_sorts_by_median_abs_t():
    stats = [
        _stat("A", "B", 10.0, conditioning="c1"),
        _stat("A", "B", -20.0, conditioning="c2"),
        _stat("C", "D", 1.0, conditioning="c1"),
        _stat("C", "D", -3.0, conditioning="c2"),
    ]

    summary = but.build_pair_summary(stats)

    # C vs D has the lower median |t| and sorts first
    assert summary["model_a"].to_list() == ["C", "A"]
    assert summary["median_abs_t"].to_list() == pytest.approx([2.0, 15.0])
    assert summary["n_cells"].to_list() == [2, 2]
    # |t| < 2 holds for one of C vs D's two cells and neither of A vs B's
    assert summary["n_weak"].to_list() == [1, 0]
    assert summary["frac_weak"].to_list() == pytest.approx([0.5, 0.0])


def test_build_pair_summary_ignores_null_relative_diffs():
    stats = [
        _stat("A", "B", 5.0, relative_diff=None, conditioning="c1"),
        _stat("A", "B", 6.0, relative_diff=0.2, conditioning="c2"),
    ]

    summary = but.build_pair_summary(stats)

    assert summary["median_abs_relative_diff"].to_list() == pytest.approx([0.2])


def test_build_summary_latex_marks_majority_unresolved_pairs():
    stats = [
        _stat("A", "B", 0.5, conditioning="c1"),
        _stat("A", "B", 0.7, conditioning="c2"),
        _stat("C", "D", 30.0, conditioning="c1"),
        _stat("C", "D", 40.0, conditioning="c2"),
    ]

    latex = but.build_summary_latex(
        but.build_pair_summary(stats),
        metric_type="sig_mmd",
        baseline_label="CNP",
        decimals=1,
    )

    assert latex.startswith(r"\begin{tabular}{lrrrr}")
    assert latex.rstrip().endswith(r"\end{tabular}")
    # A vs B is weak in both cells and is marked; C vs D is not
    assert r"\textbf{A vs B}" in latex
    assert r"\textbf{C vs D}" not in latex


@pytest.mark.parametrize(
    ("test_set_kwargs", "split", "dataloader_name"),
    [
        ({}, "plot", "synth_test_plot_dataloaders"),
        ({"test_set": "shifted"}, "seen", "synth_shifted_seen_dataloaders"),
    ],
)
def test_find_block_path_defaults_to_the_legacy_test_set(
    tmp_path, test_set_kwargs, split, dataloader_name
):
    directory = (
        tmp_path
        / dataloader_name
        / "LNP"
        / "6"
        / "checkpoint-3000000"
        / f"test_{split}"
        / "noenv_nogeno"
        / "no_context"
        / but.metric_dir_name("sig_mmd", "context_matched_blocked")
    )
    directory.mkdir(parents=True)
    written = directory / but.block_file_name("sig_mmd")
    written.write_text("source,unit_scope,unit_id,block_index,score\n")

    found = but.find_block_path(
        tmp_path,
        "LNP",
        split,
        "noenv_nogeno",
        "no_context",
        "sig_mmd",
        "context_matched_blocked",
        "checkpoint-3000000",
        **test_set_kwargs,
    )

    assert found == written
