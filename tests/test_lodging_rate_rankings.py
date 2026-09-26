from __future__ import annotations

import polars as pl
import pytest

from npnf.scripts.paper import lodging_rate_rankings as rankings


def _long_table(path, means):
    rows = [
        {
            "model_name": model,
            "prediction_method": method,
            "conditioning": conditioning,
            "mean": value,
        }
        for (method, conditioning), values in means.items()
        for model, value in values.items()
    ]
    pl.DataFrame(rows).write_csv(path)
    return path


def test_rank_models_averages_ranks_over_complete_columns(tmp_path):
    path = _long_table(
        tmp_path / "rate.csv",
        {
            ("no_context", "env_geno"): {"A": 0.1, "B": 0.2, "C": 0.3},
            ("max_height", "env_geno"): {"A": 0.3, "B": 0.2, "C": 0.1},
            ("max_height", "noenv_geno"): {"A": 0.1, "B": 0.2},
        },
    )

    ranked = rankings.rank_models(pl.read_csv(path)).sort("model_name")

    assert ranked["columns"].to_list() == [2, 2, 2]
    assert ranked["mean_rank"].to_list() == pytest.approx([2.0, 2.0, 2.0])
    assert ranked["first"].to_list() == [1, 0, 1]


def test_lodging_rate_rankings_compares_each_rate_with_the_first(tmp_path):
    first = _long_table(
        tmp_path / "first.csv",
        {("no_context", "env_geno"): {"A": 0.1, "B": 0.2, "C": 0.3}},
    )
    reversed_order = _long_table(
        tmp_path / "second.csv",
        {("no_context", "env_geno"): {"A": 0.3, "B": 0.2, "C": 0.1}},
    )

    wide, tau_lines = rankings.lodging_rate_rankings(
        {"19.7": first, "2.7": reversed_order}
    )

    assert wide["model_name"].to_list() == ["A", "B", "C"]
    assert wide["mean_rank_19.7"].to_list() == [1.0, 2.0, 3.0]
    assert wide["mean_rank_2.7"].to_list() == [3.0, 2.0, 1.0]
    assert tau_lines == [
        "Kendall tau of mean ranks (3 models), 19.7 vs 2.7: -1.000 (p = 0.333)"
    ]


def test_lodging_rate_rankings_limits_kendall_tau_to_selected_models(tmp_path):
    first = _long_table(
        tmp_path / "first.csv",
        {("no_context", "env_geno"): {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.9}},
    )
    second = _long_table(
        tmp_path / "second.csv",
        {("no_context", "env_geno"): {"A": 0.3, "B": 0.2, "C": 0.1, "D": 0.9}},
    )

    wide, tau_lines = rankings.lodging_rate_rankings(
        {"19.7": first, "2.7": second}, kendall_models=["A", "B", "C"]
    )

    assert wide["mean_rank_2.7"].to_list() == [3.0, 2.0, 1.0, 4.0]
    assert tau_lines == [
        "Kendall tau of mean ranks (3 models), 19.7 vs 2.7: -1.000 (p = 0.333)"
    ]


def test_lodging_rate_rankings_rejects_unknown_kendall_models(tmp_path):
    table = _long_table(
        tmp_path / "rate.csv", {("no_context", "env_geno"): {"A": 0.1, "B": 0.2}}
    )

    with pytest.raises(ValueError, match="not in the tables"):
        rankings.lodging_rate_rankings({"a": table, "b": table}, kendall_models=["Z"])


def test_parse_tables_rejects_entries_without_label():
    with pytest.raises(ValueError, match="LABEL=PATH"):
        rankings.parse_tables(["only_a_path.csv"])
