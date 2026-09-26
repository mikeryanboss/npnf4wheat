import subprocess
import sys

import numpy as np
import pytest
import tensordict
import torch

from npnf.data.fip1_day_grid import align_days
from npnf.metrics.blocks import UnitMember, UnitScope, unit_seed
from npnf.metrics.sig_mmd import kernel_grams
from npnf.metrics.signature import cosine_normalized, unbiased_self_term
from npnf.scripts.metrics.fip1_metric_grid import (
    heights_at_days,
    metric_paths,
    unit_yearsite_days,
)
from npnf.scripts.metrics.sig_mmd_fip1_context_matched_blocked import (
    score_unit,
    split_units,
    weighted_sig_mmd_x1000,
)
from npnf.scripts.utils.prediction import _BATCH_METADATA_FIELDS, peak_height_index


def uniform(count: int) -> np.ndarray:
    return np.full(count, 1.0 / count)


def test_fip1_batch_schema_keys_are_flattenable_metadata():
    # `height` holds ragged per-plot tensors, which breaks `_extract_metadata`'s
    # flatten(); the FIP1 schema must stay on plain list columns.
    assert _BATCH_METADATA_FIELDS["fip1"] == ("yearsite_uid", "genotype_id", "plot_uid")


def test_peak_height_index_prefers_the_clean_curve_when_present():
    heights = tensordict.TensorDict(
        {
            "Y": torch.tensor([[0.1], [0.9], [0.5]]),
            "Y_original": torch.tensor([[0.1], [0.2], [0.7]]),
        },
        batch_size=[3],
    )
    assert peak_height_index(heights) == 2


def test_peak_height_index_falls_back_to_the_robust_observed_peak():
    # Median of the top three, so the lone 2.0 spike does not become the peak.
    heights = tensordict.TensorDict(
        {"Y": torch.tensor([[0.1], [0.5], [0.6], [2.0], [0.55]])}, batch_size=[5]
    )
    assert peak_height_index(heights) == 2


def test_single_reference_path_gives_the_kernel_score():
    # n=1 must not be rejected: it is the proper kernel score against a point mass,
    # and N identical copies must give the same value.
    rng = np.random.default_rng(3)
    days = np.arange(200, 280, 4)
    truth = 0.9 / (1 + np.exp(-(days - 240) / 10))
    model = metric_paths(truth[None, :] + rng.normal(0, 0.03, (64, days.size)), days)
    one = truth[None, :] + rng.normal(0, 0.03, (1, days.size))
    single = weighted_sig_mmd_x1000(model, metric_paths(one, days), uniform(1))
    repeated = weighted_sig_mmd_x1000(
        model, metric_paths(np.repeat(one, 8, 0), days), uniform(8)
    )
    assert np.isfinite(single)
    assert single == pytest.approx(repeated, rel=1e-4, abs=1e-5)
    wrong = metric_paths(
        0.6 * truth[None, :] + rng.normal(0, 0.03, (64, days.size)), days
    )
    assert (
        weighted_sig_mmd_x1000(wrong, metric_paths(one, days), uniform(1)) > 10 * single
    )


def test_deterministic_model_reaches_self_x_exactly_one():
    # The point of the aligned grid: one curve evaluated on one grid gives identical
    # paths, so the concentration penalty cannot be avoided. Per-yearsite grids gave
    # the same curve one path per year and `self_x` fell below 1.
    days = np.array([210, 218, 229, 241, 250])
    jagged = np.array([0.05, 0.40, 0.18, 0.62, 0.55])
    model = metric_paths(np.tile(jagged, (8, 1)), days)
    reference = metric_paths(
        np.stack([jagged + 0.05 * index for index in range(4)]), days
    )
    k_xx, _, _ = kernel_grams(model, reference)
    diagonal = k_xx.diagonal()
    normalized = cosine_normalized(k_xx, diagonal, diagonal)
    assert float(unbiased_self_term(normalized)) == pytest.approx(1.0, abs=1e-6)


def test_reference_sampling_is_reproducible_across_processes():
    # `hash()` of a str is salted per process, so seeding the reference draw on it
    # made the same command return a different score per run and ignored `--seed`.
    # The seed must come from `unit_seed`, which is stable.
    script = (
        "from npnf.metrics.blocks import UnitScope, unit_seed;"
        "print(unit_seed(0, UnitScope.GLOBAL, 'global'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert runs == {str(unit_seed(0, UnitScope.GLOBAL, "global"))}
    assert unit_seed(0, UnitScope.GLOBAL, "global") != unit_seed(
        1, UnitScope.GLOBAL, "global"
    )


def test_score_unit_scores_the_model_on_the_anchors_and_the_plots_on_their_days():
    # Pins the two sides apart: the reference takes each plot's real measurement at
    # its own yearsite's assigned day, the model is evaluated at the anchor day. If
    # the model were sampled at the assigned days instead, one deterministic curve
    # would again look like one path per yearsite.
    day_axis = np.arange(200, 260)
    curve = torch.tensor(
        [[0.02 * index + 0.05 * (index % 2) for index in range(day_axis.size)]]
    )
    plots = {
        "A": ([210, 220, 230, 240], [0.05, 0.30, 0.60, 0.80]),
        "B": ([211, 221, 232, 241], [0.10, 0.35, 0.55, 0.85]),
    }
    records = [
        {
            "plot_uid": f"{yearsite}{shift}",
            "gid": "g",
            "ys": yearsite,
            "grid": curve,
            "days": np.array(days),
            "heights": np.array(heights) + shift,
            "context_days": np.array([], dtype=int),
            "context_values": np.array([], dtype=float),
        }
        for yearsite, (days, heights) in plots.items()
        for shift in (0.0, 0.03)
    ]
    members = [
        UnitMember(
            genotype_id="g",
            yearsite_uid=record["ys"],  # ty: ignore[invalid-argument-type]
            condition_index=index,
        )
        for index, record in enumerate(records)
    ]
    grid = align_days(unit_yearsite_days(members, records), tolerance=2)
    anchors, assigned = grid.anchors, grid.assigned
    assert anchors.size == 4

    rows = score_unit(
        members,
        records,
        day_axis,
        anchors,
        assigned,
        draw_count=1,
        n_blocks=16,
        block_size=512,
        seed=0,
        unit_scope=UnitScope.GLOBAL,
        unit_id="global",
        sigma=0.02,
        min_reference_plots=1,
    )
    # Four draws fit into one block, so the 16 blocks would all be the same.
    (row,) = rows
    assert row["n_days"] == anchors.size
    assert row["n_yearsites"] == 2
    assert row["n_model"] == len(records)

    positions = np.searchsorted(day_axis, anchors)
    expected_model = metric_paths(curve[:, positions].repeat(len(records), 1), anchors)
    expected_reference = metric_paths(
        np.stack(
            [
                heights_at_days(record, assigned[record["ys"]])  # ty: ignore[invalid-argument-type]
                for record in sorted(records, key=lambda r: (r["ys"], r["plot_uid"]))
            ]
        ),
        anchors,
    )
    assert row["sig_mmd_x1000"] == pytest.approx(
        weighted_sig_mmd_x1000(
            expected_model, expected_reference, uniform(len(records))
        ),
        rel=1e-5,
        abs=1e-5,
    )


def test_two_trials_of_one_harvest_year_are_one_environment():
    # FIP1 2019 has two trials with one temperature record; a model with the
    # environment covariate cannot tell them apart, so they must be one E unit,
    # and a genotype grown in both must be one E&G unit. Each plot keeps its own
    # trial, which selects the measurement days it is read at.
    records = [
        {"gid": gid, "ys": trial, "environment": year}
        for gid, trial, year in (
            ("g1", "FPWW024", "2019"),
            ("g1", "FPWW028", "2019"),
            ("g2", "FPWW028", "2019"),
            ("g1", "FPWW022", "2018"),
        )
    ]
    environment = split_units(records, UnitScope.ENVIRONMENT)
    assert sorted(environment) == ["2018", "2019"]
    assert [m.yearsite_uid for m in environment["2019"]] == [
        "FPWW024",
        "FPWW028",
        "FPWW028",
    ]
    condition = split_units(records, UnitScope.CONDITION)
    assert sorted(condition) == ["g1::2018", "g1::2019", "g2::2019"]
    assert len(condition["g1::2019"]) == 2
