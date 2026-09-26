import datasets
import numpy as np
import pytest
import torch

from npnf.data.fip1_day_grid import (
    METRIC_WINDOW,
    SharedDayGrid,
    align_days,
    observed_at_grid,
    observed_on_common_days,
    select_year,
    shared_days,
)


def test_shared_days_keeps_the_common_dates_inside_the_window():
    days = shared_days(
        [np.array([190, 210, 230, 250, 330]), np.array([210, 230, 250, 330, 340])]
    )
    assert days.tolist() == [210, 230, 250]
    assert days.min() >= METRIC_WINDOW[0]
    assert days.max() < METRIC_WINDOW[1]


def test_align_days_takes_exact_matches_first_and_uses_each_measurement_date_once():
    grid = align_days(
        {"A": np.array([10, 20, 31]), "B": np.array([11, 20, 30])},
        tolerance=1,
        min_gap=3,
    )
    assert grid.anchors.tolist() == [10, 20, 30]
    # Day 20 is exact for both, so it is matched in the first stage and neither
    # site-year can spend it on a neighbouring anchor.
    assert grid.assigned["A"].tolist() == [10, 20, 31]
    assert grid.assigned["B"].tolist() == [11, 20, 30]


def test_align_days_moves_an_anchor_that_is_too_close_and_drops_it_if_it_cannot():
    yearsite_days = {"A": np.array([10, 14]), "B": np.array([10, 15])}
    assert align_days(yearsite_days, tolerance=1, min_gap=3).anchors.tolist() == [
        10,
        14,
    ]
    # Day 14 is 4 days from the accepted 10, so the anchor goes to the first date
    # that keeps the spacing and is within tolerance of both measurement dates.
    assert align_days(yearsite_days, tolerance=1, min_gap=5).anchors.tolist() == [
        10,
        15,
    ]
    # With a wider spacing no date is left that both site-years can serve.
    assert align_days(yearsite_days, tolerance=1, min_gap=7).anchors.tolist() == [10]


def test_align_days_drops_an_anchor_that_one_site_year_cannot_serve():
    grid = align_days(
        {"A": np.array([10, 20]), "B": np.array([10])}, tolerance=1, min_gap=3
    )
    assert grid.anchors.tolist() == [10]


def test_comparison_keeps_first_measurement_on_repeated_dates() -> None:
    rows = datasets.Dataset.from_dict(
        {
            "yearsite_uid": ["A"],
            "height_days": [[200, 210, 210, 220]],
            "height_values": [[0.1, 0.2, 0.9, 0.3]],
        }
    ).with_format("torch")
    days = np.array([200, 210, 220])
    grid = SharedDayGrid(days, {"A": days})
    np.testing.assert_allclose(observed_at_grid(rows, grid), [[0.1, 0.2, 0.3]])


def test_site_assignments_preserve_all_plots_and_row_order() -> None:
    grid = align_days({"A": np.array([210, 220]), "B": np.array([213, 223])})
    rows = datasets.Dataset.from_dict(
        {
            "yearsite_uid": ["B", "A", "B"],
            "height_days": [[213, 223], [210, 220], [213, 223]],
            "height_values": [[0.3, 0.8], [0.2, 0.7], [0.4, 0.9]],
        }
    ).with_format("torch")
    np.testing.assert_allclose(
        observed_at_grid(rows, grid), [[0.3, 0.8], [0.2, 0.7], [0.4, 0.9]]
    )
    np.testing.assert_array_equal(grid.anchors, [211, 221])
    np.testing.assert_array_equal(grid.assigned["B"], [213, 223])


def test_comparison_rejects_missing_selected_measurement_instead_of_next_day() -> None:
    rows = datasets.Dataset.from_dict(
        {
            "yearsite_uid": ["A"],
            "height_days": [[210, 221]],
            "height_values": [[0.2, 0.9]],
        }
    ).with_format("torch")
    days = np.array([210, 220])
    with pytest.raises(ValueError, match="exact subset"):
        observed_at_grid(rows, SharedDayGrid(days, {"A": days}))


def test_year_selection_keeps_complete_measurement_dates_in_original_order() -> None:
    grid = SharedDayGrid(
        np.array([211, 221]), {"A": np.array([210, 220]), "B": np.array([213, 223])}
    )
    rows = datasets.Dataset.from_dict(
        {
            "harvest_year": [2016, 2016, 2017, 2016, 2016],
            "plot_uid": ["first", "missing", "other-year", "late", "last"],
            "yearsite_uid": ["B", "A", "A", "B", "A"],
            "height_days": [[213, 223], [210, 221], [210, 220], [213], [210, 220]],
            "height_values": [[0.3, 0.8], [0.2, 0.9], [0.2, 0.7], [0.4], [0.1, 0.6]],
        }
    ).with_format("torch")
    selected = select_year(rows, 2016, grid)
    assert list(selected["plot_uid"]) == ["first", "last"]
    np.testing.assert_allclose(
        observed_at_grid(selected, grid), [[0.3, 0.8], [0.1, 0.6]]
    )


def test_dense_alignment_keeps_lodged_plots_and_original_common_dates() -> None:
    rows = datasets.Dataset.from_dict(
        {
            "harvest_year": [2019, 2019, 2019],
            "plot_uid": ["first", "lodged", "last"],
            "height_days": [
                [200, 210, 220, 230],
                [200, 210, 230, 330],
                [200, 220, 230],
            ],
            "height_values": [
                [0.1, 0.2, 0.3, 0.4],
                [1.1, 1.1, 1.1, 0.43],
                [0.2, 0.4, 0.6],
            ],
        }
    ).with_format("torch")
    days, real = observed_on_common_days(rows)
    np.testing.assert_array_equal(days, [200, 230])
    np.testing.assert_allclose(real, [[0.1, 0.4], [1.1, 1.1], [0.2, 0.6]])


def test_common_observations_keep_first_values_on_unsorted_dates_in_custom_window():
    rows = [
        {
            "height_days": torch.tensor([210, 190, 190, 200]),
            "height_values": torch.tensor([0.4, 0.1, 0.9, 0.2]),
        },
        {
            "height_days": torch.tensor([180, 190, 200, 210]),
            "height_values": torch.tensor([0.3, 0.5, 0.6, 0.7]),
        },
    ]
    days, heights = observed_on_common_days(rows, window=(190, 210))
    np.testing.assert_array_equal(days, [190, 200])
    np.testing.assert_allclose(heights, [[0.1, 0.2], [0.5, 0.6]])


def test_common_observations_do_not_drop_a_plot_without_common_dates():
    rows = [
        {"height_days": torch.tensor([200]), "height_values": torch.tensor([0.1])},
        {"height_days": torch.tensor([330]), "height_values": torch.tensor([0.8])},
    ]
    days, heights = observed_on_common_days(rows)
    assert days.numel() == 0
    assert heights.shape == (2, 0)
