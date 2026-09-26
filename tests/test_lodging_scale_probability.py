import numpy as np
import pytest

from npnf.scripts.paper import lodging_scale_probability as lsp


def test_analytic_lodging_probability_matches_shifted_weibull_cdf() -> None:
    heights = np.array([0.0, 0.5, 1.1, 3.0])

    probabilities = lsp.analytic_lodging_probability(
        heights,
        lodging_weibull_scale=1.1,
        lodging_weibull_shape=7.0,
        lodging_weibull_offset=0.0,
        lodging_height_clamp=2.0,
    )

    expected_heights = np.array([0.0, 0.5, 1.1, 2.0])
    expected = 1.0 - np.exp(-((expected_heights / 1.1) ** 7.0))
    np.testing.assert_allclose(probabilities, expected)


def test_validate_input_lengths_reports_all_three_lengths() -> None:
    with pytest.raises(
        ValueError, match=r"base_folders=2.*model_names=1.*lodging_weibull_scales=3"
    ):
        lsp.validate_input_lengths(
            base_folders=["a", "b"],
            model_names=["model"],
            lodging_weibull_scales=[1.1, 1.3, 1.5],
        )


def test_validate_two_row_input_lengths_reports_all_four_lengths() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"base_folders=2.*base_folders_row2=1.*model_names=2"
            r".*lodging_weibull_scales=3"
        ),
    ):
        lsp.validate_two_row_input_lengths(
            base_folders=["a", "b"],
            base_folders_row2=["c"],
            model_names=["1.1", "1.3"],
            lodging_weibull_scales=[1.1, 1.3, 1.5],
        )


def test_calculate_rejects_row_labels_without_row2() -> None:
    with pytest.raises(ValueError, match=r"--row-labels requires"):
        lsp.calculate_lodging_scale_probability(
            base_folders=["a"],
            model_names=["1.1"],
            lodging_weibull_scales=[1.1],
            row_labels=["LNP", "ANP"],
        )


def test_calculate_requires_two_row_labels_with_row2() -> None:
    with pytest.raises(ValueError, match=r"--row-labels must be provided"):
        lsp.calculate_lodging_scale_probability(
            base_folders=["a"],
            base_folders_row2=["b"],
            model_names=["1.1"],
            lodging_weibull_scales=[1.1],
        )


def test_ground_truth_x_limit_uses_last_supported_bin() -> None:
    ground_truth = lsp.GroundTruth(
        labels=np.array([False] * 21),
        max_heights=np.array([0.62] * 10 + [0.72] * 10 + [0.82]),
    )
    bins = np.array([0.6, 0.7, 0.8, 0.9])
    centers = np.array([0.65, 0.75, 0.85])

    limits = lsp.calculate_ground_truth_x_limit(
        ground_truth, bins, centers, min_sample_fraction=0.2
    )
    assert limits is not None
    x_min, x_max = limits

    assert x_min == 0.6
    assert x_max == 0.75


def _rates(name: str) -> lsp.BinnedRates:
    return lsp.BinnedRates(
        name=name,
        rates=np.array([0.1]),
        counts=np.array([10]),
        lodged_counts=np.array([1]),
    )


def _prediction_stats(length: int) -> lsp.PredictionStats:
    max_height = np.linspace(0.65, 0.95, length)
    drop_abs = np.linspace(0.0, 0.2, length)
    return lsp.PredictionStats(
        max_height=max_height,
        final_height=max_height - drop_abs,
        drop_abs=drop_abs,
        drop_rel=drop_abs / max_height,
        is_lodged=np.arange(length) % 2 == 0,
    )


def _ground_truth(length: int) -> lsp.GroundTruth:
    return lsp.GroundTruth(
        labels=np.arange(length) % 2 == 0, max_heights=np.linspace(0.65, 0.95, length)
    )


def _out_of_ground_truth_range_prediction_stats(length: int) -> lsp.PredictionStats:
    max_height = np.linspace(1.5, 1.8, length)
    drop_abs = np.zeros(length)
    return lsp.PredictionStats(
        max_height=max_height,
        final_height=max_height,
        drop_abs=drop_abs,
        drop_rel=drop_abs,
        is_lodged=np.zeros(length, dtype=bool),
    )


def _make_base_folder(
    tmp_path, name: str, methods: tuple[str, ...] = ("no_context", "max_height")
):
    base_folder = tmp_path / name
    for method in methods:
        (base_folder / method).mkdir(parents=True)
    return base_folder


def test_single_row_calculation_uses_ground_truth_bins(tmp_path, monkeypatch) -> None:
    base_folder = _make_base_folder(tmp_path, "scale_1_1", methods=("no_context",))
    ground_truth = lsp.GroundTruth(
        labels=np.array([False, True, False, True]),
        max_heights=np.array([0.53, 0.63, 0.73, 0.83]),
    )

    monkeypatch.setattr(lsp, "_load_ground_truth", lambda _path: ground_truth)
    monkeypatch.setattr(
        lsp,
        "_load_predictions",
        lambda _path, _relative, _absolute: _out_of_ground_truth_range_prediction_stats(
            4
        ),
    )
    monkeypatch.setattr(lsp, "_plot_scale_lodging_rates", lambda **_kwargs: None)

    output_path = tmp_path / "out"
    lsp.calculate_lodging_scale_probability(
        base_folders=[str(base_folder)],
        model_names=["1.1"],
        lodging_weibull_scales=[1.1],
        output_folder=str(output_path),
        bin_width=0.1,
    )

    histogram = np.genfromtxt(
        output_path / "lodging_scale_probability.csv", delimiter=",", names=True
    )
    expected_bins, expected_centers = lsp.calculate_height_bins(
        ground_truth.max_heights, 0.1
    )
    np.testing.assert_allclose(histogram["bin_left"], expected_bins[:-1])
    np.testing.assert_allclose(histogram["bin_right"], expected_bins[1:])
    np.testing.assert_allclose(histogram["bin_center"], expected_centers)


def _two_row_panels() -> list[lsp.ScaleRowPanels]:
    return [
        lsp.ScaleRowPanels(
            label="LNP",
            prior_panel=lsp.ScalePanelData(
                title="no context",
                model_rates=[_rates("1.1")],
                gt_rates=[_rates("1.1")],
            ),
            context_panel=lsp.ScalePanelData(
                title="max-height context",
                model_rates=[_rates("1.1")],
                gt_rates=[_rates("1.1")],
            ),
            empirical_overlays={
                "1.1": lsp.EmpiricalOverlay(
                    name="1.1", rates=_rates("1.1"), color=(0.0, 0.0, 0.0)
                )
            },
        ),
        lsp.ScaleRowPanels(
            label="ANP",
            prior_panel=lsp.ScalePanelData(
                title="no context",
                model_rates=[_rates("1.1")],
                gt_rates=[_rates("1.1")],
            ),
            context_panel=lsp.ScalePanelData(
                title="max-height context",
                model_rates=[_rates("1.1")],
                gt_rates=[_rates("1.1")],
            ),
            empirical_overlays={
                "1.1": lsp.EmpiricalOverlay(
                    name="1.1", rates=_rates("1.1"), color=(0.0, 0.0, 0.0)
                )
            },
        ),
    ]


def test_two_row_calculation_reuses_shared_ground_truth(tmp_path, monkeypatch) -> None:
    base_folders = [
        _make_base_folder(tmp_path, "lnp_1_1"),
        _make_base_folder(tmp_path, "lnp_1_2"),
    ]
    base_folders_row2 = [
        _make_base_folder(tmp_path, "anp_1_1"),
        _make_base_folder(tmp_path, "anp_1_2"),
    ]
    ground_truth_calls = []
    prediction_calls = []

    def fake_load_ground_truth(path):
        ground_truth_calls.append(path)
        return _ground_truth(4)

    def fake_load_predictions(path, _relative_threshold, _absolute_threshold):
        prediction_calls.append(path)
        return _out_of_ground_truth_range_prediction_stats(4)

    monkeypatch.setattr(lsp, "_load_ground_truth", fake_load_ground_truth)
    monkeypatch.setattr(lsp, "_load_predictions", fake_load_predictions)

    output_path = tmp_path / "out"
    lsp.calculate_lodging_scale_probability(
        base_folders=[str(path) for path in base_folders],
        base_folders_row2=[str(path) for path in base_folders_row2],
        model_names=["1.1", "1.2"],
        lodging_weibull_scales=[1.1, 1.2],
        row_labels=["LNP", "ANP"],
        empirical_gt_scales=["1.1"],
        output_folder=str(output_path),
    )

    assert len(ground_truth_calls) == 1
    assert len(prediction_calls) == 8
    histogram = np.genfromtxt(
        output_path / "lodging_scale_probability.csv", delimiter=",", names=True
    )
    _, expected_centers = lsp.calculate_height_bins(_ground_truth(4).max_heights, 0.05)
    np.testing.assert_allclose(histogram["bin_center"], expected_centers)

    header = (output_path / "lodging_scale_probability.csv").read_text().splitlines()[0]
    assert "context_model_ANP_1.1_lodging_rate" in header
    assert "empirical_gt_LNP_1.1_lodging_rate" in header
    assert "empirical_gt_ANP_1.1_lodging_rate" in header
    assert "prior_model_LNP_1.1_lodging_rate" in header
    assert "context_gt_ANP_1.1_lodging_rate" in header
    assert "prior_model_1.1_lodging_rate" not in header


def test_context_length_mismatch_falls_back_to_method_ground_truth(
    tmp_path, monkeypatch
) -> None:
    base_folder = _make_base_folder(tmp_path, "scale_1_1", methods=("max_height",))
    ground_truth_lengths = [2, 3]

    def fake_load_ground_truth(_path):
        return _ground_truth(ground_truth_lengths.pop(0))

    monkeypatch.setattr(lsp, "_load_ground_truth", fake_load_ground_truth)
    monkeypatch.setattr(
        lsp,
        "_load_predictions",
        lambda _path, _relative, _absolute: _prediction_stats(3),
    )

    output_path = tmp_path / "out"
    lsp.calculate_lodging_scale_probability(
        base_folders=[str(base_folder)],
        model_names=["1.1"],
        lodging_weibull_scales=[1.1],
        output_folder=str(output_path),
    )

    assert ground_truth_lengths == []
    header = (output_path / "lodging_scale_probability.csv").read_text().splitlines()[0]
    assert "context_model_1.1_lodging_rate" in header


def test_calculate_rejects_prediction_length_mismatch(tmp_path, monkeypatch) -> None:
    base_folder = _make_base_folder(tmp_path, "scale_1_1", methods=("max_height",))

    monkeypatch.setattr(lsp, "_load_ground_truth", lambda _path: _ground_truth(2))
    monkeypatch.setattr(
        lsp,
        "_load_predictions",
        lambda _path, _relative, _absolute: _prediction_stats(3),
    )

    with pytest.raises(ValueError, match=r"1\.1.*max_height.*3 samples.*2"):
        lsp.calculate_lodging_scale_probability(
            base_folders=[str(base_folder)],
            model_names=["1.1"],
            lodging_weibull_scales=[1.1],
            output_folder=str(tmp_path / "out"),
        )


def test_two_row_plot_writes_png_and_pdf(tmp_path) -> None:
    lsp._plot_two_row_scale_lodging_rates(  # noqa: SLF001
        bin_centers=np.array([0.65]),
        row_panels=_two_row_panels(),
        model_colors={"1.1": (0.0, 0.0, 1.0)},
        output_path=tmp_path / "lodging_scale_probability",
        model_order=["1.1"],
        min_sample_fraction=0.01,
        x_limit=None,
    )

    assert (tmp_path / "lodging_scale_probability.png").stat().st_size > 0
    assert (tmp_path / "lodging_scale_probability.pdf").stat().st_size > 0
