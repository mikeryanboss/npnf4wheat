from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npnf.scripts.paper.style import log_ticks

SPLIT_DIRS = {
    "plot": ("synth_test_plot_dataloaders", "test_plot"),
    "genotype": ("synth_test_genotype_dataloaders", "test_genotype"),
    "environment": ("synth_test_environment_dataloaders", "test_environment"),
    "unseen": ("synth_test_unseen_dataloaders", "test_unseen"),
}


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "mmd_plot", Path("src/npnf/scripts/paper/mmd_plot.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_summary(
    root: Path,
    *,
    model: str = "CNP-512k",
    split: str = "plot",
    opaque: str = "6",
    checkpoint: str = "checkpoint-100",
    conditioning: str = "noenv_nogeno",
    prediction_method: str = "no_context",
    metric_type: str = "sig_mmd",
    view: str = "condition",
    mean: float = 0.001,
    include_metric: bool = True,
) -> Path:
    dataloader_dir, split_dir = SPLIT_DIRS[split]
    if metric_type == "csig_mmd":
        dir_name, summary_file = "csig_mmd", "csig_mmd_summary.csv"
    else:
        dir_name, summary_file = "sig_mmd", "sig_mmd_summary.csv"
    if view == "context_matched_blocked":
        dir_name = f"{dir_name}_context_matched_blocked"
    path = (
        root
        / dataloader_dir
        / model
        / opaque
        / checkpoint
        / split_dir
        / conditioning
        / prediction_method
        / dir_name
        / summary_file
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if include_metric:
        path.write_text(
            "view,n_units,n_conditions,mean,median,std,p95\n"
            f"{view},2,2,{mean},0.1,0.2,0.3\n"
        )
    else:
        path.write_text(f"view,n_units,n_conditions,median\n{view},2,2,0.1\n")
    return path


def _write_sig_plot_grid(root: Path, *, point_style: str) -> list[str]:
    means = {
        ("CNP-512k", "plot"): 0.001,
        ("CNP-512k", "environment"): 0.002,
        ("LNP-512k", "plot"): 0.003,
        ("LNP-512k", "environment"): 0.004,
    }
    for (model, split), mean in means.items():
        _write_summary(root, model=model, split=split, mean=mean)
    return [
        "--results-base",
        str(root),
        "--models",
        "CNP-512k",
        "LNP-512k",
        "--model-names",
        "CNP",
        "LNP",
        "--splits",
        "plot",
        "environment",
        "--split-labels",
        "seen",
        "env",
        "--conditionings",
        "noenv_nogeno",
        "--conditioning-labels",
        "P",
        "--prediction-methods",
        "no_context",
        "--checkpoint",
        "100",
        "--point-style",
        point_style,
    ]


def _write_averaging_grid(root: Path) -> list[str]:
    means = {
        ("plot", "no_context", "noenv_nogeno"): 0.001,
        ("environment", "no_context", "noenv_nogeno"): 0.003,
        ("plot", "random_context", "noenv_nogeno"): 0.011,
        ("environment", "random_context", "noenv_nogeno"): 0.013,
        ("plot", "no_context", "env_nogeno"): 0.005,
        ("environment", "no_context", "env_nogeno"): 0.007,
        ("plot", "random_context", "env_nogeno"): 0.017,
        ("environment", "random_context", "env_nogeno"): 0.019,
    }
    for (split, prediction_method, conditioning), mean in means.items():
        _write_summary(
            root,
            split=split,
            prediction_method=prediction_method,
            conditioning=conditioning,
            mean=mean,
        )
    return [
        "--results-base",
        str(root),
        "--models",
        "CNP-512k",
        "--model-names",
        "CNP",
        "--splits",
        "plot",
        "environment",
        "--split-labels",
        "seen",
        "env",
        "--conditionings",
        "noenv_nogeno",
        "env_nogeno",
        "--conditioning-labels",
        "P",
        "E",
        "--prediction-methods",
        "no_context",
        "random_context",
        "--checkpoint",
        "100",
        "--point-style",
        "swapped-fill",
    ]


def test_default_test_set_is_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv("NPNF_RESULTS_DIR", str(tmp_path))
    module = _load_module()

    cfg = module.parse_args([])
    shifted = module.parse_args(["--test-set", "shifted"])

    assert {spec.test_set for spec in cfg.specs} == {"legacy"}
    assert list(dict.fromkeys(spec.split for spec in cfg.specs)) == [
        "plot",
        "genotype",
        "environment",
        "unseen",
    ]
    assert {spec.test_set for spec in shifted.specs} == {"shifted"}
    assert list(dict.fromkeys(spec.split for spec in shifted.specs)) == [
        "seen",
        "geno",
        "env",
        "unseen",
    ]


def test_read_raw_values_discovers_sig_mmd_summaries(tmp_path):
    module = _load_module()
    means = {
        ("CNP-512k", "no_context", "plot"): 0.001,
        ("CNP-512k", "no_context", "environment"): 0.002,
        ("LNP-512k", "no_context", "plot"): 0.003,
        ("LNP-512k", "no_context", "environment"): 0.004,
        ("CNP-512k", "random_context", "plot"): 0.005,
        ("CNP-512k", "random_context", "environment"): 0.006,
        ("LNP-512k", "random_context", "plot"): 0.007,
        ("LNP-512k", "random_context", "environment"): 0.008,
    }
    for (model, prediction_method, split), mean in means.items():
        _write_summary(
            tmp_path,
            model=model,
            prediction_method=prediction_method,
            split=split,
            mean=mean,
        )
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "LNP-512k",
            "--model-names",
            "CNP",
            "LNP",
            "--splits",
            "plot",
            "environment",
            "--split-labels",
            "seen",
            "env",
            "--conditionings",
            "noenv_nogeno",
            "--conditioning-labels",
            "P",
            "--prediction-methods",
            "no_context",
            "random_context",
            "--checkpoint",
            "100",
        ]
    )

    values = module.read_raw_values(cfg)

    assert len(values) == 8
    assert "sig_mmd_x1000" in values.columns
    assert "summary_paths" not in values.columns
    actual = values.set_index(["model", "prediction_method", "split"])[
        "sig_mmd_x1000"
    ].to_dict()
    for key, value in means.items():
        assert actual[key] == pytest.approx(value * 1000.0)
    assert values["view"].drop_duplicates().tolist() == ["condition"]
    assert all("sig_mmd/sig_mmd_summary.csv" in path for path in values["summary_path"])
    assert values["prediction_method_label"].drop_duplicates().tolist() == [
        "no context",
        "random context",
    ]
    assert values["split_label"].drop_duplicates().tolist() == ["seen", "env"]


def test_read_raw_values_reports_all_missing_summaries(tmp_path):
    module = _load_module()
    _write_summary(tmp_path, model="CNP-512k", split="plot", mean=0.001)
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "LNP-512k",
            "--splits",
            "plot",
            "environment",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "--checkpoint",
            "100",
        ]
    )

    with pytest.raises(FileNotFoundError) as error_info:
        module.read_raw_values(cfg)

    message = str(error_info.value)
    assert "model=CNP-512k, split=environment" in message
    assert "model=LNP-512k, split=plot" in message
    assert "model=LNP-512k, split=environment" in message
    assert "prediction_method=no_context" in message
    assert "sig_mmd_summary.csv" in message
    assert "expected_glob=" in message


def test_read_raw_values_reports_ambiguous_and_missing_summaries(tmp_path):
    module = _load_module()
    first = _write_summary(tmp_path, model="CNP-512k", opaque="6")
    second = _write_summary(tmp_path, model="CNP-512k", opaque="7")
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "LNP-512k",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "--checkpoint",
            "100",
        ]
    )

    with pytest.raises(
        ValueError, match=r"ambiguous sig_mmd_summary\.csv"
    ) as error_info:
        module.read_raw_values(cfg)

    message = str(error_info.value)
    assert str(first) in message
    assert str(second) in message
    assert "model=LNP-512k, split=plot" in message
    assert "expected_glob=" in message


def test_read_raw_values_discovers_csig_mmd_summaries(tmp_path):
    module = _load_module()
    summary_path = _write_summary(tmp_path, metric_type="csig_mmd", mean=0.005)
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "--metric-type",
            "csig_mmd",
            "--checkpoint",
            "100",
        ]
    )

    values = module.read_raw_values(cfg)

    assert len(values) == 1
    assert values["csig_mmd_x1000"].tolist() == [5.0]
    assert values["summary_path"].tolist() == [str(summary_path)]


@pytest.mark.parametrize("metric_type", ["sig_mmd", "csig_mmd"])
def test_context_matched_blocked_view_reads_blocked_summaries_and_writes_outputs(
    tmp_path, metric_type
):
    module = _load_module()
    _write_summary(tmp_path, metric_type=metric_type, view="condition", mean=0.999)
    blocked_summary = _write_summary(
        tmp_path, metric_type=metric_type, view="context_matched_blocked", mean=0.005
    )
    output_dir = tmp_path / "out"
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "--metric-type",
            metric_type,
            "--view",
            "context_matched_blocked",
            "--checkpoint",
            "100",
            "--output-dir",
            str(output_dir),
            "--point-style",
            "gray",
        ]
    )

    values = module.read_raw_values(cfg)
    assert values["view"].tolist() == ["context_matched_blocked"]
    assert values["summary_path"].tolist() == [str(blocked_summary)]
    assert f"{metric_type}_context_matched_blocked" in values["summary_path"][0]
    assert values[f"{metric_type}_x1000"].tolist() == [5.0]

    module.write_outputs(cfg)

    stem = f"{metric_type}_context_matched_blocked_boxplot"
    expected_files = [
        f"{stem}.png",
        f"{stem}.pdf",
        f"{stem}_raw_values.csv",
        f"{stem}_model_summary.csv",
    ]
    for filename in expected_files:
        assert (output_dir / filename).is_file()
    assert not (output_dir / f"{metric_type}_boxplot_raw_values.csv").exists()

    written_values = pd.read_csv(output_dir / f"{stem}_raw_values.csv")
    assert written_values["view"].tolist() == ["context_matched_blocked"]


def test_prediction_method_view_overrides_write_mixed_outputs(tmp_path):
    module = _load_module()
    blocked_summary = _write_summary(
        tmp_path,
        prediction_method="no_context",
        view="context_matched_blocked",
        mean=0.001,
    )
    random_context_summary = _write_summary(
        tmp_path, prediction_method="random_context", view="condition", mean=0.002
    )
    max_height_summary = _write_summary(
        tmp_path, prediction_method="max_height", view="condition", mean=0.003
    )
    _write_summary(
        tmp_path, prediction_method="no_context", view="condition", mean=0.900
    )
    output_dir = tmp_path / "out"
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "random_context",
            "max_height",
            "--view",
            "condition",
            "--prediction-method-views",
            "no_context=context_matched_blocked",
            "--checkpoint",
            "100",
            "--output-dir",
            str(output_dir),
            "--point-style",
            "gray",
        ]
    )

    values = module.read_raw_values(cfg)
    view_by_method = dict(zip(values["prediction_method"], values["view"], strict=True))
    path_by_method = dict(
        zip(values["prediction_method"], values["summary_path"], strict=True)
    )
    assert view_by_method == {
        "no_context": "context_matched_blocked",
        "random_context": "condition",
        "max_height": "condition",
    }
    assert path_by_method["no_context"] == str(blocked_summary)
    assert path_by_method["random_context"] == str(random_context_summary)
    assert path_by_method["max_height"] == str(max_height_summary)

    module.write_outputs(cfg)

    stem = "sig_mmd_mixed_views_boxplot"
    assert (output_dir / f"{stem}_raw_values.csv").is_file()
    assert (output_dir / f"{stem}_model_summary.csv").is_file()
    assert (output_dir / f"{stem}.png").is_file()
    assert (output_dir / f"{stem}.pdf").is_file()
    assert not (output_dir / "sig_mmd_boxplot_raw_values.csv").exists()
    summary = pd.read_csv(output_dir / f"{stem}_model_summary.csv")
    assert summary["prediction_method_views"].tolist() == [
        (
            "max_height=condition;no_context=context_matched_blocked;"
            "random_context=condition"
        )
    ]


def test_summarize_by_model_uses_raw_values(tmp_path):
    module = _load_module()
    cfg = module.parse_args(_write_sig_plot_grid(tmp_path, point_style="gray"))
    values = module.read_raw_values(cfg)

    summary = module.summarize_by_model(values, "sig_mmd")

    assert summary["model_name"].tolist() == ["CNP", "LNP"]
    assert summary["n"].tolist() == [2, 2]
    assert summary["mean"].tolist() == [1.5, 3.5]
    assert summary["min"].tolist() == [1.0, 3.0]
    assert summary["max"].tolist() == [2.0, 4.0]


@pytest.mark.parametrize(
    (
        "extra_args",
        "expected_stem",
        "expected_index",
        "expected_values",
        "expected_n_rows",
        "expected_averaged_over",
    ),
    [
        (
            ["--average-test-sets"],
            "sig_mmd_boxplot_testset_averaged",
            ["prediction_method", "conditioning"],
            {
                ("no_context", "noenv_nogeno"): 2.0,
                ("random_context", "noenv_nogeno"): 12.0,
                ("no_context", "env_nogeno"): 6.0,
                ("random_context", "env_nogeno"): 18.0,
            },
            [2, 2, 2, 2],
            "split",
        ),
        (
            ["--average-prediction-methods"],
            "sig_mmd_boxplot_prediction_method_averaged",
            ["split", "conditioning"],
            {
                ("plot", "noenv_nogeno"): 6.0,
                ("environment", "noenv_nogeno"): 8.0,
                ("plot", "env_nogeno"): 11.0,
                ("environment", "env_nogeno"): 13.0,
            },
            [2, 2, 2, 2],
            "prediction_method",
        ),
        (
            ["--average-test-sets", "--average-prediction-methods"],
            "sig_mmd_boxplot_testset_prediction_method_averaged",
            ["conditioning"],
            {"noenv_nogeno": 7.0, "env_nogeno": 12.0},
            [4, 4],
            "split,prediction_method",
        ),
    ],
)
def test_average_plot_values_over_requested_dimensions(
    tmp_path,
    extra_args,
    expected_stem,
    expected_index,
    expected_values,
    expected_n_rows,
    expected_averaged_over,
):
    module = _load_module()
    cfg = module.parse_args([*_write_averaging_grid(tmp_path), *extra_args])
    values = module.read_raw_values(cfg)

    averaged = module.average_plot_values(
        values,
        metric=cfg.metric,
        metric_type=cfg.metric_type,
        average_test_sets=cfg.average_test_sets,
        average_prediction_methods=cfg.average_prediction_methods,
    )

    assert (
        module.output_stem(
            cfg.metric_type,
            cfg.view,
            cfg.prediction_method_views,
            average_test_sets=cfg.average_test_sets,
            average_prediction_methods=cfg.average_prediction_methods,
        )
        == expected_stem
    )
    assert len(averaged) == len(expected_values)
    assert averaged["n_rows"].tolist() == expected_n_rows
    assert averaged["averaged_over"].drop_duplicates().tolist() == [
        expected_averaged_over
    ]
    actual = averaged.set_index(expected_index)["sig_mmd_x1000"].to_dict()
    for key, expected in expected_values.items():
        assert actual[key] == pytest.approx(expected)


@pytest.mark.parametrize("point_style", ["testset-context", "gray"])
def test_average_options_require_swapped_fill(tmp_path, point_style):
    module = _load_module()

    with pytest.raises(ValueError, match="point-style swapped-fill"):
        module.parse_args(
            [
                *_write_averaging_grid(tmp_path),
                "--point-style",
                point_style,
                "--average-test-sets",
            ]
        )


@pytest.mark.parametrize("point_style", ["testset-context", "gray", "swapped-fill"])
def test_write_outputs_creates_sig_mmd_figures_and_csvs(tmp_path, point_style):
    module = _load_module()
    output_dir = tmp_path / "out"
    cfg = module.parse_args(
        [
            *_write_sig_plot_grid(tmp_path, point_style=point_style),
            "--output-dir",
            str(output_dir),
        ]
    )

    module.write_outputs(cfg)

    expected_files = [
        "sig_mmd_boxplot.png",
        "sig_mmd_boxplot.pdf",
        "sig_mmd_boxplot_raw_values.csv",
        "sig_mmd_boxplot_model_summary.csv",
    ]
    for filename in expected_files:
        assert (output_dir / filename).is_file()
    assert not (output_dir / "sig_mmd_boxplot_averaged_values.csv").exists()

    values = pd.read_csv(output_dir / "sig_mmd_boxplot_raw_values.csv")
    summary = pd.read_csv(output_dir / "sig_mmd_boxplot_model_summary.csv")
    assert len(values) == 4
    assert values["sig_mmd_x1000"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert summary["n"].tolist() == [2, 2]


def test_write_outputs_swapped_fill_custom_conditioning_labels(tmp_path):
    module = _load_module()
    conditionings = ["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"]
    labels = ["$\\emptyset$", "$e$", "$g$", "$g+e$"]
    for index, conditioning in enumerate(conditionings, start=1):
        _write_summary(tmp_path, conditioning=conditioning, mean=index / 1000.0)
    output_dir = tmp_path / "out"
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "--model-names",
            "CNP",
            "--splits",
            "plot",
            "--conditionings",
            *conditionings,
            "--conditioning-labels",
            *labels,
            "--prediction-methods",
            "no_context",
            "--checkpoint",
            "100",
            "--point-style",
            "swapped-fill",
            "--output-dir",
            str(output_dir),
        ]
    )

    module.write_outputs(cfg)

    assert (output_dir / "sig_mmd_boxplot.png").is_file()
    assert (output_dir / "sig_mmd_boxplot.pdf").is_file()
    values = pd.read_csv(output_dir / "sig_mmd_boxplot_raw_values.csv")
    assert values["conditioning_label"].tolist() == labels


def test_write_outputs_averaged_swapped_fill_custom_conditioning_labels(tmp_path):
    module = _load_module()
    conditionings = ["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"]
    labels = ["$\\emptyset$", "$e$", "$g$", "$g+e$"]
    value = 1
    for split in ("plot", "environment"):
        for method in ("no_context", "random_context"):
            for conditioning in conditionings:
                _write_summary(
                    tmp_path,
                    split=split,
                    prediction_method=method,
                    conditioning=conditioning,
                    mean=value / 1000.0,
                )
                value += 1
    output_dir = tmp_path / "out"
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "--model-names",
            "CNP",
            "--splits",
            "plot",
            "environment",
            "--conditionings",
            *conditionings,
            "--conditioning-labels",
            *labels,
            "--prediction-methods",
            "no_context",
            "random_context",
            "--checkpoint",
            "100",
            "--point-style",
            "swapped-fill",
            "--average-test-sets",
            "--average-prediction-methods",
            "--output-dir",
            str(output_dir),
        ]
    )

    module.write_outputs(cfg)

    stem = "sig_mmd_boxplot_testset_prediction_method_averaged"
    assert (output_dir / f"{stem}.png").is_file()
    assert (output_dir / f"{stem}.pdf").is_file()
    averaged = pd.read_csv(output_dir / f"{stem}_averaged_values.csv")
    assert averaged["conditioning_label"].tolist() == labels


@pytest.mark.parametrize(
    ("extra_args", "stem", "expected_len", "averaged_over"),
    [
        (["--average-test-sets"], "sig_mmd_boxplot_testset_averaged", 4, "split"),
        (
            ["--average-prediction-methods"],
            "sig_mmd_boxplot_prediction_method_averaged",
            4,
            "prediction_method",
        ),
        (
            ["--average-test-sets", "--average-prediction-methods"],
            "sig_mmd_boxplot_testset_prediction_method_averaged",
            2,
            "split,prediction_method",
        ),
    ],
)
def test_write_outputs_creates_averaged_values_and_figures(
    tmp_path, extra_args, stem, expected_len, averaged_over
):
    module = _load_module()
    output_dir = tmp_path / "out"
    cfg = module.parse_args(
        [*_write_averaging_grid(tmp_path), *extra_args, "--output-dir", str(output_dir)]
    )

    module.write_outputs(cfg)

    expected_files = [
        f"{stem}.png",
        f"{stem}.pdf",
        f"{stem}_raw_values.csv",
        f"{stem}_averaged_values.csv",
        f"{stem}_model_summary.csv",
    ]
    for filename in expected_files:
        assert (output_dir / filename).is_file()
    assert not (output_dir / "sig_mmd_boxplot_raw_values.csv").exists()

    values = pd.read_csv(output_dir / f"{stem}_raw_values.csv")
    averaged = pd.read_csv(output_dir / f"{stem}_averaged_values.csv")
    summary = pd.read_csv(output_dir / f"{stem}_model_summary.csv")
    assert len(values) == 8
    assert len(averaged) == expected_len
    assert summary["n"].tolist() == [expected_len]
    assert summary["source_n"].tolist() == [8]
    assert summary["averaged_over"].tolist() == [averaged_over]


def test_missing_metric_column_fails(tmp_path):
    module = _load_module()
    summary_path = _write_summary(tmp_path, include_metric=False)
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "--checkpoint",
            "100",
        ]
    )

    with pytest.raises(ValueError, match="Metric column 'mean' missing") as error_info:
        module.read_raw_values(cfg)

    assert str(summary_path) in str(error_info.value)


def test_log_ticks_follow_one_two_five_sequence():
    assert log_ticks(0.03, 84.0) == [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    assert log_ticks(0.13, 8.1) == [0.2, 0.5, 1, 2, 5]


@pytest.mark.parametrize(
    ("point_style", "extra_args", "tick_labels"),
    [
        ("testset-context", [], ["1", "2", "5"]),
        ("gray", [], ["1", "2", "5"]),
        ("swapped-fill", [], ["1", "2", "5"]),
        ("swapped-fill", ["--average-test-sets"], ["2"]),
    ],
)
def test_write_outputs_log_y_uses_a_log_axis(
    monkeypatch, tmp_path, point_style, extra_args, tick_labels
):
    module = _load_module()
    closed_figures = []
    monkeypatch.setattr(module.plt, "close", closed_figures.append)
    output_dir = tmp_path / "out"
    cfg = module.parse_args(
        [
            *_write_sig_plot_grid(tmp_path, point_style=point_style),
            *extra_args,
            "--log-y",
            "--output-dir",
            str(output_dir),
        ]
    )

    module.write_outputs(cfg)

    axis = closed_figures[0].axes[0]
    assert axis.get_yscale() == "log"
    assert [label.get_text() for label in axis.get_yticklabels()] == tick_labels
    assert list(output_dir.glob("*.png"))


def test_write_outputs_log_y_rejects_non_positive_values(tmp_path):
    module = _load_module()
    arguments = _write_sig_plot_grid(tmp_path, point_style="swapped-fill")
    _write_summary(tmp_path, model="CNP-512k", split="plot", mean=-0.0005)
    cfg = module.parse_args(
        [*arguments, "--log-y", "--output-dir", str(tmp_path / "out")]
    )

    with pytest.raises(ValueError, match=r"smallest plotted value is -0\.5"):
        module.write_outputs(cfg)


def _write_replicate_grid(root: Path) -> list[str]:
    """Three runs sharing one display label, on a two-cell grid."""
    means = {
        ("LNP-512k", "plot"): 0.001,
        ("LNP-512k_seed2", "plot"): 0.002,
        ("LNP-512k_seed3", "plot"): 0.003,
        ("LNP-512k", "environment"): 0.004,
        ("LNP-512k_seed2", "environment"): 0.004,
        ("LNP-512k_seed3", "environment"): 0.004,
    }
    for (model, split), mean in means.items():
        _write_summary(root, model=model, split=split, mean=mean)
    return [
        "--results-base",
        str(root),
        "--models",
        "LNP-512k",
        "LNP-512k_seed2",
        "LNP-512k_seed3",
        "--model-names",
        "LNP",
        "LNP",
        "LNP",
        "--splits",
        "plot",
        "environment",
        "--conditionings",
        "noenv_nogeno",
        "--prediction-methods",
        "no_context",
        "--point-style",
        "swapped-fill",
    ]


def test_aggregate_replicates_collapses_runs_to_one_point_per_cell(tmp_path):
    module = _load_module()
    args = _write_replicate_grid(tmp_path / "results")
    cfg = module.parse_args(
        [*args, "--aggregate-replicates", "--output-dir", str(tmp_path / "out")]
    )
    values = module.read_raw_values(cfg)
    assert len(values) == 6

    aggregated = module.aggregate_replicate_values(values, cfg.metric_type)
    assert len(aggregated) == 2
    assert set(aggregated["model_name"]) == {"LNP"}

    column = module.scaled_metric_column(cfg.metric_type)
    by_split = {row["split"]: row for row in aggregated.to_dict("records")}
    # means 1, 2, 3 (x1000) -> mean 2, sample sd 1
    assert by_split["plot"][column] == pytest.approx(2.0)
    assert by_split["plot"][module.REPLICATE_SD_COLUMN] == pytest.approx(1.0)
    assert by_split["plot"][module.REPLICATE_N_COLUMN] == 3
    # identical replicates -> zero spread, not a missing bar
    assert by_split["environment"][module.REPLICATE_SD_COLUMN] == pytest.approx(0.0)


def test_log_axis_replicates_use_geometric_mean_and_log_sd(tmp_path):
    """On a log axis a +/- sd bar is stretched below the mean, a factor is not."""
    module = _load_module()
    args = _write_replicate_grid(tmp_path / "results")
    cfg = module.parse_args(
        [
            *args,
            "--aggregate-replicates",
            "--log-y",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert module.geometric_replicates(cfg)
    aggregated = module.aggregate_replicate_values(
        module.read_raw_values(cfg), cfg.metric_type, geometric=True
    )
    column = module.scaled_metric_column(cfg.metric_type)
    by_split = {row["split"]: row for row in aggregated.to_dict("records")}
    logs = np.log([1.0, 2.0, 3.0])
    assert by_split["plot"][column] == pytest.approx(6.0 ** (1 / 3))
    assert by_split["plot"][module.REPLICATE_SD_COLUMN] == pytest.approx(
        logs.std(ddof=1)
    )
    module.write_outputs(cfg)
    assert (tmp_path / "out").is_dir()


def test_single_run_label_gets_zero_sd_rather_than_nan(tmp_path):
    module = _load_module()
    root = tmp_path / "results"
    _write_summary(root, model="LNP-512k", split="plot", mean=0.001)
    cfg = module.parse_args(
        [
            "--results-base",
            str(root),
            "--models",
            "LNP-512k",
            "--model-names",
            "LNP",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "--point-style",
            "swapped-fill",
            "--aggregate-replicates",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    aggregated = module.aggregate_replicate_values(
        module.read_raw_values(cfg), cfg.metric_type
    )
    assert len(aggregated) == 1
    assert aggregated[module.REPLICATE_SD_COLUMN].iloc[0] == pytest.approx(0.0)


def test_without_the_flag_replicates_stay_separate(tmp_path):
    """The flag must be opt-in: existing invocations keep one point per run."""
    module = _load_module()
    args = _write_replicate_grid(tmp_path / "results")
    cfg = module.parse_args([*args, "--output-dir", str(tmp_path / "out")])
    assert cfg.aggregate_replicates is False
    module.write_outputs(cfg)
    raw = pd.read_csv(tmp_path / "out" / "sig_mmd_boxplot_raw_values.csv")
    assert len(raw) == 6
    assert module.REPLICATE_SD_COLUMN not in raw.columns
    summary = pd.read_csv(tmp_path / "out" / "sig_mmd_boxplot_model_summary.csv")
    assert len(summary) == 3


def test_aggregate_replicates_writes_one_summary_row_per_label(tmp_path):
    module = _load_module()
    args = _write_replicate_grid(tmp_path / "results")
    cfg = module.parse_args(
        [*args, "--aggregate-replicates", "--output-dir", str(tmp_path / "out")]
    )
    module.write_outputs(cfg)
    summary = pd.read_csv(tmp_path / "out" / "sig_mmd_boxplot_model_summary.csv")
    assert len(summary) == 1
    assert summary["n"].iloc[0] == 2


def test_aggregate_replicates_rejects_unsupported_point_style(tmp_path):
    module = _load_module()
    args = _write_replicate_grid(tmp_path / "results")
    args[args.index("swapped-fill")] = "gray"
    with pytest.raises(ValueError, match="aggregate-replicates"):
        module.parse_args([*args, "--aggregate-replicates"])


def test_summary_by_split_rejects_unsupported_point_style(tmp_path):
    module = _load_module()
    args = _write_replicate_grid(tmp_path / "results")
    args[args.index("swapped-fill")] = "gray"
    with pytest.raises(ValueError, match="summary-by-split"):
        module.parse_args([*args, "--summary-by-split"])


def test_summary_by_split_defaults_off(tmp_path):
    module = _load_module()
    args = _write_replicate_grid(tmp_path / "results")
    cfg = module.parse_args([*args, "--output-dir", str(tmp_path / "out")])
    assert cfg.summary_by_split is False


def test_raw_ylim_covers_error_bars():
    """Limits are set explicitly, so a tall bar would be cut at the axis edge."""
    module = _load_module()
    values = pd.Series([1.0, 2.0])
    without = module.raw_ylim(values)
    with_errors = module.raw_ylim(values, pd.Series([0.0, 30.0]))
    assert without[1] < 5.0
    assert with_errors[1] >= 32.0


def test_raw_ylim_never_goes_below_zero():
    module = _load_module()
    low, _ = module.raw_ylim(pd.Series([1.0]), pd.Series([30.0]))
    assert low == 0.0


def test_raw_ylim_starts_at_zero():
    """Scores far from 0 keep 0 on the axis, so panels read on the same terms."""
    module = _load_module()
    low, _ = module.raw_ylim(pd.Series([2.5, 6.0]))
    assert low == 0.0


def test_raw_ylim_shows_negative_scores():
    """The unbiased estimator gives a near-perfect model scores below 0."""
    module = _load_module()
    low, _ = module.raw_ylim(pd.Series([-0.03, 1.0]))
    assert low < -0.03


def test_right_panel_models_get_their_own_axis(monkeypatch, tmp_path):
    module = _load_module()
    closed_figures = []
    monkeypatch.setattr(module.plt, "close", closed_figures.append)
    cfg = module.parse_args(
        [
            *_write_sig_plot_grid(tmp_path, point_style="swapped-fill"),
            "--right-panel-models",
            "CNP",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    module.write_outputs(cfg)

    left, right = closed_figures[0].axes[:2]
    assert [label.get_text() for label in left.get_xticklabels()] == ["LNP"]
    assert [label.get_text() for label in right.get_xticklabels()] == ["CNP"]
    assert right.get_ylim()[0] == 0.0


def test_right_panel_models_must_be_model_names(tmp_path):
    module = _load_module()
    with pytest.raises(ValueError, match="--right-panel-models"):
        module.parse_args(
            [
                *_write_sig_plot_grid(tmp_path, point_style="swapped-fill"),
                "--right-panel-models",
                "ACNP",
            ]
        )


def test_log_y_axis_covers_error_bars():
    import matplotlib.pyplot as plt

    module = _load_module()
    _, ax = plt.subplots()
    module._set_y_axis(  # noqa: SLF001
        ax, pd.Series([1.0, 2.0]), log_y=True, errors=pd.Series([0.0, 30.0])
    )
    assert ax.get_ylim()[1] >= 32.0
    plt.close("all")


def test_log_y_axis_covers_geometric_bars():
    import matplotlib.pyplot as plt

    module = _load_module()
    _, ax = plt.subplots()
    module._set_y_axis(  # noqa: SLF001
        ax,
        pd.Series([1.0, 2.0]),
        log_y=True,
        errors=pd.Series([np.log(10.0), 0.0]),
        geometric=True,
    )
    low, high = ax.get_ylim()
    assert low <= 0.1
    assert high >= 10.0
    plt.close("all")


def test_log_y_axis_keeps_positive_lower_bar_ends():
    import matplotlib.pyplot as plt

    module = _load_module()
    _, ax = plt.subplots()
    module._set_y_axis(  # noqa: SLF001
        ax, pd.Series([1.0, 2.0, 3.0]), log_y=True, errors=pd.Series([0.9, 0.0, 5.0])
    )
    # 1.0 - 0.9 is inside the axis, the bar through zero is not used
    assert ax.get_ylim()[0] <= 0.1
    assert ax.get_ylim()[0] > 0.05
    plt.close("all")
