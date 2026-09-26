# ruff: noqa: SLF001
from __future__ import annotations

import importlib.util
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "dataset_scaling", Path("src/npnf/scripts/paper/dataset_scaling.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATALOADERS = {
    "plot": "synth_test_plot_dataloaders",
    "environment": "synth_test_environment_dataloaders",
    "genotype": "synth_test_genotype_dataloaders",
    "unseen": "synth_test_unseen_dataloaders",
}


def _write_summary(
    results_base: Path,
    *,
    split: str,
    model: str,
    dataset_size_token: str,
    training_length_token: str,
    mean: float,
    set_mode_token: str = "nested_noprior",
    conditioning: str = "noenv_nogeno",
    prediction_method: str = "no_context",
    seed: str = "6",
    view: str = "condition",
) -> Path:
    metric_dir = (
        "sig_mmd_context_matched_blocked"
        if view == "context_matched_blocked"
        else "sig_mmd"
    )
    run_name = (
        f"{model}-{dataset_size_token}-training{training_length_token}"
        f"_set_mode_{set_mode_token}"
    )
    path = (
        results_base
        / DATALOADERS[split]
        / run_name
        / seed
        / "checkpoint-3000000"
        / f"test_{split}"
        / conditioning
        / prediction_method
        / metric_dir
        / "sig_mmd_summary.csv"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        "view,n_units,n_conditions,mean,median,std,p95\n"
        f"{view},2,2,{mean},0.1,0.2,0.3\n"
    )
    return path


def _small_config(module, tmp_path: Path):
    return module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--models",
            "LNP",
            "ANP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "8k",
            "--training-lengths",
            "100k",
            "1m",
            "--set-mode",
            "nested_noprior",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
        ]
    )


def _write_small_grid(results_base: Path, *, view: str = "condition") -> None:
    values = {
        ("LNP", "1k", "100k"): 10.0,
        ("ANP", "1k", "100k"): 8.0,
        ("LNP", "8k", "100k"): 9.0,
        ("ANP", "8k", "100k"): 7.0,
        ("LNP", "1k", "1m"): 5.0,
        ("ANP", "1k", "1m"): 4.0,
        ("LNP", "8k", "1m"): 3.0,
        ("ANP", "8k", "1m"): 2.0,
    }
    for (model, size_token, training_token), mean in values.items():
        _write_summary(
            results_base,
            split="plot",
            model=model,
            dataset_size_token=size_token,
            training_length_token=training_token,
            mean=mean,
            view=view,
        )


def test_parse_bash_args_and_size_aliases(tmp_path):
    module = _load_module()

    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "ANP",
            "LNP",
            "--splits",
            "plot",
            "unseen",
            "--dataset-sizes",
            "1k",
            "64k",
            "--training-lengths",
            "100k",
            "3m",
            "--set-mode",
            "nested_noprior",
            "--title",
            "Custom title",
        ]
    )

    assert cfg.results_base == tmp_path
    assert cfg.models == ("ANP", "LNP")
    assert cfg.splits == ("plot", "unseen")
    assert cfg.dataset_sizes == ("1k", "64k")
    assert cfg.dataset_size_labels == ("1k", "64k")
    assert [module._dataset_size_value(size) for size in cfg.dataset_sizes] == [
        1024,
        64 * 1024,
    ]
    assert [module._dataset_size_label(size) for size in cfg.dataset_sizes] == [
        "1k",
        "64k",
    ]
    assert cfg.training_lengths == ("100k", "3m")
    assert [module._training_steps(length) for length in cfg.training_lengths] == [
        100_000,
        3_000_000,
    ]
    assert [module._training_label(length) for length in cfg.training_lengths] == [
        "100k",
        "3M",
    ]
    assert cfg.set_mode == "nested-noprior"
    assert cfg.title == "Custom title"


def test_parse_accepts_custom_dataset_size_labels(tmp_path):
    module = _load_module()

    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--dataset-sizes",
            "1k",
            "8k",
            "--dataset-size-labels",
            "1K examples",
            "8K examples",
        ]
    )

    assert cfg.dataset_sizes == ("1k", "8k")
    assert cfg.dataset_size_labels == ("1K examples", "8K examples")


def test_parse_rejects_mismatched_dataset_size_labels(tmp_path):
    module = _load_module()

    with pytest.raises(ValueError, match="--dataset-sizes and --dataset-size-labels"):
        module.parse_args(
            [
                "--results-base",
                str(tmp_path),
                "--dataset-sizes",
                "1k",
                "8k",
                "--dataset-size-labels",
                "1K",
            ]
        )


def test_parse_accepts_custom_prediction_method_labels(tmp_path):
    module = _load_module()

    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--prediction-methods",
            "no_context",
            "custom_method",
            "--prediction-method-labels",
            "No context",
            "Custom method",
        ]
    )

    assert cfg.prediction_methods == ("no_context", "custom_method")
    assert cfg.prediction_method_labels == ("No context", "Custom method")


def test_parse_rejects_mismatched_conditioning_labels(tmp_path):
    module = _load_module()

    with pytest.raises(ValueError, match="--conditionings and --conditioning-labels"):
        module.parse_args(
            [
                "--results-base",
                str(tmp_path),
                "--conditionings",
                "noenv_nogeno",
                "env_nogeno",
                "--conditioning-labels",
                "P",
            ]
        )


def test_parse_rejects_mismatched_prediction_method_labels(tmp_path):
    module = _load_module()

    with pytest.raises(
        ValueError, match="--prediction-methods and --prediction-method-labels"
    ):
        module.parse_args(
            [
                "--results-base",
                str(tmp_path),
                "--prediction-methods",
                "no_context",
                "random_context",
                "--prediction-method-labels",
                "No context",
            ]
        )


def test_default_test_set_is_legacy(tmp_path):
    module = _load_module()

    cfg = module.parse_args(["--results-base", str(tmp_path)])
    shifted = module.parse_args(
        ["--results-base", str(tmp_path), "--test-set", "shifted"]
    )

    assert cfg.test_set == "legacy"
    assert cfg.splits == ("plot", "environment", "genotype", "unseen")
    assert shifted.test_set == "shifted"
    assert shifted.splits == ("seen", "env", "geno", "unseen")


def test_metric_path_construction_uses_requested_split_and_run_name(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    expected = _write_summary(
        tmp_path,
        split="plot",
        model="ANP",
        dataset_size_token="8k",
        training_length_token="1m",
        mean=0.5,
    )

    path = module._metric_csv_path(
        cfg,
        split="plot",
        model="ANP",
        dataset_size=cfg.dataset_sizes[1],
        training_length=cfg.training_lengths[1],
        conditioning="noenv_nogeno",
        prediction_method="no_context",
    )

    assert path == expected


def test_dataset_size_labels_are_display_only(tmp_path):
    module = _load_module()
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--models",
            "LNP",
            "ANP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "8k",
            "--dataset-size-labels",
            "1K examples",
            "8K examples",
            "--training-lengths",
            "100k",
            "1m",
            "--set-mode",
            "nested_noprior",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
        ]
    )
    _write_small_grid(tmp_path)

    metrics = module.collect_metrics(cfg)
    module.plot_boxplot(metrics, cfg, tmp_path / "custom_labels")

    assert set(metrics["dataset_size"]) == {1024, 8192}
    assert set(metrics["dataset_size_label"]) == {"1K examples", "8K examples"}
    assert (tmp_path / "custom_labels.png").is_file()
    assert (tmp_path / "custom_labels.pdf").is_file()


def test_prediction_method_labels_are_display_only_in_metrics_and_plot(tmp_path):
    module = _load_module()
    for method, mean in [("no_context", 1.0), ("random_context", 2.0)]:
        _write_summary(
            tmp_path,
            split="plot",
            model="LNP",
            dataset_size_token="1k",
            training_length_token="100k",
            prediction_method=method,
            mean=mean,
        )
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--models",
            "LNP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "--training-lengths",
            "100k",
            "--set-mode",
            "nested_noprior",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "random_context",
            "--prediction-method-labels",
            "No ctx",
            "Random ctx",
            "--point-style",
            "swapped-fill",
        ]
    )

    module.write_outputs(cfg)

    metrics = pd.read_csv(cfg.output_dir / "dataset_scaling_metrics.csv")
    assert metrics["prediction_method"].tolist() == ["no_context", "random_context"]
    assert metrics["prediction_method_label"].tolist() == ["No ctx", "Random ctx"]
    assert metrics["summary_path"].str.contains("/no_context/").any()
    assert metrics["summary_path"].str.contains("/random_context/").any()
    assert (cfg.output_dir / "dataset_scaling.png").is_file()
    assert (cfg.output_dir / "dataset_scaling.pdf").is_file()


def test_collect_metrics_reads_context_matched_blocked_summaries(tmp_path):
    module = _load_module()
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--models",
            "LNP",
            "ANP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "8k",
            "--training-lengths",
            "100k",
            "1m",
            "--set-mode",
            "nested_noprior",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "--view",
            "context_matched_blocked",
        ]
    )
    _write_small_grid(tmp_path, view="context_matched_blocked")
    _write_summary(
        tmp_path,
        split="plot",
        model="LNP",
        dataset_size_token="1k",
        training_length_token="100k",
        mean=99.0,
        view="condition",
    )

    metrics = module.collect_metrics(cfg)

    assert len(metrics) == 8
    assert set(metrics["view"]) == {"context_matched_blocked"}
    assert metrics["summary_path"].str.contains("sig_mmd_context_matched_blocked").all()
    assert 99.0 not in set(metrics["mean"])


def test_prediction_method_view_overrides_collect_mixed_sources(tmp_path):
    module = _load_module()
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--models",
            "LNP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "--training-lengths",
            "100k",
            "--set-mode",
            "nested_noprior",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "random_context",
            "--view",
            "condition",
            "--prediction-method-views",
            "no_context=context_matched_blocked",
        ]
    )
    _write_summary(
        tmp_path,
        split="plot",
        model="LNP",
        dataset_size_token="1k",
        training_length_token="100k",
        prediction_method="no_context",
        mean=1.0,
        view="context_matched_blocked",
    )
    _write_summary(
        tmp_path,
        split="plot",
        model="LNP",
        dataset_size_token="1k",
        training_length_token="100k",
        prediction_method="random_context",
        mean=2.0,
        view="condition",
    )
    _write_summary(
        tmp_path,
        split="plot",
        model="LNP",
        dataset_size_token="1k",
        training_length_token="100k",
        prediction_method="no_context",
        mean=99.0,
        view="condition",
    )

    module.write_outputs(cfg)

    metrics = pd.read_csv(cfg.output_dir / "dataset_scaling_metrics.csv")
    summary = pd.read_csv(cfg.output_dir / "dataset_scaling_summary.csv")
    view_by_method = {
        method: set(group["view"])
        for method, group in metrics.groupby("prediction_method", sort=False)
    }
    assert view_by_method == {
        "no_context": {"context_matched_blocked"},
        "random_context": {"condition"},
    }
    assert (
        metrics.loc[metrics["prediction_method"] == "no_context", "summary_path"]
        .str.contains("sig_mmd_context_matched_blocked")
        .all()
    )
    assert (
        metrics.loc[metrics["prediction_method"] == "random_context", "summary_path"]
        .str.endswith("sig_mmd/sig_mmd_summary.csv")
        .all()
    )
    assert 99.0 not in set(metrics["mean"])
    assert set(summary["prediction_method_views"]) == {
        "no_context=context_matched_blocked;random_context=condition"
    }


def test_collect_metrics_reads_exact_requested_cartesian_product(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    _write_small_grid(tmp_path)
    _write_summary(
        tmp_path,
        split="plot",
        model="LNP",
        dataset_size_token="512k",
        training_length_token="3m",
        mean=99.0,
    )

    metrics = module.collect_metrics(cfg)

    assert len(metrics) == 8
    assert set(metrics["model"]) == {"ANP", "LNP"}
    assert set(metrics["dataset_size"]) == {1024, 8192}
    assert set(metrics["training_steps"]) == {100_000, 1_000_000}
    assert set(metrics["conditioning"]) == {"noenv_nogeno"}
    assert set(metrics["prediction_method"]) == {"no_context"}
    assert set(metrics["view"]) == {"condition"}
    assert metrics["summary_path"].str.endswith("sig_mmd_summary.csv").all()
    assert 99.0 not in set(metrics["mean"])
    assert (metrics["sig_mmd_x1000"] == metrics["mean"] * 1000.0).all()


def test_raw_summary_groups_sig_mmd_x1000_by_budget(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    _write_small_grid(tmp_path)

    metrics = module.collect_metrics(cfg)
    summary = module.summarize_metrics(metrics)

    assert len(summary) == 4
    assert set(summary["n"]) == {2}
    row = summary[
        (summary["dataset_size"] == 8192) & (summary["training_steps"] == 1_000_000)
    ].iloc[0]
    assert row["median"] == pytest.approx(2500.0)
    assert row["mean"] == pytest.approx(2500.0)
    assert row["min"] == pytest.approx(2000.0)
    assert row["max"] == pytest.approx(3000.0)


def test_write_outputs_creates_figure_and_csvs(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    _write_small_grid(tmp_path)

    module.write_outputs(cfg)

    expected_files = [
        "dataset_scaling.png",
        "dataset_scaling.pdf",
        "dataset_scaling_metrics.csv",
        "dataset_scaling_summary.csv",
    ]
    for filename in expected_files:
        assert (cfg.output_dir / filename).is_file()
    assert not (cfg.output_dir / "dataset_scaling_reductions.csv").exists()

    metrics = pd.read_csv(cfg.output_dir / "dataset_scaling_metrics.csv")
    summary = pd.read_csv(cfg.output_dir / "dataset_scaling_summary.csv")
    assert len(metrics) == 8
    assert "sig_mmd_x1000" in metrics.columns
    assert len(summary) == 4


@pytest.mark.parametrize("point_style", ["testset-context", "swapped-fill"])
def test_write_outputs_with_point_style(tmp_path, point_style):
    module = _load_module()
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--models",
            "LNP",
            "ANP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "8k",
            "--training-lengths",
            "100k",
            "1m",
            "--set-mode",
            "nested_noprior",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "--point-style",
            point_style,
        ]
    )
    _write_small_grid(tmp_path)

    module.write_outputs(cfg)

    assert (cfg.output_dir / "dataset_scaling.png").is_file()
    assert (cfg.output_dir / "dataset_scaling.pdf").is_file()


def test_write_outputs_with_swapped_fill_custom_conditioning_labels(tmp_path):
    module = _load_module()
    conditionings = ["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"]
    labels = ["$\\emptyset$", "$e$", "$g$", "$g+e$"]
    for index, conditioning in enumerate(conditionings, start=1):
        _write_summary(
            tmp_path,
            split="plot",
            model="LNP",
            dataset_size_token="1k",
            training_length_token="100k",
            conditioning=conditioning,
            mean=float(index),
        )
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--models",
            "LNP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "--training-lengths",
            "100k",
            "--set-mode",
            "nested_noprior",
            "--conditionings",
            *conditionings,
            "--conditioning-labels",
            *labels,
            "--prediction-methods",
            "no_context",
            "--point-style",
            "swapped-fill",
        ]
    )

    module.write_outputs(cfg)

    assert cfg.conditioning_labels == tuple(labels)
    assert (cfg.output_dir / "dataset_scaling.png").is_file()
    assert (cfg.output_dir / "dataset_scaling.pdf").is_file()
    metrics = pd.read_csv(cfg.output_dir / "dataset_scaling_metrics.csv")
    assert metrics["conditioning_label"].tolist() == labels


def test_missing_metric_files_report_all_requested_cells(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    missing_cells = {("LNP", "1k", "100k"), ("ANP", "8k", "1m")}
    for model, size_token, training_token in product(
        ["LNP", "ANP"], ["1k", "8k"], ["100k", "1m"]
    ):
        if (model, size_token, training_token) in missing_cells:
            continue
        _write_summary(
            tmp_path,
            split="plot",
            model=model,
            dataset_size_token=size_token,
            training_length_token=training_token,
            mean=1.0,
        )

    with pytest.raises(FileNotFoundError) as error:
        module.collect_metrics(cfg)

    message = str(error.value)
    for model, size_token, training_token in missing_cells:
        assert f"model={model}" in message
        assert f"dataset_size={size_token}" in message
        assert f"training_length={training_token}" in message
        assert (
            f"run_name={model}-{size_token}-training{training_token}"
            "_set_mode_nested_noprior"
        ) in message
    assert "split=plot" in message
    assert "conditioning=noenv_nogeno" in message
    assert "prediction_method=no_context" in message
    assert "sig_mmd_summary.csv" in message
    assert message.count("found=0") == 2


def test_ambiguous_metric_files_report_matched_paths(tmp_path):
    module = _load_module()
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "LNP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "--training-lengths",
            "100k",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
        ]
    )
    first = _write_summary(
        tmp_path,
        split="plot",
        model="LNP",
        dataset_size_token="1k",
        training_length_token="100k",
        mean=1.0,
        seed="6",
    )
    second = _write_summary(
        tmp_path,
        split="plot",
        model="LNP",
        dataset_size_token="1k",
        training_length_token="100k",
        mean=2.0,
        seed="7",
    )

    with pytest.raises(FileNotFoundError) as error:
        module.collect_metrics(cfg)

    message = str(error.value)
    assert "found=2" in message
    assert str(first) in message
    assert str(second) in message


def test_missing_metric_column_fails(tmp_path):
    module = _load_module()
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "LNP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "--training-lengths",
            "100k",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            "--metric",
            "p95",
        ]
    )
    path = _write_summary(
        tmp_path,
        split="plot",
        model="LNP",
        dataset_size_token="1k",
        training_length_token="100k",
        mean=1.0,
    )
    path.write_text("view,mean\ncondition,1.0\n")

    with pytest.raises(KeyError, match="p95"):
        module.collect_metrics(cfg)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--point-style", "swapped-fill", "--model-rows", "LNP", "ANP"],
        ["--point-style", "swapped-fill"],
        [],
    ],
)
def test_write_outputs_log_y_uses_a_log_axis(monkeypatch, tmp_path, extra_args):
    module = _load_module()
    closed_figures = []
    monkeypatch.setattr(module.plt, "close", closed_figures.append)
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--models",
            "LNP",
            "ANP",
            "--splits",
            "plot",
            "--dataset-sizes",
            "1k",
            "8k",
            "--training-lengths",
            "100k",
            "1m",
            "--set-mode",
            "nested_noprior",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-methods",
            "no_context",
            *extra_args,
            "--log-y",
        ]
    )
    _write_small_grid(tmp_path)

    module.write_outputs(cfg)

    panels = [ax for ax in closed_figures[0].axes if ax.get_ylabel() or ax.lines]
    assert panels
    assert all(ax.get_yscale() == "log" for ax in panels)
