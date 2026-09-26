from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "context_training", Path("src/npnf/scripts/paper/context_training.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATALOADERS = {
    "plot": ("synth_test_plot_dataloaders", "test_plot"),
    "environment": ("synth_test_environment_dataloaders", "test_environment"),
}


MODELS = (
    "LNP-512k-training3m_set_mode_nested_all",
    "LNP-512k-training3m_set_mode_nested_noprior",
    "ANP-512k-training3m_set_mode_nested_all",
    "ANP-512k-training3m_set_mode_nested_noprior",
)
MODEL_NAMES = ("LNP-NP", "LNP-NP-noprior", "ANP-NP", "ANP-NP-noprior")


def _write_summary(
    results_base: Path,
    *,
    split: str,
    model: str,
    mean: float,
    conditioning: str = "noenv_nogeno",
    prediction_method: str = "no_context",
    checkpoint: str = "checkpoint-3000000",
    view: str = "condition",
) -> Path:
    dataloader_dir, split_dir = DATALOADERS[split]
    metric_dir = (
        "sig_mmd_context_matched_blocked"
        if view == "context_matched_blocked"
        else "sig_mmd"
    )
    path = (
        results_base
        / dataloader_dir
        / model
        / "6"
        / checkpoint
        / split_dir
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


def _small_config(module, tmp_path: Path, extra_args: list[str] | None = None):
    args = [
        "--results-base",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--models",
        *MODELS,
        "--model-names",
        *MODEL_NAMES,
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
        "--conditioning-labels",
        "P",
        "--prediction-methods",
        "no_context",
        "random_context",
        "--prediction-method-labels",
        "no context",
        "random context",
        "--checkpoint",
        "latest",
    ]
    if extra_args is not None:
        args.extend(extra_args)
    return module.parse_args(args)


def _write_small_grid(results_base: Path, *, view: str = "condition") -> None:
    value = 1.0
    for model in MODELS:
        for method in ("no_context", "random_context"):
            _write_summary(
                results_base,
                split="plot",
                model=model,
                prediction_method=method,
                mean=value,
                view=view,
            )
            value += 1.0


def test_default_test_set_is_legacy(tmp_path):
    module = _load_module()

    cfg = module.parse_args(["--results-base", str(tmp_path)])
    shifted = module.parse_args(
        ["--results-base", str(tmp_path), "--test-set", "shifted"]
    )

    assert cfg.test_set == "legacy"
    assert cfg.splits == ("plot", "environment", "genotype", "unseen")
    assert shifted.splits == ("seen", "env", "geno", "unseen")
    assert module.summary_pattern(
        test_set=shifted.test_set,
        split="env",
        model="LNP",
        conditioning="env_geno",
        prediction_method="no_context",
        checkpoint=None,
        view="condition",
    ).startswith("synth_shifted_env_dataloaders/LNP/")


def test_parse_omitted_labels_follow_requested_values(tmp_path):
    module = _load_module()

    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "model-a",
            "model-b",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_geno",
            "custom_conditioning",
            "--prediction-methods",
            "max_height",
            "custom_method",
        ]
    )

    assert cfg.model_names == ("model-a", "model-b")
    assert cfg.conditioning_labels == ("G", "custom_conditioning")
    assert cfg.prediction_method_labels == ("max context", "custom method")


@pytest.mark.parametrize(
    ("extra_args", "match"),
    [
        (["--row-labels", "Latent NP"], "--row-labels requires"),
        (["--architecture-rows"], "requires at least one"),
        (["--architecture-rows", ""], "requires at least one"),
        (["--architecture-rows", "LNP", "LNP"], "duplicate"),
        (
            ["--architecture-rows", "LNP", "ANP", "--row-labels", "Latent NP"],
            "exactly one label",
        ),
        (["--architecture-rows", "LNP", "--row-labels", ""], "non-empty labels"),
    ],
)
def test_parse_rejects_invalid_architecture_row_options(tmp_path, extra_args, match):
    module = _load_module()

    with pytest.raises(ValueError, match=match):
        module.parse_args(["--results-base", str(tmp_path), *extra_args])


@pytest.mark.parametrize(
    ("extra_args", "match"),
    [
        (["--recipe-labels"], "at least one label"),
        (["--recipe-labels", ""], "non-empty labels"),
        (["--recipe-labels", "baseline"], "recipes=4"),
    ],
)
def test_parse_rejects_invalid_recipe_label_options(tmp_path, extra_args, match):
    module = _load_module()

    with pytest.raises(ValueError, match=match):
        module.parse_args(["--results-base", str(tmp_path), *extra_args])


@pytest.mark.parametrize(
    ("extra_args", "match"),
    [
        (
            ["--models", "model-a", "model-b", "--model-names", "Model A"],
            "--models and --model-names",
        ),
        (
            [
                "--conditionings",
                "noenv_nogeno",
                "env_geno",
                "--conditioning-labels",
                "P",
            ],
            "--conditionings and --conditioning-labels",
        ),
        (
            [
                "--prediction-methods",
                "no_context",
                "random_context",
                "--prediction-method-labels",
                "no context",
            ],
            "--prediction-methods and --prediction-method-labels",
        ),
    ],
)
def test_parse_rejects_mismatched_labels(tmp_path, extra_args, match):
    module = _load_module()

    with pytest.raises(ValueError, match=match):
        module.parse_args(["--results-base", str(tmp_path), *extra_args])


def test_checkpoint_normalization():
    module = _load_module()

    assert module.normalize_checkpoint("latest") == "latest"
    assert module.normalize_checkpoint("3000000") == "checkpoint-3000000"
    assert module.normalize_checkpoint("checkpoint-3000000") == "checkpoint-3000000"
    assert module.checkpoint_step("checkpoint-3000000") == 3_000_000


def test_resolve_checkpoint_uses_latest_common_checkpoint(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    _write_small_grid(tmp_path)
    for model in MODELS:
        for method in ("no_context", "random_context"):
            _write_summary(
                tmp_path,
                split="plot",
                model=model,
                prediction_method=method,
                checkpoint="checkpoint-1000000",
                mean=9.0,
            )

    assert module.resolve_checkpoint(cfg) == "checkpoint-3000000"


def test_collect_metrics_reads_raw_summaries_and_labels(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    _write_small_grid(tmp_path)

    checkpoint = module.resolve_checkpoint(cfg)
    metrics = module.collect_metrics(cfg, checkpoint)

    assert len(metrics) == 8
    assert set(metrics["architecture"]) == {"ANP", "LNP"}
    assert set(metrics["recipe_label"]) == {"NP", "no prior"}
    assert set(metrics["split_label"]) == {"seen"}
    assert set(metrics["conditioning_label"]) == {"P"}
    assert set(metrics["prediction_method_label"]) == {"no context", "random context"}
    assert set(metrics["view"]) == {"condition"}
    assert (metrics["sig_mmd_x1000"] == metrics["mean"] * 1000.0).all()
    assert metrics["summary_path"].str.endswith("sig_mmd_summary.csv").all()


def test_collect_metrics_reads_context_matched_blocked_summaries(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path, ["--view", "context_matched_blocked"])
    _write_small_grid(tmp_path, view="context_matched_blocked")
    _write_summary(
        tmp_path,
        split="plot",
        model=MODELS[0],
        prediction_method="no_context",
        mean=99.0,
        view="condition",
    )

    metrics = module.collect_metrics(cfg, module.resolve_checkpoint(cfg))

    assert len(metrics) == 8
    assert set(metrics["view"]) == {"context_matched_blocked"}
    assert metrics["summary_path"].str.contains("sig_mmd_context_matched_blocked").all()
    assert 99.0 not in set(metrics["mean"])


def test_prediction_method_view_overrides_collect_mixed_sources(tmp_path):
    module = _load_module()
    cfg = _small_config(
        module,
        tmp_path,
        [
            "--view",
            "condition",
            "--prediction-method-views",
            "no_context=context_matched_blocked",
        ],
    )
    value = 1.0
    for model in MODELS:
        _write_summary(
            tmp_path,
            split="plot",
            model=model,
            prediction_method="no_context",
            mean=value,
            view="context_matched_blocked",
        )
        value += 1.0
        _write_summary(
            tmp_path,
            split="plot",
            model=model,
            prediction_method="random_context",
            mean=value,
            view="condition",
        )
        value += 1.0
    _write_summary(
        tmp_path,
        split="plot",
        model=MODELS[0],
        prediction_method="no_context",
        mean=99.0,
        view="condition",
    )

    module.write_outputs(cfg)

    metrics = pd.read_csv(cfg.output_dir / "context_training_metrics.csv")
    summary = pd.read_csv(cfg.output_dir / "context_training_summary.csv")
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


def test_summary_groups_sig_mmd_x1000_by_architecture_and_recipe(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    _write_small_grid(tmp_path)

    metrics = module.collect_metrics(cfg, module.resolve_checkpoint(cfg))
    summary = module.summarize_metrics(metrics)

    assert len(summary) == 4
    assert set(summary["n"]) == {2}
    lnp_np = summary[
        (summary["architecture"] == "LNP") & (summary["recipe_label"] == "NP")
    ].iloc[0]
    assert lnp_np["mean"] == pytest.approx(1500.0)
    assert lnp_np["median"] == pytest.approx(1500.0)


def test_write_outputs_creates_figure_and_csvs(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    _write_small_grid(tmp_path)

    module.write_outputs(cfg)

    expected_files = [
        "context_training.png",
        "context_training.pdf",
        "context_training_metrics.csv",
        "context_training_summary.csv",
    ]
    for filename in expected_files:
        assert (cfg.output_dir / filename).is_file()

    metrics = pd.read_csv(cfg.output_dir / "context_training_metrics.csv")
    summary = pd.read_csv(cfg.output_dir / "context_training_summary.csv")
    assert len(metrics) == 8
    assert "sig_mmd_x1000" in metrics.columns
    assert len(summary) == 4


def test_write_outputs_applies_recipe_labels(tmp_path):
    module = _load_module()
    cfg = _small_config(
        module, tmp_path, ["--recipe-labels", "baseline", "without prior"]
    )
    _write_small_grid(tmp_path)

    module.write_outputs(cfg)

    metrics = pd.read_csv(cfg.output_dir / "context_training_metrics.csv")
    np_labels = set(metrics.loc[metrics["recipe"] == "NP", "recipe_label"])
    noprior_labels = set(metrics.loc[metrics["recipe"] == "noprior", "recipe_label"])
    assert np_labels == {"baseline"}
    assert noprior_labels == {"without prior"}


@pytest.mark.parametrize(
    "extra_args",
    [
        [
            "--architecture-rows",
            "LNP",
            "ANP",
            "--row-labels",
            "Latent NP",
            "Attentive NP",
        ],
        ["--architecture-rows", "LNP", "ANP", "--point-style", "testset-context"],
        ["--architecture-rows", "LNP", "ANP", "--point-style", "swapped-fill"],
        ["--point-style", "testset-context"],
        ["--point-style", "swapped-fill"],
    ],
)
def test_write_outputs_renders_point_styles_and_architecture_rows(tmp_path, extra_args):
    module = _load_module()
    cfg = _small_config(module, tmp_path, extra_args)
    _write_small_grid(tmp_path)

    module.write_outputs(cfg)

    assert (cfg.output_dir / "context_training.png").is_file()
    assert (cfg.output_dir / "context_training.pdf").is_file()


def test_write_outputs_rejects_missing_architecture_row(tmp_path):
    module = _load_module()
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--models",
            MODELS[0],
            "--model-names",
            MODEL_NAMES[0],
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--conditioning-labels",
            "P",
            "--prediction-methods",
            "no_context",
            "--prediction-method-labels",
            "no context",
            "--checkpoint",
            "latest",
            "--architecture-rows",
            "LNP",
            "ANP",
        ]
    )
    _write_summary(tmp_path, split="plot", model=MODELS[0], mean=1.0)

    with pytest.raises(ValueError, match="--architecture-rows"):
        module.write_outputs(cfg)


def test_write_outputs_with_swapped_fill_custom_conditioning_labels(tmp_path):
    module = _load_module()
    conditionings = ["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"]
    labels = ["$\\emptyset$", "$e$", "$g$", "$g+e$"]
    for index, conditioning in enumerate(conditionings, start=1):
        _write_summary(
            tmp_path,
            split="plot",
            model=MODELS[0],
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
            MODELS[0],
            "--model-names",
            MODEL_NAMES[0],
            "--splits",
            "plot",
            "--conditionings",
            *conditionings,
            "--conditioning-labels",
            *labels,
            "--prediction-methods",
            "no_context",
            "--prediction-method-labels",
            "no context",
            "--checkpoint",
            "latest",
            "--point-style",
            "swapped-fill",
        ]
    )

    module.write_outputs(cfg)

    assert cfg.conditioning_labels == tuple(labels)
    assert (cfg.output_dir / "context_training.png").is_file()
    assert (cfg.output_dir / "context_training.pdf").is_file()
    metrics = pd.read_csv(cfg.output_dir / "context_training_metrics.csv")
    assert metrics["conditioning_label"].tolist() == labels


def test_missing_explicit_metric_file_fails(tmp_path):
    module = _load_module()
    cfg = _small_config(module, tmp_path)
    for model in MODELS:
        for method in ("no_context", "random_context"):
            if (model, method) == (MODELS[-1], "random_context"):
                continue
            _write_summary(
                tmp_path, split="plot", model=model, prediction_method=method, mean=1.0
            )

    with pytest.raises(FileNotFoundError, match="Expected exactly one summary"):
        module.collect_metrics(cfg, "checkpoint-3000000")


def test_missing_metric_column_fails(tmp_path):
    module = _load_module()
    cfg = module.parse_args(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            MODELS[0],
            "--model-names",
            MODEL_NAMES[0],
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--conditioning-labels",
            "P",
            "--prediction-methods",
            "no_context",
            "--prediction-method-labels",
            "no context",
            "--checkpoint",
            "checkpoint-3000000",
            "--metric",
            "p95",
        ]
    )
    path = _write_summary(tmp_path, split="plot", model=MODELS[0], mean=1.0)
    path.write_text("view,mean\ncondition,1.0\n")

    with pytest.raises(KeyError, match="p95"):
        module.collect_metrics(cfg, "checkpoint-3000000")


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--architecture-rows", "LNP", "ANP", "--point-style", "swapped-fill"],
        ["--point-style", "swapped-fill"],
        [],
    ],
)
def test_write_outputs_log_y_uses_a_log_axis(monkeypatch, tmp_path, extra_args):
    module = _load_module()
    closed_figures = []
    monkeypatch.setattr(module.plt, "close", closed_figures.append)
    cfg = _small_config(module, tmp_path, [*extra_args, "--log-y"])
    _write_small_grid(tmp_path)

    module.write_outputs(cfg)

    panels = [ax for ax in closed_figures[0].axes if ax.get_ylabel() or ax.lines]
    assert panels
    assert all(ax.get_yscale() == "log" for ax in panels)
