from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "mmd_table", Path("src/npnf/scripts/paper/mmd_table.py")
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
    mean: float = 0.1234,
    include_metric: bool = True,
) -> Path:
    if metric_type == "csig_mmd":
        dir_name, summary_file = "csig_mmd", "csig_mmd_summary.csv"
    else:
        dir_name, summary_file = "sig_mmd", "sig_mmd_summary.csv"
    if view == "context_matched_blocked":
        dir_name = f"{dir_name}_context_matched_blocked"
    path = (
        root
        / f"synth_test_{split}_dataloaders"
        / model
        / opaque
        / checkpoint
        / f"test_{split}"
        / conditioning
        / prediction_method
        / dir_name
        / summary_file
    )
    path.parent.mkdir(parents=True)
    if include_metric:
        path.write_text(
            f"view,n_units,mean,median,std,p95\n{view},2,{mean},0.2,0.3,0.4\n"
        )
    else:
        path.write_text(f"view,n_units,median\n{view},2,0.2\n")
    return path


def _args(module, *extra: str):
    return module.build_parser().parse_args([*extra])


def test_default_test_set_is_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv("NPNF_RESULTS_DIR", str(tmp_path))
    module = _load_module()

    args = module.build_parser().parse_args([])
    shifted = module.build_parser().parse_args(["--test-set", "shifted"])

    assert args.test_set == "legacy"
    assert module.resolve_splits(args) == ["plot", "genotype", "environment", "unseen"]
    assert module.resolve_splits(shifted) == ["seen", "geno", "env", "unseen"]


def test_validate_and_build_specs_accepts_explicit_label_overrides(tmp_path):
    module = _load_module()
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "CustomModel",
        "--model-names",
        "CNP",
        "Custom",
        "--splits",
        "plot",
        "unseen",
        "--split-labels",
        "Seen",
        "Fully unseen",
        "--conditionings",
        "noenv_nogeno",
        "env_geno",
        "--conditioning-labels",
        "Prior",
        "Env & Geno",
    )

    specs = module.validate_and_build_specs(args)

    assert len(specs) == 8
    assert specs[0].model == "CNP-512k"
    assert specs[0].model_name == "CNP"
    assert specs[0].split_label == "Seen"
    assert specs[0].conditioning_label == "Prior"
    assert specs[-1].model == "CustomModel"
    assert specs[-1].model_name == "Custom"
    assert specs[-1].split_label == "Fully unseen"
    assert specs[-1].conditioning_label == "Env & Geno"


@pytest.mark.parametrize(
    ("bad_args", "message"),
    [
        (("--models", "A", "B", "--model-names", "A"), "--model-names"),
        (("--splits", "plot", "unseen", "--split-labels", "Plot"), "--split-labels"),
        (
            (
                "--conditionings",
                "noenv_nogeno",
                "env_geno",
                "--conditioning-labels",
                "P",
            ),
            "--conditioning-labels",
        ),
        (("--splits", "bogus"), "Unknown split"),
    ],
)
def test_validate_and_build_specs_rejects_invalid_labels_or_splits(bad_args, message):
    module = _load_module()
    args = _args(module, *bad_args)

    with pytest.raises(ValueError, match=message):
        module.validate_and_build_specs(args)


def test_common_latest_checkpoint_uses_largest_step_available_to_all(tmp_path):
    module = _load_module()
    _write_summary(tmp_path, model="CNP-512k", checkpoint="checkpoint-100", mean=0.1)
    _write_summary(tmp_path, model="CNP-512k", checkpoint="checkpoint-200", mean=0.2)
    _write_summary(tmp_path, model="ANP-512k", checkpoint="checkpoint-100", mean=0.3)

    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "ANP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    specs = module.validate_and_build_specs(args)

    checkpoint = module.resolve_common_checkpoint(
        tmp_path, specs, "no_context", "sig_mmd", "latest"
    )
    rows = module.load_summaries(
        tmp_path,
        specs,
        prediction_method="no_context",
        metric_type="sig_mmd",
        checkpoint="latest",
        metric="mean",
    )

    assert checkpoint == "checkpoint-100"
    assert {row.checkpoint for row in rows} == {"checkpoint-100"}
    assert [row.value for row in rows] == [0.1, 0.3]


def test_explicit_checkpoint_accepts_numeric_value(tmp_path):
    module = _load_module()
    _write_summary(tmp_path, checkpoint="checkpoint-100", mean=0.1)
    selected = _write_summary(tmp_path, checkpoint="checkpoint-200", mean=0.2)
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
        "--checkpoint",
        "200",
    )
    specs = module.validate_and_build_specs(args)

    rows = module.load_summaries(
        tmp_path,
        specs,
        prediction_method="no_context",
        metric_type="sig_mmd",
        checkpoint=args.checkpoint,
        metric="mean",
    )

    assert rows[0].checkpoint == "checkpoint-200"
    assert rows[0].summary_path == selected
    assert rows[0].value == 0.2


def test_latest_checkpoint_fails_when_no_common_step(tmp_path):
    module = _load_module()
    _write_summary(tmp_path, model="CNP-512k", checkpoint="checkpoint-100")
    _write_summary(tmp_path, model="ANP-512k", checkpoint="checkpoint-200")
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "ANP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    specs = module.validate_and_build_specs(args)

    with pytest.raises(ValueError, match="No common checkpoint"):
        module.resolve_common_checkpoint(
            tmp_path, specs, "no_context", "sig_mmd", "latest"
        )


def test_load_summaries_reports_all_missing_explicit_checkpoint_cells(tmp_path):
    module = _load_module()
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "ANP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    specs = module.validate_and_build_specs(args)

    with pytest.raises(FileNotFoundError) as error_info:
        module.load_summaries(
            tmp_path,
            specs,
            prediction_method="no_context",
            metric_type="sig_mmd",
            checkpoint="100",
            metric="mean",
        )

    message = str(error_info.value)
    assert "model=CNP-512k, split=plot" in message
    assert "model=ANP-512k, split=plot" in message
    assert "checkpoint=checkpoint-100" in message
    assert "expected_glob=" in message
    assert "launch_sig_mmd_context_matched_blocked_jobs.py" not in message


def test_context_matched_missing_failure_includes_restart_summary(tmp_path):
    module = _load_module()
    models = ["CNP-512k", "ANP-512k"]
    conditionings = ["noenv_nogeno", "env_geno"]
    prediction_methods = ["no_context", "random_context"]
    for model in models:
        for conditioning in conditionings:
            for prediction_method in prediction_methods:
                _write_summary(
                    tmp_path,
                    model=model,
                    split="plot",
                    conditioning=conditioning,
                    prediction_method=prediction_method,
                    view="context_matched_blocked",
                )
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        *models,
        "--splits",
        "plot",
        "environment",
        "--conditionings",
        *conditionings,
    )
    specs = module.validate_and_build_specs(args)

    with pytest.raises(FileNotFoundError) as error_info:
        module.load_summaries_for_methods(
            tmp_path,
            specs,
            prediction_methods=prediction_methods,
            metric_type="sig_mmd",
            checkpoint="100",
            metric="mean",
            view="context_matched_blocked",
        )

    message = str(error_info.value)
    assert "Restart summary for missing context-matched blocked Sig-MMD jobs" in message
    assert "missing_cells=8" in message
    assert "models=ANP-512k CNP-512k" in message
    assert "splits=environment" in message
    assert "conditionings=env_geno noenv_nogeno" in message
    assert "prediction_methods=no_context random_context" in message
    assert (
        "uv run python scripts/launch_sig_mmd_context_matched_blocked_jobs.py"
        in message
    )
    assert message.index("Detailed lookup diagnostics:") < message.index(
        "Suggested relaunch commands"
    )
    assert message.rstrip().endswith("--checkpoint checkpoint-100")
    assert str(tmp_path) not in message[message.index("Suggested relaunch commands") :]
    assert '--results-base "$NPNF_RESULTS_DIR"' in message
    assert "--models ANP-512k CNP-512k" in message
    assert "--splits environment" in message
    assert "--conditionings env_geno noenv_nogeno" in message
    assert "--prediction-methods no_context random_context" in message
    assert "--checkpoint checkpoint-100" in message
    assert "expected_glob=" in message


def test_relaunch_command_uses_uv_env_placeholder_and_omits_defaults(tmp_path):
    module = _load_module()
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "--splits",
        "plot",
        "--conditionings",
        *module.DEFAULT_CONDITIONINGS,
    )
    specs = module.validate_and_build_specs(args)

    with pytest.raises(FileNotFoundError) as error_info:
        module.load_summaries_for_methods(
            tmp_path,
            specs,
            prediction_methods=sorted(module.PREDICTION_METHODS),
            metric_type="sig_mmd",
            checkpoint="latest",
            metric="mean",
            view="context_matched_blocked",
        )

    message = str(error_info.value)
    relaunch_section = message[message.index("Suggested relaunch commands") :]
    assert (
        "uv run python scripts/launch_sig_mmd_context_matched_blocked_jobs.py "
        '--results-base "$NPNF_RESULTS_DIR" --models CNP-512k --splits plot'
        in relaunch_section
    )
    assert str(tmp_path) not in relaunch_section
    assert "--checkpoint" not in relaunch_section
    assert "--conditionings" not in relaunch_section
    assert "--prediction-methods" not in relaunch_section


def test_latest_checkpoint_missing_failure_reports_all_cells(tmp_path):
    module = _load_module()
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "ANP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    specs = module.validate_and_build_specs(args)

    with pytest.raises(FileNotFoundError) as error_info:
        module.resolve_common_checkpoint_for_methods(
            tmp_path, specs, ["no_context", "random_context"], "sig_mmd", "latest"
        )

    message = str(error_info.value)
    assert "model=CNP-512k, split=plot" in message
    assert "prediction_method=no_context" in message
    assert "model=ANP-512k, split=plot" in message
    assert "prediction_method=random_context" in message
    assert "checkpoint=checkpoint-*" in message
    assert "expected_glob=" in message


def test_ambiguous_summary_failure_lists_paths(tmp_path):
    module = _load_module()
    first = _write_summary(tmp_path, opaque="6", checkpoint="checkpoint-100")
    second = _write_summary(tmp_path, opaque="7", checkpoint="checkpoint-100")
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    specs = module.validate_and_build_specs(args)

    with pytest.raises(
        ValueError, match=r"Ambiguous sig_mmd_summary.csv"
    ) as error_info:
        module.load_summaries(
            tmp_path,
            specs,
            prediction_method="no_context",
            metric_type="sig_mmd",
            checkpoint="latest",
            metric="mean",
        )

    message = str(error_info.value)
    assert str(first) in message
    assert str(second) in message


def test_missing_metric_failure_names_metric_and_path(tmp_path):
    module = _load_module()
    summary_path = _write_summary(tmp_path, include_metric=False)
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    specs = module.validate_and_build_specs(args)

    with pytest.raises(ValueError, match="Metric column 'mean' missing") as error_info:
        module.load_summaries(
            tmp_path,
            specs,
            prediction_method="no_context",
            metric_type="sig_mmd",
            checkpoint="latest",
            metric="mean",
        )

    assert str(summary_path) in str(error_info.value)


def test_nonfinite_metric_failure_names_metric_and_path(tmp_path):
    module = _load_module()
    summary_path = _write_summary(tmp_path, mean=float("nan"))
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    specs = module.validate_and_build_specs(args)

    with pytest.raises(
        ValueError, match="Metric column 'mean' is non-finite"
    ) as error_info:
        module.load_summaries(
            tmp_path,
            specs,
            prediction_method="no_context",
            metric_type="sig_mmd",
            checkpoint="latest",
            metric="mean",
        )

    assert str(summary_path) in str(error_info.value)


@pytest.mark.parametrize(
    ("metric_type", "metric_label"), [("sig_mmd", "Sig-MMD"), ("csig_mmd", "CSig-MMD")]
)
def test_writes_long_wide_and_latex_outputs(tmp_path, metric_type, metric_label):
    module = _load_module()
    for split, conditioning, mean in [
        ("plot", "noenv_nogeno", 0.1111),
        ("plot", "env_geno", 0.2222),
        ("unseen", "noenv_nogeno", 0.3333),
        ("unseen", "env_geno", 0.4444),
    ]:
        _write_summary(
            tmp_path,
            metric_type=metric_type,
            split=split,
            conditioning=conditioning,
            mean=mean,
        )
    output_folder = tmp_path / "out"

    module.main(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "--splits",
            "plot",
            "unseen",
            "--split-labels",
            "Seen",
            "Fully unseen",
            "--conditionings",
            "noenv_nogeno",
            "env_geno",
            "--conditioning-labels",
            "P",
            "E&G",
            "--metric-type",
            metric_type,
            "--decimals",
            "2",
            "--output-folder",
            str(output_folder),
        ]
    )

    long_path = output_folder / f"{metric_type}_no_context_long.csv"
    wide_path = output_folder / f"{metric_type}_no_context_wide.csv"
    tex_path = output_folder / f"{metric_type}_no_context.tex"
    assert long_path.is_file()
    assert wide_path.is_file()
    assert tex_path.is_file()

    long_df = pl.read_csv(long_path)
    assert long_df.columns == [
        "model",
        "model_name",
        "split",
        "split_label",
        "conditioning",
        "conditioning_label",
        "prediction_method",
        "checkpoint",
        "view",
        "mean",
        "summary_path",
    ]
    assert long_df["view"].to_list() == ["condition"] * 4
    assert long_df["mean"].to_list() == [0.1111, 0.2222, 0.3333, 0.4444]

    wide_df = pl.read_csv(wide_path)
    assert wide_df.columns == [
        "model",
        "model_name",
        "plot__noenv_nogeno",
        "plot__env_geno",
        "unseen__noenv_nogeno",
        "unseen__env_geno",
    ]
    assert wide_df["plot__noenv_nogeno"][0] == 0.1111
    assert wide_df["unseen__env_geno"][0] == 0.4444

    latex = tex_path.read_text()
    assert r"\toprule" in latex
    assert rf"Mean {metric_label}$^2 \times 10^3$ ($\downarrow$)" in latex
    assert r"\midrule" in latex
    assert r"\bottomrule" in latex
    assert r"\multicolumn{2}{c}{Seen}" in latex
    assert r"\multicolumn{2}{c}{Fully unseen}" in latex
    assert r"E\&G" in latex
    assert (
        r"CNP & \textbf{111.10} & \textbf{222.20} & \textbf{333.30} & \textbf{444.40}"
        in latex
    )
    assert (output_folder / f"{metric_type}_no_context.md").read_text() == (
        "| Model | Seen P | Seen E&G | Fully unseen P | Fully unseen E&G |\n"
        "|---|---:|---:|---:|---:|\n"
        "| CNP | **111.10** | **222.20** | **333.30** | **444.40** |\n"
    )


def test_prediction_methods_write_method_grouped_split_average_outputs(tmp_path):
    module = _load_module()
    values = {
        ("CNP-512k", "no_context", "plot", "noenv_nogeno"): 0.001,
        ("CNP-512k", "no_context", "unseen", "noenv_nogeno"): 0.003,
        ("CNP-512k", "no_context", "plot", "env_geno"): 0.004,
        ("CNP-512k", "no_context", "unseen", "env_geno"): 0.006,
        ("CNP-512k", "random_context", "plot", "noenv_nogeno"): 0.010,
        ("CNP-512k", "random_context", "unseen", "noenv_nogeno"): 0.012,
        ("CNP-512k", "random_context", "plot", "env_geno"): 0.014,
        ("CNP-512k", "random_context", "unseen", "env_geno"): 0.016,
        ("CNP-512k", "max_height", "plot", "noenv_nogeno"): 0.020,
        ("CNP-512k", "max_height", "unseen", "noenv_nogeno"): 0.022,
        ("CNP-512k", "max_height", "plot", "env_geno"): 0.024,
        ("CNP-512k", "max_height", "unseen", "env_geno"): 0.026,
        ("ANP-512k", "no_context", "plot", "noenv_nogeno"): 0.002,
        ("ANP-512k", "no_context", "unseen", "noenv_nogeno"): 0.004,
        ("ANP-512k", "no_context", "plot", "env_geno"): 0.003,
        ("ANP-512k", "no_context", "unseen", "env_geno"): 0.005,
        ("ANP-512k", "random_context", "plot", "noenv_nogeno"): 0.011,
        ("ANP-512k", "random_context", "unseen", "noenv_nogeno"): 0.013,
        ("ANP-512k", "random_context", "plot", "env_geno"): 0.013,
        ("ANP-512k", "random_context", "unseen", "env_geno"): 0.015,
        ("ANP-512k", "max_height", "plot", "noenv_nogeno"): 0.019,
        ("ANP-512k", "max_height", "unseen", "noenv_nogeno"): 0.021,
        ("ANP-512k", "max_height", "plot", "env_geno"): 0.025,
        ("ANP-512k", "max_height", "unseen", "env_geno"): 0.027,
    }
    for (model, prediction_method, split, conditioning), mean in values.items():
        _write_summary(
            tmp_path,
            model=model,
            prediction_method=prediction_method,
            split=split,
            conditioning=conditioning,
            mean=mean,
        )
    output_folder = tmp_path / "out"

    module.main(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "ANP-512k",
            "--model-names",
            "CNP",
            "ANP",
            "--splits",
            "plot",
            "unseen",
            "--split-labels",
            "Seen",
            "Fully unseen",
            "--conditionings",
            "noenv_nogeno",
            "env_geno",
            "--conditioning-labels",
            "P",
            "E&G",
            "--prediction-methods",
            "no_context",
            "random_context",
            "max_height",
            "--prediction-method-names",
            "prior",
            "random ctx",
            "height ctx",
            "--reference-model",
            "CNP",
            "--decimals",
            "2",
            "--output-folder",
            str(output_folder),
        ]
    )

    stem = "sig_mmd_no_context_random_context_max_height_avg_splits"
    long_path = output_folder / f"{stem}_long.csv"
    wide_path = output_folder / f"{stem}_wide.csv"
    tex_path = output_folder / f"{stem}.tex"
    assert long_path.is_file()
    assert wide_path.is_file()
    assert tex_path.is_file()
    assert not (output_folder / "sig_mmd_no_context.tex").exists()

    long_df = pl.read_csv(long_path)
    assert long_df.columns == [
        "model",
        "model_name",
        "prediction_method",
        "prediction_method_label",
        "conditioning",
        "conditioning_label",
        "checkpoint",
        "view",
        "source_splits",
        "source_split_labels",
        "n_splits",
        "mean",
        "summary_paths",
    ]
    cnp_no_context_p = long_df.filter(
        (pl.col("model") == "CNP-512k")
        & (pl.col("prediction_method") == "no_context")
        & (pl.col("conditioning") == "noenv_nogeno")
    )
    assert cnp_no_context_p["view"][0] == "condition"
    assert cnp_no_context_p["mean"][0] == pytest.approx(0.002)
    assert cnp_no_context_p["source_splits"][0] == "plot,unseen"
    assert cnp_no_context_p["source_split_labels"][0] == "Seen,Fully unseen"
    assert cnp_no_context_p["n_splits"][0] == 2

    wide_df = pl.read_csv(wide_path)
    assert wide_df.columns == [
        "model",
        "model_name",
        "no_context__noenv_nogeno",
        "no_context__env_geno",
        "random_context__noenv_nogeno",
        "random_context__env_geno",
        "max_height__noenv_nogeno",
        "max_height__env_geno",
    ]
    assert wide_df["no_context__noenv_nogeno"][0] == pytest.approx(0.002)
    assert wide_df["max_height__env_geno"][1] == pytest.approx(0.026)

    latex = tex_path.read_text()
    assert cnp_no_context_p["prediction_method_label"][0] == "prior"
    assert r"\multicolumn{2}{c}{prior}" in latex
    assert r"\multicolumn{2}{c}{random ctx}" in latex
    assert r"\multicolumn{2}{c}{height ctx}" in latex
    assert r"\multicolumn{2}{c}{no context}" not in latex
    assert r"\multicolumn{2}{c}{Seen}" not in latex
    assert r"P & E\&G & P & E\&G & P & E\&G" in latex
    assert r"CNP & \textbf{2.00}" in latex
    assert r"\cellcolor{red!" in latex
    assert r"\cellcolor{green!" in latex
    assert r"\textbf{4.00}" in latex


@pytest.mark.parametrize("metric_type", ["sig_mmd", "csig_mmd"])
def test_context_matched_blocked_view_reads_blocked_dir_and_writes_viewed_stem(
    tmp_path, metric_type
):
    module = _load_module()
    _write_summary(tmp_path, metric_type=metric_type, view="condition", mean=0.9999)
    blocked_summary = _write_summary(
        tmp_path, metric_type=metric_type, view="context_matched_blocked", mean=0.1234
    )
    output_folder = tmp_path / "out"

    module.main(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--metric-type",
            metric_type,
            "--view",
            "context_matched_blocked",
            "--checkpoint",
            "100",
            "--output-folder",
            str(output_folder),
        ]
    )

    stem = f"{metric_type}_context_matched_blocked_no_context"
    long_path = output_folder / f"{stem}_long.csv"
    assert long_path.is_file()
    assert (output_folder / f"{stem}_wide.csv").is_file()
    assert (output_folder / f"{stem}.tex").is_file()
    assert not (output_folder / f"{metric_type}_no_context_long.csv").exists()

    long_df = pl.read_csv(long_path)
    assert long_df["view"].to_list() == ["context_matched_blocked"]
    assert long_df["mean"].to_list() == [0.1234]
    assert long_df["summary_path"].to_list() == [str(blocked_summary)]
    assert f"{metric_type}_context_matched_blocked" in long_df["summary_path"][0]


def test_prediction_method_view_override_validation_errors():
    module = _load_module()

    cases = [
        (["no_context"], "METHOD=VIEW"),
        (["no_context=context_matched_blocked", "no_context=condition"], "duplicate"),
        (["random_context=context_matched_blocked"], "Unselected"),
        (["no_context=bogus"], "Unknown --view"),
    ]
    for raw_overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            module.resolve_prediction_method_views(
                ["no_context"],
                metric_type="sig_mmd",
                default_view="condition",
                raw_overrides=raw_overrides,
            )


def test_csig_mmd_prediction_method_view_allows_context_matched_blocked():
    module = _load_module()

    views = module.resolve_prediction_method_views(
        ["no_context"],
        metric_type="csig_mmd",
        default_view="condition",
        raw_overrides=["no_context=context_matched_blocked"],
    )

    assert views == {"no_context": "context_matched_blocked"}


def test_mixed_prediction_method_views_read_distinct_summary_dirs(tmp_path):
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
    output_folder = tmp_path / "out"

    module.main(
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
            "--output-folder",
            str(output_folder),
        ]
    )

    stem = "sig_mmd_mixed_views_no_context_random_context_max_height_avg_splits"
    long_path = output_folder / f"{stem}_long.csv"
    assert long_path.is_file()
    assert (output_folder / f"{stem}_wide.csv").is_file()
    assert (output_folder / f"{stem}.tex").is_file()
    old_stem = "sig_mmd_no_context_random_context_max_height_avg_splits"
    assert not (output_folder / f"{old_stem}_long.csv").exists()

    long_df = pl.read_csv(long_path)
    view_by_method = dict(
        zip(
            long_df["prediction_method"].to_list(),
            long_df["view"].to_list(),
            strict=True,
        )
    )
    path_by_method = dict(
        zip(
            long_df["prediction_method"].to_list(),
            long_df["summary_paths"].to_list(),
            strict=True,
        )
    )
    mean_by_method = dict(
        zip(
            long_df["prediction_method"].to_list(),
            long_df["mean"].to_list(),
            strict=True,
        )
    )
    assert view_by_method == {
        "no_context": "context_matched_blocked",
        "random_context": "condition",
        "max_height": "condition",
    }
    assert path_by_method["no_context"] == str(blocked_summary)
    assert path_by_method["random_context"] == str(random_context_summary)
    assert path_by_method["max_height"] == str(max_height_summary)
    assert mean_by_method == {
        "no_context": pytest.approx(0.001),
        "random_context": pytest.approx(0.002),
        "max_height": pytest.approx(0.003),
    }


def test_single_method_prediction_method_view_override_uses_selected_method(tmp_path):
    module = _load_module()
    _write_summary(tmp_path, prediction_method="no_context", view="condition", mean=0.9)
    blocked_summary = _write_summary(
        tmp_path,
        prediction_method="no_context",
        view="context_matched_blocked",
        mean=0.1234,
    )
    output_folder = tmp_path / "out"

    module.main(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            "CNP-512k",
            "--splits",
            "plot",
            "--conditionings",
            "noenv_nogeno",
            "--prediction-method",
            "no_context",
            "--view",
            "condition",
            "--prediction-method-views",
            "no_context=context_matched_blocked",
            "--checkpoint",
            "100",
            "--output-folder",
            str(output_folder),
        ]
    )

    long_path = output_folder / "sig_mmd_context_matched_blocked_no_context_long.csv"
    assert long_path.is_file()
    long_df = pl.read_csv(long_path)
    assert long_df["view"].to_list() == ["context_matched_blocked"]
    assert long_df["summary_path"].to_list() == [str(blocked_summary)]
    assert long_df["mean"].to_list() == [0.1234]


def test_single_method_prediction_method_view_rejects_unselected_override(tmp_path):
    module = _load_module()

    with pytest.raises(SystemExit, match="Unselected"):
        module.main(
            [
                "--results-base",
                str(tmp_path),
                "--prediction-method",
                "no_context",
                "--prediction-method-views",
                "random_context=context_matched_blocked",
            ]
        )


def test_context_matched_blocked_view_method_grouped_table_uses_viewed_stem(tmp_path):
    module = _load_module()
    for prediction_method, mean in {
        "no_context": 0.001,
        "random_context": 0.003,
    }.items():
        _write_summary(
            tmp_path,
            prediction_method=prediction_method,
            view="context_matched_blocked",
            mean=mean,
        )
    output_folder = tmp_path / "out"

    module.main(
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
            "--view",
            "context_matched_blocked",
            "--checkpoint",
            "100",
            "--output-folder",
            str(output_folder),
        ]
    )

    stem = "sig_mmd_context_matched_blocked_no_context_random_context_avg_splits"
    long_path = output_folder / f"{stem}_long.csv"
    assert long_path.is_file()
    assert (output_folder / f"{stem}_wide.csv").is_file()
    assert (output_folder / f"{stem}.tex").is_file()
    assert not (
        output_folder / "sig_mmd_no_context_random_context_avg_splits_long.csv"
    ).exists()

    long_df = pl.read_csv(long_path)
    assert long_df["view"].to_list() == [
        "context_matched_blocked",
        "context_matched_blocked",
    ]
    assert all(
        "sig_mmd_context_matched_blocked" in path for path in long_df["summary_paths"]
    )


def test_context_matched_blocked_view_missing_summary_error_names_blocked_dir(tmp_path):
    module = _load_module()
    _write_summary(tmp_path, view="condition")
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    specs = module.validate_and_build_specs(args)

    with pytest.raises(FileNotFoundError, match="sig_mmd_context_matched_blocked"):
        module.load_summaries(
            tmp_path,
            specs,
            prediction_method="no_context",
            metric_type="sig_mmd",
            checkpoint="latest",
            metric="mean",
            view="context_matched_blocked",
        )


def test_default_results_base_without_environment(monkeypatch):
    monkeypatch.delenv("NPNF_RESULTS_DIR", raising=False)
    module = _load_module()

    with pytest.raises(KeyError, match="NPNF_RESULTS_DIR"):
        module.build_parser()


def test_prediction_method_options_accept_single_or_multiple_methods():
    module = _load_module()

    assert module.resolve_prediction_method_labels(
        ["no_context", "random_context", "max_height"], None
    ) == ["no context", "random context", "max context"]


def test_prediction_method_names_must_match_prediction_methods(tmp_path):
    module = _load_module()

    with pytest.raises(SystemExit, match="--prediction-method-names"):
        module.main(
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
                "max_height",
                "--prediction-method-names",
                "prior",
            ]
        )

    with pytest.raises(SystemExit, match="requires --prediction-methods"):
        module.main(["--prediction-method-names", "prior"])


def test_import_does_not_capture_stale_results_base(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("NPNF_RESULTS_DIR", "/first")
    first = module.build_parser().parse_args([]).results_base
    monkeypatch.setenv("NPNF_RESULTS_DIR", "/second")
    second = module.build_parser().parse_args([]).results_base

    assert first == "/first"
    assert second == "/second"


def test_bold_marks_every_value_tied_with_the_lowest(tmp_path):
    """Two values differ only beyond twice the standard error of their difference,
    and a value below the split's floor counts as the floor."""
    module = _load_module()
    cells = {  # model: (plot mean, unseen mean); n_units 4 and std 0.0002 -> SE 0.0001
        "A-512k": (0.0010, -0.0010),
        "B-512k": (0.0012, 0.0005),
        "C-512k": (0.0020, 0.0009),
    }
    for model, means in cells.items():
        for split, mean in zip(("plot", "unseen"), means, strict=True):
            path = _write_summary(tmp_path, model=model, split=split, mean=mean)
            path.write_text(
                f"view,n_units,mean,median,std,p95\ncondition,4,{mean},0,0.0002,0\n"
            )
    output_folder = tmp_path / "out"
    module.main(
        [
            "--results-base",
            str(tmp_path),
            "--models",
            *cells,
            "--model-names",
            "A",
            "B",
            "C",
            "--splits",
            "plot",
            "unseen",
            "--conditionings",
            "noenv_nogeno",
            "--conditioning-labels",
            "P",
            "--floors",
            "unseen=0.5",
            "--output-folder",
            str(output_folder),
        ]
    )
    rows = {
        line.split(" | ")[0].strip("| "): line.split(" | ")[1:]
        for line in (output_folder / "sig_mmd_no_context.md").read_text().splitlines()
        if line.startswith("| ") and not line.startswith("| Model")
    }
    plot = {model: cells[0].startswith("**") for model, cells in rows.items()}
    unseen = {model: cells[1].startswith("**") for model, cells in rows.items()}
    assert plot == {"A": True, "B": True, "C": False}
    # A (-1.0) counts as the floor 0.5, so B (0.5) ties with it; C (0.9) does not.
    assert unseen == {"A": True, "B": True, "C": False}


def test_model_checkpoint_overrides_the_common_checkpoint(tmp_path):
    """A fitted baseline has checkpoint-0, which no trained model shares."""
    module = _load_module()
    _write_summary(tmp_path, model="CNP-512k", checkpoint="checkpoint-100", mean=0.1)
    _write_summary(tmp_path, model="CNP-512k", checkpoint="checkpoint-200", mean=0.2)
    _write_summary(tmp_path, model="Baseline-512k", checkpoint="checkpoint-0", mean=0.3)
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "Baseline-512k",
        "--model-checkpoints",
        "Baseline-512k=0",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    specs = module.validate_and_build_specs(args)

    rows = module.load_summaries(
        tmp_path,
        specs,
        prediction_method="no_context",
        metric_type="sig_mmd",
        checkpoint="latest",
        metric="mean",
    )

    assert [(row.spec.model, row.checkpoint, row.value) for row in rows] == [
        ("CNP-512k", "checkpoint-200", 0.2),
        ("Baseline-512k", "checkpoint-0", 0.3),
    ]


@pytest.mark.parametrize("item", ["Baseline-512k", "Baseline-512k=x", "Other=0"])
def test_model_checkpoints_reject_malformed_or_unknown_models(tmp_path, item):
    module = _load_module()
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "Baseline-512k",
        "--model-checkpoints",
        item,
    )

    with pytest.raises(ValueError, match="--model-checkpoints"):
        module.validate_and_build_specs(args)


def _folded_row(tmp_path, *, n_seeds: int, sd: str, n_units: int, floor_sds=None):
    """The standard error load_summaries reads for one folded FIP1-style cell."""
    module = _load_module()
    path = _write_summary(tmp_path, view="context_matched_blocked")
    path.write_text(
        f"n_seeds,sd,n_units,mean,std\n{n_seeds},{sd},{n_units},0.001,0.002\n"
    )
    args = _args(
        module,
        "--results-base",
        str(tmp_path),
        "--models",
        "CNP-512k",
        "--splits",
        "plot",
        "--conditionings",
        "noenv_nogeno",
    )
    (row,) = module.load_summaries(
        tmp_path,
        module.validate_and_build_specs(args),
        prediction_method="no_context",
        metric_type="sig_mmd",
        checkpoint="latest",
        metric="mean",
        view="context_matched_blocked",
        floor_sds=floor_sds,
    )
    return row


def test_folded_cell_uses_the_seed_standard_error(tmp_path):
    row = _folded_row(tmp_path, n_seeds=4, sd="0.004", n_units=5)
    assert row.standard_error == 0.004 / 2


def test_folded_single_run_uses_the_unit_standard_error(tmp_path):
    """A fitted baseline has one run and no seed spread."""
    assert _folded_row(tmp_path / "a", n_seeds=1, sd="", n_units=4).standard_error == (
        0.002 / 2
    )
    single = _folded_row(
        tmp_path / "b", n_seeds=1, sd="", n_units=1, floor_sds={"plot": 0.0001}
    )
    assert single.standard_error == 0.0001
