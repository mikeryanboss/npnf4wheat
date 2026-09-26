from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch

from npnf.metrics import csig_mmd
from npnf.metrics.blocked_scoring import BlockedScoringOptions
from npnf.metrics.csig_mmd import CensoringParameters, CensoringSweep
from npnf.metrics.signature import SYNTHETIC_SIGNATURE_HEIGHT_SCALE
from npnf.scripts.metrics import sig_mmd_context_matched_blocked as blocked


class _FakeMetadata:
    def __init__(self, data: dict[str, list[str]]):
        self._data = data

    def flatten(self) -> _FakeMetadata:
        return self

    def __getitem__(self, key: str) -> list[str]:
        return self._data[key]


class _FakePredictions:
    def __init__(self, grid: torch.Tensor):
        self._grid = grid

    def select(self, key: str) -> _FakePredictions:
        assert key == "grid"
        return self

    def flatten(self) -> dict[str, torch.Tensor]:
        return {"grid": self._grid}


class _FakeDataset:
    def __init__(self, key_order: list[tuple[str, str]]):
        self._keys = key_order
        self.num_genotypes = len({gid for gid, _ in key_order})
        self.num_yearsites = len({ys for _, ys in key_order})
        self.n_lodging_draws = 4
        self.scale_noise = 0.5
        self._height_days = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        self._subsample_indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
        # Distinct clean curves make context-aware posteriors non-uniform.
        rows = [
            torch.tensor([u0, 10.0 * u0 + 1.0, 100.0 - u0])
            for u0 in range(len(key_order))
        ]
        self._intermediate = {"heights_clean": torch.stack(rows).float()}
        self._noise = torch.zeros_like(self._intermediate["heights_clean"])

    def get_condition_metadata(self) -> dict[str, list[str]]:
        return {
            "genotype_id": [gid for gid, _ in self._keys],
            "yearsite_uid": [ys for _, ys in self._keys],
        }

    def get_height_days_all(self) -> torch.Tensor:
        return self._height_days

    def get_lodged_chunk(self, indices, draw_indices):
        # Oracle draw ``d`` of dataset condition ``c`` has height ``c * 100 + d``.
        values = torch.tensor(
            [[float(int(c) * 100 + int(d)) for d in draw_indices] for c in indices]
        )
        days = self._height_days.numel()
        return {"height_values_all_nonoise": values.unsqueeze(-1).expand(-1, -1, days)}


class _FakeOracleCfg:
    __name__ = "FakeOracleCfg"


_LEX_KEYS = [("g1", "e1"), ("g1", "e2"), ("g2", "e1"), ("g2", "e2")]


def _write_config(method_dir: Path, *, use_temperature: bool, use_marker: bool) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    method_dir.joinpath("config.json").write_text(
        json.dumps(
            {
                "use_temperature": use_temperature,
                "use_marker": use_marker,
                "dataloader_name": "fake_dataloader",
            }
        )
    )


def _fake_artifact(
    *,
    artifact_key_order: list[tuple[str, str]],
    dataset_key_order: list[tuple[str, str]],
    prediction_day_axis: np.ndarray | None = None,
) -> types.SimpleNamespace:
    dataset_index_by_key = {key: index for index, key in enumerate(dataset_key_order)}
    distances = np.empty((len(artifact_key_order), 4), dtype=np.float64)
    for artifact_index, key in enumerate(artifact_key_order):
        dataset_index = dataset_index_by_key[key]
        for draw in range(4):
            distances[artifact_index, draw] = float(dataset_index * 1000 + draw)
    return types.SimpleNamespace(
        path=Path("artifact"),
        condition_genotype_ids=np.array([gid for gid, _ in artifact_key_order]),
        condition_yearsite_uids=np.array([ys for _, ys in artifact_key_order]),
        mahalanobis_distances=distances.reshape(-1),
        has_lodged=np.tile(np.arange(4) >= 2, len(artifact_key_order)),
        prediction_day_axis=(
            np.array([0.0, 1.0, 2.0], dtype=np.float32)
            if prediction_day_axis is None
            else prediction_day_axis.astype(np.float32)
        ),
        metric_day_axis_norm=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        metadata={
            "draw_count": 4,
            "height_scale": SYNTHETIC_SIGNATURE_HEIGHT_SCALE,
            "depth": 4,
        },
        mrcd_location=np.zeros(2, dtype=np.float64),
        mrcd_precision=np.eye(2, dtype=np.float64),
    )


def _install_fake_world(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    method_name: str = "no_context",
    use_temperature: bool = False,
    use_marker: bool = False,
    dataset_key_order: list[tuple[str, str]] | None = None,
    loader_key_order: list[tuple[str, str]] | None = None,
    artifact_key_order: list[tuple[str, str]] | None = None,
    context_indices: list[list[int]] | None = None,
    prediction_day_axis: np.ndarray | None = None,
    capture: list[dict[str, Any]] | None = None,
) -> tuple[Path, types.SimpleNamespace]:
    dataset_key_order = _LEX_KEYS if dataset_key_order is None else dataset_key_order
    loader_key_order = _LEX_KEYS if loader_key_order is None else loader_key_order
    artifact_key_order = _LEX_KEYS if artifact_key_order is None else artifact_key_order
    method_dir = tmp_path / method_name
    _write_config(method_dir, use_temperature=use_temperature, use_marker=use_marker)
    dataset = _FakeDataset(dataset_key_order)

    grid = torch.stack(
        [
            torch.stack(
                [torch.full((3,), float(row * 10 + draw)) for draw in range(4)], dim=0
            )
            for row in range(len(loader_key_order))
        ],
        dim=0,
    ).unsqueeze(-1)
    batch_dict: dict[str, object] = {
        "data": _FakeMetadata(
            {
                "genotype_id": [gid for gid, _ in loader_key_order],
                "yearsite_uid": [ys for _, ys in loader_key_order],
            }
        ),
        "grid_points": {"X": dataset.get_height_days_all().unsqueeze(-1)},
    }
    if context_indices is not None:
        batch_dict["context_indices"] = context_indices
    batch = (_FakePredictions(grid), batch_dict)

    class _FakeBatchLoader:
        def __init__(self, method_dir: Path, *, load_predictions: bool = True):
            del load_predictions
            self.method_dir = Path(method_dir)

        def load_batch(self, index: int):
            assert index == 0
            return batch

        def __iter__(self):
            return iter([batch])

        def __len__(self) -> int:
            return 1

        @property
        def config(self) -> dict[str, object]:
            return json.loads((self.method_dir / "config.json").read_text())

    artifact = _fake_artifact(
        artifact_key_order=artifact_key_order,
        dataset_key_order=dataset_key_order,
        prediction_day_axis=prediction_day_axis,
    )
    monkeypatch.setattr(blocked, "BatchLoader", _FakeBatchLoader)
    monkeypatch.setattr(
        blocked,
        "get_oracle_config_for_method_dir",
        lambda method_path: (_FakeOracleCfg, "fake_dataloader"),
    )
    monkeypatch.setattr(blocked, "instantiate", lambda cfg: dataset)
    monkeypatch.setattr(
        csig_mmd, "load_signature_mahalanobis_artifact", lambda path: artifact
    )

    if capture is not None:

        def fake_score(
            self,
            oracle_paths,
            model_paths,
            oracle_distances,
            height_scale,
            t_norm,
            self_kernel_batch_size,
            *,
            label,
            sigma,
        ):
            del t_norm, self_kernel_batch_size, sigma
            raw_oracle = (oracle_paths[:, 0, 0] * height_scale).detach().cpu().numpy()
            raw_model = (model_paths[:, 0, 0] * height_scale).detach().cpu().numpy()
            capture.append(
                {
                    "label": label,
                    "raw_oracle": raw_oracle.round(3).tolist(),
                    "raw_model": raw_model.round(3).tolist(),
                    "distances": np.asarray(oracle_distances).tolist(),
                    "c_squared": self.main.c_squared,
                    "oracle_sample_size": oracle_paths.shape[0],
                    "model_sample_size": model_paths.shape[0],
                }
            )
            # The main rule scores 1.5; each rule's score and weights follow its
            # alpha and beta, so the sweep outputs can be told apart.
            return [
                {
                    "mean_w_oracle": 0.25 if index == 0 else rule.alpha,
                    "mean_w_model": 0.75 if index == 0 else rule.alpha / 2,
                    "score": 1.5 if index == 0 else 10 * rule.alpha + rule.beta,
                    "sig_score": 0.5,
                }
                for index, rule in enumerate(self.censorings)
            ]

        monkeypatch.setattr(csig_mmd.CensoringReference, "score", fake_score)

    return method_dir, artifact


@pytest.mark.parametrize(
    "kwargs", [{"oracle_self": True}, {"lead_lag": False}], ids=["oracle", "lead"]
)
def test_csig_rejects_oracle_self_and_raw_paths(tmp_path, kwargs) -> None:
    with pytest.raises(ValueError, match="mahalanobis-artifact"):
        blocked.sig_mmd_context_matched_blocked_analysis(
            method_dir=str(tmp_path / "no_context"),
            mahalanobis_artifact=str(tmp_path / "artifact"),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("use_temperature", "use_marker", "expected_scope", "expected_units"),
    [
        (False, False, "global", ["global"]),
        (True, False, "environment", ["e1", "e2"]),
        (False, True, "genotype", ["g1", "g2"]),
        (True, True, "condition", ["g1::e1", "g1::e2", "g2::e1", "g2::e2"]),
    ],
)
def test_successful_scoring_for_all_covariate_scopes(
    monkeypatch, tmp_path, use_temperature, use_marker, expected_scope, expected_units
) -> None:
    capture: list[dict[str, Any]] = []
    method_dir, _ = _install_fake_world(
        monkeypatch,
        tmp_path,
        use_temperature=use_temperature,
        use_marker=use_marker,
        capture=capture,
    )

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(method_dir),
        mahalanobis_artifact=str(tmp_path / "artifact"),
        options=BlockedScoringOptions(
            n_samples=4,
            block_size=2,
            global_blocks=2,
            environment_blocks=1,
            genotype_blocks=1,
            condition_blocks=1,
        ),
    )

    output = method_dir / "csig_mmd_context_matched_blocked"
    per_unit = pl.read_csv(output / "csig_mmd_per_unit.csv")
    blocks = pl.read_csv(output / "csig_mmd_blocks.csv")
    summary = pl.read_csv(output / "csig_mmd_summary.csv")
    assert per_unit["unit_scope"].to_list() == [expected_scope] * len(expected_units)
    assert per_unit["unit_id"].to_list() == expected_units
    assert blocks["unit_id"].unique().sort().to_list() == expected_units
    assert summary["view"].to_list() == ["context_matched_blocked"]
    assert summary["estimator"].to_list() == ["blocked"]
    assert summary["mode"].to_list() == ["no_context"]
    assert len(capture) == blocks.height

    sig_output = method_dir / "sig_mmd_context_matched_blocked"
    sig_blocks = pl.read_csv(sig_output / "sig_mmd_blocks.csv")
    sig_per_unit = pl.read_csv(sig_output / "sig_mmd_per_unit.csv")
    sig_members = pl.read_csv(sig_output / "sig_mmd_unit_members.csv")
    sig_summary = pl.read_csv(sig_output / "sig_mmd_summary.csv")
    assert sig_blocks["score"].to_list() == [0.5] * blocks.height
    assert sig_blocks.drop("score").equals(
        blocks.drop("source", "score", "mean_w_oracle", "mean_w_model")
    )
    assert sig_per_unit["unit_id"].to_list() == expected_units
    assert "source" not in sig_members.columns
    assert (
        sig_members.height == pl.read_csv(output / "csig_mmd_unit_members.csv").height
    )
    assert sig_summary["estimator"].to_list() == ["blocked"]
    assert sig_summary["mean"].to_list() == [0.5]


def test_outputs_include_required_columns_and_global_c_squared(
    monkeypatch, tmp_path
) -> None:
    capture: list[dict[str, Any]] = []
    method_dir, artifact = _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="max_height",
        use_temperature=False,
        use_marker=False,
        capture=capture,
    )

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(method_dir),
        mahalanobis_artifact=str(tmp_path / "artifact"),
        options=BlockedScoringOptions(n_samples=4, block_size=2, global_blocks=3),
        censoring=CensoringParameters(alpha=0.5, beta=3.0),
    )

    output = method_dir / "csig_mmd_context_matched_blocked"
    blocks = pl.read_csv(output / "csig_mmd_blocks.csv")
    per_unit = pl.read_csv(output / "csig_mmd_per_unit.csv")
    members = pl.read_csv(output / "csig_mmd_unit_members.csv")
    summary = pl.read_csv(output / "csig_mmd_summary.csv")

    assert {
        "source",
        "mode",
        "unit_scope",
        "unit_id",
        "block_index",
        "block_size",
        "model_sample_size",
        "oracle_sample_size",
        "score",
        "mean_w_oracle",
        "mean_w_model",
    } <= set(blocks.columns)
    assert {
        "source",
        "mode",
        "unit_scope",
        "unit_id",
        "n_members",
        "n_blocks",
        "block_size",
        "full_model_pool_size",
        "full_oracle_pool_size",
        "model_sample_size",
        "oracle_sample_size",
        "score",
        "score_std",
        "score_se",
        "mean_w_oracle",
        "mean_w_model",
        "mean_n_context",
        "mean_ess",
        "mean_post_peak_mass",
    } <= set(per_unit.columns)
    assert {
        "source",
        "mode",
        "unit_scope",
        "unit_id",
        "genotype_id",
        "yearsite_uid",
    } <= set(members.columns)
    assert {
        "source",
        "view",
        "estimator",
        "mode",
        "block_size",
        "global_blocks",
        "environment_blocks",
        "genotype_blocks",
        "condition_blocks",
        "seed",
        "requested_n_samples",
        "draw_count",
        "model_draw_count",
        "oracle_draw_count",
        "n_units",
        "n_blocks",
        "alpha",
        "beta",
        "c_squared",
        "mean",
        "median",
        "std",
        "p95",
        "mean_w_oracle",
        "mean_w_model",
        "mean_n_context",
        "mean_ess",
        "mean_post_peak_mass",
    } <= set(summary.columns)

    expected_c = float(np.quantile(artifact.mahalanobis_distances, 0.5))
    assert summary["alpha"].to_list() == [0.5]
    assert summary["beta"].to_list() == [3.0]
    assert summary["c_squared"][0] == pytest.approx(expected_c)
    assert {call["c_squared"] for call in capture} == {expected_c}
    assert summary["std"].to_list() == [0.0]
    assert summary["draw_count"].to_list() == summary["model_draw_count"].to_list()


def test_censoring_sweep_writes_one_output_per_rule_from_one_score_call(
    monkeypatch, tmp_path
) -> None:
    capture: list[dict[str, Any]] = []
    method_dir, artifact = _install_fake_world(
        monkeypatch, tmp_path, method_name="max_height", capture=capture
    )
    options = BlockedScoringOptions(n_samples=4, block_size=2, global_blocks=3)
    main_rule = CensoringParameters(alpha=0.5, beta=3.0)

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(method_dir),
        mahalanobis_artifact=str(tmp_path / "artifact"),
        options=options,
        censoring=main_rule,
        csig_output_folder=str(tmp_path / "without_sweep"),
    )
    calls_without_sweep = len(capture)
    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(method_dir),
        mahalanobis_artifact=str(tmp_path / "artifact"),
        options=options,
        censoring=main_rule,
        censoring_sweep=CensoringSweep(alphas=(0.25, 0.75), betas=(1.2,)),
        csig_output_folder=str(tmp_path / "with_sweep"),
    )

    # One score call (one Gram pass) per block, with or without the sweep.
    assert len(capture) == 2 * calls_without_sweep
    for name in ("csig_mmd_blocks.csv", "csig_mmd_per_unit.csv"):
        assert pl.read_csv(tmp_path / "with_sweep" / name).equals(
            pl.read_csv(tmp_path / "without_sweep" / name)
        )
    distances = np.asarray(artifact.mahalanobis_distances)
    for alpha in (0.25, 0.75):
        summary = pl.read_csv(
            tmp_path
            / "with_sweep"
            / "sweep"
            / f"alpha={alpha:.4f}_beta=1.2000"
            / "csig_mmd_summary.csv"
        )
        assert summary["alpha"].to_list() == [alpha]
        assert summary["beta"].to_list() == [1.2]
        assert summary["c_squared"][0] == pytest.approx(
            float(np.quantile(distances, alpha))
        )
        assert summary["mean"][0] == pytest.approx(10 * alpha + 1.2)
        assert summary["mean_w_oracle"][0] == pytest.approx(alpha)
    assert len(list((tmp_path / "with_sweep" / "sweep").iterdir())) == 2


def test_censoring_sweep_requires_artifact(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --mahalanobis-artifact"):
        blocked.sig_mmd_context_matched_blocked_analysis(
            method_dir=str(tmp_path / "max_height"),
            censoring_sweep=CensoringSweep(alphas=(0.9,)),
        )


def test_dataset_artifact_model_orders_are_mapped_by_condition_key_and_distances_pair(
    monkeypatch, tmp_path
) -> None:
    dataset_order = [("g2", "e1"), ("g2", "e2"), ("g1", "e1"), ("g1", "e2")]
    loader_order = [("g1", "e2"), ("g2", "e1"), ("g1", "e1"), ("g2", "e2")]
    artifact_order = [("g1", "e2"), ("g2", "e2"), ("g1", "e1"), ("g2", "e1")]
    capture: list[dict[str, Any]] = []
    method_dir, _ = _install_fake_world(
        monkeypatch,
        tmp_path,
        use_temperature=True,
        use_marker=True,
        dataset_key_order=dataset_order,
        loader_key_order=loader_order,
        artifact_key_order=artifact_order,
        capture=capture,
    )

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(method_dir),
        mahalanobis_artifact=str(tmp_path / "artifact"),
        options=BlockedScoringOptions(
            n_samples=4, block_size=1, condition_blocks=1, seed=7
        ),
    )

    dataset_index_by_unit = {
        f"{gid}::{ys}": index for index, (gid, ys) in enumerate(dataset_order)
    }
    seen: dict[str, int] = {}
    for call in capture:
        unit_id = str(call["label"]).split("unit_id='")[1].split("'")[0]
        raw_oracle = float(call["raw_oracle"][0])
        distance = float(call["distances"][0])
        sampled_dataset_index = int(raw_oracle // 100)
        sampled_draw = int(raw_oracle % 100)
        assert int(distance // 1000) == sampled_dataset_index
        assert int(distance % 1000) == sampled_draw
        seen[unit_id] = sampled_dataset_index

    assert seen == dataset_index_by_unit


def test_random_context_reconstruction_changes_diagnostics(
    monkeypatch, tmp_path
) -> None:
    capture: list[dict[str, Any]] = []
    method_dir, _ = _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="random_context",
        use_temperature=False,
        use_marker=False,
        context_indices=[[0, 1], [0], [1], [0, 2]],
        capture=capture,
    )

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(method_dir),
        mahalanobis_artifact=str(tmp_path / "artifact"),
        options=BlockedScoringOptions(
            n_samples=4, block_size=2, global_blocks=2, seed=0
        ),
    )

    summary = pl.read_csv(
        method_dir / "csig_mmd_context_matched_blocked" / "csig_mmd_summary.csv"
    )
    assert summary["mode"].to_list() == ["random_context"]
    assert summary["mean_n_context"][0] > 0.0
    assert summary["mean_ess"][0] < 4.0
    assert len(capture) == 2


def test_no_context_reports_empty_context_uniform_diagnostics(
    monkeypatch, tmp_path
) -> None:
    method_dir, _ = _install_fake_world(monkeypatch, tmp_path, capture=[])

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(method_dir),
        mahalanobis_artifact=str(tmp_path / "artifact"),
        options=BlockedScoringOptions(n_samples=4, block_size=2, global_blocks=1),
    )

    per_unit = pl.read_csv(
        method_dir / "csig_mmd_context_matched_blocked" / "csig_mmd_per_unit.csv"
    )
    assert per_unit["mean_n_context"].to_list() == [0.0]
    assert per_unit["mean_post_peak_mass"].to_list() == [0.0]
    assert per_unit["mean_ess"].to_list() == [4.0]


@pytest.mark.parametrize("case", ["grid", "missing-condition"])
def test_early_validation_errors_do_not_create_output(
    monkeypatch, tmp_path, case
) -> None:
    if case == "grid":
        method_dir, _ = _install_fake_world(
            monkeypatch,
            tmp_path,
            prediction_day_axis=np.array([0.0, 1.0, 3.0]),
            capture=[],
        )
        match = "prediction_day_axis differs"
    else:
        method_dir, _ = _install_fake_world(
            monkeypatch, tmp_path, artifact_key_order=_LEX_KEYS[:-1], capture=[]
        )
        match = "does not contain condition"

    with pytest.raises(ValueError, match=match):
        blocked.sig_mmd_context_matched_blocked_analysis(
            method_dir=str(method_dir),
            mahalanobis_artifact=str(tmp_path / "artifact"),
            options=BlockedScoringOptions(n_samples=4, block_size=2),
            output_folder=str(tmp_path / "out"),
            csig_output_folder=str(tmp_path / "csig-out"),
        )

    assert not list((tmp_path / "out").glob("*.csv"))
    assert not (tmp_path / "csig-out").exists()
