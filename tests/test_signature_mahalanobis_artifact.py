from __future__ import annotations

import json
import types
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
import torch

from npnf.data.configs.datasets.synthetic_test_sets import design_splits
from npnf.metrics import csig_mmd as csig_mmd_metrics
from npnf.metrics.csig_mmd import (
    CensoringParameters,
    censored_mmd_squared,
    load_censoring_reference,
    signature_kernel_gram,
)
from npnf.metrics.mahalanobis_artifact import (
    CONDITION_INDICES_FILE,
    HAS_LODGED_FILE,
    MRCD_FIT_CONDITION_INDICES_FILE,
    MRCD_FIT_DRAW_INDICES_FILE,
    MRCD_FIT_SUPPORT_FILE,
    NORMALIZED_SIGNATURES_FILE,
    ORACLE_TRAJECTORIES_ON_GRID_FILE,
    SignatureMahalanobisArtifact,
    censoring_threshold,
    load_metadata,
    load_signature_mahalanobis_artifact,
    normalize_synthetic_dataset_selectors,
    synthetic_dataset_artifact_key,
    synthetic_dataset_artifact_path,
)
from npnf.metrics.sig_mmd import normalized_sig_mmd
from npnf.metrics.signature import SYNTHETIC_SIGNATURE_HEIGHT_SCALE, to_metric_paths
from npnf.scripts.metrics import (
    precompute_sig_mahalanobis,
    score_sig_mahalanobis,
    sig_mmd,
)
from npnf.scripts.paper import signature_mahalanobis_separation as separation


class _FakeMetadata:
    def __init__(self, data: dict[str, list[str]]):
        self._data = data

    def flatten(self) -> _FakeMetadata:
        return self

    def __getitem__(self, key: str) -> list[str]:
        return self._data[key]


class _FakeDataset:
    n_lodging_draws = 2
    lodging_weibull_scale = 1.1
    scale_noise = 0.05
    lodging_seed = 42

    def __init__(
        self,
        genotype_ids: list[str] | None = None,
        yearsite_uids: list[str] | None = None,
        offset: float = 0.0,
        height_scale: float = 10.0,
        height_days: torch.Tensor | None = None,
    ):
        self._genotype_ids = genotype_ids or ["g1", "g2"]
        self._yearsite_uids = yearsite_uids or ["e1", "e2"]
        self._offset = offset
        self._height_scale = height_scale
        self._height_days = (
            torch.tensor([0.0, 1.0, 2.0]) if height_days is None else height_days
        )

    @property
    def samples(self):
        msg = "precompute must use lazy dataset helpers, not samples"
        raise AssertionError(msg)

    def get_condition_metadata(self) -> dict[str, list[str]]:
        return {"genotype_id": self._genotype_ids, "yearsite_uid": self._yearsite_uids}

    def get_height_days_all(self) -> torch.Tensor:
        return self._height_days

    def get_height_scale(self) -> torch.Tensor:
        return torch.tensor(self._height_scale)

    def get_lodged_chunk(self, indices, draw_indices):
        indices = list(indices)
        draw_indices = list(draw_indices)
        rows = []
        lodged_rows = []
        for index in range(len(self._genotype_ids)):
            base = self._offset + 1.0 + 3.0 * index
            rows.append(
                [[base, base + 1.0, base + 2.0], [base + 0.5, base + 1.5, base + 2.5]]
            )
            lodged_rows.append([index % 2 == 1, index % 2 == 0])
        base_tensor = torch.tensor(rows)
        lodged = torch.tensor(lodged_rows)
        return {
            "height_values_all_nonoise": base_tensor[indices][:, draw_indices],
            "height_days_all": self.get_height_days_all(),
            "has_lodged": lodged[indices][:, draw_indices],
            "height_values_all_nonoise_nolodge": base_tensor[indices, 0],
            "genotype_id": [self._genotype_ids[index] for index in indices],
            "yearsite_uid": [self._yearsite_uids[index] for index in indices],
        }


class _FakeBatchLoader:
    def __init__(self, method_dir: Path, *, load_predictions: bool = False):
        self.method_dir = Path(method_dir)
        self.load_predictions = load_predictions
        self.batch = {
            "data": _FakeMetadata(
                {"genotype_id": ["g1", "g2"], "yearsite_uid": ["e1", "e2"]}
            ),
            "grid_points": {"X": torch.tensor([[0.0], [1.0], [2.0]])},
        }

    def __len__(self) -> int:
        return 1

    def _predictions(self):
        grid = torch.tensor(
            [[[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], [[4.0, 5.0, 6.0], [4.5, 5.5, 6.5]]]
        ).unsqueeze(-1)
        return types.SimpleNamespace(
            select=lambda key: types.SimpleNamespace(flatten=lambda: {"grid": grid})
        )

    def load_batch(self, index: int):
        assert index == 0
        if self.load_predictions:
            return self._predictions(), self.batch
        return self.batch

    def __iter__(self):
        yield self.load_batch(0)

    @property
    def config(self) -> dict | None:
        config_path = self.method_dir / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return None


class _FakeMRCD:
    def __init__(self, support_fraction: float, random_state: int):
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.max_condition_number = 50.0

    def fit(self, x: np.ndarray) -> None:
        self.location_ = x.mean(axis=0)
        self.precision_ = np.eye(x.shape[1])
        self.support_ = np.arange(len(x)) % 2 == 0
        self.regularization_ = 0.25
        self.condition_number_ = 4.0
        self.standardized_condition_number_ = 3.0


def _fake_normalized_signatures(
    paths: np.ndarray, depth: int, self_kernel_batch_size: int
) -> np.ndarray:
    del depth, self_kernel_batch_size
    return np.stack([paths.sum(axis=(1, 2)), paths.mean(axis=(1, 2))], axis=1)


def _skip_if_cuda_tensor_would_fail_pysiglib() -> None:
    if not torch.cuda.is_available():
        return
    from pysiglib.data_handlers import BUILT_WITH_CUDA

    if not BUILT_WITH_CUDA:
        pytest.skip("pySigLib was built without CUDA support in this environment")


def _example_metric_paths(sample_count: int, offset: float) -> torch.Tensor:
    heights = torch.arange(sample_count * 3, dtype=torch.float32).reshape(
        sample_count, 3
    )
    return to_metric_paths(
        heights + offset, 20.0, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    )


def _legacy_alias_to_dataloader() -> dict[str, str]:
    return {split.alias: split.dataloader_name for split in design_splits("legacy")}


def test_synthetic_dataset_selector_helpers_are_canonical_and_strict(tmp_path: Path):
    assert normalize_synthetic_dataset_selectors(None) == (
        "plot",
        "genotype",
        "environment",
        "unseen",
    )
    assert normalize_synthetic_dataset_selectors(["unseen", "plot"]) == (
        "plot",
        "unseen",
    )
    assert normalize_synthetic_dataset_selectors(["synth_test_plot_dataloaders"]) == (
        "plot",
    )
    assert synthetic_dataset_artifact_key(None) == "all"
    assert synthetic_dataset_artifact_key(["plot"]) == "synth_test_plot_dataloaders"
    assert synthetic_dataset_artifact_key(["unseen", "plot"]) == "plot_unseen"
    assert synthetic_dataset_artifact_path(["plot"], base_path=tmp_path) == (
        tmp_path / "synth_test_plot_dataloaders" / "oracle_signature_mahalanobis"
    )

    for selector in ["site", "year", "synth_test_plot_no_lodging_dataloaders"]:
        with pytest.raises(ValueError, match=selector):
            normalize_synthetic_dataset_selectors([selector])
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_synthetic_dataset_selectors(["plot", "synth_test_plot_dataloaders"])


def test_legacy_selectors_are_the_default_and_shifted_keys_differ(tmp_path: Path):
    assert normalize_synthetic_dataset_selectors(None) == (
        "plot",
        "genotype",
        "environment",
        "unseen",
    )
    assert synthetic_dataset_artifact_key(None) == "all"
    shifted = {"test_set": "shifted"}
    assert normalize_synthetic_dataset_selectors(None, **shifted) == (
        "seen",
        "geno",
        "env",
        "unseen",
    )
    assert synthetic_dataset_artifact_key(None, **shifted) == "shifted"
    assert (
        synthetic_dataset_artifact_key(["unseen", "seen"], **shifted)
        == "shifted_seen_unseen"
    )
    assert synthetic_dataset_artifact_key(["unseen"], **shifted) == (
        "synth_shifted_unseen_dataloaders"
    )
    # A legacy alias is not a shifted split, even when the names agree.
    with pytest.raises(ValueError, match="plot"):
        normalize_synthetic_dataset_selectors(["plot"], **shifted)


def _manual_censored_mmd_squared(
    paths_x: torch.Tensor,
    paths_y: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
    height_scale: float,
    t_norm: torch.Tensor,
) -> float:
    m = paths_x.shape[0]
    n = paths_y.shape[0]
    zero_path = to_metric_paths(
        torch.zeros(1, t_norm.shape[0], dtype=torch.float32), height_scale, t_norm
    )
    if torch.cuda.is_available():
        paths_x = paths_x.cuda()
        paths_y = paths_y.cuda()
        zero_path = zero_path.cuda()
        w = w.cuda()
        v = v.cuda()

    stacked = torch.cat([paths_x, paths_y, zero_path], dim=0)
    gram = signature_kernel_gram(stacked, stacked)
    diag = gram.diagonal()
    gram = gram / torch.sqrt(torch.outer(diag, diag))

    k_xx = gram[:m, :m]
    k_yy = gram[m : m + n, m : m + n]
    k_xy = gram[:m, m : m + n]
    k_x0 = gram[:m, -1]
    k_y0 = gram[m : m + n, -1]
    k_00 = gram[-1, -1]

    def xx(i: int, j: int) -> torch.Tensor:
        return (
            w[i] * w[j] * k_xx[i, j]
            + w[i] * (1 - w[j]) * k_x0[i]
            + (1 - w[i]) * w[j] * k_x0[j]
            + (1 - w[i]) * (1 - w[j]) * k_00
        )

    def yy(i: int, j: int) -> torch.Tensor:
        return (
            v[i] * v[j] * k_yy[i, j]
            + v[i] * (1 - v[j]) * k_y0[i]
            + (1 - v[i]) * v[j] * k_y0[j]
            + (1 - v[i]) * (1 - v[j]) * k_00
        )

    def xy(i: int, j: int) -> torch.Tensor:
        return (
            w[i] * v[j] * k_xy[i, j]
            + w[i] * (1 - v[j]) * k_x0[i]
            + (1 - w[i]) * v[j] * k_y0[j]
            + (1 - w[i]) * (1 - v[j]) * k_00
        )

    if m == 1:
        pp = xx(0, 0)
    else:
        pp = sum(xx(i, j) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    if n == 1:
        qq = yy(0, 0)
    else:
        qq = sum(yy(i, j) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    pq = sum(xy(i, j) for i in range(m) for j in range(n)) / (m * n)
    return float(pp - 2 * pq + qq)


@pytest.fixture
def signature_artifact(tmp_path, monkeypatch) -> Path:
    method_dir = tmp_path / "method"
    method_dir.mkdir()
    output = tmp_path / "oracle_signature_mahalanobis"

    class _FakeOracleCfg:
        __name__ = "FakeOracleCfg"

    monkeypatch.setattr(precompute_sig_mahalanobis, "BatchLoader", _FakeBatchLoader)
    monkeypatch.setattr(
        precompute_sig_mahalanobis,
        "get_oracle_config_for_method_dir",
        lambda method_path: (_FakeOracleCfg, "synth_test_plot_dataloaders"),
    )
    monkeypatch.setattr(
        precompute_sig_mahalanobis, "instantiate", lambda cfg: _FakeDataset()
    )
    monkeypatch.setattr(
        precompute_sig_mahalanobis,
        "compute_normalized_truncated_sigs",
        _fake_normalized_signatures,
    )
    monkeypatch.setattr(precompute_sig_mahalanobis, "MRCD", _FakeMRCD)

    precompute_sig_mahalanobis.precompute_training_reference(
        method_dir=str(method_dir),
        support_fraction=0.8,
        mrcd_fit_size=4,
        self_kernel_batch_size=8,
        output=str(output),
    )
    return output


def _patch_precompute_for_union(
    monkeypatch: pytest.MonkeyPatch, datasets_by_alias: dict[str, _FakeDataset]
) -> list[str]:
    class _FakeOracleCfg:
        def __init__(self, alias: str):
            self.alias = alias
            self.__name__ = f"Fake{alias.title()}Cfg"

    cfg_by_dataloader = {
        dataloader: _FakeOracleCfg(alias)
        for alias, dataloader in _legacy_alias_to_dataloader().items()
    }
    instantiated_aliases: list[str] = []

    def fake_get_config(dataloader_name: str):
        return cfg_by_dataloader[dataloader_name]

    def fake_instantiate(cfg: _FakeOracleCfg):
        # The training reference instantiates TrainConfig_512k, which has no alias.
        alias = getattr(cfg, "alias", "train")
        instantiated_aliases.append(alias)
        return datasets_by_alias[alias]

    monkeypatch.setattr(precompute_sig_mahalanobis, "BatchLoader", _FakeBatchLoader)

    def fake_method_config(method_path: Path):
        return (
            cfg_by_dataloader["synth_test_plot_dataloaders"],
            "synth_test_plot_dataloaders",
        )

    monkeypatch.setattr(
        precompute_sig_mahalanobis,
        "get_oracle_config_for_method_dir",
        fake_method_config,
    )
    monkeypatch.setattr(
        precompute_sig_mahalanobis,
        "get_oracle_config_for_dataloader_name",
        fake_get_config,
    )
    monkeypatch.setattr(precompute_sig_mahalanobis, "instantiate", fake_instantiate)
    monkeypatch.setattr(
        precompute_sig_mahalanobis,
        "compute_normalized_truncated_sigs",
        _fake_normalized_signatures,
    )
    monkeypatch.setattr(precompute_sig_mahalanobis, "MRCD", _FakeMRCD)
    return instantiated_aliases


def test_precompute_defaults_to_primary_dataset_union(
    signature_artifact: Path, monkeypatch, tmp_path: Path
):
    output = tmp_path / "union" / "oracle_signature_mahalanobis"
    datasets_by_alias = {
        "plot": _FakeDataset(["g-plot"], ["e-train"], offset=0.0, height_scale=10.0),
        "genotype": _FakeDataset(
            ["g-genotype"], ["e-train"], offset=10.0, height_scale=20.0
        ),
        "environment": _FakeDataset(
            ["g-plot"], ["e-env"], offset=20.0, height_scale=30.0
        ),
        "unseen": _FakeDataset(["g-unseen"], ["e-env"], offset=30.0, height_scale=40.0),
    }
    instantiated_aliases = _patch_precompute_for_union(monkeypatch, datasets_by_alias)
    scale_calls: list[float] = []
    original_to_metric_paths = precompute_sig_mahalanobis.to_metric_paths

    def recording_to_metric_paths(
        heights: torch.Tensor,
        height_scale: float,
        t_norm: torch.Tensor,
        *,
        lead_lag: bool = True,
    ) -> torch.Tensor:
        scale_calls.append(height_scale)
        return original_to_metric_paths(
            heights, height_scale, t_norm, lead_lag=lead_lag
        )

    monkeypatch.setattr(
        precompute_sig_mahalanobis, "to_metric_paths", recording_to_metric_paths
    )

    precompute_sig_mahalanobis.precompute_sig_mahalanobis(
        method_dir=str(tmp_path / "method"),
        oracle_draws=64,
        self_kernel_batch_size=8,
        output=str(output),
        reference_artifact=str(signature_artifact),
    )

    artifact = load_signature_mahalanobis_artifact(output)
    assert instantiated_aliases == ["plot", "genotype", "environment", "unseen"]
    assert artifact.metadata["artifact_key"] == "all"
    assert artifact.metadata["dataset_aliases"] == [
        "plot",
        "genotype",
        "environment",
        "unseen",
    ]
    assert artifact.metadata["dataloader_names"] == [
        "synth_test_plot_dataloaders",
        "synth_test_genotype_dataloaders",
        "synth_test_environment_dataloaders",
        "synth_test_unseen_dataloaders",
    ]
    assert artifact.metadata["n_conditions"] == 4
    assert artifact.metadata["n_total_draws"] == 8
    assert artifact.metadata["height_scale"] == SYNTHETIC_SIGNATURE_HEIGHT_SCALE
    assert scale_calls
    assert set(scale_calls) == {SYNTHETIC_SIGNATURE_HEIGHT_SCALE}
    assert artifact.condition_genotype_ids.tolist() == [
        "g-plot",
        "g-genotype",
        "g-plot",
        "g-unseen",
    ]
    assert artifact.condition_yearsite_uids.tolist() == [
        "e-train",
        "e-train",
        "e-env",
        "e-env",
    ]
    assert artifact.condition_indices.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    # Test artifacts reuse the training fit and write no fit support.
    assert artifact.mrcd_fit_condition_indices is None


def test_precompute_rejects_different_union_day_axes(monkeypatch, tmp_path: Path):
    output = tmp_path / "different-days" / "oracle_signature_mahalanobis"
    datasets_by_alias = {
        "plot": _FakeDataset(["g-plot"], ["e1"]),
        "genotype": _FakeDataset(
            ["g-genotype"], ["e1"], height_days=torch.tensor([0.0, 1.0, 3.0])
        ),
    }
    _patch_precompute_for_union(monkeypatch, datasets_by_alias)

    with pytest.raises(ValueError, match="height day axis"):
        precompute_sig_mahalanobis.precompute_sig_mahalanobis(
            method_dir=str(tmp_path / "method"),
            oracle_draws=64,
            self_kernel_batch_size=8,
            output=str(output),
            datasets=["plot", "genotype"],
            reference_artifact=str(tmp_path / "reference"),
        )

    assert not output.exists()


def test_precompute_rejects_duplicate_union_conditions(monkeypatch, tmp_path: Path):
    output = tmp_path / "duplicate" / "oracle_signature_mahalanobis"
    datasets_by_alias = {
        alias: _FakeDataset(["g1"], ["e1"], offset=float(index))
        for index, alias in enumerate(_legacy_alias_to_dataloader())
    }
    _patch_precompute_for_union(monkeypatch, datasets_by_alias)

    with pytest.raises(ValueError, match="Duplicate oracle condition"):
        precompute_sig_mahalanobis.precompute_sig_mahalanobis(
            method_dir=str(tmp_path / "method"),
            oracle_draws=64,
            self_kernel_batch_size=8,
            output=str(output),
            datasets=["plot", "genotype"],
            reference_artifact=str(tmp_path / "reference"),
        )

    assert not output.exists()


def test_csig_mmd_all_one_weights_matches_normalized_sig_mmd():
    _skip_if_cuda_tensor_would_fail_pysiglib()
    paths_x = _example_metric_paths(3, 1.0)
    paths_y = _example_metric_paths(4, 5.0)
    weights_x = torch.ones(paths_x.shape[0])
    weights_y = torch.ones(paths_y.shape[0])
    t_norm = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    score = censored_mmd_squared(paths_x, paths_y, weights_x, weights_y, 20.0, t_norm)

    assert score == pytest.approx(normalized_sig_mmd(paths_x, paths_y), abs=1e-6)


def test_csig_mmd_all_zero_weights_collapses_to_pivot():
    _skip_if_cuda_tensor_would_fail_pysiglib()
    paths_x = _example_metric_paths(2, 1.0)
    paths_y = _example_metric_paths(3, 5.0)
    weights_x = torch.zeros(paths_x.shape[0])
    weights_y = torch.zeros(paths_y.shape[0])
    t_norm = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    score = censored_mmd_squared(paths_x, paths_y, weights_x, weights_y, 20.0, t_norm)

    assert score == pytest.approx(0.0, abs=1e-6)


def test_csig_mmd_mixed_weights_matches_manual_unbiased_formula():
    _skip_if_cuda_tensor_would_fail_pysiglib()
    paths_x = _example_metric_paths(3, 1.0)
    paths_y = _example_metric_paths(2, 5.0)
    weights_x = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float32)
    weights_y = torch.tensor([0.25, 0.75], dtype=torch.float32)
    t_norm = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    score = censored_mmd_squared(paths_x, paths_y, weights_x, weights_y, 20.0, t_norm)
    expected = _manual_censored_mmd_squared(
        paths_x, paths_y, weights_x, weights_y, 20.0, t_norm
    )

    assert score == pytest.approx(expected, abs=1e-6)


def test_precompute_writes_directory_artifact(signature_artifact: Path):
    artifact = load_signature_mahalanobis_artifact(signature_artifact)

    assert artifact.metadata["draw_count"] == 2
    assert artifact.metadata["height_scale"] == SYNTHETIC_SIGNATURE_HEIGHT_SCALE
    assert artifact.metadata["signature_dim"] == 2
    assert artifact.metadata["mrcd_fit_n_seen"] == 4
    assert artifact.metadata["mrcd_fit_n_samples"] == 4
    assert artifact.metadata["mrcd_fit_n_support"] == 2
    assert "alpha" not in artifact.metadata
    assert "beta" not in artifact.metadata
    assert "c_squared" not in artifact.metadata
    assert not (signature_artifact.with_suffix(".npz")).exists()
    assert np.load(
        signature_artifact / NORMALIZED_SIGNATURES_FILE, mmap_mode="r"
    ).shape == (4, 2)
    assert np.load(
        signature_artifact / ORACLE_TRAJECTORIES_ON_GRID_FILE, mmap_mode="r"
    ).shape == (2, 2, 3)
    assert np.load(signature_artifact / HAS_LODGED_FILE, mmap_mode="r").tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert np.load(
        signature_artifact / CONDITION_INDICES_FILE, mmap_mode="r"
    ).tolist() == [0, 0, 1, 1]
    assert np.load(
        signature_artifact / MRCD_FIT_CONDITION_INDICES_FILE, mmap_mode="r"
    ).tolist() == [0, 0, 1, 1]
    assert np.load(
        signature_artifact / MRCD_FIT_DRAW_INDICES_FILE, mmap_mode="r"
    ).tolist() == [0, 1, 0, 1]
    assert np.load(
        signature_artifact / MRCD_FIT_SUPPORT_FILE, mmap_mode="r"
    ).tolist() == [True, False, True, False]
    assert artifact.mrcd_fit_condition_indices is not None
    assert artifact.mrcd_fit_draw_indices is not None
    assert artifact.mrcd_fit_support is not None
    assert artifact.mrcd_fit_condition_indices.tolist() == [0, 0, 1, 1]
    assert artifact.mrcd_fit_draw_indices.tolist() == [0, 1, 0, 1]
    assert artifact.mrcd_fit_support.tolist() == [True, False, True, False]

    normalized_signatures = np.load(
        signature_artifact / NORMALIZED_SIGNATURES_FILE, mmap_mode="r"
    )
    expected_location = normalized_signatures.mean(axis=0)
    expected_distances = np.sqrt(
        ((normalized_signatures - expected_location) ** 2).sum(axis=1)
    )
    assert np.allclose(artifact.mahalanobis_distances, expected_distances)


def test_artifact_loader_accepts_old_artifacts_without_support_files(
    signature_artifact: Path,
):
    (signature_artifact / MRCD_FIT_CONDITION_INDICES_FILE).unlink()
    (signature_artifact / MRCD_FIT_DRAW_INDICES_FILE).unlink()
    (signature_artifact / MRCD_FIT_SUPPORT_FILE).unlink()

    artifact = load_signature_mahalanobis_artifact(signature_artifact)

    assert artifact.mrcd_fit_condition_indices is None
    assert artifact.mrcd_fit_draw_indices is None
    assert artifact.mrcd_fit_support is None


def test_score_sig_mahalanobis_reads_directory_artifact(
    signature_artifact: Path, tmp_path: Path
):
    output = tmp_path / "score"

    score_sig_mahalanobis.score_sig_mahalanobis(
        precomputed=str(signature_artifact), output_folder=str(output)
    )

    detail = pl.read_csv(output / "mahalanobis_vs_lodging.csv")
    assert detail["genotype_id"].to_list() == ["g1", "g1", "g2", "g2"]
    assert detail["yearsite_uid"].to_list() == ["e1", "e1", "e2", "e2"]
    assert "in_mrcd_fit" in detail.columns
    assert "in_mrcd_support" in detail.columns
    assert detail["in_mrcd_fit"].to_list() == [True, True, True, True]
    assert detail["in_mrcd_support"].to_list() == [True, False, True, False]

    summary = pl.read_csv(output / "discrimination_summary.csv")
    assert summary["n_total"].to_list() == [4]
    assert summary["n_lodged"].to_list() == [2]

    support_summary = pl.read_csv(output / "mrcd_support_summary.csv")
    assert support_summary["n_fit_samples"].to_list() == [4]
    assert support_summary["n_support"].to_list() == [2]
    assert support_summary["n_non_support"].to_list() == [2]
    assert support_summary["lodged_rate_fit"][0] == pytest.approx(0.5)
    assert support_summary["lodged_rate_support"][0] == pytest.approx(0.5)
    assert support_summary["lodged_rate_non_support"][0] == pytest.approx(0.5)
    assert (output / "mrcd_support_lodging_counts.png").is_file()
    assert (output / "mrcd_support_distance_hist.png").is_file()


class _FakeOracleCfg:
    __name__ = "FakeOracleCfg"


def _patch_condition_csig_scoring(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset: _FakeDataset | None = None,
    batch_loader: type[_FakeBatchLoader] = _FakeBatchLoader,
    csig_score: float = 1.25,
) -> list[tuple[int, ...]]:
    signature_call_shapes: list[tuple[int, ...]] = []

    def fake_normalized_signatures(
        paths: np.ndarray, depth: int, self_kernel_batch_size: int
    ) -> np.ndarray:
        signature_call_shapes.append(paths.shape)
        return _fake_normalized_signatures(paths, depth, self_kernel_batch_size)

    monkeypatch.setattr(
        sig_mmd,
        "BatchLoader",
        lambda method_path, **kwargs: batch_loader(method_path, load_predictions=True),
    )
    monkeypatch.setattr(
        sig_mmd,
        "get_oracle_config_for_method_dir",
        lambda method_path: (_FakeOracleCfg, "fake_dataloader"),
    )
    monkeypatch.setattr(sig_mmd, "instantiate", lambda cfg: dataset or _FakeDataset())
    monkeypatch.setattr(
        csig_mmd_metrics,
        "compute_normalized_truncated_sigs",
        fake_normalized_signatures,
    )
    monkeypatch.setattr(
        csig_mmd_metrics,
        "sig_and_csig_mmd_squared",
        lambda *args, sigma: (0.5, [csig_score]),
    )
    return signature_call_shapes


@pytest.mark.parametrize("explicit_output", [True, False])
def test_condition_view_csig_reads_directory_artifact(
    signature_artifact: Path, monkeypatch, tmp_path, explicit_output: bool
):
    method_dir = tmp_path / "method"
    default_output = method_dir / "csig_mmd"
    output = tmp_path / "explicit-csig_mmd" if explicit_output else default_output
    signature_call_shapes = _patch_condition_csig_scoring(monkeypatch)

    sig_mmd.sig_mmd_analysis(
        method_dir=str(method_dir),
        n_samples=2,
        mahalanobis_artifact=str(signature_artifact),
        csig_output_folder=str(output) if explicit_output else None,
    )

    per_x = pl.read_csv(output / "csig_mmd_per_x.csv")
    assert per_x.columns == [
        "source",
        "genotype_id",
        "yearsite_uid",
        "oracle_lodging_rate",
        "mean_w_oracle",
        "mean_w_model",
        "score",
    ]
    assert per_x["genotype_id"].to_list() == ["g1", "g2"]
    assert per_x["score"].to_list() == [1.25, 1.25]
    assert signature_call_shapes == [(2, 5, 3), (2, 5, 3)]
    assert (output / "csig_mmd_summary.csv").exists()
    sig_per_x = pl.read_csv(method_dir / "sig_mmd" / "sig_mmd_per_x.csv")
    assert sig_per_x["score"].to_list() == [0.5, 0.5]
    if explicit_output:
        assert not (default_output / "csig_mmd_per_x.csv").exists()


def test_condition_view_csig_nonfinite_score_fails_with_condition(
    signature_artifact: Path, monkeypatch, tmp_path
):
    output = tmp_path / "csig_mmd"
    _patch_condition_csig_scoring(monkeypatch, csig_score=float("nan"))

    with pytest.raises(ValueError, match="Non-finite CSig-MMD score") as error_info:
        sig_mmd.sig_mmd_analysis(
            method_dir=str(tmp_path / "method"),
            n_samples=2,
            mahalanobis_artifact=str(signature_artifact),
            csig_output_folder=str(output),
        )

    message = str(error_info.value)
    assert "genotype_id='g1'" in message
    assert "yearsite_uid='e1'" in message
    assert not (output / "csig_mmd_per_x.csv").exists()


def test_condition_view_csig_missing_artifact_condition_fails_clearly(
    signature_artifact: Path, monkeypatch, tmp_path
):
    class _ExtraConditionBatchLoader(_FakeBatchLoader):
        def __init__(self, method_dir: Path, *, load_predictions: bool = False):
            super().__init__(method_dir, load_predictions=load_predictions)
            self.batch["data"] = _FakeMetadata(
                {"genotype_id": ["g3"], "yearsite_uid": ["e3"]}
            )

    _patch_condition_csig_scoring(
        monkeypatch,
        dataset=_FakeDataset(["g1", "g2", "g3"], ["e1", "e2", "e3"]),
        batch_loader=_ExtraConditionBatchLoader,
    )

    with pytest.raises(ValueError, match="does not contain condition"):
        sig_mmd.sig_mmd_analysis(
            method_dir=str(tmp_path / "method"),
            n_samples=2,
            mahalanobis_artifact=str(signature_artifact),
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"oracle_self": True}, {"lead_lag": False}, {"normalize_sig_kernel": False}],
    ids=["oracle-self", "no-lead-lag", "unnormalized"],
)
def test_condition_view_csig_rejects_other_sig_modes(tmp_path, kwargs):
    with pytest.raises(ValueError, match="mahalanobis-artifact"):
        sig_mmd.sig_mmd_analysis(
            method_dir=str(tmp_path / "no_context"),
            n_samples=2,
            mahalanobis_artifact=str(tmp_path / "artifact"),
            **kwargs,
        )


def test_condition_view_csig_computes_censoring_from_scoring_args(
    signature_artifact: Path, monkeypatch, tmp_path
):
    metadata_path = signature_artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({"alpha": 0.01, "beta": 99.0, "c_squared": -123.0})
    metadata_path.write_text(json.dumps(metadata))

    output = tmp_path / "csig_mmd"
    _patch_condition_csig_scoring(monkeypatch)

    sig_mmd.sig_mmd_analysis(
        method_dir=str(tmp_path / "method"),
        n_samples=2,
        mahalanobis_artifact=str(signature_artifact),
        csig_output_folder=str(output),
        censoring=CensoringParameters(alpha=0.5, beta=3.0),
    )

    distances = load_signature_mahalanobis_artifact(
        signature_artifact
    ).mahalanobis_distances
    summary = pl.read_csv(output / "csig_mmd_summary.csv")
    assert summary["alpha"].to_list() == [0.5]
    assert summary["beta"].to_list() == [3.0]
    assert summary["c_squared"][0] == pytest.approx(float(np.quantile(distances, 0.5)))
    assert summary["c_squared"][0] != -123.0


def test_condition_view_csig_default_censoring_follows_lodged_rule(
    signature_artifact: Path, monkeypatch, tmp_path
):
    output = tmp_path / "csig_mmd"
    _patch_condition_csig_scoring(monkeypatch)

    sig_mmd.sig_mmd_analysis(
        method_dir=str(tmp_path / "method"),
        n_samples=2,
        mahalanobis_artifact=str(signature_artifact),
        csig_output_folder=str(output),
    )

    artifact = load_signature_mahalanobis_artifact(signature_artifact)
    alpha, c_squared = censoring_threshold(artifact, None, 0.975)
    summary = pl.read_csv(output / "csig_mmd_summary.csv")
    assert summary["alpha"][0] == pytest.approx(alpha)
    assert summary["c_squared"][0] == pytest.approx(c_squared)
    assert summary["beta"][0] == pytest.approx(25.6 / c_squared)


def test_separation_figure_uses_the_censoring_rule(
    signature_artifact: Path, monkeypatch, tmp_path
):
    monkeypatch.setenv("NPNF_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "signature_mahalanobis_separation.py",
            "--artifact",
            str(signature_artifact),
            "--output-dir",
            str(tmp_path / "figure"),
        ],
    )

    separation.main()

    summary = pl.read_csv(
        tmp_path / "figure" / "signature_mahalanobis_separation_summary.csv"
    )
    artifact = load_signature_mahalanobis_artifact(signature_artifact)
    alpha, c_squared = censoring_threshold(artifact, None, 0.975)
    assert summary["alpha"][0] == pytest.approx(alpha)
    assert summary["threshold"][0] == pytest.approx(c_squared)


def test_censoring_threshold_rule_keeps_lodged_recall_in_tail():
    distances = np.arange(1000, dtype=np.float64)
    namespace = types.SimpleNamespace(
        metadata={},
        mrcd_location=np.zeros(2),
        mrcd_precision=np.eye(2),
        mahalanobis_distances=distances,
        has_lodged=distances >= 800,
    )
    reference = cast(SignatureMahalanobisArtifact, namespace)

    alpha, c_squared = censoring_threshold(reference, None, 0.975)

    assert c_squared == pytest.approx(800 + 0.025 * 199)
    assert alpha == pytest.approx(0.805)
    assert np.mean(distances[distances >= 800] > c_squared) == pytest.approx(
        0.975, abs=0.005
    )
    assert censoring_threshold(reference, 0.5, 0.975) == (
        0.5,
        pytest.approx(float(np.quantile(distances, 0.5))),
    )
    namespace.has_lodged = np.zeros(1000, dtype=np.bool_)
    with pytest.raises(ValueError, match="no lodged draws"):
        censoring_threshold(reference, None, 0.975)


def test_censoring_parameters_resolve_beta_from_threshold(signature_artifact: Path):
    artifact = load_signature_mahalanobis_artifact(signature_artifact)
    rule = CensoringParameters().resolve(artifact)
    alpha, c_squared = censoring_threshold(artifact, None, 0.975)
    assert (rule.alpha, rule.c_squared) == (alpha, c_squared)
    assert rule.beta == pytest.approx(25.6 / c_squared)
    assert CensoringParameters(beta=3.0).resolve(artifact).beta == 3.0


def test_test_artifact_uses_reference_fit_and_threshold(monkeypatch, tmp_path: Path):
    reference_path = tmp_path / "train" / "oracle_signature_mahalanobis"
    datasets_by_alias = {
        "train": _FakeDataset(["g-plot", "g-genotype"], ["e-train", "e-train"]),
        "environment": _FakeDataset(["g-plot"], ["e-env"], offset=20.0),
        "unseen": _FakeDataset(["g-unseen"], ["e-env"], offset=30.0),
    }
    _patch_precompute_for_union(monkeypatch, datasets_by_alias)
    precompute_sig_mahalanobis.precompute_training_reference(
        method_dir=str(tmp_path / "method"),
        support_fraction=0.8,
        mrcd_fit_size=8,
        self_kernel_batch_size=8,
        output=str(reference_path),
    )
    test_path = tmp_path / "test" / "oracle_signature_mahalanobis"
    precompute_sig_mahalanobis.precompute_sig_mahalanobis(
        method_dir=str(tmp_path / "method"),
        oracle_draws=64,
        self_kernel_batch_size=8,
        output=str(test_path),
        datasets=["environment", "unseen"],
        reference_artifact=str(reference_path),
    )

    reference = load_signature_mahalanobis_artifact(reference_path)
    artifact = load_signature_mahalanobis_artifact(test_path)
    assert artifact.metadata["censoring_reference_artifact"] == str(
        reference_path.resolve()
    )
    np.testing.assert_array_equal(artifact.mrcd_location, reference.mrcd_location)
    np.testing.assert_array_equal(artifact.mrcd_precision, reference.mrcd_precision)
    # The threshold comes from the reference distances, not the test distances.
    assert censoring_threshold(artifact, 0.85, 0.975)[1] == pytest.approx(
        float(np.quantile(reference.mahalanobis_distances, 0.85))
    )
    assert censoring_threshold(artifact, 0.85, 0.975)[1] != pytest.approx(
        float(np.quantile(artifact.mahalanobis_distances, 0.85))
    )
    reference_distances = np.asarray(reference.mahalanobis_distances)
    lodged_distances = reference_distances[np.asarray(reference.has_lodged)]
    assert censoring_threshold(artifact, None, 0.975)[1] == pytest.approx(
        float(np.quantile(lodged_distances, 0.025))
    )
    scorer_reference = load_censoring_reference(
        test_path,
        CensoringParameters(alpha=0.85),
        prediction_day_axis=torch.from_numpy(np.asarray(artifact.prediction_day_axis)),
        metric_day_axis_norm=torch.from_numpy(
            np.asarray(artifact.metric_day_axis_norm)
        ),
        height_scale=float(artifact.metadata["height_scale"]),
    )
    assert (
        scorer_reference.main.c_squared == censoring_threshold(artifact, 0.85, 0.975)[1]
    )


@pytest.mark.parametrize(
    ("test_set", "setting", "value"),
    [
        ("lodging1_3", "lodging_weibull_scale", 1.3),
        ("lodging1_5", "lodging_weibull_scale", 1.5),
        ("noise0_02", "scale_noise", 0.02),
    ],
)
def test_setting_test_sets_change_only_their_setting(test_set, setting, value):
    from npnf.data.configs.datasets import synthetic

    splits = design_splits(test_set)
    assert [split.dataloader_name for split in splits] == [
        f"synth_test_{alias}_{test_set}_dataloaders"
        for alias in ("plot", "genotype", "environment", "unseen")
    ]
    for split in splits:
        assert split.seed_b is not None
        for config, default in (
            (split.oracle, getattr(synthetic, f"TestConfig_{split.title}")),
            (split.seed_b, getattr(synthetic, f"TestConfig_{split.title}_SeedB")),
        ):
            changed = {
                name
                for name, value in vars(config()).items()
                if value != getattr(default(), name)
            }
            assert changed == {setting}
            assert getattr(config(), setting) == value
    assert synthetic_dataset_artifact_key(None, test_set=test_set) == test_set


def test_training_reference_uses_and_records_the_lodging_scale(tmp_path, monkeypatch):
    method_dir = tmp_path / "method"
    method_dir.mkdir()
    output = tmp_path / "oracle_signature_mahalanobis"
    overrides: list[dict] = []

    def fake_instantiate(cfg, **kwargs):
        overrides.append(kwargs)
        dataset = _FakeDataset()
        dataset.lodging_weibull_scale = kwargs["lodging_weibull_scale"]
        return dataset

    monkeypatch.setattr(precompute_sig_mahalanobis, "BatchLoader", _FakeBatchLoader)
    monkeypatch.setattr(precompute_sig_mahalanobis, "instantiate", fake_instantiate)
    monkeypatch.setattr(
        precompute_sig_mahalanobis,
        "compute_normalized_truncated_sigs",
        _fake_normalized_signatures,
    )
    monkeypatch.setattr(precompute_sig_mahalanobis, "MRCD", _FakeMRCD)

    precompute_sig_mahalanobis.precompute_training_reference(
        method_dir=str(method_dir),
        support_fraction=0.8,
        mrcd_fit_size=4,
        self_kernel_batch_size=8,
        output=str(output),
        lodging_weibull_scale=1.3,
    )

    assert overrides == [{"lodging_weibull_scale": 1.3}]
    assert load_metadata(output)["lodging_weibull_scale"] == 1.3


def test_training_reference_uses_and_records_the_observation_noise(
    tmp_path, monkeypatch
):
    method_dir = tmp_path / "method"
    method_dir.mkdir()
    output = tmp_path / "oracle_signature_mahalanobis"
    overrides: list[dict] = []

    def fake_instantiate(cfg, **kwargs):
        overrides.append(kwargs)
        dataset = _FakeDataset()
        dataset.scale_noise = kwargs["scale_noise"]
        return dataset

    monkeypatch.setattr(precompute_sig_mahalanobis, "BatchLoader", _FakeBatchLoader)
    monkeypatch.setattr(precompute_sig_mahalanobis, "instantiate", fake_instantiate)
    monkeypatch.setattr(
        precompute_sig_mahalanobis,
        "compute_normalized_truncated_sigs",
        _fake_normalized_signatures,
    )
    monkeypatch.setattr(precompute_sig_mahalanobis, "MRCD", _FakeMRCD)

    precompute_sig_mahalanobis.precompute_training_reference(
        method_dir=str(method_dir),
        support_fraction=0.8,
        mrcd_fit_size=4,
        self_kernel_batch_size=8,
        output=str(output),
        scale_noise=0.02,
    )

    assert overrides == [{"scale_noise": 0.02}]
    metadata = load_metadata(output)
    assert metadata["scale_noise"] == 0.02
    assert metadata["lodging_weibull_scale"] == 1.1
