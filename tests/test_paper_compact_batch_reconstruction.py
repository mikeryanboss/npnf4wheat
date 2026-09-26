from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from npnf.data.process.collate import collate_fn_fip1_heights
from npnf.scripts.paper import (
    conditioning_sample_comparison as csc,
    lodging_probability as lp,
    prior_prediction_samples as pps,
    training_objectives as tobj,
)
from npnf.scripts.utils import synthetic_batch_reconstruction as reconstruction


class _ListLoader:
    def __init__(self, items):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def _install_loader(monkeypatch: pytest.MonkeyPatch, module, items):
    def fake_loader(_method_dir: Path, **_kwargs):
        return _ListLoader(items)

    monkeypatch.setattr(module, "BatchLoader", fake_loader)


def _full_batch() -> dict:
    x = torch.tensor([200.0, 210.0, 220.0]).reshape(1, 3, 1).expand(2, -1, -1)
    y_original = torch.tensor(
        [
            [[[1.0], [4.0], [2.0]], [[2.0], [3.0], [1.0]]],
            [[[0.5], [0.8], [0.2]], [[1.5], [1.8], [1.2]]],
        ],
        dtype=torch.float32,
    )
    return {
        "data": {
            "height": {"X": x, "Y": y_original[:, 0], "Y_original": y_original},
            "has_lodged": torch.tensor(
                [[True, False], [False, True]], dtype=torch.bool
            ),
            "genotype_id": ["G0", "G1"],
            "yearsite_uid": ["Y0", "Y1"],
        },
        "grid_points": {"X": torch.tensor([[200.0], [210.0], [220.0]])},
    }


def _compact_batch(full_batch: dict | None = None) -> dict:
    return {
        "data": {"genotype_id": ["G0", "G1"], "yearsite_uid": ["Y0", "Y1"]},
        "grid_points": {"X": torch.tensor([[200.0], [210.0], [220.0]])},
        "_full": full_batch or _full_batch(),
    }


def _predictions() -> dict:
    grid = torch.tensor(
        [
            [[[[1.0], [2.0], [1.0]], [[1.5], [2.5], [1.5]]]],
            [[[[0.2], [0.4], [0.3]], [[0.3], [0.5], [0.4]]]],
        ],
        dtype=torch.float32,
    )
    return {"grid": grid}


class _ParityDataset:
    rows = (("G0", "Y0"), ("G1", "Y1"))

    def __len__(self) -> int:
        return len(self.rows)

    def get_dataset_identity(self) -> dict:
        return {"cache_key": "paper-parity-cache", "num_conditions": len(self)}

    def get_condition_metadata(self) -> dict[str, list[str]]:
        return {
            "genotype_id": [row[0] for row in self.rows],
            "yearsite_uid": [row[1] for row in self.rows],
        }

    def __getitem__(self, index: int) -> dict:
        genotype_id, yearsite_uid = self.rows[index]
        days = torch.tensor([200.0, 210.0, 220.0])
        clean_draws = torch.tensor(
            [
                [1.0 + index, 4.0 + index, 2.0 + index],
                [2.0 + index, 3.0 + index, 1.0 + index],
            ],
            dtype=torch.float32,
        )
        return {
            "height_days": days,
            "height_days_normalized": (days - 212.5) / 151.5,
            "height_values": clean_draws[0] + 0.01,
            "height_values_nonoise": clean_draws,
            "height_lodged_mask": torch.zeros_like(clean_draws, dtype=torch.bool),
            "has_lodged": torch.tensor([index == 0, index == 1], dtype=torch.bool),
            "genotype_id": genotype_id,
            "yearsite_uid": yearsite_uid,
            "temperature_values": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        }


def _install_method_loader(monkeypatch: pytest.MonkeyPatch, module, method_items):
    def fake_loader(method_dir: Path, **_kwargs):
        return _ListLoader(method_items[Path(method_dir)])

    monkeypatch.setattr(module, "BatchLoader", fake_loader)


def _patch_reconstruction_dataset(
    monkeypatch: pytest.MonkeyPatch, dataset: _ParityDataset
) -> None:
    def fake_resolve(dataloader_name: str):
        assert dataloader_name == "parity_loader"
        return object()

    monkeypatch.setattr(
        reconstruction, "get_oracle_config_for_dataloader_name", fake_resolve
    )
    monkeypatch.setattr(reconstruction, "instantiate", lambda _config: dataset)


def _write_compact_config(method_dir: Path, dataset: _ParityDataset) -> None:
    method_dir.mkdir(parents=True)
    config = {
        "batch_schema": "compact",
        "dataloader_name": "parity_loader",
        "dataset_identity": dataset.get_dataset_identity(),
    }
    (method_dir / "config.json").write_text(json.dumps(config))


def _full_artifact_batch(dataset: _ParityDataset) -> dict:
    data = (
        collate_fn_fip1_heights([dataset[index] for index in range(len(dataset))])
        .select("height", "has_lodged", "genotype_id", "yearsite_uid")
        .cpu()
    )
    return {
        "data": data,
        "grid_points": {"X": torch.tensor([[200.0], [210.0], [220.0]])},
    }


def _compact_artifact_batch(dataset: _ParityDataset) -> dict:
    return {
        "data": {
            "genotype_id": [row[0] for row in dataset.rows],
            "yearsite_uid": [row[1] for row in dataset.rows],
        },
        "grid_points": {"X": torch.tensor([[200.0], [210.0], [220.0]])},
    }


def _assert_trajectory_data_equal(actual, expected) -> None:
    np.testing.assert_allclose(actual.x_values, expected.x_values)
    np.testing.assert_allclose(actual.sample_trajectories, expected.sample_trajectories)
    np.testing.assert_allclose(actual.mean_trajectory, expected.mean_trajectory)


def _assert_panel_equal(actual, expected) -> None:
    np.testing.assert_allclose(actual.x_values, expected.x_values)
    np.testing.assert_allclose(actual.samples, expected.samples)
    np.testing.assert_allclose(actual.mean, expected.mean)


def _patch_reconstruct(monkeypatch: pytest.MonkeyPatch, module):
    calls = []

    def fake_reconstruct(method_dir: Path, batch: dict, cache=None):
        calls.append((method_dir, cache))
        return batch.get("_full", batch)

    monkeypatch.setattr(module, "reconstruct_synthetic_batch", fake_reconstruct)
    return calls


def _fail_on_reconstruct(monkeypatch: pytest.MonkeyPatch, module) -> None:
    def fail(*_args, **_kwargs):
        pytest.fail("prediction/ID-only path should not reconstruct full data")

    monkeypatch.setattr(module, "reconstruct_synthetic_batch", fail)


def test_paper_ground_truth_paths_match_full_artifacts_after_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _ParityDataset()
    _patch_reconstruction_dataset(monkeypatch, dataset)
    full_dir = tmp_path / "full"
    compact_dir = tmp_path / "compact"
    _write_compact_config(compact_dir, dataset)
    full_batch = _full_artifact_batch(dataset)
    compact_batch = _compact_artifact_batch(dataset)

    _install_method_loader(
        monkeypatch, pps, {full_dir: [full_batch], compact_dir: [compact_batch]}
    )
    full_prior = pps._load_ground_truth_trajectories(  # noqa: SLF001
        [full_dir], num_batches=None, num_samples=10, rng=np.random.default_rng(0)
    )
    compact_prior = pps._load_ground_truth_trajectories(  # noqa: SLF001
        [compact_dir], num_batches=None, num_samples=10, rng=np.random.default_rng(0)
    )
    _assert_trajectory_data_equal(compact_prior, full_prior)

    _install_method_loader(
        monkeypatch, lp, {full_dir: [full_batch], compact_dir: [compact_batch]}
    )
    full_lodging = lp._load_ground_truth(full_dir)  # noqa: SLF001
    compact_lodging = lp._load_ground_truth(compact_dir)  # noqa: SLF001
    assert compact_lodging.labels.tolist() == full_lodging.labels.tolist()
    np.testing.assert_allclose(compact_lodging.max_heights, full_lodging.max_heights)

    _install_method_loader(
        monkeypatch,
        tobj,
        {
            full_dir: [(_predictions(), full_batch)],
            compact_dir: [(_predictions(), compact_batch)],
        },
    )
    full_sample = tobj.load_sample(full_dir, batch_index=0, sample_index=1)
    compact_sample = tobj.load_sample(compact_dir, batch_index=0, sample_index=1)
    np.testing.assert_allclose(compact_sample.grid_x, full_sample.grid_x)
    np.testing.assert_allclose(
        compact_sample.prediction_draws, full_sample.prediction_draws
    )
    np.testing.assert_allclose(compact_sample.full_x, full_sample.full_x)
    np.testing.assert_allclose(compact_sample.ground_truth, full_sample.ground_truth)
    np.testing.assert_allclose(compact_sample.context_x, full_sample.context_x)
    np.testing.assert_allclose(compact_sample.context_y, full_sample.context_y)

    _install_method_loader(
        monkeypatch, csc, {full_dir: [full_batch], compact_dir: [compact_batch]}
    )
    full_panel = csc.load_ground_truth_panel(
        full_dir,
        genotype_id="G0",
        yearsite_uid=None,
        rng=np.random.default_rng(0),
        num_samples=10,
        num_batches=None,
    )
    compact_panel = csc.load_ground_truth_panel(
        compact_dir,
        genotype_id="G0",
        yearsite_uid=None,
        rng=np.random.default_rng(0),
        num_samples=10,
        num_batches=None,
    )
    _assert_panel_equal(compact_panel, full_panel)

    full_stratified = csc.load_ground_truth_stratified(
        full_dir, [("G1", "Y1")], seed=0, num_samples=2, num_batches=None
    )
    compact_stratified = csc.load_ground_truth_stratified(
        compact_dir, [("G1", "Y1")], seed=0, num_samples=2, num_batches=None
    )
    _assert_panel_equal(compact_stratified, full_stratified)

    assert csc.select_default_ground_truth_targets(
        compact_dir, num_batches=None
    ) == csc.select_default_ground_truth_targets(full_dir, num_batches=None)


def test_prior_model_trajectories_do_not_reconstruct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_loader(monkeypatch, pps, [(_predictions(), _compact_batch())])
    _fail_on_reconstruct(monkeypatch, pps)

    data = pps._load_model_trajectories(  # noqa: SLF001
        Path("prior-method"),
        num_batches=None,
        num_samples=10,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_allclose(data.x_values, np.array([200.0, 210.0, 220.0]))


def test_lodging_predictions_do_not_reconstruct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_loader(monkeypatch, lp, [_predictions()])
    _fail_on_reconstruct(monkeypatch, lp)

    predictions = lp._load_predictions(  # noqa: SLF001
        Path("lodging-method"), relative_threshold=0.2, absolute_threshold=0.1
    )

    assert predictions.max_height.shape == (2,)


def test_conditioning_ground_truth_panel_reconstructs_only_matching_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_dir = Path("conditioning-method")
    full = _full_batch()
    nonmatching = {"data": {"genotype_id": ["G9"], "yearsite_uid": ["Y9"]}}
    _install_loader(monkeypatch, csc, [nonmatching, _compact_batch(full)])
    calls = _patch_reconstruct(monkeypatch, csc)

    panel = csc.load_ground_truth_panel(
        method_dir,
        genotype_id="G0",
        yearsite_uid=None,
        rng=np.random.default_rng(0),
        num_samples=10,
        num_batches=None,
    )

    assert len(calls) == 1
    np.testing.assert_allclose(panel.x_values, np.array([200.0, 210.0, 220.0]))
    np.testing.assert_allclose(panel.mean, np.array([1.5, 3.5, 1.5]))


def test_conditioning_id_and_prediction_paths_do_not_reconstruct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_dir = Path("conditioning-method")
    compact = _compact_batch()
    _fail_on_reconstruct(monkeypatch, csc)

    _install_loader(monkeypatch, csc, [compact])
    assert csc.collect_condition_pairs(method_dir, num_batches=None) == [
        ("G0", "Y0"),
        ("G1", "Y1"),
    ]

    _install_loader(monkeypatch, csc, [(_predictions(), compact)])
    model_panel = csc.load_model_panel(
        method_dir,
        genotype_id="G0",
        yearsite_uid=None,
        rng=np.random.default_rng(0),
        num_samples=10,
        num_batches=None,
    )
    np.testing.assert_allclose(model_panel.x_values, np.array([200.0, 210.0, 220.0]))

    stratified = csc.load_model_stratified(
        method_dir, [("G0", "Y0")], seed=0, num_samples=2, num_batches=None
    )
    np.testing.assert_allclose(stratified.x_values, np.array([200.0, 210.0, 220.0]))
