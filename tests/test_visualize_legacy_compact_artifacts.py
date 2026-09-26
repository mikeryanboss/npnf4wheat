from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensordict
import torch

from npnf.data.process.collate import collate_fn_fip1_heights
from npnf.scripts.metrics import calculate_synthetic
from npnf.scripts.utils import synthetic_batch_reconstruction as reconstruction
from npnf.scripts.visualize import create_gifs, plot_trajectories, plot_yearsite_prior


class _ListLoader:
    def __init__(self, items):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


class _ParityDataset:
    rows = (("G0", "SITE_2020"), ("G1", "SITE_2021"))

    def __len__(self) -> int:
        return len(self.rows)

    def get_dataset_identity(self) -> dict:
        return {"cache_key": "visualize-parity-cache", "num_conditions": len(self)}

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


def _install_loader(monkeypatch: pytest.MonkeyPatch, module, items_by_dir: dict):
    def fake_loader(method_dir: Path, **_kwargs):
        return _ListLoader(items_by_dir[Path(method_dir)])

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


def _full_batch(dataset: _ParityDataset) -> dict:
    data = (
        collate_fn_fip1_heights([dataset[index] for index in range(len(dataset))])
        .select("height", "has_lodged", "genotype_id", "yearsite_uid")
        .cpu()
    )
    return {
        "data": data,
        "grid_points": {"X": torch.tensor([[200.0], [210.0], [220.0]])},
        "context_indices": [[0], [1]],
    }


def _compact_batch(dataset: _ParityDataset) -> dict:
    return {
        "data": {
            "genotype_id": [row[0] for row in dataset.rows],
            "yearsite_uid": [row[1] for row in dataset.rows],
        },
        "grid_points": {"X": torch.tensor([[200.0], [210.0], [220.0]])},
        "context_indices": [[0], [1]],
    }


def _grid_predictions() -> dict:
    grid = torch.tensor(
        [[[[[1.0], [2.0], [1.0]]]], [[[[2.0], [3.0], [2.0]]]]], dtype=torch.float32
    )
    return {"grid": grid}


def _yearsite_predictions() -> tensordict.TensorDict:
    return tensordict.TensorDict(_grid_predictions(), batch_size=[2])


def _target_predictions() -> dict:
    targets = torch.tensor(
        [
            [[[2.0], [3.0], [1.0]], [[3.0], [4.0], [2.0]]],
            [[[3.0], [4.0], [2.0]], [[4.0], [5.0], [3.0]]],
        ],
        dtype=torch.float32,
    )
    return {"targets": targets}


def test_create_gifs_visualize_reconstructs_compact_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _ParityDataset()
    method_dir = tmp_path / "compact"
    _write_compact_config(method_dir, dataset)
    _patch_reconstruction_dataset(monkeypatch, dataset)
    compact_batch = _compact_batch(dataset)
    _install_loader(
        monkeypatch,
        create_gifs,
        {method_dir: [(["sample0", "sample1"], compact_batch)]},
    )

    captured_args = []

    class DummyPool:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return None

        def map(self, func, args):
            captured_args.extend(args)
            return [func(arg) for arg in args]

    monkeypatch.setattr(create_gifs, "Pool", DummyPool)
    monkeypatch.setattr(create_gifs, "process_single_sample", lambda args: args[0])

    create_gifs.visualize(str(method_dir), num_images=1)

    assert len(captured_args) == 1
    args = captured_args[0]
    assert args[0] == 0
    assert args[4] == [0]
    torch.testing.assert_close(args[5], compact_batch["grid_points"]["X"])
    torch.testing.assert_close(
        args[2]["Y"], _full_batch(dataset)["data"]["height"][0]["Y"]
    )


def test_plot_trajectories_ground_truth_matches_full_after_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _ParityDataset()
    full_dir = tmp_path / "full"
    compact_dir = tmp_path / "compact"
    _write_compact_config(compact_dir, dataset)
    _patch_reconstruction_dataset(monkeypatch, dataset)
    full_batch = _full_batch(dataset)
    compact_batch = _compact_batch(dataset)
    _install_loader(
        monkeypatch,
        plot_trajectories,
        {full_dir: [full_batch], compact_dir: [compact_batch]},
    )

    full = plot_trajectories.load_ground_truth(full_dir, np.array([0]))
    compact = plot_trajectories.load_ground_truth(compact_dir, np.array([0]))

    for actual, expected in zip(compact, full, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_plot_trajectories_predictions_do_not_reconstruct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method_dir = tmp_path / "method"
    _install_loader(monkeypatch, plot_trajectories, {method_dir: [_grid_predictions()]})

    def fail_reconstruction(*_args, **_kwargs):
        pytest.fail("prediction-only loading should not reconstruct full data")

    monkeypatch.setattr(
        plot_trajectories, "reconstruct_synthetic_batch", fail_reconstruction
    )

    trajectories = plot_trajectories.load_predictions(method_dir, np.array([0]))

    np.testing.assert_allclose(
        trajectories, np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0]])
    )


def test_plot_yearsite_prior_load_all_batches_matches_full_after_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _ParityDataset()
    full_dir = tmp_path / "full"
    compact_dir = tmp_path / "compact"
    _write_compact_config(compact_dir, dataset)
    _patch_reconstruction_dataset(monkeypatch, dataset)
    full_batch = _full_batch(dataset)
    compact_batch = _compact_batch(dataset)
    predictions = _yearsite_predictions()
    _install_loader(
        monkeypatch,
        plot_yearsite_prior,
        {
            full_dir: [(predictions, full_batch)],
            compact_dir: [(predictions, compact_batch)],
        },
    )

    full_grid_max, full_batch_max, full_metadata = (
        plot_yearsite_prior._load_all_batches(full_dir)  # noqa: SLF001
    )
    compact_grid_max, compact_batch_max, compact_metadata = (
        plot_yearsite_prior._load_all_batches(compact_dir)  # noqa: SLF001
    )

    torch.testing.assert_close(compact_grid_max, full_grid_max)
    torch.testing.assert_close(compact_batch_max, full_batch_max)
    assert compact_metadata == full_metadata


def test_calculate_synthetic_metrics_match_full_after_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _ParityDataset()
    full_dir = tmp_path / "full"
    compact_dir = tmp_path / "compact"
    _write_compact_config(compact_dir, dataset)
    _patch_reconstruction_dataset(monkeypatch, dataset)
    full_batch = _full_batch(dataset)
    compact_batch = _compact_batch(dataset)
    predictions = _target_predictions()
    _install_loader(
        monkeypatch,
        calculate_synthetic,
        {
            full_dir: [(predictions, full_batch)],
            compact_dir: [(predictions, compact_batch)],
        },
    )

    full_metrics = calculate_synthetic.calculate_single_shot_metrics(full_dir)
    compact_metrics = calculate_synthetic.calculate_single_shot_metrics(compact_dir)

    assert set(compact_metrics) == {"mae_clean", "mean_epistemic_std"}
    assert compact_metrics == pytest.approx(full_metrics)


def test_calculate_synthetic_grid_only_predictions_fail_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method_dir = tmp_path / "grid-only"
    _install_loader(
        monkeypatch,
        calculate_synthetic,
        {method_dir: [(_grid_predictions(), _compact_batch(_ParityDataset()))]},
    )

    with pytest.raises(ValueError, match=r"predictions\['targets'\].*grid-only"):
        calculate_synthetic.calculate_single_shot_metrics(method_dir)


def test_unreconstructable_compact_artifact_surfaces_reconstruction_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method_dir = tmp_path / "compact-without-config"
    _install_loader(
        monkeypatch, plot_trajectories, {method_dir: [_compact_batch(_ParityDataset())]}
    )

    with pytest.raises(ValueError, match=r"config\.json"):
        plot_trajectories.load_ground_truth(method_dir, np.array([0]))
