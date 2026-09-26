from __future__ import annotations

import json
from pathlib import Path

import pytest
import tensordict
import torch

from npnf.data.process.collate import collate_fn_fip1_heights
from npnf.scripts.utils import synthetic_batch_reconstruction as reconstruction
from tests.utils import tensor_at


class FakeSyntheticDataset:
    def __init__(
        self,
        rows: list[tuple[str, str]] | None = None,
        *,
        cache_key: str = "fake-cache-key",
    ) -> None:
        self.rows = rows or [("G0", "Y0"), ("G1", "Y1"), ("G2", "Y2")]
        self.cache_key = cache_key

    def __len__(self) -> int:
        return len(self.rows)

    def get_dataset_identity(self) -> dict:
        return {
            "dataset_type": "SyntheticDataset",
            "cache_key": self.cache_key,
            "cache_key_config": {"rows": self.rows},
            "num_conditions": len(self),
        }

    def get_condition_metadata(self) -> dict[str, list[str]]:
        return {
            "genotype_id": [row[0] for row in self.rows],
            "yearsite_uid": [row[1] for row in self.rows],
        }

    def __getitem__(self, index: int) -> dict:
        genotype_id, yearsite_uid = self.rows[index]
        days = torch.tensor([200.0, 210.0, 220.0])
        base = float(index + 1)
        clean_draws = torch.stack(
            [
                torch.tensor([base, base + 0.2, base + 0.4]),
                torch.tensor([base + 0.5, base + 0.7, base + 0.9]),
            ]
        )
        observed = clean_draws[0] + 0.01
        lodged_mask = torch.tensor(
            [[False, False, index % 2 == 0], [False, True, True]], dtype=torch.bool
        )
        has_lodged = torch.tensor([index % 2 == 0, True], dtype=torch.bool)
        return {
            "height_days": days,
            "height_days_normalized": (days - 212.5) / 151.5,
            "height_values": observed,
            "height_values_nonoise": clean_draws,
            "height_lodged_mask": lodged_mask,
            "has_lodged": has_lodged,
            "genotype_id": genotype_id,
            "yearsite_uid": yearsite_uid,
            "temperature_values": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        }


def _write_config(
    method_dir: Path,
    dataset: FakeSyntheticDataset,
    *,
    dataloader_name: str = "fake_loader",
    batch_schema: str | None = "compact",
    dataset_identity=...,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    config = {"dataloader_name": dataloader_name}
    if batch_schema is not None:
        config["batch_schema"] = batch_schema
    if dataset_identity is ...:
        config["dataset_identity"] = dataset.get_dataset_identity()
    elif dataset_identity is not None:
        config["dataset_identity"] = dataset_identity
    (method_dir / "config.json").write_text(json.dumps(config))


def _compact_batch(dataset: FakeSyntheticDataset, indices: list[int]) -> dict:
    data = tensordict.TensorDict(
        {  # ty: ignore[invalid-argument-type]
            "genotype_id": [dataset.rows[index][0] for index in indices],
            "yearsite_uid": [dataset.rows[index][1] for index in indices],
        },
        batch_size=[len(indices)],
    )
    grid_points = tensordict.TensorDict(
        {"X": torch.tensor([[200.0], [210.0], [220.0]])}, batch_size=[3]
    )
    return {
        "data": data,
        "grid_points": grid_points,
        "context_indices": [[0], [1]][: len(indices)],
    }


def _full_data(
    dataset: FakeSyntheticDataset, indices: list[int]
) -> tensordict.TensorDict:
    return (
        collate_fn_fip1_heights([dataset[index] for index in indices])
        .select("height", "has_lodged", "genotype_id", "yearsite_uid")
        .cpu()
    )


def _patch_dataset(monkeypatch: pytest.MonkeyPatch, dataset: FakeSyntheticDataset):
    calls = {"resolve": 0, "instantiate": 0}
    fake_config = object()

    def fake_resolve(dataloader_name: str):
        calls["resolve"] += 1
        if dataloader_name != "fake_loader":
            msg = f"Unknown dataloader_name {dataloader_name!r}"
            raise ValueError(msg)
        return fake_config

    def fake_instantiate(config):
        calls["instantiate"] += 1
        assert config is fake_config
        return dataset

    monkeypatch.setattr(
        reconstruction, "get_oracle_config_for_dataloader_name", fake_resolve
    )
    monkeypatch.setattr(reconstruction, "instantiate", fake_instantiate)
    return calls


def _unwrap(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.values() if tensor.is_nested else tensor


def _assert_reconstructed_matches_expected(
    actual: tensordict.TensorDict, expected: tensordict.TensorDict
) -> None:
    assert set(actual.keys()) == {"height", "has_lodged", "genotype_id", "yearsite_uid"}
    for key in ["X", "X_normalized", "Y", "Y_original", "lodged_mask"]:
        assert torch.equal(
            _unwrap(tensor_at(actual, ("height", key))),
            _unwrap(tensor_at(expected, ("height", key))),
        )
    assert torch.equal(
        tensor_at(actual, "has_lodged"), tensor_at(expected, "has_lodged")
    )
    assert list(actual["genotype_id"]) == list(expected["genotype_id"])
    assert list(actual["yearsite_uid"]) == list(expected["yearsite_uid"])


def test_full_schema_batch_returns_unchanged_without_dataset_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = FakeSyntheticDataset()
    full_batch = {"data": _full_data(dataset, [0, 1])}

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("full-schema artifacts must not instantiate a dataset")

    monkeypatch.setattr(reconstruction, "instantiate", fail_if_called)

    result = reconstruction.reconstruct_synthetic_batch(tmp_path / "method", full_batch)

    assert result is full_batch


def test_compact_batch_reconstructs_full_schema_and_preserves_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = FakeSyntheticDataset()
    method_dir = tmp_path / "method"
    _write_config(method_dir, dataset)
    calls = _patch_dataset(monkeypatch, dataset)
    compact = _compact_batch(dataset, [2, 0])

    reconstructed = reconstruction.reconstruct_synthetic_batch(method_dir, compact)

    _assert_reconstructed_matches_expected(
        reconstructed["data"], _full_data(dataset, [2, 0])
    )
    assert reconstructed["grid_points"] is compact["grid_points"]
    assert reconstructed["context_indices"] == compact["context_indices"]
    assert list(reconstructed["data"]["genotype_id"]) == ["G2", "G0"]
    assert calls == {"resolve": 1, "instantiate": 1}


def test_cache_reuses_dataset_and_id_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = FakeSyntheticDataset()
    method_dir = tmp_path / "method"
    _write_config(method_dir, dataset)
    calls = _patch_dataset(monkeypatch, dataset)
    cache = reconstruction.SyntheticBatchReconstructionCache()

    reconstruction.reconstruct_synthetic_batch(
        method_dir, _compact_batch(dataset, [0]), cache=cache
    )
    reconstruction.reconstruct_synthetic_batch(
        method_dir, _compact_batch(dataset, [1]), cache=cache
    )

    assert calls == {"resolve": 1, "instantiate": 1}


def test_dataset_identity_mismatch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = FakeSyntheticDataset()
    method_dir = tmp_path / "method"
    _write_config(
        method_dir,
        dataset,
        dataset_identity={**dataset.get_dataset_identity(), "cache_key": "wrong"},
    )
    _patch_dataset(monkeypatch, dataset)

    with pytest.raises(ValueError, match="dataset_identity cache_key mismatch"):
        reconstruction.reconstruct_synthetic_batch(
            method_dir, _compact_batch(dataset, [0])
        )


def test_missing_dataset_identity_fails_by_default_but_opt_in_allows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = FakeSyntheticDataset()
    method_dir = tmp_path / "method"
    _write_config(method_dir, dataset, dataset_identity=None)
    _patch_dataset(monkeypatch, dataset)

    with pytest.raises(ValueError, match="dataset_identity"):
        reconstruction.reconstruct_synthetic_batch(
            method_dir, _compact_batch(dataset, [0])
        )

    reconstructed = reconstruction.reconstruct_synthetic_batch(
        method_dir, _compact_batch(dataset, [0]), allow_unverified_dataset_identity=True
    )

    _assert_reconstructed_matches_expected(
        reconstructed["data"], _full_data(dataset, [0])
    )


def test_missing_config_fails(tmp_path: Path) -> None:
    dataset = FakeSyntheticDataset()

    with pytest.raises(ValueError, match=r"config\.json"):
        reconstruction.reconstruct_synthetic_batch(
            tmp_path / "method", _compact_batch(dataset, [0])
        )


def test_missing_dataloader_name_fails(tmp_path: Path) -> None:
    dataset = FakeSyntheticDataset()
    method_dir = tmp_path / "method"
    method_dir.mkdir()
    (method_dir / "config.json").write_text(
        json.dumps(
            {
                "batch_schema": "compact",
                "dataset_identity": dataset.get_dataset_identity(),
            }
        )
    )

    with pytest.raises(ValueError, match="dataloader_name"):
        reconstruction.reconstruct_synthetic_batch(
            method_dir, _compact_batch(dataset, [0])
        )


def test_unknown_dataloader_name_fails(tmp_path: Path) -> None:
    dataset = FakeSyntheticDataset()
    method_dir = tmp_path / "method"
    _write_config(method_dir, dataset, dataloader_name="unknown_loader")

    with pytest.raises(ValueError, match="cannot resolve dataloader_name"):
        reconstruction.reconstruct_synthetic_batch(
            method_dir, _compact_batch(dataset, [0])
        )


def test_missing_required_compact_fields_fail(tmp_path: Path) -> None:
    dataset = FakeSyntheticDataset()
    method_dir = tmp_path / "method"
    _write_config(method_dir, dataset)
    batch = _compact_batch(dataset, [0])
    batch["data"] = batch["data"].exclude("yearsite_uid")

    with pytest.raises(ValueError, match="yearsite_uid"):
        reconstruction.reconstruct_synthetic_batch(method_dir, batch)


def test_missing_condition_ids_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = FakeSyntheticDataset()
    method_dir = tmp_path / "method"
    _write_config(method_dir, dataset)
    _patch_dataset(monkeypatch, dataset)
    batch = _compact_batch(dataset, [0])
    batch["data"]["genotype_id"] = ["missing"]

    with pytest.raises(ValueError, match="not found"):
        reconstruction.reconstruct_synthetic_batch(method_dir, batch)


def test_duplicate_dataset_condition_ids_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = FakeSyntheticDataset(rows=[("G0", "Y0"), ("G0", "Y0")])
    method_dir = tmp_path / "method"
    _write_config(method_dir, dataset)
    _patch_dataset(monkeypatch, dataset)

    with pytest.raises(ValueError, match="duplicate"):
        reconstruction.reconstruct_synthetic_batch(
            method_dir, _compact_batch(dataset, [0])
        )


def test_missing_batch_schema_fails(tmp_path: Path) -> None:
    dataset = FakeSyntheticDataset()
    method_dir = tmp_path / "method"
    _write_config(method_dir, dataset, batch_schema=None)

    with pytest.raises(ValueError, match="batch_schema"):
        reconstruction.reconstruct_synthetic_batch(
            method_dir, _compact_batch(dataset, [0])
        )
