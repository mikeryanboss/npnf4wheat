"""Reconstruct full synthetic prediction batch metadata from compact artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from hydra_zen import instantiate

from npnf.data.datasets.utils import verify_dataset_cache_key
from npnf.data.process.collate import collate_fn_fip1_heights
from npnf.metrics.oracle_config import get_oracle_config_for_dataloader_name

_FULL_DATA_FIELDS = ("height", "has_lodged", "genotype_id", "yearsite_uid")
_COMPACT_ID_FIELDS = ("genotype_id", "yearsite_uid")


@dataclass(frozen=True)
class _DatasetState:
    dataset: Any
    cache_key: str
    id_to_index: dict[tuple[str, str], int]


@dataclass
class SyntheticBatchReconstructionCache:
    """Cache the synthetic dataset and ID map for each method directory."""

    _states: dict[Path, _DatasetState] = field(default_factory=dict)

    def get(self, method_dir: Path, dataloader_name: str) -> _DatasetState:
        method_dir = method_dir.expanduser().resolve()
        if method_dir not in self._states:
            self._states[method_dir] = _load_dataset_state(method_dir, dataloader_name)
        return self._states[method_dir]


def reconstruct_synthetic_batch(
    method_dir: Path | str,
    batch_dict: object,
    cache: SyntheticBatchReconstructionCache | None = None,
    *,
    allow_unverified_dataset_identity: bool = False,
) -> dict[str, Any]:
    """Hydrate compact one-step synthetic batch metadata to full schema.

    Existing full-schema batches are returned unchanged. Compact batches are
    reconstructed from the deterministic synthetic test dataset named in
    ``config.json``. The saved ``dataset_identity.cache_key`` must match the
    loaded dataset cache key unless ``allow_unverified_dataset_identity`` is set
    for trusted legacy compact artifacts.
    """
    method_path = Path(method_dir).expanduser().resolve()
    if not isinstance(batch_dict, dict):
        msg = f"{method_path}: batch_dict must be a dict loaded from batch.pt"
        raise TypeError(msg)
    if "data" not in batch_dict:
        msg = f"{method_path}: batch.pt is missing required top-level 'data' field"
        raise ValueError(msg)

    data = batch_dict["data"]
    if "height" in data:
        return batch_dict

    for field_name in _COMPACT_ID_FIELDS:
        if field_name not in data:
            msg = (
                f"{method_path}: compact batch data is missing required "
                f"'data/{field_name}' field"
            )
            raise ValueError(msg)

    dataloader_name, saved_identity = _compact_config(
        method_path, allow_unverified_dataset_identity=allow_unverified_dataset_identity
    )
    reconstruction_cache = cache or SyntheticBatchReconstructionCache()
    state = reconstruction_cache.get(method_path, dataloader_name)
    if saved_identity is not None:
        _verify_cache_key(method_path, saved_identity, state.cache_key)

    indices = _dataset_indices(method_path, data, state.id_to_index)
    samples = [state.dataset[index] for index in indices]
    reconstructed_data = collate_fn_fip1_heights(samples).select(*_FULL_DATA_FIELDS)

    reconstructed = dict(batch_dict)
    reconstructed["data"] = reconstructed_data.cpu()
    return reconstructed


def _compact_config(
    method_dir: Path, *, allow_unverified_dataset_identity: bool
) -> tuple[str, dict[str, Any] | None]:
    config_path = method_dir / "config.json"
    if not config_path.exists():
        msg = f"{method_dir}: compact batch reconstruction requires {config_path}"
        raise ValueError(msg)
    with config_path.open() as f:
        config = json.load(f)

    if config.get("batch_schema") != "compact":
        msg = (
            f"{method_dir}: batch.pt has compact-looking data without height, "
            f"but config.json batch_schema is {config.get('batch_schema')!r}; "
            "expected 'compact'"
        )
        raise ValueError(msg)

    dataloader_name = config.get("dataloader_name")
    if dataloader_name is None:
        msg = f"{method_dir}: config.json is missing required 'dataloader_name' field"
        raise ValueError(msg)

    saved_identity = config.get("dataset_identity")
    if saved_identity is None and not allow_unverified_dataset_identity:
        msg = (
            f"{method_dir}: compact batch reconstruction requires config.json "
            "dataset_identity; pass allow_unverified_dataset_identity=True only "
            "for trusted legacy compact artifacts"
        )
        raise ValueError(msg)
    return str(dataloader_name), saved_identity


def _verify_cache_key(
    method_dir: Path, saved_identity: dict[str, Any], loaded_cache_key: str
) -> None:
    verify_dataset_cache_key(
        method_dir, saved_identity.get("cache_key"), loaded_cache_key
    )


def _load_dataset_state(method_dir: Path, dataloader_name: str) -> _DatasetState:
    try:
        oracle_config = get_oracle_config_for_dataloader_name(dataloader_name)
    except ValueError as exc:
        msg = (
            f"{method_dir}: cannot resolve dataloader_name {dataloader_name!r} "
            f"for compact synthetic reconstruction: {exc}"
        )
        raise ValueError(msg) from exc

    dataset = instantiate(oracle_config)
    identity = dataset.get_dataset_identity()
    cache_key = identity.get("cache_key")
    if cache_key is None:
        msg = f"{method_dir}: resolved synthetic dataset identity lacks cache_key"
        raise ValueError(msg)
    return _DatasetState(
        dataset=dataset,
        cache_key=str(cache_key),
        id_to_index=build_id_to_index(method_dir, dataset),
    )


def build_id_to_index(method_dir: Path, dataset) -> dict[tuple[str, str], int]:
    metadata = dataset.get_condition_metadata()
    genotype_ids = metadata.get("genotype_id")
    yearsite_uids = metadata.get("yearsite_uid")
    if genotype_ids is None or yearsite_uids is None:
        msg = (
            f"{method_dir}: synthetic dataset metadata must contain genotype_id "
            "and yearsite_uid"
        )
        raise ValueError(msg)

    id_to_index: dict[tuple[str, str], int] = {}
    for index, (genotype_id, yearsite_uid) in enumerate(
        zip(genotype_ids, yearsite_uids, strict=True)
    ):
        key = (_normalize_id(genotype_id), _normalize_id(yearsite_uid))
        if key in id_to_index:
            msg = (
                f"{method_dir}: duplicate synthetic dataset condition ID pair "
                f"{key!r} at indices {id_to_index[key]} and {index}"
            )
            raise ValueError(msg)
        id_to_index[key] = index
    return id_to_index


def _dataset_indices(
    method_dir: Path, data, id_to_index: dict[tuple[str, str], int]
) -> list[int]:
    genotype_ids = _id_list(method_dir, data, "genotype_id")
    yearsite_uids = _id_list(method_dir, data, "yearsite_uid")
    if len(genotype_ids) != len(yearsite_uids):
        msg = (
            f"{method_dir}: compact batch ID length mismatch: "
            f"genotype_id={len(genotype_ids)}, yearsite_uid={len(yearsite_uids)}"
        )
        raise ValueError(msg)

    indices: list[int] = []
    missing: list[tuple[str, str]] = []
    for key in zip(genotype_ids, yearsite_uids, strict=True):
        index = id_to_index.get(key)
        if index is None:
            missing.append(key)
        else:
            indices.append(index)
    if missing:
        msg = (
            f"{method_dir}: {len(missing)} compact batch condition ID pairs "
            f"were not found in the resolved synthetic dataset; "
            f"examples={missing[:5]}"
        )
        raise ValueError(msg)
    return indices


def _id_list(method_dir: Path, data, field_name: str) -> list[str]:
    values = data[field_name]
    if torch.is_tensor(values):
        if values.ndim == 0:
            msg = (
                f"{method_dir}: compact batch 'data/{field_name}' must have a "
                "batch dimension"
            )
            raise ValueError(msg)
        return [_normalize_id(value) for value in values]
    return [_normalize_id(value) for value in values]


def _normalize_id(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0 or value.numel() == 1:
            return str(value.item())
        return str(value.tolist())
    return str(value)


__all__ = ["SyntheticBatchReconstructionCache", "reconstruct_synthetic_batch"]
