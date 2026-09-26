"""Synthetic oracle dataset configs for a prediction run's dataloader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from npnf.data.configs.datasets.shifted import shifted_split_config
from npnf.data.configs.datasets.synthetic_test_sets import (
    SyntheticTestSet,
    split_for_dataloader,
    splits_of,
)
from npnf.utils import file_hash


def get_oracle_config_for_dataloader_name(dataloader_name: str) -> Any:
    """Resolve the synthetic oracle dataset config for a dataloader name."""
    return split_for_dataloader(dataloader_name).oracle


def load_method_config(method_dir: Path | str) -> dict[str, Any]:
    """The ``config.json`` of a method directory; it must name its dataloader."""
    config_path = Path(method_dir) / "config.json"
    if not config_path.exists():
        msg = f"Missing config.json at {config_path}"
        raise FileNotFoundError(msg)
    config_json = json.loads(config_path.read_text())
    if config_json.get("dataloader_name") is None:
        msg = f"Missing 'dataloader_name' in {config_path}"
        raise ValueError(msg)
    return config_json


def get_oracle_config_for_method_dir(method_dir: Path | str) -> tuple[Any, str]:
    """Read a method directory config and resolve its oracle dataset config."""
    config_json = load_method_config(method_dir)
    dataloader_name = config_json["dataloader_name"]

    if "oracle_dataset" in config_json:
        reference = config_json["oracle_dataset"]
        output_dir = Path(reference["output_dir"]).resolve()
        manifest = output_dir / "manifest.json"
        if file_hash(manifest) != reference["manifest_sha256"]:
            msg = f"Frozen oracle manifest hash mismatch: {manifest}"
            raise ValueError(msg)
        return shifted_split_config(reference["variant"], output_dir), dataloader_name

    return get_oracle_config_for_dataloader_name(dataloader_name), dataloader_name


def get_seed_b_config_for_dataloader_name(dataloader_name: str) -> Any:
    """Resolve the independent SeedB config for a lodging-enabled split."""
    seed_b_cfg = split_for_dataloader(dataloader_name).seed_b
    if seed_b_cfg is None:
        known = [
            split.dataloader_name
            for test_set in SyntheticTestSet
            for split in splits_of(test_set)
            if split.seed_b is not None
        ]
        msg = (
            f"No SeedB oracle-self config for dataloader_name {dataloader_name!r}. "
            f"Known lodging-enabled dataloaders: {known}"
        )
        raise ValueError(msg)
    return seed_b_cfg
