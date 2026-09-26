"""Prevent changed input encodings and stale-oracle scoring in shift experiments."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from npnf.data.pools import GenotypePool
from npnf.data.synthetic.height.params import height_pool_params_from_mapping
from npnf.metrics.oracle_config import get_oracle_config_for_method_dir
from npnf.scripts.calibration.shifted_data import candidate_genotypes


def test_new_genotypes_keep_training_markers_and_distinct_ids():
    parameters = height_pool_params_from_mapping(
        {
            "mean_control_points": {f"cp_{i}": 0.5 for i in range(30)},
            "covariance": {
                "cp_sigma": 0.08,
                "length_scale_T": 20.0,
                "length_scale_tau": 0.2,
                "jitter": 1e-6,
            },
            "tau_max": {
                "loc": 25000.0,
                "scale": 2000.0,
                "min": 19000.0,
                "max": 31000.0,
            },
        }
    )
    training = GenotypePool.sample(16, 42, height_pool_params=parameters)
    new = candidate_genotypes(parameters, training.bounds, seed=43, count=64)
    bounds = training.bounds.to_dict()
    lower = torch.tensor(
        [bounds[f"cp_{i}"][0] for i in range(30)] + [bounds["tau_max"][0]]
    )
    upper = torch.tensor(
        [bounds[f"cp_{i}"][1] for i in range(30)] + [bounds["tau_max"][1]]
    )
    decoded = lower + new.markers * (upper - lower + 1e-8)
    torch.testing.assert_close(decoded, new.params)
    # Baseline exact-ID lookup must never treat a new genotype as a seen one.
    assert not set(new.genotype_ids) & set(training.genotype_ids)


def test_changed_oracle_manifest_cannot_fall_back_to_original_test(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"datasets": {"shifted": {"version": 1}}}')
    original_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    method = tmp_path / "no_context"
    method.mkdir()
    (method / "config.json").write_text(
        json.dumps(
            {
                "dataloader_name": "synth_test_unseen_dataloaders",
                "oracle_dataset": {
                    "output_dir": str(tmp_path),
                    "variant": "shifted",
                    "manifest_sha256": original_hash,
                },
            }
        )
    )
    manifest.write_text('{"datasets": {"shifted": {"version": 2}}}')
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        get_oracle_config_for_method_dir(method)


def test_legacy_test_set_is_default_and_keeps_old_names():
    from npnf.data.configs.datasets import synthetic
    from npnf.data.configs.datasets.synthetic_test_sets import (
        design_splits,
        split_for_dataloader,
    )
    from npnf.metrics.oracle_config import (
        get_oracle_config_for_dataloader_name,
        get_seed_b_config_for_dataloader_name,
    )

    assert [split.dataloader_name for split in design_splits("shifted")] == [
        "synth_shifted_seen_dataloaders",
        "synth_shifted_geno_dataloaders",
        "synth_shifted_env_dataloaders",
        "synth_shifted_unseen_dataloaders",
    ]
    # Saved legacy predictions must keep resolving to the old oracle.
    assert [split.alias for split in design_splits()] == [
        "plot",
        "genotype",
        "environment",
        "unseen",
    ]
    for split in design_splits("legacy"):
        assert get_oracle_config_for_dataloader_name(split.dataloader_name) is (
            getattr(synthetic, f"TestConfig_{split.title}")
        )
    unseen = split_for_dataloader("synth_shifted_unseen_dataloaders")
    assert get_seed_b_config_for_dataloader_name(unseen.dataloader_name) is (
        unseen.seed_b
    )


def test_predictions_record_the_frozen_oracle_reference():
    from types import SimpleNamespace

    from npnf.scripts.utils.prediction import prepare_batch_schema_config

    reference = {"output_dir": "bundle", "variant": "seen", "manifest_sha256": "x"}
    dataloader = SimpleNamespace(
        dataset=SimpleNamespace(
            frozen_reference=reference, get_dataset_identity=lambda: {"cache_key": "k"}
        )
    )
    config = prepare_batch_schema_config({}, "compact", dataloader)
    assert config["oracle_dataset"] == reference
