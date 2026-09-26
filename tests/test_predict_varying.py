from __future__ import annotations

import json
from pathlib import Path

import tensordict
import torch

from npnf.scripts.predict import varying
from npnf.scripts.utils.prediction import get_indices_random


class StubSyntheticDataset:
    def get_dataset_identity(self) -> dict:
        return {
            "dataset_type": "SyntheticDataset",
            "cache_key": "varying-stub-cache",
            "num_conditions": 2,
        }


class StubDataloader(list):
    def __init__(self, batches) -> None:
        super().__init__(batches)
        self.dataset = StubSyntheticDataset()


class StubModel:
    latent_dim = 2
    variance = torch.tensor(0.01)

    def __call__(
        self,
        context_priors,
        context_posteriors,
        targets,
        temperatures=None,
        markers=None,
        num_samples: int = 1,
        noise_z=None,
        get_samples: bool = True,
        get_loss: bool = False,
        sample_from_posterior: bool = False,
        genotype_ids=None,
        yearsite_ids=None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        base = targets["X_normalized"][:, None, :, :].expand(-1, num_samples, -1, -1)
        offsets = torch.arange(num_samples, device=base.device, dtype=base.dtype).view(
            1, num_samples, 1, 1
        )
        predictions = base + offsets * 0.01
        return {
            "predictions": {
                "target": predictions,
                "target_scale": torch.ones_like(predictions) * 0.1,
            }
        }


def _make_batch() -> tensordict.TensorDict:
    batch_size = 2
    seq_len = 4
    x = (
        torch.arange(seq_len, dtype=torch.float32)
        .reshape(1, seq_len, 1)
        .expand(batch_size, -1, -1)
    )
    y = torch.tensor(
        [[[0.0], [1.0], [2.0], [1.5]], [[0.0], [0.8], [1.2], [1.1]]],
        dtype=torch.float32,
    )
    height = tensordict.TensorDict(
        {"X": x, "X_normalized": x / 10.0, "Y": y, "Y_original": y.clone()},
        batch_size=[batch_size],
    )
    return tensordict.TensorDict(
        {
            "height": height,
            "yearsite_uid": torch.tensor([10, 11]),
            "has_lodged": torch.tensor([0, 1]),
            "genotype_id": torch.tensor([100, 101]),
        },
        batch_size=[batch_size],
    )


def _make_grid() -> tensordict.TensorDict:
    x = torch.linspace(0.0, 1.0, 3).reshape(3, 1)
    return tensordict.TensorDict({"X": x, "X_normalized": x}, batch_size=[3])


def _run_varying_process(
    base_path: Path,
    *,
    predict_uncertainty: bool = True,
    predict_sequential: bool = False,
    num_random_trials: int = 0,
) -> None:
    num_predictions = (
        int(predict_uncertainty) + int(predict_sequential) + num_random_trials
    )
    varying.process_dataloader(
        dataloader=StubDataloader([_make_batch()]),  # ty: ignore[invalid-argument-type]
        split_name="test",
        base_path=base_path,
        num_predictions=num_predictions,
        grid_points=_make_grid(),
        model=StubModel(),
        use_temperature=False,
        use_marker=False,
        predict_uncertainty=predict_uncertainty,
        predict_sequential=predict_sequential,
        num_random_trials=num_random_trials,
        num_samples=2,
        max_num_context_prior=1,
        num_batches=1,
        run_config={
            "num_samples": 2,
            "num_random_trials": num_random_trials,
            "max_num_context_prior": 1,
            "dataloader_name": "stub_dataloader",
            "checkpoint_folder": "stub-checkpoint",
            "model_variance": 0.01,
            "save_dtype": "float16",
        },
        save_dtype=torch.float16,
    )


def _load_config(method_dir: Path) -> dict:
    return json.loads((method_dir / "config.json").read_text())


def _load_saved(base_path: Path, method: str):
    predictions = torch.load(
        base_path / method / "predictions" / "000" / "predictions.pt",
        weights_only=False,
    )
    batch = torch.load(
        base_path / method / "predictions" / "000" / "batch.pt", weights_only=False
    )
    return predictions, batch


def _assert_grid_only_compact_artifact(
    base_path: Path, method: str, expected_batch_size: torch.Size
) -> tuple[tensordict.TensorDict, dict]:
    predictions, batch = _load_saved(base_path, method)

    assert predictions.batch_size == expected_batch_size
    assert set(predictions.keys()) == {"grid"}
    assert predictions["grid"].dtype == torch.float16

    assert set(batch["data"].keys()) == {"genotype_id", "yearsite_uid"}
    assert "height" not in batch["data"]
    assert "has_lodged" not in batch["data"]
    assert batch["grid_points"]["X"].dtype == torch.float16
    assert "max_valid_context" in batch
    assert "context_indices" in batch

    return predictions, batch


def test_uncertainty_only_output_is_compact_grid_only(tmp_path: Path) -> None:
    _run_varying_process(tmp_path, num_random_trials=0)

    config = _load_config(tmp_path / "uncertainty")
    assert config["prediction_method"] == "uncertainty"
    assert config["trials_per_sample"] == 1
    assert config["num_random_trials"] == 0
    assert config["batch_schema"] == "compact"
    assert config["dataset_identity"] == StubSyntheticDataset().get_dataset_identity()
    assert config["save_dtype"] == "float16"

    _, batch = _assert_grid_only_compact_artifact(
        tmp_path, "uncertainty", torch.Size([2, 2])
    )
    assert batch["data"]["genotype_id"].tolist() == [100, 101]
    assert batch["data"]["yearsite_uid"].tolist() == [10, 11]
    assert batch["max_valid_context"].tolist() == [4, 4]
    assert len(batch["context_indices"]) == 2
    assert all(len(indices) == 2 for indices in batch["context_indices"])


def test_mixed_random_and_uncertainty_outputs_are_compact_grid_only(
    tmp_path: Path,
) -> None:
    _run_varying_process(tmp_path, num_random_trials=2)

    uncertainty_config = _load_config(tmp_path / "uncertainty")
    random_config = _load_config(tmp_path / "random")

    assert uncertainty_config["prediction_method"] == "uncertainty"
    assert uncertainty_config["trials_per_sample"] == 1
    assert uncertainty_config["num_random_trials"] == 2
    assert uncertainty_config["batch_schema"] == "compact"
    assert random_config["prediction_method"] == "random"
    assert random_config["trials_per_sample"] == 2
    assert random_config["num_random_trials"] == 2
    assert random_config["batch_schema"] == "compact"

    _, uncertainty_batch = _assert_grid_only_compact_artifact(
        tmp_path, "uncertainty", torch.Size([2, 2])
    )
    _, random_batch = _assert_grid_only_compact_artifact(
        tmp_path, "random", torch.Size([4, 2])
    )

    assert uncertainty_batch["data"]["genotype_id"].tolist() == [100, 101]
    assert uncertainty_batch["data"]["yearsite_uid"].tolist() == [10, 11]
    assert random_batch["data"]["genotype_id"].tolist() == [100, 101, 100, 101]
    assert random_batch["data"]["yearsite_uid"].tolist() == [10, 11, 10, 11]
    assert random_batch["max_valid_context"].tolist() == [4, 4, 4, 4]
    assert len(random_batch["context_indices"]) == 4
    assert all(len(indices) == 2 for indices in random_batch["context_indices"])


def test_get_indices_random_uses_target_mask_device() -> None:
    mask = torch.tensor([False, True, False])
    assert get_indices_random([mask])[0] in {0, 2}

    if torch.cuda.is_available():
        cuda_mask = mask.cuda()
        assert get_indices_random([cuda_mask])[0] in {0, 2}
