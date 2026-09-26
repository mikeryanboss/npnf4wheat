from __future__ import annotations

import json
from pathlib import Path

import tensordict
import torch

from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.scripts.predict import random_context
from npnf.scripts.utils.utils import _synthetic_eval_day_axis, create_grid_points
from tests.utils import tensor_at, tensordict_at


class StubSyntheticDataset(SyntheticDataset):
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        return 2

    def get_dataset_identity(self) -> dict:
        return {
            "dataset_type": "SyntheticDataset",
            "cache_key": "random-context-stub",
            "cache_key_config": {"test": "random_context"},
            "num_conditions": 2,
        }


class StubDataloader(list):
    dataset = StubSyntheticDataset()


class StubModel:
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
    days = _synthetic_eval_day_axis(torch.device("cpu"))
    batch_size = 2
    x = days.reshape(1, -1, 1).expand(batch_size, -1, -1)
    x_normalized = (x - 212.5) / 151.5
    max_days = torch.tensor([280.0, 260.0]).reshape(batch_size, 1, 1)
    y = 1.0 - (x - max_days).abs() / 100.0
    height = tensordict.TensorDict(
        {"X": x, "X_normalized": x_normalized, "Y": y, "Y_original": y.clone()},
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


def _run_process(base_path: Path) -> None:
    random_context.process_dataloader(
        dataloader=StubDataloader([_make_batch()]),  # ty: ignore[invalid-argument-type]
        split_name="test",
        base_path=base_path,
        grid_points=create_grid_points(torch.device("cpu")),
        model=StubModel(),
        use_temperature=False,
        use_marker=False,
        num_samples=2,
        context_seed=42,
        min_context=1,
        max_context=10,
        num_batches=1,
        batch_schema="compact",
        config={
            "prediction_method": random_context.METHOD_NAME,
            "num_samples": 2,
            "context_seed": 42,
            "min_context": 1,
            "max_context": 10,
            "num_context_trials": 1,
            "context_pool": random_context.CONTEXT_POOL,
            "batch_schema": "compact",
        },
    )


def _load_saved(base_path: Path):
    method_dir = base_path / random_context.METHOD_NAME
    predictions = torch.load(
        method_dir / "predictions" / "000" / "predictions.pt", weights_only=False
    )
    batch = torch.load(
        method_dir / "predictions" / "000" / "batch.pt", weights_only=False
    )
    config = json.loads((method_dir / "config.json").read_text())
    return predictions, batch, config


def test_random_context_indices_can_select_grid_day() -> None:
    days = torch.tensor([[199.0], [200.0], [322.0]])
    y_original = torch.tensor([[0.0], [1.0], [0.0]])
    height = tensordict.TensorDict(
        {
            "X": days,
            "X_normalized": (days - 212.5) / 151.5,
            "Y": y_original.clone(),
            "Y_original": y_original,
        },
        batch_size=[len(days)],
    )
    grid_days = torch.tensor([200.0])

    indices = random_context._sample_context_indices(  # noqa: SLF001
        height,
        torch.Generator(device="cpu").manual_seed(0),
        min_context=1,
        max_context=1,
    )

    selected_days = days[indices, 0]
    assert indices == [1]
    assert bool(torch.isin(selected_days, grid_days).all())


def test_random_context_indices_are_deterministic_and_valid(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_process(first)
    _run_process(second)

    first_predictions, first_batch, first_config = _load_saved(first)
    second_predictions, second_batch, _ = _load_saved(second)

    assert first_batch["context_indices"] == second_batch["context_indices"]
    assert first_predictions.batch_size == torch.Size([2, 1])
    assert second_predictions.batch_size == torch.Size([2, 1])

    height = tensordict_at(_make_batch(), "height")
    for sample_index, indices in enumerate(first_batch["context_indices"]):
        assert 1 <= len(indices) <= 10
        assert indices == sorted(indices)

        days = tensor_at(tensordict_at(height, sample_index), "X")[..., 0]
        clean = tensor_at(tensordict_at(height, sample_index), "Y_original")[..., 0]
        max_height_day = days[clean.argmax()]
        selected_days = days[indices]

        assert bool(((selected_days >= 200) & (selected_days <= 321)).all())
        assert bool((selected_days <= max_height_day).all())

    assert first_config["dataset_identity"]["cache_key"] == "random-context-stub"
