from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tensordict
import torch

from npnf.data.batch_loader import BatchLoader
from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.metrics.sig_mmd import normalized_sig_mmd
from npnf.metrics.signature import to_metric_paths
from npnf.scripts.metrics import precompute_sig_mahalanobis, sig_mmd
from npnf.scripts.predict import max_height, no_context, random_context
from npnf.scripts.utils.utils import _synthetic_eval_day_axis, create_grid_points

ONE_STEP_METHODS = [
    no_context.METHOD_NAME,
    max_height.METHOD_NAME,
    random_context.METHOD_NAME,
]


class DummyAccelerator:
    device = torch.device("cpu")


class StubSyntheticDataset(SyntheticDataset):
    def __init__(
        self,
        cache_key: str = "stub-cache-key",
        cache_key_config: dict | None = None,
        num_conditions: int = 2,
    ) -> None:
        self.cache_key = cache_key
        self.cache_key_config = cache_key_config or {"split": cache_key}
        self.num_conditions = num_conditions

    def __len__(self) -> int:
        return self.num_conditions

    def get_dataset_identity(self) -> dict:
        return {
            "dataset_type": "SyntheticDataset",
            "cache_key": self.cache_key,
            "cache_key_config": self.cache_key_config,
            "num_conditions": self.num_conditions,
        }


class StubDataloader(list):
    def __init__(self, batches, dataset: SyntheticDataset | None = None) -> None:
        super().__init__(batches)
        self.dataset = dataset if dataset is not None else StubSyntheticDataset()


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
    num_draws = 3
    x = days.reshape(1, -1, 1).expand(batch_size, -1, -1)
    x_normalized = (x - 212.5) / 151.5
    max_days = torch.tensor([280.0, 260.0]).reshape(batch_size, 1, 1)
    y = 1.0 - (x - max_days).abs() / 100.0
    draw_offsets = torch.linspace(0.0, 0.02, num_draws).reshape(1, num_draws, 1, 1)
    y_original = y[:, None] + draw_offsets
    lodged_mask = x[:, None] > max_days[:, None]
    height = tensordict.TensorDict(
        {
            "X": x,
            "X_normalized": x_normalized,
            "Y": y,
            "Y_original": y_original,
            "lodged_mask": lodged_mask,
        },
        batch_size=[batch_size],
    )
    return tensordict.TensorDict(
        {
            "height": height,
            "yearsite_uid": torch.tensor([10, 11]),
            "has_lodged": torch.tensor(
                [[False, True, False], [True, False, True]], dtype=torch.bool
            ),
            "genotype_id": torch.tensor([100, 101]),
        },
        batch_size=[batch_size],
    )


def _load_saved(base_path: Path, method_name: str):
    return BatchLoader(base_path / method_name)[0]


def _run_one_step(method_name: str, base_path: Path, **kwargs) -> None:
    batch_schema = kwargs.get("batch_schema", "compact")
    dataloader = kwargs.pop(
        "dataloader", StubDataloader([_make_batch()], StubSyntheticDataset())
    )
    common: dict[str, Any] = {
        "dataloader": dataloader,
        "split_name": "test",
        "base_path": base_path,
        "grid_points": create_grid_points(torch.device("cpu")),
        "model": StubModel(),
        "use_temperature": False,
        "use_marker": False,
        "num_batches": 1,
        "config": {"batch_schema": batch_schema},
    }
    common.update(kwargs)

    if method_name == no_context.METHOD_NAME:
        no_context.process_dataloader(
            accelerator=DummyAccelerator(),  # ty: ignore[invalid-argument-type]
            **common,
        )
    elif method_name == max_height.METHOD_NAME:
        max_height.process_dataloader(**common)
    elif method_name == random_context.METHOD_NAME:
        random_context.process_dataloader(
            num_samples=2, min_context=1, max_context=2, **common
        )
    else:  # pragma: no cover - parametrization guard
        raise ValueError(method_name)


@pytest.mark.parametrize("method_name", ONE_STEP_METHODS)
def test_one_step_default_outputs_use_compact_fp16_without_std_fields(
    tmp_path: Path, method_name: str
) -> None:
    _run_one_step(method_name, tmp_path)

    predictions, batch = _load_saved(tmp_path, method_name)

    assert predictions["grid"].dtype == torch.float16
    assert "grid_std" not in predictions
    assert "targets_std" not in predictions
    assert set(batch["data"].keys()) == {"genotype_id", "yearsite_uid"}
    assert batch["data"]["genotype_id"].tolist() == [100, 101]
    assert batch["data"]["yearsite_uid"].tolist() == [10, 11]
    assert "height" not in batch["data"]
    assert "has_lodged" not in batch["data"]
    assert batch["grid_points"]["X"].dtype == torch.float16
    config = BatchLoader(tmp_path / method_name).config
    assert config is not None
    assert config["batch_schema"] == "compact"
    assert config["dataset_identity"] == StubSyntheticDataset().get_dataset_identity()
    if method_name == random_context.METHOD_NAME:
        assert "context_indices" in batch


@pytest.mark.parametrize("method_name", ONE_STEP_METHODS)
def test_one_step_full_schema_override_preserves_full_metadata(
    tmp_path: Path, method_name: str
) -> None:
    _run_one_step(method_name, tmp_path, batch_schema="full")

    predictions, batch = _load_saved(tmp_path, method_name)

    assert predictions["grid"].dtype == torch.float16
    assert "grid_std" not in predictions
    assert "targets_std" not in predictions
    assert set(batch["data"].keys()) == {
        "genotype_id",
        "has_lodged",
        "height",
        "yearsite_uid",
    }
    assert batch["data"]["height"]["Y_original"].dtype == torch.float16
    assert batch["data"]["height"]["Y"].dtype == torch.float16
    assert batch["data"]["height"]["X"].dtype == torch.float16
    assert batch["data"]["height"]["X_normalized"].dtype == torch.float16
    assert batch["data"]["height"]["lodged_mask"].dtype == torch.bool
    assert batch["data"]["has_lodged"].dtype == torch.bool
    assert batch["grid_points"]["X"].dtype == torch.float16
    config = BatchLoader(tmp_path / method_name).config
    assert config is not None
    assert config["batch_schema"] == "full"
    assert "dataset_identity" not in config
    if method_name == random_context.METHOD_NAME:
        assert "context_indices" in batch


def test_full_schema_allows_identityless_dataloader(tmp_path: Path) -> None:
    _run_one_step(
        no_context.METHOD_NAME,
        tmp_path,
        dataloader=[_make_batch()],
        batch_schema="full",
    )

    _, batch = _load_saved(tmp_path, no_context.METHOD_NAME)
    assert "height" in batch["data"]
    config = BatchLoader(tmp_path / no_context.METHOD_NAME).config
    assert config is not None
    assert "dataset_identity" not in config


def test_compact_schema_records_per_split_dataset_identity(tmp_path: Path) -> None:
    shared_config = {"batch_schema": "compact"}
    _run_one_step(
        no_context.METHOD_NAME,
        tmp_path / "split_a",
        dataloader=StubDataloader(
            [_make_batch()], StubSyntheticDataset(cache_key="split-a")
        ),
        config=shared_config,
    )
    _run_one_step(
        no_context.METHOD_NAME,
        tmp_path / "split_b",
        dataloader=StubDataloader(
            [_make_batch()], StubSyntheticDataset(cache_key="split-b")
        ),
        config=shared_config,
    )

    config_a = BatchLoader(tmp_path / "split_a" / no_context.METHOD_NAME).config
    config_b = BatchLoader(tmp_path / "split_b" / no_context.METHOD_NAME).config
    assert config_a is not None
    assert config_b is not None

    assert config_a["dataset_identity"]["cache_key"] == "split-a"
    assert config_b["dataset_identity"]["cache_key"] == "split-b"
    assert "dataset_identity" not in shared_config


def test_one_step_storage_dtype_and_std_fields_are_configurable(tmp_path: Path) -> None:
    _run_one_step(
        no_context.METHOD_NAME,
        tmp_path,
        save_targets_std=True,
        save_grid_std=True,
        save_dtype=torch.float32,
        save_batch_dtype=torch.float32,
    )

    predictions, batch = _load_saved(tmp_path, no_context.METHOD_NAME)

    assert predictions["grid"].dtype == torch.float32
    assert predictions["grid_std"].dtype == torch.float32
    assert predictions["targets_std"].dtype == torch.float32
    assert set(batch["data"].keys()) == {"genotype_id", "yearsite_uid"}
    assert batch["grid_points"]["X"].dtype == torch.float32


@pytest.mark.parametrize("method_name", ONE_STEP_METHODS)
def test_invalid_batch_schema_raises_before_writing(
    tmp_path: Path, method_name: str
) -> None:
    with pytest.raises(ValueError, match="batch schema"):
        _run_one_step(method_name, tmp_path, batch_schema="expanded")

    assert not (tmp_path / method_name).exists()


def test_metric_loader_paths_consume_fp16_compact_schema_outputs(
    tmp_path: Path,
) -> None:
    _run_one_step(no_context.METHOD_NAME, tmp_path)
    method_dir = tmp_path / no_context.METHOD_NAME
    loader = BatchLoader(method_dir)

    gids, yss, grid = next(sig_mmd._iter_condition_batches(loader))  # noqa: SLF001
    assert gids == ["tensor(100)", "tensor(101)"]
    assert yss == ["tensor(10)", "tensor(11)"]
    assert grid.dtype == torch.float16

    t_norm = torch.linspace(0.0, 1.0, grid.shape[-1])
    paths = to_metric_paths(grid[0, :2], height_scale=1.0, t_norm=t_norm)
    assert torch.isfinite(torch.tensor(normalized_sig_mmd(paths, paths)))

    precompute_loader = BatchLoader(method_dir, load_predictions=False)
    precompute_gids, precompute_yss = next(
        precompute_sig_mahalanobis._iter_batch_condition_ids(  # noqa: SLF001
            precompute_loader
        )
    )
    assert precompute_gids == ["tensor(100)", "tensor(101)"]
    assert precompute_yss == ["tensor(10)", "tensor(11)"]
