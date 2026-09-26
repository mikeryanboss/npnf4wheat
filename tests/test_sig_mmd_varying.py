from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import tensordict
import torch

from npnf.metrics.signature import SYNTHETIC_SIGNATURE_HEIGHT_SCALE
from npnf.scripts.metrics import sig_mmd_varying


class _FakeSyntheticDataset:
    n_lodging_draws = 2

    def get_dataset_identity(self) -> dict[str, object]:
        return {
            "dataset_type": "SyntheticDataset",
            "cache_key": "fake-cache",
            "num_conditions": 2,
        }

    def get_condition_metadata(self) -> dict[str, list[str]]:
        return {"genotype_id": ["g0", "g1"], "yearsite_uid": ["y0", "y1"]}

    def get_height_days_all(self) -> torch.Tensor:
        return torch.tensor([0.0, 1.0, 2.0])

    def get_height_scale(self) -> torch.Tensor:
        return torch.tensor(200.0)

    def get_lodged_chunk(self, indices, draw_indices) -> dict[str, torch.Tensor]:
        all_heights = torch.tensor(
            [
                [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
                [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
            ]
        )
        all_lodged = torch.tensor([[False, True], [True, True]])
        indices = torch.as_tensor(list(indices), dtype=torch.long)
        draw_indices = torch.as_tensor(list(draw_indices), dtype=torch.long)
        return {
            "height_values_all_nonoise": all_heights[indices][:, draw_indices],
            "has_lodged": all_lodged[indices][:, draw_indices],
        }


def _write_method_artifact(base_path: Path, method_name: str) -> Path:
    method_dir = base_path / method_name
    batch_dir = method_dir / "predictions" / "000"
    batch_dir.mkdir(parents=True)

    rows = [("g0", "y0"), ("g1", "y1"), ("g0", "y0"), ("g1", "y1")]
    grid = torch.empty((4, 3, 2, 3, 1), dtype=torch.float32)
    for row_index in range(4):
        for context_index in range(3):
            for sample_index in range(2):
                grid[row_index, context_index, sample_index, :, 0] = torch.tensor(
                    [
                        row_index * 100.0 + context_index * 10.0 + sample_index,
                        row_index * 100.0 + context_index * 10.0 + sample_index + 1,
                        row_index * 100.0 + context_index * 10.0 + sample_index + 2,
                    ]
                )

    predictions = tensordict.TensorDict({"grid": grid}, batch_size=[4, 3])
    batch = {
        "data": tensordict.TensorDict(
            {  # ty: ignore[invalid-argument-type]
                "genotype_id": [gid for gid, _ in rows],
                "yearsite_uid": [ys for _, ys in rows],
            },
            batch_size=[4],
        ),
        "grid_points": tensordict.TensorDict(
            {"X": torch.tensor([[0.0], [1.0], [2.0]])}, batch_size=[3]
        ),
        "max_valid_context": torch.tensor([1, 2, 1, 2]),
        "context_indices": [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]],
    }
    config = {
        "trials_per_sample": 2,
        "dataloader_name": "fake_dataloader",
        "dataset_identity": _FakeSyntheticDataset().get_dataset_identity(),
    }

    torch.save(predictions, batch_dir / "predictions.pt")
    torch.save(batch, batch_dir / "batch.pt")
    (method_dir / "config.json").write_text(json.dumps(config))
    return method_dir


@pytest.fixture
def fake_sig_mmd_world(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    class _FakeOracleCfg:
        pass

    score_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_to_metric_paths(
        heights: torch.Tensor,
        height_scale: float,
        t_norm: torch.Tensor,
        *,
        lead_lag: bool = True,
    ) -> torch.Tensor:
        assert height_scale == SYNTHETIC_SIGNATURE_HEIGHT_SCALE
        return heights.float()

    def fake_normalized_sig_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
        score_calls.append((x.detach().clone(), y.detach().clone()))
        return float(y.mean())

    monkeypatch.setattr(
        sig_mmd_varying,
        "get_oracle_config_for_method_dir",
        lambda method_path: (_FakeOracleCfg, "fake_dataloader"),
    )
    monkeypatch.setattr(
        sig_mmd_varying, "instantiate", lambda cfg: _FakeSyntheticDataset()
    )
    monkeypatch.setattr(sig_mmd_varying, "to_metric_paths", fake_to_metric_paths)
    monkeypatch.setattr(sig_mmd_varying, "normalized_sig_mmd", fake_normalized_sig_mmd)
    return score_calls


def test_varying_sig_mmd_scores_grid_rows_against_condition_oracle(
    tmp_path: Path, fake_sig_mmd_world: list[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    method_dir = _write_method_artifact(tmp_path, "random")

    sig_mmd_varying.sig_mmd_varying_analysis(
        str(method_dir), n_samples=2, lead_lag=False
    )

    per_context_path = method_dir / "sig_mmd_per_context_condition.csv"
    by_context_path = method_dir / "sig_mmd_by_context.csv"
    assert per_context_path.exists()
    assert by_context_path.exists()
    assert not (method_dir / "synthetic_metrics.csv").exists()

    per_context = pl.read_csv(per_context_path).sort(["prediction_row", "num_context"])
    assert per_context.columns == [
        "num_context",
        "prediction_row",
        "condition_row",
        "trial_index",
        "genotype_id",
        "yearsite_uid",
        "oracle_lodging_rate",
        "score",
    ]
    assert per_context.height == 10
    assert per_context["prediction_row"].to_list() == [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    assert per_context["condition_row"].to_list() == [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    assert per_context["trial_index"].to_list() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    g0_rows = per_context.filter(pl.col("genotype_id") == "g0")
    g1_rows = per_context.filter(pl.col("genotype_id") == "g1")
    assert g0_rows["yearsite_uid"].unique().to_list() == ["y0"]
    assert g1_rows["yearsite_uid"].unique().to_list() == ["y1"]
    assert g0_rows["oracle_lodging_rate"].unique().to_list() == [0.0]
    assert g1_rows["oracle_lodging_rate"].unique().to_list() == [1.0]
    assert g0_rows["score"].unique().to_list() == [11.0]
    assert g1_rows["score"].unique().to_list() == [101.0]

    by_context = pl.read_csv(by_context_path)
    assert by_context.columns == ["num_context", "n", "mean", "median", "std", "p95"]
    assert by_context["num_context"].to_list() == [0, 1, 2]
    assert by_context["n"].to_list() == [4, 4, 2]
    assert by_context["mean"].to_list() == [56.0, 56.0, 101.0]

    assert len(fake_sig_mmd_world) == 10
    first_model_paths, first_oracle_paths = fake_sig_mmd_world[0]
    assert first_model_paths.shape == (2, 3)
    assert first_oracle_paths.tolist() == [[10.0, 11.0, 12.0]]
    assert all(oracle_paths.shape == (1, 3) for _, oracle_paths in fake_sig_mmd_world)


def test_parent_folder_processing_writes_method_outputs_and_summary(
    tmp_path: Path, fake_sig_mmd_world: list[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    _write_method_artifact(tmp_path, "random")
    _write_method_artifact(tmp_path, "uncertainty")

    sig_mmd_varying.sig_mmd_varying_analysis(str(tmp_path), n_samples=2, lead_lag=False)

    for method_name in ["random", "uncertainty"]:
        method_dir = tmp_path / method_name
        assert (method_dir / "sig_mmd_per_context_condition.csv").exists()
        assert (method_dir / "sig_mmd_by_context.csv").exists()

    summary_path = tmp_path / "sig_mmd_varying_summary.csv"
    assert summary_path.exists()
    summary = pl.read_csv(summary_path).sort(["method", "num_context"])
    assert summary["method"].to_list() == [
        "random",
        "random",
        "random",
        "uncertainty",
        "uncertainty",
        "uncertainty",
    ]
    assert summary["num_context"].to_list() == [0, 1, 2, 0, 1, 2]
    assert summary["n"].to_list() == [4, 4, 2, 4, 4, 2]
