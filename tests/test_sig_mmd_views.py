from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import polars as pl
import pysiglib
import pytest
import tensordict
import torch

from npnf.metrics import oracle_config
from npnf.metrics.sig_mmd import normalized_sig_mmd, unnormalized_sig_mmd
from npnf.metrics.signature import SYNTHETIC_SIGNATURE_HEIGHT_SCALE, to_metric_paths
from npnf.scripts.metrics import generate_oracle_self_predictions, sig_mmd


class _FakeMetadata:
    def __init__(self, data: dict[str, list[str]]):
        self._data = data

    def flatten(self) -> _FakeMetadata:
        return self

    def __getitem__(self, key: str) -> list[str]:
        return self._data[key]


class _FakePredictions:
    def __init__(self, grid: torch.Tensor):
        self._grid = grid

    def select(self, key: str) -> _FakePredictions:
        assert key == "grid"
        return self

    def flatten(self) -> dict[str, torch.Tensor]:
        return {"grid": self._grid}


class _FakeSamples(TypedDict):
    height_values_all_nonoise: torch.Tensor
    height_days_all: torch.Tensor
    has_lodged: torch.Tensor
    genotype_id: list[str]
    yearsite_uid: list[str]
    height_values_all_nonoise_nolodge: torch.Tensor


class _FakeDataset:
    def __init__(self, samples: _FakeSamples):
        self._samples = samples
        self.n_lodging_draws = int(samples["height_values_all_nonoise"].shape[1])
        self.lodging_seed = 142

    @property
    def samples(self):
        msg = "sig_mmd must use lazy dataset helpers, not samples"
        raise AssertionError(msg)

    def get_condition_metadata(self) -> dict[str, list[str]]:
        return {
            "genotype_id": list(self._samples["genotype_id"]),
            "yearsite_uid": list(self._samples["yearsite_uid"]),
        }

    def get_height_days_all(self) -> torch.Tensor:
        days = self._samples["height_days_all"]
        return days[0] if days.ndim == 2 else days

    def get_height_scale(self) -> torch.Tensor:
        return self._samples["height_values_all_nonoise_nolodge"].float().max()

    def get_lodged_chunk(self, indices, draw_indices):
        indices = torch.as_tensor(list(indices), dtype=torch.long)
        draw_indices = torch.as_tensor(list(draw_indices), dtype=torch.long)
        if (draw_indices >= self.n_lodging_draws).any():
            msg = "draw index out of bounds"
            raise IndexError(msg)
        return {
            "height_values_all_nonoise": self._samples["height_values_all_nonoise"][
                indices
            ][:, draw_indices],
            "height_days_all": self.get_height_days_all(),
            "has_lodged": self._samples["has_lodged"][indices][:, draw_indices],
            "height_values_all_nonoise_nolodge": self._samples[
                "height_values_all_nonoise_nolodge"
            ]
            .reshape(-1)[0]
            .expand(len(indices), self.get_height_days_all().numel()),
            "genotype_id": [
                self._samples["genotype_id"][int(index)] for index in indices
            ],
            "yearsite_uid": [
                self._samples["yearsite_uid"][int(index)] for index in indices
            ],
        }


def _install_fake_world(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    method_name: str = "no_context",
    use_temperature: bool = False,
    use_marker: bool = False,
    oracle_draws: int = 2,
) -> tuple[Path, list[tuple[torch.Tensor, torch.Tensor]], float]:
    method_dir = tmp_path / method_name
    method_dir.mkdir()

    day_axis = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).unsqueeze(-1)
    sorted_ids = [("g1", "e1"), ("g1", "e2"), ("g2", "e1"), ("g2", "e2")]
    loader_order = [("g2", "e1"), ("g1", "e2"), ("g1", "e1"), ("g2", "e2")]

    model_heights = {
        ("g1", "e1"): torch.tensor([[11.0, 12.0, 13.0], [111.0, 112.0, 113.0]]),
        ("g1", "e2"): torch.tensor([[21.0, 22.0, 23.0], [121.0, 122.0, 123.0]]),
        ("g2", "e1"): torch.tensor([[31.0, 32.0, 33.0], [131.0, 132.0, 133.0]]),
        ("g2", "e2"): torch.tensor([[41.0, 42.0, 43.0], [141.0, 142.0, 143.0]]),
    }
    oracle_heights = {
        ("g1", "e1"): torch.tensor([[15.0, 16.0, 17.0], [115.0, 116.0, 117.0]]),
        ("g1", "e2"): torch.tensor([[25.0, 26.0, 27.0], [125.0, 126.0, 127.0]]),
        ("g2", "e1"): torch.tensor([[35.0, 36.0, 37.0], [135.0, 136.0, 137.0]]),
        ("g2", "e2"): torch.tensor([[45.0, 46.0, 47.0], [145.0, 146.0, 147.0]]),
    }

    grid = torch.stack([model_heights[key] for key in loader_order], dim=0).unsqueeze(
        -1
    )
    fake_batch = (
        _FakePredictions(grid),
        {
            "data": _FakeMetadata(
                {
                    "genotype_id": [gid for gid, _ in loader_order],
                    "yearsite_uid": [ys for _, ys in loader_order],
                }
            ),
            "grid_points": {"X": day_axis},
        },
    )

    class _FakeBatchLoader:
        def __init__(self, method_dir: Path):
            self.method_dir = Path(method_dir)

        def load_batch(self, index: int):
            assert index == 0
            return fake_batch

        def __iter__(self):
            return iter([fake_batch])

        def __len__(self) -> int:
            return 1

        @property
        def config(self) -> dict[str, bool]:
            return {"use_temperature": use_temperature, "use_marker": use_marker}

    dataset_samples: _FakeSamples = {
        "height_values_all_nonoise": torch.stack(
            [oracle_heights[key] for key in sorted_ids], dim=0
        )[:, :oracle_draws],
        "height_days_all": day_axis.squeeze(-1).repeat(len(sorted_ids), 1),
        "has_lodged": torch.tensor([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=torch.bool)[
            :, :oracle_draws
        ],
        "genotype_id": [gid for gid, _ in sorted_ids],
        "yearsite_uid": [ys for _, ys in sorted_ids],
        "height_values_all_nonoise_nolodge": torch.full((1, 1, 1), 200.0),
    }

    class _FakeOracleCfg:
        __name__ = "FakeOracleCfg"

    score_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_normalized_sig_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
        score_calls.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
        return float(x.shape[0] * 1000 + y.shape[0])

    def fake_unnormalized_sig_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
        score_calls.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
        return float(x.shape[0] * 1000 + y.shape[0] + 0.5)

    monkeypatch.setattr(sig_mmd, "BatchLoader", _FakeBatchLoader)
    monkeypatch.setattr(
        sig_mmd,
        "get_oracle_config_for_method_dir",
        lambda method_path: (_FakeOracleCfg, "fake_dataloader"),
    )
    monkeypatch.setattr(
        sig_mmd, "instantiate", lambda cfg: _FakeDataset(dataset_samples)
    )
    monkeypatch.setattr(
        sig_mmd, "align_trajectories_to_day_axis", lambda trajectories, *_: trajectories
    )
    monkeypatch.setattr(sig_mmd, "normalized_sig_mmd", fake_normalized_sig_mmd)
    monkeypatch.setattr(sig_mmd, "unnormalized_sig_mmd", fake_unnormalized_sig_mmd)

    return method_dir, score_calls, SYNTHETIC_SIGNATURE_HEIGHT_SCALE


def _install_fake_oracle_self_world(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    method_name: str = "no_context",
    use_temperature: bool = False,
    use_marker: bool = False,
) -> tuple[Path, list[tuple[torch.Tensor, torch.Tensor]]]:
    method_dir = tmp_path / method_name
    method_dir.mkdir()

    day_axis = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).unsqueeze(-1)
    sorted_ids = [("g1", "e1"), ("g1", "e2"), ("g2", "e1"), ("g2", "e2")]

    def oracle_heights(base: float, condition_step: float) -> torch.Tensor:
        rows = []
        for condition_index in range(len(sorted_ids)):
            draws = []
            for draw_index in range(4):
                first = base + condition_index * condition_step + draw_index
                draws.append(torch.tensor([first, first + 0.1, first + 0.2]))
            rows.append(torch.stack(draws, dim=0))
        return torch.stack(rows, dim=0)

    batch_dict = {
        "data": _FakeMetadata(
            {
                "genotype_id": [gid for gid, _ in sorted_ids],
                "yearsite_uid": [ys for _, ys in sorted_ids],
            }
        ),
        "grid_points": {"X": day_axis},
    }

    class _FakeBatchLoader:
        def __init__(self, method_dir: Path, *, load_predictions: bool = True):
            self.method_dir = Path(method_dir)
            self.load_predictions = load_predictions

        def load_batch(self, index: int):
            assert index == 0
            if not self.load_predictions:
                return batch_dict
            grid = torch.zeros((len(sorted_ids), 1, 3, 1), dtype=torch.float32)
            return _FakePredictions(grid), batch_dict

        def __iter__(self):
            return iter([self.load_batch(0)])

        def __len__(self) -> int:
            return 1

        @property
        def config(self) -> dict[str, bool]:
            return {"use_temperature": use_temperature, "use_marker": use_marker}

    seed_a_samples: _FakeSamples = {
        "height_values_all_nonoise": oracle_heights(1000.0, 100.0),
        "height_days_all": day_axis.squeeze(-1).repeat(len(sorted_ids), 1),
        "has_lodged": torch.zeros((len(sorted_ids), 4), dtype=torch.bool),
        "genotype_id": [gid for gid, _ in sorted_ids],
        "yearsite_uid": [ys for _, ys in sorted_ids],
        "height_values_all_nonoise_nolodge": torch.full((1, 1, 1), 200.0),
    }
    seed_b_samples: _FakeSamples = {
        "height_values_all_nonoise": oracle_heights(100.0, 10.0),
        "height_days_all": day_axis.squeeze(-1).repeat(len(sorted_ids), 1),
        "has_lodged": torch.zeros((len(sorted_ids), 4), dtype=torch.bool),
        "genotype_id": [gid for gid, _ in sorted_ids],
        "yearsite_uid": [ys for _, ys in sorted_ids],
        "height_values_all_nonoise_nolodge": torch.full((1, 1, 1), 200.0),
    }
    seed_a_dataset = _FakeDataset(seed_a_samples)
    seed_b_dataset = _FakeDataset(seed_b_samples)

    class _FakeSeedACfg:
        __name__ = "FakeSeedA"

    class _FakeSeedBCfg:
        __name__ = "FakeSeedB"

    def fake_instantiate(cfg):
        return seed_b_dataset if cfg is _FakeSeedBCfg else seed_a_dataset

    score_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_normalized_sig_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
        score_calls.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
        return float(x.shape[0] * 1000 + y.shape[0])

    def fake_unnormalized_sig_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
        score_calls.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
        return float(x.shape[0] * 1000 + y.shape[0] + 0.5)

    monkeypatch.setattr(sig_mmd, "BatchLoader", _FakeBatchLoader)
    monkeypatch.setattr(
        sig_mmd,
        "get_oracle_config_for_method_dir",
        lambda method_path: (_FakeSeedACfg, "fake_dataloader"),
    )
    monkeypatch.setattr(
        sig_mmd,
        "get_seed_b_config_for_dataloader_name",
        lambda dataloader_name: _FakeSeedBCfg,
    )
    monkeypatch.setattr(sig_mmd, "instantiate", fake_instantiate)
    monkeypatch.setattr(
        sig_mmd, "align_trajectories_to_day_axis", lambda trajectories, *_: trajectories
    )
    monkeypatch.setattr(sig_mmd, "normalized_sig_mmd", fake_normalized_sig_mmd)
    monkeypatch.setattr(sig_mmd, "unnormalized_sig_mmd", fake_unnormalized_sig_mmd)

    return method_dir, score_calls


def _first_heights(paths: torch.Tensor) -> list[float]:
    return (paths[:, 0, 0] * SYNTHETIC_SIGNATURE_HEIGHT_SCALE).tolist()


def _seed_b_condition_bases(paths: torch.Tensor) -> set[int]:
    return {100 + int((height - 100.0) // 10) * 10 for height in _first_heights(paths)}


def test_condition_view_writes_legacy_outputs(monkeypatch, tmp_path):
    method_dir, _, _ = _install_fake_world(monkeypatch, tmp_path)
    output_dir = tmp_path / "condition"

    sig_mmd.sig_mmd_analysis(
        method_dir=str(method_dir),
        n_samples=2,
        output_folder=str(output_dir),
        lead_lag=False,
    )

    per_x_path = output_dir / "sig_mmd_per_x.csv"
    summary_path = output_dir / "sig_mmd_summary.csv"
    assert per_x_path.exists()
    assert summary_path.exists()
    assert not (output_dir / "sig_mmd_per_unit.csv").exists()
    assert not (output_dir / "sig_mmd_unit_members.csv").exists()

    per_x_df = pl.read_csv(per_x_path)
    assert per_x_df["genotype_id"].to_list() == ["g1", "g1", "g2", "g2"]
    assert per_x_df["yearsite_uid"].to_list() == ["e1", "e2", "e1", "e2"]

    summary = pl.read_csv(summary_path)
    assert summary["view"].to_list() == ["condition"]
    assert summary["n_units"].to_list() == [4]
    assert summary["n_conditions"].to_list() == [4]


def test_condition_view_default_output_folder_uses_normalized_name(
    monkeypatch, tmp_path
):
    method_dir, _, _ = _install_fake_world(monkeypatch, tmp_path)

    sig_mmd.sig_mmd_analysis(method_dir=str(method_dir), n_samples=2, lead_lag=False)

    assert (method_dir / "sig_mmd" / "sig_mmd_per_x.csv").exists()
    assert (method_dir / "sig_mmd" / "sig_mmd_summary.csv").exists()
    assert not (method_dir / "sig_mmd_unnormalized").exists()


def test_condition_view_unnormalized_uses_separate_default_output_folder(
    monkeypatch, tmp_path
):
    method_dir, _, _ = _install_fake_world(monkeypatch, tmp_path)

    sig_mmd.sig_mmd_analysis(
        method_dir=str(method_dir),
        n_samples=2,
        lead_lag=False,
        normalize_sig_kernel=False,
    )

    per_x_path = method_dir / "sig_mmd_unnormalized" / "sig_mmd_per_x.csv"
    summary_path = method_dir / "sig_mmd_unnormalized" / "sig_mmd_summary.csv"
    assert per_x_path.exists()
    assert summary_path.exists()
    assert "score" in pl.read_csv(per_x_path).columns
    assert "mean" in pl.read_csv(summary_path).columns


def test_condition_view_unnormalized_selects_unnormalized_scorer(monkeypatch, tmp_path):
    method_dir, _, _ = _install_fake_world(monkeypatch, tmp_path)
    scorer_names: list[str] = []

    def fake_normalized(x: torch.Tensor, y: torch.Tensor) -> float:
        scorer_names.append("normalized")
        return 1.0

    def fake_unnormalized(x: torch.Tensor, y: torch.Tensor) -> float:
        scorer_names.append("unnormalized")
        return 2.0

    monkeypatch.setattr(sig_mmd, "normalized_sig_mmd", fake_normalized)
    monkeypatch.setattr(sig_mmd, "unnormalized_sig_mmd", fake_unnormalized)

    sig_mmd.sig_mmd_analysis(
        method_dir=str(method_dir),
        n_samples=2,
        output_folder=str(tmp_path / "unnormalized"),
        lead_lag=False,
        normalize_sig_kernel=False,
    )

    assert scorer_names == ["unnormalized"] * 4


@pytest.mark.parametrize(
    ("use_temperature", "use_marker", "expected_seed_b_bases"),
    [
        (False, False, {100, 110, 120, 130}),
        (False, True, {100, 110}),
        (True, False, {100, 120}),
        (True, True, {100}),
    ],
)
def test_oracle_self_uses_covariate_pool_and_exact_seed_a_target(
    monkeypatch, tmp_path, use_temperature, use_marker, expected_seed_b_bases
):
    method_dir, score_calls = _install_fake_oracle_self_world(
        monkeypatch, tmp_path, use_temperature=use_temperature, use_marker=use_marker
    )

    sig_mmd.sig_mmd_analysis(
        method_dir=str(method_dir), n_samples=4, lead_lag=False, oracle_self=True
    )

    assert len(score_calls) == 4
    seed_b_paths, seed_a_paths = score_calls[0]
    assert _seed_b_condition_bases(seed_b_paths) == expected_seed_b_bases
    assert _first_heights(seed_a_paths) == [1000.0, 1001.0, 1002.0, 1003.0]


def test_oracle_self_reuses_same_seed_b_unit_for_same_genotype(monkeypatch, tmp_path):
    method_dir, score_calls = _install_fake_oracle_self_world(
        monkeypatch, tmp_path, use_temperature=False, use_marker=True
    )

    sig_mmd.sig_mmd_analysis(
        method_dir=str(method_dir), n_samples=4, lead_lag=False, oracle_self=True
    )

    g1e1_seed_b, g1e1_seed_a = score_calls[0]
    g1e2_seed_b, g1e2_seed_a = score_calls[1]
    g2e1_seed_b, _ = score_calls[2]

    assert torch.equal(g1e1_seed_b, g1e2_seed_b)
    assert not torch.equal(g1e1_seed_b, g2e1_seed_b)
    assert _first_heights(g1e1_seed_a) == [1000.0, 1001.0, 1002.0, 1003.0]
    assert _first_heights(g1e2_seed_a) == [1100.0, 1101.0, 1102.0, 1103.0]


def test_oracle_self_default_output_folders(monkeypatch, tmp_path):
    method_dir, _ = _install_fake_oracle_self_world(monkeypatch, tmp_path)

    sig_mmd.sig_mmd_analysis(
        method_dir=str(method_dir), n_samples=4, lead_lag=False, oracle_self=True
    )
    sig_mmd.sig_mmd_analysis(
        method_dir=str(method_dir),
        n_samples=4,
        lead_lag=False,
        oracle_self=True,
        normalize_sig_kernel=False,
    )

    assert (method_dir / "sig_mmd_oracle_self" / "sig_mmd_per_x.csv").exists()
    assert (
        method_dir / "sig_mmd_oracle_self_unnormalized" / "sig_mmd_summary.csv"
    ).exists()


def test_oracle_self_rejects_context_method_dirs(monkeypatch, tmp_path):
    method_dir, _ = _install_fake_oracle_self_world(
        monkeypatch, tmp_path, method_name="random_context"
    )

    with pytest.raises(ValueError, match="context-aware oracle baselines"):
        sig_mmd.sig_mmd_analysis(
            method_dir=str(method_dir),
            n_samples=4,
            output_folder=str(tmp_path / "oracle-self"),
            lead_lag=False,
            oracle_self=True,
        )


def test_oracle_self_rejects_unsupported_seed_b_dataloader(monkeypatch, tmp_path):
    method_dir, _ = _install_fake_oracle_self_world(monkeypatch, tmp_path)

    class _FakeSeedACfg:
        __name__ = "FakeSeedA"

    monkeypatch.setattr(
        sig_mmd,
        "get_oracle_config_for_method_dir",
        lambda method_path: (_FakeSeedACfg, "synth_test_plot_no_lodging_dataloaders"),
    )
    monkeypatch.setattr(
        sig_mmd,
        "get_seed_b_config_for_dataloader_name",
        oracle_config.get_seed_b_config_for_dataloader_name,
    )

    with pytest.raises(ValueError, match="synth_test_plot_no_lodging_dataloaders"):
        sig_mmd.sig_mmd_analysis(
            method_dir=str(method_dir),
            n_samples=4,
            output_folder=str(tmp_path / "oracle-self"),
            lead_lag=False,
            oracle_self=True,
        )


def test_condition_view_clamps_requested_draws_to_oracle_availability(
    monkeypatch, tmp_path
):
    method_dir, score_calls, _ = _install_fake_world(
        monkeypatch, tmp_path, oracle_draws=1
    )
    output_dir = tmp_path / "condition-one-draw"

    sig_mmd.sig_mmd_analysis(
        method_dir=str(method_dir),
        n_samples=2,
        output_folder=str(output_dir),
        lead_lag=False,
    )

    assert score_calls
    for model_paths, oracle_paths in score_calls:
        assert model_paths.shape[0] == 1
        assert oracle_paths.shape[0] == 1


def test_to_metric_paths_uses_float32_without_lead_lag():
    paths = to_metric_paths(
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        3.0,
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        lead_lag=False,
    )

    assert paths.dtype == torch.float32


def _example_metric_paths(sample_count: int, offset: float) -> torch.Tensor:
    heights = torch.arange(sample_count * 3, dtype=torch.float32).reshape(
        sample_count, 3
    )
    heights = heights + offset
    return to_metric_paths(
        heights,
        20.0,
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
        lead_lag=False,
    )


def _raw_kernel_grams(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def kernel_gram(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        gram = pysiglib.sig_kernel_gram(
            left.float(),
            right.float(),
            dyadic_order=2,
            static_kernel=pysiglib.RBFKernel(sigma=1.0),
            time_aug=False,
            lead_lag=False,
        )
        return cast(torch.Tensor, gram)

    k_xx = kernel_gram(x, x)
    k_xy = kernel_gram(x, y)
    k_yy = kernel_gram(y, y)
    return k_xx, k_xy, k_yy


def _normalized_kernel_grams(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_xx, k_xy, k_yy = _raw_kernel_grams(x, y)

    diag_x = k_xx.diagonal()
    diag_y = k_yy.diagonal()
    k_xx = k_xx / torch.sqrt(torch.outer(diag_x, diag_x))
    k_xy = k_xy / torch.sqrt(torch.outer(diag_x, diag_y))
    k_yy = k_yy / torch.sqrt(torch.outer(diag_y, diag_y))
    return k_xx, k_xy, k_yy


@pytest.mark.parametrize(("m", "n"), [(1, 1), (1, 3), (3, 1), (3, 3)])
def test_normalized_sig_mmd_handles_singleton_sample_counts(m, n):
    # Copies of one path are the same distribution, so MMD^2 must be zero for every
    # sample count, including the singleton self-term cases.
    path = _example_metric_paths(1, 1.0)

    score = normalized_sig_mmd(path.repeat(m, 1, 1), path.repeat(n, 1, 1))

    assert score == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(("m", "n"), [(1, 1), (1, 3), (3, 1), (3, 3)])
def test_unnormalized_sig_mmd_uses_raw_kernel_grams(m, n):
    path = _example_metric_paths(1, 1.0)
    other = _example_metric_paths(1, 5.0)

    score = unnormalized_sig_mmd(path.repeat(m, 1, 1), path.repeat(n, 1, 1))

    assert score == pytest.approx(0.0, abs=1e-5)
    # Without self-kernel normalisation the score differs from the normalised one.
    assert unnormalized_sig_mmd(path, other) != pytest.approx(
        normalized_sig_mmd(path, other), abs=1e-3
    )


def test_normalized_sig_mmd_one_vs_one_uses_dirac_formula():
    paths_x = _example_metric_paths(1, 1.0)
    paths_y = _example_metric_paths(1, 5.0)

    _, k_xy, _ = _normalized_kernel_grams(paths_x, paths_y)
    expected = 2 - 2 * k_xy[0, 0]

    assert normalized_sig_mmd(paths_x, paths_y) == pytest.approx(
        float(expected), abs=1e-6
    )


def test_generate_oracle_self_predictions_streams_chunks(monkeypatch, tmp_path):
    day_axis = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).unsqueeze(-1)
    samples: _FakeSamples = {
        "height_values_all_nonoise": torch.arange(
            4 * 2 * 3, dtype=torch.float32
        ).reshape(4, 2, 3),
        "height_days_all": day_axis.squeeze(-1).repeat(4, 1),
        "has_lodged": torch.tensor([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=torch.bool),
        "genotype_id": ["g1", "g1", "g2", "g2"],
        "yearsite_uid": ["e1", "e2", "e1", "e2"],
        "height_values_all_nonoise_nolodge": torch.full((1, 1, 1), 200.0),
    }
    dataset = _FakeDataset(samples)
    grid_points = tensordict.TensorDict({"X": day_axis}, batch_size=[])
    saved_batches = []

    monkeypatch.setattr(
        generate_oracle_self_predictions,
        "load_grid_points",
        lambda reference_method_dir: grid_points,
    )
    monkeypatch.setattr(
        generate_oracle_self_predictions, "instantiate", lambda cfg: dataset
    )

    def fake_save_batch_results(**kwargs):
        saved_batches.append(kwargs)

    monkeypatch.setattr(
        generate_oracle_self_predictions, "save_batch_results", fake_save_batch_results
    )

    generate_oracle_self_predictions.generate_oracle_self_predictions(
        seed_b_config=object(),
        reference_method_dir=tmp_path / "reference",
        dataloader_name="fake_dataloader",
        batch_size=2,
    )

    assert len(saved_batches) == 2
    assert saved_batches[0]["config"]["num_samples"] == 2
    assert saved_batches[1]["config"] is None
    assert saved_batches[0]["predictions"]["grid"].shape == torch.Size([2, 1, 2, 3, 1])
    assert saved_batches[1]["predictions"]["grid"].shape == torch.Size([2, 1, 2, 3, 1])
    assert saved_batches[0]["batch_data"]["genotype_id"] == ["g1", "g1"]
    assert saved_batches[1]["batch_data"]["genotype_id"] == ["g2", "g2"]
