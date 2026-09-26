from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from npnf.metrics import blocked_scoring
from npnf.metrics.blocked_scoring import BlockedScoringOptions, context_weights
from npnf.metrics.blocks import UnitScope, build_unit_blocks
from npnf.scripts.metrics import sig_mmd_context_matched_blocked as blocked


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


class _FakeDataset:
    """Synthetic-dataset stand-in with a configurable dataset flat-index order.

    ``key_order`` is the genotype-major dataset order, so condition ``(gid, ys)`` has
    dataset flat index ``key_order.index((gid, ys))``. Oracle lodging draws encode that
    flat index as ``offset + condition * 100 + draw`` so tests can prove which dataset
    and condition were loaded.
    """

    def __init__(self, key_order: list[tuple[str, str]], *, offset: float = 0.0):
        self._keys = key_order
        self._offset = offset
        self.num_genotypes = len({gid for gid, _ in key_order})
        self.num_yearsites = len({ys for _, ys in key_order})
        self.n_lodging_draws = 4
        self.scale_noise = 0.5
        self._height_days = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        self._subsample_indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
        clean = torch.stack(
            [
                torch.tensor([u0 + 1.0, u0 + 2.0, u0 + 3.0], dtype=torch.float32)
                for u0 in range(len(key_order))
            ],
            dim=0,
        )
        self._intermediate = {"heights_clean": clean}
        self._noise = torch.zeros_like(clean)
        self.loaded_indices: list[list[int]] = []

    def get_condition_metadata(self) -> dict[str, list[str]]:
        return {
            "genotype_id": [gid for gid, _ in self._keys],
            "yearsite_uid": [ys for _, ys in self._keys],
        }

    def get_height_days_all(self) -> torch.Tensor:
        return self._height_days

    def get_lodged_chunk(self, indices, draw_indices):
        indices = [int(i) for i in indices]
        draw_indices = [int(i) for i in draw_indices]
        self.loaded_indices.append(indices)
        days = self._height_days.numel()
        draws = torch.stack(
            [
                torch.stack(
                    [
                        torch.full((days,), float(self._offset + cond * 100 + draw))
                        for draw in draw_indices
                    ],
                    dim=0,
                )
                for cond in indices
            ],
            dim=0,
        )
        return {"height_values_all_nonoise": draws}


class _FakeOracleCfg:
    __name__ = "FakeOracleCfg"


class _FakeSeedBCfg:
    __name__ = "FakeSeedBCfg"


def _install_fake_world(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    method_name: str,
    use_temperature: bool,
    use_marker: bool,
    dataset_key_order: list[tuple[str, str]],
    loader_key_order: list[tuple[str, str]],
    context_indices: list[list[int]] | None = None,
    model_draws: int = 4,
    capture: list | None = None,
    seed_b_key_order: list[tuple[str, str]] | None = None,
    forbid_prediction_reads: bool = False,
    loader_load_predictions: list[bool] | None = None,
) -> _FakeDataset:
    method_dir = tmp_path / method_name
    method_dir.mkdir(parents=True)
    dataset = _FakeDataset(dataset_key_order)
    seed_b_dataset = _FakeDataset(
        seed_b_key_order or dataset_key_order, offset=10_000.0
    )
    day_axis = dataset.get_height_days_all().unsqueeze(-1)

    # Model heights encode the predicted condition: row r, draw d -> r*10 + d.
    grid = torch.stack(
        [
            torch.stack(
                [
                    torch.full((3,), float(row * 10 + draw))
                    for draw in range(model_draws)
                ],
                dim=0,
            )
            for row in range(len(loader_key_order))
        ],
        dim=0,
    ).unsqueeze(-1)

    batch_dict: dict = {
        "data": _FakeMetadata(
            {
                "genotype_id": [gid for gid, _ in loader_key_order],
                "yearsite_uid": [ys for _, ys in loader_key_order],
            }
        ),
        "grid_points": {"X": day_axis},
    }
    if context_indices is not None:
        batch_dict["context_indices"] = context_indices
    batch = (_FakePredictions(grid), batch_dict)

    class _FakeBatchLoader:
        def __init__(
            self,
            method_dir: Path,
            *,
            load_predictions: bool = True,
            load_batch_data: bool = True,
        ):
            self.method_dir = Path(method_dir)
            self.load_predictions = load_predictions
            self.load_batch_data = load_batch_data
            if loader_load_predictions is not None:
                loader_load_predictions.append(load_predictions)

        def _item(self):
            if self.load_predictions:
                if forbid_prediction_reads:
                    msg = "predictions.pt read"
                    raise AssertionError(msg)
                return batch
            return batch_dict

        def load_batch(self, index: int):
            assert index == 0
            return self._item()

        def __iter__(self):
            return iter([self._item()])

        def __len__(self) -> int:
            return 1

        @property
        def config(self) -> dict[str, bool]:
            return {"use_temperature": use_temperature, "use_marker": use_marker}

    monkeypatch.setattr(blocked, "BatchLoader", _FakeBatchLoader)
    monkeypatch.setattr(
        blocked,
        "get_oracle_config_for_method_dir",
        lambda method_path: (_FakeOracleCfg, "fake_dataloader"),
    )
    monkeypatch.setattr(
        blocked,
        "get_seed_b_config_for_dataloader_name",
        lambda dataloader_name: _FakeSeedBCfg,
    )
    monkeypatch.setattr(
        blocked,
        "instantiate",
        lambda cfg: seed_b_dataset if cfg is _FakeSeedBCfg else dataset,
    )

    if capture is not None:

        def fake_normalized_sig_mmd(
            x: torch.Tensor, y: torch.Tensor, *, sigma: float
        ) -> float:
            del sigma
            capture.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
            return float(x.shape[0] * 1000 + y.shape[0])

        monkeypatch.setattr(blocked, "normalized_sig_mmd", fake_normalized_sig_mmd)

    return dataset


_LEX_DATASET = [("g1", "e1"), ("g1", "e2"), ("g2", "e1"), ("g2", "e2")]


def test_rejects_unknown_method_dir_name(tmp_path) -> None:
    bad = tmp_path / "some_other_method"
    bad.mkdir()
    with pytest.raises(ValueError, match="requires a method directory named"):
        blocked.sig_mmd_context_matched_blocked_analysis(method_dir=str(bad))
    assert not (bad / "sig_mmd_context_matched_blocked").exists()


@pytest.mark.parametrize(
    ("use_temperature", "use_marker", "expected_scope"),
    [
        (False, False, "global"),
        (True, False, "environment"),
        (False, True, "genotype"),
        (True, True, "condition"),
    ],
)
def test_infer_scope_for_four_covariate_settings(
    use_temperature, use_marker, expected_scope
) -> None:
    scope = UnitScope.from_config(
        {"use_temperature": use_temperature, "use_marker": use_marker},
        Path("max_height"),
    )
    assert scope == expected_scope


def test_weights_from_clean_pool_matches_context_weights() -> None:
    # P2: the local helper that reuses a precomputed clean[pool] must be bit-identical
    # to context_weights(clean, pool, ...) for both empty and non-empty context.
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(6, 5))
    pool = np.array([4, 1, 3, 0], dtype=int)
    sigma = 0.7

    ctx_idx = np.array([2, 4], dtype=int)
    ctx_vals = np.array([0.3, -1.2])
    w_ref, ess_ref = context_weights(clean, pool, ctx_idx, ctx_vals, sigma)
    w, ess = blocked_scoring.weights_from_clean_pool(
        clean[pool], ctx_idx, ctx_vals, sigma
    )
    assert np.array_equal(w, w_ref)
    assert ess == ess_ref

    empty_idx = np.array([], dtype=int)
    empty_vals = np.array([])
    w0_ref, ess0_ref = context_weights(clean, pool, empty_idx, empty_vals, sigma)
    w0, ess0 = blocked_scoring.weights_from_clean_pool(
        clean[pool], empty_idx, empty_vals, sigma
    )
    assert np.array_equal(w0, w0_ref)
    assert ess0 == ess0_ref


def _tile_inputs(seed: int):
    rng = np.random.default_rng(seed)
    clean_pool = rng.normal(size=(40, 12))
    sigma = 0.8
    ctx_idx_list = [
        np.array([0, 3, 7], dtype=int),
        np.arange(9, dtype=int),
        np.array([], dtype=int),  # empty context -> uniform, ess == pool_size
        np.array([5], dtype=int),
    ]
    ctx_vals_list = [
        clean_pool[2, idx] + 0.01 * rng.normal(size=idx.size) for idx in ctx_idx_list
    ]
    return clean_pool, ctx_idx_list, ctx_vals_list, sigma


def _run_tile(clean_pool, ctx_idx_list, ctx_vals_list, sigma, device):
    clean_t = torch.as_tensor(clean_pool, dtype=torch.float64, device=device)
    clean_t = clean_t.t().contiguous()
    return blocked_scoring.tile_posterior(
        clean_t,
        clean_t * clean_t,
        ctx_idx_list,
        ctx_vals_list,
        sigma,
        device,
        clean_pool.shape[0],
    )


def test_tile_posterior_matches_per_target_weights() -> None:
    # The vectorized tiled posterior is numerically equivalent (not bit-identical)
    # to the per-target weights_from_clean_pool, and reproduces the empty-context
    # short-circuit (uniform weights, ess == pool_size) exactly.
    clean_pool, ctx_idx_list, ctx_vals_list, sigma = _tile_inputs(1)
    pool_size = clean_pool.shape[0]
    weights, ess = _run_tile(
        clean_pool, ctx_idx_list, ctx_vals_list, sigma, torch.device("cpu")
    )
    pool = np.arange(pool_size)
    days = np.arange(clean_pool.shape[1], dtype=float)
    peak_pool = days[clean_pool.argmax(axis=1)]
    for i, (idx, vals) in enumerate(zip(ctx_idx_list, ctx_vals_list, strict=True)):
        w_ref, ess_ref = blocked_scoring.weights_from_clean_pool(
            clean_pool, idx, vals, sigma
        )
        assert np.abs(weights[i] - w_ref).max() <= 1e-9
        assert abs(ess[i] - ess_ref) / ess_ref <= 1e-9
        if idx.size == 0:
            post_peak_mass = post_peak_mass_ref = 0.0
        else:
            latest_context_day = float(days[idx].max())
            post_peak_mass = float(weights[i][peak_pool < latest_context_day].sum())
            post_peak_mass_ref = float(w_ref[peak_pool < latest_context_day].sum())
        assert abs(post_peak_mass - post_peak_mass_ref) <= 1e-9
        rng = np.random.default_rng(100 + i)
        rng_ref = np.random.default_rng(100 + i)
        assert np.array_equal(
            rng.choice(pool, size=8, p=weights[i]),
            rng_ref.choice(pool, size=8, p=w_ref),
        )
    assert np.array_equal(weights[2], np.full(pool_size, 1.0 / pool_size))
    assert ess[2] == float(pool_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tile_posterior_gpu_matches_cpu() -> None:
    # GPU and CPU posteriors are numerically equivalent (float64).
    clean_pool, ctx_idx_list, ctx_vals_list, sigma = _tile_inputs(2)
    w_cpu, ess_cpu = _run_tile(
        clean_pool, ctx_idx_list, ctx_vals_list, sigma, torch.device("cpu")
    )
    w_gpu, ess_gpu = _run_tile(
        clean_pool, ctx_idx_list, ctx_vals_list, sigma, torch.device("cuda")
    )
    assert np.abs(w_cpu - w_gpu).max() <= 1e-6
    assert np.abs(ess_cpu - ess_gpu).max() <= 1e-6


def test_load_oracle_draws_loads_only_needed_pairs_grouped_by_drawset() -> None:
    # P1: only the requested (condition, draw_index) pairs are loaded, and conditions
    # sharing an identical draw-index set are batched into a single get_lodged_chunk
    # call. Day axes match, so aligned values equal the cond*100+draw encoding.
    dataset = _FakeDataset([("g0", "y0"), ("g1", "y0"), ("g0", "y1"), ("g1", "y1")])
    day_axis = dataset.get_height_days_all()
    needed_pairs = {(0, 1), (0, 3), (2, 1), (3, 1)}

    loaded = blocked._load_oracle_draws(  # noqa: SLF001
        dataset,  # ty: ignore[invalid-argument-type]
        needed_pairs,
        oracle_day_axis=day_axis,
        prediction_day_axis=day_axis,
        chunk_size=1024,
    )

    assert set(loaded) == needed_pairs
    for (cond, draw), traj in loaded.items():
        assert torch.allclose(traj, torch.full((3,), float(cond * 100 + draw)))
    # Drawsets: {1,3} -> [0]; {1} -> [2, 3] batched into one call.
    assert sorted(dataset.loaded_indices) == [[0], [2, 3]]


def test_condition_scope_uses_dataset_flat_index_not_model_bank_order(
    monkeypatch, tmp_path
) -> None:
    # Dataset order is genotype-major with g2 first, so the dataset flat index of a
    # condition differs from its lexicographic model-bank row. Condition scope makes
    # each unit a single condition whose only oracle candidate is its own u0, and the
    # oracle draw encodes u0 as floor(height / 100). If the scorer mistakenly used the
    # sorted model-bank row as u0, the loaded oracle condition would be wrong.
    dataset_order = [("g2", "e1"), ("g2", "e2"), ("g1", "e1"), ("g1", "e2")]
    loader_order = [("g1", "e1"), ("g2", "e2"), ("g1", "e2"), ("g2", "e1")]
    capture: list = []
    dataset = _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="no_context",
        use_temperature=True,
        use_marker=True,
        dataset_key_order=dataset_order,
        loader_key_order=loader_order,
        capture=capture,
    )

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(tmp_path / "no_context"),
        options=BlockedScoringOptions(
            n_samples=4, block_size=1, condition_blocks=1, seed=7
        ),
        output_folder=str(tmp_path / "out"),
    )

    # Units are scored in sorted order: g1e1, g1e2, g2e1, g2e2.
    # Their dataset flat indices under the genotype-major dataset order are 2, 3, 0, 1.
    scale = blocked.SYNTHETIC_SIGNATURE_HEIGHT_SCALE
    oracle_first_heights = [
        round(float(oracle[:, 0, 0].max()) * scale) // 100 for _, oracle in capture
    ]
    assert oracle_first_heights == [2, 3, 0, 1]
    assert sorted(dataset.loaded_indices) == [[0], [1], [2], [3]]


def test_condition_scope_one_unit_per_condition(monkeypatch, tmp_path) -> None:
    capture: list = []
    _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="random_context",
        use_temperature=True,
        use_marker=True,
        dataset_key_order=_LEX_DATASET,
        loader_key_order=_LEX_DATASET,
        context_indices=[[0, 1], [0], [1], [0, 2]],
        capture=capture,
    )
    output_dir = tmp_path / "out"

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(tmp_path / "random_context"),
        options=BlockedScoringOptions(
            n_samples=4, block_size=2, condition_blocks=3, seed=0
        ),
        output_folder=str(output_dir),
    )

    per_unit = pl.read_csv(output_dir / "sig_mmd_per_unit.csv")
    assert per_unit["unit_scope"].unique().to_list() == ["condition"]
    assert sorted(per_unit["unit_id"].to_list()) == [
        "g1::e1",
        "g1::e2",
        "g2::e1",
        "g2::e2",
    ]
    assert per_unit["n_blocks"].to_list() == [3, 3, 3, 3]
    # 4 units x 3 blocks = 12 kernel calls, one per block, never one per condition.
    assert len(capture) == 12


def test_outputs_contain_required_columns_and_diagnostics(
    monkeypatch, tmp_path
) -> None:
    capture: list = []
    _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="max_height",
        use_temperature=False,
        use_marker=False,
        dataset_key_order=_LEX_DATASET,
        loader_key_order=_LEX_DATASET,
        capture=capture,
    )
    output_dir = tmp_path / "out"
    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(tmp_path / "max_height"),
        options=BlockedScoringOptions(
            n_samples=4, block_size=2, global_blocks=3, seed=0
        ),
        output_folder=str(output_dir),
    )

    blocks = pl.read_csv(output_dir / "sig_mmd_blocks.csv")
    per_unit = pl.read_csv(output_dir / "sig_mmd_per_unit.csv")
    members = pl.read_csv(output_dir / "sig_mmd_unit_members.csv")
    summary = pl.read_csv(output_dir / "sig_mmd_summary.csv")

    assert {
        "mode",
        "unit_scope",
        "unit_id",
        "block_index",
        "block_size",
        "model_sample_size",
        "oracle_sample_size",
        "score",
    } <= set(blocks.columns)
    assert {
        "mode",
        "unit_scope",
        "unit_id",
        "n_members",
        "n_blocks",
        "block_size",
        "full_model_pool_size",
        "full_oracle_pool_size",
        "model_sample_size",
        "oracle_sample_size",
        "score",
        "score_std",
        "score_se",
        "mean_n_context",
        "mean_ess",
        "mean_post_peak_mass",
    } <= set(per_unit.columns)
    assert {"mode", "unit_scope", "unit_id", "genotype_id", "yearsite_uid"} <= set(
        members.columns
    )
    assert {
        "view",
        "estimator",
        "mode",
        "block_size",
        "global_blocks",
        "environment_blocks",
        "genotype_blocks",
        "condition_blocks",
        "seed",
        "requested_n_samples",
        "draw_count",
        "model_draw_count",
        "oracle_draw_count",
        "n_units",
        "n_blocks",
        "mean",
        "median",
        "std",
        "p95",
        "mean_n_context",
        "mean_ess",
        "mean_post_peak_mass",
    } <= set(summary.columns)

    assert summary["view"].to_list() == ["context_matched_blocked"]
    assert summary["estimator"].to_list() == ["blocked"]
    assert summary["mode"].to_list() == ["max_height"]
    assert summary["draw_count"].to_list() == summary["model_draw_count"].to_list()
    assert blocks["mode"].unique().to_list() == ["max_height"]
    assert summary["n_blocks"].to_list() == [len(capture)]


def test_no_context_diagnostics_are_uniform(monkeypatch, tmp_path) -> None:
    def fail_posterior(*args, **kwargs):
        msg = "no_context should use the empty-context fast path"
        raise AssertionError(msg)

    monkeypatch.setattr(blocked_scoring, "tile_posterior", fail_posterior)
    capture: list = []
    _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="no_context",
        use_temperature=False,
        use_marker=False,
        dataset_key_order=_LEX_DATASET,
        loader_key_order=_LEX_DATASET,
        capture=capture,
    )
    output_dir = tmp_path / "out"
    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(tmp_path / "no_context"),
        options=BlockedScoringOptions(
            n_samples=4, block_size=2, global_blocks=2, seed=0
        ),
        output_folder=str(output_dir),
    )
    per_unit = pl.read_csv(output_dir / "sig_mmd_per_unit.csv")
    # global scope: candidate pool is all 4 conditions, empty context -> ESS == 4.
    assert per_unit["mean_n_context"].to_list() == [0.0]
    assert per_unit["mean_post_peak_mass"].to_list() == [0.0]
    assert per_unit["mean_ess"].to_list() == [4.0]


def test_diagnostics_only_for_selected_oracle_targets(monkeypatch, tmp_path) -> None:
    calls: list = []
    real_posterior = blocked_scoring.tile_posterior

    def counting_posterior(*args, **kwargs):
        calls.append(args[2])  # ctx_idx_tile: the targets processed in this tile
        return real_posterior(*args, **kwargs)

    capture: list = []
    _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="max_height",
        use_temperature=False,
        use_marker=False,
        dataset_key_order=_LEX_DATASET,
        loader_key_order=_LEX_DATASET,
        capture=capture,
    )
    monkeypatch.setattr(blocked_scoring, "tile_posterior", counting_posterior)

    seed = 0
    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(tmp_path / "max_height"),
        options=BlockedScoringOptions(
            n_samples=4, block_size=2, global_blocks=1, seed=seed, context_tile_size=1
        ),
        output_folder=str(tmp_path / "out"),
    )

    # Replicate the oracle block plan to count distinct selected target rows.
    members = blocked.group_units(UnitScope.GLOBAL, tuple(sorted(_LEX_DATASET)))[
        "global"
    ]
    oracle_blocks = build_unit_blocks(
        members,
        draws_per_member=4,
        unit_scope=UnitScope.GLOBAL,
        unit_id="global",
        n_blocks=1,
        block_size=2,
        seed=seed,
    )
    distinct_targets = {
        selection.condition_index for block in oracle_blocks for selection in block
    }
    processed = sum(len(tile) for tile in calls)
    assert processed == len(distinct_targets)
    assert processed < len(_LEX_DATASET)
    assert all(len(tile) <= 1 for tile in calls)


def _run_and_capture_oracle(monkeypatch, tmp_path, *, seed, name) -> list:
    capture: list = []
    _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name=name,
        use_temperature=False,
        use_marker=False,
        dataset_key_order=_LEX_DATASET,
        loader_key_order=_LEX_DATASET,
        capture=capture,
    )
    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(tmp_path / name),
        options=BlockedScoringOptions(
            n_samples=4, block_size=2, global_blocks=3, seed=seed
        ),
        output_folder=str(tmp_path / f"out-{name}"),
    )
    return [oracle for _, oracle in capture]


def test_sampling_is_deterministic_and_seed_sensitive(monkeypatch, tmp_path) -> None:
    run_a = _run_and_capture_oracle(
        monkeypatch, tmp_path / "a", seed=1, name="no_context"
    )
    run_b = _run_and_capture_oracle(
        monkeypatch, tmp_path / "b", seed=1, name="no_context"
    )
    run_c = _run_and_capture_oracle(
        monkeypatch, tmp_path / "c", seed=2, name="no_context"
    )

    assert len(run_a) == len(run_b) == len(run_c)
    assert all(torch.equal(a, b) for a, b in zip(run_a, run_b, strict=True))
    assert any(not torch.equal(a, c) for a, c in zip(run_a, run_c, strict=True))


def test_context_sampling_seed_preserves_normal_and_namespaces_oracle_self() -> None:
    normal = blocked._context_sampling_seed(  # noqa: SLF001
        3, "random_context", "global", "g1", "e2", None
    )
    assert normal == blocked.stable_seed(3, "random_context", "global", "g1", "e2")
    model = blocked._context_sampling_seed(  # noqa: SLF001
        3, "random_context", "global", "g1", "e2", "oracle_self_model"
    )
    reference = blocked._context_sampling_seed(  # noqa: SLF001
        3, "random_context", "global", "g1", "e2", "oracle_self_reference"
    )
    assert model != normal
    assert reference != normal
    assert model != reference


def test_oracle_self_uses_metadata_loader_and_default_output(
    monkeypatch, tmp_path
) -> None:
    capture: list = []
    load_predictions: list[bool] = []
    _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="noenv_nogeno/no_context",
        use_temperature=False,
        use_marker=False,
        dataset_key_order=_LEX_DATASET,
        loader_key_order=_LEX_DATASET,
        capture=capture,
        forbid_prediction_reads=True,
        loader_load_predictions=load_predictions,
    )
    method_dir = tmp_path / "noenv_nogeno" / "no_context"

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(method_dir),
        options=BlockedScoringOptions(
            n_samples=4, block_size=2, global_blocks=2, seed=0
        ),
        oracle_self=True,
    )

    assert load_predictions == [False]
    output_dir = (
        tmp_path
        / "sig_mmd_context_matched_blocked_oracle_self"
        / "noenv_nogeno"
        / "no_context"
    )
    assert (output_dir / "sig_mmd_blocks.csv").exists()
    assert (output_dir / "sig_mmd_per_unit.csv").exists()
    assert (output_dir / "sig_mmd_unit_members.csv").exists()
    assert (output_dir / "sig_mmd_summary.csv").exists()
    summary = pl.read_csv(output_dir / "sig_mmd_summary.csv")
    assert summary["estimator"].to_list() == ["blocked_oracle_self"]
    assert summary["oracle_self"].to_list() == [True]
    assert summary["model_source"].to_list() == ["SeedB_context_matched_oracle"]
    assert summary["reference_source"].to_list() == ["SeedA_context_matched_oracle"]
    assert len(capture) == 2

    scale = blocked.SYNTHETIC_SIGNATURE_HEIGHT_SCALE
    first_model_raw = float(capture[0][0][0, 0, 0]) * scale
    first_reference_raw = float(capture[0][1][0, 0, 0]) * scale
    assert first_model_raw >= 10_000.0
    assert first_reference_raw < 10_000.0


@pytest.mark.parametrize(
    ("method_name", "context_indices"),
    [
        ("no_context", None),
        ("random_context", [[0, 1], [0], [1], [0, 2]]),
        ("max_height", None),
    ],
)
@pytest.mark.parametrize(
    ("use_temperature", "use_marker", "expected_scope", "blocks_kw"),
    [
        (False, False, "global", {"global_blocks": 1}),
        (True, False, "environment", {"environment_blocks": 1}),
        (False, True, "genotype", {"genotype_blocks": 1}),
        (True, True, "condition", {"condition_blocks": 1}),
    ],
)
def test_oracle_self_supports_modes_and_scopes(
    monkeypatch,
    tmp_path,
    method_name,
    context_indices,
    use_temperature,
    use_marker,
    expected_scope,
    blocks_kw,
) -> None:
    capture: list = []
    _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name=method_name,
        use_temperature=use_temperature,
        use_marker=use_marker,
        dataset_key_order=_LEX_DATASET,
        loader_key_order=_LEX_DATASET,
        context_indices=context_indices,
        capture=capture,
        forbid_prediction_reads=True,
    )
    output_dir = tmp_path / "out"

    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(tmp_path / method_name),
        options=BlockedScoringOptions(n_samples=4, block_size=1, seed=4, **blocks_kw),
        output_folder=str(output_dir),
        oracle_self=True,
    )

    summary = pl.read_csv(output_dir / "sig_mmd_summary.csv")
    per_unit = pl.read_csv(output_dir / "sig_mmd_per_unit.csv")
    blocks = pl.read_csv(output_dir / "sig_mmd_blocks.csv")
    members = pl.read_csv(output_dir / "sig_mmd_unit_members.csv")
    assert summary["mode"].to_list() == [method_name]
    assert summary["estimator"].to_list() == ["blocked_oracle_self"]
    assert summary["oracle_self"].to_list() == [True]
    assert per_unit["unit_scope"].unique().to_list() == [expected_scope]
    assert set(blocks["oracle_self"].unique().to_list()) == {True}
    assert set(per_unit["oracle_self"].unique().to_list()) == {True}
    assert set(members["oracle_self"].unique().to_list()) == {True}
    assert len(capture) == int(summary["n_blocks"][0])


def test_oracle_self_validates_seed_b_flat_index_order(monkeypatch, tmp_path) -> None:
    _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="no_context",
        use_temperature=False,
        use_marker=False,
        dataset_key_order=_LEX_DATASET,
        loader_key_order=_LEX_DATASET,
        seed_b_key_order=list(reversed(_LEX_DATASET)),
    )

    with pytest.raises(ValueError, match="flat-index ordering differs"):
        blocked.sig_mmd_context_matched_blocked_analysis(
            method_dir=str(tmp_path / "no_context"),
            options=BlockedScoringOptions(n_samples=4, block_size=1, global_blocks=1),
            output_folder=str(tmp_path / "out"),
            oracle_self=True,
        )


def _run_and_capture_oracle_self(monkeypatch, tmp_path, *, seed) -> list:
    capture: list = []
    _install_fake_world(
        monkeypatch,
        tmp_path,
        method_name="no_context",
        use_temperature=False,
        use_marker=False,
        dataset_key_order=_LEX_DATASET,
        loader_key_order=_LEX_DATASET,
        capture=capture,
        forbid_prediction_reads=True,
    )
    blocked.sig_mmd_context_matched_blocked_analysis(
        method_dir=str(tmp_path / "no_context"),
        options=BlockedScoringOptions(
            n_samples=4, block_size=2, global_blocks=3, seed=seed
        ),
        output_folder=str(tmp_path / "out"),
        oracle_self=True,
    )
    return list(capture)


def test_oracle_self_sampling_is_deterministic_and_seed_sensitive(
    monkeypatch, tmp_path
) -> None:
    run_a = _run_and_capture_oracle_self(monkeypatch, tmp_path / "a", seed=1)
    run_b = _run_and_capture_oracle_self(monkeypatch, tmp_path / "b", seed=1)
    run_c = _run_and_capture_oracle_self(monkeypatch, tmp_path / "c", seed=2)

    assert len(run_a) == len(run_b) == len(run_c)
    assert all(
        torch.equal(a_model, b_model) and torch.equal(a_ref, b_ref)
        for (a_model, a_ref), (b_model, b_ref) in zip(run_a, run_b, strict=True)
    )
    assert any(
        not torch.equal(a_model, c_model) or not torch.equal(a_ref, c_ref)
        for (a_model, a_ref), (c_model, c_ref) in zip(run_a, run_c, strict=True)
    )
