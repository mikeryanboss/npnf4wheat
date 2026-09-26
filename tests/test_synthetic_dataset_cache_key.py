"""Regression tests for SyntheticDataset cache-key behavior (ETH-847, ETH-881, ETH-139).

_compute_cache_key() must distinguish datasets that share the same
(num_genotypes, num_yearsites) shape but differ in any behavior-affecting parameter.

As of ETH-132 the cache key is computed from config-derivable values only —
`_genotype_indices`, `_yearsite_indices`, `genotype_pool_seed`,
`yearsite_pool_source`, and scalar behavior params — so it can be resolved
before the pool factories are called.
"""

# ruff: noqa: SLF001

import json
import types
from typing import Any

import pytest
import tensordict
import torch

import npnf.data.datasets.synthetic as synthetic_mod
from npnf.calibration.height.constants import HeightDates
from npnf.data.datasets.synthetic import SyntheticDataset
from tests.utils import tensor_at, tensordict_at


def _make_proxy(
    genotype_indices: list[int] | None = None,
    yearsite_indices: list[int] | None = None,
    **overrides,
) -> types.SimpleNamespace:
    """Minimal namespace satisfying _compute_cache_key()'s attribute access."""
    if genotype_indices is None:
        genotype_indices = [0, 1]
    if yearsite_indices is None:
        yearsite_indices = [0]

    defaults = {
        "num_genotypes": len(genotype_indices),
        "num_yearsites": len(yearsite_indices),
        "genotype_pool_seed": 42,
        "_genotype_indices": genotype_indices,
        "yearsite_pool_source": "synthetic",
        "_yearsite_indices": yearsite_indices,
        "scale_noise": 0.01,
        "enable_lodging": False,
        "eval_mode": True,
        "num_pre_season_points": 5,
        "num_post_season_points": 5,
        "lodging_height_clamp": 1.2,
        "lodging_weibull_shape": 2.0,
        "lodging_weibull_scale": 0.3,
        "lodging_weibull_offset": 0.3,
        "lodging_severity_min": 0.3,
        "lodging_severity_max": 0.7,
        "lodging_transition_steps_min": 5,
        "lodging_transition_steps_max": 20,
        "n_lodging_draws": 1,
        "lodging_seed": None,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _key(proxy):
    return SyntheticDataset._compute_cache_key(proxy)


def _make_cache_path_proxy(**overrides):
    proxy = _make_proxy(**overrides)
    proxy._compute_cache_key_config = types.MethodType(
        SyntheticDataset._compute_cache_key_config, proxy
    )
    proxy._compute_cache_key = types.MethodType(
        SyntheticDataset._compute_cache_key, proxy
    )
    proxy._get_cache_name = types.MethodType(SyntheticDataset._get_cache_name, proxy)
    proxy._get_cache_path = types.MethodType(SyntheticDataset._get_cache_path, proxy)
    proxy._get_meta_path = types.MethodType(SyntheticDataset._get_meta_path, proxy)
    proxy._get_cache_key_config_path = types.MethodType(
        SyntheticDataset._get_cache_key_config_path, proxy
    )
    proxy._slugify_cache_part = SyntheticDataset._slugify_cache_part
    proxy._summarize_cache_indices = SyntheticDataset._summarize_cache_indices
    return proxy


def _make_stubbed_dataset(monkeypatch, **overrides):
    """Construct SyntheticDataset cheaply while exercising __init__ logic."""

    def _fake_compute_intermediate(self, genotype_pool, yearsite_pool):
        batch_size = self.num_genotypes * self.num_yearsites
        return tensordict.TensorDict(
            {
                "heights_clean": torch.zeros(batch_size, 40),
                "genotype_indices": torch.arange(self.num_genotypes).repeat_interleave(
                    self.num_yearsites
                ),
                "yearsite_indices": torch.arange(self.num_yearsites).repeat(
                    self.num_genotypes
                ),
                "markers_unique": torch.zeros(self.num_genotypes, 2),
                "temperatures_unique": torch.zeros(self.num_yearsites, 4, 24),
                "noise_seed": torch.tensor(123, dtype=torch.long),
                **(
                    {"lodging_seed": torch.tensor(self.lodging_seed, dtype=torch.long)}
                    if self.enable_lodging
                    else {}
                ),
            },
            batch_size=[],
            non_blocking=True,
        )

    monkeypatch.setattr(
        SyntheticDataset, "_compute_intermediate", _fake_compute_intermediate
    )
    monkeypatch.setattr(
        SyntheticDataset, "_compose_and_format", lambda self, intermediate: intermediate
    )

    class FakeGenotypePool:
        def __init__(self):
            self.genotype_ids = ["g0", "g1"]

        def __len__(self):
            return len(self.genotype_ids)

    class FakeYearsitePool:
        def __init__(self):
            self.yearsite_ids = ["y0"]

        def __len__(self):
            return len(self.yearsite_ids)

    defaults: dict[str, Any] = {
        "genotype_pool": FakeGenotypePool,
        "yearsite_pool": FakeYearsitePool,
        "scale_noise": 0.01,
        "enable_lodging": False,
        "eval_mode": True,
        "num_pre_season_points": 5,
        "num_post_season_points": 5,
        "lodging_height_clamp": 1.2,
        "lodging_weibull_shape": 2.0,
        "lodging_weibull_scale": 0.3,
        "lodging_weibull_offset": 0.3,
        "lodging_severity_min": 0.3,
        "lodging_severity_max": 0.7,
        "lodging_transition_steps_min": 5,
        "lodging_transition_steps_max": 20,
        "genotype_pool_seed": 42,
        "yearsite_pool_source": "synthetic",
        "n_lodging_draws": 1,
        "lodging_seed": None,
        "cache_dir": None,
    }
    defaults.update(overrides)
    return SyntheticDataset(**defaults)


def _make_compose_proxy(
    *, enable_lodging: bool, n_lodging_draws: int, scale_noise: float = 1.0
):
    """Minimal namespace satisfying _compose_and_format()'s attribute access."""
    proxy = types.SimpleNamespace(
        scale_noise=scale_noise,
        enable_lodging=enable_lodging,
        n_lodging_draws=n_lodging_draws,
        eval_mode=True,
        day_temperature_start=0,
        day_temperature_end=10,
        growing_period_start=5,
        growing_period_end=35,
        num_pre_season_points=2,
        num_post_season_points=2,
        genotype_ids=["g0", "g1"],
        yearsite_ids=["y0", "y1"],
    )
    proxy._compute_subsample_indices = types.MethodType(
        SyntheticDataset._compute_subsample_indices, proxy
    )
    return proxy


def _expected_noise(seed: int, shape: tuple[int, int], scale: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * scale


def _make_lazy_dataset_from_intermediate(proxy, intermediate):
    dataset = object.__new__(SyntheticDataset)
    for key, value in vars(proxy).items():
        if key.startswith("_"):
            continue
        setattr(dataset, key, value)
    dataset.num_genotypes = len(proxy.genotype_ids)
    dataset.num_yearsites = len(proxy.yearsite_ids)
    dataset._intermediate = intermediate
    dataset._temperatures_unique = intermediate["temperatures_unique"]
    dataset._sample_yearsite_indices = intermediate["yearsite_indices"]
    dataset._compute_subsample_indices = types.MethodType(  # ty: ignore[invalid-assignment]
        SyntheticDataset._compute_subsample_indices, dataset
    )
    dataset._sample_lodging_params = types.MethodType(  # ty: ignore[invalid-assignment]
        SyntheticDataset._sample_lodging_params, dataset
    )
    dataset._apply_lodging_from_params = types.MethodType(  # ty: ignore[invalid-assignment]
        SyntheticDataset._apply_lodging_from_params, dataset
    )
    dataset._subsample_indices = None
    dataset._height_days_all = None
    dataset._noise = None
    dataset._lodging_params = None
    dataset.samples = None
    SyntheticDataset._prepare_lazy_composition(dataset)
    return dataset


def _assert_equal_value(actual, expected, key: str) -> None:
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual), key
        if expected.is_floating_point():
            assert torch.allclose(actual, expected), key
        else:
            assert torch.equal(actual, expected), key
    else:
        assert actual == expected, key


# --- Pool identity tests (ETH-847) ---


@pytest.mark.parametrize(
    ("indices_a", "indices_b"),
    [
        pytest.param(
            {"genotype_indices": [0, 1]},
            {"genotype_indices": [2, 3]},
            id="genotype-indices",
        ),
        pytest.param(
            {"yearsite_indices": [0, 1]},
            {"yearsite_indices": [2, 3]},
            id="yearsite-indices",
        ),
        # Regression: test_plot vs test_genotype collision scenario.
        pytest.param(
            {"genotype_indices": list(range(64)), "yearsite_indices": list(range(32))},
            {
                "genotype_indices": list(range(64, 128)),
                "yearsite_indices": list(range(32, 64)),
            },
            id="equal-shape-splits",
        ),
    ],
)
def test_different_pool_indices_produce_different_keys(indices_a, indices_b):
    """Same pool shape, different indices → different cache key."""
    assert _key(_make_proxy(**indices_a)) != _key(_make_proxy(**indices_b))


def test_different_pool_seed_produces_different_keys():
    """Different genotype_pool_seed → different cache key."""
    a = _make_proxy(genotype_pool_seed=42)
    b = _make_proxy(genotype_pool_seed=43)
    assert _key(a) != _key(b)


def test_different_yearsite_source_produces_different_keys():
    """Different yearsite_pool_source → different cache key."""
    a = _make_proxy(yearsite_pool_source="synthetic")
    b = _make_proxy(yearsite_pool_source="weather_data")
    assert _key(a) != _key(b)


def test_identical_config_produces_same_key():
    """Cache key must be deterministic: same config always yields same key."""
    a = _make_proxy()
    b = _make_proxy()
    assert _key(a) == _key(b)


def test_no_lodging_init_canonicalizes_draw_count_and_seed(monkeypatch):
    """No-lodging configs must collapse draw-specific settings in __init__."""
    dataset = _make_stubbed_dataset(
        monkeypatch, enable_lodging=False, n_lodging_draws=64, lodging_seed=42
    )
    assert dataset.n_lodging_draws == 1
    assert dataset.lodging_seed is None


def test_no_lodging_draw_config_does_not_change_cache_key(monkeypatch):
    """Caller-provided no-lodging draw settings must not fork the cache."""
    baseline = _make_stubbed_dataset(
        monkeypatch, enable_lodging=False, n_lodging_draws=1, lodging_seed=None
    )
    variant = _make_stubbed_dataset(
        monkeypatch, enable_lodging=False, n_lodging_draws=64, lodging_seed=42
    )
    assert baseline._compute_cache_key() == variant._compute_cache_key()


def test_lodging_enabled_requires_seed(monkeypatch):
    """Lodging-enabled datasets must be configured with a reproducible seed."""
    with pytest.raises(
        ValueError, match="lodging_seed is required when enable_lodging=True"
    ):
        _make_stubbed_dataset(monkeypatch, enable_lodging=True, lodging_seed=None)


def test_lodging_enabled_draw_config_still_changes_cache_key():
    """When lodging is enabled, draw count and seed remain behavior-affecting."""
    baseline = _make_proxy(enable_lodging=True, n_lodging_draws=1, lodging_seed=41)
    different_n = _make_proxy(enable_lodging=True, n_lodging_draws=8, lodging_seed=41)
    different_seed = _make_proxy(
        enable_lodging=True, n_lodging_draws=8, lodging_seed=42
    )
    assert _key(baseline) != _key(different_n)
    assert _key(different_n) != _key(different_seed)


def test_public_dataset_identity_is_json_serializable(monkeypatch):
    """Prediction artifacts need a stable public identity for reconstruction."""
    dataset = _make_stubbed_dataset(monkeypatch)

    identity = dataset.get_dataset_identity()

    assert set(identity) == {
        "dataset_type",
        "cache_key",
        "cache_key_config",
        "num_conditions",
    }
    assert json.loads(json.dumps(identity, sort_keys=True)) == identity


def test_cache_path_uses_readable_directory_name_without_td_suffix():
    """Cache path should be a readable directory name, not an opaque .td path."""
    config = {
        "genotype_indices": list(range(64)),
        "yearsite_indices": list(range(32, 64)),
        "yearsite_pool_source": "weather_data",
        "eval_mode": False,
        "enable_lodging": True,
        "n_lodging_draws": 8,
        "lodging_seed": 99,
    }
    path = _make_cache_path_proxy(**config)._get_cache_path("dataset_cache")
    other_path = _make_cache_path_proxy(
        **{**config, "lodging_seed": 100}
    )._get_cache_path("dataset_cache")
    hash_suffix = path.name.rsplit("_", maxsplit=1)[-1]

    assert not path.name.endswith(".td")
    assert len(hash_suffix) == 12
    assert all(char in "0123456789abcdef" for char in hash_suffix)
    assert other_path != path


def test_saved_cache_writes_cache_key_config_inside_cache_directory(
    monkeypatch, tmp_path
):
    """Cache saves should persist the full key config inside the cache folder."""

    def _fake_save_cache(self, path, data):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(SyntheticDataset, "_save_cache", _fake_save_cache)

    dataset = _make_stubbed_dataset(monkeypatch, cache_dir=str(tmp_path))
    cache_path = dataset._get_cache_path(str(tmp_path))
    metadata_path = dataset._get_meta_path(str(tmp_path))
    cache_key_path = dataset._get_cache_key_config_path(str(tmp_path))

    assert cache_path.is_dir()
    assert cache_path.suffix != ".td"
    assert metadata_path == cache_path / "metadata.json"
    assert cache_key_path == cache_path / "cache_key.json"
    expected_config = dataset._compute_cache_key_config()
    assert json.loads(cache_key_path.read_text()) == expected_config
    assert json.loads(metadata_path.read_text()) == {
        "genotype_ids": ["g0", "g1"],
        "yearsite_ids": ["y0"],
    }


# --- Behavior-affecting parameter collision tests (ETH-881) ---

_BEHAVIOR_PARAMS: dict[str, tuple[Any, Any]] = {
    "num_pre_season_points": (5, 10),
    "num_post_season_points": (5, 10),
    "lodging_height_clamp": (1.2, 1.5),
    "lodging_weibull_shape": (2.0, 3.0),
    "lodging_weibull_scale": (0.3, 0.5),
    "lodging_weibull_offset": (0.3, 0.5),
    "lodging_severity_min": (0.3, 0.4),
    "lodging_severity_max": (0.7, 0.8),
    "lodging_transition_steps_min": (5, 10),
    "lodging_transition_steps_max": (20, 30),
}


def test_behavior_params_produce_different_keys():
    """Changing any behavior-affecting parameter must produce a different key."""
    baseline = _make_proxy()
    baseline_key = _key(baseline)

    for param_name, (default_val, alt_val) in _BEHAVIOR_PARAMS.items():
        altered = _make_proxy(**{param_name: alt_val})
        altered_key = _key(altered)
        assert altered_key != baseline_key, (
            f"Changing {param_name} from {default_val} to {alt_val} "
            f"did not change cache key"
        )


def test_compute_intermediate_skips_lodging_params_when_disabled(monkeypatch):
    """No-lodging intermediate data must omit lodging params entirely."""

    class FakeLUT:
        def __init__(self):
            self.basis_grid = torch.zeros(1)

    monkeypatch.setattr(
        synthetic_mod,
        "_precompute_yearsite_basis",
        lambda temperatures, *, dates: (torch.zeros(1, 1, 1, 1), torch.zeros(1, 1)),
    )
    monkeypatch.setattr(
        synthetic_mod,
        "_compute_growth_chunked",
        lambda **kwargs: torch.arange(
            kwargs["num_days"], dtype=torch.float32, device=kwargs["tau_max"].device
        ).repeat(kwargs["yearsite_indices"].shape[0], 1),
    )
    monkeypatch.setattr(synthetic_mod, "BSplineLUT", FakeLUT)
    monkeypatch.setattr(
        synthetic_mod.GenotypeParams, "field_names", staticmethod(lambda: ["tau_max"])
    )
    monkeypatch.setattr(
        synthetic_mod.GenotypeParams,
        "build_clamped_control_grid",
        staticmethod(
            lambda params, n_T_basis, n_tau_basis: torch.zeros(
                params.shape[0], n_T_basis, n_tau_basis
            )
        ),
    )

    proxy: Any = _make_proxy(
        genotype_indices=[0, 1],
        yearsite_indices=[0],
        num_days=4,
        dates=HeightDates(),
        n_lodging_draws=64,
        enable_lodging=False,
    )
    proxy._sample_lodging_params = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("_sample_lodging_params should not be called")
    )

    genotype_pool: Any = types.SimpleNamespace(
        params=torch.tensor([[1.0], [2.0]]), markers=torch.tensor([[0.0], [1.0]])
    )
    yearsite_pool: Any = types.SimpleNamespace(temperatures=torch.zeros(1, 4, 24))

    intermediate = SyntheticDataset._compute_intermediate(
        proxy, genotype_pool, yearsite_pool
    )

    assert "lodging_will_lodge" not in intermediate
    assert "lodging_offset" not in intermediate
    assert "lodging_severity" not in intermediate
    assert "lodging_transition_steps" not in intermediate
    assert "lodging_seed" not in intermediate
    assert "noise" not in intermediate
    assert "marker_biallelic_codes" not in intermediate
    assert "temperature_values" not in intermediate
    assert intermediate["noise_seed"].shape == torch.Size([])
    assert intermediate["markers_unique"].shape == torch.Size([2, 1])
    assert intermediate["temperatures_unique"].shape == torch.Size([1, 4, 24])
    assert intermediate.batch_size == torch.Size([])


def test_compute_intermediate_stores_lodging_seed_without_lodging_params(monkeypatch):
    """Lodging-enabled intermediate data stores only the seed, not params."""

    class FakeLUT:
        def __init__(self):
            self.basis_grid = torch.zeros(1)

    monkeypatch.setattr(
        synthetic_mod,
        "_precompute_yearsite_basis",
        lambda temperatures, *, dates: (torch.zeros(1, 1, 1, 1), torch.zeros(1, 1)),
    )
    monkeypatch.setattr(
        synthetic_mod,
        "_compute_growth_chunked",
        lambda **kwargs: torch.arange(
            kwargs["num_days"], dtype=torch.float32, device=kwargs["tau_max"].device
        ).repeat(kwargs["yearsite_indices"].shape[0], 1),
    )
    monkeypatch.setattr(synthetic_mod, "BSplineLUT", FakeLUT)
    monkeypatch.setattr(
        synthetic_mod.GenotypeParams, "field_names", staticmethod(lambda: ["tau_max"])
    )
    monkeypatch.setattr(
        synthetic_mod.GenotypeParams,
        "build_clamped_control_grid",
        staticmethod(
            lambda params, n_T_basis, n_tau_basis: torch.zeros(
                params.shape[0], n_T_basis, n_tau_basis
            )
        ),
    )

    proxy: Any = _make_proxy(
        genotype_indices=[0, 1],
        yearsite_indices=[0],
        num_days=4,
        dates=HeightDates(),
        n_lodging_draws=3,
        lodging_seed=42,
        enable_lodging=True,
    )
    proxy._sample_lodging_params = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("_sample_lodging_params should not be called")
    )

    genotype_pool: Any = types.SimpleNamespace(
        params=torch.tensor([[1.0], [2.0]]), markers=torch.tensor([[0.0], [1.0]])
    )
    yearsite_pool: Any = types.SimpleNamespace(temperatures=torch.zeros(1, 4, 24))

    intermediate = SyntheticDataset._compute_intermediate(
        proxy, genotype_pool, yearsite_pool
    )

    assert int(tensor_at(intermediate, "lodging_seed").item()) == 42
    assert "lodging_will_lodge" not in intermediate
    assert "lodging_offset" not in intermediate
    assert "lodging_severity" not in intermediate
    assert "lodging_transition_steps" not in intermediate
    assert "noise" not in intermediate
    assert "marker_biallelic_codes" not in intermediate
    assert "temperature_values" not in intermediate


def test_compose_and_format_ignores_missing_lodging_keys_when_disabled():
    """No-lodging compose path must work without any cached lodging tensors."""
    proxy = _make_compose_proxy(
        enable_lodging=False, n_lodging_draws=64, scale_noise=0.5
    )
    proxy._apply_lodging_from_params = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("_apply_lodging_from_params should not be called")
    )

    B = 2
    T = 40
    heights_clean = torch.stack(
        [
            torch.arange(T, dtype=torch.float32),
            torch.arange(T, dtype=torch.float32) + 100.0,
        ]
    )
    noise_seed = 123
    noise = _expected_noise(noise_seed, (B, T), proxy.scale_noise)
    markers_unique = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    temperatures_unique = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    intermediate = tensordict.TensorDict(
        {
            "heights_clean": heights_clean,
            "noise_seed": torch.tensor(noise_seed, dtype=torch.long),
            "genotype_indices": torch.tensor([0, 1]),
            "yearsite_indices": torch.tensor([0, 1]),
            "markers_unique": markers_unique,
            "temperatures_unique": temperatures_unique,
        },
        batch_size=[],
        non_blocking=True,
    )

    result = SyntheticDataset._compose_and_format(proxy, intermediate)
    indices = proxy._compute_subsample_indices(T).expand(B, -1)
    clean_sub = heights_clean.gather(1, indices)
    noise_sub = noise.gather(1, indices)

    assert result["height_values_nonoise"].shape == clean_sub.shape
    assert result["height_values_all_nonoise"].shape == heights_clean.shape
    assert result["has_lodged"].shape == torch.Size([B])
    assert result["height_lodged_mask"].shape == clean_sub.shape
    assert result["height_lodged_mask_all"].shape == heights_clean.shape

    assert torch.equal(tensor_at(result, "height_values_nonoise"), clean_sub)
    assert torch.equal(tensor_at(result, "height_values_all_nonoise"), heights_clean)
    assert torch.equal(
        tensor_at(result, "height_values_nonoise"),
        tensor_at(result, "height_values_nonoise_nolodge"),
    )
    assert torch.equal(
        tensor_at(result, "height_values_all_nonoise"),
        tensor_at(result, "height_values_all_nonoise_nolodge"),
    )
    assert torch.allclose(tensor_at(result, "height_values"), clean_sub + noise_sub)
    assert torch.equal(tensor_at(result, "marker_biallelic_codes"), markers_unique)
    assert "temperature_values" not in result
    assert not result["has_lodged"].any()
    assert not result["height_lodged_mask"].any()
    assert not result["height_lodged_mask_all"].any()


def test_compose_and_format_preserves_multi_draw_outputs_when_enabled():
    """Lodging-enabled compose path must keep the N axis unchanged."""
    proxy = _make_compose_proxy(
        enable_lodging=True, n_lodging_draws=2, scale_noise=0.25
    )

    B = 2
    T = 40
    heights_clean = torch.stack(
        [
            torch.arange(T, dtype=torch.float32),
            torch.arange(T, dtype=torch.float32) + 100.0,
        ]
    )
    noise_seed = 456
    noise = _expected_noise(noise_seed, (B, T), proxy.scale_noise)
    heights_lodged = torch.stack([heights_clean, heights_clean + 10.0], dim=1)
    has_lodged = torch.tensor([[False, True], [True, False]])
    lodged_mask = torch.zeros((B, 2, T), dtype=torch.bool)
    lodged_mask[:, 1, 10:] = True
    called = {"apply": False, "sample": False}
    lodging_seed = 77
    expected_lodging_draw = torch.rand(
        (), generator=torch.Generator().manual_seed(lodging_seed)
    )

    def _fake_sample(heights, n_draws, generator):
        called["sample"] = True
        assert torch.equal(heights, heights_clean)
        assert n_draws == 2
        assert torch.equal(torch.rand((), generator=generator), expected_lodging_draw)
        return {
            "will_lodge": torch.zeros((B, 2), dtype=torch.bool),
            "offset": torch.zeros((B, 2), dtype=torch.long),
            "severity": torch.zeros((B, 2)),
            "transition_steps": torch.zeros((B, 2), dtype=torch.long),
        }

    def _fake_apply(heights, lodging_params):
        called["apply"] = True
        assert torch.equal(heights, heights_clean)
        assert torch.equal(
            lodging_params["will_lodge"], torch.zeros((B, 2), dtype=torch.bool)
        )
        return heights_lodged, has_lodged, lodged_mask

    proxy._sample_lodging_params = _fake_sample
    proxy._apply_lodging_from_params = _fake_apply

    intermediate = tensordict.TensorDict(
        {
            "heights_clean": heights_clean,
            "noise_seed": torch.tensor(noise_seed, dtype=torch.long),
            "lodging_seed": torch.tensor(lodging_seed, dtype=torch.long),
            "genotype_indices": torch.tensor([0, 1]),
            "yearsite_indices": torch.tensor([0, 1]),
            "markers_unique": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "temperatures_unique": torch.zeros(B, 4),
        },
        batch_size=[],
        non_blocking=True,
    )

    result = SyntheticDataset._compose_and_format(proxy, intermediate)
    indices = proxy._compute_subsample_indices(T).expand(B, -1)
    indices_n = indices.unsqueeze(1).expand(B, 2, -1)
    lodged_sub = heights_lodged.gather(2, indices_n)
    mask_sub = lodged_mask.gather(2, indices_n)
    noise_sub = noise.gather(1, indices)

    assert called == {"apply": True, "sample": True}
    assert result["height_values_nonoise"].shape == lodged_sub.shape
    assert result["height_values_all_nonoise"].shape == heights_lodged.shape
    assert result["has_lodged"].shape == has_lodged.shape
    assert result["height_lodged_mask"].shape == mask_sub.shape
    assert result["height_lodged_mask_all"].shape == lodged_mask.shape
    assert torch.equal(tensor_at(result, "height_values_nonoise"), lodged_sub)
    assert torch.equal(tensor_at(result, "height_values_all_nonoise"), heights_lodged)
    assert torch.equal(tensor_at(result, "has_lodged"), has_lodged)
    assert torch.equal(tensor_at(result, "height_lodged_mask"), mask_sub)
    assert torch.equal(tensor_at(result, "height_lodged_mask_all"), lodged_mask)
    assert torch.allclose(
        tensor_at(result, "height_values"), lodged_sub[:, 0, :] + noise_sub
    )


def test_lazy_getitem_matches_eager_compose_for_multi_draw_outputs():
    """Lazy __getitem__ must return the same per-item data as eager compose."""
    proxy = _make_compose_proxy(enable_lodging=True, n_lodging_draws=3, scale_noise=0.1)
    proxy.lodging_height_clamp = 10.0
    proxy.lodging_weibull_shape = 2.0
    proxy.lodging_weibull_scale = 1.0
    proxy.lodging_weibull_offset = 0.0
    proxy.lodging_severity_min = 0.3
    proxy.lodging_severity_max = 0.7
    proxy.lodging_transition_steps_min = 2
    proxy.lodging_transition_steps_max = 5
    proxy._sample_lodging_params = types.MethodType(
        SyntheticDataset._sample_lodging_params, proxy
    )
    proxy._apply_lodging_from_params = types.MethodType(
        SyntheticDataset._apply_lodging_from_params, proxy
    )

    num_genotypes = 3
    num_yearsites = 2
    proxy.genotype_ids = [f"g{i}" for i in range(num_genotypes)]
    proxy.yearsite_ids = [f"y{i}" for i in range(num_yearsites)]
    B = num_genotypes * num_yearsites
    T = 40
    intermediate = tensordict.TensorDict(
        {
            "heights_clean": torch.linspace(0.1, 2.0, B * T).reshape(B, T),
            "noise_seed": torch.tensor(1234, dtype=torch.long),
            "lodging_seed": torch.tensor(5678, dtype=torch.long),
            "genotype_indices": torch.arange(num_genotypes).repeat_interleave(
                num_yearsites
            ),
            "yearsite_indices": torch.arange(num_yearsites).repeat(num_genotypes),
            "markers_unique": torch.arange(
                num_genotypes * 3, dtype=torch.float32
            ).reshape(num_genotypes, 3),
            "temperatures_unique": torch.arange(
                num_yearsites * 4 * 24, dtype=torch.float32
            ).reshape(num_yearsites, 4, 24),
        },
        batch_size=[],
        non_blocking=True,
    )

    eager = SyntheticDataset._compose_and_format(proxy, intermediate)
    lazy = _make_lazy_dataset_from_intermediate(proxy, intermediate)

    for index in [0, 1, 2]:
        expected = tensordict_at(eager, index).to_dict()
        actual = SyntheticDataset.__getitem__(lazy, index)
        for key, expected_value in expected.items():
            _assert_equal_value(actual[key], expected_value, key)
        yearsite_index = int(tensor_at(intermediate, "yearsite_indices")[index].item())
        assert torch.equal(
            actual["temperature_values"],
            tensor_at(intermediate, "temperatures_unique")[yearsite_index],
        )


def test_get_lodged_chunk_matches_eager_compose_for_selected_draws():
    """Chunked access must match eager all-draw fields for selected rows/draws."""
    proxy = _make_compose_proxy(enable_lodging=True, n_lodging_draws=3, scale_noise=0.1)
    proxy.lodging_height_clamp = 10.0
    proxy.lodging_weibull_shape = 2.0
    proxy.lodging_weibull_scale = 1.0
    proxy.lodging_weibull_offset = 0.0
    proxy.lodging_severity_min = 0.3
    proxy.lodging_severity_max = 0.7
    proxy.lodging_transition_steps_min = 2
    proxy.lodging_transition_steps_max = 5
    proxy._sample_lodging_params = types.MethodType(
        SyntheticDataset._sample_lodging_params, proxy
    )
    proxy._apply_lodging_from_params = types.MethodType(
        SyntheticDataset._apply_lodging_from_params, proxy
    )

    num_genotypes = 3
    num_yearsites = 2
    proxy.genotype_ids = [f"g{i}" for i in range(num_genotypes)]
    proxy.yearsite_ids = [f"y{i}" for i in range(num_yearsites)]
    B = num_genotypes * num_yearsites
    T = 40
    intermediate = tensordict.TensorDict(
        {
            "heights_clean": torch.linspace(0.1, 2.0, B * T).reshape(B, T),
            "noise_seed": torch.tensor(1234, dtype=torch.long),
            "lodging_seed": torch.tensor(5678, dtype=torch.long),
            "genotype_indices": torch.arange(num_genotypes).repeat_interleave(
                num_yearsites
            ),
            "yearsite_indices": torch.arange(num_yearsites).repeat(num_genotypes),
            "markers_unique": torch.arange(
                num_genotypes * 3, dtype=torch.float32
            ).reshape(num_genotypes, 3),
            "temperatures_unique": torch.zeros(num_yearsites, 4, 24),
        },
        batch_size=[],
        non_blocking=True,
    )

    eager = SyntheticDataset._compose_and_format(proxy, intermediate)
    lazy = _make_lazy_dataset_from_intermediate(proxy, intermediate)
    indices = [0, 2]
    draw_indices = [0, 1, 2]

    chunk = lazy.get_lodged_chunk(indices, draw_indices)

    assert torch.equal(
        chunk["height_values_all_nonoise"],
        tensor_at(eager, "height_values_all_nonoise")[indices][:, draw_indices],
    )
    assert torch.equal(
        chunk["has_lodged"], tensor_at(eager, "has_lodged")[indices][:, draw_indices]
    )
    assert torch.equal(
        chunk["height_lodged_mask_all"],
        tensor_at(eager, "height_lodged_mask_all")[indices][:, draw_indices],
    )
    assert torch.equal(chunk["height_days_all"], tensor_at(eager, "height_days_all")[0])
    assert torch.equal(
        chunk["height_values_all_nonoise_nolodge"],
        tensor_at(eager, "height_values_all_nonoise_nolodge")[indices],
    )
    assert chunk["genotype_id"] == [eager["genotype_id"][i] for i in indices]
    assert chunk["yearsite_uid"] == [eager["yearsite_uid"][i] for i in indices]


def test_saved_compact_intermediate_reconstructs_identical_samples(tmp_path):
    """Saved compact intermediates omit redundant keys and replay deterministically."""
    proxy = _make_compose_proxy(enable_lodging=True, n_lodging_draws=2, scale_noise=0.1)
    proxy.lodging_height_clamp = 10.0
    proxy.lodging_weibull_shape = 2.0
    proxy.lodging_weibull_scale = 1.0
    proxy.lodging_weibull_offset = 0.0
    proxy.lodging_severity_min = 0.3
    proxy.lodging_severity_max = 0.7
    proxy.lodging_transition_steps_min = 2
    proxy.lodging_transition_steps_max = 5
    proxy._sample_lodging_params = types.MethodType(
        SyntheticDataset._sample_lodging_params, proxy
    )
    proxy._apply_lodging_from_params = types.MethodType(
        SyntheticDataset._apply_lodging_from_params, proxy
    )

    num_genotypes = 8
    num_yearsites = 4
    proxy.genotype_ids = [f"g{i}" for i in range(num_genotypes)]
    proxy.yearsite_ids = [f"y{i}" for i in range(num_yearsites)]
    B = num_genotypes * num_yearsites
    T = 40
    genotype_indices = torch.arange(num_genotypes).repeat_interleave(num_yearsites)
    yearsite_indices = torch.arange(num_yearsites).repeat(num_genotypes)
    intermediate = tensordict.TensorDict(
        {
            "heights_clean": torch.linspace(0.1, 2.0, B * T).reshape(B, T),
            "noise_seed": torch.tensor(1234, dtype=torch.long),
            "lodging_seed": torch.tensor(5678, dtype=torch.long),
            "genotype_indices": genotype_indices,
            "yearsite_indices": yearsite_indices,
            "markers_unique": torch.arange(
                num_genotypes * 3, dtype=torch.float32
            ).reshape(num_genotypes, 3),
            "temperatures_unique": torch.arange(
                num_yearsites * 5, dtype=torch.float32
            ).reshape(num_yearsites, 5),
        },
        batch_size=[],
        non_blocking=True,
    )

    cache_path = tmp_path / "compact_cache"
    SyntheticDataset._save_cache(proxy, cache_path, intermediate)
    loaded = SyntheticDataset._load_cache(proxy, cache_path)

    forbidden_keys = {
        "temperature_values",
        "marker_biallelic_codes",
        "noise",
        "lodging_will_lodge",
        "lodging_offset",
        "lodging_severity",
        "lodging_transition_steps",
    }
    assert forbidden_keys.isdisjoint(set(loaded.keys()))

    fresh = SyntheticDataset._compose_and_format(proxy, intermediate)
    cached = SyntheticDataset._compose_and_format(proxy, loaded)

    keys = set(fresh.keys())
    assert keys == set(cached.keys())
    for key in keys:
        fresh_value = fresh[key]
        cached_value = cached[key]
        if isinstance(fresh_value, torch.Tensor):
            if fresh_value.is_floating_point():
                assert torch.allclose(fresh_value, tensor_at(cached, key)), key
            else:
                assert torch.equal(fresh_value, tensor_at(cached, key)), key
        else:
            assert fresh_value == cached_value


@pytest.mark.parametrize(
    ("enable_lodging", "n_lodging_draws"), [(True, 3), (True, 1), (False, 1)]
)
def test_getitems_matches_getitem(enable_lodging, n_lodging_draws):
    """Batched __getitems__ must return exactly the per-index __getitem__ samples."""
    proxy = _make_compose_proxy(
        enable_lodging=enable_lodging, n_lodging_draws=n_lodging_draws, scale_noise=0.1
    )
    proxy.lodging_height_clamp = 10.0
    proxy.lodging_weibull_shape = 2.0
    proxy.lodging_weibull_scale = 1.0
    proxy.lodging_weibull_offset = 0.0
    proxy.lodging_severity_min = 0.3
    proxy.lodging_severity_max = 0.7
    proxy.lodging_transition_steps_min = 2
    proxy.lodging_transition_steps_max = 5
    proxy._sample_lodging_params = types.MethodType(
        SyntheticDataset._sample_lodging_params, proxy
    )
    proxy._apply_lodging_from_params = types.MethodType(
        SyntheticDataset._apply_lodging_from_params, proxy
    )
    num_genotypes = 3
    num_yearsites = 2
    proxy.genotype_ids = [f"g{i}" for i in range(num_genotypes)]
    proxy.yearsite_ids = [f"y{i}" for i in range(num_yearsites)]
    B = num_genotypes * num_yearsites
    T = 40
    intermediate = tensordict.TensorDict(
        {
            "heights_clean": torch.linspace(0.1, 2.0, B * T).reshape(B, T),
            "noise_seed": torch.tensor(1234, dtype=torch.long),
            **(
                {"lodging_seed": torch.tensor(5678, dtype=torch.long)}
                if enable_lodging
                else {}
            ),
            "genotype_indices": torch.arange(num_genotypes).repeat_interleave(
                num_yearsites
            ),
            "yearsite_indices": torch.arange(num_yearsites).repeat(num_genotypes),
            "markers_unique": torch.arange(
                num_genotypes * 3, dtype=torch.float32
            ).reshape(num_genotypes, 3),
            "temperatures_unique": torch.arange(
                num_yearsites * 4 * 24, dtype=torch.float32
            ).reshape(num_yearsites, 4, 24),
        },
        batch_size=[],
        non_blocking=True,
    )
    lazy = _make_lazy_dataset_from_intermediate(proxy, intermediate)
    indices = [4, 0, 3, 3]

    batch = SyntheticDataset.__getitems__(lazy, indices)

    assert len(batch) == len(indices)
    for index, actual in zip(indices, batch, strict=True):
        expected = SyntheticDataset.__getitem__(lazy, index)
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            if torch.is_tensor(expected_value):
                assert actual[key].dtype == expected_value.dtype, key
                assert actual[key].shape == expected_value.shape, key
            _assert_equal_value(actual[key], expected_value, key)
