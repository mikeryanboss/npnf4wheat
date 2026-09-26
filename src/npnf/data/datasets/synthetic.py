"""SyntheticDataset: Synthetic plant height dataset using B-spline response surface.

This module provides a dataset that uses fixed genotype and yearsite pools
to create G x Y combinations for synthetic plant height simulation using
a B-spline temperature x thermal-time response surface.
"""

import hashlib
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

import einops as EO
import tensordict
import torch
from tqdm import tqdm

from npnf.calibration.height.constants import HeightDates
from npnf.data.pools import GenotypePool, YearsitePool
from npnf.data.synthetic.height.genotype import GenotypeParams
from npnf.data.synthetic.height.response_surface import (
    DEFAULT_DEGREE,
    DEFAULT_T_KNOTS,
    DEFAULT_TAU_KNOTS,
    BSplineLUT,
    compute_bspline_basis,
    compute_growth_from_basis,
)
from npnf.data.synthetic.lodging import apply_lodging_from_params, sample_lodging_params

logger = logging.getLogger(__name__)

_GROWTH_CHUNK_SIZE = 8192
_CACHE_KEY_HASH_LEN = 12


class LodgedChunk(TypedDict):
    """Lodged full trajectories for selected rows and draws."""

    height_values_all_nonoise: torch.Tensor
    has_lodged: torch.Tensor
    height_lodged_mask_all: torch.Tensor
    height_days_all: torch.Tensor
    height_values_all_nonoise_nolodge: torch.Tensor
    genotype_id: list[str]
    yearsite_uid: list[str]


def _get_tensor(data: tensordict.TensorDict, key: str) -> torch.Tensor:
    """Return the tensor stored under a string key."""
    return cast("torch.Tensor", data[key])


def _precompute_yearsite_basis(
    temperatures: torch.Tensor, *, dates: HeightDates
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute B-spline T basis and raw thermal time per unique yearsite.

    B_T and raw thermal time depend only on temperature (not genotype).
    Computing per unique yearsite (Y) instead of per sample (G*Y) cuts
    memory from ~82 GB to ~13 MB for a 128-yearsite pool.

    Args:
        temperatures: Hourly temps per yearsite, shape (Y, 274, 24)

    Returns:
        B_T_unique: Temperature basis, shape (Y, D_active, 24, n_T_basis)
        raw_tau_unique: Unnormalized hourly thermal time, shape (Y, D_active*24)
    """
    T_hourly_active = temperatures[:, dates.tau_start_idx :, :]  # (Y, D_active, 24)
    Y = T_hourly_active.shape[0]
    n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1
    T_flat = T_hourly_active.reshape(Y, -1)  # (Y, D_active*24)

    T_clamped = T_flat.clamp(DEFAULT_T_KNOTS[0], DEFAULT_T_KNOTS[-1])
    B_T = compute_bspline_basis(T_clamped.reshape(-1), DEFAULT_T_KNOTS, DEFAULT_DEGREE)
    B_T_unique = B_T.reshape(Y, -1, 24, n_T_basis)

    raw_tau_unique = T_flat.clamp(min=0).cumsum(dim=-1)

    return B_T_unique, raw_tau_unique


@torch.inference_mode()
def _compute_growth_chunked(
    yearsite_indices: torch.Tensor,
    B_T_unique: torch.Tensor,
    raw_tau_unique: torch.Tensor,
    tau_max: torch.Tensor,
    cps: torch.Tensor,
    num_days: int,
    tau_lut: BSplineLUT | None = None,
    *,
    dates: HeightDates,
) -> torch.Tensor:
    """Compute growth heights in memory-efficient chunks.

    Processes ~8192 samples per chunk. Each chunk indexes into the
    precomputed per-yearsite B_T and raw_tau (no recomputation), then
    computes the genotype-specific B_tau basis and factored einsum.

    All input tensors must be on the same device. When CUDA is available,
    passing GPU tensors enables GPU-accelerated einsum and basis evaluation.

    Args:
        yearsite_indices: Per-sample yearsite index, shape (B,)
        B_T_unique: Precomputed T basis per yearsite,
            shape (Y, D_active, 24, n_T_basis)
        raw_tau_unique: Precomputed raw thermal time per yearsite,
            shape (Y, D_active*24)
        tau_max: Per-genotype max thermal time, shape (B,)
        cps: Control point grids, shape (B, n_T_basis, n_tau_basis)
        num_days: Total days in output (274)
        tau_lut: Optional precomputed lookup table for B_tau basis.

    Returns:
        heights_grown: Height trajectories, shape (B, num_days)
    """
    device = cps.device
    B = yearsite_indices.shape[0]
    n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1
    heights_grown = torch.zeros(B, num_days, device=device)

    n_chunks = math.ceil(B / _GROWTH_CHUNK_SIZE)

    for start in tqdm(
        range(0, B, _GROWTH_CHUNK_SIZE),
        total=n_chunks,
        desc="SyntheticDataset growth",
        unit="chunk",
    ):
        end = min(start + _GROWTH_CHUNK_SIZE, B)
        ys_idx = yearsite_indices[start:end]
        chunk_size = end - start

        # Index into precomputed data (no recomputation)
        B_T_chunk = B_T_unique[ys_idx]  # (chunk, D_active, 24, n_T_basis)
        raw_tau_chunk = raw_tau_unique[ys_idx]  # (chunk, D_active*24)

        # Normalize thermal time per genotype
        tau_norm = (raw_tau_chunk / tau_max[start:end].unsqueeze(-1)).clamp(max=1.0)

        # Compute B_tau basis for this chunk
        tau_clamped = tau_norm.clamp(
            DEFAULT_TAU_KNOTS[0].item(), DEFAULT_TAU_KNOTS[-1].item()
        )
        if tau_lut is not None:
            B_tau = tau_lut(tau_clamped.reshape(-1))
        else:
            knots = DEFAULT_TAU_KNOTS.to(device)
            B_tau = compute_bspline_basis(
                tau_clamped.reshape(-1), knots, DEFAULT_DEGREE
            )
        B_tau = B_tau.reshape(chunk_size, -1, 24, n_tau_basis)

        # Shared einsum + clamp + maturity gate + daily sum
        cps_chunk = cps[start:end]
        tau_norm_3d = tau_norm.reshape(chunk_size, -1, 24)
        growth = compute_growth_from_basis(
            B_T_chunk, B_tau, cps_chunk, tau_norm_3d
        )  # (chunk, D_active)

        # Hard cap: zero growth after maximum active period
        growth[:, dates.maximum_growth_period :] = 0.0

        # Integrate to heights
        heights_active = growth.cumsum(dim=-1) / 1000.0
        heights_grown[start:end, dates.tau_start_idx :] = heights_active

    return heights_grown


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic plant height dataset using B-spline response surface.

    Uses a 2D B-spline surface R(T, τ_norm) where T is daily mean temperature
    and τ_norm is normalized thermal time. Growth rate at each day is evaluated
    from the surface and integrated to produce height trajectories.

    Pool instantiation is lazy: `genotype_pool` and `yearsite_pool` are
    callables that are invoked only on cache miss. When the dataset cache
    exists on disk (together with its sidecar metadata file), the factories
    are never called, so training with a pre-computed cache does not require
    the calibration files the factories would otherwise read.

    Args:
        genotype_pool: Zero-arg factory returning a GenotypePool
        yearsite_pool: Zero-arg factory returning a YearsitePool
        scale_noise: Standard deviation of measurement noise
        enable_lodging: Whether to apply lodging simulation
        eval_mode: If True, use deterministic eval subsampling
        num_pre_season_points: Number of pre-season points in eval mode
        num_post_season_points: Number of post-season points in eval mode
        lodging_*: Lodging simulation parameters
        genotype_pool_seed: Seed used to build the full genotype pool.
            Included in the cache key so different pool seeds invalidate
            the cache without inspecting the pool object.
        yearsite_pool_source: Source tag of the yearsite pool (e.g.
            "synthetic" or "weather_data"). Also included in the cache key.
        genotype_indices: Optional range of indices to subset from pool
        yearsite_indices: Optional range of indices to subset from pool
        n_lodging_draws: Number of independent lodging draws per sample
        lodging_seed: Seed for reproducible lodging draws. Required when
            enable_lodging is True and ignored otherwise.
        cache_dir: Optional directory for caching computed samples
    """

    # Set by `npnf.data.datasets.shifted.load_dataset` for frozen test bundles.
    frozen_reference: dict | None = None

    def __init__(
        self,
        genotype_pool: Callable[[], GenotypePool],
        yearsite_pool: Callable[[], YearsitePool],
        scale_noise: float,
        enable_lodging: bool,
        eval_mode: bool,
        num_pre_season_points: int,
        num_post_season_points: int,
        lodging_height_clamp: float,
        lodging_weibull_shape: float,
        lodging_weibull_scale: float,
        lodging_weibull_offset: float,
        lodging_severity_min: float,
        lodging_severity_max: float,
        lodging_transition_steps_min: int,
        lodging_transition_steps_max: int,
        genotype_pool_seed: int,
        yearsite_pool_source: str,
        genotype_indices: range | list[int] | None = None,
        yearsite_indices: range | list[int] | None = None,
        n_lodging_draws: int = 1,
        lodging_seed: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        # Pool identity scalars — included in the cache key so two runs with
        # different pool configs produce different keys without ever touching
        # the pool objects.
        self.genotype_pool_seed = genotype_pool_seed
        self.yearsite_pool_source = yearsite_pool_source

        # Normalize indices to plain lists so they're stable inputs to the
        # cache-key hash (tuples/ranges serialize differently).
        self._genotype_indices: list[int] | None = (
            list(genotype_indices) if genotype_indices is not None else None
        )
        self._yearsite_indices: list[int] | None = (
            list(yearsite_indices) if yearsite_indices is not None else None
        )

        self.scale_noise = scale_noise
        self.enable_lodging = enable_lodging
        self.eval_mode = eval_mode
        self.num_pre_season_points = num_pre_season_points
        self.num_post_season_points = num_post_season_points

        # Lodging parameters
        self.lodging_height_clamp = lodging_height_clamp
        self.lodging_weibull_shape = lodging_weibull_shape
        self.lodging_weibull_scale = lodging_weibull_scale
        self.lodging_weibull_offset = lodging_weibull_offset
        self.lodging_severity_min = lodging_severity_min
        self.lodging_severity_max = lodging_severity_max
        self.lodging_transition_steps_min = lodging_transition_steps_min
        self.lodging_transition_steps_max = lodging_transition_steps_max
        if enable_lodging and lodging_seed is None:
            msg = "lodging_seed is required when enable_lodging=True"
            raise ValueError(msg)
        # Canonicalize the no-lodging configuration so draw-specific settings
        # become true no-ops in both shapes and cache keys.
        self.n_lodging_draws = 1 if not enable_lodging else n_lodging_draws
        self.lodging_seed = None if not enable_lodging else lodging_seed

        # Temperature/day bounds
        self.day_temperature_start = 61
        self.day_temperature_end = self.day_temperature_start + 274
        self.num_days = 274

        # Growth period bounds (from shared constants)
        self.dates = HeightDates()
        self.maximum_growth_period = self.dates.maximum_growth_period
        self.growing_period_start = self.dates.tau_start_idx
        self.growing_period_end = self.dates.growing_period_end_idx

        # Sizes. If indices are given we know them up-front and can check
        # the cache before instantiating pools. Otherwise we must load the
        # pools to measure.
        self.num_genotypes: int | None = (
            len(self._genotype_indices) if self._genotype_indices is not None else None
        )
        self.num_yearsites: int | None = (
            len(self._yearsite_indices) if self._yearsite_indices is not None else None
        )

        # Per-sample string metadata (populated from cache sidecar on cache
        # hit, or from the pool objects on cache miss).
        self.genotype_ids: list[str] | None = None
        self.yearsite_ids: list[str] | None = None
        self._temperatures_unique: torch.Tensor | None = None
        self._sample_yearsite_indices: torch.Tensor | None = None

        # Two-phase init: try cache first (without touching pools), fall
        # back to computing from pools. _compute_intermediate takes the
        # pool objects as explicit arguments — they are not stored on self.
        intermediate: tensordict.TensorDict | None = None
        if (
            cache_dir is not None
            and self.num_genotypes is not None
            and self.num_yearsites is not None
        ):
            cache_path = self._get_cache_path(cache_dir)
            meta_path = self._get_meta_path(cache_dir)
            cache_key_path = self._get_cache_key_config_path(cache_dir)
            if cache_path.exists() and meta_path.exists() and cache_key_path.exists():
                intermediate = self._load_cache(cache_path)
                meta = json.loads(meta_path.read_text())
                self.genotype_ids = meta["genotype_ids"]
                self.yearsite_ids = meta["yearsite_ids"]

        if intermediate is None:
            gp_instance = genotype_pool()
            ys_instance = yearsite_pool()
            if self._genotype_indices is not None:
                gp_instance = gp_instance.subset(self._genotype_indices)
            if self._yearsite_indices is not None:
                ys_instance = ys_instance.subset(self._yearsite_indices)

            self.num_genotypes = len(gp_instance)
            self.num_yearsites = len(ys_instance)
            self.genotype_ids = list(gp_instance.genotype_ids)
            self.yearsite_ids = list(ys_instance.yearsite_ids)

            intermediate = self._compute_intermediate(gp_instance, ys_instance)

            if cache_dir is not None:
                cache_path = self._get_cache_path(cache_dir)
                meta_path = self._get_meta_path(cache_dir)
                self._save_cache(cache_path, intermediate)
                meta_path.write_text(
                    json.dumps(
                        {
                            "genotype_ids": self.genotype_ids,
                            "yearsite_ids": self.yearsite_ids,
                        }
                    )
                )
                self._get_cache_key_config_path(cache_dir).write_text(
                    json.dumps(
                        self._compute_cache_key_config(), indent=2, sort_keys=True
                    )
                )

        if "temperatures_unique" in intermediate:
            self._temperatures_unique = _get_tensor(intermediate, "temperatures_unique")
        if "yearsite_indices" in intermediate:
            self._sample_yearsite_indices = _get_tensor(
                intermediate, "yearsite_indices"
            )

        self._intermediate = intermediate
        self._subsample_indices: torch.Tensor | None = None
        self._height_days_all: torch.Tensor | None = None
        self._noise: torch.Tensor | None = None
        self._lodging_params: dict[str, torch.Tensor] | None = None
        self._prepare_lazy_composition()
        self.samples: tensordict.TensorDict | None = None

    def _compute_cache_key_config(self) -> dict[str, Any]:
        """Return the behavior-affecting config used for cache identity.

        Uses only config-derivable values so the config can be computed without
        instantiating the pools. Every behavior-affecting parameter must be
        included so different configs produce different cache keys.
        """
        return {
            "num_genotypes": self.num_genotypes,
            "num_yearsites": self.num_yearsites,
            "genotype_pool_seed": self.genotype_pool_seed,
            "genotype_indices": self._genotype_indices,
            "yearsite_pool_source": self.yearsite_pool_source,
            "yearsite_indices": self._yearsite_indices,
            "scale_noise": self.scale_noise,
            "enable_lodging": self.enable_lodging,
            "eval_mode": self.eval_mode,
            "num_pre_season_points": self.num_pre_season_points,
            "num_post_season_points": self.num_post_season_points,
            "lodging_height_clamp": self.lodging_height_clamp,
            "lodging_weibull_shape": self.lodging_weibull_shape,
            "lodging_weibull_scale": self.lodging_weibull_scale,
            "lodging_weibull_offset": self.lodging_weibull_offset,
            "lodging_severity_min": self.lodging_severity_min,
            "lodging_severity_max": self.lodging_severity_max,
            "lodging_transition_steps_min": self.lodging_transition_steps_min,
            "lodging_transition_steps_max": self.lodging_transition_steps_max,
            "n_lodging_draws": self.n_lodging_draws,
            "lodging_seed": self.lodging_seed,
            "model": "bspline_surface_v7",
        }

    def _compute_cache_key(self) -> str:
        """Compute a hash key for the current configuration."""
        config = SyntheticDataset._compute_cache_key_config(self)
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:_CACHE_KEY_HASH_LEN]

    def get_dataset_identity(self) -> dict[str, Any]:
        """Return a JSON-serializable identity for deterministic reconstruction."""
        return {
            "dataset_type": "SyntheticDataset",
            "cache_key": self._compute_cache_key(),
            "cache_key_config": self._compute_cache_key_config(),
            "num_conditions": len(self),
        }

    def _compute_noise_seed(self) -> int:
        """Derive a deterministic measurement-noise seed from the cache key."""
        cache_key = SyntheticDataset._compute_cache_key(self)
        digest = hashlib.md5(f"{cache_key}:noise".encode()).digest()
        return int.from_bytes(digest[:8], "big") % (2**63 - 1)

    @staticmethod
    def _slugify_cache_part(value: object) -> str:
        """Convert a cache-name component to a path-safe short token."""
        text = str(value).lower().replace(".", "p")
        chars = [char if char.isalnum() else "-" for char in text]
        return "-".join("".join(chars).strip("-").split("-"))

    @staticmethod
    def _summarize_cache_indices(indices: list[int] | None) -> str:
        """Summarize potentially long index lists for readable cache names."""
        if indices is None:
            return "all"
        if not indices:
            return "empty"
        if len(indices) == 1:
            return str(indices[0])
        first = indices[0]
        last = indices[-1]
        if indices == list(range(first, last + 1)):
            return f"{first}-{last}"
        digest = hashlib.md5(json.dumps(indices).encode()).hexdigest()[:6]
        return f"{first}-{last}-{len(indices)}i-{digest}"

    def _get_cache_name(self) -> str:
        """Get readable cache directory name for the current configuration."""
        config = SyntheticDataset._compute_cache_key_config(self)
        genotype_indices = SyntheticDataset._summarize_cache_indices(
            config["genotype_indices"]
        )
        yearsite_indices = SyntheticDataset._summarize_cache_indices(
            config["yearsite_indices"]
        )
        yearsite_source = SyntheticDataset._slugify_cache_part(
            config["yearsite_pool_source"]
        )
        lodging = (
            f"lodging{config['n_lodging_draws']}-seed{config['lodging_seed']}"
            if config["enable_lodging"]
            else "nolodging"
        )
        parts = [
            "factorial_synth",
            f"g{config['num_genotypes']}",
            f"gi{genotype_indices}",
            f"y{config['num_yearsites']}",
            f"yi{yearsite_indices}",
            f"src{yearsite_source}",
            "eval" if config["eval_mode"] else "train",
            SyntheticDataset._slugify_cache_part(lodging),
            SyntheticDataset._compute_cache_key(self),
        ]
        return "_".join(parts)

    def _get_cache_path(self, cache_dir: str) -> Path:
        """Get cache path for current configuration."""
        return Path(cache_dir) / SyntheticDataset._get_cache_name(self)

    def _get_meta_path(self, cache_dir: str) -> Path:
        """Get dataset metadata path for the current configuration."""
        return SyntheticDataset._get_cache_path(self, cache_dir) / "metadata.json"

    def _get_cache_key_config_path(self, cache_dir: str) -> Path:
        """Get persisted cache-key config path for the current configuration."""
        return SyntheticDataset._get_cache_path(self, cache_dir) / "cache_key.json"

    def _save_cache(self, path: Path, data: tensordict.TensorDict) -> None:
        """Save intermediate data to cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data.save(str(path))

    def _load_cache(self, path: Path) -> tensordict.TensorDict:
        """Load intermediate data from cache."""
        return tensordict.TensorDict.load(str(path))

    def _compute_intermediate(
        self, genotype_pool: GenotypePool, yearsite_pool: YearsitePool
    ) -> tensordict.TensorDict:
        """Compute cacheable intermediate data.

        Phase 1 of init. Produces the compact TensorDict that gets cached:
        per-sample clean height trajectories plus the small per-genotype
        and per-yearsite lookup tables. Noise and (when enabled) lodging
        tensors are not cached — they are regenerated deterministically
        from seeds in `_compose_and_format`, which avoids storing many
        GB of redundant per-sample data for a GxY factorial.

        Pools are passed explicitly (not read from self) so that a cache-hit
        path in __init__ never needs to instantiate them.
        """
        assert self.num_genotypes is not None
        assert self.num_yearsites is not None
        B = self.num_genotypes * self.num_yearsites

        # Factorial indices into the unique per-genotype / per-yearsite tables.
        genotype_indices = torch.arange(self.num_genotypes).repeat_interleave(
            self.num_yearsites
        )
        yearsite_indices = torch.arange(self.num_yearsites).repeat(self.num_genotypes)

        # Per-sample genotype params drive the growth model.
        params = genotype_pool.params[genotype_indices]
        param_dict = {
            name: params[:, i] for i, name in enumerate(GenotypeParams.field_names())
        }

        # ----- B-spline surface forward model -----

        # Control points (genotype-specific, built on CPU)
        n_T_basis = len(DEFAULT_T_KNOTS) + DEFAULT_DEGREE - 1  # 8
        n_tau_basis = len(DEFAULT_TAU_KNOTS) + DEFAULT_DEGREE - 1  # 7
        cps = GenotypeParams.build_clamped_control_grid(params, n_T_basis, n_tau_basis)

        # Precompute per-yearsite on CPU (DEFAULT_T_KNOTS is CPU)
        B_T_unique, raw_tau_unique = _precompute_yearsite_basis(
            yearsite_pool.temperatures, dates=self.dates
        )

        # Move tensors to GPU for accelerated growth computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            logger.info("SyntheticDataset: using GPU for growth computation")
        B_T_unique = B_T_unique.to(device)
        raw_tau_unique = raw_tau_unique.to(device)
        cps = cps.to(device)
        tau_max = param_dict["tau_max"].to(device)  # (B,)
        yearsite_indices_device = yearsite_indices.to(device)

        tau_lut = BSplineLUT()
        tau_lut.basis_grid = tau_lut.basis_grid.to(device)

        matmul_precision = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        try:
            heights_grown = _compute_growth_chunked(
                yearsite_indices=yearsite_indices_device,
                B_T_unique=B_T_unique,
                raw_tau_unique=raw_tau_unique,
                tau_max=tau_max,
                cps=cps,
                num_days=self.num_days,
                tau_lut=tau_lut,
                dates=self.dates,
            ).cpu()
        finally:
            torch.backends.cuda.matmul.fp32_precision = matmul_precision

        # Extend the 274-day temperature window with 30 flat post-season days.
        T = self.num_days + 30
        heights_clean = torch.zeros(B, T)
        heights_clean[:, : self.num_days] = heights_grown
        heights_clean[:, self.num_days :] = heights_grown[:, -1:].expand(-1, 30)

        intermediate = {
            "heights_clean": heights_clean,
            "genotype_indices": genotype_indices,
            "yearsite_indices": yearsite_indices,
            "markers_unique": genotype_pool.markers,
            "temperatures_unique": yearsite_pool.temperatures,
            "noise_seed": torch.tensor(
                SyntheticDataset._compute_noise_seed(self), dtype=torch.long
            ),
        }
        if self.enable_lodging:
            assert self.lodging_seed is not None
            intermediate["lodging_seed"] = torch.tensor(
                self.lodging_seed, dtype=torch.long
            )

        # Entries have heterogeneous leading dims (B for per-sample heights,
        # num_genotypes for markers, num_yearsites for temperatures, and scalar
        # seeds), so the intermediate has no batch dim.
        return tensordict.TensorDict(
            intermediate,  # ty: ignore[invalid-argument-type]
            batch_size=[],
            non_blocking=True,
        )

    def _sample_lodging_params(
        self,
        heights_clean: torch.Tensor,
        n_draws: int,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample shared simulator parameters without changing RNG draw order."""
        return sample_lodging_params(
            heights_clean,
            n_draws,
            generator,
            enable_lodging=self.enable_lodging,
            lodging_height_clamp=self.lodging_height_clamp,
            lodging_weibull_shape=self.lodging_weibull_shape,
            lodging_weibull_scale=self.lodging_weibull_scale,
            lodging_weibull_offset=self.lodging_weibull_offset,
            lodging_severity_min=self.lodging_severity_min,
            lodging_severity_max=self.lodging_severity_max,
            lodging_transition_steps_min=self.lodging_transition_steps_min,
            lodging_transition_steps_max=self.lodging_transition_steps_max,
        )

    def _apply_lodging_from_params(
        self, heights_clean: torch.Tensor, lodging_params: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the shared deterministic simulator lodging transformation."""
        return apply_lodging_from_params(
            heights_clean, lodging_params, enable_lodging=self.enable_lodging
        )

    def _compute_subsample_indices(
        self, num_heights: int, *, eval_mode: bool | None = None
    ) -> torch.Tensor:
        """Compute subsample indices for the given timeline length.

        Args:
            num_heights: Total number of time points (T).
            eval_mode: Override for self.eval_mode. Defaults to self.eval_mode.

        Returns:
            Indices tensor of shape (1, S), broadcastable over batch.
        """
        if eval_mode is None:
            eval_mode = self.eval_mode

        growth_indices = torch.arange(
            self.growing_period_start, self.growing_period_end
        )

        if eval_mode:
            if self.growing_period_start > 0 and self.num_pre_season_points > 0:
                pre_positions = torch.linspace(
                    0, self.growing_period_start - 1, self.num_pre_season_points
                ).long()
            else:
                pre_positions = torch.tensor([], dtype=torch.long)

            post_available = num_heights - self.growing_period_end
            if post_available > 0 and self.num_post_season_points > 0:
                post_positions = (
                    self.growing_period_end
                    + torch.linspace(
                        0, post_available - 1, self.num_post_season_points
                    ).long()
                )
            else:
                post_positions = torch.tensor([], dtype=torch.long)

            selected = torch.cat([pre_positions, growth_indices, post_positions])
        else:
            pre_indices = torch.arange(0, self.growing_period_start)
            post_indices = torch.arange(self.growing_period_end, num_heights)
            non_growth_indices = torch.cat([pre_indices, post_indices])

            num_non_growth = min(15, len(non_growth_indices))
            perm = torch.randperm(len(non_growth_indices))[:num_non_growth]
            sampled_non_growth = non_growth_indices[perm]
            selected = torch.cat([growth_indices, sampled_non_growth]).sort().values

        return selected.unsqueeze(0)  # (1, S)

    def _prepare_lazy_composition(self) -> None:
        """Precompute compact tensors needed for lazy final sample composition."""
        heights_clean = _get_tensor(self._intermediate, "heights_clean")
        B, T = heights_clean.shape

        self._subsample_indices = self._compute_subsample_indices(T)
        self._height_days_all = torch.arange(
            self.day_temperature_start,
            self.day_temperature_start + T,
            device=heights_clean.device,
        )

        noise_generator = torch.Generator(device=heights_clean.device).manual_seed(
            int(_get_tensor(self._intermediate, "noise_seed").item())
        )
        self._noise = (
            torch.randn(
                (B, T),
                dtype=heights_clean.dtype,
                device=heights_clean.device,
                generator=noise_generator,
            )
            * self.scale_noise
        )

        if self.enable_lodging:
            lodging_generator = torch.Generator().manual_seed(
                int(_get_tensor(self._intermediate, "lodging_seed").item())
            )
            self._lodging_params = self._sample_lodging_params(
                heights_clean, self.n_lodging_draws, lodging_generator
            )

    @staticmethod
    def _index_tensor(indices: int | range | list[int] | torch.Tensor) -> torch.Tensor:
        """Normalize scalar or sequence indices to a 1D CPU long tensor."""
        if isinstance(indices, int):
            return torch.tensor([indices], dtype=torch.long)
        if isinstance(indices, range):
            return torch.tensor(list(indices), dtype=torch.long)
        tensor = torch.as_tensor(indices, dtype=torch.long)
        return tensor.reshape(-1)

    def get_height_days_all(self) -> torch.Tensor:
        """Return the shared full day axis without composing final samples."""
        if self._height_days_all is None:
            msg = "Height day axis is not available"
            raise RuntimeError(msg)
        return self._height_days_all

    def get_height_scale(self) -> torch.Tensor:
        """Return the clean-height max without composing lodged trajectories."""
        return _get_tensor(self._intermediate, "heights_clean").max()

    def get_condition_metadata(self) -> dict[str, list[str]]:
        """Return genotype and yearsite IDs in dataset index order."""
        assert self.genotype_ids is not None
        assert self.yearsite_ids is not None
        genotype_indices = _get_tensor(self._intermediate, "genotype_indices")
        yearsite_indices = _get_tensor(self._intermediate, "yearsite_indices")
        return {
            "genotype_id": [
                self.genotype_ids[int(idx.item())] for idx in genotype_indices
            ],
            "yearsite_uid": [
                self.yearsite_ids[int(idx.item())] for idx in yearsite_indices
            ],
        }

    def get_lodged_chunk(
        self,
        indices: int | range | list[int] | torch.Tensor,
        draw_indices: int | range | list[int] | torch.Tensor,
    ) -> LodgedChunk:
        """Compose lodged full trajectories only for the requested rows/draws."""
        sample_indices = SyntheticDataset._index_tensor(indices)
        draw_indices_tensor = SyntheticDataset._index_tensor(draw_indices)
        if (draw_indices_tensor < 0).any() or (
            draw_indices_tensor >= self.n_lodging_draws
        ).any():
            msg = "draw_indices are out of bounds for n_lodging_draws"
            raise IndexError(msg)

        heights_clean = _get_tensor(self._intermediate, "heights_clean").index_select(
            0, sample_indices
        )
        if self.enable_lodging:
            if self._lodging_params is None:
                msg = "Lodging parameters are not available"
                raise RuntimeError(msg)
            lodging_params = {
                key: value.index_select(0, sample_indices).index_select(
                    1, draw_indices_tensor
                )
                for key, value in self._lodging_params.items()
            }
            heights_lodged, has_lodged, lodged_mask = self._apply_lodging_from_params(
                heights_clean, lodging_params
            )
        else:
            B, T = heights_clean.shape
            N = len(draw_indices_tensor)
            heights_lodged = heights_clean.unsqueeze(1).expand(-1, N, -1)
            has_lodged = torch.zeros(
                B, N, dtype=torch.bool, device=heights_clean.device
            )
            lodged_mask = torch.zeros(
                B, N, T, dtype=torch.bool, device=heights_clean.device
            )

        genotype_indices = _get_tensor(
            self._intermediate, "genotype_indices"
        ).index_select(0, sample_indices)
        yearsite_indices = _get_tensor(
            self._intermediate, "yearsite_indices"
        ).index_select(0, sample_indices)
        assert self.genotype_ids is not None
        assert self.yearsite_ids is not None

        return {
            "height_values_all_nonoise": heights_lodged,
            "has_lodged": has_lodged,
            "height_lodged_mask_all": lodged_mask,
            "height_days_all": self.get_height_days_all(),
            "height_values_all_nonoise_nolodge": heights_clean,
            "genotype_id": [
                self.genotype_ids[int(idx.item())] for idx in genotype_indices
            ],
            "yearsite_uid": [
                self.yearsite_ids[int(idx.item())] for idx in yearsite_indices
            ],
        }

    def _compose_and_format(
        self, intermediate: tensordict.TensorDict
    ) -> tensordict.TensorDict:
        """Compose lodged trajectories from intermediate data and build output.

        Phase 2 of init: applies lodging when enabled, subsamples, adds noise,
        and builds the final TensorDict.
        """
        heights_clean = _get_tensor(intermediate, "heights_clean")  # (B, T)
        B, T = heights_clean.shape
        genotype_indices = _get_tensor(intermediate, "genotype_indices")
        yearsite_indices = _get_tensor(intermediate, "yearsite_indices")

        noise_generator = torch.Generator(device=heights_clean.device).manual_seed(
            int(_get_tensor(intermediate, "noise_seed").item())
        )
        noise = (
            torch.randn(
                (B, T),
                dtype=heights_clean.dtype,
                device=heights_clean.device,
                generator=noise_generator,
            )
            * self.scale_noise
        )

        # Reconstruct day values (deterministic, not cached)
        days = EO.repeat(
            EO.rearrange(
                torch.arange(self.day_temperature_start, self.day_temperature_end + 30),
                "d -> 1 d",
            ),
            "1 d -> b d",
            b=B,
        )

        # Compute subsample indices
        indices = self._compute_subsample_indices(T)  # (1, S)
        indices_b = indices.expand(B, -1)  # (B, S)

        # Subsample 1D tensors
        days_sub = days.gather(1, indices_b)  # (B, S)
        clean_sub = heights_clean.gather(1, indices_b)  # (B, S)
        noise_sub = noise.gather(1, indices_b)  # (B, S)

        if self.enable_lodging:
            N = self.n_lodging_draws

            lodging_generator = torch.Generator().manual_seed(
                int(_get_tensor(intermediate, "lodging_seed").item())
            )
            lodging_params = self._sample_lodging_params(
                heights_clean, N, lodging_generator
            )

            # Apply lodging deterministically -> (B, N, T)
            heights_lodged, has_lodged, lodged_mask = self._apply_lodging_from_params(
                heights_clean, lodging_params
            )

            # Subsample N-draw tensors along dim=2
            indices_n = indices.unsqueeze(1).expand(B, N, -1)  # (B, N, S)
            lodged_sub = heights_lodged.gather(2, indices_n)  # (B, N, S)
            mask_sub = lodged_mask.gather(2, indices_n)  # (B, N, S)

            # Observed trajectory: draw 0 + noise (always 1D per sample)
            height_values = lodged_sub[:, 0, :] + noise_sub  # (B, S)

            if N == 1:
                heights_lodged = heights_lodged.squeeze(1)  # (B, T)
                has_lodged = has_lodged.squeeze(1)  # (B,)
                lodged_mask = lodged_mask.squeeze(1)  # (B, T)
                lodged_sub = lodged_sub.squeeze(1)  # (B, S)
                mask_sub = mask_sub.squeeze(1)  # (B, S)
        else:
            height_values = clean_sub + noise_sub
            heights_lodged = heights_clean
            lodged_sub = clean_sub
            has_lodged = torch.zeros(B, dtype=torch.bool, device=heights_clean.device)
            lodged_mask = torch.zeros(
                (B, T), dtype=torch.bool, device=heights_clean.device
            )
            mask_sub = torch.zeros_like(clean_sub, dtype=torch.bool)

        # Build string metadata from indices + cached ID lists. `self.genotype_ids`
        # and `self.yearsite_ids` are populated either from the cache sidecar
        # (cache hit) or from the pool objects (cache miss), so no pool access
        # is needed here.
        assert self.genotype_ids is not None
        assert self.yearsite_ids is not None
        genotype_ids = [self.genotype_ids[int(idx.item())] for idx in genotype_indices]
        yearsite_ids = [self.yearsite_ids[int(idx.item())] for idx in yearsite_indices]
        marker_biallelic_codes = _get_tensor(intermediate, "markers_unique")[
            genotype_indices
        ]

        result_dict = {
            # Subsampled
            "height_days": days_sub,
            "height_days_normalized": (days_sub - 212.5) / 151.5,
            "height_values": height_values,
            "height_values_nonoise": lodged_sub,
            "height_values_nonoise_nolodge": clean_sub,
            # Full timeline
            "height_days_all": days,
            "height_values_all_nonoise": heights_lodged,
            "height_values_all_nonoise_nolodge": heights_clean,
            # Lodging
            "has_lodged": has_lodged,
            "height_lodged_mask": mask_sub,
            "height_lodged_mask_all": lodged_mask,
            # Metadata
            "genotype_index": genotype_indices,
            "yearsite_index": yearsite_indices,
            "genotype_id": genotype_ids,
            "yearsite_uid": yearsite_ids,
            "marker_biallelic_codes": marker_biallelic_codes,
        }

        return tensordict.TensorDict(
            result_dict,  # ty: ignore[invalid-argument-type]
            batch_size=[B],
            non_blocking=True,
        )

    def __len__(self) -> int:
        assert self.num_genotypes is not None
        assert self.num_yearsites is not None
        return self.num_genotypes * self.num_yearsites

    def __getitem__(self, index: int) -> dict:
        """Get a single sample by factorial index.

        The index is decoded as:
        g_index = index // num_yearsites, y_index = index % num_yearsites
        """
        if self._subsample_indices is None or self._noise is None:
            msg = "Lazy sample composition tensors are not available"
            raise RuntimeError(msg)
        heights_clean = _get_tensor(self._intermediate, "heights_clean")[index]
        sample_indices = self._subsample_indices.squeeze(0)
        days_all = self.get_height_days_all()
        days_sub = days_all.index_select(0, sample_indices)
        clean_sub = heights_clean.index_select(0, sample_indices)
        noise_sub = self._noise[index].index_select(0, sample_indices)

        if self.enable_lodging:
            chunk = self.get_lodged_chunk(index, range(self.n_lodging_draws))
            heights_lodged = chunk["height_values_all_nonoise"][0]
            has_lodged = chunk["has_lodged"][0]
            lodged_mask = chunk["height_lodged_mask_all"][0]
            lodged_sub = heights_lodged.index_select(1, sample_indices)
            mask_sub = lodged_mask.index_select(1, sample_indices)
            height_values = lodged_sub[0] + noise_sub

            if self.n_lodging_draws == 1:
                heights_lodged = heights_lodged.squeeze(0)
                has_lodged = has_lodged.squeeze(0)
                lodged_mask = lodged_mask.squeeze(0)
                lodged_sub = lodged_sub.squeeze(0)
                mask_sub = mask_sub.squeeze(0)
        else:
            height_values = clean_sub + noise_sub
            heights_lodged = heights_clean
            lodged_sub = clean_sub
            has_lodged = torch.zeros((), dtype=torch.bool, device=heights_clean.device)
            lodged_mask = torch.zeros_like(heights_clean, dtype=torch.bool)
            mask_sub = torch.zeros_like(clean_sub, dtype=torch.bool)

        genotype_index = _get_tensor(self._intermediate, "genotype_indices")[index]
        yearsite_index = _get_tensor(self._intermediate, "yearsite_indices")[index]
        assert self.genotype_ids is not None
        assert self.yearsite_ids is not None
        sample = {
            "height_days": days_sub,
            "height_days_normalized": (days_sub - 212.5) / 151.5,
            "height_values": height_values,
            "height_values_nonoise": lodged_sub,
            "height_values_nonoise_nolodge": clean_sub,
            "height_days_all": days_all,
            "height_values_all_nonoise": heights_lodged,
            "height_values_all_nonoise_nolodge": heights_clean,
            "has_lodged": has_lodged,
            "height_lodged_mask": mask_sub,
            "height_lodged_mask_all": lodged_mask,
            "genotype_index": genotype_index,
            "yearsite_index": yearsite_index,
            "genotype_id": self.genotype_ids[int(genotype_index.item())],
            "yearsite_uid": self.yearsite_ids[int(yearsite_index.item())],
            "marker_biallelic_codes": _get_tensor(self._intermediate, "markers_unique")[
                genotype_index
            ],
        }
        if self._temperatures_unique is None or self._sample_yearsite_indices is None:
            msg = "Temperature lookup tensors not available"
            raise RuntimeError(msg)
        yearsite_index = int(self._sample_yearsite_indices[index].item())
        sample["temperature_values"] = self._temperatures_unique[yearsite_index]
        return sample

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batch version of ``__getitem__``, used by ``DataLoader`` per batch.

        Returns exactly ``[self[i] for i in indices]``, but composes the lodged
        draws of all indices with one ``get_lodged_chunk`` call instead of one
        call per index.
        """
        if self._subsample_indices is None or self._noise is None:
            msg = "Lazy sample composition tensors are not available"
            raise RuntimeError(msg)
        if self._temperatures_unique is None or self._sample_yearsite_indices is None:
            msg = "Temperature lookup tensors not available"
            raise RuntimeError(msg)
        index = torch.as_tensor(indices, dtype=torch.long)
        heights_clean = _get_tensor(self._intermediate, "heights_clean")[index]
        sample_indices = self._subsample_indices.squeeze(0)
        days_all = self.get_height_days_all()
        days_sub = days_all.index_select(0, sample_indices)
        days_normalized = (days_sub - 212.5) / 151.5
        clean_sub = heights_clean.index_select(1, sample_indices)
        noise_sub = self._noise[index].index_select(1, sample_indices)

        if self.enable_lodging:
            chunk = self.get_lodged_chunk(index, range(self.n_lodging_draws))
            heights_lodged = chunk["height_values_all_nonoise"]  # (B, N, T)
            has_lodged = chunk["has_lodged"]  # (B, N)
            lodged_mask = chunk["height_lodged_mask_all"]  # (B, N, T)
            lodged_sub = heights_lodged.index_select(2, sample_indices)
            mask_sub = lodged_mask.index_select(2, sample_indices)
            height_values = lodged_sub[:, 0] + noise_sub

            if self.n_lodging_draws == 1:
                heights_lodged = heights_lodged.squeeze(1)
                has_lodged = has_lodged.squeeze(1)
                lodged_mask = lodged_mask.squeeze(1)
                lodged_sub = lodged_sub.squeeze(1)
                mask_sub = mask_sub.squeeze(1)
        else:
            height_values = clean_sub + noise_sub
            heights_lodged = heights_clean
            lodged_sub = clean_sub
            has_lodged = torch.zeros(
                len(indices), dtype=torch.bool, device=heights_clean.device
            )
            lodged_mask = torch.zeros_like(heights_clean, dtype=torch.bool)
            mask_sub = torch.zeros_like(clean_sub, dtype=torch.bool)

        genotype_indices = _get_tensor(self._intermediate, "genotype_indices")[index]
        yearsite_indices = _get_tensor(self._intermediate, "yearsite_indices")[index]
        markers = _get_tensor(self._intermediate, "markers_unique")[genotype_indices]
        temperatures = self._temperatures_unique[self._sample_yearsite_indices[index]]
        assert self.genotype_ids is not None
        assert self.yearsite_ids is not None
        return [
            {
                "height_days": days_sub,
                "height_days_normalized": days_normalized,
                "height_values": height_values[row],
                "height_values_nonoise": lodged_sub[row],
                "height_values_nonoise_nolodge": clean_sub[row],
                "height_days_all": days_all,
                "height_values_all_nonoise": heights_lodged[row],
                "height_values_all_nonoise_nolodge": heights_clean[row],
                "has_lodged": has_lodged[row],
                "height_lodged_mask": mask_sub[row],
                "height_lodged_mask_all": lodged_mask[row],
                "genotype_index": genotype_indices[row],
                "yearsite_index": yearsite_indices[row],
                "genotype_id": self.genotype_ids[int(genotype_indices[row])],
                "yearsite_uid": self.yearsite_ids[int(yearsite_indices[row])],
                "marker_biallelic_codes": markers[row],
                "temperature_values": temperatures[row],
            }
            for row in range(len(indices))
        ]
