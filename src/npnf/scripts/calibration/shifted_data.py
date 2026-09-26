"""Frozen shifted test population.

The test identities are selected from the unchanged simulator: the lowest
1000 of 4096 training genotypes by mean clean maximum height and the lowest 32 of
128 training year-sites by growing-season temperature. New genotypes and
year-sites come from new pools of the unchanged generators and pass the same
thresholds. Growth, weather, observation noise and lodging are delegated
unchanged to the production simulator.

Each population is exported to `distributions/<population>.npz` for the report:
clean and lodged noise-free paths plus observed paths (lodged plus the production
measurement noise) of a fixed subsample, using the first lodging draw; lodging
events of all draws, with the lodging day in absolute days (masked by `lodging`)
and the severity as the final/peak height fraction.
"""

# This experiment reuses the simulator's private cached state without changing it.
# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra_zen import instantiate
from loguru import logger

from npnf.data.configs.datasets.synthetic import (
    TestConfig_Unseen,
    TestConfig_Unseen_SeedB,
    TrainConfig_512k,
)
from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.data.pools import GenotypePool, YearsitePool
from npnf.data.pools.genotype_pool import _normalize_tensor
from npnf.data.synthetic.height.params import load_height_pool_params
from npnf.data.synthetic.temperature.generator import SyntheticTemperatureGenerator
from npnf.data.synthetic.temperature.params import load_temperature_params
from npnf.scripts.utils.outputs import save_json
from npnf.utils import file_hash


def _settings(config):
    """Resolve the existing config, without instantiating its dataset or pools."""
    from omegaconf import OmegaConf

    settings = cast(
        "dict[str, Any]",
        OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
    )
    settings.pop("_target_")
    settings.pop("genotype_pool")
    settings.pop("yearsite_pool")
    # Hydra nested index config is resolved independently, not reimplemented.
    settings["yearsite_indices"] = list(instantiate(config.yearsite_indices))
    settings["genotype_indices"] = list(instantiate(config.genotype_indices))
    return settings


def _dataset(genotype_pool, yearsite_pool, settings):
    return SyntheticDataset(
        genotype_pool=lambda: genotype_pool,
        yearsite_pool=lambda: yearsite_pool,
        **settings,
    )


def _existing(config, cache_dir):
    """Load an original dataset from its cache; refuse to regenerate it.

    The dataset asks its pool factories for pools only when the cache is missing.
    """
    settings = _settings(config)
    settings["cache_dir"] = str(cache_dir)

    def refuse():
        msg = f"Original {config.__name__} cache unavailable; refusing regeneration"
        raise FileNotFoundError(msg)

    return SyntheticDataset(
        genotype_pool=refuse, yearsite_pool=refuse, **settings
    ), settings


def candidate_genotypes(height_params, bounds, seed=4240, count=10000):
    """New genotypes from the unchanged generator, with training normalization."""
    pool = GenotypePool.sample(count, seed, height_pool_params=height_params)
    pool.bounds = bounds
    pool._markers = _normalize_tensor(pool.params, bounds)
    pool.genotype_ids = [f"Shifted-{identity}" for identity in pool.genotype_ids]
    return pool


def candidate_yearsites(temperature_params, seed=4241, sites=32, years=20):
    """New sites and years from the unchanged temperature generator."""
    generator = SyntheticTemperatureGenerator(
        num_sites=sites,
        num_years=years,
        seed=seed,
        temperature_params=temperature_params,
    )
    return YearsitePool(
        torch.as_tensor(generator.generate_all()),
        [
            f"Shifted-S{site:02d}_Y{year:02d}"
            for site in range(sites)
            for year in range(years)
        ],
        "synthetic",
        seed,
    )


def _genotype_height(genotype_pool, yearsite_pool, genotype_indices, panel, chunk=2500):
    """Mean clean maximum height of each genotype over a fixed year-site panel."""
    settings = _settings(TestConfig_Unseen)
    settings.update(cache_dir=None, n_lodging_draws=1, yearsite_indices=panel)
    heights = []
    for start in range(0, len(genotype_indices), chunk):
        settings["genotype_indices"] = genotype_indices[start : start + chunk]
        dataset = _dataset(genotype_pool, yearsite_pool, settings)
        clean = dataset._intermediate["heights_clean"]
        peak = clean.max(1).values.double().cpu()
        heights.append(peak.reshape(dataset.num_genotypes, -1).mean(1))
        del dataset
    return torch.cat(heights).numpy()


def _growing_temperature(yearsite_pool, indices):
    """Mean temperature from day 200 to day 321 (the axis starts at day 61)."""
    values = yearsite_pool.temperatures[torch.as_tensor(indices)].double()
    return values[:, 139:261].mean((1, 2)).numpy()


def _export(output, label, dataset, genotype_pool, yearsite_pool):
    directory = output / "distributions"
    directory.mkdir(exist_ok=True)
    clean = dataset._intermediate["heights_clean"]
    days = dataset.get_height_days_all()
    sample = torch.linspace(0, len(clean) - 1, min(512, len(clean))).long()
    peak, peak_index = clean.max(1)
    timing = (clean >= peak[:, None] * 0.95).long().argmax(1)
    lodging = dataset._lodging_params
    event_day = peak_index[:, None] + 1 + lodging["offset"]
    occurred = lodging["will_lodge"] & (event_day < clean.shape[1])
    # Non-event timing/severity use explicit masks, avoiding nonfinite artifacts.
    lodged, _, _ = dataset._apply_lodging_from_params(
        clean[sample], {k: v[sample, :1] for k, v in lodging.items()}
    )
    arrays = {
        "days": days,
        "clean": clean[sample],
        "heights": lodged[:, 0],
        "observed": lodged[:, 0] + dataset._noise[sample],
        "temperatures": yearsite_pool.temperatures,
        "genotype_params": genotype_pool.params,
        "clean_max_height": peak,
        "timing": days[timing],
        "peak_day": days[peak_index],
        "lodging": occurred,
        "lodging_day": event_day + int(days[0]),
        "severity": lodging["severity"],
        "genotype_ids": np.array(genotype_pool.genotype_ids),
        "yearsite_ids": np.array(yearsite_pool.yearsite_ids),
    }
    saved: dict[str, Any] = {
        k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in arrays.items()
    }
    np.savez_compressed(directory / f"{label}.npz", **saved)
    result = {
        "num_genotypes": len(genotype_pool),
        "num_yearsites": len(yearsite_pool),
        "num_conditions": len(clean),
        "draws": dataset.n_lodging_draws,
    }
    logger.info("Distribution {}: {}", label, result)
    return result


def _select(
    output,
    genotype_pool,
    yearsite_pool,
    height_params,
    temperature_params,
    *,
    genotypes=1000,
    yearsites=32,
):
    """Count selection on the training population; same thresholds for new ones."""
    train = _settings(TrainConfig_512k)
    train_genotypes = np.array(sorted(train["genotype_indices"]))
    train_yearsites = np.array(sorted(train["yearsite_indices"]))
    panel = np.sort(
        np.random.default_rng(0).choice(train_yearsites, yearsites, replace=False)
    ).tolist()
    new_genotype_pool = candidate_genotypes(height_params, genotype_pool.bounds)
    new_yearsite_pool = candidate_yearsites(temperature_params)
    height = _genotype_height(
        genotype_pool, yearsite_pool, train_genotypes.tolist(), panel
    )
    new_height = _genotype_height(
        new_genotype_pool, yearsite_pool, list(range(len(new_genotype_pool))), panel
    )
    temperature = _growing_temperature(yearsite_pool, train_yearsites)
    new_temperature = _growing_temperature(
        new_yearsite_pool, np.arange(len(new_yearsite_pool))
    )
    height_threshold = float(np.sort(height)[genotypes - 1])
    temperature_threshold = float(np.sort(temperature)[yearsites - 1])
    available_genotypes = np.flatnonzero(new_height <= height_threshold)
    available_yearsites = np.flatnonzero(new_temperature <= temperature_threshold)
    rng = np.random.default_rng(5)
    selection = {
        "rule": (
            f"lowest {genotypes} of {len(train_genotypes)} training genotypes by mean "
            f"clean maximum height over a fixed panel of {yearsites} training "
            f"year-sites (random, seed 0); lowest {yearsites} of "
            f"{len(train_yearsites)} training year-sites by mean temperature from "
            "day 200 to day 321; new genotypes and year-sites at or below the same "
            "thresholds, random subset (seed 5)"
        ),
        "device": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        ),
        "panel_yearsites": panel,
        "genotype_height_threshold_m": height_threshold,
        "yearsite_temperature_threshold_c": temperature_threshold,
        "new_genotype_pool": {
            "seed": new_genotype_pool.seed,
            "count": len(new_genotype_pool),
        },
        "new_yearsite_pool": {
            "seed": new_yearsite_pool.seed,
            "count": len(new_yearsite_pool),
        },
        "available_new_genotypes": len(available_genotypes),
        "available_new_yearsites": len(available_yearsites),
        "training_genotypes": np.sort(
            train_genotypes[np.argsort(height, kind="stable")[:genotypes]]
        ).tolist(),
        "training_yearsites": np.sort(
            train_yearsites[np.argsort(temperature, kind="stable")[:yearsites]]
        ).tolist(),
        "new_genotypes": np.sort(
            rng.choice(available_genotypes, genotypes, replace=False)
        ).tolist(),
        "new_yearsites": np.sort(
            rng.choice(available_yearsites, yearsites, replace=False)
        ).tolist(),
    }
    save_json(output / "selection.json", selection)
    return selection, new_genotype_pool, new_yearsite_pool


def prepare(output_dir: str | Path) -> dict:
    """Select the identities and materialize the four frozen test splits.

    An existing bundle is never changed; `verify` checks it.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    if (output / "manifest.json").exists():
        logger.info("Frozen bundle exists; not regenerating {}", output)
        return json.loads((output / "manifest.json").read_text())
    source = Path(os.environ.get("NPNF_ORIGINAL_ROOT", "/workspace"))
    calibration = Path(
        os.environ.get("NPNF_CALIBRATION_DIR", source / "results/calibration")
    )
    height_params_path = calibration / "height/params_pool.json"
    temperature_params_path = calibration / "temperature/params_pool.json"
    genotype_pool_path = source / "dataset_cache/pools/genotype.pt"
    yearsite_pool_path = source / "dataset_cache/pools/yearsite.pt"
    genotype_pool = GenotypePool.load(genotype_pool_path)
    yearsite_pool = YearsitePool.load(yearsite_pool_path)
    sources = {
        str(path): file_hash(path)
        for path in (
            height_params_path,
            temperature_params_path,
            genotype_pool_path,
            yearsite_pool_path,
        )
    }
    logger.info(
        "prepare CUDA_VISIBLE_DEVICES={} original_root={}; source hashes={}",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        source,
        sources,
    )
    selection, new_genotype_pool, new_yearsite_pool = _select(
        output,
        genotype_pool,
        yearsite_pool,
        load_height_pool_params(height_params_path),
        load_temperature_params(temperature_params_path),
    )
    training_genotypes = genotype_pool.subset(selection["training_genotypes"])
    training_yearsites = yearsite_pool.subset(selection["training_yearsites"])
    new_genotypes = new_genotype_pool.subset(selection["new_genotypes"])
    new_yearsites = new_yearsite_pool.subset(selection["new_yearsites"])
    layout = {
        "seen": (training_genotypes, training_yearsites),
        "geno": (new_genotypes, training_yearsites),
        "env": (training_genotypes, new_yearsites),
        "unseen": (new_genotypes, new_yearsites),
    }
    pools = output / "pools"
    pools.mkdir(exist_ok=True)
    seed_b = _settings(TestConfig_Unseen_SeedB)["lodging_seed"]
    datasets = {}
    summaries = {}
    for variant, (genotypes, environments) in layout.items():
        genotype_file = pools / f"{variant}-genotype.pt"
        yearsite_file = pools / f"{variant}-yearsite.pt"
        genotypes.save(genotype_file)
        environments.save(yearsite_file)
        settings = _settings(TestConfig_Unseen)
        settings.update(
            genotype_indices=list(range(len(genotypes))),
            yearsite_indices=list(range(len(environments))),
            genotype_pool_seed=genotypes.seed,
            yearsite_pool_source="shifted-"
            + hashlib.sha256(
                (file_hash(genotype_file) + file_hash(yearsite_file)).encode()
            ).hexdigest(),
            cache_dir=str(output / "dataset_cache"),
        )
        pool_entries = {
            "genotype_pool": {
                "path": str(genotype_file.relative_to(output)),
                "sha256": file_hash(genotype_file),
            },
            "yearsite_pool": {
                "path": str(yearsite_file.relative_to(output)),
                "sha256": file_hash(yearsite_file),
            },
        }
        # SeedB: the same conditions with independent lodging draws (oracle-self).
        for name, lodging_seed in (
            (variant, settings["lodging_seed"]),
            (f"{variant}_seed_b", seed_b),
        ):
            settings["lodging_seed"] = lodging_seed
            dataset = _dataset(genotypes, environments, settings)
            if name == variant:
                summaries[variant] = _export(
                    output, variant, dataset, genotypes, environments
                )
            parameters = {k: v for k, v in settings.items() if k != "cache_dir"}
            datasets[name] = {
                "num_genotypes": dataset.num_genotypes,
                "num_yearsites": dataset.num_yearsites,
                "num_conditions": len(dataset),
                "parameters": parameters,
                "identity": dataset.get_dataset_identity(),
                **pool_entries,
            }
            del dataset
    identities = {}
    for label, config in (("train", TrainConfig_512k), ("original", TestConfig_Unseen)):
        dataset, settings = _existing(config, source / "dataset_cache")
        summaries[label] = _export(
            output,
            label,
            dataset,
            genotype_pool.subset(settings["genotype_indices"]),
            yearsite_pool.subset(settings["yearsite_indices"]),
        )
        identities[label] = dataset.get_dataset_identity()
        del dataset
    for path, digest in sources.items():
        if file_hash(path) != digest:
            msg = f"Original input mutated: {path}"
            raise RuntimeError(msg)
    manifest = {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        # The identity lists are in selection.json, which is a hashed artifact.
        "selection": {
            key: value
            for key, value in selection.items()
            if not isinstance(value, list)
        },
        "sources": sources,
        "datasets": datasets,
        "distributions": summaries,
        "training_identity": identities["train"],
        "original_unseen_identity": identities["original"],
        "normalization_bounds": genotype_pool.bounds.to_dict(),
        "unchanged": [
            "genotype and temperature generators and their calibrated parameters",
            "growth equations",
            "lodging law and all parameters",
            "height and temperature measurement noise",
            "weather AR process",
            "daily fluctuations",
            "marker normalization (training bounds)",
            "day units",
        ],
        "shifted": [
            (
                "selected population: shorter genotypes and cooler year-sites, by "
                "thresholds on simulator outputs"
            )
        ],
        "artifacts": [
            {"path": str(path.relative_to(output)), "sha256": file_hash(path)}
            for path in sorted(output.rglob("*"))
            if path.is_file()
        ],
    }
    save_json(output / "manifest.json", manifest)
    logger.info(
        "Frozen manifest {} ready; original inputs unchanged",
        file_hash(output / "manifest.json"),
    )
    return manifest
