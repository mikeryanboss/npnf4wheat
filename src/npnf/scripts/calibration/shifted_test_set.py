"""Generate, report and verify the shifted test population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from npnf.data.configs.datasets.synthetic_test_sets import (
    SyntheticTestSet,
    design_aliases,
)
from npnf.data.datasets.shifted import load_dataset
from npnf.scripts.calibration.shifted_report import (
    Figure,
    Population,
    load_distributions,
    report,
)
from npnf.scripts.utils.outputs import save_json
from npnf.utils import file_hash


def _check_population(population: Population, data: dict[str, np.ndarray]) -> dict:
    """Finite outputs and factorial sizes; tests have 1000 x 32 conditions, 64 draws."""
    for key in (
        "days",
        "clean",
        "heights",
        "observed",
        "genotype_params",
        "temperatures",
        "clean_max_height",
        "timing",
    ):
        if not np.isfinite(data[key]).all():
            msg = f"Nonfinite simulator output: {population}/{key}"
            raise ValueError(msg)
    num_conditions = data["clean_max_height"].size
    num_genotypes = data["genotype_params"].shape[0]
    num_environments = data["temperatures"].shape[0]
    if num_conditions != num_genotypes * num_environments:
        msg = f"Population does not match factorial dimensions: {population}"
        raise ValueError(msg)
    mask = data["lodging"].astype(bool)
    for key in ("lodging_day", "severity"):
        if not np.isfinite(data[key][mask]).all():
            msg = f"Nonfinite lodged event: {population}/{key}"
            raise ValueError(msg)
    if population != Population.TRAIN and mask.shape != (32000, 64):
        msg = f"Final test cohort or draw count changed: {population}, {mask.shape}"
        raise ValueError(msg)
    return {
        "genotypes": num_genotypes,
        "environments": num_environments,
        "conditions": num_conditions,
        "total_lodging_draws": int(mask.size),
    }


def verify(output_dir: str | Path) -> dict:
    """Reject incomplete populations, changed datasets and missing report files."""
    root = Path(output_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if file_hash(path) != artifact["sha256"]:
            msg = f"Dataset artifact checksum mismatch: {path}"
            raise ValueError(msg)
    # Every split rebuilds with its recorded pools, settings and identity.
    for variant in manifest["datasets"]:
        load_dataset(root, variant)
    splits = design_aliases(SyntheticTestSet.SHIFTED)
    # SeedB (oracle-self): same pools and settings, only another lodging seed.
    for split in splits:
        entry, seed_b = (manifest["datasets"][k] for k in (split, f"{split}_seed_b"))
        same = {k: v for k, v in entry["parameters"].items() if k != "lodging_seed"}
        other = {k: v for k, v in seed_b["parameters"].items() if k != "lodging_seed"}
        if (
            same != other
            or entry["genotype_pool"] != seed_b["genotype_pool"]
            or entry["yearsite_pool"] != seed_b["yearsite_pool"]
            or entry["parameters"]["lodging_seed"]
            == seed_b["parameters"]["lodging_seed"]
        ):
            msg = f"SeedB entry of {split} is not an independent lodging draw"
            raise ValueError(msg)
    counts = {}
    identities = {}
    for population, data in load_distributions(root).items():
        counts[population] = _check_population(population, data)
        identities[population] = (
            set(data["genotype_ids"].tolist()),
            set(data["yearsite_ids"].tolist()),
        )
    # 2x2 design: seen identities are training identities, new ones never are.
    train_genotypes, train_yearsites = identities[Population.TRAIN]
    seen_genotypes, seen_yearsites = identities[Population.SEEN]
    new_genotypes, new_yearsites = identities[Population.UNSEEN]
    if not (
        seen_genotypes <= train_genotypes
        and seen_yearsites <= train_yearsites
        and not new_genotypes & train_genotypes
        and not new_yearsites & train_yearsites
        and identities[Population.GENO] == (new_genotypes, seen_yearsites)
        and identities[Population.ENV] == (seen_genotypes, new_yearsites)
    ):
        msg = "Split identities do not follow the seen/geno/env/unseen design"
        raise ValueError(msg)
    for name in Figure:
        for extension in ("png", "pdf"):
            path = root / "figures" / f"{name}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                msg = f"Missing figure: {path}"
                raise ValueError(msg)
    for filename in ("distribution_summary.csv", "report.html", "selection.json"):
        if not (root / filename).is_file():
            msg = f"Missing report artifact: {filename}"
            raise ValueError(msg)
    result = {
        "status": "PASS",
        "populations": counts,
        "manifest_sha256": file_hash(root / "manifest.json"),
        "figures": list(Figure),
    }
    save_json(root / "verification.json", result)
    logger.info("Shifted test population verified: {}", json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "prepare", "report", "verify"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("run", "prepare"):
        from npnf.scripts.calibration.shifted_data import prepare

        prepare(args.output_dir)
    if args.command in ("run", "report"):
        report(args.output_dir)
    if args.command in ("run", "verify"):
        verify(args.output_dir)


if __name__ == "__main__":
    main()
