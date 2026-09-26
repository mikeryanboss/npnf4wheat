"""Load a split of a frozen synthetic test bundle.

A bundle holds the genotype and year-site pools of each split, the exact dataset
settings and the expected dataset identity in `manifest.json`. Create it with
`python -m npnf.scripts.calibration.shifted_test_set run`.
"""

from __future__ import annotations

import json
from pathlib import Path

from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.data.pools import GenotypePool, YearsitePool
from npnf.utils import file_hash


def frozen_reference(output_dir: str | Path, variant: str) -> dict:
    """Bundle reference that prediction configs record for their oracle."""
    return {
        "output_dir": str(output_dir),
        "variant": variant,
        "manifest_sha256": file_hash(Path(output_dir) / "manifest.json"),
    }


def load_dataset(output_dir: str | Path, variant: str) -> SyntheticDataset:
    """Reconstruct exact frozen pools, settings, identities and noise cross-process."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    entry = manifest["datasets"][variant]
    for key in ("genotype_pool", "yearsite_pool"):
        info = entry[key]
        if file_hash(output / info["path"]) != info["sha256"]:
            msg = f"Frozen {variant} {key} hash mismatch"
            raise ValueError(msg)
    genotype_pool = GenotypePool.load(output / entry["genotype_pool"]["path"])
    yearsite_pool = YearsitePool.load(output / entry["yearsite_pool"]["path"])
    settings = dict(entry["parameters"])
    settings["cache_dir"] = str(output / "dataset_cache")
    dataset = SyntheticDataset(
        genotype_pool=lambda: genotype_pool,
        yearsite_pool=lambda: yearsite_pool,
        **settings,
    )
    if dataset.get_dataset_identity() != entry["identity"]:
        msg = f"Frozen {variant} dataset identity mismatch"
        raise ValueError(msg)
    # Predictions record this so that scorers resolve the same frozen oracle.
    dataset.frozen_reference = frozen_reference(output_dir, variant)
    return dataset
