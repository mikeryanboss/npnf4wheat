"""Disk-backed artifact helpers for signature Mahalanobis metrics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from npnf.data.configs.datasets.synthetic_test_sets import (
    SyntheticTestSet,
    design_aliases,
    design_splits,
    split_named,
)

ARTIFACT_DIR_NAME = "oracle_signature_mahalanobis"
METADATA_FILE = "metadata.json"
SCHEMA_VERSION = 2

PREDICTION_DAY_AXIS_FILE = "prediction_day_axis.npy"
METRIC_DAY_AXIS_NORM_FILE = "metric_day_axis_norm.npy"
MRCD_LOCATION_FILE = "mrcd_location.npy"
MRCD_PRECISION_FILE = "mrcd_precision.npy"
MRCD_FIT_CONDITION_INDICES_FILE = "mrcd_fit_condition_indices.npy"
MRCD_FIT_DRAW_INDICES_FILE = "mrcd_fit_draw_indices.npy"
MRCD_FIT_SUPPORT_FILE = "mrcd_fit_support.npy"
CONDITION_GENOTYPE_IDS_FILE = "condition_genotype_ids.npy"
CONDITION_YEARSITE_UIDS_FILE = "condition_yearsite_uids.npy"
CONDITION_LODGING_RATES_FILE = "condition_lodging_rates.npy"
NORMALIZED_SIGNATURES_FILE = "normalized_signatures.npy"
MAHALANOBIS_DISTANCES_FILE = "mahalanobis_distances.npy"
HAS_LODGED_FILE = "has_lodged.npy"
CONDITION_INDICES_FILE = "condition_indices.npy"
DRAW_INDICES_FILE = "draw_indices.npy"
ORACLE_TRAJECTORIES_ON_GRID_FILE = "oracle_trajectories_on_grid.npy"


@dataclass
class SignatureMahalanobisArtifact:
    """Loaded signature Mahalanobis artifact arrays and metadata."""

    path: Path
    metadata: dict[str, Any]
    prediction_day_axis: np.ndarray
    metric_day_axis_norm: np.ndarray
    mrcd_location: np.ndarray
    mrcd_precision: np.ndarray
    mrcd_fit_condition_indices: np.ndarray | None
    mrcd_fit_draw_indices: np.ndarray | None
    mrcd_fit_support: np.ndarray | None
    condition_genotype_ids: np.ndarray
    condition_yearsite_uids: np.ndarray
    condition_lodging_rates: np.ndarray
    normalized_signatures: np.ndarray
    mahalanobis_distances: np.ndarray
    has_lodged: np.ndarray
    condition_indices: np.ndarray
    draw_indices: np.ndarray
    oracle_trajectories_on_grid: np.ndarray


def normalize_synthetic_dataset_selectors(
    selectors: Sequence[str] | None,
    *,
    test_set: SyntheticTestSet | str = SyntheticTestSet.LEGACY,
) -> tuple[str, ...]:
    """Return canonical synthetic dataset aliases for a selector list.

    Selectors may be the four design-split aliases of the test set or their exact
    dataloader names. The result is de-duplicated and sorted in canonical
    evaluation order.
    """
    alias_to_dataloader = {
        split.alias: split.dataloader_name for split in design_splits(test_set)
    }
    dataloader_to_alias = {
        dataloader_name: alias for alias, dataloader_name in alias_to_dataloader.items()
    }
    raw_selectors = alias_to_dataloader if selectors is None else selectors
    aliases: list[str] = []
    seen: set[str] = set()
    for raw_selector in raw_selectors:
        selector = str(raw_selector).strip().lower()
        if selector in alias_to_dataloader:
            alias = selector
        elif selector in dataloader_to_alias:
            alias = dataloader_to_alias[selector]
        else:
            valid = list(alias_to_dataloader) + list(dataloader_to_alias)
            msg = (
                f"Unsupported {test_set} synthetic dataset selector "
                f"{raw_selector!r}. Use one of {valid}; site, year, and no-lodging "
                "datasets are not part of the signature-Mahalanobis dataset union."
            )
            raise ValueError(msg)
        if alias in seen:
            msg = f"Duplicate synthetic dataset selector for {alias!r}"
            raise ValueError(msg)
        seen.add(alias)
        aliases.append(alias)

    if not aliases:
        msg = "At least one synthetic dataset selector is required"
        raise ValueError(msg)

    return tuple(alias for alias in alias_to_dataloader if alias in seen)


def synthetic_dataset_dataloader_names(
    aliases: Sequence[str],
    *,
    test_set: SyntheticTestSet | str = SyntheticTestSet.LEGACY,
) -> tuple[str, ...]:
    """Return dataloader names for canonical synthetic dataset aliases."""
    return tuple(split_named(alias, test_set).dataloader_name for alias in aliases)


def synthetic_dataset_artifact_key(
    selectors: Sequence[str] | None = None,
    *,
    test_set: SyntheticTestSet | str = SyntheticTestSet.LEGACY,
) -> str:
    """Return the artifact key for a synthetic dataset selector list.

    The legacy four-split union keeps its original key ``all``; unions of the other
    test sets are prefixed with the test-set name (``shifted``, ``lodging1_3``, ...)
    so that two test sets never share an artifact.
    """
    aliases = normalize_synthetic_dataset_selectors(selectors, test_set=test_set)
    if len(aliases) == 1:
        return split_named(aliases[0], test_set).dataloader_name
    if test_set == SyntheticTestSet.LEGACY:
        if aliases == design_aliases(test_set):
            return "all"
        return "_".join(aliases)
    if aliases == design_aliases(test_set):
        return str(test_set)
    return "_".join((str(test_set), *aliases))


def synthetic_dataset_artifact_path(
    selectors: Sequence[str] | None = None,
    *,
    test_set: SyntheticTestSet | str = SyntheticTestSet.LEGACY,
    base_path: str | Path = Path("results/sig_mahalanobis"),
) -> Path:
    """Return the artifact directory for a synthetic dataset selector list."""
    return (
        Path(base_path)
        / synthetic_dataset_artifact_key(selectors, test_set=test_set)
        / ARTIFACT_DIR_NAME
    )


def default_artifact_path(dataloader_name: str) -> Path:
    """Return the default artifact directory for a dataloader."""
    return Path("results/sig_mahalanobis") / dataloader_name / ARTIFACT_DIR_NAME


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write artifact metadata as sorted JSON."""
    (path / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_metadata(path: Path) -> dict[str, Any]:
    """Load artifact metadata."""
    return json.loads((path / METADATA_FILE).read_text())


def _load_optional_array(
    artifact_path: Path, filename: str, mmap_mode: Literal["r+", "r", "w+", "c"] | None
) -> np.ndarray | None:
    path = artifact_path / filename
    if not path.exists():
        return None
    return np.load(path, mmap_mode=mmap_mode)


def load_signature_mahalanobis_artifact(
    path: str | Path, *, mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r"
) -> SignatureMahalanobisArtifact:
    """Load the new directory-format signature Mahalanobis artifact."""
    artifact_path = Path(path)
    if not artifact_path.is_dir():
        msg = f"Expected artifact directory at {artifact_path}"
        raise NotADirectoryError(msg)

    metadata = load_metadata(artifact_path)
    return SignatureMahalanobisArtifact(
        path=artifact_path,
        metadata=metadata,
        prediction_day_axis=np.load(
            artifact_path / PREDICTION_DAY_AXIS_FILE, mmap_mode=mmap_mode
        ),
        metric_day_axis_norm=np.load(
            artifact_path / METRIC_DAY_AXIS_NORM_FILE, mmap_mode=mmap_mode
        ),
        mrcd_location=np.load(artifact_path / MRCD_LOCATION_FILE),
        mrcd_precision=np.load(artifact_path / MRCD_PRECISION_FILE),
        mrcd_fit_condition_indices=_load_optional_array(
            artifact_path, MRCD_FIT_CONDITION_INDICES_FILE, mmap_mode
        ),
        mrcd_fit_draw_indices=_load_optional_array(
            artifact_path, MRCD_FIT_DRAW_INDICES_FILE, mmap_mode
        ),
        mrcd_fit_support=_load_optional_array(
            artifact_path, MRCD_FIT_SUPPORT_FILE, mmap_mode
        ),
        condition_genotype_ids=np.load(
            artifact_path / CONDITION_GENOTYPE_IDS_FILE, mmap_mode=mmap_mode
        ),
        condition_yearsite_uids=np.load(
            artifact_path / CONDITION_YEARSITE_UIDS_FILE, mmap_mode=mmap_mode
        ),
        condition_lodging_rates=np.load(
            artifact_path / CONDITION_LODGING_RATES_FILE, mmap_mode=mmap_mode
        ),
        normalized_signatures=np.load(
            artifact_path / NORMALIZED_SIGNATURES_FILE, mmap_mode=mmap_mode
        ),
        mahalanobis_distances=np.load(
            artifact_path / MAHALANOBIS_DISTANCES_FILE, mmap_mode=mmap_mode
        ),
        has_lodged=np.load(artifact_path / HAS_LODGED_FILE, mmap_mode=mmap_mode),
        condition_indices=np.load(
            artifact_path / CONDITION_INDICES_FILE, mmap_mode=mmap_mode
        ),
        draw_indices=np.load(artifact_path / DRAW_INDICES_FILE, mmap_mode=mmap_mode),
        oracle_trajectories_on_grid=np.load(
            artifact_path / ORACLE_TRAJECTORIES_ON_GRID_FILE, mmap_mode=mmap_mode
        ),
    )


def censoring_threshold(
    artifact: SignatureMahalanobisArtifact, alpha: float | None, lodged_recall: float
) -> tuple[float, float]:
    """CSig-MMD quantile ``alpha`` and threshold ``c_squared`` of the reference.

    With ``alpha`` given, ``c_squared`` is the alpha quantile of the reference
    distances. Without it, ``c_squared`` keeps the fraction ``lodged_recall`` of the
    lodged reference draws above the threshold, and ``alpha`` is the fraction of all
    reference draws at or below it. This places the threshold just below the lodged
    draws for any lodging rate.

    Artifacts that use the training fit record the training artifact as
    ``censoring_reference_artifact``; older artifacts use their own distances.
    """
    reference_path = artifact.metadata.get("censoring_reference_artifact")
    reference = (
        artifact
        if reference_path is None
        else load_signature_mahalanobis_artifact(reference_path)
    )
    if not (
        np.array_equal(artifact.mrcd_location, reference.mrcd_location)
        and np.array_equal(artifact.mrcd_precision, reference.mrcd_precision)
    ):
        msg = "Oracle distances and censoring reference use different fits"
        raise ValueError(msg)
    distances = np.asarray(reference.mahalanobis_distances)
    if alpha is not None:
        return alpha, float(np.quantile(distances, alpha))
    lodged_distances = distances[np.asarray(reference.has_lodged, dtype=np.bool_)]
    if lodged_distances.size == 0:
        msg = "The censoring reference has no lodged draws; pass alpha explicitly"
        raise ValueError(msg)
    c_squared = float(np.quantile(lodged_distances, 1 - lodged_recall))
    return float(np.mean(distances <= c_squared)), c_squared
