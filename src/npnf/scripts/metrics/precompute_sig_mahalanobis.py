"""Precompute disk-backed signature Mahalanobis oracle artifacts.

The MRCD fit and the CSig-MMD censoring threshold come from the training
population, as in Redhead et al. (https://arxiv.org/abs/2602.10182), where the
threshold is a quantile of the Mahalanobis distances of the training signatures.
The scorers derive the threshold from these distances
(``npnf.metrics.mahalanobis_artifact.censoring_threshold``). Run once with
``--training-reference`` to fit the MRCD on the ``TrainConfig_512k`` reference
draws, then once per test set to measure its oracle draws with that fit. The
test artifact records the training artifact as ``censoring_reference_artifact``;
the CSig-MMD scorers take the threshold from it. The output is the
``oracle_signature_mahalanobis/`` directory consumed by
``score_sig_mahalanobis.py`` and by the Sig-MMD scorers for CSig-MMD
(``--mahalanobis-artifact``).
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from robustcov.mrcd import MRCD

from npnf.data.batch_loader import BatchLoader
from npnf.data.configs.datasets.synthetic import TrainConfig_512k
from npnf.data.configs.datasets.synthetic_test_sets import (
    SyntheticTestSet,
    split_for_dataloader,
)
from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.data.fip1_day_grid import align_trajectories_to_day_axis
from npnf.metrics.mahalanobis_artifact import (
    CONDITION_GENOTYPE_IDS_FILE,
    CONDITION_INDICES_FILE,
    CONDITION_LODGING_RATES_FILE,
    CONDITION_YEARSITE_UIDS_FILE,
    DRAW_INDICES_FILE,
    HAS_LODGED_FILE,
    MAHALANOBIS_DISTANCES_FILE,
    METRIC_DAY_AXIS_NORM_FILE,
    MRCD_FIT_CONDITION_INDICES_FILE,
    MRCD_FIT_DRAW_INDICES_FILE,
    MRCD_FIT_SUPPORT_FILE,
    MRCD_LOCATION_FILE,
    MRCD_PRECISION_FILE,
    NORMALIZED_SIGNATURES_FILE,
    ORACLE_TRAJECTORIES_ON_GRID_FILE,
    PREDICTION_DAY_AXIS_FILE,
    SCHEMA_VERSION,
    load_signature_mahalanobis_artifact,
    normalize_synthetic_dataset_selectors,
    synthetic_dataset_artifact_key,
    synthetic_dataset_artifact_path,
    synthetic_dataset_dataloader_names,
    write_metadata,
)
from npnf.metrics.oracle_config import (
    get_oracle_config_for_dataloader_name,
    get_oracle_config_for_method_dir,
)
from npnf.metrics.signature import (
    SYNTHETIC_SIGNATURE_HEIGHT_SCALE,
    compute_normalized_truncated_sigs,
    mahalanobis_distance,
    to_metric_paths,
)

_SIG_DEPTH = 4


@dataclass(frozen=True)
class _DatasetBundle:
    alias: str
    dataloader_name: str
    dataset: SyntheticDataset


@dataclass(frozen=True)
class _ConditionRef:
    dataset_alias: str
    dataset: SyntheticDataset
    dataset_index: int


def _iter_batch_condition_ids(
    loader: BatchLoader,
) -> Iterator[tuple[list[str], list[str]]]:
    for batch_dict in loader:
        if not isinstance(batch_dict, dict):
            msg = "Expected batch metadata only from the loader"
            raise TypeError(msg)
        metadata = batch_dict["data"].flatten()
        yield (
            [str(g) for g in metadata["genotype_id"]],
            [str(y) for y in metadata["yearsite_uid"]],
        )


def _collect_condition_order(
    loader: BatchLoader, id_to_index: dict[tuple[str, str], int]
) -> tuple[list[str], list[str], list[int]]:
    condition_genotype_ids: list[str] = []
    condition_yearsite_uids: list[str] = []
    dataset_indices: list[int] = []

    for gids, yss in _iter_batch_condition_ids(loader):
        for gid, ys in zip(gids, yss, strict=True):
            condition_genotype_ids.append(gid)
            condition_yearsite_uids.append(ys)
            dataset_indices.append(id_to_index[(gid, ys)])

    return condition_genotype_ids, condition_yearsite_uids, dataset_indices


def _collect_union_condition_order(
    datasets: Sequence[_DatasetBundle],
) -> tuple[list[_ConditionRef], list[str], list[str]]:
    condition_refs: list[_ConditionRef] = []
    condition_genotype_ids: list[str] = []
    condition_yearsite_uids: list[str] = []
    seen: dict[tuple[str, str], str] = {}

    for bundle in datasets:
        metadata = bundle.dataset.get_condition_metadata()
        for dataset_index, (gid_raw, ys_raw) in enumerate(
            zip(metadata["genotype_id"], metadata["yearsite_uid"], strict=True)
        ):
            gid = str(gid_raw)
            ys = str(ys_raw)
            key = (gid, ys)
            if key in seen:
                msg = (
                    "Duplicate oracle condition "
                    f"(genotype_id={gid!r}, yearsite_uid={ys!r}) in datasets "
                    f"{seen[key]!r} and {bundle.alias!r}"
                )
                raise ValueError(msg)
            seen[key] = bundle.alias
            condition_refs.append(
                _ConditionRef(
                    dataset_alias=bundle.alias,
                    dataset=bundle.dataset,
                    dataset_index=dataset_index,
                )
            )
            condition_genotype_ids.append(gid)
            condition_yearsite_uids.append(ys)

    if not condition_refs:
        msg = "No oracle conditions were selected"
        raise ValueError(msg)

    return condition_refs, condition_genotype_ids, condition_yearsite_uids


def _signature_chunk(
    dataset: SyntheticDataset,
    dataset_indices: list[int],
    draw_count: int,
    oracle_day_axis: torch.Tensor,
    prediction_day_axis: torch.Tensor,
    height_scale: float,
    metric_day_axis_norm: torch.Tensor,
    self_kernel_batch_size: int,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    chunk = dataset.get_lodged_chunk(dataset_indices, range(draw_count))
    oracle_grid = align_trajectories_to_day_axis(
        chunk["height_values_all_nonoise"].float(), oracle_day_axis, prediction_day_axis
    )
    B, D, G = oracle_grid.shape
    paths = to_metric_paths(
        oracle_grid.reshape(B * D, G), height_scale, metric_day_axis_norm
    )
    normalized_signatures = compute_normalized_truncated_sigs(
        paths.cpu().numpy(),
        depth=_SIG_DEPTH,
        self_kernel_batch_size=self_kernel_batch_size,
    )
    return normalized_signatures, oracle_grid, chunk["has_lodged"]


def _iter_condition_chunks(
    condition_refs: Sequence[_ConditionRef], draw_count: int, rows: int = 4096
) -> Iterator[tuple[int, SyntheticDataset, list[int]]]:
    """Consecutive conditions of one dataset, about ``rows`` draws per chunk.

    Training references have one draw per condition, so signatures are computed
    for many conditions at once; the row order is the same as one by one.
    """
    size = max(1, rows // draw_count)
    start = 0
    while start < len(condition_refs):
        dataset = condition_refs[start].dataset
        end = start
        while (
            end < len(condition_refs)
            and end - start < size
            and condition_refs[end].dataset is dataset
        ):
            end += 1
        yield (start, dataset, [ref.dataset_index for ref in condition_refs[start:end]])
        start = end


def _update_reservoir(
    reservoir: list[np.ndarray],
    reservoir_condition_indices: list[int],
    reservoir_draw_indices: list[int],
    chunk: np.ndarray,
    *,
    condition_index: int,
    max_size: int,
    n_seen: int,
    rng: np.random.RandomState,
    draw_count: int | None = None,
) -> int:
    """Reservoir-sample rows; row r is condition index + r // draw_count."""
    draw_count = len(chunk) if draw_count is None else draw_count
    for row_index, row in enumerate(chunk):
        n_seen += 1
        row_condition_index = condition_index + row_index // draw_count
        draw_index = row_index % draw_count
        if len(reservoir) < max_size:
            reservoir.append(row.copy())
            reservoir_condition_indices.append(row_condition_index)
            reservoir_draw_indices.append(draw_index)
            continue
        replace_index = rng.randint(0, n_seen)
        if replace_index < max_size:
            reservoir[replace_index] = row.copy()
            reservoir_condition_indices[replace_index] = row_condition_index
            reservoir_draw_indices[replace_index] = draw_index
    return n_seen


def _fit_mrcd_from_stream(
    condition_refs: Sequence[_ConditionRef],
    draw_count: int,
    oracle_day_axis: torch.Tensor,
    prediction_day_axis: torch.Tensor,
    height_scale: float,
    metric_day_axis_norm: torch.Tensor,
    support_fraction: float,
    mrcd_fit_size: int,
    self_kernel_batch_size: int,
) -> tuple[MRCD, int, np.dtype, np.ndarray, np.ndarray, np.ndarray, int]:
    if mrcd_fit_size <= 0:
        msg = "mrcd_fit_size must be positive for bounded streaming precompute"
        raise ValueError(msg)

    rng = np.random.RandomState(42)
    reservoir: list[np.ndarray] = []
    reservoir_condition_indices: list[int] = []
    reservoir_draw_indices: list[int] = []
    n_seen = 0
    signature_dim: int | None = None
    signature_dtype: np.dtype | None = None

    for start, dataset, dataset_indices in _iter_condition_chunks(
        condition_refs, draw_count
    ):
        normalized_signatures, _, _ = _signature_chunk(
            dataset,
            dataset_indices,
            draw_count,
            oracle_day_axis,
            prediction_day_axis,
            height_scale,
            metric_day_axis_norm,
            self_kernel_batch_size,
        )
        signature_dim = int(normalized_signatures.shape[1])
        signature_dtype = normalized_signatures.dtype
        n_seen = _update_reservoir(
            reservoir,
            reservoir_condition_indices,
            reservoir_draw_indices,
            normalized_signatures,
            condition_index=start,
            max_size=mrcd_fit_size,
            n_seen=n_seen,
            rng=rng,
            draw_count=draw_count,
        )
        logger.info(
            "MRCD pass: {}/{} conditions, reservoir={}/{}",
            start + len(dataset_indices),
            len(condition_refs),
            len(reservoir),
            mrcd_fit_size,
        )

    if signature_dim is None or signature_dtype is None:
        msg = "No oracle signatures were generated"
        raise ValueError(msg)

    fit_signatures = np.stack(reservoir, axis=0)
    logger.info("Fitting MRCD on {} / {} signatures", len(fit_signatures), n_seen)
    mrcd = MRCD(support_fraction=support_fraction, random_state=42)
    mrcd.fit(fit_signatures)
    mrcd_fit_condition_indices = np.asarray(reservoir_condition_indices, dtype=np.int64)
    mrcd_fit_draw_indices = np.asarray(reservoir_draw_indices, dtype=np.int64)
    mrcd_fit_support = np.asarray(mrcd.support_, dtype=np.bool_)
    logger.info(
        "MRCD fitted (location shape: {}, support={}/{}, rho={:.4g}, "
        "standardized cond={:.4g})",
        mrcd.location_.shape,
        int(mrcd_fit_support.sum()),
        len(mrcd_fit_support),
        float(mrcd.regularization_),
        float(mrcd.standardized_condition_number_),
    )
    return (
        mrcd,
        signature_dim,
        signature_dtype,
        mrcd_fit_condition_indices,
        mrcd_fit_draw_indices,
        mrcd_fit_support,
        n_seen,
    )


def _prepare_output_dir(output_path: Path) -> None:
    if output_path.exists():
        if not output_path.is_dir():
            msg = f"Output path exists and is not a directory: {output_path}"
            raise NotADirectoryError(msg)
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)


def _write_streamed_arrays(
    output_path: Path,
    condition_refs: Sequence[_ConditionRef],
    condition_genotype_ids: list[str],
    condition_yearsite_uids: list[str],
    draw_count: int,
    oracle_day_axis: torch.Tensor,
    prediction_day_axis: torch.Tensor,
    height_scale: float,
    metric_day_axis_norm: torch.Tensor,
    self_kernel_batch_size: int,
    mrcd_location: np.ndarray,
    mrcd_precision: np.ndarray,
    signature_dim: int,
    signature_dtype: np.dtype,
) -> None:
    n_conditions = len(condition_refs)
    n_total_draws = n_conditions * draw_count
    n_grid_points = int(prediction_day_axis.numel())

    normalized_signatures_mm = np.lib.format.open_memmap(
        output_path / NORMALIZED_SIGNATURES_FILE,
        mode="w+",
        dtype=signature_dtype,
        shape=(n_total_draws, signature_dim),
    )
    distances_mm = np.lib.format.open_memmap(
        output_path / MAHALANOBIS_DISTANCES_FILE,
        mode="w+",
        dtype=np.float64,
        shape=(n_total_draws,),
    )
    has_lodged_mm = np.lib.format.open_memmap(
        output_path / HAS_LODGED_FILE, mode="w+", dtype=np.bool_, shape=(n_total_draws,)
    )
    condition_indices_mm = np.lib.format.open_memmap(
        output_path / CONDITION_INDICES_FILE,
        mode="w+",
        dtype=np.int64,
        shape=(n_total_draws,),
    )
    draw_indices_mm = np.lib.format.open_memmap(
        output_path / DRAW_INDICES_FILE,
        mode="w+",
        dtype=np.int64,
        shape=(n_total_draws,),
    )
    trajectories_mm = np.lib.format.open_memmap(
        output_path / ORACLE_TRAJECTORIES_ON_GRID_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(n_conditions, draw_count, n_grid_points),
    )
    condition_lodging_rates = np.empty(n_conditions, dtype=np.float32)

    for first, dataset, dataset_indices in _iter_condition_chunks(
        condition_refs, draw_count
    ):
        normalized_signatures, oracle_grid, has_lodged = _signature_chunk(
            dataset,
            dataset_indices,
            draw_count,
            oracle_day_axis,
            prediction_day_axis,
            height_scale,
            metric_day_axis_norm,
            self_kernel_batch_size,
        )
        last = first + len(dataset_indices)
        start = first * draw_count
        end = last * draw_count
        normalized_signatures_mm[start:end] = normalized_signatures
        distances_mm[start:end] = mahalanobis_distance(
            normalized_signatures, mrcd_location, mrcd_precision
        )
        has_lodged_np = has_lodged.cpu().numpy().astype(np.bool_)
        has_lodged_mm[start:end] = has_lodged_np.reshape(-1)
        condition_indices_mm[start:end] = np.repeat(np.arange(first, last), draw_count)
        draw_indices_mm[start:end] = np.tile(np.arange(draw_count), last - first)
        trajectories_mm[first:last] = oracle_grid.cpu().numpy().astype(np.float32)
        condition_lodging_rates[first:last] = has_lodged_np.mean(axis=1)
        logger.info("Write pass: {}/{} conditions", last, n_conditions)

    for array in (
        normalized_signatures_mm,
        distances_mm,
        has_lodged_mm,
        condition_indices_mm,
        draw_indices_mm,
        trajectories_mm,
    ):
        array.flush()

    np.save(
        output_path / CONDITION_GENOTYPE_IDS_FILE,
        np.array(condition_genotype_ids, dtype=str),
    )
    np.save(
        output_path / CONDITION_YEARSITE_UIDS_FILE,
        np.array(condition_yearsite_uids, dtype=str),
    )
    np.save(output_path / CONDITION_LODGING_RATES_FILE, condition_lodging_rates)
    np.save(
        output_path / PREDICTION_DAY_AXIS_FILE,
        prediction_day_axis.cpu().numpy().astype(np.float32),
    )
    np.save(
        output_path / METRIC_DAY_AXIS_NORM_FILE,
        metric_day_axis_norm.cpu().numpy().astype(np.float32),
    )
    np.save(output_path / MRCD_LOCATION_FILE, mrcd_location)
    np.save(output_path / MRCD_PRECISION_FILE, mrcd_precision)


def _write_mrcd_fit_support(
    output_path: Path,
    condition_indices: np.ndarray,
    draw_indices: np.ndarray,
    support: np.ndarray,
) -> None:
    np.save(output_path / MRCD_FIT_CONDITION_INDICES_FILE, condition_indices)
    np.save(output_path / MRCD_FIT_DRAW_INDICES_FILE, draw_indices)
    np.save(output_path / MRCD_FIT_SUPPORT_FILE, support.astype(np.bool_))


def _validate_shared_dataset_axes(datasets: Sequence[_DatasetBundle]) -> None:
    reference = datasets[0].dataset
    reference_days = reference.get_height_days_all().float()
    for bundle in datasets[1:]:
        days = bundle.dataset.get_height_days_all().float()
        if not torch.equal(days, reference_days):
            msg = f"Dataset {bundle.alias!r} uses a different height day axis"
            raise ValueError(msg)


def _load_selected_datasets(
    dataset_aliases: Sequence[str], test_set: SyntheticTestSet
) -> tuple[list[_DatasetBundle], tuple[str, ...]]:
    dataloader_names = synthetic_dataset_dataloader_names(
        dataset_aliases, test_set=test_set
    )
    datasets: list[_DatasetBundle] = []
    for alias, dataloader_name in zip(dataset_aliases, dataloader_names, strict=True):
        oracle_cfg = get_oracle_config_for_dataloader_name(dataloader_name)
        logger.info(
            "Loading oracle dataset: {} (dataloader={})",
            oracle_cfg.__name__,
            dataloader_name,
        )
        datasets.append(
            _DatasetBundle(
                alias=alias,
                dataloader_name=dataloader_name,
                dataset=instantiate(oracle_cfg),
            )
        )
    return datasets, dataloader_names


def _precompute(
    method_path: Path,
    dataset_bundles: Sequence[_DatasetBundle],
    output_path: Path,
    metadata: dict,
    *,
    oracle_draws: int,
    self_kernel_batch_size: int,
    reference_artifact: str | None,
    support_fraction: float = 0.8,
    mrcd_fit_size: int = 10_000,
) -> None:
    """Write the artifact; fit the MRCD here (training reference, without
    ``reference_artifact``) or reuse the fit of ``reference_artifact``.

    ``support_fraction`` and ``mrcd_fit_size`` apply to the fit only.
    """
    loader = BatchLoader(method_path, load_predictions=False)
    first_batch = loader.load_batch(0)
    if not isinstance(first_batch, dict):
        msg = f"Expected batch metadata only from {method_path}"
        raise TypeError(msg)
    grid_x = first_batch["grid_points"]["X"].squeeze(-1).float()
    logger.info("Grid: {} points", grid_x.shape[0])

    _validate_shared_dataset_axes(dataset_bundles)
    condition_refs, condition_gids, condition_yss = _collect_union_condition_order(
        dataset_bundles
    )

    draw_count = min(
        [oracle_draws]
        + [int(bundle.dataset.n_lodging_draws) for bundle in dataset_bundles]
    )
    if draw_count <= 0:
        msg = f"No oracle draws available for oracle_draws={oracle_draws}"
        raise ValueError(msg)

    reference_dataset = dataset_bundles[0].dataset
    oracle_day_axis = reference_dataset.get_height_days_all().float()
    metric_day_axis_norm = (grid_x - oracle_day_axis.min()) / (
        oracle_day_axis.max() - oracle_day_axis.min()
    )
    height_scale = SYNTHETIC_SIGNATURE_HEIGHT_SCALE
    logger.info("Height scale: {:.4f}", height_scale)
    logger.info(
        "Params: oracle_draws={}, draw_count={}, depth={}, n_conditions={}",
        oracle_draws,
        draw_count,
        _SIG_DEPTH,
        len(condition_refs),
    )

    if reference_artifact is None:
        (
            mrcd,
            signature_dim,
            signature_dtype,
            mrcd_fit_condition_indices,
            mrcd_fit_draw_indices,
            mrcd_fit_support,
            mrcd_fit_n_seen,
        ) = _fit_mrcd_from_stream(
            condition_refs,
            draw_count,
            oracle_day_axis,
            grid_x,
            height_scale,
            metric_day_axis_norm,
            support_fraction,
            mrcd_fit_size,
            self_kernel_batch_size,
        )
        mrcd_location, mrcd_precision = mrcd.location_, mrcd.precision_
        metadata |= {
            "estimator": "robustcov.mrcd.MRCD",
            "support_fraction": support_fraction,
            "regularization": float(mrcd.regularization_),
            "max_condition_number": float(mrcd.max_condition_number),
            # max_condition_number bounds the standardized covariance, not the raw
            "standardized_condition_number": float(mrcd.standardized_condition_number_),
            "covariance_condition_number": float(mrcd.condition_number_),
            "mrcd_fit_size": mrcd_fit_size,
            "mrcd_fit_n_seen": mrcd_fit_n_seen,
            "mrcd_fit_n_samples": len(mrcd_fit_support),
            "mrcd_fit_n_support": int(mrcd_fit_support.sum()),
        }
    else:
        reference = load_signature_mahalanobis_artifact(reference_artifact)
        if not np.allclose(reference.prediction_day_axis, grid_x.numpy()):
            msg = "Reference artifact uses a different prediction grid"
            raise ValueError(msg)
        mrcd_location = np.asarray(reference.mrcd_location)
        mrcd_precision = np.asarray(reference.mrcd_precision)
        signature_dim = int(mrcd_location.shape[0])
        signature_dtype = reference.normalized_signatures.dtype
        # Scorers take the censoring threshold from the reference distances.
        metadata |= {
            key: reference.metadata[key]
            for key in ("estimator", "support_fraction", "regularization")
        }
        metadata["censoring_reference_artifact"] = str(
            Path(reference_artifact).resolve()
        )
        logger.info("Using the MRCD fit of {}", reference_artifact)

    _prepare_output_dir(output_path)
    if reference_artifact is None:
        _write_mrcd_fit_support(
            output_path,
            mrcd_fit_condition_indices,
            mrcd_fit_draw_indices,
            mrcd_fit_support,
        )
    _write_streamed_arrays(
        output_path,
        condition_refs,
        condition_gids,
        condition_yss,
        draw_count,
        oracle_day_axis,
        grid_x,
        height_scale,
        metric_day_axis_norm,
        self_kernel_batch_size,
        mrcd_location,
        mrcd_precision,
        signature_dim,
        signature_dtype,
    )

    distances = np.load(output_path / MAHALANOBIS_DISTANCES_FILE, mmap_mode="r")
    logger.info(
        "Mahalanobis distances: min={:.4f}, max={:.4f}",
        float(distances.min()),
        float(distances.max()),
    )

    metadata |= {
        "schema_version": SCHEMA_VERSION,
        "depth": _SIG_DEPTH,
        "height_scale": height_scale,
        "self_kernel_batch_size": self_kernel_batch_size,
        "n_conditions": len(condition_refs),
        "draw_count": draw_count,
        "n_total_draws": len(condition_refs) * draw_count,
        "n_grid_points": int(grid_x.numel()),
        "signature_dim": signature_dim,
    }
    write_metadata(output_path, metadata)
    logger.info("Saved to {}", output_path)


def precompute_training_reference(
    method_dir: str,
    support_fraction: float,
    mrcd_fit_size: int,
    self_kernel_batch_size: int,
    output: str,
    lodging_weibull_scale: float | None = None,
    scale_noise: float | None = None,
) -> None:
    """Fit the MRCD on the training reference draws (the censoring reference).

    ``lodging_weibull_scale`` and ``scale_noise`` replace the training lodging scale
    and observation noise, for models trained and scored at another setting.
    """
    overrides = {
        name: value
        for name, value in (
            ("lodging_weibull_scale", lodging_weibull_scale),
            ("scale_noise", scale_noise),
        )
        if value is not None
    }
    logger.info("Loading training dataset: TrainConfig_512k {}", overrides)
    bundle = _DatasetBundle(
        alias="train",
        dataloader_name="synth_train_512k_dataloaders",
        dataset=instantiate(TrainConfig_512k, **overrides),
    )
    _precompute(
        Path(method_dir),
        [bundle],
        Path(output),
        {
            "dataloader_name": "train",
            "artifact_key": "train",
            "dataset_aliases": ["train"],
            "dataloader_names": [bundle.dataloader_name],
            "lodging_weibull_scale": bundle.dataset.lodging_weibull_scale,
            "scale_noise": bundle.dataset.scale_noise,
        },
        oracle_draws=bundle.dataset.n_lodging_draws,
        self_kernel_batch_size=self_kernel_batch_size,
        reference_artifact=None,
        support_fraction=support_fraction,
        mrcd_fit_size=mrcd_fit_size,
    )


def precompute_sig_mahalanobis(
    method_dir: str,
    oracle_draws: int,
    self_kernel_batch_size: int,
    output: str | None,
    datasets: Sequence[str] | None = None,
    *,
    reference_artifact: str,
) -> None:
    """Write a test-set artifact with distances from the training reference fit.

    The oracle distances use the MRCD location and precision of
    ``reference_artifact`` (no own fit), and the scorers take the censoring
    threshold from its distances.
    """
    method_path = Path(method_dir)
    _, source_dataloader_name = get_oracle_config_for_method_dir(method_path)
    # Dataset aliases belong to the test set of the source predictions.
    test_set = split_for_dataloader(source_dataloader_name).test_set
    dataset_aliases = normalize_synthetic_dataset_selectors(datasets, test_set=test_set)
    artifact_key = synthetic_dataset_artifact_key(dataset_aliases, test_set=test_set)
    output_path = (
        Path(output)
        if output is not None
        else synthetic_dataset_artifact_path(dataset_aliases, test_set=test_set)
    )

    logger.info("Source method dataloader: {}", source_dataloader_name)
    logger.info("Selected oracle datasets: {}", list(dataset_aliases))
    logger.info("Artifact key: {}", artifact_key)

    dataset_bundles, dataloader_names = _load_selected_datasets(
        dataset_aliases, test_set
    )
    _precompute(
        method_path,
        dataset_bundles,
        output_path,
        {
            "dataloader_name": artifact_key,
            "source_method_dataloader_name": source_dataloader_name,
            "dataset_aliases": list(dataset_aliases),
            "dataloader_names": list(dataloader_names),
            "artifact_key": artifact_key,
        },
        oracle_draws=oracle_draws,
        self_kernel_batch_size=self_kernel_batch_size,
        reference_artifact=reference_artifact,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute kernel-normalised signatures and MRCD for oracle draws"
    )
    parser.add_argument(
        "--method-dir",
        type=str,
        required=True,
        help="Method dir that provides the prediction grid and source run context",
    )
    parser.add_argument(
        "--training-reference",
        action="store_true",
        help=(
            "Fit the MRCD on the TrainConfig_512k reference draws and write the "
            "training reference artifact (ignores --datasets and --oracle-draws)"
        ),
    )
    parser.add_argument(
        "--lodging-weibull-scale",
        type=float,
        default=None,
        help=(
            "Training lodging Weibull scale of the reference draws (default: the "
            "TrainConfig_512k value; --training-reference only)"
        ),
    )
    parser.add_argument(
        "--scale-noise",
        type=float,
        default=None,
        help=(
            "Training observation noise (m) of the reference draws (default: the "
            "TrainConfig_512k value; --training-reference only)"
        ),
    )
    parser.add_argument(
        "--reference-artifact",
        type=str,
        default="results/sig_mahalanobis/train/oracle_signature_mahalanobis",
        help=(
            "Training reference artifact whose MRCD fit and distances define the "
            "censoring region of a test-set artifact"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Synthetic oracle datasets to union, from the test set of "
            "--method-dir (shifted: seen geno env unseen; legacy: plot genotype "
            "environment unseen), or their exact dataloader names. Default: all "
            "four. Site, year, and no-lodging datasets are not accepted."
        ),
    )
    parser.add_argument(
        "--oracle-draws",
        type=int,
        default=64,
        help="Number of oracle trajectory draws per test condition",
    )
    parser.add_argument(
        "--support-fraction",
        type=float,
        default=0.8,
        help="MRCD support fraction (--training-reference only)",
    )
    parser.add_argument(
        "--mrcd-fit-size",
        type=int,
        default=10_000,
        help=(
            "Maximum number of signatures for MRCD fitting; must be positive "
            "(--training-reference only)"
        ),
    )
    parser.add_argument(
        "--self-kernel-batch-size",
        type=int,
        default=512,
        help="Batch size for self-kernel norm computation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output artifact directory (default: "
            "results/sig_mahalanobis/<dataset-artifact-key>/"
            "oracle_signature_mahalanobis, or the --reference-artifact path with "
            "--training-reference)"
        ),
    )
    args = parser.parse_args()
    if args.training_reference:
        precompute_training_reference(
            method_dir=args.method_dir,
            support_fraction=args.support_fraction,
            mrcd_fit_size=args.mrcd_fit_size,
            self_kernel_batch_size=args.self_kernel_batch_size,
            output=args.output or args.reference_artifact,
            lodging_weibull_scale=args.lodging_weibull_scale,
            scale_noise=args.scale_noise,
        )
    else:
        precompute_sig_mahalanobis(
            method_dir=args.method_dir,
            oracle_draws=args.oracle_draws,
            self_kernel_batch_size=args.self_kernel_batch_size,
            output=args.output,
            datasets=args.datasets,
            reference_artifact=args.reference_artifact,
        )
