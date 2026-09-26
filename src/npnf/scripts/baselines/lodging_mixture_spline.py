"""Fit the lodging-mixture spline baseline on the 512k synthetic training split or
on the FIP1 train split.

Writes the fit as a checkpoint run, ``<runs-dir>/LodgingMixtureSpline-512k`` (FIP1:
``LodgingMixtureSpline-FIP-2930``), and the model as stated (a random training
effect for unseen genotypes and year-sites) with the suffix ``-uniform``. The
prediction scripts in ``scripts/predict/`` and ``scripts/launch_prediction_jobs.py``
run both like any neural-process checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from npnf.baselines.lodging_mixture_spline.effects import cumulative_thermal_time
from npnf.baselines.lodging_mixture_spline.fit import TrainingData, fit
from npnf.baselines.lodging_mixture_spline.lodging import DropDetector
from npnf.baselines.lodging_mixture_spline.model import (
    LodgingMixtureSpline,
    UnseenEffect,
)
from npnf.data.configs.datasets.fip1 import (
    TrainHeightDatasetConfig as Fip1TrainHeightDatasetConfig,
)
from npnf.data.configs.datasets.synthetic import TrainConfig_512k
from npnf.data.process.collate import collate_fn_fip1_heights


def load_training_data(
    dataset: torch.utils.data.Dataset,
) -> tuple[TrainingData, torch.Tensor | None, torch.Tensor | None]:
    """Training trajectories from the standard collate on the sorted union of
    their days, and the simulator's lodging labels and masks for the detector
    diagnostics (None for real data). Repeated days of a row are averaged."""
    genotype_index: dict[str, int] = {}
    yearsite_index: dict[str, int] = {}
    markers: list[torch.Tensor | None] = []
    thermal_time = []
    rows, genotypes, yearsites, lodged, lodged_masks = [], [], [], [], []
    for batch in DataLoader(
        dataset, batch_size=4096, collate_fn=collate_fn_fip1_heights
    ):
        for row, (genotype_id, yearsite_id) in enumerate(
            zip(batch["genotype_id"], batch["yearsite_uid"], strict=True)
        ):
            if genotype_id not in genotype_index:
                genotype_index[genotype_id] = len(genotype_index)
                marker = batch["marker"]["Y"][row].reshape(-1)
                markers.append(marker if marker.numel() > 0 else None)
            if yearsite_id not in yearsite_index:
                yearsite_index[yearsite_id] = len(yearsite_index)
                hourly = batch["temperature"]["Y"][row].reshape(1, 274, 24)
                thermal_time.append(cumulative_thermal_time(hourly)[0])
            genotypes.append(genotype_index[genotype_id])
            yearsites.append(yearsite_index[yearsite_id])
        height = batch["height"]
        labelled = batch.get("has_lodged", None) is not None
        if labelled:
            lodged.append(batch["has_lodged"].reshape(-1).bool())
        if height["X"].is_nested:  # rows with their own days
            rows += [
                (days[None, :, 0], values[None, :, 0], None)
                for days, values in zip(
                    height["X"].unbind(), height["Y"].unbind(), strict=True
                )
            ]
        else:
            rows.append(
                (
                    height["X"][..., 0],
                    height["Y"][..., 0],
                    height["lodged_mask"][..., 0].bool() if labelled else None,
                )
            )
    days = torch.unique(torch.cat([row_days.reshape(-1) for row_days, *_ in rows]))
    heights, observed = [], []
    for row_days, values, lodged_mask in rows:
        position = torch.searchsorted(days, row_days)
        empty = torch.zeros(len(values), len(days))
        heights.append(
            empty.scatter_reduce(
                1, position, values.float(), "mean", include_self=False
            )
        )
        observed.append(empty.bool().scatter(1, position, True))
        if lodged_mask is not None:
            lodged_masks.append(empty.bool().scatter(1, position, lodged_mask))
    all_observed = torch.cat(observed)
    num_markers = max(marker.numel() for marker in markers if marker is not None)
    data = TrainingData(
        days=days,
        heights=torch.cat(heights),
        genotype_index=torch.tensor(genotypes),
        yearsite_index=torch.tensor(yearsites),
        genotype_ids=list(genotype_index),
        yearsite_ids=list(yearsite_index),
        markers=torch.stack(
            [
                torch.full((num_markers,), torch.nan) if marker is None else marker
                for marker in markers
            ]
        ),
        thermal_time=torch.stack(thermal_time),
        observed=None if all_observed.all() else all_observed,
    )
    if not lodged:
        return data, None, None
    return data, torch.cat(lodged), torch.cat(lodged_masks)


def detector_diagnostics(
    data: TrainingData, lodged: torch.Tensor, lodged_mask: torch.Tensor
) -> dict[str, float]:
    """Detected against simulated lodging (the fit itself does not use the labels)."""
    detection = DropDetector().detect(data.days, data.heights)
    true_day = torch.where(lodged_mask, data.days.float(), torch.inf).min(dim=1)
    hits = detection.lodged & lodged
    day_error = (detection.day - true_day.values)[hits]
    return {
        "precision": float(hits.sum() / detection.lodged.sum()),
        "recall": float(hits.sum() / lodged.sum()),
        "detected_rate": float(detection.lodged.float().mean()),
        "true_rate": float(lodged.float().mean()),
        "day_error_median": float(day_error.median()),
        "day_error_above_5_fraction": float((day_error.abs() > 5).float().mean()),
    }


def save_run(
    model: LodgingMixtureSpline,
    run_dir: Path,
    dataset_config: object,
    summary: dict[str, object],
) -> None:
    """Write ``model`` as a checkpoint run that ``resolve_model_from_checkpoint``
    and ``prepare_model_and_dataloaders`` load."""
    checkpoint = run_dir / "checkpoints" / "checkpoint-0"
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(checkpoint / "model.safetensors"))
    model_config = {
        "_target_": f"{LodgingMixtureSpline.__module__}.LodgingMixtureSpline",
        "genotype_ids": model.genotype_ids,
        "yearsite_ids": model.yearsite_ids,
        "num_markers": model.genotype_regression.coefficients.shape[1],
        "num_thermal_days": model.yearsite_regression.coefficients.shape[1],
        "num_drop_triples": len(model.drop_triples),
        "unseen_effect": str(model.unseen_effect),
        "pool_size": model.pool_size,
        "seed": model.seed,
    }
    config = {
        "model": model_config,
        "dataloaders": {"train": {"dataset": dataset_config}},
    }
    OmegaConf.save(OmegaConf.create(config), run_dir / "train_config.yaml")
    (run_dir / "fit.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(os.environ.get("NPNF_PROJECT_DIR", ".")),
        help="Directory of the checkpoint runs (default: $NPNF_PROJECT_DIR).",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "fip1"],
        default="synthetic",
        help="Training split: the 512k synthetic split or the FIP1 train split.",
    )
    parser.add_argument(
        "--datasets-offline-path",
        default=None,
        help="Local copy of the FIP1 dataset (default: download mikeboss/FIP1).",
    )
    args = parser.parse_args()

    if args.dataset == "synthetic":
        torch.manual_seed(0)  # the training split draws its off-season days globally
        dataset_config = TrainConfig_512k
        run_name = "LodgingMixtureSpline-512k"
    else:
        dataset_config = Fip1TrainHeightDatasetConfig(
            datasets_offline_path=args.datasets_offline_path
        )
        run_name = "LodgingMixtureSpline-FIP-2930"
    data, lodged, lodged_mask = load_training_data(instantiate(dataset_config))
    model, summary = fit(data)
    if lodged is not None and lodged_mask is not None:
        summary["detector"] = detector_diagnostics(data, lodged, lodged_mask)
    logger.info("Fit: {}", json.dumps(summary))
    structured = OmegaConf.structured(dataset_config)
    save_run(model, args.runs_dir / run_name, structured, summary)
    model.unseen_effect = UnseenEffect.UNIFORM
    save_run(model, args.runs_dir / f"{run_name}-uniform", structured, summary)


if __name__ == "__main__":
    main()
