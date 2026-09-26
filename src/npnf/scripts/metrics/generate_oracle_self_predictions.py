"""Write an independent oracle draw as a fake predictions directory.

Instantiates a seed-B synthetic test-dataset config, grid-aligns its
``height_values_all_nonoise`` trajectories to a reference prediction grid,
and writes the result in the same on-disk layout that
``src/npnf/scripts/predict/*.py`` produces. The output directory can then
be scored with ``sig_mmd.py`` as if it were any other method's predictions,
giving an oracle-vs-oracle noise-floor baseline evaluated through the exact
same pipeline as a model run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import tensordict
from hydra.utils import instantiate
from loguru import logger

from npnf.data.configs.datasets.synthetic_test_sets import SyntheticTestSet, split_named
from npnf.data.fip1_day_grid import align_trajectories_to_day_axis
from npnf.scripts.utils.prediction import load_grid_points, save_batch_results

METHOD_NAME = "oracle_self"


def generate_oracle_self_predictions(
    seed_b_config: Any,
    reference_method_dir: Path,
    *,
    dataloader_name: str,
    batch_size: int | None = None,
) -> None:
    """Write an oracle-self predictions directory for a given split.

    Output goes to ``$NPNF_RESULTS_DIR/<dataloader_name>/oracle_self/``, parallel
    to real model runs under ``$NPNF_RESULTS_DIR/<dataloader_name>/<model>/...``.

    Args:
        seed_b_config: A hydra-zen config class for a ``TestConfig_*_SeedB``
            synthetic dataset.
        reference_method_dir: Path to any real method_dir whose grid the
            oracle-self draw should be aligned to. Only batch 0's
            ``grid_points`` is consulted.
        dataloader_name: Name of the dataloader split for this oracle (e.g.
            ``"synth_test_plot_dataloaders"``). Determines the output
            subdirectory and is written into ``config.json`` so ``sig_mmd.py``
            can resolve the paired seed-A oracle config.
        batch_size: Conditions per saved batch file. ``None`` writes a single
            batch containing all conditions.
    """
    reference_method_dir = Path(reference_method_dir)
    base_path = Path(os.environ["NPNF_RESULTS_DIR"]) / dataloader_name

    grid_points = load_grid_points(reference_method_dir)
    prediction_day_axis = grid_points["X"].squeeze(-1).float()
    num_grid_points = prediction_day_axis.numel()
    logger.info("Reference grid: {} points", num_grid_points)

    logger.info("Instantiating seed-B oracle dataset")
    dataset = instantiate(seed_b_config)
    metadata = dataset.get_condition_metadata()
    genotype_ids = metadata["genotype_id"]
    yearsite_uids = metadata["yearsite_uid"]
    shared_oracle_day_axis = dataset.get_height_days_all().float()

    num_conditions = len(genotype_ids)
    num_samples_per_cond = dataset.n_lodging_draws
    logger.info(
        "Oracle samples: {} conditions x {} draws", num_conditions, num_samples_per_cond
    )

    effective_batch_size = num_conditions if batch_size is None else batch_size
    num_batches = (num_conditions + effective_batch_size - 1) // effective_batch_size
    logger.info("Writing {} batches under {}", num_batches, base_path / METHOD_NAME)

    for batch_index in range(num_batches):
        start = batch_index * effective_batch_size
        end = min(start + effective_batch_size, num_conditions)
        condition_indices = list(range(start, end))
        oracle_chunk = dataset.get_lodged_chunk(
            condition_indices, range(dataset.n_lodging_draws)
        )
        aligned = align_trajectories_to_day_axis(
            oracle_chunk["height_values_all_nonoise"].float(),
            shared_oracle_day_axis,
            prediction_day_axis,
        )
        B, draw_count, _ = aligned.shape

        grid_tensor = aligned.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, G, 1)
        predictions = tensordict.TensorDict({"grid": grid_tensor}, batch_size=(B, 1))

        batch_data = tensordict.TensorDict(
            {"has_lodged": oracle_chunk["has_lodged"]}, batch_size=(B,)
        )
        batch_data["genotype_id"] = genotype_ids[start:end]
        batch_data["yearsite_uid"] = yearsite_uids[start:end]

        lodging_seed = getattr(dataset, "lodging_seed", -1)
        config_payload = {
            "method": METHOD_NAME,
            "dataloader_name": dataloader_name,
            "num_samples": draw_count,
            "num_grid_points": num_grid_points,
            "batch_size": batch_size,
            "reference_method_dir": str(reference_method_dir.resolve()),
            "seed_b_lodging_seed": -1 if lodging_seed is None else int(lodging_seed),
        }

        save_batch_results(
            batch_index=batch_index,
            method_name=METHOD_NAME,
            base_path=base_path,
            predictions=predictions,
            batch_data=batch_data,
            grid_points=grid_points,
            config=config_payload if batch_index == 0 else None,
        )

    logger.info("Done. Output: {}", base_path / METHOD_NAME)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an oracle-self fake-predictions directory that sig_mmd.py "
            "can score like any model run."
        )
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help=(
            "Test split alias within --test-set (shifted: seen, geno, env, unseen; "
            "legacy: plot, genotype, site, year, environment, unseen)."
        ),
    )
    SyntheticTestSet.add_argument(parser)
    parser.add_argument(
        "--reference-method-dir",
        type=str,
        required=True,
        help=(
            "Path to a real method_dir (with predictions/). Its batch 0 "
            "grid_points defines the grid used for alignment."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Conditions per saved batch file. If omitted, writes a single "
            "batch containing all conditions."
        ),
    )
    args = parser.parse_args()

    split = split_named(args.split, args.test_set)
    if split.seed_b is None:
        parser.error(f"{args.test_set} split {args.split!r} has no seed-B config")
    seed_b_config = split.seed_b
    dataloader_name = split.dataloader_name

    reference_method_dir = Path(args.reference_method_dir)
    config_path = reference_method_dir / "config.json"
    if config_path.exists():
        try:
            ref_dataloader_name = json.loads(config_path.read_text()).get(
                "dataloader_name"
            )
        except json.JSONDecodeError:
            ref_dataloader_name = None
        if ref_dataloader_name and ref_dataloader_name != dataloader_name:
            logger.warning(
                "Reference method_dir dataloader_name ({}) differs from --split "
                "({}). The oracle-self grid will be taken from the reference, but "
                "the scored oracle will be the split's seed-A dataset.",
                ref_dataloader_name,
                dataloader_name,
            )

    generate_oracle_self_predictions(
        seed_b_config=seed_b_config,
        reference_method_dir=reference_method_dir,
        dataloader_name=dataloader_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    _main()
