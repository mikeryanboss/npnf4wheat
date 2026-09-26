"""Sig-MMD for varying-context synthetic predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from hydra.utils import instantiate
from loguru import logger
from tqdm import tqdm

from npnf.data.batch_loader import BatchLoader
from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.data.datasets.utils import verify_dataset_cache_key
from npnf.data.fip1_day_grid import align_trajectories_to_day_axis
from npnf.metrics.oracle_config import get_oracle_config_for_method_dir
from npnf.metrics.sig_mmd import normalized_sig_mmd
from npnf.metrics.signature import SYNTHETIC_SIGNATURE_HEIGHT_SCALE, to_metric_paths
from npnf.metrics.utils import id_to_str

SAMPLING_METHODS = ["random", "uncertainty", "sequential"]


def _score_method_dir(
    method_dir: Path, n_samples: int, *, lead_lag: bool
) -> list[dict[str, object]]:
    loader = BatchLoader(method_dir)
    config = loader.config
    if config is None:
        msg = f"Missing config.json in {method_dir}"
        raise FileNotFoundError(msg)

    _, first_batch_data = loader.load_batch(0)
    prediction_day_axis = first_batch_data["grid_points"]["X"].squeeze(-1).float()

    oracle_cfg, dataloader_name = get_oracle_config_for_method_dir(method_dir)
    logger.info(
        "{}: oracle config {} (dataloader={})",
        method_dir.name,
        oracle_cfg.__name__,
        dataloader_name,
    )

    dataset: SyntheticDataset = instantiate(oracle_cfg)
    verify_dataset_cache_key(
        method_dir,
        config["dataset_identity"]["cache_key"],
        dataset.get_dataset_identity()["cache_key"],
    )

    metadata = dataset.get_condition_metadata()
    id_to_index = {
        (id_to_str(gid), id_to_str(ys)): idx
        for idx, (gid, ys) in enumerate(
            zip(metadata["genotype_id"], metadata["yearsite_uid"], strict=True)
        )
    }

    oracle_day_axis = dataset.get_height_days_all().float()
    if n_samples <= 0:
        msg = f"No model samples available for n_samples={n_samples}"
        raise ValueError(msg)
    if dataset.n_lodging_draws <= 0:
        msg = "No oracle draw 0 available"
        raise ValueError(msg)

    height_scale = SYNTHETIC_SIGNATURE_HEIGHT_SCALE
    metric_day_axis_norm = (prediction_day_axis - oracle_day_axis.min()) / (
        oracle_day_axis.max() - oracle_day_axis.min()
    )
    trials_per_sample = config["trials_per_sample"]

    rows: list[dict[str, object]] = []
    prediction_row_offset = 0
    for batch_index, (predictions, batch_dict) in enumerate(
        tqdm(loader, total=len(loader), unit="batch")
    ):
        grid = predictions["grid"].squeeze(-1).float()
        num_rows, num_context_steps, num_model_samples, _ = grid.shape
        condition_count = num_rows // trials_per_sample
        model_sample_count = min(n_samples, num_model_samples)

        batch_metadata = batch_dict["data"].flatten()
        gids = [id_to_str(gid) for gid in batch_metadata["genotype_id"]]
        yss = [id_to_str(ys) for ys in batch_metadata["yearsite_uid"]]
        dataset_indices = [
            id_to_index[(gid, ys)] for gid, ys in zip(gids, yss, strict=True)
        ]

        oracle_chunk = dataset.get_lodged_chunk(dataset_indices, [0])
        oracle_grid = align_trajectories_to_day_axis(
            oracle_chunk["height_values_all_nonoise"].float(),
            oracle_day_axis,
            prediction_day_axis,
        )
        has_lodged = oracle_chunk["has_lodged"]

        batch_rows_start = len(rows)
        for row_index, (gid, ys) in enumerate(zip(gids, yss, strict=True)):
            oracle_heights = oracle_grid[row_index, :1]
            oracle_lodging_rate = float(has_lodged[row_index, :1].float().mean())
            oracle_paths = to_metric_paths(
                oracle_heights, height_scale, metric_day_axis_norm, lead_lag=lead_lag
            )
            max_context = min(
                int(batch_dict["max_valid_context"][row_index].item()) + 1,
                num_context_steps,
            )

            for num_context in range(max_context):
                model_paths = to_metric_paths(
                    grid[row_index, num_context, :model_sample_count],
                    height_scale,
                    metric_day_axis_norm,
                    lead_lag=lead_lag,
                )
                rows.append(
                    {
                        "num_context": num_context,
                        "prediction_row": prediction_row_offset + row_index,
                        "condition_row": row_index % condition_count,
                        "trial_index": row_index // condition_count,
                        "genotype_id": gid,
                        "yearsite_uid": ys,
                        "oracle_lodging_rate": oracle_lodging_rate,
                        "score": normalized_sig_mmd(model_paths, oracle_paths),
                    }
                )

        logger.info(
            "Batch {}: scored {} prediction rows and {} context rows",
            batch_index,
            num_rows,
            len(rows) - batch_rows_start,
        )
        prediction_row_offset += num_rows

    logger.info("{}: scored {} context-condition rows", method_dir.name, len(rows))
    return rows


def _write_method_outputs(
    method_dir: Path, output_dir: Path, n_samples: int, *, lead_lag: bool
) -> pl.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_context_df = pl.DataFrame(
        _score_method_dir(method_dir, n_samples, lead_lag=lead_lag)
    ).sort(["num_context", "prediction_row", "genotype_id", "yearsite_uid"])
    summary_df = (
        per_context_df.group_by("num_context")
        .agg(
            pl.len().alias("n"),
            pl.col("score").mean().alias("mean"),
            pl.col("score").median().alias("median"),
            pl.col("score").std().alias("std"),
            pl.col("score").quantile(0.95).alias("p95"),
        )
        .sort("num_context")
    )

    per_context_path = output_dir / "sig_mmd_per_context_condition.csv"
    summary_path = output_dir / "sig_mmd_by_context.csv"
    per_context_df.write_csv(per_context_path)
    summary_df.write_csv(summary_path)
    logger.info("Wrote {}", per_context_path)
    logger.info("Wrote {}", summary_path)
    return summary_df


def sig_mmd_varying_analysis(
    results_folder: str,
    n_samples: int = 64,
    output_folder: str | None = None,
    *,
    lead_lag: bool = True,
) -> None:
    results_path = Path(results_folder)

    if (results_path / "predictions").is_dir():
        output_path = results_path if output_folder is None else Path(output_folder)
        _write_method_outputs(results_path, output_path, n_samples, lead_lag=lead_lag)
        return

    method_dirs = [
        (method, results_path / method)
        for method in SAMPLING_METHODS
        if (results_path / method / "predictions").is_dir()
    ]
    if not method_dirs:
        msg = f"No predictions/ directory found in {results_path}"
        raise FileNotFoundError(msg)

    output_root = results_path if output_folder is None else Path(output_folder)
    summary_frames = []
    for method_name, method_dir in method_dirs:
        method_output = (
            method_dir if output_folder is None else output_root / method_name
        )
        method_summary = _write_method_outputs(
            method_dir, method_output, n_samples, lead_lag=lead_lag
        )
        summary_frames.append(
            method_summary.with_columns(pl.lit(method_name).alias("method")).select(
                ["method", "num_context", "n", "mean", "median", "std", "p95"]
            )
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "sig_mmd_varying_summary.csv"
    pl.concat(summary_frames).sort(["method", "num_context"]).write_csv(summary_path)
    logger.info("Wrote {}", summary_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sig-MMD across varying context prediction steps"
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        required=True,
        help="Method dir or parent dir containing varying prediction methods",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Number of trajectory samples per condition",
    )
    parser.add_argument(
        "--output-folder", type=str, default=None, help="Output directory for CSV files"
    )
    parser.add_argument(
        "--no-lead-lag",
        action="store_true",
        help="Disable lead-lag augmentation and use raw height-plus-time paths",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    sig_mmd_varying_analysis(
        results_folder=args.results_folder,
        n_samples=args.n_samples,
        output_folder=args.output_folder,
        lead_lag=not args.no_lead_lag,
    )


if __name__ == "__main__":
    main()
