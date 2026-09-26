"""Sig-MMD and CSig-MMD for synthetic condition-view evaluation.

Scores each exact ``(genotype_id, yearsite_uid)`` condition of a method directory
against its grid-aligned oracle draws. With ``--mahalanobis-artifact`` the scorer
also writes the Censored Signature MMD (CSig-MMD, Redhead et al.,
https://arxiv.org/abs/2602.10182) of each condition from the same Gram
matrices; the precomputed ``oracle_signature_mahalanobis/`` artifact supplies the
MRCD censoring rule.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

import polars as pl
import torch
from hydra.utils import instantiate
from loguru import logger
from tqdm import tqdm

from npnf.data.batch_loader import BatchLoader
from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.data.fip1_day_grid import align_trajectories_to_day_axis
from npnf.metrics.blocked_scoring import build_id_to_index, group_units
from npnf.metrics.blocks import Selection, UnitMember, UnitScope, build_unit_blocks
from npnf.metrics.csig_mmd import (
    CensoringParameters,
    CensoringReference,
    load_censoring_reference,
)
from npnf.metrics.oracle_config import (
    get_oracle_config_for_method_dir,
    get_seed_b_config_for_dataloader_name,
)
from npnf.metrics.sig_mmd import normalized_sig_mmd, unnormalized_sig_mmd
from npnf.metrics.signature import SYNTHETIC_SIGNATURE_HEIGHT_SCALE, to_metric_paths


def _extract_batch_metadata(batch_dict: dict) -> tuple[list[str], list[str]]:
    metadata = batch_dict["data"].flatten()
    gids = [str(g) for g in metadata["genotype_id"]]
    yss = [str(y) for y in metadata["yearsite_uid"]]
    return gids, yss


def _iter_condition_batches(
    loader: BatchLoader,
) -> Iterator[tuple[list[str], list[str], torch.Tensor]]:
    """Yield condition metadata and model grid rows per saved batch."""
    for batch_dict in loader:
        if not isinstance(batch_dict, tuple):
            msg = "sig_mmd requires both saved predictions and batch metadata"
            raise TypeError(msg)

        predictions, batch_dict = batch_dict
        grid = predictions.select("grid").flatten()["grid"].squeeze(-1)
        gids, yss = _extract_batch_metadata(batch_dict)

        yield gids, yss, grid


def _iter_condition_metadata_batches(
    loader: BatchLoader,
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield condition metadata for saved batches without loading predictions."""
    for batch_dict in loader:
        if isinstance(batch_dict, tuple):
            _, batch_dict = batch_dict
        yield _extract_batch_metadata(batch_dict)


def _oracle_chunk_on_grid(
    dataset: SyntheticDataset,
    indices: list[int],
    draw_count: int,
    oracle_day_axis: torch.Tensor,
    prediction_day_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = dataset.get_lodged_chunk(indices, range(draw_count))
    oracle_trajectories = align_trajectories_to_day_axis(
        chunk["height_values_all_nonoise"].float(), oracle_day_axis, prediction_day_axis
    )
    return oracle_trajectories, chunk["has_lodged"]


def _score_conditions(
    loader: BatchLoader,
    dataset: SyntheticDataset,
    id_to_index: dict[tuple[str, str], int],
    draw_count: int,
    oracle_day_axis: torch.Tensor,
    prediction_day_axis: torch.Tensor,
    height_scale: float,
    metric_day_axis_norm: torch.Tensor,
    *,
    lead_lag: bool = True,
    normalize_sig_kernel: bool = True,
    reference: CensoringReference | None = None,
    self_kernel_batch_size: int = 512,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score all conditions in a loader against grid-aligned oracle draws.

    Returns the Sig-MMD rows and, with a censoring ``reference``, the CSig-MMD rows.
    """
    score_fn = normalized_sig_mmd if normalize_sig_kernel else unnormalized_sig_mmd
    rows: list[dict[str, object]] = []
    csig_rows: list[dict[str, object]] = []
    n_collected = 0

    for gids, yss, model_grid in tqdm(
        _iter_condition_batches(loader), total=len(loader), unit="batch"
    ):
        indices = [id_to_index[(gid, ys)] for gid, ys in zip(gids, yss, strict=True)]
        oracle_grid, has_lodged = _oracle_chunk_on_grid(
            dataset, indices, draw_count, oracle_day_axis, prediction_day_axis
        )

        for i, (gid, ys) in enumerate(zip(gids, yss, strict=True)):
            oracle_a = oracle_grid[i, :draw_count]
            lodged = has_lodged[i, :draw_count]
            paths_x = to_metric_paths(
                model_grid[i, :draw_count],
                height_scale,
                metric_day_axis_norm,
                lead_lag=lead_lag,
            )
            paths_y = to_metric_paths(
                oracle_a, height_scale, metric_day_axis_norm, lead_lag=lead_lag
            )

            if reference is None:
                score = score_fn(paths_x, paths_y)
            else:
                artifact_row = reference.artifact_row(gid, ys)
                score_fields = reference.score(
                    paths_y,
                    paths_x,
                    reference.distances[artifact_row, :draw_count],
                    height_scale,
                    metric_day_axis_norm,
                    self_kernel_batch_size,
                    label=f"genotype_id={gid!r}, yearsite_uid={ys!r}",
                )[0]
                score = score_fields["sig_score"]
                csig_rows.append(
                    {
                        "source": "model",
                        "genotype_id": gid,
                        "yearsite_uid": ys,
                        "oracle_lodging_rate": float(
                            reference.artifact.condition_lodging_rates[artifact_row]
                        ),
                        "mean_w_oracle": score_fields["mean_w_oracle"],
                        "mean_w_model": score_fields["mean_w_model"],
                        "score": score_fields["score"],
                    }
                )
            rows.append(
                {
                    "genotype_id": gid,
                    "yearsite_uid": ys,
                    "oracle_lodging_rate": float(lodged.float().mean()),
                    "score": score,
                }
            )
            n_collected += 1

    logger.info("Model: {} conditions scored", n_collected)
    return rows, csig_rows


def _selected_oracle_grid_on_prediction_axis(
    dataset: SyntheticDataset,
    selections: Sequence[Selection],
    oracle_day_axis: torch.Tensor,
    prediction_day_axis: torch.Tensor,
) -> torch.Tensor:
    """Load selected pairwise oracle trajectories and align them to the grid."""
    bank = torch.empty(
        (len(selections), prediction_day_axis.numel()), dtype=torch.float32
    )
    selections_by_draw: dict[int, list[tuple[int, int]]] = defaultdict(list)
    selections_by_condition: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for output_index, selection in enumerate(selections):
        selections_by_draw[selection.draw_index].append(
            (output_index, selection.condition_index)
        )
        selections_by_condition[selection.condition_index].append(
            (output_index, selection.draw_index)
        )

    if len(selections_by_draw) <= len(selections_by_condition):
        for draw_index in sorted(selections_by_draw):
            output_positions = [pair[0] for pair in selections_by_draw[draw_index]]
            condition_indices = [pair[1] for pair in selections_by_draw[draw_index]]
            chunk = dataset.get_lodged_chunk(condition_indices, [draw_index])
            aligned = align_trajectories_to_day_axis(
                chunk["height_values_all_nonoise"].float(),
                oracle_day_axis,
                prediction_day_axis,
            )
            bank[torch.tensor(output_positions, dtype=torch.long)] = aligned[:, 0].cpu()
    else:
        for condition_index in sorted(selections_by_condition):
            output_positions = [
                pair[0] for pair in selections_by_condition[condition_index]
            ]
            draw_indices = [
                pair[1] for pair in selections_by_condition[condition_index]
            ]
            chunk = dataset.get_lodged_chunk([condition_index], draw_indices)
            aligned = align_trajectories_to_day_axis(
                chunk["height_values_all_nonoise"].float(),
                oracle_day_axis,
                prediction_day_axis,
            )
            bank[torch.tensor(output_positions, dtype=torch.long)] = aligned[
                0, torch.arange(len(draw_indices))
            ].cpu()

    return bank


def _score_oracle_self_conditions(
    loader: BatchLoader,
    seed_a_dataset: SyntheticDataset,
    seed_a_id_to_index: dict[tuple[str, str], int],
    seed_b_dataset: SyntheticDataset,
    seed_b_grouped_members: dict[str, list[UnitMember]],
    unit_scope: UnitScope,
    draw_count: int,
    seed_a_day_axis: torch.Tensor,
    seed_b_day_axis: torch.Tensor,
    prediction_day_axis: torch.Tensor,
    height_scale: float,
    metric_day_axis_norm: torch.Tensor,
    *,
    oracle_self_seed: int,
    lead_lag: bool = True,
    normalize_sig_kernel: bool = True,
) -> list[dict[str, object]]:
    """Score exact SeedA targets against covariate-pooled SeedB draws."""
    score_fn = normalized_sig_mmd if normalize_sig_kernel else unnormalized_sig_mmd
    rows: list[dict[str, object]] = []
    n_collected = 0
    seed_b_unit_cache: dict[str, torch.Tensor] = {}

    for gids, yss in tqdm(
        _iter_condition_metadata_batches(loader),
        total=len(loader),
        unit="batch",
        desc="oracle self",
    ):
        try:
            seed_a_indices = [
                seed_a_id_to_index[(gid, ys)] for gid, ys in zip(gids, yss, strict=True)
            ]
        except KeyError as error:
            gid, ys = error.args[0]
            msg = (
                "SeedA oracle dataset does not contain prediction condition "
                f"(genotype_id={gid!r}, yearsite_uid={ys!r})"
            )
            raise ValueError(msg) from error

        seed_a_grid, has_lodged = _oracle_chunk_on_grid(
            seed_a_dataset,
            seed_a_indices,
            draw_count,
            seed_a_day_axis,
            prediction_day_axis,
        )

        for i, (gid, ys) in enumerate(zip(gids, yss, strict=True)):
            unit_id = unit_scope.unit_id(gid, ys)
            seed_b_grid = seed_b_unit_cache.get(unit_id)
            if seed_b_grid is None:
                blocks = build_unit_blocks(
                    seed_b_grouped_members[unit_id],
                    draws_per_member=draw_count,
                    unit_scope=unit_scope,
                    unit_id=unit_id,
                    n_blocks=1,
                    block_size=draw_count,
                    seed=oracle_self_seed,
                )
                seed_b_grid = _selected_oracle_grid_on_prediction_axis(
                    seed_b_dataset, blocks[0], seed_b_day_axis, prediction_day_axis
                )
                if unit_scope != UnitScope.CONDITION:
                    seed_b_unit_cache[unit_id] = seed_b_grid

            oracle_a = seed_a_grid[i, :draw_count]
            lodged = has_lodged[i, :draw_count]
            paths_x = to_metric_paths(
                seed_b_grid, height_scale, metric_day_axis_norm, lead_lag=lead_lag
            )
            paths_y = to_metric_paths(
                oracle_a, height_scale, metric_day_axis_norm, lead_lag=lead_lag
            )

            rows.append(
                {
                    "genotype_id": gid,
                    "yearsite_uid": ys,
                    "oracle_lodging_rate": float(lodged.float().mean()),
                    "score": score_fn(paths_x, paths_y),
                }
            )
            n_collected += 1

    logger.info("Oracle self: {} conditions scored", n_collected)
    return rows


def _default_output_name(*, oracle_self: bool, normalize_sig_kernel: bool) -> str:
    if oracle_self:
        return (
            "sig_mmd_oracle_self"
            if normalize_sig_kernel
            else "sig_mmd_oracle_self_unnormalized"
        )
    return "sig_mmd" if normalize_sig_kernel else "sig_mmd_unnormalized"


def _condition_summary(per_x_df: pl.DataFrame) -> pl.DataFrame:
    n_units = per_x_df.height
    return pl.DataFrame(
        {
            "view": ["condition"],
            "n_units": [n_units],
            "n_conditions": [n_units],
            "mean": [per_x_df["score"].mean()],
            "median": [per_x_df["score"].median()],
            "std": [per_x_df["score"].std()],
            "p95": [per_x_df["score"].quantile(0.95)],
        }
    )


def _csig_condition_summary(
    per_x_df: pl.DataFrame, reference: CensoringReference
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source": ["model"],
            "n_conditions": [per_x_df.height],
            "alpha": [reference.main.alpha],
            "beta": [reference.main.beta],
            "c_squared": [reference.main.c_squared],
            "mean": [per_x_df["score"].mean()],
            "median": [per_x_df["score"].median()],
            "std": [per_x_df["score"].std()],
            "p95": [per_x_df["score"].quantile(0.95)],
        }
    )


def sig_mmd_analysis(  # noqa: PLR0912
    method_dir: str,
    n_samples: int,
    output_folder: str | None = None,
    *,
    lead_lag: bool = True,
    normalize_sig_kernel: bool = True,
    oracle_self: bool = False,
    oracle_self_seed: int = 0,
    mahalanobis_artifact: str | None = None,
    csig_output_folder: str | None = None,
    censoring: CensoringParameters | None = None,
    self_kernel_batch_size: int = 512,
) -> None:
    if mahalanobis_artifact is not None and (
        oracle_self or not lead_lag or not normalize_sig_kernel
    ):
        msg = (
            "CSig-MMD (--mahalanobis-artifact) requires normalized lead-lag model "
            "scoring; it cannot be combined with --oracle-self, --no-lead-lag or "
            "--unnormalized"
        )
        raise ValueError(msg)
    if censoring is None:
        censoring = CensoringParameters()
    method_path = Path(method_dir)
    if output_folder is None:
        output_folder = str(
            method_path
            / _default_output_name(
                oracle_self=oracle_self, normalize_sig_kernel=normalize_sig_kernel
            )
        )
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    loader = (
        BatchLoader(method_path, load_predictions=False)
        if oracle_self
        else BatchLoader(method_path)
    )
    first_batch = loader.load_batch(0)
    if isinstance(first_batch, tuple):
        _, first_batch = first_batch
    prediction_day_axis = first_batch["grid_points"]["X"].squeeze(-1).float()
    logger.info("Grid: {} points", prediction_day_axis.shape[0])

    unit_scope: UnitScope | None = None
    if oracle_self:
        if method_path.name != "no_context":
            msg = (
                "oracle-self Sig-MMD is only supported for no_context method "
                "directories; context-aware oracle baselines for methods with "
                f"height context require a separate P(H | A, C) estimator. Got "
                f"{method_path.name!r}."
            )
            raise ValueError(msg)
        unit_scope = UnitScope.from_config(loader.config, method_path)

    oracle_cfg, dataloader_name = get_oracle_config_for_method_dir(method_path)
    logger.info(
        "Oracle config: {} (dataloader={})", oracle_cfg.__name__, dataloader_name
    )
    logger.info("Loading oracle dataset")
    dataset: SyntheticDataset = instantiate(oracle_cfg)
    id_to_index = build_id_to_index(dataset)
    shared_oracle_day_axis = dataset.get_height_days_all().float()
    draw_count = min(n_samples, dataset.n_lodging_draws)

    seed_b_dataset: SyntheticDataset | None = None
    seed_b_grouped_members = None
    seed_b_day_axis: torch.Tensor | None = None
    if oracle_self:
        seed_b_cfg = get_seed_b_config_for_dataloader_name(dataloader_name)
        logger.info("Loading SeedB oracle dataset: {}", seed_b_cfg.__name__)
        seed_b_dataset = instantiate(seed_b_cfg)
        draw_count = min(draw_count, seed_b_dataset.n_lodging_draws)
        seed_b_id_to_index = build_id_to_index(seed_b_dataset)
        assert unit_scope is not None
        # The keys are in dataset index order, so a member's position is its index.
        seed_b_grouped_members = group_units(unit_scope, list(seed_b_id_to_index))
        seed_b_day_axis = seed_b_dataset.get_height_days_all().float()
        missing_in_seed_b = set(id_to_index) - set(seed_b_id_to_index)
        if missing_in_seed_b:
            gid, ys = sorted(missing_in_seed_b)[0]
            msg = (
                "SeedB oracle dataset does not contain SeedA condition "
                f"(genotype_id={gid!r}, yearsite_uid={ys!r})"
            )
            raise ValueError(msg)

    if draw_count <= 0:
        msg = f"No oracle draws available for n_samples={n_samples}"
        raise ValueError(msg)

    oracle_day_min = shared_oracle_day_axis.min()
    oracle_day_max = shared_oracle_day_axis.max()

    height_scale = SYNTHETIC_SIGNATURE_HEIGHT_SCALE
    logger.info("Height scale: {:.1f}", height_scale)
    logger.info(
        "Params: n_samples={}, oracle_draws={}, lead_lag={}, "
        "normalize_sig_kernel={}, oracle_self={}, oracle_self_seed={}, "
        "unit_scope={}",
        n_samples,
        draw_count,
        lead_lag,
        normalize_sig_kernel,
        oracle_self,
        oracle_self_seed,
        unit_scope,
    )

    metric_day_axis_norm = (prediction_day_axis - oracle_day_min) / (
        oracle_day_max - oracle_day_min
    )

    reference: CensoringReference | None = None
    if mahalanobis_artifact is not None:
        reference = load_censoring_reference(
            mahalanobis_artifact,
            censoring,
            prediction_day_axis=prediction_day_axis,
            metric_day_axis_norm=metric_day_axis_norm,
            height_scale=height_scale,
        )
        draw_count = min(draw_count, reference.draw_count)
        logger.info(
            "CSig-MMD: alpha={:.4f}, beta={:.4f}, c²={:.4f}, draws={}, artifact={}",
            reference.main.alpha,
            reference.main.beta,
            reference.main.c_squared,
            draw_count,
            mahalanobis_artifact,
        )

    if oracle_self:
        assert seed_b_dataset is not None
        assert seed_b_grouped_members is not None
        assert seed_b_day_axis is not None
        assert unit_scope is not None
        csig_rows: list[dict[str, object]] = []
        rows = _score_oracle_self_conditions(
            loader,
            dataset,
            id_to_index,
            seed_b_dataset,
            seed_b_grouped_members,
            unit_scope,
            draw_count,
            shared_oracle_day_axis,
            seed_b_day_axis,
            prediction_day_axis,
            height_scale,
            metric_day_axis_norm,
            oracle_self_seed=oracle_self_seed,
            lead_lag=lead_lag,
            normalize_sig_kernel=normalize_sig_kernel,
        )
    else:
        rows, csig_rows = _score_conditions(
            loader,
            dataset,
            id_to_index,
            draw_count,
            shared_oracle_day_axis,
            prediction_day_axis,
            height_scale,
            metric_day_axis_norm,
            lead_lag=lead_lag,
            normalize_sig_kernel=normalize_sig_kernel,
            reference=reference,
            self_kernel_batch_size=self_kernel_batch_size,
        )

    per_x_df = pl.DataFrame(rows).sort(["genotype_id", "yearsite_uid"])
    per_x_df.write_csv(output_path / "sig_mmd_per_x.csv")
    logger.info("Wrote {}", output_path / "sig_mmd_per_x.csv")
    summary = _condition_summary(per_x_df)
    summary.write_csv(output_path / "sig_mmd_summary.csv")
    logger.info("Wrote {}", output_path / "sig_mmd_summary.csv")

    if reference is not None:
        csig_output_path = (
            Path(csig_output_folder)
            if csig_output_folder is not None
            else method_path / "csig_mmd"
        )
        csig_output_path.mkdir(parents=True, exist_ok=True)
        csig_per_x_df = pl.DataFrame(csig_rows).sort(["genotype_id", "yearsite_uid"])
        csig_per_x_df.write_csv(csig_output_path / "csig_mmd_per_x.csv")
        logger.info("Wrote {}", csig_output_path / "csig_mmd_per_x.csv")
        _csig_condition_summary(csig_per_x_df, reference).write_csv(
            csig_output_path / "csig_mmd_summary.csv"
        )
        logger.info("Wrote {}", csig_output_path / "csig_mmd_summary.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sig-MMD for synthetic evaluation")
    parser.add_argument(
        "--method-dir",
        type=str,
        required=True,
        help="Path to method directory with predictions/",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Number of trajectory samples per condition",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help=(
            "Output directory for CSV files (default: <method-dir>/sig_mmd, "
            "<method-dir>/sig_mmd_unnormalized with --unnormalized, "
            "<method-dir>/sig_mmd_oracle_self with --oracle-self, or "
            "<method-dir>/sig_mmd_oracle_self_unnormalized with both flags)"
        ),
    )
    parser.add_argument(
        "--no-lead-lag",
        action="store_true",
        help="Disable lead-lag augmentation and use raw height-plus-time paths",
    )
    parser.add_argument(
        "--unnormalized",
        action="store_true",
        help="Disable Sig-MMD self-kernel normalization",
    )
    parser.add_argument(
        "--oracle-self",
        action="store_true",
        help=(
            "Score a covariate-aware SeedB oracle baseline instead of saved "
            "model predictions. Currently supported for no_context method dirs."
        ),
    )
    parser.add_argument(
        "--oracle-self-seed",
        type=int,
        default=0,
        help="Base seed for deterministic oracle-self pool sampling (default: 0).",
    )
    parser.add_argument(
        "--mahalanobis-artifact",
        type=str,
        default=None,
        help=(
            "oracle_signature_mahalanobis artifact directory. When given, the "
            "scorer also writes CSig-MMD results from the same Gram matrices."
        ),
    )
    parser.add_argument(
        "--csig-output-folder",
        type=str,
        default=None,
        help="CSig-MMD output directory (default: <method-dir>/csig_mmd).",
    )
    parser.add_argument(
        "--self-kernel-batch-size",
        type=int,
        default=512,
        help="Batch size for model signature self-kernel norms of CSig-MMD",
    )
    CensoringParameters.add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sig_mmd_analysis(
        method_dir=args.method_dir,
        n_samples=args.n_samples,
        output_folder=args.output_folder,
        lead_lag=not args.no_lead_lag,
        normalize_sig_kernel=not args.unnormalized,
        oracle_self=args.oracle_self,
        oracle_self_seed=args.oracle_self_seed,
        mahalanobis_artifact=args.mahalanobis_artifact,
        csig_output_folder=args.csig_output_folder,
        censoring=CensoringParameters.from_args(args),
        self_kernel_batch_size=args.self_kernel_batch_size,
    )


if __name__ == "__main__":
    main()
