"""Score oracle draws using precomputed signature Mahalanobis distances.

Reads the ``oracle_signature_mahalanobis/`` artifact directory produced by
``precompute_sig_mahalanobis.py`` and writes discrimination metrics
(ROC-AUC, Precision@5%) to CSV.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import matplotlib as mpl
import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import roc_auc_score

from npnf.metrics.mahalanobis_artifact import (
    SignatureMahalanobisArtifact,
    load_signature_mahalanobis_artifact,
)
from npnf.metrics.utils import mean_or_none
from npnf.scripts.utils.outputs import write_csv

mpl.use("Agg")
import matplotlib.pyplot as plt


def _support_lookup(
    artifact: SignatureMahalanobisArtifact,
) -> dict[tuple[int, int], bool] | None:
    if (
        artifact.mrcd_fit_condition_indices is None
        or artifact.mrcd_fit_draw_indices is None
        or artifact.mrcd_fit_support is None
    ):
        return None
    return {
        (int(condition_index), int(draw_index)): bool(in_support)
        for condition_index, draw_index, in_support in zip(
            artifact.mrcd_fit_condition_indices,
            artifact.mrcd_fit_draw_indices,
            artifact.mrcd_fit_support,
            strict=True,
        )
    }


def _fit_row_indices(
    artifact: SignatureMahalanobisArtifact,
) -> tuple[np.ndarray, np.ndarray] | None:
    if (
        artifact.mrcd_fit_condition_indices is None
        or artifact.mrcd_fit_draw_indices is None
        or artifact.mrcd_fit_support is None
    ):
        return None

    row_by_key = {
        (int(condition_index), int(draw_index)): row_index
        for row_index, (condition_index, draw_index) in enumerate(
            zip(artifact.condition_indices, artifact.draw_indices, strict=True)
        )
    }
    rows = np.asarray(
        [
            row_by_key[(int(condition_index), int(draw_index))]
            for condition_index, draw_index in zip(
                artifact.mrcd_fit_condition_indices,
                artifact.mrcd_fit_draw_indices,
                strict=True,
            )
        ],
        dtype=np.int64,
    )
    support = np.asarray(artifact.mrcd_fit_support, dtype=np.bool_)
    return rows, support


def _write_support_summary_and_plots(
    artifact: SignatureMahalanobisArtifact, output_path: Path
) -> None:
    fit_rows_and_support = _fit_row_indices(artifact)
    if fit_rows_and_support is None:
        return

    fit_rows, support = fit_rows_and_support
    fit_lodged = np.asarray(artifact.has_lodged[fit_rows], dtype=np.bool_)
    fit_distances = np.asarray(artifact.mahalanobis_distances[fit_rows], dtype=float)
    non_support = ~support

    summary = {
        "n_fit_samples": len(fit_rows),
        "n_support": int(support.sum()),
        "n_non_support": int(non_support.sum()),
        "lodged_rate_fit": mean_or_none(fit_lodged.astype(float)),
        "lodged_rate_support": mean_or_none(fit_lodged[support].astype(float)),
        "lodged_rate_non_support": mean_or_none(fit_lodged[non_support].astype(float)),
        "mean_distance_support": mean_or_none(fit_distances[support]),
        "mean_distance_non_support": mean_or_none(fit_distances[non_support]),
    }
    summary_path = output_path / "mrcd_support_summary.csv"
    pl.DataFrame([summary]).write_csv(summary_path)
    logger.info("Wrote {}", summary_path)

    _plot_support_lodging_counts(fit_lodged, support, output_path)
    _plot_support_distance_hist(fit_distances, support, output_path)


def _plot_support_lodging_counts(
    fit_lodged: np.ndarray, support: np.ndarray, output_path: Path
) -> None:
    non_support = ~support
    counts = np.asarray(
        [
            [int((support & ~fit_lodged).sum()), int((support & fit_lodged).sum())],
            [
                int((non_support & ~fit_lodged).sum()),
                int((non_support & fit_lodged).sum()),
            ],
        ]
    )

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    x = np.arange(2)
    ax.bar(x, counts[:, 0], label="not lodged")
    ax.bar(x, counts[:, 1], bottom=counts[:, 0], label="lodged")
    ax.set_xticks(x, ["support", "non-support"])
    ax.set_ylabel("fit samples")
    ax.set_title("MRCD fit support by lodging")
    ax.legend()
    fig.tight_layout()
    plot_path = output_path / "mrcd_support_lodging_counts.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote {}", plot_path)


def _plot_support_distance_hist(
    fit_distances: np.ndarray, support: np.ndarray, output_path: Path
) -> None:
    non_support = ~support
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    if support.any():
        ax.hist(fit_distances[support], bins=30, alpha=0.65, label="support")
    if non_support.any():
        ax.hist(fit_distances[non_support], bins=30, alpha=0.65, label="non-support")
    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("fit samples")
    ax.set_title("MRCD support distance distribution")
    ax.legend()
    fig.tight_layout()
    plot_path = output_path / "mrcd_support_distance_hist.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote {}", plot_path)


def score_sig_mahalanobis(precomputed: str, output_folder: str) -> None:
    artifact = load_signature_mahalanobis_artifact(precomputed)
    distances = artifact.mahalanobis_distances
    has_lodged = artifact.has_lodged
    condition_indices = artifact.condition_indices
    draw_indices = artifact.draw_indices
    condition_genotype_ids = artifact.condition_genotype_ids
    condition_yearsite_uids = artifact.condition_yearsite_uids
    condition_lodging_rates = artifact.condition_lodging_rates
    depth = int(artifact.metadata["depth"])
    mrcd_support_by_key = _support_lookup(artifact)

    logger.info(
        "Loaded {} draws (depth={}) from {}", len(distances), depth, precomputed
    )

    auc = roc_auc_score(has_lodged, distances)

    n_top = max(1, int(0.05 * len(distances)))
    top_indices = np.argsort(distances)[-n_top:]
    precision_at_5pct = has_lodged[top_indices].mean()

    logger.info("ROC-AUC: {:.4f}", auc)
    logger.info(
        "Precision@5%: {:.4f} ({}/{} lodged in top {})",
        precision_at_5pct,
        has_lodged[top_indices].sum(),
        n_top,
        n_top,
    )

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    detail_path = output_path / "mahalanobis_vs_lodging.csv"
    fieldnames = [
        "genotype_id",
        "yearsite_uid",
        "draw_idx",
        "has_lodged",
        "mahalanobis_distance",
        "depth",
        "oracle_lodging_rate",
    ]
    if mrcd_support_by_key is not None:
        fieldnames.extend(["in_mrcd_fit", "in_mrcd_support"])

    def detail_rows() -> Iterator[dict[str, object]]:
        for row_index in range(len(distances)):
            condition_index = int(condition_indices[row_index])
            draw_index = int(draw_indices[row_index])
            row: dict[str, object] = {
                "genotype_id": str(condition_genotype_ids[condition_index]),
                "yearsite_uid": str(condition_yearsite_uids[condition_index]),
                "draw_idx": draw_index,
                "has_lodged": bool(has_lodged[row_index]),
                "mahalanobis_distance": round(float(distances[row_index]), 6),
                "depth": depth,
                "oracle_lodging_rate": round(
                    float(condition_lodging_rates[condition_index]), 4
                ),
            }
            if mrcd_support_by_key is not None:
                support_value = mrcd_support_by_key.get((condition_index, draw_index))
                row["in_mrcd_fit"] = support_value is not None
                row["in_mrcd_support"] = "" if support_value is None else support_value
            yield row

    write_csv(detail_path, detail_rows(), fieldnames)
    logger.info("Wrote {}", detail_path)

    summary_row = {
        "depth": depth,
        "roc_auc": round(float(auc), 4),
        "precision_at_5pct": round(float(precision_at_5pct), 4),
        "n_total": len(distances),
        "n_lodged": int(has_lodged.sum()),
    }
    pl.DataFrame([summary_row]).write_csv(output_path / "discrimination_summary.csv")
    logger.info("Wrote {}", output_path / "discrimination_summary.csv")
    _write_support_summary_and_plots(artifact, output_path)

    logger.info(
        "Depth {}: ROC-AUC={}, Precision@5%={}",
        depth,
        summary_row["roc_auc"],
        summary_row["precision_at_5pct"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score oracle draws using precomputed signature Mahalanobis data"
    )
    parser.add_argument(
        "--precomputed",
        type=str,
        required=True,
        help="Path to oracle_signature_mahalanobis artifact directory",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output directory for CSV files",
    )
    args = parser.parse_args()
    score_sig_mahalanobis(
        precomputed=args.precomputed, output_folder=args.output_folder
    )
