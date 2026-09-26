"""Plot signature-Mahalanobis separation of lodged oracle trajectories."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from npnf.metrics.csig_mmd import CensoringParameters
from npnf.metrics.mahalanobis_artifact import (
    SignatureMahalanobisArtifact,
    load_signature_mahalanobis_artifact,
)
from npnf.scripts.paper.style import COLORS, setup_style

DEFAULT_OUTPUT_DIR = Path("paper/signature_mahalanobis_separation")
DEFAULT_MAX_POINTS_PER_CLASS = 60_000


@dataclass(frozen=True)
class Config:
    artifact: Path
    output_dir: Path
    alpha: float | None
    max_points_per_class: int
    seed: int
    title: str


def default_artifact_path() -> Path:
    results_base = Path(os.environ["NPNF_RESULTS_DIR"])
    return results_base / "sig_mahalanobis" / "all" / "oracle_signature_mahalanobis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot lodged/non-lodged separation by signature-Mahalanobis distance."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=default_artifact_path(),
        help=(
            "Path to oracle_signature_mahalanobis artifact. Defaults to "
            "$NPNF_RESULTS_DIR/sig_mahalanobis/all/oracle_signature_mahalanobis."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Quantile of the censoring reference distances used as the tail "
            "threshold (default: the CSig-MMD rule, which keeps 97.5 %% of the "
            "lodged censoring reference draws in the tail)."
        ),
    )
    parser.add_argument(
        "--max-points-per-class",
        type=int,
        default=DEFAULT_MAX_POINTS_PER_CLASS,
        help="Maximum lodged and non-lodged scatter points to draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic scatter subsampling.",
    )
    parser.add_argument("--title", default="", help="Optional figure title.")
    return parser


def parse_args() -> Config:
    args = build_parser().parse_args()
    return Config(
        artifact=args.artifact,
        output_dir=args.output_dir,
        alpha=args.alpha,
        max_points_per_class=args.max_points_per_class,
        seed=args.seed,
        title=args.title,
    )


def _sample_indices(
    indices: np.ndarray, max_count: int, rng: np.random.Generator
) -> np.ndarray:
    if len(indices) <= max_count:
        return indices
    return np.sort(rng.choice(indices, size=max_count, replace=False))


def _load_scatter_data(
    artifact_path: Path, max_points_per_class: int, seed: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray], SignatureMahalanobisArtifact]:
    artifact = load_signature_mahalanobis_artifact(artifact_path, mmap_mode="r")
    distances = np.asarray(artifact.mahalanobis_distances, dtype=np.float64)
    has_lodged = np.asarray(artifact.has_lodged, dtype=np.bool_)

    rng = np.random.default_rng(seed)
    lodged_indices = np.flatnonzero(has_lodged)
    non_lodged_indices = np.flatnonzero(~has_lodged)
    sampled_rows = np.sort(
        np.concatenate(
            [
                _sample_indices(non_lodged_indices, max_points_per_class, rng),
                _sample_indices(lodged_indices, max_points_per_class, rng),
            ]
        )
    )

    draw_count = int(artifact.metadata["draw_count"])
    condition_indices = sampled_rows // draw_count
    draw_indices = sampled_rows % draw_count
    trajectories = artifact.oracle_trajectories_on_grid[condition_indices, draw_indices]
    max_heights = np.asarray(trajectories.max(axis=1), dtype=np.float32)

    scatter_df = pd.DataFrame(
        {
            "row_index": sampled_rows,
            "condition_index": condition_indices,
            "draw_index": draw_indices,
            "max_height": max_heights,
            "mahalanobis_distance": distances[sampled_rows],
            "has_lodged": has_lodged[sampled_rows],
        }
    )
    return scatter_df, {"distances": distances, "has_lodged": has_lodged}, artifact


def _summary(
    distances: np.ndarray, has_lodged: np.ndarray, alpha: float, threshold: float
) -> dict[str, float]:
    tail = distances >= threshold
    n_total = len(distances)
    n_lodged = int(has_lodged.sum())
    n_non_lodged = n_total - n_lodged
    n_tail = int(tail.sum())
    lodged_in_tail = int(has_lodged[tail].sum())
    non_lodged_in_tail = n_tail - lodged_in_tail
    return {
        "alpha": alpha,
        "threshold": threshold,
        "n_total": n_total,
        "n_lodged": n_lodged,
        "n_non_lodged": n_non_lodged,
        "lodged_rate": n_lodged / n_total,
        "n_tail": n_tail,
        "tail_fraction": n_tail / n_total,
        "lodged_in_tail": lodged_in_tail,
        "non_lodged_in_tail": non_lodged_in_tail,
        "tail_lodged_rate": lodged_in_tail / n_tail,
        "lodged_recall": lodged_in_tail / n_lodged,
        "non_lodged_false_positive_rate": non_lodged_in_tail / n_non_lodged,
        "roc_auc": float(roc_auc_score(has_lodged, distances)),
    }


def _plot_scatter(
    scatter_df: pd.DataFrame, summary: dict[str, float], cfg: Config
) -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    non_lodged = scatter_df[~scatter_df["has_lodged"]]
    lodged = scatter_df[scatter_df["has_lodged"]]
    ax.scatter(
        non_lodged["max_height"],
        non_lodged["mahalanobis_distance"],
        s=5,
        color=COLORS["gray"],
        alpha=0.18,
        linewidths=0,
        rasterized=True,
        label="Non-lodged",
    )
    ax.scatter(
        lodged["max_height"],
        lodged["mahalanobis_distance"],
        s=6,
        color=COLORS["orange"],
        alpha=0.45,
        linewidths=0,
        rasterized=True,
        label="Lodged",
    )
    ax.axhline(summary["threshold"], color="#222222", linestyle="--", linewidth=1.1)

    ax.set_yscale("log")
    ax.set_xlabel("Maximum height")
    ax.set_ylabel("Mahalanobis distance")
    if cfg.title:
        ax.set_title(cfg.title)
    ax.text(
        0.98,
        summary["threshold"],
        rf"$\alpha={summary['alpha']:.3f}$",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="right",
        fontsize=9,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["gray"],
            markeredgecolor="none",
            markersize=6,
            label="Non-lodged",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["orange"],
            markeredgecolor="none",
            markersize=6,
            label="Lodged",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    return fig


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading {}", cfg.artifact)
    scatter_df, full_data, artifact = _load_scatter_data(
        cfg.artifact, cfg.max_points_per_class, cfg.seed
    )
    censoring = CensoringParameters(alpha=cfg.alpha).resolve(artifact)
    summary = _summary(
        full_data["distances"],
        full_data["has_lodged"],
        censoring.alpha,
        censoring.c_squared,
    )

    scatter_df.to_csv(
        cfg.output_dir / "signature_mahalanobis_separation_points.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(
        cfg.output_dir / "signature_mahalanobis_separation_summary.csv", index=False
    )

    fig = _plot_scatter(scatter_df, summary, cfg)
    output_path = cfg.output_dir / "signature_mahalanobis_separation"
    for fmt in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{fmt}"))
    plt.close(fig)
    logger.info("Wrote outputs to {}", cfg.output_dir)
    logger.info(
        "Artifact {}: n_conditions={}, draw_count={}, height_scale={}",
        artifact.path,
        artifact.metadata.get("n_conditions"),
        artifact.metadata.get("draw_count"),
        artifact.metadata.get("height_scale"),
    )


if __name__ == "__main__":
    main()
