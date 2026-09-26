"""Observed FIP1 plots as the reference side of the Sig-MMD protocol.

The synthetic scorer asks the simulator for a reference trajectory; FIP1 has no
simulator, so the reference is the measured plots themselves. This module loads
them, rebuilds the context each prediction was conditioned on, weights the
candidate plots by that context, and pairs the saved predictions with the plot
they belong to.
"""

from __future__ import annotations

import functools

import numpy as np
from tqdm import tqdm

from npnf.data.batch_loader import BatchLoader
from npnf.data.datasets.fip1 import get_heights_dataset
from npnf.lodging import robust_max_height
from npnf.metrics.blocked_scoring import extract_grid

SPLIT_DATALOADERS = {
    "test_plot": "fip1_test_plot_dataloaders",
    "test_genotype": "fip1_test_genotype_dataloaders",
    "test_environment": "fip1_test_environment_dataloaders",
    "test_genotype_environment": "fip1_test_genotype_environment_dataloaders",
}


def split_from_dataloader_name(dataloader_name: str) -> str:
    return {name: split for split, name in SPLIT_DATALOADERS.items()}[dataloader_name]


@functools.lru_cache(maxsize=4)
def load_observed_plots(split: str, datasets_offline_path: str) -> dict[str, dict]:
    """Observed days, heights and harvest year of every plot of ``split``, keyed by
    ``plot_uid``.

    Cached because loading the dataset costs about 40 s while scoring one method
    directory costs a few seconds, and a scoring job covers many directories of
    the same split.
    """
    dataset = get_heights_dataset(
        split=split, datasets_offline_path=datasets_offline_path
    )
    observed = {}
    for sample in dataset:
        days = sample["height_days"].numpy().astype(int)
        heights = sample["height_values"].numpy().astype(np.float64)
        observed[str(sample["plot_uid"])] = {
            "days": days,
            "heights": heights,
            "harvest_year": int(sample["harvest_year"]),
        }
    return observed


def reconstruct_context(
    mode: str, batch_dict: dict, row_index: int, plot: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Context days and values the model was given for one plot.

    Mirrors ``blocked_scoring.reconstruct_context`` but reads the observed FIP1
    trajectory instead of reconstructing ``clean + noise``.
    """
    if mode == "no_context":
        return np.array([], dtype=int), np.array([], dtype=float)
    if mode == "random_context":
        positions = np.asarray(
            batch_dict["context_indices"][row_index], dtype=int
        ).reshape(-1)
    else:  # max_height: every observation up to the height peak
        _, peak_index = robust_max_height(plot["heights"])
        positions = np.arange(int(peak_index) + 1, dtype=int)
    return plot["days"][positions], plot["heights"][positions]


def context_weights(
    pool_plots: list[dict],
    context_days: np.ndarray,
    context_values: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Posterior weights over ``pool_plots`` given one target's context.

    Gaussian likelihood of the context values under each candidate's observed
    trajectory, evaluated at the context days by linear interpolation of that
    candidate's own curve. Empty contexts fall back to uniform weight - the
    ``no_context`` degenerate case of the synthetic scorer.

    Interpolation is what makes a pooled multi-year candidate pool work, and it
    mirrors ``synthetic_metrics_utils.context_weights``, where a candidate's clean
    curve is known on the whole grid. Exact-day matching cannot: a candidate from
    another harvest year shares no measurement day with the context, and treating
    that as zero log-likelihood would give it the *highest* weight. Interpolation
    enters the weight only; scored paths stay exact observations at observed days.
    """
    if context_days.size == 0:
        return np.full(len(pool_plots), 1.0 / len(pool_plots))

    log_weights = np.empty(len(pool_plots), dtype=np.float64)
    for index, candidate in enumerate(pool_plots):
        heights = np.interp(context_days, candidate["days"], candidate["heights"])
        residual = heights - context_values
        log_weights[index] = -0.5 * float((residual**2).sum()) / (sigma**2)

    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum()


def load_records(
    loader: BatchLoader, mode: str, observed: dict[str, dict]
) -> tuple[list[dict], int]:
    """One record per predicted plot, in lexicographic ``plot_uid`` order."""
    records: list[dict] = []
    draw_counts: list[int] = []
    for predictions, batch_dict in tqdm(
        loader, total=len(loader), unit="batch", desc="fip1-blocked: load"
    ):
        grid = extract_grid(predictions).detach().cpu()
        metadata = batch_dict["data"].flatten()
        plot_uids = [str(uid) for uid in metadata["plot_uid"]]
        gids = [str(gid) for gid in metadata["genotype_id"]]
        yss = [str(ys) for ys in metadata["yearsite_uid"]]
        for row_index, (plot_uid, gid, ys) in enumerate(
            zip(plot_uids, gids, yss, strict=True)
        ):
            plot = observed[plot_uid]
            context_days, context_values = reconstruct_context(
                mode, batch_dict, row_index, plot
            )
            draw_counts.append(int(grid.shape[1]))
            records.append(
                {
                    "plot_uid": plot_uid,
                    "gid": gid,
                    "ys": ys,
                    # 2019 has two trials with one temperature record: the
                    # environment a model sees is the harvest year, not the trial.
                    "environment": str(plot["harvest_year"]),
                    "grid": grid[row_index],
                    "days": plot["days"],
                    "heights": plot["heights"],
                    "context_days": context_days,
                    "context_values": context_values,
                }
            )

    records.sort(key=lambda record: record["plot_uid"])
    return records, min(draw_counts)
