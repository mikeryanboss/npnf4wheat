# ruff: noqa: T201, TRY003, EM101, SLF001
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensordict
import torch
from safetensors.torch import load_file

from npnf.lodging import detect_lodging_grid
from npnf.models.neural_process.models.flows import transform_sample
from npnf.scripts.paper.style import setup_style
from npnf.scripts.utils.prediction import (
    get_empty_observations,
    resolve_model_from_checkpoint,
)
from npnf.scripts.utils.utils import create_grid_points
from npnf.utils import add_panel_labels

OUTPUT_DIR = Path("paper/prior_latent_pca_sweeps")
CHECKPOINT_NAME = "checkpoint-3000000"
DEFAULT_MODEL_RUNS = {
    "LNP": "LNP-512k-training3m_set_mode_nested_noprior",
    "LNP-NF-Prior-Posterior": (
        "LNP-NF-Prior-Posterior-512k-training3m_set_mode_nested_noprior"
    ),
    "ANP": "ANP-512k-training3m_set_mode_nested_noprior",
    "ANP-NF-Prior-Posterior": (
        "ANP-NF-Prior-Posterior-512k-training3m_set_mode_nested_noprior"
    ),
}
DEFAULT_MODELS = ["LNP", "LNP-NF-Prior-Posterior", "ANP", "ANP-NF-Prior-Posterior"]
METRICS = [
    "max_height",
    "final_height",
    "drop_abs",
    "drop_rel",
    "generated_lodged",
    "peak_grid_index",
]
ORIENTATION_METRICS = ["drop_abs", "max_height", "generated_lodged"]


@dataclass(frozen=True)
class PCAResult:
    mean: np.ndarray
    components: np.ndarray
    scores: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    summary: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create decoded prior-latent PCA sweep paper figures."
    )
    parser.add_argument(
        "--project-dir", type=Path, default=Path(os.environ["NPNF_PROJECT_DIR"])
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-name", default=CHECKPOINT_NAME)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=None,
        help="Optional run names matching --models. Defaults to known paper runs.",
    )
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-pcs", type=int, default=3)
    parser.add_argument("--max-alpha-sd", type=float, default=3.0)
    parser.add_argument("--num-alpha", type=int, default=31)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--write-model-rows-variant",
        action="store_true",
        help="Also write a model-rows / PC-columns comparison figure.",
    )
    parser.add_argument(
        "--write-flow-before-after-variant",
        action="store_true",
        help="Also write a pre-flow z0 vs post-flow z PCA comparison for flow models.",
    )
    return parser.parse_args()


def resolve_run_names(models: list[str], run_names: list[str] | None) -> dict[str, str]:
    if run_names is not None:
        if len(run_names) != len(models):
            message = "--run-names must have the same length as --models"
            raise ValueError(message)
        return dict(zip(models, run_names, strict=True))

    missing = [model for model in models if model not in DEFAULT_MODEL_RUNS]
    if missing:
        message = (
            "No default run names for models: "
            + ", ".join(missing)
            + ". Provide --run-names."
        )
        raise ValueError(message)
    return {model: DEFAULT_MODEL_RUNS[model] for model in models}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores = scores[order]

    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end

    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def make_targets(device: torch.device) -> tensordict.TensorDict:
    grid = create_grid_points(device)
    return tensordict.lazy_stack([grid.clone()]).densify(layout=torch.jagged)


def dense_prediction_tensor(predictions: torch.Tensor) -> torch.Tensor:
    if predictions.is_nested:
        predictions = torch.stack(list(predictions))
    if predictions.ndim == 5 and predictions.shape[2] == 1:
        predictions = predictions[:, :, 0]
    if predictions.ndim == 4:
        predictions = predictions[..., 0]
    return predictions


def squeeze_latent(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 4 and z.shape[2] == 1:
        return z[:, :, 0]
    if z.ndim == 3:
        return z
    message = f"Unexpected latent shape {tuple(z.shape)}"
    raise ValueError(message)


def extract_distribution_stats(
    prior_distribution: torch.distributions.Independent,
) -> tuple[np.ndarray, np.ndarray]:
    base = prior_distribution.base_dist
    return (
        base.loc.detach().cpu().numpy().reshape(-1),
        base.scale.detach().cpu().numpy().reshape(-1),
    )


def load_model(
    project_dir: Path, run_name: str, checkpoint_name: str, device: torch.device
):
    checkpoint_dir = project_dir / run_name / "checkpoints" / checkpoint_name
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)

    model = resolve_model_from_checkpoint(str(checkpoint_dir))
    model.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")))
    model.to(device)
    model.eval()
    return model


def uses_prior_flow(model) -> bool:
    return hasattr(model, "prior_flow")


def trajectory_metrics(predictions: np.ndarray) -> pd.DataFrame:
    tensor = torch.as_tensor(predictions, dtype=torch.float32)
    lodging = detect_lodging_grid(tensor)
    return pd.DataFrame(
        {
            "generated_lodged": lodging.is_lodged.cpu().numpy().astype(bool),
            "max_height": lodging.max_height.cpu().numpy(),
            "final_height": lodging.final_height.cpu().numpy(),
            "drop_abs": lodging.drop_abs.cpu().numpy(),
            "drop_rel": lodging.drop_rel.cpu().numpy(),
            "peak_grid_index": predictions.argmax(axis=1),
        }
    )


def sample_prior_outputs(
    model,
    *,
    num_samples: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
    is_flow: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype = torch.float32
    context = get_empty_observations(1, device=device, dtype=dtype)
    targets = make_targets(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    z_chunks = []
    prediction_chunks = []
    prior_mu = None
    prior_sigma = None

    for start in range(0, num_samples, chunk_size):
        current = min(chunk_size, num_samples - start)
        noise = torch.randn(
            (1, current, model.latent_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        kwargs = {"noise_global": noise} if is_flow else {"noise_z": noise}
        with torch.no_grad():
            output = model(
                context_priors=context,
                context_posteriors=None,
                targets=targets,
                temperatures=None,
                markers=None,
                num_samples=current,
                get_samples=True,
                get_loss=False,
                sample_from_posterior=False,
                **kwargs,
            )

        if prior_mu is None:
            prior_mu, prior_sigma = extract_distribution_stats(
                output["prior"]["distribution"]
            )

        z = squeeze_latent(output["prior"]["z"].detach()).cpu().numpy()[0]
        predictions = (
            dense_prediction_tensor(output["predictions"]["target"].detach())
            .cpu()
            .numpy()[0]
        )
        z_chunks.append(z)
        prediction_chunks.append(predictions)

    if prior_mu is None or prior_sigma is None:
        raise RuntimeError("No prior distribution statistics were captured")

    return (
        np.concatenate(z_chunks, axis=0),
        np.concatenate(prediction_chunks, axis=0),
        prior_mu,
        prior_sigma,
    )


def sample_prior_flow_outputs(
    model, *, num_samples: int, chunk_size: int, seed: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype = torch.float32
    context = get_empty_observations(1, device=device, dtype=dtype)
    targets = make_targets(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    flow_z_chunks = []
    z0_chunks = []
    prediction_chunks = []
    prior_mu = None
    prior_sigma = None

    for start in range(0, num_samples, chunk_size):
        current = min(chunk_size, num_samples - start)
        noise = torch.randn(
            (1, current, model.latent_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        with torch.no_grad():
            output = model(
                context_priors=context,
                context_posteriors=None,
                targets=targets,
                temperatures=None,
                markers=None,
                num_samples=current,
                get_samples=True,
                get_loss=False,
                sample_from_posterior=False,
                noise_global=noise,
            )

        if prior_mu is None or prior_sigma is None:
            prior_mu, prior_sigma = extract_distribution_stats(
                output["prior"]["distribution"]
            )

        flow_z = squeeze_latent(output["prior"]["z"].detach()).cpu().numpy()[0]
        noise_values = noise.detach().cpu().numpy()[0]
        z0 = prior_mu[None, :] + noise_values * prior_sigma[None, :]
        predictions = (
            dense_prediction_tensor(output["predictions"]["target"].detach())
            .cpu()
            .numpy()[0]
        )
        flow_z_chunks.append(flow_z)
        z0_chunks.append(z0)
        prediction_chunks.append(predictions)

    if prior_mu is None or prior_sigma is None:
        raise RuntimeError("No prior distribution statistics were captured")

    return (
        np.concatenate(flow_z_chunks, axis=0),
        np.concatenate(z0_chunks, axis=0),
        np.concatenate(prediction_chunks, axis=0),
        prior_mu,
        prior_sigma,
    )


def fit_pca(values: np.ndarray, metrics: pd.DataFrame, num_pcs: int) -> PCAResult:
    mean = values.mean(axis=0)
    centered = values - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    num_pcs = min(num_pcs, vt.shape[0])

    components = vt[:num_pcs].copy()
    scores = centered @ components.T
    explained_variance_all = singular_values**2 / (len(values) - 1)
    explained_variance = explained_variance_all[:num_pcs]
    explained_variance_ratio = explained_variance / float(explained_variance_all.sum())

    for pc_index in range(num_pcs):
        pc_scores = scores[:, pc_index]
        metric_corrs = {
            metric: pearson(pc_scores, metrics[metric].to_numpy(dtype=float))
            for metric in ORIENTATION_METRICS
        }
        orientation_metric = max(
            metric_corrs, key=lambda metric: abs(metric_corrs[metric])
        )
        if metric_corrs[orientation_metric] < 0:
            components[pc_index] *= -1.0
            scores[:, pc_index] *= -1.0

    rows = []
    labels = metrics["generated_lodged"].to_numpy(dtype=bool)
    for pc_index in range(num_pcs):
        pc_scores = scores[:, pc_index]
        metric_corrs = {
            metric: pearson(pc_scores, metrics[metric].to_numpy(dtype=float))
            for metric in ORIENTATION_METRICS
        }
        orientation_metric = max(
            metric_corrs, key=lambda metric: abs(metric_corrs[metric])
        )
        auc = auc_score(pc_scores, labels)
        row = {
            "pc": pc_index + 1,
            "explained_variance": float(explained_variance[pc_index]),
            "explained_variance_ratio": float(explained_variance_ratio[pc_index]),
            "pc_score_std": float(pc_scores.std(ddof=1)),
            "orientation_metric": orientation_metric,
            "orientation_abs_corr": abs(metric_corrs[orientation_metric]),
            "auc_for_lodged": float(auc),
            "auc_abs_for_lodged": float(max(auc, 1.0 - auc)),
        }
        for metric in METRICS:
            row[f"corr_{metric}"] = pearson(
                pc_scores, metrics[metric].to_numpy(dtype=float)
            )
        rows.append(row)

    return PCAResult(
        mean=mean,
        components=components,
        scores=scores,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        summary=pd.DataFrame(rows),
    )


def prepare_flow_decode(model, device: torch.device):
    dtype = torch.float32
    context = get_empty_observations(1, device=device, dtype=dtype)
    targets = make_targets(device)
    prepared = model._prepare_flow_inputs(
        context_priors=context,
        targets=targets,
        context_posteriors=None,
        temperatures=None,
        markers=None,
    )
    prior_latents = model._encode_prior_latents(
        contexts=prepared["context_priors"],
        targets=prepared["targets"],
        temperatures=prepared["temperatures"],
        markers=prepared["markers"],
        prefix_token_keys=["global", "flow_condition"],
    )
    return prior_latents, prepared["targets"]


def prepare_prior_flow_decode(model, device: torch.device):
    prior_latents, targets = prepare_flow_decode(model, device)
    with torch.no_grad():
        prior_flow_condition = model.prior_context_projection(
            prior_latents["flow_condition"]
        )
    return prior_latents, targets, prior_flow_condition


def decode_flow_z_batch(model, z_grid: np.ndarray, flow_state, device: torch.device):
    prior_latents, targets = flow_state
    z_tensor = torch.as_tensor(z_grid, dtype=torch.float32, device=device)
    z_tensor = z_tensor.unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        mus, _ = model._decode_flow_predictions(
            z=z_tensor, prior_latents=prior_latents, targets=targets
        )
    return dense_prediction_tensor(mus.transpose(1, 2).detach()).cpu().numpy()[0]


def decode_flow_z0_batch(model, z0_grid: np.ndarray, flow_state, device: torch.device):
    prior_latents, targets, prior_flow_condition = flow_state
    z0_tensor = torch.as_tensor(z0_grid, dtype=torch.float32, device=device)
    z0_tensor = z0_tensor.unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        z_grid, _ = transform_sample(
            z0_tensor,
            flow=model.prior_flow,
            condition=prior_flow_condition,
            inverse=False,
            accumulate_logdet=False,
        )
        mus, _ = model._decode_flow_predictions(
            z=z_grid, prior_latents=prior_latents, targets=targets
        )
    return dense_prediction_tensor(mus.transpose(1, 2).detach()).cpu().numpy()[0]


def decode_plain_z_batch(
    model,
    z_grid: np.ndarray,
    prior_mu: np.ndarray,
    prior_sigma: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    eps_grid = (z_grid - prior_mu[None, :]) / prior_sigma[None, :]
    context = get_empty_observations(1, device=device, dtype=torch.float32)
    targets = make_targets(device)
    eps_tensor = torch.as_tensor(
        eps_grid, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        output = model(
            context_priors=context,
            context_posteriors=None,
            targets=targets,
            temperatures=None,
            markers=None,
            num_samples=len(z_grid),
            noise_z=eps_tensor,
            get_samples=True,
            get_loss=False,
            sample_from_posterior=False,
        )
    return (
        dense_prediction_tensor(output["predictions"]["target"].detach())
        .cpu()
        .numpy()[0]
    )


def decode_pc_sweeps(
    model,
    pca: PCAResult,
    *,
    is_flow: bool,
    prior_mu: np.ndarray,
    prior_sigma: np.ndarray,
    alphas: np.ndarray,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flow_state = prepare_flow_decode(model, device) if is_flow else None
    metric_rows = []
    trajectory_rows = []

    for pc_index, component in enumerate(pca.components, start=1):
        pc_sd = float(np.sqrt(pca.explained_variance[pc_index - 1]))
        z_grid = pca.mean[None, :] + alphas[:, None] * pc_sd * component[None, :]
        if is_flow:
            predictions = decode_flow_z_batch(model, z_grid, flow_state, device)
        else:
            predictions = decode_plain_z_batch(
                model, z_grid, prior_mu, prior_sigma, device
            )

        metrics = trajectory_metrics(predictions)
        metrics["pc"] = pc_index
        metrics["alpha_sd"] = alphas
        metrics["latent_pc_score"] = alphas * pc_sd
        metrics["pc_explained_variance_ratio"] = pca.explained_variance_ratio[
            pc_index - 1
        ]
        metrics["pc_score_std"] = pc_sd
        metric_rows.append(metrics)

        for local_index, alpha in enumerate(alphas):
            for grid_index, height in enumerate(predictions[local_index]):
                trajectory_rows.append(
                    {
                        "pc": pc_index,
                        "alpha_sd": float(alpha),
                        "latent_pc_score": float(alpha * pc_sd),
                        "grid_index": grid_index,
                        "height": float(height),
                        "generated_lodged": bool(
                            metrics.iloc[local_index]["generated_lodged"]
                        ),
                    }
                )

    return pd.concat(metric_rows, ignore_index=True), pd.DataFrame(trajectory_rows)


def decode_flow_coordinate_pc_sweeps(
    model,
    pca: PCAResult,
    *,
    coordinate: str,
    coordinate_label: str,
    alphas: np.ndarray,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if coordinate == "before_flow_z0":
        flow_state = prepare_prior_flow_decode(model, device)
    elif coordinate == "after_flow_z":
        flow_state = prepare_flow_decode(model, device)
    else:
        message = f"Unknown flow coordinate {coordinate}"
        raise ValueError(message)

    metric_rows = []
    trajectory_rows = []

    for pc_index, component in enumerate(pca.components, start=1):
        pc_sd = float(np.sqrt(pca.explained_variance[pc_index - 1]))
        z_grid = pca.mean[None, :] + alphas[:, None] * pc_sd * component[None, :]
        if coordinate == "before_flow_z0":
            predictions = decode_flow_z0_batch(model, z_grid, flow_state, device)
        else:
            predictions = decode_flow_z_batch(model, z_grid, flow_state, device)

        metrics = trajectory_metrics(predictions)
        metrics["coordinate"] = coordinate
        metrics["coordinate_label"] = coordinate_label
        metrics["pc"] = pc_index
        metrics["alpha_sd"] = alphas
        metrics["latent_pc_score"] = alphas * pc_sd
        metrics["pc_explained_variance_ratio"] = pca.explained_variance_ratio[
            pc_index - 1
        ]
        metrics["pc_score_std"] = pc_sd
        metric_rows.append(metrics)

        for local_index, alpha in enumerate(alphas):
            for grid_index, height in enumerate(predictions[local_index]):
                trajectory_rows.append(
                    {
                        "coordinate": coordinate,
                        "coordinate_label": coordinate_label,
                        "pc": pc_index,
                        "alpha_sd": float(alpha),
                        "latent_pc_score": float(alpha * pc_sd),
                        "grid_index": grid_index,
                        "height": float(height),
                        "generated_lodged": bool(
                            metrics.iloc[local_index]["generated_lodged"]
                        ),
                    }
                )

    return pd.concat(metric_rows, ignore_index=True), pd.DataFrame(trajectory_rows)


def selected_alphas(alphas: np.ndarray) -> np.ndarray:
    requested = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    available = np.array(sorted(np.unique(alphas)))
    return np.unique(
        np.array([available[np.argmin(np.abs(available - a))] for a in requested])
    )


def plot_sweeps(
    trajectories: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    model_order: list[str],
    num_pcs: int,
) -> None:
    setup_style()
    snaps = selected_alphas(metrics["alpha_sd"].to_numpy(dtype=float))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=float(snaps.min()), vmax=float(snaps.max()))

    fig, axes = plt.subplots(
        num_pcs, len(model_order), figsize=(11.2, 6.8), sharex=True, sharey=True
    )
    height_max = max(1.0, float(trajectories["height"].max()) * 1.05)
    grid_index_max = int(trajectories["grid_index"].max())

    for col_index, model_name in enumerate(model_order):
        model_traj = trajectories.loc[trajectories["model"] == model_name]
        for pc in range(1, num_pcs + 1):
            ax = axes[pc - 1, col_index]
            subset = model_traj.loc[
                (model_traj["pc"] == pc) & (model_traj["alpha_sd"].isin(snaps))
            ]
            for alpha in snaps:
                line = subset.loc[np.isclose(subset["alpha_sd"], alpha)].sort_values(
                    "grid_index"
                )
                if line.empty:
                    continue
                ax.plot(
                    line["grid_index"],
                    line["height"],
                    color=cmap(norm(float(alpha))),
                    linewidth=1.2,
                    alpha=0.78,
                )

            ax.set_xlim(0, grid_index_max)
            ax.set_ylim(0.0, height_max)
            ax.tick_params(labelsize=8)

    add_panel_labels(axes.flat)
    fig.tight_layout(rect=(0.14, 0.23, 0.995, 0.985))

    grid_left = axes[0, 0].get_position().x0
    grid_right = axes[0, -1].get_position().x1
    grid_bottom = axes[-1, 0].get_position().y0
    grid_top = axes[0, 0].get_position().y1
    grid_center_x = (grid_left + grid_right) / 2

    fig.text(
        grid_left - 0.105,
        (grid_bottom + grid_top) / 2,
        "Height",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10,
    )
    for row_index in range(num_pcs):
        box = axes[row_index, 0].get_position()
        fig.text(
            grid_left - 0.058,
            (box.y0 + box.y1) / 2,
            f"PC{row_index + 1}",
            ha="right",
            va="center",
            fontsize=10,
        )

    for col_index, model_name in enumerate(model_order):
        box = axes[-1, col_index].get_position()
        fig.text(
            (box.x0 + box.x1) / 2,
            grid_bottom - 0.060,
            model_name,
            ha="center",
            va="center",
            fontsize=8,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_width = min(0.36, (grid_right - grid_left) * 0.42)
    cbar_y = max(0.045, grid_bottom - 0.160)
    cbar_ax = fig.add_axes((grid_center_x - cbar_width / 2, cbar_y, cbar_width, 0.012))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([-3, 0, 3])
    cbar.set_label("latent PC offset (SD)", fontsize=8, labelpad=2)
    cbar.outline.set_visible(False)  # ty: ignore[call-non-callable]
    cbar.ax.tick_params(labelsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"prior_latent_pca_sweeps.{suffix}", dpi=300)
    plt.close(fig)


def plot_sweeps_model_rows(
    trajectories: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    model_order: list[str],
    num_pcs: int,
) -> None:
    setup_style()
    snaps = selected_alphas(metrics["alpha_sd"].to_numpy(dtype=float))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=float(snaps.min()), vmax=float(snaps.max()))

    fig, axes = plt.subplots(
        len(model_order), num_pcs, figsize=(8.8, 7.9), sharex=True, sharey=True
    )
    height_min = min(0.0, float(trajectories["height"].min()) * 1.05)
    height_max = max(1.0, float(trajectories["height"].max()) * 1.05)

    for row_index, model_name in enumerate(model_order):
        model_traj = trajectories.loc[trajectories["model"] == model_name]
        for pc in range(1, num_pcs + 1):
            ax = axes[row_index, pc - 1]
            subset = model_traj.loc[
                (model_traj["pc"] == pc) & (model_traj["alpha_sd"].isin(snaps))
            ]
            for alpha in snaps:
                line = subset.loc[np.isclose(subset["alpha_sd"], alpha)].sort_values(
                    "grid_index"
                )
                if line.empty:
                    continue
                ax.plot(
                    line["grid_index"],
                    line["height"],
                    color=cmap(norm(float(alpha))),
                    linewidth=1.2,
                    alpha=0.78,
                )
            ax.set_ylim(height_min, height_max)
            ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0.20, 0.20, 0.995, 0.985))

    grid_left = axes[0, 0].get_position().x0
    grid_right = axes[0, -1].get_position().x1
    grid_bottom = axes[-1, 0].get_position().y0
    grid_top = axes[0, 0].get_position().y1
    grid_center_x = (grid_left + grid_right) / 2

    fig.text(
        grid_left - 0.145,
        (grid_bottom + grid_top) / 2,
        "decoded height",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10,
    )
    for row_index, model_name in enumerate(model_order):
        box = axes[row_index, 0].get_position()
        fig.text(
            grid_left - 0.075,
            (box.y0 + box.y1) / 2,
            model_name,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    fig.text(
        grid_center_x,
        grid_bottom - 0.055,
        "prediction grid index",
        ha="center",
        va="center",
        fontsize=10,
    )
    for pc in range(1, num_pcs + 1):
        box = axes[-1, pc - 1].get_position()
        fig.text(
            (box.x0 + box.x1) / 2,
            grid_bottom - 0.107,
            f"PC{pc}",
            ha="center",
            va="center",
            fontsize=10,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_width = min(0.30, (grid_right - grid_left) * 0.42)
    cbar_ax = fig.add_axes((grid_center_x - cbar_width / 2, 0.042, cbar_width, 0.012))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([-3, 0, 3])
    cbar.set_label("latent PC offset (SD)", fontsize=8, labelpad=2)
    cbar.outline.set_visible(False)  # ty: ignore[call-non-callable]
    cbar.ax.tick_params(labelsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"prior_latent_pca_sweeps_model_rows.{suffix}", dpi=300
        )
    plt.close(fig)


def plot_flow_before_after_sweeps(
    trajectories: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    model_order: list[str],
    num_pcs: int,
) -> None:
    setup_style()
    snaps = selected_alphas(metrics["alpha_sd"].to_numpy(dtype=float))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=float(snaps.min()), vmax=float(snaps.max()))
    coordinate_order = [
        ("before_flow_z0", "before flow"),
        ("after_flow_z", "after flow"),
    ]
    num_rows = len(coordinate_order) * num_pcs

    fig, axes = plt.subplots(
        num_rows, len(model_order), figsize=(7.4, 10.0), sharex=True, sharey=True
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(num_rows, len(model_order))

    height_min = min(0.0, float(trajectories["height"].min()) * 1.05)
    height_max = max(1.0, float(trajectories["height"].max()) * 1.05)

    for col_index, model_name in enumerate(model_order):
        model_traj = trajectories.loc[trajectories["model"] == model_name]
        for coordinate_index, (coordinate, _) in enumerate(coordinate_order):
            coordinate_traj = model_traj.loc[model_traj["coordinate"] == coordinate]
            for pc in range(1, num_pcs + 1):
                row_index = coordinate_index * num_pcs + pc - 1
                ax = axes[row_index, col_index]
                subset = coordinate_traj.loc[
                    (coordinate_traj["pc"] == pc)
                    & (coordinate_traj["alpha_sd"].isin(snaps))
                ]
                for alpha in snaps:
                    line = subset.loc[
                        np.isclose(subset["alpha_sd"], alpha)
                    ].sort_values("grid_index")
                    if line.empty:
                        continue
                    ax.plot(
                        line["grid_index"],
                        line["height"],
                        color=cmap(norm(float(alpha))),
                        linewidth=1.2,
                        alpha=0.78,
                    )
                ax.set_ylim(height_min, height_max)
                ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0.24, 0.15, 0.995, 0.985))

    grid_left = axes[0, 0].get_position().x0
    grid_right = axes[0, -1].get_position().x1
    grid_bottom = axes[-1, 0].get_position().y0
    grid_top = axes[0, 0].get_position().y1
    grid_center_x = (grid_left + grid_right) / 2

    fig.text(
        grid_left - 0.165,
        (grid_bottom + grid_top) / 2,
        "Height",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10,
    )
    for coordinate_index, (_, coordinate_label) in enumerate(coordinate_order):
        for pc in range(1, num_pcs + 1):
            row_index = coordinate_index * num_pcs + pc - 1
            box = axes[row_index, 0].get_position()
            fig.text(
                grid_left - 0.072,
                (box.y0 + box.y1) / 2,
                f"{coordinate_label} PC{pc}",
                ha="right",
                va="center",
                fontsize=8,
            )

    for col_index, model_name in enumerate(model_order):
        box = axes[-1, col_index].get_position()
        fig.text(
            (box.x0 + box.x1) / 2,
            grid_bottom - 0.055,
            model_name,
            ha="center",
            va="center",
            fontsize=8,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_width = min(0.30, (grid_right - grid_left) * 0.42)
    cbar_y = max(0.040, grid_bottom - 0.125)
    cbar_ax = fig.add_axes((grid_center_x - cbar_width / 2, cbar_y, cbar_width, 0.012))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([-3, 0, 3])
    cbar.set_label("latent PC offset (SD)", fontsize=8, labelpad=2)
    cbar.outline.set_visible(False)  # ty: ignore[call-non-callable]
    cbar.ax.tick_params(labelsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"prior_latent_pca_flow_before_after.{suffix}", dpi=300
        )
    plt.close(fig)


def process_model(
    model_name: str, run_name: str, args: argparse.Namespace, device: torch.device
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = load_model(args.project_dir, run_name, args.checkpoint_name, device)
    is_flow = uses_prior_flow(model)
    z_samples, predictions, prior_mu, prior_sigma = sample_prior_outputs(
        model,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
        device=device,
        is_flow=is_flow,
    )
    sample_metrics = trajectory_metrics(predictions)
    pca = fit_pca(z_samples, sample_metrics, args.num_pcs)
    alphas = np.linspace(-args.max_alpha_sd, args.max_alpha_sd, args.num_alpha)
    sweep_metrics, sweep_trajectories = decode_pc_sweeps(
        model,
        pca,
        is_flow=is_flow,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        alphas=alphas,
        device=device,
    )

    summary = pca.summary.copy()
    summary["model"] = model_name
    summary["is_flow"] = is_flow
    sweep_metrics["model"] = model_name
    sweep_metrics["is_flow"] = is_flow
    sweep_trajectories["model"] = model_name
    sweep_trajectories["is_flow"] = is_flow
    return summary, sweep_metrics, sweep_trajectories


def process_flow_before_after_model(
    model_name: str, run_name: str, args: argparse.Namespace, device: torch.device
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    model = load_model(args.project_dir, run_name, args.checkpoint_name, device)
    if not uses_prior_flow(model):
        print(f"Skipping {model_name}: no prior flow")
        return None

    flow_z_samples, z0_samples, predictions, _, _ = sample_prior_flow_outputs(
        model,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
        device=device,
    )
    sample_metrics = trajectory_metrics(predictions)
    alphas = np.linspace(-args.max_alpha_sd, args.max_alpha_sd, args.num_alpha)
    coordinate_samples = [
        ("before_flow_z0", "before flow", z0_samples),
        ("after_flow_z", "after flow", flow_z_samples),
    ]

    summaries = []
    metric_frames = []
    trajectory_frames = []
    for coordinate, coordinate_label, samples in coordinate_samples:
        pca = fit_pca(samples, sample_metrics, args.num_pcs)
        sweep_metrics, sweep_trajectories = decode_flow_coordinate_pc_sweeps(
            model,
            pca,
            coordinate=coordinate,
            coordinate_label=coordinate_label,
            alphas=alphas,
            device=device,
        )

        summary = pca.summary.copy()
        summary["model"] = model_name
        summary["is_flow"] = True
        summary["coordinate"] = coordinate
        summary["coordinate_label"] = coordinate_label
        sweep_metrics["model"] = model_name
        sweep_metrics["is_flow"] = True
        sweep_trajectories["model"] = model_name
        sweep_trajectories["is_flow"] = True
        summaries.append(summary)
        metric_frames.append(sweep_metrics)
        trajectory_frames.append(sweep_trajectories)

    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(trajectory_frames, ignore_index=True),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    run_names = resolve_run_names(args.models, args.run_names)

    summaries = []
    metric_frames = []
    trajectory_frames = []
    for model_name in args.models:
        print(f"Processing {model_name}")
        summary, metrics, trajectories = process_model(
            model_name, run_names[model_name], args, device
        )
        summaries.append(summary)
        metric_frames.append(metrics)
        trajectory_frames.append(trajectories)

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_metrics = pd.concat(metric_frames, ignore_index=True)
    combined_trajectories = pd.concat(trajectory_frames, ignore_index=True)

    combined_summary.to_csv(
        args.output_dir / "prior_latent_pca_summary.csv", index=False
    )
    combined_metrics.to_csv(
        args.output_dir / "prior_latent_pca_sweep_metrics.csv", index=False
    )
    combined_trajectories.to_csv(
        args.output_dir / "prior_latent_pca_sweep_trajectories.csv", index=False
    )

    plot_sweeps(
        combined_trajectories,
        combined_metrics,
        args.output_dir,
        model_order=args.models,
        num_pcs=args.num_pcs,
    )
    if args.write_model_rows_variant:
        plot_sweeps_model_rows(
            combined_trajectories,
            combined_metrics,
            args.output_dir,
            model_order=args.models,
            num_pcs=args.num_pcs,
        )
    if args.write_flow_before_after_variant:
        flow_summaries = []
        flow_metric_frames = []
        flow_trajectory_frames = []
        flow_model_order = []
        for model_name in args.models:
            print(f"Processing {model_name} flow before/after")
            result = process_flow_before_after_model(
                model_name, run_names[model_name], args, device
            )
            if result is None:
                continue
            summary, metrics, trajectories = result
            flow_summaries.append(summary)
            flow_metric_frames.append(metrics)
            flow_trajectory_frames.append(trajectories)
            flow_model_order.append(model_name)

        if not flow_summaries:
            message = "--write-flow-before-after-variant requires a prior-flow model"
            raise ValueError(message)

        flow_summary = pd.concat(flow_summaries, ignore_index=True)
        flow_metrics = pd.concat(flow_metric_frames, ignore_index=True)
        flow_trajectories = pd.concat(flow_trajectory_frames, ignore_index=True)
        flow_summary.to_csv(
            args.output_dir / "prior_latent_pca_flow_before_after_summary.csv",
            index=False,
        )
        flow_metrics.to_csv(
            args.output_dir / "prior_latent_pca_flow_before_after_sweep_metrics.csv",
            index=False,
        )
        flow_trajectories.to_csv(
            args.output_dir
            / "prior_latent_pca_flow_before_after_sweep_trajectories.csv",
            index=False,
        )
        plot_flow_before_after_sweeps(
            flow_trajectories,
            flow_metrics,
            args.output_dir,
            model_order=flow_model_order,
            num_pcs=args.num_pcs,
        )
    print("Wrote outputs to", args.output_dir)
    print(
        combined_summary[
            ["model", "pc", "explained_variance_ratio", "orientation_metric"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
