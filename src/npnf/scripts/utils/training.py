import contextlib
from pathlib import Path
from typing import Any

import einops as EO
import matplotlib.pyplot as plt
import seaborn as sns
import tensordict
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import tqdm
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsoluteError
from tqdm import tqdm as ProgressBar

from npnf.models.neural_process.utils import SetMode, get_sets
from npnf.scripts.utils.prediction import consolidate_nested_tensors
from npnf.scripts.utils.utils import create_grid_points
from npnf.scripts.visualize.create_gifs import create_uncertainty_plot
from npnf.utils import fig2img

LATENT_SCATTER_MIN_SAMPLES = 512
PREDICTION_PLOT_GLOBAL_SAMPLES = 32
LATENT_SCATTER_MAX_POINTS = 4000
PREDICTION_IMAGE_MAX_SIZE = (900, 560)
LATENT_IMAGE_MAX_SIZE = (750, 460)
PREDICTION_FIGSIZE = (6.5, 4.0)
PREDICTION_DPI = 90
LATENT_FIGSIZE = (4.5, 4.5)
LATENT_COMPARISON_FIGSIZE = (8.5, 4.5)
LATENT_DPI = 110
LATENT_3D_ELEVATION = 28
LATENT_3D_AZIMUTH = -60
THUMBNAIL_RESAMPLE = Image.Resampling.LANCZOS


def handle_optimizer_mode(optimizer_dict: dict, mode: str) -> None:
    """Handle optimizer mode changes (train/eval) with error suppression."""
    with contextlib.suppress(AttributeError):
        if mode == "eval":
            optimizer_dict["optimizer"].eval()
        elif mode == "train":
            optimizer_dict["optimizer"].train()


def setup_checkpoint_and_validation_schedule(initial: int) -> tuple[int, int]:
    """Setup the next checkpoint and validation steps."""
    next_validation = initial
    next_checkpoint = initial
    return next_checkpoint, next_validation


def should_checkpoint(
    progress_bar: ProgressBar, next_checkpoint: int, checkpoint_frequency: int | None
) -> bool:
    """Check if it's time to save a checkpoint."""
    return checkpoint_frequency is not None and progress_bar.n >= next_checkpoint


def should_validate(
    progress_bar: ProgressBar, next_validation: int, validation_frequency: int | None
) -> bool:
    """Check if it's time to run validation."""
    return validation_frequency is not None and progress_bar.n >= next_validation


@torch.inference_mode()
def validate_epoch(
    model,
    validation_dataloader: DataLoader,
    accelerator: Accelerator,
    progress_bar: ProgressBar,
    num_samples: int = 16,
    use_temperature: bool = False,
    use_marker: bool = False,
    max_context_prior_length: int = 32,
    max_distinct: int | None = 128,
    min_target: int = 10,
    metric_suffix: str | None = None,
) -> dict[str, torch.Tensor]:
    """Run validation epoch and return metrics."""
    generator = torch.Generator()
    generator.manual_seed(42)

    mae_epoch = MeanAbsoluteError().to(accelerator.device)
    first_batch = True

    for batch in tqdm(
        iterable=validation_dataloader,
        desc="Validation loop...",
        position=1,
        leave=False,
    ):
        temperatures = batch.get("temperature") if use_temperature else None
        markers = batch.get("marker") if use_marker else None

        if first_batch and metric_suffix is None:
            visualize_validation_predictions(
                model=model,
                batch=batch.clone(),
                accelerator=accelerator,
                progress_bar=progress_bar,
                num_samples=num_samples,
                max_context_prior_length=max_context_prior_length,
                max_distinct=max_distinct,
                min_target=min_target,
                generator=generator,
                temperatures=temperatures,
                markers=markers,
            )
            first_batch = False

        output = get_model_predictions(
            model,
            batch,
            num_samples=num_samples,
            max_context_prior_length=max_context_prior_length,
            temperatures=temperatures,
            markers=markers,
            sample_from_posterior=False,
            generator=generator,
            kl_weight=1.0,
            set_mode="disjoint-holdout",
            max_distinct=max_distinct,
            min_target=min_target,
        )

        Y_pred_targets = output["predictions"]["target"]
        Y_targets = output["nested_batch"]["target"]["Y"]

        for Y_pred, Y_target in zip(Y_pred_targets, Y_targets, strict=True):
            Y_pred = Y_pred.contiguous()
            Y_target = Y_target.contiguous()

            mae_epoch.update(
                EO.reduce(Y_pred, "n t h -> t h", "mean").contiguous(), Y_target
            )

    mae_epoch = mae_epoch.compute()  # ty: ignore[missing-argument]

    mae_key = (
        "validation/mae" if metric_suffix is None else f"validation/mae_{metric_suffix}"
    )
    metrics = {mae_key: mae_epoch}
    accelerator.log(metrics, step=progress_bar.n)

    return metrics


def visualize_validation_predictions(
    model,
    batch: dict[str, Any],
    accelerator: Accelerator,
    progress_bar: ProgressBar,
    temperatures: tensordict.TensorDict | None = None,
    markers: tensordict.TensorDict | None = None,
    num_samples: int = 16,
    num_visualize: int = 8,
    max_context_prior_length: int = 32,
    max_distinct: int | None = 128,
    min_target: int = 10,
    generator: torch.Generator | None = None,
) -> None:
    """Visualize predictions from first few validation samples."""
    nested_batch = get_sets(
        batch=batch["height"],
        max_context_prior_length=max_context_prior_length,
        set_mode="nested-holdout",
        max_distinct=max_distinct,
        min_target=min_target,
        generator=generator,
    )

    context_priors_list = list(nested_batch["context_prior"])
    targets_list = list(nested_batch["target"])
    context_posteriors_list = list(nested_batch["context_posterior"])

    context_priors = tensordict.lazy_stack(context_priors_list[:num_visualize]).densify(
        layout=torch.jagged
    )
    targets_gt = tensordict.lazy_stack(targets_list[:num_visualize]).densify(
        layout=torch.jagged
    )
    context_posteriors_gt = tensordict.lazy_stack(
        context_posteriors_list[:num_visualize]
    ).densify(layout=torch.jagged)

    context_priors, targets_gt = consolidate_nested_tensors(context_priors, targets_gt)
    context_priors, context_posteriors_gt = consolidate_nested_tensors(
        context_priors, context_posteriors_gt
    )

    grid_points = create_grid_points(targets_gt.device)
    num_grid_points = len(grid_points)

    targets_with_grid = tensordict.lazy_stack(
        [
            tensordict.TensorDict(
                {
                    k: torch.cat([targets_list[i][k], grid_points[k]])  # ty: ignore[invalid-argument-type]
                    for k in grid_points.keys()  # noqa: SIM118
                },
                device=targets_list[i].device,
            )
            for i in range(num_visualize)
        ]
    ).densify(layout=torch.jagged)

    temperatures_subset = (
        temperatures[:num_visualize] if temperatures is not None else None
    )
    markers_subset = markers[:num_visualize] if markers is not None else None

    def _run_prediction_pass(sample_from_posterior: bool) -> dict[str, Any]:
        return model(
            context_priors=context_priors,
            targets=targets_with_grid,
            context_posteriors=context_posteriors_gt,
            temperatures=temperatures_subset,
            markers=markers_subset,
            num_samples=max(num_samples, LATENT_SCATTER_MIN_SAMPLES),
            get_samples=True,
            get_loss=False,
            sample_from_posterior=sample_from_posterior,
        )

    def _slice_prediction_grid(
        sample: torch.Tensor, keep_global: int | None = None
    ) -> tuple[torch.Tensor, int]:
        num_global = len(sample)
        target_global = (
            min(num_global, keep_global)
            if keep_global is not None
            else min(PREDICTION_PLOT_GLOBAL_SAMPLES, num_global)
        )
        trimmed = sample[:target_global, -num_grid_points:, :]
        return trimmed.cpu(), target_global

    def _build_prediction_images(
        predictions: torch.Tensor,
        prediction_scales: torch.Tensor | None,
        *,
        caption_suffix: str,
    ) -> list[wandb.Image]:
        prediction_images: list[wandb.Image] = []
        for i in range(num_visualize):
            context_sample = context_priors[i].to("cpu")
            context_posterior_sample = context_posteriors_gt[i].to("cpu")
            target_sample = targets_gt[i].to("cpu")
            prediction_sample = predictions[i]
            scale_sample = (
                prediction_scales[i] if prediction_scales is not None else None
            )
            pred_grid, kept_globals = _slice_prediction_grid(prediction_sample)
            scale_grid = None
            if scale_sample is not None:
                scale_grid, _ = _slice_prediction_grid(
                    scale_sample, keep_global=kept_globals
                )
            plot_predictions: dict[str, torch.Tensor] = {"grid": pred_grid}
            if scale_grid is not None:
                plot_predictions["grid_std"] = scale_grid

            img = create_uncertainty_plot(
                context_prior=context_sample,  # ty: ignore[invalid-argument-type]
                context_posterior=context_posterior_sample,  # ty: ignore[invalid-argument-type]
                targets=target_sample,  # ty: ignore[invalid-argument-type]
                predictions=plot_predictions,
                grid_x=grid_points["X"].cpu(),  # ty: ignore[invalid-argument-type]
                x_min=200.0,
                x_max=350.0,
                y_min=-0.1,
                y_max=2.0,
                training_days_min=177.0,
                training_days_max=329.0,
                figsize=PREDICTION_FIGSIZE,
                dpi=PREDICTION_DPI,
            )
            img = _downscale_image(img, max_size=PREDICTION_IMAGE_MAX_SIZE)
            prediction_images.append(
                wandb.Image(img, caption=f"Sample {i} ({caption_suffix})")
            )
        return prediction_images

    prior_output = _run_prediction_pass(sample_from_posterior=False)
    prior_predictions = prior_output["predictions"]["target"]
    prior_prediction_scales = prior_output["predictions"].get("target_scale")

    log_images: dict[str, Any] = {
        "validation/predictions": _build_prediction_images(
            prior_predictions, prior_prediction_scales, caption_suffix="prior"
        )
    }

    if "posterior" in prior_output:
        posterior_output = _run_prediction_pass(sample_from_posterior=True)
        posterior_predictions = posterior_output["predictions"]["target"]
        posterior_prediction_scales = posterior_output["predictions"].get(
            "target_scale"
        )
        log_images["validation/predictions_posterior"] = _build_prediction_images(
            posterior_predictions,
            posterior_prediction_scales,
            caption_suffix="posterior",
        )

    if "prior" in prior_output and "posterior" in prior_output:
        latent_comparison_images: list[wandb.Image] = []

        for i, (prior_sample, posterior_sample) in enumerate(
            zip(prior_output["prior"]["z"], prior_output["posterior"]["z"], strict=True)
        ):
            latent_img = create_latent_comparison_plot(
                prior_sample,
                posterior_sample,
                title=f"Sample {i} Prior vs Posterior",
                max_points=LATENT_SCATTER_MAX_POINTS,
            )
            latent_img = _downscale_image(latent_img, max_size=LATENT_IMAGE_MAX_SIZE)
            latent_comparison_images.append(
                wandb.Image(latent_img, caption=f"Sample {i} latents")
            )

        log_images["validation/latent_space"] = latent_comparison_images

    accelerator.log(log_images, step=progress_bar.n)


def create_latent_comparison_plot(
    prior_samples: torch.Tensor,
    posterior_samples: torch.Tensor,
    *,
    max_points: int = LATENT_SCATTER_MAX_POINTS,
    title: str = "Latent Comparison",
    figsize: tuple[float, float] = LATENT_COMPARISON_FIGSIZE,
    dpi: int = LATENT_DPI,
) -> Any:
    coords_prior = prior_samples[:max_points, 0, :3].detach().cpu()
    coords_posterior = posterior_samples[:max_points, 0, :3].detach().cpu()

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax in axes:
        ax.set_zlabel("z₃", fontsize=10)
        ax.view_init(elev=LATENT_3D_ELEVATION, azim=LATENT_3D_AZIMUTH)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))

    axis_data = [
        (axes[0], coords_prior, "Prior Latents"),
        (axes[1], coords_posterior, "Posterior Latents"),
    ]
    scatter_kwargs = {"s": 18, "alpha": 0.7, "edgecolors": "none"}

    for ax, coords, subtitle in axis_data:
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2], color="#2E86AB", **scatter_kwargs
        )
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel("z₁", fontsize=10)
        ax.set_ylabel("z₂", fontsize=10)
        ax.grid(visible=True, alpha=0.3, linewidth=0.5)

    def _compute_limits(
        coords_list: list[torch.Tensor],
    ) -> tuple[float, float, float, float, float | None, float | None]:
        stacked = torch.cat(coords_list, dim=0)
        x_min = stacked[:, 0].min().item()
        x_max = stacked[:, 0].max().item()
        y_min = stacked[:, 1].min().item()
        y_max = stacked[:, 1].max().item()
        z_min = stacked[:, 2].min().item()
        z_max = stacked[:, 2].max().item()

        return x_min, x_max, y_min, y_max, z_min, z_max

    x_min, x_max, y_min, y_max, z_min, z_max = _compute_limits(
        [coords_prior, coords_posterior]
    )
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if z_min is not None and z_max is not None:
            ax.set_zlim(z_min, z_max)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    img = fig2img(fig)
    img = _downscale_image(img, max_size=LATENT_IMAGE_MAX_SIZE)
    plt.close(fig)
    return img


def _downscale_image(image: Image.Image, *, max_size: tuple[int, int]) -> Image.Image:
    """Downscale PIL images in-place to constrain storage footprint."""
    image.thumbnail(max_size, resample=THUMBNAIL_RESAMPLE)
    return image


def save_state(accelerator: Accelerator, global_step: int, project_dir: Path) -> None:
    """Save model and training state to checkpoint."""
    save_path = project_dir / "checkpoints"
    checkpoint_path = save_path / f"checkpoint-{global_step}"
    if not checkpoint_path.exists():
        accelerator.save_state(str(checkpoint_path))


def save_wandb_run_id(accelerator: Accelerator, project_dir: Path) -> None:
    """Save wandb run ID to project directory for resumption."""
    if accelerator.is_main_process:
        try:
            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
            run_id = wandb_tracker.id
            run_id_path = project_dir / "wandb_run_id.txt"
            project_dir.mkdir(parents=True, exist_ok=True)
            run_id_path.write_text(run_id)
        except (ValueError, AttributeError):
            pass  # wandb not configured or tracker unavailable  # wandb not configured


def load_pretrained_weights(
    model: torch.nn.Module, checkpoint_folder: str
) -> list[str]:
    """Load shape-compatible weights from a checkpoint; return skipped tensor names."""
    from safetensors.torch import load_file

    from npnf.scripts.utils.prediction import resolve_checkpoint

    resolved = resolve_checkpoint(Path(checkpoint_folder))
    state_dict = load_file(str(resolved / "model.safetensors"))
    target = model.state_dict()
    compatible = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in target and target[name].shape == tensor.shape
    }
    skipped = sorted(set(state_dict) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    logger.info(
        f"Loaded {len(compatible)} pretrained tensors from {resolved}; "
        f"skipped {len(skipped)}: {skipped}"
    )
    return skipped


def load_wandb_run_id(project_dir: Path) -> str | None:
    """Load wandb run ID from project directory if it exists."""
    run_id_path = project_dir / "wandb_run_id.txt"
    if run_id_path.exists():
        return run_id_path.read_text().strip()
    return None


def save_training_config(project_dir: Path, cfg) -> None:
    """Save resolved Hydra config to project directory.

    Saves the full training config once at training start for later reference
    during prediction (e.g., to extract training dataset size).

    Args:
        project_dir: Project directory where config will be saved
        cfg: The resolved Hydra config (DictConfig from zen_cfg)
    """
    config_path = project_dir / "train_config.yaml"
    if not config_path.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_path)


def get_model_predictions(
    model,
    batch: dict[str, Any],
    temperatures: torch.Tensor | None = None,
    markers: torch.Tensor | None = None,
    num_samples: int = 16,
    get_samples: bool = True,
    max_context_prior_length: int | None = None,
    set_mode: SetMode = "disjoint-holdout",
    max_distinct: int | None = 128,
    min_target: int = 10,
    generator: torch.Generator | None = None,
    **kwargs,
) -> dict[str, Any]:
    nested_batch = get_sets(
        batch=batch["height"],
        max_context_prior_length=max_context_prior_length,
        set_mode=set_mode,
        max_distinct=max_distinct,
        min_target=min_target,
        generator=generator,
    )

    output = model(
        context_priors=nested_batch["context_prior"],
        context_posteriors=nested_batch["context_posterior"],
        targets=nested_batch["target"],
        get_samples=get_samples,
        num_samples=num_samples,
        temperatures=temperatures,
        markers=markers,
        get_loss=False,
        **kwargs,
    )
    output["nested_batch"] = nested_batch
    return output
