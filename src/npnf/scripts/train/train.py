import random
from pathlib import Path
from typing import Any

import schedulefree
import torch
from accelerate import Accelerator
from accelerate.utils import tqdm
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm as ProgressBar
from transformers import get_cosine_schedule_with_warmup

from npnf.configs.neural_process import TrainConfig
from npnf.models.neural_process.utils import SetMode, get_sets
from npnf.scripts.utils.paths import format_dataset_size, get_model_name_from_hydra
from npnf.scripts.utils.prediction import resolve_checkpoint
from npnf.scripts.utils.schedulers import ContextLengthScheduler, KLWeightScheduler
from npnf.scripts.utils.training import (
    handle_optimizer_mode,
    load_pretrained_weights,
    load_wandb_run_id,
    save_state,
    save_training_config,
    save_wandb_run_id,
    setup_checkpoint_and_validation_schedule,
    should_checkpoint,
    should_validate,
    validate_epoch,
)


def train_step(
    batch: TensorDict,
    model: nn.Module,
    accelerator: Accelerator,
    optimizer_dict: dict[str, Any],
    use_temperature_probability: float = 0.0,
    use_marker_probability: float = 0.0,
    use_full_context_probability: float = 0.8,
    small_context_prior_length_upper_bound: int = 3,
    max_context_prior_length: int = 50,
    set_mode: SetMode = "nested-noprior",
    max_distinct: int | None = 128,
    min_target: int = 10,
) -> dict[str, Any]:
    temperatures = (
        batch.get("temperature")
        if torch.rand(1) < use_temperature_probability
        else None
    )
    markers = batch.get("marker") if torch.rand(1) < use_marker_probability else None

    context_scheduler = optimizer_dict.get("context_scheduler")
    scheduled_context_prior_length = (
        context_scheduler.get_length() if context_scheduler is not None else None
    )
    effective_context_prior_length = (
        scheduled_context_prior_length
        if scheduled_context_prior_length is not None
        else max_context_prior_length
    )

    if random.random() <= use_full_context_probability:
        batch = get_sets(
            batch["height"],  # ty: ignore[invalid-argument-type]
            max_context_prior_length=effective_context_prior_length,
            set_mode=set_mode,
            max_distinct=max_distinct,
            min_target=min_target,
        )
    else:
        context_prior_length_close_to_0 = min(
            random.randint(0, small_context_prior_length_upper_bound),
            effective_context_prior_length,
        )
        batch = get_sets(
            batch["height"],  # ty: ignore[invalid-argument-type]
            max_context_prior_length=context_prior_length_close_to_0,
            set_mode=set_mode,
            max_distinct=max_distinct,
            min_target=min_target,
        )

    kl_weight = (
        optimizer_dict["kl_scheduler"].get_weight()
        if "kl_scheduler" in optimizer_dict
        else 1.0
    )

    output = model(
        context_priors=batch["context_prior"],
        context_posteriors=batch["context_posterior"],
        targets=batch["target"],
        temperatures=temperatures,
        markers=markers,
        sample_from_posterior=True,
        kl_weight=kl_weight,
    )

    loss = output["loss_dict"]["loss"]

    optimizer_dict["optimizer"].zero_grad()
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), 0.1)

    optimizer_dict["optimizer"].step()
    if "lr_scheduler" in optimizer_dict:
        optimizer_dict["lr_scheduler"].step()
    if "kl_scheduler" in optimizer_dict:
        optimizer_dict["kl_scheduler"].step()
    if context_scheduler is not None:
        context_scheduler.step()

    loss_dict = {f"train/loss/{k}": v for k, v in output["loss_dict"].items()}

    if "lr_scheduler" in optimizer_dict:
        loss_dict["train/lr"] = optimizer_dict["lr_scheduler"].get_last_lr()[0]
    if "kl_scheduler" in optimizer_dict:
        loss_dict["train/kl_weight"] = kl_weight
    loss_dict["train/context_prior_length"] = float(effective_context_prior_length)

    output["metrics"] = loss_dict

    return output


def write_progress(
    progress_bar: ProgressBar, accelerator: Accelerator, message: str
) -> None:
    """Write a durable tqdm-compatible progress message from the main process."""
    if accelerator.is_main_process:
        progress_bar.write(message)


def run_training(
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    model,
    accelerator: Accelerator,
    optimizer_dict: dict[str, Any],
    progress_bar: ProgressBar,
    initial: int,
    batch_size: int,
    project_dir: Path,
    use_temperature_probability_train: float = 0.0,
    use_marker_probability_train: float = 0.0,
    use_full_context_probability_train: float = 0.8,
    small_context_prior_length_upper_bound_train: int = 3,
    use_temperature_validation: bool = False,
    use_marker_validation: bool = False,
    max_context_prior_length: int = 50,
    checkpoint_frequency: int | None = None,
    validation_frequency: int | None = None,
    set_mode: SetMode = "nested-noprior",
    max_distinct: int | None = 128,
    min_target: int = 10,
) -> None:
    train_dataloader_iter = iter(train_dataloader)

    next_checkpoint, next_validation = setup_checkpoint_and_validation_schedule(initial)

    while True:
        if should_checkpoint(progress_bar, next_checkpoint, checkpoint_frequency):
            checkpoint_path = (
                project_dir / "checkpoints" / f"checkpoint-{progress_bar.n}"
            )
            write_progress(
                progress_bar,
                accelerator,
                f"Checkpoint start: samples={progress_bar.n}/{progress_bar.total} "
                f"path={checkpoint_path}",
            )
            model.eval()
            handle_optimizer_mode(optimizer_dict, "eval")
            save_state(accelerator, progress_bar.n, project_dir)
            model.train()
            handle_optimizer_mode(optimizer_dict, "train")
            write_progress(
                progress_bar,
                accelerator,
                f"Checkpoint finished: samples={progress_bar.n}/{progress_bar.total} "
                f"path={checkpoint_path}",
            )
            assert checkpoint_frequency is not None
            next_checkpoint += checkpoint_frequency
        if should_validate(progress_bar, next_validation, validation_frequency):
            write_progress(
                progress_bar,
                accelerator,
                f"Validation start: samples={progress_bar.n}/{progress_bar.total}",
            )
            model.eval()
            handle_optimizer_mode(optimizer_dict, "eval")
            validate_epoch(
                model=model,
                validation_dataloader=validation_dataloader,
                use_temperature=use_temperature_validation,
                use_marker=use_marker_validation,
                accelerator=accelerator,
                progress_bar=progress_bar,
                max_context_prior_length=max_context_prior_length,
                max_distinct=max_distinct,
                min_target=min_target,
            )
            default_combo = (use_temperature_validation, use_marker_validation)
            for use_temperature, use_marker in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                if (use_temperature, use_marker) == default_combo:
                    continue
                suffix_parts = []
                if use_temperature:
                    suffix_parts.append("temp")
                if use_marker:
                    suffix_parts.append("marker")
                suffix = "_".join(suffix_parts) if suffix_parts else "none"
                validate_epoch(
                    model=model,
                    validation_dataloader=validation_dataloader,
                    use_temperature=use_temperature,
                    use_marker=use_marker,
                    accelerator=accelerator,
                    progress_bar=progress_bar,
                    max_context_prior_length=max_context_prior_length,
                    max_distinct=max_distinct,
                    min_target=min_target,
                    metric_suffix=suffix,
                )
            write_progress(
                progress_bar,
                accelerator,
                f"Validation finished: samples={progress_bar.n}/{progress_bar.total}",
            )
            assert validation_frequency is not None
            next_validation += validation_frequency

        model.train()
        handle_optimizer_mode(optimizer_dict, "train")
        for batch in train_dataloader_iter:
            assert progress_bar.total is not None
            if progress_bar.n >= progress_bar.total:
                return

            output = train_step(
                batch=batch,
                model=model,
                accelerator=accelerator,
                optimizer_dict=optimizer_dict,
                use_temperature_probability=use_temperature_probability_train,
                use_marker_probability=use_marker_probability_train,
                use_full_context_probability=use_full_context_probability_train,
                small_context_prior_length_upper_bound=small_context_prior_length_upper_bound_train,
                max_context_prior_length=max_context_prior_length,
                set_mode=set_mode,
                max_distinct=max_distinct,
                min_target=min_target,
            )

            accelerator.log(output["metrics"], step=progress_bar.n)
            progress_bar.update(batch_size)

            if should_validate(
                progress_bar, next_validation, validation_frequency
            ) or should_checkpoint(progress_bar, next_checkpoint, checkpoint_frequency):
                break
        else:
            train_dataloader_iter = iter(train_dataloader)


def setup_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    use_lr_schedule: bool = False,
    num_warmup_steps: int | None = None,
    num_training_steps: int | None = None,
    kl_hold_steps: int | None = None,
    kl_ramp_steps: int | None = None,
    kl_weight_initial: float = 0.0,
    kl_weight_peak: float = 0.5,
    kl_anneal_shape: str = "linear",
    context_prior_anneal_steps: int | None = None,
    context_prior_length_initial: float = 8.0,
    context_prior_length_final: float = 50.0,
    context_prior_anneal_shape: str = "linear",
) -> dict[str, Any]:
    """Setup optimizer with optional learning rate scheduling.

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ("adamw" or "schedulefree")
        learning_rate: Learning rate
        weight_decay: Weight decay (constant)
        use_lr_schedule: Enable cosine LR schedule with warmup (adamw only)
        num_warmup_steps: Number of warmup steps for LR scheduler
        num_training_steps: Total training steps for schedulers
        kl_hold_steps: Steps to hold KL weight at exactly 0
        kl_ramp_steps: Steps for KL weight ramp
        kl_weight_initial: Initial KL weight
        kl_weight_peak: Peak KL weight after ramp completes
        kl_anneal_shape: KL annealing shape ("linear", "cosine", "constant")
        context_prior_anneal_steps: Steps for context_prior length annealing
        context_prior_length_initial: Initial context_prior length
        context_prior_length_final: Final context_prior length
        context_prior_anneal_shape: Context_prior annealing shape

    Returns:
        Dictionary containing optimizer and optional schedulers
    """
    optimizer_dict: dict[str, Any] = {}

    if optimizer_type == "schedulefree":
        optimizer_dict["optimizer"] = schedulefree.RAdamScheduleFree(
            model.parameters(),
            lr=learning_rate,
            betas=(0.95, 0.98),
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        optimizer_dict["optimizer"] = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.95, 0.98),
            weight_decay=weight_decay,
        )

        if use_lr_schedule:
            if num_training_steps is None:
                msg = "num_training_steps required when use_lr_schedule=True"
                raise ValueError(msg)
            if num_warmup_steps is None:
                num_warmup_steps = 0

            optimizer_dict["lr_scheduler"] = get_cosine_schedule_with_warmup(
                optimizer_dict["optimizer"],
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
    else:
        msg = (
            f"Unknown optimizer_type: '{optimizer_type}'. "
            "Must be 'adamw' or 'schedulefree'"
        )
        raise ValueError(msg)

    if kl_hold_steps is not None or kl_ramp_steps is not None:
        optimizer_dict["kl_scheduler"] = KLWeightScheduler(
            hold_steps=0 if kl_hold_steps is None else kl_hold_steps,
            ramp_steps=0 if kl_ramp_steps is None else kl_ramp_steps,
            shape=kl_anneal_shape,
            initial_weight=kl_weight_initial,
            peak_weight=kl_weight_peak,
        )
    if context_prior_anneal_steps is not None:
        optimizer_dict["context_scheduler"] = ContextLengthScheduler(
            total_steps=context_prior_anneal_steps,
            shape=context_prior_anneal_shape,
            initial_length=context_prior_length_initial,
            final_length=context_prior_length_final,
        )

    return optimizer_dict


def train(
    run_name: str,
    dataloaders: DictConfig,
    accelerator: Accelerator,
    model,
    num_training_samples: int,
    per_device_batch_size: int,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    use_lr_schedule: bool = False,
    num_warmup_samples: int = 0,
    debug: bool = False,
    use_temperature_probability_train: float = 0.0,
    use_marker_probability_train: float = 0.0,
    use_full_context_probability_train: float = 0.8,
    small_context_prior_length_upper_bound_train: int = 3,
    use_temperature_validation: bool = False,
    use_marker_validation: bool = False,
    max_context_prior_length: int = 50,
    set_mode: SetMode = "nested-noprior",
    max_distinct: int | None = 128,
    min_target: int = 10,
    context_prior_anneal_fraction: float | None = None,
    context_prior_length_initial: float = 8.0,
    context_prior_length_final: float = 50.0,
    context_prior_anneal_shape: str = "linear",
    checkpoint_frequency: int | None = None,
    checkpoint_folder: str | None = None,
    resume_training: bool = True,
    validation_frequency: int | None = None,
    kl_hold_fraction: float | None = None,
    kl_ramp_fraction: float | None = None,
    kl_weight_initial: float = 0.0,
    kl_weight_peak: float = 0.5,
    kl_anneal_shape: str = "linear",
    zen_cfg=None,
) -> None:
    # TF32 speeds up training; prediction and metrics keep full float32 precision.
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"  # ty: ignore[unresolved-attribute]

    effective_batch_size = per_device_batch_size * accelerator.num_processes

    # Convert sample counts to step counts
    num_training_steps = num_training_samples // effective_batch_size
    num_warmup_steps = num_warmup_samples // effective_batch_size

    kl_hold_steps = (
        int(num_training_samples * kl_hold_fraction) // effective_batch_size
        if kl_hold_fraction is not None
        else None
    )
    kl_ramp_steps = (
        int(num_training_samples * kl_ramp_fraction) // effective_batch_size
        if kl_ramp_fraction is not None
        else None
    )
    context_prior_steps = (
        int(num_training_samples * context_prior_anneal_fraction)
        // effective_batch_size
        if context_prior_anneal_fraction is not None
        else None
    )

    optimizer_dict = setup_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_lr_schedule=use_lr_schedule,
        num_warmup_steps=num_warmup_steps if use_lr_schedule else None,
        num_training_steps=num_training_steps if use_lr_schedule else None,
        kl_hold_steps=kl_hold_steps,
        kl_ramp_steps=kl_ramp_steps,
        kl_weight_initial=kl_weight_initial,
        kl_weight_peak=kl_weight_peak,
        kl_anneal_shape=kl_anneal_shape,
        context_prior_anneal_steps=context_prior_steps,
        context_prior_length_initial=context_prior_length_initial,
        context_prior_length_final=context_prior_length_final,
        context_prior_anneal_shape=context_prior_anneal_shape,
    )

    if checkpoint_folder is not None and not resume_training:
        load_pretrained_weights(model, checkpoint_folder)

    model, train_dataloader, validation_dataloader = accelerator.prepare(
        model, dataloaders.train, dataloaders.validation
    )
    optimizer_dict = {k: accelerator.prepare(v) for k, v in optimizer_dict.items()}
    if "kl_scheduler" in optimizer_dict:
        accelerator.register_for_checkpointing(optimizer_dict["kl_scheduler"])
    if "context_scheduler" in optimizer_dict:
        accelerator.register_for_checkpointing(optimizer_dict["context_scheduler"])

    initial = 0
    if checkpoint_folder is not None and resume_training:
        resolved_checkpoint = resolve_checkpoint(Path(checkpoint_folder))
        checkpoint_folder = str(resolved_checkpoint)
        accelerator.load_state(checkpoint_folder, strict=False)
        initial = int(resolved_checkpoint.name.split("-")[-1])

    model_name = get_model_name_from_hydra()
    dataset_size = len(dataloaders.train.dataset)
    formatted_size = format_dataset_size(dataset_size)
    if run_name:
        full_run_name = f"{model_name}-{formatted_size}-{run_name}"
        project_dir = Path(accelerator.project_dir).parent / full_run_name
    else:
        full_run_name = f"{model_name}-{formatted_size}"
        project_dir = Path(accelerator.project_dir) / full_run_name
    wandb_init_kwargs = {"group": "train", "name": full_run_name}
    if checkpoint_folder is not None and resume_training:
        existing_run_id = load_wandb_run_id(project_dir)
        if existing_run_id is not None:
            wandb_init_kwargs["id"] = existing_run_id
            wandb_init_kwargs["resume"] = "allow"

    accelerator.init_trackers(
        project_name="neural_process", init_kwargs={"wandb": wandb_init_kwargs}
    )
    save_wandb_run_id(accelerator, project_dir)

    save_training_config(project_dir, zen_cfg)

    with (
        torch.autograd.set_detect_anomaly(debug),
        tqdm(
            total=num_training_samples,
            initial=initial,
            desc="Training",
            unit="samples",
            unit_scale=True,
            dynamic_ncols=False,
            mininterval=30,
        ) as progress_bar,
    ):
        run_training(
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            model=model,
            accelerator=accelerator,
            optimizer_dict=optimizer_dict,
            progress_bar=progress_bar,
            initial=initial,
            batch_size=effective_batch_size,
            project_dir=project_dir,
            use_temperature_probability_train=use_temperature_probability_train,
            use_marker_probability_train=use_marker_probability_train,
            use_full_context_probability_train=use_full_context_probability_train,
            small_context_prior_length_upper_bound_train=small_context_prior_length_upper_bound_train,
            use_temperature_validation=use_temperature_validation,
            use_marker_validation=use_marker_validation,
            max_context_prior_length=max_context_prior_length,
            checkpoint_frequency=checkpoint_frequency,
            validation_frequency=validation_frequency,
            set_mode=set_mode,
            max_distinct=max_distinct,
            min_target=min_target,
        )
        write_progress(
            progress_bar,
            accelerator,
            f"Training loop complete: samples={progress_bar.n}/{progress_bar.total}",
        )

        if checkpoint_frequency is not None:
            checkpoint_path = (
                project_dir / "checkpoints" / f"checkpoint-{progress_bar.n}"
            )
            sample_status = f"samples={progress_bar.n}/{progress_bar.total}"
            write_progress(
                progress_bar,
                accelerator,
                f"Final checkpoint start: {sample_status} path={checkpoint_path}",
            )
            model.eval()
            handle_optimizer_mode(optimizer_dict, "eval")
            save_state(accelerator, progress_bar.n, project_dir)
            write_progress(
                progress_bar,
                accelerator,
                f"Final checkpoint finished: {sample_status} path={checkpoint_path}",
            )

    accelerator.end_training()


if __name__ == "__main__":
    from npnf.scripts.utils.hydra import run_hydra_main

    run_hydra_main(TrainConfig, train)
