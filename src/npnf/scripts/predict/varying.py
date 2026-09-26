"""Predict lodging outcomes with varying context-window selection strategies."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import tensordict
import torch
from accelerate import Accelerator
from accelerate.utils import tqdm
from loguru import logger
from torch.utils.data import DataLoader

from npnf.configs.neural_process import TestVaryingConfig
from npnf.scripts.utils.paths import (
    build_results_path,
    extract_model_info,
    get_dataloader_name_from_hydra,
    get_dataset_size_from_checkpoint,
    parse_run_name,
)
from npnf.scripts.utils.prediction import (
    build_single_draw_height_view,
    get_indices_random,
    get_indices_sequential,
    get_indices_uncertainty,
    predict_batch_with_grid,
    prepare_batch_data,
    prepare_batch_schema_config,
    prepare_model_and_dataloaders,
    resolve_checkpoint,
    resolve_model_from_checkpoint,
    resolve_save_dtype,
    save_batch_results,
    update_context_priors_and_samples,
)
from npnf.scripts.utils.utils import create_grid_points


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for a single prediction method."""

    name: str
    start_index: int
    slice_size: int


@dataclass
class PredictionMethodsConfig:
    """Configuration for all enabled prediction methods."""

    B: int
    num_random_trials: int
    methods: list[MethodConfig]

    @classmethod
    def from_flags(
        cls,
        B: int,
        predict_sequential: bool,
        predict_uncertainty: bool,
        num_random_trials: int,
    ) -> "PredictionMethodsConfig":
        """Build config from prediction flags."""
        methods = []
        offset = 0
        if predict_sequential:
            methods.append(MethodConfig("sequential", offset, B))
            offset += B
        if predict_uncertainty:
            methods.append(MethodConfig("uncertainty", offset, B))
            offset += B
        if num_random_trials > 0:
            methods.append(MethodConfig("random", offset, B * num_random_trials))
        return cls(B=B, num_random_trials=num_random_trials, methods=methods)

    def get_method(self, name: str) -> MethodConfig | None:
        """Get method config by name."""
        return next((m for m in self.methods if m.name == name), None)


def _stack_step_predictions(step_predictions: list) -> tensordict.TensorDict:
    """Stack predictions across context steps into a single TensorDict."""
    stacked = [
        tensordict.lazy_stack(list(p)).densify(layout=torch.jagged)
        for p in step_predictions
    ]
    return tensordict.lazy_stack(stacked, dim=1)


def _transpose_context_indices(per_step_indices: list[list[int]]) -> list[list[int]]:
    """Transpose context indices from per-step to per-sample format."""
    if not per_step_indices:
        return []
    num_samples = len(per_step_indices[0]) if per_step_indices else 0
    return [
        [per_step_indices[step][sample] for step in range(len(per_step_indices))]
        for sample in range(num_samples)
    ]


def _extract_batch_features(
    batch: tensordict.TensorDict,
    use_temperature: bool,
    use_marker: bool,
    num_predictions: int,
) -> tuple[torch.Tensor | None, tensordict.TensorDict | None]:
    """Extract optional temperature and marker features from batch."""
    temperatures: torch.Tensor | None = None
    if use_temperature and "temperature" in batch:
        temperatures = batch["temperature"].repeat(num_predictions, 1)  # ty: ignore[invalid-assignment]

    markers = None
    if use_marker and "marker" in batch:
        markers = cast(tensordict.TensorDict, batch["marker"])
        if markers["Y"].is_nested:  # ty: ignore[unresolved-attribute]
            markers["Y"] = torch.nested.as_nested_tensor(
                list(markers["Y"]) * num_predictions,
                device=markers.device,
                layout=torch.jagged,
            )
        else:
            markers = markers.repeat(num_predictions)  # ty: ignore[invalid-argument-type]

    return temperatures, markers


def _get_target_lengths(batch_height: tensordict.TensorDict, B: int) -> torch.Tensor:
    """Compute target lengths (max valid context) for each sample in batch."""
    heights: torch.Tensor = batch_height["Y"]  # ty: ignore[invalid-assignment]
    if heights.is_nested:
        return torch.as_tensor(
            [len(heights[i]) for i in range(B)], device=batch_height.device
        )
    return torch.full((B,), heights.shape[1], device=batch_height.device)


def _process_step_predictions(
    predictions_list: list[tensordict.TensorDict], start: int, size: int
) -> tensordict.TensorDict:
    """Process a slice of predictions: stack, detach, clone, move to CPU, densify."""
    return (
        tensordict.lazy_stack(predictions_list[start : start + size])
        .detach()
        .clone()
        .cpu()
        .densify(layout=torch.jagged)
    )


def _extract_method_predictions(
    predictions: tensordict.TensorDict, config: PredictionMethodsConfig
) -> dict[str, tensordict.TensorDict]:
    """Extract grid predictions for each method from combined predictions."""
    predictions_list = list(predictions.select("grid"))
    return {
        method.name: _process_step_predictions(
            predictions_list, method.start_index, method.slice_size
        )
        for method in config.methods
    }


def _compute_next_indices(
    context_priors_batch: tensordict.TensorDict,
    predictions: tensordict.TensorDict,
    targets_in_context_mask_batch: torch.Tensor,
    num_context: int,
    config: PredictionMethodsConfig,
) -> list[int | None]:
    """Compute next context indices for all methods."""
    context_list = list(context_priors_batch)
    predictions_list = list(predictions)
    mask_list = list(targets_in_context_mask_batch)

    indices: list[int | None] = []
    for method in config.methods:
        start, end = method.start_index, method.start_index + method.slice_size
        if method.name == "sequential":
            indices.extend(get_indices_sequential(context_list[start:end], num_context))
        elif method.name == "uncertainty":
            indices.extend(
                get_indices_uncertainty(
                    predictions_list[start:end], mask_list[start:end]
                )
            )
        elif method.name == "random":
            indices.extend(get_indices_random(mask_list[start:end]))
    return indices


def _save_indices_by_method(
    batch_data: dict[str, dict[str, list]],
    next_context_indices: list[int | None],
    config: PredictionMethodsConfig,
) -> None:
    """Append context indices to batch_data for each method."""
    for method in config.methods:
        batch_data[method.name]["context_indices"].append(
            next_context_indices[
                method.start_index : method.start_index + method.slice_size
            ]
        )


def _prepare_method_save_data(
    method: MethodConfig,
    batch: tensordict.TensorDict,
    target_lengths: torch.Tensor,
    B: int,
    num_random_trials: int,
) -> tuple[tensordict.TensorDict, torch.Tensor]:
    """Prepare compact condition IDs and max_valid_context for saving."""
    multiplier = num_random_trials if method.name == "random" else 1

    max_valid_ctx = target_lengths.repeat(multiplier)
    compact_batch = batch.select("yearsite_uid", "genotype_id")
    method_batch_data = (
        tensordict.lazy_stack([compact_batch[i % B] for i in range(multiplier * B)])  # ty: ignore[invalid-argument-type]
        .densify(layout=torch.jagged)
        .cpu()
    )
    return method_batch_data, max_valid_ctx


def _config_for_method(
    run_config: dict | None, method: MethodConfig, num_random_trials: int
) -> dict | None:
    """Build method-specific config metadata for saved predictions."""
    if run_config is None:
        return None

    method_config = dict(run_config)
    method_config["prediction_method"] = method.name
    method_config["trials_per_sample"] = (
        num_random_trials if method.name == "random" else 1
    )
    return method_config


def _process_context_step(
    num_context: int,
    batch_data: dict[str, dict[str, list]],
    context_priors_batch: tensordict.TensorDict,
    targets_batch: tensordict.TensorDict,
    targets_in_context_prior_mask_batch: torch.Tensor,
    batch_height: tensordict.TensorDict,
    config: PredictionMethodsConfig,
    grid_points: tensordict.TensorDict,
    model,
    temperatures: torch.Tensor | None,
    markers: tensordict.TensorDict | None,
    noise_z: torch.Tensor | None,
    num_samples: int,
) -> tensordict.TensorDict:
    """Process a single context step: predict, extract, compute indices, update."""
    predictions = predict_batch_with_grid(
        context_priors_batch=context_priors_batch,
        targets_batch=targets_batch,
        grid_points=grid_points,
        model=model,
        temperatures=temperatures,
        markers=markers,
        num_samples=num_samples,
        noise_z=noise_z,
        include_targets_std=False,
        include_grid_std=False,
    )

    method_preds = _extract_method_predictions(predictions, config)
    for method_name, preds in method_preds.items():
        batch_data[method_name]["predictions"].append(preds)

    next_context_indices = _compute_next_indices(
        context_priors_batch,
        predictions,
        targets_in_context_prior_mask_batch,
        num_context,
        config,
    )
    _save_indices_by_method(batch_data, next_context_indices, config)

    return update_context_priors_and_samples(
        list(context_priors_batch),
        batch_height,
        config.B,
        list(targets_in_context_prior_mask_batch),
        next_context_indices,
    )


def process_dataloader(
    dataloader: DataLoader,
    split_name: str,
    base_path: Path,
    num_predictions: int,
    grid_points: tensordict.TensorDict,
    model,
    use_temperature: bool,
    use_marker: bool,
    predict_uncertainty: bool = False,
    predict_sequential: bool = True,
    num_random_trials: int = 0,
    num_samples: int = 64,
    max_num_context_prior: int | None = None,
    num_batches: int | None = None,
    noise_z: torch.Tensor | None = None,
    run_config: dict | None = None,
    save_dtype: torch.dtype | None = None,
) -> None:
    """Process dataloader: predict, select points, iterate context steps, save."""
    run_config = prepare_batch_schema_config(run_config, "compact", dataloader)

    for batch_index, batch in enumerate(
        tqdm(dataloader, desc=f"Testing {split_name}...", leave=False)
    ):
        B, *_ = batch.batch_size
        batch_height = (
            build_single_draw_height_view(batch["height"])
            .densify(layout=torch.jagged)
            .clone()
        )

        temperatures, markers = _extract_batch_features(
            batch, use_temperature, use_marker, num_predictions
        )

        (
            context_priors_batch,
            targets_batch,
            targets_in_context_prior_mask_batch,
            max_num_targets,
        ) = prepare_batch_data(batch=batch, num_predictions=num_predictions)
        effective_max_context = (
            min(max_num_targets, max_num_context_prior)
            if max_num_context_prior is not None
            else max_num_targets
        )

        config = PredictionMethodsConfig.from_flags(
            B, predict_sequential, predict_uncertainty, num_random_trials
        )
        target_lengths = _get_target_lengths(batch_height, B)
        batch_data: dict[str, dict[str, list]] = {
            m.name: {"predictions": [], "context_indices": []} for m in config.methods
        }

        for num_context in tqdm(
            range(effective_max_context + 1), desc="Sampling...", position=1
        ):
            context_priors_batch = _process_context_step(
                num_context=num_context,
                batch_data=batch_data,
                context_priors_batch=context_priors_batch,
                targets_batch=targets_batch,
                targets_in_context_prior_mask_batch=targets_in_context_prior_mask_batch,
                batch_height=batch_height,
                config=config,
                grid_points=grid_points,
                model=model,
                temperatures=temperatures,
                markers=markers,
                noise_z=noise_z,
                num_samples=num_samples,
            )

        for method in config.methods:
            method_data = batch_data[method.name]
            stacked_predictions = _stack_step_predictions(method_data["predictions"])
            transposed_indices = _transpose_context_indices(
                method_data["context_indices"]
            )

            method_batch_data, max_valid_ctx = _prepare_method_save_data(
                method, batch, target_lengths, B, num_random_trials
            )

            save_batch_results(
                batch_index=batch_index,
                method_name=method.name,
                base_path=base_path,
                predictions=stacked_predictions,
                batch_data=method_batch_data,
                grid_points=grid_points,
                max_valid_context=max_valid_ctx,
                context_indices=transposed_indices,
                config=_config_for_method(run_config, method, num_random_trials),
                save_dtype=save_dtype,
                save_batch_dtype=save_dtype,
            )

        if num_batches is not None and batch_index + 1 >= num_batches:
            break


def predict_varying(
    model,
    dataloaders: dict[str, DataLoader],
    accelerator: Accelerator,
    checkpoint_folder: str,
    dataloader_name: str,
    use_temperature: bool,
    use_marker: bool,
    predict_uncertainty: bool = True,
    predict_sequential: bool = False,
    num_samples: int = 64,
    num_random_trials: int = 0,
    max_num_context_prior: int | None = None,
    num_batches: int | None = None,
    save_dtype: torch.dtype | None = None,
    run_config: dict | None = None,
) -> None:
    num_predictions = num_random_trials + predict_sequential + predict_uncertainty
    if num_predictions == 0:
        msg = "No predictions to make"
        raise ValueError(msg)

    grid_points = create_grid_points(accelerator.device)
    noise_z = torch.randn((num_samples, model.latent_dim), device=accelerator.device)
    dataset_size = get_dataset_size_from_checkpoint(checkpoint_folder)
    run_name, checkpoint_name = extract_model_info(checkpoint_folder)
    model_name, run_suffix = parse_run_name(run_name)

    for split_name, dataloader in dataloaders.items():
        base_path = build_results_path(
            dataloader_name=dataloader_name,
            model=model_name,
            checkpoint=checkpoint_name,
            method="",
            split_name=split_name,
            use_temperature=use_temperature,
            use_marker=use_marker,
            dataset_size=dataset_size,
            run_suffix=run_suffix,
        )

        process_dataloader(
            dataloader=dataloader,
            split_name=split_name,
            base_path=base_path,
            num_predictions=num_predictions,
            grid_points=grid_points,
            model=model,
            use_temperature=use_temperature,
            use_marker=use_marker,
            predict_uncertainty=predict_uncertainty,
            predict_sequential=predict_sequential,
            num_random_trials=num_random_trials,
            num_samples=num_samples,
            max_num_context_prior=max_num_context_prior,
            num_batches=num_batches,
            noise_z=noise_z,
            run_config=run_config,
            save_dtype=save_dtype,
        )


def main(
    dataloaders: dict[str, DataLoader],
    accelerator: Accelerator,
    checkpoint_folder: str,
    use_temperature: bool = True,
    use_marker: bool = True,
    num_samples: int = 64,
    predict_uncertainty: bool = True,
    predict_sequential: bool = False,
    predict_random: bool = True,
    num_random_trials: int = 3,
    max_num_context_prior: int = 20,
    num_batches: int | None = None,
    save_dtype: str | None = "float16",
) -> None:
    resolved_checkpoint_folder = str(resolve_checkpoint(Path(checkpoint_folder)))
    dataloader_name = get_dataloader_name_from_hydra()

    effective_num_random_trials = num_random_trials if predict_random else 0

    model = resolve_model_from_checkpoint(resolved_checkpoint_folder)
    model, dataloaders, _ = prepare_model_and_dataloaders(
        model, dataloaders, checkpoint_folder, accelerator
    )

    resolved_dtype = resolve_save_dtype(save_dtype)

    config = {
        "num_samples": num_samples,
        "num_random_trials": effective_num_random_trials,
        "max_num_context_prior": max_num_context_prior,
        "num_batches": num_batches,
        "use_temperature": use_temperature,
        "use_marker": use_marker,
        "checkpoint_folder": resolved_checkpoint_folder,
        "dataloader_name": dataloader_name,
        "model_variance": model.variance.item(),
        "save_dtype": save_dtype,
        "batch_schema": "compact",
    }

    logger.info(
        f"""
        Testing (varying context_priors) with:
            - num_samples: {num_samples}
            - use_temperature: {use_temperature}
            - use_marker: {use_marker}
            - checkpoint_folder: {resolved_checkpoint_folder}
            - predict_uncertainty: {predict_uncertainty}
            - predict_sequential: {predict_sequential}
            - num_random_trials: {effective_num_random_trials}
            - max_num_context_prior: {max_num_context_prior}
            - save_dtype: {save_dtype}
            - batch_schema: compact
        """
    )

    with torch.inference_mode():
        predict_varying(
            dataloaders=dataloaders,
            model=model,
            accelerator=accelerator,
            checkpoint_folder=resolved_checkpoint_folder,
            dataloader_name=dataloader_name,
            use_temperature=use_temperature,
            use_marker=use_marker,
            num_samples=num_samples,
            predict_uncertainty=predict_uncertainty,
            predict_sequential=predict_sequential,
            num_random_trials=effective_num_random_trials,
            max_num_context_prior=max_num_context_prior,
            num_batches=num_batches,
            save_dtype=resolved_dtype,
            run_config=config,
        )

    accelerator.end_training()


if __name__ == "__main__":
    from npnf.scripts.utils.hydra import run_hydra_main

    run_hydra_main(TestVaryingConfig, main)
