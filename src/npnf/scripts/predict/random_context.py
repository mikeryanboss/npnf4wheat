"""Predict with deterministic random sparse height context."""

from pathlib import Path

import tensordict
import torch
from accelerate import Accelerator
from accelerate.utils import tqdm
from loguru import logger
from torch.utils.data import DataLoader

from npnf.configs.neural_process import RandomContextPredictionConfig
from npnf.models.utils import consolidate_tensordict_jagged_dim
from npnf.scripts.utils.paths import (
    build_results_path,
    extract_model_info,
    get_dataloader_name_from_hydra,
    get_dataset_size_from_checkpoint,
    parse_run_name,
)
from npnf.scripts.utils.prediction import (
    build_single_draw_height_view,
    get_empty_observations,
    peak_height_index,
    predict_batch_with_grid,
    prepare_batch_schema_config,
    prepare_model_and_dataloaders,
    resolve_batch_schema,
    resolve_checkpoint,
    resolve_model_from_checkpoint,
    resolve_save_dtype,
    save_batch_results,
    select_batch_metadata_for_schema,
)
from npnf.scripts.utils.utils import create_grid_points

METHOD_NAME = "random_context"
CONTEXT_POOL = "growing_pre_max_height"


def _sample_context_indices(
    height: tensordict.TensorDict,
    generator: torch.Generator,
    min_context: int,
    max_context: int,
) -> list[int]:
    days: torch.Tensor = height["X"][..., 0]  # ty: ignore[invalid-assignment]
    max_height_day = days[peak_height_index(height)]

    candidate_mask = (days >= 200) & (days <= 321) & (days <= max_height_day)
    candidates = candidate_mask.nonzero(as_tuple=False).squeeze(-1).cpu()

    k = int(torch.randint(min_context, max_context + 1, (), generator=generator).item())
    positions = torch.randperm(len(candidates), generator=generator)[:k]
    return candidates[positions].sort().values.tolist()


def _build_random_context_priors(
    batch_height: tensordict.TensorDict,
    generator: torch.Generator,
    min_context: int,
    max_context: int,
) -> tuple[tensordict.TensorDict, list[list[int]]]:
    height_batch = build_single_draw_height_view(batch_height)

    contexts = []
    context_indices = []
    for height in height_batch:
        height.batch_size = [len(height["X"])]
        indices = _sample_context_indices(height, generator, min_context, max_context)
        context_indices.append(indices)
        contexts.append(height.select("X", "X_normalized", "Y")[indices])

    context_priors_batch = tensordict.lazy_stack(contexts, dim=0).densify(
        layout=torch.jagged
    )
    context_priors_batch.batch_size = context_priors_batch.batch_size[:1]
    return consolidate_tensordict_jagged_dim(context_priors_batch), context_indices


def process_dataloader(
    dataloader: DataLoader,
    split_name: str,
    base_path: Path,
    grid_points: tensordict.TensorDict,
    model,
    use_temperature: bool,
    use_marker: bool,
    num_samples: int = 64,
    context_seed: int = 42,
    min_context: int = 1,
    max_context: int = 10,
    num_batches: int | None = None,
    config: dict | None = None,
    save_targets_std: bool = False,
    save_grid_std: bool = False,
    save_dtype: torch.dtype | None = torch.float16,
    save_batch_dtype: torch.dtype | None = torch.float16,
    batch_schema: str = "compact",
) -> None:
    """Predict using deterministic random sparse height context."""
    batch_schema = resolve_batch_schema(batch_schema)
    config = prepare_batch_schema_config(config, batch_schema, dataloader)
    generator = torch.Generator(device="cpu").manual_seed(context_seed)

    for batch_index, batch in enumerate(
        tqdm(dataloader, desc=f"Testing {split_name}...", leave=False)
    ):
        B, *_ = batch.batch_size

        temperatures = None
        if use_temperature and "temperature" in batch:
            temperatures = batch["temperature"]
        markers = None
        if use_marker and "marker" in batch:
            markers = batch["marker"]

        context_priors_batch, context_indices = _build_random_context_priors(
            batch["height"], generator, min_context, max_context
        )
        targets_batch = get_empty_observations(
            batch_size=B, device=grid_points.device, dtype=grid_points["X"].dtype
        )

        predictions = predict_batch_with_grid(
            context_priors_batch=context_priors_batch,
            targets_batch=targets_batch,
            grid_points=grid_points,
            model=model,
            temperatures=temperatures,
            markers=markers,
            num_samples=num_samples,
            include_targets_std=save_targets_std,
            include_grid_std=save_grid_std,
            genotype_ids=batch["genotype_id"],
            yearsite_ids=batch["yearsite_uid"],
        )

        stacked = tensordict.lazy_stack(list(predictions)).densify(layout=torch.jagged)
        stacked_predictions = tensordict.lazy_stack([stacked], dim=1)

        save_batch_results(
            batch_index=batch_index,
            method_name=METHOD_NAME,
            base_path=base_path,
            predictions=stacked_predictions,
            batch_data=select_batch_metadata_for_schema(batch, batch_schema),
            grid_points=grid_points,
            context_indices=context_indices,
            config=config,
            save_dtype=save_dtype,
            save_batch_dtype=save_batch_dtype,
        )

        if num_batches is not None and batch_index + 1 >= num_batches:
            break


def main(
    dataloaders: dict[str, DataLoader],
    accelerator: Accelerator,
    checkpoint_folder: str,
    use_temperature: bool = True,
    use_marker: bool = True,
    num_samples: int = 64,
    context_seed: int = 42,
    min_context: int = 1,
    max_context: int = 10,
    per_device_batch_size: int = 128,
    num_batches: int | None = None,
    save_targets_std: bool = False,
    save_grid_std: bool = False,
    save_dtype: str | None = "float16",
    save_batch_dtype: str | None = "float16",
    batch_schema: str = "compact",
    grid: str = "synthetic",
) -> None:
    if per_device_batch_size <= 0:
        msg = "`per_device_batch_size` must be positive."
        raise ValueError(msg)
    batch_schema = resolve_batch_schema(batch_schema)

    resolved_checkpoint_folder = str(resolve_checkpoint(Path(checkpoint_folder)))
    dataloader_name = get_dataloader_name_from_hydra()

    model = resolve_model_from_checkpoint(resolved_checkpoint_folder)
    model, dataloaders, _ = prepare_model_and_dataloaders(
        model, dataloaders, checkpoint_folder, accelerator
    )
    resolved_save_dtype = resolve_save_dtype(save_dtype)
    resolved_save_batch_dtype = resolve_save_dtype(save_batch_dtype)

    config = {
        "prediction_method": METHOD_NAME,
        "num_samples": num_samples,
        "context_seed": context_seed,
        "min_context": min_context,
        "max_context": max_context,
        "num_context_trials": 1,
        "context_pool": CONTEXT_POOL,
        "per_device_batch_size": per_device_batch_size,
        "num_batches": num_batches,
        "use_temperature": use_temperature,
        "use_marker": use_marker,
        "checkpoint_folder": resolved_checkpoint_folder,
        "dataloader_name": dataloader_name,
        "model_variance": model.variance.item(),
        "save_targets_std": save_targets_std,
        "save_grid_std": save_grid_std,
        "save_dtype": save_dtype,
        "save_batch_dtype": save_batch_dtype,
        "batch_schema": batch_schema,
        "grid": grid,
    }

    logger.info(
        f"""
        Testing (random sparse context) with:
            - num_samples: {num_samples}
            - context_seed: {context_seed}
            - min_context: {min_context}
            - max_context: {max_context}
            - use_temperature: {use_temperature}
            - use_marker: {use_marker}
            - per_device_batch_size: {per_device_batch_size}
            - save_targets_std: {save_targets_std}
            - save_grid_std: {save_grid_std}
            - save_dtype: {save_dtype}
            - save_batch_dtype: {save_batch_dtype}
            - batch_schema: {batch_schema}
            - checkpoint_folder: {resolved_checkpoint_folder}
        """
    )

    grid_points = create_grid_points(accelerator.device, grid=grid)
    dataset_size = get_dataset_size_from_checkpoint(resolved_checkpoint_folder)
    run_name, checkpoint_name = extract_model_info(resolved_checkpoint_folder)
    model_name, run_suffix = parse_run_name(run_name)

    with torch.inference_mode():
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
                grid_points=grid_points,
                model=model,
                use_temperature=use_temperature,
                use_marker=use_marker,
                num_samples=num_samples,
                context_seed=context_seed,
                min_context=min_context,
                max_context=max_context,
                num_batches=num_batches,
                config=config,
                save_targets_std=save_targets_std,
                save_grid_std=save_grid_std,
                save_dtype=resolved_save_dtype,
                save_batch_dtype=resolved_save_batch_dtype,
                batch_schema=batch_schema,
            )

    accelerator.end_training()


if __name__ == "__main__":
    from npnf.scripts.utils.hydra import run_hydra_main

    run_hydra_main(RandomContextPredictionConfig, main)
