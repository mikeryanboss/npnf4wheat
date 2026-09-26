"""Predict lodging outcomes using max-height context_priors."""

from pathlib import Path

import tensordict
import torch
from accelerate import Accelerator
from accelerate.utils import tqdm
from loguru import logger
from torch.utils.data import DataLoader

from npnf.configs.neural_process import MaxHeightPredictionConfig
from npnf.scripts.utils.paths import (
    build_results_path,
    extract_model_info,
    get_dataloader_name_from_hydra,
    get_dataset_size_from_checkpoint,
    parse_run_name,
)
from npnf.scripts.utils.prediction import (
    build_context_priors_up_to_max_height,
    predict_batch_with_grid,
    prepare_batch_data,
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

METHOD_NAME = "max_height"


def process_dataloader(
    dataloader: DataLoader,
    split_name: str,
    base_path: Path,
    grid_points: tensordict.TensorDict,
    model,
    use_temperature: bool,
    use_marker: bool,
    num_batches: int | None = None,
    noise_z: torch.Tensor | None = None,
    config: dict | None = None,
    save_targets_std: bool = False,
    save_grid_std: bool = False,
    save_dtype: torch.dtype | None = torch.float16,
    save_batch_dtype: torch.dtype | None = torch.float16,
    batch_schema: str = "compact",
) -> None:
    """Predict using observations up to and including max height as context."""
    batch_schema = resolve_batch_schema(batch_schema)
    config = prepare_batch_schema_config(config, batch_schema, dataloader)
    for batch_index, batch in enumerate(
        tqdm(dataloader, desc=f"Testing {split_name}...", leave=False)
    ):
        temperatures = None
        if use_temperature and "temperature" in batch:
            temperatures = batch["temperature"]
        markers = None
        if use_marker and "marker" in batch:
            markers = batch["marker"]

        context_priors_batch, targets_batch, _, _ = prepare_batch_data(batch=batch)

        context_priors_batch, targets_batch = build_context_priors_up_to_max_height(
            context_priors_batch, targets_batch
        )

        predictions = predict_batch_with_grid(
            context_priors_batch=context_priors_batch,
            targets_batch=targets_batch,
            grid_points=grid_points,
            model=model,
            temperatures=temperatures,
            markers=markers,
            num_samples=64,
            noise_z=noise_z,
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
    num_batches: int | None = None,
    save_targets_std: bool = False,
    save_grid_std: bool = False,
    save_dtype: str | None = "float16",
    save_batch_dtype: str | None = "float16",
    batch_schema: str = "compact",
    grid: str = "synthetic",
) -> None:
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
        "num_samples": 64,
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
        Testing (max-height context_priors) with:
            - num_samples: {64}
            - use_temperature: {use_temperature}
            - use_marker: {use_marker}
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

    run_hydra_main(MaxHeightPredictionConfig, main)
