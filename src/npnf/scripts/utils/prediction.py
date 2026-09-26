"""Shared utilities for prediction scripts."""

from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import tensordict
import torch
from accelerate import Accelerator
from tensordict import NestedKey
from tensordict.base import CompatibleType

from npnf.data.batch_loader import BatchLoader
from npnf.lodging import robust_max_height
from npnf.models.utils import consolidate_tensordict_jagged_dim
from npnf.scripts.utils.paths import save_config
from npnf.scripts.utils.utils import repeat_nested_tensordict

_SAVE_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}
_BATCH_METADATA_FIELDS = {
    "compact": ("yearsite_uid", "genotype_id"),
    "full": ("height", "yearsite_uid", "has_lodged", "genotype_id"),
    # FIP1 has no `has_lodged` column, and its replicate plots share
    # (genotype_id, yearsite_uid), so `plot_uid` identifies the condition. All three
    # are plain lists, keeping the saved batch flattenable; the ragged `height`
    # tensors are deliberately left out and read from the dataset when scoring.
    "fip1": ("yearsite_uid", "genotype_id", "plot_uid"),
}


def resolve_save_dtype(save_dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Resolve a user-facing save dtype value to a torch dtype."""
    if save_dtype is None or isinstance(save_dtype, torch.dtype):
        return save_dtype
    if save_dtype not in _SAVE_DTYPE_MAP:
        valid = sorted(_SAVE_DTYPE_MAP)
        msg = f"save dtype must be one of {valid} or None; got {save_dtype!r}"
        raise ValueError(msg)
    return _SAVE_DTYPE_MAP[save_dtype]


def resolve_batch_schema(batch_schema: str) -> str:
    """Validate the user-facing batch metadata schema name."""
    if batch_schema not in _BATCH_METADATA_FIELDS:
        valid = sorted(_BATCH_METADATA_FIELDS)
        msg = f"batch schema must be one of {valid}; got {batch_schema!r}"
        raise ValueError(msg)
    return batch_schema


def select_batch_metadata_for_schema(
    batch: tensordict.TensorDictBase, batch_schema: str
) -> tensordict.TensorDictBase:
    """Select the batch metadata saved by one-step prediction scripts."""
    resolved_schema = resolve_batch_schema(batch_schema)
    return batch.select(*_BATCH_METADATA_FIELDS[resolved_schema]).cpu()


def prepare_batch_schema_config(
    config: dict | None, batch_schema: str, dataloader
) -> dict:
    """Return method config with schema, dataset identity and frozen oracle metadata."""
    resolved_schema = resolve_batch_schema(batch_schema)
    prepared_config = dict(config or {})
    prepared_config["batch_schema"] = resolved_schema
    # Frozen test bundles: scorers resolve the oracle from this, with a hash check.
    dataset = getattr(dataloader, "dataset", None)
    reference = getattr(dataset, "frozen_reference", None)
    if reference is not None:
        prepared_config["oracle_dataset"] = reference

    if resolved_schema in {"compact", "fip1"}:
        prepared_config["dataset_identity"] = dataloader.dataset.get_dataset_identity()
    else:
        prepared_config.pop("dataset_identity", None)

    return prepared_config


def cast_floating_tensors(obj, dtype: torch.dtype | None):
    """Recursively cast floating tensors while preserving non-floating metadata."""
    if dtype is None:
        return obj

    if torch.is_tensor(obj):
        result = obj.to(dtype=dtype) if obj.is_floating_point() else obj
    elif isinstance(obj, tensordict.TensorDictBase):
        result = obj.clone(recurse=False)
        for key, value in obj.items():
            result[key] = cast_floating_tensors(value, dtype)
    elif isinstance(obj, dict):
        result = {
            key: cast_floating_tensors(value, dtype) for key, value in obj.items()
        }
    elif isinstance(obj, list):
        result = [cast_floating_tensors(value, dtype) for value in obj]
    elif isinstance(obj, tuple):
        result = tuple(cast_floating_tensors(value, dtype) for value in obj)
    else:
        result = obj
    return result


def predict_batch_with_grid(
    model,
    context_priors_batch: tensordict.TensorDict,
    targets_batch: tensordict.TensorDict,
    grid_points: tensordict.TensorDict,
    num_samples: int = 1,
    temperatures: torch.Tensor | None = None,
    markers: tensordict.TensorDict | None = None,
    noise_z: torch.Tensor | None = None,
    include_targets_std: bool = True,
    include_grid_std: bool = True,
    genotype_ids: list[str] | None = None,
    yearsite_ids: list[str] | None = None,
) -> tensordict.TensorDict:
    """Predict the grid and the targets of a batch.

    ``genotype_ids`` and ``yearsite_ids`` are passed on for models that look up
    fitted effects (the spline baseline); the neural processes ignore them.
    """
    targets_with_grid = tensordict.lazy_stack(
        [
            tensordict.TensorDict(
                {
                    k: torch.cat([grid, target[k]])  # ty: ignore[invalid-argument-type]
                    for k, grid in grid_points.items()
                },
                device=target.device,
            )
            for target in targets_batch
        ]
    ).densify(layout=torch.jagged)

    predictions = model(
        context_priors=context_priors_batch.clone(),
        context_posteriors=None,
        targets=targets_with_grid.clone(),
        temperatures=temperatures.clone() if temperatures is not None else None,
        markers=markers.clone() if markers is not None else None,
        num_samples=num_samples,
        noise_z=noise_z,
        get_samples=True,
        get_loss=False,
        sample_from_posterior=False,
        genotype_ids=genotype_ids,
        yearsite_ids=yearsite_ids,
    )["predictions"]
    target_predictions = predictions["target"]
    target_scales = predictions["target_scale"]

    targets_lengths = [len(target["X"]) for target in targets_batch]
    num_grid_points = len(grid_points)

    targets_tensor = torch.nested.as_nested_tensor(
        [
            pred[:, num_grid_points : num_grid_points + length]
            for pred, length in zip(target_predictions, targets_lengths, strict=True)
        ],
        layout=torch.jagged,
    )
    grid_tensor = torch.stack(
        [pred[:, :num_grid_points] for pred in target_predictions]
    )

    result: dict[NestedKey, CompatibleType] = {
        "targets": targets_tensor,
        "grid": grid_tensor,
    }

    if include_targets_std:
        result["targets_std"] = torch.nested.as_nested_tensor(
            [
                scale[:, num_grid_points : num_grid_points + length]
                for scale, length in zip(target_scales, targets_lengths, strict=True)
            ],
            layout=torch.jagged,
        )
    if include_grid_std:
        result["grid_std"] = torch.stack(
            [scale[:, :num_grid_points] for scale in target_scales]
        )

    return tensordict.TensorDict(
        result, batch_size=targets_batch.batch_size, device=targets_batch.device
    )


def predict_batch(
    model,
    context_priors_batch: tensordict.TensorDict,
    targets_batch: tensordict.TensorDict,
    num_samples: int = 1,
    temperatures: torch.Tensor | None = None,
    markers: tensordict.TensorDict | None = None,
    noise_z: torch.Tensor | None = None,
) -> tensordict.TensorDict:
    """Predict on targets without grid points."""
    predictions = model(
        context_priors=context_priors_batch.clone(),
        context_posteriors=None,
        targets=targets_batch.clone(),
        temperatures=temperatures.clone() if temperatures is not None else None,
        markers=markers.clone() if markers is not None else None,
        num_samples=num_samples,
        noise_z=noise_z,
        get_samples=True,
        get_loss=False,
        sample_from_posterior=False,
    )["predictions"]

    return tensordict.TensorDict(
        {"targets": predictions["target"], "targets_std": predictions["target_scale"]},
        batch_size=targets_batch.batch_size,
        device=targets_batch.device,
    )


def get_next_sets_by_index(
    context_priors: list[tensordict.TensorDict],
    ground_truth: tensordict.TensorDict,
    B: int,
    context_in_prior_masks: list[torch.Tensor],
    indices: Sequence[int | None],
) -> list[tensordict.TensorDict]:
    """Update context_prior sets by adding ground_truth points at indices.

    Args:
        context_priors: List of context prior TensorDicts (one per prediction).
        ground_truth: Shared ground truth TensorDict (original batch, not duplicated).
        B: Original batch size (before duplication for methods).
        context_in_prior_masks: List of masks tracking which points are in context.
        indices: List of indices to add to context for each prediction.

    Returns:
        Updated list of context prior TensorDicts.
    """
    ground_truth_list = list(ground_truth)
    for i in range(len(context_priors)):
        context_prior = context_priors[i]
        context_in_prior_mask = context_in_prior_masks[i]
        context_prior.batch_size = context_prior["Y"].shape[:2]  # ty: ignore[invalid-assignment]
        if not context_in_prior_mask.all():
            index = indices[i]
            context_in_prior_mask[index] = True
            # Use ground_truth[i % B] to get the original sample
            gt_sample = ground_truth_list[i % B]
            gt_sample.batch_size = gt_sample["Y"].shape[:2]
            context_prior = tensordict.cat(
                [context_prior, gt_sample[index][None]], dim=0
            )
        context_priors[i] = context_prior

    return context_priors


def get_indices_sequential(targets: list[tensordict.TensorDict], index) -> list[int]:
    """Get sequential indices for all targets."""
    return [index] * len(targets)


def get_indices_uncertainty(
    predictions: list[tensordict.TensorDict],
    target_in_context_masks: list[torch.Tensor],
) -> list[int | None]:
    """Get indices based on prediction uncertainty (highest std)."""
    indices: list[int | None] = []
    for prediction, target_in_context_mask in zip(
        predictions, target_in_context_masks, strict=True
    ):
        prediction = prediction["targets"]
        if prediction.ndim == 4:
            num_targets = prediction.shape[2]
            prediction = prediction.reshape(
                prediction.shape[0] * prediction.shape[1],
                num_targets,
                prediction.shape[3],
            )
        else:
            num_targets = prediction.shape[1]

        if num_targets == 0:
            indices.append(None)
            continue

        prediction_std = prediction.std(dim=0)
        if prediction_std.ndim > 1:
            prediction_std = prediction_std.squeeze(-1)
        prediction_std[target_in_context_mask] = -torch.inf
        index = int(prediction_std.argmax().item())  # ty: ignore[unresolved-attribute]
        indices.append(index)

    return indices


def get_indices_random(targets_in_context_masks: list[torch.Tensor]) -> list[int]:
    """Get random indices for targets not in context."""
    indices = []
    for target_in_context_mask in targets_in_context_masks:
        noise = torch.rand(
            len(target_in_context_mask), device=target_in_context_mask.device
        )
        noise[target_in_context_mask] = -torch.inf
        index = int(noise.argmax().item())
        indices.append(index)

    return indices


def save_predictions(
    pred_dict: dict,
    batch_index: int,
    start: int,
    end: int,
    context_priors: tensordict.TensorDict | None = None,
    predictions: torch.Tensor | None = None,
    context_posteriors: tensordict.TensorDict | None = None,
    context_prior_indices: list[int] | None = None,
) -> dict:
    """Save predictions to the prediction dictionary.

    Args:
        pred_dict: Dictionary to save predictions to.
        batch_index: Current batch index.
        start: Start index for slicing.
        end: End index for slicing.
        context_priors: Context prior tensors.
        predictions: Prediction tensors.
        context_posteriors: Context posterior tensors.
        context_prior_indices: Indices that will be added to context_prior.
            These are the indices selected at the current step, representing
            which points will become context_prior in the next iteration.
    """
    if context_priors is not None:
        pred_dict["context_priors"][batch_index].append(
            tensordict.lazy_stack(list(context_priors)[start:end])
            .detach()
            .clone()
            .cpu()
            .densify(layout=torch.jagged)
        )
    if context_posteriors is not None:
        pred_dict["context_posteriors"][batch_index].append(
            tensordict.lazy_stack(list(context_posteriors)[start:end])
            .detach()
            .clone()
            .cpu()
            .densify(layout=torch.jagged)
        )
    if predictions is not None:
        pred_dict["predictions"][batch_index].append(
            tensordict.lazy_stack(list(predictions)[start:end])
            .detach()
            .clone()
            .cpu()
            .densify(layout=torch.jagged)
        )
    if context_prior_indices is not None and "context_prior_indices" in pred_dict:
        pred_dict["context_prior_indices"][batch_index].append(context_prior_indices)
    return pred_dict


def get_prediction_method_indices(
    method_name: str,
    targets: list[tensordict.TensorDict],
    predictions: list[tensordict.TensorDict] | None,
    target_in_context_masks: list[torch.Tensor],
    index: int,
) -> Sequence[int | None]:
    """Get indices for the specified prediction method."""
    if method_name == "sequential":
        return get_indices_sequential(targets, index)
    if method_name == "uncertainty":
        if predictions is None:
            msg = "Uncertainty prediction method requires predictions"
            raise ValueError(msg)
        return get_indices_uncertainty(predictions, target_in_context_masks)
    if method_name == "random":
        return get_indices_random(target_in_context_masks)

    error_msg = f"Unknown prediction method: {method_name}"
    raise ValueError(error_msg)


def consolidate_nested_tensors(
    context_priors_batch: tensordict.TensorDict,
    context_posteriors_batch: tensordict.TensorDict,
) -> tuple[tensordict.TensorDict, tensordict.TensorDict]:
    """Consolidate nested tensors if needed."""
    if context_priors_batch["Y"].is_nested:  # ty: ignore[unresolved-attribute]
        context_priors_batch = consolidate_tensordict_jagged_dim(context_priors_batch)
    if context_posteriors_batch["Y"].is_nested:  # ty: ignore[unresolved-attribute]
        context_posteriors_batch = consolidate_tensordict_jagged_dim(
            context_posteriors_batch
        )
    return context_priors_batch, context_posteriors_batch


def build_single_draw_height_view(
    height: tensordict.TensorDict, draw_index: int = 0
) -> tensordict.TensorDict:
    """Project multi-draw synthetic height metadata to one realized draw.

    Synthetic test batches keep the observed trajectory in `Y` but may retain an
    extra lodging-draw axis for auxiliary fields like `lodged_mask` and
    `Y_original`. Prediction-time helper code expects all `height` fields to
    align with `Y`, so project those auxiliary fields to the same realized draw.
    """

    y = height["Y"]
    projected: dict[NestedKey, CompatibleType] = {}
    for key, value in height.items():
        projected_value = value
        if (
            key in {"lodged_mask", "Y_original"}
            and not value.is_nested  # ty: ignore[unresolved-attribute]
            and value.ndim == y.ndim + 1
            and value.shape[0] == y.shape[0]
            and value.shape[2:] == y.shape[1:]
        ):
            projected_value = value[:, draw_index]
        projected[key] = projected_value

    return tensordict.TensorDict(
        projected, batch_size=height.batch_size, device=height.device
    )


def prepare_batch_data(
    batch: tensordict.TensorDict, num_predictions: int = 1
) -> tuple[tensordict.TensorDict, tensordict.TensorDict, torch.Tensor, int]:
    """Prepare batch data for prediction."""
    YX = build_single_draw_height_view(batch["height"])  # ty: ignore[invalid-argument-type]

    context_priors_batch = tensordict.TensorDict.from_dict(
        {
            k: torch.empty((*batch.batch_size, 0, 1), device=v.device, dtype=v.dtype)
            for k, v in YX.items()
        },
        batch_size=batch.batch_size,
        device=batch.device,
    )
    targets_batch = YX.clone()
    max_num_targets = (
        targets_batch["Y"]._max_seqlen  # noqa: SLF001  # ty: ignore[unresolved-attribute]
        if targets_batch["Y"].is_nested  # ty: ignore[unresolved-attribute]
        else targets_batch["Y"].shape[1]
    )
    targets_batch.batch_size = targets_batch.batch_size[:1]  # ty: ignore[invalid-assignment]

    context_priors_batch = context_priors_batch.repeat(num_predictions)  # ty: ignore[invalid-argument-type]
    targets_batch = (
        repeat_nested_tensordict(targets_batch, num_predictions)
        if targets_batch["Y"].is_nested  # ty: ignore[unresolved-attribute]
        else targets_batch.repeat(num_predictions)  # ty: ignore[invalid-argument-type]
    )

    targets_in_context_prior_mask_batch = torch.zeros_like(
        targets_batch["Y"][..., 0], dtype=torch.bool
    )

    return (
        context_priors_batch,
        targets_batch,
        targets_in_context_prior_mask_batch,
        max_num_targets,
    )


def calculate_method_starts(
    B: int, predict_sequential: bool, predict_uncertainty: bool
) -> tuple[int, int, int]:
    """Calculate starting indices for different prediction methods."""
    sequential_start = 0
    uncertainty_start = B * predict_sequential
    random_start = B * predict_uncertainty + B * predict_sequential
    return sequential_start, uncertainty_start, random_start


def update_context_priors_and_samples(
    context_priors_batch_list: list[tensordict.TensorDict],
    ground_truth: tensordict.TensorDict,
    B: int,
    context_in_prior_mask_list: list[torch.Tensor],
    next_context_prior_indices: Sequence[int | None],
) -> tensordict.TensorDict:
    """Update context_priors based on indices, using shared ground_truth.

    Args:
        context_priors_batch_list: List of context prior TensorDicts.
        ground_truth: Shared ground truth TensorDict (original batch, not duplicated).
        B: Original batch size (before duplication for methods).
        context_in_prior_mask_list: List of masks tracking which points are in context.
        next_context_prior_indices: Indices to add to context for each prediction.

    Returns:
        Updated context_priors_batch TensorDict.
    """
    context_priors_batch_list = get_next_sets_by_index(
        context_priors=context_priors_batch_list,
        ground_truth=ground_truth,
        B=B,
        context_in_prior_masks=context_in_prior_mask_list,
        indices=next_context_prior_indices,
    )

    context_priors_batch = tensordict.lazy_stack(
        context_priors_batch_list, dim=0
    ).densify(layout=torch.jagged)
    context_priors_batch.batch_size = context_priors_batch.batch_size[:1]

    if context_priors_batch["Y"].is_nested:
        context_priors_batch = consolidate_tensordict_jagged_dim(context_priors_batch)

    return context_priors_batch


def peak_height_index(heights: tensordict.TensorDict) -> int:
    """Index of the peak height of one plot.

    Synthetic samples carry the noise-free ``Y_original``, so the peak is exact and
    this keeps the existing behaviour bit-identical. Real FIP1 plots have no clean
    curve, so the peak is taken from the observed ``Y`` with the project's robust
    FIP1 estimator (median of the top three heights, ``npnf.lodging``), which is the
    same rule the lodging detector uses on real data.
    """
    if "Y_original" in heights:
        clean: torch.Tensor = heights["Y_original"][..., 0]  # ty: ignore[invalid-assignment]
        return int(clean.argmax().item())
    values: torch.Tensor = heights["Y"][..., 0]  # ty: ignore[invalid-assignment]
    _, peak_index = robust_max_height(values)
    return int(peak_index)


def build_context_priors_up_to_max_height(
    context_priors_batch: tensordict.TensorDict, targets_batch: tensordict.TensorDict
) -> tuple[tensordict.TensorDict, tensordict.TensorDict]:
    """Fill context_priors with observations up to and including max height."""

    context_priors_to_max_height = []
    for targets in targets_batch:
        targets.batch_size = [len(targets["X"])]
        max_height_index = peak_height_index(targets)
        context_priors_to_max_height.append(targets[: (max_height_index + 1)])

    targets_batch = context_priors_batch.clone()
    context_priors_batch = tensordict.lazy_stack(
        context_priors_to_max_height, dim=0
    ).densify(layout=torch.jagged)
    context_priors_batch.batch_size = context_priors_batch.batch_size[:1]

    if context_priors_batch["Y"].is_nested:
        context_priors_batch = consolidate_tensordict_jagged_dim(context_priors_batch)

    return context_priors_batch, targets_batch


def build_context_priors_first_half(
    context_priors_batch: tensordict.TensorDict, targets_batch: tensordict.TensorDict
) -> tuple[tensordict.TensorDict, tensordict.TensorDict]:
    """Split observations: first half as context_prior, second half as targets."""
    context_priors_half = []
    targets_half = []

    for targets in targets_batch:
        targets.batch_size = [len(targets["X"])]
        n = len(targets["X"])
        midpoint = n // 2
        context_priors_half.append(targets[:midpoint])
        targets_half.append(targets[midpoint:])

    context_priors_batch = tensordict.lazy_stack(context_priors_half, dim=0).densify(
        layout=torch.jagged
    )
    context_priors_batch.batch_size = context_priors_batch.batch_size[:1]

    targets_batch = tensordict.lazy_stack(targets_half, dim=0).densify(
        layout=torch.jagged
    )
    targets_batch.batch_size = targets_batch.batch_size[:1]

    if context_priors_batch["Y"].is_nested:
        context_priors_batch = consolidate_tensordict_jagged_dim(context_priors_batch)
    if targets_batch["Y"].is_nested:
        targets_batch = consolidate_tensordict_jagged_dim(targets_batch)

    return context_priors_batch, targets_batch


def load_grid_points(method_dir: Path) -> tensordict.TensorDict:
    """``grid_points`` of batch 0 of a saved prediction directory."""
    batch = BatchLoader(method_dir, load_predictions=False).load_batch(0)
    if not isinstance(batch, dict):
        msg = f"Expected batch metadata only from {method_dir}"
        raise TypeError(msg)
    return batch["grid_points"]


def save_batch_results(
    batch_index: int,
    method_name: str,
    base_path: Path,
    predictions: tensordict.TensorDict,
    batch_data: tensordict.TensorDictBase | None = None,
    grid_points: tensordict.TensorDict | None = None,
    max_valid_context: torch.Tensor | None = None,
    context_indices: list | None = None,
    config: dict | None = None,
    save_dtype: torch.dtype | None = None,
    save_batch_dtype: torch.dtype | None = None,
) -> None:
    """Save results for a single batch to disk.

    Args:
        batch_index: Index of the batch being saved.
        method_name: Name of the prediction method.
        base_path: Base path for results.
        predictions: Prediction tensors for this batch.
        batch_data: Batch data (contexts, targets) for this batch.
        grid_points: Grid points used for prediction.
        max_valid_context: Maximum valid context indices.
        context_indices: Indices selected at each step.
        config: Configuration to save (only saved for batch 0).
        save_dtype: Optional dtype to convert predictions to before saving.
        save_batch_dtype: Optional dtype for floating tensors in batch metadata.
    """
    batch_dir = base_path / method_name / "predictions" / f"{batch_index:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred = predictions.detach().cpu()
    if save_dtype is not None:
        pred = pred.to(dtype=save_dtype)
    torch.save(pred, batch_dir / "predictions.pt")

    # Save batch data
    batch_dict = {}
    if batch_data is not None:
        batch_dict["data"] = batch_data.detach().cpu()
    if grid_points is not None:
        batch_dict["grid_points"] = grid_points.detach().cpu()
    if max_valid_context is not None:
        batch_dict["max_valid_context"] = max_valid_context.detach().cpu()
    if context_indices is not None:
        batch_dict["context_indices"] = context_indices

    if batch_dict:
        batch_dict = cast_floating_tensors(batch_dict, save_batch_dtype)
        torch.save(batch_dict, batch_dir / "batch.pt")

    # Save config once at method level
    if config is not None and batch_index == 0:
        save_config(base_path / method_name, **config)


def get_empty_observations(
    batch_size: int, device: torch.device | None, dtype: torch.dtype
) -> tensordict.TensorDict:
    """Return an empty observation TensorDict used for contexts and targets."""
    empty = torch.empty((batch_size, 0, 1), device=device, dtype=dtype)
    return tensordict.TensorDict(
        {"X": empty.clone(), "X_normalized": empty.clone(), "Y": empty.clone()},
        batch_size=[batch_size],
        device=device,
    )


def unnest_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Extract values from nested tensor if nested, otherwise return as-is."""
    return tensor.values() if tensor.is_nested else tensor


def get_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Find the checkpoint with the highest step number."""
    checkpoints = [c for c in checkpoints_dir.glob("checkpoint-*") if c.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda c: int(c.name.split("-")[-1]))


def resolve_checkpoint(checkpoint_path: Path) -> Path:
    """Resolve checkpoint path, auto-detecting latest if directory given.

    Returns an absolute resolved path.
    """
    if checkpoint_path.name.startswith("checkpoint-"):
        return checkpoint_path.resolve()
    latest = get_latest_checkpoint(checkpoint_path)
    if latest is None:
        msg = f"No checkpoints found in {checkpoint_path}"
        raise ValueError(msg)
    return latest.resolve()


def resolve_model_from_checkpoint(checkpoint_folder: str) -> torch.nn.Module:
    """Instantiate the model class encoded in a checkpoint run directory.

    Prefers the actual training config saved alongside the checkpoint
    (`train_config.yaml` in the project directory). This ensures models
    trained with non-default parameters (e.g., larger `latent_dim`) are
    reconstructed correctly.

    Falls back to longest-prefix matching against `model_store` entries
    when `train_config.yaml` is absent (legacy checkpoints).
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from npnf.models.configs.models import ANP_Config  # noqa: F401
    from npnf.models.configs.utils import model_store
    from npnf.scripts.utils.paths import extract_model_info

    resolved = str(resolve_checkpoint(Path(checkpoint_folder)))

    # Prefer actual training config when available
    project_dir = Path(resolved).parent.parent
    train_config_path = project_dir / "train_config.yaml"
    if train_config_path.exists():
        train_cfg = OmegaConf.load(train_config_path)
        if "model" in train_cfg:
            return instantiate(train_cfg.model)

    # Fall back to model store matching for legacy checkpoints
    run_name, _ = extract_model_info(resolved)

    entries = {e["name"]: e["node"] for e in model_store if e["group"] == "model"}
    matches = [
        name for name in entries if run_name == name or run_name.startswith(name + "-")
    ]
    if not matches:
        registered = sorted(entries)
        msg = (
            f"No registered model matches run name {run_name!r}. "
            f"Registered names: {registered}"
        )
        raise ValueError(msg)

    best = max(matches, key=len)
    return instantiate(entries[best])


def dataloaders_to_shallow_dict(dataloaders) -> dict:
    """Return top-level dataloader fields without copying DataLoader objects."""
    return {
        field.name: getattr(dataloaders, field.name) for field in fields(dataloaders)
    }


def prepare_model_and_dataloaders(
    model, dataloaders, checkpoint_folder: str, accelerator: Accelerator
) -> tuple:
    """Load checkpoint and prepare model/dataloaders with accelerator.

    Args:
        model: The model to load checkpoint into.
        dataloaders: Dataclass containing dataloaders to prepare.
        checkpoint_folder: Path to the checkpoint folder or checkpoints directory.
        accelerator: Accelerator instance.

    Returns:
        Tuple of (model, dataloaders_dict, resolved_checkpoint_folder).

    Note:
        We use manual safetensors loading instead of load_checkpoint_and_dispatch
        with device_map='auto' because the combination of device_map='auto' and
        accelerator.prepare() causes weights to be zeroed out in accelerate 1.11.0.
    """
    from safetensors.torch import load_file

    resolved_checkpoint_folder = str(resolve_checkpoint(Path(checkpoint_folder)))
    checkpoint_path = Path(resolved_checkpoint_folder) / "model.safetensors"
    state_dict = load_file(str(checkpoint_path))
    model.load_state_dict(state_dict)
    model = accelerator.prepare(model)

    dataloaders_dict = dataloaders_to_shallow_dict(dataloaders)
    dataloaders_prepared = accelerator.prepare(*dataloaders_dict.values())
    if len(dataloaders_dict) == 1:
        dataloaders_prepared = [dataloaders_prepared]
    dataloaders_dict = dict(
        zip(list(dataloaders_dict.keys()), dataloaders_prepared, strict=True)
    )

    model.eval()
    return model, dataloaders_dict, resolved_checkpoint_folder
