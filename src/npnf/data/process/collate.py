from typing import Any

import numpy as np
import tensordict
import torch


def _process_value_for_collate(v):
    """Process values for collate function, handling different data types."""
    if v is None:
        return torch.empty((18845, 0), dtype=torch.long)
    if isinstance(v, str | bytes):
        return v
    if isinstance(v, np.ndarray) and v.dtype.kind in ["U", "S", "O"]:
        return v
    try:
        return v[..., None]
    except (TypeError, IndexError):
        return v


def collate_fn_fip1_heights(
    batch: list[dict[str, Any]], transforms=None
) -> tensordict.TensorDict:
    metadata_keys = [
        "plot_uid",
        "harvest_year",
        "yearsite_uid",
        "genotype_id",
        "has_lodged",
    ]
    metadata = {
        key: [sample.get(key) for sample in batch]
        for key in metadata_keys
        if any(key in sample for sample in batch)
    }

    batch_stacked = (
        tensordict.lazy_stack(
            [
                tensordict.TensorDict.from_dict(
                    {k: _process_value_for_collate(v) for k, v in sample.items()}
                )
                for sample in batch
            ]
        )
        .densify(layout=torch.jagged)
        .float()
    )

    batch_stacked["height"] = batch_stacked.select(
        "height_values", "height_days", "height_days_normalized"
    )
    batch_stacked["height"].rename_key_("height_values", "Y")
    batch_stacked["height"].rename_key_("height_days", "X")
    batch_stacked["height"].rename_key_("height_days_normalized", "X_normalized")
    if "height_lodged_mask" in batch_stacked:
        batch_stacked["height"]["lodged_mask"] = batch_stacked["height_lodged_mask"].to(
            dtype=torch.bool
        )
    if "height_values_nonoise" in batch_stacked:
        batch_stacked["height"]["Y_original"] = batch_stacked["height_values_nonoise"]
    if batch_stacked["height"]["Y"].is_nested:
        offsets = batch_stacked["height"]["Y"].offsets()
        for k, v in batch_stacked["height"].items():
            batch_stacked["height"][k] = torch.nested.nested_tensor_from_jagged(
                values=v.values(),
                offsets=offsets,
                min_seqlen=v._min_seqlen,  # noqa: SLF001
                max_seqlen=v._max_seqlen,  # noqa: SLF001
            )

    temperature = batch_stacked.select("temperature_values").densify(
        layout=torch.jagged
    )
    temperature.rename_key_("temperature_values", "Y")
    temperature.batch_size = temperature["Y"].shape
    temperature = temperature.squeeze(-1)
    temperature = temperature.flatten(1, 2)
    temperature = temperature.unsqueeze(1)
    temperature.batch_size = temperature.batch_size[:2]
    batch_stacked["temperature"] = temperature

    if "marker_biallelic_codes" in batch_stacked:
        marker = batch_stacked.select("marker_biallelic_codes").densify(
            layout=torch.jagged
        )
        marker.rename_key_("marker_biallelic_codes", "Y")
        marker["Y"] = marker["Y"].transpose(-1, -2)
        marker.batch_size = marker["Y"].shape[:1]
        batch_stacked["marker"] = marker
    else:
        batch_stacked["marker"] = tensordict.TensorDict.from_dict(
            {"Y": torch.empty((*batch_stacked.batch_size, 0, 18845), dtype=torch.long)},
            batch_size=batch_stacked.batch_size,
        )

    result = batch_stacked.select("height", "temperature", "marker")
    for key, value in metadata.items():
        result[key] = value
    return result
