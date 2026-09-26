from typing import cast

import tensordict
import torch

from npnf.models.neural_process.utils import consolidate_tensordict_jagged_dim

_SYNTHETIC_FULL_DAY_START = 61
_SYNTHETIC_FULL_DAY_END = 364
_SYNTHETIC_GROWING_START_INDEX = 139
_SYNTHETIC_GROWING_END_INDEX = 261
_SYNTHETIC_NUM_PRE_SEASON_POINTS = 10
_SYNTHETIC_NUM_POST_SEASON_POINTS = 10
_SYNTHETIC_PREDICTION_NUM_POINTS = 32


def _synthetic_eval_day_axis(device: torch.device | None) -> torch.Tensor:
    """Return the fixed synthetic eval-mode day axis.

    Mirrors `SyntheticDataset._compute_subsample_indices(..., eval_mode=True)`:
    sparse pre/post-season points with every growing-season day retained.
    """
    full_days = torch.arange(
        _SYNTHETIC_FULL_DAY_START,
        _SYNTHETIC_FULL_DAY_END + 1,
        device=device,
        dtype=torch.float32,
    )
    pre_positions = torch.linspace(
        0,
        _SYNTHETIC_GROWING_START_INDEX - 1,
        _SYNTHETIC_NUM_PRE_SEASON_POINTS,
        device=device,
    ).long()
    growth_positions = torch.arange(
        _SYNTHETIC_GROWING_START_INDEX, _SYNTHETIC_GROWING_END_INDEX, device=device
    )
    post_positions = (
        _SYNTHETIC_GROWING_END_INDEX
        + torch.linspace(
            0,
            full_days.numel() - _SYNTHETIC_GROWING_END_INDEX - 1,
            _SYNTHETIC_NUM_POST_SEASON_POINTS,
            device=device,
        ).long()
    )
    selected = torch.cat([pre_positions, growth_positions, post_positions])
    return full_days[selected]


def _subset_synthetic_eval_day_axis(
    device: torch.device | None, num_points: int
) -> torch.Tensor:
    """Pick a deterministic subset of the synthetic eval day axis."""
    eval_days = _synthetic_eval_day_axis(device)
    positions = (
        torch.linspace(0, eval_days.numel() - 1, num_points, device=device)
        .round()
        .long()
    )
    positions = torch.unique(positions, sorted=True)
    if positions.numel() != num_points:
        msg = f"Could not choose {num_points} unique eval-day positions"
        raise ValueError(msg)
    return eval_days.index_select(0, positions)


def _synthetic_prediction_day_axis(device: torch.device | None) -> torch.Tensor:
    """Return the canonical synthetic prediction grid."""
    return _subset_synthetic_eval_day_axis(device, _SYNTHETIC_PREDICTION_NUM_POINTS)


FIP1_PREDICTION_DAY_START, FIP1_PREDICTION_DAY_END = 61, 365


def _fip1_prediction_day_axis(device: torch.device | None) -> torch.Tensor:
    """Return the daily FIP1 prediction grid.

    The synthetic grid is a 32-point subset of the synthetic eval day axis, which
    retains every growing-season day, so those 32 days are real observation days
    there. FIP1 measurement dates do not line up with them: only 7 of the 32 fall
    on a FIP1 observation day. Predicting daily instead lets a scorer sample each
    unit's actual shared observation days, which is what
    ``metrics/sig_mmd_fip1_context_matched_blocked.py`` does with its aligned grid.
    """
    return torch.arange(
        FIP1_PREDICTION_DAY_START,
        FIP1_PREDICTION_DAY_END,
        device=device,
        dtype=torch.float32,
    )


def create_grid_points(
    device: torch.device | None, grid: str = "synthetic"
) -> tensordict.TensorDict:
    if grid == "synthetic":
        days = _synthetic_prediction_day_axis(device)[..., None]
    elif grid == "fip1":
        days = _fip1_prediction_day_axis(device)[..., None]
    else:
        msg = f"grid must be one of ('synthetic', 'fip1'); got {grid!r}"
        raise ValueError(msg)
    days_normalized = (days - 212.5) / 151.5
    return tensordict.TensorDict(
        {"X": days, "X_normalized": days_normalized},
        device=device,
        batch_size=(days.shape[0],),
    )


def repeat_nested_tensor(nt: torch.Tensor, num_times: int) -> torch.Tensor:
    values = nt.values().repeat(num_times, *nt.values().shape[1:])
    offsets = nt.offsets()  # ty: ignore[unresolved-attribute]
    lengths = (offsets[1:] - offsets[:-1]).repeat(num_times)
    return torch.nested.nested_tensor_from_jagged(
        values=values,
        lengths=lengths,
        min_seqlen=nt._min_seqlen,  # noqa: SLF001  # ty: ignore[unresolved-attribute]
        max_seqlen=nt._max_seqlen,  # noqa: SLF001  # ty: ignore[unresolved-attribute]
    )


def repeat_nested_tensordict(td: tensordict.TensorDict, num_times: int):
    td = tensordict.TensorDict(
        {
            k: repeat_nested_tensor(cast(torch.Tensor, v), num_times)
            for k, v in td.items()
        },
        batch_size=[td.batch_size[0] * num_times, *td.batch_size[1:]],
        device=td.device,
    )
    return consolidate_tensordict_jagged_dim(td)
