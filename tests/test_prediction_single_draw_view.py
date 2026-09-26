import tensordict
import torch

from npnf.scripts.utils.prediction import (
    build_context_priors_first_half,
    build_context_priors_up_to_max_height,
    build_single_draw_height_view,
    prepare_batch_data,
    update_context_priors_and_samples,
)
from tests.utils import tensor_at, tensordict_at


def _make_batch() -> tensordict.TensorDict:
    batch_size = 2
    seq_len = 6
    num_draws = 4

    y = torch.arange(batch_size * seq_len, dtype=torch.float32).reshape(
        batch_size, seq_len, 1
    )
    x = (
        torch.arange(seq_len, dtype=torch.float32)
        .reshape(1, seq_len, 1)
        .expand(batch_size, -1, -1)
    )
    x_normalized = x / 10.0

    lodged_mask = torch.zeros((batch_size, num_draws, seq_len, 1), dtype=torch.bool)
    lodged_mask[0, 0, 3:, 0] = True
    lodged_mask[1, 0, :, 0] = False
    lodged_mask[0, 1, 1:, 0] = True

    y_original = torch.tensor(
        [
            [
                [[0.0], [2.0], [5.0], [4.0], [3.0], [1.0]],
                [[0.0], [4.0], [3.0], [2.0], [1.0], [0.0]],
                [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
                [[2.0], [2.0], [2.0], [2.0], [2.0], [2.0]],
            ],
            [
                [[0.0], [1.0], [3.0], [6.0], [4.0], [2.0]],
                [[0.0], [6.0], [3.0], [2.0], [1.0], [0.0]],
                [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
                [[2.0], [2.0], [2.0], [2.0], [2.0], [2.0]],
            ],
        ],
        dtype=torch.float32,
    )

    height = tensordict.TensorDict(
        {
            "Y": y,
            "X": x,
            "X_normalized": x_normalized,
            "lodged_mask": lodged_mask,
            "Y_original": y_original,
        },
        batch_size=[batch_size],
    )
    return tensordict.TensorDict({"height": height}, batch_size=[batch_size])


def test_build_single_draw_height_view_projects_only_multi_draw_keys() -> None:
    batch = _make_batch()

    projected = build_single_draw_height_view(tensordict_at(batch, "height"))

    assert torch.equal(tensor_at(projected, "Y"), tensor_at(batch, ("height", "Y")))
    assert torch.equal(tensor_at(projected, "X"), tensor_at(batch, ("height", "X")))
    assert torch.equal(
        tensor_at(projected, "X_normalized"),
        tensor_at(batch, ("height", "X_normalized")),
    )
    assert torch.equal(
        tensor_at(projected, "lodged_mask"),
        tensor_at(batch, ("height", "lodged_mask"))[:, 0],
    )
    assert torch.equal(
        tensor_at(projected, "Y_original"),
        tensor_at(batch, ("height", "Y_original"))[:, 0],
    )


def test_prediction_helpers_accept_single_draw_projected_height_view() -> None:
    batch = _make_batch()

    context_priors_batch, targets_batch, mask_batch, _ = prepare_batch_data(batch)
    original_targets = list(targets_batch.clone())

    sample = next(iter(targets_batch))
    assert sample["lodged_mask"].shape == sample["Y"].shape
    assert sample["Y_original"].shape == sample["Y"].shape

    half_context_batch, half_targets_batch = build_context_priors_first_half(
        context_priors_batch.clone(), targets_batch.clone()
    )
    for half_context, half_targets, original in zip(
        half_context_batch, half_targets_batch, original_targets, strict=True
    ):
        for key in ("Y", "lodged_mask", "Y_original"):
            assert torch.equal(half_context[key], original[key][:3])
            assert torch.equal(half_targets[key], original[key][3:])

    context_priors_batch, empty_targets_batch = build_context_priors_up_to_max_height(
        context_priors_batch.clone(), targets_batch.clone()
    )

    context_samples = list(context_priors_batch)
    empty_target_samples = list(empty_targets_batch)
    assert len(context_samples[0]["Y"]) == 3
    assert len(context_samples[1]["Y"]) == 4
    assert len(empty_target_samples[0]["Y"]) == 0
    assert len(empty_target_samples[1]["Y"]) == 0
    assert torch.equal(context_samples[0]["Y"], original_targets[0]["Y"][:3])
    assert torch.equal(context_samples[1]["Y"], original_targets[1]["Y"][:4])

    updated = update_context_priors_and_samples(
        context_priors_batch_list=list(context_priors_batch),
        ground_truth=build_single_draw_height_view(tensordict_at(batch, "height")),
        B=batch.batch_size[0],
        context_in_prior_mask_list=list(mask_batch),
        next_context_prior_indices=[0, 1],
    )

    updated_samples = list(updated)
    assert torch.equal(
        updated_samples[0]["Y"],
        torch.cat([context_samples[0]["Y"], original_targets[0]["Y"][0:1]]),
    )
    assert torch.equal(
        updated_samples[1]["Y"],
        torch.cat([context_samples[1]["Y"], original_targets[1]["Y"][1:2]]),
    )
    assert [mask.nonzero().flatten().tolist() for mask in mask_batch] == [[0], [1]]
