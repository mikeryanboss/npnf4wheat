from types import SimpleNamespace

import torch

from npnf.data.datasets.synthetic import SyntheticDataset


def test_synthetic_lodging_invariants():
    """
    Test that SyntheticDataset._apply_lodging_from_params enforces these invariants:
    1. A draw that does not lodge keeps its clean heights and an all-false mask.
    2. Zero transition-step lodging is treated deterministically as a lodging event.
    3. Height-drop implies has_lodged == True.
    """
    dataset = SimpleNamespace(enable_lodging=True)
    # Unique peak at index 5, so the lodging day is 5 + 1 + offset = 6.
    heights = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 4.0, 4.0, 4.0]])
    lodging_params = {
        "will_lodge": torch.tensor([[False, True]]),
        "offset": torch.tensor([[0, 0]]),
        "severity": torch.tensor([[0.5, 0.5]]),
        "transition_steps": torch.tensor([[0, 0]]),
    }

    modified_heights, has_lodged, lodged_mask = (
        SyntheticDataset._apply_lodging_from_params(  # noqa: SLF001
            dataset,  # ty: ignore[invalid-argument-type]
            heights,
            lodging_params,
        )
    )

    # Invariant 1: the draw that does not lodge is untouched.
    assert has_lodged[0].tolist() == [False, True]
    assert torch.equal(modified_heights[0, 0], heights[0])
    assert not lodged_mask[0, 0].any()

    # Invariant 2: with zero transition steps the drop to max * severity is
    # immediate after the lodging day, and the mask starts on the lodging day.
    assert lodged_mask[0, 1].tolist() == [False] * 6 + [True] * 4
    assert torch.equal(modified_heights[0, 1, :7], heights[0, :7])
    assert torch.equal(modified_heights[0, 1, 7:], torch.full((3,), 2.5))

    # Invariant 3: every draw whose heights dropped is flagged as lodged.
    dropped = (modified_heights[0] < heights[0]).any(dim=-1)
    assert torch.equal(dropped, dropped & has_lodged[0])
