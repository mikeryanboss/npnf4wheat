from __future__ import annotations

from collections.abc import Iterator

import pytest
import tensordict
import torch

from npnf.models.neural_process.utils import SetMode, get_sets

VALID_MODES = (
    "nested-all",
    "nested-noprior",
    "disjoint-noprior",
    "nested-holdout",
    "disjoint-holdout",
    "nested-nested",
    "disjoint-nested",
    "nested-noprior-holdout",
)


def _height_batch(num_points: int = 10) -> tensordict.TensorDict:
    x = torch.arange(num_points, dtype=torch.float32).reshape(1, num_points, 1)
    return tensordict.TensorDict({"X": x, "Y": x + 10.0}, batch_size=[1])


def _patch_randint(monkeypatch: pytest.MonkeyPatch, values: list[int]) -> None:
    randint_values: Iterator[int] = iter(values)

    def fake_randint(low, high, size, *, generator=None):
        del generator
        value = next(randint_values)
        low_value = int(low.item()) if torch.is_tensor(low) else int(low)
        high_value = int(high.item()) if torch.is_tensor(high) else int(high)
        assert size == (1,)
        assert low_value <= value < high_value
        return torch.tensor([value])

    monkeypatch.setattr(torch, "randint", fake_randint)


def _make_sets(
    monkeypatch: pytest.MonkeyPatch,
    set_mode: SetMode,
    *,
    P: int = 5,
    p: int = 2,
    num_points: int = 10,
    max_distinct: int | None = 10,
    min_target: int = 3,
    max_context_prior_length: int | None = 10,
) -> tensordict.TensorDict:
    _patch_randint(monkeypatch, [P, p])
    return get_sets(
        _height_batch(num_points),
        max_context_prior_length=max_context_prior_length,
        set_mode=set_mode,
        max_distinct=max_distinct,
        min_target=min_target,
        generator=torch.Generator().manual_seed(0),
    )


def _x_values(result: tensordict.TensorDict, key: str) -> tuple[float, ...]:
    sample = next(iter(result[key]))
    return tuple(sample["X"].flatten().tolist())


def _x_set(result: tensordict.TensorDict, key: str) -> set[float]:
    return set(_x_values(result, key))


def test_same_draw_has_identical_slices_and_context_sizes(monkeypatch):
    results = {mode: _make_sets(monkeypatch, mode) for mode in VALID_MODES}

    A = _x_set(results["nested-all"], "context_prior")
    B = _x_set(results["nested-noprior"], "target")
    Q = _x_set(results["disjoint-noprior"], "context_posterior") - B
    H = _x_set(results["disjoint-holdout"], "target")

    assert len(A) == len(Q) == 2
    assert len(B) == len(H) == 3

    for result in results.values():
        assert _x_set(result, "context_prior") == A
        assert len(_x_set(result, "context_posterior")) == 5

    assert _x_set(results["nested-all"], "context_posterior") == A | B
    assert _x_set(results["nested-all"], "target") == A | B
    assert _x_set(results["nested-noprior"], "context_posterior") == A | B
    assert _x_set(results["nested-nested"], "context_posterior") == A | B
    assert _x_set(results["nested-nested"], "target") == A | B | Q | H
    assert _x_set(results["nested-noprior-holdout"], "context_posterior") == A | B
    assert _x_set(results["nested-noprior-holdout"], "target") == B | Q | H
    assert _x_set(results["disjoint-noprior"], "context_posterior") == B | Q
    assert _x_set(results["disjoint-noprior"], "target") == B | Q
    assert _x_set(results["disjoint-nested"], "context_posterior") == B | Q
    assert _x_set(results["disjoint-nested"], "target") == B | Q | H
    assert _x_set(results["nested-holdout"], "target") == Q | H
    assert _x_set(results["disjoint-holdout"], "target") == H

    assert len(_x_set(results["nested-noprior"], "target")) == len(
        _x_set(results["disjoint-holdout"], "target")
    )
    assert len(_x_set(results["nested-all"], "target")) == len(
        _x_set(results["disjoint-noprior"], "target")
    )
    assert len(_x_set(results["nested-all"], "target")) == len(
        _x_set(results["nested-holdout"], "target")
    )
    assert len(_x_set(results["nested-nested"], "target")) == len(A | B | Q | H)
    assert len(_x_set(results["disjoint-nested"], "target")) == len(B | Q | H)
    assert len(_x_set(results["nested-noprior-holdout"], "target")) == len(B | Q | H)


def test_zero_prior_length_makes_q_empty_and_posteriors_identical(monkeypatch):
    nested = _make_sets(
        monkeypatch, "nested-all", P=3, p=0, num_points=6, max_distinct=6, min_target=3
    )
    disjoint = _make_sets(
        monkeypatch,
        "disjoint-noprior",
        P=3,
        p=0,
        num_points=6,
        max_distinct=6,
        min_target=3,
    )

    assert _x_values(nested, "context_prior") == ()
    assert _x_values(disjoint, "context_prior") == ()
    assert _x_set(nested, "context_posterior") == _x_set(disjoint, "context_posterior")
    assert _x_set(disjoint, "target") == _x_set(disjoint, "context_posterior")


def test_max_distinct_none_allows_half_sample_posterior(monkeypatch):
    result = _make_sets(
        monkeypatch,
        "nested-all",
        P=4,
        p=0,
        num_points=9,
        max_distinct=None,
        min_target=1,
        max_context_prior_length=0,
    )

    assert len(_x_values(result, "context_posterior")) == 4
    assert len(_x_values(result, "target")) == 4


def test_too_small_samples_or_caps_raise_value_error():
    with pytest.raises(ValueError, match="too short for symmetric set sampling"):
        get_sets(_height_batch(5), max_distinct=None, min_target=3)

    with pytest.raises(ValueError, match="too short for symmetric set sampling"):
        get_sets(_height_batch(20), max_distinct=5, min_target=3)


def test_invalid_set_mode_lists_valid_modes():
    with pytest.raises(ValueError, match="invalid set_mode") as exc_info:
        get_sets(_height_batch(), set_mode="unknown")  # ty: ignore[invalid-argument-type]

    message = str(exc_info.value)
    for mode in VALID_MODES:
        assert mode in message


def test_min_target_must_be_positive():
    with pytest.raises(ValueError, match="min_target must be >= 1"):
        get_sets(_height_batch(), min_target=0)


def test_max_context_prior_length_must_be_non_negative_or_none():
    with pytest.raises(
        ValueError, match="max_context_prior_length must be >= 0 or None"
    ):
        get_sets(_height_batch(), max_context_prior_length=-1)


def test_max_distinct_must_be_non_negative_or_none():
    with pytest.raises(ValueError, match="max_distinct must be >= 0 or None"):
        get_sets(_height_batch(), max_distinct=-1)
