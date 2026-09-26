"""Type-narrowing helpers for TensorDict lookups in tests."""

import tensordict
import torch
from tensordict.utils import IndexType


def tensor_at(collection: tensordict.TensorDictBase, index: IndexType) -> torch.Tensor:
    value = collection[index]
    assert isinstance(value, torch.Tensor)
    return value


def tensordict_at(
    collection: tensordict.TensorDictBase, index: IndexType
) -> tensordict.TensorDict:
    value = collection[index]
    assert isinstance(value, tensordict.TensorDict)
    return value
