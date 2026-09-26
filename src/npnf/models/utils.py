import tensordict
import torch


def consolidate_tensordict_jagged_dim(td: tensordict.TensorDict):
    offsets = None
    for value in td.values():
        if getattr(value, "is_nested", False):
            offsets = value.offsets()  # ty: ignore[unresolved-attribute]
            break
    if offsets is None:
        return td

    for key, value in td.items():
        if getattr(value, "is_nested", False):
            td[key] = torch.nested.nested_tensor_from_jagged(
                values=value.values(),  # ty: ignore[invalid-argument-type]
                offsets=offsets,
                min_seqlen=value._min_seqlen,  # noqa: SLF001  # ty: ignore[unresolved-attribute]
                max_seqlen=value._max_seqlen,  # noqa: SLF001  # ty: ignore[unresolved-attribute]
            )
    return td
