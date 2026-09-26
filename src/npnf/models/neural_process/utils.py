from typing import Literal

import tensordict
import torch

from npnf.models.utils import consolidate_tensordict_jagged_dim

SetMode = Literal[
    "nested-all",
    "nested-noprior",
    "disjoint-noprior",
    "nested-holdout",
    "disjoint-holdout",
    "nested-nested",
    "disjoint-nested",
    "nested-noprior-holdout",
]

VALID_SET_MODES = (
    "nested-all",
    "nested-noprior",
    "disjoint-noprior",
    "nested-holdout",
    "disjoint-holdout",
    "nested-nested",
    "disjoint-nested",
    "nested-noprior-holdout",
)


def get_sets(  # noqa: PLR0912
    batch: tensordict.TensorDict,
    *,
    max_context_prior_length: int | None = 50,
    set_mode: SetMode = "nested-noprior",
    max_distinct: int | None = 128,
    min_target: int = 10,
    generator: torch.Generator | None = None,
) -> tensordict.TensorDict:
    if set_mode not in VALID_SET_MODES:
        msg = f"invalid set_mode {set_mode!r}; expected one of {VALID_SET_MODES}"
        raise ValueError(msg)
    if min_target < 1:
        msg = "min_target must be >= 1"
        raise ValueError(msg)
    if max_context_prior_length is not None and max_context_prior_length < 0:
        msg = "max_context_prior_length must be >= 0 or None"
        raise ValueError(msg)
    if max_distinct is not None and max_distinct < 0:
        msg = "max_distinct must be >= 0 or None"
        raise ValueError(msg)

    context_priors = []
    context_posteriors = []
    targets = []
    for sample in batch:
        sample.batch_size = sample["Y"].shape[:2]

        D_cap = (
            min(max_distinct, len(sample)) if max_distinct is not None else len(sample)
        )
        P_cap = D_cap // 2
        if P_cap < min_target:
            msg = "sample/caps are too short for symmetric set sampling"
            raise ValueError(msg)

        P = int(torch.randint(min_target, P_cap + 1, (1,), generator=generator).item())

        p_upper = P - min_target
        if max_context_prior_length is not None:
            p_upper = min(p_upper, max_context_prior_length)
        p = int(torch.randint(0, p_upper + 1, (1,), generator=generator).item())

        noise = torch.rand(
            sample["Y"].shape,
            dtype=sample["Y"].dtype,
            device="cpu",
            generator=generator,
        ).to(sample["Y"].device)
        indices = noise.argsort(dim=0)

        context_prior_indices, _ = indices[:p].sort(dim=0)
        nested_posterior_indices, _ = indices[:P].sort(dim=0)
        disjoint_posterior_indices, _ = indices[p : P + p].sort(dim=0)

        if set_mode == "nested-all":
            context_posterior_indices = nested_posterior_indices
            target_indices = nested_posterior_indices
        elif set_mode == "nested-noprior":
            context_posterior_indices = nested_posterior_indices
            target_indices, _ = indices[p:P].sort(dim=0)
        elif set_mode == "disjoint-noprior":
            context_posterior_indices = disjoint_posterior_indices
            target_indices = disjoint_posterior_indices
        elif set_mode == "nested-holdout":
            context_posterior_indices = nested_posterior_indices
            target_indices, _ = indices[P : 2 * P].sort(dim=0)
        elif set_mode == "nested-nested":
            context_posterior_indices = nested_posterior_indices
            target_indices, _ = indices[: 2 * P].sort(dim=0)
        elif set_mode == "disjoint-nested":
            context_posterior_indices = disjoint_posterior_indices
            target_indices, _ = indices[p : 2 * P].sort(dim=0)
        elif set_mode == "nested-noprior-holdout":
            context_posterior_indices = nested_posterior_indices
            target_indices, _ = indices[p : 2 * P].sort(dim=0)
        else:
            context_posterior_indices = disjoint_posterior_indices
            target_indices, _ = indices[P + p : 2 * P].sort(dim=0)

        # Bug in tensordict gather with no indices, works in torch
        if len(context_prior_indices) > 0:
            context_prior = sample.gather(dim=0, index=context_prior_indices)
        else:
            context_prior = sample[:0]
        context_posterior = sample.gather(dim=0, index=context_posterior_indices)
        target = sample.gather(dim=0, index=target_indices)

        context_prior.batch_size = []
        context_posterior.batch_size = []
        sample.batch_size = []
        target.batch_size = []

        context_priors.append(context_prior)
        context_posteriors.append(context_posterior)
        targets.append(target)

    context_priors = tensordict.lazy_stack(context_priors).densify(layout=torch.jagged)
    context_posteriors = tensordict.lazy_stack(context_posteriors).densify(
        layout=torch.jagged
    )
    targets = tensordict.lazy_stack(targets).densify(layout=torch.jagged)

    if context_priors["Y"].is_nested:
        context_priors = consolidate_tensordict_jagged_dim(context_priors)
    if context_posteriors["Y"].is_nested:
        context_posteriors = consolidate_tensordict_jagged_dim(context_posteriors)
    if targets["Y"].is_nested:
        targets = consolidate_tensordict_jagged_dim(targets)

    batch["context_prior"] = context_priors
    batch["context_posterior"] = context_posteriors
    batch["target"] = targets

    return batch
