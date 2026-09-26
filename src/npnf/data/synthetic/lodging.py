"""Shared stochastic lodging draws and deterministic trajectory transformation."""

from __future__ import annotations

import torch


def lodging_probability(
    max_heights: torch.Tensor,
    *,
    lodging_height_clamp: float,
    lodging_weibull_shape: float,
    lodging_weibull_scale: float,
    lodging_weibull_offset: float,
) -> torch.Tensor:
    """Lodging probability of each maximum clean height (Weibull CDF)."""
    clamped_heights = max_heights.clamp(max=lodging_height_clamp)
    effective_height = torch.clamp(clamped_heights - lodging_weibull_offset, min=0.0)
    scale = max(float(lodging_weibull_scale), 1e-6)
    normalized = (effective_height / scale).clamp(min=0.0)
    lodging_probs = 1 - torch.exp(-(normalized ** float(lodging_weibull_shape)))
    return lodging_probs.clamp(0.0, 1.0)


def sample_lodging_params(
    heights_clean: torch.Tensor,
    n_draws: int,
    generator: torch.Generator | None = None,
    *,
    enable_lodging: bool,
    lodging_height_clamp: float,
    lodging_weibull_shape: float,
    lodging_weibull_scale: float,
    lodging_weibull_offset: float,
    lodging_severity_min: float,
    lodging_severity_max: float,
    lodging_transition_steps_min: int,
    lodging_transition_steps_max: int,
) -> dict[str, torch.Tensor]:
    """Sample compact lodging parameters.

    Args:
        heights_clean: Clean height trajectories, shape (B, T).
        n_draws: Number of independent lodging draws.
        generator: Optional seeded generator for reproducibility.

    Returns:
        Dict of (B, N) tensors: will_lodge, offset, severity, transition_steps.
    """
    B, T = heights_clean.shape
    N = n_draws

    if not enable_lodging:
        return {
            "will_lodge": torch.zeros(B, N, dtype=torch.bool),
            "offset": torch.zeros(B, N, dtype=torch.long),
            "severity": torch.zeros(B, N),
            "transition_steps": torch.zeros(B, N, dtype=torch.long),
        }

    max_heights, max_indices = heights_clean.max(dim=1)  # (B,)

    lodging_probs = lodging_probability(
        max_heights,
        lodging_height_clamp=lodging_height_clamp,
        lodging_weibull_shape=lodging_weibull_shape,
        lodging_weibull_scale=lodging_weibull_scale,
        lodging_weibull_offset=lodging_weibull_offset,
    )

    # Draw 1: will_lodge (B, N)
    will_lodge = torch.bernoulli(
        lodging_probs.unsqueeze(1).expand(-1, N), generator=generator
    ).bool()

    # Draw 2: offset (B, N), clamped by remaining_days
    remaining_days = T - max_indices - 1  # (B,)
    offset = torch.abs(torch.randn(B, N, generator=generator) * 10.0).long()
    offset = torch.minimum(
        offset, remaining_days.unsqueeze(1).expand(-1, N).clamp(min=0)
    )

    # Draw 3: severity (B, N)
    severity = (
        torch.rand(B, N, generator=generator)
        * (lodging_severity_max - lodging_severity_min)
        + lodging_severity_min
    )

    # Draw 4: transition_steps (B, N) — stored pre-clamp
    transition_steps = torch.randint(
        lodging_transition_steps_min,
        max(lodging_transition_steps_max, lodging_transition_steps_min + 1),
        (B, N),
        generator=generator,
    )

    return {
        "will_lodge": will_lodge,
        "offset": offset,
        "severity": severity,
        "transition_steps": transition_steps,
    }


def apply_lodging_from_params(
    heights_clean: torch.Tensor,
    lodging_params: dict[str, torch.Tensor],
    *,
    enable_lodging: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically apply lodging from pre-sampled parameters.

    Args:
        heights_clean: Clean height trajectories, shape (B, T).
        lodging_params: Dict of (B, N) tensors from sample_lodging_params.

    Returns:
        Tuple of (heights_lodged, has_lodged, lodged_mask) with shapes
        (B, N, T), (B, N), (B, N, T).
    """
    B, T = heights_clean.shape
    N = lodging_params["will_lodge"].shape[1]
    will_lodge = lodging_params["will_lodge"]  # (B, N)

    if not enable_lodging:
        return (
            heights_clean.unsqueeze(1).expand(-1, N, -1),
            torch.zeros(B, N, dtype=torch.bool),
            torch.zeros(B, N, T, dtype=torch.bool),
        )

    max_heights, max_indices = heights_clean.max(dim=1)  # (B,)

    # Lodging day: max_index + 1 + offset
    lodging_day = max_indices.unsqueeze(1) + 1 + lodging_params["offset"]  # (B, N)

    # Clamp transition_steps by remaining time after lodging_day
    transition_steps = torch.minimum(
        lodging_params["transition_steps"], (T - lodging_day).clamp(min=0)
    )  # (B, N)

    # Expand clean heights to (B, N, T) — view, no copy
    heights_n = heights_clean.unsqueeze(1).expand(-1, N, -1)

    # Gather start heights at lodging day
    lodging_day_clamped = lodging_day.clamp(0, T - 1)  # (B, N)
    start_heights = heights_n.gather(2, lodging_day_clamped.unsqueeze(2)).squeeze(
        2
    )  # (B, N)

    # Final heights after lodging
    final_heights = max_heights.unsqueeze(1) * lodging_params["severity"]  # (B, N)

    # Time grid (1, 1, T) for broadcasting
    time_grid = torch.arange(T).reshape(1, 1, T)
    t_since_lodging = (time_grid - lodging_day.unsqueeze(2)).float()  # (B, N, T)

    # Decay computation
    safe_transition = transition_steps.float().clamp(min=1.0)  # (B, N)
    decay_rates = (
        -torch.log((final_heights / start_heights.clamp(min=1e-6)).clamp(min=1e-6))
        / safe_transition
    )  # (B, N)

    in_transition = (t_since_lodging > 0) & (
        t_since_lodging <= transition_steps.unsqueeze(2)
    )

    decay_factors = torch.exp(-decay_rates.unsqueeze(2) * t_since_lodging.clamp(min=0))
    decayed_heights = start_heights.unsqueeze(2) * decay_factors

    post_transition = t_since_lodging > transition_steps.unsqueeze(2).float()

    # Apply lodging via torch.where (creates new tensors, no in-place mutation)
    lodge_mask_expanded = will_lodge.unsqueeze(2) & (lodging_day.unsqueeze(2) < T)

    heights_n = torch.where(
        lodge_mask_expanded & in_transition, decayed_heights, heights_n
    )
    heights_n = torch.where(
        lodge_mask_expanded & post_transition, final_heights.unsqueeze(2), heights_n
    )

    # Output masks
    has_lodged = will_lodge & (lodging_day < T)  # (B, N)
    lodged_mask = (
        (time_grid >= lodging_day.unsqueeze(2))
        & (lodging_day.unsqueeze(2) < T)
        & will_lodge.unsqueeze(2)
    )  # (B, N, T)

    return heights_n, has_lodged, lodged_mask
