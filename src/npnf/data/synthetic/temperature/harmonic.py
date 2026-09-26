"""Shared harmonic functions for temperature generation and calibration.

Pure mathematical functions used by both the calibration and generation pipelines.

- `harmonic_components()`: Individual harmonic terms
- `harmonic_model()`: Two-harmonic seasonal temperature model
"""

from __future__ import annotations

import numpy as np


def harmonic_components(
    days: np.ndarray, amp1: float, phase1: float, amp2: float, phase2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two harmonic terms separately.

    Args:
        days: Day values (array of day numbers).
        amp1: Primary (annual) harmonic amplitude (C).
        phase1: Primary harmonic phase (day of peak).
        amp2: Secondary (semi-annual) harmonic amplitude (C).
        phase2: Secondary harmonic phase.

    Returns:
        (h1, h2) where h1 is the annual and h2 the semi-annual component.
    """
    omega = 2 * np.pi / 365
    h1 = amp1 * np.cos(omega * (days - phase1))
    h2 = amp2 * np.cos(2 * omega * (days - phase2))
    return h1, h2


def harmonic_model(
    days: np.ndarray,
    base: float,
    amp1: float,
    phase1: float,
    amp2: float,
    phase2: float,
) -> np.ndarray:
    """Two-harmonic seasonal temperature model.

    Args:
        days: Day values (array of day numbers).
        base: Base/mean temperature (C).
        amp1: Primary (annual) harmonic amplitude (C).
        phase1: Primary harmonic phase (day of peak).
        amp2: Secondary (semi-annual) harmonic amplitude (C).
        phase2: Secondary harmonic phase.

    Returns:
        Temperature values at each day.
    """
    h1, h2 = harmonic_components(days, amp1, phase1, amp2, phase2)
    return base + h1 + h2
