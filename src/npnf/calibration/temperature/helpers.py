"""Shared utilities for temperature calibration subpackages."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from npnf.calibration.temperature.data import HarmonicFit
from npnf.data.synthetic.temperature import HarmonicParams, harmonic_model


def acf_lag1(daily_values: np.ndarray) -> np.ndarray:
    """Compute lag-1 autocorrelation for each row of a 2-D array."""
    mean = daily_values.mean(axis=1, keepdims=True)
    centered = daily_values - mean
    var = (centered**2).mean(axis=1)
    cov = (centered[:, :-1] * centered[:, 1:]).mean(axis=1)
    return np.where(var > 0, cov / var, 0.0)


def fit_harmonic(signal: np.ndarray) -> tuple[HarmonicFit, float]:
    """Fit 2-harmonic seasonal model to a 1-D signal over 274 days.

    Returns:
        (harmonic_fit, fit_rmse) tuple. Fitted values are in ``fit.values``.
    """
    days = np.arange(274)
    data_mean = np.mean(signal)
    data_range = np.ptp(signal)
    peak_day = np.argmax(signal)
    amp1_guess = data_range / 2

    p0 = [data_mean, amp1_guess, peak_day, amp1_guess * 0.05, peak_day / 2]
    bounds = (
        [data_mean - data_range, 0.0, max(0.0, peak_day - 60), 0.0, 0.0],
        [
            data_mean + data_range,
            data_range,
            min(365.0, peak_day + 60),
            data_range / 2,
            365.0,
        ],
    )

    popt, _ = optimize.curve_fit(
        harmonic_model, days, signal, p0=p0, bounds=bounds, maxfev=10000
    )

    fit = HarmonicFit(
        params=HarmonicParams(
            base=float(popt[0]),
            amplitude=float(popt[1]),
            phase=float(popt[2]),
            amplitude_2=float(popt[3]),
            phase_2=float(popt[4]),
        )
    )
    rmse = float(np.sqrt(np.mean((signal - fit.values) ** 2)))
    return fit, rmse
