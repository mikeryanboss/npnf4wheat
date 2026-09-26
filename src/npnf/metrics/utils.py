"""Small domain-free helpers shared by the metric modules and scripts."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import numpy as np
import polars as pl
import torch


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = f"{value!r} must be > 0"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def id_to_str(value: object) -> str:
    return str(value.item()) if torch.is_tensor(value) else str(value)


def mean_or_none(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return float(values.mean())


def std_and_se(values: Sequence[float]) -> tuple[float, float]:
    if len(values) <= 1:
        return 0.0, 0.0
    std = float(np.std(np.asarray(values, dtype=float), ddof=1))
    return std, std / math.sqrt(len(values))


def summary_value(values: pl.Series, op: str) -> float:
    if values.len() == 0:
        return float("nan")
    if op == "std" and values.len() <= 1:
        return 0.0
    value = getattr(values, op)()
    return float(value) if value is not None else 0.0


def quantile_value(values: pl.Series, quantile: float) -> float:
    if values.len() == 0:
        return float("nan")
    value = values.quantile(quantile)
    if value is None:
        msg = "Cannot compute a quantile of an all-null series."
        raise ValueError(msg)
    return value
