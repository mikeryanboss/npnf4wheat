"""Parameter dataclasses and I/O for height pool generation.

This module contains:
- Parameter dataclasses: `TruncatedNormalDist`, `MeanControlPointParams`,
  `ControlPointCovarianceParams`, `HeightPoolParams`
- Parameter loading: `height_pool_params_from_mapping()`,
  `height_pool_params_from_trial_params()`, `load_height_pool_params()`
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CALIBRATION_PARAMS_ENV_VAR = "NPNF_HEIGHT_PARAMS_PATH"


@dataclass(frozen=True)
class TruncatedNormalDist:
    """Parameters for a truncated-normal distribution."""

    loc: float
    scale: float
    min: float
    max: float


@dataclass(frozen=True)
class MeanControlPointParams:
    """Mean free control-point values (6x5 grid, row-major)."""

    cp_0: float
    cp_1: float
    cp_2: float
    cp_3: float
    cp_4: float
    cp_5: float
    cp_6: float
    cp_7: float
    cp_8: float
    cp_9: float
    cp_10: float
    cp_11: float
    cp_12: float
    cp_13: float
    cp_14: float
    cp_15: float
    cp_16: float
    cp_17: float
    cp_18: float
    cp_19: float
    cp_20: float
    cp_21: float
    cp_22: float
    cp_23: float
    cp_24: float
    cp_25: float
    cp_26: float
    cp_27: float
    cp_28: float
    cp_29: float


@dataclass(frozen=True)
class ControlPointCovarianceParams:
    """Covariance hyperparameters for the Gaussian CP prior."""

    cp_sigma: float
    length_scale_T: float  # noqa: N815
    length_scale_tau: float
    jitter: float


@dataclass(frozen=True)
class HeightPoolParams:
    """Top-level pool sampling parameters for genotype generation.

    Analogous to TemperatureParams. Composed of frozen sub-structs.
    Calibration produces this; GenotypePool/SyntheticGenotypeGenerator consume it.
    """

    mean_control_points: MeanControlPointParams
    covariance: ControlPointCovarianceParams
    tau_max: TruncatedNormalDist


def height_pool_params_from_mapping(values: Mapping[str, Any]) -> HeightPoolParams:
    """Build HeightPoolParams from a nested mapping with strict key checking."""
    return HeightPoolParams(
        mean_control_points=MeanControlPointParams(
            **{f"cp_{i}": values["mean_control_points"][f"cp_{i}"] for i in range(30)}
        ),
        covariance=ControlPointCovarianceParams(**values["covariance"]),
        tau_max=TruncatedNormalDist(**values["tau_max"]),
    )


def height_pool_params_from_trial_params(flat: dict[str, float]) -> HeightPoolParams:
    """Build HeightPoolParams from Optuna trial parameters.

    Expects flat keys: cp_0..cp_29, cp_sigma, length_scale_T,
    length_scale_tau, tau_max_loc, tau_max_scale.
    """
    return HeightPoolParams(
        mean_control_points=MeanControlPointParams(
            **{f"cp_{i}": flat[f"cp_{i}"] for i in range(30)}
        ),
        covariance=ControlPointCovarianceParams(
            cp_sigma=flat["cp_sigma"],
            length_scale_T=flat["length_scale_T"],
            length_scale_tau=flat["length_scale_tau"],
            jitter=1e-6,
        ),
        tau_max=TruncatedNormalDist(
            loc=flat["tau_max_loc"],
            scale=flat["tau_max_scale"],
            min=flat["tau_max_loc"] - 3 * flat["tau_max_scale"],
            max=flat["tau_max_loc"] + 3 * flat["tau_max_scale"],
        ),
    )


def resolve_height_pool_params_path(
    params_path: str | Path | None, env_var: str = CALIBRATION_PARAMS_ENV_VAR
) -> Path:
    """Resolve params path from argument or env variable."""
    if params_path is None:
        params_path = os.environ.get(env_var)
    if params_path is None:
        msg = f"No params path given and {env_var} is not set"
        raise ValueError(msg)
    return Path(params_path)


def load_height_pool_params(path: str | Path) -> HeightPoolParams:
    """Load HeightPoolParams from a calibration JSON file."""
    with Path(path).open() as f:
        data = json.load(f)
    payload = data.get("parameters", data)
    return height_pool_params_from_mapping(payload)
