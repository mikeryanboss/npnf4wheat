"""Fitting the lodging-mixture spline in closed form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sklearn.linear_model import LogisticRegression

from npnf.baselines.lodging_mixture_spline.lodging import DropDetector, observed_peak
from npnf.baselines.lodging_mixture_spline.model import Conditions, LodgingMixtureSpline


@dataclass(frozen=True)
class TrainingData:
    """Noisy height trajectories on one set of days. Rows with their own days
    mark them in ``observed``; None means every row is observed on every day."""

    days: torch.Tensor  # (D,)
    heights: torch.Tensor  # (N, D), finite also where not observed
    genotype_index: torch.Tensor  # (N,)
    yearsite_index: torch.Tensor  # (N,)
    genotype_ids: list[str]
    yearsite_ids: list[str]
    markers: torch.Tensor  # (G, M), one row per genotype id, NaN without markers
    thermal_time: torch.Tensor  # (Y, T), one row per year-site id
    observed: torch.Tensor | None = None  # (N, D) bool


def fit(
    data: TrainingData, calibration_draws: int = 65536
) -> tuple[LodgingMixtureSpline, dict[str, Any]]:
    """Fit the model and return it with a summary of the fit.

    Lodging is detected from the heights alone. The spline fits use every
    observation day, also those outside the growth window, which observe the
    constant edge levels. The genotype regression uses the genotypes with
    markers. Detected lodged observations are left out of the fits,
    so a lodged trajectory has no observations of its growth curve after the
    drop; the residual covariance therefore comes from the trajectories without
    a detected drop. The drop triples of the detected lodged trajectories are
    corrected for the detector bias (``calibrate_drop_triples``).
    """
    model = LodgingMixtureSpline(
        data.genotype_ids,
        data.yearsite_ids,
        num_markers=data.markers.shape[1],
        num_thermal_days=data.thermal_time.shape[1],
        num_drop_triples=0,
    )
    spline, geometry, detector = model.spline, model.geometry, DropDetector()
    days, values, observed = data.days, data.heights, data.observed
    detection = detector.detect(days, values, observed)
    weights = (~detection.mask).float()
    if observed is not None:
        weights *= observed
    coefficients = spline.fit(days, values, weights)
    residual = (values - coefficients @ spline.basis(days).T) * weights
    noise_scale = (residual.square().sum() / weights.sum()).sqrt()
    model.noise_scale.copy_(noise_scale)
    effects = model.effects
    growing = ~detection.lodged
    effects.fit(
        coefficients,
        data.genotype_index,
        data.yearsite_index,
        spline.estimation_covariance(days, weights[growing], float(noise_scale)),
        growing,
    )
    with_markers = data.markers.isfinite().all(dim=1)
    genotype_regression = model.genotype_regression.fit(
        data.markers[with_markers], effects.genotype[with_markers]
    )
    yearsite_regression = model.yearsite_regression.fit(
        data.thermal_time, effects.yearsite
    )

    peak = observed_peak(data.heights, detection.mask, observed)
    logistic = LogisticRegression(penalty=None).fit(
        peak[:, None].numpy(), detection.lodged.numpy()
    )
    model.lodging_intercept.fill_(float(logistic.intercept_[0]))
    model.lodging_slope.fill_(float(logistic.coef_[0, 0]))

    smooth = (
        effects.mean
        + effects.genotype[data.genotype_index]
        + effects.yearsite[data.yearsite_index]
    )
    growth_end, plateau_height = geometry.growth_end(
        spline.heights(smooth, spline.growth_days), spline.growth_days
    )
    lodged = detection.lodged
    triples = geometry.extract(
        data.days,
        data.heights[lodged],
        detection.day[lodged],
        plateau_height[lodged],
        growth_end[lodged],
        None if observed is None else observed[lodged],
    )
    model.drop_triples = triples
    correction, calibration = calibrate_drop_triples(
        model, data.days, detector, calibration_draws, observed
    )
    model.drop_triples = geometry.clip(triples + correction)
    summary = {
        "num_trajectories": len(data.heights),
        "num_detected_lodged": int(lodged.sum()),
        "num_genotypes_with_markers": int(with_markers.sum()),
        "num_drop_triples": len(triples),
        "noise_scale": float(noise_scale),
        "residual_std": effects.residual_covariance.diag().sqrt().tolist(),
        "mean_lodging_probability": float(
            torch.sigmoid(model.lodging_intercept + model.lodging_slope * peak).mean()
        ),
        "mean_drop_triple": model.drop_triples.mean(dim=0).tolist(),
        "calibration": calibration,
        "genotype_regression": genotype_regression,
        "yearsite_regression": yearsite_regression,
    }
    return model, summary


def calibrate_drop_triples(
    model: LodgingMixtureSpline,
    days: torch.Tensor,
    detector: DropDetector,
    num_draws: int,
    observed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Correction of the detector and extraction bias of the drop triples.

    Trajectories sampled from ``model`` with observation noise on ``days`` go
    through the same detector and extraction. With ``observed`` (N, D), each
    trajectory is observed on the days of a random training row. The correction
    is the mean of the triples that generated them minus the mean of the
    recovered triples, for example the detection lag. It uses no ground-truth
    labels.
    """
    generator = torch.Generator().manual_seed(0)
    sampled = model.sample(Conditions(1), num_draws, generator)
    heights = sampled.heights(model.spline, days)[0]
    heights = heights + model.noise_scale * torch.randn(
        heights.shape, generator=generator
    )
    if observed is not None:
        observed = observed[
            torch.randint(len(observed), (num_draws,), generator=generator)
        ]
    detection = detector.detect(days, heights, observed)
    rows = detection.lodged & sampled.lodged[0]
    true_mean = torch.stack(
        [
            sampled.lodging_day - sampled.growth_end,
            sampled.final_height / sampled.plateau_height.clamp(min=1e-6),
            sampled.transition,
        ],
        dim=-1,
    )[0, rows].mean(dim=0)
    recovered_mean = model.geometry.extract(
        days,
        heights[rows],
        detection.day[rows],
        sampled.plateau_height[0, rows],
        sampled.growth_end[0, rows],
        None if observed is None else observed[rows],
    ).mean(dim=0)
    correction = true_mean - recovered_mean
    return correction, {
        "sampled_lodged_rate": float(sampled.lodged.float().mean()),
        "detected_rate": float(detection.lodged.float().mean()),
        "correction": correction.tolist(),
    }
