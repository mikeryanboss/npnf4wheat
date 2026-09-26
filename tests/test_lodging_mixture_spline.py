"""Tests for the lodging-mixture spline baseline."""

from __future__ import annotations

from pathlib import Path

import tensordict
import torch
from safetensors.torch import load_file

from npnf.baselines.lodging_mixture_spline.effects import (
    AdditiveEffects,
    EffectRegression,
    cumulative_thermal_time,
)
from npnf.baselines.lodging_mixture_spline.fit import TrainingData, fit
from npnf.baselines.lodging_mixture_spline.lodging import (
    DropDetector,
    DropGeometry,
    observed_peak,
)
from npnf.baselines.lodging_mixture_spline.model import (
    Conditions,
    LodgingMixtureSpline,
    UnseenEffect,
)
from npnf.baselines.lodging_mixture_spline.pspline import PSpline
from npnf.calibration.height.constants import HeightDates
from npnf.data.datasets.synthetic import _precompute_yearsite_basis
from npnf.scripts.baselines.lodging_mixture_spline import save_run
from npnf.scripts.utils.prediction import (
    get_empty_observations,
    predict_batch_with_grid,
    resolve_model_from_checkpoint,
)


def _growth_days() -> torch.Tensor:
    return PSpline().growth_days.float()


def _model(
    lodging_probability: float,
    genotype_scale: float = 0.0,
    num_markers: int = 4,
    unseen_effect: UnseenEffect = UnseenEffect.REGRESSION,
) -> LodgingMixtureSpline:
    """Three genotypes, two year-sites, no residual; genotype effects shift the
    curve by ``genotype_scale`` times (-1, 0, 1)."""
    model = LodgingMixtureSpline(
        genotype_ids=["G_0000", "G_0001", "G_0002"],
        yearsite_ids=["Synth00_2010", "Synth00_2011"],
        num_markers=num_markers,
        num_thermal_days=123,
        num_drop_triples=2,
        unseen_effect=unseen_effect,
    )
    size = model.spline.num_coefficients
    model.effects.mean.copy_(torch.linspace(0.0, 1.0, size))
    model.effects.genotype.copy_(
        genotype_scale * torch.tensor([[-1.0], [0.0], [1.0]]).expand(3, size)
    )
    model.lodging_intercept.fill_(float(torch.logit(torch.tensor(lodging_probability))))
    model.drop_triples.copy_(torch.tensor([[5.0, 0.5, 10.0], [0.0, 0.3, 0.0]]))
    model.noise_scale.fill_(0.05)
    return model


def test_fit_recovers_noise_free_spline_coefficients() -> None:
    days = _growth_days()
    spline = PSpline(penalty=0.0)
    coefficients = torch.rand(
        6, spline.num_coefficients, generator=torch.Generator().manual_seed(1)
    )
    values = coefficients @ spline.basis(days).T
    weights = torch.ones_like(values)
    weights[0, 40:60] = 0.0  # left-out (lodged) days must not break the fit
    assert torch.allclose(spline.fit(days, values, weights), coefficients, atol=1e-4)


def test_penalty_fits_curves_without_late_observations() -> None:
    days = _growth_days()
    curve = torch.sigmoid((days - 240.0) / 10.0)
    weights = (days < 250).float()[None]  # nothing observed after day 250
    coefficients = PSpline().fit(days, curve[None], weights)
    fitted = PSpline().heights(coefficients, days)[0]
    assert (fitted[days < 250] - curve[days < 250]).abs().max() < 0.02


def test_basis_is_constant_outside_the_growth_window() -> None:
    outside = torch.tensor([61.0, 137.0, 199.0, 323.0, 364.0])
    edges = PSpline().basis(torch.tensor([200.0, 200.0, 200.0, 322.0, 322.0]))
    assert torch.equal(PSpline().basis(outside), edges)


def test_fit_uses_observations_outside_the_growth_window() -> None:
    """Zero heights before day 200 hold the start level of the fitted curves at 0;
    from the window days alone, the start level depends on the growth start."""
    days = torch.cat(
        [
            torch.tensor([85.0, 96.0, 117.0, 137.0, 162.0, 181.0]),
            torch.arange(200.0, 322.0),
            torch.tensor([352.0, 353.0]),
        ]
    )
    growth_start = torch.linspace(205.0, 215.0, 8)
    final_height = torch.linspace(0.8, 1.1, 4)
    genotype_index = torch.arange(4).repeat_interleave(8)
    yearsite_index = torch.arange(8).repeat(4)
    progress = ((days - growth_start[yearsite_index, None]) / 60.0).clamp(0.0, 1.0)
    heights = (
        final_height[genotype_index, None] * progress.square() * (3 - 2 * progress)
    )
    lodged = (genotype_index >= 2) & (yearsite_index % (5 - genotype_index) == 0)
    heights[lodged] = torch.where(days > 300.0, 0.5 * heights[lodged], heights[lodged])
    data = TrainingData(
        days=days,
        heights=heights,
        genotype_index=genotype_index,
        yearsite_index=yearsite_index,
        genotype_ids=[f"G_{index:04d}" for index in range(4)],
        yearsite_ids=[f"Synth00_{2010 + index}" for index in range(8)],
        markers=torch.randn(4, 3, generator=torch.Generator().manual_seed(0)),
        thermal_time=torch.linspace(0.0, 1.0, 123).expand(8, 123).clone(),
    )
    model, _ = fit(data, calibration_draws=512)
    effects = model.effects
    start_level = (
        effects.mean[0]
        + effects.genotype[genotype_index, 0]
        + effects.yearsite[yearsite_index, 0]
    )
    assert start_level.abs().max() < 0.003


def test_residual_covariance_comes_from_the_residual_rows_only() -> None:
    """Rows outside ``residual_rows`` (lodged trajectories, whose late coefficients
    have no observations) set the group means but not the residual covariance."""
    genotype_index = torch.arange(3).repeat_interleave(8)
    yearsite_index = torch.arange(4).repeat_interleave(2).repeat(3)
    generator = torch.Generator().manual_seed(0)
    coefficients = (
        torch.randn(3, 5, generator=generator)[genotype_index]
        + torch.randn(4, 5, generator=generator)[yearsite_index]
    )
    excluded = torch.zeros(24, dtype=torch.bool)
    excluded[:4] = True
    coefficients[:4, -2:] += torch.tensor([1.0, -1.0, 2.0, -2.0])[:, None]
    effects = AdditiveEffects(3, 4, 5)
    effects.fit(
        coefficients, genotype_index, yearsite_index, torch.zeros(5, 5), ~excluded
    )
    assert effects.residual_covariance.abs().max() < 1e-6
    effects.fit(
        coefficients,
        genotype_index,
        yearsite_index,
        torch.zeros(5, 5),
        torch.ones(24, dtype=torch.bool),
    )
    assert effects.residual_covariance[-1, -1] > 0.1


def test_growth_end_is_start_of_plateau() -> None:
    days = _growth_days()
    curve = torch.sigmoid((days - 250.0) / 8.0).clamp(max=0.9)  # flat from ~268
    coefficients = PSpline().fit(days, curve[None], torch.ones(1, len(days)))
    day, height = DropGeometry().growth_end(PSpline().heights(coefficients, days), days)
    assert abs(height.item() - 0.9) < 0.02
    assert 264 <= day.item() <= 276


def test_detector_finds_drops_from_heights_only() -> None:
    generator = torch.Generator().manual_seed(2)
    off_season = torch.tensor([70.0, 120.0, 180.0, 330.0, 350.0, 360.0])
    days = torch.cat([off_season, _growth_days()[:-1]])  # unsorted on purpose
    curve = torch.sigmoid((days - 250.0) / 12.0) * (days >= 200)
    values = curve.repeat(4, 1) + 0.05 * torch.randn(4, len(days), generator=generator)
    lodging_days = [290.0, 305.0]
    for row, day in enumerate(lodging_days):
        after = days >= day
        values[row, after] = 0.45 + 0.05 * torch.randn(
            int(after.sum()), generator=generator
        )
    detection = DropDetector().detect(days, values)
    assert detection.lodged.tolist() == [True, True, False, False]
    assert torch.all((detection.day[:2] - torch.tensor(lodging_days)).abs() <= 3)
    assert not detection.mask[2:].any()
    assert bool(detection.mask[0][days >= 293].all())
    assert not bool(detection.mask[0][days < 287].any())


def test_thermal_time_matches_the_simulator() -> None:
    temperatures = 10 * torch.randn(
        3, 274, 24, generator=torch.Generator().manual_seed(5)
    )
    dates = HeightDates()
    _, hourly = _precompute_yearsite_basis(temperatures, dates=dates)
    daily_end = hourly.view(3, -1, 24)[:, : dates.maximum_growth_period + 1, -1]
    assert torch.allclose(cumulative_thermal_time(temperatures), daily_end)


def test_sampler_matches_lodging_probability_and_keeps_growth_monotone() -> None:
    grid = torch.tensor([61.0, 137.0, 199.0, 204.0, 250.0, 300.0, 322.0, 340.0, 364.0])
    conditions = Conditions(
        2,
        genotype_ids=["G_0000", "G_9999"],
        markers=torch.zeros(2, 4),
        yearsite_ids=["Synth00_2010", "Synth99_2099"],
        thermal_time=torch.zeros(2, 123),
    )
    model = _model(0.3, unseen_effect=UnseenEffect.UNIFORM)
    draws = model.sample(conditions, 2000, torch.Generator().manual_seed(0))
    heights = draws.heights(model.spline, grid)
    assert heights.shape == (2, 2000, len(grid))
    assert abs(draws.lodged.float().mean().item() - 0.3) < 0.03
    peak = heights.max(dim=2).values
    assert bool((heights[draws.lodged][:, -1] < peak[draws.lodged] - 1e-6).all())
    growing = heights[~draws.lodged]
    assert bool((growing[:, 1:] >= growing[:, :-1] - 1e-6).all())


def test_regression_predicts_unseen_genotype_effects() -> None:
    generator = torch.Generator().manual_seed(4)
    size = PSpline().num_coefficients
    markers = torch.randn(300, 8, generator=generator)
    effects = 0.1 * markers @ torch.randn(8, size, generator=generator)
    regression = EffectRegression(8, size)
    assert regression.fit(markers[:200], effects[:200])["loo_r2"] > 0.95
    predicted = regression.predict(markers[200:])
    error = (predicted - effects[200:]).square().sum() / effects[200:].square().sum()
    assert error < 0.05

    grid = torch.tensor([250.0, 300.0])
    conditions = Conditions(2, ["G_9998", "G_9999"], markers[200:202])

    def mean_error(unseen_effect: UnseenEffect) -> float:
        model = _model(1e-6, num_markers=8, unseen_effect=unseen_effect)
        model.genotype_regression = regression
        truth = model.spline.heights(model.effects.mean + effects[200:202], grid)
        draws = model.sample(conditions, 256, torch.Generator().manual_seed(0))
        return float(
            (draws.heights(model.spline, grid).mean(dim=1) - truth).abs().max()
        )

    assert mean_error(UnseenEffect.REGRESSION) < 0.05
    assert mean_error(UnseenEffect.UNIFORM) > mean_error(UnseenEffect.REGRESSION)


def test_context_conditioning_selects_matching_genotype() -> None:
    days = _growth_days()
    model = _model(1e-6, genotype_scale=0.2)
    target = model.effects.mean + model.effects.genotype[2]
    mask = torch.zeros(1, len(days), dtype=torch.bool)
    mask[0, 40:80] = True
    draws, sample_size = model.condition(
        Conditions(1),  # genotype unknown: it must come from the context
        days,
        mask,
        model.spline.heights(target, days)[None],
        num_draws=64,
        pool_size=512,
        generator=torch.Generator().manual_seed(0),
        max_pool_trajectories=256,  # two chunks
    )
    grid = torch.tensor([250.0, 300.0, 340.0])
    expected = model.spline.heights(target, grid).expand(64, -1)
    assert torch.allclose(draws.heights(model.spline, grid)[0], expected, atol=1e-5)
    assert 100 < sample_size.item() < 250  # about one third of the pool


def _observations(
    days: list[list[float]], heights: list[list[float]]
) -> tensordict.TensorDict:
    rows = [
        tensordict.TensorDict(
            {
                "X": torch.tensor(x)[:, None],
                "X_normalized": (torch.tensor(x)[:, None] - 212.5) / 151.5,
                "Y": torch.tensor(y)[:, None],
            },
            batch_size=[len(x)],
        )
        for x, y in zip(days, heights, strict=True)
    ]
    context = tensordict.lazy_stack(rows, dim=0).densify(layout=torch.jagged)
    context.batch_size = context.batch_size[:1]
    return context


def test_prediction_call_uses_ids_only_with_their_covariate(tmp_path: Path) -> None:
    model = _model(1e-6, genotype_scale=0.2)
    grid_days = torch.tensor([250.0, 300.0])
    grid = tensordict.TensorDict(
        {"X": grid_days[:, None], "X_normalized": (grid_days[:, None] - 212.5) / 151.5},
        batch_size=[2],
    )
    empty = get_empty_observations(
        batch_size=2, device=torch.device("cpu"), dtype=torch.float32
    )
    markers = tensordict.TensorDict({"Y": torch.zeros(2, 1, 4)}, batch_size=[2, 1])
    ids = {"genotype_ids": ["G_0002", "G_0000"], "yearsite_ids": ["Synth00_2010"] * 2}

    def predict(context: tensordict.TensorDict, **covariates) -> torch.Tensor:
        return predict_batch_with_grid(
            model, context, empty, grid, num_samples=64, **covariates, **ids
        )["grid"][..., 0]  # ty: ignore[invalid-return-type]

    with_markers = predict(empty, markers=markers)
    truth = model.spline.heights(
        model.effects.mean + model.effects.genotype[[2, 0]], grid_days
    )
    assert torch.allclose(with_markers, truth[:, None].expand(-1, 64, -1), atol=1e-5)
    without_markers = predict(empty)  # the ids alone must not select the genotype
    assert without_markers.std(dim=1).max() > 0.05

    days = _growth_days()[40:80]
    heights = model.spline.heights(model.effects.mean + model.effects.genotype[2], days)
    context = _observations([days.tolist()] * 2, [heights.tolist()] * 2)
    conditioned = predict(context)
    assert torch.allclose(conditioned[0], truth[0].expand(64, -1), atol=1e-4)

    save_run(model, tmp_path / "Spline", {}, {})
    loaded = resolve_model_from_checkpoint(str(tmp_path / "Spline" / "checkpoints"))
    state = load_file(
        str(tmp_path / "Spline/checkpoints/checkpoint-0/model.safetensors")
    )
    loaded.load_state_dict(state)
    assert isinstance(loaded, LodgingMixtureSpline)
    assert loaded.genotype_ids == model.genotype_ids
    assert torch.equal(loaded.drop_triples, model.drop_triples)
    assert float(loaded.variance) == float(model.variance)


def test_detector_counts_only_the_observed_entries() -> None:
    """Rows with their own days on a shared grid give the same detection as each
    row on its own days."""
    generator = torch.Generator().manual_seed(3)
    days = torch.cat(
        [torch.tensor([150.0, 180.0]), _growth_days(), torch.tensor([340.0])]
    )
    curve = torch.sigmoid((days - 250.0) / 12.0) * (days >= 200)
    values = curve.repeat(6, 1) + 0.02 * torch.randn(6, len(days), generator=generator)
    for row, day in enumerate([280.0, 295.0, 310.0]):
        values[row, days >= day] = 0.4
    observed = torch.rand(6, len(days), generator=generator) < 0.4
    detection = DropDetector().detect(days, values, observed)
    for row in range(6):
        alone = DropDetector().detect(
            days[observed[row]], values[row, observed[row]][None]
        )
        assert bool(detection.lodged[row]) == bool(alone.lodged[0])
        if alone.lodged[0]:
            assert detection.day[row] == alone.day[0]
            assert torch.equal(detection.mask[row, observed[row]], alone.mask[0])
    assert detection.lodged[:3].all()
    assert not detection.lodged[3:].any()
    assert not (detection.mask & ~observed).any()
    peak = observed_peak(values, detection.mask, observed)
    for row in range(3, 6):
        alone = observed_peak(
            values[row, observed[row]][None],
            torch.zeros(1, 1, dtype=torch.bool).expand(1, int(observed[row].sum())),
        )
        assert torch.allclose(peak[row], alone[0])


def test_fit_accepts_rows_with_own_days_and_genotypes_without_markers() -> None:
    generator = torch.Generator().manual_seed(5)
    days = torch.arange(180.0, 330.0)
    genotype_index = torch.arange(6).repeat_interleave(4)
    yearsite_index = torch.arange(4).repeat(6)
    final_height = torch.linspace(0.7, 1.1, 6)[genotype_index, None]
    growth_start = torch.linspace(205.0, 215.0, 4)[yearsite_index, None]
    progress = ((days - growth_start) / 60.0).clamp(0.0, 1.0)
    heights = final_height * progress + 0.01 * torch.randn(
        24, len(days), generator=generator
    )
    heights[::5] = torch.where(days > 300.0, 0.4 * heights[::5], heights[::5])
    observed = torch.rand(24, len(days), generator=generator) < 0.3
    markers = torch.randn(6, 5, generator=generator)
    markers[1] = torch.nan
    data = TrainingData(
        days=days,
        heights=torch.where(observed, heights, 0.0),
        genotype_index=genotype_index,
        yearsite_index=yearsite_index,
        genotype_ids=[f"G_{index:04d}" for index in range(6)],
        yearsite_ids=[f"FPWW{index:03d}" for index in range(4)],
        markers=markers,
        thermal_time=torch.rand(4, 123, generator=generator).cumsum(dim=1),
        observed=observed,
    )
    model, summary = fit(data, calibration_draws=512)
    assert summary["num_genotypes_with_markers"] == 5
    assert summary["noise_scale"] < 0.02
    fitted = model.spline.heights(
        model.effects.mean
        + model.effects.genotype[genotype_index]
        + model.effects.yearsite[yearsite_index],
        days,
    )
    assert (fitted - final_height * progress).abs().mean() < 0.02

    conditions = Conditions(
        2, ["G_9998", "G_9999"], torch.stack([markers[0], torch.full((5,), torch.nan)])
    )
    draws = model.sample(conditions, 256, torch.Generator().manual_seed(0))
    spread = draws.heights(model.spline, torch.tensor([320.0]))[..., 0].std(dim=1)
    assert spread[1] > spread[0]  # no markers: a random training genotype


def test_prediction_call_accepts_genotypes_without_markers() -> None:
    """FIP1 batches give a genotype without markers no marker row."""
    model = _model(1e-6, genotype_scale=0.2)
    grid_days = torch.tensor([250.0, 300.0])
    grid = tensordict.TensorDict(
        {"X": grid_days[:, None], "X_normalized": (grid_days[:, None] - 212.5) / 151.5},
        batch_size=[2],
    )
    empty = get_empty_observations(
        batch_size=2, device=torch.device("cpu"), dtype=torch.float32
    )
    rows = [torch.zeros(1, 4), torch.zeros(0, 4)]
    markers = tensordict.TensorDict(
        {"Y": torch.nested.nested_tensor(rows, layout=torch.jagged)}, batch_size=[2]
    )
    output = predict_batch_with_grid(
        model,
        empty,
        empty,
        grid,
        num_samples=64,
        markers=markers,
        genotype_ids=["G_0002", "G_9999"],
        yearsite_ids=["Synth00_2010"] * 2,
    )
    predicted: torch.Tensor = output["grid"][..., 0]  # ty: ignore[invalid-assignment]
    truth = model.spline.heights(
        model.effects.mean + model.effects.genotype[2], grid_days
    )
    assert torch.allclose(predicted[0], truth.expand(64, -1), atol=1e-5)
    assert predicted[1].std(dim=0).max() > 0.05


def test_additive_effects_are_exact_in_an_unbalanced_design() -> None:
    """Not every genotype is in every year-site; plain group means would mix the
    year-site effects into the genotype effects."""
    generator = torch.Generator().manual_seed(6)
    genotype_index = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 4])
    yearsite_index = torch.tensor([0, 1, 1, 2, 0, 2, 0, 2, 1])
    genotype = torch.randn(5, 3, generator=generator)
    yearsite = 2 * torch.randn(3, 3, generator=generator)
    coefficients = genotype[genotype_index] + yearsite[yearsite_index]
    effects = AdditiveEffects(5, 3, 3)
    effects.fit(
        coefficients,
        genotype_index,
        yearsite_index,
        torch.zeros(3, 3),
        torch.ones(9, dtype=torch.bool),
    )
    fitted = (
        effects.mean
        + effects.genotype[genotype_index]
        + effects.yearsite[yearsite_index]
    )
    assert torch.allclose(fitted, coefficients, atol=1e-4)
    assert effects.residual_covariance.abs().max() < 1e-6
