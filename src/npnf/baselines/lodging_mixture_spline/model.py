"""The lodging-mixture spline: prior sampling, context conditioning and the
neural-process call of the prediction scripts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any

import tensordict
import torch
from torch import nn

from npnf.baselines.lodging_mixture_spline.effects import (
    AdditiveEffects,
    EffectRegression,
    cumulative_thermal_time,
)
from npnf.baselines.lodging_mixture_spline.lodging import DropGeometry
from npnf.baselines.lodging_mixture_spline.pspline import PSpline
from npnf.baselines.lodging_mixture_spline.utils import running_mean, sample_gaussian


class UnseenEffect(StrEnum):
    """Effect of a genotype or year-site that is not in the training data."""

    REGRESSION = "regression"  # predicted from its markers or thermal time
    UNIFORM = "uniform"  # a random training effect, the model as stated


@dataclass(frozen=True)
class Conditions:
    """Covariates of the rows to sample. ``genotype_ids`` with ``markers`` (B, M)
    and ``yearsite_ids`` with ``thermal_time`` (B, T) are None when the covariate
    is not used; the row then gets a random training effect. A NaN row of
    ``markers`` marks a genotype without markers."""

    num_rows: int
    genotype_ids: list[str] | None = None
    markers: torch.Tensor | None = None
    yearsite_ids: list[str] | None = None
    thermal_time: torch.Tensor | None = None

    def rows(self, start: int, stop: int) -> Conditions:
        return Conditions(
            len(range(self.num_rows)[start:stop]),
            None if self.genotype_ids is None else self.genotype_ids[start:stop],
            None if self.markers is None else self.markers[start:stop],
            None if self.yearsite_ids is None else self.yearsite_ids[start:stop],
            None if self.thermal_time is None else self.thermal_time[start:stop],
        )


@dataclass(frozen=True)
class Trajectories:
    """(B, N) sampled trajectories: a spline curve with an optional linear drop."""

    coefficients: torch.Tensor  # (B, N, K)
    lodged: torch.Tensor
    growth_end: torch.Tensor
    plateau_height: torch.Tensor
    lodging_day: torch.Tensor
    start_height: torch.Tensor  # height on the lodging day
    final_height: torch.Tensor
    transition: torch.Tensor  # days from the lodging day to the final height

    def heights(self, spline: PSpline, days: torch.Tensor) -> torch.Tensor:
        """(B, N, D) heights on ``days``."""
        curve = spline.heights(self.coefficients, days)
        elapsed = days.float().to(curve.device) - self.lodging_day[..., None]
        transition = self.transition[..., None]
        progress = torch.where(
            transition > 0, (elapsed / transition.clamp(min=1)).clamp(0, 1), 1.0
        )
        start = self.start_height[..., None]
        dropped = start + (self.final_height[..., None] - start) * progress
        return torch.where(self.lodged[..., None] & (elapsed >= 0), dropped, curve)

    def take(self, draws: torch.Tensor) -> Trajectories:
        """The draws ``draws`` (B, N') of every row."""
        rows = torch.arange(len(draws), device=draws.device)[:, None]
        return Trajectories(
            **{
                field.name: getattr(self, field.name)[rows, draws]
                for field in fields(self)
            }
        )

    @staticmethod
    def cat(parts: list[Trajectories]) -> Trajectories:
        """The rows of ``parts`` in order."""
        return Trajectories(
            **{
                field.name: torch.cat([getattr(part, field.name) for part in parts])
                for field in fields(Trajectories)
            }
        )


class LodgingMixtureSpline(nn.Module):
    """``P(H | u) = (1 - π(u)) P_nonlodged(H | u) + π(u) P_lodged(H | u)``.

    ``P_nonlodged``: P-spline curves with additive genotype and year-site effects.
    ``π``: logistic in the observed peak height (3-day running mean of noisy
    heights). ``P_lodged``: a linear drop with an empirical (delay, severity,
    transition) triple. The constructor arguments are the model configuration;
    the fitted values are buffers, set by ``fit`` or loaded from a checkpoint.
    """

    lodging_intercept: torch.Tensor
    lodging_slope: torch.Tensor
    drop_triples: torch.Tensor  # (M, 3)
    noise_scale: torch.Tensor

    def __init__(
        self,
        genotype_ids: list[str],
        yearsite_ids: list[str],
        num_markers: int,
        num_thermal_days: int,
        num_drop_triples: int,
        unseen_effect: UnseenEffect = UnseenEffect.REGRESSION,
        pool_size: int = 16384,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.spline = PSpline()
        self.geometry = DropGeometry()
        self.genotype_ids = list(genotype_ids)
        self.yearsite_ids = list(yearsite_ids)
        self.unseen_effect = UnseenEffect(unseen_effect)
        self.pool_size = pool_size
        self.seed = seed
        self.generator: torch.Generator | None = None
        size = self.spline.num_coefficients
        self.effects = AdditiveEffects(len(genotype_ids), len(yearsite_ids), size)
        self.genotype_regression = EffectRegression(num_markers, size)
        self.yearsite_regression = EffectRegression(num_thermal_days, size)
        self.register_buffer("lodging_intercept", torch.tensor(0.0))
        self.register_buffer("lodging_slope", torch.tensor(0.0))
        self.register_buffer("drop_triples", torch.zeros(num_drop_triples, 3))
        self.register_buffer("noise_scale", torch.tensor(0.0))

    @property
    def variance(self) -> torch.Tensor:
        """Observation noise variance, as the neural processes report it."""
        return self.noise_scale.square()

    def forward(
        self,
        context_priors: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        temperatures: tensordict.TensorDict | None = None,
        markers: tensordict.TensorDict | None = None,
        num_samples: int = 64,
        genotype_ids: list[str] | None = None,
        yearsite_ids: list[str] | None = None,
        **ignored: Any,
    ) -> dict[str, Any]:
        """The neural-process call of the prediction scripts.

        ``markers`` and ``temperatures`` are None when the covariate is not used;
        the ids are used only together with their covariate. Rows with context
        points are conditioned on them.
        """
        if self.generator is None:
            self.generator = torch.Generator(self.noise_scale.device)
            self.generator.manual_seed(self.seed)
        num_rows = len(targets)
        marker_rows = thermal_time = None
        if markers is not None:
            marker_values: torch.Tensor = markers["Y"]  # ty: ignore[invalid-assignment]
            marker_rows = torch.full(
                (num_rows, self.genotype_regression.coefficients.shape[1]), torch.nan
            )
            for row, values in enumerate(marker_values.unbind()):
                if values.numel() > 0:  # genotypes without markers have no row
                    marker_rows[row] = values.reshape(-1).float().cpu()
        if temperatures is not None:
            hourly: torch.Tensor = temperatures["Y"]  # ty: ignore[invalid-assignment]
            thermal_time = cumulative_thermal_time(hourly.reshape(num_rows, 274, 24))
        conditions = Conditions(
            num_rows,
            None if markers is None else genotype_ids,
            marker_rows,
            None if temperatures is None else yearsite_ids,
            thermal_time,
        )
        context = [(row["X"][..., 0], row["Y"][..., 0]) for row in context_priors]
        if all(len(days) == 0 for days, _heights in context):
            draws = self.sample(conditions, num_samples, self.generator)
        else:
            context_days = torch.unique(torch.cat([days for days, _heights in context]))
            mask = torch.zeros(num_rows, len(context_days), dtype=torch.bool)
            values = torch.zeros(num_rows, len(context_days))
            for row, (days, heights) in enumerate(context):
                position = torch.searchsorted(context_days, days).cpu()
                mask[row, position] = True
                values[row, position] = heights.float().cpu()
            draws = self.condition(
                conditions,
                context_days,
                mask,
                values,
                num_samples,
                self.pool_size,
                self.generator,
            )[0]
        target_days = [row["X"][..., 0] for row in targets]
        all_days = torch.unique(torch.cat(target_days))
        heights = draws.heights(self.spline, all_days)
        target = [
            heights[row, :, torch.searchsorted(all_days, days)][..., None]
            for row, days in enumerate(target_days)
        ]
        scale = [torch.zeros_like(row) for row in target]
        return {"predictions": {"target": target, "target_scale": scale}}

    def sample(
        self, conditions: Conditions, num_draws: int, generator: torch.Generator
    ) -> Trajectories:
        """Draw ``num_draws`` prior trajectories for every row of ``conditions``."""
        device = generator.device
        shape = (conditions.num_rows, num_draws)
        coefficients = (
            self.effects.mean
            + self._draw_effects(
                self.effects.genotype,
                self.genotype_ids,
                self.genotype_regression,
                conditions.genotype_ids,
                conditions.markers,
                shape,
                generator,
            )
            + self._draw_effects(
                self.effects.yearsite,
                self.yearsite_ids,
                self.yearsite_regression,
                conditions.yearsite_ids,
                conditions.thermal_time,
                shape,
                generator,
            )
            + sample_gaussian(self.effects.residual_covariance, shape, generator)
        )
        days = self.spline.growth_days
        daily = self.spline.heights(coefficients, days)
        growth_end, plateau_height = self.geometry.growth_end(daily, days)
        noise = torch.randn(daily.shape, generator=generator, device=device)
        peak = running_mean(daily + self.noise_scale * noise).max(dim=-1).values
        probability = torch.sigmoid(
            self.lodging_intercept + self.lodging_slope * peak.clamp(min=0)
        )
        lodged = torch.rand(shape, generator=generator, device=device) < probability
        drawn = torch.randint(
            len(self.drop_triples), shape, generator=generator, device=device
        )
        delay, severity, transition = self.drop_triples[drawn].unbind(-1)
        lodging_day = growth_end + delay
        start_index = (
            (lodging_day - self.spline.start_day).long().clamp(max=len(days) - 1)
        )
        return Trajectories(
            coefficients=coefficients,
            lodged=lodged,
            growth_end=growth_end,
            plateau_height=plateau_height,
            lodging_day=lodging_day,
            start_height=daily.gather(-1, start_index[..., None])[..., 0],
            final_height=severity * plateau_height,
            transition=transition,
        )

    def condition(
        self,
        conditions: Conditions,
        context_days: torch.Tensor,
        context_mask: torch.Tensor,
        context_values: torch.Tensor,
        num_draws: int,
        pool_size: int,
        generator: torch.Generator,
        max_pool_trajectories: int = 2**20,
    ) -> tuple[Trajectories, torch.Tensor]:
        """Importance-sampled posterior draws given the context heights.

        ``context_mask`` (B, D) marks each row's context among ``context_days``
        (D,), with heights ``context_values`` (B, D). A prior pool of
        ``pool_size`` trajectories per row is weighted by the Gaussian
        likelihood ``exp(-SSE / (2 noise_scale²))`` and resampled. Returns the
        draws and the effective sample size of every row. Rows are processed in
        chunks of at most ``max_pool_trajectories`` pool trajectories.
        """
        device = generator.device
        draws, sample_sizes = [], []
        rows_per_chunk = max(1, max_pool_trajectories // pool_size)
        for start in range(0, conditions.num_rows, rows_per_chunk):
            stop = start + rows_per_chunk
            pool = self.sample(conditions.rows(start, stop), pool_size, generator)
            error = pool.heights(self.spline, context_days) - context_values[
                start:stop, None
            ].to(device)
            squared_error = (
                (error * context_mask[start:stop, None].to(device)).square().sum(dim=-1)
            )
            weights = torch.softmax(-squared_error / (2 * self.variance), dim=1)
            drawn = torch.multinomial(
                weights, num_draws, replacement=True, generator=generator
            )
            draws.append(pool.take(drawn))
            sample_sizes.append(1 / weights.square().sum(dim=1))
        return Trajectories.cat(draws), torch.cat(sample_sizes)

    def _draw_effects(
        self,
        table: torch.Tensor,
        known_ids: list[str],
        regression: EffectRegression,
        row_ids: list[str] | None,
        covariates: torch.Tensor | None,
        shape: tuple[int, int],
        generator: torch.Generator,
    ) -> torch.Tensor:
        """(B, N, K) effects: random training effects when ``row_ids`` is None,
        otherwise the fitted effect of a known id and for an unseen id the
        ``unseen_effect`` from the row's ``covariates``. An unseen id without
        covariates (a NaN row) gets random training effects."""
        device = generator.device
        drawn = torch.randint(len(table), shape, generator=generator, device=device)
        if row_ids is None:
            return table[drawn]
        index = {identifier: row for row, identifier in enumerate(known_ids)}
        known = torch.tensor([identifier in index for identifier in row_ids])
        drawn[known.to(device)] = torch.tensor(
            [index[identifier] for identifier in row_ids if identifier in index],
            dtype=torch.long,
            device=device,
        )[:, None]
        effects = table[drawn]
        if self.unseen_effect is UnseenEffect.REGRESSION and not known.all():
            assert covariates is not None
            predicted = ~known & covariates.isfinite().all(dim=1).cpu()
            if predicted.any():
                effects[predicted.to(device)] = regression.sample(
                    covariates[predicted.to(covariates.device)], shape[1], generator
                )
        return effects
