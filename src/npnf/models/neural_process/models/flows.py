from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, TypedDict

import einops as EO
import tensordict
import torch
import zuko.flows
import zuko.transforms
from torch import nn
from torch.distributions import Independent, Normal
from torchvision.ops import MLP

from npnf.models.neural_process.models.neural_process import ANP, LNP


class CouplingNSFDistribution:
    """Wrapper to match zuko's distribution interface expected by _run_flow."""

    def __init__(self, transform: "CouplingNSFTransform"):
        self.transform = transform


class CouplingNSFTransform(zuko.transforms.Transform):
    """Transform produced by CouplingNSF.

    Coupling-based spline transform with symmetric forward/inverse speed.
    Unlike autoregressive transforms, both directions are O(L) where L is layers.
    """

    def __init__(self, flow: "CouplingNSF", context: torch.Tensor | None):
        super().__init__()
        self.flow = flow
        self.context = context

    def _get_spline_params(
        self, x_identity: torch.Tensor, layer_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get spline parameters from conditioner network."""
        conditioner = self.flow.conditioners[layer_index]

        if self.context is not None:
            inp = torch.cat([x_identity, self.context], dim=-1)
        else:
            inp = x_identity

        out = conditioner(inp)

        if layer_index % 2 == 0:
            out_features = self.flow.features - self.flow.features // 2
        else:
            out_features = self.flow.features // 2

        out = out.view(*out.shape[:-1], out_features, 3 * self.flow.num_bins + 1)

        widths = (
            out[..., : self.flow.num_bins].softmax(dim=-1) * 2 * self.flow.tail_bound
        )
        heights = (
            out[..., self.flow.num_bins : 2 * self.flow.num_bins].softmax(dim=-1)
            * 2
            * self.flow.tail_bound
        )
        derivatives = out[..., 2 * self.flow.num_bins :].exp()

        return widths, heights, derivatives

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base -> target (sampling direction)."""
        for i in range(len(self.flow.conditioners)):
            if i % 2 == 0:
                x_id = x[..., : self.flow.features // 2]
                x_tr = x[..., self.flow.features // 2 :]
            else:
                x_id = x[..., self.flow.features // 2 :]
                x_tr = x[..., : self.flow.features // 2]

            widths, heights, derivatives = self._get_spline_params(x_id, i)
            spline = zuko.transforms.MonotonicRQSTransform(
                widths, heights, derivatives, bound=self.flow.tail_bound
            )
            y_tr = spline(x_tr)

            if i % 2 == 0:
                x = torch.cat([x_id, y_tr], dim=-1)
            else:
                x = torch.cat([y_tr, x_id], dim=-1)

        return x

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass: target -> base (encoding direction) - equally fast."""
        for i in reversed(range(len(self.flow.conditioners))):
            if i % 2 == 0:
                y_id = y[..., : self.flow.features // 2]
                y_tr = y[..., self.flow.features // 2 :]
            else:
                y_id = y[..., self.flow.features // 2 :]
                y_tr = y[..., : self.flow.features // 2]

            widths, heights, derivatives = self._get_spline_params(y_id, i)
            spline = zuko.transforms.MonotonicRQSTransform(
                widths, heights, derivatives, bound=self.flow.tail_bound
            )
            x_tr = spline.inv(y_tr)

            if i % 2 == 0:
                y = torch.cat([y_id, x_tr], dim=-1)
            else:
                y = torch.cat([x_tr, y_id], dim=-1)

        return y

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log absolute determinant of Jacobian."""
        ladj = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)

        current = x
        for i in range(len(self.flow.conditioners)):
            if i % 2 == 0:
                x_id = current[..., : self.flow.features // 2]
                x_tr = current[..., self.flow.features // 2 :]
            else:
                x_id = current[..., self.flow.features // 2 :]
                x_tr = current[..., : self.flow.features // 2]

            widths, heights, derivatives = self._get_spline_params(x_id, i)
            spline = zuko.transforms.MonotonicRQSTransform(
                widths, heights, derivatives, bound=self.flow.tail_bound
            )

            y_tr = spline(x_tr)
            ladj = ladj + spline.log_abs_det_jacobian(x_tr, y_tr).sum(dim=-1)

            if i % 2 == 0:
                current = torch.cat([x_id, y_tr], dim=-1)
            else:
                current = torch.cat([y_tr, x_id], dim=-1)

        return ladj


class CouplingNSF(nn.Module):
    """Coupling-based Neural Spline Flow with symmetric forward/inverse speed.

    Uses simple MLP conditioners (like zuko's autoregressive NSF) combined with
    zuko's efficient spline transforms. Unlike autoregressive NSF, both forward
    and inverse are O(L) where L is the number of layers.

    Args:
        features: Dimension of the input/output
        context: Dimension of the conditioning context
        num_layers: Number of coupling layers
        hidden_dim: Hidden dimension of the conditioner networks
        num_bins: Number of spline bins
        tail_bound: Bound of the spline tails
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        num_layers: int = 6,
        hidden_dim: int = 64,
        num_bins: int = 8,
        tail_bound: float = 5.0,
    ):
        super().__init__()
        self.features = features
        self.context_features = context if context > 0 else None
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        self.conditioners = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                in_features = features // 2
                out_features = features - features // 2
            else:
                in_features = features - features // 2
                out_features = features // 2

            param_dim = out_features * (3 * num_bins + 1)
            in_dim = in_features + (context if context > 0 else 0)

            self.conditioners.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, param_dim),
                )
            )

    def forward(self, context: torch.Tensor | None = None) -> CouplingNSFDistribution:
        """Returns a distribution-like object with transform conditioned on context."""
        return CouplingNSFDistribution(CouplingNSFTransform(self, context))


def build_prior_flow(features: int, context: int = 0) -> CouplingNSF:
    """Build coupling NSF for prior (fast inverse for KL computation)."""
    return CouplingNSF(
        features=features,
        context=context,
        num_layers=6,
        hidden_dim=64,
        num_bins=8,
        tail_bound=5.0,
    )


def build_posterior_flow(features: int, context: int = 0) -> zuko.flows.Flow:
    """Build autoregressive NSF for posterior (only needs fast forward)."""
    return zuko.flows.NSF(features=features, context=context)


def _run_flow(
    flow: zuko.flows.Flow | CouplingNSF,
    z: torch.Tensor,
    context: torch.Tensor | None = None,
    *,
    inverse: bool = False,
    accumulate_logdet: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    distribution = flow() if context is None else flow(context.float())
    transform = distribution.transform

    if inverse:
        z_transformed = transform.inv(z.float())
        if accumulate_logdet:
            logdet = -transform.log_abs_det_jacobian(z_transformed, z.float())
            return z_transformed, logdet
        return z_transformed, None

    z_transformed = transform(z.float())
    if accumulate_logdet:
        logdet = transform.log_abs_det_jacobian(z.float(), z_transformed)
        return z_transformed, logdet
    return z_transformed, None


def transform_sample(
    z: torch.Tensor,
    flow: zuko.flows.Flow | CouplingNSF,
    inverse: bool,
    condition: torch.Tensor | None = None,
    accumulate_logdet: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, num_samples, *_ = z.shape
    z_flat = EO.rearrange(z, "b s 1 h -> (b s 1) h")

    z_transformed, log_det = _run_flow(
        flow=flow,
        z=z_flat,
        context=EO.repeat(condition, "b 1 h -> (b s) h", s=num_samples)
        if condition is not None
        else None,
        inverse=inverse,
        accumulate_logdet=accumulate_logdet,
    )
    z_transformed = EO.rearrange(
        z_transformed, "(b s 1) h -> b s 1 h", b=B, s=num_samples
    )

    if not accumulate_logdet:
        return z_transformed, None

    log_det = EO.rearrange(log_det, "(b s 1) -> b s 1", b=B, s=num_samples)
    return z_transformed, log_det


class _FlowInputs(TypedDict):
    context_priors: tensordict.TensorDict
    targets: tensordict.TensorDict
    context_posteriors: tensordict.TensorDict | None
    context_posteriors_has_y: bool
    temperatures: torch.Tensor
    markers: list[torch.Tensor]


class _FlowProcessMixin(LNP):
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        state_dict = self._map_legacy_posterior_flow_base_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _map_legacy_posterior_flow_base_keys(
        self, state_dict: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        key_mapping = {
            "posterior_flow.base._0": "posterior_flow.base.loc",
            "posterior_flow.base._1": "posterior_flow.base.scale",
        }
        if not any(key in state_dict for key in key_mapping):
            return state_dict

        mapped_state_dict = OrderedDict(state_dict)
        for old_key, new_key in key_mapping.items():
            if old_key in mapped_state_dict and new_key not in mapped_state_dict:
                mapped_state_dict[new_key] = mapped_state_dict[old_key]
            mapped_state_dict.pop(old_key, None)

        if hasattr(state_dict, "_metadata"):
            mapped_state_dict._metadata = state_dict._metadata  # noqa: SLF001  # ty: ignore[unresolved-attribute]

        return mapped_state_dict

    def _prepare_flow_inputs(
        self,
        context_priors: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        context_posteriors: tensordict.TensorDict | None,
        temperatures: tensordict.TensorDict | None,
        markers: tensordict.TensorDict | None,
    ) -> _FlowInputs:
        context_priors, _ = self._project_points(context_priors)
        targets, _ = self._project_points(targets)
        context_posteriors_has_y = False
        if context_posteriors is not None:
            context_posteriors, context_posteriors_has_y = self._project_points(
                context_posteriors
            )
        temperature_tokens = self._prepare_temperatures(
            temperatures=temperatures, reference=context_priors
        )
        marker_tokens = self._prepare_markers(markers=markers, reference=context_priors)

        return {
            "context_priors": context_priors,
            "targets": targets,
            "context_posteriors": context_posteriors,
            "context_posteriors_has_y": context_posteriors_has_y,
            "temperatures": temperature_tokens,
            "markers": marker_tokens,
        }

    def _encode_posterior_latents(
        self,
        contexts: tensordict.TensorDict,
        temperatures: torch.Tensor,
        markers: list[torch.Tensor],
        prefix_token_keys: list[str],
    ) -> tensordict.TensorDict:
        return self._encode(
            contexts=contexts,
            targets=None,
            temperatures=temperatures,
            markers=markers,
            prefix_token_keys=prefix_token_keys,
        )

    def _decode_flow_predictions(
        self,
        z: torch.Tensor,
        prior_latents: tensordict.TensorDict,
        targets: tensordict.TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.latent_decoder(z)
        decode_targets = self._get_decode_targets(
            prior_latents=prior_latents, targets=targets
        )
        combined = self.combine_paths(z=z, targets=decode_targets)
        mus = self._predict_mean(combined, encoder=self.decoder_mu)
        variances = self._predict_variance(combined)
        return mus, variances

    def _format_flow_predictions(
        self, mus: torch.Tensor, variances: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "target": mus.transpose(1, 2),
            "target_scale": variances.sqrt().transpose(1, 2),
        }


class _PriorFlowMixin(_FlowProcessMixin):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_transformer_layers: int = 12,
        num_transformer_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_transformer_layers=num_transformer_layers,
            num_transformer_heads=num_transformer_heads,
            **kwargs,
        )

        self.prior_flow = build_prior_flow(
            features=self.latent_dim, context=self.latent_dim
        )
        self.prior_context_projection = MLP(
            in_channels=self.hidden_dim,
            hidden_channels=[self.hidden_dim, self.latent_dim],
            activation_layer=nn.SiLU,
            bias=True,
        )

    def forward(  # ty: ignore[invalid-method-override]
        self,
        context_priors: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        context_posteriors: tensordict.TensorDict | None = None,
        temperatures: tensordict.TensorDict | None = None,
        markers: tensordict.TensorDict | None = None,
        get_loss: bool = True,
        get_samples: bool = False,
        num_samples: int = 8,
        sample_from_posterior: bool = False,
        kl_weight: float = 1.0,
        noise_global: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        prepared = self._prepare_flow_inputs(
            context_priors=context_priors,
            targets=targets,
            context_posteriors=context_posteriors,
            temperatures=temperatures,
            markers=markers,
        )
        context_priors = prepared["context_priors"]
        targets = prepared["targets"]
        context_posteriors = prepared["context_posteriors"]
        context_posteriors_has_y = prepared["context_posteriors_has_y"]
        temperature_tokens = prepared["temperatures"]
        marker_tokens = prepared["markers"]

        prior_latents = self._encode_prior_latents(
            contexts=context_priors,
            targets=targets,
            temperatures=temperature_tokens,
            markers=marker_tokens,
            prefix_token_keys=["global", "flow_condition"],
        )

        if not get_samples and not get_loss:
            return {"prior": {"latents": prior_latents}}

        prior = self.predict_distribution(
            prior_latents["global"],  # ty: ignore[invalid-argument-type]
            encoder=self.latent_encoder,
        )
        z0_prior = self.sample_latent(
            prior, num_samples=num_samples, noise=noise_global
        )
        prior_flow_condition = self.prior_context_projection(
            prior_latents["flow_condition"]
        )

        z_prior = None
        if not self.training:
            z_prior, _ = transform_sample(
                z0_prior,
                flow=self.prior_flow,
                condition=prior_flow_condition,
                inverse=False,
                accumulate_logdet=self.training,
            )

        if context_posteriors is not None and context_posteriors_has_y:
            posterior_latents = self._encode_posterior_latents(
                contexts=context_posteriors,
                temperatures=temperature_tokens,
                markers=marker_tokens,
                prefix_token_keys=["global"],
            )
            posterior = self.predict_distribution(
                posterior_latents["global"],  # ty: ignore[invalid-argument-type]
                encoder=self.latent_encoder,
            )
            z_posterior = self.sample_latent(
                posterior, num_samples=num_samples, noise=noise_global
            )

            z_posterior_inv, logdet_p_inv = transform_sample(
                z_posterior,
                flow=self.prior_flow,
                condition=prior_flow_condition,
                inverse=True,
                accumulate_logdet=self.training,
            )

        z = z_posterior if sample_from_posterior else z_prior
        assert z is not None
        mus, variances = self._decode_flow_predictions(
            z=z, prior_latents=prior_latents, targets=targets
        )

        output = {
            "prior": {"latents": prior_latents, "distribution": prior, "z": z_prior}
        }
        if context_posteriors is not None and context_posteriors_has_y:
            output["posterior"] = {
                "latents": posterior_latents,
                "distribution": posterior,
                "z": z_posterior,
            }

        if get_samples:
            output["predictions"] = self._format_flow_predictions(
                mus=mus, variances=variances
            )

        if get_loss:
            assert logdet_p_inv is not None
            output["loss_dict"] = self.get_loss(
                mus=mus,
                targets=targets,
                prior=prior,
                posterior=posterior,
                z_posterior=z_posterior,
                z_posterior_inv=z_posterior_inv,
                logdet_p_inv=logdet_p_inv,
                variances=variances,
                kl_weight=kl_weight,
            )

        return output

    def get_loss(  # ty: ignore[invalid-method-override]
        self,
        mus: torch.Tensor,
        targets: tensordict.TensorDict,
        prior: Independent,
        posterior: Independent,
        z_posterior: torch.Tensor,
        z_posterior_inv: torch.Tensor,
        logdet_p_inv: torch.Tensor,
        variances: torch.Tensor,
        kl_weight: float = 1.0,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        likelihood_loss = self.get_likelihood_loss(
            mus=mus, targets=targets, variances=variances
        )
        kl_loss, kl_debug = self.get_kl_loss(
            prior=prior,
            posterior=posterior,
            z_posterior=z_posterior,
            z_posterior_inv=z_posterior_inv,
            logdet_p_inv=logdet_p_inv,
        )

        total_loss = likelihood_loss + kl_weight * kl_loss

        return {
            "loss": total_loss,
            "kl": kl_loss,
            "likelihood": likelihood_loss,
            **kl_debug,
        }

    def get_kl_loss(  # ty: ignore[invalid-method-override]
        self,
        prior: Independent,
        posterior: Independent,
        z_posterior: torch.Tensor,
        z_posterior_inv: torch.Tensor,
        logdet_p_inv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """KL divergence for prior-only flow.

        KL[q||p] = E_q[log q(z) - log p0(f^{-1}(z)) - log|det J^{-1}|]
        """
        posterior_samples = Independent(
            Normal(posterior.mean[:, None], posterior.stddev[:, None]), 1
        )
        prior_samples = Independent(
            Normal(prior.mean[:, None], prior.stddev[:, None]), 1
        )

        log_q = posterior_samples.log_prob(z_posterior).mean(dim=1)
        log_p0 = prior_samples.log_prob(z_posterior_inv).mean(dim=1)
        logdet_p = logdet_p_inv.mean(dim=1)

        kl_loss = (log_q - log_p0 - logdet_p).mean()

        debug_metrics = {
            "prior_std": prior.stddev.mean(),
            "posterior_std": posterior.stddev.mean(),
            "log_q": log_q.mean(),
            "log_p0": log_p0.mean(),
            "logdet_p": logdet_p.mean(),
        }

        return kl_loss, debug_metrics


class _PosteriorFlowMixin(_FlowProcessMixin):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_transformer_layers: int = 12,
        num_transformer_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_transformer_layers=num_transformer_layers,
            num_transformer_heads=num_transformer_heads,
            **kwargs,
        )

        self.posterior_flow = build_posterior_flow(
            features=self.latent_dim, context=self.latent_dim
        )
        self.posterior_context_projection = MLP(
            in_channels=self.hidden_dim,
            hidden_channels=[self.hidden_dim, self.latent_dim],
            activation_layer=nn.SiLU,
            bias=True,
        )

    def forward(  # ty: ignore[invalid-method-override]
        self,
        context_priors: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        context_posteriors: tensordict.TensorDict | None = None,
        temperatures: tensordict.TensorDict | None = None,
        markers: tensordict.TensorDict | None = None,
        get_loss: bool = True,
        get_samples: bool = False,
        num_samples: int = 8,
        sample_from_posterior: bool = False,
        kl_weight: float = 1.0,
        noise_global: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        prepared = self._prepare_flow_inputs(
            context_priors=context_priors,
            targets=targets,
            context_posteriors=context_posteriors,
            temperatures=temperatures,
            markers=markers,
        )
        context_priors = prepared["context_priors"]
        targets = prepared["targets"]
        context_posteriors = prepared["context_posteriors"]
        context_posteriors_has_y = prepared["context_posteriors_has_y"]
        temperature_tokens = prepared["temperatures"]
        marker_tokens = prepared["markers"]

        prior_latents = self._encode_prior_latents(
            contexts=context_priors,
            targets=targets,
            temperatures=temperature_tokens,
            markers=marker_tokens,
            prefix_token_keys=["global"],
        )

        if not get_samples and not get_loss:
            return {"prior": {"latents": prior_latents}}

        prior = self.predict_distribution(
            prior_latents["global"],  # ty: ignore[invalid-argument-type]
            encoder=self.latent_encoder,
        )
        z_prior = self.sample_latent(prior, num_samples=num_samples, noise=noise_global)

        if context_posteriors is not None and context_posteriors_has_y:
            posterior_latents = self._encode_posterior_latents(
                contexts=context_posteriors,
                temperatures=temperature_tokens,
                markers=marker_tokens,
                prefix_token_keys=["global", "flow_condition"],
            )
            posterior = self.predict_distribution(
                posterior_latents["global"],  # ty: ignore[invalid-argument-type]
                encoder=self.latent_encoder,
            )
            z0_posterior = self.sample_latent(
                posterior, num_samples=num_samples, noise=noise_global
            )

            posterior_flow_condition = self.posterior_context_projection(
                posterior_latents["flow_condition"]
            )
            z_posterior, logdet_q = transform_sample(
                z0_posterior,
                flow=self.posterior_flow,
                condition=posterior_flow_condition,
                inverse=False,
                accumulate_logdet=self.training,
            )

        z = z_posterior if sample_from_posterior else z_prior
        mus, variances = self._decode_flow_predictions(
            z=z, prior_latents=prior_latents, targets=targets
        )

        output = {
            "prior": {"latents": prior_latents, "distribution": prior, "z": z_prior}
        }
        if context_posteriors is not None and context_posteriors_has_y:
            output["posterior"] = {
                "latents": posterior_latents,
                "distribution": posterior,
                "z": z_posterior,
            }

        if get_samples:
            output["predictions"] = self._format_flow_predictions(
                mus=mus, variances=variances
            )

        if get_loss:
            assert logdet_q is not None
            output["loss_dict"] = self.get_loss(
                mus=mus,
                targets=targets,
                prior=prior,
                posterior=posterior,
                z0_posterior=z0_posterior,
                z_posterior=z_posterior,
                logdet_q=logdet_q,
                variances=variances,
                kl_weight=kl_weight,
            )

        return output

    def get_loss(  # ty: ignore[invalid-method-override]
        self,
        mus: torch.Tensor,
        targets: tensordict.TensorDict,
        prior: Independent,
        posterior: Independent,
        z0_posterior: torch.Tensor,
        z_posterior: torch.Tensor,
        logdet_q: torch.Tensor,
        variances: torch.Tensor,
        kl_weight: float = 1.0,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        likelihood_loss = self.get_likelihood_loss(
            mus=mus, targets=targets, variances=variances
        )
        kl_loss, kl_debug = self.get_kl_loss(
            prior=prior,
            posterior=posterior,
            z0_posterior=z0_posterior,
            z_posterior=z_posterior,
            logdet_q=logdet_q,
        )

        total_loss = likelihood_loss + kl_weight * kl_loss

        return {
            "loss": total_loss,
            "kl": kl_loss,
            "likelihood": likelihood_loss,
            **kl_debug,
        }

    def get_kl_loss(  # ty: ignore[invalid-method-override]
        self,
        prior: Independent,
        posterior: Independent,
        z0_posterior: torch.Tensor,
        z_posterior: torch.Tensor,
        logdet_q: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """KL divergence for posterior-only flow.

        KL[q||p] = E_q0[log q0(z0) - log|det J| - log p(f(z0))]
        """
        posterior_samples = Independent(
            Normal(posterior.mean[:, None], posterior.stddev[:, None]), 1
        )
        prior_samples = Independent(
            Normal(prior.mean[:, None], prior.stddev[:, None]), 1
        )

        log_q0 = posterior_samples.log_prob(z0_posterior).mean(dim=1)
        log_p = prior_samples.log_prob(z_posterior).mean(dim=1)
        logdet_q_mean = logdet_q.mean(dim=1)

        kl_loss = (log_q0 - logdet_q_mean - log_p).mean()

        debug_metrics = {
            "prior_std": prior.stddev.mean(),
            "posterior_std": posterior.stddev.mean(),
            "log_q0": log_q0.mean(),
            "log_p": log_p.mean(),
            "logdet_q": logdet_q_mean.mean(),
        }

        return kl_loss, debug_metrics


class _PriorPosteriorFlowMixin(_PriorFlowMixin, _PosteriorFlowMixin):
    def forward(
        self,
        context_priors: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        context_posteriors: tensordict.TensorDict | None = None,
        temperatures: tensordict.TensorDict | None = None,
        markers: tensordict.TensorDict | None = None,
        get_loss: bool = True,
        get_samples: bool = False,
        num_samples: int = 8,
        sample_from_posterior: bool = False,
        kl_weight: float = 1.0,
        noise_global: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        prepared = self._prepare_flow_inputs(
            context_priors=context_priors,
            targets=targets,
            context_posteriors=context_posteriors,
            temperatures=temperatures,
            markers=markers,
        )
        context_priors = prepared["context_priors"]
        targets = prepared["targets"]
        context_posteriors = prepared["context_posteriors"]
        context_posteriors_has_y = prepared["context_posteriors_has_y"]
        temperature_tokens = prepared["temperatures"]
        marker_tokens = prepared["markers"]

        prior_latents = self._encode_prior_latents(
            contexts=context_priors,
            targets=targets,
            temperatures=temperature_tokens,
            markers=marker_tokens,
            prefix_token_keys=["global", "flow_condition"],
        )

        if not get_samples and not get_loss:
            return {"prior": {"latents": prior_latents}}

        prior = self.predict_distribution(
            prior_latents["global"],  # ty: ignore[invalid-argument-type]
            encoder=self.latent_encoder,
        )
        z0_prior = self.sample_latent(
            prior, num_samples=num_samples, noise=noise_global
        )
        prior_flow_condition = self.prior_context_projection(
            prior_latents["flow_condition"]
        )

        z_prior = None
        if not self.training:
            z_prior, _ = transform_sample(
                z0_prior,
                flow=self.prior_flow,
                condition=prior_flow_condition,
                inverse=False,
                accumulate_logdet=self.training,
            )

        if context_posteriors is not None and context_posteriors_has_y:
            posterior_latents = self._encode_posterior_latents(
                contexts=context_posteriors,
                temperatures=temperature_tokens,
                markers=marker_tokens,
                prefix_token_keys=["global", "flow_condition"],
            )
            posterior = self.predict_distribution(
                posterior_latents["global"],  # ty: ignore[invalid-argument-type]
                encoder=self.latent_encoder,
            )
            z0_posterior = self.sample_latent(
                posterior, num_samples=num_samples, noise=noise_global
            )

            posterior_flow_condition = self.posterior_context_projection(
                posterior_latents["flow_condition"]
            )
            z_posterior, logdet_q = transform_sample(
                z0_posterior,
                flow=self.posterior_flow,
                condition=posterior_flow_condition,
                inverse=False,
                accumulate_logdet=self.training,
            )

            z_posterior_inv, logdet_p_inv = transform_sample(
                z_posterior,
                flow=self.prior_flow,
                condition=prior_flow_condition,
                inverse=True,
                accumulate_logdet=self.training,
            )

        z = z_posterior if sample_from_posterior else z_prior
        assert z is not None
        mus, variances = self._decode_flow_predictions(
            z=z, prior_latents=prior_latents, targets=targets
        )

        output = {
            "prior": {"latents": prior_latents, "distribution": prior, "z": z_prior}
        }
        if context_posteriors is not None and context_posteriors_has_y:
            output["posterior"] = {
                "latents": posterior_latents,
                "distribution": posterior,
                "z": z_posterior,
            }

        if get_samples:
            output["predictions"] = self._format_flow_predictions(
                mus=mus, variances=variances
            )

        if get_loss:
            assert logdet_q is not None
            assert logdet_p_inv is not None
            output["loss_dict"] = self.get_loss(
                mus=mus,
                targets=targets,
                prior=prior,
                posterior=posterior,
                z0_posterior=z0_posterior,
                z_posterior_inv=z_posterior_inv,
                logdet_q=logdet_q,
                logdet_p_inv=logdet_p_inv,
                variances=variances,
                kl_weight=kl_weight,
            )

        return output

    def get_loss(  # ty: ignore[invalid-method-override]
        self,
        mus: torch.Tensor,
        targets: tensordict.TensorDict,
        prior: Independent,
        posterior: Independent,
        z0_posterior: torch.Tensor,
        z_posterior_inv: torch.Tensor,
        logdet_q: torch.Tensor,
        logdet_p_inv: torch.Tensor,
        variances: torch.Tensor,
        kl_weight: float = 1.0,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        likelihood_loss = self.get_likelihood_loss(
            mus=mus, targets=targets, variances=variances
        )
        kl_loss, kl_debug = self.get_kl_loss(
            prior=prior,
            posterior=posterior,
            z0_posterior=z0_posterior,
            z_posterior_inv=z_posterior_inv,
            logdet_q=logdet_q,
            logdet_p_inv=logdet_p_inv,
        )

        total_loss = likelihood_loss + kl_weight * kl_loss

        return {
            "loss": total_loss,
            "kl": kl_loss,
            "likelihood": likelihood_loss,
            **kl_debug,
        }

    def get_kl_loss(  # ty: ignore[invalid-method-override]
        self,
        prior: Independent,
        posterior: Independent,
        z0_posterior: torch.Tensor,
        z_posterior_inv: torch.Tensor,
        logdet_q: torch.Tensor,
        logdet_p_inv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """KL divergence for separate prior and posterior flows.

        KL[q||p] = E_q0[log q0(z0) - log p0(f_p^{-1}(f_q(z0)))
                        - logdet_q - logdet_p_inv]
        """
        prior_samples = Independent(
            Normal(prior.mean[:, None], prior.stddev[:, None]), 1
        )
        posterior_samples = Independent(
            Normal(posterior.mean[:, None], posterior.stddev[:, None]), 1
        )

        log_q0 = posterior_samples.log_prob(z0_posterior).mean(dim=1)
        log_p0 = prior_samples.log_prob(z_posterior_inv).mean(dim=1)

        logdet_q_mean = logdet_q.mean(dim=1)
        logdet_p_mean = logdet_p_inv.mean(dim=1)

        kl_loss = (log_q0 - log_p0 - logdet_q_mean - logdet_p_mean).mean()

        debug_metrics = {
            "prior_std": prior.stddev.mean(),
            "posterior_std": posterior.stddev.mean(),
            "log_q0": log_q0.mean(),
            "log_p0": log_p0.mean(),
            "logdet_q": logdet_q_mean.mean(),
            "logdet_p": logdet_p_mean.mean(),
        }

        return kl_loss, debug_metrics


class ANP_NF_Prior(_PriorFlowMixin, _FlowProcessMixin, ANP):  # noqa: N801
    pass


class ANP_NF_Posterior(_PosteriorFlowMixin, _FlowProcessMixin, ANP):  # noqa: N801
    pass


class ANP_NF_Prior_Posterior(  # noqa: N801
    _PriorPosteriorFlowMixin, ANP_NF_Prior, ANP_NF_Posterior
):
    pass


class LNP_NF_Prior(_PriorFlowMixin, _FlowProcessMixin, LNP):  # noqa: N801
    pass


class LNP_NF_Posterior(_PosteriorFlowMixin, _FlowProcessMixin, LNP):  # noqa: N801
    pass


class LNP_NF_Prior_Posterior(  # noqa: N801
    _PriorPosteriorFlowMixin, LNP_NF_Prior, LNP_NF_Posterior
):
    pass
