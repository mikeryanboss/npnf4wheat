from typing import Any, Literal

import einops as EO
import tensordict
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.nn.functional import gaussian_nll_loss
from torchvision.ops import MLP

from npnf.models.neural_process.base import TransformerProcessBase


class CNP(TransformerProcessBase):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        activation_layer: type[nn.Module] = nn.SiLU,
        num_latent_encoder_layers: int = 2,
        num_latent_decoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_transformer_layers: int = 12,
        num_transformer_heads: int = 4,
        noise_type: Literal["homoscedastic", "heteroscedastic"] = "homoscedastic",
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            activation_layer=activation_layer,
            num_transformer_layers=num_transformer_layers,
            num_transformer_heads=num_transformer_heads,
            **kwargs,
        )

        self.noise_type = noise_type

        self.latent_encoder = MLP(
            in_channels=hidden_dim,
            hidden_channels=[
                *([hidden_dim] * (num_latent_encoder_layers - 1)),
                2 * latent_dim,
            ],
            activation_layer=activation_layer,
            bias=True,
        )
        self.latent_decoder = MLP(
            in_channels=latent_dim,
            hidden_channels=[
                *([hidden_dim] * (num_latent_decoder_layers - 1)),
                hidden_dim,
            ],
            activation_layer=activation_layer,
            bias=False,
        )

        self.decoder_mu = nn.Sequential(
            MLP(
                in_channels=hidden_dim,
                hidden_channels=[hidden_dim] * num_decoder_layers,
                activation_layer=activation_layer,
                bias=False,
            ),
            activation_layer(),
            nn.Linear(hidden_dim, 1),
        )

        if noise_type == "heteroscedastic":
            self.decoder_var = nn.Sequential(
                MLP(
                    in_channels=hidden_dim,
                    hidden_channels=[hidden_dim] * num_decoder_layers,
                    activation_layer=activation_layer,
                    bias=False,
                ),
                activation_layer(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.var = nn.Parameter(torch.empty(1))
            self._init_parameter(self.var)

    @property
    def variance(self):
        if self.noise_type == "heteroscedastic":
            msg = "Use _predict_variance() for heteroscedastic models"
            raise RuntimeError(msg)
        return torch.nn.functional.softplus(self.var) + 1e-8

    def _predict_variance(self, combined: torch.Tensor) -> torch.Tensor:
        """Predict variance from combined representation."""
        if self.noise_type == "heteroscedastic":
            raw_var = self.decoder_var(combined)
            return torch.nn.functional.softplus(raw_var) + 1e-8
        # Use ones_like to match nested tensor structure instead of expand
        return self.variance * torch.ones_like(combined[..., :1])

    def forward(
        self,
        context_priors: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        temperatures: tensordict.TensorDict | None = None,
        markers: tensordict.TensorDict | None = None,
        get_loss: bool = True,
        get_samples: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        context_priors, _ = self._project_points(context_priors)
        targets, _ = self._project_points(targets)
        temperature_tokens = self._prepare_temperatures(
            temperatures=temperatures, reference=context_priors
        )
        marker_tokens = self._prepare_markers(markers=markers, reference=context_priors)

        latents = self._encode_prior_latents(
            contexts=context_priors,
            targets=targets,
            temperatures=temperature_tokens,
            markers=marker_tokens,
        )

        if not get_samples and not get_loss:
            return {"prior": {"latents": latents}}

        # to make it similar between models
        z = self.latent_encoder(latents["global"])
        z, _ = torch.chunk(z, 2, dim=-1)
        z = self.latent_decoder(z)

        decode_targets = self._get_decode_targets(
            prior_latents=latents, targets=targets
        )
        combined = self.combine_paths(z=z, targets=decode_targets)
        mus = self._predict_mean(combined, encoder=self.decoder_mu)
        variances = self._predict_variance(combined)

        output = {}
        if get_loss:
            output["loss_dict"] = self.get_loss(
                mus=mus, targets=targets, variances=variances
            )
        if get_samples:
            output["predictions"] = {
                "target": mus.unsqueeze(1),
                "target_scale": variances.sqrt().unsqueeze(1),
            }

        return output

    def _encode_prior_latents(
        self,
        contexts: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        temperatures: torch.Tensor,
        markers: list[torch.Tensor],
        prefix_token_keys: list[str] | None = None,
    ) -> tensordict.TensorDict:
        return self._encode(
            contexts=contexts,
            targets=None,
            temperatures=temperatures,
            markers=markers,
            prefix_token_keys=prefix_token_keys or ["global"],
        )

    def _get_decode_targets(
        self, prior_latents: tensordict.TensorDict, targets: tensordict.TensorDict
    ) -> torch.Tensor:
        return targets["X_projected"]  # ty: ignore[invalid-return-type]

    def combine_paths(self, z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_items = list(targets)
        z = torch.nested.as_nested_tensor(
            [
                EO.repeat(rep, "1 h -> t h", t=len(target))
                for rep, target in zip(z, target_items, strict=True)
            ],
            layout=torch.jagged,
        )
        targets = torch.nested.nested_tensor_from_jagged(
            values=torch.cat(target_items, dim=0),
            offsets=z.offsets(),  # ty: ignore[unresolved-attribute]
            min_seqlen=z._min_seqlen,  # noqa: SLF001  # ty: ignore[unresolved-attribute]
            max_seqlen=z._max_seqlen,  # noqa: SLF001  # ty: ignore[unresolved-attribute]
        )

        return z + targets

    def _predict_mean(
        self, parameters: torch.Tensor, encoder: nn.Module
    ) -> torch.Tensor:
        return encoder(parameters)

    def get_loss(
        self, mus: torch.Tensor, targets: tensordict.TensorDict, variances: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        likelihood_loss = self.get_likelihood_loss(
            mus=mus, targets=targets, variances=variances
        )
        return {"loss": likelihood_loss, "likelihood": likelihood_loss}

    def get_likelihood_loss(
        self, mus: torch.Tensor, targets: tensordict.TensorDict, variances: torch.Tensor
    ) -> torch.Tensor:
        nll = torch.as_tensor(0.0, device=mus.device)
        for mu, Y, var in zip(mus, targets["Y"], variances, strict=True):
            nll = nll + gaussian_nll_loss(mu, Y, var)

        return nll / len(mus)


class ACNP(CNP):
    def _encode_prior_latents(
        self,
        contexts: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        temperatures: torch.Tensor,
        markers: list[torch.Tensor],
        prefix_token_keys: list[str] | None = None,
    ) -> tensordict.TensorDict:
        return self._encode(
            contexts=contexts,
            targets=targets,
            temperatures=temperatures,
            markers=markers,
            prefix_token_keys=prefix_token_keys or ["global"],
        )

    def _get_decode_targets(
        self, prior_latents: tensordict.TensorDict, targets: tensordict.TensorDict
    ) -> torch.Tensor:
        return prior_latents["target"]  # ty: ignore[invalid-return-type]


class LNP(CNP):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        activation_layer: type[nn.Module] = nn.SiLU,
        num_latent_encoder_layers: int = 2,
        num_latent_decoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_transformer_layers: int = 12,
        num_transformer_heads: int = 4,
        noise_type: Literal["homoscedastic", "heteroscedastic"] = "homoscedastic",
        free_bits: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            activation_layer=activation_layer,
            num_latent_encoder_layers=num_latent_encoder_layers,
            num_latent_decoder_layers=num_latent_decoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_transformer_layers=num_transformer_layers,
            num_transformer_heads=num_transformer_heads,
            noise_type=noise_type,
            **kwargs,
        )

        self.free_bits = free_bits

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
        noise_z: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        context_priors, _ = self._project_points(context_priors)
        targets, _ = self._project_points(targets)
        if context_posteriors is not None:
            context_posteriors, context_posteriors_has_y = self._project_points(
                context_posteriors
            )
        temperature_tokens = self._prepare_temperatures(
            temperatures=temperatures, reference=context_priors
        )
        marker_tokens = self._prepare_markers(markers=markers, reference=context_priors)

        prior_latents = self._encode_prior_latents(
            contexts=context_priors,
            targets=targets,
            temperatures=temperature_tokens,
            markers=marker_tokens,
        )

        if not get_samples and not get_loss:
            return {"prior": {"latents": prior_latents}}

        prior = self.predict_distribution(
            prior_latents["global"],  # ty: ignore[invalid-argument-type]
            encoder=self.latent_encoder,
        )
        with torch.no_grad():
            z_prior = self.sample_latent(prior, num_samples=num_samples, noise=noise_z)
        if context_posteriors is not None and context_posteriors_has_y:
            posterior_latents = self._encode(
                contexts=context_posteriors,
                targets=None,
                temperatures=temperature_tokens,
                markers=marker_tokens,
                prefix_token_keys=["global"],
            )
            posterior = self.predict_distribution(
                posterior_latents["global"],  # ty: ignore[invalid-argument-type]
                encoder=self.latent_encoder,
            )
            z_posterior = self.sample_latent(
                posterior, num_samples=num_samples, noise=noise_z
            )

        z = z_posterior if sample_from_posterior else z_prior
        z = self.latent_decoder(z)

        decode_targets = self._get_decode_targets(
            prior_latents=prior_latents, targets=targets
        )
        combined = self.combine_paths(z=z, targets=decode_targets)
        mus = self._predict_mean(combined, encoder=self.decoder_mu)
        variances = self._predict_variance(combined)

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
            output["predictions"] = {
                "target": mus.transpose(1, 2),
                "target_scale": variances.sqrt().transpose(1, 2),
            }

        if get_loss:
            output["loss_dict"] = self.get_loss(
                mus=mus,
                targets=targets,
                priors=prior,
                posteriors=posterior,
                variances=variances,
                kl_weight=kl_weight,
            )

        return output

    def predict_distribution(
        self, parameters: torch.Tensor, encoder: nn.Module
    ) -> Independent:
        output = encoder(parameters)
        mus, scales = output.chunk(2, dim=-1)
        scales = 0.1 + 0.9 * torch.sigmoid(scales)
        return Independent(Normal(mus, scales), 1)

    def sample_latent(
        self,
        distribution: Independent,
        num_samples: int = 1,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training:
            samples = distribution.rsample((num_samples,))
        elif noise is not None:
            mu = distribution.base_dist.loc
            std = distribution.base_dist.scale
            samples = EO.rearrange(mu + noise * std, " b s h -> s b 1 h")
        else:
            samples = distribution.sample((num_samples,))

        return EO.rearrange(samples, "s b ... -> b s ...")

    def combine_paths(self, z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_samples = z.shape[1]
        z = torch.nested.as_nested_tensor(
            [
                EO.repeat(rep, "s 1 h -> t s h", t=len(target))
                for rep, target in zip(z, targets, strict=True)
            ],
            layout=torch.jagged,
        )
        targets = torch.nested.nested_tensor_from_jagged(
            values=torch.cat(
                [EO.repeat(target, "t h -> t s h", s=num_samples) for target in targets]
            ),
            offsets=z.offsets(),  # ty: ignore[unresolved-attribute]
            min_seqlen=z._min_seqlen,  # noqa: SLF001  # ty: ignore[unresolved-attribute]
            max_seqlen=z._max_seqlen,  # noqa: SLF001  # ty: ignore[unresolved-attribute]
        )

        return z + targets

    def _encode_prior_latents(
        self,
        contexts: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        temperatures: torch.Tensor,
        markers: list[torch.Tensor],
        prefix_token_keys: list[str] | None = None,
    ) -> tensordict.TensorDict:
        return self._encode(
            contexts=contexts,
            targets=None,
            temperatures=temperatures,
            markers=markers,
            prefix_token_keys=prefix_token_keys or ["global"],
        )

    def _get_decode_targets(
        self, prior_latents: tensordict.TensorDict, targets: tensordict.TensorDict
    ) -> torch.Tensor:
        return targets["X_projected"]  # ty: ignore[invalid-return-type]

    def get_loss(  # ty: ignore[invalid-method-override]
        self,
        mus: torch.Tensor,
        targets: tensordict.TensorDict,
        priors: Independent,
        posteriors: Independent,
        variances: torch.Tensor,
        kl_weight: float = 1.0,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        likelihood_loss = self.get_likelihood_loss(
            mus=mus, targets=targets, variances=variances
        )
        kl_loss, debug_metrics = self.get_kl_loss(
            priors=priors, posteriors=posteriors, **kwargs
        )

        total_loss = likelihood_loss + kl_weight * kl_loss

        return {
            "loss": total_loss,
            "kl": kl_loss,
            "likelihood": likelihood_loss,
            **debug_metrics,
        }

    def get_kl_loss(
        self, priors: Independent, posteriors: Independent, **kwargs
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prior_mu = priors.base_dist.loc
        prior_std = priors.base_dist.scale
        post_mu = posteriors.base_dist.loc
        post_std = posteriors.base_dist.scale

        kl_per_dim = (
            prior_std.log()
            - post_std.log()
            + (post_std**2 + (post_mu - prior_mu) ** 2) / (2 * prior_std**2)
            - 0.5
        )

        kl_loss = kl_per_dim.clamp(min=self.free_bits).sum(dim=-1).mean()

        debug_metrics = {
            "active_units": (kl_per_dim.mean(dim=0) > 0.01).float().sum(),
            "kl_per_dim_mean": kl_per_dim.mean(dim=0),
            "posterior_std": post_std.mean(),
            "prior_std": prior_std.mean(),
        }

        return kl_loss, debug_metrics

    def get_likelihood_loss(
        self, mus: torch.Tensor, targets: tensordict.TensorDict, variances: torch.Tensor
    ) -> torch.Tensor:
        nll = torch.as_tensor(0.0, device=mus.device)
        for mu, Y, var in zip(mus, targets["Y"], variances, strict=True):
            nll = nll + gaussian_nll_loss(mu, Y[:, None], var)

        return nll / len(mus)


class ANP(LNP):
    def _encode_prior_latents(
        self,
        contexts: tensordict.TensorDict,
        targets: tensordict.TensorDict,
        temperatures: torch.Tensor,
        markers: list[torch.Tensor],
        prefix_token_keys: list[str] | None = None,
    ) -> tensordict.TensorDict:
        return self._encode(
            contexts=contexts,
            targets=targets,
            temperatures=temperatures,
            markers=markers,
            prefix_token_keys=prefix_token_keys or ["global"],
        )

    def _get_decode_targets(
        self, prior_latents: tensordict.TensorDict, targets: tensordict.TensorDict
    ) -> torch.Tensor:
        return prior_latents["target"]  # ty: ignore[invalid-return-type]
