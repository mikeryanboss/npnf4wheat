import math

import einops as EO
import tensordict
import tensordict.nn as tdnn
import torch
from torch import nn
from torchvision.ops import MLP

from npnf.models.transformer import RMSNorm, TransformerBlock


class HierarchicalTemperatureEncoder(nn.Module):
    """Encodes temperature time series into multiple tokens.

    Uses hierarchical convolutions for temporal feature extraction.

    Architecture:
    - Stage 1: Hourly -> Daily aggregation (kernel=24, stride=24)
    - Stage 2: Weekly pattern extraction (kernel=7)
    - Stage 3: Learned strided convolution to fixed token count
    - Learnable positional encoding

    Input:  (B, 6576) flattened hourly temperatures (274 days * 24 hours)
    Output: (B, num_output_tokens, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_output_tokens: int = 6,
        intermediate_channels: int = 64,
        daily_kernel_size: int = 24,
        weekly_kernel_size: int = 7,
    ):
        super().__init__()

        self.num_output_tokens = num_output_tokens

        self.hourly_to_daily = nn.Sequential(
            nn.Conv1d(
                1,
                intermediate_channels,
                kernel_size=daily_kernel_size,
                stride=daily_kernel_size,
            ),
            nn.SiLU(),
        )

        self.weekly_conv = nn.Sequential(
            nn.Conv1d(
                intermediate_channels,
                intermediate_channels * 2,
                kernel_size=weekly_kernel_size,
                padding=weekly_kernel_size // 2,
            ),
            nn.SiLU(),
        )

        num_daily = 6576 // daily_kernel_size  # 274
        pool_stride = num_daily // num_output_tokens
        pool_kernel = math.ceil(num_daily / num_output_tokens)
        self.pool = nn.Sequential(
            nn.Conv1d(
                intermediate_channels * 2,
                intermediate_channels * 2,
                kernel_size=pool_kernel,
                stride=pool_stride,
            ),
            nn.SiLU(),
        )
        self.project = nn.Linear(intermediate_channels * 2, hidden_dim)

        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_output_tokens, hidden_dim) * (hidden_dim**-0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)  # (B, 1, 6576)
        x = self.hourly_to_daily(x)  # (B, 64, 274)
        x = self.weekly_conv(x)  # (B, 128, 274)
        x = self.pool(x)  # (B, 128, 6)
        x = x.transpose(1, 2)  # (B, 6, 128)
        x = self.project(x)  # (B, 6, hidden_dim)
        return x + self.pos_encoding  # (B, 6, hidden_dim)


def get_attention_mask(
    contexts: tensordict.TensorDict,
    targets: tensordict.TensorDict | None,
    num_tokens: int,
    tokens_lengths: torch.Tensor,
    num_prefix_tokens: torch.Tensor,
) -> torch.Tensor:
    """Create attention mask for transformer with target masking.

    Tokens can attend to each other except to targets.
    Each target additionally attends to itself.

    Args:
        num_prefix_tokens: Per-sample tensor of prefix token counts.
    """
    arange_T = torch.arange(num_tokens, device=tokens_lengths.device)

    context_lengths = torch.as_tensor(
        [len(x) for x in contexts["X"]], device=tokens_lengths.device
    )
    target_lengths = (
        torch.as_tensor([len(x) for x in targets["X"]], device=tokens_lengths.device)
        if targets is not None
        else torch.zeros(len(contexts), dtype=torch.long, device=tokens_lengths.device)
    )

    target_cols = (
        arange_T[None, :] >= (num_prefix_tokens + context_lengths)[:, None]
    ) & (
        arange_T[None, :]
        < (num_prefix_tokens + context_lengths + target_lengths)[:, None]
    )
    pad_cols = arange_T[None, :] >= tokens_lengths[:, None]

    masks = torch.ones(
        (len(contexts), num_tokens, num_tokens),
        dtype=torch.bool,
        device=tokens_lengths.device,
    )

    masks.masked_fill_(
        mask=target_cols.unsqueeze(1).expand(-1, num_tokens, -1), value=False
    )

    diag_target = torch.diag_embed(target_cols)
    masks |= diag_target

    masks.masked_fill_(
        mask=pad_cols.unsqueeze(1).expand(-1, num_tokens, -1), value=False
    )
    masks.masked_fill_(
        mask=pad_cols.unsqueeze(2).expand(-1, -1, num_tokens), value=False
    )

    return masks


class TransformerProcessBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 4,
        activation_layer: type[nn.Module] = nn.SiLU,
        num_transformer_layers: int = 12,
        num_transformer_heads: int = 4,
        *,
        markers_input_dim: int,
        markers_hidden_channels: list[int],
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.Y_projection = tdnn.TensorDictModule(
            nn.Linear(1, hidden_dim, bias=False),
            in_keys=["Y"],
            out_keys=["Y_projected"],
        )
        self.X_projection = tdnn.TensorDictModule(
            nn.Linear(1, hidden_dim, bias=False),
            in_keys=["X_normalized"],
            out_keys=["X_projected"],
        )

        self.temperature_encoder = HierarchicalTemperatureEncoder(
            hidden_dim=hidden_dim, num_output_tokens=6
        )

        self.markers_encoder = tdnn.TensorDictModule(
            MLP(
                in_channels=markers_input_dim,
                hidden_channels=[*markers_hidden_channels, hidden_dim],
                activation_layer=activation_layer,
                bias=False,
            ),
            in_keys=["Y"],
            out_keys=["Y_projected"],
        )

        self.encoder_layers = nn.ModuleList(
            TransformerBlock(
                dim=hidden_dim, num_heads=num_transformer_heads, multiple_of=hidden_dim
            )
            for _ in range(num_transformer_layers)
        )
        self.final_norm = RMSNorm(hidden_dim)

        self.prefix_tokens = nn.ParameterDict(
            {
                "global": nn.Parameter(torch.empty(hidden_dim)),
                "flow_condition": nn.Parameter(torch.empty(hidden_dim)),
            }
        )

        self.apply(self._init_weights)
        self._init_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=module.in_features**-0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_parameters(self) -> None:
        for token in self.prefix_tokens.values():
            self._init_parameter(token)

    def _init_parameter(self, parameter: nn.Parameter) -> None:
        parameter.data.normal_(mean=0.0, std=parameter.shape[-1] ** -0.5)

    def _project_points(
        self, points: tensordict.TensorDict
    ) -> tuple[tensordict.TensorDict, bool]:
        points = self.X_projection(points)
        has_y = "Y" in points and points["Y"] is not None
        if has_y:
            points = self.Y_projection(points)
            points["YX_projected"] = points["Y_projected"] + points["X_projected"]
        return points, has_y

    def _prepare_temperatures(
        self,
        *,
        temperatures: tensordict.TensorDict | None,
        reference: tensordict.TensorDict,
    ) -> torch.Tensor:
        device = reference["X"].device
        dtype = reference["X"].dtype

        if temperatures is None:
            return torch.empty(
                len(reference), 0, self.hidden_dim, device=device, dtype=dtype
            )

        return self.temperature_encoder(
            temperatures["Y"].to(device=device, dtype=dtype)
        )

    def _prepare_markers(
        self, *, markers: tensordict.TensorDict | None, reference: tensordict.TensorDict
    ) -> list[torch.Tensor]:
        device = reference["X"].device
        dtype = reference["X"].dtype

        if markers is None:
            return [
                torch.empty(0, self.hidden_dim, device=device, dtype=dtype)
                for _ in range(len(reference))
            ]

        markers = self.markers_encoder(markers.float())
        return [
            sample
            if len(sample) > 0
            else torch.empty(0, self.hidden_dim, device=device, dtype=dtype)
            for sample in markers["Y_projected"]
        ]

    def _encode(
        self,
        contexts: tensordict.TensorDict,
        targets: tensordict.TensorDict | None,
        temperatures: torch.Tensor,
        markers: list[torch.Tensor],
        prefix_token_keys: list[str],
    ) -> tensordict.TensorDict:
        tokens, tokens_lengths = self._assemble_tokens(
            contexts=contexts,
            targets=targets,
            temperatures=temperatures,
            markers=markers,
            prefix_token_keys=prefix_token_keys,
        )
        _, num_tokens, _ = tokens.shape
        num_temperature_tokens = temperatures.shape[1]
        num_marker_tokens = torch.tensor(
            [len(m) for m in markers], device=tokens.device
        )
        num_prefix_tokens = (
            len(prefix_token_keys) + num_temperature_tokens + num_marker_tokens
        )

        attention_mask = get_attention_mask(
            contexts=contexts,
            targets=targets,
            num_tokens=num_tokens,
            tokens_lengths=tokens_lengths,
            num_prefix_tokens=num_prefix_tokens,
        )

        tokens = self._run_transformer(tokens=tokens, attn_mask=attention_mask)

        return self._split_tokens(
            tokens=tokens,
            targets=targets,
            contexts=contexts,
            prefix_token_keys=prefix_token_keys,
            num_temps=num_temperature_tokens,
            num_markers=num_marker_tokens,
        )

    def _assemble_tokens(
        self,
        *,
        contexts: tensordict.TensorDict,
        targets: tensordict.TensorDict | None,
        temperatures: torch.Tensor,
        markers: list[torch.Tensor],
        prefix_token_keys: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = contexts["YX_projected"].device

        prefix_tokens_list = [
            EO.repeat(
                self.prefix_tokens[key].to(
                    device=device, dtype=contexts["YX_projected"].dtype
                ),
                "r -> b 1 r",
                b=len(contexts),
            )
            for key in prefix_token_keys
        ]

        tokens = [*prefix_tokens_list, temperatures, markers, contexts["YX_projected"]]

        if targets is not None:
            tokens.append(targets["X_projected"])

        tokens = torch.nested.as_nested_tensor(
            [torch.cat(t) for t in zip(*tokens, strict=True)], layout=torch.jagged
        )
        tokens_lengths = torch.as_tensor([len(t) for t in tokens], device=device)
        tokens = tokens.to_padded_tensor(0.0)

        return tokens, tokens_lengths

    def _run_transformer(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.encoder_layers:
            tokens = layer(tokens=tokens, attn_mask=attn_mask)
        return self.final_norm(tokens)

    def _split_tokens(
        self,
        tokens: torch.Tensor,
        targets: tensordict.TensorDict | None,
        contexts: tensordict.TensorDict,
        prefix_token_keys: list[str],
        num_temps: int,
        num_markers: torch.Tensor,
    ) -> tensordict.TensorDict:
        target_lengths = (
            torch.as_tensor([len(x) for x in targets["X"]], device=tokens.device)
            if targets is not None
            else torch.zeros(len(contexts), dtype=torch.long, device=tokens.device)
        )
        context_lengths = torch.as_tensor(
            [len(x) for x in contexts["X"]], device=tokens.device
        )
        num_prefix = len(prefix_token_keys)

        result = {key: tokens[:, i : i + 1] for i, key in enumerate(prefix_token_keys)}
        result["temperature"] = tokens[:, num_prefix : num_prefix + num_temps]

        marker_start = num_prefix + num_temps
        result["marker"] = torch.nested.as_nested_tensor(
            [
                tokens[i, marker_start : marker_start + num_markers[i]]
                for i in range(len(tokens))
            ],
            layout=torch.jagged,
        )

        context_start = marker_start + num_markers
        result["context"] = torch.nested.as_nested_tensor(
            [
                tokens[i, context_start[i] : context_start[i] + context_lengths[i]]
                for i in range(len(tokens))
            ],
            layout=torch.jagged,
        )

        target_start = context_start + context_lengths
        result["target"] = torch.nested.as_nested_tensor(
            [
                tokens[i, target_start[i] : target_start[i] + target_lengths[i]]
                for i in range(len(tokens))
            ],
            layout=torch.jagged,
        )

        return tensordict.TensorDict(
            result,  # ty: ignore[invalid-argument-type]
            batch_size=[len(contexts)],
        )
