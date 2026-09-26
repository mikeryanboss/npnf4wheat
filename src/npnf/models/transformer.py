import einops as EO
import torch
from torch import nn

from npnf.models.layers import FeedForward


class RMSNorm(nn.RMSNorm):
    """RMSNorm with float32 computation for numerical stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=module.in_features**-0.5)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        qkv = self.input_to_qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        query = EO.rearrange(query, "b s (d h) -> b d s h", d=self.num_heads)
        key = EO.rearrange(key, "b s (d h) -> b d s h", d=self.num_heads)
        value = EO.rearrange(value, "b s (d h) -> b d s h", d=self.num_heads)

        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        )

        output = EO.rearrange(output, "b d s h -> b s (d h)")

        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attention = Attention(dim=dim, num_heads=num_heads)

        self.feed_forward = FeedForward(
            in_dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, seq_length, _ = tokens.shape
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = EO.repeat(
                    attn_mask, "b s -> b d l s", d=self.num_heads, l=seq_length
                )
            else:
                attn_mask = EO.repeat(attn_mask, "b l s -> b d l s", d=self.num_heads)

        tokens = tokens + self.attention(
            self.attention_norm(tokens), attn_mask=attn_mask
        )
        return tokens + self.feed_forward(self.feed_forward_norm(tokens))
