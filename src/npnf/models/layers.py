import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        bias: bool = False,
        activation_fn=F.silu,
    ) -> None:
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if out_dim is None:
            out_dim = in_dim

        self.w13 = nn.Linear(in_dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=bias)

        self.activation_fn = activation_fn

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=module.in_features**-0.5)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(self.activation_fn(x1) * x3)
