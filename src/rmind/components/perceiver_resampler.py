from typing import final, override

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import Tensor, einsum
from torch.nn import GELU, LayerNorm, Linear, Module, ModuleList, Parameter, Sequential


def feed_forward(dim: int, mult: int = 4) -> Module:
    inner_dim = int(dim * mult)
    return Sequential(
        LayerNorm(dim),
        Linear(dim, inner_dim, bias=False),
        GELU(),
        Linear(inner_dim, dim, bias=False),
    )


@final
class PerceiverAttention(Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = LayerNorm(dim)
        self.norm_latents = LayerNorm(dim)

        self.to_q = Linear(dim, inner_dim, bias=False)
        self.to_kv = Linear(dim, inner_dim * 2, bias=False)
        self.to_out = Linear(inner_dim, dim, bias=False)

    @override
    def forward(self, x: Tensor, latents: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latents (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q *= self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim -= sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


@final
class PerceiverResampler(Module):
    def __init__(  # noqa: PLR0913
        self,
        *,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,  # output num of embeddings
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.latents = Parameter(torch.randn(num_latents, dim))

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(  # pyright: ignore[reportUnusedCallResult]
                ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    feed_forward(dim=dim, mult=ff_mult),
                ])
            )

        self.norm = LayerNorm(dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, t, num_tokens_in, dim)
        Returns:
            shape (b, t, num_tokens_out, dim)
        """
        b, t = x.shape[:2]

        # blocks
        latents = repeat(self.latents, "n d -> b t n d", b=b, t=t)
        for attn, ff in self.layers:  # pyright: ignore[reportGeneralTypeIssues]
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)
