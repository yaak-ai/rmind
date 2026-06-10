from typing import final

import torch
from torch import Tensor
from torch.nn import Module
from vector_quantize_pytorch import ResidualVQ as _ResidualVQ


@final
class ResidualVQ(Module):
    """Residual vector quantizer from VQ-BeT (https://arxiv.org/pdf/2403.03181).
    """

    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int,
        num_quantizers: int,
        decay: float = 0.99,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: float = 2.0,
        kmeans_init: bool = True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.vq = _ResidualVQ(
            dim=dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
        )

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,) * self.num_quantizers

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        _, codes, commit = self.vq(z)
        z_q = self.lookup(codes)
        return codes, z_q, {"codebook": z.new_zeros(()), "commit": commit.sum()}

    def lookup(self, codes: Tensor) -> Tensor:
        return self.vq.get_output_from_indices(codes)

    def codebook(self, level: int) -> Tensor:
        return self.vq.layers[level]._codebook.embed.reshape(-1, self.dim)

    @torch.no_grad()
    def perplexity(self, codes: Tensor) -> Tensor:
        out: list[Tensor] = []
        for q, size in enumerate(self.codebook_sizes):
            counts = torch.bincount(
                codes[..., q].reshape(-1), minlength=size
            ).float()
            p = counts / counts.sum().clamp_min(1.0)
            entropy = -(p * p.clamp_min(1e-10).log()).sum()
            out.append(entropy.exp())
        return torch.stack(out)
