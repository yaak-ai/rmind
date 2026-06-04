from typing import final

import torch
from torch import Tensor
from torch.nn import Module
from vector_quantize_pytorch import ResidualVQ as _ResidualVQ


@final
class ResidualVQ(Module):
    """Residual vector quantizer from VQ-BeT (https://arxiv.org/pdf/2403.03181).

    Thin wrapper over ``vector_quantize_pytorch.ResidualVQ`` — EMA codebook
    updates, k-means init, and dead-code revival come for free (this is the same
    library VQ-BeT uses). Keeps the rmind interface so ``ActionTokenizer`` and the
    config are unchanged: ``forward → (codes, z_q, {commit})``, ``lookup``,
    ``perplexity``.

    ``codebook_sizes`` must be uniform (the library uses one ``codebook_size`` for
    all residual quantizers); the tuple length sets ``num_quantizers``.
    """

    def __init__(
        self,
        *,
        dim: int,
        codebook_sizes: tuple[int, ...],
        decay: float = 0.99,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: float = 2.0,
        kmeans_init: bool = True,
    ) -> None:
        super().__init__()

        sizes = tuple(codebook_sizes)
        if not sizes:
            msg = "codebook_sizes must be non-empty"
            raise ValueError(msg)
        if len(set(sizes)) != 1:
            msg = (
                "vector_quantize_pytorch.ResidualVQ uses a uniform codebook_size; "
                f"pass equal codebook_sizes (got {sizes})"
            )
            raise ValueError(msg)

        self.dim = dim
        self.codebook_sizes = sizes
        self.vq = _ResidualVQ(
            dim=dim,
            num_quantizers=len(sizes),
            codebook_size=sizes[0],
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
        )

    @property
    def num_quantizers(self) -> int:
        return len(self.codebook_sizes)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        # The library's `quantized` is straight-through; we return the *hard* z_q
        # via get_output_from_indices so the model's own STE
        # (z + (z_q - z).detach()) stays correct. `commit` is per-quantizer → sum.
        # The codebook is EMA-updated (no codebook loss), so report 0 for it.
        _, codes, commit = self.vq(z)
        z_q = self.lookup(codes)
        return codes, z_q, {"codebook": z.new_zeros(()), "commit": commit.sum()}

    def lookup(self, codes: Tensor) -> Tensor:
        """``codes``: ``(..., num_quantizers)`` → quantized latent ``(..., dim)``."""
        return self.vq.get_output_from_indices(codes)

    @torch.no_grad()
    def perplexity(self, codes: Tensor) -> Tensor:
        """Per-quantizer codebook perplexity ``exp(H(code usage))`` — a usage/health
        diagnostic; low values flag codebook collapse / dead codes."""
        out: list[Tensor] = []
        for q, size in enumerate(self.codebook_sizes):
            counts = torch.bincount(
                codes[..., q].reshape(-1), minlength=size
            ).float()
            p = counts / counts.sum().clamp_min(1.0)
            entropy = -(p * p.clamp_min(1e-10).log()).sum()
            out.append(entropy.exp())
        return torch.stack(out)
