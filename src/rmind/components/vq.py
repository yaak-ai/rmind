from collections.abc import Callable
from typing import final

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn import functional as F

from rmind.components.nn import default_weight_init_fn


@final
class _EMACodebook(Module):
    """A single VQ codebook with EMA updates + dead-code revival.

    The codebook is *not* trained by gradient — it's tracked as an exponential
    moving average of the encoder outputs assigned to each code (van den Oord
    2017 / VQ-VAE-2). Codes whose EMA cluster size falls below
    ``threshold_ema_dead_code`` are reseeded from random current-batch latents,
    which is what keeps the codebook from collapsing on imbalanced data.
    """

    def __init__(
        self,
        *,
        num_codes: int,
        dim: int,
        decay: float,
        eps: float,
        threshold_ema_dead_code: float,
        weight_init_fn: Callable[[Tensor], None],
    ) -> None:
        super().__init__()

        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        embed = torch.empty(num_codes, dim)
        weight_init_fn(embed)
        self.register_buffer("embed", embed)  # (K, dim) — the codebook
        self.register_buffer("cluster_size", torch.zeros(num_codes))  # EMA counts
        self.register_buffer("embed_avg", embed.clone())  # EMA latent sums

    @property
    def num_embeddings(self) -> int:
        return self.embed.shape[0]

    def lookup(self, idx: Tensor) -> Tensor:
        return F.embedding(idx, self.embed)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: (N, dim) → (idx (N,), quantized (N, dim)); EMA-updates in train mode."""
        idx = torch.cdist(x, self.embed).argmin(dim=-1)
        quantized = self.lookup(idx)
        if self.training:
            self._ema_update(x, idx)
        return idx, quantized

    @torch.no_grad()
    def _ema_update(self, x: Tensor, idx: Tensor) -> None:
        k = self.embed.shape[0]
        onehot = F.one_hot(idx, k).type_as(x)  # (N, K)
        counts = onehot.sum(0)  # (K,)
        embed_sum = onehot.t() @ x  # (K, dim)

        self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        # Laplace-smoothed cluster sizes to avoid divide-by-zero
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (total + k * self.eps) * total
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(-1))

        if self.threshold_ema_dead_code > 0:
            self._revive_dead_codes(x)

    @torch.no_grad()
    def _revive_dead_codes(self, x: Tensor) -> None:
        dead = self.cluster_size < self.threshold_ema_dead_code
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        # reseed dead codes from random live batch latents
        samples = x[torch.randint(0, x.shape[0], (n_dead,), device=x.device)]
        self.embed[dead] = samples
        self.embed_avg[dead] = samples
        self.cluster_size[dead] = 1.0


@final
class ResidualVQ(Module):
    """Residual vector quantizer from VQ-BeT (https://arxiv.org/pdf/2403.03181),
    with EMA codebook updates + dead-code revival.

    A cascade of EMA codebooks where each quantizes the residual left by the
    previous ones (primary mode + secondary refinement, …). Since the codebooks
    are updated by EMA (not gradient), the only loss returned is the
    ``commitment`` term that pulls the encoder toward the chosen codes;
    ``codebook`` is kept (as 0) for interface compatibility.
    """

    def __init__(
        self,
        *,
        dim: int,
        codebook_sizes: tuple[int, ...],
        decay: float = 0.99,
        eps: float = 1e-5,
        threshold_ema_dead_code: float = 2.0,
        weight_init_fn: Callable[[Tensor], None] = default_weight_init_fn,  # ty:ignore[invalid-parameter-default]
    ) -> None:
        super().__init__()

        if not codebook_sizes:
            msg = "codebook_sizes must be non-empty"
            raise ValueError(msg)

        self.dim = dim
        self.codebook_sizes = tuple(codebook_sizes)
        self.codebooks = ModuleList(
            _EMACodebook(
                num_codes=n,
                dim=dim,
                decay=decay,
                eps=eps,
                threshold_ema_dead_code=threshold_ema_dead_code,
                weight_init_fn=weight_init_fn,
            )
            for n in self.codebook_sizes
        )

    @property
    def num_quantizers(self) -> int:
        return len(self.codebooks)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        codes: list[Tensor] = []
        z_q = torch.zeros_like(z)
        residual = z
        commit_loss = z.new_zeros(())

        for codebook in self.codebooks:
            idx, e = codebook(residual)
            # commitment loss → trains the encoder (codebook itself is EMA-updated)
            commit_loss = commit_loss + F.mse_loss(residual, e.detach())
            codes.append(idx)
            z_q = z_q + e
            residual = residual - e

        return (
            torch.stack(codes, dim=-1),
            z_q,
            {"codebook": z.new_zeros(()), "commit": commit_loss},
        )

    def lookup(self, codes: Tensor) -> Tensor:
        """``codes``: ``(..., num_quantizers)`` long → quantized latent ``(..., dim)``."""
        return sum(
            (self.codebooks[q].lookup(codes[..., q]) for q in range(self.num_quantizers)),
            start=torch.zeros((), device=codes.device),
        )

    @torch.no_grad()
    def perplexity(self, codes: Tensor) -> Tensor:
        """Per-quantizer codebook perplexity ``exp(H(code usage))`` — a usage/health
        diagnostic; low values flag codebook collapse / dead codes."""
        out: list[Tensor] = []
        for q, codebook in enumerate(self.codebooks):
            counts = torch.bincount(
                codes[..., q].reshape(-1), minlength=codebook.num_embeddings
            ).float()
            p = counts / counts.sum().clamp_min(1.0)
            entropy = -(p * p.clamp_min(1e-10).log()).sum()
            out.append(entropy.exp())
        return torch.stack(out)
