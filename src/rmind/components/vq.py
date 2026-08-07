from typing import TYPE_CHECKING, cast, final

import torch
from torch import Tensor
from torch.nn import Module
from vector_quantize_pytorch import ResidualVQ as RVQ  # noqa: N817

if TYPE_CHECKING:
    from collections.abc import Callable


@final
class ResidualVQ(Module):
    """Residual vector quantizer from VQ-BeT (https://arxiv.org/pdf/2403.03181)."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        dim: int,
        codebook_size: int,
        num_quantizers: int,
        decay: float = 0.99,
        threshold_ema_dead_code: float = 2.0,
        kmeans_init: bool = True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        # the caller weights the returned commitment loss (WaypointsTokenizer._step);
        # leave RVQ's internal commitment_weight at its 1.0 default to avoid scaling it
        # twice.
        self.vq = RVQ(
            dim=dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            decay=decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
        )

        # The library codebook lazily runs kmeans-init on the first forward, guarded
        # by a data-dependent `if self.initted` (a tensor buffer) that
        # `torch.export` can't trace (ONNX export fails with "Data-dependent
        # branching"). That init is only needed while training from scratch -- an
        # eval/inference model always loads an already-initialized codebook -- so
        # gate it on the Python `training` flag, which export specializes as a
        # constant instead of failing on a tensor guard.
        for layer in self.vq.layers:
            self._guard_kmeans_init(cast("Module", layer._codebook))  # noqa: SLF001

    @staticmethod
    def _guard_kmeans_init(codebook: Module) -> None:
        init_embed_ = cast("Callable[..., object]", codebook.init_embed_)

        def guarded(*args: object, **kwargs: object) -> None:
            if codebook.training:
                init_embed_(*args, **kwargs)

        codebook.init_embed_ = guarded  # ty:ignore[unresolved-attribute]

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,) * self.num_quantizers

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # codebook loss is handled by the EMA codebook update, so only the
        # commitment loss is returned for the caller to weight.
        _, codes, commit = self.vq(z)
        z_q = self.lookup(codes)
        return codes, z_q, commit.sum()

    def lookup(self, codes: Tensor) -> Tensor:
        return self.vq.get_output_from_indices(codes)

    @torch.no_grad()
    def perplexity(self, codes: Tensor) -> Tensor:
        out: list[Tensor] = []
        for q, size in enumerate(self.codebook_sizes):
            counts = torch.bincount(codes[..., q].reshape(-1), minlength=size).float()
            p = counts / counts.sum().clamp_min(1.0)
            entropy = -(p * p.clamp_min(1e-10).log()).sum()
            out.append(entropy.exp())
        return torch.stack(out)
