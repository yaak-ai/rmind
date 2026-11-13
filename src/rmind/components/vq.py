from functools import lru_cache
from typing import TYPE_CHECKING, cast, final

import torch
from torch import Tensor
from torch.nn import Module
from vector_quantize_pytorch import ResidualVQ as RVQ  # noqa: N817
from vector_quantize_pytorch import vector_quantize_pytorch as _vqp

if TYPE_CHECKING:
    from collections.abc import Callable


@lru_cache(maxsize=None)
def _cached_zero(device: torch.device) -> Tensor:
    return torch.zeros((), device=device)


def _patch_vq_loss_init() -> None:
    """Patch `VectorQuantize.forward`'s per-call loss-accumulator init.

    `vector_quantize_pytorch.py:1263` does
    `loss = tensor(0., device=device, requires_grad=self.training)` on
    *every* forward call, for *every* quantizer level, on *every* step --
    a host-allocated Python float copied to GPU synchronously. Profiling
    (PROFILING_NOTES.md §7) found this to be the single largest source of
    per-step stall: `cudaStreamSynchronize` drains the entire pending kernel
    queue at that point, so the deepest/last-called quantizer level pays for
    the whole step's accumulated GPU work so far (~123ms + ~33ms, ~52% of
    step wall time combined).

    The library itself already registers a `self.zero` buffer for exactly
    this purpose (line 954) but doesn't reuse it at line 1263 -- this looks
    like an oversight upstream rather than an intentional per-call
    allocation. Not easily patched in-place (third-party, and the exact line
    is buried inside a large `forward`), so this replaces the module-level
    `tensor` symbol `vector_quantize_pytorch.py` calls into with a wrapper
    that intercepts only that exact call signature (`tensor(0., device=...,
    requires_grad=...)` -- distinctive within that file, see the assertion
    below) and returns a cached GPU-resident zero instead. Every other
    `tensor(...)` call in that module (kmeans-init flags, buffer inits,
    etc.) is passed through unchanged.
    """
    real_tensor = _vqp.tensor

    def patched_tensor(*args: object, **kwargs: object) -> Tensor:
        if (
            len(args) == 1
            and args[0] == 0.0
            and set(kwargs) <= {"device", "requires_grad"}
            and "device" in kwargs
        ):
            zero = _cached_zero(cast("torch.device", kwargs["device"]))
            return zero.requires_grad_(bool(kwargs.get("requires_grad", False)))
        return real_tensor(*args, **kwargs)

    _vqp.tensor = patched_tensor


_patch_vq_loss_init()


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
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: float = 2.0,
        kmeans_init: bool = True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.vq = RVQ(
            dim=dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
        )

        # The library codebook lazily runs kmeans-init on the first forward,
        # guarded by a data-dependent `if self.initted` (a tensor buffer) that
        # `torch.export` can't trace. That init is only needed while training
        # from scratch -- an eval/inference model always loads an
        # already-initialized codebook -- so gate it on the Python `training`
        # flag, which export specializes as a constant. (Ported from
        # feat/wpts-rvq.)
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

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        _, codes, commit = self.vq(z)
        z_q = self.lookup(codes)
        return codes, z_q, {"codebook": z.new_zeros(()), "commit": commit.sum()}

    def lookup(self, codes: Tensor) -> Tensor:
        return self.vq.get_output_from_indices(codes)

    def codebook(self, level: int) -> Tensor:
        return self.vq.layers[level]._codebook.embed.reshape(  # noqa: SLF001
            -1, self.dim
        )

    @torch.no_grad()
    def perplexity(self, codes: Tensor) -> Tensor:
        out: list[Tensor] = []
        for q, size in enumerate(self.codebook_sizes):
            counts = torch.bincount(codes[..., q].reshape(-1), minlength=size).float()
            p = counts / counts.sum().clamp_min(1.0)
            entropy = -(p * p.clamp_min(1e-10).log()).sum()
            out.append(entropy.exp())
        return torch.stack(out)
