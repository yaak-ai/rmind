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
        codebook = cast("Module", self.vq.layers[level]._codebook)  # noqa: SLF001
        return cast("Tensor", codebook.embed).reshape(-1, self.dim)

    @torch.no_grad()
    def perplexity(self, codes: Tensor) -> Tensor:
        out: list[Tensor] = []
        for q, size in enumerate(self.codebook_sizes):
            counts = torch.bincount(codes[..., q].reshape(-1), minlength=size).float()
            p = counts / counts.sum().clamp_min(1.0)
            entropy = -(p * p.clamp_min(1e-10).log()).sum()
            out.append(entropy.exp())
        return torch.stack(out)


@final
class GroupedResidualVQ(Module):
    """Independent residual VQs over consecutive latent slices (one per axis group).

    Presents the `ResidualVQ` interface over the concatenation, so a consumer that
    only knows "g levels x c codes over a `dim` latent" (the patch policy's code
    head, `lookup`, `perplexity`) is unchanged: `num_quantizers` is
    `groups * depth` and the code columns are group-major ([g0 q0, g0 q1, g1 q0,
    g1 q1, ...]). Each group has its own codebooks; `dim` is `groups * group_dim`.
    """

    def __init__(
        self,
        *,
        groups: int,
        group_dim: int,
        codebook_size: int,
        num_quantizers_per_group: int,
        **kwargs: float | bool,
    ) -> None:
        super().__init__()
        self.groups = groups
        self.group_dim = group_dim
        self.depth = num_quantizers_per_group
        self.dim = groups * group_dim
        self.codebook_size = codebook_size
        self.num_quantizers = groups * num_quantizers_per_group
        self.quantizers: list[ResidualVQ] = torch.nn.ModuleList(  # ty:ignore[invalid-assignment]
            ResidualVQ(
                dim=group_dim,
                codebook_size=codebook_size,
                num_quantizers=num_quantizers_per_group,
                **kwargs,  # ty:ignore[invalid-argument-type]
            )
            for _ in range(groups)
        )

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,) * self.num_quantizers

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        codes, z_q, commit = [], [], z.new_zeros(())
        for quantizer, zg in zip(
            self.quantizers, z.split(self.group_dim, dim=-1), strict=True
        ):
            c, q, vq = quantizer(zg)
            codes.append(c)
            z_q.append(q)
            commit += vq["commit"]
        return (
            torch.cat(codes, dim=-1),
            torch.cat(z_q, dim=-1),
            {"codebook": z.new_zeros(()), "commit": commit},
        )

    def lookup(self, codes: Tensor) -> Tensor:
        return torch.cat(
            [
                quantizer.lookup(c.contiguous())
                for quantizer, c in zip(
                    self.quantizers, codes.split(self.depth, dim=-1), strict=True
                )
            ],
            dim=-1,
        )

    def codebook(self, level: int) -> Tensor:
        """Level `level`'s codebook embedded in the full latent `(c, dim)` (zeros elsewhere)."""
        g, q = divmod(level, self.depth)
        book = self.quantizers[g].codebook(q)
        out = book.new_zeros(book.shape[0], self.dim)
        out[:, g * self.group_dim : (g + 1) * self.group_dim] = book
        return out

    @torch.no_grad()
    def perplexity(self, codes: Tensor) -> Tensor:
        return torch.cat([
            quantizer.perplexity(c)
            for quantizer, c in zip(
                self.quantizers, codes.split(self.depth, dim=-1), strict=True
            )
        ])
