from collections.abc import Mapping
from itertools import accumulate, pairwise
from typing import final, override

import more_itertools as mit
import torch
from einops import pack, repeat
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from tensordict import TensorClass, TensorDict
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import (  # noqa: PLC2701
    MappingKey,
    key_get,
    tree_leaves,
    tree_map,
    tree_map_with_path,
)

from rmind.components.base import TensorTree, TokenMeta
from rmind.components.containers import ModuleDict
from rmind.components.mask import (
    FactorizedAttentionMask,
    FactorizedAttentionMaskBuilder,
    TorchAttentionMaskLegend,
)
from rmind.utils.pytree import unflatten_keys

logger = get_logger(__name__)


class Index(TensorClass["frozen"]):  # ty:ignore[unsupported-base]
    continuous: TensorDict
    context: TensorDict
    discrete: TensorDict
    foresight: TensorDict
    image: TensorDict
    summary: TensorDict

    def parse(self, src: Tensor, dim: int = 1) -> TensorDict:
        shape_left, shape_right = src.shape[:dim], src.shape[dim + 1 :]
        batch_size = (*shape_left, *self.batch_size)

        return self.to_tensordict(retain_none=False).apply(
            lambda index: src.index_select(dim, index.flatten()).view(
                *shape_left, *index.shape, *shape_right
            ),
            batch_size=batch_size,
            device=src.device,
            inplace=False,
        )


class Episode(TensorClass["frozen"]):  # ty:ignore[unsupported-base]
    input: TensorDict
    input_tokens: TensorDict
    input_embeddings: TensorDict
    embeddings: TensorDict
    index: Index
    embeddings_flattened: Tensor
    attention_mask: FactorizedAttentionMask



@final
class EpisodeBuilder(Module):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        timestep: tuple[TokenMeta, ...],
        special_tokens: Mapping[str, Mapping[str, tuple[int, ...]]],
        input_transform: InstanceOf[Module],
        tokenizers: InstanceOf[ModuleDict],
        embeddings: InstanceOf[ModuleDict],
        projections: InstanceOf[ModuleDict],
        role_encoding: InstanceOf[Module],
        attention_mask_builder: InstanceOf[FactorizedAttentionMaskBuilder],
    ) -> None:
        super().__init__()

        self.special_tokens: Mapping[str, Mapping[str, tuple[int, ...]]] = (
            special_tokens
        )
        self.timestep: tuple[TokenMeta, ...] = timestep
        self._timestep_keys: tuple[tuple[str, str], ...] = tuple(
            (token.modality.value, token.name) for token in timestep
        )
        self.input_transform: Module = input_transform
        self.tokenizers: ModuleDict = tokenizers
        self.embeddings: ModuleDict = embeddings
        self.projections: ModuleDict = projections
        self.role_encoding: Module = role_encoding
        self.attention_mask_builder: FactorizedAttentionMaskBuilder = (
            attention_mask_builder
        )
        self.register_buffer("_attention_mask_spatial", None, persistent=False)
        self.register_buffer("_attention_mask_temporal", None, persistent=False)
        role_idx_by_type_modality: dict[tuple[str, str], int] = {}
        self._role_idx_by_path: dict[tuple[MappingKey, MappingKey], int] = {
            (
                MappingKey(token.modality.value),
                MappingKey(token.name),  # https://github.com/yaak-ai/rmind/issues/204
            ): role_idx_by_type_modality.setdefault(
                (token.type.value, token.modality.value), len(role_idx_by_type_modality)
            )
            for token in timestep
        }

    @property
    def attention_mask_cache_is_warm(self) -> bool:
        return (
            self._attention_mask_spatial is not None
            and self._attention_mask_temporal is not None
        )

    @override
    def forward(self, batch: TensorTree) -> Episode:
        input = self.input_transform(batch)
        input_tokens = self.tokenizers(input)

        first_leaf = mit.first(
            leaf for leaf in tree_leaves(input_tokens) if leaf is not None
        )
        b, t, device = first_leaf.shape[0], first_leaf.shape[1], first_leaf.device

        input_tokens.update(
            tree_map(
                lambda x: torch.tensor(x, device=device).expand(b, t, -1),
                self.special_tokens,
                is_leaf=lambda x: isinstance(x, tuple),
            )
        )
        input_embeddings = self.embeddings(input_tokens)
        projected_embeddings = self.projections(input_embeddings)

        index = self._build_index(projected_embeddings)

        attention_mask = self._build_attention_mask(index=index, timestep=self.timestep)

        role_embeddings = self._build_role_embeddings(projected_embeddings)

        embeddings = tree_map(
            lambda p, r: p + r if p is not None and r is not None else None,
            projected_embeddings,
            role_embeddings,
        )

        embeddings_flattened, _ = pack(
            [
                key_get(embeddings, (MappingKey(k[0]), MappingKey(k[1])))  # ty:ignore[invalid-argument-type]
                for k in self._timestep_keys
            ],
            "b t * d",
        )

        return Episode(
            input=TensorDict(input, batch_size=[b, t]).filter_non_tensor_data(),
            input_tokens=TensorDict(
                input_tokens, batch_size=[b, t]
            ).filter_non_tensor_data(),
            input_embeddings=TensorDict(
                input_embeddings, batch_size=[b, t]
            ).filter_non_tensor_data(),
            embeddings=TensorDict.from_dict(
                embeddings, batch_dims=2
            ).filter_non_tensor_data(),
            index=Index.from_tensordict(TensorDict(index, batch_size=[t])),  # ty:ignore[invalid-argument-type]
            embeddings_flattened=embeddings_flattened,
            attention_mask=attention_mask,
            device=device,
        )

    def _build_attention_mask(
        self, *, index: TensorTree, timestep: tuple[TokenMeta, ...]
    ) -> FactorizedAttentionMask:
        """Build (or return cached) spatial and temporal attention masks.

        WARNING: attention_mask_builder is not trace-friendly, so torch.export relies
        on this cache being warm. An eager forward pass must run *before*
        torch.export.export() — see export_onnx.py.
        """
        if self.attention_mask_cache_is_warm:
            return FactorizedAttentionMask.from_tensors(
                spatial_mask_tensor=self._attention_mask_spatial,
                temporal_mask_tensor=self._attention_mask_temporal,
                legend=TorchAttentionMaskLegend,
            )

        if torch.compiler.is_exporting() and not torch.compiler.is_compiling():
            # Guard with is_compiling(): structlog's logger is not dynamo-traceable
            # (_thread.allocate_lock breaks strict export). The warning still fires
            # in fake-export paths where is_exporting()=True but dynamo isn't active.
            logger.warning(
                "building attention mask during export; "
                "run an eager forward pass first to populate the cache"
            )

        attention_mask = self.attention_mask_builder(
            index=index, timestep=timestep, legend=TorchAttentionMaskLegend
        )

        if not torch.compiler.is_exporting():
            self._attention_mask_spatial = attention_mask.spatial.mask_tensor
            self._attention_mask_temporal = attention_mask.temporal.mask_tensor

        return attention_mask

    def _build_index(self, embeddings: TensorTree) -> TensorTree:
        first_leaf = mit.first(
            leaf for leaf in tree_leaves(embeddings) if leaf is not None
        )
        t, device = first_leaf.shape[1], first_leaf.device

        lengths = [
            key_get(
                embeddings,
                (MappingKey(token.modality.value), MappingKey(token.name)),  # ty:ignore[invalid-argument-type]
            ).shape[2]
            for token in self.timestep
        ]

        timestep_length = sum(lengths)
        ranges = pairwise(accumulate(lengths, initial=0))

        timestep_index = unflatten_keys({
            (token.modality.value, token.name): torch.arange(*range_, device=device)
            for token, range_ in zip(self.timestep, ranges, strict=True)
        })

        return tree_map(
            lambda x: torch.stack([x + i * timestep_length for i in range(t)]),
            timestep_index,
        )

    def _build_role_embeddings(self, embeddings: TensorTree) -> TensorTree:
        first_leaf = mit.first(
            leaf for leaf in tree_leaves(embeddings) if leaf is not None
        )
        t, device = first_leaf.shape[1], first_leaf.device

        roles = torch.arange(self.role_encoding.num_embeddings, device=device)  # ty:ignore[no-matching-overload]
        role_embeddings = self.role_encoding(roles)

        return tree_map_with_path(
            lambda path, leaf: (
                repeat(
                    role_embeddings[self._role_idx_by_path[path]],
                    "d -> b t n d",
                    b=leaf.shape[0],
                    t=t,
                    n=leaf.shape[-2],
                )
                if leaf is not None
                and path is not None
                and path in self._role_idx_by_path
                else (torch.zeros_like(leaf) if leaf is not None else None)
            ),
            embeddings,
        )
