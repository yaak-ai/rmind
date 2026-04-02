from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate, pairwise
from operator import itemgetter
from typing import Any, NamedTuple, final, override

import more_itertools as mit
import torch
from einops import pack, rearrange, repeat
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from tensordict import TensorClass, TensorDict
from tensordict._pytree import (  # noqa: PLC2701
    _td_flatten_with_keys,
    _tensordict_flatten,
    _tensordict_unflatten,
)
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import (  # noqa: PLC2701
    MappingKey,
    key_get,
    register_pytree_node,
    tree_leaves,
    tree_leaves_with_path,
    tree_map,
    tree_map_with_path,
)

from rmind.components.base import Modality, TensorTree, TokenType
from rmind.components.containers import ModuleDict
from rmind.components.mask import (
    AttentionMask,
    AttentionMaskBuilder,
    TorchAttentionMaskLegend,
)
from rmind.utils.pytree import unflatten_keys

logger = get_logger(__name__)


class TokenMeta(NamedTuple):
    type: TokenType
    modality: Modality
    name: str


class Index(TensorClass["frozen"]):
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


class Timestep(TensorDict):
    pass


register_pytree_node(
    Timestep,
    _tensordict_flatten,
    _tensordict_unflatten,  # ty:ignore[invalid-argument-type]
    flatten_with_keys_fn=_td_flatten_with_keys,
)

TimestepExport = dict[str, dict[tuple[Modality, str], int]]


class Episode(TensorClass["frozen"]):
    input: TensorDict
    input_tokens: TensorDict
    input_embeddings: TensorDict
    projected_embeddings: TensorDict
    role_embeddings: TensorDict
    index: Index
    timestep: Timestep
    attention_mask: AttentionMask

    @property
    def embeddings(self) -> TensorDict:
        return self.projected_embeddings + self.role_embeddings

    @property
    def embeddings_packed(self) -> Tensor:
        embeddings = self.embeddings
        keys = (
            (modality, name)
            for (_token_type, modality, name), _pos in sorted(
                self.timestep.items(include_nested=True, leaves_only=True),
                key=itemgetter(1),
            )
        )
        packed, _ = pack([embeddings[key] for key in keys], "b t * d")

        return rearrange(packed, "b t s d -> b (t s) d")  # ty:ignore[invalid-return-type]


@dataclass(frozen=True, kw_only=True)
class EpisodeExport:
    input: TensorTree
    input_tokens: TensorTree
    input_embeddings: TensorTree
    projected_embeddings: TensorTree
    role_embeddings: TensorTree
    index: TensorTree
    timestep: TimestepExport
    attention_mask: AttentionMask

    @property
    def embeddings(self) -> TensorTree:
        return tree_map(
            lambda left, right: (
                left + right if left is not None and right is not None else None
            ),
            self.projected_embeddings,
            self.role_embeddings,
        )

    @property
    def embeddings_packed(self) -> Tensor:
        embeddings = self.embeddings
        paths = (
            (modality, name)
            for (_token_type, modality, name), _pos in sorted(
                tree_leaves_with_path(self.timestep), key=itemgetter(1)
            )
        )

        packed, _ = pack([key_get(embeddings, path) for path in paths], "b t * d")

        return rearrange(packed, "b t s d -> b (t s) d")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


torch.export.register_dataclass(
    (cls := EpisodeExport), serialized_type_name=cls.__name__
)


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
        attention_mask_builder: InstanceOf[AttentionMaskBuilder],
        freeze: bool | None = None,
    ) -> None:
        super().__init__()

        self.special_tokens: Mapping[str, Mapping[str, tuple[int, ...]]] = (
            special_tokens
        )
        self.timestep: tuple[TokenMeta, ...] = timestep
        self.input_transform: Module = input_transform
        self.tokenizers: ModuleDict = tokenizers
        self.embeddings: ModuleDict = embeddings
        self.projections: ModuleDict = projections
        self.role_encoding: Module = role_encoding
        self.attention_mask_builder: AttentionMaskBuilder = attention_mask_builder
        self.register_buffer("_attention_mask", None, persistent=False)
        role_idx_by_type_modality: dict[tuple[str, str], int] = {}
        self._role_idx_by_path: dict[tuple[MappingKey, MappingKey], int] = {
            (
                MappingKey(token.modality.value),
                MappingKey(
                    str(token.name)
                ),  # https://github.com/yaak-ai/rmind/issues/204
            ): role_idx_by_type_modality.setdefault(
                (token.type.value, token.modality.value), len(role_idx_by_type_modality)
            )
            for token in timestep
        }
        if freeze is not None:
            if freeze is False and (
                params_to_unfreeze := tuple(
                    k
                    for (k, v) in self.named_parameters(recurse=True)
                    if not v.requires_grad
                )
            ):
                logger.warning("unfreezing", params=params_to_unfreeze)

            self.requires_grad_(not freeze).train(not freeze)

    @override
    def forward(self, batch: TensorTree) -> Episode | EpisodeExport:
        input = self.input_transform(batch)
        input_tokens = self.tokenizers(input)

        (b, t), device = mit.one({
            (leaf.shape[:2], leaf.device)
            for leaf in tree_leaves(input_tokens)
            if leaf is not None
        })

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

        timestep = unflatten_keys({
            tuple(map(str, k)): idx for idx, k in enumerate(self.timestep)
        })

        attention_mask = AttentionMask.from_tensor(
            mask_tensor=self._build_attention_mask_tensor(
                index=index, timestep=timestep
            ),
            legend=TorchAttentionMaskLegend,
        )

        role_embeddings = self._build_role_embeddings(projected_embeddings)
        return (
            EpisodeExport(
                input=input,
                input_tokens=input_tokens,
                input_embeddings=input_embeddings,
                projected_embeddings=projected_embeddings,
                role_embeddings=role_embeddings,
                index=index,
                timestep=timestep,
                attention_mask=attention_mask,
            )
            if torch.compiler.is_exporting()
            else Episode(
                input=TensorDict.from_dict(
                    input, batch_dims=2
                ).filter_non_tensor_data(),
                input_tokens=TensorDict.from_dict(
                    input_tokens, batch_dims=2
                ).filter_non_tensor_data(),
                input_embeddings=TensorDict.from_dict(
                    input_embeddings, batch_dims=2
                ).filter_non_tensor_data(),
                projected_embeddings=TensorDict.from_dict(
                    projected_embeddings, batch_dims=2
                ).filter_non_tensor_data(),
                role_embeddings=TensorDict.from_dict(
                    role_embeddings,  # ty:ignore[invalid-argument-type]
                    batch_dims=2,
                ).filter_non_tensor_data(),
                index=Index.from_dict(index, batch_dims=1),
                timestep=Timestep.from_dict(timestep),
                attention_mask=attention_mask,
                device=device,
            )
        )

    def _build_attention_mask_tensor(
        self, *, index: TensorTree, timestep: TimestepExport
    ) -> Tensor:
        """Build (or return cached) attention mask tensor.

        WARNING: attention_mask_builder is not trace-friendly, so torch.export relies
        on this cache being warm. An eager forward pass must run *before*
        torch.export.export() — see export_onnx.py.
        """
        if self._attention_mask is not None:
            return self._attention_mask

        if torch.compiler.is_exporting():
            logger.warning(
                "building attention mask during export; "
                "run an eager forward pass first to populate the cache"
            )

        attention_mask_tensor = self.attention_mask_builder(
            index=index, timestep=timestep, legend=TorchAttentionMaskLegend
        )
        if not torch.compiler.is_exporting():
            self._attention_mask = attention_mask_tensor
        return attention_mask_tensor

    def _build_index(self, embeddings: TensorTree) -> TensorTree:
        (_, t), device = mit.one({
            (leaf.shape[:2], leaf.device)
            for leaf in tree_leaves(embeddings)
            if leaf is not None
        })

        lengths = [
            key_get(
                embeddings,
                (MappingKey(token.modality.value), MappingKey(str(token.name))),  # ty:ignore[invalid-argument-type]
            ).shape[2]
            for token in self.timestep
        ]

        timestep_length = sum(lengths)
        ranges = pairwise(accumulate(lengths, initial=0))

        timestep_index = unflatten_keys({
            (token.modality.value, str(token.name)): torch.arange(
                *range_, device=device
            )
            for token, range_ in zip(self.timestep, ranges, strict=True)
        })

        return tree_map(
            lambda x: torch.stack([x + i * timestep_length for i in range(t)]),
            timestep_index,
        )

    def _build_role_embeddings(self, embeddings: TensorTree) -> TensorTree:
        (_, t), device = mit.one({
            (leaf.shape[:2], leaf.device)
            for leaf in tree_leaves(embeddings)
            if leaf is not None
        })

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
