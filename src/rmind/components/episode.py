from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum, auto, unique
from itertools import accumulate, pairwise
from operator import itemgetter
from typing import Any, NamedTuple, final

from typing_extensions import override

import torch
from einops import pack, rearrange, repeat
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from tensordict import TensorClass, TensorDict
from tensordict._pytree import (
    _td_flatten_with_keys,  # noqa: PLC2701
    _tensordict_flatten,  # noqa: PLC2701
    _tensordict_unflatten,  # noqa: PLC2701
)
from tensordict.tensorclass import (
    _eq as tensorclass_eq,  # noqa: PLC2701  # ty:ignore[unresolved-import]
)
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import (
    MappingKey,  # noqa: PLC2701
    key_get,  # noqa: PLC2701
    register_pytree_node,  # noqa: PLC2701
    tree_leaves,  # noqa: PLC2701
    tree_leaves_with_path,  # noqa: PLC2701
    tree_map,  # noqa: PLC2701
    tree_map_with_path,  # noqa: PLC2701
)

from rmind.components.base import TensorTree
from rmind.components.containers import ModuleDict
from rmind.utils.pytree import tree_paths, unflatten_keys

logger = get_logger(__name__)


def _get_batch_info_from_tree(tree: Any) -> tuple[tuple[int, int], torch.device]:
    """Get batch_size (shape[:2]) and device from the first tensor leaf in a tree.

    This is a trace-friendly replacement for mit.one() which fails during ONNX
    export because set operations on traced tensor shapes don't work correctly.

    Args:
        tree: A nested structure containing tensors

    Returns:
        Tuple of (batch_size as (B, T), device)
    """
    for leaf in tree_leaves(tree):
        if leaf is not None and isinstance(leaf, Tensor):
            return leaf.shape[:2], leaf.device
    msg = "No tensor leaves found in tree"
    raise ValueError(msg)


def _is_exporting() -> bool:
    """Check if we're in an export context (torch.export or JIT tracing).

    Returns True during:
    - torch.export.export() (dynamo export) - detected by torch.compiler.is_exporting()
    - torch.onnx.export() with dynamo=False (JIT tracing) - detected by torch.jit.is_tracing()

    This is used to select EpisodeExport (plain tensors) vs Episode (TensorDict).
    """
    return torch.compiler.is_exporting() or torch.jit.is_tracing()  # ty:ignore[possibly-missing-attribute]


class StrEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name.lower()

    def __str__(self) -> str:
        return self.value


@unique
class TokenType(StrEnum):
    OBSERVATION = auto()
    ACTION = auto()
    SPECIAL = auto()


@unique
class Modality(StrEnum):
    IMAGE = auto()
    CONTINUOUS = auto()
    DISCRETE = auto()
    SPECIAL = auto()
    CONTEXT = auto()


@unique
class SpecialToken(StrEnum):
    OBSERVATION_SUMMARY = auto()
    OBSERVATION_HISTORY = auto()
    ACTION_SUMMARY = auto()


@unique
class PositionEncoding(StrEnum):
    OBSERVATIONS = auto()
    ACTIONS = auto()
    SPECIAL = auto()
    TIMESTEP = auto()
    CONTEXT = auto()


class TokenMeta(NamedTuple):
    type: TokenType
    modality: Modality
    name: str


class Index(TensorClass["frozen"]):
    image: TensorDict
    continuous: TensorDict
    discrete: TensorDict
    special: TensorDict
    context: TensorDict

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

    @override
    def __hash__(self) -> int:
        items = tuple(
            (k, tuple(v.flatten().tolist()))
            for k, v in sorted(
                self.items(include_nested=True, leaves_only=True), key=itemgetter(0)
            )
        )

        return hash(items)


# HACK: need Index.__eq__ for @lru_cache but @tensorclass overrides it  # noqa: FIX004
Index.__eq__ = lambda self, other: tensorclass_eq(self, other).all()  # ty:ignore[invalid-assignment]


class Timestep(TensorDict, Hashable):
    @override
    def __eq__(self, other: object) -> bool:  # ty:ignore[invalid-method-override]
        return super().__eq__(other).all()

    @override
    def __hash__(self) -> int:
        return hash(
            tuple(
                k
                for k, _ in sorted(
                    self.items(include_nested=True, leaves_only=True), key=itemgetter(1)
                )
            )
        )


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
    position_embeddings: TensorDict
    index: Index
    timestep: Timestep

    @property
    def embeddings(self) -> TensorDict:
        return self.projected_embeddings + self.position_embeddings

    @property
    def projected_embeddings_packed(self) -> Tensor:
        """Packed projected embeddings WITHOUT position embeddings.

        Use this for caching in sliding window inference where position
        embeddings need to be applied after concatenation.
        """
        keys = (
            (modality, name)
            for (_token_type, modality, name), _pos in sorted(
                self.timestep.items(include_nested=True, leaves_only=True),
                key=itemgetter(1),
            )
        )
        packed, _ = pack([self.projected_embeddings[key] for key in keys], "b t * d")

        return rearrange(packed, "b t s d -> b (t s) d")

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

        return rearrange(packed, "b t s d -> b (t s) d")


@dataclass(frozen=True, kw_only=True)
class EpisodeExport:
    input: TensorTree
    input_tokens: TensorTree
    input_embeddings: TensorTree
    projected_embeddings: TensorTree
    position_embeddings: TensorTree
    index: TensorTree
    timestep: TimestepExport

    @property
    def embeddings(self) -> TensorTree:
        return tree_map(
            lambda left, right: (
                left + right if left is not None and right is not None else None
            ),
            self.projected_embeddings,
            self.position_embeddings,
        )

    @property
    def projected_embeddings_packed(self) -> Tensor:
        """Packed projected embeddings WITHOUT position embeddings.

        Use this for caching in sliding window inference where position
        embeddings need to be applied after concatenation.
        """
        paths = (
            (modality, name)
            for (_token_type, modality, name), _pos in sorted(
                tree_leaves_with_path(self.timestep), key=itemgetter(1)
            )
        )

        packed, _ = pack(
            [key_get(self.projected_embeddings, path) for path in paths], "b t * d"
        )

        return rearrange(packed, "b t s d -> b (t s) d")

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


@dataclass(frozen=True, kw_only=True)
class TimestepEmbeddings:
    """Intermediate representation of computed embeddings for a single or multiple timesteps.

    Used for caching embeddings during inference to avoid recomputation.
    """

    input: TensorTree
    input_tokens: TensorTree
    input_embeddings: TensorTree
    projected_embeddings: TensorTree
    batch_size: tuple[int, int]
    device: torch.device


torch.export.register_dataclass(
    (cls := TimestepEmbeddings), serialized_type_name=cls.__name__
)


@final
class EpisodeBuilder(Module):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        timestep: tuple[TokenMeta, ...],
        special_tokens: Mapping[SpecialToken, int],
        input_transform: InstanceOf[Module],
        tokenizers: InstanceOf[ModuleDict],
        embeddings: InstanceOf[ModuleDict],
        projections: InstanceOf[ModuleDict],
        position_encoding: InstanceOf[ModuleDict],
        freeze: bool | None = None,
    ) -> None:
        super().__init__()

        self.special_tokens: Mapping[SpecialToken, int] = special_tokens
        self.timestep: tuple[TokenMeta, ...] = timestep
        self.input_transform: Module = input_transform
        self.tokenizers: ModuleDict = tokenizers
        self.embeddings: ModuleDict = embeddings
        self.projections: ModuleDict = projections
        self.position_encoding: ModuleDict = position_encoding

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

    def compute_embeddings(self, batch: TensorTree) -> TimestepEmbeddings:
        """Compute embeddings for a batch without building the full episode.

        This method performs the expensive embedding computation (input transform,
        tokenization, embedding lookup, projection) but stops before building
        position embeddings and the final Episode structure.

        Useful for caching embeddings during inference.

        Args:
            batch: Input batch data

        Returns:
            TimestepEmbeddings containing computed embeddings and metadata
        """
        input = self.input_transform(batch)
        input_tokens = self.tokenizers(input)

        batch_size, device = _get_batch_info_from_tree(input_tokens)

        input_tokens[Modality.SPECIAL.value] = {
            k.value: torch.tensor(v, device=device).expand(*batch_size, 1)
            for k, v in self.special_tokens.items()
        }

        input_embeddings = self.embeddings(input_tokens)
        projected_embeddings = self.projections(input_embeddings)

        return TimestepEmbeddings(
            input=input,
            input_tokens=input_tokens,
            input_embeddings=input_embeddings,
            projected_embeddings=projected_embeddings,
            batch_size=batch_size,
            device=device,
        )

    def assemble_episode(
        self, timestep_embeddings: TimestepEmbeddings, timestep_offset: int | None = None
    ) -> Episode | EpisodeExport:
        """Assemble an Episode from pre-computed embeddings.

        This method takes pre-computed embeddings and builds the final Episode
        structure including position embeddings and index.

        Args:
            timestep_embeddings: Pre-computed embeddings from compute_embeddings()
            timestep_offset: Offset for timestep position encoding (for incremental inference).
                             When processing timestep 5 alone, set offset=5 so it gets position 5.
                             When None, use random position during training or 0 during export.

        Returns:
            Episode or EpisodeExport depending on export context
        """
        input = timestep_embeddings.input
        input_tokens = timestep_embeddings.input_tokens
        input_embeddings = timestep_embeddings.input_embeddings
        projected_embeddings = timestep_embeddings.projected_embeddings
        batch_size = timestep_embeddings.batch_size
        device = timestep_embeddings.device

        index = self._build_index(projected_embeddings)
        timestep_index = tree_map(itemgetter(0), index)

        timestep = unflatten_keys({
            tuple(map(str, k)): idx for idx, k in enumerate(self.timestep)
        })

        position_embeddings = self._build_position_embeddings(
            input_embeddings, timestep_index, timestep, timestep_offset
        )

        return (
            EpisodeExport(
                input=input,
                input_tokens=input_tokens,
                input_embeddings=input_embeddings,
                projected_embeddings=projected_embeddings,
                position_embeddings=position_embeddings,
                index=index,
                timestep=timestep,
            )
            if _is_exporting()
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
                position_embeddings=TensorDict.from_dict(
                    position_embeddings,  # ty:ignore[invalid-argument-type]
                    batch_dims=2,
                ).filter_non_tensor_data(),
                index=Index.from_dict(index, batch_dims=1),
                timestep=Timestep.from_dict(timestep),
                device=device,
            )
        )

    @override
    def forward(
        self, batch: TensorTree, timestep_offset: int | None = None
    ) -> Episode | EpisodeExport:
        """Build an Episode from input batch.

        This method has all logic inline for export compatibility.
        For inference with caching, use compute_embeddings() and
        assemble_episode() separately.

        Args:
            batch: Input batch dict
            timestep_offset: Offset for timestep position encoding (for incremental inference).
                             When processing timestep 5 alone, set offset=5 so it gets position 5.
                             When None, use random position during training or 0 during export.
        """
        input = self.input_transform(batch)
        input_tokens = self.tokenizers(input)

        batch_size, device = _get_batch_info_from_tree(input_tokens)

        input_tokens[Modality.SPECIAL.value] = {
            k.value: torch.tensor(v, device=device).expand(*batch_size, 1)
            for k, v in self.special_tokens.items()
        }

        input_embeddings = self.embeddings(input_tokens)
        projected_embeddings = self.projections(input_embeddings)

        index = self._build_index(projected_embeddings)
        timestep_index = tree_map(itemgetter(0), index)

        timestep = unflatten_keys({
            tuple(map(str, k)): idx for idx, k in enumerate(self.timestep)
        })

        position_embeddings = self._build_position_embeddings(
            input_embeddings, timestep_index, timestep, timestep_offset
        )

        return (
            EpisodeExport(
                input=input,
                input_tokens=input_tokens,
                input_embeddings=input_embeddings,
                projected_embeddings=projected_embeddings,
                position_embeddings=position_embeddings,
                index=index,
                timestep=timestep,
            )
            if _is_exporting()
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
                position_embeddings=TensorDict.from_dict(
                    position_embeddings,  # ty:ignore[invalid-argument-type]
                    batch_dims=2,
                ).filter_non_tensor_data(),
                index=Index.from_dict(index, batch_dims=1),
                timestep=Timestep.from_dict(timestep),
                device=device,
            )
        )

    def _build_index(self, embeddings: TensorTree) -> TensorTree:
        (_, t), device = _get_batch_info_from_tree(embeddings)

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

    def _build_position_embeddings(
        self,
        embeddings: TensorTree,
        timestep_index: TensorTree,
        timestep: TimestepExport,
        timestep_offset: int | None = None,
    ) -> TensorTree:
        """Build position embeddings for the episode.

        Args:
            embeddings: Token embeddings
            timestep_index: Index mapping for timesteps
            timestep: Timestep structure
            timestep_offset: Offset for timestep position encoding (for incremental inference).
                             When processing timestep 5 alone, set offset=5 so it gets position 5.
                             When None, use random position during training or 0 during export.
        """
        position_embeddings = {}

        (_, t), device = _get_batch_info_from_tree(embeddings)

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.TIMESTEP.value, default=None
            )
        ) is not None:
            if timestep_offset is not None or torch.compiler.is_exporting():  # ty:ignore[possibly-missing-attribute]
                # Use explicit timestep_offset for incremental inference or export
                offset = timestep_offset if timestep_offset is not None else 0
                position = torch.arange(
                    start=offset, end=offset + t, device=device
                )
            else:
                # build a sequence starting from a random index (simplified [0])
                # e.g. given num_embeddings=20 and t=6, sample from ([0, 5], [1, 6], ..., [14, 19])
                # ---
                # [0] Randomized Positional Encodings Boost Length Generalization of Transformers (https://arxiv.org/abs/2305.16843)

                low, high = 0, mod_pe.num_embeddings - t + 1  # ty:ignore[unresolved-attribute]
                start = torch.randint(low, high, (1,)).item()
                position = torch.arange(start=start, end=start + t, device=device)

            position_embeddings[k_pe] = tree_map(
                lambda leaf: (
                    repeat(
                        mod_pe(position),  # ty:ignore[call-non-callable]
                        "... t d -> ... t n d",
                        n=leaf.shape[-2],
                    )
                    if leaf is not None
                    else None
                ),
                embeddings,
            )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := (PositionEncoding.CONTEXT.value, "waypoints"), default=None
            )
        ) is not None:
            position = torch.arange(mod_pe.num_embeddings, device=device)  # ty:ignore[unresolved-attribute]
            position_embedding = mod_pe(position)
            paths = tuple(
                (modality, name)
                for (_, modality, name) in tree_paths(timestep)
                if modality.key == Modality.CONTEXT.value  # ty:ignore[unresolved-attribute]
            )

            position_embeddings[k_pe] = tree_map_with_path(
                lambda path, _: position_embedding if path in paths else None,
                embeddings,
            )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.OBSERVATIONS.value, default=None
            )
        ) is not None:
            paths = tree_paths(timestep[TokenType.OBSERVATION.value])
            position_embeddings[k_pe] = tree_map_with_path(
                lambda path, _: (
                    mod_pe(key_get(timestep_index, path))  # ty:ignore[call-non-callable]
                    if path in paths
                    else None
                ),
                embeddings,
            )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.ACTIONS.value, default=None
            )
        ) is not None:
            position = torch.arange(mod_pe.num_embeddings, device=device)  # ty:ignore[unresolved-attribute]
            position_embedding = mod_pe(position)
            paths = tree_paths(timestep[TokenType.ACTION.value])
            position_embeddings[k_pe] = tree_map_with_path(
                lambda path, _: position_embedding if path in paths else None,
                embeddings,
            )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.SPECIAL.value, default=None
            )
        ) is not None:
            position = torch.arange(mod_pe.num_embeddings, device=device)  # ty:ignore[unresolved-attribute]
            position_embedding = mod_pe(position)
            paths = tree_paths(timestep[TokenType.SPECIAL.value])
            position_embeddings[k_pe] = tree_map_with_path(
                lambda path, _: position_embedding if path in paths else None,
                embeddings,
            )

        return tree_map(
            lambda *xs: (
                sum(leaves) if (leaves := [x for x in xs if x is not None]) else None
            ),
            *position_embeddings.values(),
        )

    def apply_timestep_position_embeddings(
        self,
        packed_embeddings: Tensor,
        num_timesteps: int,
        timestep_offset: int = 0,
    ) -> Tensor:
        """Apply timestep position embeddings to a packed embedding tensor.

        This method is used for incremental inference where we concatenate
        cached projected embeddings with new ones, then apply position embeddings
        to the full sequence.

        Args:
            packed_embeddings: [B, S, D] packed projected embeddings (without position embeddings)
            num_timesteps: Number of timesteps in the packed tensor
            timestep_offset: Starting position for timestep encoding (default 0)

        Returns:
            [B, S, D] embeddings with timestep position embeddings added
        """
        mod_pe = self.position_encoding.get(PositionEncoding.TIMESTEP.value, default=None)
        if mod_pe is None:
            return packed_embeddings

        device = packed_embeddings.device
        batch_size, seq_len, embed_dim = packed_embeddings.shape
        tokens_per_timestep = seq_len // num_timesteps

        # Get timestep positions
        positions = torch.arange(
            start=timestep_offset,
            end=timestep_offset + num_timesteps,
            device=device,
        )

        # Get position embeddings [T, D]
        pos_emb = mod_pe(positions)

        # Expand to [T, tokens_per_timestep, D] then reshape to [S, D]
        pos_emb_expanded = pos_emb.unsqueeze(1).expand(-1, tokens_per_timestep, -1)
        pos_emb_packed = pos_emb_expanded.reshape(seq_len, embed_dim)

        # Add to embeddings [B, S, D] + [S, D] -> [B, S, D]
        return packed_embeddings + pos_emb_packed
