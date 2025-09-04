from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum, auto, unique
from itertools import accumulate, pairwise
from operator import itemgetter
from typing import Any, NamedTuple, final, override

import more_itertools as mit
import torch
from einops import pack, rearrange, repeat
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from tensordict import TensorClass, TensorDict
from tensordict._pytree import (
    _td_flatten_with_keys,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
    _tensordict_flatten,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
    _tensordict_unflatten,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
)
from tensordict.tensorclass import (
    _eq as tensorclass_eq,  # pyright: ignore[reportAttributeAccessIssue]  # noqa: PLC2701
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
    IMAGE = auto()
    OBSERVATIONS = auto()
    ACTIONS = auto()
    SPECIAL = auto()
    TIMESTEP = auto()


class TokenMeta(NamedTuple):
    type: TokenType
    modality: Modality
    name: str


class Index(TensorClass["frozen"]):  # pyright: ignore[reportInvalidTypeArguments]
    image: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    continuous: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    discrete: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    special: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    context: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]

    def parse(self, src: Tensor, dim: int = 1) -> TensorDict:
        # https://github.com/pytorch/pytorch/issues/30574

        fn: Callable[[int], Tensor]
        match dim:
            case 0:
                fn = lambda idx: src[idx]  # noqa: E731

            case 1:
                fn = lambda idx: src[:, idx]  # noqa: E731

            case 2:
                fn = lambda idx: src[:, :, idx]  # noqa: E731

            case 3:
                fn = lambda idx: src[:, :, :, idx]  # noqa: E731

            case _:
                raise NotImplementedError

        batch_size = [*src.shape[:dim], *self.batch_size]

        return self.to_tensordict(retain_none=False).apply(
            fn, batch_size=batch_size, device=src.device, inplace=False
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
Index.__eq__ = lambda self, other: tensorclass_eq(self, other).all()


class Timestep(TensorDict, Hashable):
    @override
    def __eq__(self, other: object) -> bool:
        return super().__eq__(other).all()  # pyright: ignore[reportAttributeAccessIssue]

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
    _tensordict_unflatten,  # pyright: ignore[reportArgumentType]
    flatten_with_keys_fn=_td_flatten_with_keys,  # pyright: ignore[reportArgumentType]
)

TimestepExport = dict[str, dict[tuple[Modality, str], int]]


class Episode(TensorClass["frozen"]):  # pyright: ignore[reportInvalidTypeArguments]
    input: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    input_tokens: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    input_embeddings: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    position_embeddings: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    index: Index  # pyright: ignore[reportUninitializedInstanceVariable]
    timestep: Timestep  # pyright: ignore[reportUninitializedInstanceVariable]

    @property
    def embeddings(self) -> TensorDict:
        return self.input_embeddings + self.position_embeddings

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
    position_embeddings: TensorTree
    index: TensorTree
    timestep: TimestepExport

    @property
    def embeddings(self) -> TensorTree:
        return tree_map(
            lambda left, right: left + right
            if left is not None and right is not None
            else None,
            self.input_embeddings,
            self.position_embeddings,
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
        special_tokens: Mapping[SpecialToken, int],
        input_transform: InstanceOf[Module],
        tokenizers: InstanceOf[ModuleDict],
        embeddings: InstanceOf[ModuleDict],
        position_encoding: InstanceOf[ModuleDict],
        freeze: bool | None = None,
        modality_dropouts: dict[str, dict[str, float]] | None = None,  # fix
    ) -> None:
        super().__init__()

        self.special_tokens = special_tokens
        self.timestep = timestep
        self.input_transform = input_transform
        self.tokenizers = tokenizers
        self.embeddings = embeddings
        self.position_encoding = position_encoding
        self.modality_dropouts = modality_dropouts

        if freeze is not None:
            if freeze is False and (
                params_to_unfreeze := tuple(
                    k
                    for (k, v) in self.named_parameters(recurse=True)
                    if not v.requires_grad
                )
            ):
                logger.warning("unfreezing", params=params_to_unfreeze)

            self.requires_grad_(not freeze).train(not freeze)  # pyright: ignore[reportUnusedCallResult]

    @override
    def forward(self, batch: TensorTree) -> Episode | EpisodeExport:
        input = self.input_transform(batch)
        input_tokens = self.tokenizers(input)

        for key, dropout_probability in tree_leaves_with_path(self.modality_dropouts):
            if not batch.get("mask_token"):
                # we are not in training mode
                continue

            token = key_get(input_tokens, key)
            mask_token = key_get(batch["mask_token"], key)

            # Generate random mask per sample in batch
            batch_size = token.shape[0]
            sample_mask = (
                torch.rand(batch_size, device=token.device) < dropout_probability
            )

            masked_token = mask_token.expand(token.shape).to(
                dtype=token.dtype, device=token.device
            )
            input_tokens[key[0].key][key[1].key] = torch.where(
                sample_mask.view(-1, *([1] * (token.ndim - 1))), masked_token, token
            )

        batch_size, device = mit.one({
            (leaf.shape[:2], leaf.device)
            for leaf in tree_leaves(input_tokens)
            if leaf is not None
        })

        input_tokens[Modality.SPECIAL.value] = {
            k.value: torch.tensor(v, device=device).expand(*batch_size, 1)
            for k, v in self.special_tokens.items()
        }

        input_embeddings = self.embeddings(input_tokens)

        index = self._build_index(input_embeddings)
        timestep_index = tree_map(itemgetter(0), index)

        timestep = unflatten_keys({
            tuple(map(str, k)): idx for idx, k in enumerate(self.timestep)
        })

        position_embeddings = self._build_position_embeddings(
            input_embeddings, timestep_index, timestep
        )

        return (
            EpisodeExport(
                input=input,
                input_tokens=input_tokens,
                input_embeddings=input_embeddings,
                position_embeddings=position_embeddings,
                index=index,
                timestep=timestep,
            )
            if torch.compiler.is_exporting()
            else Episode(
                input=TensorDict.from_dict(
                    input, batch_dims=2
                ).filter_non_tensor_data(),  # pyright: ignore[reportAttributeAccessIssue]
                input_tokens=TensorDict.from_dict(
                    input_tokens, batch_dims=2
                ).filter_non_tensor_data(),  # pyright: ignore[reportAttributeAccessIssue]
                input_embeddings=TensorDict.from_dict(
                    input_embeddings, batch_dims=2
                ).filter_non_tensor_data(),  # pyright: ignore[reportAttributeAccessIssue]
                position_embeddings=TensorDict.from_dict(
                    position_embeddings,  # pyright: ignore[reportArgumentType]
                    batch_dims=2,
                ).filter_non_tensor_data(),  # pyright: ignore[reportAttributeAccessIssue]
                index=Index.from_dict(index, batch_dims=1),
                timestep=Timestep.from_dict(timestep),
                device=device,
            )
        )

    def _build_index(self, embeddings: TensorTree) -> TensorTree:
        (_, t), device = mit.one({
            (leaf.shape[:2], leaf.device)
            for leaf in tree_leaves(embeddings)
            if leaf is not None
        })

        lengths = [
            key_get(
                embeddings,
                (MappingKey(token.modality.value), MappingKey(str(token.name))),  # pyright: ignore[reportArgumentType]
            ).shape[2]
            for token in self.timestep
        ]

        timestep_length = sum(lengths)
        ranges = pairwise(accumulate(lengths, initial=0))

        timestep_index = unflatten_keys({
            (token.modality.value, str(token.name)): torch.arange(
                *_range, device=device
            )
            for token, _range in zip(self.timestep, ranges, strict=True)
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
    ) -> TensorTree:
        position_embeddings = {}

        (_, t), device = mit.one({
            (leaf.shape[:2], leaf.device)
            for leaf in tree_leaves(embeddings)
            if leaf is not None
        })

        if (
            mod_pe := self.position_encoding.get(
                (k_pe := (PositionEncoding.IMAGE.value, "patch")), default=None
            )
        ) is not None:
            num_rows = mod_pe.row.num_embeddings  # pyright: ignore[reportAttributeAccessIssue]
            num_cols = mod_pe.col.num_embeddings  # pyright: ignore[reportAttributeAccessIssue]
            row_pe = mod_pe.row(torch.arange(num_rows, device=device))  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue, reportArgumentType]
            col_pe = mod_pe.col(torch.arange(num_cols, device=device))  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue, reportArgumentType]
            row_pe = repeat(row_pe, "h d -> (h w) d", w=num_cols)
            col_pe = repeat(col_pe, "w d -> (h w) d", h=num_rows)
            position_embedding = row_pe + col_pe

            paths = tuple(
                (modality, name)
                for (_, modality, name) in tree_paths(timestep)
                if modality.key == Modality.IMAGE.value  # pyright: ignore[reportAttributeAccessIssue]
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
                lambda path, _: mod_pe(key_get(timestep_index, path))  # pyright: ignore[reportCallIssue]
                if path in paths
                else None,
                embeddings,
            )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.ACTIONS.value, default=None
            )
        ) is not None:
            position = torch.arange(mod_pe.num_embeddings, device=device)  # pyright: ignore[reportCallIssue, reportArgumentType, reportAttributeAccessIssue]
            position_embedding = mod_pe(position)  # pyright: ignore[reportCallIssue]
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
            position = torch.arange(mod_pe.num_embeddings, device=device)  # pyright: ignore[reportCallIssue, reportArgumentType, reportAttributeAccessIssue]
            position_embedding = mod_pe(position)  # pyright: ignore[reportCallIssue]
            paths = tree_paths(timestep[TokenType.SPECIAL.value])
            position_embeddings[k_pe] = tree_map_with_path(
                lambda path, _: position_embedding if path in paths else None,
                embeddings,
            )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.TIMESTEP.value, default=None
            )
        ) is not None:
            if not torch.compiler.is_exporting():
                # build a sequence starting from a random index (simplified [0])
                # e.g. given num_embeddings=20 and t=6, sample from ([0, 5], [1, 6], ..., [14, 19])
                # ---
                # [0] Randomized Positional Encodings Boost Length Generalization of Transformers (https://arxiv.org/abs/2305.16843)

                low, high = 0, mod_pe.num_embeddings - t + 1  # pyright: ignore[reportAttributeAccessIssue]
                start = torch.randint(low, high, (1,)).item()
                position = torch.arange(start=start, end=start + t, device=device)
            else:
                position = torch.arange(t, device=device)

            position_embedding = rearrange(mod_pe(position), "t d -> t 1 d")  # pyright: ignore[reportCallIssue]

            position_embeddings[k_pe] = tree_map(
                lambda leaf: position_embedding if leaf is not None else None,
                embeddings,
            )

        return tree_map(
            lambda *xs: sum(leaves)
            if (leaves := [x for x in xs if x is not None])
            else None,
            *position_embeddings.values(),
        )
