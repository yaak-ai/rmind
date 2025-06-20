from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum, auto, unique
from itertools import accumulate, pairwise
from operator import itemgetter
from typing import NamedTuple, final, overload, override

import more_itertools as mit
import torch
from einops import pack, rearrange, repeat
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from tensordict import TensorClass, TensorDict
from tensordict.tensorclass import (
    _eq as tensorclass_eq,  # pyright: ignore[reportAttributeAccessIssue]  # noqa: PLC2701
)
from torch import Tensor, nn
from torch.utils._pytree import (
    MappingKey,  # noqa: PLC2701
    key_get,  # noqa: PLC2701
    tree_leaves,  # noqa: PLC2701
    tree_map,  # noqa: PLC2701
    tree_map_with_path,  # noqa: PLC2701
)

from rmind.components.base import TensorDictExport
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


torch.export.register_dataclass((cls := TokenMeta), serialized_type_name=cls.__name__)


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


class Episode(TensorClass["frozen"]):  # pyright: ignore[reportInvalidTypeArguments]
    input: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    input_tokens: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    input_embeddings: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    position_embeddings: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]
    index: Index  # pyright: ignore[reportUninitializedInstanceVariable]
    timestep: TensorDict  # pyright: ignore[reportUninitializedInstanceVariable]

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


type Timestep = dict[TokenType, dict[tuple[Modality, str], int]]


@dataclass(frozen=True, kw_only=True)
class EpisodeExport:
    input: TensorDictExport
    input_tokens: TensorDictExport
    input_embeddings: TensorDictExport
    position_embeddings: TensorDictExport
    index: TensorDictExport
    timestep: Timestep

    @property
    def embeddings(self) -> TensorDictExport:
        return tree_map(torch.add, self.input_embeddings, self.position_embeddings)

    @property
    def embeddings_packed(self) -> Tensor:
        embeddings = self.embeddings
        paths = (
            (modality, name)
            for (_token_type, modality, name) in tree_paths(self.timestep)
        )
        packed, _ = pack([key_get(embeddings, path) for path in paths], "b t * d")

        return rearrange(packed, "b t s d -> b (t s) d")


torch.export.register_dataclass(
    (cls := EpisodeExport), serialized_type_name=cls.__name__
)


@final
class EpisodeBuilder(nn.Module):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        special_tokens: Mapping[SpecialToken, int],
        timestep: tuple[TokenMeta, ...],
        tokenizers: InstanceOf[ModuleDict],
        embeddings: InstanceOf[ModuleDict],
        position_encoding: InstanceOf[ModuleDict],
        freeze: bool | None = None,
    ) -> None:
        super().__init__()

        self.special_tokens = special_tokens
        self.timestep_flat = timestep
        self.tokenizers = tokenizers
        self.embeddings = embeddings
        self.position_encoding = position_encoding

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

    @overload
    def forward(self, input: TensorDict) -> Episode: ...

    @overload
    def forward(self, input: TensorDictExport) -> EpisodeExport: ...

    @override
    def forward(self, input: TensorDict | TensorDictExport) -> Episode | EpisodeExport:
        input_tokens = self.tokenizers.forward(input)

        if not (is_exporting := torch.compiler.is_exporting()):
            batch_size, device = input_tokens.batch_size, input_tokens.device
            input_tokens[Modality.SPECIAL] = (
                TensorDict(self.special_tokens, device=device)  # pyright: ignore[reportArgumentType]
                .expand(*batch_size, 1)
                .auto_batch_size_(input_tokens.batch_dims)
            )
        else:
            batch_size, device = mit.one({
                (tensor.shape[:2], tensor.device)
                for tensor in tree_leaves(input_tokens)
            })

            input_tokens[Modality.SPECIAL.value] = {
                k.value: torch.tensor(v, device=device).expand(*batch_size, 1)
                for k, v in self.special_tokens.items()
            }

        input_embeddings = self.embeddings.forward(input_tokens)

        index = self._build_index(input_embeddings)
        timestep_index = (
            index[0] if not is_exporting else tree_map(itemgetter(0), index)
        )

        timestep = (
            TensorDict({k: idx for idx, k in enumerate(self.timestep_flat)})
            if not is_exporting
            else unflatten_keys({
                tuple(map(str, k)): torch.tensor(idx)
                for idx, k in enumerate(self.timestep_flat)
            })
        )

        position_embeddings = self._build_position_embeddings(
            input_embeddings, timestep_index, timestep
        )

        kwargs = {
            "input": input,
            "input_tokens": input_tokens,
            "input_embeddings": input_embeddings,
            "position_embeddings": position_embeddings,
            "index": index,
            "timestep": timestep,
        }

        return (
            Episode(**kwargs, device=input.device)
            if not is_exporting
            else EpisodeExport(**kwargs)
        )

    @overload
    def _build_index(self, embeddings: TensorDict) -> Index: ...

    @overload
    def _build_index(self, embeddings: TensorDictExport) -> TensorDictExport: ...

    def _build_index(
        self, embeddings: TensorDict | TensorDictExport
    ) -> Index | TensorDictExport:
        if not (is_exporting := torch.compiler.is_exporting()):
            (_, t), device = embeddings.batch_size, embeddings.device

            lengths = [
                embeddings.get_item_shape((token.modality, token.name))[2]
                for token in self.timestep_flat
            ]

        else:
            (_, t), device = mit.one({
                (tensor.shape[:2], tensor.device) for tensor in tree_leaves(embeddings)
            })

            lengths = [
                key_get(
                    embeddings,
                    (MappingKey(token.modality.value), MappingKey(str(token.name))),
                ).shape[2]
                for token in self.timestep_flat
            ]

        timestep_length = sum(lengths)
        ranges = pairwise(accumulate(lengths, initial=0))

        if not is_exporting:
            timestep_index = Index.from_dict(
                {
                    (token.modality, token.name): torch.arange(*_range)
                    for token, _range in zip(self.timestep_flat, ranges, strict=True)
                },
                device=embeddings.device,
            )

            index = torch.stack([
                timestep_index + i * timestep_length for i in range(t)
            ])
        else:
            timestep_index = unflatten_keys({
                (token.modality.value, str(token.name)): torch.arange(
                    *_range, device=device
                )
                for token, _range in zip(self.timestep_flat, ranges, strict=True)
            })

            index = tree_map(
                lambda x: torch.stack([x + i * timestep_length for i in range(t)]),
                timestep_index,
            )

        return index

    @overload
    def _build_position_embeddings(
        self, src: TensorDict, timestep_index: Index, timestep: TensorDict
    ) -> TensorDict: ...

    @overload
    def _build_position_embeddings(
        self,
        src: TensorDictExport,
        timestep_index: TensorDictExport,
        timestep: TensorDictExport,
    ) -> TensorDictExport: ...

    def _build_position_embeddings(  # noqa: PLR0912
        self,
        src: TensorDict | TensorDictExport,
        timestep_index: Index | TensorDictExport,
        timestep: TensorDict | TensorDictExport,
    ) -> TensorDict | TensorDictExport:
        position_embeddings = {}

        (_, t), device = (
            (src.batch_size, src.device)
            if not (is_exporting := torch.compiler.is_exporting())
            else mit.one({
                (tensor.shape[:2], tensor.device) for tensor in tree_leaves(src)
            })
        )

        if (
            mod_pe := self.position_encoding.get(
                (k_pe := (PositionEncoding.IMAGE.value, "patch")), default=None
            )
        ) is not None:
            num_rows = mod_pe.row.num_embeddings
            num_cols = mod_pe.col.num_embeddings
            row_pe = mod_pe.row(torch.arange(num_rows, device=device))
            col_pe = mod_pe.col(torch.arange(num_cols, device=device))
            row_pe = repeat(row_pe, "h d -> (h w) d", w=num_cols)
            col_pe = repeat(col_pe, "w d -> (h w) d", h=num_rows)
            position_embedding = row_pe + col_pe

            if not is_exporting:
                keys = (
                    (modality, name)
                    for (_, modality, name) in timestep.keys(True, True)
                    if modality == Modality.IMAGE
                )
                position_embeddings[k_pe] = src.select(*keys).new_tensor(
                    position_embedding
                )
            else:
                paths = tuple(
                    (modality, name)
                    for (_, modality, name) in tree_paths(timestep)
                    if modality.key == Modality.IMAGE.value
                )

                position_embeddings[k_pe] = tree_map_with_path(
                    lambda path, _: position_embedding if path in paths else None, src
                )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.OBSERVATIONS.value, default=None
            )
        ) is not None:
            if not is_exporting:
                keys = timestep[TokenType.OBSERVATION].keys(True, True)
                position_embeddings[k_pe] = (
                    timestep_index.select(*keys)
                    .to_tensordict(retain_none=False)
                    .apply(mod_pe)
                )
            else:
                paths = tree_paths(timestep[TokenType.OBSERVATION.value])
                position_embeddings[k_pe] = tree_map_with_path(
                    lambda path, x: mod_pe(x) if path in paths else None, timestep_index
                )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.ACTIONS.value, default=None
            )
        ) is not None:
            position = torch.arange(mod_pe.num_embeddings, device=device)
            position_embedding = mod_pe(position)

            if not is_exporting:
                keys = timestep[TokenType.ACTION].keys(True, True)
                position_embeddings[k_pe] = src.select(*keys).new_tensor(
                    position_embedding
                )

            else:
                paths = tree_paths(timestep[TokenType.ACTION.value])
                position_embeddings[k_pe] = tree_map_with_path(
                    lambda path, _: position_embedding if path in paths else None, src
                )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.SPECIAL.value, default=None
            )
        ) is not None:
            position = torch.arange(mod_pe.num_embeddings, device=device)
            position_embedding = mod_pe(position)

            if not is_exporting:
                keys = timestep[TokenType.SPECIAL].keys(True, True)
                position_embeddings[k_pe] = src.select(*keys).new_tensor(
                    position_embedding
                )
            else:
                paths = tree_paths(timestep[TokenType.SPECIAL.value])
                position_embeddings[k_pe] = tree_map_with_path(
                    lambda path, _: position_embedding if path in paths else None, src
                )

        if (
            mod_pe := self.position_encoding.get(
                k_pe := PositionEncoding.TIMESTEP.value, default=None
            )
        ) is not None:
            # build a sequence starting from a random index (simplified [0])
            # e.g. given num_embeddings=20 and t=6, sample from ([0, 5], [1, 6], ..., [14, 19])
            # ---
            # [0] Randomized Positional Encodings Boost Length Generalization of Transformers (https://arxiv.org/abs/2305.16843)

            low, high = 0, mod_pe.num_embeddings - t + 1
            start = torch.randint(low, high, (1,)).item()
            position = torch.arange(start=start, end=start + t, device=device)
            position_embedding = rearrange(mod_pe(position), "t d -> t 1 d")

            position_embeddings[k_pe] = (
                src.new_tensor(position_embedding)
                if not is_exporting
                else tree_map(lambda _: position_embedding, src)
            )

        return (
            src.new_zeros().apply(
                lambda *xs: sum(xs), *position_embeddings.values(), default=0
            )
            if not is_exporting
            else tree_map(
                lambda *xs: sum(x for x in xs if x is not None),
                *position_embeddings.values(),
            )
        )
