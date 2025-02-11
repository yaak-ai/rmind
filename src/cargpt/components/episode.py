from collections.abc import Hashable, Mapping, Sequence
from enum import StrEnum, auto, unique
from functools import cached_property
from itertools import accumulate, pairwise, starmap
from operator import add, attrgetter, itemgetter
from typing import override

import more_itertools as mit
import torch
from einops import pack, rearrange, repeat
from jaxtyping import Float, Shaped
from loguru import logger
from pydantic import ConfigDict, RootModel, model_validator, validate_call
from pydantic.dataclasses import dataclass
from tensordict import TensorClass, TensorDict
from tensordict.tensorclass import (
    _eq as tensorclass_eq,  # pyright: ignore[reportAttributeAccessIssue]
)
from torch import Tensor
from torch.nn import Embedding, Module

from cargpt.utils import ModuleDict


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


@dataclass(frozen=True)
class TokenMeta:
    type: TokenType
    modality: Modality
    name: str

    @property
    def key(self) -> tuple[Modality, str]:
        return (self.modality, self.name)


class Timestep(RootModel[tuple[TokenMeta, ...]], Hashable):
    model_config = ConfigDict(frozen=True, ignored_types=(cached_property,))

    def __iter__(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    @model_validator(mode="before")
    @classmethod
    def validate_tokens(cls, v: Sequence[Sequence[str]]) -> tuple[TokenMeta, ...]:
        return tuple(starmap(TokenMeta, v))

    @cached_property
    def keys_by_type(self) -> Mapping[TokenType, Sequence[tuple[str, str]]]:
        return mit.map_reduce(
            self, keyfunc=attrgetter("type"), valuefunc=attrgetter("key")
        )


class Index(TensorClass["frozen"], Hashable):  # pyright: ignore[reportInvalidTypeArguments]
    image: TensorDict
    continuous: TensorDict
    discrete: TensorDict
    special: TensorDict

    def parse(self, src: Shaped[Tensor, "..."], dim: int = 1) -> TensorDict:
        # https://github.com/pytorch/pytorch/issues/30574

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

    def __hash__(self) -> int:
        items = tuple(
            (k, tuple(v.flatten().tolist()))
            for k, v in sorted(
                self.items(include_nested=True, leaves_only=True), key=itemgetter(0)
            )
        )

        return hash(items)


# HACK: need Index.__eq__ for @lru_cache but @tensorclass overrides it
Index.__eq__ = lambda self, other: tensorclass_eq(self, other).all()


class Episode(TensorClass["frozen"]):  # pyright: ignore[reportInvalidTypeArguments]
    input: TensorDict
    input_tokens: TensorDict
    input_embeddings: TensorDict
    position_embeddings: TensorDict
    index: Index
    timestep: Timestep

    @property
    def embeddings(self) -> TensorDict:
        return self.input_embeddings + self.position_embeddings

    @property
    def embeddings_packed(self) -> Float[Tensor, "b s d"]:
        embeddings = self.embeddings
        packed, _ = pack([embeddings[token.key] for token in self.timestep], "b t * d")

        return rearrange(packed, "b t s d -> b (t s) d")


class EpisodeBuilder(Module):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        special_tokens: Mapping[SpecialToken, int],
        timestep: Timestep,
        tokenizers: ModuleDict,
        embeddings: ModuleDict,
        position_encoding: ModuleDict,
        freeze: bool | None = None,
    ) -> None:
        super().__init__()

        self.timestep = timestep
        self.special_tokens = special_tokens
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

            self.requires_grad_(not freeze).train(not freeze)

    @override
    def forward(self, input: TensorDict) -> Episode:
        batch_size, device = input.batch_size, input.device
        special_tokens = (
            TensorDict(self.special_tokens, device=device)  # pyright: ignore[reportArgumentType]
            .expand(*batch_size, 1)
            .auto_batch_size_(input.batch_dims)
        )

        input_tokens = self.tokenizers.forward(
            input, batch_size=batch_size, device=device
        ).update({Modality.SPECIAL: special_tokens})

        input_embeddings = self.embeddings.forward(
            input_tokens.select(*self.embeddings.tree_paths()),
            batch_size=batch_size,
            device=device,
        )

        index = self._build_index(input_embeddings)

        position_embeddings = self._build_position_embeddings(
            input_embeddings, timestep_index=index[0]
        )

        return Episode(
            input=input,
            input_tokens=input_tokens,
            input_embeddings=input_embeddings,
            position_embeddings=position_embeddings,
            index=index,
            timestep=self.timestep,
            device=input.device,
        )

    def _build_index(self, embeddings: TensorDict) -> Index:
        _, t = embeddings.batch_size

        lengths = [embeddings.get_item_shape(token.key)[2] for token in self.timestep]
        timestep_length = sum(lengths)
        ranges = pairwise(accumulate(lengths, initial=0))

        timestep_index = Index.from_dict(
            {
                token.key: torch.arange(*_range)
                for token, _range in zip(self.timestep, ranges, strict=True)
            },
            device=embeddings.device,
        )

        return TensorDict.stack([  # pyright: ignore[reportReturnType]
            timestep_index + i * timestep_length for i in range(t)
        ])

    def _build_position_embeddings(
        self, src: TensorDict, timestep_index: Index
    ) -> TensorDict:
        (_, t), device = src.batch_size, src.device

        position_embeddings: TensorDict = src.new_zeros(t)  # pyright: ignore[reportAssignmentType, reportArgumentType]

        match pe_mod := self.position_encoding.get(
            (pe_k := PositionEncoding.IMAGE, "patch"), default=None
        ):
            case ModuleDict():
                num_rows = pe_mod.row.num_embeddings
                num_cols = pe_mod.col.num_embeddings
                row_pe = pe_mod.row(torch.arange(num_rows, device=device))
                col_pe = pe_mod.col(torch.arange(num_cols, device=device))
                row_pe = repeat(row_pe, "h d -> (h w) d", w=num_cols)
                col_pe = repeat(col_pe, "w d -> (h w) d", h=num_rows)

                position_embeddings.select(pe_k).apply(
                    lambda x: x + row_pe + col_pe,
                    inplace=True,  # NOTE
                )

            case None:
                pass

            case _:
                msg = f"position encoding for `{pe_k}`: {pe_mod}"
                raise NotImplementedError(msg)

        match pe_mod := self.position_encoding.get(
            pe_k := PositionEncoding.OBSERVATIONS, default=None
        ):
            case Embedding():
                keys = self.timestep.keys_by_type[TokenType.OBSERVATION]
                position = timestep_index.select(*keys).to_tensordict(retain_none=False)
                position_embedding = position.apply(pe_mod)
                position_embeddings.select(*keys).apply(
                    add,
                    position_embedding,
                    inplace=True,  # NOTE
                )

            case None:
                pass

            case _:
                raise NotImplementedError

        match pe_mod := self.position_encoding.get(
            pe_k := PositionEncoding.ACTIONS, default=None
        ):
            case Embedding():
                keys = self.timestep.keys_by_type[TokenType.ACTION]
                position = torch.arange(pe_mod.num_embeddings, device=device)
                position_embedding = pe_mod(position)
                position_embeddings.select(*keys).apply(
                    lambda emb: emb + position_embedding,
                    inplace=True,  # NOTE
                )

            case None:
                pass

            case _:
                msg = f"position encoding for `{pe_k}`: {pe_mod}"
                raise NotImplementedError(msg)

        match pe_mod := self.position_encoding.get(
            pe_k := PositionEncoding.SPECIAL, default=None
        ):
            case Embedding():
                keys = self.timestep.keys_by_type[TokenType.SPECIAL]
                position = torch.arange(pe_mod.num_embeddings, device=device)
                position_embedding = pe_mod(position)
                position_embeddings.select(*keys).apply(
                    lambda emb: emb + position_embedding,
                    inplace=True,  # NOTE
                )

            case None:
                pass

            case _:
                msg = f"position encoding for `{pe_k}`: {pe_mod}"
                raise NotImplementedError(msg)

        match pe_mod := self.position_encoding.get(
            pe_k := PositionEncoding.TIMESTEP, default=None
        ):
            case Embedding():
                # build a sequence starting from a random index (simplified [0])
                # e.g. given num_embeddings=20 and t=6, sample from ([0, 5], [1, 6], ..., [14, 19])
                # ---
                # [0] Randomized Positional Encodings Boost Length Generalization of Transformers (https://arxiv.org/abs/2305.16843)

                low, high = 0, pe_mod.num_embeddings - t + 1
                start = torch.randint(low, high, (1,)).item()
                position = torch.arange(start=start, end=start + t, device=device)
                position_embedding = rearrange(pe_mod(position), "t d -> t 1 d")
                position_embeddings.apply(
                    lambda emb: emb + position_embedding,
                    inplace=True,  # NOTE
                )

            case None:
                pass

            case _:
                msg = f"position encoding for `{pe_k}`: {pe_mod}"
                raise NotImplementedError(msg)

        return position_embeddings
