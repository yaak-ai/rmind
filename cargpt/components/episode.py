from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import cache
from itertools import accumulate, pairwise
from operator import add
from typing import cast

import torch
from einops import pack, rearrange, repeat
from jaxtyping import Float, Int, Shaped
from loguru import logger
from tensordict import TensorDict, tensorclass
from tensordict.utils import NestedKey
from torch import Tensor
from torch.nn import Module, ModuleDict


class TokenType(StrEnum):
    OBSERVATION = auto()
    ACTION = auto()
    SPECIAL = auto()


class Modality(StrEnum):
    IMAGE = auto()
    CONTINUOUS = auto()
    DISCRETE = auto()
    SPECIAL = auto()


class SpecialToken(StrEnum):
    OBSERVATION_SUMMARY = auto()
    OBSERVATION_HISTORY = auto()
    ACTION_SUMMARY = auto()


class PositionEncoding(StrEnum):
    OBSERVATIONS = auto()
    ACTIONS = auto()
    SPECIAL = auto()
    TIMESTEP = auto()


@dataclass(frozen=True, kw_only=True)
class Token:
    type: TokenType
    name: str
    modality: Modality

    @property
    def key(self) -> tuple[Modality, str]:
        return (self.modality, self.name)


@dataclass(frozen=True)
class Timestep:
    tokens: tuple[Token, ...]

    @classmethod
    def build(cls, *args: Sequence[str]):
        tokens: list[Token] = []
        for arg in args:
            match arg:
                case (type, modality, key):
                    token = Token(
                        type=TokenType(type),
                        modality=Modality(modality),
                        name=key,
                    )

                case ("special", key):
                    token = Token(
                        type=TokenType.SPECIAL,
                        modality=Modality.SPECIAL,
                        name=key,
                    )

                case _:
                    msg = f"invalid token: {arg}"
                    raise ValueError(msg)

            tokens.append(token)

        return cls(tuple(tokens))

    @cache  # noqa: B019
    def keys(self, token_type: TokenType) -> tuple[tuple[Modality, str], ...]:
        return tuple(token.key for token in self.tokens if token.type is token_type)


@tensorclass  # pyright: ignore
class Index:
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

        return self.to_tensordict().apply(
            fn,
            batch_size=batch_size,
            device=src.device,
            inplace=False,
        )

    @property
    def all_values(self) -> Int[Tensor, "d"]:
        return torch.cat(
            list(
                self.values(
                    include_nested=True,
                    leaves_only=True,
                )
            ),
            -1,
        ).flatten()

    @property
    def max(self) -> int:
        return max(
            self.apply(torch.max, batch_size=[]).values(
                include_nested=True,
                leaves_only=True,
            )
        ).item()


# NOTE: need Index.__hash__ and Index.__eq__ for @lru_cache'ing methods with Index arguments
# defining/assigning these outside of Index since @tensordict overrides most methods
from tensordict.tensorclass import _eq  # noqa: E402, PLC2701


def _index_hash(self) -> int:
    items = tuple(
        (k, tuple(v.flatten().tolist()))
        for k, v in sorted(
            self.items(
                include_nested=True,
                leaves_only=True,
            ),
            key=lambda x: x[0],
        )
    )

    return hash(items)


def _index_eq(self, other: Index) -> bool:
    return _eq(self, other).all()


Index.__hash__ = _index_hash
Index.__eq__ = _index_eq


@tensorclass  # pyright: ignore
class Episode:
    inputs: TensorDict
    tokenized: TensorDict
    embedded_nope: TensorDict
    embedded: TensorDict
    index: Index
    timestep: Timestep

    @property  # TODO: cache?
    def packed_embeddings(self) -> Float[Tensor, "b s d"]:
        embeddings, _ = pack(
            [self.embedded[token.key] for token in self.timestep.tokens],
            "b t * d",
        )
        return rearrange(embeddings, "b t s d -> b (t s) d")


class EpisodeBuilder(Module):
    def __init__(
        self,
        *,
        timestep: Timestep,
        special_tokens: Mapping[str, int],
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

            self.requires_grad_(not freeze).train(not freeze)  # pyright: ignore[reportUnusedCallResult]

    def build_episode(
        self,
        inputs: TensorDict,
        *,
        # TODO: make this less jarring
        masked_action_timestep_idx: list[int] | None = None,
        masked_observation_timestep_idx: list[int] | None = None,
    ) -> Episode:
        tokenized = inputs.named_apply(
            lambda nested_key, tensor: (
                self.tokenizers.get(nested_key, default=None)
                or self.tokenizers.get(nested_key[0])
            )(tensor),
            nested_keys=True,
        )

        tokenized[Modality.SPECIAL] = {
            k: torch.tensor(v).expand(*tokenized.batch_size, 1)
            for k, v in self.special_tokens.items()
        }

        embedded_nope = tokenized.named_apply(
            lambda nested_key, tensor: (
                self.embeddings.get(nested_key, default=None)
                or self.embeddings.get(nested_key[0])
            )(tensor),
            nested_keys=True,
        )

        # TODO: learnable mask token?
        if masked_action_timestep_idx is not None:
            embedded_nope.select(*self.timestep.keys(TokenType.ACTION))[
                :, masked_action_timestep_idx
            ] = -1.0

        if masked_observation_timestep_idx is not None:
            embedded_nope.select(*self.timestep.keys(TokenType.OBSERVATION))[
                :, masked_observation_timestep_idx
            ] = -1.0

        lengths = {
            token.key: embedded_nope.get_item_shape(token.key)[2]
            for token in self.timestep.tokens
        }
        timestep_index = self._build_timestep_index(lengths).to(embedded_nope.device)  # pyright: ignore
        timestep_length = sum(lengths.values())
        _, t = embedded_nope.batch_size
        index = timestep_index.apply(
            lambda x: torch.stack([x + i * timestep_length for i in range(t)]),
            batch_size=[t],
        )

        embedded = self._position_encode(embedded_nope, timestep_index)

        return Episode(  # pyright: ignore
            inputs=inputs,
            tokenized=tokenized,
            embedded_nope=embedded_nope,
            embedded=embedded,
            index=index,
            timestep=self.timestep,
            batch_size=[],
            device=inputs.device,
        )

    def _build_timestep_index(self, lengths: dict[NestedKey, int]) -> Index:
        ranges = dict(
            zip(
                lengths.keys(),
                pairwise(accumulate(lengths.values(), initial=0)),
            )
        )
        return Index.from_dict(
            {k: torch.arange(*v) for k, v in ranges.items()},
            batch_size=[],
        )

    def _position_encode(
        self,
        embeddings: TensorDict,
        index: Index,
    ) -> TensorDict:
        position_embeddings = cast(TensorDict, torch.zeros_like(embeddings[0]))
        device = position_embeddings.device

        if module := self.position_encoding.get(
            (key := Modality.IMAGE, "patch"), default=None
        ):
            num_rows = module.row.num_embeddings
            num_cols = module.col.num_embeddings
            row_pe = module.row(torch.arange(num_rows, device=device))
            col_pe = module.col(torch.arange(num_cols, device=device))
            row_pe = repeat(row_pe, "h d -> (h w) d", w=num_cols)
            col_pe = repeat(col_pe, "w d -> (h w) d", h=num_rows)

            position_embeddings.select(key).apply(
                lambda x: x + row_pe + col_pe,
                inplace=True,  # NOTE
            )

        if module := self.position_encoding.get(
            PositionEncoding.OBSERVATIONS, default=None
        ):
            keys = self.timestep.keys(TokenType.OBSERVATION)
            position = index.select(*keys).to_tensordict()
            position_embedding = position.apply(module)
            position_embeddings.select(*keys).apply(
                add,
                position_embedding,
                inplace=True,  # NOTE
            )

        if module := self.position_encoding.get(PositionEncoding.ACTIONS, default=None):
            keys = self.timestep.keys(TokenType.ACTION)
            position = torch.arange(module.num_embeddings, device=device)
            position_embedding = module(position)
            position_embeddings.select(*keys).apply(
                lambda emb: emb + position_embedding,
                inplace=True,  # NOTE
            )

        if module := self.position_encoding.get(PositionEncoding.SPECIAL, default=None):
            keys = self.timestep.keys(TokenType.SPECIAL)
            position = torch.arange(module.num_embeddings, device=device)
            position_embedding = module(position)
            position_embeddings.select(*keys).apply(
                lambda emb: emb + position_embedding,
                inplace=True,  # NOTE
            )

        if module := self.position_encoding.get(
            PositionEncoding.TIMESTEP, default=None
        ):
            position = torch.arange(module.num_embeddings, device=device)
            position_embedding = rearrange(module(position), "t d -> t 1 d")
            position_embeddings.apply(
                lambda emb: emb + position_embedding,
                inplace=True,  # NOTE
            )

        return embeddings.apply(add, position_embeddings)
