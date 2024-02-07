import operator
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum, auto
from itertools import accumulate, pairwise
from typing import Annotated, Dict, List, Mapping, Tuple

import torch
from beartype.vale import Is
from einops import pack, rearrange
from jaxtyping import Float, Int, Shaped
from tensordict import TensorDict, tensorclass
from tensordict.utils import NestedKey
from torch import Tensor
from torch.nn import Module, ModuleDict


class TokenType(StrEnum):
    IMAGE = auto()
    CONTINUOUS = auto()
    DISCRETE = auto()
    SPECIAL = auto()


TokenModuleDict = Annotated[ModuleDict, Is[lambda d: d.keys() <= set(TokenType)]]


class PositionEncoding(StrEnum):
    OBSERVATIONS = auto()
    ACTIONS = auto()
    TIMESTEP = auto()


PositionEncodingModuleDict = Annotated[
    ModuleDict, Is[lambda d: d.keys() <= set(PositionEncoding)]
]


@dataclass(frozen=True)
class Timestep:
    observations: Tuple[Tuple[TokenType, str], ...]
    actions: Tuple[Tuple[TokenType, str], ...]

    @classmethod
    def build(
        cls,
        observations: Iterable[Tuple[str, str]],
        actions: Iterable[Tuple[str, str]],
    ):
        return cls(
            observations=tuple((TokenType(a), b) for (a, b) in observations),
            actions=tuple((TokenType(a), b) for (a, b) in actions),
        )

    @property
    def keys(self):
        return self.observations + self.actions


@tensorclass  # pyright: ignore
class Index:
    image: TensorDict
    continuous: TensorDict
    discrete: TensorDict
    special: TensorDict

    def parse(self, src: Shaped[Tensor, "b s ..."]) -> TensorDict:
        b, *_ = src.shape

        return self.to_tensordict().apply(  # pyright: ignore
            lambda idx: src[:, idx],
            batch_size=[b],
            device=src.device,
            inplace=False,
        )

    @property
    def all_values(self) -> Int[Tensor, "d"]:
        return torch.cat(
            list(
                self.values(  # pyright: ignore
                    include_nested=True,
                    leaves_only=True,
                )
            ),
            -1,
        ).flatten()

    @property
    def max(self) -> int:
        return max(
            self.apply(torch.max, batch_size=[]).values(  # pyright: ignore
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
    return _eq(self, other).all()  # pyright: ignore


Index.__hash__ = _index_hash
Index.__eq__ = _index_eq  # pyright: ignore


@tensorclass  # pyright: ignore
class Episode:
    embeddings: TensorDict
    labels: TensorDict
    index: Index
    timestep: Timestep

    @property  # TODO: cache?
    def packed_embeddings(self) -> Float[Tensor, "b s d"]:
        embeddings, _ = pack(
            [self.embeddings[k] for k in self.timestep.keys],
            "b t * d",
        )
        return rearrange(embeddings, "b t s d -> b (t s) d")


class EpisodeBuilder(Module):
    def __init__(
        self,
        *,
        timestep: Timestep,
        transforms: TokenModuleDict,
        special_tokens: Mapping[str, int],
        tokenizers: TokenModuleDict,
        embeddings: TokenModuleDict,
        position_encoding: PositionEncodingModuleDict,
    ) -> None:
        super().__init__()
        self.timestep = timestep
        self.transforms = transforms
        self.special_tokens = special_tokens
        self.tokenizers = tokenizers
        self.embeddings = embeddings
        self.position_encoding = position_encoding

    def build_episode(
        self,
        inputs: TensorDict,
        *,
        # TODO: make this less jarring
        masked_action_timestep_idx: List[int] | None = None,
        masked_observation_timestep_idx: List[int] | None = None,
    ) -> Episode:
        transformed = self._transform(inputs)
        tokens = self._tokenize(transformed)
        embeddings = self._embed(tokens)

        # TODO: learnable mask token?
        if masked_action_timestep_idx is not None:
            embeddings.select(*self.timestep.actions)[
                :, masked_action_timestep_idx
            ] = -1.0

        if masked_observation_timestep_idx is not None:
            embeddings.select(*self.timestep.observations)[
                :, masked_observation_timestep_idx
            ] = -1.0

        lengths = {k: embeddings.get_item_shape(k)[2] for k in self.timestep.keys}
        timestep_index = self._build_timestep_index(lengths).to(embeddings.device)  # pyright: ignore
        timestep_length = sum(lengths.values())
        _, t = embeddings.batch_size
        index = timestep_index.apply(
            lambda x: torch.stack([x + i * timestep_length for i in range(t)]),
            batch_size=[t],
        )

        embeddings = self._position_encode(embeddings, timestep_index)

        return Episode(  # pyright: ignore
            embeddings=embeddings,
            labels=tokens,
            index=index,
            timestep=self.timestep,
            batch_size=[],
            device=inputs.device,
        )

    def _transform(self, inputs: TensorDict) -> TensorDict:
        # TODO: named_apply instead?
        transformed = inputs.clone(recurse=True)  # TODO: avoid cloning if noop?
        for t, transforms in self.transforms.items():
            for n, transform in transforms.items():
                transformed[(t, n)] = transform(inputs[t, n])

        return transformed

    def _tokenize(self, inputs: TensorDict) -> TensorDict:
        tokens = inputs.named_apply(
            lambda nested_key, tensor: self.tokenizers[nested_key[0]](tensor),
            nested_keys=True,
        )

        tokens[TokenType.SPECIAL] = {
            k: torch.tensor(v).expand(*tokens.batch_size, 1)
            for k, v in self.special_tokens.items()
        }

        return tokens

    def _embed(self, tokens: TensorDict) -> TensorDict:
        return tokens.named_apply(
            lambda nested_key, tensor: self.embeddings[nested_key[0]](tensor),
            nested_keys=True,
        )

    def _build_timestep_index(self, lengths: Dict[NestedKey, int]) -> Index:
        ranges = dict(
            zip(
                lengths.keys(),
                pairwise(accumulate(lengths.values(), initial=0)),
            )
        )
        return Index.from_dict(  # pyright: ignore
            {k: torch.arange(*v) for k, v in ranges.items()},
            batch_size=[],
        )

    def _position_encode(
        self,
        embeddings: TensorDict,
        index: Index,
    ) -> TensorDict:
        embeddings = embeddings.clone(recurse=True)  # TODO: avoid cloning if noop?

        if module := getattr(
            self.position_encoding, PositionEncoding.OBSERVATIONS, None
        ):
            position = index.select(*self.timestep.observations).to_tensordict()  # pyright: ignore
            position_embedding = position.apply(module)
            embeddings.select(*self.timestep.observations).apply(
                operator.add,
                position_embedding,
                inplace=True,  # NOTE
            )

        if module := getattr(self.position_encoding, PositionEncoding.ACTIONS, None):
            position = torch.arange(module.num_embeddings, device=embeddings.device)
            position_embedding = module(position)
            embeddings.select(*self.timestep.actions).apply(
                lambda emb: emb + position_embedding,
                inplace=True,  # NOTE
            )

        if module := getattr(self.position_encoding, PositionEncoding.TIMESTEP, None):
            position = torch.arange(module.num_embeddings, device=embeddings.device)
            position_embedding = rearrange(module(position), "t d -> t 1 d")
            embeddings.apply(
                lambda emb: emb + position_embedding,
                inplace=True,  # NOTE
            )

        return embeddings
