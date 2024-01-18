import operator
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import accumulate, pairwise
from typing import Annotated, Dict, List, Tuple, no_type_check

import more_itertools as mit
import torch
from beartype.vale import Is
from einops import pack, rearrange
from jaxtyping import Float, Int, Shaped
from tensordict import TensorDict, tensorclass
from tensordict.utils import NestedKey
from torch import Tensor
from torch.nn import Module, ModuleDict


class Token(str, Enum):
    IMAGE = "image"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


TokenModuleDict = Annotated[ModuleDict, Is[lambda d: d.keys() <= set(Token)]]


class PositionEncoding(str, Enum):
    OBSERVATIONS = "observations"
    ACTIONS = "actions"
    TIMESTEP = "timestep"


PositionEncodingModuleDict = Annotated[
    ModuleDict, Is[lambda d: d.keys() <= set(PositionEncoding)]
]


@dataclass(frozen=True)
class Timestep:
    observations: Tuple[Tuple[Token, str], ...]
    actions: Tuple[Tuple[Token, str], ...]

    @classmethod
    def build(
        cls,
        observations: Iterable[Tuple[str, str]],
        actions: Iterable[Tuple[str, str]],
    ):
        return cls(
            observations=tuple((Token(a), b) for (a, b) in observations),
            actions=tuple((Token(a), b) for (a, b) in actions),
        )

    @property
    def keys(self):
        return self.observations + self.actions


@no_type_check
@tensorclass  # pyright: ignore
class Index:
    image: TensorDict
    continuous: TensorDict
    discrete: TensorDict

    def parse(self, src: Shaped[Tensor, "b s ..."]) -> TensorDict:
        b, *_ = src.shape

        return self.apply(  # pyright: ignore
            lambda idx: src[:, idx],
            batch_size=[b],
            device=src.device,
            inplace=False,
        ).to_tensordict()

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

    def __hash__(self) -> int:
        items = tuple(
            (k, tuple(v.flatten().tolist()))
            for k, v in sorted(
                self.items(  # pyright: ignore
                    include_nested=True,
                    leaves_only=True,
                ),
                key=lambda x: x[0],
            )
        )

        return hash(items)


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
        tokenizers: TokenModuleDict,
        embeddings: TokenModuleDict,
        position_encoding: PositionEncodingModuleDict,
    ) -> None:
        super().__init__()

        self.timestep = timestep
        self.transforms = transforms
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
        timestep_count = mit.one({
            embeddings.get_item_shape(k)[1] for k in self.timestep.keys
        })
        index = timestep_index.apply(
            lambda x: torch.stack([
                x + i * timestep_length for i in range(timestep_count)
            ]),
            batch_size=[timestep_count],
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
        transformed = inputs.clone(recurse=True)  # TODO: avoid cloning if noop?
        for t, transforms in self.transforms.items():
            for n, transform in transforms.items():  # pyright: ignore
                transformed[(t, n)] = transform(inputs[t, n])

        return transformed

    def _tokenize(self, inputs: TensorDict) -> TensorDict:
        return TensorDict(
            {k: v.apply(self.tokenizers[k]) for k, v in inputs.items()},
            batch_size=inputs.batch_size,
            device=inputs.device,
        )

    def _embed(self, tokens: TensorDict) -> TensorDict:
        return TensorDict(
            {k: v.apply(self.embeddings[k]) for k, v in tokens.items()},
            batch_size=tokens.batch_size,
            device=tokens.device,
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
