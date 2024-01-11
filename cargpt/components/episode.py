import operator
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import accumulate, pairwise
from typing import Annotated, List, Tuple, no_type_check

import more_itertools as mit
import torch
from beartype.vale import Is
from einops import pack, rearrange
from jaxtyping import Float, Int, Shaped
from tensordict import TensorDict, tensorclass
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
class EpisodeIndex:
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
    index: EpisodeIndex
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

        embedding_shapes = embeddings.apply(lambda x: x.shape, batch_size=[])
        index = self._build_index(embedding_shapes)

        embeddings = self._position_encode(embeddings, index)

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

    def _build_index(self, shapes: TensorDict) -> EpisodeIndex:
        # NOTE: ordered by timestep keys
        step_shapes = {k: shapes[k].tolist() for k in self.timestep.keys}
        step_lengths = {k: s for k, (_, _, s, _) in step_shapes.items()}
        step_ranges = dict(
            zip(
                step_lengths.keys(),
                pairwise(accumulate(step_lengths.values(), initial=0)),
            )
        )
        step_index = EpisodeIndex.from_dict(  # pyright: ignore
            {k: torch.arange(*v) for k, v in step_ranges.items()},
            batch_size=[],
            device=shapes.device,
        )
        step_length = step_index.max + 1
        step_count = mit.one({t for (_, t, _, _) in step_shapes.values()})

        return step_index.apply(
            lambda x: torch.stack([x + i * step_length for i in range(step_count)]),
            batch_size=[step_count],
        )

    def _position_encode(
        self,
        embeddings: TensorDict,
        index: EpisodeIndex,
    ) -> TensorDict:
        embeddings = embeddings.clone(recurse=True)  # TODO: avoid cloning if noop?

        if module := getattr(
            self.position_encoding, PositionEncoding.OBSERVATIONS, None
        ):
            observation_index = index.select(*self.timestep.observations).apply(  # pyright: ignore
                # shift indices from global (episode) to local (timestep)
                lambda idx: idx - idx[:, [0]]
            )

            position_embedding = observation_index.to_tensordict().apply(module)
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
