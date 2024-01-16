from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import accumulate, pairwise, starmap
from typing import Annotated, Tuple, no_type_check

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
    def max(self) -> int:  # noqa: A003
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

    def build_episode(self, inputs: TensorDict) -> Episode:
        if self.transforms:
            inputs = inputs.clone(recurse=True)
            for t, transforms in self.transforms.items():
                for n, transform in transforms.items():  # pyright: ignore
                    inputs.select((t, n)).apply(transform, inplace=True)

        tokens = TensorDict(
            {k: v.apply(self.tokenizers[k]) for k, v in inputs.items()},
            batch_size=inputs.batch_size,
            device=inputs.device,
        )

        embeddings = TensorDict(
            {k: v.apply(self.embeddings[k]) for k, v in tokens.items()},
            batch_size=tokens.batch_size,
            device=tokens.device,
        )

        index = self._build_index(embeddings)
        # Disable for hindsign control where we apply [-1, -1, -,1 .... -1] as masked token
        # embeddings = self._apply_position_encoding(
        #     embeddings=embeddings,
        #     step_index=index[0],  # pyright: ignore
        # )

        return Episode(  # pyright: ignore
            embeddings=embeddings,
            labels=tokens,
            index=index,
            timestep=self.timestep,
            batch_size=[],
            device=inputs.device,
        )

    def _build_index(self, embeddings: TensorDict) -> EpisodeIndex:
        step_index_counts = [
            embeddings.get_item_shape(k)[2] for k in self.timestep.keys
        ]
        step_index_ranges = pairwise(accumulate(step_index_counts, initial=0))
        step_index = TensorDict.from_dict(
            dict(zip(self.timestep.keys, starmap(torch.arange, step_index_ranges))),
            batch_size=[],
            device=embeddings.device,
        )
        step_len = sum(step_index_counts)
        step_count = mit.one(
            {embeddings.get_item_shape(k)[1] for k in self.timestep.keys}
        )

        return EpisodeIndex.from_tensordict(  # pyright: ignore
            step_index.apply(
                lambda idx: torch.stack(
                    [idx + i * step_len for i in range(step_count)]
                ),
                batch_size=[step_count],
            )
        )

    def _apply_position_encoding(
        self,
        *,
        embeddings: TensorDict,
        step_index: EpisodeIndex,
    ) -> TensorDict:
        if module := getattr(
            self.position_encoding, PositionEncoding.OBSERVATIONS, None
        ):
            observation_embeddings = embeddings.select(*self.timestep.observations)
            observation_step_index = step_index.select(
                *self.timestep.observations
            )  # pyright: ignore
            embeddings = embeddings.update(
                observation_embeddings.apply(
                    lambda emb, pos: emb + module(pos),  # pyright: ignore
                    observation_step_index._tensordict,
                )
            )

        if module := getattr(self.position_encoding, PositionEncoding.ACTIONS, None):
            pos = torch.arange(module.num_embeddings, device=embeddings.device)
            pos_encd = module(pos)
            action_embeddings = embeddings.select(*self.timestep.actions)
            embeddings = embeddings.update(
                action_embeddings.apply(lambda emb: emb + pos_encd)
            )

        if module := getattr(self.position_encoding, PositionEncoding.TIMESTEP, None):
            pos = torch.arange(module.num_embeddings, device=embeddings.device)
            pos_encd = rearrange(module(pos), "t d -> t 1 d")
            embeddings = embeddings.update(embeddings.apply(lambda emb: emb + pos_encd))

        return embeddings
