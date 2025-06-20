from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto, unique
from functools import cached_property
from itertools import accumulate, starmap
from operator import add, attrgetter, itemgetter
from typing import ClassVar, override

import more_itertools as mit
import torch
from einops import pack, rearrange, repeat
from pydantic import ConfigDict, InstanceOf, RootModel, model_validator, validate_call
from structlog import get_logger
from tensordict import TensorClass, TensorDict
from torch import Tensor, nn

from rmind.utils import ModuleDict

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


@dataclass(frozen=True)
class TokenMeta:
    type: TokenType
    modality: Modality
    name: str

    @property
    def key(self) -> tuple[Modality, str]:
        return (self.modality, self.name)


class Timestep(RootModel[tuple[TokenMeta, ...]]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, ignored_types=(cached_property,)
    )

    @override
    def __iter__(self) -> Iterator[TokenMeta]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return iter(self.root)

    def __getitem__(self, item: int) -> TokenMeta:
        return self.root[item]

    @override
    def __hash__(self) -> int:
        return hash(self.root)

    @model_validator(mode="before")
    @classmethod
    def validate_tokens(cls, v: Sequence[Sequence[str]]) -> tuple[TokenMeta, ...]:
        return tuple(starmap(TokenMeta, v))

    @cached_property
    def keys_by_type(self) -> Mapping[TokenType, Sequence[tuple[str, str]]]:
        return mit.map_reduce(
            self, keyfunc=attrgetter("type"), valuefunc=attrgetter("key")
        )

    @cached_property
    def keys_by_modality(self) -> Mapping[TokenType, Sequence[tuple[str, str]]]:
        return mit.map_reduce(
            self, keyfunc=attrgetter("modality"), valuefunc=attrgetter("key")
        )


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


def parse_index(index: TensorDict, src: Tensor, dim: int = 1) -> TensorDict:
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

    batch_size = [*src.shape[:dim], *index.batch_size]

    return index.apply(fn, batch_size=batch_size, device=src.device, inplace=False)


@dataclass(frozen=True)
class Episode:
    input: TensorDict
    input_tokens: TensorDict
    input_embeddings: TensorDict
    position_embeddings: TensorDict
    index: TensorDict
    timestep: Timestep

    @property
    def embeddings(self) -> TensorDict:
        return self.input_embeddings + self.position_embeddings

    @property
    def embeddings_packed(self) -> Tensor:
        embeddings = self.embeddings
        packed, _ = pack([embeddings[token.key] for token in self.timestep], "b t * d")

        return rearrange(packed, "b t s d -> b (t s) d")


torch.export.register_dataclass(Episode)


class EpisodeBuilder(nn.Module):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        special_tokens: Mapping[SpecialToken, int],
        timestep: Timestep,
        tokenizers: InstanceOf[ModuleDict],
        embeddings: InstanceOf[ModuleDict],
        position_encoding: InstanceOf[ModuleDict],
        freeze: bool | None = None,
    ) -> None:
        super().__init__()

        self.special_tokens: Mapping[SpecialToken, int] = special_tokens
        self.timestep: Timestep = timestep
        self.tokenizers: ModuleDict = tokenizers
        self.embeddings: ModuleDict = embeddings
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

            self.requires_grad_(not freeze).train(not freeze)  # pyright: ignore[reportUnusedCallResult]

    @override
    def forward(self, input: TensorDict) -> Episode:
        batch_size, device = input.batch_size, input.device

        input_tokens = self.tokenizers.forward(
            input, batch_size=batch_size, device=device
        )
        input_tokens._tensordict[Modality.SPECIAL] = (
            TensorDict.from_dict(self.special_tokens, device=device)
            .expand(*batch_size, 1)
            .auto_batch_size_(input.batch_dims)
        )

        input_embeddings = self.embeddings.forward(
            input_tokens.select(*self.embeddings.tree_paths(), strict=False),
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
        )

    def _build_index(self, embeddings: TensorDict) -> TensorDict:
        _, t = embeddings.batch_size

        # lengths = [embeddings[token.key].shape[2] for token in self.timestep]
        lengths = [180, 1, 1, 10, 1, 1, 1, 1, 1, 1]
        boundaries = list(accumulate(lengths, initial=0))
        timestep_length = boundaries[-1]
        ranges = zip(boundaries[:-1], boundaries[1:], strict=True)  # noqa: RUF007

        timestep_index = TensorDict(
            {
                token.key: torch.arange(*_range)
                for token, _range in zip(self.timestep.root, ranges, strict=True)
            }
            # device=embeddings.device,
        )

        return torch.stack([  # pyright: ignore[reportReturnType]
            timestep_index + i * timestep_length for i in range(t)
        ])

    def _build_position_embeddings(
        self, src: TensorDict, timestep_index: Index
    ) -> TensorDict:
        (_, t), device = src.batch_size, src.device

        position_embeddings: TensorDict = src.new_zeros(t)  # pyright: ignore[reportAssignmentType, reportArgumentType]

        if (
            pe_mod := self.position_encoding.get(
                (pe_k := PositionEncoding.IMAGE, "patch"), default=None
            )
        ) is not None:
            num_rows = pe_mod.row.num_embeddings
            num_cols = pe_mod.col.num_embeddings
            row_pe = pe_mod.row(torch.arange(num_rows, device=device))
            col_pe = pe_mod.col(torch.arange(num_cols, device=device))
            row_pe = repeat(row_pe, "h d -> (h w) d", w=num_cols)
            col_pe = repeat(col_pe, "w d -> (h w) d", h=num_rows)

            position_embeddings.select(pe_k, strict=False).apply(
                lambda x: x + row_pe + col_pe,
                inplace=True,  # NOTE
            )

        if (
            pe_mod := self.position_encoding.get(
                pe_k := PositionEncoding.OBSERVATIONS, default=None
            )
        ) is not None:
            keys = self.timestep.keys_by_type[TokenType.OBSERVATION]
            position = timestep_index.select(*keys).to_tensordict(retain_none=False)
            position_embedding = position.apply(pe_mod)
            position_embeddings.select(*keys, strict=False).apply(
                add,
                position_embedding,
                inplace=True,  # NOTE
            )

        if (
            pe_mod := self.position_encoding.get(
                pe_k := PositionEncoding.ACTIONS, default=None
            )
        ) is not None:
            keys = self.timestep.keys_by_type[TokenType.ACTION]
            position = torch.arange(pe_mod.num_embeddings, device=device)
            position_embedding = pe_mod(position)
            position_embeddings.select(*keys, strict=False).apply(
                lambda emb: emb + position_embedding,
                inplace=True,  # NOTE
            )

        if (
            pe_mod := self.position_encoding.get(
                pe_k := PositionEncoding.SPECIAL, default=None
            )
        ) is not None:
            keys = self.timestep.keys_by_type[TokenType.SPECIAL]
            position = torch.arange(pe_mod.num_embeddings, device=device)
            position_embedding = pe_mod(position)
            position_embeddings.select(*keys, strict=False).apply(
                lambda emb: emb + position_embedding,
                inplace=True,  # NOTE
            )

        if (
            pe_mod := self.position_encoding.get(
                pe_k := PositionEncoding.TIMESTEP, default=None
            )
        ) is not None:
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

        return position_embeddings
