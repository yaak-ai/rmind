from collections.abc import Callable, Mapping
from itertools import accumulate
from typing import Any, final, override

import more_itertools as mit
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from structlog import get_logger
from torch import Tensor
from torch.nn import Identity, Module, ModuleDict, Sequential
from torch.utils._pytree import (
    KeyEntry,
    MappingKey,
    PyTree,
    tree_leaves,
    tree_map,
    tree_map_with_path,
)
from torchvision.transforms.v2 import CenterCrop, Normalize, ToDtype

from rmind.components.episode import Modality, PositionEncoding, TokenType

from .control_transformer import ControlTransformer

logger = get_logger(__name__)


def _fn_to_module(fn: Callable[..., Tensor], **static_kwargs: Any) -> type[Module]:
    class _Fn(Module):
        @override
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return fn(*args, **(static_kwargs | kwargs))

    _Fn.__name__ = fn.__name__.capitalize() + "Layer"

    return _Fn


@final
class ControlTransformerExportable(Module):
    def __init__(self, module: ControlTransformer) -> None:
        super().__init__()

        self._encoder = module.encoder
        self._mask = torch.load("./data/mask.pt")

        atleast_3d = _fn_to_module(torch.atleast_3d)

        self._input_transforms = ModuleDict({
            "image": Sequential(
                Rearrange("... h w c -> ... c h w"),
                CenterCrop([320, 576]),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ),
            "continuous": atleast_3d(),
            "discrete": atleast_3d(),
            "context": Identity(),
        })

        self._module = module

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        (b, t), device = mit.one({
            (tensor.shape[:2], tensor.device) for tensor in tree_leaves(inputs)
        })

        inputs_transformed = tree_map_with_path(
            lambda path, x: get_deepest(path, self._input_transforms)(x), inputs
        )

        input_tokens = tree_map_with_path(
            lambda path, x: (
                get_deepest(path, self._module.episode_builder.tokenizers)
            )(x),
            inputs_transformed,
        )

        input_tokens["special"] = tree_map(
            lambda x: x.expand(b, t, 1).to(device),
            {
                "observation_summary": torch.tensor(0),
                "observation_history": torch.tensor(1),
                "action_summary": torch.tensor(2),
            },
        )

        input_embeddings = tree_map_with_path(
            lambda path, x: (
                get_deepest(path, self._module.episode_builder.embeddings)
            )(x),
            input_tokens,
        )

        # index
        timestep = self._module.episode_builder.timestep
        lengths = [
            input_embeddings[(k := token.key)[0]][k[1]].shape[2]
            for token in timestep.root
        ]
        boundaries = tuple(accumulate(lengths, initial=0))
        timestep_length = boundaries[-1]
        ranges = tuple(zip(boundaries[:-1], boundaries[1:], strict=True))  # noqa: RUF007
        timestep_index = unflatten_keys({
            token.key: torch.arange(*_range)
            for token, _range in zip(timestep.root, ranges, strict=True)
        })

        index = tree_map(
            lambda x: torch.stack([x + i * timestep_length for i in range(t)]),
            timestep_index,
        )

        # position_embeddings
        position_encoding = self._module.episode_builder.position_encoding
        position_embeddings_accum = {}

        if (
            mod_pe := position_encoding.get(
                (k_pe := PositionEncoding.IMAGE.value, "patch"), default=None
            )
        ) is not None:
            paths = tree_map(
                MappingKey, timestep.keys_by_modality[Modality.IMAGE.value]
            )

            num_rows = mod_pe.row.num_embeddings
            num_cols = mod_pe.col.num_embeddings
            row_pe = mod_pe.row(torch.arange(num_rows, device=device))
            col_pe = mod_pe.col(torch.arange(num_cols, device=device))
            row_pe = repeat(row_pe, "h d -> (h w) d", w=num_cols)
            col_pe = repeat(col_pe, "w d -> (h w) d", h=num_rows)
            position_embedding = row_pe + col_pe

            position_embeddings_accum[k_pe] = tree_map_with_path(
                lambda path, _: position_embedding if path in paths else None,
                input_embeddings,
            )

        if (
            mod_pe := position_encoding.get(
                k_pe := PositionEncoding.OBSERVATIONS.value, default=None
            )
        ) is not None:
            paths = tree_map(
                MappingKey, timestep.keys_by_type[TokenType.OBSERVATION.value]
            )

            position_embeddings_accum[k_pe] = tree_map_with_path(
                lambda path, x: mod_pe(x) if path in paths else None, timestep_index
            )

        if (
            mod_pe := position_encoding.get(
                k_pe := PositionEncoding.ACTIONS.value, default=None
            )
        ) is not None:
            paths = tree_map(MappingKey, timestep.keys_by_type[TokenType.ACTION.value])
            position = torch.arange(mod_pe.num_embeddings, device=device)
            position_embedding = mod_pe(position)
            position_embeddings_accum[k_pe] = tree_map_with_path(
                lambda path, _: position_embedding if path in paths else None,
                input_embeddings,
            )

        if (
            mod_pe := position_encoding.get(
                k_pe := PositionEncoding.SPECIAL.value, default=None
            )
        ) is not None:
            paths = tree_map(MappingKey, timestep.keys_by_type[TokenType.SPECIAL.value])
            position = torch.arange(mod_pe.num_embeddings, device=device)
            position_embedding = mod_pe(position)
            position_embeddings_accum[k_pe] = tree_map_with_path(
                lambda path, _: position_embedding if path in paths else None,
                input_embeddings,
            )

        if (
            mod_pe := position_encoding.get(
                k_pe := PositionEncoding.TIMESTEP.value, default=None
            )
        ) is not None:
            # build a sequence starting from a random index (simplified [0])
            # e.g. given num_embeddings=20 and t=6, sample from ([0, 5], [1, 6], ..., [14, 19])
            # ---
            # [0] Randomized Positional Encodings Boost Length Generalization of Transformers (https://arxiv.org/abs/2305.16843)

            low, high = 0, mod_pe.num_embeddings - t + 1
            # TODO: rand vs inference?
            start = torch.randint(low, high, (1,)).item()
            position = torch.arange(start=start, end=start + t, device=device)
            position_embedding = rearrange(mod_pe(position), "t d -> t 1 d")
            position_embeddings_accum[k_pe] = tree_map(
                lambda _: position_embedding, input_embeddings
            )

        # return tree_map(
        #     lambda *xs: sum(x for x in xs if x is not None),
        #     *position_embeddings_accum.values(),
        # )

        return index


def get_deepest(path: tuple[KeyEntry, ...], modules: ModuleDict) -> Module:
    # TODO: for whatever reason a recursive version is not exportable
    if len(path) == 1:
        return path[0].get(modules)

    if len(path) == 2:
        module = path[0].get(modules)
        if isinstance(module, ModuleDict):
            module = path[1].get(module)

        return module

    raise RuntimeError


def unflatten_keys(data: Mapping[tuple[str, ...], Any]) -> PyTree:
    out = {}
    for k, v in data.items():
        last = out
        for k_elem in k[:-1]:
            if k_elem not in out:
                out[k_elem] = {}

            last = out[k_elem]

        last[k[-1]] = v

    return out
