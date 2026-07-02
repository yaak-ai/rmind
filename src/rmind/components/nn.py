from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, final, override

import torch
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import Module
from torch.utils._pytree import (  # noqa: PLC2701
    MappingKey,
    PyTree,
    tree_map,
    tree_map_with_path,
)

from rmind.utils.functional import diff_last
from rmind.utils.pytree import key_get_default

from .base import Invertible

default_weight_init_fn = partial(
    nn.init.trunc_normal_, mean=0.0, std=0.02, a=-0.04, b=0.04
)
default_linear_weight_init_fn = nn.init.xavier_uniform_
default_linear_bias_init_fn = partial(nn.init.constant_, val=0.0)


@final
class Embedding(nn.Embedding):
    def __init__(
        self,
        *args: Any,
        weight_init_fn: Callable[[Tensor], None] = default_weight_init_fn,  # ty:ignore[invalid-parameter-default]
        **kwargs: Any,
    ) -> None:
        self.weight_init_fn: Callable[[Tensor], None] = weight_init_fn

        super().__init__(*args, **kwargs)

    @override
    def reset_parameters(self) -> None:
        self.weight_init_fn(self.weight)
        self._fill_padding_idx_with_zero()


@final
class Linear(nn.Linear):
    def __init__(
        self,
        *args: Any,
        weight_init_fn: Callable[[Tensor], None] = default_linear_weight_init_fn,  # ty:ignore[invalid-parameter-default]
        bias_init_fn: Callable[[Tensor], None] = default_linear_bias_init_fn,  # ty:ignore[invalid-parameter-default]
        **kwargs: Any,
    ) -> None:
        self.weight_init_fn: Callable[[Tensor], None] = weight_init_fn
        self.bias_init_fn: Callable[[Tensor], None] = bias_init_fn

        super().__init__(*args, **kwargs)

    @override
    def reset_parameters(self) -> None:
        self.weight_init_fn(self.weight)
        if self.bias is not None:
            self.bias_init_fn(self.bias)


class Sequential(nn.Sequential, Invertible):
    @override
    def invert(self, input: Tensor) -> Tensor:
        for module in reversed(self):
            input = module.invert(input)
        return input


class Identity(nn.Identity, Invertible):
    @override
    def invert(self, input: Tensor) -> Tensor:
        return input


type Paths = Mapping[str, tuple[str, ...] | Paths]


@final
class Remapper(Module):
    @validate_call
    def __init__(self, paths: Paths) -> None:
        super().__init__()

        self._paths = tree_map(
            lambda path: tuple(map(MappingKey, path)),
            paths,
            is_leaf=lambda x: isinstance(x, tuple),
        )

    @property
    def paths(self) -> PyTree:
        return self._paths

    @override
    def extra_repr(self) -> str:
        return str(
            tree_map(
                lambda path: tuple(x.key for x in path),
                self._paths,
                is_leaf=lambda x: isinstance(x, tuple),
            )
        )

    @override
    def forward(self, input: PyTree) -> PyTree:
        return tree_map(
            lambda path: key_get_default(input, path, None),
            self._paths,
            is_leaf=lambda x: isinstance(x, tuple),
        )


@final
class Frozen(Module):
    """Wrap a module so it never trains: params frozen and kept in eval mode."""

    @validate_call
    def __init__(self, *, module: InstanceOf[Module]) -> None:
        super().__init__()

        self.module = module.requires_grad_(False).eval()  # noqa: FBT003

    @override
    def train(self, mode: bool = True) -> "Frozen":
        super().train(mode)
        self.module.eval()
        return self

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


@final
class StackFields(Module):
    """Gather ordered `paths` and stack them on a trailing axis under `out_key`.

    Each field is `(..., chunk)` (chunk == 1 for the immediate action), so the
    result is `(..., chunk, fields)` — e.g. `(B, T, 6, 4)` or `(B, T, 1, 4)`. The
    chunk/field axes are flattened into the action vector downstream by the
    `ActionTokenizer`. Emits `None` when a field is absent.
    """

    @validate_call
    def __init__(self, *, paths: Mapping[str, tuple[str, ...]], out_key: str) -> None:
        super().__init__()

        self._paths = {
            name: tuple(map(MappingKey, path)) for name, path in paths.items()
        }
        self.out_key = out_key

    @override
    def forward(self, input: PyTree) -> PyTree:
        fields = [key_get_default(input, path, None) for path in self._paths.values()]

        if any(value is None for value in fields):
            return {**input, self.out_key: None}

        stacked = torch.stack(fields, dim=-1)
        return {**input, self.out_key: stacked}


@final
class SliceFields(Module):
    """Narrow each path in `paths` to a length-1 slice along `dim` (keeps the axis).

    Leaves all other fields untouched. Used to take the immediate action
    `chunk[..., 0:1]` for the per-timestep tokens while `joint_actions` keeps the
    full action chunk.
    """

    @validate_call
    def __init__(
        self, *, paths: list[tuple[str, ...]], dim: int = -1, index: int = 0
    ) -> None:
        super().__init__()

        self._paths = {tuple(path) for path in paths}
        self.dim = dim
        self.index = index

    @override
    def forward(self, input: PyTree) -> PyTree:
        def fn(key_path: Any, value: Any) -> Any:
            names = tuple(entry.key for entry in key_path)
            if names in self._paths and value is not None:
                return value.narrow(self.dim, self.index, 1)
            return value

        return tree_map_with_path(fn, input)


@final
class ChunkFields(Module):
    """Build per-timestep action chunks from a flat time axis.

    Inputs span a fixed flat window over the time axis (`dim`) sized for the
    largest horizon, so the build is shared across configs. Each path in
    `unfold_paths` is unfolded into a sliding window of length `action_horizon`
    (step 1) and then truncated to the first `episode_length` windows, yielding
    `(..., episode_length, action_horizon)` — the action chunk starting at each of
    the `episode_length` timesteps. Every other field is narrowed to the first
    `episode_length` steps, dropping the tail kept only to form the chunks.

    For `action_horizon == 1` this yields `(..., episode_length, 1)`, i.e. the
    immediate action per timestep.
    """

    @validate_call
    def __init__(
        self,
        *,
        episode_length: int,
        action_horizon: int,
        unfold_paths: list[tuple[str, ...]],
        dim: int = 1,
    ) -> None:
        super().__init__()

        self.episode_length = episode_length
        self.action_horizon = action_horizon
        self._unfold_paths = {tuple(path) for path in unfold_paths}
        self.dim = dim

    @override
    def forward(self, input: PyTree) -> PyTree:
        def fn(key_path: Any, value: Any) -> Any:
            if value is None:
                return value

            names = tuple(entry.key for entry in key_path)
            if names in self._unfold_paths:
                return value.unfold(self.dim, self.action_horizon, 1).narrow(
                    self.dim, 0, self.episode_length
                )

            return value.narrow(self.dim, 0, self.episode_length)

        return tree_map_with_path(fn, input)


def _module_wrapper(
    fn: Callable[..., Tensor], *, name: str | None = None
) -> type[nn.Module]:
    @final
    class _Fn(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

            self._kwargs: Any = kwargs

        @override
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return fn(*args, **(self._kwargs | kwargs))

    if name is not None:
        _Fn.__name__ = name

    return _Fn


AtLeast3D = _module_wrapper(torch.atleast_3d, name="AtLeast3D")
DiffLast = _module_wrapper(diff_last, name="DiffLast")
Squeeze = _module_wrapper(torch.squeeze, name="Squeeze")


@final
class OnnxOutputUnpacker(Module):
    """Unpack ONNX model output (joint_actions) into individual action components."""

    def __init__(self, **_kwargs: Any) -> None:
        super().__init__()

    @override
    def forward(self, input: dict[str, Tensor]) -> dict:
        joint_actions = input["policy.joint_actions"]
        return {
            "policy": {
                "prediction_value": {
                    "value": TensorDict({
                        "continuous": TensorDict({
                            "gas_pedal": joint_actions[..., 0],
                            "brake_pedal": joint_actions[..., 1],
                            "steering_angle": joint_actions[..., 2],
                        }),
                        "discrete": TensorDict({
                            "turn_signal": torch.bucketize(
                                joint_actions[..., 3] * 2,
                                torch.tensor([0.5, 1.5], device=joint_actions.device),
                            )
                        }),
                    })
                }
            }
        }
