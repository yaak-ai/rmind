from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, final, override

import torch
from pydantic import validate_call
from torch import Tensor, nn
from torch.nn import Module
from torch.utils._pytree import MappingKey, PyTree, tree_map  # noqa: PLC2701

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
    def forward(self, input: PyTree) -> PyTree:
        return tree_map(
            lambda path: key_get_default(input, path, None),
            self._paths,
            is_leaf=lambda x: isinstance(x, tuple),
        )


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


@final
class GRUTrajectoryHead(Module):
    """Autoregressive GRU that predicts future trajectory waypoints step-by-step.

    At each step the previously predicted means are fed back as input. Returns
    both the per-step logits and the GRU hidden states so the companion action
    heads can be conditioned on the trajectory's internal representations.

    Output logits layout per step: [mean_x, logvar_x, mean_y, logvar_y] (4
    values), or with `predict_yaw=True`: [mean_x, logvar_x, mean_y, logvar_y,
    mean_yaw, logvar_yaw] (6 values). Either way this matches GaussianNLLLoss,
    which expects input[..., 0]=mean, input[..., 1]=logvar per pair.
    """

    @validate_call
    def __init__(
        self,
        *,
        in_features: int,
        hidden_size: int,
        num_steps: int,
        predict_yaw: bool = False,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.predict_yaw = predict_yaw
        self._hidden_size = hidden_size
        pose_dim = 3 if predict_yaw else 2  # (x, y[, yaw])
        self.hidden_proj = Linear(in_features, hidden_size)
        self.input_proj = Linear(pose_dim, hidden_size)  # embed prev pose means
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_proj = Linear(hidden_size, 2 * pose_dim)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (b, 1, in_features)
        b = x.size(0)
        pose_dim = 3 if self.predict_yaw else 2
        pose_idx = [0, 2, 4] if self.predict_yaw else [0, 2]
        h = self.hidden_proj(x[:, 0]).unsqueeze(0)        # (1, b, H)
        prev_pose = torch.zeros(b, pose_dim, device=x.device, dtype=x.dtype)
        preds, hs = [], []
        for _ in range(self.num_steps):
            inp = self.input_proj(prev_pose).unsqueeze(1)  # (b, 1, H)
            out, h = self.gru(inp, h)                      # (b, 1, H), (1, b, H)
            hs.append(h[0])                                # (b, H)
            logits = self.output_proj(out[:, 0])           # (b, 2 * pose_dim)
            prev_pose = logits[:, pose_idx]                # (b, pose_dim) means only
            preds.append(logits)
        return torch.stack(preds, dim=1), torch.stack(hs, dim=1)
        # (b, num_steps, 2 * pose_dim),  (b, num_steps, H)


@final
class GRUHead(Module):
    """GRU decoder that generates `num_steps` action predictions.

    Context features are projected to the GRU initial hidden state; per-step
    positional embeddings serve as inputs so each decoding step can specialize.
    When `h_traj` is provided (from a companion GRUTrajectoryHead), it is
    projected and added to the step embeddings before each GRU step.
    """

    @validate_call
    def __init__(
        self,
        *,
        in_features: int,
        hidden_size: int,
        out_features: int,
        num_steps: int,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.out_features = out_features
        self._hidden_size = hidden_size
        self.hidden_proj = Linear(in_features, hidden_size)
        self.step_embed = nn.Embedding(num_steps, hidden_size)
        self.traj_proj = Linear(hidden_size, hidden_size)  # trajectory conditioning
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_proj = Linear(hidden_size, out_features)

    @override
    def forward(self, x: Tensor, h_traj: Tensor | None = None) -> Tensor:
        # x: (b, 1, in_features);  h_traj: (b, num_steps, hidden_size) or None
        b = x.size(0)
        h0 = self.hidden_proj(x[:, 0]).unsqueeze(0)                          # (1, b, H)
        step_ids = torch.arange(self.num_steps, device=x.device)
        inp = self.step_embed(step_ids).unsqueeze(0).expand(b, -1, -1)       # (b, steps, H)
        if h_traj is not None:
            inp = inp + self.traj_proj(h_traj)                                # (b, steps, H)
        out, _ = self.gru(inp, h0)                                            # (b, steps, H)
        return self.output_proj(out)                                           # (b, steps, out_features)


@final
class MLPHead(Module):
    """Pointwise per-step MLP decoder.

    Unlike `GRUHead`, which generates its own per-step variation from a single
    context vector, this expects an input that already varies per predicted
    step (e.g. an inverse-dynamics decode conditioned on a companion head's
    own multi-step forecast) and applies the same small MLP at every step.
    """

    @validate_call
    def __init__(self, *, in_features: int, hidden_size: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Linear(in_features, hidden_size),
            nn.GELU(),
            Linear(hidden_size, out_features),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (b, steps, in_features) -> (b, steps, out_features)
        return self.net(x)
