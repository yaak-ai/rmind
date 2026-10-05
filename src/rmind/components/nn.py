import math
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, final, override

import torch
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
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


# --------------------------------------------------------------------------
# Action-chunk codec modules, ported verbatim from origin/feat/palletjack-3cam-
# retile (the D12 tokenizer playbook). Embodiment-agnostic: flat chunks are laid
# out (timestep, axis) with the axis fastest-varying.
# --------------------------------------------------------------------------

def _dct_ii_basis(num_steps: int, num_coefficients: int) -> Tensor:
    """First `num_coefficients` rows of the orthonormal DCT-II matrix, `(k, T)`.

    `D[k, t] = a_k cos(pi (2t + 1) k / 2T)`, with `a_0 = sqrt(1/T)` and
    `a_k = sqrt(2/T)`. The full matrix is orthogonal, so the inverse transform is
    the transpose and a truncated basis is a least-squares projection.
    """
    t = torch.arange(num_steps, dtype=torch.float64)
    k = torch.arange(num_coefficients, dtype=torch.float64).unsqueeze(1)
    basis = torch.cos(torch.pi * (2 * t + 1) * k / (2 * num_steps))
    basis *= torch.sqrt(torch.tensor(2.0 / num_steps, dtype=torch.float64))
    if num_coefficients > 0:
        basis[0] /= torch.sqrt(torch.tensor(2.0, dtype=torch.float64))

    return basis.float()


@final
class ChunkDCT(Module):
    """Flat time-domain action chunk -> low-frequency DCT-II coefficients.

    WHY. The tokenizer's encoder is an MLP over a chunk flattened to `T * A` numbers,
    with nothing telling it that element `i` and `i + A` are adjacent in TIME. Measured
    on the d12 holdout, that costs it: a fixed rank-32 linear map of the same chunks
    reaches EV +0.968 while the trained autoencoder reaches +0.92, and widening the MLP
    4x moved it +0.003. The gap is the basis, not the capacity.

    A DCT is the basis these signals want. 96% of the traction AC power and 99.6% of the
    steering sits below 1 Hz, so nearly all of a 5 s chunk lands in the first few
    coefficients: measured per axis, 32 coefficients reconstruct traction to +0.997,
    steering to +1.000 and fork1 to +0.987. It matches a PCA fit on the training drives
    to within 0.003 at every budget -- as expected, the DCT is asymptotically the KLT for
    smooth signals -- while fitting nothing and needing no artifact.

    Pairs with `ChunkIDCT`, which must be the LAST module of the decoder so the
    reconstruction loss is still taken in the time domain. Scoring in coefficient space
    would weight error per-coefficient rather than per-sample, and the fork1 event
    weighting in `ActionTokenizer._weighted_l1` would stop meaning anything.
    """

    @validate_call
    def __init__(self, *, num_steps: int, num_axes: int, num_coefficients: int) -> None:
        super().__init__()

        if not 0 < num_coefficients <= num_steps:
            msg = f"num_coefficients {num_coefficients} not in (0, {num_steps}]"
            raise ValueError(msg)

        self.num_steps = num_steps
        self.num_axes = num_axes
        self.num_coefficients = num_coefficients
        self.register_buffer("basis", _dct_ii_basis(num_steps, num_coefficients))

    @override
    def forward(self, x: Tensor) -> Tensor:
        # `_gather_actions` stacks axes on the last dim before flattening, so the flat
        # layout is (timestep, axis) with axis fastest-varying.
        chunk = x.reshape(*x.shape[:-1], self.num_steps, self.num_axes)

        return torch.einsum("kt,...ta->...ka", self.basis, chunk).reshape(
            *x.shape[:-1], self.num_coefficients * self.num_axes
        )


@final
class ChunkIDCT(Module):
    """DCT-II coefficients -> flat time-domain action chunk. Inverse of `ChunkDCT`.

    The DCT matrix is orthogonal, so this is its transpose; with `num_coefficients <
    num_steps` it is the least-squares reconstruction from the retained band.
    """

    @validate_call
    def __init__(self, *, num_steps: int, num_axes: int, num_coefficients: int) -> None:
        super().__init__()

        self.num_steps = num_steps
        self.num_axes = num_axes
        self.num_coefficients = num_coefficients
        self.register_buffer("basis", _dct_ii_basis(num_steps, num_coefficients))

    @override
    def forward(self, c: Tensor) -> Tensor:
        coeffs = c.reshape(*c.shape[:-1], self.num_coefficients, self.num_axes)

        return torch.einsum("kt,...ka->...ta", self.basis, coeffs).reshape(
            *c.shape[:-1], self.num_steps * self.num_axes
        )


@final
class _DilatedResidualUnit(Module):
    """SoundStream's residual unit: dilated conv, pointwise conv, skip.

    `Conv1d(k=3, dilation=d) . ELU . Conv1d(k=1)` added back to the input, exactly the
    unit in arXiv:2107.03312 Fig. 3 (there at dilations 1, 3, 9). Padding keeps the
    length so a stack of these leaves the time axis intact.
    """

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation),
            nn.ELU(),
            nn.Conv1d(channels, channels, 1),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


@final
class ChunkConvEncoder(Module):
    """Flat action chunk -> latent, through a dilated temporal conv over the STEP axis.

    WHY, given the DCT arms. `ChunkDCT` handed the MLP a better global basis and the
    lossless control (k=50, a pure rotation) moved nothing: the basis was not the
    constraint. That result says nothing about LOCAL temporal structure, which is a
    different claim -- a convolution shares one set of weights across every position, so
    it can represent an edge wherever it occurs, while an MLP over a flattened window
    has to learn each position separately and a fixed cosine basis has to spend many
    coefficients on any edge at all.

    That is why a codec built for sharp transients uses this and not an MLP
    (arXiv:2107.03312 §III-A: `Conv1d(k=7)` then residual units at dilations 1, 3, 9,
    ELU, no normalization). The receptive field here is +/-16 steps of the 50 -- 3 from
    the input conv, then 1 + 3 + 9 from the dilated units -- so a unit at any position
    sees ~3.3 s of the 5 s chunk.

    Emits the same flat latent the MLP encoder did, so the quantizer, the decoder and
    the loss are untouched.
    """

    @validate_call
    def __init__(
        self,
        *,
        num_steps: int,
        num_axes: int,
        out_features: int,
        channels: int = 32,
        dilations: tuple[int, ...] = (1, 3, 9),
    ) -> None:
        super().__init__()

        self.num_steps = num_steps
        self.num_axes = num_axes
        self.conv = nn.Sequential(
            nn.Conv1d(num_axes, channels, 7, padding=3),
            nn.ELU(),
            *[_DilatedResidualUnit(channels, d) for d in dilations],
            nn.ELU(),
        )
        self.project = nn.Linear(channels * num_steps, out_features)

    @override
    def forward(self, x: Tensor) -> Tensor:
        *batch, _ = x.shape
        # flat layout is (timestep, axis); conv wants (batch, axis, timestep)
        chunk = x.reshape(-1, self.num_steps, self.num_axes).transpose(1, 2)

        return self.project(self.conv(chunk).flatten(1)).reshape(*batch, -1)


@final
class ChunkConvDecoder(Module):
    """Latent -> flat action chunk. Mirror of `ChunkConvEncoder`."""

    @validate_call
    def __init__(
        self,
        *,
        num_steps: int,
        num_axes: int,
        in_features: int,
        channels: int = 32,
        dilations: tuple[int, ...] = (9, 3, 1),
    ) -> None:
        super().__init__()

        self.num_steps = num_steps
        self.num_axes = num_axes
        self.channels = channels
        self.project = nn.Linear(in_features, channels * num_steps)
        self.conv = nn.Sequential(
            nn.ELU(),
            *[_DilatedResidualUnit(channels, d) for d in dilations],
            nn.ELU(),
            nn.Conv1d(channels, num_axes, 7, padding=3),
        )

    @override
    def forward(self, z: Tensor) -> Tensor:
        *batch, _ = z.shape
        h = self.project(z).reshape(-1, self.channels, self.num_steps)

        return (
            self.conv(h).transpose(1, 2).reshape(*batch, self.num_steps * self.num_axes)
        )


@final
class AxisShrinkage(Module):
    """Soft-threshold selected axes of a flat chunk, so quiet regions decode to EXACTLY zero.

    WHY. Measured on the d12 holdout, fork1's reconstruction is not too smooth -- it is
    too JAGGED. Total variation of the reconstruction over total variation of ground
    truth runs 1.09-1.39 on fork1 across every arm (against 0.40-0.57 on traction, which
    genuinely is over-smoothed). Ground truth fork1 is exactly 0.000 for 91.7% of
    messages; a continuous decoder head cannot emit exact zeros, so it sprays
    low-amplitude noise across the quiet stretches while separately attenuating the real
    events. Two errors in opposite directions on one axis.

    Rate does not fix it -- fork1's TV ratio RISES with rate (1.02 at 16 bits to 1.39 at
    32), i.e. extra capacity buys more wiggle, not sharper events. It is a parameterization
    problem: the output family cannot represent the spike at zero.

    Soft-thresholding is the minimal fix and the principled one -- `sign(x) * relu(|x| -
    tau)` is the proximal operator of the L1 norm, the standard map for a
    sparsity-inducing prior. Everything within +/-tau becomes exactly zero; everything
    outside is shifted by tau, which the decoder trivially compensates for on events an
    order of magnitude larger. `tau` is learned per axis through a softplus so it stays
    positive, and only the named axes are touched: traction and steering have no zero
    atom and shrinking them would only bias them.

    Belongs LAST in the decoder, after any `ChunkIDCT`, so it acts on time-domain samples.
    """

    @validate_call
    def __init__(
        self,
        *,
        num_steps: int,
        num_axes: int,
        axes: tuple[int, ...],
        init_threshold: float = 0.05,
    ) -> None:
        super().__init__()

        if not axes:
            msg = "`axes` must name at least one axis to shrink"
            raise ValueError(msg)
        if any(not 0 <= a < num_axes for a in axes):
            msg = f"axes {axes} out of range for num_axes {num_axes}"
            raise ValueError(msg)

        self.num_steps = num_steps
        self.num_axes = num_axes
        self.axes = axes
        # softplus^-1 so the stored parameter is unconstrained and tau starts at
        # `init_threshold`
        inv = math.log(math.expm1(init_threshold))
        self.raw_threshold = nn.Parameter(torch.full((len(axes),), inv))

        mask = torch.zeros(num_axes)
        mask[list(axes)] = 1.0
        self.register_buffer("axis_mask", mask)

    @property
    def thresholds(self) -> Tensor:
        return F.softplus(self.raw_threshold)

    @override
    def forward(self, x: Tensor) -> Tensor:
        *batch, _ = x.shape
        chunk = x.reshape(*batch, self.num_steps, self.num_axes)

        # `.to(chunk.dtype)`: the parameter is fp32 but under `bf16-mixed` the chunk
        # arrives as bfloat16, and `index_copy` requires matching dtypes
        tau = chunk.new_zeros(self.num_axes)
        tau = tau.index_copy(
            0,
            torch.as_tensor(self.axes, device=chunk.device, dtype=torch.long),
            self.thresholds.to(chunk.dtype),
        )
        shrunk = torch.sign(chunk) * F.relu(chunk.abs() - tau)

        return shrunk.reshape(*batch, self.num_steps * self.num_axes)


def linear_keyframe_matrix(num_steps: int, stride: int) -> Tensor:
    """`(num_steps, K)` linear-interpolation matrix from keyframes `0, s, 2s, ...`.

    Keyframe `j` sits at step `j * stride`; `(num_steps - 1) % stride == 0` so the
    last step is a keyframe and nothing is extrapolated. Rows at keyframes are
    one-hot, so interpolating the keyframes of a signal returns them exactly.
    """
    if stride < 1 or (num_steps - 1) % stride:
        msg = f"(num_steps - 1) = {num_steps - 1} must be a multiple of stride {stride}"
        raise ValueError(msg)
    keys = (num_steps - 1) // stride + 1
    out = torch.zeros(num_steps, keys, dtype=torch.float64)
    for t in range(num_steps):
        lo, rem = divmod(t, stride)
        if rem == 0:
            out[t, lo] = 1.0
        else:
            u = rem / stride
            out[t, lo] = 1.0 - u
            out[t, lo + 1] = u
    return out.float()


@final
class KeyframeInterpolation(Module):
    """Flat keyframe chunk `(..., K * A)` -> flat dense chunk `(..., T * A)`.

    A FIXED linear interpolation (a buffer, never trained), so the tokenizer can
    code a 100-step 30 Hz chunk at 10 Hz (34 keyframes) and still be scored --
    and served -- at 30 Hz in one graph. Its error floor on ground truth is the
    tokenizer's best case and is reported separately (`interp_floor_ev`).
    """

    @validate_call
    def __init__(self, *, num_steps: int, num_axes: int, stride: int) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.num_axes = num_axes
        self.stride = stride
        self.register_buffer("weight", linear_keyframe_matrix(num_steps, stride))

    @property
    def num_keyframes(self) -> int:
        return self.weight.shape[1]

    @override
    def forward(self, x: Tensor) -> Tensor:
        keys = x.reshape(*x.shape[:-1], self.num_keyframes, self.num_axes)
        dense = torch.einsum("tk,...ka->...ta", self.weight.to(keys.dtype), keys)
        return dense.reshape(*x.shape[:-1], self.num_steps * self.num_axes)


@final
class NormedTokenEmbedding(Module):
    """`MLP -> LayerNorm -> learnable gain`: one scale-matched token.

    WHY. #276 measured the driving speed token entering the trunk ~20x below the
    patch tokens; a bare MLP embedding of a low-dimensional vector (state, hand)
    has the same failure mode, and 480 patch tokens then drown it (reliance
    metrics read ~0). The output LayerNorm pins the token's per-element RMS at
    ~1 and `token_gain` sets it; the gain is initialised from a MEASURED patch
    token RMS (`calibrate`), the `fusion_goal_rms` lesson -- a Monte-Carlo guess
    was 1.86x off there. The gain is excluded from weight decay (SelectiveAdamW).

    `embed_learned(raw)` maps a learned d-dim vector (e.g. `no_hand`) through the
    SAME output norm and gain, so a substituted token sits at the same scale as a
    real one.

    NO INPUT NORM by default (`input_norm=False`). A per-sample LayerNorm over a
    heterogeneous physical vector (hand current / pos_err / pos, or arm q +
    hand_prev) makes the embedding exactly invariant to `x -> a*x + b*1`: two
    valid hand readings that differ only in their overall current / pos_err
    LEVEL -- the thing the token is meant to carry -- collapse to the same token.
    The inputs already arrive on fixed scales (hand_features' fixed scales; the
    state is standardized in-graph), so only the OUTPUT norm + gain are needed
    to fix the token-scale problem. `input_norm=True` is kept for ablation only.
    """

    @validate_call
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_features: tuple[int, ...] = (512,),
        gain_init: float = 1.0,
        input_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_norm: Module = (
            nn.LayerNorm(in_features) if input_norm else nn.Identity()
        )
        layers: list[Module] = []
        width = in_features
        for hidden in hidden_features:
            layers += [nn.Linear(width, hidden), nn.GELU()]
            width = hidden
        layers.append(nn.Linear(width, out_features))
        self.mlp = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(out_features)
        self.token_gain = nn.Parameter(torch.tensor(float(gain_init)))

    def embed_learned(self, raw: Tensor) -> Tensor:
        return self.token_gain * self.out_norm(raw)

    @torch.no_grad()
    def calibrate(self, target_rms: float) -> None:
        """Set the gain so the output per-element RMS equals `target_rms`."""
        self.token_gain.fill_(float(target_rms))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.token_gain * self.out_norm(self.mlp(self.in_norm(x)))
