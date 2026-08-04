from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, final, override

import torch
from einops import rearrange
from pydantic import InstanceOf, validate_call
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
class MultiModalGRUTrajectoryHead(Module):
    """Winner-takes-all multi-modal counterpart of GRUTrajectoryHead: predicts
    `num_modes` candidate trajectories from a single shared autoregressive GRU
    decoder, the candidates distinguished only by a learned per-mode embedding
    added to the initial hidden state. `hidden_proj`/`input_proj`/`gru`/
    `output_proj` match GRUTrajectoryHead's parameter names and shapes exactly
    (the only new parameter is `mode_embed`), so a single-mode checkpoint
    warm-starts directly under `strict=false` -- no `state_dict_drop` needed,
    unlike a head whose I/O projections actually change shape.

    Output layout as GRUTrajectoryHead, with an extra leading mode axis: (b,
    num_modes, num_steps, 2*pose_dim). Which candidate is "the" prediction is
    the caller's responsibility (see PolicyObjective's winner-takes-all
    selection) -- this head only proposes candidates.
    """

    @validate_call
    def __init__(
        self,
        *,
        in_features: int,
        hidden_size: int,
        num_steps: int,
        num_modes: int = 4,
        predict_yaw: bool = False,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.num_modes = num_modes
        self.predict_yaw = predict_yaw
        self._hidden_size = hidden_size
        pose_dim = 3 if predict_yaw else 2  # (x, y[, yaw])
        self.hidden_proj = Linear(in_features, hidden_size)
        self.input_proj = Linear(pose_dim, hidden_size)  # embed prev pose means
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_proj = Linear(hidden_size, 2 * pose_dim)
        self.mode_embed = Embedding(num_modes, hidden_size)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (b, 1, in_features)
        b, k = x.size(0), self.num_modes
        pose_dim = 3 if self.predict_yaw else 2
        pose_idx = [0, 2, 4] if self.predict_yaw else [0, 2]

        base_h = self.hidden_proj(x[:, 0])  # (b, H)
        mode_ids = torch.arange(k, device=x.device)
        mode_h = self.mode_embed(mode_ids)  # (k, H)
        h = rearrange(
            base_h.unsqueeze(1) + mode_h.unsqueeze(0), "b k h -> 1 (b k) h"
        )  # (1, b*k, H)

        prev_pose = torch.zeros(b * k, pose_dim, device=x.device, dtype=x.dtype)
        preds, hs = [], []
        for _ in range(self.num_steps):
            inp = self.input_proj(prev_pose).unsqueeze(1)  # (b*k, 1, H)
            out, h = self.gru(inp, h)                      # (b*k, 1, H), (1, b*k, H)
            hs.append(h[0])                                # (b*k, H)
            logits = self.output_proj(out[:, 0])           # (b*k, 2*pose_dim)
            prev_pose = logits[:, pose_idx]                # means only
            preds.append(logits)

        traj = rearrange(torch.stack(preds, dim=1), "(b k) s d -> b k s d", b=b, k=k)
        h_traj = rearrange(torch.stack(hs, dim=1), "(b k) s h -> b k s h", b=b, k=k)
        return traj, h_traj
        # (b, num_modes, num_steps, 2*pose_dim), (b, num_modes, num_steps, H)


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


@final
class FeatureFusionPool(Module):
    """Fuses several heterogeneous feature sources into one pooled vector via
    cross-attention instead of concatenation.

    Each source contributes one or more `embedding_dim`-wide tokens: the
    observation-summary history window contributes one (already-embedded)
    token per tick, tagged with a tick-offset positional embedding; raw
    waypoints are projected to `embedding_dim` and contribute one token per
    point, tagged with a point-index positional embedding (necessary, since
    the shared `waypoint_proj` has no other way to distinguish point order);
    raw speed contributes a single projected token; raw image patches are
    optionally compressed into a small, fixed number of learned "register"
    tokens first (see `image_patch_pool`), each tagged with a slot-index
    positional embedding. Every token also gets a shared per-source-type
    embedding, so the pool can tell sources apart even where a positional
    embedding doesn't apply (e.g. the singleton speed token). All tokens are
    concatenated into one sequence and pooled by `pool` (an
    `AttentionPoolHead`, optionally with `num_queries>1`); a multi-query
    `(b, num_queries, d)` output is flattened to `(b, 1, num_queries*d)` so
    it's a drop-in replacement for a concatenated `features` vector
    everywhere downstream.
    """

    _OBS_SUMMARY, _RAW_WAYPOINTS, _RAW_SPEED, _IMAGE_PATCH = range(4)

    @validate_call
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_waypoints: int,
        history_steps: int,
        pool: InstanceOf[Module],
        # per-token modality dropout, distinct from raw_waypoints_dropout/
        # raw_speed_dropout (which zero a whole feature stream): randomly
        # excludes individual tokens from the cross-attention pooling step
        # each training forward, so the pool can't over-rely on any single
        # token. Each waypoint token is dropped independently at
        # waypoint_token_dropout; the single speed token is dropped at
        # speed_token_dropout. obs_summary/history tokens are never
        # dropped. No-op at eval (self.training is False) or when 0.0
        # (both default).
        waypoint_token_dropout: float = 0.0,
        speed_token_dropout: float = 0.0,
        # register-style compression of raw image patch tokens (e.g. frozen
        # DINOv3 patches read straight off episode.input_embeddings, before
        # the main cross-modal encoder ever attends over them -- see
        # PolicyObjective.raw_image_patches_key) into num_image_patch_tokens
        # learned "register" tokens via cross-attention, mirroring DrivoR
        # (arXiv:2601.05083)'s camera-aware register compression: cheaper
        # than feeding all (e.g. 256) raw patches into this pool's own
        # cross-attention directly, and reuses embeddings the frozen
        # backbone already computed for the main encoder -- no extra vision
        # backbone forward pass. None (default) disables image tokens
        # entirely, unchanged from before this option existed.
        image_patch_pool: InstanceOf[Module] | None = None,
        # must match image_patch_pool's num_queries -- sizes
        # image_patch_pos_embed. Ignored when image_patch_pool is None.
        num_image_patch_tokens: int = 0,
        image_patch_token_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.waypoint_proj = Linear(2, embedding_dim)
        self.waypoint_pos_embed = Embedding(num_waypoints, embedding_dim)
        self.speed_proj = Linear(1, embedding_dim)
        self.history_pos_embed = Embedding(history_steps, embedding_dim)
        self.image_patch_pool: Module | None = image_patch_pool
        self.image_patch_pos_embed = (
            Embedding(num_image_patch_tokens, embedding_dim)
            if image_patch_pool is not None
            else None
        )
        self.source_type_embed = Embedding(4, embedding_dim)
        self.waypoint_token_dropout = waypoint_token_dropout
        self.speed_token_dropout = speed_token_dropout
        self.image_patch_token_dropout = image_patch_token_dropout

    @override
    def forward(
        self,
        *,
        obs_summary_history: Tensor,
        raw_waypoints: Tensor,
        raw_speed: Tensor,
        raw_image_patches: Tensor | None = None,
    ) -> Tensor:
        # obs_summary_history: (b, history_steps, d)
        # raw_waypoints: (b, num_waypoints, 2)
        # raw_speed: (b, 1)
        # raw_image_patches: (b, num_raw_patches, d), required iff image_patch_pool is set
        history_ids = torch.arange(obs_summary_history.shape[1], device=obs_summary_history.device)
        obs_summary_tokens = (
            obs_summary_history
            + self.history_pos_embed(history_ids)
            + self.source_type_embed.weight[self._OBS_SUMMARY]
        )

        waypoint_ids = torch.arange(raw_waypoints.shape[1], device=raw_waypoints.device)
        waypoint_tokens = (
            self.waypoint_proj(raw_waypoints)
            + self.waypoint_pos_embed(waypoint_ids)
            + self.source_type_embed.weight[self._RAW_WAYPOINTS]
        )

        speed_tokens = self.speed_proj(raw_speed).unsqueeze(1) + self.source_type_embed.weight[
            self._RAW_SPEED
        ]  # (b, 1, d)

        tokens = torch.cat([obs_summary_tokens, waypoint_tokens, speed_tokens], dim=1)

        image_patch_tokens = None
        if self.image_patch_pool is not None:
            if raw_image_patches is None:
                msg = "image_patch_pool is set but raw_image_patches was not provided"
                raise ValueError(msg)
            compressed = self.image_patch_pool(raw_image_patches)  # (b, num_image_patch_tokens, d)
            patch_ids = torch.arange(compressed.shape[1], device=compressed.device)
            image_patch_tokens = (
                compressed
                + self.image_patch_pos_embed(patch_ids)
                + self.source_type_embed.weight[self._IMAGE_PATCH]
            )
            tokens = torch.cat([tokens, image_patch_tokens], dim=1)

        key_padding_mask = None
        if self.training and (
            self.waypoint_token_dropout > 0.0
            or self.speed_token_dropout > 0.0
            or (image_patch_tokens is not None and self.image_patch_token_dropout > 0.0)
        ):
            b = tokens.shape[0]
            h, w = obs_summary_tokens.shape[1], waypoint_tokens.shape[1]
            key_padding_mask = torch.zeros(
                b, tokens.shape[1], dtype=torch.bool, device=tokens.device
            )
            if self.waypoint_token_dropout > 0.0:
                key_padding_mask[:, h : h + w] = (
                    torch.rand(b, w, device=tokens.device) < self.waypoint_token_dropout
                )
            if self.speed_token_dropout > 0.0:
                key_padding_mask[:, h + w : h + w + 1] = (
                    torch.rand(b, 1, device=tokens.device) < self.speed_token_dropout
                )
            if image_patch_tokens is not None and self.image_patch_token_dropout > 0.0:
                p = image_patch_tokens.shape[1]
                key_padding_mask[:, h + w + 1 : h + w + 1 + p] = (
                    torch.rand(b, p, device=tokens.device) < self.image_patch_token_dropout
                )
            fully_masked = key_padding_mask.all(dim=1)
            if fully_masked.any():
                # avoid an all-masked row -> every key/value ignored -> NaN softmax
                # (obs_summary tokens, indices [0, h), are never dropped above, so
                # this only guards the degenerate h == 0 case)
                key_padding_mask[fully_masked, 0] = False

        pooled = self.pool(tokens, key_padding_mask)  # (b, num_queries, d)
        return rearrange(pooled, "b q d -> b 1 (q d)")
