"""KV-cached one-tick decode step for `PatchPolicy` -- the deployment export target.

Per tick the graph encodes exactly ONE new frame (1 speed token prepended to 256
goal-fused patch tokens = 257 queries), runs those 257 queries against the cached
K/V of the past frames, and returns the action chunk plus the new frame's K/V.
Old frames are never re-encoded and never re-attended to each other.

Runtime contract (drivr)
------------------------
Inputs, all bound **by name**:

| name | shape | notes |
| --- | --- | --- |
| `image` | `(1, 1, 3, H, W)` | ONE frame, `[0, 1]` float. ImageNet norm is in-graph, so serve with `--image-norm unit`. `H = W = 224` (dinov2) / `256` (dinov3) -- `Resize=0`, a wrong size is silently sheared. |
| `speed` | `(1, 1, 1)` | km/h |
| `waypoints` | `(1, 1, 10, 2)` | ego-frame, /100 |
| `past_k`, `past_v` | `(L, 1, heads, cache_frames * 257, head_dim)` | read-only |
| `cache_bias` | `(1, 1, 1, cache_frames * 257)` | `0` = filled slot, `-1e4` = empty |
| `rope_cos`, `rope_sin` | `(1, head_dim)` | this frame's rotation, from the episode frame counter |

Outputs: `policy.joint_actions` `(1, horizon, action_features)`, and `new_k` /
`new_v` `(L, 1, heads, 257, head_dim)`.

Checkpoints trained with the auxiliary trajectory head (`trajectory_head`, DrivoR
arXiv:2601.05083) emit one more output, `policy.trajectory`
`(1, num_trajectory_hypotheses, trajectory_horizon, 3)` -- `num_trajectory_hypotheses`
multi-hypothesis ego-frame `(x, y, heading)` forecasts for frames
`t0+1 .. t0+trajectory_horizon`, unranked by default (the training loss is
winner-takes-all, so there is no confidence output to pick a mode by;
hypothesis 0 is not "the best" -- UNLESS a `mode_head` is also present, see
below). Checkpoints without the head export the three outputs above and
nothing else, so consumers must bind outputs by NAME and treat
`policy.trajectory` as optional.

Checkpoints additionally trained with a trajectory-MODE classifier
(`mode_head`, via `PatchPolicy.load_for_mode_head_training`) emit one more
output still, `policy.trajectory_mode_logits` `(1, num_trajectory_hypotheses)`
-- unnormalized logits scoring each `policy.trajectory` hypothesis, so the
mode a downstream consumer would run with is `argmax(policy.trajectory_mode_logits,
dim=-1)`. Only present when both `trajectory_head` AND `mode_head` are set.

The host owns the ring buffer: it shifts `new_k`/`new_v` into `past_k`/`past_v`
and appends `257` zeros to the filled region of `cache_bias`. Nothing is written
inside the graph -- no `ScatterElements`, and the cache tensors are ordinary
engine I/O.

⚠️ `TRTEngine.run` binds via `set_tensor_address`, a raw pointer with **no size
validation** (hand-off §3.4). A cache allocated for a different `cache_frames`,
`L`, `heads` or dtype is *not* an error: TRT reinterprets the buffer and the model
merely looks weak. Validate every binding's shape against
`engine.get_tensor_shape(name)` before the first `run`.

⚠️ Reset `past_k`/`past_v` (i.e. set `cache_bias` fully to `-1e4`) and restart the
frame counter on every episode boundary -- engage, disengage and manual override.
drivr already clears the action plan on those transitions; hook the same paths.

Why `rope_cos`/`rope_sin` are inputs
------------------------------------
They are the only positional state, they are two `head_dim` vectors, and computing
them host-side in float64 keeps a long-episode absolute frame counter exact while
leaving the TRT graph with **zero `Sin`/`Cos` nodes** -- trigonometric ops are the
most fp16-fragile part of this model family (see the trt-export skill §2).
"""

from collections.abc import Mapping
from typing import final, override

import torch
from tensordict import TensorDict
from torch import Tensor, nn

from rmind.components.transformer.causal_frame import (
    CausalFrameTransformer,
    frame_rope_cos_sin,
)
from rmind.models.patch_policy import PatchPolicy

__all__ = ["PatchPolicyDecoderStep"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@final
class PatchPolicyDecoderStep(nn.Module):
    """Export wrapper around a `PatchPolicy` whose trunk is a `CausalFrameTransformer`.

    Everything except the trunk is reused unchanged: the frozen ViT encoder, the
    frozen goal encoder and its RVQ, the fusion, `patch_projection`,
    `speed_embedding`, `norm`, and the VQ-BeT `code_head`/`offset_head`/tokenizer.
    """

    def __init__(
        self, *, policy: PatchPolicy, readout_only_final_block: bool = True
    ) -> None:
        super().__init__()
        if not isinstance(policy.encoder, CausalFrameTransformer):
            msg = (
                "policy.encoder must be a CausalFrameTransformer; "
                f"got {type(policy.encoder).__name__}"
            )
            raise TypeError(msg)
        self.policy = policy.eval()
        self.trunk: CausalFrameTransformer = policy.encoder
        # §3.3: the head reads one token per frame, so the final block's attention
        # output and MLP for the other 256 positions are discarded. K/V are still
        # produced for all 257 -- future frames attend to them.
        self.readout_only_final_block = readout_only_final_block
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).reshape(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).reshape(1, 1, 3, 1, 1)
        )

    # ---------------------------------------------------------------- host side

    def empty_cache(
        self,
        *,
        cache_frames: int,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """`(past_k, past_v, cache_bias)` for a cold cache -- the engage-time state."""
        return self.trunk.empty_cache(
            batch_size=batch_size, cache_frames=cache_frames, device=device, dtype=dtype
        )

    def rope(self, frame_index: int) -> tuple[Tensor, Tensor]:
        """`(rope_cos, rope_sin)` `(1, head_dim)` for the episode-absolute frame index."""
        cos, sin = frame_rope_cos_sin(
            torch.tensor(frame_index),
            head_dim=self.trunk.head_dim,
            base=self.trunk.rope_base,
        )
        return cos.reshape(1, -1), sin.reshape(1, -1)

    @staticmethod
    def advance(
        past: tuple[Tensor, Tensor, Tensor], new_k: Tensor, new_v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Ring-buffer update, i.e. what drivr does between ticks.

        Shift left by one frame block and write the new K/V into the freed tail.
        In the runtime this is a device-to-device copy of `257` tokens per layer,
        not of the whole cache.
        """
        past_k, past_v, bias = past
        k = new_k.shape[-2]
        return (
            torch.cat((past_k[..., k:, :], new_k), dim=-2),
            torch.cat((past_v[..., k:, :], new_v), dim=-2),
            torch.cat((bias[..., k:], torch.zeros_like(bias[..., :k])), dim=-1),
        )

    # -------------------------------------------------------------------- graph

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> TensorDict:
        policy = self.policy
        image = (inputs["image"] - self.image_mean) / self.image_std

        tokens = policy._frame_tokens(  # ruff: ignore[private-member-access]
            image, inputs["speed"], inputs["waypoints"]
        )  # (b, 1, 257, d)

        out, new_k, new_v = self.trunk.step(
            tokens[:, 0],
            past_k=inputs["past_k"],
            past_v=inputs["past_v"],
            cos=inputs["rope_cos"],
            sin=inputs["rope_sin"],
            cache_bias=inputs["cache_bias"],
            readout_only_final_block=self.readout_only_final_block,
        )

        features = out[:, -1]  # the frame's last patch token = the readout position
        if policy.norm is not None:
            features = policy.norm(features)

        out_policy: dict[str, Tensor] = {
            "joint_actions": policy._predict_chunk(features)  # ruff: ignore[private-member-access]
        }
        if policy.trajectory_head is not None:
            # auxiliary direct-regression forecast (DrivoR): reads the SAME
            # readout `features` as the VQ-BeT chunk, so it costs one MLP and
            # cannot perturb `joint_actions`.
            out_policy["trajectory"] = policy._predict_trajectory(features)  # ruff: ignore[private-member-access]
            if policy.mode_head is not None:
                # trajectory-mode classifier: also reads the SAME readout
                # `features`, costs one more MLP, and cannot perturb
                # `joint_actions` or `trajectory` either.
                out_policy["trajectory_mode_logits"] = policy._predict_mode(  # ruff: ignore[private-member-access]
                    features
                )

        return TensorDict({"policy": out_policy, "new_k": new_k, "new_v": new_v})
