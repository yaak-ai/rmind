"""KV-cached one-tick decode step for the robot-native `NeroPatchPolicy` (contract v3).

The serving graph of the patch family. Per 10 Hz observation tick it encodes ONE
frame -- `[state][hand][patches(3 cams x P)]` -- runs those tokens against the
cached K/V of the previous `cache_frames = window - 1` frames, and returns the
100-step action chunk plus the new frame's K/V. It is a separate class from the
driving `PatchPolicyDecoderStep` on purpose: that one is wired to PatchPolicy's
single image, speed and waypoints inputs and 257 tokens per frame.

I/O (bound BY NAME; shapes are nutron-cli `policy_contract.patch_io_shapes`):

| input        | shape                                   | notes |
|--------------|-----------------------------------------|-------|
| `images`     | `(1, n_cams, 3, H, W)` float32          | unit RGB (`uint8 / 255`) on the model grid, i.e. AFTER `rmind.data.nero_image.preprocess`; ImageNet norm is in-graph |
| `state`      | `(1, 26)` float32                       | RAW side-major state (measured q + hand_prev/1000); the state standardizer is in-graph |
| `side_valid` | `(1, 2)` float32                        | 1 = valid side |
| `hand_token` | `(1, dim)` float32                      | `hf.build_token` vector, `hand_valid` LAST (only with a hand token) |
| `past_k/v`   | `(L, 1, heads, cache_frames*T, head_dim)` | read-only ring, oldest first |
| `cache_bias` | `(1, 1, 1, cache_frames*T)`             | 0 = filled, `-1e4` = empty |
| `rope_cos/sin` | `(1, head_dim)` float32               | host-computed (float64) from the int64 frames-since-reset counter |

| output    | shape                    | notes |
|-----------|--------------------------|-------|
| `actions` | `(1, 100, 26)` float32   | in the ACTION STANDARDIZER's (relative-mode) space; the host unstandardizes and adds the anchor (`relative_mask`, anchor = the state of THIS observation) |
| `new_k/v` | `(L, 1, heads, T, head_dim)` | the host shifts them into its ring |
| `codes`   | `(1, num_quantizers)` int64 | the first VALID side's argmax codes (diagnostic) |

Goal: none (`goal_mode` no_goal/none) -- there is no goal input. `camera_cond` is
a CONSTANT of the graph (the contract carries the same array verbatim).
"""

from __future__ import annotations

from typing import Any, override

import torch
from torch import Tensor, nn

from rmind.components.transformer.causal_frame import (
    CausalFrameTransformer,
    frame_rope_cos_sin,
)
from rmind.models.nero_patch_policy import NeroPatchPolicy

__all__ = ["NeroPatchPolicyDecoderStep"]


class NeroPatchPolicyDecoderStep(nn.Module):
    """Export wrapper: one 10 Hz frame through a trained robot-native `NeroPatchPolicy`."""

    def __init__(
        self,
        *,
        policy: NeroPatchPolicy,
        camera_cond: Tensor | None = None,
        readout_only_final_block: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(policy.encoder, CausalFrameTransformer):
            msg = f"policy.encoder must be a CausalFrameTransformer, got {type(policy.encoder).__name__}"
            raise TypeError(msg)
        if not policy.robot:
            msg = "the decoder step serves the robot action space only"
            raise ValueError(msg)
        if policy.goal_mode == "image":
            msg = "goal_mode='image' has no goal source at serving; export no_goal/none"
            raise ValueError(msg)
        if policy.use_depth:
            msg = "depth is not part of serving contract v3 (depth: null)"
            raise ValueError(msg)
        self.policy = policy.eval()
        self.trunk: CausalFrameTransformer = policy.encoder
        self.readout_only_final_block = readout_only_final_block
        n_cams = len(policy.cameras)
        cond = torch.zeros(1, n_cams, 13) if camera_cond is None else camera_cond.reshape(1, n_cams, 13)
        self.register_buffer("camera_cond", cond.float())
        self.use_hand = policy.use_hand

    # ---------------------------------------------------------------- host side

    @property
    def tokens_per_frame(self) -> int:
        return int(self.trunk.tokens_per_frame)

    def empty_cache(
        self,
        *,
        cache_frames: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Cold `(past_k, past_v, cache_bias)` -- the rollout-start / reset state."""
        return self.trunk.empty_cache(
            batch_size=1, cache_frames=cache_frames, dtype=dtype, device=device
        )

    def rope(self, frame_index: int) -> tuple[Tensor, Tensor]:
        """`(rope_cos, rope_sin)` `(1, head_dim)` for frames-since-reset `frame_index`."""
        cos, sin = frame_rope_cos_sin(
            torch.tensor(frame_index, dtype=torch.int64),
            head_dim=self.trunk.head_dim,
            base=self.trunk.rope_base,
        )
        return cos.reshape(1, -1).float(), sin.reshape(1, -1).float()

    @staticmethod
    def advance(
        past: tuple[Tensor, Tensor, Tensor], new_k: Tensor, new_v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Ring update (`ring: oldest_first_shift`): drop the oldest frame, append."""
        past_k, past_v, bias = past
        k = new_k.shape[-2]
        return (
            torch.cat((past_k[..., k:, :], new_k), dim=-2),
            torch.cat((past_v[..., k:, :], new_v), dim=-2),
            torch.cat((bias[..., k:], torch.zeros_like(bias[..., :k])), dim=-1),
        )

    def frame_inputs(self, batch: dict[str, Any], frame: int) -> dict[str, Tensor]:
        """Training-batch frame `frame` (sample 0) -> this graph's named inputs.

        `image.*` uint8 -> unit float, `state (S, A)` -> flat 26, `hand_token`
        composed exactly as training does. For the streaming gate and replay.
        """
        policy = self.policy
        images = torch.stack(
            [
                batch[policy.image_key.format(camera=c)][0, frame].float() / 255.0
                for c in policy.cameras
            ]
        ).unsqueeze(0)
        out = {
            "images": images,
            "state": batch[policy.state[0]][0, frame].reshape(1, -1).float(),
            "side_valid": batch[policy.side_valid[0]][0].reshape(1, -1).float(),
        }
        if self.use_hand:
            vec = policy.hand_vector(batch)
            if vec is None:
                vec = torch.zeros(1, 1, 1)
            out["hand_token"] = vec[0, frame].reshape(1, -1).float()
        return out

    # -------------------------------------------------------------------- graph

    @override
    def forward(  # noqa: PLR0913, PLR0917
        self,
        images: Tensor,
        state: Tensor,
        side_valid: Tensor,
        past_k: Tensor,
        past_v: Tensor,
        cache_bias: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        hand_token: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        policy = self.policy
        n_sides = side_valid.shape[-1]
        valid = side_valid > 0.5  # noqa: PLR2004
        batch: dict[str, Tensor] = {
            policy.state[0]: state.reshape(1, 1, n_sides, -1),
            policy.side_valid[0]: valid,
            policy.camera_cond[0]: self.camera_cond,
        }
        for index, camera in enumerate(policy.cameras):
            batch[policy.image_key.format(camera=camera)] = images[:, index : index + 1]
        if self.use_hand:
            if hand_token is None:
                msg = "this policy has a hand token; `hand_token` is required"
                raise ValueError(msg)
            batch[policy.hand_token_key] = hand_token.reshape(1, 1, -1)

        tokens = policy._frame_tokens(batch)  # noqa: SLF001  (1, 1, T, d)
        out, new_k, new_v = self.trunk.step(
            tokens[:, 0],
            past_k=past_k,
            past_v=past_v,
            cos=rope_cos,
            sin=rope_sin,
            cache_bias=cache_bias,
            readout_only_final_block=self.readout_only_final_block,
        )
        features = out[:, -1]
        if policy.norm is not None:
            features = policy.norm(features)
        chunk, codes = policy._predict_chunk_and_codes(features)  # noqa: SLF001
        # (1, S, H, A) -> (1, H, S*A): the contract's side-major flat layout
        actions = chunk.permute(0, 2, 1, 3).reshape(1, chunk.shape[2], -1)
        first_valid = valid.to(torch.int64).argmax(dim=-1)  # (1,)
        side_codes = codes.gather(
            1, first_valid.reshape(1, 1, 1).expand(1, 1, codes.shape[-1])
        ).squeeze(1)
        return actions, new_k, new_v, side_codes
