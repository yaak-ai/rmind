"""One-frame D12 inference with external KV state and explicit flow start noise."""

import torch
from torch import Tensor, nn

from rmind.components.rtc_flow_head import TokenTimeFlowHead
from rmind.components.transformer.causal_frame import frame_rope_cos_sin
from rmind.models.patch_policy_continuous import PatchPolicyContinuous


class D12FlowDecoderStep(nn.Module):
    input_names: tuple[str, ...]
    image_mean: Tensor
    image_std: Tensor

    def __init__(self, policy: PatchPolicyContinuous) -> None:
        super().__init__()
        if policy.fm_head is None:
            message = "D12 flow export requires a flow head"
            raise ValueError(message)
        if tuple(policy.cameras) not in {
            ("cam_fork",),
            ("cam_fork", "cam_left_forward", "cam_right_forward"),
            ("cam_fork", "cam_left_backward", "cam_right_backward"),
        }:
            message = "D12 flow export requires fork-only or the configured three-camera order"
            raise ValueError(message)
        self.policy = policy.eval()
        self.input_names = (*policy.cameras, *INPUT_NAMES[3:])
        self.rtc_enabled = isinstance(policy.fm_head, TokenTimeFlowHead)
        if self.rtc_enabled:
            self.input_names = (*self.input_names, "clean_prefix", "prefix_mask")
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3, 1, 1)
        )

    def empty_cache(self) -> tuple[Tensor, Tensor, Tensor]:
        past_k, past_v, bias = self.policy.encoder.empty_cache(
            batch_size=1, cache_frames=5
        )
        return past_k.transpose(0, 1), past_v.transpose(0, 1), bias

    def rope(self, frame_index: int) -> tuple[Tensor, Tensor]:
        cosine, sine = frame_rope_cos_sin(
            torch.tensor(frame_index),
            head_dim=self.policy.encoder.head_dim,
            base=self.policy.encoder.rope_base,
        )
        return cosine.reshape(1, -1).float(), sine.reshape(1, -1).float()

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tensors = dict(zip(self.input_names, inputs, strict=True))
        images = torch.stack([tensors[camera] for camera in self.policy.cameras], dim=2)
        images = (images - self.image_mean) / self.image_std
        tokens = self.policy.frame_tokens(
            images,
            tensors["speed"],
            {
                "fork_above_300": tensors["fork_above_300"],
                "relative_ego_pos": tensors["relative_ego_pos"],
                "relative_dropoff_pos": tensors["relative_dropoff_pos"],
                "fork_height": tensors["fork_height"],
            },
        )
        encoded, new_k, new_v = self.policy.encoder.step(
            tokens[:, 0],
            past_k=tensors["past_k"].transpose(0, 1),
            past_v=tensors["past_v"].transpose(0, 1),
            cache_bias=tensors["cache_bias"],
            cos=tensors["rope_cos"],
            sin=tensors["rope_sin"],
            readout_only_final_block=True,
        )
        features = encoded[:, -1]
        if self.policy.norm is not None:
            features = self.policy.norm(features)
        prefix_arguments = {}
        if self.rtc_enabled:
            prefix_arguments = {
                "clean_prefix": (tensors["clean_prefix"] - self.policy.action_mean)
                / self.policy.action_std,
                "prefix_mask": tensors["prefix_mask"],
            }
        actions = self.policy.fm_sample(
            features, x0=tensors["fm_x0"], **prefix_arguments
        )
        actions = actions * self.policy.action_std + self.policy.action_mean
        return actions, new_k.transpose(0, 1), new_v.transpose(0, 1)


INPUT_NAMES = (
    "cam_fork",
    "cam_left_forward",
    "cam_right_forward",
    "speed",
    "fork_above_300",
    "relative_ego_pos",
    "relative_dropoff_pos",
    "fork_height",
    "past_k",
    "past_v",
    "cache_bias",
    "rope_cos",
    "rope_sin",
    "fm_x0",
)
OUTPUT_NAMES = ("actions", "new_k", "new_v")
