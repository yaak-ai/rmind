"""Unit tests for `PatchPolicyDecoderStep` -- the KV-cached deployment export.

The decoder step's output set is the ONNX graph's output set, so it is a
deployment contract: consumers (rsim's `onnx_decoder`, drivr) bind by name and
must keep working against checkpoints trained before the auxiliary trajectory
head existed.
"""

from typing import override

import torch
from torch import Tensor
from torch.nn import Identity, L1Loss, LayerNorm, Linear, Module
from torchvision.ops import MLP

from rmind.components.containers import ModuleDict
from rmind.components.loss import FocalLoss, WinnerTakesAllPoseLoss
from rmind.components.nn import Embedding
from rmind.components.norm import UniformBinner
from rmind.components.transformer.causal_frame import CausalFrameTransformer
from rmind.models.control_transformer import PredictionConfig
from rmind.models.patch_policy import PatchPolicy
from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep
from tests.test_patch_policy import (
    ACTION_DIM,
    CODEBOOK_SIZE,
    GOAL_DIM,
    IMAGE_DIM,
    NUM_PATCHES,
    NUM_QUANTIZERS,
    NUM_TRAJECTORY_HYPOTHESES,
    POLICY_DIM,
    SPEED_BINS,
    TRAJECTORY_HORIZON,
    _GoalEncoderStub,
    _make_tokenizer,
)

WINDOW = 3
NUM_HEADS = 2
TOKENS_PER_FRAME = NUM_PATCHES + 1
IMAGE_HW = 4


class _ImageEncoderStub(Module):
    """`(b, t, c, h, w)` -> `(b, t, p, d_img)`, like the frozen ViT. The decoder
    step normalizes with a `(1, 1, 3, 1, 1)` buffer, so the graph's `image` input
    is a real 5-D frame here rather than pre-extracted patches."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = Linear(3 * IMAGE_HW * IMAGE_HW, NUM_PATCHES * IMAGE_DIM)

    @override
    def forward(self, images: Tensor) -> Tensor:
        b, t = images.shape[:2]
        flat = self.projection(images.reshape(b, t, -1))
        return flat.reshape(b, t, NUM_PATCHES, IMAGE_DIM)


def _make_causal_policy(*, with_trajectory_head: bool) -> PatchPolicy:
    """A `PatchPolicy` with a `CausalFrameTransformer` trunk -- the only trunk the
    decoder step accepts. Mirrors `test_patch_policy._make_model` otherwise; the
    image encoder is `Identity`, so the step is fed pre-extracted patch features.
    """
    losses = {"code": FocalLoss(), "offset": L1Loss()}
    if with_trajectory_head:
        losses["trajectory"] = WinnerTakesAllPoseLoss()

    return PatchPolicy(
        input_transform=Identity(),
        image_encoder=_ImageEncoderStub(),
        goal_encoder=_GoalEncoderStub(),
        patch_projection=Linear(IMAGE_DIM + GOAL_DIM, POLICY_DIM),
        speed_tokenizer=UniformBinner(range=(0.0, 130.0), bins=SPEED_BINS),
        speed_embedding=Embedding(SPEED_BINS, POLICY_DIM),
        encoder=CausalFrameTransformer(
            dim_model=POLICY_DIM,
            num_layers=2,
            num_heads=NUM_HEADS,
            tokens_per_frame=TOKENS_PER_FRAME,
            window=WINDOW,
            attn_dropout=0.0,
            resid_dropout=0.0,
            mlp_dropout=0.0,
            checkpoint=False,
        ),
        tokenizer=_make_tokenizer(),
        code_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE]),
        offset_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE * ACTION_DIM]),
        trajectory_head=(
            MLP(POLICY_DIM, [16, NUM_TRAJECTORY_HYPOTHESES * TRAJECTORY_HORIZON * 3])
            if with_trajectory_head
            else None
        ),
        num_trajectory_hypotheses=NUM_TRAJECTORY_HYPOTHESES,
        losses=ModuleDict(modules=losses),
        norm=LayerNorm(POLICY_DIM),
        sample_codes=False,
        prediction_config=PredictionConfig(objectives=set()),
    ).eval()


def _step_inputs(step: PatchPolicyDecoderStep) -> dict[str, torch.Tensor]:
    past_k, past_v, cache_bias = step.empty_cache(cache_frames=WINDOW - 1)
    rope_cos, rope_sin = step.rope(WINDOW - 1)
    generator = torch.Generator().manual_seed(0)
    return {
        "image": torch.rand(1, 1, 3, IMAGE_HW, IMAGE_HW, generator=generator),
        "speed": torch.rand(1, 1, 1, generator=generator) * 130,
        "waypoints": torch.rand(1, 1, 10, 2, generator=generator) * 2 - 1,
        "past_k": torch.randn(past_k.shape, generator=generator),
        "past_v": torch.randn(past_v.shape, generator=generator),
        "cache_bias": torch.zeros_like(cache_bias),
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
    }


@torch.inference_mode()
def test_decoder_step_omits_trajectory_without_the_head() -> None:
    """Pre-trajectory checkpoints must export the original three outputs and
    nothing else, or every existing consumer's output binding shifts."""
    step = PatchPolicyDecoderStep(
        policy=_make_causal_policy(with_trajectory_head=False)
    ).eval()

    out = step(_step_inputs(step))

    assert set(out.keys(include_nested=True, leaves_only=True)) == {
        ("policy", "joint_actions"),
        "new_k",
        "new_v",
    }


@torch.inference_mode()
def test_decoder_step_emits_trajectory_with_the_head() -> None:
    step = PatchPolicyDecoderStep(
        policy=_make_causal_policy(with_trajectory_head=True)
    ).eval()
    inputs = _step_inputs(step)

    out = step(inputs)

    assert set(out.keys(include_nested=True, leaves_only=True)) == {
        ("policy", "joint_actions"),
        ("policy", "trajectory"),
        "new_k",
        "new_v",
    }
    assert out["policy", "trajectory"].shape == (
        1,
        NUM_TRAJECTORY_HYPOTHESES,
        TRAJECTORY_HORIZON,
        3,
    )
    assert out["policy", "trajectory"].isfinite().all()


@torch.inference_mode()
def test_trajectory_head_does_not_perturb_joint_actions() -> None:
    """The head is a pure read of the readout features -- adding it must leave the
    action chunk and the K/V written back to the cache bit-identical."""
    policy = _make_causal_policy(with_trajectory_head=True)
    step_with = PatchPolicyDecoderStep(policy=policy).eval()
    inputs = _step_inputs(step_with)
    with_head = step_with(inputs)

    policy.trajectory_head = None
    without_head = PatchPolicyDecoderStep(policy=policy).eval()(inputs)

    for key in (("policy", "joint_actions"), "new_k", "new_v"):
        torch.testing.assert_close(with_head[key], without_head[key], rtol=0, atol=0)
