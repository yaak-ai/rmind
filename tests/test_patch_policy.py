"""Unit tests for `PatchPolicy` (https://arxiv.org/pdf/2607.18236).

Covers the block-causal attention mask, per-frame readout/loss shapes, the
teacher-forced offset semantics inherited from `JointPolicyObjective` (#258),
and equivalence of the batched `_gather_offset` with the joint-policy original.
"""

from typing import override

import torch
from torch import Tensor
from torch.nn import Identity, L1Loss, Linear, Module, Sequential
from torchvision.ops import MLP

from rmind.components.base import Modality
from rmind.components.containers import ModuleDict
from rmind.components.loss import FocalLoss
from rmind.components.nn import Embedding
from rmind.components.norm import Scaler, UniformBinner
from rmind.components.objectives.base import ObjectivePredictionKey
from rmind.components.objectives.joint_policy import JointPolicyObjective
from rmind.components.vq import ResidualVQ
from rmind.models.action_tokenizer import ActionTokenizer
from rmind.models.control_transformer import PredictionConfig
from rmind.models.patch_policy import (
    BlockCausalTransformer,
    PatchPolicy,
    block_causal_mask,
)

BATCH_SIZE = 2
EPISODE_LENGTH = 3
NUM_PATCHES = 4
IMAGE_DIM = 8
GOAL_DIM = 8
POLICY_DIM = 16
ACTION_HORIZON = 3
ACTION_FIELDS = 4
ACTION_DIM = ACTION_HORIZON * ACTION_FIELDS
LATENT_DIM = 8
NUM_QUANTIZERS = 2
CODEBOOK_SIZE = 4
SPEED_BINS = 8


def _make_tokenizer() -> ActionTokenizer:
    """Tiny ActionTokenizer mirroring config/model/yaak/action_tokenizer/raw.yaml."""
    return ActionTokenizer(
        input_transform=Sequential(
            Identity(),
            ModuleDict(
                modules={
                    Modality.CONTINUOUS: Identity(),
                    Modality.DISCRETE: {
                        "turn_signal": Scaler(in_range=(0.0, 2.0), out_range=(0.0, 1.0))
                    },
                }
            ),
        ),
        encoder=Linear(ACTION_DIM, LATENT_DIM),
        quantizer=ResidualVQ(
            dim=LATENT_DIM,
            codebook_size=CODEBOOK_SIZE,
            num_quantizers=NUM_QUANTIZERS,
            kmeans_init=False,
        ),
        decoder=Linear(LATENT_DIM, ACTION_DIM),
        targets={
            Modality.CONTINUOUS: {
                "gas_pedal": ("continuous", "gas_pedal"),
                "brake_pedal": ("continuous", "brake_pedal"),
                "steering_angle": ("continuous", "steering_angle"),
            },
            Modality.DISCRETE: {"turn_signal": ("discrete", "turn_signal")},
        },
    )


class _GoalEncoderStub(Module):
    """Maps waypoints `(b, t, n, 2)` -> a deterministic latent `(b, t, GOAL_DIM)`."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = Linear(2, GOAL_DIM)

    def encode(self, waypoints: Tensor) -> Tensor:
        return self.proj(waypoints).mean(dim=-2)

    @override
    def forward(self, waypoints: Tensor) -> Tensor:
        return self.encode(waypoints)


def _make_model(*, teacher_force_offset: bool = True) -> PatchPolicy:
    return PatchPolicy(
        input_transform=Identity(),
        # tests feed pre-extracted patch features (b, t, p, d) directly
        image_encoder=Identity(),
        goal_encoder=_GoalEncoderStub(),
        patch_projection=Linear(IMAGE_DIM + GOAL_DIM, POLICY_DIM),
        speed_tokenizer=UniformBinner(range=(0.0, 130.0), bins=SPEED_BINS),
        speed_embedding=Embedding(SPEED_BINS, POLICY_DIM),
        encoder=BlockCausalTransformer(
            dim_model=POLICY_DIM,
            num_layers=2,
            num_heads=2,
            max_sequence_length=EPISODE_LENGTH * (NUM_PATCHES + 1),
        ),
        tokenizer=_make_tokenizer(),
        code_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE]),
        offset_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE * ACTION_DIM]),
        losses=ModuleDict(modules={"code": FocalLoss(), "offset": L1Loss()}),
        norm=torch.nn.LayerNorm(POLICY_DIM),
        sample_codes=False,
        teacher_force_offset=teacher_force_offset,
        prediction_config=PredictionConfig(
            objectives={
                ObjectivePredictionKey.GROUND_TRUTH,
                ObjectivePredictionKey.PREDICTION_VALUE,
                ObjectivePredictionKey.SCORE_L1,
                ObjectivePredictionKey.SCORE_SIGNED_ERROR,
            }
        ),
    ).eval()


def _make_batch() -> dict:
    generator = torch.Generator().manual_seed(0)
    chunk = torch.rand(
        (BATCH_SIZE, EPISODE_LENGTH, ACTION_HORIZON, ACTION_FIELDS), generator=generator
    )
    chunk[..., 3] = torch.randint(
        0, 3, chunk[..., 3].shape, generator=generator
    ).float()
    return {
        "image": {
            "cam_front_left": torch.randn(
                (BATCH_SIZE, EPISODE_LENGTH, NUM_PATCHES, IMAGE_DIM),
                generator=generator,
            )
        },
        "continuous": {
            "speed": torch.rand((BATCH_SIZE, EPISODE_LENGTH, 1), generator=generator)
            * 130.0
        },
        "context": {
            "waypoints": torch.randn(
                (BATCH_SIZE, EPISODE_LENGTH, 10, 2), generator=generator
            )
        },
        "joint_actions": chunk,
    }


def test_block_causal_mask() -> None:
    mask = block_causal_mask(3, 2)

    frames = torch.arange(6) // 2
    for i in range(6):
        for j in range(6):
            blocked = frames[j] > frames[i]
            assert mask[i, j].item() == blocked, (i, j)

    # within-frame: bidirectional (nothing blocked)
    assert not mask[0, 1]
    assert not mask[1, 0]
    # across frames: strictly causal
    assert mask[0, 2]
    assert not mask[2, 0]


def test_features_and_metrics_shapes() -> None:
    model = _make_model()
    batch = _make_batch()

    features, chunk = model._features(batch)  # noqa: SLF001
    assert features.shape == (BATCH_SIZE, EPISODE_LENGTH, POLICY_DIM)
    assert chunk.shape == (BATCH_SIZE, EPISODE_LENGTH, ACTION_HORIZON, ACTION_FIELDS)

    metrics = model._compute_metrics(batch)  # noqa: SLF001
    losses = metrics["policy", "loss"]
    assert set(losses.keys()) == {
        *(f"code_{q}" for q in range(NUM_QUANTIZERS)),
        "offset",
    }
    for value in losses.values():
        assert value.isfinite()
    assert metrics["policy", "metric", "offset_sampled_recon"].isfinite()


def test_frozen_modules_receive_no_grad() -> None:
    model = _make_model()

    assert all(not p.requires_grad for p in model.tokenizer.parameters())
    assert all(not p.requires_grad for p in model.goal_encoder.parameters())

    model.train()
    assert not model.tokenizer.training
    assert not model.goal_encoder.training
    assert not model.image_encoder.training

    loss = model._compute_metrics(_make_batch())["policy", "loss"].sum(  # noqa: SLF001
        reduce=True
    )
    loss.backward()

    assert all(p.grad is None for p in model.tokenizer.parameters())
    assert all(p.grad is None for p in model.goal_encoder.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.code_head.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.offset_head.parameters()
    )


def test_gather_offset_matches_joint_policy() -> None:
    generator = torch.Generator().manual_seed(1)
    offsets = torch.randn(
        (BATCH_SIZE, NUM_QUANTIZERS, CODEBOOK_SIZE, ACTION_DIM), generator=generator
    )
    codes = torch.randint(
        0, CODEBOOK_SIZE, (BATCH_SIZE, NUM_QUANTIZERS), generator=generator
    )

    torch.testing.assert_close(
        PatchPolicy._gather_offset(offsets, codes),  # noqa: SLF001
        JointPolicyObjective._gather_offset(offsets, codes),  # noqa: SLF001
    )

    # the batched form must equal per-frame application
    offsets_bt = torch.randn(
        (BATCH_SIZE, EPISODE_LENGTH, NUM_QUANTIZERS, CODEBOOK_SIZE, ACTION_DIM),
        generator=generator,
    )
    codes_bt = torch.randint(
        0,
        CODEBOOK_SIZE,
        (BATCH_SIZE, EPISODE_LENGTH, NUM_QUANTIZERS),
        generator=generator,
    )
    gathered = PatchPolicy._gather_offset(offsets_bt, codes_bt)  # noqa: SLF001
    for t in range(EPISODE_LENGTH):
        torch.testing.assert_close(
            gathered[:, t],
            JointPolicyObjective._gather_offset(offsets_bt[:, t], codes_bt[:, t]),  # noqa: SLF001
        )


def test_teacher_forcing_gradient_routing() -> None:
    """With teacher forcing, the offset loss must not push gradient through
    entries at non-target codes."""
    model = _make_model(teacher_force_offset=True)
    batch = _make_batch()

    features, chunk = model._features(batch)  # noqa: SLF001
    with torch.no_grad():
        target_codes = model.tokenizer(chunk)
        target = model.tokenizer._normalize(chunk.flatten(-2, -1))  # noqa: SLF001

    _, offsets = model._heads(features)  # noqa: SLF001
    offsets = offsets.detach().requires_grad_()

    predicted = model.tokenizer.invert(target_codes) + model._offset(  # noqa: SLF001
        offsets, target_codes
    )
    model.losses["offset"](predicted, target).backward()

    grad = offsets.grad
    assert grad is not None
    index = target_codes[..., None, None].expand(*target_codes.shape, 1, ACTION_DIM)
    target_grad = grad.gather(-2, index)
    assert target_grad.abs().sum() > 0
    # zero out target-code entries: everything else must have zero gradient
    grad_zeroed = grad.scatter(-2, index, torch.zeros_like(target_grad))
    assert grad_zeroed.abs().sum() == 0


def test_forward_and_predict_step() -> None:
    model = _make_model()
    batch = _make_batch()

    output = model.forward(batch)
    assert output["policy", "joint_actions"].shape == (
        BATCH_SIZE,
        ACTION_HORIZON,
        ACTION_FIELDS,
    )

    predictions = model.predict_step(batch)
    for key in (
        ObjectivePredictionKey.GROUND_TRUTH,
        ObjectivePredictionKey.PREDICTION_VALUE,
        ObjectivePredictionKey.SCORE_L1,
        ObjectivePredictionKey.SCORE_SIGNED_ERROR,
    ):
        prediction = predictions["policy", key]
        assert prediction.value["continuous", "gas_pedal"].shape == (
            BATCH_SIZE,
            ACTION_HORIZON,
        )
        assert prediction.value["discrete", "turn_signal"].shape == (
            BATCH_SIZE,
            ACTION_HORIZON,
        )

    score = predictions["policy", ObjectivePredictionKey.SCORE_L1]
    assert (score.value["continuous", "gas_pedal"] >= 0).all()


def test_readout_is_causally_valid() -> None:
    """Frame t's readout must not change when future frames' inputs change."""
    model = _make_model()
    batch = _make_batch()

    features, _ = model._features(batch)  # noqa: SLF001

    perturbed = _make_batch()
    perturbed["image"]["cam_front_left"] = batch["image"]["cam_front_left"].clone()
    perturbed["continuous"]["speed"] = batch["continuous"]["speed"].clone()
    perturbed["context"]["waypoints"] = batch["context"]["waypoints"].clone()
    perturbed["joint_actions"] = batch["joint_actions"].clone()
    # change only the LAST frame's observations
    perturbed["image"]["cam_front_left"][:, -1] += 1.0
    perturbed["continuous"]["speed"][:, -1] = 100.0
    perturbed["context"]["waypoints"][:, -1] += 1.0

    features_perturbed, _ = model._features(perturbed)  # noqa: SLF001

    torch.testing.assert_close(features[:, :-1], features_perturbed[:, :-1])
    assert not torch.allclose(features[:, -1], features_perturbed[:, -1])
