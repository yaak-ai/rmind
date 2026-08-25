"""Unit tests for `PatchPolicy` (https://arxiv.org/pdf/2607.18236).

Covers the block-causal attention mask, per-frame readout/loss shapes, the
teacher-forced offset semantics inherited from `JointPolicyObjective` (#258),
and equivalence of the batched `_gather_offset` with the joint-policy original.
"""

from typing import Any, cast, override

import pytest
import torch
from tensordict import TensorDict
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
from rmind.components.transformer.causal_frame import (
    CausalFrameTransformer,
    frame_rope_cos_sin,
)
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
        # fusion_norm calibrates its goal gain from the quantizer codebooks
        self.quantizer = ResidualVQ(
            dim=GOAL_DIM, codebook_size=4, num_quantizers=2, kmeans_init=False
        )

    def encode(self, waypoints: Tensor) -> Tensor:
        return self.proj(waypoints).mean(dim=-2)

    @override
    def forward(self, waypoints: Tensor) -> Tensor:
        return self.encode(waypoints)


def _make_model(  # noqa: PLR0913
    *,
    teacher_force_offset: bool = True,
    fusion_norm: bool = False,
    sample_codes: bool = False,
    neighbor_smoothing_tau: float | None = None,
    losses: ModuleDict | None = None,
    use_readout_token: bool = False,
    num_register_tokens: int = 0,
    encoder: Module | None = None,
) -> PatchPolicy:
    tokens_per_frame = NUM_PATCHES + 1
    if use_readout_token:
        tokens_per_frame += num_register_tokens + 1
    return PatchPolicy(
        fusion_norm=fusion_norm,
        neighbor_smoothing_tau=neighbor_smoothing_tau,
        use_readout_token=use_readout_token,
        num_register_tokens=num_register_tokens,
        input_transform=Identity(),
        # tests feed pre-extracted patch features (b, t, p, d) directly
        image_encoder=Identity(),
        goal_encoder=_GoalEncoderStub(),
        patch_projection=Linear(IMAGE_DIM + GOAL_DIM, POLICY_DIM),
        speed_tokenizer=UniformBinner(range=(0.0, 130.0), bins=SPEED_BINS),
        speed_embedding=Embedding(SPEED_BINS, POLICY_DIM),
        encoder=encoder
        or BlockCausalTransformer(
            dim_model=POLICY_DIM,
            num_layers=2,
            num_heads=2,
            max_sequence_length=EPISODE_LENGTH * tokens_per_frame,
        ),
        tokenizer=_make_tokenizer(),
        code_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE]),
        offset_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE * ACTION_DIM]),
        losses=losses
        if losses is not None
        else ModuleDict(modules={"code": FocalLoss(), "offset": L1Loss()}),
        norm=torch.nn.LayerNorm(POLICY_DIM),
        sample_codes=sample_codes,
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
    losses = cast("dict[str, Tensor]", metrics["policy", "loss"])
    assert set(losses.keys()) == {
        *(f"code_{q}" for q in range(NUM_QUANTIZERS)),
        "offset",
    }
    for value in losses.values():
        assert value.isfinite()
    # sample_codes=False: sampling is an eval-only decode mode, so the sampled
    # metrics are neither computed nor logged (they would just duplicate
    # offset_argmax_recon* while misrepresenting argmax serving)
    metric_keys = set(cast("dict[str, Tensor]", metrics["policy", "metric"]).keys())
    assert "offset_sampled_recon" not in metric_keys
    assert "offset_sampled_recon_last" not in metric_keys
    assert metrics["policy", "metric", "offset_argmax_recon"].isfinite()


def test_sampled_metrics_present_when_sampling() -> None:
    model = _make_model(sample_codes=True)
    metrics = model._compute_metrics(_make_batch())  # noqa: SLF001

    assert metrics["policy", "metric", "offset_sampled_recon"].isfinite()
    assert metrics["policy", "metric", "offset_sampled_recon_last"].isfinite()


def test_non_teacher_forced_offset_loss_without_sampling() -> None:
    # sample_codes=False + teacher_force_offset=False: the offset loss is
    # supervised at the argmax decode; it must still exist and be finite even
    # though the sampled METRICS are dropped
    model = _make_model(sample_codes=False, teacher_force_offset=False)
    metrics = model._compute_metrics(_make_batch())  # noqa: SLF001

    assert metrics["policy", "loss", "offset"].isfinite()
    assert "offset_sampled_recon" not in set(
        cast("dict[str, Tensor]", metrics["policy", "metric"]).keys()
    )


def test_neighbor_smoothing_targets_and_loss() -> None:
    model = _make_model(
        neighbor_smoothing_tau=0.02,
        losses=ModuleDict(
            modules={"code": FocalLoss(label_smoothing=0.1), "offset": L1Loss()}
        ),
    )
    batch = _make_batch()

    target_codes = model.tokenizer(batch["joint_actions"])
    weights = model._neighbor_smoothing_targets(target_codes)  # noqa: SLF001
    assert weights.shape == (BATCH_SIZE, EPISODE_LENGTH, NUM_QUANTIZERS, CODEBOOK_SIZE)
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    # the ground-truth code is at decoded distance 0 -> largest weight
    assert torch.equal(weights.argmax(-1), target_codes)

    # the smoothing term must actually change the code losses vs uniform
    uniform_model = _make_model(
        losses=ModuleDict(
            modules={"code": FocalLoss(label_smoothing=0.1), "offset": L1Loss()}
        )
    )
    uniform_model.load_state_dict(model.state_dict())
    smoothed = cast("dict[str, Tensor]", model._compute_metrics(batch)["policy", "loss"])  # noqa: SLF001
    uniform = cast("dict[str, Tensor]", uniform_model._compute_metrics(batch)["policy", "loss"])  # noqa: SLF001
    assert not torch.allclose(smoothed["code_0"], uniform["code_0"])
    # the offset loss is untouched by the code-smoothing change
    assert torch.allclose(smoothed["offset"], uniform["offset"])


def test_neighbor_smoothing_requires_focal_loss() -> None:
    with pytest.raises(TypeError, match="neighbor_smoothing_tau"):
        _make_model(
            neighbor_smoothing_tau=0.02,
            losses=ModuleDict(
                modules={"code": torch.nn.CrossEntropyLoss(), "offset": L1Loss()}
            ),
        )
    with pytest.raises(ValueError, match="neighbor_smoothing_tau"):
        _make_model(neighbor_smoothing_tau=0.0)


def test_frozen_modules_receive_no_grad() -> None:
    model = _make_model()

    assert all(not p.requires_grad for p in model.tokenizer.parameters())
    assert all(not p.requires_grad for p in model.goal_encoder.parameters())

    model.train()
    assert not model.tokenizer.training
    assert not model.goal_encoder.training
    assert not model.image_encoder.training

    loss = cast(  # noqa: SLF001
        "TensorDict", model._compute_metrics(_make_batch())["policy", "loss"]
    ).sum(reduce=True)
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


def test_fusion_norm_balances_sources() -> None:
    model = _make_model(fusion_norm=True)
    batch = _make_batch()

    inputs = model.input_transform(batch)
    patches = model.image_encoder(inputs["image"]["cam_front_left"])
    goal = model.goal_encoder.encode(inputs["context"]["waypoints"])

    normed = model.fusion_patch_norm(patches) * model.fusion_patch_gain
    goal * model.fusion_goal_gain

    patch_rms = normed.pow(2).mean().sqrt().item()
    # the goal gain is calibrated from codebook combinations, the stub's encode
    # is a different map -- assert ballpark scale match, not equality
    codes = model.goal_encoder.quantizer.lookup(
        torch.randint(0, 4, (256, 2), generator=torch.Generator().manual_seed(1))
    )
    zq_rms = (codes * model.fusion_goal_gain).pow(2).mean().sqrt().item()
    tolerance = 0.2
    assert abs(patch_rms - 1.0) < tolerance
    assert abs(zq_rms - 1.0) < tolerance

    assert model.fusion_patch_gain.requires_grad
    assert model.fusion_goal_gain.requires_grad
    loss = cast(  # noqa: SLF001
        "TensorDict", model._compute_metrics(batch)["policy", "loss"]
    ).sum(reduce=True)
    loss.backward()
    assert model.fusion_goal_gain.grad is not None

    # flag off -> attributes absent
    off = _make_model()
    assert off.fusion_patch_norm is None


def test_fusion_norm_calibration_deterministic() -> None:
    """Same goal-encoder weights (the real-world case: loaded from a checkpoint)
    must yield the identical calibrated gain on every construction/DDP rank."""
    torch.manual_seed(7)
    a = _make_model(fusion_norm=True)
    torch.manual_seed(7)
    b = _make_model(fusion_norm=True)
    torch.testing.assert_close(a.fusion_goal_gain, b.fusion_goal_gain)


def test_argmax_decode_metrics_are_the_deployment_path() -> None:
    """`offset_argmax_recon*` must be the ARGMAX decode -- what inference serves with
    `sample_codes=false` -- and not an alias of the sampled-decode metric.

    The distinction is the point of the metric: on dashing-dream-514 the val code losses
    rose 256% across training while the argmax decode IMPROVED 13%, so selecting on the
    code loss picks the worse checkpoint.
    """
    model = _make_model()
    model.sample_codes = True  # make the sampled path genuinely differ from argmax
    batch = _make_batch()

    metrics = model._compute_metrics(batch)["policy", "metric"]  # noqa: SLF001

    expected_keys = {
        "offset_argmax_recon",
        "offset_argmax_recon_last",
        "code_acc_joint_last",
        *(f"code_acc_{q}_last" for q in range(NUM_QUANTIZERS)),
    }
    assert expected_keys <= set(cast("dict[str, Tensor]", metrics).keys()), (
        f"missing: {expected_keys - set(cast("dict[str, Tensor]", metrics).keys())}"
    )
    for key in expected_keys:
        assert metrics[key].isfinite()

    # accuracies are proportions, and joint correctness cannot exceed any marginal
    marginals = [float(metrics[f"code_acc_{q}_last"]) for q in range(NUM_QUANTIZERS)]
    joint = float(metrics["code_acc_joint_last"])
    for acc in [*marginals, joint]:
        assert 0.0 <= acc <= 1.0
    assert joint <= min(marginals) + 1e-6

    _assert_argmax_decode_matches(model, batch, metrics)


def _assert_argmax_decode_matches(
    model: PatchPolicy, batch: dict[str, Any], metrics: TensorDict
) -> None:
    """Recompute the argmax decode independently -- it must match exactly."""
    with torch.no_grad():
        features, chunk = model._features(batch)  # noqa: SLF001
        target = model.tokenizer._normalize(chunk.flatten(-2, -1))  # noqa: SLF001
        code_logits, offsets = model._heads(features)  # noqa: SLF001
        codes = code_logits.argmax(dim=-1)
        decoded = model.tokenizer.invert(codes) + model._offset(offsets, codes)  # noqa: SLF001
        correct = codes[:, -1] == model.tokenizer(chunk)[:, -1]

        torch.testing.assert_close(
            metrics["offset_argmax_recon_last"],
            model.losses["offset"](decoded[:, -1], target[:, -1]),
        )
        torch.testing.assert_close(
            metrics["offset_argmax_recon"], model.losses["offset"](decoded, target)
        )

    # the accuracies must be the argmax hit-rate against the tokenized targets
    for q in range(NUM_QUANTIZERS):
        torch.testing.assert_close(
            metrics[f"code_acc_{q}_last"], correct[:, q].float().mean()
        )


def test_readout_is_the_last_patch_token_not_the_speed_token() -> None:
    """The speed token is PREPENDED, so each frame block must END on a patch token.

    `test_readout_is_causally_valid` cannot catch a readout moved to index 0: the
    speed token attends bidirectionally within its frame, so it changes in
    lockstep with the patch tokens. This pins the index arithmetic instead.
    """
    model = _make_model()
    batch = _make_batch()
    tokens_per_frame = NUM_PATCHES + 1

    captured: dict[str, Tensor] = {}
    encoder_forward = model.encoder.forward

    def _capture(src: Tensor, *, num_frames: int) -> Tensor:
        out = encoder_forward(src, num_frames=num_frames)
        captured["embedding"] = out
        return out

    model.encoder.forward = _capture  # type: ignore[method-assign]
    features, _ = model._features(batch)  # noqa: SLF001
    model.encoder.forward = encoder_forward  # type: ignore[method-assign]

    embedding = captured["embedding"]  # (b, t * k, d), pre-norm
    assert embedding.shape[1] == EPISODE_LENGTH * tokens_per_frame

    # the readout of frame t must be flat index t*(P+1) + P, i.e. its LAST token
    for t in range(EPISODE_LENGTH):
        expected = embedding[:, t * tokens_per_frame + NUM_PATCHES]
        if model.norm is not None:
            expected = model.norm(expected)
        torch.testing.assert_close(features[:, t], expected)

    # and NOT the frame's first token (the speed token)
    speed_token = embedding[:, 0 * tokens_per_frame]
    if model.norm is not None:
        speed_token = model.norm(speed_token)
    assert not torch.allclose(features[:, 0], speed_token)


def test_teacher_forcing_routes_the_offset_loss_through_ground_truth_codes() -> None:
    """`_compute_metrics` must gather offsets at GROUND-TRUTH codes when
    `teacher_force_offset=True`.

    The existing gradient test builds `predicted` by hand, so it would still pass
    if `_compute_metrics` silently used the sampled codes. This asserts on the
    loss value produced by `_compute_metrics` itself.
    """
    torch.manual_seed(0)
    model = _make_model(teacher_force_offset=True)
    model.sample_codes = True  # make the sampled path genuinely differ from argmax
    batch = _make_batch()

    offset_loss = model._compute_metrics(batch)["policy", "loss", "offset"]  # noqa: SLF001

    with torch.no_grad():
        features, chunk = model._features(batch)  # noqa: SLF001
        target = model.tokenizer._normalize(chunk.flatten(-2, -1))  # noqa: SLF001
        _, offsets = model._heads(features)  # noqa: SLF001
        target_codes = model.tokenizer(chunk)
        teacher_chunk = model.tokenizer.invert(target_codes) + model._offset(  # noqa: SLF001
            offsets, target_codes
        )
        expected = model.losses["offset"](teacher_chunk, target)

    torch.testing.assert_close(offset_loss, expected)

    # the same model with teacher forcing OFF must NOT produce the teacher value
    # (guards against the flag being ignored entirely)
    torch.manual_seed(0)
    free = _make_model(teacher_force_offset=False)
    free.load_state_dict(model.state_dict())
    free.sample_codes = True
    free_loss = free._compute_metrics(batch)["policy", "loss", "offset"]  # noqa: SLF001
    assert not torch.allclose(free_loss, expected)


# --------------------------------------------------------------------------- #
# dedicated readout + register tokens (opt-in)
# --------------------------------------------------------------------------- #

NUM_REGISTERS = 2
READOUT_TOKENS_PER_FRAME = NUM_PATCHES + 1 + NUM_REGISTERS + 1


def _frame_inputs(batch: dict) -> tuple[Tensor, Tensor, Tensor]:
    return (
        batch["image"]["cam_front_left"],
        batch["continuous"]["speed"],
        batch["context"]["waypoints"],
    )


def test_readout_and_register_token_layout() -> None:
    """[speed, patches..., register_0, register_1, READOUT]: the readout is LAST
    (so `[:, :, -1]` picks the learned token) and the registers sit just before
    it, never at the readout position.
    """
    model = _make_model(use_readout_token=True, num_register_tokens=NUM_REGISTERS)
    tokens = model._frame_tokens(*_frame_inputs(_make_batch()))  # noqa: SLF001

    assert tokens.shape == (
        BATCH_SIZE,
        EPISODE_LENGTH,
        READOUT_TOKENS_PER_FRAME,
        POLICY_DIM,
    )
    assert model.readout_token is not None
    assert model.register_tokens is not None
    torch.testing.assert_close(
        tokens[:, :, -1],
        model.readout_token.reshape(1, 1, -1).expand(BATCH_SIZE, EPISODE_LENGTH, -1),
    )
    torch.testing.assert_close(
        tokens[:, :, -(NUM_REGISTERS + 1) : -1],
        model.register_tokens.reshape(1, 1, NUM_REGISTERS, -1).expand(
            BATCH_SIZE, EPISODE_LENGTH, -1, -1
        ),
    )

    # hparams round-trip (load_for_export / continuation reload from these)
    assert model.hparams["use_readout_token"] is True
    assert model.hparams["num_register_tokens"] == NUM_REGISTERS


def test_readout_token_default_off_preserves_current_layout() -> None:
    """Default-off: existing arms keep the last-image-patch readout, no new
    parameters, identical token block."""
    model = _make_model()
    assert model.readout_token is None
    assert model.register_tokens is None
    assert model.hparams["use_readout_token"] is False

    tokens = model._frame_tokens(*_frame_inputs(_make_batch()))  # noqa: SLF001
    assert tokens.shape == (BATCH_SIZE, EPISODE_LENGTH, NUM_PATCHES + 1, POLICY_DIM)
    assert not any(
        name in {"readout_token", "register_tokens"}
        for name, _ in model.named_parameters()
    )


def test_register_tokens_without_readout_are_rejected() -> None:
    """A register at `[:, :, -1]` would be read from, which registers must never
    be -- constructor-time error, not a silent mis-readout."""
    with pytest.raises(ValueError, match="use_readout_token"):
        _make_model(num_register_tokens=1)


def test_readout_and_register_tokens_receive_gradient() -> None:
    """The readout token feeds the heads directly; the registers feed them via
    attention (K/V) -- both must train."""
    model = _make_model(use_readout_token=True, num_register_tokens=NUM_REGISTERS)
    model.train()
    loss = cast(  # noqa: SLF001
        "TensorDict", model._compute_metrics(_make_batch())["policy", "loss"]
    ).sum(reduce=True)
    loss.backward()
    assert model.readout_token is not None
    assert model.register_tokens is not None
    assert model.readout_token.grad is not None
    assert model.readout_token.grad.abs().sum() > 0
    assert model.register_tokens.grad is not None
    assert model.register_tokens.grad.abs().sum() > 0


def test_readout_token_metrics_and_losses_finite() -> None:
    # both decode modes: the sampled metrics only exist under sample_codes=True,
    # so assert finiteness over whatever the mode actually emits.
    for sample_codes in (False, True):
        model = _make_model(
            use_readout_token=True,
            num_register_tokens=NUM_REGISTERS,
            sample_codes=sample_codes,
        )
        metrics = model._compute_metrics(_make_batch())  # noqa: SLF001
        for value in metrics["policy", "loss"].values():
            assert value.isfinite()
        for value in metrics["policy", "metric"].values():
            assert value.isfinite()
        assert ("offset_sampled_recon" in metrics["policy", "metric"]) is sample_codes


def test_token_norm_tracking() -> None:
    """B2: per-token-type norms for the quality metrics -- patch/speed/goal always,
    register/readout only when the opt-in layout is on."""
    batch = _make_batch()

    norms: dict[str, Tensor] = {}
    model = _make_model(use_readout_token=True, num_register_tokens=NUM_REGISTERS)
    model._compute_metrics(batch, token_norms=norms)  # noqa: SLF001
    assert set(norms) == {"speed", "patch", "goal", "register", "readout"}
    for name, value in norms.items():
        assert value.isfinite(), name
        assert float(value) > 0, name
        assert not value.requires_grad, name

    off_norms: dict[str, Tensor] = {}
    off = _make_model()
    off._compute_metrics(batch, token_norms=off_norms)  # noqa: SLF001
    assert set(off_norms) == {"speed", "patch", "goal"}


def test_encoder_tokens_per_frame_mismatch_raises() -> None:
    """Enabling the readout layout without re-gearing the trunk must fail loudly
    at the first forward, not diverge at serving time."""
    trunk = CausalFrameTransformer(
        dim_model=POLICY_DIM,
        num_layers=1,
        num_heads=2,
        tokens_per_frame=NUM_PATCHES + 1,  # stale: misses registers + readout
        window=2,
    )
    model = _make_model(
        use_readout_token=True, num_register_tokens=NUM_REGISTERS, encoder=trunk
    )
    with pytest.raises(ValueError, match="tokens_per_frame"):
        model._features(_make_batch())  # noqa: SLF001


def test_readout_token_streaming_matches_full_forward() -> None:  # noqa: PLR0914
    """KV-cache gate for the widened frame: streaming one widened frame block
    per tick against a ring of `window - 1` frames equals the full windowed
    forward at every frame's readout -- the same equivalence
    tests/test_causal_frame.py gates at k=17/257, here through the actual
    `PatchPolicy` token pipeline with registers + readout appended.
    """
    window = 2
    trunk = CausalFrameTransformer(
        dim_model=POLICY_DIM,
        num_layers=2,
        num_heads=2,
        tokens_per_frame=READOUT_TOKENS_PER_FRAME,
        window=window,
    )
    model = _make_model(
        use_readout_token=True, num_register_tokens=NUM_REGISTERS, encoder=trunk
    )
    batch = _make_batch()

    with torch.no_grad():
        features, _ = model._features(batch)  # noqa: SLF001  # full windowed forward

        tokens = model._frame_tokens(*_frame_inputs(batch))  # noqa: SLF001
        k = READOUT_TOKENS_PER_FRAME
        past_k, past_v, bias = trunk.empty_cache(
            batch_size=BATCH_SIZE, cache_frames=window - 1
        )
        readouts = []
        for t in range(EPISODE_LENGTH):
            cos, sin = frame_rope_cos_sin(
                torch.tensor(t), head_dim=trunk.head_dim, base=trunk.rope_base
            )
            out, new_k, new_v = trunk.step(
                tokens[:, t],
                past_k=past_k,
                past_v=past_v,
                cos=cos,
                sin=sin,
                cache_bias=bias,
            )
            readouts.append(out[:, -1])
            past_k = torch.cat((past_k[..., k:, :], new_k), dim=-2)
            past_v = torch.cat((past_v[..., k:, :], new_v), dim=-2)
            bias = torch.cat((bias[..., k:], torch.zeros_like(bias[..., :k])), dim=-1)
        streamed = torch.stack(readouts, dim=1)
        if model.norm is not None:
            streamed = model.norm(streamed)

    torch.testing.assert_close(streamed, features, rtol=0, atol=1e-5)
