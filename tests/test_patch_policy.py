"""Unit tests for `PatchPolicy` (https://arxiv.org/pdf/2607.18236).

Covers the block-causal attention mask, per-frame readout/loss shapes, the
teacher-forced offset semantics inherited from `JointPolicyObjective` (#258),
and equivalence of the batched `_gather_offset` with the joint-policy original.
"""

from typing import Any, override

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
AUX_SEG_CLASSES = 7
AUX_GRID = (2, 2)  # H * W must equal NUM_PATCHES


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


def _make_model(
    *,
    teacher_force_offset: bool = True,
    fusion_norm: bool = False,
    aux_heads: ModuleDict | None = None,
    aux_weights: dict[str, float] | None = None,
    aux_purity_min: float = 0.6,
) -> PatchPolicy:
    return PatchPolicy(
        fusion_norm=fusion_norm,
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
        aux_heads=aux_heads,
        aux_weights=aux_weights,
        aux_purity_min=aux_purity_min,
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


def _make_aux_heads() -> ModuleDict:
    return ModuleDict(
        modules={
            "segmentation": Linear(POLICY_DIM, AUX_SEG_CLASSES),
            "motion": Linear(POLICY_DIM, 1),
        }
    )


def _make_lfg_labels(generator: torch.Generator) -> Tensor:
    """`(b, t, 4, *AUX_GRID)` uint8, matching `rmind.utils.lfg_labels.decode_lfg_label`'s
    layout (seg_label, seg_purity, motion, confidence)."""
    seg_label = torch.randint(
        0,
        AUX_SEG_CLASSES,
        (BATCH_SIZE, EPISODE_LENGTH, 1, *AUX_GRID),
        generator=generator,
    )
    purity_motion_conf = torch.randint(
        0, 256, (BATCH_SIZE, EPISODE_LENGTH, 3, *AUX_GRID), generator=generator
    )
    return torch.cat([seg_label, purity_motion_conf], dim=2).to(torch.uint8)


def _make_batch(*, with_lfg: bool = False) -> dict:
    generator = torch.Generator().manual_seed(0)
    chunk = torch.rand(
        (BATCH_SIZE, EPISODE_LENGTH, ACTION_HORIZON, ACTION_FIELDS), generator=generator
    )
    chunk[..., 3] = torch.randint(
        0, 3, chunk[..., 3].shape, generator=generator
    ).float()
    batch = {
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
    if with_lfg:
        batch["context"]["lfg"] = _make_lfg_labels(generator)
    return batch


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

    features, _blocks, chunk = model._features(batch)  # noqa: SLF001
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

    features, _blocks, chunk = model._features(batch)  # noqa: SLF001
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

    features, _blocks, _ = model._features(batch)  # noqa: SLF001

    perturbed = _make_batch()
    perturbed["image"]["cam_front_left"] = batch["image"]["cam_front_left"].clone()
    perturbed["continuous"]["speed"] = batch["continuous"]["speed"].clone()
    perturbed["context"]["waypoints"] = batch["context"]["waypoints"].clone()
    perturbed["joint_actions"] = batch["joint_actions"].clone()
    # change only the LAST frame's observations
    perturbed["image"]["cam_front_left"][:, -1] += 1.0
    perturbed["continuous"]["speed"][:, -1] = 100.0
    perturbed["context"]["waypoints"][:, -1] += 1.0

    features_perturbed, _blocks, _ = model._features(perturbed)  # noqa: SLF001

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
    loss = model._compute_metrics(batch)["policy", "loss"].sum(reduce=True)  # noqa: SLF001
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
    assert expected_keys <= set(metrics.keys()), (
        f"missing: {expected_keys - set(metrics.keys())}"
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
        features, _blocks, chunk = model._features(batch)  # noqa: SLF001
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
    features, _blocks, _ = model._features(batch)  # noqa: SLF001
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
        features, _blocks, chunk = model._features(batch)  # noqa: SLF001
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


# --- LFG auxiliary supervision (lfg_aux_supervision_task.md, stage 3) ---


def test_aux_heads_absent_by_default() -> None:
    """No `aux_heads` -> no `"aux"` group at all, and the trunk output block's
    non-readout positions are simply unused (as before this stage)."""
    model = _make_model()
    metrics = model._compute_metrics(_make_batch())  # noqa: SLF001
    assert "aux" not in metrics.keys()  # noqa: SIM118


def test_aux_weights_missing_entry_raises() -> None:
    """`aux_weights` must cover every `aux_heads` key -- a silently-unweighted
    (i.e. weight=1) aux term would be an easy way to blow up the total loss."""
    with pytest.raises(ValueError, match="aux_weights"):
        _make_model(aux_heads=_make_aux_heads(), aux_weights={"segmentation": 0.1})


def test_aux_metrics_shapes_and_weighting() -> None:
    model = _make_model(
        aux_heads=_make_aux_heads(),
        aux_weights={"segmentation": 0.1, "motion": 0.3},
        aux_purity_min=0.0,  # keep every patch supervised for this shape/weight check
    )
    batch = _make_batch(with_lfg=True)

    metrics = model._compute_metrics(batch)  # noqa: SLF001
    aux_losses = metrics["aux", "loss"]
    aux_metrics = metrics["aux", "metric"]

    assert set(aux_losses.keys()) == {"segmentation", "motion"}
    for value in aux_losses.values():
        assert value.isfinite()
        assert value.ndim == 0

    for key in ("segmentation_acc", "motion_mae", "supervised_fraction"):
        assert aux_metrics[key].isfinite()
    torch.testing.assert_close(  # aux_purity_min=0.0
        aux_metrics["supervised_fraction"], torch.tensor(1.0)
    )


def test_aux_weight_scaling_matches_unweighted_terms() -> None:
    """Recompute the aux losses with `aux_weights=1` and confirm the weighted
    model's logged losses equal `weight * unweighted_term` -- `_step` sums
    `loss/*` unweighted, so the weighting MUST happen inside `_aux_metrics`."""
    torch.manual_seed(0)
    weights = {"segmentation": 0.1, "motion": 0.3}
    weighted = _make_model(
        aux_heads=_make_aux_heads(), aux_weights=weights, aux_purity_min=0.0
    )
    torch.manual_seed(0)
    unweighted = _make_model(
        aux_heads=_make_aux_heads(),
        aux_weights={"segmentation": 1.0, "motion": 1.0},
        aux_purity_min=0.0,
    )
    unweighted.load_state_dict(weighted.state_dict())

    batch = _make_batch(with_lfg=True)
    weighted_losses = weighted._compute_metrics(batch)["aux", "loss"]  # noqa: SLF001
    unweighted_losses = unweighted._compute_metrics(batch)["aux", "loss"]  # noqa: SLF001

    for key, weight in weights.items():
        torch.testing.assert_close(
            weighted_losses[key], weight * unweighted_losses[key]
        )


def test_aux_purity_min_masks_low_purity_patches() -> None:
    """Patches below `aux_purity_min` must be dropped entirely -- confirmed by
    checking `supervised_fraction`, not just that the loss changed."""
    model = _make_model(
        aux_heads=_make_aux_heads(),
        aux_weights={"segmentation": 1.0, "motion": 1.0},
        aux_purity_min=0.5,
    )
    batch = _make_batch(with_lfg=True)
    labels = batch["context"]["lfg"]

    # force half the patches (by flat index within the grid) below/above the
    # purity threshold, deterministically
    purity = labels[:, :, 1].flatten(-2).clone()
    low = torch.zeros_like(purity, dtype=torch.bool)
    low[..., : low.shape[-1] // 2] = True
    purity_flat = torch.where(
        low, torch.tensor(50, dtype=torch.uint8), torch.tensor(200, dtype=torch.uint8)
    )
    labels[:, :, 1] = purity_flat.reshape(labels[:, :, 1].shape)
    # confidence must be > 0 for the surviving patches to actually contribute
    labels[:, :, 3] = 255

    metrics = model._compute_metrics(batch)  # noqa: SLF001
    expected_fraction = 1.0 - (low.float().mean().item())
    torch.testing.assert_close(
        metrics["aux", "metric", "supervised_fraction"], torch.tensor(expected_fraction)
    )


def test_aux_zero_confidence_masks_patches() -> None:
    """`confidence=0` must zero a patch's contribution even at full purity."""
    model = _make_model(
        aux_heads=_make_aux_heads(),
        aux_weights={"segmentation": 1.0, "motion": 1.0},
        aux_purity_min=0.0,
    )
    batch = _make_batch(with_lfg=True)
    batch["context"]["lfg"][:, :, 3] = 0  # confidence

    metrics = model._compute_metrics(batch)  # noqa: SLF001
    torch.testing.assert_close(
        metrics["aux", "metric", "supervised_fraction"], torch.tensor(0.0)
    )
    for key in ("segmentation", "motion"):
        torch.testing.assert_close(metrics["aux", "loss", key], torch.tensor(0.0))


def test_aux_gradients_reach_trunk_not_frozen_modules() -> None:
    """Aux gradients must flow into the trunk / aux heads and NOT into the
    permanently-frozen `image_encoder`, `goal_encoder`, `tokenizer` (brief §5.4)."""
    model = _make_model(
        aux_heads=_make_aux_heads(),
        aux_weights={"segmentation": 0.1, "motion": 0.1},
        aux_purity_min=0.0,
    )
    batch = _make_batch(with_lfg=True)

    metrics = model._compute_metrics(batch)  # noqa: SLF001
    aux_loss = metrics["aux", "loss"].sum(reduce=True)
    aux_loss.backward()

    assert model.aux_heads is not None
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.aux_heads.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    for frozen in (model.tokenizer, model.goal_encoder):
        assert all(p.grad is None for p in frozen.parameters())
    # image_encoder is Identity() in this test (no parameters) -- assert the
    # no_grad contract at the source instead
    assert all(not p.requires_grad for p in model.image_encoder.parameters())


def test_aux_loss_does_not_perturb_policy_metrics() -> None:
    """A `PatchPolicy` with `aux_heads` attached must reproduce the exact same
    `policy` loss/metric values as one without -- the aux branch is additive and
    must not perturb `_encode`'s shared computation (brief §7.3's premise)."""
    torch.manual_seed(0)
    plain = _make_model()
    torch.manual_seed(0)
    with_aux = _make_model(
        aux_heads=_make_aux_heads(), aux_weights={"segmentation": 0.0, "motion": 0.0}
    )
    with_aux.load_state_dict(plain.state_dict(), strict=False)

    batch = _make_batch(with_lfg=True)
    plain_metrics = plain._compute_metrics(batch)  # noqa: SLF001
    aux_metrics = with_aux._compute_metrics(batch)  # noqa: SLF001

    torch.testing.assert_close(
        plain_metrics["policy"].to_dict(), aux_metrics["policy"].to_dict()
    )
