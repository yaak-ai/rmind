"""Causality, `side_valid` masking and conditioning flow for `NeroPatchPolicy`.

The three properties that a shape test would pass vacuously, and that this file
makes falsifiable:

* **causality of the NEW token layout** -- a frame's readout must not move when a
  LATER frame's pixels change. The layout differs from PR #265's (769 tokens per
  frame across 3 cameras, state token first instead of a speed token), so the
  trunk's guarantee has to be re-checked against this packing, not assumed.
* **`side_valid`** -- perturbing an invalid side's state must leave the output
  BIT-IDENTICAL, and the invalid side must contribute no loss rows. Note the
  converse is deliberately NOT asserted: the state token is shared, so a valid
  side's prediction legitimately depends on both sides' inputs.
* **camera conditioning** -- `camera_cond` must actually reach the trunk. A
  shape-only test passes even if the tensor is dropped on the floor.
"""

from pathlib import Path
from typing import Any, override

import pytest
import torch
from torch import Tensor, nn

from rmind.components.containers import ModuleDict
from rmind.components.image import LetterboxResize
from rmind.components.loss import FocalLoss
from rmind.components.transformer.causal_frame import CausalFrameTransformer
from rmind.components.vq import ResidualVQ
from rmind.data.nero import (
    CAMERA_COND_DIM,
    NUM_SIDES,
    SIDE_DIM,
    STATE_QUAT_DIM,
    DisparityStandardizer,
)
from rmind.datamodules.nero_random import CAMERA_NAMES, nero_random_batch
from rmind.models.nero_patch_policy import NeroPatchPolicy
from rmind.models.nero_pose_tokenizer import NeroPoseTokenizer

PATCH_GRID = (2, 3)  # 6 patches per camera, standing in for the real 10x16 = 160
NUM_PATCHES = PATCH_GRID[0] * PATCH_GRID[1]
# the three cameras arrive on DIFFERENT grids (rbyte isotropic downscale); the
# model's LetterboxResize is what unifies them, so the test uses one too
TEST_GRIDS = {"base": (9, 16), "side_left": (10, 16), "side_right": (10, 16)}
UNIFIED = (10, 16)
IMAGE_DIM = 8
POLICY_DIM = 16
NUM_HEADS = 2  # head_dim = 8: divisible by 8 (fused SDPA) and even (RoPE)
NUM_LAYERS = 2
HORIZON = 2
EPISODE_LENGTH = 3
CODEBOOK_SIZE = 4
NUM_QUANTIZERS = 2


class _Unify(nn.Module):
    """The real pipeline minus the ImageNet normalisation: letterbox, then scale.

    The letterbox is not decoration here -- `base` and `side_*` arrive on
    different grids, so without it the per-camera patch tensors cannot be
    concatenated at all.
    """

    def __init__(self) -> None:
        super().__init__()
        self.letterbox = LetterboxResize(size=UNIFIED)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.letterbox(x).float() / 255.0


class _TinyImageEncoder(nn.Module):
    """Deterministic stand-in for the frozen ViT: `(..., 3, H, W)` -> `(..., P, D)`."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, IMAGE_DIM)

    @override
    def forward(self, x: Tensor) -> Tensor:
        *batch, c, h, w = x.shape
        pooled = nn.functional.adaptive_avg_pool2d(x.reshape(-1, c, h, w), PATCH_GRID)
        pooled = pooled.flatten(-2, -1).transpose(-2, -1)  # (n, P, 3)
        return self.projection(pooled).reshape(*batch, NUM_PATCHES, IMAGE_DIM)


def _tokenizer() -> NeroPoseTokenizer:
    action_dim = HORIZON * SIDE_DIM
    return NeroPoseTokenizer(
        encoder=nn.Sequential(nn.Linear(action_dim, 32), nn.GELU(), nn.Linear(32, 16)),
        # kmeans_init=False so the codebook is meaningfully initialised without a
        # data-dependent first forward (the repo guards that on `training`)
        quantizer=ResidualVQ(
            dim=16,
            codebook_size=CODEBOOK_SIZE,
            num_quantizers=NUM_QUANTIZERS,
            kmeans_init=False,
        ),
        decoder=nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, action_dim)),
        action_features=SIDE_DIM,
        action_horizon=HORIZON,
    )


def _policy(**kwargs: Any) -> NeroPatchPolicy:
    torch.manual_seed(0)
    action_dim = HORIZON * SIDE_DIM
    policy = NeroPatchPolicy(
        image_transform=_Unify(),
        image_encoder=_TinyImageEncoder(),
        patch_projection=nn.Linear(2 * IMAGE_DIM + CAMERA_COND_DIM, POLICY_DIM),
        state_embedding=nn.Linear(NUM_SIDES * SIDE_DIM + NUM_SIDES, POLICY_DIM),
        encoder=CausalFrameTransformer(
            dim_model=POLICY_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            tokens_per_frame=len(CAMERA_NAMES) * NUM_PATCHES + 1,
            window=EPISODE_LENGTH,
        ),
        tokenizer=_tokenizer(),
        code_head=nn.Linear(POLICY_DIM, NUM_QUANTIZERS * CODEBOOK_SIZE),
        offset_head=nn.Linear(POLICY_DIM, NUM_QUANTIZERS * CODEBOOK_SIZE * action_dim),
        losses=ModuleDict(modules={"code": FocalLoss(), "offset": nn.L1Loss()}),
        image_embedding_dim=IMAGE_DIM,
        policy_embedding_dim=POLICY_DIM,
        sample_codes=False,  # determinism
        **({"goal_dropout": 0.0} | kwargs),
    )
    return policy.eval()  # no dropout, no goal dropout


def _batch(*, both_sides: bool = True, seed: int = 0) -> dict[str, Any]:
    # ⚠️ `seed` only seeds the POSE generator. The images, `camera_cond` and the
    # depth stream are drawn from the GLOBAL torch RNG, so two calls with the
    # same `seed` do NOT agree unless the global stream is pinned too. That is
    # pre-existing behaviour of `nero_random_batch`, not something depth
    # introduced -- but any test that compares two batches depends on it.
    torch.manual_seed(seed)
    return nero_random_batch(
        batch_size=2,
        episode_length=EPISODE_LENGTH,
        action_horizon=HORIZON,
        grids=TEST_GRIDS,
        both_sides=both_sides,
        seed=seed,
    )


# ---------------------------------------------------------------- shape flow


def test_token_layout_and_readout_shapes() -> None:
    policy = _policy()
    batch = _batch()
    tokens = policy._frame_tokens(batch)  # noqa: SLF001
    assert tokens.shape == (
        2,
        EPISODE_LENGTH,
        len(CAMERA_NAMES) * NUM_PATCHES + 1,
        POLICY_DIM,
    )

    features = policy._features(batch)  # noqa: SLF001
    assert features.shape == (2, EPISODE_LENGTH, POLICY_DIM)

    out = policy(batch)["policy", "action"]
    assert out.shape == (2, NUM_SIDES, HORIZON, SIDE_DIM)


def test_loader_shapes_are_what_rbyte_actually_emits() -> None:
    """Storage form (46/side), per-camera grids, per-camera goal keys."""
    batch = _batch()
    assert batch["state.pose"].shape == (2, EPISODE_LENGTH, NUM_SIDES, STATE_QUAT_DIM)
    assert batch["action.future_state"].shape == (
        2,
        EPISODE_LENGTH,
        HORIZON,
        NUM_SIDES,
        STATE_QUAT_DIM,
    )
    assert batch["camera_cond"].shape == (2, len(CAMERA_NAMES), CAMERA_COND_DIM)
    assert batch["side_valid"].shape == (2, NUM_SIDES)
    for camera in CAMERA_NAMES:
        h, w = TEST_GRIDS[camera]
        assert batch[f"image.{camera}"].shape == (2, EPISODE_LENGTH, 3, h, w)
        # three SEPARATE goal keys, each on its own grid
        assert batch[f"goal.image.{camera}"].shape == (2, 3, h, w)
    # the cameras really are on different grids -- otherwise the letterbox path
    # this test exercises would be vacuous
    assert len({TEST_GRIDS[c] for c in CAMERA_NAMES}) > 1
    # §6.2: action(t) really is the future state chunk [t+1 .. t+H]
    assert torch.allclose(
        batch["action.future_state"][:, 0, 0], batch["state.pose"][:, 1], atol=1e-6
    )
    # §6.2 alias, materialised by rbyte as a byte-identical duplicate
    assert torch.equal(batch["action.commanded"], batch["action.future_state"])


def test_storage_form_is_expanded_to_the_model_facing_9d_form() -> None:
    policy = _policy()
    batch = _batch()
    assert policy._state(batch).shape == (  # noqa: SLF001
        2,
        EPISODE_LENGTH,
        NUM_SIDES,
        SIDE_DIM,
    )
    assert policy._chunk(batch).shape == (  # noqa: SLF001
        2,
        EPISODE_LENGTH,
        HORIZON,
        NUM_SIDES,
        SIDE_DIM,
    )


# ----------------------------------------------------------------- causality


def test_future_frames_do_not_change_earlier_readouts() -> None:
    """Causality of the NEW 3-camera / state-token layout, not the trunk's alone."""
    policy = _policy()
    batch = _batch()
    baseline = policy._features(batch)  # noqa: SLF001

    perturbed = dict(batch)
    images = batch["image.side_right"].clone()
    images[:, -1] = torch.randint_like(images[:, -1], 0, 256)  # newest frame only
    perturbed["image.side_right"] = images
    after = policy._features(perturbed)  # noqa: SLF001

    assert torch.allclose(baseline[:, :-1], after[:, :-1], atol=1e-6)
    # falsifiability: the perturbation must actually matter SOMEWHERE
    assert not torch.allclose(baseline[:, -1], after[:, -1], atol=1e-4)


def test_state_of_a_future_frame_does_not_change_earlier_readouts() -> None:
    policy = _policy()
    batch = _batch()
    baseline = policy._features(batch)  # noqa: SLF001

    perturbed = dict(batch)
    state = batch["state.pose"].clone()
    state[:, -1] += 1.0
    perturbed["state.pose"] = state
    after = policy._features(perturbed)  # noqa: SLF001

    assert torch.allclose(baseline[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(baseline[:, -1], after[:, -1], atol=1e-4)


# ---------------------------------------------------------------- side_valid


def test_invalid_side_state_cannot_influence_the_output() -> None:
    """The discriminating `side_valid` test: bit-identical, not merely 'it runs'."""
    policy = _policy()
    batch = _batch(both_sides=False)  # left absent, as in the dummy recording
    assert not bool(batch["side_valid"][0, 0])

    baseline = policy._features(batch)  # noqa: SLF001

    perturbed = dict(batch)
    state = batch["state.pose"].clone()
    state[:, :, 0] = torch.randn_like(state[:, :, 0]) * 10  # the INVALID side
    perturbed["state.pose"] = state
    assert torch.equal(policy._features(perturbed), baseline)  # noqa: SLF001

    # falsifiable control: the VALID side does influence the output
    control = dict(batch)
    state = batch["state.pose"].clone()
    state[:, :, 1] = torch.randn_like(state[:, :, 1]) * 10
    control["state.pose"] = state
    assert not torch.allclose(policy._features(control), baseline, atol=1e-4)  # noqa: SLF001


def test_invalid_side_contributes_no_loss_rows() -> None:
    policy = _policy()
    batch = _batch(both_sides=False)
    metrics = policy._compute_metrics(batch)  # noqa: SLF001
    # 2 samples x 3 frames x 1 valid side
    assert metrics["policy", "metric", "valid_rows"].item() == 2 * EPISODE_LENGTH * 1

    both = policy._compute_metrics(_batch(both_sides=True))  # noqa: SLF001
    assert (
        both["policy", "metric", "valid_rows"].item() == 2 * EPISODE_LENGTH * NUM_SIDES
    )


def test_invalid_side_action_values_do_not_move_the_loss() -> None:
    """`sum/count` normalisation: garbage in the masked-out rows must be inert."""
    policy = _policy()
    batch = _batch(both_sides=False)
    baseline = policy._compute_metrics(batch)["policy", "loss"].sum(reduce=True)  # noqa: SLF001

    poisoned = dict(batch)
    action = batch["action.future_state"].clone()
    action[:, :, :, 0] = 1e3  # the invalid side's target
    poisoned["action.future_state"] = action
    assert torch.allclose(
        policy._compute_metrics(poisoned)["policy", "loss"].sum(reduce=True),  # noqa: SLF001
        baseline,
        atol=1e-6,
    )


# ------------------------------------------------------ conditioning reaches


def test_camera_conditioning_reaches_the_trunk() -> None:
    policy = _policy()
    batch = _batch()
    baseline = policy._features(batch)  # noqa: SLF001

    perturbed = dict(batch)
    perturbed["camera_cond"] = batch["camera_cond"] + 1.0
    assert not torch.allclose(policy._features(perturbed), baseline, atol=1e-4)  # noqa: SLF001


def test_per_camera_conditioning_is_bound_to_that_camera() -> None:
    """Conditioning is concatenated PER CAMERA, so swapping two cameras' vectors
    must change the output -- an implementation that broadcast one shared vector
    to all patches would pass a naive 'it changes something' test.
    """
    policy = _policy()
    batch = _batch()
    baseline = policy._features(batch)  # noqa: SLF001

    swapped = dict(batch)
    cond = batch["camera_cond"].clone()
    cond[:, [1, 2]] = cond[:, [2, 1]]
    swapped["camera_cond"] = cond
    assert not torch.allclose(policy._features(swapped), baseline, atol=1e-4)  # noqa: SLF001


def test_goal_image_reaches_the_trunk_and_can_be_disabled() -> None:
    policy = _policy()
    batch = _batch()
    baseline = policy._features(batch)  # noqa: SLF001

    perturbed = dict(batch)
    for camera in CAMERA_NAMES:
        key = f"goal.image.{camera}"
        perturbed[key] = torch.randint_like(batch[key], 0, 256)
    assert not torch.allclose(policy._features(perturbed), baseline, atol=1e-4)  # noqa: SLF001


def test_goal_dropout_replaces_the_goal_with_a_learned_embedding() -> None:
    """Dropout must be TRAIN-only and must use `no_goal`, not zeros."""
    policy = _policy(goal_dropout=1.0)
    batch = _batch()

    policy.train()
    policy.image_encoder.eval()
    dropped = policy._frame_tokens(batch)  # noqa: SLF001

    policy.eval()
    kept = policy._frame_tokens(batch)  # noqa: SLF001
    assert not torch.allclose(dropped, kept, atol=1e-4)

    # with every goal dropped the goal slice is the same learned vector for all
    # cameras and patches, so two different goal images give the SAME tokens
    policy.train()
    policy.image_encoder.eval()
    other = dict(batch)
    for camera in CAMERA_NAMES:
        key = f"goal.image.{camera}"
        other[key] = torch.randint_like(batch[key], 0, 256)
    assert torch.allclose(policy._frame_tokens(other), dropped, atol=1e-6)  # noqa: SLF001


# ------------------------------------------------------------------ training


def test_gradients_flow_to_the_trunk_and_not_to_frozen_modules() -> None:
    policy = _policy()
    policy.train()
    policy.image_encoder.eval()
    loss = policy._compute_metrics(_batch())["policy", "loss"].sum(reduce=True)  # noqa: SLF001
    loss.backward()

    assert policy.encoder.layers[0].attn.in_proj_weight.grad is not None
    assert policy.patch_projection.weight.grad is not None
    assert policy.side_embedding.weight.grad is not None
    assert all(p.grad is None for p in policy.tokenizer.parameters())
    assert all(p.grad is None for p in policy.image_encoder.parameters())


def test_head_is_weight_shared_but_side_distinguishing() -> None:
    policy = _policy()
    features = torch.randn(2, EPISODE_LENGTH, POLICY_DIM)
    per_side = policy._per_side_features(features)  # noqa: SLF001
    assert per_side.shape == (2, EPISODE_LENGTH, NUM_SIDES, POLICY_DIM)
    assert not torch.allclose(per_side[..., 0, :], per_side[..., 1, :])


@pytest.mark.parametrize("action_features", [SIDE_DIM, 12])
def test_action_dimensionality_is_a_config_seam(action_features: int) -> None:
    """Contract §11: swapping to the Revo2 joint-angle space must be config-only."""
    tokenizer = NeroPoseTokenizer(
        encoder=nn.Linear(HORIZON * action_features, 16),
        quantizer=ResidualVQ(
            dim=16,
            codebook_size=CODEBOOK_SIZE,
            num_quantizers=NUM_QUANTIZERS,
            kmeans_init=False,
        ),
        decoder=nn.Linear(16, HORIZON * action_features),
        action_features=action_features,
        action_horizon=HORIZON,
    ).eval()
    assert tokenizer.has_pose_layout == (action_features == SIDE_DIM)

    chunk = torch.randn(5, HORIZON, action_features)
    codes = tokenizer(chunk)
    assert codes.shape == (5, NUM_QUANTIZERS)
    assert tokenizer.invert(codes).shape == (5, HORIZON * action_features)


# ------------------------------------------------------- action-space seam


def _joint_policy(*, action_features: int, state_features: int) -> NeroPatchPolicy:
    """The §13.3 option-A configuration: robot joint commands, ~12 dims per side."""
    torch.manual_seed(0)
    action_dim = HORIZON * action_features
    tokenizer = NeroPoseTokenizer(
        encoder=nn.Linear(action_dim, 16),
        quantizer=ResidualVQ(
            dim=16,
            codebook_size=CODEBOOK_SIZE,
            num_quantizers=NUM_QUANTIZERS,
            kmeans_init=False,
        ),
        decoder=nn.Linear(16, action_dim),
        action_features=action_features,
        action_horizon=HORIZON,
    )
    return NeroPatchPolicy(
        image_transform=_Unify(),
        image_encoder=_TinyImageEncoder(),
        patch_projection=nn.Linear(2 * IMAGE_DIM + CAMERA_COND_DIM, POLICY_DIM),
        state_embedding=nn.Linear(NUM_SIDES * state_features + NUM_SIDES, POLICY_DIM),
        encoder=CausalFrameTransformer(
            dim_model=POLICY_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            tokens_per_frame=len(CAMERA_NAMES) * NUM_PATCHES + 1,
            window=EPISODE_LENGTH,
        ),
        tokenizer=tokenizer,
        code_head=nn.Linear(POLICY_DIM, NUM_QUANTIZERS * CODEBOOK_SIZE),
        offset_head=nn.Linear(POLICY_DIM, NUM_QUANTIZERS * CODEBOOK_SIZE * action_dim),
        losses=ModuleDict(modules={"code": FocalLoss(), "offset": nn.L1Loss()}),
        image_embedding_dim=IMAGE_DIM,
        policy_embedding_dim=POLICY_DIM,
        sample_codes=False,
        goal_dropout=0.0,
    )


def test_joint_angle_action_space_needs_no_code_change() -> None:
    """Contract §10 A1 / §13.3 option A -- LOAD-BEARING, not precautionary.

    The glove SE(3) parameterisation is a stand-in; iteration-1 teleop records
    robot joint values (~12 per side, 24 bimanual). Swapping to that must be a
    config change plus a new tokenizer checkpoint, so this exercises a full
    forward AND backward in the joint-angle space with the SAME model code.
    """
    action_features, state_features = 12, 24
    policy = _joint_policy(
        action_features=action_features, state_features=state_features
    )
    policy.train()
    policy.image_encoder.eval()

    generator = torch.Generator().manual_seed(7)
    batch = dict(_batch())
    batch["state.pose"] = torch.randn(
        2, EPISODE_LENGTH, NUM_SIDES, state_features, generator=generator
    )
    batch["action.future_state"] = torch.randn(
        2, EPISODE_LENGTH, HORIZON, NUM_SIDES, action_features, generator=generator
    )

    # the storage-form conversion must NOT fire for a non-46-dim space
    assert policy._state(batch).shape[-1] == state_features  # noqa: SLF001
    assert policy._chunk(batch).shape[-1] == action_features  # noqa: SLF001
    # ... and the pose-layout metrics must switch themselves off
    assert not policy.tokenizer.has_pose_layout

    loss = policy._compute_metrics(batch)["policy", "loss"].sum(reduce=True)  # noqa: SLF001
    loss.backward()
    assert policy.encoder.layers[0].attn.in_proj_weight.grad is not None
    assert policy.offset_head.weight.grad is not None

    policy.eval()
    out = policy(batch)["policy", "action"]
    assert out.shape == (2, NUM_SIDES, HORIZON, action_features)


def test_head_widths_derive_from_the_action_space_in_config() -> None:
    """The seam is only real if the CONFIG derives the head widths from it."""
    from hydra import compose, initialize_config_dir  # noqa: PLC0415

    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    widths: dict[int, tuple[int, int]] = {}
    for action_features in (60, 12):
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=yaak/nero_arms/causal",
                    f"action_features={action_features}",
                ],
            )
        widths[action_features] = (
            cfg.model.offset_head.hidden_channels[-1],
            cfg.model.tokenizer.decoder.hidden_channels[-1],
        )
        assert widths[action_features] == (
            cfg.num_quantizers
            * cfg.codebook_size
            * cfg.action_horizon
            * action_features,
            cfg.action_horizon * action_features,
        )
    assert widths[60] != widths[12]  # the override actually propagates


# --------------------------------------------------------------- goal_valid


def test_goal_valid_absent_is_backward_compatible() -> None:
    """An existing batch with no `goal_valid` must behave exactly as before."""
    policy = _policy()
    batch = _batch()
    policy.eval()
    with torch.no_grad():
        base = policy(batch)["policy"]["action"]
        explicit = policy(
            dict(batch) | {"goal_valid": torch.ones(2, 3, dtype=torch.bool)}
        )["policy"]["action"]
    assert torch.equal(base, explicit)


def test_no_goal_is_reachable_at_inference() -> None:
    """§22.8: the learned `no_goal` embedding must be usable at serving.

    Before `goal_valid`, the only routes were training-time dropout and a
    model-wide config switch, so a serving app could never say "no goal" -- and
    omitting the frames raised KeyError while a black image was read as a real
    goal.
    """
    policy = _policy()
    batch = _batch()
    policy.eval()
    lean = {k: v for k, v in batch.items() if not k.startswith("goal.image")}
    lean["goal_valid"] = torch.zeros(2, 3, dtype=torch.bool)
    with torch.no_grad():
        goal_free = policy(lean)["policy"]["action"]
        goal_on = policy(batch)["policy"]["action"]
    assert goal_free.shape == goal_on.shape
    assert not torch.equal(goal_free, goal_on)


def test_missing_goal_frames_still_raise_when_claimed() -> None:
    """A missing key with `goal_valid` TRUE is an error, not an implicit no-goal."""
    policy = _policy()
    batch = _batch()
    policy.eval()
    broken = {k: v for k, v in batch.items() if k != "goal.image.base"}
    broken["goal_valid"] = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(KeyError), torch.no_grad():
        policy(broken)


def test_goal_dropout_does_not_mutate_the_loader_flag() -> None:
    """Regression: `present &= keep` would alias and corrupt `batch["goal_valid"]`.

    `.to()` returns *self* when dtype and device already match, so an in-place op
    permanently zeroes the loader's tensor -- and with a cached TensorDict the
    corruption outlives the step. A ruff PLR6104 autofix introduced exactly this
    bug in the depth path.
    """
    policy = _policy()
    batch = _batch()
    policy.train()
    flag = torch.ones(2, 3, dtype=torch.bool)
    batch = dict(batch) | {"goal_valid": flag}
    for _ in range(5):
        _ = policy._goal_features(  # noqa: SLF001
            batch, batch_size=2, device=torch.device("cpu")
        )
    assert bool(flag.all()), "goal dropout mutated the caller's goal_valid tensor"


# --------------------------------------------------------------- depth (§22)
#
# The depth arm is opt-in (§22.6), so the FIRST thing these assert is that it
# does not exist unless asked for. The rest make the four §22 claims falsifiable:
# depth is a fourth CAMERA (not a fourth channel), invalid pixels are masked (not
# zero-filled), absence is handled by a LEARNED embedding (not zeros), and the
# per-sample flag is genuinely per-sample.

#: §22.7: the depth stream's own patch embedding. A 4x12 input at patch 4 is a
#: 1x3 = 3-token grid -- coarser than a camera's 6, as §22.2 asks.
DEPTH_PATCH_SIZE = 4
DEPTH_INPUT = (4, 12)
DEPTH_PATCH_GRID = (
    DEPTH_INPUT[0] // DEPTH_PATCH_SIZE,
    DEPTH_INPUT[1] // DEPTH_PATCH_SIZE,
)
DEPTH_PATCHES = DEPTH_PATCH_GRID[0] * DEPTH_PATCH_GRID[1]
#: 8x24 -> 4x12 is exactly isotropic, as the contract's 400x640 -> 80x128 is
DEPTH_TEST_GRIDS = {"base": (8, 24)}
DEPTH_TOKENS_PER_FRAME = len(CAMERA_NAMES) * NUM_PATCHES + 1 + DEPTH_PATCHES


def _depth_policy(**kwargs: Any) -> NeroPatchPolicy:
    torch.manual_seed(0)
    action_dim = HORIZON * SIDE_DIM
    policy = NeroPatchPolicy(
        image_transform=_Unify(),
        image_encoder=_TinyImageEncoder(),
        patch_projection=nn.Linear(2 * IMAGE_DIM + CAMERA_COND_DIM, POLICY_DIM),
        state_embedding=nn.Linear(NUM_SIDES * SIDE_DIM + NUM_SIDES, POLICY_DIM),
        encoder=CausalFrameTransformer(
            dim_model=POLICY_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            tokens_per_frame=DEPTH_TOKENS_PER_FRAME,
            window=EPISODE_LENGTH,
        ),
        tokenizer=_tokenizer(),
        code_head=nn.Linear(POLICY_DIM, NUM_QUANTIZERS * CODEBOOK_SIZE),
        offset_head=nn.Linear(POLICY_DIM, NUM_QUANTIZERS * CODEBOOK_SIZE * action_dim),
        losses=ModuleDict(modules={"code": FocalLoss(), "offset": nn.L1Loss()}),
        image_embedding_dim=IMAGE_DIM,
        policy_embedding_dim=POLICY_DIM,
        sample_codes=False,
        use_depth=True,
        depth_transform=LetterboxResize(size=DEPTH_INPUT),
        depth_standardizer=DisparityStandardizer(mean=48.0, std=27.4),
        # §22.7: trainable, and 2 input channels (disparity + validity mask)
        depth_patch_embedding=nn.Linear(
            DEPTH_PATCH_SIZE * DEPTH_PATCH_SIZE * 2 + CAMERA_COND_DIM, POLICY_DIM
        ),
        depth_patch_size=DEPTH_PATCH_SIZE,
        depth_patch_grid=DEPTH_PATCH_GRID,
        **({"goal_dropout": 0.0, "depth_dropout": 0.0} | kwargs),
    )
    return policy.eval()


def _depth_batch(*, seed: int = 0, depth_present: float = 1.0) -> dict[str, Any]:
    torch.manual_seed(seed)  # see `_batch`: the global stream must be pinned too
    return nero_random_batch(
        batch_size=2,
        episode_length=EPISODE_LENGTH,
        action_horizon=HORIZON,
        grids=TEST_GRIDS,
        depth=True,
        depth_grids=DEPTH_TEST_GRIDS,
        depth_present=depth_present,
        seed=seed,
    )


def test_depth_off_constructs_nothing_at_all() -> None:
    """§22.6: the opt-in arm must not touch the existing model.

    Not "the depth branch is skipped" -- there must be no depth PARAMETER, no
    depth module, and no extra token. A depth-off model that merely never calls
    the depth path would still have shifted every subsequent init draw.
    """
    policy = _policy()
    assert not policy.use_depth
    assert not hasattr(policy, "no_depth")
    assert policy.depth_transform is None
    assert policy.depth_patch_embedding is None
    assert policy.depth_standardizer is None
    assert not [k for k in policy.state_dict() if "depth" in k]

    tokens = policy._frame_tokens(_batch())  # noqa: SLF001
    assert tokens.shape[-2] == len(CAMERA_NAMES) * NUM_PATCHES + 1


def test_a_depth_off_model_ignores_depth_keys_in_the_batch() -> None:
    """Depth-off must be indifferent to a loader that happens to emit depth."""
    policy = _policy()
    with torch.no_grad():
        plain = policy(_batch(seed=3))["policy", "action"]
        withdepth = policy(_depth_batch(seed=3))["policy", "action"]
    assert torch.equal(plain, withdepth)


def test_depth_is_a_fourth_camera_with_its_own_coarser_token_block() -> None:
    """§22.1/§22.2: separate tokens on a COARSER grid, not a channel on the RGB.

    Also pins the placement: depth sits BETWEEN the state token and the RGB
    patches, so the readout (last token) is still an RGB patch.
    """
    policy = _depth_policy()
    batch = _depth_batch()
    tokens = policy._frame_tokens(batch)  # noqa: SLF001
    assert tokens.shape == (2, EPISODE_LENGTH, DEPTH_TOKENS_PER_FRAME, POLICY_DIM)
    # the depth block is strictly coarser than a camera's block
    assert DEPTH_PATCHES < NUM_PATCHES

    perturbed = dict(batch)
    disparity = batch["disparity.base"].clone()
    disparity[:] = 7
    perturbed["disparity.base"] = disparity
    other = policy._frame_tokens(perturbed)  # noqa: SLF001

    depth_slice = slice(1, 1 + DEPTH_PATCHES)
    rgb_slice = slice(1 + DEPTH_PATCHES, None)
    assert not torch.equal(tokens[:, :, depth_slice], other[:, :, depth_slice])
    # depth must not leak into the RGB tokens or the state token
    assert torch.equal(tokens[:, :, rgb_slice], other[:, :, rgb_slice])
    assert torch.equal(tokens[:, :, 0], other[:, :, 0])


def test_no_depth_embedding_receives_gradient_when_depth_is_absent() -> None:
    """§22.5: the learned `no_depth` must TRAIN, through a real backward pass.

    The depth-absent case is the normal one -- none of the 104 existing episodes
    carry depth -- so this embedding is what the model actually sees most of the
    time. A `no_depth` that never received gradient would be a fancy constant.
    """
    policy = _depth_policy()
    policy.train()
    batch = _batch()  # NO disparity keys at all: the common real case
    assert "disparity.base" not in batch

    loss = policy._compute_metrics(batch)["policy", "loss"].sum(reduce=True)  # noqa: SLF001
    loss.backward()

    assert policy.no_depth.grad is not None
    assert torch.any(policy.no_depth.grad != 0)
    # ...but the trainable patch embedding gets NO gradient from a batch with no
    # depth in it, which is correct: nothing routed through it. §22.7's embedding
    # trains on the depth-PRESENT samples; `no_depth` covers the rest.
    assert policy.depth_patch_embedding.weight.grad is None


def test_absent_depth_key_and_depth_valid_false_are_the_same_state() -> None:
    """The two ways depth can be missing must not be two different models."""
    policy = _depth_policy()
    absent = _batch(seed=5)
    flagged = _depth_batch(seed=5)
    flagged["depth_valid"] = torch.zeros_like(flagged["depth_valid"])

    with torch.no_grad():
        a = policy(absent)["policy", "action"]
        b = policy(flagged)["policy", "action"]
    # the depth CONDITIONING vector is still present in `flagged`, so allow it to
    # differ there; compare the depth FEATURE path by zeroing the cond too
    flagged["disparity_cond"] = torch.zeros_like(flagged["disparity_cond"])
    with torch.no_grad():
        c = policy(flagged)["policy", "action"]
    assert torch.equal(a, c)
    assert b.shape == a.shape


def test_invalid_disparity_pixels_cannot_influence_the_output() -> None:
    """§22.4: `disparity == 0` is NO MEASUREMENT, so its value must not be read.

    The strong form: change the disparity values ONLY where the mask says
    invalid, and the output must be BIT-identical. A zero-filling implementation
    passes a "the mask reaches the model" test but fails this one.
    """
    policy = _depth_policy()
    batch = _depth_batch(seed=1)
    mask = batch["disparity_valid.base"]
    assert not bool(mask.all()), "test needs some invalid pixels"

    tampered = dict(batch)
    disparity = batch["disparity.base"].clone()
    # garbage, but only under the invalid mask
    noise = torch.randint(0, 96, disparity.shape, dtype=disparity.dtype)
    tampered["disparity.base"] = torch.where(mask, disparity, noise)

    with torch.no_grad():
        a = policy(batch)["policy", "action"]
        b = policy(tampered)["policy", "action"]
    assert torch.equal(a, b)


def test_the_validity_mask_itself_reaches_the_model() -> None:
    """Control for the test above: the MASK must matter, not just be respected."""
    policy = _depth_policy()
    batch = _depth_batch(seed=1)
    flipped = dict(batch)
    flipped["disparity_valid.base"] = torch.ones_like(batch["disparity_valid.base"])

    with torch.no_grad():
        a = policy(batch)["policy", "action"]
        b = policy(flipped)["policy", "action"]
    assert not torch.equal(a, b)


def test_depth_valid_is_genuinely_per_sample() -> None:
    """§22.5: a MIXED batch is the realistic case, not all-on or all-off."""
    policy = _depth_policy()
    batch = _depth_batch(seed=2)
    mixed = dict(batch)
    flag = torch.tensor([True, False])
    mixed["depth_valid"] = flag

    none = dict(batch)
    none["depth_valid"] = torch.zeros(2, dtype=torch.bool)

    with torch.no_grad():
        mixed_out = policy(mixed)["policy", "action"]
        none_out = policy(none)["policy", "action"]
        all_out = policy(batch)["policy", "action"]

    # sample 1 has no depth in either -> identical; sample 0 differs
    assert torch.equal(mixed_out[1], none_out[1])
    assert not torch.equal(mixed_out[0], none_out[0])
    assert torch.equal(mixed_out[0], all_out[0])


def test_depth_dropout_replaces_depth_with_the_learned_embedding() -> None:
    """§22.5: dropout applies in TRAINING only, and lands on `no_depth`."""
    policy = _depth_policy(depth_dropout=1.0)
    batch = _depth_batch(seed=4)

    absent = dict(batch)
    absent["depth_valid"] = torch.zeros(2, dtype=torch.bool)

    policy.train()
    with torch.no_grad():
        dropped = policy._frame_tokens(batch)  # noqa: SLF001
        never = policy._frame_tokens(absent)  # noqa: SLF001
    assert torch.equal(dropped, never)

    # ...and NOT in eval: serving must use the depth it was given
    policy.eval()
    with torch.no_grad():
        kept = policy._frame_tokens(batch)  # noqa: SLF001
    assert not torch.equal(kept, never)


def test_depth_camera_cond_is_bound_to_the_depth_tokens() -> None:
    """§22.2: depth carries its own 13-dim cond, from the MONO intrinsics."""
    policy = _depth_policy()
    batch = _depth_batch(seed=6)
    changed = dict(batch)
    changed["disparity_cond"] = torch.randn_like(batch["disparity_cond"])

    with torch.no_grad():
        a = policy._frame_tokens(batch)  # noqa: SLF001
        b = policy._frame_tokens(changed)  # noqa: SLF001

    depth_slice = slice(1, 1 + DEPTH_PATCHES)
    rgb_slice = slice(1 + DEPTH_PATCHES, None)
    assert not torch.equal(a[:, :, depth_slice], b[:, :, depth_slice])
    # the RGB cameras' geometry must be untouched by the depth camera's
    assert torch.equal(a[:, :, rgb_slice], b[:, :, rgb_slice])


def test_a_disparity_stream_without_its_mask_is_rejected() -> None:
    """§21.4/§22.4: never infer the mask from `disparity != 0`.

    A real loader marks additional pixels invalid (confidence, left-right check)
    that are NOT zero, so a silent fallback would read fabricated measurements.
    """
    policy = _depth_policy()
    batch = _depth_batch(seed=7)
    del batch["disparity_valid.base"]
    with pytest.raises(KeyError, match="validity mask"):
        _ = policy._frame_tokens(batch)  # noqa: SLF001


def test_depth_dropout_does_not_mutate_the_batch() -> None:
    """Regression: dropout must not write through to the loader's own tensor.

    `present.to(dtype=bool, device=cpu)` returns the SAME tensor, not a copy, so
    an in-place `&=` for the dropout mask zeroes `batch["depth_valid"]`
    permanently. With a cached or reused TensorDict that corruption outlives the
    step and is invisible -- the batch simply claims it never had depth. Caught
    by a test that failed for a completely unrelated-looking reason.
    """
    policy = _depth_policy(depth_dropout=1.0)
    batch = _depth_batch(seed=4)
    before = batch["depth_valid"].clone()
    assert bool(before.all()), "test needs depth present to begin with"

    policy.train()
    with torch.no_grad():
        _ = policy._frame_tokens(batch)  # noqa: SLF001

    assert torch.equal(batch["depth_valid"], before)
