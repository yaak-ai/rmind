"""Streaming/full-forward equivalence for `PatchPolicyDecoderStep`, multi-camera.

`tests/test_causal_frame.py` gates the same equivalence at the bare-trunk level
(synthetic `(b, seq, d)` tensors). This exercises it end-to-end through
`PatchPolicy` + `PatchPolicyDecoderStep`, with 3 cameras -- there is no test for
`PatchPolicyDecoderStep` at any camera count today.

Everything runs at float64 (weights, activations, images) so the two paths
agree to machine epsilon rather than a "close enough" tolerance -- the same
technique `rmind.scripts.decoder_only_verify.cmd_gates` uses, and for the same
reason: `sample_codes=False` argmax-decodes the joint head, and even a tiny
float32 residual can flip an argmax and blow up `joint_actions` while the
underlying features were fine. At float64 that's not a practical risk.
"""

import torch
from torch import Tensor
from torch.nn import Identity, L1Loss, Linear, Module, Sequential
from torchvision.ops import MLP

from rmind.components.base import Modality
from rmind.components.containers import ModuleDict
from rmind.components.loss import FocalLoss
from rmind.components.nn import Embedding
from rmind.components.norm import Scaler, UniformBinner
from rmind.components.transformer.causal_frame import CausalFrameTransformer
from rmind.components.vq import ResidualVQ
from rmind.models.action_tokenizer import ActionTokenizer
from rmind.models.patch_policy import PatchPolicy
from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep

BATCH_SIZE = 2
CAMERAS = ("cam_front_left", "cam_left_forward", "cam_right_forward")
IMG_SIZE = 4
IMG_CHANNELS = 3
NUM_PATCHES = 1  # per camera, from the tiny image encoder below
IMAGE_DIM = 6
GOAL_DIM = 6
POLICY_DIM = 8
NUM_LAYERS = 2
NUM_HEADS = 2
NUM_WAYPOINTS = 5
ACTION_HORIZON = 2
ACTION_FIELDS = 4
ACTION_DIM = ACTION_HORIZON * ACTION_FIELDS
LATENT_DIM = 6
NUM_QUANTIZERS = 2
CODEBOOK_SIZE = 4
SPEED_BINS = 8
WINDOW = 3
NUM_FRAMES = 5  # > WINDOW: exercises both the partial- and full-window regime


class _TinyImageEncoder(Module):
    """Maps raw `(..., c, h, w)` pixels to one patch token `(..., 1, d)`.

    Unlike `test_patch_policy.py`'s `image_encoder=Identity()` (which feeds
    pre-extracted patch features directly), `PatchPolicyDecoderStep.forward`
    normalizes raw pixels itself, so this test needs a real -- if tiny --
    pixel-space encoder to exercise that step.
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.proj = Linear(IMG_CHANNELS * IMG_SIZE * IMG_SIZE, out_dim)

    def forward(self, images: Tensor) -> Tensor:
        *b, c, h, w = images.shape
        return self.proj(images.reshape(*b, c * h * w)).unsqueeze(-2)  # (..., 1, d)


class _GoalEncoderStub(Module):
    """Maps waypoints `(b, t, n, 2)` -> a deterministic latent `(b, t, GOAL_DIM)`."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = Linear(2, GOAL_DIM)
        self.quantizer = ResidualVQ(
            dim=GOAL_DIM, codebook_size=4, num_quantizers=2, kmeans_init=False
        )

    def encode(self, waypoints: Tensor) -> Tensor:
        return self.proj(waypoints).mean(dim=-2)

    def forward(self, waypoints: Tensor) -> Tensor:
        return self.encode(waypoints)


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


def _make_model() -> PatchPolicy:
    tokens_per_frame = len(CAMERAS) * NUM_PATCHES + 1
    return PatchPolicy(
        input_transform=Identity(),
        image_encoder=_TinyImageEncoder(IMAGE_DIM),
        goal_encoder=_GoalEncoderStub(),
        patch_projection=Linear(IMAGE_DIM + GOAL_DIM, POLICY_DIM),
        speed_tokenizer=UniformBinner(range=(0.0, 130.0), bins=SPEED_BINS),
        speed_embedding=Embedding(SPEED_BINS, POLICY_DIM),
        cameras=CAMERAS,
        encoder=CausalFrameTransformer(
            dim_model=POLICY_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            tokens_per_frame=tokens_per_frame,
            window=WINDOW,
            attn_dropout=0.0,
            resid_dropout=0.0,
            mlp_dropout=0.0,
        ),
        tokenizer=_make_tokenizer(),
        code_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE]),
        offset_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE * ACTION_DIM]),
        losses=ModuleDict(modules={"code": FocalLoss(), "offset": L1Loss()}),
        norm=torch.nn.LayerNorm(POLICY_DIM),
        sample_codes=False,  # argmax: deterministic, and required for the float64
        # equivalence check below to be meaningful (see module docstring)
    ).eval()


def test_streamed_decode_matches_full_windowed_forward() -> None:  # noqa: PLR0914
    generator = torch.Generator().manual_seed(0)
    raw_images = {
        camera: torch.rand(
            (BATCH_SIZE, NUM_FRAMES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
            generator=generator,
            dtype=torch.float64,
        )
        for camera in CAMERAS
    }
    speed = (
        torch.rand(
            (BATCH_SIZE, NUM_FRAMES, 1), generator=generator, dtype=torch.float64
        )
        * 130.0
    )
    waypoints = torch.randn(
        (BATCH_SIZE, NUM_FRAMES, NUM_WAYPOINTS, 2),
        generator=generator,
        dtype=torch.float64,
    )

    step = PatchPolicyDecoderStep(policy=_make_model()).double().eval()

    # --- reference: one full forward over the whole clip, windowed mask -----
    # `step.image_mean`/`image_std` are the SAME buffers `step.forward` uses --
    # reusing them (rather than re-deriving the ImageNet constants) guarantees
    # this is exactly the normalization the streamed path applies internally.
    normalized_images = {
        camera: (raw_images[camera] - step.image_mean) / step.image_std
        for camera in CAMERAS
    }
    batch = {
        "image": normalized_images,
        "continuous": {"speed": speed},
        "context": {"waypoints": waypoints},
    }
    features, _ = step.policy._features(batch, require_chunk=False)  # noqa: SLF001
    reference_chunk = step.policy._predict_chunk(features)  # noqa: SLF001

    # --- streamed: one new frame per tick against a ring of WINDOW - 1 frames
    cache_frames = WINDOW - 1
    past_k, past_v, cache_bias = step.empty_cache(
        cache_frames=cache_frames, batch_size=BATCH_SIZE, dtype=torch.float64
    )
    streamed_chunks = []
    for t in range(NUM_FRAMES):
        cos, sin = step.rope(t)
        inputs = {
            **{
                f"image_{camera}": raw_images[camera][:, t : t + 1]
                for camera in CAMERAS
            },
            "speed": speed[:, t : t + 1],
            "waypoints": waypoints[:, t : t + 1],
            "past_k": past_k,
            "past_v": past_v,
            "cache_bias": cache_bias,
            "rope_cos": cos,
            "rope_sin": sin,
        }
        out = step(inputs)
        streamed_chunks.append(out["policy", "joint_actions"])
        past_k, past_v, cache_bias = step.advance(
            (past_k, past_v, cache_bias), out["new_k"], out["new_v"]
        )
    streamed_chunk = torch.stack(streamed_chunks, dim=1)

    assert streamed_chunk.shape == reference_chunk.shape
    torch.testing.assert_close(streamed_chunk, reference_chunk, atol=1e-8, rtol=1e-6)


def test_streamed_decode_is_sensitive_to_camera_identity() -> None:
    """Swapping two cameras' frames must change the streamed decode -- i.e. the
    per-camera inputs are actually distinguished, not just counted (same
    property `test_multi_camera_stacks_patches_in_camera_order` checks for the
    non-decoder path in `test_patch_policy.py`)."""
    generator = torch.Generator().manual_seed(1)
    raw_images = {
        camera: torch.rand(
            (1, 1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
            generator=generator,
            dtype=torch.float64,
        )
        for camera in CAMERAS
    }
    speed = torch.rand((1, 1, 1), generator=generator, dtype=torch.float64) * 130.0
    waypoints = torch.randn(
        (1, 1, NUM_WAYPOINTS, 2), generator=generator, dtype=torch.float64
    )

    step = PatchPolicyDecoderStep(policy=_make_model()).double().eval()
    past_k, past_v, cache_bias = step.empty_cache(
        cache_frames=WINDOW - 1, batch_size=1, dtype=torch.float64
    )
    cos, sin = step.rope(0)

    def _decode(images: dict[str, Tensor]) -> Tensor:
        inputs = {
            **{f"image_{camera}": images[camera] for camera in CAMERAS},
            "speed": speed,
            "waypoints": waypoints,
            "past_k": past_k,
            "past_v": past_v,
            "cache_bias": cache_bias,
            "rope_cos": cos,
            "rope_sin": sin,
        }
        return step(inputs)["policy", "joint_actions"]

    baseline = _decode(raw_images)
    swapped = {**raw_images}
    swapped["cam_left_forward"], swapped["cam_right_forward"] = (
        raw_images["cam_right_forward"],
        raw_images["cam_left_forward"],
    )
    assert not torch.allclose(baseline, _decode(swapped))
