"""Causal patch policy for the nero-arms bimanual manipulation contract.

Reuses the decoder-only trunk from `feat/patch-policy-decoder-only` unchanged --
`rmind.components.transformer.causal_frame.CausalFrameTransformer` (frame-RoPE +
tiled intra-frame embedding, bidirectional intra-frame / causal inter-frame,
KV-cacheable) -- and replaces everything above and below it to match the
nero-arms data contract.

Token layout (one frame block)
------------------------------
::

    [ state token (1) ][ base patches (P) ][ side_left (P) ][ side_right (P) ]
                                                          tokens_per_frame = 3P + 1

with `P = 160` -- a 10x16 grid at DINOv2's patch 14 on a 140x224 input, which is
the cameras' own 5:8 aspect -> **481 tokens per frame**. At `episode_length = 6`
the flattened sequence is **2886**, versus 1542 for the 6-frame driving arm and
4112 for the 16-frame causal driving arm.

Forcing the usual SQUARE 224x224 input would put identical image content on a
16x16 grid with 6 of 16 rows pure letterbox padding: 769 tokens per frame, 4614
flattened, ~2.6x the attention cost for no extra information.

The state token goes FIRST so that each frame block ends on a patch token, which
is the readout position (unchanged from PR #265).

Design decisions, and why
-------------------------

**Camera conditioning (contract §7.1) -> per-patch concatenation, not extra
tokens and not FiLM.** Each camera's 13-dim vector is concatenated to *that
camera's* patch tokens before `patch_projection`. Reasons, in order of weight:

1. *zero sequence cost*. At 481 tokens/frame attention is the binding constraint;
   3 extra tokens per frame is cheap but the pattern does not stay cheap, and
   FiLM would need a broadcast anyway.
2. *it binds the geometry to the tokens it describes*. A patch of `side_left`
   carries `side_left`'s extrinsics; an extra token or a FiLM vector would carry
   all three cameras' geometry to all three cameras' patches, and the trunk would
   have to learn the routing.
3. *it doubles as camera identity*. The tiled intra-frame positional embedding
   gives each slot an index, but that index is setup-specific; the conditioning
   vector is what generalises across camera-setup changes, which is the stated
   point of §7.1.

**Goal conditioning (contract §9) -> same-index concatenation of the goal
image's patch features.** The goal image is the episode's final frame *from the
same camera*, so goal patch `(c, p)` and observation patch `(c, p)` are the same
ray through the same lens. Concatenating them index-aligned preserves that
spatial correspondence, which is exactly the "where did the object end up"
signal; mean-pooling the goal (the obvious cheap alternative) discards it. Cost
is identical -- both are a channel concat, no extra tokens. This is the natural
generalisation of the paper's `T x P x (D + G)` scheme, with `G = D` and the
goal constant over `t` (the driving arm's `g_t` varied per frame because
waypoints are ego-frame; a goal image does not).

**Goal dropout.** With probability `goal_dropout` (per sample, per camera) the
goal features are replaced by a *learned* `no_goal` embedding rather than zeros,
so "no goal supplied" is distinguishable from "goal that happens to encode near
zero". Without this the policy becomes goal-dependent for basic motion (§9).

**Bimanual `side_valid` (contract §6.1).** Consumed in two places, both
falsifiable:

* the state token is built from `state * side_valid` with the 2-dim mask
  appended, so perturbing an invalid side's state cannot change any output;
* the action loss selects only valid `(batch, frame, side)` rows. Normalisation
  is `sum / count`, never `mean` over a zero-padded tensor -- the latter silently
  halves the loss on right-only data, which changes the effective LR and makes
  the curve incomparable to a future bimanual run.

**Per-side, weight-shared head.** One readout token per frame feeds a shared
`code_head`/`offset_head`, applied twice with a learned per-side embedding added
to the feature. This halves the head parameter count versus two independent
heads and -- more importantly -- lets right-only dummy data train a head that is
immediately meaningful for the left hand, matching the weight-shared tokenizer.

Depth (contract §22) -- OPT-IN, off by default
----------------------------------------------
`use_depth=False` is the default and nothing below is constructed in that case,
so the depth-off model is bit-identical to the pre-depth one (same parameters,
same init RNG stream, same forward). The four decisions, all from §22:

**Depth is a FOURTH CAMERA, not a 4th channel on the RGB image (§22.1/§22.2).**
The disparity stream's metadata declares `reference_frame: rectified_CAM_B`,
`aligned_to: none` -- it lives in the rectified LEFT MONO camera's frame, a
different sensor at a different position with a much wider lens (96.0 deg HFOV
against `CAM_A`'s 73.7 deg). `setDepthAlign` is deliberately not applied upstream
because warping invalidates the `fx_mono * baseline / disparity` conversion. So
stacking it as an extra channel on the overhead RGB would silently misregister
every pixel. Instead it gets its own patch tokens and its own 13-dim
`camera_cond` built from the MONO intrinsics at the recorded depth resolution
(§21.11), which is the same machinery that already handles three heterogeneous
cameras.

**A TRAINABLE patch embedding, not the frozen DINOv2 encoder (§22.7).** The
first implementation here routed disparity through the RGB ViT; that was wrong.
DINOv2 is trained on natural RGB -- photometric statistics, texture, colour,
semantics -- while disparity is a smooth, single-channel, purely *geometric*
signal, so replicated to 3 channels it is a grey image with none of the
statistics the encoder expects. The encoder is also **frozen**, so nothing
downstream can adapt the mismatch away. And the cost ran backwards: a fourth ViT
pass is +33% frozen compute per frame to buy only ~+8% sequence, i.e. paying the
larger cost for the less appropriate representation. Instead the disparity is
patchified and projected by a `Linear` -- what a ViT does at its own input, and
the standard treatment for a non-RGB modality: ~262k **trainable** parameters
against ~22M frozen ones, and cheaper in FLOPs than the pass it replaces.

The coarse depth grid §22.2 asks for then falls out of the patch size directly,
with no pooling: an 80x128 letterboxed input at patch 16 is a 5x8 = 40-token
grid. `(5, 16)` -- §22.2's literal "half the patches" -- is a config change.

**Normalised DISPARITY, not metric depth (§22.3), with the validity mask as a
SECOND INPUT CHANNEL (§22.4).** `disparity == 0` means *no measurement*, not zero
distance, and stereo drops out precisely at the depth discontinuities around a
grasped object -- i.e. exactly where the signal matters. Making the mask a
first-class input channel is what lets "unmeasured" be its own state instead of
something the network has to infer from a fill value; the disparity channel is
additionally filled with the train-split mean where invalid, so the two channels
agree on a neutral value plus an explicit flag rather than asserting a surface at
zero distance. See `rmind.data.nero.DisparityStandardizer`.

**Depth is ABSENT from most data and that is the normal case (§22.5).** None of
the 104 existing episodes have depth and the recorder's `--depth` is off by
default, so a depth-enabled model trains on a mixture. THREE cases, all handled:
the key is missing from the batch entirely (no encoder call at all), the key is
present but a given sample has `depth_valid=False`, and `depth_dropout` forcing
the second case during training so the policy never becomes depth-DEPENDENT and
degrades gracefully when the stream is missing at serving time. In every case the
depth tokens carry a **learned `no_depth` embedding, never zeros**, plus an
explicit availability flag -- the same pattern as the `no_goal` goal-dropout
embedding and `side_valid`.

Depth tokens are placed BETWEEN the state token and the RGB patches, so the
readout (the last token of a frame block) is still a `side_right` patch exactly
as in the depth-off model, rather than becoming a constant `no_depth` token on
the majority of samples.

Configuration seam (contract §11)
---------------------------------
`action_features` (per-side action dimensionality) and `action_horizon` come from
the tokenizer, and every head width is derived in config from
`num_quantizers * codebook_size * action_horizon * action_features`. Swapping to
§11 option (B) -- Revo2 joint targets, ~12 dims per side instead of 60 -- is a
new tokenizer checkpoint plus the derived config values; no code change here.
What WOULD change: `rbyte` must apply the glove-SE(3) -> Revo2 retargeting at
ingestion, `state.pose` would become joint angles (or stay SE(3) as a separate
observation block, which this model supports by pointing `state` elsewhere), and
`NeroPoseTokenizer.has_pose_layout` goes False so the mm/degree metrics are
replaced by joint-angle degrees.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, final, override

import pytorch_lightning as pl
import torch
from einops import rearrange
from pydantic import Field, InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from structlog import get_logger
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Optimizer

from rmind.components import optimizers
from rmind.components.containers import ModuleDict
from rmind.components.nn import NormedTokenEmbedding
from rmind.config import HydraConfig, init_hydra_param
from rmind.data.nero import (
    CAMERA_COND_DIM,
    NUM_SIDES,
    STATE_QUAT_DIM,
    pose_error_metrics,
    state_quat_to_9d,
)
from rmind.data.nero_robot import (
    FINGER_AXES,
    RELATIVE_MODES,
    HandTokenStandardizer,
    compose_hand_token,
    compose_hand_tokens,
    hand_token_dim,
    normalize_hand_groups,
    normalize_hand_sides,
    to_absolute,
)
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.models.control_transformer import PredictionConfig
from rmind.models.nero_quality import (
    alarm_metrics,
    code_confidence_metrics,
    grasp_event_metrics,
    grasp_window_mask,
    horizon_ev_metrics,
    per_axis_ev,
)
from rmind.utils._wandb import LoadableFromArtifact

__all__ = ["NeroPatchPolicy"]

logger = get_logger(__name__)

type Path = tuple[str, ...]


@final
class NeroPatchPolicy(pl.LightningModule, LoadableFromArtifact):
    """Causal patch policy over 3 cameras + bimanual SE(3) state. See module docstring."""

    @validate_call
    def __init__(  # ruff: ignore[too-many-arguments, too-many-statements]
        self,
        *,
        image_transform: HydraConfig[Module] | InstanceOf[Module],
        image_encoder: HydraConfig[Module] | InstanceOf[Module],
        patch_projection: HydraConfig[Module] | InstanceOf[Module],
        state_embedding: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module],
        tokenizer: HydraConfig[Module] | InstanceOf[Module],
        code_head: HydraConfig[Module] | InstanceOf[Module],
        offset_head: HydraConfig[Module] | InstanceOf[Module],
        losses: HydraConfig[ModuleDict] | InstanceOf[ModuleDict],
        image_embedding_dim: int,
        policy_embedding_dim: int,
        norm: HydraConfig[Module] | InstanceOf[Module] | None = None,
        cameras: Sequence[str] = ("base", "side_left", "side_right"),
        image_key: str = "image.{camera}",
        goal_image_key: str = "goal.image.{camera}",
        camera_cond: Path = ("camera_cond",),
        state: Path = ("state.pose",),
        side_valid: Path = ("side_valid",),
        goal_valid: Path = ("goal_valid",),
        chunk: Path = ("action.future_state",),
        use_goal_image: bool = True,
        # --- contract §22 depth, OFF BY DEFAULT (§22.6). Nothing below is
        # constructed unless `use_depth`, so the depth-off model is bit-identical
        # to the pre-depth one.
        use_depth: bool = False,
        depth_transform: HydraConfig[Module] | InstanceOf[Module] | None = None,
        depth_standardizer: HydraConfig[Module] | InstanceOf[Module] | None = None,
        depth_patch_embedding: HydraConfig[Module] | InstanceOf[Module] | None = None,
        #: §21.3: the overhead `base` device only -- the SR side cameras sit
        #: near-tangent and localise on-plane objects poorly.
        depth_cameras: Sequence[str] = ("base",),
        depth_key: str = "disparity.{camera}",
        #: PER-PIXEL validity (§21.4/§22.4), not to be confused with...
        depth_mask_key: str = "disparity_valid.{camera}",
        #: ...this, the PER-SAMPLE "does this episode have depth at all" flag (§22.5)
        depth_valid: Path = ("depth_valid",),
        depth_camera_cond: Path = ("disparity_cond",),
        #: side of the square depth patch, e.g. 16 (§22.7)
        depth_patch_size: int | None = None,
        #: the depth token grid, e.g. (5, 8) for an 80x128 input at patch 16
        depth_patch_grid: tuple[int, int] | None = None,
        depth_dropout: float = 0.25,
        # rbyte emits the contract §5.2 STORAGE form (46 per side: 6 poses x 7 +
        # a 4-dim hub quaternion). The 9D expansion happens here, at the model
        # boundary -- set False if a loader ever hands over 60 directly.
        convert_state_to_9d: bool = True,
        goal_dropout: float = 0.15,
        # Argmax by default, not sampling. No loss depends on this while
        # `teacher_force_offset` is True (the default): the code losses are
        # cross-entropy against tokenizer-encoded `target_codes`, and the offset
        # loss is teacher-forced from those same codes. So sampling only feeds
        # the reported `offset_sampled_recon` / pose-error metrics -- and those
        # are more useful computed the way SERVING decodes, which is argmax.
        # It also makes inference deterministic; see docs/nero_serving_handover.md.
        sample_codes: bool = False,
        teacher_force_offset: bool = True,
        offset_scale: float | None = None,
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
        prediction_config: Annotated[
            PredictionConfig, Field(default_factory=PredictionConfig)
        ],
        # --- robot-native patch family (P3-P7, P9). ALL OPT-IN, and everything
        # they construct is built AFTER depth, so a model that leaves them at
        # their defaults is bit-identical to the glove model (same parameters,
        # same init RNG stream, same forward). See docs/nero_robot_patch_policy.md.
        #: "glove": the §6 SE(3) stand-in (tokenizer standardizes internally).
        #: "robot": §11(B) robot-native 13-d per side; the tokenizer
        #: (`NeroChunkTokenizer`) owns the relative transform + standardizer and
        #: the batch carries `action.is_pad`.
        action_space: Literal["glove", "robot"] = "glove",
        #: P7. None = legacy (`use_goal_image` decides: "image" or "none").
        #: "no_goal": the goal channel exists but is ALWAYS the learned `no_goal`
        #: (no goal frames are read; the export has no goal input).
        goal_mode: Literal["image", "no_goal", "none"] | None = None,
        #: P6: hand token groups (hand_features.SELECTABLE_GROUPS); () = no token
        hand_groups: Sequence[str] = (),
        #: WP5: () = ONE untagged hand token read from `hand.*` (single arm, the
        #: pre-bimanual layout and checkpoints); ("left", "right") = one
        #: side-tagged token PER SIDE read from `hand.{side}.*` (bimanual,
        #: contract v3 `hand_sides`), all through the one pooled
        #: `hand_embedding` + `hand_standardizer`, each with its own `no_hand`
        #: substitution and dropout.
        hand_sides: Sequence[str] = (),
        hand_embedding: HydraConfig[Module] | InstanceOf[Module] | None = None,
        hand_prefix: str = "hand.",
        hand_token_key: str = "hand_token",
        hand_dropout_sample: float = 0.2,
        hand_dropout_frame: float = 0.1,
        #: robot: train-split per-axis state standardizer (in-graph at serving)
        state_standardizer: HydraConfig[Module] | InstanceOf[Module] | None = None,
        #: P6: fixed per-column affine for the hand token's feature columns
        #: (`HandTokenStandardizer`, in-graph; hand_age/hand_valid untouched).
        #: None = identity -- only for checkpoints trained before it existed.
        hand_standardizer: HydraConfig[Module] | InstanceOf[Module] | None = None,
        #: P5 ablation flag; must equal the tokenizer's
        relative_mode: Literal["none", "hand", "all"] = "none",
        #: "table" = VQ-BeT per-code full-chunk offsets (glove default);
        #: "latent" = one offset in tokenizer latent space, decoded through the
        #: frozen tokenizer decoder (robot; the table is ~340M params at 100x13)
        offset_mode: Literal["table", "latent"] = "table",
        #: latent only: feed the stop-grad quantized latent of the codes
        #: (`tokenizer.lookup(codes)`) into the offset head, concatenated to the
        #: features -- TARGET codes in training (teacher forcing), the ARGMAX
        #: codes at serving / in the decoder step. Without it the offset never
        #: sees the codes: it is fitted to the target codes' residual but added
        #: to the argmax codes' latent, so when the code head picks a different
        #: mode the offset is the residual averaged across modes, not a
        #: refinement inside the chosen one. The offset head then takes
        #: `policy_embedding_dim + latent_dim` inputs.
        offset_code_conditioning: bool = False,
        #: robot: measure the patch-token RMS on the first training batch and
        #: set the state/hand `NormedTokenEmbedding` gains to it
        calibrate_token_gain: bool = False,
        #: P9: token norms, code confidence, per-horizon EV, grasp, alarms
        quality_metrics: bool = False,
        #: P9: hand reliance deltas (3 extra no-grad forwards per val batch)
        reliance_metrics: bool = False,
        #: an expected action-standardizer digest; refused if the tokenizer's differs
        action_standardizer_sha256: str | None = None,
    ) -> None:
        super().__init__()

        hparams: dict[str, Any] = {}

        self.image_transform = init_hydra_param(
            hparams, "image_transform", image_transform
        )
        # frozen feature extractor: never trains, never leaves eval mode (see train())
        self.image_encoder = (
            init_hydra_param(hparams, "image_encoder", image_encoder)
            .requires_grad_(False)  # ruff: ignore[boolean-positional-value-in-call]
            .eval()
        )
        self.tokenizer = (
            init_hydra_param(hparams, "tokenizer", tokenizer)
            .requires_grad_(False)  # ruff: ignore[boolean-positional-value-in-call]
            .eval()
        )

        self.patch_projection = init_hydra_param(
            hparams, "patch_projection", patch_projection
        )
        self.state_embedding = init_hydra_param(
            hparams, "state_embedding", state_embedding
        )
        self.encoder: Module = init_hydra_param(hparams, "encoder", encoder)
        self.code_head = init_hydra_param(hparams, "code_head", code_head)
        self.offset_head = init_hydra_param(hparams, "offset_head", offset_head)
        self.losses: ModuleDict = init_hydra_param(hparams, "losses", losses)
        self.norm: Module | None = init_hydra_param(hparams, "norm", norm)

        # learned "no goal supplied" feature -- see the docstring on goal dropout
        self.no_goal = nn.Parameter(torch.zeros(image_embedding_dim))
        nn.init.trunc_normal_(self.no_goal, std=0.02)
        # per-side identity for the weight-shared head
        self.side_embedding = nn.Embedding(NUM_SIDES, policy_embedding_dim)
        nn.init.trunc_normal_(self.side_embedding.weight, std=0.02)

        # --- contract §22 depth. ⚠️ CONSTRUCTED LAST, AND ONLY IF `use_depth`.
        # Last, so that every module above draws the SAME init RNG whether depth
        # is on or off -- which is what makes the §22.6 A/B a controlled
        # comparison instead of two differently-initialised models. Only if
        # `use_depth`, so the depth-off model has no extra parameters at all.
        self.use_depth = use_depth
        self.depth_transform: Module | None = None
        self.depth_standardizer: Module | None = None
        self.depth_patch_embedding: Module | None = None
        self.depth_patch_size: int | None = None
        self.depth_patch_grid: tuple[int, int] | None = None
        if use_depth:
            missing = [
                name
                for name, value in (
                    ("depth_transform", depth_transform),
                    ("depth_standardizer", depth_standardizer),
                    ("depth_patch_embedding", depth_patch_embedding),
                    ("depth_patch_size", depth_patch_size),
                    ("depth_patch_grid", depth_patch_grid),
                )
                if value is None
            ]
            if missing:
                msg = f"use_depth=True requires {missing}"
                raise ValueError(msg)
            self.depth_transform = init_hydra_param(
                hparams, "depth_transform", depth_transform
            )
            # not a Parameter: train-split statistics, registered buffers, so it
            # travels inside the checkpoint (see DisparityStandardizer)
            self.depth_standardizer = init_hydra_param(
                hparams, "depth_standardizer", depth_standardizer
            ).requires_grad_(False)  # ruff: ignore[boolean-positional-value-in-call]
            self.depth_patch_embedding = init_hydra_param(
                hparams, "depth_patch_embedding", depth_patch_embedding
            )
            self.depth_patch_size = int(depth_patch_size)  # ty:ignore[invalid-argument-type]
            self.depth_patch_grid = tuple(depth_patch_grid)  # ty:ignore[invalid-argument-type]
            # §22.5: learned "no depth supplied", NEVER zeros -- so "this episode
            # has no depth stream" is distinguishable from "a depth map that
            # happens to encode near zero". It is a learned TOKEN (d_model), the
            # same role a mask token plays in MAE/BERT.
            self.no_depth = nn.Parameter(torch.zeros(policy_embedding_dim))
            nn.init.trunc_normal_(self.no_depth, std=0.02)

        # --- robot-native / hand / goal mode. ⚠️ CONSTRUCTED AFTER DEPTH and
        # only when enabled (see the argument comment).
        if action_space not in {"glove", "robot"}:
            msg = f"action_space {action_space!r}"
            raise ValueError(msg)
        if relative_mode not in RELATIVE_MODES:
            msg = f"relative_mode {relative_mode!r} not in {RELATIVE_MODES}"
            raise ValueError(msg)
        self.action_space = action_space
        self.robot = action_space == "robot"
        if goal_mode is None:
            goal_mode = "image" if use_goal_image else "none"
        if goal_mode == "image" and not use_goal_image:
            msg = "goal_mode='image' needs use_goal_image=True"
            raise ValueError(msg)
        self.goal_mode = goal_mode
        self.hand_groups = normalize_hand_groups(hand_groups)
        self.use_hand = bool(self.hand_groups)
        self.hand_sides = normalize_hand_sides(hand_sides)
        if self.hand_sides and not self.use_hand:
            msg = f"hand_sides {list(self.hand_sides)} without hand_groups (no hand token)"
            raise ValueError(msg)
        #: hand tokens per frame: 0 (no hand), 1 (untagged), len(hand_sides)
        self.n_hand_tokens = (len(self.hand_sides) or 1) if self.use_hand else 0
        self.hand_embedding: Module | None = None
        if self.use_hand:
            if hand_embedding is None:
                msg = "hand_groups set but no hand_embedding"
                raise ValueError(msg)
            self.hand_embedding = init_hydra_param(
                hparams, "hand_embedding", hand_embedding
            )
            width = getattr(self.hand_embedding, "in_features", None)
            if width is not None and width != hand_token_dim(self.hand_groups):
                msg = (
                    f"hand_embedding takes {width} inputs but hand_groups "
                    f"{self.hand_groups} give a {hand_token_dim(self.hand_groups)}-dim token"
                )
                raise ValueError(msg)
            # learned "no hand reading" (refused / stale / dropped). For a
            # NormedTokenEmbedding it is mapped through the SAME output norm and
            # gain as a real token, so it sits at the same scale.
            self.no_hand = nn.Parameter(torch.zeros(policy_embedding_dim))
            nn.init.normal_(self.no_hand, std=1.0)
        self.state_standardizer: Module | None = init_hydra_param(
            hparams, "state_standardizer", state_standardizer
        )
        self.hand_standardizer = self._init_hand_standardizer(
            hparams, hand_standardizer
        )
        if self.state_standardizer is not None:
            self.state_standardizer.requires_grad_(False)  # ruff: ignore[boolean-positional-value-in-call]
        self.hand_prefix = hand_prefix
        self.hand_token_key = hand_token_key
        self.hand_dropout_sample = hand_dropout_sample
        self.hand_dropout_frame = hand_dropout_frame
        self.relative_mode = relative_mode
        tokenizer_mode = getattr(self.tokenizer, "relative_mode", "none")
        if tokenizer_mode != relative_mode:
            msg = (
                f"policy relative_mode {relative_mode!r} != tokenizer's "
                f"{tokenizer_mode!r}: each relative mode needs its own tokenizer"
            )
            raise ValueError(msg)
        if action_standardizer_sha256 is not None:
            digest = getattr(self.tokenizer, "standardizer_digest", None)
            if digest != action_standardizer_sha256:
                msg = (
                    f"tokenizer action standardizer {digest} != the pinned "
                    f"{action_standardizer_sha256}"
                )
                raise ValueError(msg)
        if offset_mode == "latent" and not hasattr(self.tokenizer, "decode_latent"):
            msg = "offset_mode='latent' needs a tokenizer with decode_latent"
            raise ValueError(msg)
        if offset_code_conditioning and offset_mode != "latent":
            msg = "offset_code_conditioning needs offset_mode='latent'"
            raise ValueError(msg)
        self.offset_mode = offset_mode
        self.offset_code_conditioning = offset_code_conditioning
        if offset_mode == "latent":
            first = next(
                (m for m in self.offset_head.modules() if isinstance(m, nn.Linear)),
                None,
            )
            latent = int(getattr(self.tokenizer, "latent_dim", 0))
            expected = policy_embedding_dim + (
                latent if offset_code_conditioning else 0
            )
            if first is not None and first.in_features != expected:
                msg = (
                    f"offset_head takes {first.in_features} inputs, expected {expected} "
                    f"(policy_embedding_dim {policy_embedding_dim}"
                    + (f" + latent {latent}" if offset_code_conditioning else "")
                    + ")"
                )
                raise ValueError(msg)
        self.calibrate_token_gain = calibrate_token_gain
        self.quality_metrics = quality_metrics
        self.reliance_metrics = reliance_metrics
        self.action_standardizer_sha256 = action_standardizer_sha256
        if calibrate_token_gain:
            self.register_buffer("token_gain_calibrated", torch.zeros(()))
        self._calibrate_now = False

        self.depth_cameras = tuple(depth_cameras)
        self.depth_key = depth_key
        self.depth_mask_key = depth_mask_key
        self.depth_valid: Path = depth_valid
        self.depth_camera_cond: Path = depth_camera_cond
        self.depth_dropout = depth_dropout

        self.cameras = tuple(cameras)
        self.image_key = image_key
        self.goal_image_key = goal_image_key
        self.camera_cond: Path = camera_cond
        self.state: Path = state
        self.side_valid: Path = side_valid
        self.goal_valid: Path = goal_valid
        self.chunk: Path = chunk
        self.use_goal_image = use_goal_image
        self.convert_state_to_9d = convert_state_to_9d
        self.goal_dropout = goal_dropout
        self.sample_codes = sample_codes
        self.teacher_force_offset = teacher_force_offset
        self.offset_scale = offset_scale
        self.image_embedding_dim = image_embedding_dim
        self.policy_embedding_dim = policy_embedding_dim
        hparams |= {
            "cameras": self.cameras,
            "image_key": image_key,
            "goal_image_key": goal_image_key,
            "camera_cond": camera_cond,
            "state": state,
            "side_valid": side_valid,
            "goal_valid": goal_valid,
            "chunk": chunk,
            "use_goal_image": use_goal_image,
            "use_depth": use_depth,
            "depth_cameras": self.depth_cameras,
            "depth_key": depth_key,
            "depth_mask_key": depth_mask_key,
            "depth_valid": depth_valid,
            "depth_camera_cond": depth_camera_cond,
            "depth_patch_size": self.depth_patch_size,
            "depth_patch_grid": self.depth_patch_grid,
            "depth_dropout": depth_dropout,
            "convert_state_to_9d": convert_state_to_9d,
            "goal_dropout": goal_dropout,
            "sample_codes": sample_codes,
            "teacher_force_offset": teacher_force_offset,
            "offset_scale": offset_scale,
            "image_embedding_dim": image_embedding_dim,
            "policy_embedding_dim": policy_embedding_dim,
            "action_space": action_space,
            "goal_mode": goal_mode,
            "hand_groups": self.hand_groups,
            "hand_sides": self.hand_sides,
            "hand_prefix": hand_prefix,
            "hand_token_key": hand_token_key,
            "hand_dropout_sample": hand_dropout_sample,
            "hand_dropout_frame": hand_dropout_frame,
            "relative_mode": relative_mode,
            "offset_mode": offset_mode,
            "offset_code_conditioning": offset_code_conditioning,
            "calibrate_token_gain": calibrate_token_gain,
            "quality_metrics": quality_metrics,
            "reliance_metrics": reliance_metrics,
            "action_standardizer_sha256": action_standardizer_sha256,
        }

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.prediction_config = prediction_config

        # WP5: the per-side hand tag. ⚠️ CONSTRUCTED LAST and ONLY when sided,
        # so a single-token (or hand-off) model draws the identical init RNG
        # stream and has the identical state_dict it had before hand_sides
        # existed. It is added BEFORE the hand embedding's output norm (see
        # `_embed_hand`), to the real token and to `no_hand` alike, so the trunk
        # sees WHICH hand is refused. (The trunk's learned intra-frame slot
        # embedding also separates the slots; the tag makes the identity part
        # of the token itself.)
        self.hand_side_embedding: nn.Embedding | None = None
        if self.hand_sides:
            self.hand_side_embedding = nn.Embedding(
                len(self.hand_sides), policy_embedding_dim
            )
            nn.init.trunc_normal_(self.hand_side_embedding.weight, std=0.02)

        self.save_hyperparameters(hparams)

    @override
    def train(self, mode: bool = True) -> "NeroPatchPolicy":
        super().train(mode)
        self.image_encoder.eval()
        self.tokenizer.eval()
        return self

    # ------------------------------------------------------------------ input

    @staticmethod
    def _lookup(inputs: Mapping[str, Any], path: Path) -> Tensor | None:
        value: Any = inputs
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
        return value

    @classmethod
    def _get(cls, inputs: Mapping[str, Any], path: Path) -> Tensor:
        """Fetch a required contract key, raising rather than propagating `None`.

        Raises:
            KeyError: if the path is absent from the batch.
        """
        value = cls._lookup(inputs, path)
        if value is None:
            msg = f"input {path!r} missing from batch"
            raise KeyError(msg)
        return value

    def _encode_images(self, images: Tensor) -> Tensor:
        """`(..., 3, H, W)` uint8 -> frozen patch features `(..., P, D)`."""
        with torch.no_grad():
            return self.image_encoder(self.image_transform(images))

    def _goal_features(
        self, batch: Any, *, batch_size: int, device: torch.device
    ) -> Tensor | None:
        """`(b, n_cameras, P, D)` goal patch features, or None when goals are off.

        ⚠️ The goal frames are THREE SEPARATE KEYS, not one stacked tensor: rbyte
        cannot index one stream by several columns, and the final frame index
        differs per camera (199 vs 200 in the dummy). They are also on different
        native grids, so each is letterboxed by `image_transform` before being
        stacked -- which is only valid because the transform lands them all on
        the same grid.
        """
        if self.goal_mode in {"none", "no_goal"}:
            # "no_goal" (P7: one policy per task) reads no goal frame at all;
            # `_frame_tokens` expands the learned `no_goal` onto the patch grid.
            return None

        n_cam = len(self.cameras)
        # (b, n_cam) bool. Absent => all-true, so existing batches are unaffected.
        supplied = self._lookup(batch, self.goal_valid)
        if supplied is None:
            present = torch.ones(batch_size, n_cam, dtype=torch.bool, device=device)
        else:
            present = supplied.to(device=device, dtype=torch.bool)
            if present.ndim == 1:  # (b,) broadcasts to every camera
                present = present[:, None].expand(batch_size, n_cam)
            present = present.reshape(batch_size, n_cam)

        encoded: list[Tensor | None] = []
        for index, camera in enumerate(self.cameras):
            images = self._lookup(batch, (self.goal_image_key.format(camera=camera),))
            if images is None:
                # Omitting the frames is only legal where nothing claims a goal:
                # a missing key with `goal_valid` true is a real error, not an
                # implicit "no goal".
                if bool(present[:, index].any()):
                    self._get(batch, (self.goal_image_key.format(camera=camera),))
                encoded.append(None)
                continue
            encoded.append(self._encode_images(images))

        reference = next((e for e in encoded if e is not None), None)
        if reference is None:
            # Every goal omitted. `num_patches` is not knowable from the goal
            # side alone, so borrow it from the observation stream, which is
            # always present and lands on the same grid via `image_transform`.
            reference = self._encode_images(
                self._get(batch, (self.image_key.format(camera=self.cameras[0]),))[
                    :, :1
                ]
            ).squeeze(1)
        features = torch.stack(
            [self.no_goal.expand_as(reference) if e is None else e for e in encoded],
            dim=1,
        )  # (b, n_cam, P, D)

        if self.training and self.goal_dropout > 0:
            keep = torch.rand(batch_size, n_cam, device=device) >= self.goal_dropout
            # ⚠️ NOT `present &= keep`. `present` may ALIAS `batch["goal_valid"]`
            # (`.to()` returns self when dtype/device already match), so an
            # in-place op would corrupt the loader's tensor for the rest of its
            # life -- invisibly, and worse with a cached TensorDict. Same bug a
            # ruff PLR6104 autofix introduced in the depth path.
            present &= keep

        return torch.where(
            present[:, :, None, None], features, self.no_goal.to(features.dtype)
        )

    # ------------------------------------------------------------------ depth

    def _patchify_depth(self, disparity: Tensor, mask: Tensor) -> Tensor:
        """`(b, T, 1, H, W)` disparity + mask -> `(b, T, P, patch * patch * 2)`.

        Contract §22.7: depth gets a **trainable patch embedding, NOT the frozen
        DINOv2 encoder**. Routing disparity through the RGB ViT was the first
        implementation here and it was wrong on all three counts: DINOv2 is
        trained on natural RGB (photometric statistics, texture, colour,
        semantics) while disparity is a smooth single-channel *geometric* signal,
        so replicated to 3 channels it is a grey image with none of the
        statistics the encoder expects; the encoder is **frozen**, so nothing
        downstream can adapt away the mismatch; and the cost runs the wrong way,
        a fourth ViT pass being +33% frozen compute per frame to buy only ~+8%
        sequence. A `Linear` over flattened patches is what a ViT does at its own
        input anyway, is ~262k TRAINABLE parameters against ~22M frozen ones, and
        is cheaper in FLOPs than the pass it replaces.

        **Two input channels: standardised disparity and the validity mask**
        (§22.4). The mask being a first-class input is the point -- it is what
        lets the model treat "no measurement" as its own state rather than having
        to infer it from a fill value. The disparity channel is still filled with
        the train-split mean where invalid (`DisparityStandardizer`), so the two
        channels agree: a neutral value plus an explicit "this is not a
        measurement" bit, never a fabricated surface at zero distance.

        Order is load-bearing: the fill and the standardisation happen at native
        resolution, **before** any resampling. The other way round would let the
        invalid zeros bleed into their valid neighbours during interpolation -- a
        small, plausible-looking, entirely fabricated depth gradient exactly at
        the object boundaries where stereo drops out.

        `depth_transform` (a `LetterboxResize`) is applied to BOTH channels and
        its zero padding is correct for both by construction: 0 is the train mean
        in standardised space, and 0 in the mask is "invalid".

        Raises:
            ValueError: if the transformed size is not an exact multiple of
                `depth_patch_size`, or disagrees with `depth_patch_grid`. A
                mismatched reshape would scramble the spatial layout silently.
        """
        assert self.depth_standardizer is not None  # noqa: S101
        assert self.depth_transform is not None  # noqa: S101
        assert self.depth_patch_size is not None  # noqa: S101
        standardized = self.depth_standardizer(disparity, mask)
        small = self.depth_transform(standardized)  # (b, T, 1, h, w)
        valid = self.depth_transform(mask.to(small.dtype))  # (b, T, 1, h, w)
        stacked = torch.cat([small, valid], dim=-3)  # (b, T, 2, h, w)

        size = self.depth_patch_size
        *_, height, width = stacked.shape
        grid = (height // size, width // size)
        if (
            grid[0] * size != height
            or grid[1] * size != width
            or grid != self.depth_patch_grid
        ):
            msg = (
                f"depth input {height}x{width} at patch {size} gives grid {grid}, "
                f"which is not an exact tiling or disagrees with "
                f"depth_patch_grid {self.depth_patch_grid}"
            )
            raise ValueError(msg)
        return rearrange(
            stacked, "... c (gh p1) (gw p2) -> ... (gh gw) (c p1 p2)", p1=size, p2=size
        )

    def _depth_tokens(
        self, batch: Any, *, batch_size: int, num_frames: int, device: torch.device
    ) -> Tensor | None:
        """`(b, T, n_depth_cameras * Pd, d)` depth patch tokens, or None when off.

        Per-token layout into the trainable embedding (§22.7)::

            [ flattened patch: patch * patch * 2 channels | camera_cond (13) ]

        The per-PATCH validity is already inside that first block, as the second
        channel of every pixel (§22.4) -- so a patch that is entirely unmeasured
        is representable, and so is one that is half unmeasured, which is the
        common case at an object boundary.

        The per-SAMPLE `depth_valid` (§22.5) is a different statement -- "this
        episode has no depth stream at all" -- and it is consumed by SUBSTITUTION
        rather than as an input dimension: the whole token is replaced by the
        learned `no_depth`. Feeding it as an extra input dimension as well would
        be dead weight, since it is 1 for every token that survives the
        substitution.

        Raises:
            KeyError: if a disparity stream is present without its validity mask.
                Falling back to `disparity != 0` would look right and would
                silently discard the loader's additional invalidations
                (confidence threshold, left-right check), which are not zero.
        """
        if not self.use_depth:
            return None
        assert self.depth_patch_grid is not None  # ruff: ignore[assert]
        assert self.depth_patch_embedding is not None  # ruff: ignore[assert]
        b, t = batch_size, num_frames
        num_patches = self.depth_patch_grid[0] * self.depth_patch_grid[1]

        # ⚠️ §22.5: `depth_valid` ABSENT means "no depth", not "assume depth".
        # None of the 104 existing episodes carry the key at all.
        present = self._lookup(batch, self.depth_valid)
        present = (
            torch.zeros(b, dtype=torch.bool, device=device)
            if present is None
            else present.to(device=device, dtype=torch.bool).reshape(b)
        )
        # §22.5 depth DROPOUT: applied in training regardless of how much depth
        # the data actually has, so the policy never becomes depth-dependent for
        # basic motion and degrades gracefully when the stream is missing at
        # serving time.
        if self.training and self.depth_dropout > 0:
            # ⚠️ NOT `present &= ...`. `present` may ALIAS `batch["depth_valid"]`
            # -- `.to()` with a matching dtype and device returns the same tensor,
            # not a copy -- so an in-place op here permanently zeroes the loader's
            # own flag for that batch. With a cached or reused TensorDict that
            # corruption outlives the step, and it is invisible: the batch simply
            # claims it never had depth.
            present = present & (  # noqa: PLR6104
                torch.rand(b, device=device) >= self.depth_dropout
            )

        cond = self._lookup(batch, self.depth_camera_cond)
        if cond is None:
            cond = torch.zeros(
                b, len(self.depth_cameras), CAMERA_COND_DIM, device=device
            )

        blocks: list[Tensor] = []
        for index, camera in enumerate(self.depth_cameras):
            disparity = self._lookup(batch, (self.depth_key.format(camera=camera),))
            if disparity is None:
                # the key is missing from the batch entirely -- the NORMAL case
                # on today's data (§22.5). No embedding call at all.
                tokens = self.no_depth.expand(b, t, num_patches, -1)
            else:
                mask = self._lookup(batch, (self.depth_mask_key.format(camera=camera),))
                if mask is None:
                    msg = (
                        f"{self.depth_key.format(camera=camera)!r} present without "
                        f"{self.depth_mask_key.format(camera=camera)!r}: contract "
                        "§21.4/§22.4 requires the explicit validity mask"
                    )
                    raise KeyError(msg)
                patches = self._patchify_depth(disparity, mask)
                tokens = self.depth_patch_embedding(
                    torch.cat(
                        [
                            patches,
                            cond[:, index][:, None, None, :].expand(
                                b, t, num_patches, -1
                            ),
                        ],
                        dim=-1,
                    )
                )

            # per-sample substitution, for a MIXED batch (some episodes have
            # depth, some do not) and for the dropout above
            available = present[:, None, None, None]
            tokens = torch.where(available, tokens, self.no_depth.to(tokens.dtype))
            blocks.append(tokens)

        return torch.cat(blocks, dim=-2)

    def _state(self, batch: Any) -> Tensor:
        """`state.pose` in the model-facing 9D form, converting from storage if needed.

        Robot action space: the raw `(b, T, S, 13)` state, standardized in-graph
        with the train-split `state_standardizer` when one is configured.
        """
        state = self._get(batch, self.state)
        if self.robot:
            state = state.float()
            if self.state_standardizer is not None:
                state = self.state_standardizer(state)
            return state
        if self.convert_state_to_9d and state.shape[-1] == STATE_QUAT_DIM:
            return state_quat_to_9d(state)
        return state

    # ------------------------------------------------------------------- hand

    def _init_hand_standardizer(
        self, hparams: dict[str, Any], config: Any
    ) -> HandTokenStandardizer | None:
        """The hand token's in-graph affine, recorded SELF-CONTAINED in hparams.

        Raises:
            TypeError: when it is not a `HandTokenStandardizer`.
            ValueError: when its groups differ from `hand_groups`.
        """
        if not self.use_hand:
            return None
        std = init_hydra_param(hparams, "hand_standardizer", config)
        if std is None:
            logger.warning(
                "hand token without a hand_standardizer: raw /1000 columns "
                "(identity) -- only correct for checkpoints trained that way"
            )
            return None
        if not isinstance(std, HandTokenStandardizer):
            msg = f"hand_standardizer must be a HandTokenStandardizer, got {type(std)}"
            raise TypeError(msg)
        if std.groups != self.hand_groups:
            msg = f"hand_standardizer groups {std.groups} != hand_groups {self.hand_groups}"
            raise ValueError(msg)
        # the checkpoint re-creates it from these values, never from a stats
        # path that may be gone (the buffers also restore from the state_dict)
        hparams["hand_standardizer"] = {
            "_target_": "rmind.data.nero_robot.HandTokenStandardizer",
            "groups": list(std.groups),
            "mean": std.full_mean.tolist(),
            "std": std.full_std.tolist(),
            "source": std.source,
        }
        return std

    def hand_vector(self, batch: Any) -> Tensor | None:
        """Hand token input, `hand_valid` last in every row; None if absent.

        Untagged (`hand_sides == ()`): `(b, T, dim)` from `hand.*`. Sided:
        `(b, T, S, dim)` from `hand.{side}.*`, one row per entry of
        `hand_sides`. Serving/export hands the composed vector over as
        `hand_token`; training composes it from rbyte's per-group blocks
        (`compose_hand_token(s)`, the torch twin of `hf.TokenBlocks.compose`).
        A batch with no hand stream at all gives None (`no_hand` everywhere).

        Raises:
            ValueError: when the batch's hand columns are the other layout
                (per-side columns for an untagged model, or vice versa), or a
                sided batch lacks one of `hand_sides`, or a sided model in
                training mode gets a batch with no hand columns at all --
                silently substituting `no_hand` would train a hand_on run as a
                no-hand model.
        """
        vec = self._lookup(batch, (self.hand_token_key,))
        if vec is not None:
            vec = vec.float()
            want = 4 if self.hand_sides else 3
            if vec.dim() != want or (
                self.hand_sides and vec.shape[-2] != len(self.hand_sides)
            ):
                msg = (
                    f"`{self.hand_token_key}` is {tuple(vec.shape)}; this model takes "
                    + (
                        f"(b, T, {len(self.hand_sides)}, dim) for hand_sides "
                        f"{list(self.hand_sides)}"
                        if self.hand_sides
                        else "(b, T, dim) (one untagged hand token)"
                    )
                )
                raise ValueError(msg)
            return vec
        unsided = self._lookup(batch, (f"{self.hand_prefix}motor_ok",)) is not None
        sided = [
            side
            for side in ("left", "right")
            if self._lookup(batch, (f"{self.hand_prefix}{side}.motor_ok",)) is not None
        ]
        if self.hand_sides:
            missing = [s for s in self.hand_sides if s not in sided]
            if not sided and not unsided:
                if self.training:
                    # a datamodule that lost the hand columns would silently
                    # train this sided hand_on model as a no-hand model
                    msg = (
                        f"hand_sides {list(self.hand_sides)} but the training "
                        f"batch has no {self.hand_prefix}* columns at all"
                    )
                    raise ValueError(msg)
                return None
            if missing:
                msg = (
                    f"hand_sides {list(self.hand_sides)} but the batch has no "
                    f"{', '.join(f'{self.hand_prefix}{s}.*' for s in missing)} columns"
                    + (
                        f" (it carries the unsided {self.hand_prefix}*)"
                        if unsided
                        else ""
                    )
                )
                raise ValueError(msg)
            return compose_hand_tokens(
                batch, self.hand_groups, self.hand_sides, prefix=self.hand_prefix
            )
        if not unsided:
            if sided:
                # A bimanual rbyte row carries `hand.{left,right}.*`, never the
                # unsided `hand.*`.
                msg = (
                    f"hand token enabled (hand_groups={list(self.hand_groups)}) but "
                    f"the batch carries only per-side hand columns "
                    f"({', '.join(f'{self.hand_prefix}{s}.*' for s in sided)}), "
                    f"no `{self.hand_prefix}motor_ok`: set hand_sides "
                    f"{sided} for the per-side hand token (bimanual_causal)"
                )
                raise ValueError(msg)
            return None
        return compose_hand_token(batch, self.hand_groups, prefix=self.hand_prefix)

    def _no_hand_token(self) -> Tensor:
        """`(d,)` untagged, `(S, d)` sided (each side's tag added pre-norm)."""
        embed = self.hand_embedding
        raw: Tensor = self.no_hand
        tag = self._hand_side_tag()
        if tag is not None:
            if embed is not None and hasattr(embed, "embed_learned"):
                return embed.embed_learned(raw + tag)
            return raw + tag
        if embed is not None and hasattr(embed, "embed_learned"):
            return embed.embed_learned(raw)
        return raw

    def _hand_side_tag(self) -> Tensor | None:
        if self.hand_side_embedding is None:
            return None
        return self.hand_side_embedding.weight  # (S, d)

    def _embed_hand(self, vec: Tensor) -> Tensor:
        """`hand_embedding(vec)`, with the per-side tag added BEFORE the output
        norm when sided (`NormedTokenEmbedding`; after it for any other module),
        so the tag survives at the token's own scale."""
        embed = self.hand_embedding
        assert embed is not None  # noqa: S101
        tag = self._hand_side_tag()
        if tag is None:
            return embed(vec)
        if isinstance(embed, NormedTokenEmbedding):
            hidden = embed.mlp(embed.in_norm(vec)) + tag.to(vec.dtype)
            return embed.token_gain * embed.out_norm(hidden)
        return embed(vec) + tag.to(vec.dtype)

    def _hand_tokens(
        self,
        batch: Any,
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        token_norms: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """`(b, T, n, d)`: the embedded newest-sample hand token(s), or `no_hand`.

        `n = 1` untagged, `n = len(hand_sides)` sided (side-major, one token per
        side). `no_hand` replaces a token wherever ITS `hand_valid` is 0
        (refused/stale -- a KV stream cannot drop or hold a frame, P6), the hand
        stream is absent from the batch, or hand dropout fires (training only):
        per SAMPLE (`hand_dropout_sample`, the whole sequence) and per FRAME
        (`hand_dropout_frame`), so the policy never becomes hand-DEPENDENT for
        basic motion and degrades gracefully at serving. Sided, validity and
        both dropouts are drawn PER SIDE: one hand's refusal never touches the
        other's token.
        """
        assert self.hand_embedding is not None  # noqa: S101
        b, t = batch_size, num_frames
        n = len(self.hand_sides)
        vec = self.hand_vector(batch)
        no_hand = self._no_hand_token()  # (d,) | (S, d)
        if vec is None:
            out = no_hand.expand(b, t, *no_hand.shape)
            valid = torch.zeros(
                b, t, *([n] if n else []), dtype=torch.bool, device=device
            )
        else:
            vec = vec.to(device)
            valid = vec[..., -1] > 0.5  # noqa: PLR2004  (b, T) | (b, T, S)
            side = (n,) if n else ()
            if self.training and self.hand_dropout_sample > 0:
                keep = (
                    torch.rand(b, 1, *side, device=device) >= self.hand_dropout_sample
                )
                valid = valid & keep  # noqa: PLR6104  (never in place: may alias the batch)
            if self.training and self.hand_dropout_frame > 0:
                keep = torch.rand(b, t, *side, device=device) >= self.hand_dropout_frame
                valid = valid & keep  # noqa: PLR6104
            # fixed in-graph affine on the feature columns (hand_valid, read
            # above, and hand_age pass through unchanged); ONE pooled affine
            # for every side
            if self.hand_standardizer is not None:
                vec = self.hand_standardizer(vec)
            tokens = self._embed_hand(vec)
            out = torch.where(valid.unsqueeze(-1), tokens, no_hand.to(tokens.dtype))
        if token_norms is not None:
            with torch.no_grad():
                if bool(valid.any()):
                    token_norms["hand"] = out.detach()[valid].norm(dim=-1).mean()
                token_norms["no_hand"] = no_hand.detach().norm(dim=-1).mean()
                token_norms["hand_valid_frac"] = valid.float().mean()
                for i, name in enumerate(self.hand_sides):
                    token_norms[f"hand_valid_frac/{name}"] = (
                        valid[..., i].float().mean()
                    )
        return out if n else out.unsqueeze(-2)

    def _chunk(self, batch: Any) -> Tensor:
        """The action chunk in the model-facing 9D form.

        Contract §6.2 reserves `action.commanded` as an alias of
        `action.future_state`; rbyte currently materialises BOTH as
        byte-identical tensors (~199 MB of a ~470 MB TensorDict). This model
        reads exactly one path, so the duplicate is never paid for downstream --
        point `chunk` at whichever slot is populated.
        """
        chunk = self._get(batch, self.chunk)
        if self.convert_state_to_9d and chunk.shape[-1] == STATE_QUAT_DIM:
            return state_quat_to_9d(chunk)
        return chunk

    def _frame_tokens(  # ruff: ignore[too-many-locals]
        self, batch: Any, *, token_norms: dict[str, Tensor] | None = None
    ) -> Tensor:
        """Per-frame token blocks `(b, T, 1 + n_hand + 3P, d)` -- everything below the trunk.

        Factored out so a KV-cached one-frame decode step (the
        `PatchPolicyDecoderStep` equivalent) can run the identical pipeline on a
        single frame; nothing here is temporal.

        Raises:
            ValueError: when the built token count differs from the trunk's
                configured `tokens_per_frame`.
        """
        state = self._state(batch)  # (b, T, 2, 60)
        valid = self._get(batch, self.side_valid)  # (b, 2) bool
        cond = self._get(batch, self.camera_cond)  # (b, n_cam, 13)
        b, t = state.shape[0], state.shape[1]
        device = state.device

        goal = self._goal_features(batch, batch_size=b, device=device)

        per_camera: list[Tensor] = []
        for index, camera in enumerate(self.cameras):
            images = self._get(batch, (self.image_key.format(camera=camera),))
            patches = self._encode_images(images)  # (b, T, P, D)
            parts = [patches]
            if goal is not None:
                # same-index concat: obs patch (c, p) <-> goal patch (c, p)
                parts.append(goal[:, index].unsqueeze(1).expand(-1, t, -1, -1))
            elif self.goal_mode == "no_goal":
                # P7: the goal channel is always the learned "no goal supplied"
                parts.append(self.no_goal.to(patches.dtype).expand_as(patches))
            parts.append(
                cond[:, index][:, None, None, :].expand(b, t, patches.shape[-2], -1)
            )
            per_camera.append(torch.cat(parts, dim=-1))

        patch_tokens = self.patch_projection(
            torch.cat(per_camera, dim=-2)
        )  # (b, T, 3P, d)

        if self._calibrate_now:
            self._calibrate_token_gains(patch_tokens)

        # `side_valid` consumption #1: the invalid side is zeroed BEFORE it can
        # reach any token, and the mask itself is appended so "zero because
        # absent" is distinguishable from "zero because at the origin".
        mask = valid.to(state.dtype)  # (b, 2)
        masked_state = state * mask[:, None, :, None]
        state_input = torch.cat(
            [masked_state.flatten(-2, -1), mask[:, None, :].expand(b, t, NUM_SIDES)],
            dim=-1,
        )  # (b, T, 2 * A_state + 2)
        state_token = self.state_embedding(state_input).unsqueeze(-2)  # (b, T, 1, d)

        # ⚠️ depth goes BETWEEN the state token and the RGB patches, not after
        # them: the readout is the LAST token of a frame block, and appending
        # depth would make it a constant `no_depth` token on every sample without
        # a depth stream -- which is most of them (§22.5). This way the readout
        # stays the last `side_right` patch, exactly as in the depth-off model.
        depth_tokens = self._depth_tokens(
            batch, batch_size=b, num_frames=t, device=device
        )
        # the hand token (P6) sits right after the state token, for the same
        # reason: a missing hand (`no_hand`) can never become the readout
        hand_tokens = (
            self._hand_tokens(
                batch,
                batch_size=b,
                num_frames=t,
                device=device,
                token_norms=token_norms,
            )
            if self.use_hand
            else None
        )

        if token_norms is not None:
            with torch.no_grad():
                token_norms["patch"] = patch_tokens.detach().norm(dim=-1).mean()
                token_norms["state"] = state_token.detach().norm(dim=-1).mean()
                if depth_tokens is not None:
                    token_norms["depth"] = depth_tokens.detach().norm(dim=-1).mean()

        blocks = [state_token]
        if hand_tokens is not None:
            blocks.append(hand_tokens)
        if depth_tokens is not None:
            blocks.append(depth_tokens)
        # state first, so each frame block ends on a patch token (the readout)
        blocks.append(patch_tokens)
        tokens = torch.cat(blocks, dim=-2)
        configured = getattr(self.encoder, "tokens_per_frame", None)
        if isinstance(configured, int) and tokens.shape[-2] != configured:
            # a stale trunk `tokens_per_frame` (e.g. a sided hand token with
            # num_hand_tokens still 1) would tile the slot embedding / frame
            # mask wrong without raising
            msg = (
                f"built {tokens.shape[-2]} tokens per frame, the trunk is configured "
                f"for {configured} (state 1 + hand {self.n_hand_tokens} + patches)"
            )
            raise ValueError(msg)
        return tokens

    @torch.no_grad()
    def _calibrate_token_gains(self, patch_tokens: Tensor) -> None:
        """Set the state/hand token gains to the MEASURED patch-token RMS (once).

        The `fusion_goal_rms` lesson: a Monte-Carlo estimate of the scale was
        1.86x off, so the gain is measured on a real batch, after the frozen ViT
        and the patch projection, not guessed.
        """
        rms = float(patch_tokens.detach().float().pow(2).mean().sqrt())
        for module in (self.state_embedding, self.hand_embedding):
            if module is not None and hasattr(module, "calibrate"):
                module.calibrate(rms)
        if hasattr(self, "token_gain_calibrated"):
            self.token_gain_calibrated.fill_(1.0)
        self._calibrate_now = False

    def tokens_per_frame(self) -> int:
        """`1 state + n_hand_tokens + depth + n_cameras * P`; for budget checks."""
        return int(self.encoder.tokens_per_frame)

    def _features(
        self, batch: Any, *, token_norms: dict[str, Tensor] | None = None
    ) -> Tensor:
        """Per-frame readout features `(b, T, d)`."""
        tokens = self._frame_tokens(batch, token_norms=token_norms)
        _, num_frames, _, _ = tokens.shape
        embedding = self.encoder(
            rearrange(tokens, "b t k d -> b (t k) d"), num_frames=num_frames
        )
        frames = rearrange(embedding, "b (t k) d -> b t k d", t=num_frames)
        features = frames[:, :, -1]
        if token_norms is not None:
            with torch.no_grad():
                out = frames.detach().norm(dim=-1)  # (b, T, k)
                token_norms["out/state"] = out[..., 0].mean()
                if self.use_hand:
                    token_norms["out/hand"] = out[
                        ..., 1 : 1 + self.n_hand_tokens
                    ].mean()
                token_norms["out/readout"] = out[..., -1].mean()
                token_norms["out/patch"] = out[..., -self._num_patch_tokens() :].mean()
        if self.norm is not None:
            features = self.norm(features)
        return features

    def _num_patch_tokens(self) -> int:
        depth = 0
        if self.use_depth and self.depth_patch_grid is not None:
            depth = len(self.depth_cameras) * (
                self.depth_patch_grid[0] * self.depth_patch_grid[1]
            )
        return self.tokens_per_frame() - 1 - self.n_hand_tokens - depth

    # ------------------------------------------------------- VQ-BeT head

    def _per_side_features(self, features: Tensor) -> Tensor:
        """`(b, T, d)` -> `(b, T, 2, d)`: the shared readout plus a side identity."""
        sides = torch.arange(NUM_SIDES, device=features.device)
        return features.unsqueeze(-2) + self.side_embedding(sides)

    def _heads(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Code logits `(..., g, c)` and the offset table `(..., g, c, action_dim)`.

        `offset_mode='latent'`: the second output is the offset CONTEXT (the
        features themselves); the offset is computed per chosen codes in
        `_latent_offset`, because with `offset_code_conditioning` it depends on
        which codes were picked.
        """
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size
        code_logits = rearrange(
            self.code_head(features), "... (g c) -> ... g c", g=g, c=c
        )
        if self.offset_mode == "latent":
            # ONE offset in the tokenizer's latent space `(..., L)`, decoded
            # through the frozen tokenizer decoder (`_decode`)
            return code_logits, features
        offsets = rearrange(
            self.offset_head(features), "... (g c a) -> ... g c a", g=g, c=c
        )
        return code_logits, offsets

    @staticmethod
    def _gather_offset(offsets: Tensor, codes: Tensor) -> Tensor:
        index = codes[..., None, None].expand(*codes.shape, 1, offsets.shape[-1])
        # https://arxiv.org/pdf/2403.03181 Figure 2.
        return offsets.gather(-2, index).squeeze(-2).sum(dim=-2)

    def _offset(self, offsets: Tensor, codes: Tensor) -> Tensor:
        offset = self._gather_offset(offsets, codes)
        if self.offset_scale is None:
            return offset
        return torch.tanh(offset / self.offset_scale) * self.offset_scale

    def _sample_codes(self, code_logits: Tensor) -> Tensor:
        *batch, g, c = code_logits.shape
        if self.sample_codes:
            return rearrange(
                torch.multinomial(code_logits.softmax(dim=-1).reshape(-1, c), 1),
                "(b g) 1 -> b g",
                g=g,
            ).reshape(*batch, g)
        return code_logits.argmax(dim=-1)

    def _predict_chunk(self, features: Tensor) -> Tensor:
        """`(..., d)` -> STANDARDISED chunk `(..., 2, horizon, action_features)`."""
        return self._predict_chunk_and_codes(features)[0]

    def _predict_chunk_and_codes(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """`(..., d)` -> (STANDARDISED `(..., 2, H, A)` chunk, `(..., 2, g)` codes)."""
        code_logits, offsets = self._heads(self._per_side_features(features))
        codes = self._sample_codes(code_logits)
        if self.robot:
            *batch, g = codes.shape
            chunk = self._decode(
                offsets.reshape(-1, *offsets.shape[len(batch) :]), codes.reshape(-1, g)
            )
            return chunk.reshape(*batch, *chunk.shape[1:]), codes
        offset = self._offset(offsets, codes)
        return (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # ruff: ignore[private-member-access]
        ), codes

    def _latent_offset(self, context: Tensor, codes: Tensor) -> Tensor:
        """Latent offset `(n, L)` for `codes (n, g)` from the offset context `(n, d)`."""
        inputs = context
        if self.offset_code_conditioning:
            z_q = self.tokenizer.lookup(codes).detach().to(context.dtype)
            inputs = torch.cat([context, z_q], dim=-1)
        offset = self.offset_head(inputs)
        if self.offset_scale is not None:
            offset = torch.tanh(offset / self.offset_scale) * self.offset_scale
        return offset

    def _decode(self, offsets: Tensor, codes: Tensor) -> Tensor:
        """Robot rows: codes `(n, g)` + offsets -> STANDARDIZED `(n, H, A)`.

        `offsets` is the offset table (table mode) or the offset context
        (latent mode, see `_heads`).
        """
        tokenizer = self.tokenizer
        if self.offset_mode == "latent":
            offset = self._latent_offset(offsets, codes)
            return tokenizer.decode_latent(tokenizer.lookup(codes) + offset)
        flat = tokenizer.invert(codes) + self._offset(offsets, codes)
        return flat.reshape(-1, tokenizer.action_horizon, tokenizer.action_features)

    @staticmethod
    def _elementwise(loss: Module, pred: Tensor, target: Tensor) -> Tensor:
        """The configured offset loss, unreduced (for the `action_is_pad` mask)."""
        if isinstance(loss, nn.SmoothL1Loss):
            return F.smooth_l1_loss(pred, target, beta=loss.beta, reduction="none")
        if isinstance(loss, nn.MSELoss):
            return F.mse_loss(pred, target, reduction="none")
        if isinstance(loss, nn.L1Loss):
            return F.l1_loss(pred, target, reduction="none")
        msg = f"offset loss {type(loss).__name__} has no unreduced form here"
        raise TypeError(msg)

    def _masked_offset_loss(self, pred: Tensor, target: Tensor, real: Tensor) -> Tensor:
        w = real.unsqueeze(-1).to(pred.dtype).expand_as(pred)
        err = self._elementwise(self.losses["offset"], pred, target)
        return (err * w).sum() / w.sum().clamp_min(1.0)

    def _anchor_rows(self, batch: Any, row_valid: Tensor) -> Tensor:
        """RAW per-frame state `(n, A)` of every valid row -- the relative anchor."""
        state = self._get(batch, self.state).float()  # (b, T, S, A)
        return state.reshape(-1, state.shape[-1])[row_valid]

    def _absolute(self, chunk: Tensor, anchor: Tensor, side: Tensor) -> Tensor:
        """STANDARDIZED `(n, H, A)` rows -> ABSOLUTE robot units (rad, counts/1000)."""
        std = self.tokenizer.standardizer
        mean = std.mean[side].unsqueeze(1).to(chunk.dtype)  # (n, 1, A)
        scale = std.std[side].unsqueeze(1).to(chunk.dtype)
        raw = chunk * scale + mean
        return to_absolute(
            raw.unsqueeze(-2), anchor.unsqueeze(-2), self.relative_mode
        ).squeeze(-2)

    # ------------------------------------------------------------------ loss

    def _robot_rows(self, batch: Any, features: Tensor) -> dict[str, Tensor]:
        """Everything the robot losses/metrics need, per valid (b, frame, side) row."""
        tokenizer = self.tokenizer
        b, t = features.shape[0], features.shape[1]
        valid = self._get(batch, self.side_valid).bool()  # (b, S)
        row_valid = valid[:, None, :].expand(b, t, NUM_SIDES).reshape(-1)
        side = (
            torch
            .arange(NUM_SIDES, device=features.device)
            .expand(b, t, NUM_SIDES)
            .reshape(-1)[row_valid]
        )
        with torch.no_grad():
            target, real = tokenizer.prepare(batch)  # (n, H, A), (n, H)
            target_codes = tokenizer.encode(target)  # (n, g)
        side_features = self._per_side_features(features).reshape(
            -1, features.shape[-1]
        )[row_valid]
        code_logits, offsets = self._heads(side_features)
        return {
            "row_valid": row_valid,
            "side": side,
            "target": target,
            "real": real,
            "target_codes": target_codes,
            "code_logits": code_logits,
            "offsets": offsets,
        }

    def _compute_robot_metrics(  # ruff: ignore[too-many-locals]
        self, batch: Any, *, token_norms: dict[str, Tensor] | None = None
    ) -> TensorDict:
        tokenizer = self.tokenizer
        features = self._features(batch, token_norms=token_norms)  # (b, T, d)
        rows = self._robot_rows(batch, features)
        target, real = rows["target"], rows["real"]
        target_codes, code_logits = rows["target_codes"], rows["code_logits"]
        offsets = rows["offsets"]
        g = tokenizer.quantizer.num_quantizers

        losses: dict[str, Tensor] = {
            f"code_{q}": self.losses["code"](code_logits[:, q], target_codes[:, q])
            for q in range(g)
        }
        codes = self._sample_codes(code_logits)
        if self.teacher_force_offset:
            predicted = self._decode(offsets, target_codes)
        else:
            predicted = self._decode(offsets, codes)
        losses["offset"] = self._masked_offset_loss(predicted, target, real)

        metrics: dict[str, Tensor] = {
            "valid_rows": rows["row_valid"].sum().to(features.dtype)
        }
        with torch.no_grad():
            argmax = code_logits.argmax(dim=-1)
            decoded = self._decode(offsets, argmax)
            metrics["offset_argmax_recon"] = self._masked_offset_loss(
                decoded, target, real
            )
            correct = argmax == target_codes
            for q in range(g):
                metrics[f"code_acc_{q}"] = correct[:, q].float().mean()
            metrics["code_acc_joint"] = correct.all(dim=-1).float().mean()
            metrics["code_acc_dependence"] = metrics[
                "code_acc_joint"
            ] / correct.float().mean(dim=0).prod().clamp_min(1e-8)
            if self.quality_metrics:
                anchor = self._anchor_rows(batch, rows["row_valid"])
                pred_abs = self._absolute(decoded, anchor, rows["side"])
                gt_abs = self._absolute(target, anchor, rows["side"])
                confidence = code_confidence_metrics(code_logits, argmax)
                metrics |= confidence
                metrics |= horizon_ev_metrics(pred_abs, gt_abs, real)
                metrics |= grasp_event_metrics(pred_abs, gt_abs, real)
                if self.offset_mode == "latent":
                    # the SERVING offset: the argmax codes' (conditioned) offset
                    z_q = tokenizer.lookup(argmax)
                    offset = self._latent_offset(offsets, argmax)
                    metrics["offset_to_code_norm"] = (
                        offset.norm(dim=-1) / z_q.norm(dim=-1).clamp_min(1e-8)
                    ).mean()
                metrics |= alarm_metrics(
                    code_usage=confidence,
                    codebook_size=tokenizer.quantizer.codebook_size,
                    pred=pred_abs,
                    target=gt_abs,
                    real=real,
                    token_norms=token_norms,
                    features=features,
                )
        return TensorDict(
            {"policy": {"loss": losses, "metric": metrics}}, batch_size=[]
        )

    # ---------------------------------------------------------- reliance (P9)

    def _hand_override(self, batch: Any, how: str, shift: int = 10) -> dict[str, Any]:
        """A copy of `batch` whose `hand_token` is ablated `how`.

        `no_hand`: every frame refused (the learned no_hand everywhere);
        `shuffled`: hand vectors permuted across the batch;
        `shift_plus`/`shift_minus`: hand vectors moved +-`shift` frames in time
        (1 s at the 10 Hz frame grid), frames shifted in from outside the
        window refused;
        `no_hand_<side>` (sided models only): that side's token refused on every
        frame, the other side's untouched.

        Works on both layouts (`(b, T, dim)` and the sided `(b, T, S, dim)`):
        every ablation but `no_hand_<side>` acts on all sides together.
        """
        vec = self.hand_vector(batch)
        if vec is None:
            msg = "reliance metrics need hand inputs in the batch"
            raise KeyError(msg)
        match how:
            case "no_hand":
                alt = torch.zeros_like(vec)
            case _ if how.startswith("no_hand_") and how[8:] in self.hand_sides:
                alt = vec.clone()
                alt[:, :, self.hand_sides.index(how[8:])] = 0
            case "shuffled":
                perm = torch.roll(torch.arange(vec.shape[0], device=vec.device), 1)
                alt = vec[perm]
            case "shift_plus" | "shift_minus":
                k = shift if how == "shift_plus" else -shift
                alt = torch.roll(vec, k, dims=1)
                if k > 0:
                    alt[:, :k] = 0
                else:
                    alt[:, k:] = 0
            case _:
                msg = f"unknown hand ablation {how!r}"
                raise ValueError(msg)
        return dict(batch) | {self.hand_token_key: alt}

    @torch.no_grad()
    def _eval_scores(self, batch: Any) -> dict[str, Tensor]:
        features = self._features(batch)
        rows = self._robot_rows(batch, features)
        logits, target_codes = rows["code_logits"], rows["target_codes"]
        nll = torch.stack([
            F.cross_entropy(logits[:, q], target_codes[:, q])
            for q in range(logits.shape[1])
        ]).sum()
        decoded = self._decode(rows["offsets"], logits.argmax(dim=-1))
        anchor = self._anchor_rows(batch, rows["row_valid"])
        pred = self._absolute(decoded, anchor, rows["side"])
        gt = self._absolute(rows["target"], anchor, rows["side"])
        real = rows["real"]
        window = grasp_window_mask(gt, real)
        out = {
            "code_nll": nll,
            "finger_ev": per_axis_ev(pred, gt, real)[list(FINGER_AXES)].mean(),
            "loss": self._masked_offset_loss(decoded, rows["target"], real),
        }
        if bool(window.any()):
            out["finger_ev_grasp"] = per_axis_ev(pred, gt, window)[
                list(FINGER_AXES)
            ].mean()
        return out

    @torch.no_grad()
    def reliance_metrics_for(self, batch: Any) -> dict[str, Tensor]:
        """Hand reliance deltas (nutron_act port): ablated minus clean, >0 = relied on."""
        base = self._eval_scores(batch)
        out: dict[str, Tensor] = {}
        hows = ["no_hand", "shuffled", "shift_plus", "shift_minus"]
        hows += [f"no_hand_{side}" for side in self.hand_sides]
        for how in hows:
            alt = self._eval_scores(self._hand_override(batch, how))
            out[f"reliance/{how}/code_nll"] = alt["code_nll"] - base["code_nll"]
            out[f"reliance/{how}/offset"] = alt["loss"] - base["loss"]
            out[f"reliance/{how}/finger_ev"] = base["finger_ev"] - alt["finger_ev"]
            if "finger_ev_grasp" in base and "finger_ev_grasp" in alt:
                out[f"reliance/{how}/finger_ev_grasp"] = (
                    base["finger_ev_grasp"] - alt["finger_ev_grasp"]
                )
        return out

    def _compute_metrics(  # ruff: ignore[too-many-locals]
        self, batch: Any, *, token_norms: dict[str, Tensor] | None = None
    ) -> TensorDict:
        if self.robot:
            return self._compute_robot_metrics(batch, token_norms=token_norms)
        tokenizer = self.tokenizer
        features = self._features(batch, token_norms=token_norms)  # (b, T, d)
        chunk = self._chunk(batch)  # (b, T, H, 2, 60)
        valid = self._get(batch, self.side_valid)  # (b, 2)

        b, t = features.shape[0], features.shape[1]
        # (b, T, 2, H, A) -> select valid (batch, frame, side) rows
        per_side_chunk = chunk.permute(0, 1, 3, 2, 4)
        row_valid = valid[:, None, :].expand(b, t, NUM_SIDES).reshape(-1)
        flat_chunk = per_side_chunk.reshape(-1, *per_side_chunk.shape[-2:])[row_valid]

        with torch.no_grad():
            target_codes = tokenizer(flat_chunk)  # (n, g)
            target = tokenizer._normalize(flat_chunk.flatten(-2, -1))  # ruff: ignore[private-member-access]

        side_features = self._per_side_features(features).reshape(
            -1, features.shape[-1]
        )[row_valid]
        code_logits, offsets = self._heads(side_features)

        losses: dict[str, Tensor] = {}
        for q in range(tokenizer.quantizer.num_quantizers):
            losses[f"code_{q}"] = self.losses["code"](
                code_logits[..., q, :], target_codes[..., q]
            )

        codes = self._sample_codes(code_logits)
        sampled_chunk = tokenizer.invert(codes) + self._offset(offsets, codes)

        if self.teacher_force_offset:
            predicted_chunk = tokenizer.invert(target_codes) + self._offset(
                offsets, target_codes
            )
        else:
            predicted_chunk = sampled_chunk

        losses["offset"] = self.losses["offset"](predicted_chunk, target)

        with torch.no_grad():
            metrics: dict[str, Tensor] = {
                "offset_sampled_recon": self.losses["offset"](
                    sampled_chunk.detach(), target
                ),
                "valid_rows": row_valid.sum().to(features.dtype),
            }
            # ⚠️ contract §5.5: translation and rotation, ALWAYS separately.
            if getattr(tokenizer, "has_pose_layout", False):
                shape = (-1, tokenizer.action_horizon, tokenizer.action_features)
                metrics |= pose_error_metrics(
                    tokenizer._denormalize(sampled_chunk.detach()).reshape(shape),  # ruff: ignore[private-member-access]
                    flat_chunk.reshape(shape),
                )

        return TensorDict(
            {"policy": {"loss": losses, "metric": metrics}}, batch_size=[]
        )

    def _step(self, batch: Any, prefix: str) -> STEP_OUTPUT:
        token_norms: dict[str, Tensor] | None = {} if self.quality_metrics else None
        metrics = self._compute_metrics(batch, token_norms=token_norms)
        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # ruff: ignore[in-dict-keys]
        metrics["loss", "total"] = losses.sum(reduce=True)
        self.log_dict(
            {
                "/".join([prefix, *k]): v
                for k, v in metrics.detach().items(
                    include_nested=True, leaves_only=True
                )
            },
            sync_dist=True,
        )
        if token_norms:
            self.log_dict(
                {
                    f"quality/token_norm/{prefix}/{name}": value
                    for name, value in token_norms.items()
                },
                sync_dist=True,
            )
        return {"loss": metrics["loss", "total"]}

    def compute_metrics(
        self, batch: Any, *, token_norms: dict[str, Tensor] | None = None
    ) -> TensorDict:
        """Public `_compute_metrics` (+ the one-shot token-gain calibration)."""
        if self.training:
            self._arm_calibration()
        return self._compute_metrics(batch, token_norms=token_norms)

    def _arm_calibration(self) -> None:
        if self.calibrate_token_gain and not bool(self.token_gain_calibrated):
            self._calibrate_now = True

    @override
    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        self._arm_calibration()
        return self._step(batch, "train")

    @override
    def validation_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        if self.trainer.sanity_checking:
            return {
                "loss": self._compute_metrics(batch)["policy", "loss"].sum(reduce=True)
            }
        out = self._step(batch, "val")
        if self.reliance_metrics and self.robot and self.use_hand:
            self.log_dict(
                {f"val/{k}": v for k, v in self.reliance_metrics_for(batch).items()},
                sync_dist=True,
            )
        return out

    @override
    def on_train_start(self) -> None:
        """Training-health alarm: the cosine schedule has no clamp past its end."""
        if self.lr_scheduler is None or self.trainer is None:
            return
        planned = self.lr_scheduler.scheduler.model_dump().get("num_training_steps")
        actual = self.trainer.estimated_stepping_batches
        if planned is not None and actual and int(planned) != int(actual):
            from structlog import get_logger  # noqa: PLC0415

            get_logger(__name__).warning(
                "lr_total_steps does not match the trainer's step count",
                lr_total_steps=planned,
                estimated_stepping_batches=actual,
            )
            self.lr_steps_mismatch = True

    @override
    def forward(self, batch: Any) -> TensorDict:
        """Newest frame's bimanual action chunk, `(b, 2, horizon, action_features)`."""
        chunk = self._predict_chunk(self._features(batch)[:, -1])
        return TensorDict({"policy": {"action": chunk}}, batch_size=[])

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer is None:
            msg = "optimizer not specified"
            raise ValueError(msg)

        match self.optimizer.target:
            case optimizers.SelectiveAdamW:
                optimizer = self.optimizer.instantiate(module=self)
            case _:
                optimizer = self.optimizer.instantiate(params=self.parameters())

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler.scheduler.instantiate(optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler} | self.lr_scheduler.model_dump(
                exclude={"scheduler"}
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        return {"optimizer": optimizer}
