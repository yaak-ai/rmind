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
from typing import Annotated, Any, final, override

import pytorch_lightning as pl
import torch
from einops import rearrange
from pydantic import Field, InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import Module
from torch.optim import Optimizer

from rmind.components import optimizers
from rmind.components.containers import ModuleDict
from rmind.config import HydraConfig, init_hydra_param
from rmind.data.nero import (
    CAMERA_COND_DIM,
    NUM_SIDES,
    STATE_QUAT_DIM,
    pose_error_metrics,
    state_quat_to_9d,
)
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.models.control_transformer import PredictionConfig
from rmind.utils._wandb import LoadableFromArtifact

__all__ = ["NeroPatchPolicy"]

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
        }

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.prediction_config = prediction_config

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
        if not self.use_goal_image:
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
        """`state.pose` in the model-facing 9D form, converting from storage if needed."""
        state = self._get(batch, self.state)
        if self.convert_state_to_9d and state.shape[-1] == STATE_QUAT_DIM:
            return state_quat_to_9d(state)
        return state

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

    def _frame_tokens(self, batch: Any) -> Tensor:  # ruff: ignore[too-many-locals]
        """Per-frame token blocks `(b, T, 3P + 1, d)` -- everything below the trunk.

        Factored out so a KV-cached one-frame decode step (the
        `PatchPolicyDecoderStep` equivalent) can run the identical pipeline on a
        single frame; nothing here is temporal.
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
            parts.append(
                cond[:, index][:, None, None, :].expand(b, t, patches.shape[-2], -1)
            )
            per_camera.append(torch.cat(parts, dim=-1))

        patch_tokens = self.patch_projection(
            torch.cat(per_camera, dim=-2)
        )  # (b, T, 3P, d)

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
        if depth_tokens is not None:
            return torch.cat([state_token, depth_tokens, patch_tokens], dim=-2)

        # state first, so each frame block ends on a patch token (the readout)
        return torch.cat([state_token, patch_tokens], dim=-2)

    def _features(self, batch: Any) -> Tensor:
        """Per-frame readout features `(b, T, d)`."""
        tokens = self._frame_tokens(batch)
        _, num_frames, _, _ = tokens.shape
        embedding = self.encoder(
            rearrange(tokens, "b t k d -> b (t k) d"), num_frames=num_frames
        )
        features = rearrange(embedding, "b (t k) d -> b t k d", t=num_frames)[:, :, -1]
        if self.norm is not None:
            features = self.norm(features)
        return features

    # ------------------------------------------------------- VQ-BeT head

    def _per_side_features(self, features: Tensor) -> Tensor:
        """`(b, T, d)` -> `(b, T, 2, d)`: the shared readout plus a side identity."""
        sides = torch.arange(NUM_SIDES, device=features.device)
        return features.unsqueeze(-2) + self.side_embedding(sides)

    def _heads(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Code logits `(..., g, c)` and the offset table `(..., g, c, action_dim)`."""
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size
        code_logits = rearrange(
            self.code_head(features), "... (g c) -> ... g c", g=g, c=c
        )
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
        code_logits, offsets = self._heads(self._per_side_features(features))
        codes = self._sample_codes(code_logits)
        offset = self._offset(offsets, codes)
        return (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # ruff: ignore[private-member-access]
        )

    # ------------------------------------------------------------------ loss

    def _compute_metrics(self, batch: Any) -> TensorDict:  # ruff: ignore[too-many-locals]
        tokenizer = self.tokenizer
        features = self._features(batch)  # (b, T, d)
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
        metrics = self._compute_metrics(batch)
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
        return {"loss": metrics["loss", "total"]}

    @override
    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        return self._step(batch, "train")

    @override
    def validation_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        if self.trainer.sanity_checking:
            return {
                "loss": self._compute_metrics(batch)["policy", "loss"].sum(reduce=True)
            }
        return self._step(batch, "val")

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
