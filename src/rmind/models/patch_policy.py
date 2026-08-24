from collections.abc import Mapping
from itertools import chain
from typing import Annotated, Any, final, override

import pytorch_lightning as pl
import torch
from einops import rearrange
from pydantic import Field, InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils._pytree import MappingKey  # noqa: PLC2701
from torch.utils.checkpoint import checkpoint

from rmind.components import optimizers
from rmind.components.containers import ModuleDict
from rmind.components.objectives.base import ObjectivePredictionKey, Prediction
from rmind.config import HydraConfig, init_hydra_param
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.models.control_transformer import PredictionConfig
from rmind.utils._wandb import LoadableFromArtifact
from rmind.utils.profiling import maybe_profile
from rmind.utils.pytree import key_get_default

type Path = tuple[str, ...]


def block_causal_mask(
    num_frames: int, tokens_per_frame: int, *, device: torch.device | None = None
) -> Tensor:
    """Bool mask (True = blocked) over the flattened `num_frames * tokens_per_frame`
    sequence: full bidirectional attention within a frame, causal across frames
    (https://arxiv.org/pdf/2607.18236).
    """
    frames = (
        torch.arange(num_frames * tokens_per_frame, device=device) // tokens_per_frame
    )
    return frames[None, :] > frames[:, None]


@final
class TransformerBlock(nn.Module):
    """Pre-LN GPT block (minGPT-style, as used by the VQ-BeT policy trunk)."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        dim_model: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 4,
    ) -> None:
        super().__init__()

        self.attn_norm = nn.LayerNorm(dim_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.resid_drop = nn.Dropout(resid_dropout)
        self.mlp_norm = nn.LayerNorm(dim_model)
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, hidden_layer_multiplier * dim_model),
            nn.GELU(),
            nn.Linear(hidden_layer_multiplier * dim_model, dim_model),
            nn.Dropout(mlp_dropout),
        )

    @override
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        attn_out, _ = self.attn(
            *(self.attn_norm(x),) * 3, attn_mask=mask, need_weights=False
        )
        # NOTE: no in-place ops on the residual stream (autograd + checkpointing)
        h = x + self.resid_drop(attn_out)
        return h + self.mlp(self.mlp_norm(h))


@final
class BlockCausalTransformer(nn.Module):
    """Flattened-sequence encoder over `num_frames * tokens_per_frame` tokens with a
    learned 1D positional embedding and a block-causal attention mask
    (https://arxiv.org/pdf/2607.18236).

    `checkpoint` sets the activation-checkpointing policy used during training:
    `True` wraps every block, `False` none, an int `k` every k-th block. Wrapping
    a block trades a full extra forward of it for the memory of its activations
    -- worth paying only when memory is actually scarce.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        max_sequence_length: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 4,
        checkpoint: bool | Annotated[int, Field(ge=1)] = True,
    ) -> None:
        super().__init__()

        # normalized to "checkpoint every k-th block", 0 = never
        match checkpoint:
            case bool():
                self._checkpoint_every: int = 1 if checkpoint else 0
            case _:
                self._checkpoint_every = checkpoint

        self.position_embedding = nn.Embedding(max_sequence_length, dim_model)
        nn.init.trunc_normal_(
            self.position_embedding.weight, mean=0.0, std=0.02, a=-0.04, b=0.04
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                mlp_dropout=mlp_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)

    def _should_checkpoint(self, index: int) -> bool:
        return (
            self.training
            and self._checkpoint_every > 0
            and index % self._checkpoint_every == 0
        )

    @override
    def forward(self, src: Tensor, *, num_frames: int) -> Tensor:
        _, seq_len, _ = src.shape
        x = src + self.position_embedding(torch.arange(seq_len, device=src.device))
        mask = block_causal_mask(num_frames, seq_len // num_frames, device=src.device)
        # NOTE: deliberately not `run_layer_stack` -- that helper is shared with
        # ControlTransformer's encoder/decoder, so an all-or-nothing checkpointing
        # policy there is not the right one here
        for i, layer in enumerate(self.layers):
            x = (
                checkpoint(layer, x, mask, use_reentrant=False)
                if self._should_checkpoint(i)
                else layer(x, mask)
            )
        return self.norm(x)


class PatchPolicy(pl.LightningModule, LoadableFromArtifact):
    """Patch Policy (https://arxiv.org/pdf/2607.18236) with a VQ-BeT action head.

    Per frame: frozen ViT patch features `(P, D)` -- concatenated across `cameras`
    when more than one is configured, each camera contributing its own `P` patches
    -- get the frozen waypoints-tokenizer latent `g_t` (that frame's goal vector)
    concatenated to every patch token -- the paper's `T x P x (D + G)` scheme --
    then projected to the policy width. An embedded speed token is prepended, the
    `T x (P + 1)` sequence is flattened,
    given a learned 1D positional embedding, and run through a block-causal
    transformer (bidirectional intra-frame, causal inter-frame). Each frame's LAST
    token predicts that frame's action chunk with the VQ-BeT joint head from
    `JointPolicyObjective` (frozen residual-VQ chunk tokenizer; focal code loss +
    teacher-forced L1 offset).

    Readout position (opt-in, `use_readout_token`): by default the last token of a
    frame is the last image patch -- fragile, and with multiple cameras arbitrary
    (it depends on which camera happens to be last). With `use_readout_token` the
    frame layout becomes

        [ speed, camera patches..., register_0..register_{R-1}, READOUT ]

    so `[:, :, -1]` picks a LEARNED readout token instead of a picture of the
    road. The `num_register_tokens` register tokens exist to absorb the
    attention-sink role (https://arxiv.org/abs/2309.16588) and are never read
    from -- they sit BEFORE the readout token precisely so the readout stays
    last. This changes `tokens_per_frame` (257 -> 257 + R + 1), which the
    encoder/serving configs must mirror (`CausalFrameTransformer.tokens_per_frame`,
    `BlockCausalTransformer.max_sequence_length`, KV-cache/export geometry).
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        input_transform: HydraConfig[Module] | InstanceOf[Module],
        image_encoder: HydraConfig[Module] | InstanceOf[Module],
        goal_encoder: HydraConfig[Module] | InstanceOf[Module],
        patch_projection: HydraConfig[Module] | InstanceOf[Module],
        speed_tokenizer: HydraConfig[Module] | InstanceOf[Module],
        speed_embedding: HydraConfig[Module] | InstanceOf[Module],
        # BlockCausalTransformer, or the decoder-only
        # components.transformer.causal_frame.CausalFrameTransformer (same
        # `forward(src, *, num_frames)` contract, plus a KV-cached `step`)
        encoder: HydraConfig[Module] | InstanceOf[Module],
        tokenizer: HydraConfig[Module] | InstanceOf[Module],
        code_head: HydraConfig[Module] | InstanceOf[Module],
        offset_head: HydraConfig[Module] | InstanceOf[Module],
        losses: HydraConfig[ModuleDict] | InstanceOf[ModuleDict],
        norm: HydraConfig[Module] | InstanceOf[Module] | None = None,
        cameras: tuple[str, ...] = ("cam_front_left",),
        speed: Path = ("continuous", "speed"),
        waypoints: Path = ("context", "waypoints"),
        chunk: Path = ("joint_actions",),
        sample_codes: bool = True,
        teacher_force_offset: bool = True,
        offset_scale: float | None = None,
        neighbor_smoothing_tau: float | None = None,
        fusion_norm: bool = False,
        fusion_goal_rms: float | None = None,
        use_readout_token: bool = False,
        num_register_tokens: int = 0,
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
        prediction_config: Annotated[
            PredictionConfig, Field(default_factory=PredictionConfig)
        ],
    ) -> None:
        super().__init__()

        hparams: dict[str, Any] = {}

        self.input_transform = init_hydra_param(
            hparams, "input_transform", input_transform
        )
        # frozen feature extractors: never train, never leave eval mode (see train())
        self.image_encoder = (
            init_hydra_param(hparams, "image_encoder", image_encoder)
            .requires_grad_(False)  # noqa: FBT003
            .eval()
        )
        self.goal_encoder = (
            init_hydra_param(hparams, "goal_encoder", goal_encoder)
            .requires_grad_(False)  # noqa: FBT003
            .eval()
        )
        self.tokenizer = (
            init_hydra_param(hparams, "tokenizer", tokenizer)
            .requires_grad_(False)  # noqa: FBT003
            .eval()
        )

        self.patch_projection = init_hydra_param(
            hparams, "patch_projection", patch_projection
        )
        self.speed_tokenizer = init_hydra_param(
            hparams, "speed_tokenizer", speed_tokenizer
        )
        self.speed_embedding = init_hydra_param(
            hparams, "speed_embedding", speed_embedding
        )
        self.encoder: Module = init_hydra_param(hparams, "encoder", encoder)
        self.code_head = init_hydra_param(hparams, "code_head", code_head)
        self.offset_head = init_hydra_param(hparams, "offset_head", offset_head)
        self.losses: ModuleDict = init_hydra_param(hparams, "losses", losses)
        self.norm: Module | None = init_hydra_param(hparams, "norm", norm)

        self.cameras: tuple[str, ...] = cameras
        self.speed: Path = speed
        self.waypoints: Path = waypoints
        self.chunk: Path = chunk
        self.sample_codes = sample_codes
        self.teacher_force_offset = teacher_force_offset
        self.offset_scale = offset_scale
        # neighbour-aware label smoothing (OPT-IN, default None = off): spread the
        # code loss's smoothing mass by decoded-action distance instead of
        # uniformly. See _neighbor_smoothing_targets for the construction and
        # FocalLoss.forward for how the target replaces the uniform term.
        self.neighbor_smoothing_tau = neighbor_smoothing_tau
        if neighbor_smoothing_tau is not None:
            # deferred to dodge a circular import (loss.py has no such cycle
            # today, but this mirrors load_for_export's local-import pattern)
            from rmind.components.loss import FocalLoss  # noqa: PLC0415

            if neighbor_smoothing_tau <= 0.0:
                msg = (
                    f"neighbor_smoothing_tau must be > 0, got {neighbor_smoothing_tau}"
                )
                raise ValueError(msg)
            if not isinstance(self.losses["code"], FocalLoss):
                msg = (
                    "neighbor_smoothing_tau requires losses['code'] to be a "
                    "rmind.components.loss.FocalLoss (it is the only code loss "
                    f"accepting a smoothing target), got {type(self.losses['code'])}"
                )
                raise TypeError(msg)
        hparams |= {
            "cameras": cameras,
            "speed": speed,
            "waypoints": waypoints,
            "chunk": chunk,
            "sample_codes": sample_codes,
            "teacher_force_offset": teacher_force_offset,
            "offset_scale": offset_scale,
            "neighbor_smoothing_tau": neighbor_smoothing_tau,
            "fusion_norm": fusion_norm,
            "use_readout_token": use_readout_token,
            "num_register_tokens": num_register_tokens,
        }

        # opt-in dedicated readout + register tokens (default off: existing arms
        # and checkpoints keep the last-image-patch readout unchanged)
        self.use_readout_token = use_readout_token
        self.num_register_tokens = num_register_tokens
        self._init_readout_tokens(
            use_readout_token=use_readout_token, num_register_tokens=num_register_tokens
        )

        # scale-balanced feature fusion: LayerNorm + learnable gain on the patch
        # side (encoder-agnostic scale; DINO token-norm spread is negligible so
        # nothing informative is lost), and a learnable gain on the goal side
        # initialized to 1/RMS so that RMS(gain * z_q) ~= RMS(LN(patches)) ~= 1 --
        # calibrated from the frozen RVQ codebooks (seeded, data-free, identical
        # across DDP ranks). Per-sample code-norm information passes through
        # untouched.
        #
        # CAVEAT on the estimate: the codes below are drawn INDEPENDENTLY and
        # UNIFORMLY per quantizer, whereas a real z_q is a residual-VQ tuple
        # (level q+1 corrects level q's residual) with strongly non-uniform code
        # usage. Measured on gzxgumtf:v9 the estimate is 1.86x too large
        # (0.143 vs 0.077 on real data), which lands the goal stream ~2x quieter
        # than intended. Prefer passing a measured value where it is known.
        if fusion_norm:
            self._init_fusion_norm(fusion_goal_rms)
        else:
            self.fusion_patch_norm = None
            self.fusion_patch_gain = None
            self.fusion_goal_gain = None

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.prediction_config = prediction_config

        self.save_hyperparameters(hparams)

    @override
    def train(self, mode: bool = True) -> "PatchPolicy":
        super().train(mode)
        self.image_encoder.eval()
        self.goal_encoder.eval()
        self.tokenizer.eval()
        return self

    @staticmethod
    def _get(
        inputs: Mapping[str, Any], path: Path, *, required: bool = True
    ) -> Any | None:
        value = key_get_default(inputs, tuple(map(MappingKey, path)), None)
        if value is None and required:
            msg = f"input {path!r} missing from transformed batch"
            raise KeyError(msg)
        return value

    def _init_readout_tokens(
        self, *, use_readout_token: bool, num_register_tokens: int
    ) -> None:
        """Learned per-frame READOUT and REGISTER token embeddings (see class doc).

        Deliberately EXCLUDED from the `fusion_norm` scale calibration:
        `fusion_patch_norm`/`fusion_goal_gain` balance the two FROZEN encoder
        streams that are concatenated ahead of `patch_projection`, where no
        gradient can ever fix a scale mismatch. The readout/register tokens are
        free parameters born directly in model space (post-projection) -- the same
        status as `speed_embedding` and the trunk's positional embeddings, neither
        of which participates in the calibration -- so gradient descent sets their
        scale, and folding them through the 1/RMS goal gain would merely rescale
        their init. They are initialized with the trunc_normal(std=0.02) the
        trunks already use for learned positional tokens, and their norms are
        logged in `quality/token_norm/*` next to patch/speed/goal so any drift in
        the balance is visible rather than silent.

        Raises:
            ValueError: on a negative register count, on registers without the
                readout token (a register would then occupy the `[:, :, -1]`
                readout slot and be read from, which registers must never be),
                or when the policy width cannot be inferred.
        """
        if num_register_tokens < 0:
            msg = f"num_register_tokens must be >= 0, got {num_register_tokens}"
            raise ValueError(msg)
        if num_register_tokens and not use_readout_token:
            msg = (
                "num_register_tokens > 0 requires use_readout_token=True: the "
                "readout is the LAST token of a frame, so without a dedicated "
                "readout token the last register would be read from -- and "
                "registers exist precisely to absorb attention without being read"
            )
            raise ValueError(msg)

        if not use_readout_token:
            self.readout_token: nn.Parameter | None = None
            self.register_tokens: nn.Parameter | None = None
            return

        dim = getattr(self.patch_projection, "out_features", None) or getattr(
            self.speed_embedding, "embedding_dim", None
        )
        if dim is None:
            msg = (
                "use_readout_token: cannot infer the policy width -- "
                "patch_projection has no out_features and speed_embedding has "
                "no embedding_dim"
            )
            raise ValueError(msg)

        self.readout_token = nn.Parameter(torch.empty(dim))
        nn.init.trunc_normal_(self.readout_token, mean=0.0, std=0.02, a=-0.04, b=0.04)
        if num_register_tokens:
            self.register_tokens = nn.Parameter(torch.empty(num_register_tokens, dim))
            nn.init.trunc_normal_(
                self.register_tokens, mean=0.0, std=0.02, a=-0.04, b=0.04
            )
        else:
            self.register_tokens = None

    def _frame_tokens(
        self,
        images: Tensor,
        speed: Tensor,
        waypoints: Tensor,
        *,
        token_norms: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Per-frame token blocks `(b, t, k, d)` -- everything below the trunk.

        `k = cam*p + 1` by default; `k = cam*p + 1 + num_register_tokens + 1`
        with `use_readout_token` (see the class docstring for the layout).

        `images` is `(b, t, cam, c, h, w)`, `cam = len(self.cameras)` -- even a
        single-camera model stacks to a `cam=1` axis, so this is the one code path
        for both. Factored out of `_features` so the KV-cached decode step
        (`rmind.models.patch_policy_decoder.PatchPolicyDecoderStep`) runs the
        identical per-frame pipeline on ONE frame. Nothing here is temporal, which
        is exactly why one new frame per tick is sufficient.

        `token_norms`, if given, is filled with gradient-free mean L2 norms per
        token type (`speed`/`patch`/`goal` activations entering the trunk, plus
        the `register`/`readout` embedding parameters when enabled) -- the
        training-quality signal that guards against one token type's scale
        running away. It is an out-parameter (rather than module state) so the
        `torch.export`ed serving graphs never see a module mutation.
        """
        with torch.no_grad():
            patches = self.image_encoder(images)  # (b, t, cam, p, d_img)
            goal = self.goal_encoder.encode(waypoints)  # (b, t, g)

        # each camera contributes its own `p` patches through the same frozen
        # encoder/patch_projection -- no new parameters, just a longer per-frame
        # token block (https://arxiv.org/pdf/2607.18236 section 2.1 generalizes
        # trivially: concatenate cameras along the patch axis, not the channel one)
        patches = rearrange(patches, "b t cam p d -> b t (cam p) d")

        if self.fusion_patch_norm is not None:
            # NOTE: no in-place ops -- these tensors come from a no_grad block
            # and the gains must receive gradients
            patches = self.fusion_patch_norm(patches) * self.fusion_patch_gain
            goal = torch.mul(goal, self.fusion_goal_gain)

        _, _, num_patches, _ = patches.shape
        patches = torch.cat(
            [patches, goal.unsqueeze(-2).expand(-1, -1, num_patches, -1)], dim=-1
        )  # T x P x (D + G), https://arxiv.org/pdf/2607.18236 section 2.1
        patches = self.patch_projection(patches)  # (b, t, p, d)

        speed_token = self.speed_embedding(self.speed_tokenizer(speed))  # (b, t, 1, d)

        # speed first so the frame block ENDS on the readout position: the learned
        # readout token when `use_readout_token`, else the last patch token
        parts = [speed_token, patches]
        b, t = patches.shape[:2]
        if self.register_tokens is not None:
            parts.append(
                self.register_tokens.reshape(1, 1, -1, patches.shape[-1]).expand(
                    b, t, -1, -1
                )
            )
        if self.readout_token is not None:
            parts.append(self.readout_token.reshape(1, 1, 1, -1).expand(b, t, 1, -1))

        if token_norms is not None:
            with torch.no_grad():
                token_norms["speed"] = speed_token.detach().norm(dim=-1).mean()
                token_norms["patch"] = patches.detach().norm(dim=-1).mean()
                token_norms["goal"] = goal.detach().norm(dim=-1).mean()
                if self.register_tokens is not None:
                    token_norms["register"] = (
                        self.register_tokens.detach().norm(dim=-1).mean()
                    )
                if self.readout_token is not None:
                    token_norms["readout"] = self.readout_token.detach().norm()

        return torch.cat(parts, dim=-2)  # (b, t, k, d)

    def _init_fusion_norm(self, fusion_goal_rms: float | None) -> None:
        """Build the scale-balanced fusion parameters (see __init__).

        Raises:
            ValueError: if the goal-gain calibration RMS is not finite/positive
                (1/RMS would be inf, NaN-ing the first step).
        """
        goal_dim = self.goal_encoder.quantizer.dim
        patch_dim = self.patch_projection.in_features - goal_dim
        self.fusion_patch_norm: Module | None = nn.LayerNorm(patch_dim)
        self.fusion_patch_gain: nn.Parameter | None = nn.Parameter(torch.tensor(1.0))
        if fusion_goal_rms is not None:
            # data-measured element-RMS of z_q. The uniform-random-code MC
            # below overestimates it (real codebook usage is non-uniform:
            # measured 1.86x on gzxgumtf — 0.143 MC vs 0.077 real), landing
            # the goal stream ~2x quieter than intended. Prefer the
            # measured value when known (H3 eval, 2026-08-06).
            rms = torch.tensor(fusion_goal_rms)
        else:
            with torch.no_grad():
                quantizer = self.goal_encoder.quantizer
                generator = torch.Generator().manual_seed(0)
                codes = torch.stack(
                    [
                        torch.randint(
                            0, quantizer.codebook_size, (1024,), generator=generator
                        )
                        for _ in range(quantizer.num_quantizers)
                    ],
                    dim=-1,
                )
                # the CPU generator fixes the seed sequence; lookup must run
                # on whatever device the loaded codebooks landed on
                quantizer_device = next(
                    chain(quantizer.parameters(), quantizer.buffers())
                ).device
                rms = (
                    quantizer
                    .lookup(codes.to(quantizer_device))
                    .pow(2)
                    .mean()
                    .sqrt()
                    .cpu()
                )
        if not bool(rms.isfinite()) or not float(rms) > 0.0:
            msg = (
                f"fusion_norm goal-gain calibration produced RMS={float(rms)}; "
                "the goal codebooks are probably uninitialized (1/RMS is not finite)"
            )
            raise ValueError(msg)
        self.fusion_goal_gain: nn.Parameter | None = nn.Parameter(1.0 / rms)

    def _features(
        self,
        batch: Any,
        *,
        require_chunk: bool = True,
        token_norms: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Per-frame readout features `(b, t, d)` and the action chunks `(b, t, h, a)`.

        The chunk is only a TARGET (and never feeds the features), so callers on
        the inference path (`forward`, ONNX export) pass `require_chunk=False`
        and may omit the action series from the batch entirely.

        Raises:
            ValueError: if the encoder was built for a different
                `tokens_per_frame` than the frame layout produces (e.g.
                `use_readout_token`/`num_register_tokens` changed without
                updating the encoder/serving geometry).
        """
        inputs = self.input_transform(batch)

        image_by_camera = self._get(inputs, ("image",))  # {camera: (b, t, c, h, w)}
        speed = self._get(inputs, self.speed)  # (b, t, 1)
        waypoints = self._get(inputs, self.waypoints)  # (b, t, n, 2)
        chunk = self._get(inputs, self.chunk, required=require_chunk)

        images = torch.stack(
            [image_by_camera[camera] for camera in self.cameras], dim=2
        )  # (b, t, cam, c, h, w)

        tokens = self._frame_tokens(images, speed, waypoints, token_norms=token_norms)
        _, num_frames, tokens_per_frame, _ = tokens.shape

        # the causal trunk's mask/RoPE/KV-cache geometry is built from ITS
        # `tokens_per_frame`; a layout mismatch must fail here, loudly, not
        # diverge at serving time
        encoder_k = getattr(self.encoder, "tokens_per_frame", None)
        if encoder_k is not None and encoder_k != tokens_per_frame:
            msg = (
                f"frame layout produces {tokens_per_frame} tokens per frame "
                f"(use_readout_token={self.use_readout_token}, "
                f"num_register_tokens={self.num_register_tokens}) but the "
                f"encoder was built with tokens_per_frame={encoder_k}; update "
                "the encoder config (and the KV-cache/export geometry with it)"
            )
            raise ValueError(msg)

        embedding = self.encoder(
            rearrange(tokens, "b t k d -> b (t k) d"), num_frames=num_frames
        )
        features = rearrange(embedding, "b (t k) d -> b t k d", t=num_frames)[
            :, :, -1
        ]  # last token per frame: the learned readout token when
        # `use_readout_token`, else the last patch token

        if self.norm is not None:
            features = self.norm(features)

        return features, chunk

    # VQ-BeT joint head -- mirrors `JointPolicyObjective` (incl. the teacher-forced
    # offset fix) with a leading (b, t) batch instead of (b,).

    def _heads(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Code logits (*b, g, c) and the full offset table (*b, g, c, action_dim)."""
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
        """Select each quantizer's offset at `codes` and sum over quantizers."""
        index = codes[..., None, None].expand(*codes.shape, 1, offsets.shape[-1])
        # https://arxiv.org/pdf/2403.03181 Figure 2.
        return offsets.gather(-2, index).squeeze(-2).sum(dim=-2)  # (*b, action_dim)

    def _offset(self, offsets: Tensor, codes: Tensor) -> Tensor:
        offset = self._gather_offset(offsets, codes)
        if self.offset_scale is None:
            return offset
        return torch.tanh(offset / self.offset_scale) * self.offset_scale

    def _sample_codes(self, code_logits: Tensor) -> Tensor:
        if self.sample_codes:
            # Gumbel-max: (logits + Gumbel noise).argmax(-1) draws from the same
            # categorical as torch.multinomial(logits.softmax(-1), 1), but skips the
            # softmax kernel and, critically, the multinomial CPU sync -- its validity
            # check does `.item<bool>()` on a bool scalar, forcing a
            # cudaStreamSynchronize every step this runs (every train step, via the
            # sampled_recon diagnostic in `_compute_metrics`).
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(code_logits)))
            return (code_logits + gumbel_noise).argmax(dim=-1)
        return code_logits.argmax(dim=-1)

    def _neighbor_smoothing_targets(self, target_codes: Tensor) -> Tensor:
        """Distance-aware smoothing targets `W = softmax(-d_q / tau)`, `(*b, g, c)`.

        Uniform smoothing puts half its mass on the farther half of the codebook.
        Instead, weight each candidate code by how close its DECODED action chunk
        is to the ground truth, using EXACT PREFIX-CONDITIONED distances

            d_q(c) = mean_a | invert(gt_codes with q<-c) - invert(gt_codes) |

        i.e. substitute candidate `c` at quantizer `q` while holding the OTHER
        quantizers at ground truth, and decode through the frozen RVQ. A static
        per-quantizer 16x16 codebook-distance table is NOT valid here: RVQ
        levels q>=1 encode residuals, so a code's decoded contribution depends
        on the prefix (measured additivity error 2.5e-2, about half the L1 at
        stake). Distances are the MEAN of |.| over the action dim -- the same
        reduction as the L1 offset loss -- so `tau` is in normalized-action
        units per dim.

        Cost: g*c (= 64) vectorised `invert` calls over the full batch,
        measured ~70-76 ms/batch, ~4% of a 1.7 s step.
        """
        tokenizer = self.tokenizer
        quantizer = tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size

        with torch.no_grad():
            base = tokenizer.invert(target_codes)  # (*b, action_dim)
            distances = base.new_empty(*target_codes.shape[:-1], g, c)
            for q in range(g):
                for code in range(c):
                    candidate = target_codes.clone()
                    candidate[..., q] = code
                    distances[..., q, code] = (
                        (tokenizer.invert(candidate) - base).abs().mean(dim=-1)
                    )
            return torch.softmax(-distances / self.neighbor_smoothing_tau, dim=-1)

    def _predict_chunk(self, features: Tensor) -> Tensor:
        """Decode `invert(codes) + offset` -> `(*b, horizon, action_features)`."""
        code_logits, offsets = self._heads(features)
        codes = self._sample_codes(code_logits)
        offset = self._offset(offsets, codes)

        return (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # noqa: SLF001
        )

    def _code_losses(
        self, code_logits: Tensor, target_codes: Tensor
    ) -> dict[str, Tensor]:
        """Per-quantizer classification against the ground-truth codes, supervised
        at every frame's readout token (https://arxiv.org/pdf/2607.18236 section
        2.2). With `neighbor_smoothing_tau` set, FocalLoss's uniform smoothing
        term is replaced by the distance-aware target from
        `_neighbor_smoothing_targets`.
        """
        smoothing_targets = (
            self._neighbor_smoothing_targets(target_codes)
            if self.neighbor_smoothing_tau is not None
            else None
        )

        losses: dict[str, Tensor] = {}
        for q in range(self.tokenizer.quantizer.num_quantizers):
            code_loss_args = (
                rearrange(code_logits[..., q, :], "b t c -> (b t) c"),
                rearrange(target_codes[..., q], "b t -> (b t)"),
            )
            losses[f"code_{q}"] = (
                self.losses["code"](*code_loss_args)
                if smoothing_targets is None
                else self.losses["code"](
                    *code_loss_args,
                    rearrange(smoothing_targets[..., q, :], "b t c -> (b t) c"),
                )
            )
        return losses

    def _compute_metrics(
        self, batch: Any, *, token_norms: dict[str, Tensor] | None = None
    ) -> TensorDict:
        features, chunk = self._features(
            batch, token_norms=token_norms
        )  # (b, t, d), (b, t, h, a)
        tokenizer = self.tokenizer

        with torch.no_grad():
            target_codes = tokenizer(chunk)  # (b, t, num_quantizers)
            target = tokenizer._normalize(  # noqa: SLF001
                chunk.flatten(-2, -1)
            )  # (b, t, action_dim)

        code_logits, offsets = self._heads(features)  # (b, t, g, c), (b, t, g, c, a)

        losses: dict[str, Tensor] = self._code_losses(code_logits, target_codes)

        # reconstruction as inference does it, logged for train-curve comparability.
        # Only meaningful when codes are actually SAMPLED: with sample_codes=false
        # `_sample_codes` degenerates to argmax and this would duplicate the
        # offset_argmax_recon* metrics while misrepresenting the argmax nature of
        # serving -- so skip the whole computation (multinomial + invert + gather +
        # two L1s), not just the logging.
        sampled_chunk: Tensor | None = None
        sampled_recon: Tensor | None = None
        if self.sample_codes or not self.teacher_force_offset:
            codes = self._sample_codes(code_logits)
            sampled_chunk = tokenizer.invert(codes) + self._offset(offsets, codes)
            if self.sample_codes:
                sampled_recon = self.losses["offset"](sampled_chunk.detach(), target)

        if self.teacher_force_offset:
            # offset supervised at the GROUND-TRUTH codes (teacher forcing), so each
            # code's offset entry only sees residuals of actions that quantized to it
            predicted_chunk = tokenizer.invert(target_codes) + self._offset(
                offsets, target_codes
            )
        elif sampled_chunk is not None:
            predicted_chunk = sampled_chunk
        else:  # unreachable: the decode above always runs when not teacher forcing
            msg = "sampled_chunk must exist when teacher_force_offset is False"
            raise RuntimeError(msg)

        losses["offset"] = self.losses["offset"](predicted_chunk, target)

        metrics = self._readout_metrics(
            code_logits=code_logits,
            offsets=offsets,
            target_codes=target_codes,
            target=target,
            predicted_chunk=predicted_chunk,
            sampled_chunk=sampled_chunk,
            sampled_recon=sampled_recon,
        )

        return TensorDict({"policy": {"loss": losses, "metric": metrics}})

    def _readout_metrics(  # noqa: PLR0913
        self,
        *,
        code_logits: Tensor,
        offsets: Tensor,
        target_codes: Tensor,
        target: Tensor,
        predicted_chunk: Tensor,
        sampled_chunk: Tensor | None = None,
        sampled_recon: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Gradient-free diagnostics at the deployed readout.

        The training losses average over all T readouts (contexts of 1..T
        frames), whereas `JointPolicyObjective` only ever scores the newest
        frame -- these make the two comparable.

        `sampled_chunk`/`sampled_recon` are None when `sample_codes` is false:
        sampling is an eval-only decode mode, and with argmax serving the
        sampled metrics would just duplicate `offset_argmax_recon*`.
        """
        tokenizer = self.tokenizer
        with torch.no_grad():
            metrics: dict[str, Tensor] = {}
            if sampled_recon is not None:
                metrics["offset_sampled_recon"] = sampled_recon
            for q in range(tokenizer.quantizer.num_quantizers):
                metrics[f"code_{q}_last"] = self.losses["code"](
                    code_logits[:, -1, q, :], target_codes[:, -1, q]
                )
            metrics["offset_last"] = self.losses["offset"](
                predicted_chunk[:, -1], target[:, -1]
            )
            if sampled_recon is not None and sampled_chunk is not None:
                metrics["offset_sampled_recon_last"] = self.losses["offset"](
                    sampled_chunk[:, -1].detach(), target[:, -1]
                )

            # ARGMAX decode -- exactly what inference emits when `sample_codes=false`,
            # i.e. what the exported engine actually serves. This is a DEPLOYMENT-aligned
            # number and it does not track the code losses above. Measured on
            # dashing-dream-514 (wxyp0bzq) v0 -> v9: val code_0 rose 0.811 -> 2.885
            # (+256%) while this metric IMPROVED 13% (0.0456 -> 0.0395) and p_gt improved
            # 20% -- the rising NLL is tail miscalibration, not capability loss. Across
            # arms it is worse than uninformative: dinov2_smalltrunk has the BEST val
            # code_0 (0.890) and the WORST argmax recon (0.0462), and it underperformed
            # in rsim. Select checkpoints on this, not on val/loss/code_*.
            argmax_codes = code_logits.argmax(dim=-1)
            argmax_chunk = tokenizer.invert(argmax_codes) + self._offset(
                offsets, argmax_codes
            )
            metrics["offset_argmax_recon"] = self.losses["offset"](argmax_chunk, target)
            metrics["offset_argmax_recon_last"] = self.losses["offset"](
                argmax_chunk[:, -1], target[:, -1]
            )

            # code accuracy at the deployed readout. Without it the code losses cannot
            # separate "argmax still right, tails miscalibrated" from "argmax now wrong",
            # which is the distinction that decides whether a rising val code loss
            # matters. Chance is 1/num_codes marginally, (1/num_codes)**g jointly.
            correct = argmax_codes[:, -1] == target_codes[:, -1]  # (b, g)
            for q in range(tokenizer.quantizer.num_quantizers):
                metrics[f"code_acc_{q}_last"] = correct[:, q].float().mean()
            metrics["code_acc_joint_last"] = correct.all(dim=-1).float().mean()

            # context-depth localizer for windowed causal trunks: readouts at
            # positions < window-1 train under a PARTIAL window, positions
            # >= window-1 under the FULL window served at inference. A gap
            # between the two buckets says WHERE a causal arm is failing:
            # full-window >> partial-window means long-context conditioning
            # itself is the problem, not the heads or the features.
            window = getattr(self.encoder, "window", None)
            num_frames = code_logits.shape[1]
            if window is not None and num_frames > window - 1:
                buckets = {
                    "partial_window": slice(None, window - 1),
                    "full_window": slice(window - 1, None),
                }
                for bucket, sl in buckets.items():
                    if code_logits[:, sl].shape[1] == 0:
                        continue
                    metrics[f"code_{bucket}"] = torch.stack([
                        self.losses["code"](
                            rearrange(code_logits[:, sl, q, :], "b t c -> (b t) c"),
                            rearrange(target_codes[:, sl, q], "b t -> (b t)"),
                        )
                        for q in range(tokenizer.quantizer.num_quantizers)
                    ]).mean()
                    metrics[f"offset_{bucket}"] = self.losses["offset"](
                        predicted_chunk[:, sl], target[:, sl]
                    )

        return metrics

    def _step(self, batch: Any, prefix: str) -> STEP_OUTPUT:
        token_norms: dict[str, Tensor] = {}
        # Optional profiling: set environment variable `TORCH_PROFILER` to enable.
        # If `TORCH_PROFILER_DIR` is set, a chrome trace will be written there.
        with maybe_profile(f"{prefix}_step"):
            metrics = self._compute_metrics(batch, token_norms=token_norms)

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
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

        # per-token-type norm tracking under the TrainingQualityLogger's
        # `quality/` prefix -- the guard against a token type's scale (readout/
        # register embeddings especially) drifting away from the fusion balance
        if token_norms:
            self.log_dict(
                {
                    f"quality/token_norm/{prefix}/{name}": value
                    for name, value in token_norms.items()
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
        features, _ = self._features(batch, require_chunk=False)
        chunk = self._predict_chunk(features[:, -1])
        return TensorDict({"policy": {"joint_actions": chunk}})

    @classmethod
    def load_for_export(cls, artifact: str, **kwargs: Any) -> "PatchPolicy":
        """Load a checkpoint configured for deployment export (ONNX).

        Mirrors the control-transformer export conventions
        (config/export/yaak/control_transformer/finetuned.yaml):
        - argmax code decoding (`sample_codes=False`) for determinism;
        - the in-model image pipeline (Rearrange/CenterCrop/Resize/ToDtype)
          is replaced by ImageNet `Normalize` only -- deployment supplies
          already-cropped/resized `[0, 1]` float frames in `(b, t, c, h, w)`.

        The exported graph needs no action series in the batch (`forward`
        passes `require_chunk=False`).
        """
        from torchvision.transforms.v2 import Normalize  # noqa: PLC0415

        model = cls.load_from_wandb_artifact(
            artifact, filename="model.ckpt", map_location="cpu", weights_only=False
        )
        for key, value in kwargs.items():
            setattr(model, key, value)
        model.sample_codes = False
        # index 2 of the input_transform Sequential is the per-modality ModuleDict
        model.input_transform[2]["image"] = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return model.eval()

    @classmethod
    def load_for_continuation(
        cls,
        artifact: str,
        *,
        optimizer: Any,
        lr_scheduler: Any | None = None,
        **_ignored: Any,
    ) -> "PatchPolicy":
        """Warm-start continuation training from a finished run's checkpoint.

        Loads weights AND saved hparams from the artifact, then replaces only
        the optimization config: fresh optimizer state (Adam moments reset) and
        the given LR/schedule. `lr_scheduler=None` -> constant LR, no warmup.
        The replacement is written back into `hparams` so checkpoints saved by
        the continuation run reload with the continuation settings.

        `**_ignored` swallows architecture keys that parent experiments inline
        under `model.*` (e.g. dinov2_dinowm's `image_encoder`): on continuation
        the architecture comes from the CHECKPOINT hparams, not the config.
        """
        if not isinstance(optimizer, HydraConfig):
            optimizer = HydraConfig[Optimizer].model_validate(optimizer)
        if lr_scheduler is not None and not isinstance(
            lr_scheduler, LRSchedulerHydraConfig
        ):
            lr_scheduler = LRSchedulerHydraConfig.model_validate(lr_scheduler)

        model = cls.load_from_wandb_artifact(
            artifact, filename="model.ckpt", map_location="cpu", weights_only=False
        )
        model.optimizer = optimizer
        model.lr_scheduler = lr_scheduler
        model.hparams["optimizer"] = optimizer.model_dump()
        model.hparams["lr_scheduler"] = (
            lr_scheduler.model_dump() if lr_scheduler is not None else None
        )
        return model

    @classmethod
    def load_head_for_export(cls, artifact: str) -> "PatchPolicyHead":
        """Head-only export for on-the-fly decoding (see `PatchPolicyHead`)."""
        model = cls.load_from_wandb_artifact(
            artifact, filename="model.ckpt", map_location="cpu", weights_only=False
        ).eval()
        return PatchPolicyHead(
            norm=model.norm,
            code_head=model.code_head,
            offset_head=model.offset_head,
            tokenizer=model.tokenizer,
        ).eval()

    @staticmethod
    def _structure(chunk: Tensor) -> TensorDict:
        """Split a normalized `(b, horizon, action_features)` chunk into named fields."""
        return TensorDict({
            "continuous": TensorDict({
                "gas_pedal": chunk[..., 0],
                "brake_pedal": chunk[..., 1],
                "steering_angle": chunk[..., 2],
            }),
            "discrete": TensorDict({
                "turn_signal": torch.bucketize(
                    chunk[..., 3] * 2, torch.tensor([0.5, 1.5], device=chunk.device)
                )
            }),
        })

    @override
    def predict_step(self, batch: dict[str, Any]) -> TensorDict:
        keys = frozenset(self.prediction_config.objectives)
        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        tokenizer = self.tokenizer

        features, chunk = self._features(batch)
        features = features[:, -1]  # predict from the newest frame only

        b, t = chunk.shape[:2]
        time_index = torch.arange(t, device=features.device).expand(b, -1)[:, -1:]

        ground_truth = tokenizer._normalize(  # noqa: SLF001
            chunk[:, -1].flatten(-2, -1)
        ).unflatten(-1, (-1, tokenizer._action_features))  # noqa: SLF001

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            gt = self._structure(ground_truth)
            gt["discrete", "turn_signal"] = chunk[:, -1, :, 3].long()
            predictions[key] = Prediction(value=gt, time_index=time_index)

        needs_prediction = keys & {
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.SCORE_L1,
            ObjectivePredictionKey.SCORE_SIGNED_ERROR,
        }
        if needs_prediction:
            predicted = self._predict_chunk(features)

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=self._structure(predicted), time_index=time_index
                )

            if (key := ObjectivePredictionKey.SCORE_L1) in keys:
                predictions[key] = Prediction(
                    value=self._structure(predicted).apply(
                        lambda p, g: F.l1_loss(p.float(), g.float(), reduction="none"),
                        self._structure(ground_truth),
                    ),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.SCORE_SIGNED_ERROR) in keys:
                predictions[key] = Prediction(
                    value=self._structure(predicted).apply(
                        lambda p, g: p.float() - g.float(),
                        self._structure(ground_truth),
                    ),
                    time_index=time_index,
                )

        return TensorDict({"policy": predictions}).auto_batch_size_(2)

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


@final
class PatchPolicyHead(pl.LightningModule):
    """The VQ-BeT head alone, for split deployment with on-the-fly decoding.

    Consumes the trunk's readout token (the `BlockCausalTransformer` output at a
    frame's last patch token, BEFORE `PatchPolicy.norm` -- the LayerNorm is part
    of this module) plus caller-chosen codes, and emits everything a custom
    decode strategy needs:

    - ``code_logits`` `(b, g, c)`: pick codes however you like (argmax,
      temperature sampling, entropy gating, ...) -- code selection is
      deliberately OUTSIDE the graph;
    - ``offsets`` `(b, g, c, action_dim)`: the full state-conditioned offset
      table, for offset inspection or custom gathering;
    - ``chunk`` `(b, horizon, action_features)`: `decode(codes) + offset@codes`
      for the codes passed IN. Two-pass usage: run once (any codes) to read
      logits, choose codes, run again to decode -- the head is a few MLPs, so a
      second pass costs microseconds.
    """

    @validate_call
    def __init__(
        self,
        *,
        # optional, mirroring PatchPolicy.norm -- head-only export of an arm
        # trained with `norm: null` must not fail validation
        norm: InstanceOf[Module] | None,
        code_head: InstanceOf[Module],
        offset_head: InstanceOf[Module],
        tokenizer: InstanceOf[Module],
    ) -> None:
        super().__init__()
        self.norm = norm
        self.code_head = code_head
        self.offset_head = offset_head
        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> TensorDict:
        features, codes = inputs["features"], inputs["codes"]
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size

        if self.norm is not None:
            features = self.norm(features)
        code_logits = rearrange(
            self.code_head(features), "... (g c) -> ... g c", g=g, c=c
        )
        offsets = rearrange(
            self.offset_head(features), "... (g c a) -> ... g c a", g=g, c=c
        )

        offset = PatchPolicy._gather_offset(offsets, codes)  # noqa: SLF001
        chunk = (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # noqa: SLF001
        )
        return TensorDict({
            "code_logits": code_logits,
            "offsets": offsets,
            "chunk": chunk,
        })
