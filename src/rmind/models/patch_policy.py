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
from torch.utils._pytree import MappingKey  # ruff: ignore[import-private-name]
from torch.utils.checkpoint import checkpoint

from rmind.components import optimizers
from rmind.components.containers import ModuleDict
from rmind.components.loss import (
    winner_takes_all_pose_l1,
    winner_takes_all_pose_l1_components,
)
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


def modality_transform(input_transform: Module) -> ModuleDict:
    """The per-modality `ModuleDict` stage of an `input_transform` Sequential.

    Its position shifts with optional stages ahead of it (e.g. `TrajectoryTarget`
    ahead of `ChunkFields`), so callers that need to reach into it (export's
    `Normalize` swap) must look it up by type rather than assume a fixed index.

    Raises:
        ValueError: if no `ModuleDict` stage is found.
    """
    for module in input_transform.modules():
        if isinstance(module, ModuleDict):
            return module
    msg = f"no ModuleDict stage found in {input_transform!r}"
    raise ValueError(msg)


@final
class TransformerBlock(nn.Module):
    """Pre-LN GPT block (minGPT-style, as used by the VQ-BeT policy trunk)."""

    def __init__(  # ruff: ignore[too-many-arguments]
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
    def __init__(  # ruff: ignore[too-many-arguments]
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

    Per frame: frozen ViT patch features `(P, D)` get the frozen waypoints-tokenizer
    latent `g_t` (that frame's goal vector) concatenated to every patch token --
    the paper's `T x P x (D + G)` scheme -- then projected to the policy width. An
    embedded speed token is prepended, the `T x (P + 1)` sequence is flattened,
    given a learned 1D positional embedding, and run through a block-causal
    transformer (bidirectional intra-frame, causal inter-frame). Each frame's LAST
    patch token predicts that frame's action chunk with the VQ-BeT joint head from
    `JointPolicyObjective` (frozen residual-VQ chunk tokenizer; focal code loss +
    teacher-forced L1 offset).
    """

    @validate_call
    def __init__(  # ruff: ignore[too-many-arguments, too-many-statements]
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
        # auxiliary, non-VQ trajectory head (DrivoR, arXiv:2601.05083):
        # `None` (default) leaves this model identical to before it existed --
        # backward compatible with checkpoints/configs saved without it.
        trajectory_head: HydraConfig[Module] | InstanceOf[Module] | None = None,
        # trajectory-MODE classifier: `None` (default) leaves this model
        # unchanged. Reads the SAME readout features as `trajectory_head` and
        # is trained (see `_compute_metrics`) to predict which hypothesis the
        # winner-takes-all oracle would pick -- i.e. it distills a ground-truth
        # -dependent oracle into something that can select a mode with no
        # ground truth at deployment. Requires `trajectory_head is not None`.
        mode_head: HydraConfig[Module] | InstanceOf[Module] | None = None,
        image: Path = ("image", "cam_front_left"),
        speed: Path = ("continuous", "speed"),
        waypoints: Path = ("context", "waypoints"),
        chunk: Path = ("joint_actions",),
        trajectory_target: Path = ("context", "trajectory_target"),
        num_trajectory_hypotheses: int = 5,
        sample_codes: bool = True,
        teacher_force_offset: bool = True,
        offset_scale: float | None = None,
        fusion_norm: bool = False,
        fusion_goal_rms: float | None = None,
        # freeze everything except `mode_head` (and the always-frozen
        # image/goal/tokenizer): the lever `load_for_mode_head_training` uses
        # to train a mode classifier on top of an otherwise-frozen checkpoint.
        freeze_base: bool = False,
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
            .requires_grad_(False)  # ruff: ignore[boolean-positional-value-in-call]
            .eval()
        )
        self.goal_encoder = (
            init_hydra_param(hparams, "goal_encoder", goal_encoder)
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
        self.trajectory_head: Module | None = init_hydra_param(
            hparams, "trajectory_head", trajectory_head
        )
        self.mode_head: Module | None = init_hydra_param(
            hparams, "mode_head", mode_head
        )

        self.image: Path = image
        self.speed: Path = speed
        self.waypoints: Path = waypoints
        self.chunk: Path = chunk
        self.trajectory_target: Path = trajectory_target
        self.num_trajectory_hypotheses = num_trajectory_hypotheses
        self.sample_codes = sample_codes
        self.teacher_force_offset = teacher_force_offset
        self.offset_scale = offset_scale
        self.freeze_base = freeze_base
        hparams |= {
            "image": image,
            "speed": speed,
            "waypoints": waypoints,
            "chunk": chunk,
            "trajectory_target": trajectory_target,
            "num_trajectory_hypotheses": num_trajectory_hypotheses,
            "sample_codes": sample_codes,
            "teacher_force_offset": teacher_force_offset,
            "offset_scale": offset_scale,
            "fusion_norm": fusion_norm,
            "freeze_base": freeze_base,
        }

        # scale-balanced feature fusion: LayerNorm + learnable gain on the patch
        # side (encoder-agnostic scale; DINO token-norm spread is negligible so
        # nothing informative is lost), and a learnable gain on the goal side
        # initialized so RMS(gain * z_q) ~= RMS(LN(patches)) ~= 1 -- calibrated
        # from the frozen RVQ codebooks (seeded, data-free, identical across
        # DDP ranks). Per-sample code-norm information passes through untouched.
        if fusion_norm:
            goal_dim = self.goal_encoder.quantizer.dim
            patch_dim = self.patch_projection.in_features - goal_dim
            self.fusion_patch_norm: Module | None = nn.LayerNorm(patch_dim)
            self.fusion_patch_gain: nn.Parameter | None = nn.Parameter(
                torch.tensor(1.0)
            )
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
            self.fusion_goal_gain: nn.Parameter | None = nn.Parameter(1.0 / rms)
        else:
            self.fusion_patch_norm = None
            self.fusion_patch_gain = None
            self.fusion_goal_gain = None

        if freeze_base:
            for module in self._base_modules():
                if module is not None:
                    module.requires_grad_(False).eval()  # ruff: ignore[boolean-positional-value-in-call]
            for param in (self.fusion_patch_gain, self.fusion_goal_gain):
                if param is not None:
                    param.requires_grad_(False)  # ruff: ignore[boolean-positional-value-in-call]

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.prediction_config = prediction_config

        self.save_hyperparameters(hparams)

    def _base_modules(self) -> tuple[Module | None, ...]:
        """Every module `freeze_base` freezes -- everything except `mode_head`
        (the new trainable head) and the always-frozen image/goal/tokenizer
        (handled separately, see `train`).
        """
        return (
            self.patch_projection,
            self.speed_tokenizer,
            self.speed_embedding,
            self.encoder,
            self.code_head,
            self.offset_head,
            self.norm,
            self.trajectory_head,
            self.fusion_patch_norm,
        )

    @override
    def train(self, mode: bool = True) -> "PatchPolicy":
        super().train(mode)
        self.image_encoder.eval()
        self.goal_encoder.eval()
        self.tokenizer.eval()
        if self.freeze_base:
            for module in self._base_modules():
                if module is not None:
                    module.eval()
        return self

    @staticmethod
    def _get(
        inputs: Mapping[str, Any], path: Path, *, required: bool = True
    ) -> Tensor | None:
        value = key_get_default(inputs, tuple(map(MappingKey, path)), None)
        if value is None and required:
            msg = f"input {path!r} missing from transformed batch"
            raise KeyError(msg)
        return value

    def _frame_tokens(self, images: Tensor, speed: Tensor, waypoints: Tensor) -> Tensor:
        """Per-frame token blocks `(b, t, p + 1, d)` -- everything below the trunk.

        Factored out of `_features` so the KV-cached decode step
        (`rmind.models.patch_policy_decoder.PatchPolicyDecoderStep`) runs the
        identical per-frame pipeline on ONE frame. Nothing here is temporal, which
        is exactly why one new frame per tick is sufficient.
        """
        with torch.no_grad():
            patches = self.image_encoder(images)  # (b, t, p, d_img)
            goal = self.goal_encoder.encode(waypoints)  # (b, t, g)

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

        # speed first so the frame block ends on a patch token (the readout position)
        return torch.cat([speed_token, patches], dim=-2)  # (b, t, p + 1, d)

    def _features(
        self, batch: Any, *, require_chunk: bool = True
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Per-frame readout features `(b, t, d)`, the action chunks
        `(b, t, h, a)`, and (when `trajectory_head` is configured) the
        per-frame trajectory targets `(b, t, num_poses, 3)`.

        Both targets never feed the features, so callers on the inference
        path (`forward`, ONNX export) pass `require_chunk=False` and may omit
        the action/trajectory series from the batch entirely.
        """
        inputs = self.input_transform(batch)

        images = self._get(inputs, self.image)  # (b, t, c, h, w)
        speed = self._get(inputs, self.speed)  # (b, t, 1)
        waypoints = self._get(inputs, self.waypoints)  # (b, t, n, 2)
        chunk = self._get(inputs, self.chunk, required=require_chunk)
        trajectory_target = self._get(
            inputs,
            self.trajectory_target,
            required=require_chunk and self.trajectory_head is not None,
        )

        tokens = self._frame_tokens(images, speed, waypoints)
        _, num_frames, _, _ = tokens.shape

        embedding = self.encoder(
            rearrange(tokens, "b t k d -> b (t k) d"), num_frames=num_frames
        )
        features = rearrange(embedding, "b (t k) d -> b t k d", t=num_frames)[
            :, :, -1
        ]  # last patch token per frame

        if self.norm is not None:
            features = self.norm(features)

        return features, chunk, trajectory_target

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

    def _predict_chunk(self, features: Tensor) -> Tensor:
        """Decode `invert(codes) + offset` -> `(*b, horizon, action_features)`."""
        code_logits, offsets = self._heads(features)
        codes = self._sample_codes(code_logits)
        offset = self._offset(offsets, codes)

        return (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # ruff: ignore[private-member-access]
        )

    def _predict_trajectory(self, features: Tensor) -> Tensor:
        """Direct-regression multi-hypothesis trajectory forecast (DrivoR,
        arXiv:2601.05083) from the SAME readout `features` as `_predict_chunk`
        -- no VQ, no cross-attention decoder, just an MLP over the trunk's
        per-frame feature vector. `(*b, num_trajectory_hypotheses, num_poses, 3)`.

        Only call this when `self.trajectory_head is not None`.
        """
        pred = self.trajectory_head(features)  # ty:ignore[reportOptionalCall]
        return rearrange(
            pred, "... (q p c) -> ... q p c", q=self.num_trajectory_hypotheses, c=3
        )

    def _predict_mode(self, features: Tensor) -> Tensor:
        """Classifier logits over trajectory hypotheses, `(*b,
        num_trajectory_hypotheses)`. Reads the SAME readout `features` as
        `_predict_trajectory`; trained (see `_compute_metrics`) to predict the
        winner-takes-all oracle's `best_index` -- the hypothesis a
        ground-truth-supervised loss would have picked -- so a mode can be
        chosen at deployment with no access to ground truth.

        Only call this when `self.mode_head is not None`.
        """
        return self.mode_head(features)  # ty:ignore[reportOptionalCall]

    def _compute_metrics(  # ruff: ignore[too-many-locals, complex-structure]
        self, batch: Any
    ) -> TensorDict:
        features, chunk, trajectory_target = self._features(batch)  # (b, t, d), ...
        tokenizer = self.tokenizer

        with torch.no_grad():
            target_codes = tokenizer(chunk)  # (b, t, num_quantizers)
            target = tokenizer._normalize(  # ruff: ignore[private-member-access]
                chunk.flatten(-2, -1)
            )  # (b, t, action_dim)

        code_logits, offsets = self._heads(features)  # (b, t, g, c), (b, t, g, c, a)

        losses: dict[str, Tensor] = {}

        # per-quantizer classification against the ground-truth codes, supervised at
        # every frame's readout token (https://arxiv.org/pdf/2607.18236 section 2.2)
        for q in range(tokenizer.quantizer.num_quantizers):
            losses[f"code_{q}"] = self.losses["code"](
                rearrange(code_logits[..., q, :], "b t c -> (b t) c"),
                rearrange(target_codes[..., q], "b t -> (b t)"),
            )

        # reconstruction as inference does it, logged for train-curve comparability
        codes = self._sample_codes(code_logits)
        sampled_chunk = tokenizer.invert(codes) + self._offset(offsets, codes)
        sampled_recon = self.losses["offset"](sampled_chunk.detach(), target)

        if self.teacher_force_offset:
            # offset supervised at the GROUND-TRUTH codes (teacher forcing), so each
            # code's offset entry only sees residuals of actions that quantized to it
            predicted_chunk = tokenizer.invert(target_codes) + self._offset(
                offsets, target_codes
            )
        else:
            predicted_chunk = sampled_chunk

        losses["offset"] = self.losses["offset"](predicted_chunk, target)

        # auxiliary, non-VQ trajectory head (DrivoR, arXiv:2601.05083):
        # direct multi-hypothesis regression against the dead-reckoned
        # per-frame trajectory target, winner-takes-all over hypotheses
        trajectory_pred = None
        if self.trajectory_head is not None:
            trajectory_pred = self._predict_trajectory(features)  # (b, t, q, p, 3)
            losses["trajectory"] = self.losses["trajectory"](
                trajectory_pred, trajectory_target
            )

        # trajectory-mode classifier: distills the winner-takes-all oracle's
        # best_index (computed below, from the GT-supervised trajectory_pred)
        # into a head that needs no ground truth at deployment.
        mode_logits = None
        if self.mode_head is not None:
            if trajectory_pred is None:
                msg = (
                    "mode_head requires a trajectory_head to supervise against "
                    "(its label is the winner-takes-all oracle's best_index)"
                )
                raise RuntimeError(msg)
            mode_logits = self._predict_mode(features)  # (b, t, q)

        # gradient-free LAST-FRAME metrics: the training losses above average over
        # all T readouts (contexts of 1..T frames), whereas JointPolicyObjective
        # only ever scores the newest frame -- these make the two comparable
        with torch.no_grad():
            metrics: dict[str, Tensor] = {"offset_sampled_recon": sampled_recon}
            for q in range(tokenizer.quantizer.num_quantizers):
                metrics[f"code_{q}_last"] = self.losses["code"](
                    code_logits[:, -1, q, :], target_codes[:, -1, q]
                )
            metrics["offset_last"] = self.losses["offset"](
                predicted_chunk[:, -1], target[:, -1]
            )
            metrics["offset_sampled_recon_last"] = self.losses["offset"](
                sampled_chunk[:, -1].detach(), target[:, -1]
            )

            if trajectory_pred is not None:
                trajectory_loss_module = self.losses["trajectory"]
                _, best_index, _, xy_loss, heading_loss = (
                    winner_takes_all_pose_l1_components(
                        trajectory_pred,
                        trajectory_target,
                        heading_weight=getattr(
                            trajectory_loss_module, "heading_weight", 0.1
                        ),
                        reduction=getattr(trajectory_loss_module, "reduction", "mean"),
                    )
                )
                metrics["trajectory_loss_xy"] = xy_loss.mean()
                metrics["trajectory_loss_heading"] = heading_loss.mean()
                metrics["trajectory_best_index_unique_frac"] = torch.tensor(
                    best_index.unique().numel() / best_index.numel(),
                    device=best_index.device,
                )

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

        # cross-entropy against the oracle `best_index` computed above -- OUTSIDE
        # the no_grad block above, since this is the one term that must backprop
        # into mode_logits (and thus mode_head)
        if mode_logits is not None:
            losses["mode"] = self.losses["mode"](
                rearrange(mode_logits, "b t q -> (b t) q"),
                rearrange(best_index, "b t -> (b t)"),
            )
            with torch.no_grad():
                metrics["mode_accuracy"] = (
                    (mode_logits.argmax(dim=-1) == best_index).float().mean()
                )

        return TensorDict({"policy": {"loss": losses, "metric": metrics}})

    def _step(self, batch: Any, prefix: str) -> STEP_OUTPUT:
        # Optional profiling: set environment variable `TORCH_PROFILER` to enable.
        # If `TORCH_PROFILER_DIR` is set, a chrome trace will be written there.
        with maybe_profile(f"{prefix}_step"):
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
        features, _, _ = self._features(batch, require_chunk=False)
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
        from torchvision.transforms.v2 import (  # ruff: ignore[import-outside-top-level]
            Normalize,
        )

        model = cls.load_from_wandb_artifact(
            artifact, filename="model.ckpt", map_location="cpu", weights_only=False
        )
        for key, value in kwargs.items():
            setattr(model, key, value)
        model.sample_codes = False
        # the per-modality ModuleDict's position in the input_transform Sequential
        # shifts with optional stages ahead of it (e.g. TrajectoryTarget), so find
        # it by type rather than a fixed index.
        modality_transform(model.input_transform)["image"] = Normalize(
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
    def load_for_mode_head_training(
        cls,
        artifact: str,
        *,
        mode_head: Any,
        mode_loss: Any,
        optimizer: Any,
        lr_scheduler: Any | None = None,
        **_ignored: Any,
    ) -> "PatchPolicy":
        """Train a trajectory-mode classifier on top of an otherwise-FROZEN
        checkpoint (e.g. the `trajectory_head` arm exported as `v9y58oei`).

        Loads weights AND saved hparams from the artifact like
        `load_for_continuation`, but additionally:

        - sets `freeze_base=True`, which freezes (`requires_grad_(False)` +
          permanent `.eval()`, see `_base_modules`/`train`) every module except
          the new `mode_head` -- the image/goal/tokenizer encoders were already
          frozen by construction, this extends that to the trunk and the
          code/offset/trajectory heads;
        - instantiates `mode_head` and attaches it, plus a `"mode"` loss
          (typically `nn.CrossEntropyLoss`) added into `losses` -- distilling
          the winner-takes-all oracle's `best_index` (see `_compute_metrics`)
          into a head that runs with no ground truth at deployment;
        - replaces the optimizer/lr_scheduler exactly like
          `load_for_continuation` (fresh Adam moments; with only `mode_head`
          unfrozen, `SelectiveAdamW`'s param groups over the rest are inert).

        Requires the checkpoint to have been trained with a `trajectory_head`
        (there is no oracle winner to distill without one).

        Raises:
            ValueError: if the loaded checkpoint has no `trajectory_head`.
        """
        if not isinstance(optimizer, HydraConfig):
            optimizer = HydraConfig[Optimizer].model_validate(optimizer)
        if lr_scheduler is not None and not isinstance(
            lr_scheduler, LRSchedulerHydraConfig
        ):
            lr_scheduler = LRSchedulerHydraConfig.model_validate(lr_scheduler)
        if not isinstance(mode_head, HydraConfig):
            mode_head = HydraConfig[Module].model_validate(mode_head)
        if not isinstance(mode_loss, HydraConfig):
            mode_loss = HydraConfig[Module].model_validate(mode_loss)

        model = cls.load_from_wandb_artifact(
            artifact, filename="model.ckpt", map_location="cpu", weights_only=False
        )
        if model.trajectory_head is None:
            msg = (
                "load_for_mode_head_training requires a checkpoint trained "
                "with a trajectory_head (the mode classifier distills its "
                "winner-takes-all winner)"
            )
            raise ValueError(msg)

        model.freeze_base = True
        for module in model._base_modules():  # ruff: ignore[private-member-access]
            if module is not None:
                module.requires_grad_(False).eval()  # ruff: ignore[boolean-positional-value-in-call]
        for param in (model.fusion_patch_gain, model.fusion_goal_gain):
            if param is not None:
                param.requires_grad_(False)  # ruff: ignore[boolean-positional-value-in-call]
        model.hparams["freeze_base"] = True

        model.mode_head = mode_head.instantiate()
        model.hparams["mode_head"] = mode_head.model_dump()

        model.losses["mode"] = mode_loss.instantiate()
        losses_hparams = dict(model.hparams["losses"])
        losses_hparams["modules"] = {
            **losses_hparams.get("modules", {}),
            "mode": mode_loss.model_dump(),
        }
        model.hparams["losses"] = losses_hparams

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
    def predict_step(self, batch: dict[str, Any]) -> TensorDict:  # ruff: ignore[too-many-locals]
        keys = frozenset(self.prediction_config.objectives)
        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        tokenizer = self.tokenizer

        features, chunk, trajectory_target = self._features(batch)
        features = features[:, -1]  # predict from the newest frame only

        b, t = chunk.shape[:2]
        time_index = torch.arange(t, device=features.device).expand(b, -1)[:, -1:]

        ground_truth = tokenizer._normalize(  # ruff: ignore[private-member-access]
            chunk[:, -1].flatten(-2, -1)
        ).unflatten(-1, (-1, tokenizer._action_features))  # ruff: ignore[private-member-access]

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

        result: dict[str, Any] = {"policy": predictions}

        if self.trajectory_head is not None:
            trajectory_pred = self._predict_trajectory(features)  # (b, q, p, 3)
            trajectory_loss_module = self.losses["trajectory"]
            _, best_index, per_candidate_loss = winner_takes_all_pose_l1(
                trajectory_pred,
                trajectory_target[:, -1],
                heading_weight=getattr(trajectory_loss_module, "heading_weight", 0.1),
                reduction=getattr(trajectory_loss_module, "reduction", "mean"),
            )
            best_prediction = trajectory_pred.gather(
                1,
                best_index[:, None, None, None].expand(-1, 1, *trajectory_pred.shape[-2:]),
            ).squeeze(1)
            # explicit batch_size=[b]: `best_index`/`per_candidate_loss` are rank
            # 1-2, not the `(b, horizon)` shape `auto_batch_size_` below would
            # otherwise infer from the "policy" branch -- fixing it here keeps
            # that inference (and the "policy" branch's own batch_size) unaffected
            result["trajectory"] = TensorDict(
                {
                    "prediction": trajectory_pred,
                    "best_prediction": best_prediction,
                    "best_index": best_index,
                    "per_candidate_loss": per_candidate_loss,
                    "ground_truth": trajectory_target[:, -1],
                },
                batch_size=[trajectory_pred.shape[0]],
            )

            if self.mode_head is not None:
                # the classifier's own pick, alongside the oracle `best_index`
                # above -- their agreement rate is `mode_accuracy` (train/val)
                mode_logits = self._predict_mode(features)  # (b, q)
                result["trajectory"]["mode_logits"] = mode_logits
                result["trajectory"]["predicted_index"] = mode_logits.argmax(dim=-1)

        return TensorDict(result).auto_batch_size_(2)

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
        norm: InstanceOf[Module],
        code_head: InstanceOf[Module],
        offset_head: InstanceOf[Module],
        tokenizer: InstanceOf[Module],
    ) -> None:
        super().__init__()
        self.norm = norm
        self.code_head = code_head
        self.offset_head = offset_head
        self.tokenizer = tokenizer.requires_grad_(False).eval()  # ruff: ignore[boolean-positional-value-in-call]

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> TensorDict:
        features, codes = inputs["features"], inputs["codes"]
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size

        features = self.norm(features)
        code_logits = rearrange(
            self.code_head(features), "... (g c) -> ... g c", g=g, c=c
        )
        offsets = rearrange(
            self.offset_head(features), "... (g c a) -> ... g c a", g=g, c=c
        )

        offset = PatchPolicy._gather_offset(offsets, codes)  # ruff: ignore[private-member-access]
        chunk = (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # ruff: ignore[private-member-access]
        )
        return TensorDict({
            "code_logits": code_logits,
            "offsets": offsets,
            "chunk": chunk,
        })
