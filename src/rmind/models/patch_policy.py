from collections.abc import Mapping
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

from rmind.components import optimizers
from rmind.components.containers import ModuleDict
from rmind.components.objectives.base import ObjectivePredictionKey, Prediction
from rmind.components.transformer.utils import run_layer_stack
from rmind.config import HydraConfig, init_hydra_param
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.models.control_transformer import PredictionConfig
from rmind.utils._wandb import LoadableFromArtifact
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
    ) -> None:
        super().__init__()

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

    @override
    def forward(self, src: Tensor, *, num_frames: int) -> Tensor:
        _, seq_len, _ = src.shape
        x = src + self.position_embedding(torch.arange(seq_len, device=src.device))
        mask = block_causal_mask(num_frames, seq_len // num_frames, device=src.device)
        x = run_layer_stack(self.layers, x, mask, training=self.training)
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

    Optionally (map-context Arm M), a per-frame max-speed token is prepended as
    well (`T x (P + 2)`): pass `max_speed_tokenizer` + `max_speed_embedding`
    (see `rmind.components.map_context`) and size the encoder's
    `max_sequence_length` accordingly. The token reads `[context, max_speed]`
    from the batch (all-UNKNOWN when absent) and can be pinned at inference via
    `max_speed_override` (km/h) WITHOUT retraining -- see `_max_speed_token`.
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
        encoder: HydraConfig[BlockCausalTransformer]
        | InstanceOf[BlockCausalTransformer],
        tokenizer: HydraConfig[Module] | InstanceOf[Module],
        code_head: HydraConfig[Module] | InstanceOf[Module],
        offset_head: HydraConfig[Module] | InstanceOf[Module],
        losses: HydraConfig[ModuleDict] | InstanceOf[ModuleDict],
        norm: HydraConfig[Module] | InstanceOf[Module] | None = None,
        max_speed_tokenizer: HydraConfig[Module] | InstanceOf[Module] | None = None,
        max_speed_embedding: HydraConfig[Module] | InstanceOf[Module] | None = None,
        max_speed_dropout: float = 0.3,
        max_speed_override: float | None = None,
        image: Path = ("image", "cam_front_left"),
        speed: Path = ("continuous", "speed"),
        max_speed: Path = ("context", "max_speed"),
        waypoints: Path = ("context", "waypoints"),
        chunk: Path = ("joint_actions",),
        sample_codes: bool = True,
        teacher_force_offset: bool = True,
        offset_scale: float | None = None,
        fusion_norm: bool = False,
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
        self.encoder: BlockCausalTransformer = init_hydra_param(
            hparams, "encoder", encoder
        )
        # optional per-frame max-speed conditioning token (map context, Arm M);
        # both args None (the default) -> the token is absent and the model is
        # byte-identical (state dict + forward) to pre-map-context checkpoints
        if (max_speed_tokenizer is None) != (max_speed_embedding is None):
            msg = (
                "max_speed_tokenizer and max_speed_embedding must be "
                "provided together (or both omitted)"
            )
            raise ValueError(msg)
        self.max_speed_tokenizer: Module | None = init_hydra_param(
            hparams, "max_speed_tokenizer", max_speed_tokenizer
        )
        self.max_speed_embedding: Module | None = init_hydra_param(
            hparams, "max_speed_embedding", max_speed_embedding
        )
        self.max_speed_dropout = max_speed_dropout
        self.max_speed_override = max_speed_override

        self.code_head = init_hydra_param(hparams, "code_head", code_head)
        self.offset_head = init_hydra_param(hparams, "offset_head", offset_head)
        self.losses: ModuleDict = init_hydra_param(hparams, "losses", losses)
        self.norm: Module | None = init_hydra_param(hparams, "norm", norm)

        self.image: Path = image
        self.speed: Path = speed
        self.max_speed: Path = max_speed
        self.waypoints: Path = waypoints
        self.chunk: Path = chunk
        self.sample_codes = sample_codes
        self.teacher_force_offset = teacher_force_offset
        self.offset_scale = offset_scale
        hparams |= {
            "image": image,
            "speed": speed,
            "max_speed": max_speed,
            "max_speed_dropout": max_speed_dropout,
            "max_speed_override": max_speed_override,
            "waypoints": waypoints,
            "chunk": chunk,
            "sample_codes": sample_codes,
            "teacher_force_offset": teacher_force_offset,
            "offset_scale": offset_scale,
            "fusion_norm": fusion_norm,
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
                rms = quantizer.lookup(codes).pow(2).mean().sqrt()
            self.fusion_goal_gain: nn.Parameter | None = nn.Parameter(1.0 / rms)
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
    ) -> Tensor | None:
        value = key_get_default(inputs, tuple(map(MappingKey, path)), None)
        if value is None and required:
            msg = f"input {path!r} missing from transformed batch"
            raise KeyError(msg)
        return value

    def _features(
        self, batch: Any, *, require_chunk: bool = True
    ) -> tuple[Tensor, Tensor | None]:
        """Per-frame readout features `(b, t, d)` and the action chunks `(b, t, h, a)`.

        The chunk is only a TARGET (and never feeds the features), so callers on
        the inference path (`forward`, ONNX export) pass `require_chunk=False`
        and may omit the action series from the batch entirely.
        """
        inputs = self.input_transform(batch)

        images = self._get(inputs, self.image)  # (b, t, c, h, w)
        speed = self._get(inputs, self.speed)  # (b, t, 1)
        waypoints = self._get(inputs, self.waypoints)  # (b, t, n, 2)
        chunk = self._get(inputs, self.chunk, required=require_chunk)

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

        # non-patch tokens first so the frame block ends on a patch token (the
        # readout position); the block-causal mask derives tokens_per_frame from
        # the flattened length, so the optional max-speed token joins its frame's
        # bidirectional block automatically
        if self.max_speed_embedding is not None:
            max_speed_token = self._max_speed_token(inputs, reference=speed)
            tokens = torch.cat(
                [max_speed_token, speed_token, patches], dim=-2
            )  # (b, t, p + 2, d)
        else:
            tokens = torch.cat([speed_token, patches], dim=-2)  # (b, t, p + 1, d)
        _, num_frames, _, _ = tokens.shape

        embedding = self.encoder(
            rearrange(tokens, "b t k d -> b (t k) d"), num_frames=num_frames
        )
        features = rearrange(embedding, "b (t k) d -> b t k d", t=num_frames)[
            :, :, -1
        ]  # last patch token per frame

        if self.norm is not None:
            features = self.norm(features)

        return features, chunk

    def _max_speed_token(
        self, inputs: Mapping[str, Any], *, reference: Tensor
    ) -> Tensor:
        """Embedded per-frame max-speed token `(b, t, 1, d)`.

        Resolution order: `max_speed_override` (a constant km/h applied to every
        frame -- e.g. drive the private test ground "as if" it were a 30 km/h
        zone) > the `[context, max_speed]` batch input > all-UNKNOWN (today's
        datasets, which carry no map context). During training each frame's
        token is independently replaced with UNKNOWN with probability
        `max_speed_dropout`, so the policy stays robust to missing map data and
        the UNKNOWN inference path remains in-distribution.
        """
        b, t = reference.shape[:2]
        device = reference.device

        if self.max_speed_override is not None:
            max_speed = torch.full(
                (b, t, 1),
                float(self.max_speed_override),
                dtype=torch.float32,
                device=device,
            )
        else:
            max_speed = self._get(inputs, self.max_speed, required=False)

        if max_speed is None:
            ids = torch.zeros((b, t, 1), dtype=torch.long, device=device)
        else:
            ids = self.max_speed_tokenizer(max_speed)
            if ids.ndim == 2:  # (b, t) -> one token per frame
                ids = ids.unsqueeze(-1)
            ids = ids[..., :1]

        if self.training and self.max_speed_dropout > 0:
            drop = torch.rand(b, t, 1, device=device) < self.max_speed_dropout
            ids = torch.where(drop, torch.zeros_like(ids), ids)

        return self.max_speed_embedding(ids)

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
        *batch, g, c = code_logits.shape
        if self.sample_codes:
            return rearrange(
                torch.multinomial(code_logits.softmax(dim=-1).reshape(-1, c), 1),
                "(b g) 1 -> b g",
                g=g,
            ).reshape(*batch, g)
        return code_logits.argmax(dim=-1)

    def _predict_chunk(self, features: Tensor) -> Tensor:
        """Decode `invert(codes) + offset` -> `(*b, horizon, action_features)`."""
        code_logits, offsets = self._heads(features)
        codes = self._sample_codes(code_logits)
        offset = self._offset(offsets, codes)

        return (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # noqa: SLF001
        )

    def _compute_metrics(self, batch: Any) -> TensorDict:
        features, chunk = self._features(batch)  # (b, t, d), (b, t, h, a)
        tokenizer = self.tokenizer

        with torch.no_grad():
            target_codes = tokenizer(chunk)  # (b, t, num_quantizers)
            target = tokenizer._normalize(  # noqa: SLF001
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

        return TensorDict({"policy": {"loss": losses, "metric": metrics}})

    def _step(self, batch: Any, prefix: str) -> STEP_OUTPUT:
        metrics = self._compute_metrics(batch)

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

        Extra `kwargs` are set as attributes; for map-context checkpoints pass
        `max_speed_override=<kmh>` (hydra: `+model.max_speed_override=30`) to
        bake a constant zone speed into the graph, or leave it unset and feed
        `data.meta/MapContext/max_speed` `(1, t, 1)` float km/h (NaN = unknown,
        -1 = unlimited) as an explicit graph input (see
        config/export/yaak/patch_policy/finetuned_maxspeed.yaml).
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
        norm: InstanceOf[Module],
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
