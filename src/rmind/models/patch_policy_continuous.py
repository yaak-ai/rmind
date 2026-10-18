import operator
from collections.abc import Mapping
from typing import Annotated, Any, Literal, final, override

import pytorch_lightning as pl
import torch
from einops import rearrange
from pydantic import Field, InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils._pytree import (  # ruff: ignore[import-private-name]
    MappingKey,
    tree_map,
)

from rmind.components import optimizers
from rmind.components.containers import ModuleDict
from rmind.components.fm_action_head import ActionHead
from rmind.components.rtc_flow_head import TokenTimeFlowHead
from rmind.config import HydraConfig, init_hydra_param
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.models.control_transformer import PredictionConfig
from rmind.utils._wandb import LoadableFromArtifact
from rmind.utils.pytree import key_get_default

type Path = tuple[str, ...]


# What a `CausalFrameTransformer` did before the car's intra-position work added
# these options (whose defaults are `norm_gain` + per-block checkpointing). Every
# palletjack checkpoint trained before then stores a trunk config WITHOUT them, and
# under the new defaults would ask for an `intra_position_gain` it never trained.
_PRE_MERGE_TRUNK: dict[str, Any] = {
    "intra_position_scaling": "none",
    "intra_position_factorization": "flat",
    "checkpoint": False,
}


def _pre_merge_trunk[T](encoder: "HydraConfig[T] | T") -> "HydraConfig[T] | T":
    """Give a trunk config that predates the intra-position options its old meaning.

    Only fills keys the config does not set, so a config that names them (as
    `config/model/palletjack/patch_policy/policy.yaml` does) is left alone.
    """
    if not isinstance(encoder, HydraConfig) or "intra_position_scaling" in (
        dumped := encoder.model_dump()
    ):
        return encoder

    return HydraConfig.model_validate(_PRE_MERGE_TRUNK | dumped)


@final
class PatchPolicyContinuous(pl.LightningModule, LoadableFromArtifact):
    """Patch-policy trunk with continuous Gaussian heads or a joint flow head.

    The trunk is `patch_policy.PatchPolicy`'s, unchanged in shape: frozen ViT
    patch features `(P, D)` - concatenated along the patch axis across `cameras`,
    each contributing its own `P` - projected to the policy width, an embedded
    speed token prepended so each frame block ends on a patch token, the
    flattened `T x (cam * P + 1 + O)` sequence run through a block-causal trunk
    (bidirectional intra-frame, causal inter-frame), and each frame's LAST patch
    token taken as that frame's readout. Multiple cameras therefore cost sequence
    length, not parameters, and the trunk needs no change - it sees only a larger
    `tokens_per_frame`. Frames must already be aligned across cameras; for real
    recordings `rmind.data.d12` does that by timestamp. `encoder` accepts either
    `patch_policy.BlockCausalTransformer` or
    `components.transformer.causal_frame.CausalFrameTransformer` - the same
    `forward(src, *, num_frames)` contract.

    `observations` adds `O` further scalar state tokens beside speed - on the D12,
    `fork_above_300`, the mast height switch - each named, looked up by path in
    the transformed batch and embedded by its entry in `observation_embeddings`.
    They are inputs, never targets: the model is told the fork's height band so it
    can condition on it, and separately predicts `fork1`, the command that changes
    it. Every one costs a token per frame, so the trunk's `tokens_per_frame` has to
    account for them, and adding or removing one invalidates a checkpoint's token
    positions.

    What differs is the head, and why:

    `PatchPolicy` reads out with VQ-BeT - a frozen residual-VQ chunk tokenizer
    plus code and offset heads - which requires an `ActionTokenizer` fitted to
    real action chunks. An embodiment with no recorded actions has nothing to fit
    one to, so this variant regresses each actuation directly with a Gaussian
    head (mean, log-variance), the same parameterization the control-transformer
    policy uses. That keeps the deployed contract to named scalars per actuation
    rather than codes to be decoded on the vehicle.

    The paper's goal-latent fusion is also absent: there is no route to encode
    indoors, so patches project from `D` alone rather than `D + G`.

    Every frame is a supervised readout, as in `PatchPolicy` - a `T`-frame clip
    yields `T` targets from one frozen-ViT pass. `forward` returns only the last
    frame's, which is what the vehicle acts on.

    `action_horizon` > 1 predicts an action CHUNK per frame instead of the current
    action alone - the same chunk the VQ-BeT `PatchPolicy` codes, as an
    independent Gaussian per chunk step. Each head then outputs `2 * H` (a mean and
    log-variance per step) and every target must arrive as `(b, t, H)`, i.e. the
    config unfolds the action columns with `components.nn.ChunkFields`. The loss
    is the same Gaussian NLL, now over every step; `chunk_l1*` metrics score the
    predicted means against the target chunk in the units `offset_argmax_recon*`
    uses, the first `near_horizon` steps separately (what is executed before
    re-planning). With the default `H == 1` nothing - heads, loss, metrics,
    checkpoints - differs from the single-step model.

    Supplying `fm_head` replaces the Gaussian heads and NLL with rectified-flow
    matching over the joint flattened action chunk. Training uses train-only
    action standardization; predictions are converted back to raw command units.
    Serving uses explicit Gaussian start noise and fixed-step Euler integration.
    """

    action_mean: Tensor
    action_std: Tensor

    @validate_call
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        input_transform: HydraConfig[Module] | InstanceOf[Module],
        image_encoder: HydraConfig[Module] | InstanceOf[Module],
        patch_projection: HydraConfig[Module] | InstanceOf[Module],
        speed_tokenizer: HydraConfig[Module] | InstanceOf[Module],
        speed_embedding: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module],
        heads: HydraConfig[ModuleDict] | InstanceOf[ModuleDict] | None,
        losses: HydraConfig[ModuleDict] | InstanceOf[ModuleDict] | None,
        targets: Mapping[str, Mapping[str, Path]],
        norm: HydraConfig[Module] | InstanceOf[Module] | None = None,
        cameras: tuple[str, ...] = ("cam_left_backward",),
        speed: Path = ("continuous", "speed"),
        observations: Mapping[str, Path] | None = None,
        observation_embeddings: HydraConfig[ModuleDict]
        | InstanceOf[ModuleDict]
        | None = None,
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
        prediction_config: Annotated[
            PredictionConfig, Field(default_factory=PredictionConfig)
        ],
        action_horizon: Annotated[int, Field(ge=1)] = 1,
        near_horizon: Annotated[int, Field(ge=1)] = 3,
        fm_head: HydraConfig[ActionHead] | InstanceOf[ActionHead] | None = None,
        fm_inference_steps: Annotated[int, Field(ge=1)] = 4,
        fm_x0: Literal["sample", "zero"] = "sample",
        action_mean: tuple[float, ...] | None = None,
        action_std: tuple[float, ...] | None = None,
        fm_action_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()

        hparams: dict[str, Any] = {}

        self.input_transform = init_hydra_param(
            hparams, "input_transform", input_transform
        )
        self.image_encoder = init_hydra_param(hparams, "image_encoder", image_encoder)
        self.patch_projection = init_hydra_param(
            hparams, "patch_projection", patch_projection
        )
        self.speed_tokenizer = init_hydra_param(
            hparams, "speed_tokenizer", speed_tokenizer
        )
        self.speed_embedding = init_hydra_param(
            hparams, "speed_embedding", speed_embedding
        )
        self.encoder = init_hydra_param(hparams, "encoder", _pre_merge_trunk(encoder))
        self.heads = (
            init_hydra_param(hparams, "heads", heads) if heads is not None else None
        )
        self.losses = (
            init_hydra_param(hparams, "losses", losses) if losses is not None else None
        )
        if fm_head is None and (heads is None or losses is None):
            message = "Gaussian heads and losses are required without a flow head"
            raise ValueError(message)
        hparams.setdefault("heads", None)
        hparams.setdefault("losses", None)
        self.norm = (
            init_hydra_param(hparams, "norm", norm) if norm is not None else None
        )

        # the ViT is a feature extractor here, exactly as in PatchPolicy
        self.image_encoder.requires_grad_(requires_grad=False).eval()

        self.observation_embeddings: ModuleDict | None = (
            init_hydra_param(hparams, "observation_embeddings", observation_embeddings)
            if observation_embeddings is not None
            else None
        )

        self.targets: Mapping[str, Mapping[str, Path]] = targets
        self.cameras: tuple[str, ...] = cameras
        self.speed: Path = speed
        self.observations: Mapping[str, Path] = observations or {}
        self.action_horizon: int = action_horizon
        self.near_horizon: int = min(near_horizon, action_horizon)
        self.fm_head = (
            init_hydra_param(hparams, "fm_head", fm_head)
            if fm_head is not None
            else None
        )
        self.fm_inference_steps = fm_inference_steps
        self.fm_x0 = fm_x0
        self.action_names = tuple(
            (modality, name) for modality, names in targets.items() for name in names
        )
        if self.fm_head is not None:
            self._configure_flow_head(
                hparams, action_mean, action_std, fm_action_weights
            )

        # an observation with no embedding would be read and silently discarded,
        # and an embedding with no observation never called; both are config bugs
        embedded = (
            set(self.observation_embeddings) if self.observation_embeddings else set()
        )
        if embedded != set(self.observations):
            msg = (
                "`observations` and `observation_embeddings` must name the same "
                f"observations; got {sorted(self.observations)} against "
                f"{sorted(embedded)}"
            )
            raise ValueError(msg)

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.prediction_config = prediction_config
        hparams["targets"] = targets
        hparams["cameras"] = cameras
        hparams["speed"] = speed
        hparams["observations"] = self.observations
        hparams["action_horizon"] = action_horizon
        hparams["near_horizon"] = near_horizon
        self.save_hyperparameters(hparams)

    def _configure_flow_head(
        self,
        hparams: dict[str, Any],
        action_mean: tuple[float, ...] | None,
        action_std: tuple[float, ...] | None,
        fm_action_weights: tuple[float, ...] | None,
    ) -> None:
        """Validate flow dimensions and persist action normalization.

        Raises:
            ValueError: If head dimensions or action statistics are invalid.
        """
        head = self.fm_head
        if head is None:
            return
        action_count = len(self.action_names)
        mean = torch.tensor(
            action_mean if action_mean is not None else [0.0] * action_count
        )
        std = torch.tensor(
            action_std if action_std is not None else [1.0] * action_count
        )
        if mean.shape != (action_count,) or std.shape != (action_count,):
            message = "action statistics must match the target count"
            raise ValueError(message)
        if not mean.isfinite().all() or not std.isfinite().all() or not (std > 0).all():
            message = (
                "action statistics must be finite with positive standard deviations"
            )
            raise ValueError(message)
        if (
            head.kind != "fm"
            or self.action_horizon * action_count != head.D
            or (
                isinstance(head, TokenTimeFlowHead)
                and (
                    head.action_horizon != self.action_horizon
                    or head.action_dim != action_count
                )
            )
        ):
            message = "flow head output must match horizon times target count"
            raise ValueError(message)
        self.register_buffer("action_mean", mean)
        self.register_buffer("action_std", std)
        self.fm_action_weights = tuple(fm_action_weights or [1.0] * action_count)
        weights = torch.tensor(self.fm_action_weights)
        if (
            weights.shape != (action_count,)
            or not torch.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            message = "FM weights must be finite positive values, one per action"
            raise ValueError(message)
        head.fm_steps = self.fm_inference_steps
        hparams.update(
            fm_inference_steps=self.fm_inference_steps,
            fm_x0=self.fm_x0,
            action_mean=mean.tolist(),
            action_std=std.tolist(),
            fm_action_weights=self.fm_action_weights,
        )

    @classmethod
    def load_for_export(
        cls,
        *,
        checkpoint_path: str | None = None,
        artifact: str | None = None,
        **kwargs: Any,
    ) -> "PatchPolicyContinuous":
        """Load a checkpoint configured for deployment export (ONNX).

        Follows the control-transformer export convention rather than
        `PatchPolicy.load_for_export`'s: the in-model image pipeline is replaced
        by Identity, so deployment supplies frames that are already cropped,
        resized AND ImageNet-normalized. That is what the kit's existing binding
        produces.

        The action fields stay in the `Remapper` but resolve to `None` at
        inference, which the per-modality `ModuleDict` passes through - they are
        training targets only, and `forward` never reads them.

        Raises:
            ValueError: unless exactly one of `checkpoint_path`, `artifact` is given.
        """
        from torch.nn import Identity  # ruff: ignore[import-outside-top-level]

        match (checkpoint_path, artifact):
            case (str() as path, None):
                model = cls.load_from_checkpoint(
                    path, map_location="cpu", weights_only=False
                )

            case (None, str() as ref):
                model = cls.load_from_wandb_artifact(
                    ref, filename="model.ckpt", map_location="cpu", weights_only=False
                )

            case _:
                msg = "specify exactly one of `checkpoint_path`, `artifact`"
                raise ValueError(msg)
        for key, value in kwargs.items():
            setattr(model, key, value)

        # the per-modality ModuleDict: index 1 of the input_transform Sequential,
        # or 2 when a chunk config unfolds the actions (`ChunkFields`) before it
        modalities = next(
            module for module in model.input_transform if isinstance(module, ModuleDict)
        )
        modalities["image"] = Identity()

        # RoPE defaults to float64 for exact long-episode frame counters; neither
        # onnxruntime-CPU nor TensorRT has a float64 Cos kernel, and a fixed
        # 6-frame serving buffer is nowhere near float32's rounding threshold
        if hasattr(model.encoder, "rope_compute_dtype"):
            model.encoder.rope_compute_dtype = torch.float32

        return model.eval()

    @override
    def train(self, mode: bool = True) -> "PatchPolicyContinuous":
        super().train(mode)
        self.image_encoder.eval()
        return self

    @staticmethod
    def _get(inputs: Mapping[str, Any], path: Path) -> Tensor:
        """Fetch a required input from the transformed batch.

        Raises:
            KeyError: if `path` is absent, which for this model is always a
                configuration error rather than an optional field.
        """
        value = key_get_default(inputs, tuple(map(MappingKey, path)), None)
        if value is None:
            msg = f"input {path!r} missing from transformed batch"
            raise KeyError(msg)

        return value

    def frame_tokens(
        self, images: Tensor, speed: Tensor, observations: Mapping[str, Tensor]
    ) -> Tensor:
        """Build a frame's tokens for export and cached decoding."""
        return self._frame_tokens(images, speed, observations)

    def _frame_tokens(
        self, images: Tensor, speed: Tensor, observations: Mapping[str, Tensor]
    ) -> Tensor:
        """Per-frame token blocks `(b, t, cam * p + 1 + o, d)`, nothing temporal.

        `images` is `(b, t, cam, c, h, w)`; a single-camera model still stacks a
        `cam=1` axis so there is one code path for both. Each camera contributes
        its own `p` patches through the same frozen encoder and projection - no
        new parameters, just a longer per-frame token block, which the trunk sees
        only as a larger `tokens_per_frame`.

        `o` is one token per extra observation, embedded exactly as speed is. They
        likewise count towards `tokens_per_frame`, so a config adding one must
        widen the trunk's to match.
        """
        with torch.no_grad():
            patches = self.image_encoder(images)  # (b, t, cam, p, d_img)

        patches = rearrange(patches, "b t cam p d -> b t (cam p) d")
        patches = self.patch_projection(patches)  # (b, t, cam * p, d)
        speed_token = self.speed_embedding(self.speed_tokenizer(speed))  # (b, t, 1, d)
        # `self.observations` order, not the batch's, so the token positions are
        # the same on every step and match what a checkpoint was trained with
        state_tokens = [
            self.observation_embeddings[name](observations[name])  # ty:ignore[not-subscriptable]
            for name in self.observations
        ]

        # scalars first so the frame block ends on a patch token (the readout)
        return torch.cat([speed_token, *state_tokens, patches], dim=-2)

    def _features(self, batch: Any) -> Tensor:
        """Per-frame readout features `(b, t, d)`."""
        inputs = self.input_transform(batch)
        # one entry per configured camera, stacked on a dedicated axis so the
        # frames stay aligned frame-for-frame across cameras
        images = torch.stack(
            [self._get(inputs, ("image", camera)) for camera in self.cameras], dim=2
        )  # (b, t, cam, c, h, w)
        tokens = self._frame_tokens(
            images,
            self._get(inputs, self.speed),
            {name: self._get(inputs, path) for name, path in self.observations.items()},
        )
        _, num_frames, _, _ = tokens.shape

        embedding = self.encoder(
            rearrange(tokens, "b t k d -> b (t k) d"), num_frames=num_frames
        )
        features = rearrange(embedding, "b (t k) d -> b t k d", t=num_frames)[:, :, -1]

        return self.norm(features) if self.norm is not None else features

    def features(self, batch: Any) -> Tensor:
        """Return full-window readout features for export validation."""
        return self._features(batch)

    def _heads(self, features: Tensor) -> Any:
        """Per-actuation `(mean, log-variance)` on a trailing axis of 2.

        A chunk head's flat `2 * H` output is split to `(..., H, 2)`, so the loss
        and the mean readout index the last axis exactly as for one step.

        Raises:
            ValueError: If Gaussian heads are not configured.
        """
        if self.heads is None:
            message = "Gaussian heads are not configured"
            raise ValueError(message)
        logits = self.heads(features)
        if self.action_horizon == 1:
            return logits

        return tree_map(lambda x: rearrange(x, "... (h p) -> ... h p", p=2), logits)

    def _predict(self, features: Tensor) -> TensorDict:
        """Gaussian mean per actuation: `features`' leading axes, `+ (H,)` if chunked."""
        if self.fm_head is not None:
            chunk = self.fm_sample(features) * self.action_std + self.action_mean
            return TensorDict({
                key: chunk[..., index]
                if self.action_horizon > 1
                else chunk[..., 0, index]
                for index, key in enumerate(self.action_names)
            })
        logits = self._heads(features)

        return TensorDict(logits).apply(operator.itemgetter((..., 0)))  # ty:ignore[invalid-return-type]

    @override
    def forward(self, batch: Any) -> TensorDict:
        # only the last frame's readout is acted on
        features = self._features(batch)[:, -1:]

        return TensorDict({"policy": self._predict(features)})

    def _step(self, batch: Any, prefix: str) -> STEP_OUTPUT:
        if self.fm_head is not None:
            return self._fm_step(batch, prefix)
        inputs = self.input_transform(batch)
        features = self._features(batch)
        logits = self._heads(features)

        # every frame is a readout, so targets keep their full time axis
        targets = {
            modality: {
                name: self._get(inputs, path).squeeze(-1)
                for name, path in names.items()
            }
            for modality, names in self.targets.items()
        }
        # before the loss, whose broadcasting would hide a target of the wrong shape
        chunk_metrics = (
            self._chunk_metrics(logits, targets) if self.action_horizon > 1 else None
        )
        if self.losses is None:
            message = "Gaussian losses are not configured"
            raise ValueError(message)
        losses = self.losses(logits, targets)

        metrics = TensorDict({"loss": losses})
        total = metrics.sum(reduce=True)
        metrics["loss", "total"] = total
        if chunk_metrics is not None:
            metrics["metric"] = chunk_metrics

        self.log_dict(
            {
                "/".join([prefix, *k]): v
                for k, v in metrics.detach().items(
                    include_nested=True, leaves_only=True
                )
            },
            sync_dist=True,
        )

        return {"loss": total}

    def fm_sample(
        self,
        features: Tensor,
        x0: Tensor | None = None,
        steps: int | None = None,
        clean_prefix: Tensor | None = None,
        prefix_mask: Tensor | None = None,
    ) -> Tensor:
        """Integrate in standardized units, including an optional RTC prefix.

        Raises:
            ValueError: If the head, step count, or initial noise shape is invalid.
        """
        if self.fm_head is None:
            message = "flow head is not configured"
            raise ValueError(message)
        count = self.fm_inference_steps if steps is None else steps
        if count < 1:
            message = "Euler steps must be positive"
            raise ValueError(message)
        flattened = features.reshape(-1, features.shape[-1]).float()
        width = self.action_horizon * len(self.action_names)
        if x0 is None:
            state = flattened.new_zeros(flattened.shape[0], width)
            if self.fm_x0 == "sample":
                state = torch.randn_like(state)
        else:
            if x0.shape != (*features.shape[:-1], width):
                message = "initial noise shape must match features and chunk width"
                raise ValueError(message)
            state = (
                x0
                .to(device=features.device, dtype=torch.float32)
                .reshape(-1, width)
                .clone()
            )
        if isinstance(self.fm_head, TokenTimeFlowHead):
            if (clean_prefix is None) != (prefix_mask is None):
                raise ValueError(
                    "RTC prefix actions and mask must be supplied together"
                )
            chunk_shape = (
                flattened.shape[0],
                self.action_horizon,
                len(self.action_names),
            )
            prefix = (
                state.new_zeros(chunk_shape)
                if clean_prefix is None
                else clean_prefix.reshape(chunk_shape)
            )
            mask = (
                state.new_zeros(chunk_shape[:2])
                if prefix_mask is None
                else prefix_mask.reshape(chunk_shape[:2])
            )
            result = self.fm_head.sample(flattened, state, prefix, mask, count)
            return result.reshape(
                *features.shape[:-1], self.action_horizon, len(self.action_names)
            )
        if clean_prefix is not None or prefix_mask is not None:
            raise ValueError("RTC conditioning requires a trained token-time head")
        for index in range(count):
            time = flattened.new_full((flattened.shape[0],), index / count)
            state = torch.add(
                state, self.fm_head.velocity(flattened, state, time) / count
            )
        return state.reshape(
            *features.shape[:-1], self.action_horizon, len(self.action_names)
        )

    def _fm_step(self, batch: Any, prefix: str) -> STEP_OUTPUT:
        if self.fm_head is None:
            message = "flow head is not configured"
            raise ValueError(message)
        inputs = self.input_transform(batch)
        features = self._features(batch)
        target = torch.stack(
            [
                self._get(inputs, self.targets[modality][name])
                for modality, name in self.action_names
            ],
            dim=-1,
        ).float()
        expected = (*features.shape[:-1], self.action_horizon, len(self.action_names))
        if target.shape != expected:
            message = f"action chunk shape {target.shape} != {expected}"
            raise ValueError(message)
        normalized = (target - self.action_mean) / self.action_std
        flattened = features.reshape(-1, features.shape[-1]).float()
        loss_mask = None
        if self.fm_action_weights != (1.0,) * len(self.action_names):
            loss_mask = (
                normalized
                .new_tensor(self.fm_action_weights)
                .repeat(self.action_horizon)
                .expand(flattened.shape[0], -1)
            )
        loss, _ = self.fm_head.loss(
            flattened,
            None,
            normalized.flatten(-2).reshape(flattened.shape[0], -1),
            mask=loss_mask,
        )
        metrics = {
            f"{prefix}/loss/fm_mse": loss.detach(),
            f"{prefix}/loss/total": loss.detach(),
        }
        if prefix == "val":
            with torch.no_grad():
                prediction = (
                    self.fm_sample(features) * self.action_std + self.action_mean
                )
                errors = (prediction - target).abs()
                metrics.update({
                    "val/metric/chunk_l1": errors.mean(),
                    "val/metric/chunk_l1_last": errors[:, -1].mean(),
                    "val/metric/chunk_l1_near": errors[
                        :, :, : self.near_horizon
                    ].mean(),
                    "val/metric/chunk_l1_near_last": errors[
                        :, -1, : self.near_horizon
                    ].mean(),
                })
                for index, (_, name) in enumerate(self.action_names):
                    metrics[f"val/metric/chunk_l1/{name}"] = errors[..., index].mean()
        self.log_dict(metrics, sync_dist=True, batch_size=features.shape[0])
        return {"loss": loss}

    def _chunk_metrics(
        self,
        logits: Mapping[str, Mapping[str, Tensor]],
        targets: Mapping[str, Mapping[str, Tensor]],
    ) -> dict[str, Tensor]:
        """L1 of the predicted chunk MEANS against the target chunk.

        Over every actuation in `targets` order, so in raw action units, the same
        space as `PatchPolicy`'s `offset_argmax_recon*` (its tokenizer's
        normalizer is the identity on palletjack): `chunk_l1` over all frames and
        steps, `_last` at the newest frame only (the served readout), `_near` over
        the first `near_horizon` steps (what runs before re-planning).

        Raises:
            ValueError: if a target is not the `(b, t, H)` chunk the head predicts,
                e.g. a config that skipped `ChunkFields`.
        """
        with torch.no_grad():
            errors = []
            for modality, names in targets.items():
                for name, target in names.items():
                    mean = logits[modality][name][..., 0]
                    if mean.shape != target.shape:
                        msg = (
                            f"target {modality}/{name} has shape "
                            f"{tuple(target.shape)}, expected the predicted chunk's "
                            f"{tuple(mean.shape)}"
                        )
                        raise ValueError(msg)
                    errors.append((mean.float() - target.float()).abs())

            error = torch.stack(errors, dim=-1)  # (b, t, h, actuations)
            near = error[:, :, : self.near_horizon]

            return {
                "chunk_l1": error.mean(),
                "chunk_l1_last": error[:, -1].mean(),
                "chunk_l1_near": near.mean(),
                "chunk_l1_near_last": near[:, -1].mean(),
            }

    @override
    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        return self._step(batch, "train")

    @override
    def validation_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        return self._step(batch, "val")

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
