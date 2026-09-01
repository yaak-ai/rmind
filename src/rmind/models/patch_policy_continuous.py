from collections.abc import Mapping
from typing import Annotated, Any, Final, final, override

import pytorch_lightning as pl
import torch
from einops import rearrange
from pydantic import Field, InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils._pytree import MappingKey  # noqa: PLC2701

from rmind.components import optimizers
from rmind.components.containers import ModuleDict
from rmind.config import HydraConfig, init_hydra_param
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.models.control_transformer import PredictionConfig
from rmind.utils._wandb import LoadableFromArtifact
from rmind.utils.pytree import key_get_default

type Path = tuple[str, ...]

# modality key whose heads are classifiers rather than Gaussians
DISCRETE: Final = "discrete"


@final
class PatchPolicyContinuous(pl.LightningModule, LoadableFromArtifact):
    """Patch-policy trunk with one continuous Gaussian head per actuation.

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
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        input_transform: HydraConfig[Module] | InstanceOf[Module],
        image_encoder: HydraConfig[Module] | InstanceOf[Module],
        patch_projection: HydraConfig[Module] | InstanceOf[Module],
        speed_tokenizer: HydraConfig[Module] | InstanceOf[Module],
        speed_embedding: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module],
        heads: HydraConfig[ModuleDict] | InstanceOf[ModuleDict],
        losses: HydraConfig[ModuleDict] | InstanceOf[ModuleDict],
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
        self.encoder = init_hydra_param(hparams, "encoder", encoder)
        self.heads = init_hydra_param(hparams, "heads", heads)
        self.losses = init_hydra_param(hparams, "losses", losses)
        self.norm = (
            init_hydra_param(hparams, "norm", norm) if norm is not None else None
        )

        # the ViT is a feature extractor here, exactly as in PatchPolicy
        self.image_encoder.requires_grad_(False).eval()  # noqa: FBT003

        self.observation_embeddings: ModuleDict | None = (
            init_hydra_param(hparams, "observation_embeddings", observation_embeddings)
            if observation_embeddings is not None
            else None
        )

        self.targets: Mapping[str, Mapping[str, Path]] = targets
        self.cameras: tuple[str, ...] = cameras
        self.speed: Path = speed
        self.observations: Mapping[str, Path] = observations or {}

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
        self.save_hyperparameters(hparams)

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
        from torch.nn import Identity  # noqa: PLC0415

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

        # index 1 of the input_transform Sequential is the per-modality ModuleDict
        model.input_transform[1]["image"] = Identity()

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

    def _predict(self, features: Tensor) -> TensorDict:
        """One scalar command per actuation, shaped like `features`' leading axes.

        `continuous` heads are Gaussian `(mean, log_var)` and contribute their MEAN.
        `discrete` heads are classifiers over `num_classes` and contribute the argmax
        decoded to a command: for a 3-class fork the classes are
        `{lower, hold, raise}` and decode to `{-1, 0, +1}` -- the bin CENTRES would
        give +/-0.67, but the fork command is a rate and the operator data is
        saturated at +/-1, so the extremes are the honest decode.

        The predicted variance is deliberately not returned: it is a training signal,
        not something the vehicle acts on.
        """
        logits = self.heads(features)
        out: dict[str, dict[str, Tensor]] = {}
        for modality, heads in logits.items():
            if modality == DISCRETE:
                out[modality] = {
                    name: (head.argmax(dim=-1) - (head.shape[-1] // 2)).to(head.dtype)
                    for name, head in heads.items()
                }
            else:
                out[modality] = {
                    name: head[..., 0] for name, head in heads.items()
                }

        return TensorDict(out)  # ty:ignore[invalid-return-type]

    @override
    def forward(self, batch: Any) -> TensorDict:
        # only the last frame's readout is acted on
        features = self._features(batch)[:, -1:]

        return TensorDict({"policy": self._predict(features)})

    def _step(self, batch: Any, prefix: str) -> STEP_OUTPUT:
        inputs = self.input_transform(batch)
        features = self._features(batch)
        logits = self.heads(features)

        # every frame is a readout, so targets keep their full time axis
        targets = {
            modality: {
                name: self._get(inputs, path).squeeze(-1)
                for name, path in names.items()
            }
            for modality, names in self.targets.items()
        }
        losses = self.losses(logits, targets)

        metrics = TensorDict({"loss": losses})
        total = metrics.sum(reduce=True)
        metrics["loss", "total"] = total

        # Interpretable companions to the losses, in the ACTUATION's own units.
        # A Gaussian NLL can move for two reasons -- the mean moved, or the predicted
        # variance moved -- and only the first one steers the truck. `fork1` made that
        # concrete: its val NLL went -1.09 -> +41.08 (over 100% of the total, with the
        # other two heads negative) purely through variance collapse on a target that
        # is ~87% zero and ~11% saturated at +/-1. L1 on the mean is immune to that,
        # so it is the metric to read when deciding whether an arm is improving.
        with torch.no_grad():
            scores: dict[str, dict[str, Tensor]] = {}
            for modality, heads in logits.items():
                for name, head in heads.items():
                    target = targets[modality][name]
                    if modality == DISCRETE:
                        pred = head.argmax(dim=-1)
                        scores.setdefault("acc", {})[f"{modality}/{name}"] = (
                            (pred == target.long()).float().mean()
                        )
                    else:
                        scores.setdefault("l1", {})[f"{modality}/{name}"] = (
                            (head[..., 0] - target).abs().mean()
                        )
            for kind, values in scores.items():
                for name, value in values.items():
                    metrics[kind, *name.split("/")] = value

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
