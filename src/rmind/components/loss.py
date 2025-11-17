from collections.abc import Callable
from typing import Any, final, override

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module


class FocalLoss(Module):
    """https://arxiv.org/pdf/1708.02002.pdf."""

    def __init__(self, *, gamma: float = 2.0) -> None:
        super().__init__()

        self.gamma: float = gamma

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_loss)

        return ((1 - pt).pow(self.gamma) * ce_loss).mean()


class LogitBiasMixin:
    _logit_bias: Tensor | None  # pyright: ignore[reportUninitializedInstanceVariable]

    @property
    def logit_bias(self) -> Tensor | None:
        return self._logit_bias

    @logit_bias.setter
    def logit_bias(self, value: Tensor | None) -> None:
        match value:
            case Tensor():
                if hasattr(self, "_logit_bias"):
                    del self._logit_bias

                self.register_buffer("_logit_bias", value, persistent=False)  # pyright: ignore[reportAttributeAccessIssue]

            case None:
                self._logit_bias = None


class LogitBiasFocalLoss(LogitBiasMixin, FocalLoss):
    def __init__(self, *, logit_bias: Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__(gamma=gamma)

        self._logit_bias: Tensor | None = logit_bias

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # pyright: ignore[reportOperatorIssue]


class LogitBiasCrossEntropyLoss(LogitBiasMixin, CrossEntropyLoss):
    def __init__(
        self, *args: Any, logit_bias: Tensor | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self._logit_bias: Tensor | None = logit_bias

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # pyright: ignore[reportOperatorIssue]


class GaussianNLLLoss(torch.nn.GaussianNLLLoss):
    def __init__(
        self,
        *args: Any,
        # NOTE: use torch.ones_like to get vanilla MSE
        var_pos_function: Callable[[Tensor], Tensor] = torch.exp,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.var_pos_function: Callable[[Tensor], Tensor] = var_pos_function

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, input: Tensor, target: Tensor, var: Tensor | None = None
    ) -> Tensor:
        if var is not None:
            raise ValueError

        mean, log_var = input[..., 0], input[..., 1]
        var = self.var_pos_function(log_var)

        return super().forward(input=mean, target=target, var=var)


# https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py
@final
class GramAnchoringObjective(Module):
    def __init__(
        self,
        *args: Any,
        weight: float = 1.0,
        patches: int = 256,
        timestep: int = 6,
        gamma: int = 2,
        tau: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.patches = patches
        self.timestep = timestep
        self.gamma = gamma
        self.tau = tau
        self.mse_loss = torch.nn.MSELoss()

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Trust but ~~verify~~ detach
        target = target.detach()

        input = rearrange(
            input, "(b t p) d -> b t p d", t=self.timestep, p=self.patches
        )
        target = rearrange(
            target, "(b t p) d -> b t p d", t=self.timestep, p=self.patches
        )

        # B T P D
        input = F.normalize(input, dim=-1)
        target = F.normalize(target, dim=-1)

        # B T P P
        cross_time_pred = torch.matmul(input[:, 1:], target[:, :-1].transpose(-1, -2))
        cross_time_gt = torch.matmul(target[:, 1:], target[:, :-1].transpose(-1, -2))

        # B T P
        similarity = (target[:, 1:] * target[:, :-1]).sum(dim=-1)
        gating_weight = (1.0 - similarity).pow(self.gamma)

        # B T P
        gt_prob = torch.softmax(cross_time_gt / self.tau, dim=-1)
        pred_prob = torch.softmax(cross_time_pred / self.tau, dim=-1)

        kl = (gt_prob * (gt_prob.add(1e-8).log() - pred_prob.add(1e-8).log())).sum(
            dim=-1
        )  # [B,T-1,P]
        loss_kl = (gating_weight * kl).sum() / (gating_weight.sum() + 1e-6)

        # B T P
        sim_loss = (
            gating_weight * (1.0 - (input[:, 1:] * target[:, 1:]).sum(dim=-1))
        ).sum() / (gating_weight.sum() + 1e-6)
        return self.weight * (sim_loss + loss_kl)


@final
class FocalCLIPbjective(Module):  # ignore typos
    def __init__(
        self,
        *args: Any,
        weight: float = 1.0,
        patches: int = 256,
        timestep: int = 6,
        gamma: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.patches = patches
        self.timestep = timestep
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
        self.gamma = gamma
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L295C14-L295C75
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Trust but ~~verify~~ detach
        target = target.detach()

        input = rearrange(
            input, "(b t p) d -> b t p d", t=self.timestep, p=self.patches
        )
        target = rearrange(
            target, "(b t p) d -> b t p d", t=self.timestep, p=self.patches
        )

        # B T P D
        input = F.normalize(input, dim=-1)
        target = F.normalize(target, dim=-1)

        # https://github.com/openai/CLIP/blob/main/clip/model.py#L366
        logit_scale = self.logit_scale.exp()
        # B T P P
        logits = logit_scale * torch.matmul(input, target.transpose(-1, -2))

        # B T P P
        labels = torch.arange(self.patches, device=input.device)
        labels = repeat(labels, "p -> b t p", b=logits.shape[0], t=self.timestep)

        logits = rearrange(logits, "b t p0 p1 -> (b t p0) p1")
        labels = rearrange(labels, "b t p -> (b t p)")

        clip_loss = self.ce(logits, labels)
        pt = torch.exp(-clip_loss)

        return self.weight * ((1 - pt).pow(self.gamma) * clip_loss).mean()


@final
class SoftFocalSigLIPObjective(Module):  # ignore typos
    def __init__(
        self,
        *args: Any,
        weight: float = 1.0,
        patches: int = 256,
        timestep: int = 6,
        gamma: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.patches = patches
        self.timestep = timestep
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L295C14-L295C75
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.1))
        self.logit_bias = torch.nn.Parameter(torch.ones([]) * -10)

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Trust but ~~verify~~ detach
        target = target.detach()

        input = rearrange(
            input, "(b t p) d -> b t p d", t=self.timestep, p=self.patches
        )
        target = rearrange(
            target, "(b t p) d -> b t p d", t=self.timestep, p=self.patches
        )

        # B T P D
        input = F.normalize(input, dim=-1)
        target = F.normalize(target, dim=-1)

        # https://github.com/openai/CLIP/blob/main/clip/model.py#L366
        logit_scale = self.logit_scale.exp()
        # B T P P
        logits = logit_scale * torch.matmul(input, target.transpose(-1, -2))

        # B T P P
        # Similarity as targets instead of 1-hot    targets
        labels = torch.matmul(target, target.transpose(-1, -2)).clamp(0, 1)

        logits = rearrange(logits, "b t p0 p1 -> (b t p0) p1")
        labels = rearrange(labels, "b t p0 p1 -> (b t p0) p1")

        # BCE is equivalent to sigmoid loss in SigLIP (we have soft targets)
        pred = torch.sigmoid(logits)
        soft_siglip_loss = self.bce(logits, labels)

        focal_siglip_loss = (
            ((labels - pred).abs().pow(self.gamma) * soft_siglip_loss)
            .sum(dim=-1)
            .mean()
        )

        return self.weight * focal_siglip_loss
