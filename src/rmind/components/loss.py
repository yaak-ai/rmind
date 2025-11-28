from collections.abc import Callable
from typing import Any, final, override

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

# Its getting out of hand


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
        weight_sim: float = 1.0,
        weight_cross: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.patches = patches
        self.timestep = timestep
        self.weight_sim = weight_sim
        self.weight_cross = weight_cross

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
        cross_time_pred = torch.matmul(input[:, :-1], target[:, 1:].transpose(-1, -2))
        cross_time_gt = torch.matmul(target[:, :-1], target[:, 1:].transpose(-1, -2))

        # B T P
        cross_time_gram = ((cross_time_pred - cross_time_gt) ** 2).mean()

        # B T P
        sim = (input * target).sum(dim=-1)

        # B T P
        sim_loss = (1.0 - sim).mean()

        return self.weight_sim * sim_loss + self.weight_cross * cross_time_gram


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
class SoftFocalGramAnchoringObjective(Module):
    def __init__(  # noqa: PLR0913
        self,
        *args: Any,
        weight_sim: float = 1.0,
        weight_gram: float = 10.0,
        patches: int = 256,
        timestep: int = 6,
        gamma: float = 2.0,
        gram_logit_scale: float = 10.0,
        gram_logit_bias: float = -10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patches = patches
        self.timestep = timestep
        self.weight_sim = weight_sim
        self.weight_gram = weight_gram
        self.gamma = gamma

        self.gram_logit_scale = torch.nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(gram_logit_scale))
        )
        self.gram_logit_bias = torch.nn.Parameter(torch.ones([]) * gram_logit_bias)

        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

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

        # 1. Cosine sim loss
        # B T P
        sim = (input * target).sum(dim=-1)
        loss_sim = (1.0 - sim).mean()

        # 2.Focal Gram Anchoring

        # B T P P
        cross_time_pred = torch.matmul(input[:, :-1], target[:, 1:].transpose(-1, -2))
        cross_time_pred = (cross_time_pred + 1.0) / 2.0
        cross_time_gt = torch.matmul(target[:, :-1], target[:, 1:].transpose(-1, -2))
        # [-1, 1] -> [0, 1]
        cross_time_gt = (cross_time_gt + 1.0) / 2.0

        cross_time_pred = (
            cross_time_pred * self.gram_logit_scale.exp() + self.gram_logit_bias
        )

        gram_loss = self.bce(cross_time_pred, cross_time_gt)
        # base entropy we can get this is the lower bound
        self_entropy = torch.special.entr(cross_time_gt) + torch.special.entr(
            1.0 - cross_time_gt
        )

        # B T P
        focal_weight = (gram_loss - self_entropy).clamp(min=0).pow(self.gamma)

        # B T P
        focal_gram_loss = (focal_weight * gram_loss).sum(dim=-1).mean()

        return self.weight_sim * loss_sim + self.weight_gram * focal_gram_loss


@final
class SoftFocalSigLIPObjective(Module):
    def __init__(  # noqa: PLR0913
        self,
        *args: Any,
        gram_weight: float = 1.0,
        clip_weight: float = 1.0,
        siglip_weight: float = 1.0,
        patches: int = 256,
        timestep: int = 5,
        gamma: int = 2,
        siglip_logit_scale: float = 10,
        siglip_logit_bias: float = 0,
        clip_logit_scale: float = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gram_weight = gram_weight
        self.clip_weight = clip_weight
        self.siglip_weight = siglip_weight
        self.patches = patches
        self.timestep = timestep
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.mse = torch.nn.MSELoss()
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L295C14-L295C75
        # Simoid loss sets to np.log(10) and -10 but thats for hard labels
        # TODO : principled way to set this # noqa: FIX002
        self.siglip_logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(siglip_logit_scale), requires_grad=False
        )
        self.siglip_logit_bias = torch.nn.Parameter(
            torch.ones([]) * siglip_logit_bias, requires_grad=False
        )
        self.clip_logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(clip_logit_scale), requires_grad=False
        )
        self.register_buffer(
            "patch_labels", torch.arange(patches, dtype=torch.long), persistent=False
        )

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Trust but ~~verify~~ detach
        target = target.detach()

        # letz keep everything in (B T) since torch is better with big rather then nested tensors
        # (B T P) D
        input = F.normalize(input, dim=-1)
        target = F.normalize(target, dim=-1)

        input = rearrange(input, "(bt p) d -> bt p d", p=self.patches)
        target = rearrange(target, "(bt p) d -> bt p d", p=self.patches)

        # B T P P
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L366
        # https://arxiv.org/pdf/2303.15343 Algorithm: 1

        cross_similarity = torch.matmul(input, target.transpose(-1, -2))
        # B T P P
        # Similarity as targets instead of 1-hot targets
        # we should ? [-1, 1] -> [0, 1] to match soft assignment simoid labels
        labels_raw = torch.matmul(target, target.transpose(-1, -2))

        # 1.Gram loss
        self_similarity = torch.matmul(input, input.transpose(-1, -2))
        gram_loss = self.mse(self_similarity, labels_raw)
        del self_similarity

        sig_logits = (
            self.siglip_logit_scale.exp() * cross_similarity + self.siglip_logit_bias
        )

        # 2.Soft focal SigLIP loss
        sig_logits = rearrange(sig_logits, "bt p0 p1 -> (bt p0) p1")
        sig_labels = rearrange(labels_raw, "bt p0 p1 -> (bt p0) p1").clamp(0, 1)

        # BCE is equivalent to sigmoid loss in SigLIP (we have soft targets)
        soft_siglip_loss = self.bce(sig_logits, sig_labels)

        # best entropy we can get this is the lower bound
        self_entropy = torch.special.entr(sig_labels) + torch.special.entr(
            1.0 - sig_labels
        )

        focal_weight = (soft_siglip_loss - self_entropy).clamp(0)

        focal_siglip_loss = (
            (focal_weight.pow(self.gamma) * soft_siglip_loss).sum(dim=-1)
        ).mean()

        del sig_logits, sig_labels, focal_weight, soft_siglip_loss

        # 3. Focal Clip loss

        clip_scale = self.clip_logit_scale.exp()
        clip_logits = clip_scale * cross_similarity

        clip_logits = rearrange(clip_logits, "bt p0 p1 -> (bt p0) p1")
        current_batch_size = clip_logits.shape[0]
        targets = repeat(
            self.patch_labels, "p -> (bt p)", bt=current_batch_size // self.patches
        )

        log_probs = F.log_softmax(clip_logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # pyright: ignore[reportCallIssue]
        pt = log_pt.exp()

        clip_loss = ((1 - pt).pow(self.gamma) * (-log_pt)).mean()

        return (
            self.siglip_weight * focal_siglip_loss
            + self.gram_weight * gram_loss
            + self.clip_weight * clip_loss
        )


@final
class SoftFocalSigLIPObjectiveOG(Module):
    def __init__(  # noqa: PLR0913
        self,
        *args: Any,
        gram_weight: float = 1.0,
        clip_weight: float = 1.0,
        siglip_weight: float = 1.0,
        patches: int = 256,
        timestep: int = 5,
        gamma: int = 2,
        siglip_logit_scale: float = 10,
        siglip_logit_bias: float = 0,
        clip_logit_scale: float = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gram_weight = gram_weight
        self.clip_weight = clip_weight
        self.siglip_weight = siglip_weight
        self.patches = patches
        self.timestep = timestep
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L295C14-L295C75
        # Simoid loss sets to np.log(10) and -10 but thats for hard labels
        # TODO : principled way to set this # noqa: FIX002
        self.siglip_logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(siglip_logit_scale), requires_grad=False
        )
        self.siglip_logit_bias = torch.nn.Parameter(
            torch.ones([]) * siglip_logit_bias, requires_grad=False
        )
        self.clip_logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(clip_logit_scale), requires_grad=False
        )

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa: PLR0914
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
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L366
        # https://arxiv.org/pdf/2303.15343 Algorithm: 1
        logit_scale = self.siglip_logit_scale
        logit_bias = self.siglip_logit_bias

        cross_similarity = torch.matmul(input, target.transpose(-1, -2))
        self_similarity = torch.matmul(input, input.transpose(-1, -2))

        # [-1, 1] -> [0, 1] to match soft assignment simoid labels
        # similarity = (1 + similarity) / 2.0 # noqa:  ERA001
        logits = logit_scale.exp() * cross_similarity + logit_bias

        # B T P P
        # Similarity as targets instead of 1-hot targets
        # we should ? [-1, 1] -> [0, 1] to match soft assignment simoid labels
        labels = torch.matmul(target, target.transpose(-1, -2))

        gram_loss = self.mse(self_similarity, labels)

        logits = rearrange(logits, "b t p0 p1 -> (b t p0) p1")
        labels = rearrange(labels, "b t p0 p1 -> (b t p0) p1").clamp(0, 1)

        # BCE is equivalent to sigmoid loss in SigLIP (we have soft targets)
        soft_siglip_loss = self.bce(logits, labels)
        eps = 1e-8

        # best entropy we can get this is the lower bound
        self_entropy = -(
            labels * torch.log(labels.clamp(min=eps))
            + (1 - labels) * torch.log((1 - labels).clamp(min=eps))
        )

        # downweight samples which are already close to lower bound
        focal_weight = (soft_siglip_loss - self_entropy).clamp(0)

        # TBD: https://github.com/google-research/big_vision/blob/6d6c28a9/big_vision/trainers/proj/image_text/siglip.py#L302
        focal_siglip_loss = (
            (focal_weight.pow(self.gamma) * soft_siglip_loss).sum(dim=-1).mean()
        )

        patch_labels = torch.arange(self.patches, device=input.device)
        patch_labels = repeat(
            patch_labels, "p -> b t p", b=input.shape[0], t=input.shape[1]
        )
        patch_labels = rearrange(patch_labels, "b t p -> (b t p)")

        logit_scale = self.clip_logit_scale.exp()
        clip_logits = logit_scale * cross_similarity

        clip_logits = rearrange(clip_logits, "b t p0 p1 -> (b t p0) p1")

        clip_loss = self.ce(clip_logits, patch_labels)
        pt = torch.exp(-clip_loss)

        clip_loss = ((1 - pt).pow(self.gamma) * clip_loss).mean()

        return (
            self.siglip_weight * focal_siglip_loss
            + self.gram_weight * gram_loss
            + self.clip_weight * clip_loss
        )


if __name__ == "__main__":
    import time

    b, t, p, d = 128, 5, 256, 1024
    import numpy as np

    num_iterations = 10
    original_times = []
    optimized_times = []

    for _i in range(num_iterations):
        input = torch.randn(b * t * p, d)
        target = torch.randn(b * t * p, d)
        original_loss = SoftFocalSigLIPObjectiveOG()
        optimized_loss = SoftFocalSigLIPObjective()

        # Time original loss computation
        start = time.time()
        out1 = original_loss(input, target)
        elapsed1 = time.time() - start
        original_times.append(elapsed1)

        # Time optimized loss computation
        start = time.time()
        out2 = optimized_loss(input, target)
        elapsed2 = time.time() - start
        optimized_times.append(elapsed2)
        torch.testing.assert_close(out1, out2)

    orig_mean = np.mean(original_times)
    orig_std = np.std(original_times)
    opt_mean = np.mean(optimized_times)
    opt_std = np.std(optimized_times)
    percent_decrease = ((orig_mean - opt_mean) / orig_mean) * 100

    """
    Original loss mean time: 1.651532s ± 0.334539s
    Optimized loss mean time: 1.397571s ± 0.119567s
    Time decrease: 15.38%
    """
