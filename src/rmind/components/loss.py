from collections.abc import Callable
from typing import Any, Protocol, override, runtime_checkable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module


@runtime_checkable
class HasLogitBias(Protocol):
    logit_bias: Tensor | None


class _LogitBiasBufferMixin(Module):
    logit_bias: Tensor | None

    @override
    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # a logit_bias set during training (e.g. by LogitBiasSetter) is persisted
        # in the checkpoint, but the buffer is registered as None on init and
        # would otherwise be rejected as an unexpected key
        if self.logit_bias is None and (key := f"{prefix}logit_bias") in state_dict:
            self.logit_bias = torch.empty_like(state_dict[key])

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class FocalLoss(Module):
    """https://arxiv.org/pdf/1708.02002.pdf."""

    def __init__(self, *, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()

        self.gamma: float = gamma
        self.reduction: str = reduction

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = (1 - pt).pow(self.gamma) * ce_loss

        match self.reduction:
            case "none":
                return loss
            case "sum":
                return loss.sum()
            case _:
                return loss.mean()


class LogitBiasFocalLoss(_LogitBiasBufferMixin, FocalLoss, HasLogitBias):
    logit_bias: Tensor | None

    def __init__(self, *, logit_bias: Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__(gamma=gamma)

        self.register_buffer("logit_bias", logit_bias)

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.logit_bias is not None:
            # not in-place: callers may reuse the logits after loss computation
            input = input + self.logit_bias  # noqa: PLR6104
        return super().forward(input, target)


class LogitBiasCrossEntropyLoss(_LogitBiasBufferMixin, CrossEntropyLoss, HasLogitBias):
    logit_bias: Tensor | None

    def __init__(
        self, *args: Any, logit_bias: Tensor | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer("logit_bias", logit_bias)

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.logit_bias is not None:
            # not in-place: callers may reuse the logits after loss computation
            input = input + self.logit_bias  # noqa: PLR6104
        return super().forward(input, target)


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
    def forward(
        self, input: Tensor, target: Tensor, var: Tensor | None = None
    ) -> Tensor:  # ty:ignore[invalid-method-override]
        if var is not None:
            raise ValueError

        mean, log_var = input[..., 0], input[..., 1]
        var = self.var_pos_function(log_var)

        return super().forward(input=mean, target=target, var=var)


class L1Loss(Module):
    """L1 loss for a continuous head, as a drop-in swap for GaussianNLLLoss.

    Reads only the mean channel (input[..., 0]) of the standard [mean,
    log_var] head output layout and ignores the rest -- so it plugs into a
    `heads.continuous.*` config entry without changing the head's
    out_features, and predict()'s PREDICTION_STD/PREDICTION_PROBS/
    SCORE_LOGPROB (which read the same [mean, log_var] layout) keep working;
    their variance channel just goes untrained (no gradient reaches it) since
    this loss never reads it.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction: str = reduction

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input[..., 0], target, reduction=self.reduction)


class ActivityWeightedL1Loss(L1Loss):
    """L1Loss with per-sample weights proportional to target magnitude.

    Mirrors ActivityWeightedGaussianNLLLoss's rationale (samples where the
    control is actively engaged get higher gradient weight, counteracting
    zero-inflation in the dataset) for the L1 loss family.
    """

    def __init__(self, *args: Any, activity_weight: float = 5.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.activity_weight = activity_weight

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        per_sample = F.l1_loss(input[..., 0], target, reduction="none")  # [B]
        weights = 1.0 + self.activity_weight * target.abs()  # [B]
        weights /= weights.mean()  # keep loss scale stable
        loss = weights * per_sample

        match self.reduction:
            case "none":
                return loss
            case "sum":
                return loss.sum()
            case _:
                return loss.mean()


class HurdleGaussianNLLLoss(Module):
    """Zero-inflated (hurdle) Gaussian NLL for point-mass-at-zero controls (e.g. gas).

    Head output layout: [..., 0] = mean, [..., 1] = log_var (as GaussianNLLLoss),
    [..., 2] = gate logit (press vs no-press). The Gaussian NLL is computed on
    pressed samples only, so the mean models press magnitude instead of splitting
    between the zero mode and the press mode.
    """

    def __init__(
        self,
        *,
        gate_weight: float = 1.0,
        press_threshold: float = 0.01,
        var_pos_function: Callable[[Tensor], Tensor] = torch.exp,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.gate_weight: float = gate_weight
        self.press_threshold: float = press_threshold
        self.var_pos_function: Callable[[Tensor], Tensor] = var_pos_function
        self.reduction: str = reduction

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mean, log_var, gate = input[..., 0], input[..., 1], input[..., 2]
        press = target.abs() > self.press_threshold

        gate_loss = F.binary_cross_entropy_with_logits(
            gate, press.float(), reduction="none"
        )

        nll = F.gaussian_nll_loss(
            mean, target, self.var_pos_function(log_var), reduction="none"
        )

        if self.reduction == "none":
            return self.gate_weight * gate_loss + nll * press

        # mean: NLL is averaged over pressed samples only, gate over all
        nll_press = (nll * press).sum() / press.sum().clamp(min=1)

        return self.gate_weight * gate_loss.mean() + nll_press


class MaskedGaussianNLLLoss(Module):
    """Gaussian NLL on actively-pressed samples only (|target| > press_threshold).

    The magnitude half of a hurdle factorization: pairs with a separate
    press/no-press classifier (e.g. PolicyObjective's longitudinal mode head),
    so the mean models press magnitude instead of splitting between the zero
    mode and the press mode. Head output layout as GaussianNLLLoss:
    [..., 0] = mean, [..., 1] = log_var.
    """

    def __init__(
        self,
        *,
        press_threshold: float = 0.01,
        var_pos_function: Callable[[Tensor], Tensor] = torch.exp,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.press_threshold: float = press_threshold
        self.var_pos_function: Callable[[Tensor], Tensor] = var_pos_function
        self.reduction: str = reduction

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mean, log_var = input[..., 0], input[..., 1]
        press = target.abs() > self.press_threshold

        nll = F.gaussian_nll_loss(
            mean, target, self.var_pos_function(log_var), reduction="none"
        )

        if self.reduction == "none":
            return nll * press

        return (nll * press).sum() / press.sum().clamp(min=1)


class ActivityWeightedGaussianNLLLoss(GaussianNLLLoss):
    """GaussianNLLLoss with per-sample weights proportional to target magnitude.

    Samples where the control is actively engaged (high brake, large steer, etc.)
    receive higher gradient weight, counteracting zero-inflation in the dataset.
    """

    def __init__(self, *args: Any, activity_weight: float = 5.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.activity_weight = activity_weight

    @override
    def forward(
        self, input: Tensor, target: Tensor, var: Tensor | None = None
    ) -> Tensor:
        if var is not None:
            raise ValueError

        mean, log_var = input[..., 0], input[..., 1]
        per_sample = F.gaussian_nll_loss(
            mean,
            target,
            self.var_pos_function(log_var),
            full=self.full,
            eps=self.eps,
            reduction="none",
        )  # [B]
        weights = 1.0 + self.activity_weight * target.abs()  # [B]
        weights /= weights.mean()  # keep loss scale stable
        loss = weights * per_sample

        match self.reduction:
            case "none":
                return loss
            case "sum":
                return loss.sum()
            case _:
                return loss.mean()


class GramAnchoringLoss(Module):
    """
    Gram-based anchoring loss for feature matching.
    Based on DINOv3 implementation:
    https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py
    Uses target-driven within-frame patch uniqueness weights.
    """

    def __init__(
        self,
        *args: Any,
        patches: int,
        weight_sim: float = 1.0,
        weight_gram: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight_sim: float = weight_sim
        self.weight_gram: float = weight_gram
        self.patches: int = patches

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.detach()
        eps = 1e-6

        input_view = rearrange(input, "(bt p) d -> bt p d", p=self.patches)
        target_view = rearrange(target, "(bt p) d -> bt p d", p=self.patches)
        input_n = F.normalize(input_view, dim=-1)
        target_n = F.normalize(target_view, dim=-1)

        # Target-driven within-frame patch uniqueness weights.
        # Patches similar to many others in the same frame are downweighted.
        frame_sim = torch.einsum("bpd,bqd->bpq", target_n, target_n).clamp_min(0.0)
        eye = torch.eye(self.patches, dtype=torch.bool, device=target.device)
        weights = 1.0 / (
            frame_sim.masked_fill(eye, 0.0).sum(dim=-1) / (self.patches - 1) + eps
        )
        weights /= weights.sum(dim=1, keepdim=True) + eps

        patch_loss = F.mse_loss(input_view, target_view, reduction="none").mean(dim=-1)
        sim_loss = (weights * patch_loss).sum(dim=1).mean()

        if self.weight_gram <= 0:
            return self.weight_sim * sim_loss

        # Gram on L2-normed features weighted by patch uniqueness.
        gram_pred = torch.einsum("bpd,bqd->bpq", input_n, input_n)
        gram_gt = torch.einsum("bpd,bqd->bpq", target_n, target_n)
        pair_weights = torch.einsum("bp,bq->bpq", weights, weights)  # (bt, p, p)
        gram_loss = (pair_weights * (gram_pred - gram_gt).pow(2)).sum(dim=(1, 2)).mean()

        return self.weight_sim * sim_loss + self.weight_gram * gram_loss
