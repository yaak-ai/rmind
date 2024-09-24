import os
from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from cargpt.utils.camera import Camera
from cargpt.utils.inverse_warp import inverse_warp


class FocalLoss(Module):
    """https://arxiv.org/pdf/1708.02002.pdf"""

    def __init__(self, *, gamma: float = 2.0):
        super().__init__()

        self.gamma = gamma

    @override
    def forward(
        self, input: Float[Tensor, "b d"], target: Int[Tensor, "b"]
    ) -> Float[Tensor, ""]:
        ce_loss = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_loss)

        return ((1 - pt).pow(self.gamma) * ce_loss).mean()


class LogitBiasMixin:
    @property
    def logit_bias(self) -> Float[Tensor, "d"] | None:
        return self._logit_bias

    @logit_bias.setter
    def logit_bias(self, value: Float[Tensor, "d"] | None):
        match value:
            case Tensor():
                if hasattr(self, "_logit_bias"):
                    del self._logit_bias

                self.register_buffer("_logit_bias", value, persistent=True)  # pyright: ignore[reportAttributeAccessIssue]

            case None:
                self._logit_bias = None


class LogitBiasFocalLoss(LogitBiasMixin, FocalLoss):
    def __init__(
        self, *, logit_bias: Float[Tensor, "d"] | None = None, gamma: float = 2.0
    ):
        super().__init__(gamma=gamma)

        self.logit_bias = logit_bias

    @override
    def forward(self, input: Float[Tensor, "b d"], target: Int[Tensor, "b"]):
        return super().forward(input + self.logit_bias, target)


class LogitBiasCrossEntropyLoss(LogitBiasMixin, CrossEntropyLoss):
    def __init__(self, *args, logit_bias: Float[Tensor, "d"] | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.logit_bias = logit_bias

    @override
    def forward(self, input: Float[Tensor, "b d"], target: Int[Tensor, "b"]):
        return super().forward(input + self.logit_bias, target)


class GaussianNLLLoss(torch.nn.GaussianNLLLoss):
    """
    Class that makes vanilla torch.nn.GaussianNLLLoss compatible with carGPT pipeline
    """

    def __init__(
        self,
        *args,
        var_pos_function: Callable[
            [Tensor], Tensor
        ] = torch.exp,  # NOTE: use torch.ones_like to get vanilla MSE
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.var_pos_function = var_pos_function

    @override
    def forward(self, input: Tensor, target: Float[Tensor, "b"], var=None):
        mean, log_var = input[..., 0], input[..., 1]
        return super().forward(
            input=mean, target=target, var=self.var_pos_function(log_var)
        )


class PhotoGeometryLoss(Module):
    def __init__(
        self,
        with_ssim: bool,
        with_mask: bool,
        with_auto_mask: bool,
        camera_calibration_path: str,
    ):
        self.ssim_loss = SSIM().cuda() if with_ssim else None
        self.with_auto_mask = with_auto_mask
        self.with_mask = with_mask
        self.camera_model = Camera.from_params(camera_calibration_path)

    def forward(
        self, tgt_img, ref_imgs, tgt_disparity, ref_disparities, camera_model, poses
    ):
        ref_imgs = torch.cat([ref_img.unsqueeze(1) for ref_img in ref_imgs], axis=1)
        poses = torch.cat([pose.unsqueeze(1) for pose in poses], axis=1)
        ref_disparities = torch.cat(
            [ref_disparity.unsqueeze(1) for ref_disparity in ref_disparities], axis=1
        )

        _B, V, _C, _H, _W = ref_disparities.shape

        tgt_warped_list = []
        diff_identity_list = []
        diff_img_list = []
        valid_mask_list = []
        diff_disp_list = []
        projected_disp_list = []
        computed_disp_list = []
        self_mask_list = []

        # currently only one view is possible, but I will leave it
        for view in range(V):
            (tgt_warped_from_ref, projected_disp, computed_disp) = inverse_warp(
                ref_imgs[:, view],
                tgt_disparity,
                ref_disparities[:, view],
                poses[:, view],
                camera_model,
            )

            (diff_img, diff_identity, diff_disp, valid_mask, self_mask) = (
                self.projection_and_disparity_diff(
                    tgt_img,
                    ref_imgs[:, view],
                    tgt_warped_from_ref,
                    projected_disp,
                    computed_disp,
                )
            )

            tgt_warped_list.append(tgt_warped_from_ref)
            diff_img_list.append(diff_img)
            diff_identity_list.append(diff_identity)
            valid_mask_list.append(valid_mask)
            diff_disp_list.append(diff_disp)
            self_mask_list.append(self_mask)
            projected_disp_list.append(projected_disp)
            computed_disp_list.append(computed_disp)

        inputs = [
            tgt_warped_list,
            diff_img_list,
            diff_identity_list,
            valid_mask_list,
            diff_disp_list,
            self_mask_list,
            projected_disp_list,
            computed_disp_list,
        ]
        outputs = list(map(torch.cat, inputs, [1] * len(inputs)))
        (
            tgt_warped,
            diff_img,
            diff_identity,
            valid_mask,
            diff_disp,
            self_mask,
            projected_disp,
            computed_disp,
        ) = outputs

        # Eq 5. in monodepth2 paper (minimum of all views)
        # https://arxiv.org/pdf/1806.01260.pdf
        if self.with_auto_mask:
            diff_warped = torch.gather(
                diff_img, 1, torch.argmin(diff_img, axis=1, keepdim=True)
            )
            diff_identity = torch.gather(
                diff_identity, 1, torch.argmin(diff_identity, axis=1, keepdim=True)
            )
            # add random values to break ties
            # https://github.com/nianticlabs/monodepth2/blob/master/trainer.py#L467
            diff_identity += (
                torch.randn(diff_identity.shape, device=diff_identity.device) * 0.00001
            )
            auto_mask = (diff_identity > diff_warped).float()
            valid_mask = auto_mask.expand_as(valid_mask) * valid_mask

        # [B, 2, H, W]
        # minium photometric reprojection error match in multiple views
        # Eq 4 in monodepth2 paper
        # https://arxiv.org/pdf/1806.01260.pdf
        indices = torch.argmin(diff_img, dim=1, keepdim=True)
        diff_img = torch.gather(diff_img, 1, indices)
        diff_disp = torch.gather(diff_disp, 1, indices)
        valid_mask_min = torch.gather(valid_mask, 1, indices)

        # Weight mask Eq 7) in https://jwbian.net/Papers/SC_Depth_IJCV_21.pdf
        # Object boundary which are in motion have a large change in depth
        # So we mask them out
        if self.with_mask:
            diff_img = diff_img * self_mask

        # Photo loss goes of pixels which have minimum reprojection loss
        photometric_loss = self.mean_on_mask_d(diff_img, valid_mask_min, device=device)
        # geometric loss only over pixels which
        geometric_loss = self.mean_on_mask_d(diff_disp, valid_mask_min, device=device)

        return (
            photometric_loss,
            geometric_loss,
            tgt_warped,
            projected_disp,
            computed_disp,
            valid_mask,
            self_mask,
            valid_mask_min,
        )

    @staticmethod
    def mean_on_mask_d(diff, valid_mask, device=torch.device("cpu"), min_sum=100):
        """Compute mean value given a binary mask."""

        mask = valid_mask.expand_as(diff)
        if mask.sum() > min_sum:
            mean_value = (diff * mask).sum() / mask.sum()
        else:
            mean_value = torch.tensor(0).float().to(device)
        return mean_value

    def projection_and_disparity_diff(
        self,
        tgt_img,
        ref_img,
        tgt_warped_from_ref,
        projected_disparity,
        computed_disparity,
    ):
        # Mask of pixels which are set to zero in the warped image f.ex don't map between views
        valid_mask_ref = (
            tgt_warped_from_ref.abs().mean(dim=1, keepdim=True) > 1e-3
        ).float()
        valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask = valid_mask_tgt * valid_mask_ref

        # L1 photometric projection loss
        diff_img = (tgt_img - tgt_warped_from_ref).abs().mean(dim=1, keepdim=True)
        # L1 photometric projection identity loss
        diff_identity = (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)

        # SSIM Loss
        if self.ssim_loss is not None:
            diff_ssim_map = self.ssim_loss(tgt_img, tgt_warped_from_ref).mean(
                dim=1, keepdim=True
            )
            diff_img = 0.15 * diff_img + 0.85 * diff_ssim_map
            identity_ssim_map = self.ssim_loss(tgt_img, ref_img).mean(
                dim=1, keepdim=True
            )
            diff_identity = 0.15 * diff_identity + 0.85 * identity_ssim_map

        diff_disparity = (computed_disparity - projected_disparity).abs() / (
            computed_disparity + projected_disparity
        )
        # Weight mask Eq 7) in https://jwbian.net/Papers/SC_Depth_IJCV_21.pdf
        # Object boundary which are in motion have a large change in depth
        # So we mask them out
        self_mask = (1 - diff_disparity).detach()
        return diff_img, diff_identity, diff_disparity, valid_mask, self_mask


class SSIM(Module):
    """
    Layer to compute the Structural Similarity Index (SSIM) loss between a pair of images.
    """

    def __init__(self):
        super().__init__()
        k = 3
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
