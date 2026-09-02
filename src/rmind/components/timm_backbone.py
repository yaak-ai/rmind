from math import isqrt, prod
from typing import override

import torch
from timm import create_model
from torch import Tensor, nn


class TimmBackbone(nn.Module):
    """Frozen-friendly timm feature extractor with three output modes.

    - default (`out_indices`): timm `features_only` intermediates, PRE final-norm
      -- the historical rmind convention.
    - `norm_patch_tokens=True`: the FINAL block's patch tokens after the model's
      final LayerNorm -- DINOv2's `x_norm_patchtokens`, as consumed by DINO-WM
      (https://github.com/gaoyuezhou/dino_wm); prefix (cls/register) tokens are
      dropped. `out_indices` is ignored.
    - `norm_indices=[i, ...]`: block-`i` intermediates WITH the final LayerNorm
      applied (timm `forward_intermediates(norm=True)`), channel-concatenated
      when more than one index is given -- e.g. `[10]` for "layer 10 + norm",
      `[10, 11]` for "layer 10 + norm ⊕ final layer + norm" (C doubles).

    All modes return `(..., C, H, W)`.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        *,
        out_indices: list[int] | None = None,
        img_size: list[int] | None = None,
        norm_patch_tokens: bool = False,
        norm_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        if norm_patch_tokens and norm_indices is not None:
            msg = "norm_patch_tokens and norm_indices are mutually exclusive"
            raise ValueError(msg)
        self.norm_patch_tokens = norm_patch_tokens
        self.norm_indices = norm_indices
        self.model: nn.Module = (
            create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
            if norm_patch_tokens or norm_indices is not None
            else create_model(
                model_name,
                pretrained=True,
                features_only=True,
                out_indices=out_indices,
                img_size=img_size,
            )
        )

    @override
    def forward(self, input: Tensor) -> Tensor:
        *b, c, h, w = input.shape
        x = input.view(prod(b), c, h, w)

        if self.norm_indices is not None:
            feats = self.model.forward_intermediates(
                x,
                indices=self.norm_indices,
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
            x = torch.cat(feats, dim=1)
        elif self.norm_patch_tokens:
            # (B, prefix + P, D), final norm applied by timm's forward_features
            tokens = self.model.forward_features(x)
            tokens = tokens[:, self.model.num_prefix_tokens :]
            # NON-SQUARE inputs are legal (nero-arms feeds a 10x16 grid), so take
            # the grid from the patch embedding rather than assuming isqrt(P).
            # Identical result for square inputs.
            grid = getattr(getattr(self.model, "patch_embed", None), "grid_size", None)
            if grid is None or grid[0] * grid[1] != tokens.shape[1]:
                side = isqrt(tokens.shape[1])
                grid = (side, side)
            x = tokens.transpose(1, 2).reshape(-1, tokens.shape[-1], *grid)
        else:
            x = self.model(x)[-1]

        *_, c, h, w = x.shape
        return x.view(*b, c, h, w)
