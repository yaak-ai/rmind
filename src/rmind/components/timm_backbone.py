from math import isqrt, prod
from typing import override

from timm import create_model
from torch import Tensor, nn


class TimmBackbone(nn.Module):
    """Frozen-friendly timm feature extractor with two output modes.

    - `norm_patch_tokens=False` (default): timm `features_only` intermediates at
      `out_indices` (pre final-norm), the historical rmind convention.
    - `norm_patch_tokens=True`: the FINAL block's patch tokens after the model's
      final LayerNorm -- DINOv2's `x_norm_patchtokens`, as consumed by DINO-WM
      (https://github.com/gaoyuezhou/dino_wm); prefix (cls/register) tokens are
      dropped. `out_indices` is ignored in this mode.

    Both modes return `(..., C, H, W)`.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        *,
        out_indices: list[int] | None = None,
        img_size: list[int] | None = None,
        norm_patch_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.norm_patch_tokens = norm_patch_tokens
        self.model: nn.Module = (
            create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
            if norm_patch_tokens
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

        if self.norm_patch_tokens:
            # (B, prefix + P, D), final norm applied by timm's forward_features
            tokens = self.model.forward_features(x)
            tokens = tokens[:, self.model.num_prefix_tokens :]
            grid = isqrt(tokens.shape[1])
            x = tokens.transpose(1, 2).reshape(-1, tokens.shape[-1], grid, grid)
        else:
            x = self.model(x)[-1]

        *_, c, h, w = x.shape
        return x.view(*b, c, h, w)
