"""Image-pipeline pieces that the driving configs never needed."""

from typing import final, override

from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

__all__ = ["LetterboxResize"]


@final
class LetterboxResize(Module):
    """Uniform-scale resize to a square with symmetric padding.

    Contract §7.2: the nero-arms cameras have DIFFERENT aspect ratios (`base`
    1920x1080 = 16:9, `side_left`/`side_right` 1280x800 = 16:10). Feeding all
    three through the driving configs' plain `Resize(S)` would scale x and y by
    different factors, which:

    * scales `fx` and `fy` independently, so the resolution-normalised
      intrinsics in `camera_cond` (§7.1) no longer describe the pixels the
      policy sees -- the conditioning vector's whole purpose is defeated;
    * makes the same physical object a different shape in `base` than in the
      side views, which a shared frozen ViT has no way to undo.

    Letterboxing keeps the scale isotropic; the matching rewrite of the
    conditioning vector is `rmind.data.nero.letterbox_camera_cond`.

    NOTE: all three cameras still land on the SAME patch grid, so `base`'s wider
    field of view gets coarser real-world coverage per patch than the side
    cameras. Expected, and worth watching if the overhead view underperforms.
    """

    def __init__(self, *, size: int, fill: float = 0.0) -> None:
        super().__init__()
        self.size = size
        self.fill = fill

    @override
    def forward(self, x: Tensor) -> Tensor:
        *batch, c, h, w = x.shape
        scale = min(self.size / w, self.size / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        flat = x.reshape(-1, c, h, w)
        if (new_w, new_h) != (w, h):
            flat = F.interpolate(
                flat.float(), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).to(x.dtype)
        pad_w, pad_h = self.size - new_w, self.size - new_h
        left, top = pad_w // 2, pad_h // 2
        flat = F.pad(flat, (left, pad_w - left, top, pad_h - top), value=self.fill)
        return flat.reshape(*batch, c, self.size, self.size)
