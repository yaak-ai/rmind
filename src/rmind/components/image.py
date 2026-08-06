"""Image-pipeline pieces that the driving configs never needed."""

from typing import final, override

from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

__all__ = ["LetterboxResize"]


@final
class LetterboxResize(Module):
    """Uniform-scale resize to a fixed grid with symmetric padding.

    Why nero-arms needs this and the driving configs did not
    -------------------------------------------------------
    The three nero-arms cameras have different aspect ratios (contract §7.2:
    `base` 1920x1080 = 16:9, `side_left`/`side_right` 1280x800 = 16:10), and
    rbyte delivers each one **isotropically downscaled to its own grid** --
    `base` `(270, 480)`, `side_*` `(300, 480)`. That choice keeps the
    resolution-normalised intrinsics in `camera_cond` exactly valid with no
    propagation work, but it means the three image tensors do **not** share H/W,
    so unifying them is rmind's job, here.

    A plain anisotropic `Resize` would be wrong twice over: it scales `fx` and
    `fy` by different factors, so the conditioning vector stops describing the
    pixels the policy sees; and it makes the same physical object a different
    shape in `base` than in the side views, which a shared frozen ViT cannot
    undo. Letterboxing keeps the scale isotropic. The matching rewrite of the
    conditioning vector is `rmind.data.nero.letterbox_camera_cond`.

    `size` may be a single int (square) or `(height, width)`. With
    `(300, 480)` this pads `base` by 15 rows top and bottom and leaves the side
    cameras untouched; a following isotropic resize to the ViT grid then needs
    no further intrinsics correction.

    NOTE: all three cameras land on the same patch grid, so `base`'s wider field
    of view gets coarser real-world coverage per patch than the side cameras --
    and `base` additionally spends 2 of its 10 patch rows on letterbox padding.
    Expected; the first thing to look at if the overhead view underperforms.
    """

    def __init__(self, *, size: int | tuple[int, int], fill: float = 0.0) -> None:
        super().__init__()
        self.size: tuple[int, int] = (
            (size, size) if isinstance(size, int) else tuple(size)
        )
        self.fill = fill

    @override
    def forward(self, x: Tensor) -> Tensor:
        *batch, c, h, w = x.shape
        target_h, target_w = self.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        flat = x.reshape(-1, c, h, w)
        if (new_w, new_h) != (w, h):
            flat = F.interpolate(
                flat.float(), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).to(x.dtype)
        pad_w, pad_h = target_w - new_w, target_h - new_h
        left, top = pad_w // 2, pad_h // 2
        flat = F.pad(flat, (left, pad_w - left, top, pad_h - top), value=self.fill)
        return flat.reshape(*batch, c, target_h, target_w)
