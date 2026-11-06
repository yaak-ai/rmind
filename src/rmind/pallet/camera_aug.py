"""Make rendered frames look like the real fork camera's: motion blur, sensor noise, video
compression, exposure / white balance and lens vignetting.

The synthetic renders are clean: no blur while driving, no noise in dim aisles, no h264
artefacts, perfect exposure. Real cam_fork frames have all of these. Applied to synthetic
samples only (the real ones already have them), each effect independently with its own
probability, on a `(3, H, W)` float image in [0, 1].
"""

from typing import Final

import cv2
import numpy as np
import torch
from torch import Tensor

# probability of each degradation
P_BLUR: Final = 0.5
P_EXPOSURE: Final = 0.5
P_VIGNETTE: Final = 0.4
P_NOISE: Final = 0.6
P_COMPRESSION: Final = 0.6
P_MOSTLY_HORIZONTAL: Final = 0.7  # driving and turning blur mostly along image rows


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    d = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    for t in np.linspace(-c, c, 4 * length):
        x, y = np.round(c + t * d).astype(int)
        k[y, x] = 1
    return k / k.sum()


def camera_realism(image: Tensor, rng: np.random.Generator) -> Tensor:
    """`(3, H, W)` float in [0, 1] -> the same with random real-camera degradations.

    Blur lengths are relative to the image width, so crops and full frames get
    comparable amounts.
    """
    img = image.permute(1, 2, 0).numpy().astype(np.float32)
    h, w = img.shape[:2]
    if rng.random() < P_BLUR:
        length = int(max(3, round(w * rng.uniform(0.002, 0.01))))
        length += 1 - length % 2
        angle = (
            rng.normal(0, 25)
            if rng.random() < P_MOSTLY_HORIZONTAL
            else rng.uniform(0, 180)
        )
        img = cv2.filter2D(img, -1, _motion_kernel(length, angle))
    if rng.random() < P_EXPOSURE:  # gamma and white balance
        img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.4)
        img *= rng.uniform(0.9, 1.1, size=3).astype(np.float32)
    if rng.random() < P_VIGNETTE:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
        img *= (1 - rng.uniform(0.1, 0.4) * r**2)[..., None]
    if rng.random() < P_NOISE:  # shot noise (brightness dependent) + read noise
        scale = rng.uniform(0.005, 0.04)
        img += rng.normal(0, 1, img.shape).astype(np.float32) * (
            scale * np.sqrt(np.clip(img, 0, 1)) + scale / 3
        )
    img = np.clip(img, 0, 1)
    if rng.random() < P_COMPRESSION:
        ok, buf = cv2.imencode(
            ".jpg",
            (img * 255).astype(np.uint8),
            [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(25, 75))],
        )
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
