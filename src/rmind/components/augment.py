"""Train-time augmentations for `PatchPolicy` against trajectory-head lateral
drift (`/nasa/alex/docs/traj_drift_rmind_experiments_plan.md`, task R1).

`YawShiftCrop` and `ClipPhotometric` are identity in eval mode, so val and
the export graph (which swaps the image `ModuleDict` entry for
`Normalize`/`Identity`) never see them. `FixedFrameMask` is applied in train
and eval alike and is part of the export graph.
"""

import math
from collections.abc import Mapping
from typing import Any, final, override

import kornia
import torch
from pydantic import validate_call
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import PyTree


def _get(tree: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(tree, Mapping) or tree.get(key) is None:
            return None
        tree = tree[key]
    return tree


def _assoc(tree: Mapping[str, Any], path: tuple[str, ...], value: Any) -> PyTree:
    key, *rest = path
    if not rest:
        return {**tree, key: value}
    return {**tree, key: _assoc(tree[key], tuple(rest), value)}


def rotate_ego_frame(xy: Tensor, psi: Tensor) -> Tensor:
    """Re-express ego-frame points in a frame yawed by `psi` radians.

    Conventions are those of `rmind.components.dead_reckoning` and
    `waypoints/xy_normalized` (`ST_Rotate(ST_Translate(geom, -ego), radians(heading))`):
    x = right, y = forward, and heading is a compass bearing, so `psi > 0` is
    the ego yawed to the RIGHT. A point dead ahead `(0, d)` then lands at
    `(-d sin psi, d cos psi)`: to the left of the new forward axis.

    Args:
        xy: `(*batch, n, 2)`.
        psi: `(*batch,)` radians.

    Returns:
        `(*batch, n, 2)`.
    """
    cos, sin = psi.cos()[..., None], psi.sin()[..., None]
    x, y = xy[..., 0], xy[..., 1]
    return torch.stack([x * cos - y * sin, x * sin + y * cos], dim=-1)


@final
class YawShiftCrop(Module):
    """Synthetic heading offset: crop a horizontally shifted window out of the
    raw frame (a small-angle yaw of the camera) and rotate the per-frame
    targets to match, so the trajectory head learns to steer back onto the
    human path instead of holding its current heading.

    Placement: after `ChunkFields` (the per-anchor `trajectory_target` and the
    images are both `(b, t, ...)` by then) and before the image `ModuleDict`,
    whose `CenterCrop` must be `(H, crop_width)`. This module only narrows the
    width; in eval it is identity and that `CenterCrop` takes the centre
    window, so train at `psi = 0` and eval agree exactly.

    Sign: `psi > 0` is the camera yawed RIGHT (compass sense, like `heading`).
    The window then moves right, the scene appears shifted left in the image,
    and the targets bend LEFT (back toward the recorded path):
      - `trajectory_target` `(b, t, P, 3)`, `(x, y)` rotated by
        `rotate_ego_frame`, theta minus psi (re-wrapped);
      - `waypoints` `(b, t, n, 2)`, rotated the same way (the `/100` scale
        commutes with rotation).

    Shift <-> angle: equidistant fisheye near the centre, `shift_px = fx * psi`,
    with `fx` in pixels of the frame being cropped. The shift is rounded to
    whole pixels and the labels use the realised `psi = shift / fx`, so image
    and labels agree exactly.

    Sampling: per clip, with probability `probability`, a smooth random walk
    (`psi_0 ~ U(-init_max_deg, init_max_deg)`, increments `N(0, step_std_deg)`
    per frame, clipped to the crop margin). A per-frame independent offset
    would make the causal window see a jittering camera, which never happens
    on the road.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        crop_width: int,
        fx: float,
        cameras: list[str],
        trajectory_target: tuple[str, ...] = ("context", "trajectory_target"),
        waypoints: tuple[str, ...] = ("context", "waypoints"),
        probability: float = 0.5,
        init_max_deg: float = 4.0,
        step_std_deg: float = 0.5,
    ) -> None:
        super().__init__()

        self.crop_width = crop_width
        self.fx = fx
        self.cameras = cameras
        self._trajectory_target = trajectory_target
        self._waypoints = waypoints
        self.probability = probability
        self.init_max_deg = init_max_deg
        self.step_std_deg = step_std_deg

    def sample_psi(
        self, b: int, t: int, *, max_rad: float, device: torch.device
    ) -> Tensor:
        """`(b, t)` radians, zero for clips not selected."""
        init = math.radians(self.init_max_deg)
        psi0 = torch.empty(b, 1, device=device).uniform_(-init, init)
        steps = torch.randn(b, t - 1, device=device) * math.radians(self.step_std_deg)
        psi = torch.cat([psi0, psi0 + steps.cumsum(dim=1)], dim=1)
        psi = psi.clamp(-max_rad, max_rad)
        selected = torch.rand(b, 1, device=device) < self.probability
        return psi * selected

    def crop(self, image: Tensor, shift: Tensor) -> Tensor:
        """`image` `(b, t, h, w, c)`, `shift` `(b, t)` integer pixels."""
        b, t, h, w, c = image.shape
        offset = (w - self.crop_width) // 2 + shift
        cols = offset[..., None] + torch.arange(self.crop_width, device=image.device)
        index = cols[:, :, None, :, None].expand(b, t, h, self.crop_width, c)
        return image.gather(3, index)

    @override
    def forward(self, input: PyTree) -> PyTree:
        if not self.training:
            return input

        images = {camera: _get(input, ("image", camera)) for camera in self.cameras}
        reference = next(iter(images.values()))
        b, t, _h, w, _c = reference.shape
        margin = (w - self.crop_width) // 2
        if margin < 0:
            msg = f"crop_width {self.crop_width} wider than the frame ({w})"
            raise ValueError(msg)

        psi = self.sample_psi(b, t, max_rad=margin / self.fx, device=reference.device)
        shift = (psi * self.fx).round().long().clamp(-margin, margin)
        psi = shift.to(torch.float32) / self.fx

        output = input
        for camera, image in images.items():
            output = _assoc(output, ("image", camera), self.crop(image, shift))

        return self.rotate_labels(output, psi)

    def rotate_labels(self, input: PyTree, psi: Tensor) -> PyTree:
        """Rotate `trajectory_target` and `waypoints` by the `(b, t)` yaw `psi`.

        Unperturbed frames keep their labels bit-for-bit (the theta re-wrap is
        not exact at `psi = 0`).
        """
        perturbed = psi.ne(0)[..., None, None]
        output = input

        target = _get(output, self._trajectory_target)
        if target is not None:
            xy = rotate_ego_frame(target[..., :2], psi)
            theta = target[..., 2] - psi[..., None]
            theta = torch.atan2(theta.sin(), theta.cos())
            rotated = torch.cat([xy, theta[..., None]], dim=-1)
            output = _assoc(
                output, self._trajectory_target, torch.where(perturbed, rotated, target)
            )

        waypoints = _get(output, self._waypoints)
        if waypoints is not None:
            rotated = rotate_ego_frame(waypoints.float(), psi).to(waypoints.dtype)
            output = _assoc(
                output, self._waypoints, torch.where(perturbed, rotated, waypoints)
            )

        return output


@final
class ClipPhotometric(Module):
    """Photometric jitter with parameters held constant within a clip.

    Expects `(b, t, c, h, w)` float in `[0, 1]`, i.e. it sits in the image
    `Sequential` between `ToDtype` and `ImageNormalize` (so export, which
    replaces that whole `Sequential`, drops it too). Identity in eval.

    Per clip: brightness (multiplicative), contrast, saturation, and hue
    (fraction of a full turn, torchvision units) always; Gaussian blur, additive
    Gaussian noise and a JPEG round trip each with their own probability.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.02,
        blur_p: float = 0.3,
        blur_sigma: tuple[float, float] = (0.1, 1.0),
        blur_kernel: int = 5,
        noise_p: float = 0.3,
        noise_std: float = 0.02,
        jpeg_p: float = 0.3,
        jpeg_quality: tuple[float, float] = (60.0, 95.0),
    ) -> None:
        super().__init__()

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.blur_p = blur_p
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel
        self.noise_p = noise_p
        self.noise_std = noise_std
        self.jpeg_p = jpeg_p
        self.jpeg_quality = jpeg_quality

    @staticmethod
    def _per_clip(values: Tensor, t: int) -> Tensor:
        """`(b,)` -> `(b * t,)`, matching `x.flatten(0, 1)`."""
        return values.repeat_interleave(t)

    @staticmethod
    def _uniform(b: int, low: float, high: float, device: torch.device) -> Tensor:
        return torch.empty(b, device=device).uniform_(low, high)

    @override
    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input

        # kornia's blur/JPEG convolutions would come back bf16 under the
        # trainer's bf16-mixed autocast; keep the pixel math in fp32
        with torch.autocast(device_type=input.device.type, enabled=False):
            return self.jitter(input.float()).to(input.dtype)

    def jitter(self, input: Tensor) -> Tensor:
        """Train-mode transform; `input` `(b, t, c, h, w)` fp32 in `[0, 1]`."""
        b, t = input.shape[:2]
        device = input.device
        x = input.flatten(0, 1)  # (b * t, c, h, w)

        def factor(spread: float) -> Tensor:
            return self._per_clip(
                self._uniform(b, 1 - spread, 1 + spread, device), t
            ).view(-1, 1, 1, 1)

        x *= factor(self.brightness)
        gray = kornia.color.rgb_to_grayscale(x)
        mean = gray.mean(dim=(-3, -2, -1), keepdim=True)
        x = (x - mean) * factor(self.contrast) + mean
        gray = kornia.color.rgb_to_grayscale(x)
        x = (x - gray) * factor(self.saturation) + gray
        x = x.clamp(0, 1)
        hue = self._per_clip(self._uniform(b, -self.hue, self.hue, device), t)
        x = kornia.enhance.adjust_hue(x, hue * 2 * math.pi)

        clip = torch.arange(b, device=device).repeat_interleave(t)

        blur = (torch.rand(b, device=device) < self.blur_p)[clip]
        if blur.any():
            sigma = self._per_clip(self._uniform(b, *self.blur_sigma, device), t)[blur]
            x[blur] = kornia.filters.gaussian_blur2d(
                x[blur],
                (self.blur_kernel, self.blur_kernel),
                torch.stack([sigma, sigma], dim=-1),
            )

        noise = (torch.rand(b, device=device) < self.noise_p)[clip]
        if noise.any():
            std = self._per_clip(self._uniform(b, 0, self.noise_std, device), t)
            x += torch.randn_like(x) * (std * noise).view(-1, 1, 1, 1)

        x = x.clamp(0, 1)

        jpeg = (torch.rand(b, device=device) < self.jpeg_p)[clip]
        if jpeg.any():
            quality = self._per_clip(self._uniform(b, *self.jpeg_quality, device), t)
            x[jpeg] = kornia.enhance.jpeg_codec_differentiable(
                x[jpeg], quality[jpeg]
            ).clamp(0, 1)

        return x.unflatten(0, (b, t))


@final
class FixedFrameMask(Module):
    """Blank the pixels that would give `YawShiftCrop`'s offset away: the car
    hood band and the black fisheye arcs. Both are fixed to the camera, so a
    shifted crop moves them by exactly the shift; left visible, the model can
    read psi off them instead of off the scene, and in closed loop they are
    always centred.

    Applied in train AND eval, after the image `ModuleDict` (images are
    `(b, t, c, h, w)`, ImageNet-normalised, so 0 fills with the mean colour).
    It stays in the export graph, which only swaps the `ModuleDict`'s image
    entry, so serving needs no change for it.

    The mask is built in raw-frame pixels (`frame_size`), then taken into crop
    coordinates as the union over every shift in `[-max_shift, max_shift]`, so
    it is identical for every psi, and finally any-pooled to the output `size`:
      - hood: rows `>= hood_row`, full width (covers the hood edge of every
        vehicle in the fleet, rows ~258-300 of 324, and the bottom arc);
      - top arc: rows `< arc_depth * (1 - ((x - w / 2) / arc_radius) ** 2)`.
    """

    keep: Tensor

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        cameras: list[str],
        size: tuple[int, int],
        frame_size: tuple[int, int] = (324, 576),
        crop_size: tuple[int, int] = (320, 512),
        max_shift: int = 32,
        hood_row: int = 255,
        arc_depth: float = 35.0,
        arc_radius: float = 240.0,
    ) -> None:
        super().__init__()

        self.cameras = cameras
        frame_h, frame_w = frame_size
        crop_h, crop_w = crop_size
        rows = torch.arange(frame_h, dtype=torch.float32)[:, None]
        cols = torch.arange(frame_w, dtype=torch.float32)[None, :]
        arc = arc_depth * (1 - ((cols - frame_w / 2) / arc_radius) ** 2)
        frame = (rows >= hood_row) | (rows < arc)  # (frame_h, frame_w)

        top, left = (frame_h - crop_h) // 2, (frame_w - crop_w) // 2
        crop = torch.zeros(crop_h, crop_w, dtype=torch.bool)
        for shift in range(-max_shift, max_shift + 1):
            x0 = left + shift
            crop |= frame[top : top + crop_h, x0 : x0 + crop_w]

        pooled = F.interpolate(crop[None, None].float(), size=size, mode="area")
        # one pixel of slack for Resize's antialiased (wider than area) kernel
        masked = F.max_pool2d((pooled > 0).float(), 3, stride=1, padding=1)
        self.register_buffer(
            "keep", 1 - masked[None], persistent=False
        )  # (1, 1, 1, h, w)

    @override
    def forward(self, input: PyTree) -> PyTree:
        output = input
        for camera in self.cameras:
            image = _get(output, ("image", camera))
            if image is not None:
                output = _assoc(
                    output, ("image", camera), image * self.keep.to(image.dtype)
                )
        return output
