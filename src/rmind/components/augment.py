from typing import final, override

import torch
from kornia.geometry.transform import warp_perspective
from pydantic import validate_call
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import PyTree


def _get_in(tree: PyTree, path: tuple[str, ...]) -> Tensor | None:
    node = tree
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set_in(tree: PyTree, path: tuple[str, ...], value: Tensor) -> PyTree:
    key, *rest = path
    node = dict(tree)
    node[key] = value if not rest else _set_in(node.get(key, {}), tuple(rest), value)
    return node


def _yaw_homographies(
    dyaw_deg: Tensor, *, fx: float, fy: float, cx: float, cy: float
) -> Tensor:
    """Per-sample 3x3 homographies mapping source pixel coords to the pixel
    coords a camera additionally yawed by `dyaw_deg` would see.

    `dyaw_deg` follows the same compass convention (positive = clockwise, i.e.
    a rightward turn) as `headings_denoised/heading` -- see
    `HistoryViewpointPerturbation` docstring for the derivation. `dyaw_deg ==
    0` gives the identity homography. `fx`/`fy` may differ (anisotropic
    resize, e.g. a non-square crop resized to a square) -- rotation direction
    is unaffected, only the apparent magnitude per degree along each axis.

    Args:
        dyaw_deg: (n,) yaw offsets in degrees.
        fx: horizontal focal length, pixels.
        fy: vertical focal length, pixels.
        cx: principal point x, pixels.
        cy: principal point y, pixels.

    Returns:
        (n, 3, 3) homographies.
    """
    device, dtype = dyaw_deg.device, dyaw_deg.dtype
    k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    k_inv = torch.inverse(k)

    # Camera yaws opposite the compass sign: a rightward (positive) turn
    # brings what used to be to the camera's right into frame, i.e. the
    # camera's own optical axis rotates toward its left in its old frame.
    phi = -torch.deg2rad(dyaw_deg)
    cos_p, sin_p, zero, one = phi.cos(), phi.sin(), torch.zeros_like(phi), torch.ones_like(phi)
    r = torch.stack(
        [
            torch.stack([cos_p, zero, sin_p], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-sin_p, zero, cos_p], dim=-1),
        ],
        dim=-2,
    )  # (n, 3, 3)

    return k @ r @ k_inv


@final
class HistoryViewpointPerturbation(Module):
    """Training-time augmentation simulating a momentary heading offset at
    the last history tick, in place of swapping in a genuinely off-axis
    (~70 deg) side camera. That swap has no physically consistent steering
    target -- an instantaneous ~70 deg yaw isn't achievable by any real
    steering input over one frame. A *small* synthetic yaw (a handful of
    degrees) keeps the perturbation in the regime a real steering correction
    could produce, so the geometric trajectory target derived from the
    perturbed heading (see `build_relative_trajectory`,
    `build_local_trajectory`) remains a valid "recover, then resume the
    recorded path" label.

    Deliberately placed AFTER the crop/resize/normalize `ModuleDict` step in
    `input_transform` (not between `Remapper` and it) so that step's index
    stays put -- ~20 `policy_finetune_*.yaml` configs patch
    `input_transform._args_[0].paths...` / `input_transform._args_[1].modules...`
    by fixed index via `hparams_jq`, and inserting here would silently
    misdirect all of them. This means `image_key` is read as an already
    `(b, t, c, h, w)` normalized float tensor, and `reference_width`/
    `reference_height` describe the pre-resize crop's size (the frame the
    stated `fov_degrees` was measured on), not the native decoded frame.

    Only two leaves are touched:
      - `image_key` (default `("image", "cam_front_left")`): the last
        history tick, `[:, pivot_index]`, is warped by a homography
        simulating the camera yawing by the sampled offset.
      - `heading_key` (default `("trajectory", "heading")`): the same tick's
        heading gets the same offset added, so
        `build_relative_trajectory`/`build_local_trajectory` (which read
        `heading_deg[:, history_steps - 1]` as their pivot) rotate the
        downstream trajectory target to match.

    Steering-angle history passes through unmodified -- a heading offset a
    few degrees wide doesn't imply anything determinate about the wheel
    angle at that exact instant (that's the point: history looks ordinary,
    only the image/trajectory say a correction is due). The sampled offset
    is however exposed as a third leaf, `dyaw_key` (default
    `("viewpoint_aug", "dyaw")`, degrees, 0 for non-perturbed samples,
    broadcast across the full `t` window so it satisfies
    `TensorDict(batch_size=[b, t])`, always present whenever this module is
    active) -- `PolicyObjective` reads it to nudge the *future*
    steering-angle GT into agreeing with the trajectory head's implicit
    correction, see its `dyaw_key`/`steering_correction_gain`.

    `pivot_index` must equal `history_steps - 1` of the `PolicyObjective`
    this episode feeds; it is not read from that config to avoid coupling
    `episode_builder` to a specific objective.
    """

    @validate_call
    def __init__(
        self,
        *,
        image_key: tuple[str, ...] = ("image", "cam_front_left"),
        heading_key: tuple[str, ...] = ("trajectory", "heading"),
        dyaw_key: tuple[str, ...] = ("viewpoint_aug", "dyaw"),
        pivot_index: int = 4,
        probability: float = 0.0,
        max_degrees: float = 5.0,
        fov_degrees: float = 110.0,
        reference_width: int = 576,
        reference_height: int = 320,
    ) -> None:
        super().__init__()
        self.image_key = image_key
        self.heading_key = heading_key
        self.dyaw_key = dyaw_key
        self.pivot_index = pivot_index
        self.probability = probability
        self.max_degrees = max_degrees
        self.fov_degrees = fov_degrees
        self.reference_width = reference_width
        self.reference_height = reference_height

    @override
    def forward(self, input: PyTree) -> PyTree:
        image = _get_in(input, self.image_key)
        heading = _get_in(input, self.heading_key)
        if image is None or heading is None:
            return input

        b, t, device = image.shape[0], image.shape[1], image.device
        dyaw_full = torch.zeros(b, device=device, dtype=torch.float32)
        out = input

        if self.training and self.probability > 0:
            selected = torch.rand(b, device=device) < self.probability
            idx = selected.nonzero(as_tuple=True)[0]

            if idx.numel() > 0:
                dyaw = (
                    torch.empty(idx.numel(), device=device, dtype=torch.float32)
                    .uniform_(-self.max_degrees, self.max_degrees)
                )
                dyaw_full[idx] = dyaw

                pivot = image[:, self.pivot_index]  # (b, c, h, w)
                h, w = pivot.shape[-2], pivot.shape[-1]
                chosen = pivot[idx]  # (k, c, h, w)

                fov_rad = torch.deg2rad(
                    torch.as_tensor(self.fov_degrees, device=device, dtype=dyaw.dtype)
                )
                f_ref = (self.reference_width / 2) / torch.tan(fov_rad / 2)
                fx = f_ref * (w / self.reference_width)
                fy = f_ref * (h / self.reference_height)

                homographies = _yaw_homographies(
                    dyaw, fx=fx.item(), fy=fy.item(), cx=w / 2, cy=h / 2
                )
                warped = warp_perspective(
                    chosen,
                    homographies,
                    dsize=(h, w),
                    mode="bilinear",
                    # "border" (not "zeros"): a black wedge at the frame edge
                    # would be an easy, ungeneralizable tell that a sample
                    # was perturbed.
                    padding_mode="border",
                ).to(chosen.dtype)

                new_pivot = pivot.clone()
                new_pivot[idx] = warped
                new_image = image.clone()
                new_image[:, self.pivot_index] = new_pivot

                new_heading = heading.clone()
                dyaw_ = dyaw.to(heading.dtype).reshape(
                    (idx.numel(),) + (1,) * (heading.dim() - 2)
                )
                new_heading[idx, self.pivot_index] = (
                    new_heading[idx, self.pivot_index] + dyaw_
                )

                out = _set_in(out, self.image_key, new_image)
                out = _set_in(out, self.heading_key, new_heading)

        # Broadcast across `t` (this is a per-episode, not per-tick,
        # quantity) so it satisfies `TensorDict(batch_size=[b, t])`; always
        # attached (even when idx is empty this batch, or probability == 0)
        # so its presence in `episode.input` is a static property of config,
        # not a stochastic one.
        dyaw_leaf = dyaw_full.to(heading.dtype).reshape(b, 1, 1).expand(b, t, 1)
        return _set_in(out, self.dyaw_key, dyaw_leaf)
