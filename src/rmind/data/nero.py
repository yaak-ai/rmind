"""nero-arms data primitives: SE(3) pose representations, standardisation, layout.

This module is the *shared boundary* named by the nero-arms data contract §5:
`pose_quat_to_9d` / `pose_9d_to_quat` are referenced by the rbyte ingestion side
and used identically by the policy and the action tokenizer here. Nothing in
this file may become model-specific.

Representation rules (contract §5)
----------------------------------
* storage / dataframe form: canonicalised quaternion, **7 floats** per pose
  `(x, y, z, qx, qy, qz, qw)` with `qw >= 0`;
* model-facing form: **9 floats** per pose, `3` translation + the **6D continuous
  rotation** of Zhou et al. (the first two *columns* of `R`). Quaternions are
  discontinuous as a regression/reconstruction target; 6D is not.

Channel layout of the 60-dim per-side vector (contract §6.1)
------------------------------------------------------------
The contract specifies the *blocks* but not their order within the 60 dims, so
this module **defines** it, and `POSE_BLOCK_LAYOUT` is the single source of truth
that both sides must agree on:

    [ arm_world      : 9  ]  (t3 + r6)   indices  0..8
    [ thumb_rel_hub  : 9  ]              indices  9..17
    [ index_rel_hub  : 9  ]              indices 18..26
    [ middle_rel_hub : 9  ]              indices 27..35
    [ ring_rel_hub   : 9  ]              indices 36..44
    [ little_rel_hub : 9  ]              indices 45..53
    [ hub_orientation: 6  ]  (r6 only)   indices 54..59
                       ---
                        60

⚠️ CONTRACT GAP: §5 names only the 7<->9 pose pair, but the hub-orientation block
is rotation-only, so a 4<->6 pair is also needed. It is provided here as
`quat_to_rot6d` / `rot6d_to_quat`; the contract should name it.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self, final, override

import torch
from torch import Tensor, nn

__all__ = [
    "BIMANUAL_DIM",
    "NUM_SIDES",
    "POSE_9D_DIM",
    "POSE_BLOCK_LAYOUT",
    "POSE_QUAT_DIM",
    "ROTATION_INDICES",
    "SIDE_DIM",
    "STATE_QUAT_DIM",
    "TRANSLATION_INDICES",
    "PoseStandardizer",
    "canonicalize_quat",
    "geodesic_angle_error",
    "letterbox_camera_cond",
    "pose_9d_to_quat",
    "pose_quat_to_9d",
    "quat_to_rot6d",
    "quat_to_rotmat",
    "rot6d_to_quat",
    "rot6d_to_rotmat",
    "rotmat_to_quat",
    "rotmat_to_rot6d",
    "state_9d_to_quat",
    "state_quat_to_9d",
    "translation_rotation_split",
]


@final
@dataclass(frozen=True)
class PoseBlock:
    name: str
    offset: int
    has_translation: bool

    @property
    def width(self) -> int:
        return 9 if self.has_translation else 6

    @property
    def translation_slice(self) -> slice:
        if not self.has_translation:
            msg = f"block {self.name} has no translation"
            raise ValueError(msg)
        return slice(self.offset, self.offset + 3)

    @property
    def rotation_slice(self) -> slice:
        start = self.offset + (3 if self.has_translation else 0)
        return slice(start, start + 6)


def _layout() -> tuple[PoseBlock, ...]:
    names = (
        "arm_world",
        "thumb_rel_hub",
        "index_rel_hub",
        "middle_rel_hub",
        "ring_rel_hub",
        "little_rel_hub",
    )
    blocks: list[PoseBlock] = []
    offset = 0
    for name in names:
        blocks.append(PoseBlock(name=name, offset=offset, has_translation=True))
        offset += 9
    blocks.append(
        PoseBlock(name="hub_orientation", offset=offset, has_translation=False)
    )
    return tuple(blocks)


POSE_BLOCK_LAYOUT: Final[tuple[PoseBlock, ...]] = _layout()
SIDE_DIM: Final[int] = sum(block.width for block in POSE_BLOCK_LAYOUT)  # 60
NUM_SIDES: Final[int] = 2
#: storage form (contract §5.2) and model-facing form (§5.3) widths, per pose
POSE_QUAT_DIM: Final[int] = 7
POSE_9D_DIM: Final[int] = 9
#: per-side width of the STORAGE form: 6 poses x 7 + a 4-dim hub quaternion.
#: rbyte emits this (contract §5.2); the model-facing 60 is produced here.
STATE_QUAT_DIM: Final[int] = 6 * POSE_QUAT_DIM + 4  # 46
BIMANUAL_DIM: Final[int] = NUM_SIDES * SIDE_DIM  # 120

TRANSLATION_INDICES: Final[tuple[int, ...]] = tuple(
    i
    for block in POSE_BLOCK_LAYOUT
    if block.has_translation
    for i in range(block.translation_slice.start, block.translation_slice.stop)
)  # 18 dims
ROTATION_INDICES: Final[tuple[int, ...]] = tuple(
    i
    for block in POSE_BLOCK_LAYOUT
    for i in range(block.rotation_slice.start, block.rotation_slice.stop)
)  # 42 dims

# ------------------------------------------------------------------ rotations


def canonicalize_quat(q: Tensor) -> Tensor:
    """Remove the quaternion double cover: flip so `qw >= 0`.

    `q` is `(..., 4)` ordered `(qx, qy, qz, qw)` -- the contract's storage order.
    13-21% of the dummy's arm samples have `qw < 0`; without this every
    downstream regressor/quantiser wastes capacity on the sign flip (§5).
    """
    sign = torch.where(
        q[..., 3:4] < 0, -torch.ones_like(q[..., 3:4]), torch.ones_like(q[..., 3:4])
    )
    return q * sign


def quat_to_rotmat(q: Tensor) -> Tensor:
    """`(..., 4)` `(qx, qy, qz, qw)` -> `(..., 3, 3)` rotation matrix."""
    # NOTE: a NEW name, never `q /= ...`. `q` is frequently a view into a caller's
    # pose tensor and an in-place normalisation would silently mutate it.
    unit = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = unit.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).unflatten(-1, (3, 3))


def rotmat_to_quat(r: Tensor) -> Tensor:
    """`(..., 3, 3)` -> canonical `(..., 4)` `(qx, qy, qz, qw)` with `qw >= 0`.

    Shepperd's method: pick the largest of the four candidate denominators, so
    the result is numerically stable for every rotation including `w ~ 0`.
    """
    m = r.reshape(*r.shape[:-2], 9).unbind(-1)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m

    # the four candidates, each stable near its own maximum
    candidates = torch.stack(
        [
            torch.stack([m21 - m12, m02 - m20, m10 - m01, 1 + m00 + m11 + m22], dim=-1),
            torch.stack([1 + m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12], dim=-1),
            torch.stack([m01 + m10, 1 - m00 + m11 - m22, m12 + m21, m02 - m20], dim=-1),
            torch.stack([m02 + m20, m12 + m21, 1 - m00 - m11 + m22, m10 - m01], dim=-1),
        ],
        dim=-2,
    )  # (..., 4, 4)
    trace = 1 + m00 + m11 + m22
    magnitudes = torch.stack(
        [trace, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1
    )
    best = magnitudes.argmax(dim=-1, keepdim=True)[..., None].expand(
        *magnitudes.shape[:-1], 1, 4
    )
    picked = candidates.gather(-2, best).squeeze(-2)
    return canonicalize_quat(
        picked / picked.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    )


def rotmat_to_rot6d(r: Tensor) -> Tensor:
    """`(..., 3, 3)` -> `(..., 6)`: the first two COLUMNS of `R` (Zhou et al.)."""
    return r[..., :, :2].transpose(-2, -1).reshape(*r.shape[:-2], 6)


def rot6d_to_rotmat(x: Tensor) -> Tensor:
    """`(..., 6)` -> `(..., 3, 3)` via Gram-Schmidt.

    This is a PROJECTION, not an inverse: an arbitrary 6-vector is mapped onto
    the nearest (in the Gram-Schmidt sense) element of SO(3). `rotmat_to_rot6d`
    followed by this is the identity; the reverse composition is not.
    """
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # NOTE: a NEW name. `a2` is a VIEW into the caller's `x`; `a2 -= ...` would
    # silently corrupt the caller's tensor (and this function is called on
    # prediction/target slices inside the error metrics).
    orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = orthogonal / orthogonal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def quat_to_rot6d(q: Tensor) -> Tensor:
    """Rotation-only 4 -> 6. Needed by the hub-orientation block (contract gap)."""
    return rotmat_to_rot6d(quat_to_rotmat(canonicalize_quat(q)))


def rot6d_to_quat(x: Tensor) -> Tensor:
    """Rotation-only 6 -> 4, canonical (`qw >= 0`)."""
    return rotmat_to_quat(rot6d_to_rotmat(x))


# ---------------------------------------------------------------------- poses


def pose_quat_to_9d(pose: Tensor) -> Tensor:
    """Contract §5: `(..., 7)` `(x,y,z,qx,qy,qz,qw)` -> `(..., 9)` `(t3, rot6d)`.

    Canonicalises the quaternion on the way through, so this is safe to call on
    raw storage-form data that has not been canonicalised at ingestion.

    Raises:
        ValueError: if the last dimension is not `POSE_QUAT_DIM`.
    """
    if pose.shape[-1] != POSE_QUAT_DIM:
        msg = f"expected a 7-dim pose, got {pose.shape[-1]}"
        raise ValueError(msg)
    return torch.cat([pose[..., :3], quat_to_rot6d(pose[..., 3:])], dim=-1)


def pose_9d_to_quat(pose: Tensor) -> Tensor:
    """Contract §5: `(..., 9)` `(t3, rot6d)` -> `(..., 7)` with a canonical quaternion.

    Raises:
        ValueError: if the last dimension is not `POSE_9D_DIM`.
    """
    if pose.shape[-1] != POSE_9D_DIM:
        msg = f"expected a 9-dim pose, got {pose.shape[-1]}"
        raise ValueError(msg)
    return torch.cat([pose[..., :3], rot6d_to_quat(pose[..., 3:])], dim=-1)


def state_quat_to_9d(state: Tensor) -> Tensor:
    """Per-side STORAGE form `(..., 46)` -> model-facing `(..., 60)`.

    ⚠️ CONTRACT INCONSISTENCY (resolved in rbyte's favour): §5.2 specifies
    canonical-quaternion storage (7 floats per pose) while §8 tabulated
    `state.pose` as `(T, 2, 60)`. rbyte implements §5.2, so the loader emits
    **46** per side -- 6 poses x 7 plus the hub's 4-dim orientation quaternion --
    and the 9D expansion happens **here**, at the model boundary. This is the
    mirror of `rbyte.io.nero.state_quat_to_9d`; the two must stay in step.

    Raises:
        ValueError: if the last dimension is not `STATE_QUAT_DIM`.
    """
    if state.shape[-1] != STATE_QUAT_DIM:
        msg = f"expected a {STATE_QUAT_DIM}-dim per-side state, got {state.shape[-1]}"
        raise ValueError(msg)
    poses = state[..., : 6 * POSE_QUAT_DIM].unflatten(-1, (6, POSE_QUAT_DIM))
    hub = state[..., 6 * POSE_QUAT_DIM :]
    return torch.cat(
        [pose_quat_to_9d(poses).flatten(-2, -1), quat_to_rot6d(hub)], dim=-1
    )


def state_9d_to_quat(state: Tensor) -> Tensor:
    """Inverse of `state_quat_to_9d`: `(..., 60)` -> `(..., 46)`.

    Raises:
        ValueError: if the last dimension is not `SIDE_DIM`.
    """
    if state.shape[-1] != SIDE_DIM:
        msg = f"expected a {SIDE_DIM}-dim per-side state, got {state.shape[-1]}"
        raise ValueError(msg)
    poses = state[..., : 6 * POSE_9D_DIM].unflatten(-1, (6, POSE_9D_DIM))
    hub = state[..., 6 * POSE_9D_DIM :]
    return torch.cat(
        [pose_9d_to_quat(poses).flatten(-2, -1), rot6d_to_quat(hub)], dim=-1
    )


def translation_rotation_split(x: Tensor) -> tuple[Tensor, Tensor]:
    """Split a `(..., 60)` per-side vector into its translation and rotation channels.

    Raises:
        ValueError: if the last dimension is not `SIDE_DIM`.

    Returns `(translation (..., 18), rotation (..., 42))`. This is what makes the
    contract's "report translation and rotation error SEPARATELY" (§5.5)
    mechanically possible.
    """
    if x.shape[-1] != SIDE_DIM:
        msg = f"expected a {SIDE_DIM}-dim per-side vector, got {x.shape[-1]}"
        raise ValueError(msg)
    index = torch.as_tensor(TRANSLATION_INDICES, device=x.device)
    rot_index = torch.as_tensor(ROTATION_INDICES, device=x.device)
    return x.index_select(-1, index), x.index_select(-1, rot_index)


def pose_error_metrics(pred: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Physical-unit reconstruction error for `(..., 60)` UNSTANDARDISED pose vectors.

    Contract §5.5 demands translation and rotation error be reported separately.
    A standardised scalar L1 is not interpretable, so this reports:

    * `translation_mm` -- mean Euclidean distance per translation block, in mm;
    * `rotation_deg`   -- mean geodesic angle per rotation block, in degrees.

    Both are averaged over blocks and over the leading batch dimensions.
    """
    translation: list[Tensor] = []
    rotation: list[Tensor] = []
    for block in POSE_BLOCK_LAYOUT:
        if block.has_translation:
            delta = (
                pred[..., block.translation_slice]
                - target[..., block.translation_slice]
            )
            translation.append(delta.norm(dim=-1))
        rotation.append(
            geodesic_angle_error(
                pred[..., block.rotation_slice], target[..., block.rotation_slice]
            )
        )
    return {
        "translation_mm": torch.stack(translation, dim=-1).mean() * 1000.0,
        "rotation_deg": torch.stack(rotation, dim=-1).mean(),
    }


def geodesic_angle_error(a: Tensor, b: Tensor) -> Tensor:
    """Per-rotation geodesic angle in DEGREES between two `(..., 6)` 6D rotations.

    Both sides are projected onto SO(3) first, so this is well-defined even for a
    reconstruction that has drifted off the manifold.
    """
    ra, rb = rot6d_to_rotmat(a), rot6d_to_rotmat(b)
    trace = (ra.transpose(-2, -1) @ rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1.0, 1.0)))


# ------------------------------------------------------------- standardisation


@final
class PoseStandardizer(nn.Module):
    """Per-channel standardisation over the 60-dim per-side pose vector (§5.4).

    Translations are ~0.06-0.7 m while 6D rotation components are ~1, so an
    unstandardised L2/L1 reconstruction loss lets rotation error swamp
    translation error. Statistics are computed over the **training split only**
    and shipped as a versioned artifact alongside the tokenizer checkpoint
    (`save` / `load`).

    The buffers are registered (not parameters), so they travel inside the
    checkpoint as well -- the JSON artifact is the human-auditable copy and the
    provenance record.
    """

    VERSION: Final[int] = 1

    # declared so the type checker sees the registered buffers as tensors
    mean: Tensor
    std: Tensor

    def __init__(
        self,
        *,
        mean: Tensor | Sequence[float] | None = None,
        std: Tensor | Sequence[float] | None = None,
        dim: int = SIDE_DIM,
        eps: float = 1e-6,
        source: str = "identity",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.source = source
        self.register_buffer(
            "mean",
            torch.zeros(dim)
            if mean is None
            else torch.as_tensor(mean, dtype=torch.float32),
        )
        self.register_buffer(
            "std",
            torch.ones(dim)
            if std is None
            else torch.as_tensor(std, dtype=torch.float32),
        )

    @classmethod
    def from_samples(
        cls, samples: Tensor, *, source: str = "train-split", **kwargs: object
    ) -> Self:
        """Fit from `(..., dim)` training-split samples (masked rows removed by the caller)."""
        flat = samples.reshape(-1, samples.shape[-1]).float()
        return cls(
            mean=flat.mean(0),
            std=flat.std(0).clamp_min(1e-6),
            dim=samples.shape[-1],
            source=source,
            **kwargs,  # ty:ignore[invalid-argument-type]
        )

    def standardize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def unstandardize(self, x: Tensor) -> Tensor:
        return x * (self.std + self.eps) + self.mean

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.standardize(x)

    # -------------------------------------------------------------- artifact

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.VERSION,
            "dim": self.dim,
            "eps": self.eps,
            "source": self.source,
            "layout": [
                {"name": b.name, "offset": b.offset, "width": b.width}
                for b in POSE_BLOCK_LAYOUT
            ],
            "translation_indices": list(TRANSLATION_INDICES),
            "rotation_indices": list(ROTATION_INDICES),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload["version"]
        if version != cls.VERSION:
            msg = f"pose standardisation artifact version {version} != {cls.VERSION}"
            raise ValueError(msg)
        return cls(
            mean=payload["mean"],
            std=payload["std"],
            dim=payload["dim"],
            eps=payload["eps"],
            source=payload["source"],
        )


# ------------------------------------------------------- camera conditioning

#: contract §7.1: `fx/W, fy/H, cx/W, cy/H` (4) + `t_world_cam` (3) + `R_world_cam` 6D (6)
CAMERA_COND_DIM: Final[int] = 13


def letterbox_camera_cond(
    cond: Tensor, *, source_size: Tensor, target_size: tuple[int, int] | int
) -> Tensor:
    """Rewrite the 4 resolution-normalised intrinsics of `camera_cond` for a letterbox.

    Contract §7.2 + the rbyte hand-off: rbyte delivers each camera **isotropically
    downscaled to its own H/W** (`base` 270x480, `side_*` 300x480), which keeps
    the normalised intrinsics exactly valid but leaves the three streams on
    DIFFERENT grids. Unifying them is rmind's job, and it must be a letterbox
    (uniform scale + symmetric padding): an anisotropic resize would scale `fx`
    and `fy` independently and destroy the geometric meaning of this vector.
    Padding does change the normalisation denominator, so the vector is rewritten
    to match -- otherwise the policy is told intrinsics that no longer describe
    the pixels it is looking at.

    With target `(TH, TW)`, `s = min(TW/W, TH/H)`, `rx = sW/TW`, `ry = sH/TH`:

        fx/TW = rx * (fx/W)          cx/TW = rx * (cx/W) + (1 - rx)/2
        fy/TH = ry * (fy/H)          cy/TH = ry * (cy/H) + (1 - ry)/2

    A subsequent ISOTROPIC resize with no padding leaves all four untouched
    (both numerator and denominator scale together), which is why the pipeline
    pads first and resizes second.

    Extrinsics are untouched -- letterboxing does not move the camera.

    Args:
        cond: `(..., n_cameras, 13)`.
        source_size: `(..., n_cameras, 2)` native `(W, H)` in pixels.
        target_size: the target `(height, width)`, or one int for a square.
    """
    height, width = (
        (target_size, target_size) if isinstance(target_size, int) else target_size
    )
    w, h = source_size[..., 0], source_size[..., 1]
    scale = torch.minimum(width / w, height / h)
    rx, ry = scale * w / width, scale * h / height
    fx, fy, cx, cy = cond[..., 0], cond[..., 1], cond[..., 2], cond[..., 3]
    return torch.cat(
        [
            torch.stack(
                [rx * fx, ry * fy, rx * cx + (1 - rx) / 2, ry * cy + (1 - ry) / 2],
                dim=-1,
            ),
            cond[..., 4:],
        ],
        dim=-1,
    )
