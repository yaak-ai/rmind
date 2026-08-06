"""Contract §5 / §7 primitives: rotation representations, standardisation, letterbox.

The round-trip direction matters. `quat -> 9D -> quat` from a CANONICALISED unit
quaternion is an identity and is tested to 1e-6. The reverse from an arbitrary
9-vector is NOT an identity -- Gram-Schmidt is a projection -- so that direction
is tested for the property that actually holds: the result is a valid rotation.
"""

import pytest
import torch

from rmind.components.image import LetterboxResize
from rmind.data.nero import (
    BIMANUAL_DIM,
    POSE_BLOCK_LAYOUT,
    ROTATION_INDICES,
    SIDE_DIM,
    TRANSLATION_INDICES,
    PoseStandardizer,
    canonicalize_quat,
    geodesic_angle_error,
    letterbox_camera_cond,
    pose_9d_to_quat,
    pose_error_metrics,
    pose_quat_to_9d,
    quat_to_rot6d,
    quat_to_rotmat,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_quat,
    translation_rotation_split,
)

TOL = 1e-6


def _random_poses(n: int = 512, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    q = canonicalize_quat(q / q.norm(dim=-1, keepdim=True))
    t = torch.randn(n, 3, generator=generator, dtype=torch.float64) * 0.3
    return torch.cat([t, q], dim=-1)


# ------------------------------------------------------------------- layout


def test_layout_partitions_the_60_dims() -> None:
    assert SIDE_DIM == 60  # noqa: PLR2004
    assert BIMANUAL_DIM == 120  # noqa: PLR2004
    assert len(TRANSLATION_INDICES) == 18  # noqa: PLR2004
    assert len(ROTATION_INDICES) == 42  # noqa: PLR2004
    assert set(TRANSLATION_INDICES).isdisjoint(ROTATION_INDICES)
    assert set(TRANSLATION_INDICES) | set(ROTATION_INDICES) == set(range(SIDE_DIM))
    # exactly one rotation-only block (the hub orientation, contract §6.1)
    assert sum(not b.has_translation for b in POSE_BLOCK_LAYOUT) == 1


def test_translation_rotation_split_selects_the_right_columns() -> None:
    x = torch.arange(SIDE_DIM, dtype=torch.float32)
    t, r = translation_rotation_split(x)
    assert t.tolist() == list(TRANSLATION_INDICES)
    assert r.tolist() == list(ROTATION_INDICES)


# ---------------------------------------------------------------- rotations


def test_pose_quat_9d_round_trip_is_identity() -> None:
    poses = _random_poses()
    assert torch.allclose(pose_9d_to_quat(pose_quat_to_9d(poses)), poses, atol=TOL)


def test_quaternion_double_cover_is_removed() -> None:
    """`q` and `-q` are the same rotation and MUST map to the same 9D vector."""
    poses = _random_poses()
    flipped = poses.clone()
    flipped[:, 3:] *= -1
    assert torch.allclose(pose_quat_to_9d(poses), pose_quat_to_9d(flipped), atol=TOL)
    assert (canonicalize_quat(flipped[:, 3:])[:, 3] >= 0).all()


def test_rot6d_projects_arbitrary_vectors_onto_so3() -> None:
    """The reverse direction is a PROJECTION, so assert the invariant, not identity."""
    generator = torch.Generator().manual_seed(1)
    perturbed = quat_to_rot6d(_random_poses()[:, 3:]) + 0.3 * torch.randn(
        512, 6, generator=generator, dtype=torch.float64
    )
    r = rot6d_to_rotmat(perturbed)
    eye = torch.eye(3, dtype=r.dtype).expand_as(r)
    assert torch.allclose(r.transpose(-2, -1) @ r, eye, atol=1e-10)
    assert torch.allclose(
        torch.linalg.det(r), torch.ones(512, dtype=r.dtype), atol=1e-10
    )


def test_rotmat_quat_round_trip_including_near_zero_w() -> None:
    """Shepperd's method must stay stable for 180-degree rotations (`qw ~ 0`)."""
    axis = torch.tensor(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64
    )
    axis /= axis.norm(dim=-1, keepdim=True)
    q = torch.cat([axis, torch.zeros(4, 1, dtype=torch.float64)], dim=-1)  # 180 deg
    assert torch.allclose(rotmat_to_quat(quat_to_rotmat(q)).abs(), q.abs(), atol=1e-8)


def test_rot6d_quat_round_trip_for_the_hub_block() -> None:
    """Contract GAP: §5 names only the 7<->9 pair; the hub block needs 4<->6."""
    q = _random_poses()[:, 3:]
    assert torch.allclose(rot6d_to_quat(quat_to_rot6d(q)), q, atol=TOL)


def test_geodesic_angle_error_matches_a_known_rotation() -> None:
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    angle = torch.tensor(0.5, dtype=torch.float64)  # 0.5 rad about z
    rotated = torch.stack([
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.sin(angle / 2),
        torch.cos(angle / 2),
    ])
    error = geodesic_angle_error(quat_to_rot6d(identity), quat_to_rot6d(rotated))
    assert error.item() == pytest.approx(torch.rad2deg(angle).item(), abs=1e-6)


# --------------------------------------------------------------- error split


def test_pose_error_metrics_separates_translation_from_rotation() -> None:
    """The §5.5 requirement, made falsifiable: a translation-only error must not
    show up as rotation error, and vice versa.
    """
    generator = torch.Generator().manual_seed(2)
    target = torch.zeros(64, SIDE_DIM, dtype=torch.float64)
    for block in POSE_BLOCK_LAYOUT:
        target[..., block.rotation_slice] = quat_to_rot6d(
            canonicalize_quat(
                torch.nn.functional.normalize(
                    torch.randn(64, 4, generator=generator, dtype=torch.float64), dim=-1
                )
            )
        )
        if block.has_translation:
            target[..., block.translation_slice] = torch.randn(
                64, 3, generator=generator, dtype=torch.float64
            )

    translated = target.clone()
    translated[..., list(TRANSLATION_INDICES)] += 0.01  # 10 mm per axis
    metrics = pose_error_metrics(translated, target)
    assert metrics["translation_mm"].item() == pytest.approx(10 * 3**0.5, abs=1e-3)
    assert metrics["rotation_deg"].item() == pytest.approx(0.0, abs=1e-6)

    rotated = target.clone()
    for block in POSE_BLOCK_LAYOUT:
        angle = torch.tensor(0.2, dtype=torch.float64)
        delta = torch.stack([
            torch.zeros((), dtype=torch.float64),
            torch.zeros((), dtype=torch.float64),
            torch.sin(angle / 2),
            torch.cos(angle / 2),
        ])
        r = quat_to_rotmat(delta) @ rot6d_to_rotmat(target[..., block.rotation_slice])
        rotated[..., block.rotation_slice] = (
            r[..., :, :2].transpose(-2, -1).reshape(64, 6)
        )
    metrics = pose_error_metrics(rotated, target)
    assert metrics["translation_mm"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["rotation_deg"].item() == pytest.approx(
        torch.rad2deg(torch.tensor(0.2)).item(), abs=1e-4
    )


# ------------------------------------------------------------ standardisation


def test_standardizer_round_trip_and_channel_scaling() -> None:
    generator = torch.Generator().manual_seed(3)
    samples = torch.randn(4096, SIDE_DIM, generator=generator)
    samples[:, list(TRANSLATION_INDICES)] *= 0.05  # metres-scale translations
    standardizer = PoseStandardizer.from_samples(samples)

    standardized = standardizer(samples)
    assert torch.allclose(standardizer.unstandardize(standardized), samples, atol=1e-4)
    # the point of §5.4: after standardisation the two channel groups have
    # comparable scale, so an L1 objective weights them comparably
    t, r = translation_rotation_split(standardized)
    assert t.std().item() == pytest.approx(r.std().item(), rel=0.05)


def test_standardizer_artifact_round_trip_and_version_gate(tmp_path) -> None:  # noqa: ANN001
    generator = torch.Generator().manual_seed(4)
    standardizer = PoseStandardizer.from_samples(
        torch.randn(256, SIDE_DIM, generator=generator), source="unit-test"
    )
    path = standardizer.save(tmp_path / "stats.json")
    loaded = PoseStandardizer.load(path)
    assert torch.allclose(loaded.mean, standardizer.mean)
    assert torch.allclose(loaded.std, standardizer.std)
    assert loaded.source == "unit-test"

    import json  # noqa: PLC0415

    payload = json.loads(path.read_text())
    payload["version"] = 999
    _ = path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        _ = PoseStandardizer.load(path)


# ------------------------------------------------------- camera conditioning


def test_letterbox_resize_preserves_aspect_ratio() -> None:
    """Contract §7.2: base is 16:9 and side_* are 16:10 -- an anisotropic resize
    would make the same object a different shape in different cameras.
    """
    transform = LetterboxResize(size=224)
    for w, h in ((1920, 1080), (1280, 800)):
        x = torch.zeros(2, 3, 3, h, w)
        x[..., h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
        out = transform(x)
        assert out.shape == (2, 3, 3, 224, 224)
        # the padded rows/cols are exactly the letterbox bars
        scale = min(224 / w, 224 / h)
        pad = 224 - max(1, round(h * scale))
        if pad:
            assert out[..., : pad // 2, :].abs().max().item() == pytest.approx(0.0)


def test_letterbox_camera_cond_rewrites_intrinsics_consistently() -> None:
    """A point at the image centre must stay at the letterboxed image centre."""
    cond = torch.zeros(1, 2, 13)
    cond[..., 0], cond[..., 1] = 0.5, 0.9  # fx/W, fy/H
    cond[..., 2], cond[..., 3] = 0.5, 0.5  # principal point at the centre
    cond[..., 4:] = torch.arange(9, dtype=torch.float32)  # extrinsics
    size = torch.tensor([[[1920.0, 1080.0], [1280.0, 800.0]]])

    out = letterbox_camera_cond(cond, source_size=size, target_size=224)
    assert torch.allclose(out[..., 2], torch.full((1, 2), 0.5))  # centre stays centred
    assert torch.allclose(out[..., 3], torch.full((1, 2), 0.5))
    assert torch.allclose(out[..., 4:], cond[..., 4:])  # extrinsics untouched
    # width is the limiting dimension for both cameras -> fx/S is unchanged
    assert torch.allclose(out[..., 0], cond[..., 0])
    # ... and fy/S shrinks by the letterbox ratio
    assert out[0, 0, 1].item() == pytest.approx(0.9 * (1080 / 1920), abs=1e-6)
    assert out[0, 1, 1].item() == pytest.approx(0.9 * (800 / 1280), abs=1e-6)
