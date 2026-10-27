import dataclasses

import numpy as np
import pytest
import torch

from rmind.pallet.geometry import (
    CAM_FORK,
    FLIP_INDEX,
    FORK_LENGTH,
    KEYPOINT_NAMES,
    KEYPOINTS_3D,
    FisheyeCamera,
    Pose2D,
)
from rmind.pallet.keypoints import (
    Crop,
    PalletKeypointNet,
    decode,
    gaussian_targets,
    heatmap_loss,
)
from rmind.pallet.labels import (
    facing_truck_mount,
    keypoint_labels,
    pallet_pose_6dof,
    tag_corners_robot,
    tag_ground_truth,
)
from rmind.pallet.pose import closed_form, solve
from rmind.pallet.silhouette import fit, silhouette_polygon

# a Kannala-Brandt camera with real distortion, next to the rational `CAM_FORK`
DISTORTED = FisheyeCamera(
    fx=850.0,
    fy=850.0,
    cx=960.0,
    cy=540.0,
    width=1920,
    height=1080,
    k=(0.05, -0.01, 0.002, 0.0),
    position=(0.35, 0.0, 0.43),
)
APPROACH = (
    Pose2D(FORK_LENGTH + 2.5, 0.2, np.deg2rad(8)),
    Pose2D(FORK_LENGTH + 1.0, -0.1, np.deg2rad(-4)),
    Pose2D(FORK_LENGTH + 0.2, 0.03, np.deg2rad(1)),
)


def _assert_pose(
    actual: Pose2D, expected: Pose2D, m: float = 1e-3, deg: float = 0.05
) -> None:
    assert actual.x == pytest.approx(expected.x, abs=m)
    assert actual.y == pytest.approx(expected.y, abs=m)
    assert np.rad2deg(actual.yaw) == pytest.approx(np.rad2deg(expected.yaw), abs=deg)


@pytest.mark.parametrize("camera", [CAM_FORK, DISTORTED])
def test_rays_invert_projection(camera: FisheyeCamera) -> None:
    # inside the rational model's valid field (the extreme corners are not)
    pixels = np.array([[150.0, 900.0], [960.0, 540.0], [1750.0, 200.0], [300.0, 800.0]])
    projected, visible = camera.project(camera.origin + 2.5 * camera.rays(pixels))
    assert visible.all()
    np.testing.assert_allclose(projected, pixels, atol=1e-6)


def test_level_camera_puts_the_horizon_on_the_centre_row() -> None:
    far = np.array([[100.0, 0.0, CAM_FORK.position[2]]])
    pixels, _ = CAM_FORK.project(far)
    np.testing.assert_allclose(pixels[0], [CAM_FORK.cx, CAM_FORK.cy], atol=1e-6)
    # +y is left in the robot frame, so a point to the left lands left of centre
    left, _ = CAM_FORK.project(np.array([[3.0, 1.0, 0.0]]))
    assert left[0, 0] < CAM_FORK.cx


def test_flip_index_mirrors_the_keypoints() -> None:
    mirrored = KEYPOINTS_3D * np.array([1.0, -1.0, 1.0])
    np.testing.assert_allclose(KEYPOINTS_3D[list(FLIP_INDEX)], mirrored)
    assert len(KEYPOINT_NAMES) == len(KEYPOINTS_3D)


@pytest.mark.parametrize("truth", APPROACH)
def test_closed_form_is_exact_without_noise(truth: Pose2D) -> None:
    pixels, visible = DISTORTED.project(truth.apply(KEYPOINTS_3D))
    pose = closed_form(DISTORTED, pixels[visible], KEYPOINTS_3D[visible])
    assert pose is not None
    _assert_pose(pose, truth, 1e-9, 1e-7)


@pytest.mark.parametrize("truth", APPROACH)
def test_solve_rejects_a_bad_keypoint(truth: Pose2D) -> None:
    rng = np.random.default_rng(0)
    pixels, visible = DISTORTED.project(truth.apply(KEYPOINTS_3D))
    noisy = pixels + rng.normal(0.0, 0.5, pixels.shape)
    noisy[3] += [80.0, -60.0]
    estimate = solve(DISTORTED, noisy, visible)
    assert estimate is not None
    assert not estimate.inliers[3]
    _assert_pose(estimate.pose, truth, m=0.01, deg=0.5)
    assert estimate.std[1] < 0.01  # ruff: ignore[magic-value-comparison]


def test_solve_absorbs_a_camera_pitch_error() -> None:
    truth = APPROACH[1]
    pixels, visible = CAM_FORK.project(truth.apply(KEYPOINTS_3D))
    miscalibrated = dataclasses.replace(CAM_FORK, pitch=np.deg2rad(0.7))
    fixed = solve(miscalibrated, pixels, visible, free_pitch=False)
    free = solve(miscalibrated, pixels, visible)
    assert fixed is not None
    assert free is not None
    _assert_pose(free.pose, truth)
    assert np.rad2deg(free.pitch_offset) == pytest.approx(-0.7, abs=0.01)
    assert abs(fixed.pose.x - truth.x) > 0.02  # ruff: ignore[magic-value-comparison]


def test_solve_reports_nothing_below_three_keypoints() -> None:
    pixels, _ = CAM_FORK.project(APPROACH[0].apply(KEYPOINTS_3D))
    visible = np.zeros(len(KEYPOINTS_3D), dtype=bool)
    visible[:2] = True
    assert solve(CAM_FORK, pixels, visible) is None


@pytest.mark.parametrize("truth", APPROACH[:2])
def test_tag_gives_the_pallet_pose(truth: Pose2D) -> None:
    mount = facing_truck_mount(tag_id=0, size=0.08, y=0.0, z=0.072)
    corners, visible = DISTORTED.project(
        tag_corners_robot(pallet_pose_6dof(truth), mount)
    )
    assert visible.all()
    gt = tag_ground_truth(DISTORTED, corners, mount)
    _assert_pose(gt.pose, truth, m=1e-4, deg=0.01)
    np.testing.assert_allclose(gt.off_floor, 0.0, atol=1e-4)

    pixels, _ = keypoint_labels(DISTORTED, gt.pose)
    expected, _ = DISTORTED.project(truth.apply(KEYPOINTS_3D))
    np.testing.assert_allclose(pixels, expected, atol=0.05)


def test_tag_reports_a_tilted_pallet() -> None:
    truth = APPROACH[1]
    mount = facing_truck_mount(tag_id=0, size=0.08, y=0.0, z=0.072)
    tilted = pallet_pose_6dof(truth, z=0.02, pitch=np.deg2rad(2.0))
    corners, _ = CAM_FORK.project(tag_corners_robot(tilted, mount))
    z, roll, pitch = tag_ground_truth(CAM_FORK, corners, mount).off_floor
    assert z == pytest.approx(0.02, abs=1e-3)
    assert np.rad2deg(pitch) == pytest.approx(2.0, abs=0.05)
    assert roll == pytest.approx(0.0, abs=1e-3)


def test_heatmap_targets_decode_to_subpixel_keypoints() -> None:
    keypoints = torch.tensor([[[100.3, 57.9], [250.0, 300.6], [10.0, 10.0]]])
    visible = torch.tensor([[True, True, False]])
    heatmaps = gaussian_targets(keypoints, visible)
    pixels, confidence = decode(heatmaps)
    torch.testing.assert_close(pixels[0, :2], keypoints[0, :2], atol=0.3, rtol=0)
    assert confidence[0, 2] == 0
    assert heatmap_loss(heatmaps, heatmaps, torch.ones_like(visible)) == 0


def test_crop_round_trips_and_samples_the_right_pixels() -> None:
    crop = Crop.around((800.0, 600.0, 1200.0, 900.0))
    points = np.array([[800.0, 600.0], [1100.5, 850.25]])
    np.testing.assert_allclose(crop.to_frame(crop.to_crop(points)), points)
    image = torch.zeros(1, 1080, 1920)
    image[0, 700, 1000] = 1.0
    sampled = crop.extract(image, size=round(crop.side))
    peak = np.unravel_index(int(sampled[0].argmax()), sampled.shape[1:])
    np.testing.assert_allclose(
        crop.to_frame(
            np.array([[peak[1] + 0.5, peak[0] + 0.5]]), size=round(crop.side)
        ),
        [[1000, 700]],
        atol=1.0,
    )


def test_keypoint_net_output_shape() -> None:
    net = PalletKeypointNet("resnet18", pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 384, 384))
    assert out.shape == (2, len(KEYPOINT_NAMES), 96, 96)


def test_rational_camera_matches_opencv() -> None:
    cv2 = pytest.importorskip("cv2")
    k = np.array([
        [CAM_FORK.fx, 0, CAM_FORK.cx],
        [0, CAM_FORK.fy, CAM_FORK.cy],
        [0, 0, 1],
    ])
    pixels = np.array([
        [100.0, 900.0],
        [974.0, 1079.0],
        [1700.0, 300.0],
        [500.0, 560.0],
    ])
    rays = CAM_FORK.rays(pixels) @ CAM_FORK.rotation  # optical frame
    projected, _ = cv2.projectPoints(
        rays, np.zeros(3), np.zeros(3), k, np.array(CAM_FORK.k)
    )
    np.testing.assert_allclose(projected.reshape(-1, 2), pixels, atol=1e-6)


def test_rational_camera_refuses_pixels_past_its_valid_field() -> None:
    assert np.isnan(CAM_FORK.rays(np.array([[0.0, 0.0]]))).all()


@pytest.mark.parametrize("truth", APPROACH[:2])
def test_silhouette_fit_recovers_the_pose_from_a_rendered_mask(truth: Pose2D) -> None:
    cv2 = pytest.importorskip("cv2")
    polygon = silhouette_polygon(CAM_FORK, truth)
    assert polygon is not None
    mask = np.zeros((CAM_FORK.height, CAM_FORK.width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    ys, xs = np.nonzero(mask)
    box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    result = fit(CAM_FORK, mask.astype(bool), box)
    assert result is not None
    assert result.iou > 0.95  # ruff: ignore[magic-value-comparison]
    _assert_pose(result.pose, truth, m=0.05, deg=3.0)
