import torch
from kornia.geometry.transform import warp_perspective
from tensordict import TensorDict
from torch.testing import assert_close, make_tensor

from rmind.components.augment import HistoryViewpointPerturbation, _yaw_homographies
from rmind.utils.functional import build_relative_trajectory


def _batch(device: torch.device, b: int = 8, t: int = 6, h: int = 32, w: int = 32) -> dict:
    return {
        "image": {
            "cam_front_left": make_tensor(
                (b, t, 3, h, w), dtype=torch.float32, device=device, low=-2.0, high=2.0
            )
        },
        "trajectory": {
            "heading": make_tensor(
                (b, t, 1), dtype=torch.float32, device=device, low=0.0, high=360.0
            )
        },
    }


def test_eval_mode_leaves_image_and_heading_untouched(device: torch.device) -> None:
    module = HistoryViewpointPerturbation(probability=1.0).to(device).eval()
    input = _batch(device)

    out = module(input)

    assert_close(out["image"]["cam_front_left"], input["image"]["cam_front_left"])
    assert_close(out["trajectory"]["heading"], input["trajectory"]["heading"])


def test_zero_probability_leaves_image_and_heading_untouched(device: torch.device) -> None:
    module = HistoryViewpointPerturbation(probability=0.0).to(device).train()
    input = _batch(device)

    out = module(input)

    assert_close(out["image"]["cam_front_left"], input["image"]["cam_front_left"])
    assert_close(out["trajectory"]["heading"], input["trajectory"]["heading"])


def test_dyaw_leaf_always_attached_with_episode_compatible_shape(
    device: torch.device,
) -> None:
    """`dyaw_key` must be a static property of config (present/absent
    regardless of the random draw or train/eval), broadcast over `t`, or
    `EpisodeBuilder`'s `TensorDict(input, batch_size=[b, t])` breaks --
    that's what actually enforces this shape, so build one here too.
    """
    b, t = 8, 6
    for probability, training in [(1.0, True), (0.0, True), (1.0, False)]:
        module = (
            HistoryViewpointPerturbation(probability=probability)
            .to(device)
            .train(training)
        )
        input = _batch(device, b=b, t=t)

        out = module(input)

        dyaw = out["viewpoint_aug"]["dyaw"]
        assert dyaw.shape == (b, t, 1)
        TensorDict(out, batch_size=[b, t])  # must not raise

        if not training or probability <= 0:
            assert torch.equal(dyaw, torch.zeros_like(dyaw))


def test_probability_one_perturbs_pivot_tick_only(device: torch.device) -> None:
    pivot_index = 4
    module = (
        HistoryViewpointPerturbation(probability=1.0, pivot_index=pivot_index, max_degrees=8.0)
        .to(device)
        .train()
    )
    input = _batch(device, t=6)

    out = module(input)

    image, heading = input["image"]["cam_front_left"], input["trajectory"]["heading"]
    out_image, out_heading = out["image"]["cam_front_left"], out["trajectory"]["heading"]
    dyaw = out["viewpoint_aug"]["dyaw"]

    assert not torch.equal(dyaw, torch.zeros_like(dyaw))
    for t in range(image.shape[1]):
        if t == pivot_index:
            assert not torch.equal(out_image[:, t], image[:, t])
            assert not torch.equal(out_heading[:, t], heading[:, t])
        else:
            assert_close(out_image[:, t], image[:, t])
            assert_close(out_heading[:, t], heading[:, t])


def test_missing_keys_pass_through(device: torch.device) -> None:
    module = HistoryViewpointPerturbation(probability=1.0).to(device).train()
    input = {"image": {}, "trajectory": {}}

    out = module(input)

    assert out is input


def test_positive_dyaw_shifts_scene_left_in_image(device: torch.device) -> None:
    """Sign-convention lock: a positive (compass, rightward) yaw offset must
    make the forward camera look like the car panned right, which shifts
    scene content toward smaller x (left) in the image -- see
    `HistoryViewpointPerturbation`'s docstring for why this must match
    `heading`'s sign convention.
    """
    h, w = 32, 64
    img = torch.zeros(1, 3, h, w, device=device)
    img[:, :, :, 48:52] = 1.0  # bright stripe right of center

    def centroid_x(dyaw: float) -> float:
        homography = _yaw_homographies(
            torch.tensor([dyaw], device=device), fx=60.0, fy=60.0, cx=w / 2, cy=h / 2
        )
        out = warp_perspective(img, homography, dsize=(h, w), padding_mode="border")
        row = out[0, 0, h // 2]
        xs = torch.arange(w, device=device, dtype=torch.float32)
        return ((row * xs).sum() / row.sum().clamp(min=1e-6)).item()

    assert centroid_x(10.0) < 50.0 < centroid_x(-10.0)


def test_heading_perturbation_implies_recovery_trajectory() -> None:
    """A perturbed pivot heading, with the recorded continuation unchanged,
    must make `build_relative_trajectory`'s first step point back toward the
    real path -- otherwise the augmentation and the trajectory GT it feeds
    disagree about which way is "recovering".
    """
    xy = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]])
    heading = torch.tensor([[[10.0], [0.0]]])  # perturbed +10 (compass, "right") at pivot

    gt = build_relative_trajectory(xy, heading, history_steps=1)

    dx, _dy, dyaw = gt[0, 0]
    assert dx < 0  # correct back left
    assert dyaw < 0  # ... by steering left
