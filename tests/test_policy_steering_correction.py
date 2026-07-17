from typing import Any

import torch
from torch.nn import Identity

from rmind.components.base import Modality
from rmind.components.containers import ModuleDict
from rmind.components.objectives import PolicyObjective


class _FakeEpisode:
    """Minimal `episode.get(key, default=None)` stand-in -- exercising
    `_apply_steering_correction` doesn't need a real `Episode` (attention
    masks, token index, etc.), just nested-key lookup.
    """

    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, key: tuple[str, ...], default: Any = None) -> Any:
        node = self._data
        for k in key:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def _objective(device: torch.device, **kwargs: Any) -> PolicyObjective:
    return PolicyObjective(
        heads=ModuleDict(modules={Modality.CONTINUOUS: {"steering_angle": Identity()}}),
        history_steps=5,
        **kwargs,
    ).to(device)


def test_disabled_by_default_is_identity(device: torch.device) -> None:
    objective = _objective(device)
    targets = {
        Modality.CONTINUOUS: {"steering_angle": torch.zeros(2, 6, 1, device=device)}
    }

    out = objective._apply_steering_correction(targets, episode=_FakeEpisode({}))  # noqa: SLF001

    assert out is targets


def test_missing_dyaw_or_speed_is_identity(device: torch.device) -> None:
    objective = _objective(
        device, dyaw_key=("input", "viewpoint_aug", "dyaw"), steering_correction_gain=1.5
    )
    targets = {
        Modality.CONTINUOUS: {"steering_angle": torch.zeros(2, 6, 1, device=device)}
    }

    out = objective._apply_steering_correction(targets, episode=_FakeEpisode({}))  # noqa: SLF001

    assert out is targets


def test_positive_dyaw_corrects_steering_left_and_decays(device: torch.device) -> None:
    """Positive dyaw (compass, rightward) must subtract from steering_angle
    (steer left to recover), matching the sign established for the
    trajectory-head correction in `test_augment.py`, and decay to exactly 0
    by `steering_correction_decay_steps`.
    """
    objective = _objective(
        device,
        dyaw_key=("input", "viewpoint_aug", "dyaw"),
        steering_correction_gain=1.5,
        steering_correction_decay_steps=2,
    )
    b, h, t = 2, 6, 11
    steering = torch.zeros(b, h, 1, device=device)
    targets = {Modality.CONTINUOUS: {"steering_angle": steering}}
    dyaw = torch.tensor([10.0, -10.0], device=device).reshape(b, 1, 1).expand(b, t, 1)
    episode = _FakeEpisode({
        "input": {
            "viewpoint_aug": {"dyaw": dyaw},
            "continuous": {"speed": torch.full((b, t), 60.0, device=device)},
        }
    })

    out = objective._apply_steering_correction(targets, episode=episode)  # noqa: SLF001
    corrected = out[Modality.CONTINUOUS]["steering_angle"]

    assert corrected.shape == steering.shape
    assert corrected[0, 0, 0] < 0  # turn right (dyaw>0) -> correct left
    assert corrected[1, 0, 0] > 0  # turn left (dyaw<0) -> correct right
    assert torch.equal(corrected[:, 2:], steering[:, 2:])  # decayed to 0 by step 2


def test_single_step_path_shape(device: torch.device) -> None:
    """The eval/action_horizon==1 path reads targets as (b, 1) (no `h` axis)
    -- must not break the (b, h, 1) assumption used for the horizon decay.
    """
    objective = _objective(
        device,
        dyaw_key=("input", "viewpoint_aug", "dyaw"),
        steering_correction_gain=1.5,
    )
    b, t = 2, 11
    steering = torch.zeros(b, 1, device=device)
    targets = {Modality.CONTINUOUS: {"steering_angle": steering}}
    dyaw = torch.tensor([10.0, -10.0], device=device).reshape(b, 1, 1).expand(b, t, 1)
    episode = _FakeEpisode({
        "input": {
            "viewpoint_aug": {"dyaw": dyaw},
            "continuous": {"speed": torch.full((b, t), 60.0, device=device)},
        }
    })

    out = objective._apply_steering_correction(targets, episode=episode)  # noqa: SLF001
    corrected = out[Modality.CONTINUOUS]["steering_angle"]

    assert corrected.shape == steering.shape
    assert corrected[0, 0] < 0
    assert corrected[1, 0] > 0
