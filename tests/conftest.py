from collections.abc import Generator
from typing import Any

import pytest
import torch
from rbyte.batch import Batch
from tensordict import TensorDict
from torch.testing import make_tensor


@pytest.fixture(scope="session", autouse=True)
def _set_float32_matmul_precision() -> Generator[None, Any, None]:  # pyright: ignore[reportUnusedFunction]
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(prev)


@pytest.fixture
def batch() -> Batch:
    return Batch(
        data=TensorDict(
            {
                "cam_front_left": make_tensor(
                    (1, 6, 324, 576, 3),
                    dtype=torch.uint8,
                    device="cpu",
                    low=0,
                    high=256,
                ),
                "meta/ImageMetadata.cam_front_left/frame_idx": make_tensor(
                    (1, 6), dtype=torch.int32, device="cpu", low=0
                ),
                "meta/ImageMetadata.cam_front_left/time_stamp": make_tensor(
                    (1, 6), dtype=torch.int64, device="cpu", low=0
                ),
                "meta/VehicleMotion/brake_pedal_normalized": make_tensor(
                    (1, 6), dtype=torch.float32, device="cpu", low=0.0, high=1.0
                ),
                "meta/VehicleMotion/gas_pedal_normalized": make_tensor(
                    (1, 6), dtype=torch.float32, device="cpu", low=0.0, high=1.0
                ),
                "meta/VehicleMotion/steering_angle_normalized": make_tensor(
                    (1, 6), dtype=torch.float32, device="cpu", low=-1.0, high=1.0
                ),
                "meta/VehicleMotion/speed": make_tensor(
                    (1, 6), dtype=torch.float32, device="cpu", low=0.0, high=130.0
                ),
                "meta/VehicleState/turn_signal": make_tensor(
                    (1, 6), dtype=torch.int64, device="cpu", low=0, high=3
                ),
                "waypoints/waypoints_normalized": make_tensor(
                    (1, 6, 10, 2), dtype=torch.float32, device="cpu", low=0.0, high=20.0
                ),
            },
            batch_size=[1],
            device=None,
        ),
        batch_size=[1],
        device=None,
    )
