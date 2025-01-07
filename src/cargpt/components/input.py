from typing import override

import torch
from pydantic import ConfigDict, validate_call
from rbyte.batch import Batch
from tensordict import TensorDict
from torch.nn import Module

from cargpt.components.episode import Modality
from cargpt.utils import ModuleDict
from cargpt.utils.functional import nan_padder


class InputBuilder(Module):
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, transforms: ModuleDict) -> None:
        super().__init__()

        self.transforms = transforms

    @override
    def forward(self, batch: Batch) -> TensorDict:
        data = batch.data
        input: TensorDict = (
            TensorDict.from_dict(
                {
                    Modality.IMAGE: {"cam_front_left": data["cam_front_left"]},
                    Modality.CONTINUOUS: {
                        "speed": data["VehicleMotion.speed"],
                        "gas_pedal": data["VehicleMotion.gas_pedal_normalized"],
                        "brake_pedal": data["VehicleMotion.brake_pedal_normalized"],
                        "steering_angle": data[
                            "VehicleMotion.steering_angle_normalized"
                        ],
                    },
                    Modality.DISCRETE: {
                        "turn_signal": data["VehicleState.turn_signal"]
                    },
                },
                device=batch.device,  # pyright: ignore[reportAttributeAccessIssue]
            )
            .auto_batch_size_(batch_dims=2)
            .refine_names("b", "t")
        )

        b, t = input.batch_size
        diff_keys = tuple(
            (Modality.CONTINUOUS, k)
            for k in ("gas_pedal", "brake_pedal", "steering_angle")
        )
        diffs: TensorDict = (
            input.select(*diff_keys)
            .apply(torch.diff, batch_size=[b, t - 1])
            .apply(nan_padder(pad=(0, 1), dim=-1), batch_size=[b, t])
        )

        for k in diff_keys:
            diffs = diffs.rename_key_(k, (*k[:-1], f"{k[-1]}_diff"))

        input = input.update(diffs)

        return self.transforms(input).lock_()
