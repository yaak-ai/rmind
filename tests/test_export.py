from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import pytest
import torch
from einops.layers.torch import Rearrange
from pytest_lazy_fixtures import lf
from rbyte.batch import Batch
from tensordict import TensorClass, TensorDict
from torch import Tensor
from torch.nn import Identity, Linear, Module, Sequential
from torch.testing import make_tensor
from torch.utils._pytree import tree_leaves, tree_map  # noqa: PLC2701
from torchvision.models import resnet18
from torchvision.ops import MLP
from torchvision.transforms.v2 import CenterCrop, Normalize, ToDtype

from rmind.components.base import TensorDictExport
from rmind.components.containers import ModuleDict
from rmind.components.episode import (
    Episode,
    EpisodeBuilder,
    EpisodeExport,
    Modality,
    PositionEncoding,
    SpecialToken,
    TokenMeta,
    TokenType,
)
from rmind.components.llm import IdentityEncoder
from rmind.components.nn import AtLeast3D, Embedding, Remapper
from rmind.components.norm import UniformBinner
from rmind.components.objectives.policy import PolicyObjective
from rmind.components.resnet import ResnetBackbone
from rmind.models.control_transformer import ControlTransformer


@pytest.fixture
def input_remapper() -> Module:
    return Remapper({
        Modality.IMAGE: {"cam_front_left": ("data", "cam_front_left")},
        Modality.CONTINUOUS: {
            "speed": ("data", "meta/VehicleMotion/speed"),
            "gas_pedal": ("data", "meta/VehicleMotion/gas_pedal_normalized"),
            "brake_pedal": ("data", "meta/VehicleMotion/brake_pedal_normalized"),
            "steering_angle": ("data", "meta/VehicleMotion/steering_angle_normalized"),
        },
        Modality.CONTEXT: {"waypoints": ("data", "waypoints/waypoints_normalized")},
        Modality.DISCRETE: {"turn_signal": ("data", "meta/VehicleState/turn_signal")},
    })


@pytest.fixture
def input_transforms() -> Module:
    return ModuleDict(
        modules={
            "image": {
                "cam_front_left": Sequential(
                    Rearrange("... h w c -> ... c h w"),
                    CenterCrop([320, 576]),
                    ToDtype(dtype=torch.float32, scale=True),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                )
            },
            "continuous": AtLeast3D(),
            "discrete": AtLeast3D(),
            "context": {"waypoints": Identity()},
        }
    )


@pytest.fixture
def input_builder(input_remapper: Module, input_transforms: Module) -> Module:
    return torch.nn.Sequential(input_remapper, input_transforms)


@pytest.fixture
def episode_builder() -> Module:
    return EpisodeBuilder(
        special_tokens={
            SpecialToken.OBSERVATION_SUMMARY: 0,
            SpecialToken.OBSERVATION_HISTORY: 1,
            SpecialToken.ACTION_SUMMARY: 2,
        },
        timestep=(
            TokenMeta(TokenType.OBSERVATION, Modality.IMAGE, "cam_front_left"),
            TokenMeta(TokenType.OBSERVATION, Modality.CONTINUOUS, "speed"),
            TokenMeta(TokenType.OBSERVATION, Modality.CONTEXT, "waypoints"),
            TokenMeta(
                TokenType.SPECIAL, Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY
            ),
            TokenMeta(
                TokenType.SPECIAL, Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY
            ),
            TokenMeta(TokenType.ACTION, Modality.CONTINUOUS, "gas_pedal"),
            TokenMeta(TokenType.ACTION, Modality.CONTINUOUS, "brake_pedal"),
            TokenMeta(TokenType.ACTION, Modality.CONTINUOUS, "steering_angle"),
            TokenMeta(TokenType.ACTION, Modality.DISCRETE, "turn_signal"),
            TokenMeta(TokenType.SPECIAL, Modality.SPECIAL, "action_summary"),
        ),
        tokenizers=ModuleDict({
            Modality.IMAGE: Identity(),
            Modality.CONTINUOUS: {
                "speed": UniformBinner(range=(0.0, 130.0), bins=512),
                "gas_pedal": UniformBinner(range=(0.0, 1.0), bins=512),
                "brake_pedal": UniformBinner(range=(0.0, 1.0), bins=512),
                "steering_angle": UniformBinner(range=(-1.0, 1.0), bins=512),
            },
            Modality.DISCRETE: Identity(),
            Modality.CONTEXT: {"waypoints": Identity()},
        }),
        embeddings=ModuleDict({
            Modality.IMAGE: Sequential(
                ResnetBackbone(resnet18("IMAGENET1K_V1"), freeze=True),
                Rearrange("... c h w -> ... (h w) c"),
            ),
            Modality.CONTINUOUS: {
                "speed": Embedding(512, 512),
                "gas_pedal": Embedding(512, 512),
                "brake_pedal": Embedding(512, 512),
                "steering_angle": Embedding(512, 512),
            },
            Modality.CONTEXT: {"waypoints": Linear(2, 512)},
            Modality.DISCRETE: {"turn_signal": Embedding(3, 512)},
            Modality.SPECIAL: Embedding(3, 512),
        }),
        position_encoding=ModuleDict({
            PositionEncoding.IMAGE: {
                "patch": {"row": Embedding(10, 512), "col": Embedding(18, 512)}
            },
            PositionEncoding.OBSERVATIONS: Embedding(192, 512),
            PositionEncoding.ACTIONS: Embedding(1, 512),
            PositionEncoding.SPECIAL: Embedding(1, 512),
            PositionEncoding.TIMESTEP: Embedding(6, 512),
        }),
    )


@pytest.fixture
def batch() -> TensorDict:
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
    ).to_tensordict()


@pytest.fixture
def batch_export(batch: Batch) -> dict[str, Any]:
    return batch.to_dict()


@pytest.fixture
def input(input_builder: Module, batch: TensorDict) -> TensorDict:
    with torch.inference_mode():
        return input_builder(batch).auto_batch_size_(2)


@pytest.fixture
def input_export(
    input_builder: Module, batch_export: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    with torch.inference_mode(), monkeypatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        return input_builder(batch_export)


@pytest.fixture
def episode(episode_builder: EpisodeBuilder, input: TensorDict) -> Episode:
    with torch.inference_mode():
        return episode_builder(input)


@pytest.fixture
def episode_export(
    episode_builder: EpisodeBuilder,
    input_export: TensorDict,
    monkeypatch: pytest.MonkeyPatch,
) -> Episode:
    with torch.inference_mode(), monkeypatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        return episode_builder(input_export)


@pytest.fixture
def encoder() -> Module:
    return IdentityEncoder()


@pytest.fixture
def policy_objective(encoder: Module) -> PolicyObjective:
    return PolicyObjective(
        heads=ModuleDict(
            modules={
                "continuous": {
                    "gas_pedal": MLP(1536, [512, 2], bias=False),
                    "brake_pedal": MLP(1536, [512, 2], bias=False),
                    "steering_angle": MLP(1536, [512, 2], bias=False),
                },
                "discrete": {"turn_signal": MLP(1536, [512, 3], bias=False)},
            }
        ),
        encoder=encoder,
        mask=None,
    )


@pytest.fixture
def control_transformer(
    input_builder: Module, episode_builder: Module, policy_objective: Module
) -> ControlTransformer:
    return ControlTransformer(
        input_builder=input_builder,
        episode_builder=episode_builder,
        objectives=ModuleDict({"policy": policy_objective}),
    )


def is_equal(a: Any, b: Any) -> bool:
    match a, b:
        case Tensor(), Tensor():
            return a.equal(b)

        case Tensor(), int() | float():
            return a.equal(torch.tensor(b))

        case int() | float(), Tensor():
            return torch.tensor(a).equal(b)

        case _:
            raise NotImplementedError


def to_dict(value: Any) -> dict[str, Any]:
    match value:
        case TensorDict() | TensorClass():
            return value.to_dict()

        case dict():
            return value

        case Mapping():
            return dict(value)

        case _ if is_dataclass(value):
            return asdict(value)

        case _:
            raise NotImplementedError


@pytest.mark.parametrize(
    ("module", "args", "args_export"),
    [
        (Identity(), (lf("batch"),), (lf("batch_export"),)),
        (lf("input_builder"), (lf("batch"),), (lf("batch_export"),)),
        (lf("episode_builder"), (lf("input"),), (lf("input_export"),)),
        (lf("policy_objective"), (lf("episode"),), (lf("episode_export"),)),
        (lf("control_transformer"), (lf("batch"),), (lf("batch_export"),)),
    ],
)
def test_module_outputs(
    module: Module,
    args: tuple[Any],
    args_export: tuple[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.inference_mode(), monkeypatch.context() as m:
        output = module(*args)
        m.setattr("torch.compiler._is_exporting_flag", True)
        output_export = module(*args_export)

    assert all(tree_leaves(tree_map(is_equal, to_dict(output), to_dict(output_export))))


@pytest.mark.parametrize(
    ("module", "args", "args_export"),
    [
        (lf("input_builder"), (lf("batch"),), (lf("batch_export"),)),
        (lf("episode_builder"), (lf("input"),), (lf("input_export"),)),
        (lf("policy_objective"), (lf("episode"),), (lf("episode_export"),)),
        (lf("control_transformer"), (lf("batch"),), (lf("batch_export"),)),
    ],
)
def test_module_export_aoti(
    module: Module, args: tuple[Any], args_export: tuple[Any]
) -> None:
    with torch.inference_mode():
        module_output = module(*args)

    exported = torch.export.export(module, args=args_export)
    with torch.inference_mode():
        exported_output = exported.module()(*args_export)

    assert all(
        tree_leaves(
            tree_map(is_equal, to_dict(module_output), to_dict(exported_output))
        )
    )

    package_path = torch._inductor.aoti_compile_and_package(exported)
    package = torch._inductor.aoti_load_package(package_path)
    package_output = package(*args_export)

    assert all(
        tree_leaves(
            tree_map(
                lambda x, y: torch.allclose(x, y, atol=1e-5)
                if isinstance(x, Tensor) and isinstance(y, Tensor)
                else is_equal(x, y),
                to_dict(module_output),
                to_dict(package_output),
            )
        )
    )
