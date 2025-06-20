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
from torch.utils._pytree import PyTree, tree_leaves, tree_map  # noqa: PLC2701
from torchvision.models import resnet18
from torchvision.ops import MLP
from torchvision.transforms.v2 import CenterCrop, Normalize, ToDtype

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
from rmind.components.llm import TransformerEncoder
from rmind.components.mask import TorchAttentionMaskLegend
from rmind.components.nn import AtLeast3D, Embedding, Remapper
from rmind.components.norm import Normalize as _Normalize
from rmind.components.norm import UniformBinner
from rmind.components.objectives.policy import PolicyObjective
from rmind.components.resnet import ResnetBackbone
from rmind.models.control_transformer import ControlTransformer


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


@pytest.fixture
def batch_dict(batch: Batch) -> PyTree:
    return batch.to_dict()


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
            "image": Sequential(
                Rearrange("... h w c -> ... c h w"),
                CenterCrop([320, 576]),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ),
            "continuous": AtLeast3D(),
            "discrete": AtLeast3D(),
            "context": Identity(),
        }
    )


@pytest.fixture
def input_builder(input_remapper: Module, input_transforms: Module) -> Module:
    return torch.nn.Sequential(input_remapper, input_transforms)


@pytest.fixture
def input(input_builder: Module, batch_dict: PyTree) -> PyTree:
    with torch.inference_mode():
        return input_builder(batch_dict)


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
                _Normalize(p=2, dim=-1),
            ),
            Modality.CONTINUOUS: {
                "speed": Embedding(512, 512),
                "gas_pedal": Embedding(512, 512),
                "brake_pedal": Embedding(512, 512),
                "steering_angle": Embedding(512, 512),
            },
            Modality.CONTEXT: {
                "waypoints": Sequential(Linear(2, 512), _Normalize(p=2, dim=-1))
            },
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
def episode(episode_builder: EpisodeBuilder, input: PyTree) -> Episode:
    with torch.inference_mode():
        return episode_builder(input)


@pytest.fixture
def episode_export(
    episode_builder: EpisodeBuilder, input: PyTree, monkeypatch: pytest.MonkeyPatch
) -> EpisodeExport:
    with torch.inference_mode(), monkeypatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        return episode_builder(input)


@pytest.fixture
def encoder() -> Module:
    return TransformerEncoder(
        dim_model=512,
        num_heads=4,
        num_layers=8,
        attn_dropout=0.1,
        resid_dropout=0.1,
        mlp_dropout=0.1,
        hidden_layer_multiplier=1,
    )


@pytest.fixture
def policy_mask(episode: Episode) -> Tensor:
    return PolicyObjective.build_attention_mask(
        episode.index, episode.timestep, legend=TorchAttentionMaskLegend
    ).mask


@pytest.fixture
def policy_objective(encoder: Module, policy_mask: Tensor) -> PolicyObjective:
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
        mask=policy_mask,
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


def to_tree(value: Any) -> dict[str, Any]:
    match value:
        case TensorDict() | TensorClass():
            return value.to_dict()

        case dict():
            return value

        case Mapping():
            return dict(value)

        case _ if is_dataclass(value):
            return asdict(value)  # pyright: ignore[reportArgumentType]

        case _:
            return value


@pytest.mark.parametrize(
    ("module", "args", "args_export"),
    [
        (lf("policy_objective"), (lf("episode"),), (lf("episode_export"),)),
        (lf("control_transformer"), (lf("batch"),), (lf("batch_dict"),)),
    ],
)
def test_module_export_aoti(
    module: Module, args: tuple[Any], args_export: tuple[Any]
) -> None:
    module = module.eval()
    with torch.inference_mode():
        module_output = module(*args)
        exported = torch.export.export(module, args=args_export)
        exported_output = exported.module()(*args_export)

    assert all(
        tree_leaves(
            tree_map(is_equal, to_tree(module_output), to_tree(exported_output))
        )
    )

    package_path = torch._inductor.aoti_compile_and_package(exported)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    package = torch._inductor.aoti_load_package(package_path)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    package_output = package(*args_export)

    assert all(
        tree_leaves(
            tree_map(
                lambda x, y: torch.allclose(x, y, atol=1e-5)
                if isinstance(x, Tensor) and isinstance(y, Tensor)
                else is_equal(x, y),
                to_tree(module_output),
                to_tree(package_output),
            )
        )
    )
