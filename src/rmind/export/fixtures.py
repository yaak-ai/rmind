from collections.abc import Generator
from typing import Any

import pytest
import torch
from einops.layers.torch import Rearrange
from rbyte.batch import Batch
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Linear, Module
from torch.testing import make_tensor
from torchvision.models import resnet18
from torchvision.ops import MLP
from torchvision.transforms.v2 import CenterCrop, Normalize, ToDtype

from rmind.components.base import TensorTree
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
from rmind.components.nn import (
    AtLeast3D,
    DiffLast,
    Embedding,
    Identity,
    Remapper,
    Sequential,
)
from rmind.components.norm import MuLawEncoding, Scaler, UniformBinner
from rmind.components.norm import Normalize as _Normalize
from rmind.components.objectives import PolicyObjective
from rmind.components.resnet import ResnetBackbone
from rmind.models.control_transformer import ControlTransformer

EMBEDDING_DIM = 512


def _set_float32_matmul_precision() -> Generator[None, Any, None]:  # pyright: ignore[reportUnusedFunction]
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(prev)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return torch.device(device)


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


def batch_dict(batch: Batch) -> TensorTree:
    return batch.to_dict()


def tokenizers() -> ModuleDict:
    return ModuleDict({
        Modality.IMAGE: Identity(),
        Modality.CONTINUOUS: {
            "speed": UniformBinner(range=(0.0, 130.0), bins=EMBEDDING_DIM),
            "gas_pedal": UniformBinner(range=(0.0, 1.0), bins=EMBEDDING_DIM),
            "gas_pedal_diff": MuLawEncoding(quantization_channels=EMBEDDING_DIM),
            "brake_pedal": UniformBinner(range=(0.0, 1.0), bins=EMBEDDING_DIM),
            "brake_pedal_diff": MuLawEncoding(quantization_channels=EMBEDDING_DIM),
            "steering_angle": UniformBinner(range=(-1.0, 1.0), bins=EMBEDDING_DIM),
            "steering_angle_diff": Sequential(
                Scaler(in_range=(-2.0, 2.0), out_range=(-1.0, 1.0)),
                MuLawEncoding(quantization_channels=EMBEDDING_DIM),
            ),
        },
        Modality.DISCRETE: Identity(),
        Modality.CONTEXT: {"waypoints": Identity()},
    })


def episode_builder(tokenizers: ModuleDict) -> EpisodeBuilder:
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
        input_transform=Sequential(
            Remapper({
                Modality.IMAGE: {"cam_front_left": ("data", "cam_front_left")},
                Modality.CONTINUOUS: {
                    "speed": ("data", "meta/VehicleMotion/speed"),
                    "gas_pedal": ("data", "meta/VehicleMotion/gas_pedal_normalized"),
                    "gas_pedal_diff": (
                        "data",
                        "meta/VehicleMotion/gas_pedal_normalized",
                    ),
                    "brake_pedal": (
                        "data",
                        "meta/VehicleMotion/brake_pedal_normalized",
                    ),
                    "brake_pedal_diff": (
                        "data",
                        "meta/VehicleMotion/brake_pedal_normalized",
                    ),
                    "steering_angle": (
                        "data",
                        "meta/VehicleMotion/steering_angle_normalized",
                    ),
                    "steering_angle_diff": (
                        "data",
                        "meta/VehicleMotion/steering_angle_normalized",
                    ),
                },
                Modality.CONTEXT: {
                    "waypoints": ("data", "waypoints/waypoints_normalized")
                },
                Modality.DISCRETE: {
                    "turn_signal": ("data", "meta/VehicleState/turn_signal")
                },
            }),
            ModuleDict(
                modules={
                    Modality.IMAGE: Sequential(
                        Rearrange("... h w c -> ... c h w"),
                        CenterCrop([320, 576]),
                        ToDtype(dtype=torch.float32, scale=True),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ),
                    Modality.CONTINUOUS: {
                        "speed": AtLeast3D(),
                        "gas_pedal": AtLeast3D(),
                        "gas_pedal_diff": DiffLast(append=torch.nan),
                        "brake_pedal": AtLeast3D(),
                        "brake_pedal_diff": DiffLast(append=torch.nan),
                        "steering_angle": AtLeast3D(),
                        "steering_angle_diff": DiffLast(append=torch.nan),
                    },
                    Modality.DISCRETE: AtLeast3D(),
                    Modality.CONTEXT: Identity(),
                }
            ),
        ),
        tokenizers=tokenizers,
        embeddings=ModuleDict({
            Modality.IMAGE: Sequential(
                ResnetBackbone(resnet18("IMAGENET1K_V1"), freeze=True),
                Rearrange("... c h w -> ... (h w) c"),
                _Normalize(p=2, dim=-1),
            ),
            Modality.CONTINUOUS: {
                "speed": Embedding(EMBEDDING_DIM, EMBEDDING_DIM),
                "gas_pedal": Embedding(EMBEDDING_DIM, EMBEDDING_DIM),
                "gas_pedal_diff": None,
                "brake_pedal": Embedding(EMBEDDING_DIM, EMBEDDING_DIM),
                "brake_pedal_diff": None,
                "steering_angle": Embedding(EMBEDDING_DIM, EMBEDDING_DIM),
                "steering_angle_diff": None,
            },
            Modality.CONTEXT: {
                "waypoints": Sequential(
                    Linear(2, EMBEDDING_DIM), _Normalize(p=2, dim=-1)
                )
            },
            Modality.DISCRETE: {"turn_signal": Embedding(3, EMBEDDING_DIM)},
            Modality.SPECIAL: Embedding(3, EMBEDDING_DIM),
        }),
        position_encoding=ModuleDict({
            PositionEncoding.IMAGE: {
                "patch": {
                    "row": Embedding(10, EMBEDDING_DIM),
                    "col": Embedding(18, EMBEDDING_DIM),
                }
            },
            PositionEncoding.OBSERVATIONS: Embedding(192, EMBEDDING_DIM),
            PositionEncoding.ACTIONS: Embedding(1, EMBEDDING_DIM),
            PositionEncoding.SPECIAL: Embedding(1, EMBEDDING_DIM),
            PositionEncoding.TIMESTEP: Embedding(6, EMBEDDING_DIM),
        }),
    )


def episode(episode_builder: EpisodeBuilder, batch_dict: TensorTree) -> Episode:
    with torch.inference_mode():
        return episode_builder(batch_dict)


def encoder() -> Module:
    return TransformerEncoder(
        dim_model=EMBEDDING_DIM,
        num_heads=4,
        num_layers=8,
        attn_dropout=0.1,
        resid_dropout=0.1,
        mlp_dropout=0.1,
        hidden_layer_multiplier=1,
    )


def policy_objective(encoder: Module, policy_mask: Tensor) -> PolicyObjective:
    return PolicyObjective(
        encoder=encoder,
        mask=policy_mask,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": MLP(EMBEDDING_DIM * 3, [512, 2], bias=False),
                    "brake_pedal": MLP(EMBEDDING_DIM * 3, [512, 2], bias=False),
                    "steering_angle": MLP(EMBEDDING_DIM * 3, [512, 2], bias=False),
                },
                Modality.DISCRETE: {"turn_signal": MLP(EMBEDDING_DIM * 3, [512, 3], bias=False)},
            }
        ),
    )


def episode_export(
    episode_builder: EpisodeBuilder,
    batch_dict: TensorTree,
    monkeypatch: pytest.MonkeyPatch,
) -> EpisodeExport:
    with torch.inference_mode(), monkeypatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        return episode_builder(batch_dict)


def policy_mask(episode: Episode) -> Tensor:
    return PolicyObjective.build_attention_mask(
        episode.index, episode.timestep, legend=TorchAttentionMaskLegend
    ).mask


def objectives(policy_objective: Module) -> ModuleDict:
    return ModuleDict({"policy": policy_objective})


def control_transformer(
    episode_builder: Module, objectives: ModuleDict
) -> ControlTransformer:
    return ControlTransformer(episode_builder=episode_builder, objectives=objectives)
