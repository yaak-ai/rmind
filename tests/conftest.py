from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from einops.layers.torch import Rearrange
from rbyte.types import Batch
from tensordict import TensorDict
from torch.nn import LayerNorm, Linear, Module, MSELoss
from torch.testing import make_tensor
from torchvision.ops import MLP
from torchvision.transforms.v2 import CenterCrop, Normalize, Resize, ToDtype

from rmind.components.base import TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.episode import (
    Episode,
    EpisodeBuilder,
    Modality,
    PositionEncoding,
    SpecialToken,
    TokenMeta,
    TokenType,
)
from rmind.components.llm import TransformerEncoder
from rmind.components.loss import GaussianNLLLoss, LogitBiasCrossEntropyLoss
from rmind.components.nn import (
    AtLeast3D,
    DiffLast,
    Embedding,
    Identity,
    Remapper,
    Sequential,
)
from rmind.components.norm import MuLawEncoding, Scaler, UniformBinner
from rmind.components.objectives import (
    ForwardDynamicsPredictionObjective,
    InverseDynamicsPredictionObjective,
    MemoryExtractionObjective,
    PolicyObjective,
    RandomMaskedHindsightControlObjective,
)
from rmind.components.timm_backbone import TimmBackbone


@dataclass
class NumBins:
    speed: int = 512
    gas_pedal: int = 255
    brake_pedal: int = 165
    steering_angle: int = 961


@pytest.fixture(scope="module")
def num_bins() -> NumBins:
    return NumBins()


@pytest.fixture(scope="module")
def embedding_dim() -> int:
    return 384


def _set_float32_matmul_precision() -> Generator[None, Any, None]:  # pyright: ignore[reportUnusedFunction]
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(prev)


@pytest.fixture(
    scope="module",
    params=[
        next(
            device
            for device in ("cuda", "mps", "cpu")
            if getattr(torch, device).is_available()
        )
    ],
)
def device(request) -> torch.device:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]  # noqa: ANN001
    return torch.device(request.param)


@pytest.fixture(scope="module")
def batch(device: torch.device) -> Batch:
    b, t = 2, 6
    return Batch(
        data=TensorDict(
            {
                "cam_front_left": make_tensor(
                    (b, t, 324, 576, 3),
                    dtype=torch.uint8,
                    device=device,
                    low=0,
                    high=256,
                ),
                "meta/VehicleMotion/brake_pedal_normalized": make_tensor(
                    (b, t), dtype=torch.float32, device=device, low=0.0, high=1.0
                ),
                "meta/VehicleMotion/gas_pedal_normalized": make_tensor(
                    (b, t), dtype=torch.float32, device=device, low=0.0, high=1.0
                ),
                "meta/VehicleMotion/steering_angle_normalized": make_tensor(
                    (b, t), dtype=torch.float32, device=device, low=-1.0, high=1.0
                ),
                "meta/VehicleMotion/speed": make_tensor(
                    (b, t), dtype=torch.float32, device=device, low=0.0, high=130.0
                ),
                "meta/VehicleState/turn_signal": make_tensor(
                    (b, t), dtype=torch.int64, device=device, low=0, high=3
                ),
                "waypoints/xy_normalized": make_tensor(
                    (b, t, 10, 2),
                    dtype=torch.float32,
                    device=device,
                    low=0.0,
                    high=20.0,
                ),
            },
            batch_size=[b],
            device=device,
        ),
        batch_size=[b],
        device=device,
    )


@pytest.fixture(scope="module")
def batch_dict(batch: Batch) -> TensorTree:
    return batch.to_dict(retain_none=False)


@pytest.fixture(scope="module")
def tokenizers(device: torch.device, num_bins: NumBins) -> ModuleDict:
    return ModuleDict({
        Modality.IMAGE: Identity(),
        Modality.CONTINUOUS: {
            "speed": UniformBinner(range=(0.0, 130.0), bins=num_bins.speed),
            "gas_pedal": UniformBinner(range=(0.0, 1.0), bins=num_bins.gas_pedal),
            "gas_pedal_diff": MuLawEncoding(quantization_channels=num_bins.gas_pedal),
            "brake_pedal": UniformBinner(range=(0.0, 1.0), bins=num_bins.brake_pedal),
            "brake_pedal_diff": MuLawEncoding(
                quantization_channels=num_bins.brake_pedal
            ),
            "steering_angle": UniformBinner(
                range=(-1.0, 1.0), bins=num_bins.steering_angle
            ),
            "steering_angle_diff": Sequential(
                Scaler(in_range=(-2.0, 2.0), out_range=(-1.0, 1.0)),
                MuLawEncoding(quantization_channels=num_bins.steering_angle),
            ),
        },
        Modality.DISCRETE: Identity(),
        Modality.CONTEXT: {"waypoints": Identity()},
    }).to(device)


@pytest.fixture(scope="module")
def episode_builder(
    tokenizers: ModuleDict, device: torch.device, num_bins: NumBins, embedding_dim: int
) -> Module:
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
                Modality.CONTEXT: {"waypoints": ("data", "waypoints/xy_normalized")},
                Modality.DISCRETE: {
                    "turn_signal": ("data", "meta/VehicleState/turn_signal")
                },
            }),
            ModuleDict(
                modules={
                    Modality.IMAGE: Sequential(
                        Rearrange("... h w c -> ... c h w"),
                        CenterCrop([320, 576]),
                        Resize([256, 256]),
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
                TimmBackbone(
                    "vit_small_patch16_dinov3.lvd1689m",
                    freeze=True,
                    out_indices=[10],
                    img_size=[256, 256],
                ),
                Rearrange("... c h w -> ... (h w) c"),
            ),
            Modality.CONTINUOUS: {
                "speed": Embedding(num_bins.speed, embedding_dim),
                "gas_pedal": Embedding(num_bins.gas_pedal, embedding_dim),
                "brake_pedal": Embedding(num_bins.brake_pedal, embedding_dim),
                "steering_angle": Embedding(num_bins.steering_angle, embedding_dim),
                "gas_pedal_diff": None,
                "brake_pedal_diff": None,
                "steering_angle_diff": None,
            },
            Modality.CONTEXT: {"waypoints": Linear(2, embedding_dim)},
            Modality.DISCRETE: {"turn_signal": Embedding(3, embedding_dim)},
            Modality.SPECIAL: Embedding(3, embedding_dim),
        }),
        projections=ModuleDict({
            Modality.IMAGE: Sequential(
                LayerNorm(embedding_dim), Linear(embedding_dim, embedding_dim)
            ),
            Modality.CONTINUOUS: {
                "speed": Linear(embedding_dim, embedding_dim),
                "gas_pedal": Linear(embedding_dim, embedding_dim),
                "brake_pedal": Linear(embedding_dim, embedding_dim),
                "steering_angle": Linear(embedding_dim, embedding_dim),
                "gas_pedal_diff": None,
                "brake_pedal_diff": None,
                "steering_angle_diff": None,
            },
            Modality.CONTEXT: {"waypoints": Linear(embedding_dim, embedding_dim)},
            Modality.DISCRETE: {"turn_signal": Linear(embedding_dim, embedding_dim)},
            Modality.SPECIAL: Linear(embedding_dim, embedding_dim),
        }),
        position_encoding=ModuleDict({
            PositionEncoding.CONTEXT: {"waypoints": Embedding(10, embedding_dim)},
            PositionEncoding.ACTIONS: Embedding(1, embedding_dim),
            PositionEncoding.SPECIAL: Embedding(1, embedding_dim),
            PositionEncoding.TIMESTEP: Embedding(6, embedding_dim),
            PositionEncoding.OBSERVATIONS: None,
        }),
    ).to(device)


@pytest.fixture(scope="module")
def episode(episode_builder: EpisodeBuilder, batch_dict: TensorTree) -> Episode:
    return episode_builder(batch_dict)


@pytest.fixture(scope="module")
def encoder(embedding_dim: int) -> Module:
    return TransformerEncoder(
        dim_model=embedding_dim,
        num_heads=2,
        num_layers=2,
        attn_dropout=0.1,
        resid_dropout=0.1,
        mlp_dropout=0.1,
        hidden_layer_multiplier=1,
    )


@pytest.fixture(scope="module")
def inverse_dynamics_prediction_objective(
    encoder: Module, device: torch.device, embedding_dim: int, num_bins: NumBins
) -> InverseDynamicsPredictionObjective:
    logit_bias = torch.tensor(0)

    return InverseDynamicsPredictionObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": Linear(
                        2 * embedding_dim, num_bins.gas_pedal, bias=False
                    ),
                    "brake_pedal": Linear(
                        2 * embedding_dim, num_bins.brake_pedal, bias=False
                    ),
                    "steering_angle": Linear(
                        2 * embedding_dim, num_bins.steering_angle, bias=False
                    ),
                },
                Modality.DISCRETE: {
                    "turn_signal": Linear(2 * embedding_dim, 3, bias=False)
                },
            }
        ),
        losses=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": LogitBiasCrossEntropyLoss(logit_bias=logit_bias),
                    "brake_pedal": LogitBiasCrossEntropyLoss(logit_bias=logit_bias),
                    "steering_angle": LogitBiasCrossEntropyLoss(logit_bias=logit_bias),
                },
                Modality.DISCRETE: {
                    "turn_signal": LogitBiasCrossEntropyLoss(logit_bias=logit_bias)
                },
            }
        ),
        targets={
            (modality := Modality.CONTINUOUS): {
                "gas_pedal": ("input_tokens", modality, "gas_pedal"),
                "brake_pedal": ("input_tokens", modality, "brake_pedal"),
                "steering_angle": ("input_tokens", modality, "steering_angle"),
            },
            (modality := Modality.DISCRETE): {
                "turn_signal": ("input_tokens", modality, "turn_signal")
            },
        },
    ).to(device)


@pytest.fixture(scope="module")
def forward_dynamics_prediction_objective(
    encoder: Module, device: torch.device, embedding_dim: int, num_bins: NumBins
) -> ForwardDynamicsPredictionObjective:
    logit_bias = torch.tensor(0)

    return ForwardDynamicsPredictionObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.IMAGE: {
                    "cam_front_left": Linear(
                        3 * embedding_dim, embedding_dim, bias=False
                    )
                },
                Modality.CONTINUOUS: {
                    "speed": Linear(3 * embedding_dim, num_bins.speed, bias=False)
                },
            }
        ),
        losses=ModuleDict(
            modules={
                Modality.IMAGE: {"cam_front_left": MSELoss()},
                Modality.CONTINUOUS: {
                    "speed": LogitBiasCrossEntropyLoss(logit_bias=logit_bias)
                },
            }
        ),
        targets={
            (modality := Modality.IMAGE): {
                "cam_front_left": ("input_embeddings", modality, "cam_front_left")
            },
            (modality := Modality.CONTINUOUS): {
                "speed": ("input_tokens", modality, "speed")
            },
        },
    ).to(device)


@pytest.fixture(scope="module")
def random_masked_hindsight_control_objective(
    encoder: Module, device: torch.device, embedding_dim: int, num_bins: NumBins
) -> RandomMaskedHindsightControlObjective:
    logit_bias = torch.tensor(0)

    return RandomMaskedHindsightControlObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": Linear(embedding_dim, num_bins.gas_pedal, bias=False),
                    "brake_pedal": Linear(
                        embedding_dim, num_bins.brake_pedal, bias=False
                    ),
                    "steering_angle": Linear(
                        embedding_dim, num_bins.steering_angle, bias=False
                    ),
                },
                Modality.DISCRETE: {
                    "turn_signal": Linear(embedding_dim, 3, bias=False)
                },
            }
        ),
        losses=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": LogitBiasCrossEntropyLoss(logit_bias=logit_bias),
                    "brake_pedal": LogitBiasCrossEntropyLoss(logit_bias=logit_bias),
                    "steering_angle": LogitBiasCrossEntropyLoss(logit_bias=logit_bias),
                },
                Modality.DISCRETE: {
                    "turn_signal": LogitBiasCrossEntropyLoss(logit_bias=logit_bias)
                },
            }
        ),
        targets={
            (modality := Modality.CONTINUOUS): {
                "gas_pedal": ("input_tokens", modality, "gas_pedal"),
                "brake_pedal": ("input_tokens", modality, "brake_pedal"),
                "steering_angle": ("input_tokens", modality, "steering_angle"),
            },
            (modality := Modality.DISCRETE): {
                "turn_signal": ("input_tokens", modality, "turn_signal")
            },
        },
    ).to(device)


@pytest.fixture(scope="module")
def memory_extraction_objective(
    encoder: Module, device: torch.device, embedding_dim: int, num_bins: NumBins
) -> MemoryExtractionObjective:
    logit_bias = torch.tensor(0)

    return MemoryExtractionObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal_diff": Linear(
                        embedding_dim, num_bins.gas_pedal, bias=False
                    ),
                    "brake_pedal_diff": Linear(
                        embedding_dim, num_bins.brake_pedal, bias=False
                    ),
                    "steering_angle_diff": Linear(
                        embedding_dim, num_bins.steering_angle, bias=False
                    ),
                }
            }
        ),
        losses=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal_diff": LogitBiasCrossEntropyLoss(logit_bias=logit_bias),
                    "brake_pedal_diff": LogitBiasCrossEntropyLoss(
                        logit_bias=logit_bias
                    ),
                    "steering_angle_diff": LogitBiasCrossEntropyLoss(
                        logit_bias=logit_bias
                    ),
                }
            }
        ),
        targets={
            (modality := Modality.CONTINUOUS): {
                "gas_pedal_diff": ("input_tokens", modality, "gas_pedal_diff"),
                "brake_pedal_diff": ("input_tokens", modality, "brake_pedal_diff"),
                "steering_angle_diff": (
                    "input_tokens",
                    modality,
                    "steering_angle_diff",
                ),
            }
        },
    ).to(device)


@pytest.fixture(scope="module")
def policy_objective(
    encoder: Module, device: torch.device, embedding_dim: int
) -> PolicyObjective:
    logit_bias = torch.tensor(0)

    return PolicyObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": MLP(3 * embedding_dim, [embedding_dim, 2], bias=False),
                    "brake_pedal": MLP(
                        3 * embedding_dim, [embedding_dim, 2], bias=False
                    ),
                    "steering_angle": MLP(
                        3 * embedding_dim, [embedding_dim, 2], bias=False
                    ),
                },
                Modality.DISCRETE: {
                    "turn_signal": MLP(
                        3 * embedding_dim, [embedding_dim, 3], bias=False
                    )
                },
            }
        ),
        losses=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": GaussianNLLLoss(),
                    "brake_pedal": GaussianNLLLoss(),
                    "steering_angle": GaussianNLLLoss(),
                },
                Modality.DISCRETE: {
                    "turn_signal": LogitBiasCrossEntropyLoss(logit_bias=logit_bias)
                },
            }
        ),
        targets={
            (modality := Modality.CONTINUOUS): {
                "gas_pedal": ("input", modality, "gas_pedal"),
                "brake_pedal": ("input", modality, "brake_pedal"),
                "steering_angle": ("input", modality, "steering_angle"),
            },
            (modality := Modality.DISCRETE): {
                "turn_signal": ("input", modality, "turn_signal")
            },
        },
    ).to(device)
