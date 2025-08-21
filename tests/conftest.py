from collections.abc import Generator
from typing import Any

import pytest
import torch
from einops.layers.torch import Rearrange
from rbyte.batch import Batch
from tensordict import TensorDict
from torch.nn import Linear, Module, MSELoss
from torch.testing import make_tensor
from torchvision.models import resnet18
from torchvision.ops import MLP
from torchvision.transforms.v2 import CenterCrop, Normalize, ToDtype

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
from rmind.components.norm import Normalize as _Normalize
from rmind.components.objectives import (
    ForwardDynamicsPredictionObjective,
    InverseDynamicsPredictionObjective,
    MemoryExtractionObjective,
    PolicyObjective,
    RandomMaskedHindsightControlObjective,
)
from rmind.components.resnet import ResnetBackbone

EMBEDDING_DIM = 512


@pytest.fixture(scope="session", autouse=True)
def _set_float32_matmul_precision() -> Generator[None, Any, None]:  # pyright: ignore[reportUnusedFunction]
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(prev)


@pytest.fixture
def device() -> torch.device:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return torch.device(device)


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
                "waypoints/xy_normalized": make_tensor(
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
def batch_dict(batch: Batch) -> TensorTree:
    return batch.to_dict(retain_none=False)


@pytest.fixture
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


@pytest.fixture
def episode_builder(tokenizers: ModuleDict) -> Module:
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


@pytest.fixture
def episode(episode_builder: EpisodeBuilder, batch_dict: TensorTree) -> Episode:
    return episode_builder(batch_dict)


@pytest.fixture
def encoder() -> Module:
    return TransformerEncoder(
        dim_model=EMBEDDING_DIM,
        num_heads=1,
        num_layers=1,
        attn_dropout=0.1,
        resid_dropout=0.1,
        mlp_dropout=0.1,
        hidden_layer_multiplier=1,
    )


@pytest.fixture
def inverse_dynamics_prediction_objective(
    encoder: Module,
) -> InverseDynamicsPredictionObjective:
    logit_bias = torch.tensor(0)

    return InverseDynamicsPredictionObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM, bias=False),
                    "brake_pedal": Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM, bias=False),
                    "steering_angle": Linear(
                        2 * EMBEDDING_DIM, EMBEDDING_DIM, bias=False
                    ),
                },
                Modality.DISCRETE: {
                    "turn_signal": Linear(2 * EMBEDDING_DIM, 3, bias=False)
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
    )


@pytest.fixture
def forward_dynamics_prediction_objective(
    encoder: Module,
) -> ForwardDynamicsPredictionObjective:
    logit_bias = torch.tensor(0)

    return ForwardDynamicsPredictionObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.IMAGE: {
                    "cam_front_left": Linear(
                        3 * EMBEDDING_DIM, EMBEDDING_DIM, bias=False
                    )
                },
                Modality.CONTINUOUS: {
                    "speed": Linear(3 * EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
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
    )


@pytest.fixture
def random_masked_hindsight_control_objective(
    encoder: Module,
) -> RandomMaskedHindsightControlObjective:
    logit_bias = torch.tensor(0)

    return RandomMaskedHindsightControlObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False),
                    "brake_pedal": Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False),
                    "steering_angle": Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False),
                },
                Modality.DISCRETE: {
                    "turn_signal": Linear(EMBEDDING_DIM, 3, bias=False)
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
    )


@pytest.fixture
def memory_extraction_objective(encoder: Module) -> MemoryExtractionObjective:
    logit_bias = torch.tensor(0)

    return MemoryExtractionObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal_diff": Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False),
                    "brake_pedal_diff": Linear(
                        EMBEDDING_DIM, EMBEDDING_DIM, bias=False
                    ),
                    "steering_angle_diff": Linear(
                        EMBEDDING_DIM, EMBEDDING_DIM, bias=False
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
    )


@pytest.fixture
def policy_objective(encoder: Module) -> PolicyObjective:
    logit_bias = torch.tensor(0)

    return PolicyObjective(
        encoder=encoder,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": MLP(3 * EMBEDDING_DIM, [EMBEDDING_DIM, 2], bias=False),
                    "brake_pedal": MLP(
                        3 * EMBEDDING_DIM, [EMBEDDING_DIM, 2], bias=False
                    ),
                    "steering_angle": MLP(
                        3 * EMBEDDING_DIM, [EMBEDDING_DIM, 2], bias=False
                    ),
                },
                Modality.DISCRETE: {
                    "turn_signal": MLP(
                        3 * EMBEDDING_DIM, [EMBEDDING_DIM, 3], bias=False
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
    )
