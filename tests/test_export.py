from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
import torch
from pytest_lazy_fixtures import lf
from torch import Tensor
from torch.nn import LayerNorm, Module
from torch.testing import assert_close
from torch.utils._pytree import (  # noqa: PLC2701
    key_get,
    keystr,
    tree_flatten_with_path,
    tree_map,
)
from torchvision.ops import MLP

from rmind.components.base import Modality, TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode, EpisodeBuilder
from rmind.components.objectives.policy import PolicyObjective
from rmind.models.control_transformer import ControlTransformer, PredictionConfig

if TYPE_CHECKING:
    from tests.conftest import EmbeddingDims


@pytest.fixture
def episode(episode_builder: EpisodeBuilder, batch_dict: TensorTree) -> Episode:
    with torch.inference_mode():
        return episode_builder(batch_dict)


@pytest.fixture
def episode_export(
    episode_builder: EpisodeBuilder,
    batch_dict: TensorTree,
    monkeypatch: pytest.MonkeyPatch,
) -> Episode:
    with torch.inference_mode(), monkeypatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        return episode_builder(batch_dict)


@pytest.fixture
def policy_objective(
    device: torch.device, request: pytest.FixtureRequest
) -> PolicyObjective:
    embedding_dims: EmbeddingDims = request.getfixturevalue("embedding_dims")

    return PolicyObjective(
        norm=LayerNorm(embedding_dims.encoder),
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": MLP(
                        3 * embedding_dims.encoder,
                        [embedding_dims.encoder, 2],
                        bias=False,
                    ),
                    "brake_pedal": MLP(
                        3 * embedding_dims.encoder,
                        [embedding_dims.encoder, 2],
                        bias=False,
                    ),
                    "steering_angle": MLP(
                        3 * embedding_dims.encoder,
                        [embedding_dims.encoder, 2],
                        bias=False,
                    ),
                },
                Modality.DISCRETE: {
                    "turn_signal": MLP(
                        3 * embedding_dims.encoder,
                        [embedding_dims.encoder, 3],
                        bias=False,
                    )
                },
            }
        ),
    ).to(device)


@pytest.fixture
def encoder_eval(encoder: Module) -> Module:
    return encoder.eval()


@pytest.fixture
def policy_embedding(encoder_eval: Module, episode: Episode) -> Tensor:
    return encoder_eval(src=episode.embeddings_flattened, mask=episode.attention_mask)


@pytest.fixture
def policy_embedding_export(encoder_eval: Module, episode_export: Episode) -> Tensor:
    return encoder_eval(
        src=episode_export.embeddings_flattened, mask=episode_export.attention_mask
    )


@pytest.fixture
def objectives(policy_objective: Module, device: torch.device) -> ModuleDict:
    return ModuleDict({"policy": policy_objective}).to(device)


@pytest.fixture
def control_transformer(
    episode_builder: Module,
    objectives: ModuleDict,
    encoder: Module,
    device: torch.device,
) -> ControlTransformer:
    return ControlTransformer(
        episode_builder=episode_builder,
        encoder=encoder,
        objectives=objectives,
        prediction_config=PredictionConfig(),
    ).to(device)


def test_episode(episode: Episode, episode_export: Episode) -> None:
    def ep_dict(src: Episode) -> dict:
        return src.to_dict() | {
            "embeddings": src.embeddings.to_dict(),
            "embeddings_flattened": src.embeddings_flattened,
        }

    episode_dict = ep_dict(episode)
    episode_export_dict = ep_dict(episode_export)

    for kp, expected in tree_flatten_with_path(episode_dict)[0]:
        actual = key_get(episode_export_dict, kp)
        match expected, actual:
            case (Tensor(shape=[]), int() | float()):
                expected = expected.item()  # noqa: PLW2901

            case _:
                pass

        assert_close(
            actual,
            expected=expected,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
            msg=lambda msg, kp=kp: f"{msg}\nkeypath: {keystr(kp)}",
        )


@pytest.mark.parametrize(
    ("module", "args", "args_export"),
    [
        (lf("episode_builder"), (lf("batch_dict"),), (lf("batch_dict"),)),
        (
            lf("policy_objective"),
            (lf("episode"), lf("policy_embedding")),
            (lf("episode_export"), lf("policy_embedding_export")),
        ),
        (lf("control_transformer"), (lf("batch_dict"),), (lf("batch_dict"),)),
    ],
    ids=["episode_builder", "policy_objective", "control_transformer"],
)
@torch.inference_mode()
def test_torch_export_fake(
    module: Module,
    args: tuple[Any],
    args_export: tuple[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = module.eval()

    module_output = module(*args)
    module_output_items, _ = tree_flatten_with_path(module_output)

    with monkeypatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        # so we can breakpoint() inside export code paths
        module_export_output = module(*args_export)

    module_export_output_items, _ = tree_flatten_with_path(module_export_output)

    for (kp, expected), (_, actual) in zip(
        module_output_items, module_export_output_items, strict=True
    ):
        if not isinstance(expected, Tensor):
            continue

        match expected, actual:
            case (Tensor(shape=[]), int() | float()):
                expected = expected.item()  # noqa: PLW2901

            case _:
                pass

        assert_close(
            actual,
            expected,
            equal_nan=True,
            msg=lambda msg, kp=kp: f"{msg}\nkeypath: {keystr(kp)}",
        )


@torch.inference_mode()
def test_episode_builder_warms_attention_mask_cache(
    episode_builder: EpisodeBuilder,
    batch_dict: TensorTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = episode_builder(batch_dict)

    assert episode_builder.attention_mask_cache_is_warm

    with monkeypatch.context() as m:
        m.setattr(
            episode_builder.attention_mask_builder,
            "forward",
            Mock(side_effect=AssertionError("cache was not used during export")),
        )
        m.setattr("torch.compiler._is_exporting_flag", True)
        _ = episode_builder(batch_dict)


@pytest.mark.parametrize(
    ("module", "args"),
    [
        (lf("episode_builder"), (lf("batch_dict"),)),
        (lf("policy_objective"), (lf("episode_export"), lf("policy_embedding_export"))),
        (lf("control_transformer"), (lf("batch_dict"),)),
    ],
    ids=["episode_builder", "policy_objective", "control_transformer"],
)
@torch.inference_mode()
def test_torch_export(module: Module, args: tuple[Any]) -> None:
    module = module.eval()
    module(*args)  # warm internal caches (e.g. attention mask) before export
    torch.export.export(module, args=args, strict=True)


@pytest.mark.xfail(
    reason="TensorDict batch_size rejects SymInt in export trace — pytorch/tensordict#1003",
    strict=True,
)
@torch.inference_mode()
def test_torch_export_dynamic_shapes(
    control_transformer: ControlTransformer, batch_dict: TensorTree
) -> None:
    module = control_transformer.eval()
    module(batch_dict)  # warm caches
    batch_dim = torch.export.Dim("batch", min=1)
    dynamic_shapes = (tree_map(lambda _: {0: batch_dim}, batch_dict),)
    torch.export.export(
        module, args=(batch_dict,), dynamic_shapes=dynamic_shapes, strict=True
    )


@pytest.mark.parametrize(
    ("module", "args"),
    [(lf("control_transformer"), (lf("batch_dict"),))],
    ids=["control_transformer"],
)
@torch.inference_mode()
def test_onnx_export(module: Module, args: tuple[Any]) -> None:
    module = module.eval()
    exported_program = torch.export.export(module, args=args, strict=True)
    program = torch.onnx.export(
        model=exported_program,
        external_data=False,
        dynamo=True,
        optimize=True,
        verify=True,
    )

    assert program is not None


@pytest.mark.parametrize(
    ("module", "args"),
    [(lf("control_transformer"), (lf("batch_dict"),))],
    ids=["control_transformer"],
)
@torch.inference_mode()
def test_onnx_inference(module: Module, args: tuple[Any]) -> None:
    """ORT inference on the exported ONNX model must match PyTorch eager numerically."""
    module = module.eval()

    eager_output = module(*args)
    eager_leaves, _ = tree_flatten_with_path(eager_output)

    exported_program = torch.export.export(module, args=args, strict=True)
    onnx_program = torch.onnx.export(
        model=exported_program,
        external_data=False,
        dynamo=True,
        optimize=True,
        verify=False,
    )
    assert onnx_program is not None

    onnx_output = onnx_program(*args)
    onnx_leaves, _ = tree_flatten_with_path(onnx_output)

    for (kp, expected), (_, actual) in zip(eager_leaves, onnx_leaves, strict=True):
        assert_close(
            torch.as_tensor(actual).float(),
            expected.cpu().float(),
            rtol=1e-2,
            atol=1e-3,
            msg=lambda msg, kp=kp: f"{msg}\nkeypath: {keystr(kp)}",
        )
