from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest
import torch
from pytest_lazy_fixtures import lf
from torch import Tensor
from torch.nn import Module
from torch.testing import assert_close
from torch.utils._pytree import key_get, keystr, tree_flatten_with_path  # noqa: PLC2701
from torchvision.ops import MLP

from rmind.components.base import TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode, EpisodeBuilder, EpisodeExport
from rmind.components.objectives.policy import PolicyObjective
from rmind.components.tokens import Modality
from rmind.models.control_transformer import ControlTransformer

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
) -> EpisodeExport:
    with torch.inference_mode(), monkeypatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        return episode_builder(batch_dict)


@pytest.fixture
def policy_objective(
    device: torch.device, request: pytest.FixtureRequest
) -> PolicyObjective:
    embedding_dims: EmbeddingDims = request.getfixturevalue("embedding_dims")

    return PolicyObjective(
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
        )
    ).to(device)


@pytest.fixture
def encoder_eval(encoder: Module) -> Module:
    return encoder.eval()


@pytest.fixture
def policy_embedding(encoder_eval: Module, episode: Episode) -> Tensor:
    return encoder_eval(src=episode.embeddings_packed, mask=episode.attention_mask)


@pytest.fixture
def policy_embedding_export(
    encoder_eval: Module, episode_export: EpisodeExport
) -> Tensor:
    return encoder_eval(
        src=episode_export.embeddings_packed, mask=episode_export.attention_mask
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
        episode_builder=episode_builder, encoder=encoder, objectives=objectives
    ).to(device)


def test_episode(episode: Episode, episode_export: EpisodeExport) -> None:
    episode_dict = (src := episode).to_dict() | {
        "embeddings": src.embeddings.to_dict(),
        "embeddings_packed": src.embeddings_packed,
    }
    episode_export_dict = asdict(src := episode_export) | {
        "embeddings": src.embeddings,
        "embeddings_packed": src.embeddings_packed,
    }

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

    for kp, expected in module_output_items:
        actual = key_get(module_export_output, kp)
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
    torch.export.export(module.eval(), args=args, strict=True)


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
