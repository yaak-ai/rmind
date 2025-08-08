from dataclasses import asdict
from typing import Any

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
from rmind.components.episode import Episode, EpisodeBuilder, EpisodeExport, Modality
from rmind.components.mask import TorchAttentionMaskLegend
from rmind.components.objectives.policy import PolicyObjective
from rmind.models.control_transformer import ControlTransformer


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
def policy_mask(episode: Episode) -> Tensor:
    return PolicyObjective.build_attention_mask(
        episode.index, episode.timestep, legend=TorchAttentionMaskLegend
    ).mask


@pytest.fixture
def policy_objective(encoder: Module, policy_mask: Tensor) -> PolicyObjective:
    return PolicyObjective(
        encoder=encoder,
        mask=policy_mask,
        heads=ModuleDict(
            modules={
                Modality.CONTINUOUS: {
                    "gas_pedal": MLP(1536, [512, 2], bias=False),
                    "brake_pedal": MLP(1536, [512, 2], bias=False),
                    "steering_angle": MLP(1536, [512, 2], bias=False),
                },
                Modality.DISCRETE: {"turn_signal": MLP(1536, [512, 3], bias=False)},
            }
        ),
    )


@pytest.fixture
def objectives(policy_objective: Module) -> ModuleDict:
    return ModuleDict({"policy": policy_objective})


@pytest.fixture
def control_transformer(
    episode_builder: Module, objectives: ModuleDict
) -> ControlTransformer:
    return ControlTransformer(episode_builder=episode_builder, objectives=objectives)


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
        if isinstance(actual, (int, float)):
            actual = torch.tensor(actual, dtype=expected.dtype, device=expected.device)

        assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
            check_dtype=True,
            msg=lambda msg, kp=kp: f"{msg}\nkeypath: {keystr(kp)}",
        )


@pytest.mark.parametrize(
    ("module", "args", "args_export"),
    [
        (lf("episode_builder"), (lf("batch_dict"),), (lf("batch_dict"),)),
        (lf("policy_objective"), (lf("episode"),), (lf("episode_export"),)),
        (lf("control_transformer"), (lf("batch_dict"),), (lf("batch_dict"),)),
    ],
)
@torch.inference_mode()
def test_module_export_aoti(
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
        match actual := key_get(module_export_output, kp):
            case int() | float():
                actual = torch.tensor(actual)

            case _:
                pass

        assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
            check_dtype=True,
            msg=lambda msg, kp=kp: f"{msg}\nkeypath: {keystr(kp)}",
        )

    exported = torch.export.export(module, args=args_export, strict=True)
    exported_output = exported.module()(*args_export)

    for kp, expected in module_output_items:
        actual = key_get(exported_output, kp)
        if isinstance(actual, (int, float)):
            actual = torch.tensor(actual, dtype=expected.dtype, device=expected.device)

        assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
            check_dtype=True,
            msg=lambda msg, kp=kp: f"{msg}\nkeypath: {keystr(kp)}",
        )

    package_path = torch._inductor.aoti_compile_and_package(exported)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    package = torch._inductor.aoti_load_package(package_path)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    package_output = package(*args_export)

    for kp, expected in module_output_items:
        actual = key_get(package_output, kp)
        if isinstance(actual, (int, float)):
            actual = torch.tensor(actual, dtype=expected.dtype, device=expected.device)

        assert_close(
            actual,
            expected,
            equal_nan=True,
            check_dtype=True,
            msg=lambda msg, kp=kp: f"{msg}\nkeypath: {keystr(kp)}",
        )
