from collections.abc import Generator
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
import torch
from pytest_lazy_fixtures import lf
from torch import Tensor
from torch.nn import LayerNorm, Module
from torch.testing import assert_close
from torch.utils._pytree import (  # noqa: PLC2701
    keystr,
    tree_flatten_with_path,
    tree_map,
)
from torchvision.ops import MLP

from rmind.components.base import Modality, TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode, EpisodeBuilder
from rmind.components.loss import FlowMatchingLoss
from rmind.components.objectives.policy import PolicyObjective
from rmind.components.transformer import FlowActionDecoder
from rmind.models.control_transformer import ControlTransformer, PredictionConfig

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import EmbeddingDims


@pytest.fixture(autouse=True, scope="session")  # noqa: RUF076
def _soft_pending_unbacked() -> None:
    # tensordict's _device_recorder and _CONSTRUCTORS are global dicts mutated
    # during export tracing (dynamo side-effect). This creates a spurious
    # "pending unbacked symbol u0" error even though the exported graph is valid.
    # Demote to warning so torch.export tests are not falsely blocked.
    import logging  # noqa: PLC0415

    import torch.fx.experimental._config as _fx_config  # noqa: PLC0415, PLC2701

    _fx_config.soft_pending_unbacked_not_found_error = True  # ty:ignore[invalid-assignment]
    # ...then mute the now-harmless demoted warning (it's a logging record, not a warning).
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)


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
    assert_close(episode, episode_export)


@pytest.fixture
def _no_tf32_matmul() -> Generator[None, Any, None]:
    """Force fp32 matmul (no tf32) for the flag-invariance assertion below.

    The session-level conftest fixture sets matmul_precision="high", which lets
    PyTorch use tf32 for fp32 matmuls. tf32 dispatch differs under
    torch.compiler.is_exporting() — some kernels fall back to fp32 when the
    flag is set — producing ~2e-4 drift on the encoder output between the
    eager call (flag=False) and the export-flag call (flag=True). That drift
    is well within bf16 training noise and ORT verify tolerance, but breaks
    this test's strict bit-exact check. Forcing "highest" here keeps the test
    measuring what its name claims — that the *rmind* code paths are
    flag-invariant — independent of PyTorch's tf32 dispatch.
    """
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


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
@pytest.mark.usefixtures("_no_tf32_matmul")
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


@torch.inference_mode()
def test_torch_export_dynamic_shapes(
    control_transformer: ControlTransformer, batch_dict: TensorTree
) -> None:
    """Dynamic-shape export works with the tensordict#1003 patch applied."""
    from rmind.utils.tensordict_export_patch import apply  # noqa: PLC0415

    apply()
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


_FLOW_ACTION_TARGETS = {
    "gas_pedal": ("data", "meta/VehicleMotion/gas_pedal_normalized_target"),
    "brake_pedal": ("data", "meta/VehicleMotion/brake_pedal_normalized_target"),
    "steering_angle": ("data", "meta/VehicleMotion/steering_angle_normalized_target"),
}
_FLOW_CONDITION_TOKENS = (
    (Modality.SUMMARY, "observation_summary"),
    (Modality.SUMMARY, "observation_history"),
    (Modality.CONTEXT, "waypoints"),
)


@pytest.fixture
def flow_control_transformer(
    episode_builder: EpisodeBuilder,
    encoder: Module,
    device: torch.device,
    request: pytest.FixtureRequest,
) -> ControlTransformer:
    """A ControlTransformer whose sole objective is the flow PolicyObjective,
    wired to the shared episode builder + encoder (dims must line up)."""
    embedding_dims: EmbeddingDims = request.getfixturevalue("embedding_dims")
    objective = PolicyObjective(
        history_steps=1,
        targets=_FLOW_ACTION_TARGETS,
        condition_tokens=_FLOW_CONDITION_TOKENS,
        loss=FlowMatchingLoss(),
        decoder=FlowActionDecoder(
            condition_dim=embedding_dims.encoder,
            dim_model=32,
            action_dim=3,
            action_horizon=6,
            flow_sampling_steps=2,
            num_layers=1,
            num_heads=2,
            attn_dropout=0.0,
            resid_dropout=0.0,
            mlp_dropout=0.0,
        ),
    )
    return ControlTransformer(
        episode_builder=episode_builder,
        encoder=encoder,
        objectives=ModuleDict({"policy": objective}),
        prediction_config=PredictionConfig(),
    ).to(device)


@torch.inference_mode()
def test_exportable_control_policy_full_pipeline(
    flow_control_transformer: ControlTransformer, batch_dict: TensorTree
) -> None:
    """The edge-deployment graph — raw sensor batch (+ noise) -> action draws,
    with episode builder + encoder + flow tail in one module — torch.exports
    cleanly, keeps the winner-take-all readout out of the graph, and the
    exported program matches eager."""
    exportable = flow_control_transformer.eval().exportable_policy().eval()

    b = batch_dict["data"]["meta/VehicleMotion/speed"].shape[0]
    k = 4
    noise = torch.randn(
        b, k, 6, 3, device=batch_dict["data"]["waypoints/xy_normalized"].device
    )

    eager = exportable(batch_dict, noise)  # also warms the attention-mask cache
    assert eager.shape == (b, k, 6, 3)

    program = torch.export.export(exportable, (batch_dict, noise), strict=True)

    # the readout's data-dependent ops must stay host-side, not leak into the graph
    targets = {str(n.target) for n in program.graph.nodes if n.op == "call_function"}
    assert not [t for t in targets if "sort" in t or "nonzero" in t], (
        "winner-take-all readout leaked into the export graph"
    )

    exported = program.module()(batch_dict, noise)
    assert_close(exported, eager, rtol=1e-2, atol=1e-3)


@torch.inference_mode()
def test_exportable_control_policy_onnx(
    flow_control_transformer: ControlTransformer, batch_dict: TensorTree
) -> None:
    """The full policy pipeline lowers to ONNX and ORT matches PyTorch eager —
    the exact artifact `flow_export.py --full` produces for the edge device."""
    exportable = flow_control_transformer.eval().exportable_policy().eval()

    b = batch_dict["data"]["meta/VehicleMotion/speed"].shape[0]
    noise = torch.randn(
        b, 4, 6, 3, device=batch_dict["data"]["waypoints/xy_normalized"].device
    )

    eager = exportable(batch_dict, noise)  # warms the attention-mask cache

    exported_program = torch.export.export(exportable, (batch_dict, noise), strict=True)
    onnx_program = torch.onnx.export(
        model=exported_program,
        dynamo=True,
        optimize=True,
        verify=False,
        output_names=["action_draws"],
    )
    assert onnx_program is not None

    (onnx_draws,) = onnx_program(batch_dict, noise)
    assert_close(
        torch.as_tensor(onnx_draws).float(), eager.cpu().float(), rtol=1e-2, atol=1e-3
    )


@torch.inference_mode()
def test_flow_export_compare_matches_saved_onnx(
    flow_control_transformer: ControlTransformer,
    batch_dict: TensorTree,
    tmp_path: "Path",
) -> None:
    """The comparator's saved-file round-trip: export to a .onnx on disk, reload
    via ONNX Runtime, feed identical inputs through the positional mapping used by
    flow_export_compare, and confirm ORT matches eager. Guards both the on-disk
    artifact and the input-name/order contract the edge host relies on."""
    import onnxruntime as ort  # noqa: PLC0415

    from rmind.scripts.flow_export_compare import build_feed  # noqa: PLC0415

    exportable = flow_control_transformer.eval().exportable_policy().eval()

    b = batch_dict["data"]["meta/VehicleMotion/speed"].shape[0]
    noise = torch.randn(
        b, 4, 6, 3, device=batch_dict["data"]["waypoints/xy_normalized"].device
    )
    example_args = (batch_dict, noise)

    eager = exportable(*example_args)  # warms the attention-mask cache

    exported_program = torch.export.export(exportable, example_args, strict=True)
    onnx_path = tmp_path / "policy_full.onnx"
    torch.onnx.export(
        model=exported_program,
        f=str(onnx_path),
        dynamo=True,
        optimize=False,
        verify=False,
        output_names=["action_draws"],
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_draws,) = session.run(None, build_feed(session, example_args))
    assert_close(
        torch.as_tensor(onnx_draws).float(), eager.cpu().float(), rtol=1e-2, atol=1e-3
    )
