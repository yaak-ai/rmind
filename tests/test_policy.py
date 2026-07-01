from typing import TYPE_CHECKING, Any, cast

import jq  # ty:ignore[unresolved-import]
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.testing import assert_close, make_tensor

from rmind.components.action_transform import (
    MERGE_MODEL_KEYS,
    ActionTransform,
    GasBrakeMerge,
    Gaussianize,
)
from rmind.components.base import Modality
from rmind.components.episode import Episode
from rmind.components.loss import FlowMatchingLoss
from rmind.components.objectives import PolicyObjective
from rmind.components.objectives.base import ObjectivePredictionKey, Prediction
from rmind.components.objectives.policy import ExportablePolicy, FlowTimeSampling
from rmind.components.optimizers import SelectiveAdamW
from rmind.components.readout import AxisModeReadout, SingleReadout
from rmind.components.transformer import FlowActionDecoder, TransformerEncoder
from rmind.components.transformer.decoder import FlowSamplingMethod

if TYPE_CHECKING:
    from tests.conftest import EmbeddingDims

CONFIG_PATH = "../config"
FLOW_TIME_EMBEDDING_SCALE = 1000.0
FLOW_TIME_LOGIT_SCALE = 0.25
ACTION_TARGETS = {
    "gas_pedal": ("data", "meta/VehicleMotion/gas_pedal_normalized_target"),
    "brake_pedal": ("data", "meta/VehicleMotion/brake_pedal_normalized_target"),
    "steering_angle": ("data", "meta/VehicleMotion/steering_angle_normalized_target"),
}
CONDITION_TOKENS = (
    (Modality.SUMMARY, "observation_summary"),
    (Modality.SUMMARY, "observation_history"),
    (Modality.CONTEXT, "waypoints"),
)


def _flow_policy_loss() -> FlowMatchingLoss:
    return FlowMatchingLoss()


def _flow_policy_batch(
    *, device: torch.device, batch_size: int = 2, action_horizon: int = 6
) -> dict[str, dict[str, torch.Tensor]]:
    target = torch.linspace(
        0.0, 1.0, steps=batch_size * action_horizon, dtype=torch.float32, device=device
    ).reshape(batch_size, action_horizon)
    return {
        "data": {
            "meta/VehicleMotion/gas_pedal_normalized_target": target,
            "meta/VehicleMotion/brake_pedal_normalized_target": target.flip(dims=(1,)),
            "meta/VehicleMotion/steering_angle_normalized_target": target * 2.0 - 1.0,
        }
    }


@pytest.mark.parametrize("action_horizon", [6, 12])
@pytest.mark.parametrize("sampling_method", ["euler", "heun"])
def test_flow_action_decoder_shape(
    device: torch.device, action_horizon: int, sampling_method: FlowSamplingMethod
) -> None:
    batch_size = 2
    condition_dim = 16
    decoder = FlowActionDecoder(
        condition_dim=condition_dim,
        dim_model=condition_dim,
        action_horizon=action_horizon,
        num_layers=1,
        num_heads=2,
        flow_sampling_steps=2,
        flow_sampling_method=sampling_method,
    ).to(device)
    condition_tokens = make_tensor(
        (batch_size, 2, condition_dim),
        dtype=torch.float32,
        device=device,
        low=-1.0,
        high=1.0,
    )
    noised_actions = make_tensor(
        (batch_size, action_horizon, 3),
        dtype=torch.float32,
        device=device,
        low=-1.0,
        high=1.0,
    )
    flow_time = make_tensor(
        (batch_size,), dtype=torch.float32, device=device, low=0.0, high=1.0
    )

    action_flow = decoder(
        condition_tokens=condition_tokens,
        noised_actions=noised_actions,
        flow_time=flow_time,
    )
    sample = decoder.sample(condition_tokens=condition_tokens, steps=2)

    assert action_flow.shape == noised_actions.shape
    assert sample.shape == noised_actions.shape


def test_flow_action_decoder_rejects_invalid_sample_steps() -> None:
    decoder = FlowActionDecoder(
        condition_dim=16, dim_model=16, action_horizon=6, num_layers=1, num_heads=2
    )
    condition_tokens = torch.zeros(1, 2, 16)

    with pytest.raises(ValueError, match="steps must be positive"):
        decoder.sample(condition_tokens=condition_tokens, steps=0)


def test_flow_action_decoder_supports_min_even_time_embedding_dim() -> None:
    decoder = FlowActionDecoder(
        condition_dim=2, dim_model=2, action_horizon=2, num_layers=1, num_heads=1
    )

    action_flow = decoder(
        condition_tokens=torch.zeros(1, 2, 2),
        noised_actions=torch.zeros(1, 2, 3),
        flow_time=torch.tensor([0.5]),
    )

    assert action_flow.shape == (1, 2, 3)


def test_flow_action_decoder_selective_adamw_compatible() -> None:
    decoder = FlowActionDecoder(
        condition_dim=16, dim_model=16, action_horizon=6, num_layers=1, num_heads=2
    )

    optimizer = SelectiveAdamW(
        module=decoder,
        lr=1e-4,
        weight_decay_module_blacklist=(torch.nn.Embedding, torch.nn.LayerNorm),
    )

    assert optimizer.param_groups


def test_flow_policy_loss_has_decoder_gradients(
    device: torch.device,
    embedding_dims: "EmbeddingDims",
    episode: Episode,
    encoder: TransformerEncoder,
) -> None:
    objective = PolicyObjective(
        history_steps=1,
        targets=ACTION_TARGETS,
        condition_tokens=CONDITION_TOKENS,
        loss=_flow_policy_loss(),
        decoder=FlowActionDecoder(
            condition_dim=embedding_dims.encoder,
            dim_model=32,
            action_horizon=6,
            num_layers=1,
            num_heads=2,
            flow_sampling_steps=2,
        ),
    ).to(device)
    embedding = encoder(src=episode.embeddings_flattened, mask=episode.attention_mask)
    embedding = embedding.detach()
    batch = _flow_policy_batch(device=device)

    loss = objective.compute_metrics(episode=episode, embedding=embedding, batch=batch)[
        "loss"
    ]
    assert isinstance(loss, torch.Tensor)
    loss.backward()

    decoder_grads = [
        param.grad
        for param in objective.decoder.parameters()
        if param.requires_grad and param.grad is not None
    ]

    assert torch.isfinite(loss)
    assert decoder_grads
    assert all(torch.isfinite(grad).all() for grad in decoder_grads)
    assert any(grad.abs().sum() > 0 for grad in decoder_grads)


@torch.inference_mode()
def test_flow_policy_eval_metrics_are_deterministic(
    device: torch.device,
    embedding_dims: "EmbeddingDims",
    episode: Episode,
    encoder: TransformerEncoder,
) -> None:
    objective = PolicyObjective(
        history_steps=1,
        targets=ACTION_TARGETS,
        condition_tokens=CONDITION_TOKENS,
        flow_time_sampling="logit-normal",
        loss=_flow_policy_loss(),
        decoder=FlowActionDecoder(
            condition_dim=embedding_dims.encoder,
            dim_model=32,
            action_horizon=6,
            num_layers=1,
            num_heads=2,
            flow_sampling_steps=2,
        ),
    ).to(device)
    embedding = encoder(src=episode.embeddings_flattened, mask=episode.attention_mask)
    batch = _flow_policy_batch(device=device)

    objective.eval()
    first = objective.compute_metrics(episode=episode, embedding=embedding, batch=batch)
    second = objective.compute_metrics(
        episode=episode, embedding=embedding, batch=batch
    )

    assert "sample_l1" in first
    assert isinstance(first["sample_l1"], torch.Tensor)
    assert isinstance(second["sample_l1"], torch.Tensor)
    assert isinstance(first["loss"], torch.Tensor)
    assert isinstance(second["loss"], torch.Tensor)
    assert_close(first["loss"], second["loss"])
    assert_close(first["sample_l1"], second["sample_l1"])

    objective.train()
    train_metrics = objective.compute_metrics(
        episode=episode, embedding=embedding, batch=batch
    )

    assert "sample_l1" not in train_metrics


@pytest.mark.parametrize("flow_time_sampling", ["uniform", "logit-normal"])
def test_flow_policy_samples_configured_flow_time(
    device: torch.device, flow_time_sampling: FlowTimeSampling
) -> None:
    objective = PolicyObjective(
        history_steps=1,
        targets=ACTION_TARGETS,
        condition_tokens=CONDITION_TOKENS,
        flow_time_sampling=flow_time_sampling,
        loss=_flow_policy_loss(),
        decoder=FlowActionDecoder(
            condition_dim=16, dim_model=16, action_horizon=2, num_layers=1, num_heads=2
        ),
    ).to(device)

    flow_time = objective.sample_flow_time(
        flow_time_sampling, 32, dtype=torch.float32, device=device
    )

    assert flow_time.shape == (32,)
    assert torch.isfinite(flow_time).all()
    assert (flow_time >= 0.0).all()
    assert (flow_time <= 1.0).all()


def test_flow_policy_time_sampling_respects_generator(device: torch.device) -> None:
    # The fixed-seed validation generator must make every sampler reproducible.
    for method in ("uniform", "logit-normal"):
        gen_a = torch.Generator(device=device).manual_seed(0)
        gen_b = torch.Generator(device=device).manual_seed(0)
        a = PolicyObjective.sample_flow_time(
            method, 64, dtype=torch.float32, device=device, generator=gen_a
        )
        b = PolicyObjective.sample_flow_time(
            method, 64, dtype=torch.float32, device=device, generator=gen_b
        )
        assert torch.equal(a, b)


def test_flow_policy_rejects_invalid_flow_time_sampling() -> None:
    with pytest.raises(ValueError, match="flow_time_sampling"):
        PolicyObjective(
            history_steps=1,
            targets=ACTION_TARGETS,
            condition_tokens=CONDITION_TOKENS,
            # intentionally invalid: exercises @validate_call's runtime rejection
            flow_time_sampling="invalid",  # ty: ignore[invalid-argument-type]
            loss=_flow_policy_loss(),
            decoder=FlowActionDecoder(
                condition_dim=16,
                dim_model=16,
                action_horizon=2,
                num_layers=1,
                num_heads=2,
            ),
        )


@torch.inference_mode()
def test_flow_policy_predicts_trajectory_values(
    device: torch.device,
    embedding_dims: "EmbeddingDims",
    episode: Episode,
    encoder: TransformerEncoder,
) -> None:
    objective = PolicyObjective(
        history_steps=1,
        targets=ACTION_TARGETS,
        condition_tokens=CONDITION_TOKENS,
        loss=_flow_policy_loss(),
        decoder=FlowActionDecoder(
            condition_dim=embedding_dims.encoder,
            dim_model=32,
            action_horizon=6,
            num_layers=1,
            num_heads=2,
            flow_sampling_steps=2,
        ),
    ).to(device)
    embedding = encoder(src=episode.embeddings_flattened, mask=episode.attention_mask)
    batch = _flow_policy_batch(device=device)

    predictions = objective.predict(
        episode=episode,
        embedding=embedding,
        batch=batch,
        keys={
            ObjectivePredictionKey.GROUND_TRUTH,
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.PREDICTION_PROBS,
            ObjectivePredictionKey.SCORE_LOGPROB,
            ObjectivePredictionKey.SCORE_L1,
        },
    )

    assert set(predictions.keys()) == {
        ObjectivePredictionKey.GROUND_TRUTH,
        ObjectivePredictionKey.PREDICTION_VALUE,
        ObjectivePredictionKey.SCORE_L1,
    }

    for key in list(predictions.keys()):
        prediction = cast("Prediction", predictions[key])
        assert torch.equal(prediction.time_index, torch.arange(1, 7).expand(2, -1))
        assert isinstance(prediction.value, TensorDict)
        continuous_prediction = cast(
            "TensorDict", prediction.value[Modality.CONTINUOUS]
        )
        assert set(continuous_prediction.keys()) == {
            "gas_pedal",
            "brake_pedal",
            "steering_angle",
        }
        for action_key in ACTION_TARGETS:
            value = cast("torch.Tensor", continuous_prediction[action_key])
            assert value.shape == (2, 6)

    score_l1 = cast("Prediction", predictions[ObjectivePredictionKey.SCORE_L1]).value
    for value in score_l1[Modality.CONTINUOUS].values():
        assert (value >= 0).all()

    ground_truth = cast(
        "Prediction", predictions[ObjectivePredictionKey.GROUND_TRUTH]
    ).value
    assert_close(
        ground_truth[Modality.CONTINUOUS, "gas_pedal"],
        batch["data"]["meta/VehicleMotion/gas_pedal_normalized_target"],
    )


@torch.inference_mode()
def test_flow_policy_mode_readout_predicts_trajectory(
    device: torch.device,
    embedding_dims: "EmbeddingDims",
    episode: Episode,
    encoder: TransformerEncoder,
) -> None:
    # Mode readout: the policy draws K candidates, the injected readout collapses
    # them by consensus (steering = last channel, mode_axis=-1). Output shape must
    # match the single-draw path.
    objective = PolicyObjective(
        history_steps=1,
        targets=ACTION_TARGETS,
        condition_tokens=CONDITION_TOKENS,
        loss=_flow_policy_loss(),
        readout=AxisModeReadout(num_samples=4),
        decoder=FlowActionDecoder(
            condition_dim=embedding_dims.encoder,
            dim_model=32,
            action_horizon=6,
            num_layers=1,
            num_heads=2,
            flow_sampling_steps=2,
        ),
    ).to(device)
    embedding = encoder(src=episode.embeddings_flattened, mask=episode.attention_mask)
    batch = _flow_policy_batch(device=device)

    predictions = objective.predict(
        episode=episode,
        embedding=embedding,
        batch=batch,
        keys={ObjectivePredictionKey.PREDICTION_VALUE},
    )

    value = cast(
        "Prediction", predictions[ObjectivePredictionKey.PREDICTION_VALUE]
    ).value
    continuous = cast("TensorDict", value[Modality.CONTINUOUS])
    assert set(continuous.keys()) == {"gas_pedal", "brake_pedal", "steering_angle"}
    for action_key in ACTION_TARGETS:
        assert cast("torch.Tensor", continuous[action_key]).shape == (2, 6)


@pytest.mark.parametrize(("action_horizon", "sequence_length"), [(6, 12), (12, 18)])
def test_flow_policy_config_instantiates(
    action_horizon: int, sequence_length: int
) -> None:
    history_steps = 6
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "model/yaak/control_transformer/policy_finetune",
            overrides=[
                "+encoder_embedding_dim=384",
                f"+history_steps={history_steps}",
                f"+action_horizon={action_horizon}",
                f"+sequence_length={sequence_length}",
                # Empty stats path (no fitted file) => Gaussianize off, so this
                # exercises the merge-only transform: 3 raw channels -> 2 merged
                # (action_dim is 2). merge itself is config-driven (file-less).
                "+paths.action_norm_stats=''",
            ],
        )

    cfg_container = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True))
    hparams_jq = cfg_container["model"]["yaak"]["control_transformer"]["hparams_jq"]
    hparams = jq.compile(hparams_jq).input_value({}).first()
    objective = instantiate(hparams["objectives"]["modules"]["policy"])

    assert isinstance(objective, PolicyObjective)
    assert objective.history_steps == history_steps
    # merge-only (file-less): GasBrakeMerge present, Gaussianize absent (no /paths).
    assert isinstance(objective.action_transform, ActionTransform)
    assert isinstance(objective.action_transform.merge, GasBrakeMerge)
    assert objective.action_transform.gaussianize is None
    assert objective.action_transform.model_dim == len(MERGE_MODEL_KEYS)
    # No readout injected by the train config => the single-draw default.
    assert isinstance(objective.readout, SingleReadout)
    assert objective.flow_time_sampling == "logit-normal"
    assert isinstance(objective.loss, FlowMatchingLoss)
    assert objective.decoder.action_horizon == action_horizon
    assert objective.decoder.flow_sampling_method == "heun"
    assert objective.decoder.time_embedding.scale == pytest.approx(
        FLOW_TIME_EMBEDDING_SCALE
    )
    assert objective.decoder.time_embedding.logit_scale == pytest.approx(
        FLOW_TIME_LOGIT_SCALE
    )
    assert cfg.sequence_length == history_steps + action_horizon


def test_flow_policy_keeps_inverse_dynamics_config_path() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "model/yaak/control_transformer/raw",
            overrides=[
                "+num_heads=1",
                "+num_layers=1",
                "+encoder_embedding_dim=384",
                "+image_embedding_dim=384",
                "+speed_bins=512",
                "+gas_pedal_bins=255",
                "+brake_pedal_bins=165",
                "+steering_angle_bins=961",
            ],
        )

    inverse_dynamics = cfg.model.yaak.control_transformer.objectives.modules[
        "inverse_dynamics"
    ]

    assert (
        inverse_dynamics._target_
        == "rmind.components.objectives.InverseDynamicsPredictionObjective"
    )
    assert "turn_signal" in inverse_dynamics.heads.modules.discrete


def _synthetic_transform() -> ActionTransform:
    # uniform grid + strictly-increasing knots (longitudinal, steering), merge on
    k = 32
    grid = torch.linspace(0.0, 1.0, k)
    base = torch.linspace(-1.0, 1.0, k)
    knots = torch.stack([base, base * 0.5], dim=0)
    return ActionTransform(
        merge=GasBrakeMerge(),
        gaussianize=Gaussianize(grid=grid, knots=knots, model_keys=MERGE_MODEL_KEYS),
    )


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_policy_export_emits_k_draws(
    device: torch.device, method: FlowSamplingMethod
) -> None:
    # The deployable inference graph: noise-as-input, fixed NFE, inverse
    # transform, WTA left host-side. Must torch.export cleanly and match eager.
    decoder = FlowActionDecoder(
        condition_dim=16,
        dim_model=16,
        action_dim=2,
        action_horizon=6,
        flow_sampling_steps=2,
        flow_sampling_method=method,
        num_layers=1,
        num_heads=2,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        rope=True,
    ).to(device)
    exportable = ExportablePolicy(decoder, _synthetic_transform().to(device)).eval()

    cond = torch.randn(2, 4, 16, device=device)
    noise = torch.randn(2, 8, 6, 2, device=device)  # (B, K, H, A_model)
    with torch.inference_mode():
        eager = exportable(cond, noise)
    assert eager.shape == (2, 8, 6, 3)  # raw space: gas/brake/steering

    program = torch.export.export(exportable, (cond, noise))

    # the deploy path must be searchsorted-free (the one op that doesn't lower
    # to ONNX); the WTA control flow (sort/nonzero) must not be in the graph
    targets = {str(n.target) for n in program.graph.nodes if n.op == "call_function"}
    assert not [t for t in targets if "searchsorted" in t], (
        "searchsorted in export graph"
    )
    assert not [t for t in targets if "sort" in t or "nonzero" in t], (
        "WTA leaked into graph"
    )

    with torch.inference_mode():
        exported = program.module()(cond, noise)
    assert_close(exported, eager, atol=1e-5, rtol=1e-4)
