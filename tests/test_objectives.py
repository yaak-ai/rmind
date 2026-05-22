from typing import TYPE_CHECKING, cast

import pytest
import torch
from pytest_lazy_fixtures import lf

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import Objective, ObjectivePredictionKey
from rmind.components.objectives.policy import PolicyObjective
from rmind.components.transformer import TransformerEncoder

if TYPE_CHECKING:
    from rmind.components.base import TensorTree


@pytest.mark.parametrize(
    "objective",
    [
        lf("inverse_dynamics_prediction_objective"),
        lf("forward_dynamics_prediction_objective"),
        lf("memory_extraction_objective"),
        lf("policy_objective"),
    ],
)
def test_compute_losses_and_metrics(
    objective: Objective, episode: Episode, encoder: TransformerEncoder
) -> None:
    embedding = encoder(src=episode.embeddings_flattened, mask=episode.attention_mask)
    assert "loss" in objective.compute(episode=episode, embedding=embedding)


@torch.no_grad()
def test_policy_distribution_metrics(
    policy_objective: PolicyObjective, episode: Episode, encoder: TransformerEncoder
) -> None:
    embedding = encoder(src=episode.embeddings_flattened, mask=episode.attention_mask)
    result = policy_objective.compute(episode=episode, embedding=embedding)

    assert "metrics" in result
    dist = result["metrics"]

    expected_keys = {
        "pred_std",
        "gt_std",
        "std_ratio",
        "pred_mean",
        "gt_mean",
        "gt_diff",
        "pred_diff",
    }
    assert set(dist.keys()) == expected_keys

    expected_actions = {"gas_pedal", "brake_pedal", "steering_angle"}
    for metric_key in expected_keys:
        action_values = cast("TensorTree", dist[metric_key])
        assert set(action_values.keys()) == expected_actions
        for value in action_values.values():
            assert cast("torch.Tensor", value).ndim == 0
            assert cast("torch.Tensor", value).isfinite()

    for value in cast("TensorTree", dist["std_ratio"]).values():
        assert cast("torch.Tensor", value) >= 0


@pytest.mark.parametrize(
    "objective",
    [
        lf("inverse_dynamics_prediction_objective"),
        lf("forward_dynamics_prediction_objective"),
        lf("memory_extraction_objective"),
        lf("policy_objective"),
    ],
)
@pytest.mark.parametrize(
    ("keys", "expect_empty"),
    [(frozenset(), True), (frozenset(ObjectivePredictionKey), False)],
    ids=["no_keys", "all_keys"],
)
@torch.inference_mode()
def test_predict(  # noqa: PLR0913, PLR0917
    objective: Objective,
    episode: Episode,
    tokenizers: ModuleDict,
    encoder: TransformerEncoder,
    keys: frozenset[ObjectivePredictionKey],
    expect_empty: bool,  # noqa: FBT001
) -> None:
    predictions = objective.predict(
        episode=episode,
        keys=keys,
        embedding=encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        ),
        tokenizers=tokenizers,
    )
    prediction_keys = set(predictions.keys())
    assert prediction_keys.issubset(keys)
    assert (len(prediction_keys) == 0) is expect_empty
