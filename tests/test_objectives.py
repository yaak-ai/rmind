import pytest
import torch
from pytest_lazy_fixtures import lf

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import Objective, PredictionKey


@pytest.mark.parametrize(
    "objective",
    [
        lf("inverse_dynamics_prediction_objective"),
        lf("forward_dynamics_prediction_objective"),
        lf("random_masked_hindsight_control_objective"),
        lf("memory_extraction_objective"),
        lf("policy_objective"),
    ],
)
def test_compute_metrics(objective: Objective, episode: Episode) -> None:
    assert "loss" in objective.compute_metrics(episode)


@pytest.mark.parametrize(
    "objective",
    [
        lf("inverse_dynamics_prediction_objective"),
        lf("forward_dynamics_prediction_objective"),
        lf("random_masked_hindsight_control_objective"),
        lf("memory_extraction_objective"),
        lf("policy_objective"),
    ],
)
@torch.inference_mode()
def test_predict(
    objective: Objective, episode: Episode, tokenizers: ModuleDict
) -> None:
    keys = set(PredictionKey)
    predictions = objective.predict(episode, keys=keys, tokenizers=tokenizers)
    assert set(predictions.keys()).issubset(keys)
