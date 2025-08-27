import pytest
import torch
from pytest_lazy_fixtures import lf

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import Objective, PredictionResultKey


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
def test_compute_metrics(
    objective: Objective, episode: Episode, device: torch.device
) -> None:
    objective = objective.to(device)
    episode = episode.to(device)
    metrics = objective.compute_metrics(episode)
    assert "loss" in metrics


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
    objective: Objective, episode: Episode, tokenizers: ModuleDict, device: torch.device
) -> None:
    objective = objective.to(device).eval()
    episode = episode.to(device)
    tokenizers = tokenizers.to(device)

    result_keys = set(PredictionResultKey)
    predictions = objective.predict(
        episode, result_keys=result_keys, tokenizers=tokenizers
    )
    assert set(predictions.keys()).issubset(result_keys)  # pyright: ignore[reportArgumentType]
