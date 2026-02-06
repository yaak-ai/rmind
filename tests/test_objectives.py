import pytest
import torch
from pytest_lazy_fixtures import lf

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.llm import TransformerEncoder
from rmind.components.objectives.base import Objective, PredictionKey


@pytest.mark.parametrize(
    "objective",
    [
        lf("inverse_dynamics_prediction_objective"),
        lf("forward_dynamics_prediction_objective"),
        lf("memory_extraction_objective"),
        lf("policy_objective"),
    ],
)
def test_compute_metrics(
    objective: Objective, episode: Episode, encoder: TransformerEncoder
) -> None:
    embedding = encoder(src=episode.embeddings_packed, mask=episode.attention_mask)
    assert "loss" in objective.compute_metrics(episode, embedding=embedding)


@pytest.mark.parametrize(
    "objective",
    [
        lf("inverse_dynamics_prediction_objective"),
        lf("forward_dynamics_prediction_objective"),
        lf("memory_extraction_objective"),
        lf("policy_objective"),
    ],
)
@torch.inference_mode()
def test_predict(
    objective: Objective,
    episode: Episode,
    tokenizers: ModuleDict,
    encoder: TransformerEncoder,
) -> None:
    keys = set(PredictionKey)
    attention_rollout = None
    if PredictionKey.ATTENTION_ROLLOUT in keys and (
        objective.supports_attention_rollout
    ):
        attention_rollout = encoder.compute_attention_rollout(
            src=episode.embeddings_packed,
            mask=episode.attention_mask,
            head_fusion="max",
            discard_ratio=0.9,
        )
    predictions = objective.predict(
        episode,
        keys=keys,
        embedding=encoder(src=episode.embeddings_packed, mask=episode.attention_mask),
        tokenizers=tokenizers,
        attention_rollout=attention_rollout
        if objective.supports_attention_rollout
        else None,
    )
    assert set(predictions.keys()).issubset(keys)
