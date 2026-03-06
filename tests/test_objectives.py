import pytest
import torch
from pytest_lazy_fixtures import lf

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.llm import TransformerEncoder
from rmind.components.objectives.base import Objective, ObjectivePredictionKey


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
    embedding = encoder(
        src=episode.embeddings_packed, mask=episode.attention_mask.mask_tensor
    )
    assert "loss" in objective.compute_metrics(episode=episode, embedding=embedding)


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
            src=episode.embeddings_packed, mask=episode.attention_mask.mask_tensor
        ),
        tokenizers=tokenizers,
    )
    prediction_keys = set(predictions.keys())
    assert prediction_keys.issubset(keys)
    assert (len(prediction_keys) == 0) is expect_empty
