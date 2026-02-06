import pytest
import torch
from pytest_lazy_fixtures import lf

from rmind.components.attention_mask import build_attention_mask
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.llm import TransformerEncoder
from rmind.components.mask import AttentionMask, TorchAttentionMaskLegend
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
    mask = build_attention_mask(
        episode.index, episode.timestep, legend=TorchAttentionMaskLegend
    )
    embedding = encoder(
        src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
    )
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
    mask = build_attention_mask(
        episode.index, episode.timestep, legend=TorchAttentionMaskLegend
    )
    embedding = encoder(
        src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
    )
    attention_rollout = None
    if PredictionKey.ATTENTION_ROLLOUT in keys and getattr(
        objective, "NEEDS_ATTENTION_ROLLOUT", False
    ):
        mask_tensor = mask.mask.to(episode.device)
        attention_rollout = encoder.compute_attention_rollout(
            src=episode.embeddings_packed,
            mask=AttentionMask(
                mask=mask_tensor, legend=mask.legend, device=mask_tensor.device
            ),
            head_fusion="max",
            discard_ratio=0.9,
        )

    predictions = objective.predict(
        episode,
        keys=keys,
        tokenizers=tokenizers,
        embedding=embedding,
        attention_rollout=attention_rollout,
    )
    assert set(predictions.keys()).issubset(keys)
