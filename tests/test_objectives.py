import pytest
import torch
from pytest_lazy_fixtures import lf

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives import PolicyObjective
from rmind.components.objectives.base import Objective, ObjectivePredictionKey
from rmind.components.transformer import TransformerEncoder


@pytest.mark.parametrize(
    ("idx", "window", "expected"),
    [
        (4, 5, slice(0, 5)),  # training fixed idx, exact history_steps window
        (-1, 5, slice(-5, None)),  # inference, last tick
        (-4, 3, slice(-6, -3)),  # arbitrary negative idx, no special case
        (1, 3, slice(0, 2)),  # window > idx + 1: clamped, not wrapped
        (0, 1, slice(0, 1)),  # single-tick window, degenerates to today's idx-only read
    ],
)
def test_history_window_slice(
    idx: int, window: int, expected: slice, policy_objective: PolicyObjective
) -> None:
    assert policy_objective._history_window_slice(idx, window) == expected  # noqa: SLF001


@pytest.mark.parametrize(
    ("idx", "window", "t", "expected_len"),
    [
        (4, 5, 6, 5),  # normal training case
        (-1, 5, 6, 5),  # normal inference case
        (1, 3, 6, 2),  # window wider than available history at a fixed idx: clamps
        (-1, 999, 6, 6),  # window wider than the whole episode at idx=-1: clamps
    ],
)
def test_history_window_slice_against_real_sequence(
    idx: int, window: int, t: int, expected_len: int, policy_objective: PolicyObjective
) -> None:
    sl = policy_objective._history_window_slice(idx, window)  # noqa: SLF001
    assert len(torch.arange(t)[sl]) == expected_len


@pytest.mark.parametrize(
    "objective",
    [
        lf("inverse_dynamics_prediction_objective"),
        lf("forward_dynamics_prediction_objective"),
        lf("memory_extraction_objective"),
        lf("policy_objective"),
        lf("policy_objective_with_history_attn"),
        lf("policy_objective_multimodal"),
    ],
)
def test_compute_metrics(
    objective: Objective, episode: Episode, encoder: TransformerEncoder
) -> None:
    embedding = encoder(src=episode.embeddings_flattened, mask=episode.attention_mask)
    assert "loss" in objective.compute_metrics(episode=episode, embedding=embedding)


@pytest.mark.parametrize(
    "objective",
    [
        lf("inverse_dynamics_prediction_objective"),
        lf("forward_dynamics_prediction_objective"),
        lf("memory_extraction_objective"),
        lf("policy_objective"),
        lf("policy_objective_with_history_attn"),
        lf("policy_objective_multimodal"),
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
