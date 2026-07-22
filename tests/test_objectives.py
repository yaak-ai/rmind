import pytest
import torch
from pytest_lazy_fixtures import lf

from rmind.components.base import Modality, TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode, EpisodeBuilder
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
        lf("policy_objective_raw_waypoints"),
        lf("policy_objective_raw_speed"),
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
        lf("policy_objective_raw_waypoints"),
        lf("policy_objective_raw_speed"),
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


def test_raw_waypoints_dropout(
    policy_objective_raw_waypoints: PolicyObjective,
    episode_builder: EpisodeBuilder,
    encoder: TransformerEncoder,
    batch_dict: TensorTree,
) -> None:
    """raw_waypoints_dropout=1.0 (this fixture's setting) must zero the raw
    waypoints vector during training -- two episodes differing only in
    waypoints, but sharing one `embedding` tensor (so feature_keys' pooled
    parts are identical and only the raw_waypoints_key path can differ), must
    then produce identical logits -- but leave it untouched in eval mode,
    where the same setup must produce different logits.
    """
    alt_dict = dict(batch_dict)
    alt_dict["data"] = dict(batch_dict["data"])
    alt_dict["data"]["waypoints/xy_normalized"] = torch.rand_like(
        batch_dict["data"]["waypoints/xy_normalized"]
    )

    episode_a = episode_builder(batch_dict)
    episode_b = episode_builder(alt_dict)
    embedding = encoder(
        src=episode_a.embeddings_flattened, mask=episode_a.attention_mask
    )

    def gas_pedal_logits(episode: Episode) -> torch.Tensor:
        logits, *_ = policy_objective_raw_waypoints._compute_logits(  # noqa: SLF001
            episode=episode, embedding=embedding
        )
        return logits[Modality.CONTINUOUS]["gas_pedal"]

    try:
        policy_objective_raw_waypoints.eval()
        assert not torch.allclose(gas_pedal_logits(episode_a), gas_pedal_logits(episode_b))

        policy_objective_raw_waypoints.train()
        assert torch.allclose(gas_pedal_logits(episode_a), gas_pedal_logits(episode_b))
    finally:
        policy_objective_raw_waypoints.train()


def test_raw_speed_dropout(
    policy_objective_raw_speed: PolicyObjective,
    episode_builder: EpisodeBuilder,
    encoder: TransformerEncoder,
    batch_dict: TensorTree,
) -> None:
    """Same as test_raw_waypoints_dropout, but for raw_speed_key/
    raw_speed_dropout: two episodes differing only in speed, sharing one
    `embedding` tensor, must produce identical logits under dropout=1.0
    training but different logits in eval.
    """
    alt_dict = dict(batch_dict)
    alt_dict["data"] = dict(batch_dict["data"])
    alt_dict["data"]["meta/VehicleMotion/speed"] = torch.rand_like(
        batch_dict["data"]["meta/VehicleMotion/speed"]
    )

    episode_a = episode_builder(batch_dict)
    episode_b = episode_builder(alt_dict)
    embedding = encoder(
        src=episode_a.embeddings_flattened, mask=episode_a.attention_mask
    )

    def gas_pedal_logits(episode: Episode) -> torch.Tensor:
        logits, *_ = policy_objective_raw_speed._compute_logits(  # noqa: SLF001
            episode=episode, embedding=embedding
        )
        return logits[Modality.CONTINUOUS]["gas_pedal"]

    try:
        policy_objective_raw_speed.eval()
        assert not torch.allclose(gas_pedal_logits(episode_a), gas_pedal_logits(episode_b))

        policy_objective_raw_speed.train()
        assert torch.allclose(gas_pedal_logits(episode_a), gas_pedal_logits(episode_b))
    finally:
        policy_objective_raw_speed.train()
