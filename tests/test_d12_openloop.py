import numpy as np
import pytest

from rmind.scripts.eval_d12_fm import action_metrics


def test_zero_predictor_is_identified_even_with_imbalanced_idle_targets() -> None:
    targets = np.zeros((10, 6, 3))
    targets[-1] = 0.5
    predictions = np.zeros((10, 8, 6, 3))
    metrics = action_metrics(predictions, targets)
    assert metrics["mae"] == metrics["zero_action_mae"]
    assert metrics["deadbands"]["0.05"]["predicted_all_commands_idle"] == 1
    assert metrics["deadbands"]["0.05"]["recorded_all_commands_idle"] == pytest.approx(
        0.9
    )
    assert metrics["per_action"]["traction"]["missed_active_fraction"] == 1
    assert metrics["between_sample_std"] == 0


def test_perfect_active_predictor_beats_zero_and_preserves_sign() -> None:
    targets = np.tile([0.8, -0.3, 0.0], (4, 6, 1))
    predictions = np.repeat(targets[:, None], 8, axis=1)
    metrics = action_metrics(predictions, targets)
    assert metrics["mae"] == 0
    assert metrics["zero_action_mae"] > 0
    assert metrics["per_action"]["steering"]["missed_active_fraction"] == 0
    assert metrics["per_action"]["steering"]["wrong_sign_on_active_fraction"] == 0
    assert metrics["per_action"]["fork1"]["active_mae"] is None


def test_sample_average_does_not_hide_opposing_active_modes() -> None:
    targets = np.ones((1, 6, 3)) * 0.5
    predictions = np.stack([targets, -targets], axis=1)
    metrics = action_metrics(predictions, targets)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["between_sample_std"] == pytest.approx(0.5)
    assert metrics["best_of_samples_mae_optimistic"] == 0
    assert metrics["deadbands"]["0.05"]["predicted_all_commands_idle"] == 0


def test_nonfinite_predictions_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        action_metrics(np.full((1, 2, 6, 3), np.nan), np.zeros((1, 6, 3)))
