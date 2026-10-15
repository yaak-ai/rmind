from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from rmind.data.d12_sampling import sampling_weights
from rmind.scripts.audit_d12_actions import nearest_offsets
from rmind.scripts.d12_fm_config import configure_cameras, dataset_overrides
from rmind.scripts.eval_d12_fm import hold_events, replan_metrics, resample_features
from rmind.scripts.probe_d12_fm import (
    paired_metrics,
    plot_goal_sensitivity,
    summarize_probes,
    world_offset_token,
)


def test_balancing_preserves_uniform_mass_and_upsamples_holds():
    actions = np.ones((100, 50, 3)) * 0.4
    actions[:5] = 0
    weights, labels = sampling_weights(actions)
    assert weights.sum() == pytest.approx(1)
    assert weights[:5].sum() == pytest.approx(0.275)
    assert weights.min() >= 0.005
    assert np.all(labels[:5] == 1)


def test_hold_does_not_require_centered_steering():
    actions = np.zeros((1, 50, 3))
    actions[..., 1] = 0.7
    _, labels = sampling_weights(actions)
    assert labels.tolist() == [1]


def test_transition_category_precedes_lift_category():
    actions = np.zeros((2, 50, 3))
    actions[0, 4:, 2] = 0.9
    actions[1, :, 2] = 0.9
    _, labels = sampling_weights(actions)
    assert labels.tolist() == [4, 2]


def test_world_offsets_rotate_and_scale_for_each_observation():
    np.testing.assert_allclose(
        world_offset_token((1, 0), [0, 90, 180]),
        [[0.1, 0], [0, -0.1], [-0.1, 0]],
        atol=1e-8,
    )


def test_paired_sensitivity_detects_idle_collapse_not_just_difference():
    baseline = np.ones((2, 8, 50, 3)) * 0.5
    assert paired_metrics(baseline, baseline)["paired_near_abs_change"] == 0
    assert (
        paired_metrics(baseline, np.zeros_like(baseline))[
            "collapse_to_idle_given_baseline_active"
        ]
        == 1
    )


def test_nearest_offsets_expose_future_joins():
    np.testing.assert_array_equal(nearest_offsets([1, 6, 11], [0, 10]), [-1, 4, -1])


def test_sensitivity_separates_actions_and_only_scores_near_commands():
    baseline = np.zeros((2, 8, 50, 3))
    baseline[..., 0] = 0.4
    baseline[..., 1] = -0.5
    changed = baseline.copy()
    changed[..., :6, 1] = 0
    changed[..., 6:, :] = 99
    metrics = paired_metrics(baseline, changed)["per_action"]
    assert metrics["traction"]["paired_abs_change"] == 0
    assert metrics["traction"]["neutral_given_baseline_active"] == 0
    assert metrics["steering"]["paired_abs_change"] == pytest.approx(0.5)
    assert metrics["steering"]["neutral_given_baseline_active"] == 1
    assert metrics["fork1"]["neutral_given_baseline_active"] is None
    assert metrics["fork1"]["baseline_active_count"] == 0


def test_sensitivity_summary_keeps_splits_and_plot_handles_no_active(tmp_path):
    baseline = np.zeros((2, 8, 50, 3))
    changed = baseline.copy()
    changed[0, ..., 0] = 0.5
    stacked = {"baseline": baseline, "pickup/0.1/+x": changed}
    contexts = [{"split": split, "category": "hold"} for split in ("train", "val")]
    metrics = summarize_probes(stacked, contexts)
    assert metrics["val/all"]["pickup/0.1/+x"]["paired_near_abs_change"] == 0
    assert (
        metrics["train/hold"]["pickup/0.1/+x"]["per_action"]["traction"][
            "paired_abs_change"
        ]
        == 0.5
    )
    plot_goal_sensitivity(
        {
            "metrics": metrics,
            "cases": {"pickup/0.1/+x": {"field": "pickup", "radius_m": 0.1}},
        },
        tmp_path,
    )
    assert (tmp_path / "goal_sensitivity.png").is_file()


def test_hold_events_do_not_cross_cache_resets():
    arrays = {
        "target": np.zeros((2, 50, 3)),
        "stream": np.ones((2, 2, 50, 3)),
        "reset": np.array([True, True]),
    }
    metrics = hold_events(arrays)["traction"]
    assert metrics["hold_sample_pairs"] == 4
    assert metrics["false_activation_duration_seconds_p50_p95_max"] == pytest.approx([
        0.2,
        0.2,
        0.2,
    ])


def test_cached_feature_resampling_couples_noise_and_resets(monkeypatch):
    import torch

    class NoisePolicy:
        action_horizon = 50
        action_std = torch.ones(3)
        action_mean = torch.zeros(3)

        def fm_sample(self, features, x0):
            return x0.reshape(len(features), 50, 3)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    arrays = {
        "reset": np.array([True, False, True]),
        "window_features": np.zeros((3, 4), dtype=np.float32),
        "stream_features": np.zeros((3, 4), dtype=np.float32),
    }
    fixed = resample_features(NoisePolicy(), arrays, 2, 1, "fixed", "cpu")
    shifted = resample_features(NoisePolicy(), arrays, 2, 1, "shift", "cpu")
    np.testing.assert_array_equal(fixed["stream"][0], fixed["stream"][1])
    assert not np.array_equal(fixed["stream"][0], fixed["stream"][2])
    np.testing.assert_array_equal(
        shifted["stream"][0, :, 6:], shifted["stream"][1, :, :-6]
    )
    np.testing.assert_array_equal(fixed["window"], fixed["stream"])


def test_replan_metric_excludes_reset_and_aligns_overlap():
    chunks = np.zeros((3, 2, 50, 3))
    chunks[0, :, 6] = 0.4
    chunks[1, :, 0] = 0.4
    chunks[2] = -1
    metrics = replan_metrics({
        "stream": chunks,
        "reset": np.array([True, False, True]),
        "seconds": np.array([0, 0.2, 0.4]),
    })
    assert metrics["pairs"] == 1
    assert metrics["traction"]["overlap_revision_mean"] == 0
    assert metrics["traction"]["boundary_jump_mean"] == pytest.approx(0.4)


@pytest.mark.parametrize("layout", ["fork", "forward", "backward"])
def test_camera_layout_config(layout):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config"),
        version_base=None,
    ):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=palletjack/patch_policy/d12_fm",
                "action_mean=[0,0,0]",
                "action_std=[1,1,1]",
                *dataset_overrides(layout),
            ],
        )
    configure_cameras(config, layout)
    assert list(config.image_remap) == list(config.cameras)
    assert config.num_cameras == len(config.cameras)
    for split in ("train", "val"):
        dataset = config.datamodule[split].dataset
        assert set(config.cameras).issubset(dataset.streams)
        if layout == "backward":
            assert set(dataset.streams) == set(config.cameras)
            assert list(dataset.samples.pipeline.functions[0].func.cameras) == list(
                config.cameras
            )
