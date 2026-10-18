import os
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from rmind.components.rtc_flow_head import TokenTimeFlowHead
from rmind.scripts.evaluate_d12_rtc import explicit_noise
from rmind.scripts.replay_d12_rtc import (
    Controller,
    command_metrics,
    sample_plan,
    tick_ns,
)
from rmind.scripts import replay_d12_rtc


def test_noise_is_explicit_paired_and_identity_bound():
    first = explicit_noise(131, "recording", 4)
    np.testing.assert_array_equal(first, explicit_noise(131, "recording", 4))
    assert first.dtype == np.float32
    assert not np.array_equal(first, explicit_noise(137, "recording", 4))
    assert not np.array_equal(first, explicit_noise(131, "other", 4))


def test_rational_action_boundaries():
    assert [tick_ns(tick) for tick in range(4)] == [0, 33333334, 66666667, 100000000]
    assert tick_ns(6) == 200000000


def test_raw_committed_prefix_normalized_once_then_preserved():
    head = TokenTimeFlowHead(4, 3, 50, channel_dim=16, num_layers=1, max_prefix=6)
    policy = SimpleNamespace(
        action_mean=torch.tensor([0.1, -0.2, 0.3]),
        action_std=torch.tensor([0.4, 0.5, 0.6]),
    )
    policy.fm_sample = lambda features, x0, clean_prefix, prefix_mask: head.sample(
        features, x0, clean_prefix, prefix_mask
    )
    prefix = np.zeros((50, 3), dtype=np.float32)
    prefix[:6] = [0.3, 0.2, -0.4]
    request = {"prefix": prefix.tolist(), "mask": [1.0] * 6 + [0.0] * 44}
    with torch.inference_mode():
        actions = sample_plan(
            policy, torch.zeros(1, 4), explicit_noise(131, "test", 1)[0], request, True
        )
    np.testing.assert_allclose(actions[:6], prefix[:6], atol=1e-7, rtol=0)


def test_metrics_keep_gaps_separate_from_zero_commands():
    arrays = {
        "commands": np.asarray(
            [[0.0, 0.0, 0.0], [np.nan] * 3, [0.2, 0.1, 0.0], [0.3, 0.4, 0.0]],
            dtype=np.float32,
        ),
        "scored": np.ones(4, dtype=bool),
        "plan_ids": np.asarray([1, -1, 2, 3]),
    }
    metrics = command_metrics(arrays)
    assert metrics["command_coverage"] == 0.75
    assert metrics["seams"] == 1
    assert metrics["per_action"]["steering"]["seam_delta_mean"] == pytest.approx(0.3)


@pytest.fixture
def controller():
    binary = os.environ.get("D12_RTC_STREAM_BINARY")
    if binary is None:
        pytest.skip("Build the unchanged Coppernic controller wrapper first")
    instance = Controller(Path(binary), 6)
    yield instance
    instance.close()


def test_actual_controller_commits_old_aligned_prefix(controller):
    request = controller.call({
        "op": "request",
        "now_ns": 0,
        "observation_ns": 0,
        "generation": 0,
    })["ok"]
    assert request["bootstrap"] and request["forecast"] == 0
    actions = [[index / 100, 0.0, 0.0] for index in range(50)]
    complete = controller.call({
        "op": "complete",
        "now_ns": 50_000_000,
        "generation": 0,
        "request_id": request["id"],
        "actions": actions,
    })
    assert complete["ok"] == 2
    output = controller.call({"op": "action", "now_ns": tick_ns(2), "generation": 0})[
        "ok"
    ]
    assert output["action_index"] == 2
    request = controller.call({
        "op": "request",
        "now_ns": tick_ns(6),
        "observation_ns": tick_ns(6),
        "generation": 0,
    })["ok"]
    assert request["forecast"] == 3
    np.testing.assert_allclose(request["prefix"][:3], actions[6:9])
    assert sum(request["mask"]) == 3
    assert np.count_nonzero(np.asarray(request["prefix"])[3:]) == 0


def test_actual_controller_expired_observation_fails_closed(controller):
    request = controller.call({
        "op": "request",
        "now_ns": 0,
        "observation_ns": 0,
        "generation": 0,
    })["ok"]
    output = controller.call({
        "op": "complete",
        "now_ns": 450_000_000,
        "generation": 0,
        "request_id": request["id"],
        "actions": [[0.0, 0.0, 0.0]] * 50,
    })
    assert output["error"] == "stale_observation"
    assert (
        controller.call({"op": "action", "now_ns": 500_000_000, "generation": 0})[
            "error"
        ]
        == "failed"
    )
    controller.call({"op": "reset", "generation": 1})
    assert "ok" in controller.call({
        "op": "request",
        "now_ns": 600_000_000,
        "observation_ns": 600_000_000,
        "generation": 1,
    })


def test_paced_replay_rebases_rational_clock_after_generation_reset(monkeypatch):
    binary = os.environ.get("D12_RTC_STREAM_BINARY")
    if binary is None:
        pytest.skip("Build the unchanged Coppernic controller wrapper first")
    decoder = SimpleNamespace(
        to=lambda device: decoder,
        eval=lambda: decoder,
        empty_cache=lambda: (torch.zeros(1, 1),) * 3,
        rope=lambda counter: (torch.zeros(1), torch.zeros(1)),
    )
    monkeypatch.setattr(replay_d12_rtc, "D12FlowDecoderStep", lambda policy: decoder)
    monkeypatch.setattr(
        replay_d12_rtc.PatchPolicyDecoderStep, "advance", lambda cache, *args: cache
    )
    monkeypatch.setattr(
        replay_d12_rtc,
        "sample_plan",
        lambda *args: np.full((50, 3), 0.2, dtype=np.float32),
    )
    policy = SimpleNamespace(
        norm=None,
        encoder=SimpleNamespace(
            step=lambda *args, **kwargs: (
                torch.zeros(1, 2, 4),
                torch.zeros(1, 1),
                torch.zeros(1, 1),
            )
        ),
    )
    rows = [
        {
            "wall_ns": tick_ns(1) + index * 200_000_000,
            "frame_wall_ns": tick_ns(1) + index * 200_000_000,
            "generation": index // 4,
        }
        for index in range(11)
    ]
    segment = (
        "fixture",
        "fixture",
        "1970-01-01T00:00:01+00:00",
        "1970-01-01T00:00:02+00:00",
    )
    arrays, trace = replay_d12_rtc.replay_one(
        policy,
        np.zeros((11, 1, 1), dtype=np.float32),
        rows,
        segment,
        "baseline",
        "zero",
        131,
        [0.05],
        Path(binary),
        "cpu",
        clock="paced",
    )
    actions = [event for event in trace["events"] if event["input"]["op"] == "action"]
    assert not any(event["output"].get("ok", False) is None for event in actions)
    assert not trace["errors"]
    assert np.isfinite(arrays["commands"][arrays["scored"]]).all()
