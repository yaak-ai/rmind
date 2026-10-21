import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch

from rmind.components.rtc_flow_head import TokenTimeFlowHead
from rmind.scripts import evaluate_d12_rtc, replay_d12_rtc
from rmind.scripts.audit_d12_rtc import require
from rmind.scripts.evaluate_d12_rtc import explicit_noise, sha256, verify
from rmind.scripts.replay_d12_rtc import (
    Controller,
    command_metrics,
    sample_plan,
    tick_ns,
)

if TYPE_CHECKING:
    from rmind.models.patch_policy_continuous import PatchPolicyContinuous


def test_audit_predicate_never_accepts_failed_evidence() -> None:
    require(condition=True, context="valid")
    with pytest.raises(ValueError, match="D12 evidence audit failed: corrupted"):
        require(condition=False, context="corrupted")


def test_audit_predicate_survives_python_optimized_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from rmind.scripts.audit_d12_rtc import require\nimport sys\ntry:\n require(False, 'corrupt')\nexcept ValueError:\n sys.exit(0)\nsys.exit(1)\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_loader_uses_verified_protocol_artifacts_not_default_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "alternate.ckpt"
    checkpoint.write_bytes(b"selected")
    config = tmp_path / "alternate.yaml"
    config.write_text("selected: true\n")
    protocol = {
        "models": {
            "rtc": {
                "checkpoint": str(checkpoint),
                "sha256": sha256(checkpoint),
                "config": str(config),
                "config_sha256": sha256(config),
            }
        }
    }
    loaded = []
    policy = SimpleNamespace()
    policy.eval = lambda: policy
    policy.to = lambda _device: policy

    def loader(path: str, **_kwargs: Any) -> SimpleNamespace:
        loaded.append(path)
        return policy

    monkeypatch.setattr(
        evaluate_d12_rtc.PatchPolicyContinuous, "load_from_checkpoint", loader
    )
    assert evaluate_d12_rtc.load_policy(protocol, "rtc", "cpu") is policy
    assert loaded == [str(checkpoint)]
    assert evaluate_d12_rtc.load_config(protocol, "rtc").selected is True
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        evaluate_d12_rtc.load_policy(protocol, "rtc", "cpu")
    config.write_text("selected: false\n")
    with pytest.raises(ValueError, match="Configuration changed"):
        evaluate_d12_rtc.load_config(protocol, "rtc")


@pytest.mark.parametrize("changed", ["array", "model", "missing_model"])
def test_token_cache_rejects_changed_array_or_model(
    tmp_path: Path, changed: str
) -> None:
    path = tmp_path / "tokens.npy"
    np.save(path, np.zeros((2, 3)))
    model = {"sha256": "checkpoint", "config_sha256": "config"}
    metadata = {"model_artifacts": model, "tokens_sha256": sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    replay_d12_rtc.load_tokens(path, model)
    if changed == "array":
        np.save(path, np.ones((2, 3)))
    elif changed == "model":
        model = {"sha256": "other", "config_sha256": "config"}
    else:
        del metadata["model_artifacts"]
        path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Token"):
        replay_d12_rtc.load_tokens(path, model)


def test_token_integrity_check_survives_python_optimized_mode(tmp_path: Path) -> None:
    path = tmp_path / "tokens.npy"
    np.save(path, np.zeros((2, 3)))
    path.with_suffix(".json").write_text(
        json.dumps({"model_artifacts": {}, "tokens_sha256": "not-the-array-hash"})
    )
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from pathlib import Path\nimport sys\nfrom rmind.scripts.replay_d12_rtc import load_tokens\ntry:\n load_tokens(Path(sys.argv[1]), {})\nexcept ValueError:\n sys.exit(0)\nsys.exit(1)\n",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("changed", ["checkpoint", "config", "latency"])
def test_provenance_rejects_changed_inputs(tmp_path: Path, changed: str) -> None:
    paths = {name: tmp_path / name for name in ("checkpoint", "config", "latency")}
    for path in paths.values():
        path.write_bytes(b"original")
    protocol = {
        "models": {
            "baseline": {
                "checkpoint": str(paths["checkpoint"]),
                "sha256": sha256(paths["checkpoint"]),
                "config": str(paths["config"]),
                "config_sha256": sha256(paths["config"]),
            }
        },
        "latency_source": str(paths["latency"]),
        "latency_sha256": sha256(paths["latency"]),
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    assert verify(tmp_path) == protocol
    paths[changed].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify(tmp_path)


def test_noise_is_explicit_paired_and_identity_bound() -> None:
    first = explicit_noise(131, "recording", 4)
    np.testing.assert_array_equal(first, explicit_noise(131, "recording", 4))
    assert first.dtype == np.float32
    assert not np.array_equal(first, explicit_noise(137, "recording", 4))
    assert not np.array_equal(first, explicit_noise(131, "other", 4))


def test_rational_action_boundaries() -> None:
    assert [tick_ns(tick) for tick in range(4)] == [0, 33333334, 66666667, 100000000]
    expected_six_ticks = 200000000
    assert tick_ns(6) == expected_six_ticks


def test_raw_committed_prefix_normalized_once_then_preserved() -> None:
    head = TokenTimeFlowHead(4, 3, 50, channel_dim=16, num_layers=1, max_prefix=6)
    policy = SimpleNamespace(
        action_mean=torch.tensor([0.1, -0.2, 0.3]),
        action_std=torch.tensor([0.4, 0.5, 0.6]),
    )

    def sample(
        features: torch.Tensor,
        x0: torch.Tensor,
        clean_prefix: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        return head.sample(features, x0, clean_prefix, prefix_mask)

    policy.fm_sample = sample
    prefix = np.zeros((50, 3), dtype=np.float32)
    prefix[:6] = [0.3, 0.2, -0.4]
    request = {"prefix": prefix.tolist(), "mask": [1.0] * 6 + [0.0] * 44}
    with torch.inference_mode():
        actions = sample_plan(
            cast("PatchPolicyContinuous", policy),
            torch.zeros(1, 4),
            explicit_noise(131, "test", 1)[0],
            request,
            conditional=True,
        )
    np.testing.assert_allclose(actions[:6], prefix[:6], atol=1e-7, rtol=0)


def test_metrics_keep_gaps_separate_from_zero_commands() -> None:
    arrays = {
        "commands": np.asarray(
            [[0.0, 0.0, 0.0], [np.nan] * 3, [0.2, 0.1, 0.0], [0.3, 0.4, 0.0]],
            dtype=np.float32,
        ),
        "scored": np.ones(4, dtype=bool),
        "plan_ids": np.asarray([1, -1, 2, 3]),
    }
    metrics = command_metrics(arrays)
    assert metrics["command_coverage"].hex() == (0.75).hex()
    assert metrics["seams"] == 1
    assert metrics["per_action"]["steering"]["seam_delta_mean"] == pytest.approx(0.3)


@pytest.fixture
def controller() -> Iterator[Controller]:
    binary = os.environ.get("D12_RTC_STREAM_BINARY")
    if binary is None:
        pytest.skip("Build the unchanged Coppernic controller wrapper first")
    instance = Controller(Path(binary), 6)
    yield instance
    instance.close()


def test_actual_controller_commits_old_aligned_prefix(controller: Controller) -> None:
    request = controller.call({
        "op": "request",
        "now_ns": 0,
        "observation_ns": 0,
        "generation": 0,
    })["ok"]
    assert request["bootstrap"]
    assert request["forecast"] == 0
    actions = [[index / 100, 0.0, 0.0] for index in range(50)]
    complete = controller.call({
        "op": "complete",
        "now_ns": 50_000_000,
        "generation": 0,
        "request_id": request["id"],
        "actions": actions,
    })
    expected_start_tick = 2
    assert complete["ok"] == expected_start_tick
    output = controller.call({"op": "action", "now_ns": tick_ns(2), "generation": 0})[
        "ok"
    ]
    assert output["action_index"] == expected_start_tick
    request = controller.call({
        "op": "request",
        "now_ns": tick_ns(6),
        "observation_ns": tick_ns(6),
        "generation": 0,
    })["ok"]
    expected_forecast = 3
    assert request["forecast"] == expected_forecast
    np.testing.assert_allclose(request["prefix"][:3], actions[6:9])
    assert sum(request["mask"]) == expected_forecast
    assert np.count_nonzero(np.asarray(request["prefix"])[3:]) == 0


def test_actual_controller_expired_observation_fails_closed(
    controller: Controller,
) -> None:
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


def test_paced_replay_rebases_rational_clock_after_generation_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = os.environ.get("D12_RTC_STREAM_BINARY")
    if binary is None:
        pytest.skip("Build the unchanged Coppernic controller wrapper first")
    decoder = SimpleNamespace(
        to=lambda _device: decoder,
        eval=lambda: decoder,
        empty_cache=lambda: (torch.zeros(1, 1),) * 3,
        rope=lambda _counter: (torch.zeros(1), torch.zeros(1)),
    )
    monkeypatch.setattr(replay_d12_rtc, "D12FlowDecoderStep", lambda _policy: decoder)
    monkeypatch.setattr(
        replay_d12_rtc.PatchPolicyDecoderStep, "advance", lambda cache, *_args: cache
    )
    monkeypatch.setattr(
        replay_d12_rtc,
        "sample_plan",
        lambda *_args: np.full((50, 3), 0.2, dtype=np.float32),
    )
    policy = SimpleNamespace(
        norm=None,
        encoder=SimpleNamespace(
            step=lambda *_args, **_kwargs: (
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
        cast("PatchPolicyContinuous", policy),
        np.zeros((11, 1, 1), dtype=np.float32),
        rows,
        segment,
        "baseline",
        "zero",
        131,
        np.array([0.05]),
        Path(binary),
        "cpu",
        clock="paced",
    )
    actions = [event for event in trace["events"] if event["input"]["op"] == "action"]
    assert not any(event["output"].get("ok", False) is None for event in actions)
    assert not trace["errors"]
    assert np.isfinite(arrays["commands"][arrays["scored"]]).all()
