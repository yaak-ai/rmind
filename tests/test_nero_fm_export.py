"""The flow-matching (action_head="fm") serving export: the `fm_x0` graph input,
the Euler steps unrolled at export, the manifest's `fm` block, and the offline
replay fed the bundle's own noise."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from rmind.models.nero_patch_policy_decoder import NeroPatchPolicyDecoderStep
from tests.test_fm_head import _fm_policy
from tests.test_nero_robot import CHUNK, NUTRON_CLI, _batch, _export

TOL = 1e-4

if TYPE_CHECKING:
    from pathlib import Path


def test_decoder_step_needs_fm_x0_and_unrolls_the_euler_steps() -> None:
    policy = _fm_policy(hand=False).eval()
    policy.fm_inference_steps = 3
    step = NeroPatchPolicyDecoderStep(policy=policy).eval()
    batch = _batch(t=2, b=1)
    inputs = step.frame_inputs(batch, 0)
    past = step.empty_cache()
    cos, sin = step.rope(0)
    args = (
        inputs["images"],
        inputs["state"],
        inputs["side_valid"],
        *past,
        cos,
        sin,
        None,
    )
    with pytest.raises(ValueError, match="needs `fm_x0`"):
        step(*args)
    n_sides = inputs["side_valid"].shape[-1]
    x0 = torch.randn(1, n_sides, CHUNK * 13, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        actions = step(*args, x0)[0]
        # the step's chunk IS fm_sample on the same features and noise
        features = policy._features(batch)[:, 0]  # noqa: SLF001
        want = policy.fm_sample(policy._per_side_features(features), x0=x0, steps=3)  # noqa: SLF001
    torch.testing.assert_close(
        actions, want.permute(0, 2, 1, 3).reshape(1, CHUNK, -1), rtol=0, atol=1e-5
    )
    assert torch.equal(x0, x0.clone())  # the caller's noise is never written into


@pytest.mark.parametrize("hand", [True, False])
def test_fm_export_has_the_noise_input_and_the_fm_block(
    tmp_path: Path, *, hand: bool
) -> None:
    """With and without a hand token (without: `hand_token` is a None hole before
    `fm_x0` in the step's positional inputs -- the paper2d_lh50_fm layout)."""
    pytest.importorskip("onnxruntime")
    nutron = (
        NUTRON_CLI
        if (NUTRON_CLI / "runtime/jetson/policy_contract.py").exists()
        else None
    )
    policy = _fm_policy(hand=hand)
    policy.fm_inference_steps = 2
    nero_export, report, out = _export(tmp_path, policy, nutron_cli=nutron)
    assert report["streaming_vs_windowed"]["max_abs"] <= TOL
    assert report["streaming_vs_windowed"]["action_head"] == "fm"
    assert max(report["ort_vs_eager"]["max_abs"].values()) <= TOL
    manifest = json.loads((out / "policy_manifest.json").read_text())
    assert manifest["action_head"] == "fm"
    assert manifest["fm"] == {"steps": 2, "x0": "sample", "seed": 0}
    io = manifest["io"]
    assert io["inputs"]["fm_x0"]["shape"] == [1, 2, CHUNK * 13]
    assert ("hand_token" in io["inputs"]) is hand
    assert set(io["outputs"]) == {"actions", "new_k", "new_v"}
    # a nutron-cli that knows the head accepts it; an older one refuses (recorded)
    gate3 = report["nutron_cli"]
    if nutron is not None and gate3["status"] == "ok":
        assert report["failures"] == [], report
        assert gate3["binding_problems"] == []

    # ORT: zero noise is just another x0 (the x0=0 serving mode), same graph
    import onnxruntime as ort  # noqa: PLC0415

    step = NeroPatchPolicyDecoderStep(policy=policy.eval()).eval()
    batch = _batch(t=2, b=1)
    named = nero_export.ort_inputs(
        step, step.frame_inputs(batch, 0), step.empty_cache(), 0
    )
    named["fm_x0"] = torch.zeros_like(named["fm_x0"])
    session = ort.InferenceSession(
        str(out / "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    bound = {i.name for i in session.get_inputs()}
    got = session.run(
        ["actions"], {k: v.numpy() for k, v in named.items() if k in bound}
    )[0]
    with torch.no_grad():
        eager = step(*nero_export.positional(named))[0]
    np.testing.assert_allclose(np.asarray(got), eager.numpy(), atol=TOL)


def test_replay_expected_feeds_the_bundles_noise() -> None:
    from rmind.scripts import nero_replay_expected as rx  # noqa: PLC0415

    policy = _fm_policy(hand=False).eval()
    batch = _batch(t=3, b=1)
    n = 3
    manifest = {
        "sides": ["left", "right"],
        "side_valid": [bool(v) for v in batch["side_valid"][0].tolist()],
        "camera_cond": {"values": torch.zeros(3, 13).tolist()},
        "fm": {"steps": 2},
    }
    bundle = {
        "state": batch["state"][0].reshape(n, -1).numpy().astype(np.float32),
        "images_u8": torch.stack(
            [batch[policy.image_key.format(camera=c)][0] for c in policy.cameras], 1
        ).numpy(),
        "reset": np.array([True, False, False]),
    }
    with pytest.raises(ValueError, match="needs the bundle's fm_x0"):
        rx.expected(policy, bundle, manifest, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    bundle["fm_x0"] = rng.standard_normal((n, 2, CHUNK * 13)).astype(np.float32)
    a, codes = rx.expected(policy, bundle, manifest, device=torch.device("cpu"))
    b, _ = rx.expected(policy, bundle, manifest, device=torch.device("cpu"))
    assert a.shape == (n, CHUNK, 26)
    assert codes.shape == (n, 0)
    np.testing.assert_array_equal(a, b)  # deterministic given the noise
    assert policy.fm_inference_steps == manifest["fm"]["steps"]
    bundle["fm_x0"] = np.zeros_like(bundle["fm_x0"])
    c, _ = rx.expected(policy, bundle, manifest, device=torch.device("cpu"))
    assert not np.allclose(a, c)  # the noise is really used
