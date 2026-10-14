"""The flow-matching action head (action_head="fm") is the toy-verified bimod-study head.

`rmind.components.fm_action_head` is `heads.py` of the bimod study at tag
`fm-toy-verified-v1.1` (sisyphos:~/bimod-study/wta, commit 472e3a7), vendored byte for
byte and pinned here by SHA256. The parity fixtures written by that tag's
`parity/fm_fixture.py make` (trained BlockPush W=10 head, robot-size H50 / H100 heads:
fixed inputs + the tag's velocity, Euler 1 / 4 / 10 samples and CFM losses with and
without a pad mask) are checked against rmind's code path
(`NeroPatchPolicy.fm_sample`, `NeroPatchPolicy._fm_loss`). The fixtures are too large
for the repo (check-added-large-files); they are read from `RMIND_FM_PARITY_DIR`
(default the study's NAS directory) and those tests skip when it is absent.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor

from rmind.components import fm_action_head
from rmind.components.fm_action_head import ActionHead
from rmind.models.nero_patch_policy import NeroPatchPolicy
from tests.test_nero_robot import CHUNK, POLICY_DIM, _batch, _policy

#: sha256 of heads.py at bimod-study tag fm-toy-verified-v1.1 (= v1, 7d91894)
FM_HEAD_SHA256 = "8a99bd91f12807fc7e1ccee690d8ac4acc9e4c09121b369b435312f90aa65582"
PARITY_DIR = Path(
    os.environ.get("RMIND_FM_PARITY_DIR", "/nasa/max/bimod-study/wta/parity")
)
FIXTURES = {
    "blockpush_w10_trained": ("fm_fixture_bp_w10_s0_v1.1.pt", 10, 2),
    "robot_h50": ("fm_fixture_robot_h50_v1.1.pt", 50, 13),
    "robot_h100": ("fm_fixture_robot_h100_v1.1.pt", 100, 13),
}
TOL = 1e-5


def _stand_in(head: ActionHead, horizon: int, features: int, **kw: Any) -> Any:
    """The attributes `NeroPatchPolicy.fm_sample` / `_fm_loss` read, nothing else."""
    return SimpleNamespace(
        fm_head=head,
        tokenizer=SimpleNamespace(action_horizon=horizon, action_features=features),
        fm_inference_steps=kw.get("steps", 4),
        fm_x0=kw.get("x0", "sample"),
    )


def test_vendored_head_is_the_tagged_file() -> None:
    src = Path(fm_action_head.__file__).read_bytes()
    assert hashlib.sha256(src).hexdigest() == FM_HEAD_SHA256


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fm_parity_with_the_toy_verified_head(name: str) -> None:  # noqa: PLR0914
    path = PARITY_DIR / FIXTURES[name][0]
    if not path.exists():
        pytest.skip(f"parity fixture {path} not available")
    fx = torch.load(path, map_location="cpu", weights_only=False)
    horizon, features = FIXTURES[name][1:]
    assert fx["D"] == horizon * features
    head = ActionHead("fm", fx["feat"], 1, fx["D"]).eval()
    head.load_state_dict(fx["state_dict"])
    inp, ref = fx["inputs"], fx["ref"]
    f, x0, t, target = inp["f"], inp["x0"], inp["t"], inp["target"]
    me = _stand_in(head, horizon, features)
    with torch.no_grad():
        worst = {
            "velocity": (head.velocity(f, inp["xt"], t) - ref["velocity"]).abs().max()
        }
        for s in (1, 4, 10):
            got = NeroPatchPolicy.fm_sample(me, f, x0=x0, steps=s).flatten(1)
            worst[f"euler{s}"] = (got - ref[f"euler{s}"]).abs().max()
            zero = NeroPatchPolicy.fm_sample(me, f, x0=torch.zeros_like(x0), steps=s)
            worst[f"euler{s}_x0zero"] = (
                (zero.flatten(1) - ref[f"euler{s}_x0zero"]).abs().max()
            )
        # the CFM loss at the fixture's fixed (x0, t): unmasked and with its pad mask
        xt = (1 - t[:, None]) * x0 + t[:, None] * target
        err = (head.velocity(f, xt, t) - (target - x0)).pow(2)
        worst["cfm_loss"] = (err.mean() - ref["cfm_loss"]).abs()
        m = inp["mask"]
        worst["cfm_loss_masked"] = (
            (err * m).sum() / m.sum() - ref["cfm_loss_masked"]
        ).abs()
    bad = {k: float(v) for k, v in worst.items() if float(v) > TOL}
    assert not bad, bad


def test_fm_sample_is_the_heads_own_euler_loop() -> None:
    torch.manual_seed(3)
    head = ActionHead("fm", 16, 1, 4 * 3, hidden=32).eval()
    f = torch.randn(5, 16)
    for steps in (1, 4, 10):
        head.fm_steps = steps
        with torch.no_grad():
            ref = head.act(f, "argmax")[0]
            got = NeroPatchPolicy.fm_sample(
                _stand_in(head, 4, 3, x0="zero"), f, steps=steps
            )
            assert torch.equal(got.flatten(1), ref)
            torch.manual_seed(11)
            ref_s = head.act(f, "sample")[0]
            torch.manual_seed(11)
            got_s = NeroPatchPolicy.fm_sample(_stand_in(head, 4, 3), f, steps=steps)
            assert torch.equal(got_s.flatten(1), ref_s)


def test_fm_loss_is_the_heads_loss_with_the_step_mask() -> None:
    """`_fm_loss` = ActionHead.loss (same x0, t draws) with `real (n, H)` expanded
    over the action features: padded steps' velocity errors are excluded.

    Unlike the l1 head, the padded target values are not inert: they enter the CFM
    interpolant x_t = (1 - t) x0 + t a, the velocity MLP's input (exactly as in the
    toy-verified head, incl. its BlockPush W=50 pad-mask runs). The robot reader
    hold-pads (last action), so those values are in-distribution."""
    torch.manual_seed(4)
    h, a = 5, 3
    head = ActionHead("fm", 8, 1, h * a, hidden=32)
    f, target = torch.randn(6, 8), torch.randn(6, h, a)
    real = torch.ones(6, h, dtype=torch.bool)
    real[:3, 3:] = False
    rows = {"side_features": f, "target": target, "real": real}
    torch.manual_seed(9)
    got = NeroPatchPolicy._fm_loss(_stand_in(head, h, a), rows)  # noqa: SLF001
    torch.manual_seed(9)
    x0 = torch.randn(6, h * a)
    t = torch.rand(6)
    xt = (1 - t[:, None]) * x0 + t[:, None] * target.flatten(1)
    err = (head.velocity(f, xt, t) - (target.flatten(1) - x0)).pow(2)
    w = real[..., None].float().expand(6, h, a).flatten(1)
    assert torch.allclose(got, (err * w).sum() / w.sum(), rtol=0, atol=1e-6)
    assert not torch.allclose(got, err.mean(), rtol=0, atol=1e-6)


def _fm_policy(**kw: Any) -> NeroPatchPolicy:
    torch.manual_seed(0)
    return _policy(
        action_head="fm",
        code_head=None,
        offset_head=None,
        fm_head=ActionHead("fm", POLICY_DIM, 1, CHUNK * 13, hidden=32),
        **kw,
    )


def test_fm_head_trains_with_a_single_masked_cfm_loss() -> None:
    policy = _fm_policy(quality_metrics=True).train()
    assert policy.code_head is None
    assert policy.l1_head is None
    out = policy.compute_metrics(_batch(t=4))
    losses = out["policy", "loss"]
    assert isinstance(losses, TensorDict)
    assert set(losses.keys()) == {"action_fm"}
    loss = losses["action_fm"]
    assert isinstance(loss, Tensor)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.fm_head is not None
    assert all(p.grad is not None for p in policy.fm_head.parameters())
    metric = out["policy", "metric"]
    assert "ev/arm/h10_29" in metric
    assert "action_l1_sample" in metric


def test_fm_forward_takes_explicit_noise_and_is_deterministic_with_it() -> None:
    policy = _fm_policy()
    batch = _batch(t=3)
    x0 = torch.randn(2, 2, CHUNK * 13)
    with torch.no_grad():
        a = policy(batch, fm_x0=x0)["policy", "action"]
        b = policy(batch, fm_x0=x0)["policy", "action"]
        c = policy(batch, fm_x0=torch.zeros_like(x0))["policy", "action"]
    assert a.shape == (2, 2, CHUNK, 13)
    assert torch.equal(a, b)
    policy.fm_x0 = "zero"
    with torch.no_grad():
        d = policy(batch)["policy", "action"]
    assert torch.equal(c, d)


def test_fm_head_config_is_checked() -> None:
    with pytest.raises(ValueError, match="needs fm_head"):
        _policy(action_head="fm", code_head=None, offset_head=None)
    with pytest.raises(ValueError, match="out_dim"):
        _policy(
            action_head="fm",
            code_head=None,
            offset_head=None,
            fm_head=ActionHead("fm", POLICY_DIM, 1, 7, hidden=32),
        )
    with pytest.raises(ValueError, match="fm_head needs"):
        _policy(fm_head=ActionHead("fm", POLICY_DIM, 1, CHUNK * 13, hidden=32))
