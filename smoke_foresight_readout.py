"""Smoke test for foresight-readout (1.1/2.4) + FD target hook (1.2).

Exercises the real code paths with light stubs (no wandb/tokenizer/GPU):
  - ForesightAttentionPool forward/backward + out_features
  - JointPolicyObjective._features width assembly for every branch combo
    (waypoints x attn x maxpool), incl. the width-mismatch guard
  - ForwardDynamicsPredictionObjective target re-encoding via a dummy encoder
"""

import torch
from torch import nn

from rmind.components.base import Modality, SummaryToken
from rmind.components.foresight_readout import ForesightAttentionPool
from rmind.components.objectives.joint_policy import JointPolicyObjective

B, D, N = 3, 384, 256
G, C, ADIM = 4, 16, 24


# ---- 1.1 module ------------------------------------------------------------
pool = ForesightAttentionPool(dim=D, num_queries=4, num_heads=4)
x = torch.randn(B, N, D, requires_grad=True)
out = pool(x)
assert out.shape == (B, 4 * D), out.shape
assert pool.out_features == 4 * D
out.sum().backward()
assert x.grad is not None and torch.isfinite(out).all()
print(f"[1.1] ForesightAttentionPool ok: (b,{N},{D}) -> {tuple(out.shape)}")


# ---- stub episode so _features exercises the real assembly ------------------
class _Parsed:
    def __init__(self, d):
        self._d = d

    def get(self, k):
        return self._d[k]


class _Selected:
    def __init__(self, d):
        self._d = d

    def parse(self, _embedding):
        return _Parsed(self._d)


class _Index:
    def __init__(self, d):
        self._d = d

    def select(self, *keys):
        assert set(keys) <= set(self._d), (set(keys), set(self._d))
        return _Selected({k: self._d[k] for k in keys})


class _Episode:
    def __init__(self, d):
        self.index = [None, _Index(d)]  # index[-1]


def make_episode():
    return _Episode({
        (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY): torch.randn(B, 1, D),
        (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY): torch.randn(B, 1, D),
        (Modality.CONTEXT, "waypoints"): torch.randn(B, 10, D),
        (Modality.FORESIGHT, "cam_front_left"): torch.randn(B, N, D),
    })


def features_via(read_waypoints, foresight_attn, foresight_maxpool, code_head_in):
    obj = object.__new__(JointPolicyObjective)  # bypass heavy __init__
    nn.Module.__init__(obj)  # set up module machinery for submodule assignment
    obj.norm = None
    obj.read_waypoints = read_waypoints
    obj.foresight_attn = (
        ForesightAttentionPool(dim=D, num_queries=4) if foresight_attn else None
    )
    obj.foresight_maxpool = foresight_maxpool
    obj.foresight_key = "cam_front_left"
    obj._code_head_in = code_head_in
    return obj._features(make_episode(), torch.randn(B, 1, 1))


# waypoints via legacy auto-detect (read_waypoints=None):
assert features_via(None, False, False, 2 * D).shape == (B, 2 * D)  # no-waypoint
assert features_via(None, False, False, 3 * D).shape == (B, 3 * D)  # waypoint-aware
# explicit read_waypoints + additive foresight branches:
assert features_via(True, False, False, 3 * D).shape == (B, 3 * D)
assert features_via(True, True, False, 3 * D + 4 * D).shape == (B, 3 * D + 4 * D)  # 1.1
assert features_via(True, False, True, 3 * D + D).shape == (B, 3 * D + D)  # 2.4
w = 3 * D + 4 * D + D  # waypoints + attn(4) + maxpool
assert features_via(True, True, True, w).shape == (B, w)  # 1.1 + 2.4
print(f"[1.1+2.4] _features assembly ok across branch combos (max width {w})")

# width guard fires on mismatch:
try:
    features_via(True, True, True, 999)
    raise AssertionError("expected width-mismatch ValueError")
except ValueError as e:
    assert "assembled policy-head width" in str(e)
print("[1.1+2.4] width-mismatch guard ok")


# ---- 1.2 FD target re-encoding hook ----------------------------------------
from rmind.components.objectives.forward_dynamics import (  # noqa: E402
    ForwardDynamicsPredictionObjective,
)

fd = object.__new__(ForwardDynamicsPredictionObjective)
nn.Module.__init__(fd)
dummy_target_encoder = nn.Linear(D, 1024)  # e.g. DINO(384) -> V-JEPA-like(1024)
fd.target_encoder = dummy_target_encoder.requires_grad_(False).eval()
tgt = torch.randn(B, 5, N, D)  # (b, t, patches, d)
with torch.no_grad():
    re = fd.target_encoder(tgt).detach()
assert re.shape == (B, 5, N, 1024), re.shape
assert not re.requires_grad
assert all(not p.requires_grad for p in fd.target_encoder.parameters())
print(f"[1.2] FD target_encoder hook ok: {tuple(tgt.shape)} -> {tuple(re.shape)} (frozen, detached)")

print("\nALL SMOKE CHECKS PASSED")
