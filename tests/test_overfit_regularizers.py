"""The anti-overfitting package for the causal arms: stochastic depth,
label-smoothed focal code loss, and per-module weight-decay overrides."""

import pytest
import torch
from torch import nn

from rmind.components.loss import FocalLoss
from rmind.components.optimizers.selective_adamw import SelectiveAdamW
from rmind.components.transformer.causal_frame import CausalFrameTransformer, DropPath

RESCALED = 2.0  # 1 / keep for drop_prob 0.5
DROP_RATE_BAND = (0.35, 0.65)
MEAN_TOL = 0.15
NEAR_ZERO = 1e-6


def test_drop_path_eval_is_exact_identity() -> None:
    dp = DropPath(0.5).eval()
    x = torch.randn(4, 3, 8)
    assert torch.equal(dp(x), x)


def test_drop_path_zero_rate_is_identity_in_train() -> None:
    dp = DropPath(0.0).train()
    x = torch.randn(4, 3, 8)
    assert torch.equal(dp(x), x)


def test_drop_path_drops_whole_samples_and_rescales() -> None:
    torch.manual_seed(0)
    dp = DropPath(0.5).train()
    x = torch.ones(512, 2, 4)
    y = dp(x)
    per_sample = y.flatten(1)
    zeroed = (per_sample == 0).all(dim=1)
    kept = (per_sample == RESCALED).all(dim=1)
    # every sample is either fully dropped or fully kept-and-rescaled
    assert (zeroed | kept).all()
    # unbiased in expectation: drop rate near 0.5, mean near 1
    lo, hi = DROP_RATE_BAND
    assert lo < zeroed.float().mean().item() < hi
    assert abs(y.mean().item() - 1.0) < MEAN_TOL


def test_drop_path_invalid_rate_raises() -> None:
    with pytest.raises(ValueError, match="drop_prob"):
        DropPath(1.0)


def test_trunk_ramp_is_linear_and_eval_output_unchanged() -> None:
    kwargs = {
        "dim_model": 64,
        "num_layers": 4,
        "num_heads": 4,
        "tokens_per_frame": 5,
        "window": 2,
    }
    torch.manual_seed(7)
    plain = CausalFrameTransformer(**kwargs)
    torch.manual_seed(7)
    reg = CausalFrameTransformer(**kwargs, drop_path_rate=0.1)

    rates = [blk.drop_path.drop_prob for blk in reg.layers]
    assert rates == pytest.approx([0.0, 0.1 / 3, 0.2 / 3, 0.1])

    # same seed -> same weights; in eval, drop-path must be invisible
    x = torch.randn(2, 3 * 5, 64)
    out_plain = plain.eval()(x, num_frames=3)
    out_reg = reg.eval()(x, num_frames=3)
    torch.testing.assert_close(out_plain, out_reg, rtol=0, atol=0)


def test_trunk_train_mode_actually_drops() -> None:
    torch.manual_seed(3)
    trunk = CausalFrameTransformer(
        dim_model=64,
        num_layers=4,
        num_heads=4,
        tokens_per_frame=5,
        window=2,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        drop_path_rate=0.9,
    ).train()
    x = torch.randn(8, 3 * 5, 64)
    a, b = trunk(x, num_frames=3), trunk(x, num_frames=3)
    assert not torch.allclose(a, b)


def test_zero_smoothing_matches_legacy_exactly() -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, 16)
    target = torch.randint(0, 16, (64,))
    new = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, target)
    # legacy formula, inlined
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    legacy = ((1 - pt).pow(2.0) * ce).mean()
    torch.testing.assert_close(new, legacy, rtol=0, atol=0)


def test_smoothing_penalizes_overconfidence() -> None:
    # near-one-hot confident-correct predictions: unsmoothed focal ~ 0, and the
    # smoothed loss must NOT be cancelled by the (1-pt)^gamma factor -- growing
    # the margin must grow the penalty
    target = torch.arange(16)
    confident = torch.full((16, 16), -20.0)
    confident[torch.arange(16), target] = 20.0
    overconfident = confident * 2
    plain = FocalLoss(label_smoothing=0.0)(confident, target)
    smoothed = FocalLoss(label_smoothing=0.1)
    assert plain.item() < NEAR_ZERO
    assert smoothed(confident, target).item() > NEAR_ZERO
    assert smoothed(overconfident, target).item() > smoothed(confident, target).item()


def test_smoothed_focal_gradient_flows() -> None:
    logits = torch.randn(8, 16, requires_grad=True)
    loss = FocalLoss(label_smoothing=0.1)(logits, torch.randint(0, 16, (8,)))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 4)
        self.code_head = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
        self.norm = nn.LayerNorm(4)


def _opt(**kw: object) -> SelectiveAdamW:
    return SelectiveAdamW(
        _Toy(),
        lr=1e-4,
        weight_decay=0.1,
        weight_decay_module_blacklist=(nn.LayerNorm, nn.Embedding),
        **kw,
    )


def test_override_group_gets_its_decay() -> None:
    opt = _opt(weight_decay_overrides={"code_head": 0.2})
    decays = [g["weight_decay"] for g in opt.param_groups]
    sizes = [len(g["params"]) for g in opt.param_groups]
    # [blacklist 0.0, whitelist 0.1, code_head 0.2]
    assert decays == [0.0, 0.1, 0.2]
    # code_head has its 2 Linear weights here; biases stay in the 0.0 group
    assert sizes[2] == len([
        n for n, _ in _Toy().named_parameters() if "code_head" in n and "weight" in n
    ])


BASE_GROUPS = 2  # [no-decay blacklist, decayed whitelist]


def test_no_override_is_two_groups() -> None:
    assert len(_opt().param_groups) == BASE_GROUPS


def test_unmatched_override_prefix_raises() -> None:
    with pytest.raises(ValueError, match="matched no decayed params"):
        _opt(weight_decay_overrides={"nonexistent_head": 0.2})


def test_overlapping_override_prefixes_raise() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _opt(weight_decay_overrides={"code_head": 0.2, "code_head.0": 0.3})
