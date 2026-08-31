"""The anti-overfitting package for the causal arms: stochastic depth,
label-smoothed focal code loss, and per-module weight-decay overrides."""

import pytest
import torch
from torch import nn

from rmind.components.loss import FocalLoss
from rmind.components.optimizers.selective_adamw import SelectiveAdamW
from rmind.components.transformer.causal_frame import CausalFrameTransformer, DropPath
from rmind.models.patch_policy import PatchPolicy
from tests.test_patch_policy import (
    EPISODE_LENGTH,
    NUM_PATCHES,
    POLICY_DIM,
    _make_batch,
    _make_model,
)

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


# ---------------------------------------------------------------------------
# The pre-launch gate: a full PatchPolicy training step through the causal
# trunk with ALL THREE regularizers active, backward, and a SelectiveAdamW
# step with the code_head decay override. Runs on CPU with sdpa; the flex
# variant needs a GPU (same gate, attention_impl="flex").
# ---------------------------------------------------------------------------


def _causal_regularized_model(attention_impl: str) -> PatchPolicy:
    model = _make_model()
    model.encoder = CausalFrameTransformer(
        dim_model=POLICY_DIM,
        num_layers=2,
        # FlexAttention requires head_dim >= 16 (production: 512/8 = 64);
        # 1 head keeps the toy POLICY_DIM=16 legal on the flex path
        num_heads=1,
        tokens_per_frame=NUM_PATCHES + 1,
        window=2,
        max_sequence_length=EPISODE_LENGTH * (NUM_PATCHES + 1),
        attn_dropout=0.0,
        attention_impl=attention_impl,  # type: ignore[arg-type]
        drop_path_rate=0.1,
    )
    model.losses["code"] = FocalLoss(label_smoothing=0.1)
    return model


def _train_step(model: PatchPolicy, device: str) -> None:
    model = model.to(device).train()
    opt = SelectiveAdamW(
        model,
        lr=1e-4,
        weight_decay=0.1,
        weight_decay_overrides={"code_head": 0.2},
        weight_decay_module_blacklist=(nn.LayerNorm, nn.Embedding),
    )
    batch = {}
    src = _make_batch()

    def _to(x: object) -> object:
        return (
            {k: _to(v) for k, v in x.items()} if isinstance(x, dict) else x.to(device)
        )  # type: ignore[union-attr]

    batch = _to(src)
    loss = model._compute_metrics(batch)["policy", "loss"].sum(reduce=True)  # noqa: SLF001
    assert torch.isfinite(loss)
    loss.backward()
    grads = {
        n: p.grad is not None and bool(torch.isfinite(p.grad).all())
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    assert grads
    assert all(grads.values()), [n for n, ok in grads.items() if not ok]
    assert any("encoder.intra_position_embedding" in n for n in grads)
    assert any(n.startswith("code_head") for n in grads)
    opt.step()
    opt.zero_grad()


def test_full_model_train_step_causal_sdpa_cpu() -> None:
    torch.manual_seed(0)
    _train_step(_causal_regularized_model("sdpa"), "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex needs a GPU")
def test_full_model_train_step_causal_flex_gpu() -> None:
    torch.manual_seed(0)
    _train_step(_causal_regularized_model("flex"), "cuda")


def test_context_depth_bucket_metrics_present() -> None:
    # window=2, episode_length=3: position 0 is partial-window, 1..2 full
    model = _causal_regularized_model("sdpa").eval()
    metrics = model._compute_metrics(_make_batch())  # noqa: SLF001
    for bucket in ("partial_window", "full_window"):
        assert ("policy", "metric", f"code_{bucket}") in metrics.keys(
            include_nested=True
        )
        assert ("policy", "metric", f"offset_{bucket}") in metrics.keys(
            include_nested=True
        )


def test_context_depth_buckets_absent_for_block_causal_trunk() -> None:
    metrics = _make_model()._compute_metrics(_make_batch())  # noqa: SLF001
    assert ("policy", "metric", "code_full_window") not in metrics.keys(
        include_nested=True
    )
