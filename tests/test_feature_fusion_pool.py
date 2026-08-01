import torch

from rmind.components.nn import FeatureFusionPool
from rmind.components.transformer import AttentionPoolHead, CrossAttentionDecoder

EMBEDDING_DIM = 8
NUM_WAYPOINTS = 3
HISTORY_STEPS = 2
BATCH = 4


def _make_pool(
    *,
    history_steps: int = HISTORY_STEPS,
    waypoint_token_dropout: float = 0.0,
    speed_token_dropout: float = 0.0,
) -> FeatureFusionPool:
    torch.manual_seed(0)
    return FeatureFusionPool(
        embedding_dim=EMBEDDING_DIM,
        num_waypoints=NUM_WAYPOINTS,
        history_steps=history_steps,
        waypoint_token_dropout=waypoint_token_dropout,
        speed_token_dropout=speed_token_dropout,
        pool=AttentionPoolHead(
            decoder=CrossAttentionDecoder(
                dim_model=EMBEDDING_DIM,
                num_layers=1,
                num_heads=2,
                # zeroed so the *pool's own* dropout can't confound the
                # token-dropout-specific assertions below
                attn_dropout=0.0,
                resid_dropout=0.0,
                mlp_dropout=0.0,
                hidden_layer_multiplier=1,
            ),
            embedding_dim=EMBEDDING_DIM,
            num_queries=1,
        ),
    )


def _inputs(*, history_steps: int = HISTORY_STEPS) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "obs_summary_history": torch.randn(BATCH, history_steps, EMBEDDING_DIM),
        "raw_waypoints": torch.randn(BATCH, NUM_WAYPOINTS, 2),
        "raw_speed": torch.randn(BATCH, 1),
    }


def test_token_dropout_is_noop_at_eval() -> None:
    pool = _make_pool(waypoint_token_dropout=1.0, speed_token_dropout=1.0).eval()
    inputs = _inputs()

    out_dropout = pool(**inputs)
    pool.waypoint_token_dropout = 0.0
    pool.speed_token_dropout = 0.0
    out_no_dropout = pool(**inputs)

    torch.testing.assert_close(out_dropout, out_no_dropout)


def test_waypoint_token_dropout_is_stochastic_in_train() -> None:
    pool = _make_pool(waypoint_token_dropout=0.5).train()
    inputs = _inputs()

    torch.manual_seed(2)
    first = pool(**inputs)
    torch.manual_seed(3)
    second = pool(**inputs)

    assert not torch.allclose(first, second)


def test_speed_token_dropout_is_stochastic_in_train() -> None:
    pool = _make_pool(speed_token_dropout=0.5).train()
    inputs = _inputs()

    torch.manual_seed(2)
    first = pool(**inputs)
    torch.manual_seed(3)
    second = pool(**inputs)

    assert not torch.allclose(first, second)


def test_obs_summary_tokens_are_never_dropped(monkeypatch) -> None:
    # torch.rand -> 0 everywhere makes every "< dropout" comparison True, so
    # waypoint/speed tokens are all masked; obs_summary tokens (history_steps=2)
    # stay unmasked, so the all-masked guard never triggers and there's no NaN.
    pool = _make_pool(waypoint_token_dropout=1.0, speed_token_dropout=1.0).train()
    inputs = _inputs()

    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(*args, **kwargs))

    out = pool(**inputs)

    assert not out.isnan().any()


def test_fully_masked_row_guard_has_no_nan(monkeypatch) -> None:
    # with history_steps=0 there are no obs_summary tokens to fall back on, so
    # waypoint_token_dropout=speed_token_dropout=1.0 masks every token -> the
    # explicit unmask-one-token guard must kick in to avoid a NaN softmax.
    pool = _make_pool(
        history_steps=0, waypoint_token_dropout=1.0, speed_token_dropout=1.0
    ).train()
    inputs = _inputs(history_steps=0)

    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(*args, **kwargs))

    out = pool(**inputs)

    assert not out.isnan().any()
