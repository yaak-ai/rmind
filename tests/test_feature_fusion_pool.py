import torch

from rmind.components.nn import FeatureFusionPool
from rmind.components.transformer import AttentionPoolHead, CrossAttentionDecoder

EMBEDDING_DIM = 8
NUM_WAYPOINTS = 3
HISTORY_STEPS = 2
NUM_RAW_PATCHES = 5
NUM_IMAGE_PATCH_TOKENS = 2
BATCH = 4


def _make_image_patch_pool() -> AttentionPoolHead:
    return AttentionPoolHead(
        decoder=CrossAttentionDecoder(
            dim_model=EMBEDDING_DIM,
            num_layers=1,
            num_heads=2,
            attn_dropout=0.0,
            resid_dropout=0.0,
            mlp_dropout=0.0,
            hidden_layer_multiplier=1,
        ),
        embedding_dim=EMBEDDING_DIM,
        num_queries=NUM_IMAGE_PATCH_TOKENS,
    )


def _make_pool(
    *,
    history_steps: int = HISTORY_STEPS,
    waypoint_token_dropout: float = 0.0,
    speed_token_dropout: float = 0.0,
    image_patch_token_dropout: float = 0.0,
    with_image_patches: bool = False,
) -> FeatureFusionPool:
    torch.manual_seed(0)
    return FeatureFusionPool(
        embedding_dim=EMBEDDING_DIM,
        num_waypoints=NUM_WAYPOINTS,
        history_steps=history_steps,
        waypoint_token_dropout=waypoint_token_dropout,
        speed_token_dropout=speed_token_dropout,
        image_patch_pool=_make_image_patch_pool() if with_image_patches else None,
        num_image_patch_tokens=NUM_IMAGE_PATCH_TOKENS if with_image_patches else 0,
        image_patch_token_dropout=image_patch_token_dropout,
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


def _inputs(
    *, history_steps: int = HISTORY_STEPS, with_image_patches: bool = False
) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    inputs = {
        "obs_summary_history": torch.randn(BATCH, history_steps, EMBEDDING_DIM),
        "raw_waypoints": torch.randn(BATCH, NUM_WAYPOINTS, 2),
        "raw_speed": torch.randn(BATCH, 1),
    }
    if with_image_patches:
        inputs["raw_image_patches"] = torch.randn(BATCH, NUM_RAW_PATCHES, EMBEDDING_DIM)
    return inputs


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


def test_image_patches_are_ignored_without_image_patch_pool() -> None:
    # image_patch_pool=None (default) must ignore raw_image_patches entirely,
    # not error -- forward() only requires it when image_patch_pool is set.
    pool = _make_pool().eval()
    inputs = _inputs()

    out = pool(**inputs)

    torch.testing.assert_close(out, pool(**inputs, raw_image_patches=None))


def test_image_patch_pool_requires_raw_image_patches() -> None:
    pool = _make_pool(with_image_patches=True).eval()
    inputs = _inputs()  # no raw_image_patches

    try:
        pool(**inputs)
    except ValueError:
        pass
    else:
        msg = "expected ValueError when image_patch_pool is set but raw_image_patches is None"
        raise AssertionError(msg)


def test_image_patch_pool_compresses_patches_to_fixed_token_count() -> None:
    pool = _make_pool(with_image_patches=True).eval()
    inputs = _inputs(with_image_patches=True)

    out = pool(**inputs)

    assert out.shape == (BATCH, 1, EMBEDDING_DIM)
    assert not out.isnan().any()


def test_image_patch_token_dropout_is_noop_at_eval() -> None:
    pool = _make_pool(with_image_patches=True, image_patch_token_dropout=1.0).eval()
    inputs = _inputs(with_image_patches=True)

    out_dropout = pool(**inputs)
    pool.image_patch_token_dropout = 0.0
    out_no_dropout = pool(**inputs)

    torch.testing.assert_close(out_dropout, out_no_dropout)


def test_image_patch_token_dropout_is_stochastic_in_train() -> None:
    pool = _make_pool(with_image_patches=True, image_patch_token_dropout=0.5).train()
    inputs = _inputs(with_image_patches=True)

    torch.manual_seed(2)
    first = pool(**inputs)
    torch.manual_seed(3)
    second = pool(**inputs)

    assert not torch.allclose(first, second)


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
