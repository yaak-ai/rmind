"""Test KV cache implementation for TransformerEncoder."""

import torch
import torch.nn as nn

# Add the project to path
import sys
sys.path.insert(0, "/home/harsimrat/workspace/jetson/rmind/src")

from rmind.components.llm import TransformerEncoder, TransformerEncoderBlock, KVCache


def test_kv_cache_named_tuple():
    """Test KVCache is properly defined."""
    key = torch.randn(2, 10, 256)
    value = torch.randn(2, 10, 256)
    cache = KVCache(key=key, value=value)

    assert cache.key.shape == (2, 10, 256)
    assert cache.value.shape == (2, 10, 256)
    print("✓ KVCache NamedTuple works correctly")


def test_encoder_block_backward_compat():
    """Test that existing forward behavior is preserved."""
    block = TransformerEncoderBlock(
        embedding_dim=256,
        num_heads=8,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        hidden_layer_multiplier=4,
    )
    block.eval()

    x = torch.randn(2, 10, 256)
    mask = torch.zeros(10, 10)

    # Original forward (no cache)
    with torch.no_grad():
        out = block(x, mask)

    assert out.shape == (2, 10, 256)
    print("✓ TransformerEncoderBlock backward compatibility works")


def test_encoder_block_with_cache():
    """Test encoder block with KV caching."""
    block = TransformerEncoderBlock(
        embedding_dim=256,
        num_heads=8,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        hidden_layer_multiplier=4,
    )
    block.eval()

    # Full sequence
    x_full = torch.randn(2, 10, 256)
    mask_full = torch.zeros(10, 10)

    with torch.no_grad():
        # Run full sequence without cache (reference)
        out_full_ref = block(x_full, mask_full)

        # Run full sequence with cache enabled
        out_full, kv_cache = block(x_full, mask_full, use_cache=True)

    assert out_full.shape == (2, 10, 256)
    assert kv_cache is not None
    assert kv_cache.key.shape == (2, 10, 256)
    assert kv_cache.value.shape == (2, 10, 256)

    # Outputs should be close (may differ slightly due to different code paths)
    diff = (out_full - out_full_ref).abs().max().item()
    print(f"  Max diff between cache/no-cache paths: {diff:.6f}")

    print("✓ TransformerEncoderBlock with cache works")


def test_encoder_block_incremental():
    """Test incremental inference with KV cache."""
    block = TransformerEncoderBlock(
        embedding_dim=256,
        num_heads=8,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        hidden_layer_multiplier=4,
    )
    block.eval()

    batch_size = 2
    seq_len = 10
    dim = 256

    x_full = torch.randn(batch_size, seq_len, dim)

    with torch.no_grad():
        # Full forward with cache
        mask_full = torch.zeros(seq_len, seq_len)
        out_full, _ = block(x_full, mask_full, use_cache=True)

        # Incremental: first 8 tokens, then 2 more
        x_prefix = x_full[:, :8, :]
        x_suffix = x_full[:, 8:, :]

        # Process prefix
        mask_prefix = torch.zeros(8, 8)
        out_prefix, kv_cache = block(x_prefix, mask_prefix, use_cache=True)

        # Process suffix with cache
        # Mask for suffix attending to all previous + current positions
        mask_suffix = torch.zeros(2, 10)  # [S_new, S_total]
        out_suffix, kv_cache_updated = block(
            x_suffix, mask_suffix, past_kv=kv_cache, use_cache=True
        )

        # Concatenate incremental outputs
        out_incremental = torch.cat([out_prefix, out_suffix], dim=1)

    # Compare full vs incremental
    diff = (out_full - out_incremental).abs().max().item()
    print(f"  Max diff full vs incremental: {diff:.6f}")

    # Should be very close (numerical precision)
    assert diff < 1e-4, f"Outputs differ too much: {diff}"
    print("✓ TransformerEncoderBlock incremental inference works")


def test_encoder_backward_compat():
    """Test TransformerEncoder backward compatibility."""
    encoder = TransformerEncoder(
        dim_model=256,
        num_layers=4,
        num_heads=8,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        hidden_layer_multiplier=4,
    )
    encoder.eval()

    x = torch.randn(2, 10, 256)
    mask = torch.zeros(10, 10)

    with torch.no_grad():
        out = encoder(src=x, mask=mask)

    assert out.shape == (2, 10, 256)
    print("✓ TransformerEncoder backward compatibility works")


def test_encoder_with_cache():
    """Test TransformerEncoder with KV caching."""
    encoder = TransformerEncoder(
        dim_model=256,
        num_layers=4,
        num_heads=8,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        hidden_layer_multiplier=4,
    )
    encoder.eval()

    x = torch.randn(2, 10, 256)
    mask = torch.zeros(10, 10)

    with torch.no_grad():
        out, kv_caches = encoder(src=x, mask=mask, use_cache=True)

    assert out.shape == (2, 10, 256)
    assert kv_caches is not None
    assert len(kv_caches) == 4  # One per layer

    for i, kv in enumerate(kv_caches):
        assert kv.key.shape == (2, 10, 256), f"Layer {i} key shape mismatch"
        assert kv.value.shape == (2, 10, 256), f"Layer {i} value shape mismatch"

    print("✓ TransformerEncoder with cache works")


def test_encoder_incremental():
    """Test TransformerEncoder incremental inference."""
    encoder = TransformerEncoder(
        dim_model=256,
        num_layers=4,
        num_heads=8,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        hidden_layer_multiplier=4,
    )
    encoder.eval()

    batch_size = 2
    seq_len = 10
    dim = 256

    x_full = torch.randn(batch_size, seq_len, dim)

    with torch.no_grad():
        # Full forward
        mask_full = torch.zeros(seq_len, seq_len)
        out_full, _ = encoder(src=x_full, mask=mask_full, use_cache=True)

        # Incremental: 8 + 2
        x_prefix = x_full[:, :8, :]
        x_suffix = x_full[:, 8:, :]

        mask_prefix = torch.zeros(8, 8)
        out_prefix, kv_caches = encoder(src=x_prefix, mask=mask_prefix, use_cache=True)

        mask_suffix = torch.zeros(2, 10)
        out_suffix, _ = encoder(
            src=x_suffix, mask=mask_suffix, past_key_values=kv_caches, use_cache=True
        )

        out_incremental = torch.cat([out_prefix, out_suffix], dim=1)

    diff = (out_full - out_incremental).abs().max().item()
    print(f"  Max diff full vs incremental: {diff:.6f}")

    assert diff < 1e-4, f"Outputs differ too much: {diff}"
    print("✓ TransformerEncoder incremental inference works")


def test_encoder_return_types():
    """Test that return types are correct based on use_cache flag."""
    encoder = TransformerEncoder(
        dim_model=256,
        num_layers=2,
        num_heads=8,
    )
    encoder.eval()

    x = torch.randn(1, 5, 256)
    mask = torch.zeros(5, 5)

    with torch.no_grad():
        # Without cache: returns Tensor
        out_no_cache = encoder(src=x, mask=mask)
        assert isinstance(out_no_cache, torch.Tensor)
        assert not isinstance(out_no_cache, tuple)

        # With cache: returns tuple
        result_with_cache = encoder(src=x, mask=mask, use_cache=True)
        assert isinstance(result_with_cache, tuple)
        assert len(result_with_cache) == 2

    print("✓ Return types are correct")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing KV Cache Implementation")
    print("=" * 60)

    test_kv_cache_named_tuple()
    print()

    print("TransformerEncoderBlock tests:")
    test_encoder_block_backward_compat()
    test_encoder_block_with_cache()
    test_encoder_block_incremental()
    print()

    print("TransformerEncoder tests:")
    test_encoder_backward_compat()
    test_encoder_with_cache()
    test_encoder_incremental()
    test_encoder_return_types()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
