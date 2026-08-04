from typing import override

import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from rmind.callbacks.lora import LoraInjector
from rmind.components.lora import LoRALinear, convert_multihead_attention
from rmind.components.transformer.encoder import FactorizedTransformerEncoderBlock


@pytest.fixture(scope="module")
def trainer() -> pl.Trainer:
    return pl.Trainer(
        logger=False, enable_progress_bar=False, enable_model_summary=False
    )


def test_lora_linear_is_zero_init_noop() -> None:
    base = nn.Linear(8, 4)
    x = torch.randn(3, 8)
    expected = base(x)

    wrapped = LoRALinear(base, r=2, alpha=4, dropout=0.0)

    assert torch.allclose(wrapped(x), expected)


def test_lora_linear_freezes_base_keeps_lora_trainable() -> None:
    base = nn.Linear(8, 4)
    wrapped = LoRALinear(base, r=2, alpha=4, dropout=0.0)

    assert not wrapped.base.weight.requires_grad
    assert not wrapped.base.bias.requires_grad
    assert wrapped.lora_A.requires_grad
    assert wrapped.lora_B.requires_grad


def test_convert_multihead_attention_matches_original() -> None:
    torch.manual_seed(0)
    embed_dim, num_heads, seq_len, batch = 8, 2, 5, 3
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, batch_first=True
    ).eval()

    packed = convert_multihead_attention(mha).eval()

    block_probability = 0.5
    x = torch.randn(batch, seq_len, embed_dim)
    mask = torch.rand(seq_len, seq_len) > block_probability
    mask.fill_diagonal_(False)  # noqa: FBT003

    expected, _ = mha(query=x, key=x, value=x, attn_mask=mask, need_weights=False)
    actual = packed(x, mask)

    assert torch.allclose(actual, expected, atol=1e-5)


class ToyEncoderModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FactorizedTransformerEncoderBlock(
            embedding_dim=8, num_heads=2, rope=None
        )
        self.head = nn.Linear(8, 2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


@pytest.fixture
def module() -> ToyEncoderModule:
    return ToyEncoderModule()


def _block_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, t, s, d = 2, 3, 4, 8
    x = torch.randn(b, t, s, d)
    spatial_mask = torch.zeros(s, s, dtype=torch.bool)
    temporal_mask = torch.zeros(t, t, dtype=torch.bool)
    return x, spatial_mask, temporal_mask


def test_lora_injector_is_noop_at_injection_time(
    trainer: pl.Trainer, module: ToyEncoderModule
) -> None:
    module.eval()
    x, spatial_mask, temporal_mask = _block_inputs()
    before = module.encoder(x, spatial_mask, temporal_mask)

    LoraInjector(paths={"encoder"}, r=2, alpha=4, dropout=0.0).setup(
        trainer, module, "fit"
    )

    after = module.encoder(x, spatial_mask, temporal_mask)
    assert torch.allclose(after, before, atol=1e-5)


def test_lora_injector_freezes_base_leaves_lora_trainable(
    trainer: pl.Trainer, module: ToyEncoderModule
) -> None:
    LoraInjector(paths={"encoder"}, r=2, alpha=4, dropout=0.05).setup(
        trainer, module, "fit"
    )

    lora_params = [
        p
        for name, p in module.encoder.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    base_params = [
        p
        for name, p in module.encoder.named_parameters()
        if "lora_A" not in name and "lora_B" not in name
    ]

    assert lora_params
    assert all(p.requires_grad for p in lora_params)
    assert base_params
    assert not any(p.requires_grad for p in base_params)

    assert all(p.requires_grad for p in module.head.parameters())

    assert isinstance(module.encoder.temporal_mha.attn.qkv, LoRALinear)
    assert isinstance(module.encoder.temporal_mha.attn.out_proj, LoRALinear)
    assert isinstance(module.encoder.spatial_mha.attn.qkv, LoRALinear)
    assert isinstance(module.encoder.spatial_mha.attn.out_proj, LoRALinear)
    assert isinstance(module.encoder.mlp.l1, LoRALinear)
    assert isinstance(module.encoder.mlp.l2, LoRALinear)
