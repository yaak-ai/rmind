"""CPU-only tests for the TransformerPolicyHead architecture knobs and the
WeightGradNormLogger norm helper. These deliberately pin device="cpu" so they can
run while the GPU is occupied by training.
"""

import torch
from torch import nn

from rmind.callbacks.weight_grad_norm import compute_weight_norms
from rmind.components.containers import ModuleDict
from rmind.components.transformer_policy_head import TransformerPolicyHead
from rmind.components.waypoint_pooler import _Block

CPU = torch.device("cpu")


def _head(**kwargs) -> TransformerPolicyHead:  # noqa: ANN003
    return TransformerPolicyHead(out_dim=2, **kwargs).to(CPU).eval()


def test_head_default_shape_and_zero_init() -> None:
    head = _head()
    x = torch.randn(4, 12, 384)
    out = head(x)
    assert out.shape == (4, 1, 2)
    assert torch.all(out == 0)


def test_head_deepnarrow_shape_and_zero_init() -> None:
    head = _head(dim_inner=192, num_layers=4)
    x = torch.randn(4, 12, 384)
    out = head(x)
    assert out.shape == (4, 1, 2)
    assert torch.all(out == 0)
    # blocks run at dim_inner
    assert head.input_projection is not None
    assert head.input_projection.out_features == 192
    assert head.output_projection.in_features == 192


def test_head_skip_shape_and_zero_init() -> None:
    head = _head(skip=True)
    x = torch.randn(4, 12, 384)
    out = head(x)
    assert out.shape == (4, 1, 2)
    assert torch.all(out == 0)


def test_head_skip_with_dim_inner_zero_init() -> None:
    head = _head(skip=True, dim_inner=192, num_layers=4)
    x = torch.randn(4, 12, 384)
    out = head(x)
    assert out.shape == (4, 1, 2)
    assert torch.all(out == 0)
    assert head.skip_projection is not None
    assert head.skip_projection.out_features == 192


def test_head_dropout_shape_and_zero_init() -> None:
    head = _head(dropout=0.3)
    x = torch.randn(4, 12, 384)
    out = head(x)
    assert out.shape == (4, 1, 2)
    assert torch.all(out == 0)


def test_block_dropout_keeps_state_dict_layout() -> None:
    # dropout knob must NOT renumber the MLP Sequential (mlp.0, mlp.2) or add keys,
    # so existing _Block checkpoints (shared with WaypointTransformerPooler) load.
    base = set(_Block(384, 4, 4).state_dict().keys())
    dropped = set(_Block(384, 4, 4, dropout=0.3).state_dict().keys())
    assert base == dropped
    assert any(k.startswith("mlp.2.") for k in base)


def test_compute_weight_norms_on_tiny_module() -> None:
    module = nn.Linear(4, 4).to(CPU)
    nn.init.ones_(module.weight)
    nn.init.zeros_(module.bias)
    norms = compute_weight_norms(module)
    assert "train/weight_norm/total" in norms
    # 16 ones => sqrt(16) = 4.0
    assert norms["train/weight_norm/total"] == 4.0


def test_compute_weight_norms_per_head_keys() -> None:
    heads = ModuleDict(
        modules={
            "continuous": {
                "gas_pedal": nn.Linear(2, 2),
                "brake_pedal": nn.Linear(2, 2),
                "steering_angle": nn.Linear(2, 2),
            },
            "discrete": {"turn_signal": nn.Linear(2, 3)},
        }
    )
    norms = compute_weight_norms(heads)
    assert "train/weight_norm/total" in norms
    assert "train/weight_norm/continuous/gas_pedal" in norms
    assert "train/weight_norm/continuous/brake_pedal" in norms
    assert "train/weight_norm/continuous/steering_angle" in norms
    assert "train/weight_norm/discrete/turn_signal" in norms
