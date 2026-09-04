import pytest
import torch
from torch import nn
from torch.testing import assert_close

from rmind.components.backbone_registers import RegisterViTBackbone
from rmind.components.lora import LoRALinear, apply_lora, merge_lora
from rmind.components.optimizers import SelectiveAdamW

# --- RegisterViTBackbone -- DINOv2 branch (plain VisionTransformer): the arm
# this repo actually uses (plans/effervescent-chasing-seahorse.md). Needs
# network access to download the pretrained checkpoint from HF Hub, same as
# `TimmBackbone` already does for the `episode_builder` fixture in
# tests/conftest.py. ---


@pytest.fixture(scope="module")
def register_backbone_vit(device: torch.device) -> RegisterViTBackbone:
    return RegisterViTBackbone(
        model_name="vit_small_patch14_dinov2.lvd142m",
        img_size=[224, 224],
        num_registers=4,
        lora_rank=4,
        lora_alpha=4.0,
    ).to(device)


def test_backbone_forward_shape_vit(
    device: torch.device, register_backbone_vit: RegisterViTBackbone
) -> None:
    x = torch.randn(2, 3, 224, 224, device=device)
    out = register_backbone_vit(x)
    assert out.shape == (2, 4, register_backbone_vit.model.embed_dim)


def test_backbone_forward_shape_vit_leading_dims(
    device: torch.device, register_backbone_vit: RegisterViTBackbone
) -> None:
    """`(b, t, cam, c, h, w)`-style leading dims collapse to `(..., R, d)` --
    the shape `PatchPolicy._frame_tokens` actually feeds `image_encoder`."""
    x = torch.randn(2, 3, 3, 3, 224, 224, device=device)
    out = register_backbone_vit(x)
    assert out.shape == (2, 3, 3, 4, register_backbone_vit.model.embed_dim)


def test_backbone_freeze_contract_vit(
    register_backbone_vit: RegisterViTBackbone,
) -> None:
    for name, p in register_backbone_vit.model.named_parameters():
        if "lora_" not in name:
            assert not p.requires_grad, name

    assert register_backbone_vit.camera_reg_token.requires_grad

    lora_param_names = [
        name
        for name, _ in register_backbone_vit.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    assert lora_param_names
    for name, p in register_backbone_vit.named_parameters():
        if name in lora_param_names:
            assert p.requires_grad, name


def test_backbone_timm_internals_contract_vit(
    register_backbone_vit: RegisterViTBackbone,
) -> None:
    """DINOv2 ViT-S is a plain `VisionTransformer`: NO per-block
    `num_prefix_tokens` patching -- that only applies to the Eva/DINOv3 branch
    (see `RegisterViTBackbone._forward_eva`'s docstring and
    `test_backbone_timm_internals_contract_eva` below)."""
    model = register_backbone_vit.model
    assert hasattr(model, "_pos_embed")
    assert hasattr(model, "norm_pre")
    assert hasattr(model, "norm")
    assert hasattr(model, "patch_embed")
    assert hasattr(model, "num_prefix_tokens")
    # NOTE: unlike Eva's `blocks` (a ModuleList, see the Eva-branch test
    # below), timm's `VisionTransformer.blocks` is an `nn.Sequential` --
    # iterable the same way, just not the same container type
    assert isinstance(model.blocks, nn.Sequential)
    for block in model.blocks:
        # plain `Attention` has no per-block prefix-token bookkeeping AT ALL
        # (unlike `EvaAttention.num_prefix_tokens`) -- __init__ must not have
        # tried to patch it in
        assert not hasattr(block.attn, "num_prefix_tokens")


def test_backbone_train_keeps_base_frozen(
    register_backbone_vit: RegisterViTBackbone,
) -> None:
    register_backbone_vit.train()
    assert register_backbone_vit.training
    assert not register_backbone_vit.model.training
    register_backbone_vit.eval()  # leave the module-scoped fixture as found


# --- RegisterViTBackbone -- Eva branch (DINOv3, the default `model_name`):
# ported from feat/drivor unchanged. One smoke test that dispatch still
# reaches it correctly -- not full re-coverage, that lives on feat/drivor's
# tests/test_drivor.py. ---


@pytest.fixture(scope="module")
def register_backbone_eva(device: torch.device) -> RegisterViTBackbone:
    return RegisterViTBackbone(
        img_size=[256, 256], num_registers=4, lora_rank=4, lora_alpha=4.0
    ).to(device)


def test_backbone_forward_shape_eva(
    device: torch.device, register_backbone_eva: RegisterViTBackbone
) -> None:
    x = torch.randn(2, 3, 256, 256, device=device)
    out = register_backbone_eva(x)
    assert out.shape == (2, 4, register_backbone_eva.model.embed_dim)


def test_backbone_timm_internals_contract_eva(
    register_backbone_eva: RegisterViTBackbone,
) -> None:
    """The Eva/DINOv3 branch DOES patch `attn.num_prefix_tokens` -- RoPE only
    applies past the (now-extended) prefix, see `_forward_eva`'s docstring."""
    model = register_backbone_eva.model
    for block in model.blocks:
        assert (
            block.attn.num_prefix_tokens
            == model.num_prefix_tokens + register_backbone_eva.num_registers
        )


# --- merge_lora -------------------------------------------------------------


def test_merge_lora_numerical_parity(device: torch.device) -> None:
    base = nn.Linear(8, 8).to(device)
    base.requires_grad_(False)  # noqa: FBT003
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():  # non-zero lora_B so the merge isn't a trivial no-op
        wrapped.lora_B.normal_()

    x = torch.randn(4, 8, device=device)
    before = wrapped(x)

    container = nn.Module()
    container.linear = wrapped
    merge_lora(container)

    assert isinstance(container.linear, nn.Linear)
    assert not isinstance(container.linear, LoRALinear)
    after = container.linear(x)
    assert_close(after, before)


def test_apply_lora_target_suffixes() -> None:
    module = nn.ModuleDict({
        "attn": nn.ModuleDict({"qkv": nn.Linear(4, 4), "proj": nn.Linear(4, 4)}),
        "other": nn.Linear(4, 4),
    })
    apply_lora(module, target_suffixes=("attn.qkv", "attn.proj"), rank=2, alpha=2.0)
    assert isinstance(module["attn"]["qkv"], LoRALinear)
    assert isinstance(module["attn"]["proj"], LoRALinear)
    assert not isinstance(module["other"], LoRALinear)


# --- SelectiveAdamW: lora_A/lora_B vs. camera_reg_token dispatch ------------


def test_selective_adamw_lora_and_camera_registers(
    register_backbone_vit: RegisterViTBackbone,
) -> None:
    """The naming-collision guard (plans/effervescent-chasing-seahorse.md):
    `camera_reg_token` must land decay-free while `lora_A`/`lora_B` land in the
    normal decayed group -- and never in the same group as each other."""
    optimizer = SelectiveAdamW(
        register_backbone_vit,
        weight_decay=0.1,
        weight_decay_module_blacklist=(nn.LayerNorm, nn.Embedding),
    )
    param_name = {id(p): name for name, p in register_backbone_vit.named_parameters()}

    no_decay_group = next(
        g
        for g in optimizer.param_groups
        if any(param_name[id(p)] == "camera_reg_token" for p in g["params"])
    )
    decayed_group = next(
        g
        for g in optimizer.param_groups
        if any("lora_A" in param_name[id(p)] for p in g["params"])
    )

    assert no_decay_group["weight_decay"] == pytest.approx(0.0)
    assert decayed_group["weight_decay"] == pytest.approx(0.1)
    assert no_decay_group is not decayed_group
    # every lora_B lands in the SAME group as lora_A (both weight-like, both decayed)
    lora_b_names = {
        param_name[id(p)]
        for p in decayed_group["params"]
        if "lora_B" in param_name[id(p)]
    }
    assert lora_b_names
