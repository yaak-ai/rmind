from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from rmind.components.backbone_registers import RegisterViTBackbone
from rmind.components.lora import LoRALinear, apply_lora, merge_lora
from rmind.components.optimizers import SelectiveAdamW
from tests.test_patch_policy import (
    BATCH_SIZE,
    EPISODE_LENGTH,
    NUM_PATCHES,
    _make_batch,
    _make_model,
)

if TYPE_CHECKING:
    from tensordict import TensorDict

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


# --- per-camera registers (DrivoR's own formulation) ------------------------


@pytest.fixture(scope="module")
def register_backbone_vit_per_camera(device: torch.device) -> RegisterViTBackbone:
    return RegisterViTBackbone(
        model_name="vit_small_patch14_dinov2.lvd142m",
        img_size=[224, 224],
        num_registers=4,
        num_cameras=3,
        lora_rank=4,
        lora_alpha=4.0,
    ).to(device)


def test_backbone_shared_registers_are_one_set(
    register_backbone_vit: RegisterViTBackbone,
) -> None:
    """`num_cameras=None` (the default) keeps ONE set for every view."""
    assert register_backbone_vit.num_cameras is None
    assert register_backbone_vit.camera_reg_token.shape == (
        1,
        4,
        register_backbone_vit.model.embed_dim,
    )


def test_backbone_per_camera_register_shape(
    register_backbone_vit_per_camera: RegisterViTBackbone,
) -> None:
    backbone = register_backbone_vit_per_camera
    assert backbone.camera_reg_token.shape == (3, 4, backbone.model.embed_dim)
    # independently drawn, not one set copied -- what lets the cameras diverge
    assert not torch.allclose(
        backbone.camera_reg_token[0], backbone.camera_reg_token[1]
    )


@torch.inference_mode()
def test_backbone_per_camera_register_routing(
    device: torch.device, register_backbone_vit_per_camera: RegisterViTBackbone
) -> None:
    """Flattened image `i` must get camera `i % num_cameras`'s registers.

    `forward` reshapes `(b, cam, c, h, w)` row-major, so the camera axis varies
    fastest. Feeding the SAME image at every `(b, cam)` slot makes the registers
    the only thing that can differ, which pins the routing in both directions:
    a `i // num_cameras` mapping (what `expand` would give) breaks the
    same-camera-across-batch equality AND the across-camera inequality.
    """
    backbone = register_backbone_vit_per_camera.eval()
    # the N(0, 1e-6) init makes every camera's output identical to fp tolerance;
    # separate them by hand so the routing is observable at all
    original = backbone.camera_reg_token.clone()
    backbone.camera_reg_token.copy_(
        torch.stack([torch.full_like(original[0], 0.1 * (cam + 1)) for cam in range(3)])
    )

    try:
        image = torch.randn(3, 224, 224, device=device)
        out = backbone(image.expand(2, 3, 3, 224, 224).contiguous())
        assert out.shape == (2, 3, 4, backbone.model.embed_dim)

        # same camera, different batch row: nothing differs, so must be equal
        assert_close(out[0], out[1])
        # different camera: only the registers differ, so must NOT be equal
        for i, j in ((0, 1), (0, 2), (1, 2)):
            assert not torch.allclose(out[0, i], out[0, j])
    finally:
        backbone.camera_reg_token.copy_(original)


@pytest.mark.parametrize(
    "shape",
    [(2, 2, 3, 224, 224), (3, 224, 224)],
    ids=["wrong_camera_count", "no_camera_axis"],
)
@torch.inference_mode()
def test_backbone_per_camera_rejects_bad_camera_axis(
    device: torch.device,
    register_backbone_vit_per_camera: RegisterViTBackbone,
    shape: tuple[int, ...],
) -> None:
    """Without the guard this hands images the WRONG camera's registers (or
    silently broadcasts), which is invisible at the shape level."""
    with pytest.raises(ValueError, match="camera axis LAST"):
        register_backbone_vit_per_camera(torch.randn(*shape, device=device))


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


@torch.inference_mode()
def test_merge_lora_backbone_export_parity(device: torch.device) -> None:
    """The export contract (plans/effervescent-chasing-seahorse.md, "Export"):
    `rmind.scripts.export_onnx` and `rmind.scripts.decoder_only_export` call
    `merge_lora` before tracing, so the served graph holds plain `nn.Linear`s
    and pays nothing for the adapters. Both run under `torch.inference_mode()`,
    which is why this test does too -- the in-place weight update has to be
    legal there.

    Built with non-trivial `lora_B`/registers: zero-init `lora_B` makes the
    whole merge a no-op, so a parity check on a fresh backbone would pass even
    if `merge_lora` did nothing at all.
    """
    backbone = (
        RegisterViTBackbone(
            model_name="vit_small_patch14_dinov2.lvd142m",
            img_size=[224, 224],
            num_registers=4,
            num_cameras=3,
            lora_rank=4,
            lora_alpha=4.0,
        )
        .to(device)
        .eval()
    )
    for name, param in backbone.named_parameters():
        if name.endswith("lora_B"):
            param.normal_(std=0.02)
    backbone.camera_reg_token.normal_(std=0.02)

    x = torch.randn(1, 3, 3, 224, 224, device=device)
    before = backbone(x).clone()

    merged = merge_lora(backbone)

    # qkv + proj in every block -- `apply_lora`'s default target suffixes
    assert merged == 2 * len(backbone.model.blocks)
    assert not any(isinstance(m, LoRALinear) for m in backbone.modules())
    # rtol/atol above float32 default: the merged weight reassociates
    # `x @ W + scale * (x @ A.T) @ B.T` into `x @ (W + scale * B @ A).T`, and
    # the drift compounds over 12 blocks (and over tf32, which conftest enables)
    assert_close(backbone(x), before, rtol=1e-3, atol=1e-4)


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


# --- end-to-end: gradients actually reach the trainable encoder -------------
#
# plans/effervescent-chasing-seahorse.md, Verification 4. The risk this guards
# is `PatchPolicy.trainable_image_encoder`: the default arm wraps
# `image_encoder` in `torch.no_grad()` and forces it to `.eval()`, so a missed
# gate leaves the LoRA adapters and registers receiving nothing, silently,
# while the run still trains (the trunk and heads do). Nothing about the loss
# curve would show it.

_GRAD_CAMERAS = ("cam_front_left", "cam_left_forward")


def _image_batch(device: torch.device) -> dict:
    """`_make_batch` with the pre-extracted patch features replaced by RAW
    frames -- what a trainable encoder is actually fed."""
    batch = _make_batch(cameras=_GRAD_CAMERAS)
    generator = torch.Generator().manual_seed(0)
    batch["image"] = {
        camera: torch.randn(
            (BATCH_SIZE, EPISODE_LENGTH, 3, 224, 224), generator=generator
        ).to(device)
        for camera in _GRAD_CAMERAS
    }
    return {
        key: (
            {k: v.to(device) for k, v in value.items()}
            if isinstance(value, dict)
            else value.to(device)
        )
        if key != "image"
        else value
        for key, value in batch.items()
    }


def _policy_loss(model: object, batch: dict) -> torch.Tensor:
    return cast(
        "TensorDict",
        model._compute_metrics(batch)["policy", "loss"],  # noqa: SLF001
    ).sum(reduce=True)


def test_trainable_encoder_receives_gradient(device: torch.device) -> None:
    """LoRA + registers must get gradient from the POLICY loss, and the frozen
    base must not.

    `lora_A` is deliberately checked only after an optimizer step: `lora_B` is
    zero-initialized, so `dL/dA = scale * B^T @ ...` is exactly zero at step 0.
    That is correct LoRA behaviour, not a broken graph -- but it means a naive
    "every trainable param has non-zero grad at step 0" assertion would fail
    here, and the reverse (asserting it stays zero) would hide a real
    disconnection. Both steps are checked.
    """
    backbone = RegisterViTBackbone(
        model_name="vit_small_patch14_dinov2.lvd142m",
        img_size=[224, 224],
        num_registers=NUM_PATCHES,
        num_cameras=len(_GRAD_CAMERAS),
        lora_rank=4,
        lora_alpha=4.0,
    )
    model = _make_model(
        cameras=_GRAD_CAMERAS,
        image_encoder=backbone,
        trainable_image_encoder=True,
        image_dim=backbone.model.embed_dim,
    ).to(device)
    model.train()

    # the frozen base stays in eval even though the outer module is training
    assert not backbone.model.training

    batch = _image_batch(device)
    _policy_loss(model, batch).backward()

    assert backbone.camera_reg_token.grad is not None
    assert backbone.camera_reg_token.grad.abs().sum() > 0

    grads = {name: p.grad for name, p in backbone.named_parameters()}
    lora_b = {n: g for n, g in grads.items() if n.endswith("lora_B")}
    assert lora_b
    assert all(g is not None and g.abs().sum() > 0 for g in lora_b.values())

    # frozen base: no grad at all, not merely small
    assert all(
        g is None
        for name, g in grads.items()
        if "lora_" not in name and name != "camera_reg_token"
    )

    # zero-init lora_B => zero lora_A grad on the FIRST step, non-zero after
    lora_a = {n: g for n, g in grads.items() if n.endswith("lora_A")}
    assert lora_a
    assert all(g is not None and g.abs().sum() == 0 for g in lora_a.values())

    optimizer = SelectiveAdamW(
        model, weight_decay=0.1, weight_decay_module_blacklist=(nn.LayerNorm,)
    )
    optimizer.step()
    model.zero_grad(set_to_none=True)
    _policy_loss(model, batch).backward()

    lora_a_after = {
        name: p.grad
        for name, p in backbone.named_parameters()
        if name.endswith("lora_A")
    }
    assert all(g is not None and g.abs().sum() > 0 for g in lora_a_after.values()), (
        "lora_A never picked up gradient: the adapter is a permanent no-op"
    )


def test_frozen_encoder_default_receives_no_gradient(device: torch.device) -> None:
    """The control: `trainable_image_encoder=False` (every existing arm) must
    leave the encoder's params untouched, so this opt-in changes nothing for
    them."""
    backbone = RegisterViTBackbone(
        model_name="vit_small_patch14_dinov2.lvd142m",
        img_size=[224, 224],
        num_registers=NUM_PATCHES,
        num_cameras=len(_GRAD_CAMERAS),
        lora_rank=4,
        lora_alpha=4.0,
    )
    model = _make_model(
        cameras=_GRAD_CAMERAS,
        image_encoder=backbone,
        trainable_image_encoder=False,
        image_dim=backbone.model.embed_dim,
    ).to(device)
    model.train()

    assert not backbone.training  # forced to eval by PatchPolicy.train()

    _policy_loss(model, _image_batch(device)).backward()

    assert all(g is None for g in (p.grad for p in backbone.parameters()))
