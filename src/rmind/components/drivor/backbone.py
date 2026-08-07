from typing import TYPE_CHECKING, cast, override

import torch
from timm import create_model
from torch import Tensor, nn

from rmind.components.lora import apply_lora

if TYPE_CHECKING:
    from timm.models.eva import Eva


class RegisterViTBackbone(nn.Module):
    """Pretrained ViT backbone with an extra set of learnable "compression"
    register tokens, per DrivoR (arXiv:2601.05083).

    The pretrained base (`self.model`) is frozen and LoRA-adapted (rank
    `lora_rank`, on `attn.qkv`/`attn.proj` only); only the new `reg_token`
    parameter and the LoRA adapters are trainable.

    NOTE on why this can't be a naive "concat new tokens after `_pos_embed`":
    `vit_small_patch16_dinov3.lvd1689m` is backed by timm's `Eva` class (not
    `VisionTransformer`), and its factory config
    (`num_reg_tokens=4, use_rot_pos_emb=True, use_abs_pos_emb=False`) means:
      - the checkpoint already ships 4 of its OWN pretrained register/"storage"
        tokens (Meta's attention-sink registers, remapped from `storage_tokens`
        in `checkpoint_filter_fn`) plus a cls token, i.e. `num_prefix_tokens=5`
        baked into the model at construction time;
      - position information is rotary (RoPE), not an absolute `pos_embed`;
      - each block's `EvaAttention.forward` slices `q[:, :, :npt, :]` (left
        untouched) vs `q[:, :, npt:, :]` (RoPE applied), where `npt =
        self.num_prefix_tokens` is a plain int fixed at construction on every
        attention submodule -- it does NOT adapt to a longer sequence at
        forward time. `rope` itself is sized to exactly the patch-grid length
        (256 for a 256x256/patch16 input).
    Naively inserting new tokens between the existing prefix and the patches
    (without adjusting `npt`) would push them into the "patch" slice, where
    `apply_rot_embed_cat` would try to apply a 256-length rope tensor to a
    `256 + num_registers`-length slice -- a shape-mismatch crash, not a silent
    bug. The fix: insert the new registers right after the *existing* prefix
    tokens, and extend every block's `attn.num_prefix_tokens` by
    `num_registers` so RoPE continues to apply only to the real 256 patch
    tokens. This leaves the pretrained cls/reg tokens, rope, and patch
    embedding entirely untouched -- only attention's bookkeeping of "how many
    leading tokens to skip" changes.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        img_size: list[int] | None = None,
        num_registers: int = 16,
        lora_rank: int = 32,
        lora_alpha: float = 32.0,
        lora_target_suffixes: tuple[str, ...] = ("attn.qkv", "attn.proj"),
    ) -> None:
        super().__init__()

        # cast: this backbone specifically requires an Eva-backed model (RoPE,
        # native register tokens) -- see class docstring. `create_model`'s
        # return type is generically `nn.Module` since it supports any
        # registered timm model.
        self.model: Eva = cast(
            "Eva",
            create_model(model_name, pretrained=True, img_size=img_size, num_classes=0),
        )
        self.model.requires_grad_(False)  # noqa: FBT003

        self.num_registers = num_registers
        self.reg_token = nn.Parameter(
            torch.empty(1, num_registers, self.model.embed_dim)
        )
        # matches timm's own register-token init convention (see
        # `timm.models.vision_transformer.VisionTransformer.init_weights`)
        nn.init.normal_(self.reg_token, std=1e-6)

        blocks = cast("nn.ModuleList", self.model.blocks)
        apply_lora(
            blocks,
            target_suffixes=lora_target_suffixes,
            rank=lora_rank,
            alpha=lora_alpha,
        )

        # extend every block's RoPE prefix-token count so our new registers
        # (inserted in `forward`, right after the model's own prefix tokens)
        # are excluded from RoPE the same way the pretrained cls/reg tokens
        # are -- see class docstring.
        for block in blocks:
            block.attn.num_prefix_tokens += num_registers  # ty:ignore[unresolved-attribute, unsupported-operator]

    @override
    def train(self, mode: bool = True) -> "RegisterViTBackbone":
        super().train(mode)
        self.model.eval()  # frozen base always in eval mode, regardless of outer module state
        return self

    @override
    def forward(self, input: Tensor) -> Tensor:
        *b, c, h, w = input.shape
        x = input.reshape(-1, c, h, w)

        x = self.model.patch_embed(x)
        x, rope = self.model._pos_embed(x)  # noqa: SLF001
        x = self.model.norm_pre(x)

        n = x.shape[0]
        npt = (
            self.model.num_prefix_tokens
        )  # original prefix count (cls + pretrained registers), unmodified
        reg = self.reg_token.expand(n, -1, -1)
        x = torch.cat([x[:, :npt], reg, x[:, npt:]], dim=1)

        for block in cast("nn.ModuleList", self.model.blocks):
            x = block(x, rope=rope, attn_mask=None, is_causal=False)
        x = self.model.norm(x)

        reg_out = x[:, npt : npt + self.num_registers]
        return reg_out.reshape(*b, *reg_out.shape[-2:])
