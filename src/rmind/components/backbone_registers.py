from typing import TYPE_CHECKING, cast, override

import torch
from timm import create_model
from timm.models.eva import Eva
from torch import Tensor, nn

from rmind.components.lora import apply_lora
from rmind.components.transformer.utils import run_layer_stack

if TYPE_CHECKING:
    from timm.models.vision_transformer import VisionTransformer


class RegisterViTBackbone(nn.Module):
    """Pretrained ViT backbone with an extra set of learnable per-camera
    "compression" register tokens, per DrivoR (arXiv:2601.05083).

    `num_cameras` selects the register parameterization:

    - `None` (default): ONE shared set of `num_registers` tokens, used for every
      image regardless of which camera it came from. Camera identity then has to
      come from downstream -- in the `PatchPolicy` arms, the trunk's intra-frame
      position table, whose camera bands already carry it.
    - `int`: a SEPARATE set per camera, `(num_cameras, num_registers, D)`, each
      independently initialized. This is DrivoR's own formulation ("registers
      are per-camera, not shared -- this is what gives the output tokens their
      camera identity"), and it makes the per-camera register cosine-similarity
      diagnostic (DrivoR Fig. 4) meaningful, which the shared variant cannot be.
      Requires the camera axis to be the LAST leading dim of `forward`'s input,
      i.e. `(..., cam, c, h, w)` -- which is how `PatchPolicy._frame_tokens`
      calls it, `(b, t, cam, c, h, w)`. See `_registers` for the flattening
      order this relies on.

    The pretrained base (`self.model`) is frozen and LoRA-adapted (rank
    `lora_rank`, on `attn.qkv`/`attn.proj` only); only `camera_reg_token` and
    the LoRA adapters are trainable.

    Dispatches on the created model's concrete timm type -- `Eva`
    (`vit_small_patch16_dinov3.lvd1689m`: RoPE, prefix tokens fixed at
    construction) vs. plain `VisionTransformer`
    (`vit_small_patch14_dinov2.lvd142m`: absolute `pos_embed`, no RoPE, no
    per-block prefix-token bookkeeping). See `_forward_eva`/`_forward_vit` for
    why the two families need different token-splicing logic.

    NOTE on the `camera_reg_token` name: NOT `reg_token`. `SelectiveAdamW`
    dispatches purely on trailing param name, and the pre-existing `reg_token`
    case (trunk sink registers, other callers) is decayed; these compression
    registers must be decay-free (`N(0, 1e-6)` init -- decay would fight the
    gradient that has to grow them from ~0). Sharing the name would make
    `SelectiveAdamW` unable to tell them apart. See
    plans/effervescent-chasing-seahorse.md's naming-collision note (which
    covers `num_camera_registers` vs. `PatchPolicy.num_register_tokens` --
    this is the analogous collision one level down, on the parameter itself).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        img_size: list[int] | None = None,
        num_registers: int = 16,
        num_cameras: int | None = None,
        lora_rank: int = 32,
        lora_alpha: float = 32.0,
        lora_target_suffixes: tuple[str, ...] = ("attn.qkv", "attn.proj"),
    ) -> None:
        super().__init__()

        # cast: `create_model`'s return type is generically `nn.Module` since
        # it supports any registered timm model; every model this backbone
        # accepts is one of these two concrete families -- see class docstring.
        self.model: Eva | VisionTransformer = cast(
            "Eva | VisionTransformer",
            create_model(model_name, pretrained=True, img_size=img_size, num_classes=0),
        )
        self.model.requires_grad_(False)  # noqa: FBT003

        self.num_registers = num_registers
        self.num_cameras = num_cameras
        # (num_cameras, R, D) per-camera, or (1, R, D) shared -- see class
        # docstring. One `normal_` over the whole tensor gives each camera its
        # own independent draw, which is what lets them diverge in training.
        self.camera_reg_token = nn.Parameter(
            torch.empty(num_cameras or 1, num_registers, self.model.embed_dim)
        )
        # matches timm's own register-token init convention (see
        # timm.models.vision_transformer.VisionTransformer.init_weights)
        nn.init.normal_(self.camera_reg_token, std=1e-6)

        apply_lora(
            self.model.blocks,
            target_suffixes=lora_target_suffixes,
            rank=lora_rank,
            alpha=lora_alpha,
        )

        if isinstance(self.model, Eva):
            # extend every block's RoPE prefix-token count so our new
            # registers (inserted in `_forward_eva`, right after the model's
            # own prefix tokens) are excluded from RoPE the same way the
            # pretrained cls/reg tokens are -- see `_forward_eva`'s docstring.
            for block in self.model.blocks:
                block.attn.num_prefix_tokens += num_registers  # ty:ignore[unresolved-attribute, unsupported-operator]

    @override
    def train(self, mode: bool = True) -> "RegisterViTBackbone":
        super().train(mode)
        self.model.eval()  # frozen base always in eval mode, regardless of outer module state
        return self

    @override
    def forward(self, input: Tensor) -> Tensor:
        *b, c, h, w = input.shape
        if self.num_cameras is not None and (not b or b[-1] != self.num_cameras):
            # `_registers` derives each flattened image's camera from its
            # position, so a mismatched (or missing) camera axis would silently
            # hand images the wrong camera's registers rather than fail.
            msg = (
                "per-camera registers require the camera axis LAST among the "
                f"leading dims -- (..., cam, c, h, w) with cam == num_cameras "
                f"== {self.num_cameras}; got leading dims {tuple(b)}"
            )
            raise ValueError(msg)
        x = input.reshape(-1, c, h, w)
        reg_out = (
            self._forward_eva(x)
            if isinstance(self.model, Eva)
            else self._forward_vit(x)
        )
        return reg_out.reshape(*b, *reg_out.shape[-2:])

    def _registers(self, n: int) -> Tensor:
        """The `(n, R, D)` register tokens for a batch of `n` flattened images.

        Shared (`num_cameras is None`): the single set, broadcast to every image.

        Per-camera: `forward` reshapes `(..., cam, c, h, w)` row-major, so the
        camera axis varies FASTEST -- flattened image `i` comes from camera
        `i % num_cameras`. `repeat` tiles the whole `(cam, R, D)` block in
        exactly that order; `expand` would NOT (it repeats along a new leading
        axis, i.e. `i // num_cameras`), which is a silent camera-permutation
        bug rather than a shape error. Guarded by
        `test_backbone_per_camera_register_routing`.
        """
        if self.num_cameras is None:
            return self.camera_reg_token.expand(n, -1, -1)

        return self.camera_reg_token.repeat(n // self.num_cameras, 1, 1)

    def _forward_eva(self, x: Tensor) -> Tensor:
        """`vit_small_patch16_dinov3.lvd1689m`-style: RoPE, `num_prefix_tokens`
        baked in at construction, no per-forward absolute pos_embed.

        NOTE on why this can't be a naive "concat new tokens after
        `_pos_embed`": the checkpoint already ships its own pretrained
        register/"storage" tokens (Meta's attention-sink registers, remapped
        from `storage_tokens` in `checkpoint_filter_fn`) plus a cls token,
        i.e. `num_prefix_tokens=5` fixed at construction on every attention
        submodule. Each block's `EvaAttention.forward` slices
        `q[:, :, :npt, :]` (left untouched) vs `q[:, :, npt:, :]` (RoPE
        applied), where `npt` does NOT adapt to a longer sequence at forward
        time -- naively inserting new tokens between the existing prefix and
        the patches would push them into the "patch" slice, where
        `apply_rot_embed_cat` would try to apply a patch-grid-length rope
        tensor to a longer slice: a shape-mismatch crash, not a silent bug.
        The fix (applied in `__init__`): insert the new registers right after
        the *existing* prefix tokens, and extend every block's
        `attn.num_prefix_tokens` by `num_registers` so RoPE continues to apply
        only to the real patch tokens. This leaves the pretrained cls/reg
        tokens, rope, and patch embedding entirely untouched -- only
        attention's bookkeeping of "how many leading tokens to skip" changes.
        """
        model = cast("Eva", self.model)
        x = model.patch_embed(x)
        x, rope = model._pos_embed(x)  # noqa: SLF001
        x = model.norm_pre(x)

        n = x.shape[0]
        npt = model.num_prefix_tokens  # original prefix count, unmodified
        reg = self._registers(n)
        x = torch.cat([x[:, :npt], reg, x[:, npt:]], dim=1)

        # `rope`/`attn_mask`/`is_causal` are `EvaBlock.forward`'s next three
        # positional params (see module docstring) -- run_layer_stack forwards
        # them unchanged to every block, checkpointed while training
        attn_mask, is_causal = None, False
        x = run_layer_stack(
            model.blocks, x, rope, attn_mask, is_causal, training=self.training
        )
        x = model.norm(x)

        return x[:, npt : npt + self.num_registers]

    def _forward_vit(self, x: Tensor) -> Tensor:
        """`vit_small_patch14_dinov2.lvd142m`-style plain `VisionTransformer`:
        absolute `pos_embed`, `num_prefix_tokens == 1` (cls only, `reg_token
        is None`), no RoPE, no per-block prefix-token bookkeeping -- simpler
        than `_forward_eva`; none of its `attn.num_prefix_tokens` patching
        applies here (asserted by `test_backbone_no_prefix_patching_on_vit`).
        """
        model = cast("VisionTransformer", self.model)
        x = model.patch_embed(x)
        x = model._pos_embed(x)  # noqa: SLF001 -- returns Tensor, not (x, rope)
        x = model.norm_pre(x)

        n = x.shape[0]
        npt = model.num_prefix_tokens
        reg = self._registers(n)
        x = torch.cat([x[:, :npt], reg, x[:, npt:]], dim=1)

        # cast: timm's `VisionTransformer.blocks` is an `nn.Sequential` (unlike
        # Eva's `nn.ModuleList` -- see `test_backbone_timm_internals_contract_vit`)
        blocks = cast("nn.Sequential", model.blocks)
        x = run_layer_stack(blocks, x, training=self.training)
        x = model.norm(x)

        return x[:, npt : npt + self.num_registers]
