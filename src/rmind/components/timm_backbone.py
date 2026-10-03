from math import isqrt, prod
from typing import override

import torch
from huggingface_hub import hf_hub_download
from timm import create_model
from timm.layers import resample_abs_pos_embed
from torch import Tensor, nn


class TimmBackbone(nn.Module):
    """Frozen-friendly timm feature extractor with three output modes.

    - default (`out_indices`): timm `features_only` intermediates, PRE final-norm
      -- the historical rmind convention.
    - `norm_patch_tokens=True`: the FINAL block's patch tokens after the model's
      final LayerNorm -- DINOv2's `x_norm_patchtokens`, as consumed by DINO-WM
      (https://github.com/gaoyuezhou/dino_wm); prefix (cls/register) tokens are
      dropped. `out_indices` is ignored.
    - `norm_indices=[i, ...]`: block-`i` intermediates WITH the final LayerNorm
      applied (timm `forward_intermediates(norm=True)`), channel-concatenated
      when more than one index is given -- e.g. `[10]` for "layer 10 + norm",
      `[10, 11]` for "layer 10 + norm ⊕ final layer + norm" (C doubles).

    All modes return `(..., C, H, W)`.

    Set `hf_repo_id`/`hf_filename` to overwrite the timm-pretrained weights with an
    external checkpoint whose ViT encoder lives under `checkpoint_key_prefix` --
    e.g. Depth-Anything-V2 (https://github.com/DepthAnything/Depth-Anything-V2),
    whose DINOv2 encoder is stored under `pretrained.` alongside a `depth_head.*`
    DPT head we drop. Only valid with `norm_patch_tokens=True`, so the model stays
    a vanilla ViT whose keys map 1:1 onto the checkpoint.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        *,
        out_indices: list[int] | None = None,
        img_size: list[int] | None = None,
        norm_patch_tokens: bool = False,
        norm_indices: list[int] | None = None,
        hf_repo_id: str | None = None,
        hf_filename: str | None = None,
        checkpoint_key_prefix: str = "pretrained.",
    ) -> None:
        super().__init__()
        if norm_patch_tokens and norm_indices is not None:
            msg = "norm_patch_tokens and norm_indices are mutually exclusive"
            raise ValueError(msg)
        if hf_repo_id is not None and not norm_patch_tokens:
            msg = "hf_repo_id requires norm_patch_tokens=True"
            raise ValueError(msg)
        self.norm_patch_tokens = norm_patch_tokens
        self.norm_indices = norm_indices
        # skip the timm download when an external checkpoint replaces the weights
        pretrained = hf_repo_id is None
        self.model: nn.Module = (
            create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
            if norm_patch_tokens or norm_indices is not None
            else create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                img_size=img_size,
            )
        )

        if hf_repo_id is not None:
            if hf_filename is None:
                msg = "hf_filename is required when hf_repo_id is set"
                raise ValueError(msg)
            self._load_hf_encoder(hf_repo_id, hf_filename, checkpoint_key_prefix)

    def _load_hf_encoder(self, repo_id: str, filename: str, key_prefix: str) -> None:
        path = hf_hub_download(repo_id, filename)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self._load_prefixed_encoder(
            checkpoint, key_prefix, source=f"{repo_id}/{filename}"
        )

    def _load_prefixed_encoder(
        self, checkpoint: dict[str, Tensor], key_prefix: str, *, source: str = ""
    ) -> None:
        """Load the ViT encoder stored under `key_prefix` from a foreign checkpoint.

        Strips the prefix, drops the DPT head (anything not under the prefix) and the
        `mask_token` (timm's ViT has none), resamples the absolute `pos_embed` from
        the checkpoint's native grid to this model's configured grid, and requires an
        exact key match (fails loud if a timm-version rename drifts the layout).

        Raises:
            RuntimeError: if any encoder key is missing from or unexpected in the model.
        """
        state_dict = {
            key.removeprefix(key_prefix): value
            for key, value in checkpoint.items()
            if key.startswith(key_prefix)
        }
        state_dict.pop("mask_token", None)

        if (pos_embed := state_dict.get("pos_embed")) is not None:
            state_dict["pos_embed"] = resample_abs_pos_embed(
                pos_embed,
                new_size=self.model.patch_embed.grid_size,  # ty:ignore[unresolved-attribute]
                num_prefix_tokens=self.model.num_prefix_tokens,
            )

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            msg = (
                f"key mismatch loading encoder {source or '(checkpoint)'} "
                f"(prefix {key_prefix!r}): "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
            raise RuntimeError(msg)

    @override
    def forward(self, input: Tensor) -> Tensor:
        *b, c, h, w = input.shape
        x = input.view(prod(b), c, h, w)

        if self.norm_indices is not None:
            feats = self.model.forward_intermediates(
                x,
                indices=self.norm_indices,
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
            x = torch.cat(feats, dim=1)
        elif self.norm_patch_tokens:
            # (B, prefix + P, D), final norm applied by timm's forward_features
            tokens = self.model.forward_features(x)
            tokens = tokens[:, self.model.num_prefix_tokens :]
            grid = isqrt(tokens.shape[1])
            x = tokens.transpose(1, 2).reshape(-1, tokens.shape[-1], grid, grid)
        else:
            x = self.model(x)[-1]

        *_, c, h, w = x.shape
        return x.view(*b, c, h, w)
