from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from hydra.utils import instantiate
from jaxtyping import Float
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import Tensor, nn
from torch.nn.functional import gelu
from xformers.components import (
    Activation,
    ResidualNormStyle,
    build_activation,
)
from xformers.components.attention import ScaledDotProduct
from xformers.components.attention.core import scaled_query_key_softmax
from xformers.components.feedforward import register_feedforward
from xformers.components.feedforward.mlp import Feedforward, MlpConfig
from xformers.components.reversible import ReversibleSequence
from xformers.factory.model_factory import (
    get_weight_init_fn,
    xFormerEncoderBlock,
    xFormerEncoderConfig,
    xFormerWeightInit,
)

if TYPE_CHECKING:
    from transformers import GPT2LMHeadModel


class HFGPT2(pl.LightningModule):
    hparams: AttributeDict

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.llm: GPT2LMHeadModel = instantiate(self.hparams.llm)

    def forward(
        self,
        inputs_embeds: Tensor,
        labels: Tensor | None = None,
        **_kwargs,
    ) -> Any:
        return self.llm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
            use_cache=True,
        )

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()


class TorchGPT2(pl.LightningModule):
    hparams: AttributeDict

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.llm = instantiate(self.hparams.llm)
        self.classifier = instantiate(self.hparams.classifier)
        self.loss_fn = instantiate(self.hparams.loss)

    def forward(
        self,
        inputs_embeds: Tensor,
        episode_mask: Tensor | None,
        labels: Tensor | None = None,
    ) -> Any:
        output = {}
        x = self.llm(src=inputs_embeds, mask=episode_mask)
        logits = self.classifier(x)
        output["logits"] = logits
        output["hidden_states"] = [x]
        # right shift logits

        if labels is not None:
            shifted_logits = logits[:, :-1, :].contiguous()
            b, t, c = shifted_logits.shape
            # left shift labels
            shifted_labels = labels[:, 1:].contiguous()
            # flatten on batch dimension
            logits_flattened = shifted_logits.view(b * t, c)
            labels_flattened = shifted_labels.view(b * t)
            loss = self.loss_fn(logits_flattened, labels_flattened)

            output["loss"] = loss

        return output

    def get_output_embeddings(self):
        return self.classifier


class TransformerEncoderLayerGEGLU(torch.nn.TransformerEncoderLayer):
    """Replace the original linear transformation + ReLu with GEGLU.

    The paper: https://arxiv.org/pdf/2002.05202v1.pdf
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=lambda x: x,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        # Set activation to None, as GEGLU replaces both linear1 + activation
        self.linear1 = Linear(d_model, dim_feedforward * 2, **factory_kwargs)  # type: ignore
        self.activation = None

    def _ff_block(self, x: Tensor) -> Tensor:
        # FFN_GEGLU eq. 6, https://arxiv.org/pdf/2002.05202v1.pdf
        x = self.linear1(x)
        xW, xV = x.chunk(2, dim=-1)
        geglu = gelu(xW) * xV
        # The original implementation with replacement
        return self.linear2(self.dropout(geglu))


# replicating https://github.com/yaak-ai/carGPT/blob/feat/xformers/cargpt/models/llm.py#L124
@register_feedforward("MLPGLU", MlpConfig)
class MLPGLU(Feedforward):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: str,
        hidden_layer_multiplier: int,
        bias: bool = True,  # noqa: FBT001, FBT002
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        dim_mlp = hidden_layer_multiplier * dim_model
        self.l1 = nn.Linear(in_features=dim_model, out_features=dim_mlp * 2, bias=bias)
        self.a1 = build_activation(Activation(activation))
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(in_features=dim_mlp, out_features=dim_model, bias=bias)
        self.d2 = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        # FFN_GEGLU eq. 6, https://arxiv.org/pdf/2002.05202v1.pdf
        x = self.l1(inputs)
        xW, xV = x.chunk(2, dim=-1)
        geglu = self.a1(xW) * xV
        return self.l2(self.d1(geglu))


class xFormerEncoder(nn.Module):
    def __init__(
        self,
        config: xFormerEncoderConfig,
        weight_init: xFormerWeightInit = xFormerWeightInit.ViT,
    ):
        super().__init__()

        if any((
            not config.reversible,
            config.residual_norm_style is ResidualNormStyle.DeepNorm,
            config.position_encoding_config is not None,
        )):
            raise NotImplementedError

        self.encoders = ReversibleSequence(
            nn.ModuleList(
                nn.ModuleList(xFormerEncoderBlock.get_reversible_layer(config))
                for _ in range(config.num_layers)
            )
        )

        init_fn = get_weight_init_fn(weight_init)
        for name, module in self.encoders.named_children():
            init_fn(module=module, name=name, gain=1.0)

    def forward(
        self,
        src: Float[Tensor, "b s d"],
        mask: Float[Tensor, "s s"],
    ) -> Float[Tensor, "b s d"]:
        x = torch.cat([src, src], dim=-1)
        x = self.encoders(x, att_mask=mask)
        x = torch.stack(x.chunk(2, dim=-1))

        return x.mean(dim=0)

    def compute_attention_rollout(
        self,
        src: Float[Tensor, "b s d"],
        mask: Float[Tensor, "s s"],
        *,
        drop_ratio: float | None = None,
    ) -> Float[Tensor, "b s s"]:
        """
        [1] Quantifying Attention Flow in Transformers (https://arxiv.org/abs/2005.00928)
        [2] Exploring Explainability for Vision Transformers (https://jacobgil.github.io/deeplearning/vision-transformer-explainability)
        """
        b, s, _ = src.shape
        identity = torch.eye(s, s, device=mask.device)
        attn_rollout = repeat(identity, "s_from s_to -> b s_from s_to", b=b)

        seq = repeat(src, "b s d -> b s (n d)", n=2)
        for block in self.encoders.blocks:
            match (net := block.f.net).sublayer.attention:
                case ScaledDotProduct():
                    _, x = rearrange(seq, "b s (n d) -> n b s d", n=2)
                    x = net.norm(x)
                    q = net.sublayer.in_proj_container.q_proj(x)
                    k = net.sublayer.in_proj_container.k_proj(x)
                    attn = scaled_query_key_softmax(q, k, att_mask=mask)

                    if drop_ratio is not None:
                        drop_count = int(attn.nelement() * drop_ratio)
                        attn_flat = rearrange(attn, "b s_from s_to -> b (s_from s_to)")
                        drop_indices = attn_flat.topk(
                            k=drop_count, dim=-1, largest=False
                        ).indices
                        for batch_idx, batch_drop_indices in enumerate(drop_indices):
                            attn_flat[batch_idx, batch_drop_indices] = 0

                        attn = rearrange(
                            attn_flat,
                            "b (s_from s_to) -> b s_from s_to",
                            s_from=s,
                            s_to=s,
                        )

                    attn = (attn + identity) * 0.5
                    # [2] > We also have to normalize the rows, to keep the total attention flow 1.
                    attn /= attn.sum(dim=-1, keepdim=True)
                    attn_rollout @= attn

                case _:
                    raise NotImplementedError

            seq = block(seq, f_args={"att_mask": mask})

        return attn_rollout
