from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import gelu
from xformers.components.attention import AttentionMask
from xformers.components.attention import attention_patterns as AP
from xformers.components.attention._sputnik_sparse import SparseCS
from xformers.factory.block_configs import (
    xFormerBlockConfig,
)
from xformers.factory.model_factory import xFormer
from xformers.factory.weight_init import xFormerWeightInit
from xformers.ops.fmha.attn_bias import AttentionBias


class HFGPT2(pl.LightningModule):
    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.llm = instantiate(self.hparams.llm)

    def forward(
        self,
        inputs_embeds: Tensor,
        return_dict: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        past_key_values: Optional[List[Tensor]] = None,
        episode_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Any:
        output = self.llm(
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            labels=labels,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        return output

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()


class TorchGPT2(pl.LightningModule):
    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.llm = instantiate(self.hparams.llm)
        self.classifier = instantiate(self.hparams.classifier)
        self.loss_fn = instantiate(self.hparams.loss)

    def forward(
        self,
        inputs_embeds: Tensor,
        return_dict: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        past_key_values: Optional[List[Tensor]] = None,
        episode_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
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
        self.activation = None  # type: ignore[assignment]

    def _ff_block(self, x: Tensor) -> Tensor:
        # FFN_GEGLU eq. 6, https://arxiv.org/pdf/2002.05202v1.pdf
        x = self.linear1(x)
        xW, xV = x.chunk(2, dim=-1)
        geglu = gelu(xW) * xV
        # The original implementation with replacement
        x = self.linear2(self.dropout(geglu))
        return x


class xFormerGPT(pl.LightningModule):
    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        config = instantiate(self.hparams.llm.config)  # type: ignore[union-attr]
        self.llm = SparseFormer.from_config(config)
        # decoder head
        self.classifier = instantiate(self.hparams.classifier)
        self.loss = instantiate(self.hparams.loss)
        # TODO: Not used as of now attention mask should be an input param independent of model
        self.mask = instantiate(self.hparams.mask.attn_config)  # type: ignore[union-attr]

    def forward(
        self,
        inputs_embeds: Tensor,
        return_dict: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,  # Not used
        past_key_values: Optional[List[Tensor]] = None,  # Not used
        episode_mask: Optional[Tensor] = None,  # Not used
        labels: Optional[Tensor] = None,
    ) -> Any:
        output = {}
        x = self.llm(inputs_embeds=inputs_embeds, labels=labels, att_mask=episode_mask)
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
            loss = self.loss(logits_flattened, labels_flattened)

            output["loss"] = loss

        return output

    def get_output_embeddings(self):
        return self.classifier[1]


# Wrapper around https://github.com/facebookresearch/xformers/blob/main/xformers/factory/model_factory.py#L106
# They don't support sparse attention mask as input yet
class SparseFormer(xFormer):
    def __init__(
        self,
        stack_configs: Union[
            xFormerBlockConfig, List[xFormerBlockConfig], Dict[str, xFormerBlockConfig]
        ],
        tie_embedding_weights: bool = False,
        weight_init: xFormerWeightInit = xFormerWeightInit.ViT,
    ):
        super().__init__(
            stack_configs=stack_configs,
            tie_embedding_weights=tie_embedding_weights,
            weight_init=weight_init,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        att_mask: Optional[
            Union[torch.Tensor, SparseCS, AttentionMask, AttentionBias]
        ] = None,
        encoder_input_mask: Optional[torch.Tensor] = None,
        decoder_input_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # Encode to latent space if encoder is present
        memory = inputs_embeds.clone()
        if len(list(self.encoders.parameters())) > 0:
            encoders = self.encoders
            if isinstance(encoders, torch.nn.ModuleList):
                for encoder in encoders:
                    memory = encoder(
                        memory, att_mask=att_mask, input_mask=encoder_input_mask
                    )
            else:
                if self.rev_enc_pose_encoding:
                    memory = self.rev_enc_pose_encoding(inputs_embeds)

                # Reversible Encoder
                x = torch.cat([memory, memory], dim=-1)

                # Apply the optional input masking
                if encoder_input_mask is not None:
                    if x.dim() - encoder_input_mask.dim() > 1:
                        encoder_input_mask.unsqueeze(0)
                    x += encoder_input_mask.unsqueeze(-1)

                x = encoders(x, att_mask=att_mask)
                memory = torch.stack(x.chunk(2, dim=-1)).mean(dim=0)

            if not self.decoders:
                return memory

        # If decoder: either use the encoder ouput, or just decode, both options are possible
        if len(self.decoders) > 0:
            tgt = inputs_embeds.clone() if labels is None else labels

            for decoder in self.decoders:
                tgt = decoder(
                    target=tgt,
                    # pyre-fixme[61]: `memory` is not always initialized here.
                    memory=memory,
                    att_mask=att_mask,
                    input_mask=decoder_input_mask,
                )

            return tgt

        return None
