from typing import override

from pydantic import validate_call
from torch import Tensor, nn

from rmind.components.transformer.decoder import CrossAttentionDecoder


class CrossAttentionPolicyHead(nn.Module):
    """Cross-attention policy head ("Approach A2"): a learned query decodes the
    policy logits by attending over the encoder token *context*.

    Unlike the mean/pooler path (which reduces the per-waypoint tokens to a
    single token, then concatenates with the two summary tokens and feeds a
    per-action MLP), this head consumes the full token sequence as cross-attention
    context ``[b, S, dim_model]`` (e.g. ``S = 10 waypoints + obs_summary + obs_history
    = 12``). A per-head learned query ``[1, 1, dim_model]`` is expanded to the
    batch and cross-attends over the context; the decoder output is projected to
    ``out_dim`` (2 = mean,logvar for continuous; 3 = classes for turn_signal).

    Warm-start: the ``output_projection`` is zero-initialised so step-0 outputs are
    ~0 (mild, stable start). NOTE: this is NOT mean-equivalent (unlike the
    WaypointTransformerPooler, which warm-starts to ``mean`` exactly). Here the
    decoder/query are randomly initialised and step-0 outputs are a constant zero
    vector, not the pretrained policy's prediction -- the head learns from scratch
    but starts from a stable zero.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        dim_model: int = 384,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
    ) -> None:
        super().__init__()

        self.query = nn.Parameter(nn.init.trunc_normal_(Tensor(1, 1, dim_model)))
        self.decoder = CrossAttentionDecoder(
            dim_model=dim_model,
            num_layers=num_layers,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            mlp_dropout=mlp_dropout,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )
        self.output_projection = nn.Linear(dim_model, out_dim)
        # zero-init => step-0 output is ~0 (mild, stable warm-start; NOT mean-eq)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @override
    def forward(self, context: Tensor) -> Tensor:
        b = context.shape[0]
        query = self.query.expand(b, -1, -1)
        decoded = self.decoder(query, context)
        return self.output_projection(decoded)
