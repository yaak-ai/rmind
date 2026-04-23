from typing import Any, final

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from structlog import get_logger
from torch import Tensor
from wandb import Image

from rmind.components.base import TokenType
from rmind.components.episode import Episode, EpisodeBuilder, TokenMeta

from .common import _figure_to_rgba, _get_wandb_loggers

logger = get_logger(__name__)

_TOKEN_TYPE_COLORS: dict[TokenType, str] = {
    TokenType.OBSERVATION: "#3498db",  # blue
    TokenType.ACTION: "#e74c3c",  # red
    TokenType.SPECIAL: "#2ecc71",  # green
}

_SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉"


def _subscript(n: int) -> str:
    return "".join(_SUBSCRIPTS[int(d)] for d in str(n))


def visualize_attention_mask(  # noqa: PLR0914
    mask: Tensor, timestep_meta: tuple[TokenMeta, ...], index: dict, num_timesteps: int
) -> plt.Figure:
    """
    CODEX GENERATED
    """
    # Build block definitions: (label, token_type, num_tokens)
    blocks: list[tuple[str, TokenType, int]] = []
    for t in range(num_timesteps):
        t_sub = _subscript(t)
        for token in timestep_meta:
            num_tokens = index[token.modality.value][str(token.name)].shape[-1]
            name = str(token.name)
            label = (
                f"{name}[{num_tokens}]{t_sub}" if num_tokens > 1 else f"{name}{t_sub}"
            )
            blocks.append((label, token.type, num_tokens))

    # Build block ranges for collapsing
    block_ranges: list[tuple[int, int]] = []
    offset = 0
    for _, _, num_tokens in blocks:
        block_ranges.append((offset, offset + num_tokens))
        offset += num_tokens

    # Collapse mask via max pooling over block ranges
    n_blocks = len(blocks)
    mask_cpu = mask.detach().cpu().float()
    collapsed = torch.zeros(n_blocks, n_blocks)
    for i, (i_start, i_end) in enumerate(block_ranges):
        for j, (j_start, j_end) in enumerate(block_ranges):
            collapsed[i, j] = mask_cpu[i_start:i_end, j_start:j_end].max()

    # Color by destination (column) token type
    can_attend = collapsed > 0
    type_to_val = {TokenType.OBSERVATION: 1, TokenType.ACTION: 2, TokenType.SPECIAL: 3}
    colored = np.zeros((n_blocks, n_blocks), dtype=np.int32)
    for j, (_, token_type, _) in enumerate(blocks):
        colored[:, j] = np.where(
            can_attend[:, j].numpy(), type_to_val.get(token_type, 3), 0
        )

    # Render
    cmap = ListedColormap(["black", "#3498db", "#e74c3c", "#2ecc71"])
    norm = BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5], cmap.N)

    labels = [b[0] for b in blocks]
    blocks_per_timestep = len(timestep_meta)

    fig, ax = plt.subplots(figsize=(max(14, n_blocks * 0.6), max(12, n_blocks * 0.55)))
    ax.imshow(colored, cmap=cmap, norm=norm, aspect="equal")

    ax.set_xticks(range(n_blocks))
    ax.set_yticks(range(n_blocks))
    ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Cell grid lines
    ax.set_xticks([i - 0.5 for i in range(n_blocks + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(n_blocks + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", length=0)

    # Timestep boundary lines
    for t in range(1, num_timesteps):
        pos = t * blocks_per_timestep - 0.5
        ax.axvline(pos, color="white", linewidth=2)
        ax.axhline(pos, color="white", linewidth=2)

    ax.set_xlabel("Destination (being attended to)", fontsize=10)
    ax.set_ylabel("Source (attending)", fontsize=10)

    legend_elements = [
        Patch(facecolor=_TOKEN_TYPE_COLORS[TokenType.OBSERVATION], label="Observation"),
        Patch(facecolor=_TOKEN_TYPE_COLORS[TokenType.ACTION], label="Action"),
        Patch(facecolor=_TOKEN_TYPE_COLORS[TokenType.SPECIAL], label="Special"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    fig.suptitle("Causal Attention Mask", fontsize=14)
    fig.tight_layout()

    return fig


@final
class WandbAttentionMaskLogger(Callback):
    def __init__(self, *, num_vis_timesteps: int = 2) -> None:
        self._num_vis_timesteps = num_vis_timesteps

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        loggers = _get_wandb_loggers(pl_module)
        if (
            (batch_idx != 0 or trainer.current_epoch != 0)
            or trainer.sanity_checking
            or not loggers
            or not trainer.is_global_zero
        ):
            return

        match outputs:
            case {"_internal": {"episode": Episode() as episode}}:
                pass
            case _:
                logger.warning(
                    "attention mask logger skipped: missing `_internal.episode`"
                )
                return

        index = episode.index.to_dict()
        episode_builder = getattr(pl_module, "episode_builder", None)
        if not isinstance(episode_builder, EpisodeBuilder):
            logger.warning(
                "attention mask logger skipped: `pl_module.episode_builder` is not an `EpisodeBuilder`"
            )
            return

        timestep_meta = episode_builder.timestep

        # Determine actual timesteps and clamp to vis limit
        leaves = [v for mod in index.values() for v in mod.values()]
        actual_t = leaves[0].shape[0] if leaves else 0
        n = min(self._num_vis_timesteps, actual_t)
        if n == 0:
            return

        # Slice index to vis timesteps
        sliced_index = {
            mod: {name: tensor[:n] for name, tensor in names.items()}
            for mod, names in index.items()
        }

        seq_len = sum(
            tensor.numel()
            for names in sliced_index.values()
            for tensor in names.values()
        )
        mask = (
            episode.attention_mask.mask_tensor[:seq_len, :seq_len]
            == episode.attention_mask.legend.DO_ATTEND
        )

        fig = visualize_attention_mask(mask, timestep_meta, sliced_index, n)
        try:
            for logger_ in loggers:
                logger_.log_image(
                    "attention_mask",
                    [Image(_figure_to_rgba(fig))],
                    step=trainer.global_step,
                )
        finally:
            plt.close(fig)
