from collections.abc import Set as AbstractSet
from typing import final, override

import torch
import torch.nn.functional as F
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Embedding, Module

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Targets,
)


@final
class GoalConditionedForesightObjective(Objective):
    """Multi-step latent foresight + goal-conditioned inverse dynamics.

    For every within-episode timestep pair (i, j), i < j:

    - **latent foresight**: from the (causal) OBSERVATION_SUMMARY embedding at
      timestep `i` plus a horizon embedding for `j - i`, predict the
      block-pooled (grid_pool x grid_pool) DINOv3 latent grid of the frame at
      timestep `j` (cosine loss per block). This is *latent* multi-step
      foresight: no pixel/patch detail, only a compact future scene state.

    - **goal-conditioned inverse dynamics**: from (summary_i, DINOv3 latent of
      the *goal* frame at `j`, horizon embedding), predict the next action
      tokens (at timestep `i + 1`, mirroring `InverseDynamicsPredictionObjective`).
      The hindsight goal supplies exactly the information that makes the action
      well-determined (e.g. turn vs straight at an intersection), pushing the
      backbone to encode goal-relevant, action-predictive state.

    The DINOv3 goal/target latents come from `episode.input_embeddings` (the
    frozen image backbone, *before* the temporal encoder), so they contain no
    executed-action information — the "legitimate" flavor of goal signal that
    can be substituted at inference by a route/waypoint-derived target.

    Also emits non-loss `metric` entries with representation-health signals
    (feature std / effective rank of the summary token, foresight prediction
    spread) to catch representation collapse early.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        foresight_head: InstanceOf[Module],
        inverse_heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict],
        targets: Targets,
        horizon_embedding: InstanceOf[Embedding],
        norm: InstanceOf[Module] | None = None,
        image_key: str = "cam_front_left",
        grid_pool: int = 2,
        foresight_weight: float = 1.0,
        inverse_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.foresight_head = foresight_head
        self.inverse_heads = inverse_heads
        self.losses = losses  # inverse-dynamics losses, mirrors `targets` paths
        self.targets: Targets = targets
        self.horizon_embedding = horizon_embedding
        self.image_key = image_key
        # DINO latents are block-pooled to a (grid_pool x grid_pool) spatial grid:
        # full-image mean-pooling was measured to destroy most action-relevant
        # scene information (probe: pooled current-frame DINO ~= majority class)
        self.grid_pool = grid_pool
        self.foresight_weight = foresight_weight
        self.inverse_weight = inverse_weight

    def _summaries(self, episode: Episode, embedding: Tensor) -> Tensor:
        if self.norm is not None:
            embedding = self.norm(embedding)
        k = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
        # (b, t, 1, d) -> (b, t, d)
        return episode.index.select(k).parse(embedding).get(k).squeeze(-2)

    @staticmethod
    def _pairs(t: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """All (i, j) with 0 <= i < j <= t - 1."""
        idx = torch.arange(t, device=device)
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        keep = ii < jj
        return ii[keep], jj[keep]

    @staticmethod
    def _effective_rank(x: Tensor) -> Tensor:
        """exp(entropy of normalized squared singular values) of (n, d) features."""
        x = (x - x.mean(0)).float()
        s = torch.linalg.svdvals(x)
        p = s.square() / s.square().sum().clamp_min(1e-12)
        return torch.exp(-(p * p.clamp_min(1e-12).log()).sum())

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:  # noqa: PLR0914
        summaries = self._summaries(episode, embedding)  # (b, t, d)
        b, t, _ = summaries.shape

        # frozen DINOv3 latents, block-pooled to a (g x g) grid then flattened:
        # (b, t, g*g*d_img); action-free by construction
        patches = episode.input_embeddings.get((Modality.IMAGE, self.image_key))
        n, d_img = patches.shape[-2], patches.shape[-1]
        side, g = int(n**0.5), self.grid_pool
        block = side // g
        dino = (
            patches.detach()
            .view(b, t, g, block, g, block, d_img)
            .mean(dim=(3, 5))
            .reshape(b, t, g * g, d_img)
        )

        ii, jj = self._pairs(t, summaries.device)
        horizon = self.horizon_embedding(jj - ii).expand(b, -1, -1)  # (b, p, dh)

        src = summaries[:, ii]  # (b, p, d)
        goal = dino[:, jj]  # (b, p, g*g, d_img)

        # --- multi-step latent foresight ---------------------------------
        foresight_pred = self.foresight_head(
            torch.cat([src, horizon], dim=-1)
        ).unflatten(-1, (g * g, d_img))
        foresight_loss = (
            1.0 - F.cosine_similarity(foresight_pred, goal, dim=-1)
        ).mean()

        # --- goal-conditioned inverse dynamics ---------------------------
        # predict action tokens at timestep i + 1 (as InverseDynamicsPredictionObjective)
        inverse_features = torch.cat([src, goal.flatten(-2), horizon], dim=-1)
        losses: dict[str, Tensor | dict[str, Tensor]] = {
            "foresight": self.foresight_weight * foresight_loss
        }
        inverse: dict[str, dict[str, Tensor]] = {}
        for modality, names in self.targets.items():
            for name, path in names.items():
                logits = self.inverse_heads.get((modality, name))(inverse_features)
                target = episode.get(path).squeeze(-1)[:, ii + 1]  # (b, p)
                loss = self.losses.get((modality, name))(
                    logits.flatten(0, 1), target.flatten()
                )
                inverse.setdefault(modality, {})[name] = self.inverse_weight * loss
        losses["inverse"] = inverse  # ty:ignore[invalid-assignment]

        # --- representation-health metrics (logged, not part of the loss) ---
        with torch.no_grad():
            last = summaries[:, -1]
            metrics = {
                "summary_feature_std": last.std(dim=0).mean(),
                "summary_effective_rank": self._effective_rank(last),
                "foresight_pred_std": foresight_pred.flatten(0, 2).std(dim=0).mean(),
                "foresight_cos_to_goal": 1.0 - foresight_loss.detach(),
            }

        return {"loss": losses, "metric": metrics}

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        return TensorDict({}, batch_size=episode.input.batch_size[:1])
