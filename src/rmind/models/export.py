"""Export-only wrappers that augment eager forward() output for ONNX/torch.export
without touching production model or objective forward()/predict() code paths.
"""

from typing import final, override

import pytorch_lightning as pl
from tensordict import TensorDict

from rmind.components.base import TensorTree
from rmind.components.objectives.policy import PolicyObjective
from rmind.models.control_transformer import ControlTransformer


@final
class PolicyTrajectoryExportWrapper(pl.LightningModule):
    """Wraps ControlTransformer so the exported graph also emits the policy
    objective's predicted trajectory.

    PolicyObjective.forward() (what ControlTransformer.forward()/ONNX export
    trace) computes `traj_logits` via GRUTrajectoryHead internally, but only
    uses its hidden states to condition the action heads and discards the
    logits themselves — the trajectory is normally only surfaced through
    PolicyObjective.predict()'s TRAJECTORY_VALUE branch, a path ONNX export
    never traces (ControlTransformer.predict_step(), not forward()). This
    wrapper recomputes that branch (mean x/y columns of traj_logits) alongside
    the normal forward() so it appears in the exported graph as
    `trajectory.xy`.
    """

    def __init__(self, *, model: ControlTransformer, policy_key: str = "policy") -> None:
        super().__init__()
        self.model = model
        self.policy_key = policy_key

    @override
    def forward(self, batch: TensorTree) -> TensorTree:
        episode = self.model.episode_builder(batch)
        embedding = self.model.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )

        outputs = {
            name: objective(episode=episode, embedding=embedding)
            for name, objective in self.model.objectives.items()
        }

        policy = self.model.objectives[self.policy_key]
        assert isinstance(policy, PolicyObjective)  # noqa: S101

        policy_embedding = policy.norm(embedding) if policy.norm is not None else embedding
        _, traj_logits, _, _ = policy._compute_logits(  # noqa: SLF001
            episode=episode, embedding=policy_embedding, keep_horizon=True
        )
        if traj_logits is not None:
            # traj_logits is (b, steps, 4): mean_x, logvar_x, mean_y, logvar_y per
            # step; keep the mean columns only, matching predict()'s TRAJECTORY_VALUE
            outputs["trajectory"] = {"xy": traj_logits[..., [0, 2]]}

        return TensorDict(outputs)
