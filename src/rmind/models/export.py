"""Export-only wrappers that augment eager forward() output for ONNX/torch.export
without touching production model or objective forward()/predict() code paths.
"""

from typing import final, override

import pytorch_lightning as pl
from tensordict import TensorDict

from rmind.components.base import Modality, TensorTree
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
        _, traj_logits, _, _, _ = policy._compute_logits(  # noqa: SLF001
            episode=episode, embedding=policy_embedding, keep_horizon=True
        )
        if traj_logits is not None:
            # traj_logits is (b, steps, 4): mean_x, logvar_x, mean_y, logvar_y per
            # step; keep the mean columns only, matching predict()'s TRAJECTORY_VALUE
            outputs["trajectory"] = {"xy": traj_logits[..., [0, 2]]}

        return TensorDict(outputs)


@final
class PolicyLongitudinalModeExportWrapper(pl.LightningModule):
    """Wraps ControlTransformer so the exported graph also emits the coupled
    longitudinal mode head's per-tick coast/gas/brake probabilities, plus the
    *ungated* gas/brake magnitude means.

    PolicyObjective.forward() computes `mode_logits` via longitudinal_mode_head
    and uses only the argmax to gate the gas/brake means (see the
    `_LONGITUDINAL_MODES` branch) — by the time gas/brake reach the ONNX
    output, one of them is already zeroed by that stateless per-tick decision,
    and the underlying softmax probabilities that produced it are gone. A
    stateful decision hysteresis downstream (e.g. rsim) needs both: the raw
    probabilities to decide whether to override this tick's argmax, and the
    *raw* (pre-gating) magnitude to apply when it does — there's nothing left
    in an already-zeroed pedal to resurrect. This wrapper recomputes
    mode_logits and the raw per-pedal means alongside the normal forward() so
    they appear in the exported graph as `mode.probs` ((b, 3) =
    [coast, gas, brake] — see MODE_COAST/MODE_GAS/MODE_BRAKE in
    rmind.components.objectives.policy), `mode.raw_gas`, `mode.raw_brake`
    ((b,) each, first predicted step, ungated).
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
        raw_logits, _, _, mode_logits, _ = policy._compute_logits(  # noqa: SLF001
            episode=episode, embedding=policy_embedding, keep_horizon=True
        )
        if mode_logits is not None:
            _, probs = policy._longitudinal_mode(mode_logits)  # noqa: SLF001
            outputs["mode"] = {
                "probs": probs[:, 0],  # (b, 3): coast, gas, brake
                "raw_gas": raw_logits[Modality.CONTINUOUS]["gas_pedal"][:, 0, 0],
                "raw_brake": raw_logits[Modality.CONTINUOUS]["brake_pedal"][:, 0, 0],
            }

        return TensorDict(outputs)
