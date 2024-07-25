import torch
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch.nn import Module

from cargpt.components.loss import LossType
from cargpt.utils import ModuleDict


class Branch(Module):
    def __init__(self, heads: Module | ModuleDict, losses: Module | ModuleDict):
        super().__init__()
        self.heads = heads if isinstance(heads, ModuleDict) else ModuleDict(val=heads)
        self.losses = (
            losses if isinstance(losses, ModuleDict) else ModuleDict(loss=losses)
        )

    def calculate_loss(self, features, episode, values_key):
        branch_loss_val = []
        logits = TensorDict.from_dict(
            {key: head(features) for key, head in self.heads.flatten_md()},
            batch_size=[],
        )

        logits = logits.apply(Rearrange("b 1 d -> b d"), batch_size=[])

        target_labels = episode.tokenized[values_key][:, -1]
        target_values = episode.inputs[values_key][:, -1]

        target_labels = Rearrange("b 1 -> b")(target_labels)
        # TODO: rewrite with named_apply?
        for loss in self.losses.values():
            match loss.loss_type:
                case LossType.CLASSIFICATION:
                    branch_loss_val.append(
                        loss(input=logits, target_labels=target_labels)
                    )
                case LossType.REGRESSION:
                    branch_loss_val.append(
                        loss(input=logits, target_values=target_values)
                    )
                case _:
                    raise NotImplementedError
        return torch.stack(branch_loss_val).mean()
