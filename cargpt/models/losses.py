import torch
import torch.nn as nn
import torch.nn.functional as F


class DetokenizedL1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, logits, labels, labels_shift, detokenizer, metadata_keys, action_keys
    ):
        logits = logits.clone()
        labels = labels.clone()

        b, t, c = logits.shape
        # flatten on batch dimension
        logits = logits.view(b * t, c)
        tgt_labels = labels.view(b * t)
        labels_shift = labels_shift.view(b * t)

        # Kick out ignore_index labels (-1)
        labels_mask = torch.where(tgt_labels >= 0)
        #
        # Softmax and sample from distribution
        #
        pred_labels = torch.multinomial(F.softmax(logits, dim=1), 1).view(b * t)

        # unshift pred, tgt labels to bring it to [0, 1024)
        pred_labels_shifted = pred_labels - labels_shift
        tgt_labels_shifted = tgt_labels - labels_shift

        masked_pred_labels = pred_labels_shifted[labels_mask]
        masked_tgt_labels = tgt_labels_shifted[labels_mask]

        # reverse view
        parts = [len(metadata_keys), 1, len(action_keys)]
        view_reshape = (b, -1, sum(parts))

        pred_observations_labels, _, pred_actions_labels = torch.split(
            masked_pred_labels.view(*view_reshape), parts, dim=2
        )
        tgt_observations_labels, _, tgt_actions_labels = torch.split(
            masked_tgt_labels.view(*view_reshape), parts, dim=2
        )

        # detokenize (tokens to real values)
        pred_observations_values = torch.zeros_like(
            pred_observations_labels, dtype=torch.float
        )
        pred_actions_values = torch.zeros_like(pred_actions_labels, dtype=torch.float)
        tgt_observations_values = torch.zeros_like(
            tgt_observations_labels, dtype=torch.float
        )
        tgt_actions_values = torch.zeros_like(tgt_actions_labels, dtype=torch.float)

        for idx, key in enumerate(metadata_keys):
            inv_func = detokenizer[key]
            pred_observations_values[:, :, idx] = inv_func(
                pred_observations_labels[:, :, idx]
            )
            tgt_observations_values[:, :, idx] = inv_func(
                tgt_observations_labels[:, :, idx]
            )

        for idx, key in enumerate(action_keys):
            inv_func = detokenizer[key]
            pred_actions_values[:, :, idx] = inv_func(pred_actions_labels[:, :, idx])
            tgt_actions_values[:, :, idx] = inv_func(tgt_actions_labels[:, :, idx])

        # Calculate L1
        obs_diff = torch.abs(tgt_observations_values - pred_observations_values)
        action_diff = torch.abs(tgt_actions_values - pred_actions_values)

        l1_loss = {}  # type: ignore[assignment]
        values = {}  # type: ignore[assignment]
        for idx, key in enumerate(metadata_keys):
            l1_loss[key] = obs_diff[:, :, idx].mean()
            values[f"{key}_pred"] = pred_observations_values[:, :, idx]
            values[f"{key}_tgt"] = tgt_observations_values[:, :, idx]

        for idx, key in enumerate(action_keys):
            l1_loss[key] = action_diff[:, :, idx].mean()
            values[f"{key}_pred"] = pred_actions_values[:, :, idx]
            values[f"{key}_tgt"] = tgt_actions_values[:, :, idx]

        return l1_loss, values
