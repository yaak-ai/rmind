import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Returns a learning rate scheduler that first linearly increases the learning rate from 0.0 to the initial lr over `num_warmup_steps`, then decreases it to 0.0 following a cosine curve for the remaining `num_training_steps - num_warmup_steps` steps.

    This implementation is adapted from Hugging Face's Transformers:
    https://github.com/huggingface/transformers/blob/d79b2d981f28b2730d402244ac3c2e9a8c054eee/src/transformers/optimization.py#L141

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
        num_warmup_steps (int): Number of steps to linearly increase the learning rate.
        num_training_steps (int): Total number of training steps.
        num_cycles (float, optional): Number of cosine cycles. Default is 0.5 (one half-cycle).
        last_epoch (int, optional): The index of the last epoch when resuming training. Default is -1.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Configured learning rate scheduler.
    """

    def linear_warmup(current_step: int) -> float:
        return current_step / max(1, num_warmup_steps)

    def cosine_decay(current_step: int) -> float:
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    def lr_lambda(current_step: int) -> float:
        return (
            linear_warmup(current_step)
            if current_step < num_warmup_steps
            else cosine_decay(current_step)
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
