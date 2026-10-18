"""Flow Q-learning with explicit randomness and a canonical [-1, 1] action space."""

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.func import functional_call

MATRIX_RANK = 2


@dataclass(frozen=True)
class FQLConfig:
    observation_dim: int
    action_dim: int
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    value_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    actor_layer_norm: bool = False
    critic_layer_norm: bool = True
    flow_steps: int = 10
    alpha: float = 10.0
    discount: float = 0.99
    tau: float = 0.005
    q_aggregation: Literal["mean", "min"] = "mean"
    normalize_q_loss: bool = False

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.actor_layer_norm,
                self.critic_layer_norm,
                self.normalize_q_loss,
            )
        ):
            message = "FQL boolean settings must be actual booleans"
            raise ValueError(message)
        dimensions = (
            self.observation_dim,
            self.action_dim,
            self.flow_steps,
            *self.actor_hidden_dims,
            *self.value_hidden_dims,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            message = "FQL dimensions and flow_steps must be positive integers"
            raise ValueError(message)
        if not self.actor_hidden_dims or not self.value_hidden_dims:
            message = "FQL networks require hidden layers"
            raise ValueError(message)
        if any(
            isinstance(value, bool) for value in (self.alpha, self.discount, self.tau)
        ) or not all(
            math.isfinite(value) for value in (self.alpha, self.discount, self.tau)
        ):
            message = "FQL coefficients must be finite"
            raise ValueError(message)
        if self.alpha < 0 or not 0 <= self.discount <= 1 or not 0 <= self.tau <= 1:
            message = "Invalid FQL loss or target-update coefficients"
            raise ValueError(message)
        if self.q_aggregation not in {"mean", "min"}:
            message = "Unsupported target Q aggregation"
            raise ValueError(message)


@dataclass(frozen=True)
class FQLNoise:
    next_actions: Tensor
    prior: Tensor
    times: Tensor
    distillation: Tensor

    @classmethod
    def draw(
        cls, actions: Tensor, generator: torch.Generator | None = None
    ) -> "FQLNoise":
        return cls(
            next_actions=torch.randn(
                actions.shape,
                device=actions.device,
                dtype=actions.dtype,
                generator=generator,
            ),
            prior=torch.randn(
                actions.shape,
                device=actions.device,
                dtype=actions.dtype,
                generator=generator,
            ),
            times=torch.rand(
                (*actions.shape[:-1], 1),
                device=actions.device,
                dtype=actions.dtype,
                generator=generator,
            ),
            distillation=torch.randn(
                actions.shape,
                device=actions.device,
                dtype=actions.dtype,
                generator=generator,
            ),
        )


def fql_network(
    input_dim: int, output_dim: int, hidden_dims: tuple[int, ...], *, layer_norm: bool
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden_dims:
        linear = nn.Linear(input_dim, width)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.GELU(approximate="tanh")))
        if layer_norm:
            layers.append(nn.LayerNorm(width, eps=1e-6))
        input_dim = width
    final = nn.Linear(input_dim, output_dim)
    nn.init.xavier_uniform_(final.weight)
    nn.init.zeros_(final.bias)
    return nn.Sequential(*layers, final)


class FQLPolicy(nn.Module):
    """Official low-dimensional architecture; not an RTC or chunk-policy head.

    Inputs are standardized observations and canonical actions. Explicit noise
    makes numerical replay independent of framework PRNG implementations.
    """

    def __init__(self, config: FQLConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.observation_dim + config.action_dim
        self.prior = fql_network(
            input_dim + 1,
            config.action_dim,
            config.actor_hidden_dims,
            layer_norm=config.actor_layer_norm,
        )
        self.actor = fql_network(
            input_dim,
            config.action_dim,
            config.actor_hidden_dims,
            layer_norm=config.actor_layer_norm,
        )
        self.critics = nn.ModuleList([
            fql_network(
                input_dim,
                1,
                config.value_hidden_dims,
                layer_norm=config.critic_layer_norm,
            )
            for _ in range(2)
        ])
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(
            requires_grad=False
        )

    def velocity(self, observations: Tensor, actions: Tensor, times: Tensor) -> Tensor:
        return self.prior(torch.cat((observations, actions, times), dim=-1))

    def sample_actions(self, observations: Tensor, noise: Tensor) -> Tensor:
        return self.actor(torch.cat((observations, noise), dim=-1)).clamp(-1, 1)

    def forward(self, observations: Tensor, noise: Tensor) -> Tensor:
        return self.sample_actions(observations, noise)

    @torch.no_grad()
    def compute_flow_actions(self, observations: Tensor, noise: Tensor) -> Tensor:
        actions = noise
        for step in range(self.config.flow_steps):
            times = actions.new_full(
                (*actions.shape[:-1], 1), step / self.config.flow_steps
            )
            actions = torch.add(
                actions,
                self.velocity(observations, actions, times) / self.config.flow_steps,
            )
        return actions.clamp(-1, 1)

    def values(
        self,
        observations: Tensor,
        actions: Tensor,
        *,
        target: bool = False,
        detached_parameters: bool = False,
    ) -> Tensor:
        inputs = torch.cat((observations, actions), dim=-1)
        networks = self.target_critics if target else self.critics
        values = []
        for network in networks:
            if detached_parameters:
                parameters = {
                    name: parameter.detach()
                    for name, parameter in network.named_parameters()
                }
                output = functional_call(network, parameters, (inputs,))
            else:
                output = network(inputs)
            values.append(output.squeeze(-1))
        return torch.stack(values)

    def validate_batch(self, batch: dict[str, Tensor], noise: FQLNoise) -> None:
        actions = batch["actions"]
        if (
            actions.ndim != MATRIX_RANK
            or actions.shape[1] != self.config.action_dim
            or actions.shape[0] == 0
        ):
            message = "FQL actions must be a nonempty [batch, action_dim] matrix"
            raise ValueError(message)
        expected = {
            "observations": (len(actions), self.config.observation_dim),
            "next_observations": (len(actions), self.config.observation_dim),
            "rewards": (len(actions),),
            "masks": (len(actions),),
        }
        for name, shape in expected.items():
            if batch[name].shape != shape:
                message = f"Invalid FQL {name} shape"
                raise ValueError(message)
        tensors = [
            *batch.values(),
            noise.next_actions,
            noise.prior,
            noise.times,
            noise.distillation,
        ]
        if any(
            tensor.dtype != actions.dtype or tensor.device != actions.device
            for tensor in tensors
        ):
            message = "FQL batch and noise must share device and floating dtype"
            raise ValueError(message)
        if not actions.is_floating_point() or any(
            not torch.isfinite(tensor).all() for tensor in tensors
        ):
            message = "FQL batch and noise must be finite floating tensors"
            raise ValueError(message)
        if (actions.abs() > 1).any() or (
            (batch["masks"] != 0) & (batch["masks"] != 1)
        ).any():
            message = "Actions must be canonical [-1,1]; masks must explicitly specify binary bootstrapping"
            raise ValueError(message)
        if (
            noise.times.shape != (len(actions), 1)
            or ((noise.times < 0) | (noise.times > 1)).any()
        ):
            message = "FQL interpolation times must be [batch,1] in [0,1]"
            raise ValueError(message)
        if any(
            tensor.shape != actions.shape
            for tensor in (noise.next_actions, noise.prior, noise.distillation)
        ):
            message = "FQL action noise shape mismatch"
            raise ValueError(message)

    def losses(
        self, batch: dict[str, Tensor], noise: FQLNoise
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self.validate_batch(batch, noise)
        observations, actions = batch["observations"], batch["actions"]
        with torch.no_grad():
            next_actions = self.sample_actions(
                batch["next_observations"], noise.next_actions
            )
            next_values = self.values(
                batch["next_observations"], next_actions, target=True
            )
            next_value = (
                next_values.amin(0)
                if self.config.q_aggregation == "min"
                else next_values.mean(0)
            )
            targets = (
                batch["rewards"] + self.config.discount * batch["masks"] * next_value
            )
            flow_actions = self.compute_flow_actions(observations, noise.distillation)
        critic_loss = (self.values(observations, actions) - targets).square().mean()
        interpolated = (1 - noise.times) * noise.prior + noise.times * actions
        prior_loss = (
            (
                self.velocity(observations, interpolated, noise.times)
                - (actions - noise.prior)
            )
            .square()
            .mean()
        )
        actor_actions = self.actor(
            torch.cat((observations, noise.distillation), dim=-1)
        )
        distillation_loss = (actor_actions - flow_actions).square().mean()
        action_values = self.values(
            observations, actor_actions.clamp(-1, 1), detached_parameters=True
        ).mean(0)
        q_loss = -action_values.mean()
        if self.config.normalize_q_loss:
            q_loss = torch.div(q_loss, action_values.abs().mean().detach())
        total = (
            critic_loss + prior_loss + self.config.alpha * distillation_loss + q_loss
        )
        return total, {
            "critic_loss": critic_loss,
            "prior_loss": prior_loss,
            "distillation_loss": distillation_loss,
            "q_loss": q_loss,
            "loss": total,
        }

    @torch.no_grad()
    def update_targets(self) -> None:
        """Call after backward, BEFORE optimizer.step: reference uses old critics."""
        for target, source in zip(
            self.target_critics.parameters(), self.critics.parameters(), strict=True
        ):
            target.copy_(source * self.config.tau + target * (1 - self.config.tau))
