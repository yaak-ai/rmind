from typing import override

from torch import Tensor, nn


class EgoStateEncoder(nn.Module):
    """Encode ego state into a single embedding, added to the trajectory
    decoder's learnable queries (DrivoR, arXiv:2601.05083).

    `route_embedding` stands in for the paper's discrete driving command --
    see `rmind.models.drivor` module docstring for why `waypoints/xy_normalized`
    (encoded upstream by a frozen `WaypointsTokenizer`) is used for this,
    rather than `turn_signal` alone.
    """

    def __init__(
        self,
        *,
        continuous_dim: int = 4,
        num_turn_signal_classes: int = 3,
        route_embedding_dim: int = 384,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.turn_signal_embedding = nn.Embedding(
            num_turn_signal_classes, embedding_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(continuous_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.route_projection = nn.Linear(route_embedding_dim, embedding_dim)

    @override
    def forward(
        self, *, continuous: Tensor, turn_signal: Tensor, route_embedding: Tensor
    ) -> Tensor:
        # continuous: (B, continuous_dim); turn_signal: (B,) long; route_embedding: (B, route_embedding_dim)
        return (
            self.mlp(continuous)
            + self.turn_signal_embedding(turn_signal)
            + self.route_projection(route_embedding)
        )
