"""SIGReg: Sketched Isotropic Gaussian Regularization (LeJEPA).

Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics", arXiv:2511.08544.

LeJEPA objective:
    L = (1-lambda)/B * sum_n L_pred(z_n)  +  lambda/V * sum_v SIGReg({z_n})
with the recommended trade-off lambda = 0.05.

SIGReg pushes the encoder's embedding distribution toward an isotropic
*standard* Gaussian N(0, I). It does so by "slicing": drawing V unit-norm
random directions a_v on the hypersphere S^{D-1}, projecting the batch of
embeddings onto each direction, and applying a 1-D Gaussianity test T to each
projected set of scalars:

    SIGReg(Z) = (1/V) * sum_v  T( { a_v^T z_n : n } )

The 1-D test is the Epps-Pulley statistic, which compares the empirical
characteristic function phi_hat(t) = (1/N) sum_n exp(i t p_n) of the
projections p_n against the standard-normal characteristic function
phi(t) = exp(-t^2/2), integrated against a Gaussian weight w(t):

    T = N * integral | phi_hat(t) - phi(t) |^2 w(t) dt

The integral is approximated on a fixed grid of `num_points` abscissae.
Testing against the *fixed* N(0,1) target (rather than standardizing the
projections first) is what makes SIGReg anti-collapse: a collapsed embedding
(near-constant projections) and an over/under-dispersed embedding both incur a
large statistic, so every direction is driven to unit variance -> full
effective rank, no dimensional collapse, no stop-gradient / teacher needed.

This is a faithful re-implementation of the method as described in the paper
(defaults: num_slices=1024 directions, num_points=17); the exact quadrature
grid may differ from the authors' reference repo but preserves the statistic.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn


class SIGReg(nn.Module):
    """Sliced Epps-Pulley isotropic-Gaussian regularizer.

    Args:
        num_slices: number of random unit projection directions V (default 1024).
        num_points: quadrature abscissae for the Epps-Pulley integral (default 17).
        t_max: half-width of the quadrature grid over the characteristic-function
            argument t; the Gaussian weight makes the integrand negligible beyond.
        resample: redraw the random directions every forward (synchronised across
            ranks via a shared generator seed in distributed training).
    """

    def __init__(
        self,
        *,
        num_slices: int = 1024,
        num_points: int = 17,
        t_max: float = 5.0,
        resample: bool = True,
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.resample = resample
        t = torch.linspace(-t_max, t_max, num_points)
        # trapezoidal integration weights * Gaussian weight w(t)=exp(-t^2/2)
        dt = float(t[1] - t[0])
        trap = torch.full((num_points,), dt)
        trap[0] = trap[-1] = dt / 2.0
        weight = trap * torch.exp(-0.5 * t.pow(2))
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("weight", weight, persistent=False)
        # standard-normal characteristic function is real: phi(t)=exp(-t^2/2)
        self.register_buffer("phi", torch.exp(-0.5 * t.pow(2)), persistent=False)
        self._directions: Tensor | None = None

    def _get_directions(self, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.resample or self._directions is None or self._directions.shape[1] != dim:
            a = torch.randn(self.num_slices, dim, device=device, dtype=dtype)
            a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-8)
            if not self.resample:
                self._directions = a
            return a
        return self._directions

    @override
    def forward(self, z: Tensor) -> Tensor:
        """z: (N, D) batch of embeddings. Returns scalar SIGReg loss."""
        z = z.flatten(0, -2) if z.dim() > 2 else z  # (N, D)
        n, d = z.shape
        a = self._get_directions(d, z.device, z.dtype)  # (V, D)
        proj = z @ a.t()  # (N, V) projection onto each direction
        t = self.t.to(z.dtype)  # (P,)
        # empirical characteristic function phi_hat_v(t_k) = mean_n exp(i t_k p_nv)
        # shape (V, P): arg = proj (N,V) outer t (P,)
        arg = proj.unsqueeze(-1) * t.view(1, 1, -1)  # (N, V, P)
        re = torch.cos(arg).mean(dim=0)  # (V, P)
        im = torch.sin(arg).mean(dim=0)  # (V, P)
        diff_sq = (re - self.phi.to(z.dtype)).pow(2) + im.pow(2)  # (V, P)
        stat = n * (diff_sq * self.weight.to(z.dtype)).sum(dim=-1)  # (V,)
        return stat.mean()


def lejepa_loss(
    prediction_loss: Tensor,
    embeddings: Tensor,
    sigreg: SIGReg,
    *,
    lam: float = 0.05,
) -> Tensor:
    """Combine JEPA prediction loss with SIGReg per the LeJEPA objective.

    L = (1 - lam) * prediction_loss + lam * SIGReg(embeddings)
    with lam = 0.05 recommended by the paper.
    """
    return (1.0 - lam) * prediction_loss + lam * sigreg(embeddings)
