from typing import cast

import numpy as np
import torch
from numpy.testing import assert_allclose
from torch.testing import assert_close, make_tensor

from rmind.components.nn import ChunkDCT, ChunkIDCT

STEPS, AXES = 50, 3

# The 5090 runs einsum in TF32 (~10 mantissa bits), so a round trip on device lands
# around 1e-3 rather than 1e-6. That is 30x below the ~0.03 reconstruction error the
# tokenizer is aiming at, so it is irrelevant to training -- but it means the exactness
# of the basis has to be checked in float64 on CPU, and the device tests carry a
# TF32-sized tolerance.
TF32_ATOL, TF32_RTOL = 5e-3, 1e-2


def _pair(k: int) -> tuple[ChunkDCT, ChunkIDCT]:
    kw = {"num_steps": STEPS, "num_axes": AXES, "num_coefficients": k}
    return ChunkDCT(**kw), ChunkIDCT(**kw)


def test_full_basis_round_trips(device: torch.device) -> None:
    fwd, inv = _pair(STEPS)
    x = make_tensor(8, STEPS * AXES, dtype=torch.float, device=device, low=-1, high=1)

    assert_close(inv.to(device)(fwd.to(device)(x)), x, atol=TF32_ATOL, rtol=TF32_RTOL)


def test_basis_is_orthonormal() -> None:
    # float64 on CPU: this is the claim the whole module rests on, so check it exactly.
    # `basis` is a registered buffer, which ty widens to `Tensor | Module`
    basis = cast(
        "torch.Tensor",
        ChunkDCT(num_steps=STEPS, num_axes=AXES, num_coefficients=STEPS).basis,
    ).double()
    gram = basis @ basis.T

    assert_close(gram, torch.eye(STEPS, dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_truncation_is_a_least_squares_projection(device: torch.device) -> None:
    # Projecting an already-band-limited chunk must leave it untouched.
    fwd, inv = _pair(8)
    fwd, inv = fwd.to(device), inv.to(device)
    full = ChunkDCT(num_steps=STEPS, num_axes=AXES, num_coefficients=STEPS).to(device)
    band = inv(fwd(make_tensor(4, STEPS * AXES, dtype=torch.float, device=device)))

    assert_close(inv(fwd(band)), band, atol=TF32_ATOL, rtol=TF32_RTOL)
    # and the discarded coefficients really are ~zero
    assert full(band).reshape(4, STEPS, AXES)[:, 8:].abs().max() < 5e-3  # noqa: PLR2004


def test_matches_a_reference_dct(device: torch.device) -> None:
    fwd = ChunkDCT(num_steps=STEPS, num_axes=AXES, num_coefficients=STEPS).to(device)
    x = make_tensor(2, STEPS * AXES, dtype=torch.float, device=device, low=-1, high=1)

    t = np.arange(STEPS)
    k = np.arange(STEPS)[:, None]
    ref = np.cos(np.pi * (2 * t + 1) * k / (2 * STEPS)) * np.sqrt(2 / STEPS)
    ref[0] /= np.sqrt(2)
    want = np.einsum("kt,bta->bka", ref, x.cpu().numpy().reshape(2, STEPS, AXES))

    assert_allclose(fwd(x).cpu().numpy(), want.reshape(2, -1), atol=5e-3, rtol=1e-2)


def test_axis_layout_is_preserved(device: torch.device) -> None:
    # A chunk that is constant in time must put all its energy in coefficient 0,
    # per axis -- which only holds if the (timestep, axis) layout is read correctly.
    fwd = ChunkDCT(num_steps=STEPS, num_axes=AXES, num_coefficients=STEPS).to(device)
    per_axis = torch.tensor([0.25, -0.5, 1.0], device=device)
    x = per_axis.repeat(STEPS).unsqueeze(0)

    c = fwd(x).reshape(1, STEPS, AXES)

    assert_close(
        c[0, 0], per_axis * float(np.sqrt(STEPS)), atol=TF32_ATOL, rtol=TF32_RTOL
    )
    assert c[0, 1:].abs().max() < 5e-3  # noqa: PLR2004
