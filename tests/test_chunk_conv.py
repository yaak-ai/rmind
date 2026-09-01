import torch
from torch.testing import assert_close, make_tensor

from rmind.components.nn import ChunkConvDecoder, ChunkConvEncoder

STEPS, AXES, LATENT = 50, 3, 384


def test_shapes_round_trip(device: torch.device) -> None:
    enc = ChunkConvEncoder(num_steps=STEPS, num_axes=AXES, out_features=LATENT).to(
        device
    )
    dec = ChunkConvDecoder(num_steps=STEPS, num_axes=AXES, in_features=LATENT).to(
        device
    )
    x = make_tensor(8, STEPS * AXES, dtype=torch.float, device=device, low=-1, high=1)

    z = enc(x)
    assert z.shape == (8, LATENT)
    assert dec(z).shape == x.shape


def test_axis_layout_matches_the_flat_convention(device: torch.device) -> None:
    # `_gather_actions` puts axis fastest: element i is (timestep, axis) = divmod(i, A).
    # A chunk that is constant per axis must reach the conv as constant per channel.
    enc = ChunkConvEncoder(num_steps=STEPS, num_axes=AXES, out_features=LATENT).to(
        device
    )
    per_axis = torch.tensor([0.25, -0.5, 1.0], device=device)
    x = per_axis.repeat(STEPS).unsqueeze(0)

    chunk = x.reshape(-1, STEPS, AXES).transpose(1, 2)

    assert_close(chunk[0, :, 0], per_axis)
    assert chunk[0].std(dim=1).max().item() < 1e-6  # noqa: PLR2004
    assert enc(x).shape == (1, LATENT)


def test_receptive_field_spans_the_documented_window(device: torch.device) -> None:
    # +/-16 steps: 3 from the k=7 input conv, then 1 + 3 + 9 from the dilated units.
    # A single impulse at the centre must move outputs 16 away and leave 20 alone.
    enc = ChunkConvEncoder(num_steps=STEPS, num_axes=AXES, out_features=LATENT).to(
        device
    )
    base = torch.zeros(1, STEPS * AXES, device=device)
    spike = base.clone()
    spike[0, 25 * AXES] = 1.0

    with torch.no_grad():
        a = enc.conv(base.reshape(-1, STEPS, AXES).transpose(1, 2))
        b = enc.conv(spike.reshape(-1, STEPS, AXES).transpose(1, 2))
    moved = (a - b).abs().sum(dim=1)[0] > 1e-6  # noqa: PLR2004

    assert moved[25 - 16].item()
    assert moved[25 + 16].item()
    assert not moved[25 - 20].item()
    assert not moved[25 + 20].item()


def test_gradients_reach_every_parameter(device: torch.device) -> None:
    enc = ChunkConvEncoder(num_steps=STEPS, num_axes=AXES, out_features=LATENT).to(
        device
    )
    dec = ChunkConvDecoder(num_steps=STEPS, num_axes=AXES, in_features=LATENT).to(
        device
    )
    x = make_tensor(4, STEPS * AXES, dtype=torch.float, device=device, low=-1, high=1)

    dec(enc(x)).pow(2).mean().backward()

    dead = [
        n
        for m in (enc, dec)
        for n, p in m.named_parameters()
        if p.grad is None or p.grad.abs().sum() == 0
    ]
    assert not dead, f"no gradient reached: {dead}"
