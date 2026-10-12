"""FiniteScalarQuantizer: index <-> code round trip, straight-through gradients,
and the NeroChunkTokenizer drop-in (codes as indices, no VQ losses)."""

from typing import Any

import pytest
import torch

from rmind.components.vq import FiniteScalarQuantizer


@pytest.mark.parametrize(
    ("levels", "g"), [((8, 5, 5, 5), 1), ((8, 8, 8, 5, 5), 1), ((5, 5, 5), 3)]
)
def test_index_code_round_trip(levels: tuple[int, ...], g: int) -> None:
    torch.manual_seed(0)
    q = FiniteScalarQuantizer(dim=32, levels=levels, num_quantizers=g)
    k = q.codebook_size
    assert k == torch.tensor(levels).prod().item()
    # every index decodes to a code that re-encodes to the same index
    idx = torch.randint(0, k, (256, g))
    level_codes = q.fsq._indices_to_codes(idx)  # noqa: SLF001  (n, g, len(levels))
    assert torch.equal(q.fsq.codes_to_indices(level_codes), idx)
    # forward codes are valid indices and lookup(codes) == forward z_q
    z = torch.randn(64, 32)
    codes, z_q, aux = q(z)
    assert codes.shape == (64, g)
    assert codes.dtype == torch.long
    assert int(codes.min()) >= 0
    assert int(codes.max()) < k
    torch.testing.assert_close(q.lookup(codes), z_q)
    assert not aux["codebook"].any()
    assert not aux["commit"].any()


def test_straight_through_gradients() -> None:
    torch.manual_seed(0)
    q = FiniteScalarQuantizer(dim=16, levels=(8, 5, 5, 5))
    z = torch.randn(32, 16, requires_grad=True)
    _, z_q, _ = q(z)
    z_q.pow(2).sum().backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0
    fsq: Any = q.fsq
    assert fsq.project_in.weight.grad.abs().sum() > 0
    assert fsq.project_out.weight.grad.abs().sum() > 0


def test_perplexity_and_codebook() -> None:
    q = FiniteScalarQuantizer(dim=8, levels=(5, 5))
    codes = torch.arange(25)[:, None]
    torch.testing.assert_close(q.perplexity(codes), torch.tensor([25.0]))
    assert q.codebook(0).shape == (25, 8)
    assert q.straight_through


def test_nero_chunk_tokenizer_drop_in(tmp_path) -> None:  # noqa: ANN001
    from rmind.models.nero_chunk_tokenizer import NeroChunkTokenizer  # noqa: PLC0415

    h, a, latent = 10, 13, 32
    config: dict[str, Any] = {
        "encoder": {
            "_target_": "torchvision.ops.MLP",
            "in_channels": h * a,
            "hidden_channels": [64, latent],
        },
        "quantizer": {
            "_target_": "rmind.components.vq.FiniteScalarQuantizer",
            "dim": latent,
            "levels": [8, 5, 5, 5],
        },
        "decoder": {
            "_target_": "torchvision.ops.MLP",
            "in_channels": latent,
            "hidden_channels": [64, h * a],
        },
        "action_horizon": h,
        "action_features": a,
        "keyframe_stride": 1,
        "event_weight": None,
    }
    tok = NeroChunkTokenizer(**config)
    assert tok.bits == pytest.approx(torch.tensor(1000.0).log2().item())
    x = torch.randn(8, h, a)
    # the training path: gradients reach the encoder AND the FSQ projections
    z = tok.encode_latent(x)
    _, z_q, _ = tok.quantizer(z)
    tok.decode_latent(z_q).pow(2).sum().backward()
    enc: Any = tok.encoder
    quantizer: Any = tok.quantizer
    assert enc[0].weight.grad.abs().sum() > 0
    assert quantizer.fsq.project_in.weight.grad.abs().sum() > 0
    tok.eval()
    codes = tok(x)
    assert codes.shape == (8, 1)
    assert tok.invert(codes).shape == (8, h * a)
    path = tmp_path / "tok.ckpt"
    torch.save(
        {
            "state_dict": tok.state_dict(),
            "hyper_parameters": dict(tok.hparams),
            "pytorch-lightning_version": "2.5.0",
        },
        path,
    )
    loaded = NeroChunkTokenizer.load_from_checkpoint(path, map_location="cpu").eval()
    assert isinstance(loaded.quantizer, FiniteScalarQuantizer)
    torch.testing.assert_close(loaded(x), codes)
    torch.testing.assert_close(loaded.invert(codes), tok.invert(codes))


def test_codebook_size_is_checked() -> None:
    k = 1000
    assert (
        FiniteScalarQuantizer(
            dim=8, levels=(8, 5, 5, 5), codebook_size=1000
        ).codebook_size
        == k
    )
    with pytest.raises(ValueError, match="prod"):
        FiniteScalarQuantizer(dim=8, levels=(8, 5, 5, 5), codebook_size=64)
