"""Separate arm / hand codes: AxisGroupChunkMLP{Encoder,Decoder} + GroupedResidualVQ."""

from pathlib import Path

import pytest
import torch

from rmind.components.nn import AxisGroupChunkMLPDecoder, AxisGroupChunkMLPEncoder
from rmind.components.vq import GroupedResidualVQ
from rmind.models.nero_chunk_tokenizer import NeroChunkTokenizer

H, A, LATENT, CODEBOOK, DEPTH = 10, 13, 32, 16, 2
GROUPS = ((0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12))


def _config() -> dict:
    mlp = {
        "num_steps": H,
        "num_axes": A,
        "axis_groups": [list(g) for g in GROUPS],
        "hidden_channels": [64],
        "latent_dim": LATENT,
    }
    return {
        "encoder": {"_target_": "rmind.components.nn.AxisGroupChunkMLPEncoder"} | mlp,
        "quantizer": {
            "_target_": "rmind.components.vq.GroupedResidualVQ",
            "groups": 2,
            "group_dim": LATENT,
            "codebook_size": CODEBOOK,
            "num_quantizers_per_group": DEPTH,
            "kmeans_init": False,
        },
        "decoder": {"_target_": "rmind.components.nn.AxisGroupChunkMLPDecoder"} | mlp,
        "action_horizon": H,
        "action_features": A,
        "keyframe_stride": 1,
        "event_weight": None,
    }


def test_axis_groups_must_partition() -> None:
    with pytest.raises(ValueError, match="partition"):
        AxisGroupChunkMLPEncoder(
            num_steps=H,
            num_axes=A,
            axis_groups=((0, 1), (2,)),
            hidden_channels=(8,),
            latent_dim=4,
        )


def test_groups_are_independent() -> None:
    torch.manual_seed(0)
    enc = AxisGroupChunkMLPEncoder(
        num_steps=H, num_axes=A, axis_groups=GROUPS, hidden_channels=(16,), latent_dim=8
    )
    dec = AxisGroupChunkMLPDecoder(
        num_steps=H, num_axes=A, axis_groups=GROUPS, hidden_channels=(16,), latent_dim=8
    )
    x = torch.randn(4, H * A)
    y = x.reshape(4, H, A).clone()
    y[..., 9] += 1.0  # a finger axis
    z, zy = enc(x), enc(y.flatten(1))
    assert z.shape == (4, 16)
    torch.testing.assert_close(z[:, :8], zy[:, :8])  # the arm latent is untouched
    assert not torch.allclose(z[:, 8:], zy[:, 8:])
    out, out_y = dec(z).reshape(4, H, A), dec(zy).reshape(4, H, A)
    torch.testing.assert_close(out[..., :7], out_y[..., :7])  # arm axes in place
    assert not torch.allclose(out[..., 7:], out_y[..., 7:])


def test_grouped_rvq_interface() -> None:
    torch.manual_seed(0)
    q = GroupedResidualVQ(
        groups=2,
        group_dim=LATENT,
        codebook_size=CODEBOOK,
        num_quantizers_per_group=DEPTH,
        kmeans_init=False,
    )
    assert (q.dim, q.num_quantizers, q.codebook_size) == (2 * LATENT, 4, CODEBOOK)
    z = torch.randn(64, 2 * LATENT)
    codes, z_q, vq = q(z)
    assert codes.shape == (64, 4)
    assert codes.max() < CODEBOOK
    torch.testing.assert_close(q.lookup(codes), z_q)
    assert q.perplexity(codes).shape == (4,)
    assert set(vq) == {"codebook", "commit"}
    # level 2 is the hand group's first level, in the second latent slice
    book = q.codebook(2)
    assert book.shape == (CODEBOOK, 2 * LATENT)
    assert torch.all(book[:, :LATENT] == 0)


def test_split_tokenizer_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    tok = NeroChunkTokenizer(**_config()).eval()
    assert tok.bits == 4 * 4  # 2 groups x depth 2 x log2(16)
    assert tok.latent_dim == 2 * LATENT
    x = torch.randn(8, H, A)
    codes = tok(x)
    assert codes.shape == (8, 4)
    decoded = tok.invert(codes)
    assert decoded.shape == (8, H * A)

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
    assert isinstance(loaded.quantizer, GroupedResidualVQ)
    torch.testing.assert_close(loaded(x), codes)
    torch.testing.assert_close(loaded.invert(codes), decoded)
