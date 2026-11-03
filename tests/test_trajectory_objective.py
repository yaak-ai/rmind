import numpy as np
import torch

from rmind.components.objectives.trajectory import IGNORE, TrajectoryObjective
from rmind.components.transformer import CrossAttentionDecoder

DIM = 32
FAST_TOLERANCE_M = 0.1  # FAST rounding at gamma 10 (x, y) / 100 (theta)
BATCH = 2


def _decoder(*, causal: bool) -> CrossAttentionDecoder:
    return CrossAttentionDecoder(dim_model=DIM, num_layers=2, num_heads=4, causal=causal).eval()


def test_causal_decoder_ignores_future_queries() -> None:
    torch.manual_seed(0)
    decoder = _decoder(causal=True)
    key = torch.randn(2, 5, DIM)
    x = torch.randn(2, 6, DIM)
    y = x.clone()
    y[:, 4:] = torch.randn(2, 2, DIM)  # change only the last two queries
    with torch.no_grad():
        a, b = decoder(x, key, key), decoder(y, key, key)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6)
    assert not torch.allclose(a[:, 4:], b[:, 4:])


def test_non_causal_decoder_is_unchanged_by_default() -> None:
    torch.manual_seed(0)
    decoder = _decoder(causal=False)
    key = torch.randn(2, 5, DIM)
    x = torch.randn(2, 6, DIM)
    y = x.clone()
    y[:, 5] = torch.randn(2, DIM)
    with torch.no_grad():
        a, b = decoder(x, key, key), decoder(y, key, key)
    assert not torch.allclose(a[:, 0], b[:, 0])  # every query sees every other one


def _objective() -> TrajectoryObjective:
    return TrajectoryObjective(decoder=_decoder(causal=True), tokenizer_dir="assets/fast_trajectory_h10", dim=DIM)


def test_targets_are_shifted_inputs_and_decode_back() -> None:
    obj = _objective()
    t = torch.arange(1, 11, dtype=torch.float32) / 3
    poses = torch.stack([torch.zeros(10), 10 * t, torch.zeros(10)], -1)[None].repeat(3, 1, 1)  # 36 km/h straight
    inputs, targets = obj._encode(poses)  # noqa: SLF001
    assert (inputs[:, 0] == obj.bos).all()
    valid = targets != IGNORE
    n = valid.sum(1)
    assert (targets[torch.arange(3), n - 1] == obj.eos).all()
    # inputs after BOS are the targets before EOS
    assert torch.equal(inputs[0, 1 : n[0]], targets[0, : n[0] - 1])
    ids = [targets[0, : n[0] - 1].tolist()]
    rec = obj.fast.decode(ids)[0]
    assert np.abs(rec - poses[0].numpy()).max() < FAST_TOLERANCE_M


def test_compute_and_generate_shapes() -> None:
    obj = _objective().eval()
    latent = torch.randn(BATCH, 6, 8, DIM)
    poses, ids = obj.generate(latent)
    assert poses.shape == (BATCH, 10, 3)
    assert len(ids) == BATCH


def test_tied_output_layer_is_the_embedding_and_never_emits_bos_pad() -> None:
    obj = _objective().eval()
    assert obj.head is None  # tied by default
    key = torch.randn(BATCH, 8, DIM)
    tokens = torch.full((BATCH, 1), obj.bos)
    logits = obj._logits(key, tokens)  # noqa: SLF001
    assert logits.shape[-1] == obj.token_embedding.num_embeddings  # same ids as the input
    with torch.no_grad():
        obj.head_bias[obj.bos] = 1e4  # even a huge BOS / PAD logit must not be emitted
        obj.head_bias[obj.pad] = 1e4
    _, ids = obj.generate_from_key(key)
    assert all(obj.bos not in row and obj.pad not in row for row in ids)
