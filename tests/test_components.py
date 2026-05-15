from itertools import pairwise

import torch
from torch.testing import assert_close, make_tensor

from rmind.components.base import Modality, SummaryToken, TokenType
from rmind.components.episode import Episode, EpisodeBuilder
from rmind.components.loss import GramAnchoringLoss
from rmind.components.mask import (
    CausalAttentionMaskBuilder,
    FactorizedCausalAttentionMaskBuilder,
    TorchAttentionMaskLegend,
)
from rmind.components.nn import Sequential
from rmind.components.norm import Scaler, UniformBinner


def test_scaler(device: torch.device) -> None:
    module = Scaler(in_range=(0.0, 100.0), out_range=(-1.0, 1.0)).to(device)

    x = make_tensor(
        1024,
        dtype=torch.float,
        device=device,
        low=module.in_range[0].item(),
        high=module.in_range[1].item(),
    )

    x_rt = module.invert(module(x))

    assert_close(x_rt, x)


def test_uniform_binner(device: torch.device) -> None:
    module = UniformBinner(range=(5.0, 130.0), bins=1024).to(device)
    x = make_tensor(
        1024,
        dtype=torch.float,
        device=device,
        low=module.range[0].item(),
        high=module.range[1].item(),
    )
    x_rt = module.invert(module(x))

    bin_width = (module.range[1] - module.range[0]) / module.bins
    assert_close(x_rt, x, rtol=0.0, atol=(bin_width / 2.0).item())


def test_sequential(device: torch.device) -> None:
    module = Sequential(
        *(
            Scaler(in_range=in_range, out_range=out_range)
            for in_range, out_range in pairwise((0.0, 10.0**x) for x in range(1, 6))
        )
    ).to(device)

    x = make_tensor(
        1024,
        dtype=torch.float,
        device=device,
        low=module[0].in_range[0].item(),  # ty:ignore[not-subscriptable]
        high=module[0].in_range[1].item(),  # ty:ignore[not-subscriptable]
    )

    x_rt = module.invert(module(x))

    assert_close(x_rt, x)


def test_gram_anchoring_loss_zero_for_matching_features(device: torch.device) -> None:
    target = make_tensor(
        (8, 16), dtype=torch.float32, device=device, low=-1.0, high=1.0
    )
    module = GramAnchoringLoss(patches=4).to(device)

    loss = module(target, target.clone())

    assert_close(loss, torch.zeros((), dtype=loss.dtype, device=device))


def test_gram_anchoring_loss_runs_without_gram(device: torch.device) -> None:
    input = make_tensor((8, 16), dtype=torch.float32, device=device, low=-1.0, high=1.0)
    target = make_tensor(
        (8, 16), dtype=torch.float32, device=device, low=-1.0, high=1.0
    )
    module = GramAnchoringLoss(patches=4, weight_gram=0.0).to(device)

    loss = module(input, target)

    assert loss.isfinite()


def _causal_mask_inputs(device: torch.device) -> tuple[dict, dict]:
    index = {
        Modality.IMAGE.value: {"obs": torch.tensor([[0], [6]], device=device)},
        Modality.CONTINUOUS.value: {"act": torch.tensor([[4], [10]], device=device)},
        Modality.CONTEXT.value: {},
        Modality.DISCRETE.value: {},
        Modality.FORESIGHT.value: {"future": torch.tensor([[3], [9]], device=device)},
        Modality.SUMMARY.value: {
            SummaryToken.OBSERVATION_SUMMARY.value: torch.tensor(
                [[1], [7]], device=device
            ),
            SummaryToken.OBSERVATION_HISTORY.value: torch.tensor(
                [[2], [8]], device=device
            ),
            SummaryToken.ACTION_SUMMARY.value: torch.tensor([[5], [11]], device=device),
        },
    }
    timestep = {
        TokenType.OBSERVATION.value: {Modality.IMAGE.value: {"obs": 0}},
        TokenType.ACTION.value: {Modality.CONTINUOUS.value: {"act": 4}},
        TokenType.SPECIAL.value: {
            Modality.FORESIGHT.value: {"future": 3},
            Modality.SUMMARY.value: {
                SummaryToken.OBSERVATION_SUMMARY.value: 1,
                SummaryToken.OBSERVATION_HISTORY.value: 2,
                SummaryToken.ACTION_SUMMARY.value: 5,
            },
        },
    }
    return index, timestep


def test_causal_attention_mask_builder_keeps_full_history_edges(
    device: torch.device,
) -> None:
    index, timestep = _causal_mask_inputs(device)

    mask = CausalAttentionMaskBuilder()(
        index=index, timestep=timestep, legend=TorchAttentionMaskLegend
    )

    assert mask.shape == (12, 12)
    assert not mask[6, 0]
    assert not mask[9, 0]
    assert not mask[7, 3]
    assert not mask[10, 4]
    assert not mask[11, 4]


def test_factorized_causal_attention_mask_builder(device: torch.device) -> None:
    index, timestep = _causal_mask_inputs(device)

    mask = FactorizedCausalAttentionMaskBuilder()(
        index=index, timestep=timestep, legend=TorchAttentionMaskLegend
    )

    assert mask.spatial.mask_tensor.shape == (6, 6)
    assert mask.temporal.mask_tensor.shape == (2, 2)
    assert_close(
        mask.temporal.mask_tensor,
        torch.tensor([[False, True], [False, False]], device=device),
    )


def test_embeddings_unpacked_order(
    episode_builder: EpisodeBuilder, episode: Episode
) -> None:
    """embeddings_unpacked must pack tokens in the order declared in episode_builder.timestep.

    The index assigns flat positions by enumerating episode_builder.timestep, so the
    slice at each offset must match the corresponding token's embeddings. Removing
    sorted() from embeddings_unpacked causes TensorDict to iterate alphabetically,
    placing e.g. action tokens before image tokens and failing this check.
    """
    unpacked = episode.embeddings_unpacked  # (b, t, s, d)
    embeddings = episode.embeddings

    pos = 0
    for idx, token in enumerate(episode_builder.timestep):
        key = (token.modality.value, str(token.name))
        expected = embeddings[key]  # (b, t, n, d)
        n = expected.shape[2]
        actual = unpacked[:, :, pos : pos + n, :]
        assert_close(
            actual, expected, msg=f"wrong slice at timestep position {idx} {token}"
        )
        pos += n


def test_index_unpacked_alignment(episode: Episode) -> None:
    """index.parse(flat embeddings_unpacked) must recover the per-token embeddings.

    This is the load-bearing invariant for every downstream `index.parse(encoder_output)`
    call: a token's position in the flat (t*s,) sequence — assigned by the builder when
    constructing the index — must match where that token actually sits in
    embeddings_unpacked. If the two disagree, every objective reads the wrong slice.
    """
    unpacked = episode.embeddings_unpacked  # (b, t, s, d)
    b, t, s, d = unpacked.shape
    flat = unpacked.reshape(b, t * s, d)  # same shape the encoder returns

    recovered = episode.index.parse(flat)  # nested TensorDict, leaves (b, t, n, d)
    expected = episode.embeddings

    for key in recovered.keys(include_nested=True, leaves_only=True):
        assert_close(
            recovered[key], expected[key], msg=f"index/unpacked misaligned for {key}"
        )
