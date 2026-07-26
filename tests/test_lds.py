import json
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from rmind.components.lds import LDSWeights

NBINS = 8


def _lds_stats() -> dict:
    # Two channels. Channel 0: density decays toward the tail (cruise-heavy, like
    # real steering) so the tail weight is high. Channel 1: uniform density (flat
    # weights). Edges span [0, 1].
    edges = [list(torch.linspace(0.0, 1.0, NBINS + 1)) for _ in range(2)]
    decaying = torch.linspace(1.0, 0.05, NBINS)
    decaying = (decaying / decaying.sum()).tolist()
    uniform = (torch.ones(NBINS) / NBINS).tolist()
    return {
        "lds": {
            "edges": [[float(x) for x in e] for e in edges],
            "emp": [decaying, uniform],
            "smooth": [decaying, uniform],
            "model_keys": ["longitudinal", "steering_angle"],
        }
    }


def _stats_file(tmp_path: Path) -> str:
    path = tmp_path / "action_norm.json"
    path.write_text(json.dumps(_lds_stats()))
    return str(path)


def test_weights_mean_one_over_empirical(tmp_path: Path) -> None:
    # E_emp[w] == 1 per channel by construction (scale-preserving normalization).
    w = LDSWeights.from_stats_file(_stats_file(tmp_path), alpha=1.0, cap=1e9)
    emp = torch.tensor(_lds_stats()["lds"]["emp"])
    mean = (emp * w.bin_weight).sum(dim=1)
    assert_close(mean, torch.ones(2), atol=1e-5, rtol=0)


def test_rare_tail_weighted_more_than_dense_bulk(tmp_path: Path) -> None:
    # Channel 0 density decays, so the high-intensity (rare) bin must outweigh
    # the low-intensity (dense) bin; channel 1 is uniform so weights are flat.
    w = LDSWeights.from_stats_file(_stats_file(tmp_path), alpha=1.0, cap=1e9)
    assert w.bin_weight[0, -1] > w.bin_weight[0, 0]
    assert_close(w.bin_weight[1], torch.ones(NBINS), atol=1e-5, rtol=0)


def test_alpha_zero_is_uniform(tmp_path: Path) -> None:
    # alpha=0 => (1/density)^0 == 1 everywhere => no reweighting.
    w = LDSWeights.from_stats_file(_stats_file(tmp_path), alpha=0.0, cap=1e9)
    assert_close(w.bin_weight, torch.ones(2, NBINS), atol=1e-5, rtol=0)


def test_cap_bounds_weights(tmp_path: Path) -> None:
    cap = 2.0
    w = LDSWeights.from_stats_file(_stats_file(tmp_path), alpha=1.0, cap=cap)
    # Cap applies pre-normalization; post-normalization the max can only shrink.
    assert w.bin_weight.max() <= cap + 1e-5


def test_lookup_buckets_and_clamps(tmp_path: Path) -> None:
    w = LDSWeights.from_stats_file(_stats_file(tmp_path), alpha=1.0, cap=1e9)
    label = torch.tensor([
        [0.01, 0.5],
        [0.99, 0.5],
        [5.0, 0.5],
    ])  # last row out-of-range
    out = w(label)
    assert out.shape == label.shape
    # out-of-range high label -> top bin (highest channel-0 weight)
    assert_close(out[2, 0], w.bin_weight[0, -1], atol=0, rtol=0)
    # near-zero label -> first (lowest-weight, dense) channel-0 bin
    assert_close(out[0, 0], w.bin_weight[0, 0], atol=0, rtol=0)


def test_rejects_wrong_channel_count(tmp_path: Path) -> None:
    w = LDSWeights.from_stats_file(_stats_file(tmp_path), alpha=0.5, cap=15.0)
    with pytest.raises(ValueError, match="channels"):
        w(torch.zeros(4, 3))  # 3 channels, weighter expects 2
