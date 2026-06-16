"""Winner-take-all consensus over K action-chunk draws (MBR-style readout).

Plain mean-of-K is a mode-averaging estimator: on a bimodal conditional (turn
left XOR right; or, per the copycat finding, follow-history XOR follow-route)
it splits the difference — the regression-to-the-mean failure a generative
policy exists to avoid. Measured on the held-out val drive: 77% of spike
frames are bimodal (mode separation ~0.6), and committing to the dominant mode
cuts spike steering L1 from 0.53 (mean-of-K) to 0.34.

Procedure per frame: cluster the K draws on a 1-D maneuver signature
(chunk-mean steering) via the largest gap in the sorted values; declare two
modes when the gap exceeds `gap_thr` AND the minority side holds at least
`min_frac` of draws (outlier guard); commit to the dominant cluster (draw
count ~ probability mass) and average the full chunks within it. Unimodal
frames fall back exactly to mean-of-K.

Precedents: propose-then-select in motion forecasting (MultiPath anchors,
MTR/DenseTNT NMS, Trajectron++ "most likely" deployment; MDN dominant
component), Minimum-Bayes-Risk consensus decoding (Kumar & Byrne 2004), and
sample-then-select policies (Implicit BC, SfBC, Diffusion-ES).
"""

import torch
from torch import Tensor

MODE_GAP_THRESHOLD = 0.15
MODE_MIN_FRACTION = 0.125


def split_modes(
    samples: Tensor,
    steer_idx: int,
    *,
    gap_thr: float = MODE_GAP_THRESHOLD,
    min_frac: float = MODE_MIN_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Largest-gap mode split of K draws per frame (at most 2 modes).

    samples: (B, K, H, A) raw-space action chunks.
    Returns (bimodal (B,) bool, order (B, K) sort indices on the signature,
    left_n (B,) draws left of the split, mode_sep (B,) largest gap). For a
    bimodal row, the dominant cluster is order[:left_n] when left_n >= K-left_n
    else order[left_n:], the minority cluster the complement.
    """
    if samples.ndim != 4:  # noqa: PLR2004
        msg = f"samples must be (B, K, H, A), got {tuple(samples.shape)}"
        raise ValueError(msg)
    k = samples.shape[1]
    sig = samples[..., steer_idx].mean(dim=2)  # (B, K) chunk-mean steering
    sig_sorted, order = torch.sort(sig, dim=1)
    gaps = sig_sorted.diff(dim=1)  # (B, K-1)
    mode_sep, split = gaps.max(dim=1)  # largest gap and its index
    left_n = split + 1
    minority = torch.minimum(left_n, k - left_n).float() / k
    bimodal = (
        (mode_sep > gap_thr) & (minority >= min_frac) & torch.isfinite(mode_sep)
    )
    return bimodal, order, left_n, mode_sep


def _members(
    order: Tensor, left_n: Tensor, i: int, *, dominant: bool, k: int
) -> Tensor:
    n_left = int(left_n[i])
    left_is_dominant = n_left >= k - n_left
    take_left = left_is_dominant == dominant
    return order[i, :n_left] if take_left else order[i, n_left:]


def mode_aware_anchor(
    samples: Tensor,
    steer_idx: int,
    *,
    gap_thr: float = MODE_GAP_THRESHOLD,
    min_frac: float = MODE_MIN_FRACTION,
    anchor: str = "mean",
) -> tuple[Tensor, Tensor, Tensor]:
    """Winner-take-all consensus over draws.

    samples: (B, K, H, A) raw-space action chunks (K draws per frame).
    steer_idx: index of the steering channel in A (the mode axis).
    anchor: "mean" = average the dominant cluster's chunks (lowest variance;
        a synthetic chunk). "medoid" = the actual draw closest to that mean
        (guaranteed model sample, dynamically coherent by construction; keeps
        one draw's sampling noise). Unimodal frames use all K draws either way.

    Returns (anchor (B, H, A), bimodal (B,) bool, mode_sep (B,) largest gap).
    Frames whose draws contain non-finite values get a non-finite mean anchor
    (consistent with single-draw behavior on poisoned conditions).
    """
    if anchor not in {"mean", "medoid"}:
        msg = f"anchor must be 'mean' or 'medoid', got {anchor!r}"
        raise ValueError(msg)
    b, k = samples.shape[:2]
    bimodal, order, left_n, mode_sep = split_modes(
        samples, steer_idx, gap_thr=gap_thr, min_frac=min_frac
    )

    out = samples.mean(dim=1)  # (B, H, A) default: mean-of-K
    for i in torch.nonzero(bimodal).flatten().tolist():
        members = _members(order, left_n, i, dominant=True, k=k)
        out[i] = samples[i, members].mean(dim=0)
    if anchor == "medoid":
        all_idx = torch.arange(k, device=samples.device)
        for i in range(b):
            members = (
                _members(order, left_n, i, dominant=True, k=k)
                if bool(bimodal[i])
                else all_idx
            )
            member_chunks = samples[i, members]  # (M, H, A)
            if not torch.isfinite(member_chunks).all():
                continue  # keep the (non-finite) mean, mirroring single-draw
            dists = (member_chunks - out[i]).flatten(1).norm(dim=1)
            out[i] = member_chunks[dists.argmin()]
    return out, bimodal, mode_sep
