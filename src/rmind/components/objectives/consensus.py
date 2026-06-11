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


def mode_aware_anchor(
    samples: Tensor,
    steer_idx: int,
    *,
    gap_thr: float = MODE_GAP_THRESHOLD,
    min_frac: float = MODE_MIN_FRACTION,
) -> tuple[Tensor, Tensor, Tensor]:
    """Winner-take-all consensus over draws.

    samples: (B, K, H, A) raw-space action chunks (K draws per frame).
    steer_idx: index of the steering channel in A (the mode axis).

    Returns (anchor (B, H, A), bimodal (B,) bool, mode_sep (B,) largest gap).
    Frames whose draws contain non-finite values get a non-finite mean anchor
    (consistent with single-draw behavior on poisoned conditions).
    """
    if samples.ndim != 4:  # noqa: PLR2004
        msg = f"samples must be (B, K, H, A), got {tuple(samples.shape)}"
        raise ValueError(msg)
    b, k = samples.shape[:2]
    sig = samples[..., steer_idx].mean(dim=2)  # (B, K) chunk-mean steering
    sig_sorted, order = torch.sort(sig, dim=1)
    gaps = sig_sorted.diff(dim=1)  # (B, K-1)
    mode_sep, split = gaps.max(dim=1)  # largest gap and its index
    left_n = split + 1
    minority = torch.minimum(left_n, k - left_n).float() / k
    bimodal = (
        (mode_sep > gap_thr) & (minority >= min_frac) & torch.isfinite(mode_sep)
    )

    anchor = samples.mean(dim=1)  # (B, H, A) default: mean-of-K
    for i in torch.nonzero(bimodal).flatten().tolist():
        n_left = int(left_n[i])
        members = order[i, :n_left] if n_left >= k - n_left else order[i, n_left:]
        anchor[i] = samples[i, members].mean(dim=0)
    return anchor, bimodal, mode_sep
