"""The playbook's instruments for a robot-native chunk tokenizer (P3), on a holdout.

    python -m rmind.scripts.nero_tokenizer_report --ckpt TOK.ckpt \\
        --experiment yaak/nero_robot/tokenizer --out report.json
    python -m rmind.scripts.nero_tokenizer_report --ckpt TOK.ckpt --synthetic --out r.json

Everything is PER AXIS, at 30 Hz after the keyframe interpolation, on REAL
(non-padded) steps, in the tokenizer's own (relative-mode, standardized) space:

1. explained variance EV = 1 - MSE/Var -- never an aggregate L1;
2. QUANTIZED vs UNQUANTIZED EV (decode z_q vs z): a big gap is a codebook/rate
   limit, both low is an autoencoder limit;
3. train vs holdout EV (equal = underfit: fix lr/schedule, not regularization);
4. TV ratio TV(recon)/TV(gt): < 0.6 over-smoothed (rate), > 1.1 an atom the
   output family cannot represent;
5. exact-atom share of recon vs target (the finger open atom);
6. a RATE-MATCHED DCT baseline: the 100-step chunk's DCT-II coefficients,
   scalar-quantized to the same bit budget (num_quantizers * log2(codebook)),
   best (coefficients, bits) split -- never a real-valued PCA;
7. residual-depth ladder EV@k and per-level perplexity (collapse tripwire only);
8. the interpolation floor: EV of interp(10 Hz keyframes) vs the 30 Hz truth;
9. recon_sd / target_sd (< 0.10 = dead channel), event-conditioned magnitude
   and an invariance probe per finger -- the only reliable dead-channel detectors.

The acceptance gates are evaluated and reported, but with 4 pulled episodes this
is a smoke; the numbers are recorded, not claimed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rmind.components.nn import _dct_ii_basis, linear_keyframe_matrix
from rmind.data.nero_robot import ARM_AXES, AXIS_NAMES, FINGER_AXES
from rmind.models.nero_chunk_tokenizer import NeroChunkTokenizer, total_variation
from rmind.models.nero_quality import per_axis_ev

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
GATES = {
    "ev_arm_min": 0.95,
    "ev_finger_min": 0.90,
    "quant_gap_max": 0.03,
    "tv_ratio": (0.8, 1.1),
    "dead_ratio": 0.10,
}


def collect(tokenizer: NeroChunkTokenizer, batches: Any, max_batches: int) -> tuple[Tensor, Tensor]:
    rows, reals = [], []
    for i, batch in enumerate(batches):
        if i >= max_batches:
            break
        with torch.no_grad():
            target, real = tokenizer.prepare(batch)
        rows.append(target)
        reals.append(real)
    return torch.cat(rows), torch.cat(reals)


@torch.no_grad()
def decode_both(tokenizer: NeroChunkTokenizer, x: Tensor) -> dict[str, Tensor]:
    z = tokenizer.encode_latent(x)
    codes, z_q, _ = tokenizer.quantizer(z)
    return {
        "z": z,
        "codes": codes,
        "quantized": tokenizer.decode_latent(z_q),
        "unquantized": tokenizer.decode_latent(z),
    }


@torch.no_grad()
def depth_ladder(tokenizer: NeroChunkTokenizer, codes: Tensor, x: Tensor, real: Tensor) -> list[float]:
    out = []
    partial = torch.zeros(codes.shape[0], tokenizer.latent_dim, device=codes.device)
    for level in range(tokenizer.quantizer.num_quantizers):
        partial += tokenizer.quantizer.codebook(level)[codes[:, level]]
        out.append(float(per_axis_ev(tokenizer.decode_latent(partial), x, real).mean()))
    return out


def dct_baseline(
    train: Tensor, holdout: Tensor, real: Tensor, *, bits: float
) -> dict[str, Any]:
    """Best rate-matched DCT-II scalar quantizer: 13 axes x k coefficients x b bits <= bits."""
    h, a = holdout.shape[1], holdout.shape[2]
    basis = _dct_ii_basis(h, h).to(holdout)  # (H, H) rows = coefficients
    c_train = torch.einsum("kt,nta->nka", basis, train)
    c_hold = torch.einsum("kt,nta->nka", basis, holdout)
    best: dict[str, Any] = {"ev": -math.inf}
    for k, b in itertools.product(range(1, 9), range(1, 9)):
        if a * k * b > bits:
            continue
        lo = c_train[:, :k].amin(dim=0)
        hi = c_train[:, :k].amax(dim=0)
        levels = 2**b
        step = (hi - lo).clamp_min(1e-9) / (levels - 1)
        q = ((c_hold[:, :k] - lo) / step).round().clamp(0, levels - 1) * step + lo
        recon = torch.einsum("kt,nka->nta", basis[:k], q)
        ev = per_axis_ev(recon, holdout, real)
        score = float(ev.mean())
        if score > best["ev"]:
            best = {
                "ev": score,
                "coefficients_per_axis": k,
                "bits_per_coefficient": b,
                "bits_used": a * k * b,
                "ev_per_axis": ev.tolist(),
            }
    return best


@torch.no_grad()
def invariance_probe(tokenizer: NeroChunkTokenizer, x: Tensor, axes: tuple[int, ...]) -> dict[str, float]:
    """Set one axis to its atom vs to atom + 2 (std units) over the 2nd half: the
    reconstruction of THAT axis must move (a dead channel's output does not)."""
    ref = torch.nan_to_num(tokenizer.event_reference)
    out = {}
    for axis in axes:
        low, high = x.clone(), x.clone()
        low[:, :, axis] = ref[axis]
        high[:, :, axis] = ref[axis]
        high[:, x.shape[1] // 2 :, axis] = ref[axis] + 2.0
        r_low = tokenizer.decode_latent(tokenizer.quantizer(tokenizer.encode_latent(low))[1])
        r_high = tokenizer.decode_latent(tokenizer.quantizer(tokenizer.encode_latent(high))[1])
        moved = (r_high[:, x.shape[1] // 2 :, axis] - r_low[:, x.shape[1] // 2 :, axis]).mean()
        out[AXIS_NAMES[axis]] = float(moved / 2.0)
    return out


@torch.no_grad()
def report(  # noqa: PLR0914
    tokenizer: NeroChunkTokenizer, train: tuple[Tensor, Tensor], holdout: tuple[Tensor, Tensor]
) -> dict[str, Any]:
    x, real = holdout
    xt, real_t = train
    hold = decode_both(tokenizer, x)
    tr = decode_both(tokenizer, xt)
    ev_q = per_axis_ev(hold["quantized"], x, real)
    ev_u = per_axis_ev(hold["unquantized"], x, real)
    ev_train = per_axis_ev(tr["quantized"], xt, real_t)
    tv_ratio = total_variation(hold["quantized"], real) / total_variation(x, real).clamp_min(1e-9)
    interp = linear_keyframe_matrix(tokenizer.action_horizon, tokenizer.keyframe_stride).to(x)
    floor = torch.einsum("tk,nka->nta", interp, x[:, :: tokenizer.keyframe_stride])
    ev_floor = per_axis_ev(floor, x, real)
    w = real.bool()
    sd_ratio = hold["quantized"][w].std(0) / x[w].std(0).clamp_min(1e-9)
    ref = torch.nan_to_num(tokenizer.event_reference)
    atom_t = ((x - ref).abs() < 1e-3)[w].float().mean(0)  # noqa: PLR2004
    atom_r = ((hold["quantized"] - ref).abs() < 1e-3)[w].float().mean(0)  # noqa: PLR2004
    event = ((x - ref).abs() > tokenizer.event_threshold) & w[..., None]
    event_mag = {}
    for axis in FINGER_AXES:
        e = event[..., axis]
        if bool(e.any()):
            num = (hold["quantized"][..., axis] - ref[axis]).abs()[e].mean()
            den = (x[..., axis] - ref[axis]).abs()[e].mean()
            event_mag[AXIS_NAMES[axis]] = float(num / den.clamp_min(1e-9))
    bits = tokenizer.bits
    dct = dct_baseline(xt, x, real, bits=bits)
    perplexity = tokenizer.quantizer.perplexity(hold["codes"]).tolist()
    invariance = invariance_probe(tokenizer, x[: min(len(x), 256)], tuple(FINGER_AXES))
    arm, fin = list(ARM_AXES), list(FINGER_AXES)
    gates = {
        "ev_arm": float(ev_q[arm].min()) >= GATES["ev_arm_min"],
        "ev_finger": float(ev_q[fin].min()) >= GATES["ev_finger_min"],
        "beats_dct": float(ev_q.mean()) > dct["ev"],
        "quant_gap": float((ev_u - ev_q).max()) <= GATES["quant_gap_max"],
        "tv_ratio": bool(
            ((tv_ratio >= GATES["tv_ratio"][0]) & (tv_ratio <= GATES["tv_ratio"][1])).all()
        ),
        "no_dead_channel": bool((sd_ratio >= GATES["dead_ratio"]).all()),
        "finger_invariance": all(v > 0.1 for v in invariance.values()),  # noqa: PLR2004
    }
    return {
        "relative_mode": tokenizer.relative_mode,
        "bits": bits,
        "bits_per_value": bits / (tokenizer.action_horizon * tokenizer.action_features),
        "num_quantizers": tokenizer.quantizer.num_quantizers,
        "codebook_size": tokenizer.quantizer.codebook_size,
        "holdout_rows": len(x),
        "train_rows": len(xt),
        "axes": list(AXIS_NAMES),
        "ev_quantized": ev_q.tolist(),
        "ev_unquantized": ev_u.tolist(),
        "ev_train_quantized": ev_train.tolist(),
        "quant_gap": (ev_u - ev_q).tolist(),
        "tv_ratio": tv_ratio.tolist(),
        "interp_floor_ev": ev_floor.tolist(),
        "recon_sd_over_target_sd": sd_ratio.tolist(),
        "atom_share_target": atom_t.tolist(),
        "atom_share_recon": atom_r.tolist(),
        "event_magnitude_ratio": event_mag,
        "invariance_probe": invariance,
        "dct_rate_matched": dct,
        "depth_ladder_ev": depth_ladder(tokenizer, hold["codes"], x, real),
        "perplexity_per_level": perplexity,
        "gates": gates,
        "gates_note": "smoke only: too little data to claim these",
    }


def batches_for(args: argparse.Namespace, split: str) -> Any:
    if args.synthetic:
        from rmind.datamodules.nero_robot_random import NeroRobotRandomDataLoader  # noqa: PLC0415

        return NeroRobotRandomDataLoader(
            num_batches=args.max_batches,
            batch_size=16,
            num_frames=4,
            images=False,
            seed=0 if split == "train" else 100_000,
        )
    import rmind  # noqa: F401, PLC0415
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from hydra.utils import instantiate  # noqa: PLC0415

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={args.experiment}", *args.override])
    dm = instantiate(cfg.datamodule)
    return dm.train_dataloader() if split == "train" else dm.val_dataloader()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = NeroChunkTokenizer.load_from_checkpoint(args.ckpt, map_location="cpu").eval()
    result = report(
        tokenizer,
        collect(tokenizer, batches_for(args, "train"), args.max_batches),
        collect(tokenizer, batches_for(args, "val"), args.max_batches),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("relative_mode", "bits", "gates")}, indent=1))  # noqa: T201
    q = result["ev_quantized"]
    print(  # noqa: T201
        "EV quantized arm %.3f finger %.3f | DCT %.3f | gap max %.3f"
        % (
            sum(q[:7]) / 7,
            sum(q[7:]) / 6,
            result["dct_rate_matched"]["ev"],
            max(result["quant_gap"]),
        )
    )


if __name__ == "__main__":
    main()
