# Exporting decode strategies (argmax · chain_greedy · entropy-gated)

Pull-and-run guide for exporting the fixed joint policy under three decode
strategies to ONNX (and onward to TensorRT on the dev-kit). Extends the standard
[Export runbook](https://app.notion.com/p/Export-38ed658ccf8780f39581cfd35edd706a);
only the differences are here.

Branch: `feat/decode-strategy-eval`. Shipped in it (tokenizer `model-y74asdtd:v9`):
`luts_y74asdtd.pt` (chain conditionals), `decode_table_y74asdtd.pt` (codebook
decode, host-side only). Regenerate for a different tokenizer with
`scripts.calibrate_decode_luts --codes-cache ...` and the decode-table snippet below.

## TL;DR — which to export
| strategy | determinism | TRT engine | how | recommendation |
|---|---|---|---|---|
| **argmax** | deterministic | ✅ standalone | existing `finetuned.yaml` | today's export; baseline |
| **chain_greedy** | deterministic | ✅ standalone | `finetuned_chain_greedy.yaml` (needs calibrated ckpt) | **recommended deployable** — 3× lower conflict than argmax, mutex-redundant |
| **entropy-gated** | stochastic (RNG) | ⚠ not standalone | `finetuned_heads.yaml` + host-side `decode.py` | experimental; recovers launches but adds harsh brakes; drivr integration needed |

RNG cannot live in a deterministic TRT engine, so entropy-gated (and any sampled
decode) is served by exporting a **heads** engine (raw code-logits + offset table)
and decoding host-side in drivr. argmax/chain_greedy also work host-side off the
same heads engine if you prefer one engine + switchable decode.

## 1. argmax (unchanged)
```bash
just export-onnx export=yaak/control_transformer/finetuned \
    model.artifact=<FIX_ARTIFACT> f=<ONNX>
```

## 2. chain_greedy  (deterministic, recommended)
```bash
# a) bake the shipped chain LUTs + decode_strategy into a self-contained ckpt
#    (no training data needed; --luts reuses the committed file)
uv run python -m rmind.scripts.calibrate_decode_luts \
    --ckpt <FIX_CKPT> --luts luts_y74asdtd.pt --beta 1.0 \
    --out model.chain_greedy.ckpt
# (to recompute LUTs from scratch instead: --codes-cache <train code cache> --save-luts luts.pt)

# b) export (checkpoint path via env var read by the config)
CHAIN_GREEDY_CKPT=$PWD/model.chain_greedy.ckpt \
just export-onnx export=yaak/control_transformer/finetuned_chain_greedy f=<ONNX>
#   expect: [torch.onnx] Verify output accuracy... ✅   (verified here on v9)

# c) TRT on the dev-kit — identical to the standard runbook (drivr build_trt_engine.py --fp16).
#    The engine is a drop-in replacement for the argmax engine (same I/O contract).
```
`decode_beta=1.0` was the sweep winner at v9 (re-tune with the sweep engine if the
code head changes materially). Uncalibrated chain buffers are zeros ⇒ chain_greedy
degrades exactly to argmax, so a mis-calibration fails safe.

## 3. entropy-gated  (stochastic → heads engine + host-side decode)
```bash
# a) export the deterministic heads engine (any finetuned ckpt; no calibration)
HEADS_CKPT=$PWD/<FIX_CKPT> \
just export-onnx export=yaak/control_transformer/finetuned_heads f=<HEADS_ONNX>
#   outputs: code_logits (1,G,C), offsets (1,G,C,action_dim). Verified deterministic.

# b) TRT the heads engine on the dev-kit (standard build_trt_engine.py --fp16).

# c) drivr applies decode host-side on the engine outputs:
#    (port src/rmind/inference/decode.py — pure torch, no rmind-model dep)
python - <<'PY'
import torch; from rmind.inference.decode import decode
dt = torch.load("decode_table_y74asdtd.pt")["decode_table"]
luts = torch.load("luts_y74asdtd.pt")["luts"]
gen = torch.Generator().manual_seed(0)                       # seed for reproducibility
# code_logits, offsets = <heads engine outputs>
action = decode(code_logits, offsets, dt,
                strategy="entropy_gated", tau=0.7, gate_nats=1.56, generator=gen)
PY
```
`decode.py` implements all three strategies over the heads outputs, so one heads
engine + host-side decode covers argmax/chain_greedy/entropy-gated without
re-exporting. gate_nats=1.56 = the val q0-entropy 90th percentile (top-decile-
ambiguous states); tau=0.7. Stochastic: pin drivr's RNG and log the seed.

## Verification (do before trusting an engine — mirrors the standard runbook)
- rmind side: `just benchmark-onnx onnx=<ONNX> export=<the config used> ...` →
  expect "ONNX vs PyTorch agree within tolerance". chain_greedy verified here:
  `Verify output accuracy... ✅`.
- drivr side (dev-kit): `benchmark_models.py --onnx-full ... --trt-full ...` →
  ONNX vs TRT match + equal values across the rmind/drivr benchmarks.
- For the heads engine, also numerically check host-side `decode(strategy="argmax")`
  reproduces the argmax engine's action on the same inputs.

## Regenerate artifacts for a new tokenizer
```bash
# chain LUTs from a train code cache (offset_head extract output):
uv run python -m rmind.scripts.calibrate_decode_luts --ckpt <ckpt> \
    --codes-cache <cache> --save-luts luts_<tok>.pt --out /dev/null || true
# decode table (invert all C**G combos):
uv run python -c "import torch;from rmind.scripts.offset_diag import load_policy;\
tok=load_policy('<ckpt>','cpu').objectives['policy'].tokenizer;G,C=4,16;\
cb=torch.cartesian_prod(*[torch.arange(C)]*G);\
d=torch.cat([tok.invert(b) for b in cb.split(4096)]);\
torch.save({'decode_table':d.float(),'G':G,'C':C},'decode_table_<tok>.pt')"
```
