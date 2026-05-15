# CLAUDE.md

Codebase notes for AI assistants working in this repo.

## Export architecture

The model has a single forward path used for both training and ONNX export.
Training calls `training_step` / `validation_step` directly (bypassing `forward`);
export calls `ControlTransformer.forward` via `torch.export.export`.

`EpisodeBuilder.forward` always returns `Episode` (TensorClass). The
`EpisodeExport` dataclass that previously mirrored it for export has been removed.

### Remaining dual-path guards

One `torch.compiler.is_exporting()` guard survives intentionally:

| File | Guard | Effect |
|------|-------|--------|
| `components/episode.py` | `is_exporting()` in `_build_attention_mask` | Attention-mask cache: written on eager pass, read-only during export trace |

The attention-mask cache guard is **intentional and correct** regardless of
tensordict fixes: `AttentionMaskBuilder` is not trace-friendly, so the export
script always runs an eager forward first to warm the cache.

### What was unified

- `EpisodeBuilder.forward` — always returns `Episode` (TensorClass); removed
  `EpisodeExport` dataclass and `torch.export.register_dataclass` call.
- `PolicyObjective.forward` and `_compute_logits` — single `Episode` path;
  removed `isinstance(episode, EpisodeExport)` branches.
- `ControlTransformer.forward` — always returns `TensorDict`; removed
  `is_exporting()` guard on return type.
- `Scaler.forward` and `UniformBinner.forward` (`components/norm.py`) — always
  clamp; removed `is_exporting()` / `ValueError` split.

### TensorDict construction

`TensorDict.from_dict` is not Dynamo-safe: its internal `from_any` path calls
`dataclasses.is_dataclass → hasattr` which Dynamo cannot trace. All
`EpisodeBuilder.forward` TensorDict construction uses the direct constructor
`TensorDict(d, batch_size=[b, t])` instead.

### Export pipeline

```
model.eval()
model(*args)                          # warm attention-mask cache (eager)
torch.export.export(model, args)      # strict-mode trace
torch.onnx.export(..., dynamo=True)   # lower to ONNX via torch.onnx
onnxruntime.InferenceSession(...)     # serve
```

See `src/rmind/scripts/export_onnx.py` and `tests/test_export.py`.
