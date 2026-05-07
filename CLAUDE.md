# CLAUDE.md

Codebase notes for AI assistants working in this repo.

## Export architecture

The model has a single forward path used for both training and ONNX export.
Training calls `training_step` / `validation_step` directly (bypassing `forward`);
export calls `ControlTransformer.forward` via `torch.export.export`.

### Remaining dual-path guards

Three `torch.compiler.is_exporting()` guards and one `isinstance` split survive.
They exist because `tensordict` cannot construct a `TensorDict` / `TensorClass`
with a *dynamic* batch size inside a `torch.export` trace (upstream issue
[pytorch/tensordict#1003](https://github.com/pytorch/tensordict/issues/1003) —
`_parse_batch_size` chokes on a scalar `SymInt`).

| File | Lines | Guard | Effect |
|------|-------|-------|--------|
| `components/episode.py` | 249 | `is_exporting()` | `EpisodeBuilder.forward` returns `EpisodeExport` (dataclass) instead of `Episode` (TensorClass) |
| `components/episode.py` | 290, 300 | `is_exporting()` | Attention-mask cache: written on eager pass, read-only during export trace |
| `models/control_transformer.py` | 278 | `is_exporting()` | `ControlTransformer.forward` returns plain `dict` instead of `TensorDict` |
| `components/objectives/policy.py` | 60, 87 | `isinstance(episode, Episode)` | `PolicyObjective` has separate index-lookup logic for each episode type |

The attention-mask cache guard (lines 290/300) is **intentional and correct**
regardless of the tensordict fix: `AttentionMaskBuilder` is not trace-friendly,
so the export script always runs an eager forward first to warm the cache.

#### What happens when the guards are removed

Removing all guards except the attention-mask cache and running
`torch.export.export` produces two Dynamo errors, both rooted in
`TensorDict.from_dict` being called inside the traced graph:

**Error 1 — `hasattr` on `NoneType` (graph break)**

```
torch._dynamo.exc.Unsupported: Unsupported hasattr call
  Hint: Avoid calling hasattr(UserDefinedClassVariable, __dataclass_fields__)
from: episode.py → TensorDict.from_dict → from_any → _is_dataclass
      → dataclasses.is_dataclass → hasattr(cls, '__dataclass_fields__')
```

Dynamo cannot trace `hasattr` on a class whose runtime type is unknown.
This fires the moment `TensorDict.from_dict(input_embeddings, batch_dims=2)`
is called inside the export trace.

**Error 2 — `torch.*` op returning non-Tensor**

```
torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor
  example_value type: bool; op: call_function; target: <function is_available>
```

A device-availability check (`torch.cuda.is_available()` or similar) inside
tensordict's construction path returns a plain `bool`, which Dynamo cannot
represent as a graph node.

Both errors confirm that the entire `TensorDict.from_dict` / `from_any`
call chain is not Dynamo-safe. The guards are load-bearing until this is
fixed in tensordict.

### What was already unified

- `Scaler.forward` and `UniformBinner.forward` (`components/norm.py`) — now
  always clamp; the `is_exporting()` / `ValueError` split is gone.

### Unification plan

1. Fix `TensorDict.from_dict` / `from_any` Dynamo-safety in tensordict
   (PR to pytorch/tensordict — covers the `hasattr` graph break and the
   `is_available` non-Tensor return; supersedes the narrower `_parse_batch_size`
   + `SymInt` description of #1003).
   **Or** make `EpisodeBuilder.forward` always return `EpisodeExport` and
   update all training code to use its dict-based index API.
2. Remove `isinstance(episode, Episode)` branches from `PolicyObjective`.
3. Remove the `is_exporting()` guard in `ControlTransformer.forward`.
4. Keep the attention-mask cache guard in `EpisodeBuilder`.

### Export pipeline

```
model.eval()
model(*args)                          # warm attention-mask cache (eager)
torch.export.export(model, args)      # strict-mode trace
torch.onnx.export(..., dynamo=True)   # lower to ONNX via torch.onnx
onnxruntime.InferenceSession(...)     # serve
```

See `src/rmind/scripts/export_onnx.py` and `tests/test_export.py`.
