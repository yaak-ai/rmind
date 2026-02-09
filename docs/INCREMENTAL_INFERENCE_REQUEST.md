# Incremental Inference ONNX Export

## Status: ✅ IMPLEMENTED

The dual model export solution has been implemented. Use the following commands:

```bash
# Model 1: Full forward (6 timesteps, empty cache) - for first inference
just export-onnx-cache

# Model 2: Incremental (1 timestep, 5 cached) - for subsequent inferences
just export-onnx-incremental
```

### Exported Model Shapes

| Model | Command | Batch | Cache | Mask |
|-------|---------|-------|-------|------|
| `ControlTransformer_cache.onnx` | `export-onnx-cache` | `[1,6,...]` | `[1,0,384]` | `[1644,1644]` |
| `ControlTransformer_cache_incremental.onnx` | `export-onnx-incremental` | `[1,1,...]` | `[1,1370,384]` | `[274,1644]` |

### TensorRT Deployment

```bash
# Build TensorRT engines
trtexec --onnx=ControlTransformer_cache.onnx --saveEngine=full_forward.trt --fp16
trtexec --onnx=ControlTransformer_cache_incremental.onnx --saveEngine=incremental.trt --fp16
```

### Why Single Model with Dynamic Shapes Doesn't Work

We attempted multiple approaches to get a single ONNX model with dynamic shapes:

#### Approach 1: `dynamic_axes` with Dynamo Export

```bash
# This was implemented but does NOT produce dynamic shapes
just export-onnx-cache +dynamic_batch=true
```

**The flag adds the correct `dynamic_axes` specification:**

```python
dynamic_axes = {
    "batch_data_cam_front_left": {1: "timesteps"},
    "cached_projected_embeddings": {1: "cached_seq"},
    "cached_kv": {3: "cached_seq"},
    "mask": {0: "new_seq", 1: "total_seq"},
}
```

**But it doesn't work** because with dynamo export, shapes are baked in during `torch.export.export()`:

```python
# Step 1: torch.export.export() traces with concrete shapes
exported = torch.export.export(
    mod=cache_model,
    args=(batch, cache, kv, mask),  # Shapes BAKED IN here
    strict=True,
)

# Step 2: torch.onnx.export() converts to ONNX
torch.onnx.export(
    model=exported,      # ExportedProgram already has static shapes
    dynamic_axes={...},  # IGNORED - too late, shapes already fixed
)
```

#### Approach 2: `torch.export.Dim()` for Dynamo

We attempted using `torch.export.Dim()` to specify dynamic dimensions:

```python
from torch.export import Dim

s_cached = Dim("s_cached", min=0, max=4096)
timesteps = Dim("timesteps", min=1, max=32)

export_dynamic_shapes = {
    "batch": _build_batch_shapes(batch, timesteps),
    "cached_projected_embeddings": {1: s_cached},
    "cached_kv": {3: s_cached},
    "mask": {0: s_new, 1: s_total},
}

exported = torch.export.export(
    mod=cache_model,
    args=export_args,
    dynamic_shapes=export_dynamic_shapes,
    strict=False,  # Required for this model
)
```

**This fails with multiple errors:**

| Error | Cause |
|-------|-------|
| `ConstraintViolationError: s_cached specialized to constant (0/274)` | Model code has operations that specialize dimensions (integer division, slicing with concrete values) |
| `'dict' object has no attribute 'shape'` | Model code accesses `.shape` on dicts during FakeTensor tracing |

The model code contains patterns incompatible with `torch.export` dynamic shapes:
- `total_timesteps = total_seq_len // self._tokens_per_timestep` - integer division specializes dims
- `new_embeddings = all_embeddings[:, -new_seq_len:]` - negative indexing with concrete value
- `mit.one()` in `EpisodeBuilder` - fails during symbolic tracing

#### Approach 3: Legacy ONNX Export (`dynamo=False`)

```python
torch.onnx.export(
    model=cache_model,  # nn.Module directly
    args=export_args,
    dynamic_axes=dynamic_axes,
    ...
)
```

**This fails with multiple issues:**

1. **`mit.one()` issue (FIXED)**: The `EpisodeBuilder` used `mit.one()` which failed during tracing. Fixed by adding `_get_batch_info_from_tree()` helper.

2. **`EpisodeExport` not triggered (FIXED)**: Legacy export uses JIT tracing, not `torch.export`, so `torch.compiler.is_exporting()` returned False. Fixed by adding `torch.jit.is_tracing()` check in `_is_exporting()`.

3. **Unsupported operator (NOT FIXED)**:
```
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator
'aten::_native_multi_head_attention' to ONNX opset version 17 is not supported
```

The `nn.MultiheadAttention` uses a native fast path in eval mode that isn't supported in ONNX. Dynamo export handles this through its own code generation, but legacy export requires symbolic handlers that don't exist for this operator.

#### Summary of Failures

| Export Method | `dynamic_axes` Support | Why It Fails |
|--------------|------------------------|--------------|
| Dynamo + `dynamic_axes` | ❌ Ignored | Shapes baked in during `torch.export.export()` |
| Dynamo + `torch.export.Dim()` | ✓ Supported | Model code patterns incompatible with symbolic tracing |
| Legacy (`dynamo=False`) | ✓ Supported | `aten::_foreach_add` from TensorDict not supported in ONNX |

#### Potential Future Solutions

1. **Refactor model code** to avoid:
   - `mit.one()` → use explicit tensor indexing
   - Integer division on shapes → use symbolic-friendly operations
   - Negative indexing with concrete values → use `tensor.shape[1] - value`

2. **Wait for PyTorch improvements** to `torch.export`'s handling of complex model patterns

**For now, the dual model approach is the working solution.**

---

## Original Request

The current `export-onnx-cache` produces a model with **static batch input shapes**, which prevents true incremental inference on Jetson Orin.

**Solution: Dual Model Export**

Export a second ONNX model (`ControlTransformer_cache_incremental.onnx`) with:
- 1 timestep batch inputs (instead of 6)
- 5 cached timesteps cache inputs (instead of empty)
- Incremental mask [274, 1644] (instead of [1644, 1644])

This avoids dynamic shape complexity and allows optimal TensorRT optimization for each use case.

## Current State

### What Works
- `ControlTransformer_cache.onnx` exports successfully
- Full forward inference (6 timesteps) works with TensorRT
- Cache outputs (`projected_embeddings`, `kv_cache`) are produced correctly

### What Doesn't Work
Incremental inference (1 new timestep + 5 cached) fails because:

| Input | Current Shape | Needed for Incremental |
|-------|---------------|----------------------|
| `batch_data_cam_front_left` | `[1, 6, 3, 256, 256]` **static** | `[1, 1, 3, 256, 256]` |
| `batch_data_meta_*` | `[1, 6, 1]` **static** | `[1, 1, 1]` |
| `batch_data_waypoints_xy_normalized` | `[1, 6, 10, 2]` **static** | `[1, 1, 10, 2]` |
| `cached_projected_embeddings` | `[1, ?, 384]` dynamic ✓ | `[1, 1370, 384]` (5 ts) |
| `cached_kv` | `[8, 2, 1, ?, 384]` dynamic ✓ | `[8, 2, 1, 1370, 384]` |
| `mask` | `[1644, 1644]` **static** | `[274, 1644]` |

The cache inputs are already dynamic, but batch inputs and mask are static.

## Attempted Solutions

### TensorRT Optimization Profiles (Does NOT Work)

We attempted to use TensorRT optimization profiles as documented in the rmind ONNX_EXPORT.md, but this approach **fails** because TRT profiles can only define min/opt/max ranges for **dynamic** ONNX dimensions - they cannot override **static** dimensions.

```
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error
(Input tensor batch_data_cam_front_left has static dimensions that don't
match kMIN dimensions in profile index 1. Input dimensions are
[1,6,3,256,256] but profile dimensions are [1,1,3,256,256].)
```

**Key insight**: The ONNX model must be exported with dynamic axes for the timestep dimension. TensorRT optimization profiles only work when the ONNX model declares those dimensions as dynamic (using `-1` or named axes in `dynamic_axes`).

### Current ONNX Model Analysis

The exported `ControlTransformer_cache.onnx` has:
- **Static batch inputs**: `batch_data_cam_front_left` shape `[1, 6, 3, 256, 256]` (6 is hardcoded)
- **Static mask**: `mask` shape `[1644, 1644]` (full sequence hardcoded)
- **Dynamic cache inputs**: `cached_projected_embeddings` shape `[1, -1, 384]` (sequence is dynamic)
- **Dynamic cache inputs**: `cached_kv` shape `[8, 2, 1, -1, 384]` (sequence is dynamic)

The fix must happen at the ONNX export stage, not at TensorRT conversion.

## Performance Impact

### Current (Full Forward Only)
```
TensorRT (no cache):  31.5 ms/inference
TensorRT (cache):     59.9 ms/inference  (+28ms overhead, no benefit)
```

### With Incremental Support (Estimated)
```
Full forward (cold):   31.5 ms  (first inference)
Incremental (warm):    ~6.9 ms  (subsequent inferences)
Speedup:               4.5x
```

For real-time driving at 10 Hz:
- Current: 31.5ms leaves 68.5ms margin ✓
- Incremental: 6.9ms leaves 93.1ms margin ✓✓ (more headroom for other tasks)

## Requested Changes

### Option 1: Add Dynamic Export Flag (Preferred)

Add a flag to `export-onnx-cache` for dynamic batch dimensions:

```bash
just export-onnx-cache +dynamic_batch=true
```

Implementation in export script:

```python
# Current (static)
dynamic_axes = {
    "cached_projected_embeddings": {1: "cached_seq"},
    "cached_kv": {3: "cached_seq"},
}

# With +dynamic_batch=true
dynamic_axes = {
    # Batch inputs - dynamic timestep dimension
    "batch_data_cam_front_left": {1: "timesteps"},
    "batch_data_meta_vehiclemotion_speed": {1: "timesteps"},
    "batch_data_meta_vehiclemotion_gas_pedal_normalized": {1: "timesteps"},
    "batch_data_meta_vehiclemotion_brake_pedal_normalized": {1: "timesteps"},
    "batch_data_meta_vehiclemotion_steering_angle_normalized": {1: "timesteps"},
    "batch_data_meta_vehiclestate_turn_signal": {1: "timesteps"},
    "batch_data_waypoints_xy_normalized": {1: "timesteps"},
    # Cache inputs - already dynamic
    "cached_projected_embeddings": {1: "cached_seq"},
    "cached_kv": {3: "cached_seq"},
    # Mask - dynamic for incremental
    "mask": {0: "new_seq", 1: "total_seq"},
}

torch.onnx.export(
    model,
    dummy_inputs,
    output_path,
    dynamic_axes=dynamic_axes,
    ...
)
```

### Option 2: Dual Model Export (Recommended)

Based on rmind docs, export **two separate ONNX models** with static shapes:

```bash
# Model 1: Full forward (cold start)
just export-onnx-cache  # Already exists: ControlTransformer_cache.onnx

# Model 2: Incremental (warm inference)
just export-onnx-incremental  # NEW: ControlTransformer_cache_incremental.onnx
```

**Model 1 (`ControlTransformer_cache.onnx`)** - Already exists:
- Batch inputs: `[1, 6, ...]` (6 timesteps)
- Cache inputs: `[1, 0, 384]`, `[8, 2, 1, 0, 384]` (empty)
- Mask: `[1644, 1644]` (full causal)

**Model 2 (`ControlTransformer_cache_incremental.onnx`)** - Needed:
- Batch inputs: `[1, 1, ...]` (1 timestep)
- Cache inputs: `[1, 1370, 384]`, `[8, 2, 1, 1370, 384]` (5 cached timesteps)
- Mask: `[274, 1644]` (incremental causal)

This avoids dynamic shape issues entirely by using static shapes optimized for each use case.

## TensorRT Integration

### Option A: Single Model with Dynamic Shapes (Requires ONNX re-export)

If ONNX is exported with `dynamic_axes` for batch inputs, TensorRT optimization profiles can handle both cases:

```python
# Profile 0: Full forward
profile.set_shape("batch_data_cam_front_left",
    min=(1, 6, 3, 256, 256), opt=(1, 6, 3, 256, 256), max=(1, 6, 3, 256, 256))
profile.set_shape("cached_projected_embeddings",
    min=(1, 0, 384), opt=(1, 0, 384), max=(1, 0, 384))

# Profile 1: Incremental
profile.set_shape("batch_data_cam_front_left",
    min=(1, 1, 3, 256, 256), opt=(1, 1, 3, 256, 256), max=(1, 1, 3, 256, 256))
profile.set_shape("cached_projected_embeddings",
    min=(1, 274, 384), opt=(1, 1370, 384), max=(1, 1370, 384))
```

### Option B: Dual Model Approach (Recommended - No ONNX changes needed)

Build separate TensorRT engines from each ONNX model:

```python
# Build from ControlTransformer_cache.onnx (full forward)
trtexec --onnx=ControlTransformer_cache.onnx --saveEngine=full_forward.trt --fp16

# Build from ControlTransformer_cache_incremental.onnx (incremental)
trtexec --onnx=ControlTransformer_cache_incremental.onnx --saveEngine=incremental.trt --fp16
```

This approach:
- Avoids dynamic shape complexity
- Each engine is optimized for its specific use case
- Simpler inference code (no profile switching)
- Works with current static ONNX exports

## Inference Pattern

### Dual Model Inference Loop

```python
# Load both TensorRT engines
full_forward_engine = load_trt_engine("full_forward.trt")
incremental_engine = load_trt_engine("incremental.trt")

# First inference (cold start) - use full forward engine
batch = get_initial_batch(timesteps=6)
predictions, proj_emb, kv_cache = full_forward_engine.run(
    batch,
    empty_cache_proj=torch.zeros(1, 0, 384),
    empty_cache_kv=torch.zeros(8, 2, 1, 0, 384),
    mask=build_causal_mask(1644, 1644)
)

# Streaming loop - use incremental engine
while streaming:
    # Trim oldest timestep (sliding window)
    proj_emb = proj_emb[:, 274:]      # Remove oldest: [1, 1644, 384] -> [1, 1370, 384]
    kv_cache = kv_cache[:, :, :, 274:]  # Remove oldest: [8, 2, 1, 1644, 384] -> [8, 2, 1, 1370, 384]

    # Get new single timestep
    new_batch = get_single_timestep()

    # Incremental inference (~6.9ms instead of ~31.5ms)
    predictions, proj_emb, kv_cache = incremental_engine.run(
        new_batch,
        cached_proj=proj_emb,       # [1, 1370, 384]
        cached_kv=kv_cache,         # [8, 2, 1, 1370, 384]
        mask=build_causal_mask(274, 1644)  # [274, 1644]
    )

    # Use predictions for control
    apply_control(predictions)
```

## Test Cases

The dynamic export should pass these tests:

```python
def test_full_forward():
    """6 timesteps, empty cache -> works"""
    batch = create_batch(timesteps=6)
    empty_cache = create_empty_cache()
    mask = create_mask(new=1644, total=1644)
    outputs = model(batch, empty_cache, mask)
    assert outputs["predictions"].shape == (1, 1)
    assert outputs["proj_emb"].shape == (1, 1644, 384)

def test_incremental():
    """1 timestep, 5 cached -> works"""
    batch = create_batch(timesteps=1)
    cache = create_cache(seq_len=1370)  # 5 timesteps
    mask = create_mask(new=274, total=1644)
    outputs = model(batch, cache, mask)
    assert outputs["predictions"].shape == (1, 1)
    assert outputs["proj_emb"].shape == (1, 1644, 384)

def test_incremental_matches_full():
    """Incremental produces same predictions as full forward"""
    # Run full forward on 6 timesteps
    full_pred = model.full_forward(batch_6ts)

    # Run incremental: 5 cached + 1 new
    incr_pred = model.incremental(batch_1ts, cache_5ts)

    # Predictions should match (within tolerance)
    assert torch.allclose(full_pred, incr_pred, atol=1e-5)
```

## Files We Have Ready

Once dynamic export is available, we have inference code ready:

| File | Purpose |
|------|---------|
| `onnx_cache_inference.py` | ONNX Runtime inference with cache |
| `build_cache_trt_engine.py` | TensorRT engine builder with profiles |
| `run_inference_with_cache.py` | End-to-end streaming inference |
| `benchmark_incremental.py` | Performance benchmarking |

## Implementation Guide: Dynamic ONNX Export with Cache

### Model Wrapper with Cache Support

The model wrapper needs to handle variable-length inputs and cache concatenation:

```python
class CacheEnabledControlTransformer(nn.Module):
    """Wrapper that adds KV cache support for incremental inference."""

    def __init__(self, model: ControlTransformer):
        super().__init__()
        self.model = model
        self.embed_dim = 384
        self.num_layers = 8

    def forward(
        self,
        batch_data: dict[str, torch.Tensor],      # [B, T, ...] where T is dynamic
        cached_projected_embeddings: torch.Tensor, # [B, S_cached, D]
        cached_kv: torch.Tensor,                   # [L, 2, B, S_cached, D]
        mask: torch.Tensor,                        # [S_new, S_total]
    ):
        # 1. Process new timesteps through encoder (ViT, etc.)
        new_embeddings = self.model.encode_inputs(batch_data)  # [B, S_new, D]

        # 2. Concatenate with cached embeddings
        all_embeddings = torch.cat([cached_projected_embeddings, new_embeddings], dim=1)
        # Result: [B, S_cached + S_new, D] = [B, S_total, D]

        # 3. Add position embeddings to ALL tokens (cached + new)
        all_embeddings = self.model.add_position_embeddings(all_embeddings)

        # 4. Run transformer with KV cache
        # The transformer only computes attention for new tokens,
        # but attends to all tokens (cached + new)
        output, new_kv = self.model.transformer_with_cache(
            all_embeddings,
            cached_kv=cached_kv,
            mask=mask,
        )

        # 5. Get predictions from final token
        predictions = self.model.head(output[:, -1:, :])

        # 6. Return predictions + updated cache
        new_projected = torch.cat([cached_projected_embeddings, new_embeddings], dim=1)

        return predictions, new_projected, new_kv
```

### Transformer Layer with KV Cache

```python
class TransformerLayerWithCache(nn.Module):
    def forward(self, x, cached_kv=None, mask=None):
        B, S_new, D = x.shape

        # Compute Q, K, V for new tokens only
        q = self.q_proj(x)  # [B, S_new, D]
        k = self.k_proj(x)  # [B, S_new, D]
        v = self.v_proj(x)  # [B, S_new, D]

        # Concatenate with cached K, V
        if cached_kv is not None:
            cached_k, cached_v = cached_kv[0], cached_kv[1]  # Each [B, S_cached, D]
            k = torch.cat([cached_k, k], dim=1)  # [B, S_total, D]
            v = torch.cat([cached_v, v], dim=1)  # [B, S_total, D]

        # Attention: Q attends to all K, V
        # Q: [B, S_new, D], K: [B, S_total, D], V: [B, S_total, D]
        attn_output = self.attention(q, k, v, mask=mask)  # [B, S_new, D]

        # Return output and updated KV cache
        new_kv = torch.stack([k, v], dim=0)  # [2, B, S_total, D]
        return attn_output, new_kv
```

### ONNX Export with Dynamic Axes

The key is specifying `dynamic_axes` for ALL variable dimensions:

```python
def export_onnx_with_dynamic_cache(
    model: CacheEnabledControlTransformer,
    output_path: str,
    num_timesteps: int = 6,
):
    model.eval()

    # Create dummy inputs (shapes used for tracing, not runtime)
    dummy_batch = {
        "batch_data_cam_front_left": torch.rand(1, num_timesteps, 3, 256, 256),
        "batch_data_meta_vehiclemotion_speed": torch.rand(1, num_timesteps, 1),
        # ... other batch inputs
    }
    dummy_cache_proj = torch.rand(1, 0, 384)  # Empty cache for export
    dummy_cache_kv = torch.rand(8, 2, 1, 0, 384)
    dummy_mask = torch.ones(1644, 1644, dtype=torch.bool)

    # Define dynamic axes - THIS IS THE KEY
    dynamic_axes = {
        # Batch inputs: dimension 1 (timesteps) is dynamic
        "batch_data_cam_front_left": {1: "num_timesteps"},
        "batch_data_meta_vehiclemotion_speed": {1: "num_timesteps"},
        "batch_data_meta_vehiclemotion_gas_pedal_normalized": {1: "num_timesteps"},
        "batch_data_meta_vehiclemotion_brake_pedal_normalized": {1: "num_timesteps"},
        "batch_data_meta_vehiclemotion_steering_angle_normalized": {1: "num_timesteps"},
        "batch_data_meta_vehiclestate_turn_signal": {1: "num_timesteps"},
        "batch_data_waypoints_xy_normalized": {1: "num_timesteps"},

        # Cache inputs: sequence dimension is dynamic
        "cached_projected_embeddings": {1: "cached_seq"},
        "cached_kv": {3: "cached_seq"},

        # Mask: both dimensions are dynamic
        "mask": {0: "new_seq", 1: "total_seq"},

        # Outputs: sequence dimensions are dynamic
        "projected_embeddings": {1: "total_seq"},
        "kv_cache": {3: "total_seq"},
    }

    torch.onnx.export(
        model,
        (dummy_batch, dummy_cache_proj, dummy_cache_kv, dummy_mask),
        output_path,
        input_names=[
            "batch_data_cam_front_left",
            "batch_data_meta_vehiclemotion_speed",
            # ... other inputs
            "cached_projected_embeddings",
            "cached_kv",
            "mask",
        ],
        output_names=[
            "predictions",
            "projected_embeddings",
            "kv_cache",
        ],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
```

### Verify Dynamic Shapes in Exported ONNX

```python
import onnx

model = onnx.load("model_dynamic.onnx")

print("Input shapes:")
for inp in model.graph.input:
    shape = [d.dim_param if d.dim_param else d.dim_value
             for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {shape}")

# Expected output (dimensions with names are dynamic):
# batch_data_cam_front_left: [1, 'num_timesteps', 3, 256, 256]
# cached_projected_embeddings: [1, 'cached_seq', 384]
# cached_kv: [8, 2, 1, 'cached_seq', 384]
# mask: ['new_seq', 'total_seq']
```

### Current vs Required Export

| Aspect | Current Export | Required for Incremental |
|--------|----------------|--------------------------|
| `dynamic_axes` | Only cache dims | All variable dims |
| Batch inputs | `[1, 6, ...]` fixed | `[1, T, ...]` variable |
| Mask | `[1644, 1644]` fixed | `[S_new, S_total]` variable |
| TRT profiles | Cannot override static | Can define min/opt/max |

### Common Issues to Avoid

1. **PyTorch dynamo limitations** - Complex models may not trace cleanly with dynamic shapes. Use `torch.onnx.export` with `dynamo=False` if needed.

2. **Control flow** - Avoid Python `if` statements on tensor shapes. Use `torch.where` or ensure shapes are handled at graph level.

3. **Empty tensors** - Test with `cached_seq=0` (empty cache) during export to ensure the model handles concatenation with empty tensors correctly.

## Questions

1. Is dynamic batch export feasible with the current `CacheEnabledControlTransformer` wrapper?
2. Are there any concerns about numerical accuracy with incremental inference?
3. Should we target a specific ONNX opset version for best TensorRT compatibility?

## Contact

For questions about the Jetson Orin integration, reach out to the drivr team.

---

**Priority**: High - This enables 4.5x inference speedup for real-time control on edge devices.
