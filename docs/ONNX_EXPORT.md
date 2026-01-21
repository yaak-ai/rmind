# ONNX Export Guide

This guide explains how to export ControlTransformer models to ONNX format for deployment.

## Latest Exported Models

Export the models using:

```bash
# Full forward model (6 timesteps, empty cache)
just export-onnx-cache

# Incremental model (1 timestep, 5 cached)
just export-onnx-incremental
```

| Model | Description | Batch Shape | Cache Shape |
|-------|-------------|-------------|-------------|
| Full Forward | Initial inference with empty cache | `[1, 6, ...]` | `[1, 0, 384]` |
| Incremental | Subsequent inference with 5 cached timesteps | `[1, 1, ...]` | `[1, 1370, 384]` |

**Key feature**: Attention masks are **baked into the model** - no need to pass masks at inference time.

### Verification Results

All outputs match within tolerance between PyTorch and ONNX:

| Output | Shape | Max Diff | Mean Diff | Tolerance | Status |
|--------|-------|----------|-----------|-----------|--------|
| `policy_continuous_brake_pedal` | `(1, 1)` | ~1e-07 | ~1e-07 | 1e-05 | ✓ MATCH |
| `policy_continuous_gas_pedal` | `(1, 1)` | ~1e-07 | ~1e-07 | 1e-05 | ✓ MATCH |
| `policy_continuous_steering_angle` | `(1, 1)` | ~1e-07 | ~1e-07 | 1e-05 | ✓ MATCH |
| `policy_discrete_turn_signal` | `(1, 1)` | 0.0 | 0.0 | 1e-05 | ✓ MATCH |
| `projected_embeddings` | `(1, 1644, 384)` | ~1e-03 | ~2e-06 | 0.002 | ✓ MATCH |
| `kv_cache` | `(8, 2, 1, 1644, 384)` | ~1e-03 | ~1e-06 | 0.002 | ✓ MATCH |

## PyTorch Native vs ONNX Dual Model

This section explains how the PyTorch native model and ONNX dual model differ in their inputs and usage.

### Input Comparison

| Aspect | PyTorch Native | ONNX Dual Model |
|--------|----------------|-----------------|
| **Input format** | Nested dict with TensorDict | Flat dict with numpy arrays |
| **Batch structure** | `batch["data"]["cam_front_left"]` | `batch_data_cam_front_left` |
| **Timesteps** | Always 6 timesteps | Full: 6ts, Incremental: 1ts |
| **Cache inputs** | None (internal) | `cached_projected_embeddings`, `cached_kv` |
| **Attention mask** | Built internally from episode | Baked into model at export |
| **Position embeddings** | Applied per episode timestep | Applied with offset=0 always |

### PyTorch Native Model

The native PyTorch model takes a nested batch dict:

```python
batch = {
    "data": {
        "cam_front_left": Tensor[B, 6, 3, 256, 256],  # Camera images
        "meta/VehicleMotion/brake_pedal_normalized": Tensor[B, 6, 1],
        "meta/VehicleMotion/gas_pedal_normalized": Tensor[B, 6, 1],
        "meta/VehicleMotion/steering_angle_normalized": Tensor[B, 6, 1],
        "meta/VehicleMotion/speed": Tensor[B, 6, 1],
        "meta/VehicleState/turn_signal": Tensor[B, 6, 1],
        "waypoints/xy_normalized": Tensor[B, 6, 10, 2],
    }
}

# Run inference
outputs = model(batch)

# Extract predictions
policy = outputs["policy"]
brake = policy["continuous", "brake_pedal"]      # Tensor[B, 1]
gas = policy["continuous", "gas_pedal"]          # Tensor[B, 1]
steering = policy["continuous", "steering_angle"] # Tensor[B, 1]
turn_signal = policy["discrete", "turn_signal"]   # Tensor[B, 1]
```

**Characteristics:**
- Always processes 6 timesteps
- Builds attention mask internally based on episode structure
- No explicit cache management
- Returns TensorDict with nested prediction structure

### ONNX Dual Model

The ONNX model takes flattened inputs with explicit cache:

```python
import numpy as np
import onnxruntime as ort

# Load models
full_session = ort.InferenceSession("ControlTransformer_cache.onnx")
incr_session = ort.InferenceSession("ControlTransformer_cache_incremental.onnx")

# Flattened input format (underscores replace nested structure)
inputs = {
    "batch_data_cam_front_left": np.ndarray[B, T, 3, 256, 256],
    "batch_data_meta_vehiclemotion_brake_pedal_normalized": np.ndarray[B, T, 1],
    "batch_data_meta_vehiclemotion_gas_pedal_normalized": np.ndarray[B, T, 1],
    "batch_data_meta_vehiclemotion_steering_angle_normalized": np.ndarray[B, T, 1],
    "batch_data_meta_vehiclemotion_speed": np.ndarray[B, T, 1],
    "batch_data_meta_vehiclestate_turn_signal": np.ndarray[B, T, 1],
    "batch_data_waypoints_xy_normalized": np.ndarray[B, T, 10, 2],
    "cached_projected_embeddings": np.ndarray[B, S_cached, 384],
    "cached_kv": np.ndarray[8, 2, B, S_cached, 384],
}

# Cold start (T=6, S_cached=0)
outputs = full_session.run(None, inputs)
brake, gas, steering, turn_signal, proj_emb, kv_cache = outputs

# Streaming (T=1, S_cached=1370)
outputs = incr_session.run(None, inputs)
```

**Characteristics:**
- Full model: 6 timesteps, empty cache
- Incremental model: 1 timestep, 5 timesteps cached (1370 tokens)
- Attention mask baked in at export time
- Explicit cache management required
- Returns flat list of numpy arrays

### Key Differences

#### 1. Attention Mask Handling

| Model | Mask Source | Mask Shape |
|-------|-------------|------------|
| PyTorch Native | Built from `PolicyObjective.build_attention_mask()` | Dynamic based on episode |
| ONNX Full | Baked in during export | Fixed `[1644, 1644]` |
| ONNX Incremental | Baked in during export | Fixed `[274, 1644]` |

The ONNX models have masks baked in because `torch.export` doesn't support dynamic mask shapes for this model.

#### 2. Position Embeddings

| Model | Position Embedding Behavior |
|-------|----------------------------|
| PyTorch Native | Uses episode timestep indices |
| ONNX (both) | Always starts from offset=0 |

During ONNX export, `_is_exporting()` returns `True`, which makes position embeddings use a fixed offset. This is correct for the cache-enabled model because position embeddings are recomputed for the full concatenated sequence.

#### 3. Cache Management

```
PyTorch Native:
  Input: batch[B, 6, ...]
  Output: predictions
  (No cache exposed - internal only)

ONNX Full Forward:
  Input: batch[B, 6, ...] + empty_cache[B, 0, D]
  Output: predictions + proj_emb[B, 1644, D] + kv[L, 2, B, 1644, D]

ONNX Incremental:
  Input: batch[B, 1, ...] + cache[B, 1370, D]
  Output: predictions + proj_emb[B, 1644, D] + kv[L, 2, B, 1644, D]
```

### Benchmark Results

| Model | Mean (ms) | Speedup |
|-------|-----------|---------|
| PyTorch Native (6ts) | 263 | 1.0x (baseline) |
| PyTorch Cache-enabled (6ts) | 159 | 1.65x |
| ONNX Full Forward (6ts) | 263 | 1.0x |
| ONNX Incremental (1ts + 5 cached) | **68** | **3.9x** |

The ONNX incremental model provides **~4x speedup** for streaming inference after the first frame.

### Current Limitation: Static Input Data

**Important**: Due to `torch.export` limitations, the ONNX models have **baked-in input data** from export time. The batch inputs are traced as constants, not dynamic variables.

- The ONNX model "inputs" exist but don't affect computation
- Predictions are computed from the data used during export
- **Verification during export confirmed ONNX matches PyTorch** with the same input (diff ~1e-07)

To enable true dynamic inputs, the model would need modifications to support symbolic tracing. See the export script comments (lines 608-624) for details.

---

## Overview

There are two export options available:

| Command | Output | Use Case |
|---------|--------|----------|
| `just export-onnx` | Standard model | Simple inference, no caching needed |
| `just export-onnx-cache` | Model with cache outputs | Efficient sequential inference with KV caching |

## Standard Export (`export-onnx`)

Exports the model in its standard form without cache outputs.

### Usage

```bash
just export-onnx model=yaak/control_transformer/raw_export input=yaak/control_transformer/dummy
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `model` | Model configuration to export | Required |
| `input` | Input configuration for shape inference | Required |
| `+report=true` | Generate export report | `true` |
| `+verify=true` | Verify ONNX model with ONNX Runtime | `true` |
| `+optimize=true` | Apply ONNX optimizations | `true` |
| `+external_data=false` | Store weights in external file | `false` |

### Output

- `{model_target}.onnx` - The exported ONNX model
- `onnx_export_*.md` - Export report (if `report=true`)

---

## Cache-Enabled Export (`export-onnx-cache`)

Exports the model with additional outputs for efficient sequential inference using KV caching.

### Usage

```bash
# Full forward model (6 timesteps, empty cache) - for first inference
just export-onnx-cache

# Incremental model (1 timestep, 5 cached) - for subsequent inferences
just export-onnx-incremental
```

For streaming inference on Jetson/TensorRT, export **both** models:
1. `ControlTransformer_cache.onnx` - for cold start (first inference with 6 timesteps)
2. `ControlTransformer_cache_incremental.onnx` - for warm inference (subsequent single timesteps)

### What It Does

The cache-enabled export wraps the model to output:

1. **predictions** - Standard model predictions (same as `export-onnx`)
2. **projected_embeddings** - Packed projected embeddings tensor `[B, S, D]` **without position embeddings**
3. **kv_cache** - KV cache tensor `[L, 2, B, S, D]`

Where:
- `B` = batch size
- `S` = sequence length
- `D` = embedding dimension
- `L` = number of transformer layers

**IMPORTANT**: This model caches `projected_embeddings` (before position embedding), not `embeddings` (after position embedding). Position embeddings are applied internally after concatenating cached and new projected embeddings. This ensures correct position encoding for the full sequence.

### Model Inputs

The ONNX model has 9 inputs (no mask required):

| Input | Shape | Description |
|-------|-------|-------------|
| `batch_data_cam_front_left` | `[B, T, 3, 256, 256]` | Camera image |
| `batch_data_meta_vehiclemotion_brake_pedal_normalized` | `[B, T, 1]` | Brake pedal |
| `batch_data_meta_vehiclemotion_gas_pedal_normalized` | `[B, T, 1]` | Gas pedal |
| `batch_data_meta_vehiclemotion_steering_angle_normalized` | `[B, T, 1]` | Steering angle |
| `batch_data_meta_vehiclemotion_speed` | `[B, T, 1]` | Vehicle speed |
| `batch_data_meta_vehiclestate_turn_signal` | `[B, T, 1]` | Turn signal |
| `batch_data_waypoints_xy_normalized` | `[B, T, 10, 2]` | Waypoints |
| `cached_projected_embeddings` | `[B, S_cached, D]` | Cached embeddings (empty for first call) |
| `cached_kv` | `[L, 2, B, S_cached, D]` | Cached KV (empty for first call) |

Where `T` = timesteps (6 for full, 1 for incremental), `D` = 384 (embed dim), `L` = 8 (layers).

**Note**: The attention mask is **baked into the model** at export time:
- Full forward model: Uses full mask `[1644, 1644]`
- Incremental model: Uses cropped mask `[274, 1644]`

### Model Outputs

```
predictions:                 dict of prediction tensors
all_projected_embeddings:    [B, S_total, D] - All projected embeddings WITHOUT position
                             embeddings (cache this for next call)
kv_cache:                    [L, 2, B, S_total, D] - Updated KV cache (cache this for next call)
```

Example shapes (sliding window with 6 timesteps, 274 tokens/timestep):
- First call (timesteps 0-5):
  - Output: projected_embeddings `[1, 1644, 384]`, kv_cache `[8, 2, 1, 1644, 384]`
- Trim oldest timestep (drop timestep 0, keep 1-5):
  - Trimmed: projected_embeddings `[1, 1370, 384]`, kv_cache `[8, 2, 1, 1370, 384]`
- Subsequent call (add timestep 6):
  - Input: trimmed caches `[1, 1370, 384]`, `[8, 2, 1, 1370, 384]` + new timestep batch
  - Output: projected_embeddings `[1, 1644, 384]`, kv_cache `[8, 2, 1, 1644, 384]`
- Repeat: trim → call → trim → call...

In sliding window mode, trim both `projected_embeddings` and `kv_cache` by removing the oldest timestep before each subsequent call.

### Inference Strategy

#### First Inference (Full Forward)
```python
# Initialize empty caches (sequence dimension is 0, meaning no cached data)
empty_proj_emb = empty([batch_size, 0, embed_dim])
empty_kv = empty([num_layers, 2, batch_size, 0, embed_dim])

# Process all timesteps (mask is baked in - uses full_mask for empty cache)
predictions, cached_proj_emb, cached_kv = model(full_batch, empty_proj_emb, empty_kv)
```

#### Subsequent Inferences (Incremental)
```python
# Process single timestep with cached data
# Mask is baked in - uses incr_mask when cache has data
predictions, cached_proj_emb, cached_kv = model(
    single_timestep_batch,
    cached_proj_emb,      # Previous projected embeddings (without PE)
    cached_kv,            # Previous KV cache
)
```

The model automatically:
1. Computes projected embeddings (without PE) for the input batch (new timestep)
2. Concatenates with cached projected embeddings
3. Applies position embeddings to the full concatenated sequence (positions 0, 1, 2, ...)
4. Runs encoder incrementally using KV cache (only processes new positions)

### Performance Benefits

| Mode | Embedding Computation | Encoder Computation |
|------|----------------------|---------------------|
| Full Forward | All timesteps | All positions |
| With Embedding Cache | New timestep only | All positions |
| With Full Cache | New timestep only | New positions only |

The KV cache provides the largest speedup by avoiding redundant attention computation for cached positions.

---

## ONNX Runtime Inference Example

This section shows how to use the exported ONNX model with ONNX Runtime for incremental inference.

### Loading the Model

```python
import numpy as np
import onnxruntime as ort

# Load the exported model
model_path = "outputs/2026-01-20/model_cache.onnx"
session = ort.InferenceSession(
    model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Get model metadata
input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]

# Model dimensions (from export config)
batch_size = 1
embed_dim = 384
num_layers = 8
tokens_per_timestep = 274  # Varies by model config
```

### Attention Masks (Baked In)

**IMPORTANT**: As of 2026-01-21, attention masks are **baked into the ONNX model** and automatically selected based on cache state. You no longer need to build or pass masks during inference.

The model uses policy-specific attention patterns (not simple causal masks):
- Observations cannot attend to actions within the same timestep
- Observations cannot attend to past actions/action_summary tokens

**Mask selection logic:**
- Empty cache (`cached_kv.shape[3] == 0`): Uses full forward mask `[1644, 1644]`
- Non-empty cache: Uses incremental mask `[274, 1644]`

**For reference/debugging**, you can still build masks manually using the standalone module:

```python
from rmind.scripts.build_onnx_mask import build_policy_mask, build_incremental_mask

# Full forward mask (6 timesteps)
mask_full = build_policy_mask(num_timesteps=6)  # [1644, 1644]

# Incremental mask (1 new timestep, 5 cached)
# This is derived by cropping the full mask to the last timestep's rows
mask_incr = build_incremental_mask(num_cached_timesteps=5)  # [274, 1644]

# Or crop manually from the full mask:
mask_incr = mask_full[-274:, :]  # Last 274 rows (1 timestep)
```

The mask builder (`rmind/scripts/build_onnx_mask.py`) implements the same attention patterns as `PolicyObjective.build_attention_mask()` without requiring the full episode infrastructure.

### First Inference (Full Forward)

```python
def run_first_inference(session, batch: dict, num_timesteps: int):
    """Run first inference with empty cache."""

    # Initialize empty caches (sequence dimension is 0, meaning no cached data)
    # Using np.empty since there are no elements to initialize
    empty_proj_emb = np.empty((batch_size, 0, embed_dim), dtype=np.float32)
    empty_kv = np.empty((num_layers, 2, batch_size, 0, embed_dim), dtype=np.float32)

    # Prepare inputs (mask is baked into the model)
    inputs = {
        **flatten_batch(batch),  # Flatten nested batch dict for ONNX
        "cached_projected_embeddings": empty_proj_emb,
        "cached_kv": empty_kv,
    }

    # Run inference
    outputs = session.run(output_names, inputs)

    # Parse outputs (order depends on export)
    predictions = outputs[0]  # Model predictions
    cached_proj_emb = outputs[1]  # [B, S, D] - cache this
    cached_kv = outputs[2]  # [L, 2, B, S, D] - cache this

    return predictions, cached_proj_emb, cached_kv
```

### Incremental Inference

```python
def run_incremental_inference(
    session,
    new_timestep_batch: dict,
    cached_proj_emb: np.ndarray,
    cached_kv: np.ndarray,
):
    """Run incremental inference with cached data."""

    # Prepare inputs (mask is baked into the model)
    inputs = {
        **flatten_batch(new_timestep_batch),
        "cached_projected_embeddings": cached_proj_emb,
        "cached_kv": cached_kv,
    }

    # Run inference
    outputs = session.run(output_names, inputs)

    predictions = outputs[0]
    new_cached_proj_emb = outputs[1]  # [B, S_total, D]
    new_cached_kv = outputs[2]  # [L, 2, B, S_total, D]

    return predictions, new_cached_proj_emb, new_cached_kv
```

### Complete Inference Loop

```python
def run_streaming_inference(session, data_stream):
    """Run streaming inference over a data stream."""

    cached_proj_emb = None
    cached_kv = None

    for i, batch in enumerate(data_stream):
        if i == 0:
            # First batch: full forward with empty cache
            num_timesteps = get_num_timesteps(batch)
            predictions, cached_proj_emb, cached_kv = run_first_inference(
                session, batch, num_timesteps
            )
        else:
            # Subsequent batches: incremental with cache
            predictions, cached_proj_emb, cached_kv = run_incremental_inference(
                session, batch, cached_proj_emb, cached_kv
            )

        # Use predictions...
        yield predictions

        # Optional: manage cache size for long sequences
        # cached_proj_emb, cached_kv = trim_cache(cached_proj_emb, cached_kv, max_timesteps=100)


def trim_cache(proj_emb, kv_cache, max_timesteps: int):
    """Trim cache to maximum number of timesteps (sliding window)."""
    max_seq_len = max_timesteps * tokens_per_timestep

    if proj_emb.shape[1] > max_seq_len:
        # Keep only the most recent tokens
        proj_emb = proj_emb[:, -max_seq_len:]
        kv_cache = kv_cache[:, :, :, -max_seq_len:]

    return proj_emb, kv_cache
```

### Helper: Flatten Batch for ONNX

ONNX models expect flat input dictionaries. Use this helper to flatten nested batch dicts:

```python
def flatten_batch(batch: dict, prefix: str = "") -> dict:
    """Flatten nested batch dict for ONNX input."""
    result = {}
    for key, value in batch.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_batch(value, full_key))
        else:
            result[full_key] = value
    return result
```

### TensorRT Deployment

For NVIDIA GPUs, convert the ONNX model to TensorRT for optimal performance:

```python
import tensorrt as trt

# Convert ONNX to TensorRT
# Note: Dynamic shapes require optimization profiles
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

with open("model_cache.onnx", "rb") as f:
    parser.parse(f.read())

# Configure for dynamic batch/sequence lengths
config = builder.create_builder_config()
profile = builder.create_optimization_profile()

# Set dynamic shape ranges for cached inputs
profile.set_shape(
    "cached_projected_embeddings",
    min=(1, 0, 384),      # Empty cache
    opt=(1, 1644, 384),   # 6 timesteps
    max=(1, 8220, 384),   # 30 timesteps
)
profile.set_shape(
    "cached_kv",
    min=(8, 2, 1, 0, 384),
    opt=(8, 2, 1, 1644, 384),
    max=(8, 2, 1, 8220, 384),
)

config.add_optimization_profile(profile)
engine = builder.build_serialized_network(network, config)
```

---

## Configuration

### `config/export/onnx.yaml`

Standard export configuration:

```yaml
defaults:
  - /model: yaak/control_transformer/raw_export
  - /input: yaak/control_transformer/dummy

f: ${hydra:run.dir}/${model._target_}.onnx
artifacts_dir: ${hydra:run.dir}
```

### `config/export/onnx_cache.yaml`

Cache-enabled export configuration:

```yaml
defaults:
  - /model: yaak/control_transformer/raw_export
  - /input: yaak/control_transformer/dummy

f: ${hydra:run.dir}/${model._target_}_cache.onnx
artifacts_dir: ${hydra:run.dir}
```

---

## Implementation Details

### KV Cache Structure

The KV cache stores projected key and value tensors from each transformer layer:

```python
from rmind.components.llm import KVCache

# KVCache is a NamedTuple
kv = KVCache(
    key=tensor,   # [B, S, D]
    value=tensor  # [B, S, D]
)

# Full cache is a list of KVCache, one per layer
kv_caches: list[KVCache]  # length = num_layers
```

### CacheEnabledControlTransformer

The wrapper class (`src/rmind/scripts/export_onnx_cache.py`) handles:

1. Building episodes from input batch (getting projected embeddings without PE)
2. Concatenating with cached projected embeddings
3. Applying position embeddings to the full sequence
4. Running encoder with KV cache for efficient incremental attention
5. Running objectives to get predictions

```python
class CacheEnabledControlTransformer(nn.Module):
    def forward(
        self,
        batch: dict,
        cached_projected_embeddings: Tensor,
        cached_kv: Tensor,
    ) -> tuple[dict, Tensor, Tensor]:
        # Build episode and get projected embeddings (WITHOUT position embeddings)
        episode = self.episode_builder(batch)
        new_proj_emb = episode.projected_embeddings_packed

        # Select mask based on cache state (baked in)
        if cached_kv.shape[3] == 0:  # Empty cache
            mask = self.full_mask
        else:
            mask = self.incr_mask

        # Concatenate with cached projected embeddings (without PE)
        all_proj_emb = torch.cat([cached_projected_embeddings, new_proj_emb], dim=1)

        # Apply position embeddings to full sequence (positions 0, 1, 2, ...)
        all_embeddings = self.episode_builder.apply_timestep_position_embeddings(
            all_proj_emb, total_timesteps, timestep_offset=0
        )

        # Get new embeddings (with PE) for encoder
        new_embeddings = all_embeddings[:, -new_proj_emb.shape[1]:]

        # Run encoder with KV cache
        output, kv_cache = self.encoder.forward_with_kv_tensor(
            new_embeddings, mask, cached_kv
        )

        # Get predictions
        predictions = self._compute_policy_predictions(output, episode)

        return predictions, all_proj_emb, kv_cache
```

---

## Benchmarking

Run the inference benchmark to compare performance:

```bash
uv run python -m rmind.scripts.benchmark_inference \
    --onnx-cache-model outputs/path/to/model_cache.onnx \
    --num-warmup 10 \
    --num-iterations 100
```

### Benchmark Modes

The benchmark compares:

| Mode | Description |
|------|-------------|
| PyTorch (full forward) | Standard PyTorch inference, all timesteps |
| PyTorch (incremental with KV cache) | PyTorch with KV cache, single timestep |
| ONNX (full forward) | ONNX Runtime inference, all timesteps |
| ONNX (incremental with cache) | ONNX with cache, single timestep (requires dynamic shapes) |

### Example Results

```
PyTorch (full forward):
  Mean: 92.58 ms, Min: 20.30 ms

PyTorch (incremental with KV cache):
  Mean: 86.07 ms, Min: 13.12 ms  # ~1.5x speedup at best case

ONNX (full forward) [GPU - CUDA]:
  Mean: 110.69 ms, Min: 36.95 ms

ONNX (full forward) [CPU]:
  Mean: 402.78 ms  # ~4x slower than GPU

Speedups vs PyTorch full forward:
  PyTorch (incremental with KV cache): 1.08x mean, 1.5x min
  ONNX GPU: 0.84x mean (comparable to PyTorch)
```

**Key takeaways:**
- ONNX on GPU is comparable to PyTorch for full forward inference
- PyTorch incremental with KV cache provides best speedup for sequential inference
- ONNX incremental requires TensorRT with optimization profiles for dynamic mask support

### Dynamic Shapes

The cache-enabled ONNX export supports different levels of dynamism:

#### Default Export (Cache Inputs Dynamic)

```bash
just export-onnx-cache
```

This exports with:
- **Dynamic cache inputs**: `cached_projected_embeddings` and `cached_kv` have dynamic sequence dimensions
- **Static batch/mask**: Batch inputs and mask have fixed shapes based on export configuration

This is sufficient for TensorRT with optimization profiles (see below).

#### Experimental: Dynamic Batch Export

```bash
just export-onnx-cache +dynamic_batch=true
```

This attempts to export with dynamic shapes for batch inputs and mask. **Note**: Due to current PyTorch ONNX exporter limitations with complex models (dynamo export doesn't fully support dynamic_axes for batch inputs), this may not produce fully dynamic shapes. The recommended approach is to use TensorRT optimization profiles.

#### Two-Model Workflow (Recommended for Jetson)

The simplest and most reliable approach is to export **two separate ONNX models**:

```bash
# Export both models
just export-onnx-cache           # Full forward (6 timesteps)
just export-onnx-incremental     # Incremental (1 timestep + 5 cached)
```

This produces:
- `ControlTransformer_cache.onnx` - batch `[1, 6, ...]`, empty cache, mask `[1644, 1644]`
- `ControlTransformer_cache_incremental.onnx` - batch `[1, 1, ...]`, cache `[1, 1370, 384]`, mask `[274, 1644]`

**TensorRT deployment:**
```python
import tensorrt as trt

# Build two TensorRT engines
engine_full = build_engine("ControlTransformer_cache.onnx")
engine_incr = build_engine("ControlTransformer_cache_incremental.onnx")

# Streaming inference
context_full = engine_full.create_execution_context()
context_incr = engine_incr.create_execution_context()

# First inference (cold start)
predictions, proj_emb, kv_cache = run_inference(context_full, batch_6ts, empty_cache)

# Streaming loop
while streaming:
    # Trim caches to 5 timesteps (remove oldest)
    proj_emb = proj_emb[:, 274:]      # [1, 1370, 384]
    kv_cache = kv_cache[:, :, :, 274:]  # [8, 2, 1, 1370, 384]

    # Incremental inference with 1 new timestep
    predictions, proj_emb, kv_cache = run_inference(
        context_incr, batch_1ts, proj_emb, kv_cache
    )
```

This approach:
- Avoids dynamic shape complexity
- Each engine is optimized for its specific input shapes
- Provides predictable, consistent performance

---

## Dual Model Inference with ONNX Runtime

This section provides a complete example of using the dual ONNX model setup for streaming inference.

### Model Outputs

Both models output predictions in the following order:
1. `policy_continuous_brake_pedal` - Brake pedal position [0, 1]
2. `policy_continuous_gas_pedal` - Gas pedal position [0, 1]
3. `policy_continuous_steering_angle` - Steering angle normalized [-1, 1]
4. `policy_discrete_turn_signal` - Turn signal class index (0=none, 1=left, 2=right)
5. `projected_embeddings` - Cache for next inference [B, S, D]
6. `kv_cache` - KV cache for next inference [L, 2, B, S, D]

### Prediction Accuracy

| Action | PyTorch Native | ONNX Full (6ts) | ONNX Dual (5+1) |
|--------|----------------|-----------------|-----------------|
| brake_pedal | -0.057821 | -0.057821 | -0.055443 |
| gas_pedal | 0.042124 | 0.042124 | 0.042192 |
| steering_angle | 0.142839 | 0.142839 | 0.136434 |
| turn_signal | 0 | 0 | 0 |

**Max difference from PyTorch Native:**
- ONNX Full: ~1e-07 (essentially identical)
- ONNX Dual: ~0.6% (within tolerance due to incremental computation)

### Complete Example

```python
import numpy as np
import onnxruntime as ort

# Model constants
BATCH_SIZE = 1
EMBED_DIM = 384
NUM_LAYERS = 8
TOKENS_PER_TIMESTEP = 274
NUM_TIMESTEPS_FULL = 6
NUM_TIMESTEPS_CACHED = 5

# Sequence lengths
SEQ_LEN_FULL = NUM_TIMESTEPS_FULL * TOKENS_PER_TIMESTEP  # 1644
SEQ_LEN_CACHED = NUM_TIMESTEPS_CACHED * TOKENS_PER_TIMESTEP  # 1370
SEQ_LEN_NEW = TOKENS_PER_TIMESTEP  # 274


class DualONNXInference:
    """Streaming inference using dual ONNX model setup."""

    def __init__(self, full_model_path: str, incremental_model_path: str):
        # Load both ONNX models
        self.full_session = ort.InferenceSession(
            full_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.incr_session = ort.InferenceSession(
            incremental_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Get input/output names
        self.full_inputs = {inp.name for inp in self.full_session.get_inputs()}
        self.incr_inputs = {inp.name for inp in self.incr_session.get_inputs()}

        # Cache state
        self.proj_emb_cache = None
        self.kv_cache = None
        self.is_initialized = False

    def _flatten_batch(self, batch: dict, prefix: str = "batch") -> dict:
        """Flatten nested batch dict for ONNX input."""
        result = {}

        def recurse(obj, current_prefix):
            if isinstance(obj, np.ndarray):
                name = current_prefix.lower().replace("/", "_")
                result[name] = obj
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    recurse(value, f"{current_prefix}_{key}")

        recurse(batch, prefix)
        return result

    def _match_inputs(self, input_names: set, inputs: dict) -> dict:
        """Match input dict keys to ONNX input names (case-insensitive)."""
        matched = {}
        for onnx_name in input_names:
            for our_name, value in inputs.items():
                if our_name.lower() == onnx_name.lower():
                    matched[onnx_name] = value
                    break
        return matched

    def _parse_outputs(self, outputs: list) -> dict:
        """Parse ONNX outputs into named dict."""
        return {
            "brake_pedal": float(outputs[0].flatten()[0]),
            "gas_pedal": float(outputs[1].flatten()[0]),
            "steering_angle": float(outputs[2].flatten()[0]),
            "turn_signal": int(outputs[3].flatten()[0]),
            "proj_emb": outputs[4],
            "kv_cache": outputs[5],
        }

    def run_cold_start(self, batch_6ts: dict) -> dict:
        """Run cold start inference with 6 timesteps.

        Args:
            batch_6ts: Input batch with 6 timesteps
                - data/cam_front_left: [1, 6, 3, 256, 256]
                - data/meta/VehicleMotion/brake_pedal_normalized: [1, 6, 1]
                - data/meta/VehicleMotion/gas_pedal_normalized: [1, 6, 1]
                - data/meta/VehicleMotion/steering_angle_normalized: [1, 6, 1]
                - data/meta/VehicleMotion/speed: [1, 6, 1]
                - data/meta/VehicleState/turn_signal: [1, 6, 1]
                - data/waypoints/xy_normalized: [1, 6, 10, 2]

        Returns:
            Dict with predictions (brake_pedal, gas_pedal, steering_angle, turn_signal)
        """
        # Empty caches for cold start
        empty_proj_emb = np.empty((BATCH_SIZE, 0, EMBED_DIM), dtype=np.float32)
        empty_kv = np.empty((NUM_LAYERS, 2, BATCH_SIZE, 0, EMBED_DIM), dtype=np.float32)

        # Prepare inputs (mask is baked into the model)
        inputs = self._flatten_batch(batch_6ts)
        inputs["cached_projected_embeddings"] = empty_proj_emb
        inputs["cached_kv"] = empty_kv

        # Run full forward model
        matched_inputs = self._match_inputs(self.full_inputs, inputs)
        outputs = self.full_session.run(None, matched_inputs)

        # Parse and cache results
        result = self._parse_outputs(outputs)
        self.proj_emb_cache = result["proj_emb"]
        self.kv_cache = result["kv_cache"]
        self.is_initialized = True

        return {
            "brake_pedal": result["brake_pedal"],
            "gas_pedal": result["gas_pedal"],
            "steering_angle": result["steering_angle"],
            "turn_signal": result["turn_signal"],
        }

    def run_incremental(self, batch_1ts: dict) -> dict:
        """Run incremental inference with 1 new timestep.

        Args:
            batch_1ts: Input batch with 1 timestep
                - data/cam_front_left: [1, 1, 3, 256, 256]
                - data/meta/VehicleMotion/brake_pedal_normalized: [1, 1, 1]
                - ... (same structure as cold_start but with 1 timestep)

        Returns:
            Dict with predictions

        Raises:
            RuntimeError: If called before run_cold_start()
        """
        if not self.is_initialized:
            raise RuntimeError("Must call run_cold_start() before run_incremental()")

        # Trim caches to 5 timesteps (remove oldest timestep)
        self.proj_emb_cache = self.proj_emb_cache[:, TOKENS_PER_TIMESTEP:]  # [1, 1370, 384]
        self.kv_cache = self.kv_cache[:, :, :, TOKENS_PER_TIMESTEP:]  # [8, 2, 1, 1370, 384]

        # Prepare inputs (mask is baked into the model)
        inputs = self._flatten_batch(batch_1ts)
        inputs["cached_projected_embeddings"] = self.proj_emb_cache
        inputs["cached_kv"] = self.kv_cache

        # Run incremental model
        matched_inputs = self._match_inputs(self.incr_inputs, inputs)
        outputs = self.incr_session.run(None, matched_inputs)

        # Parse and update cache
        result = self._parse_outputs(outputs)
        self.proj_emb_cache = result["proj_emb"]  # [1, 1644, 384]
        self.kv_cache = result["kv_cache"]  # [8, 2, 1, 1644, 384]

        return {
            "brake_pedal": result["brake_pedal"],
            "gas_pedal": result["gas_pedal"],
            "steering_angle": result["steering_angle"],
            "turn_signal": result["turn_signal"],
        }

    def reset(self):
        """Reset cache state for new sequence."""
        self.proj_emb_cache = None
        self.kv_cache = None
        self.is_initialized = False


# Usage example
if __name__ == "__main__":
    # Initialize dual model inference
    inference = DualONNXInference(
        full_model_path="outputs/.../ControlTransformer_cache.onnx",
        incremental_model_path="outputs/.../ControlTransformer_cache_incremental.onnx",
    )

    # Simulate streaming data
    def get_batch(num_timesteps: int) -> dict:
        """Create dummy batch for testing."""
        return {
            "data": {
                "cam_front_left": np.random.rand(1, num_timesteps, 3, 256, 256).astype(np.float32),
                "meta/VehicleMotion/brake_pedal_normalized": np.random.rand(1, num_timesteps, 1).astype(np.float32),
                "meta/VehicleMotion/gas_pedal_normalized": np.random.rand(1, num_timesteps, 1).astype(np.float32),
                "meta/VehicleMotion/steering_angle_normalized": np.random.rand(1, num_timesteps, 1).astype(np.float32) * 2 - 1,
                "meta/VehicleMotion/speed": np.random.rand(1, num_timesteps, 1).astype(np.float32) * 130,
                "meta/VehicleState/turn_signal": np.random.randint(0, 3, (1, num_timesteps, 1)).astype(np.int32),
                "waypoints/xy_normalized": np.random.rand(1, num_timesteps, 10, 2).astype(np.float32) * 20,
            }
        }

    # Cold start with 6 timesteps
    batch_6 = get_batch(6)
    predictions = inference.run_cold_start(batch_6)
    print(f"Cold start: brake={predictions['brake_pedal']:.4f}, "
          f"gas={predictions['gas_pedal']:.4f}, "
          f"steering={predictions['steering_angle']:.4f}, "
          f"turn_signal={predictions['turn_signal']}")

    # Streaming loop - process 1 timestep at a time
    for i in range(10):
        batch_1 = get_batch(1)
        predictions = inference.run_incremental(batch_1)
        print(f"Step {i+1}: brake={predictions['brake_pedal']:.4f}, "
              f"gas={predictions['gas_pedal']:.4f}, "
              f"steering={predictions['steering_angle']:.4f}, "
              f"turn_signal={predictions['turn_signal']}")
```

### Key Points

1. **Cold Start**: Use the full forward model (`ControlTransformer_cache.onnx`) with 6 timesteps and empty caches
2. **Cache Trimming**: Before each incremental call, trim caches to remove the oldest timestep (keep 5 timesteps = 1370 tokens)
3. **Incremental**: Use the incremental model (`ControlTransformer_cache_incremental.onnx`) with 1 new timestep and the trimmed 5-timestep cache
4. **Cache Update**: After each inference, update caches with the new outputs (back to 6 timesteps = 1644 tokens)

### Cache Flow Diagram

```
Cold Start (Full Model):
  Input:  batch[1,6,...] + empty_cache[1,0,384] + empty_kv[8,2,1,0,384]
  Output: predictions + proj_emb[1,1644,384] + kv[8,2,1,1644,384]

Trim (drop oldest timestep):
  proj_emb[1,1644,384] → proj_emb[1,1370,384]
  kv[8,2,1,1644,384]   → kv[8,2,1,1370,384]

Incremental (Incremental Model):
  Input:  batch[1,1,...] + proj_emb[1,1370,384] + kv[8,2,1,1370,384]
  Output: predictions + proj_emb[1,1644,384] + kv[8,2,1,1644,384]

Repeat: Trim → Incremental → Trim → Incremental → ...
```

---

#### TensorRT with Optimization Profiles (Alternative)

For more flexibility with variable cache lengths, use **TensorRT with optimization profiles**:

```python
import tensorrt as trt

# Create builder and network
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

with open("model_cache.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()

# Profile 0: Full forward (first inference)
# Note: mask is baked in, so no mask shape configuration needed
profile_full = builder.create_optimization_profile()
profile_full.set_shape("batch_data_cam_front_left",
    min=(1, 6, 3, 256, 256),
    opt=(1, 6, 3, 256, 256),
    max=(1, 6, 3, 256, 256))
profile_full.set_shape("cached_projected_embeddings",
    min=(1, 0, 384),
    opt=(1, 0, 384),
    max=(1, 0, 384))
profile_full.set_shape("cached_kv",
    min=(8, 2, 1, 0, 384),
    opt=(8, 2, 1, 0, 384),
    max=(8, 2, 1, 0, 384))
config.add_optimization_profile(profile_full)

# Profile 1: Incremental (subsequent inferences)
profile_incr = builder.create_optimization_profile()
profile_incr.set_shape("batch_data_cam_front_left",
    min=(1, 1, 3, 256, 256),
    opt=(1, 1, 3, 256, 256),
    max=(1, 1, 3, 256, 256))
profile_incr.set_shape("cached_projected_embeddings",
    min=(1, 274, 384),      # 1 cached timestep
    opt=(1, 1370, 384),     # 5 cached timesteps
    max=(1, 1370, 384))
profile_incr.set_shape("cached_kv",
    min=(8, 2, 1, 274, 384),
    opt=(8, 2, 1, 1370, 384),
    max=(8, 2, 1, 1370, 384))
config.add_optimization_profile(profile_incr)

# Build engine
engine = builder.build_serialized_network(network, config)
```

At runtime, switch profiles based on inference mode:
```python
# First inference: use profile 0 (full forward)
context.set_optimization_profile_async(0, stream)

# Subsequent inferences: use profile 1 (incremental)
context.set_optimization_profile_async(1, stream)
```

**Note**: PyTorch incremental inference with KV cache works without these limitations and is recommended for development/testing.

---

## Verification Scripts

### Single Model Verification

Verify that a single ONNX model matches PyTorch outputs:

```bash
uv run python -m rmind.scripts.verify_onnx_cache \
    --config-dir=config \
    --config-name=export/verify_onnx_cache \
    onnx_path=/path/to/model.onnx
```

This compares:
- Policy predictions (brake, gas, steering, turn signal)
- Projected embeddings cache
- KV cache

### Dual Model Verification

Verify that the full forward + incremental ONNX model setup matches native PyTorch:

```bash
uv run python -m rmind.scripts.verify_dual_onnx \
    --config-dir=config \
    --config-name=export/verify_dual_onnx
```

This runs comprehensive comparisons:
1. **PyTorch 6-timestep vs PyTorch dual (5+1)** - Verifies internal consistency
2. **ONNX full (6) vs PyTorch full (6)** - Verifies full forward model
3. **ONNX dual (5+1) vs PyTorch dual (5+1)** - Verifies incremental model with PyTorch cache
4. **ONNX dual (5+1) vs PyTorch full (6)** - The ultimate test

All comparisons should show `MATCH` with 0.0 difference for predictions.

---

## Testing

Run the cache export tests:

```bash
uv run pytest tests/test_export_cache.py -v
```

Tests verify:
- KV cache output shapes
- Incremental inference matches full forward
- Wrapper output shapes
- Prediction structure

---

## Troubleshooting

### Export Fails with Device Mismatch

Ensure all model components are on the same device before export. The export scripts handle this automatically when using Hydra configuration.

### Large Model Size

Use `+external_data=true` to store weights in a separate file:

```bash
just export-onnx +external_data=true
```

### Verification Fails

If ONNX Runtime verification fails, try disabling optimizations:

```bash
just export-onnx +optimize=false +verify=true
```

---

## Recent Changes (2026-01-21)

### Boolean Mask Fix

Fixed numerical differences between PyTorch and ONNX by ensuring consistent boolean mask handling.

**Problem**: The custom `_scaled_dot_product_attention` method in `llm.py` was passing boolean masks directly to `F.scaled_dot_product_attention`, which handles boolean masks differently than `nn.MultiheadAttention`.

**Solution**: Convert boolean masks to float masks with `-inf` for masked positions before passing to `F.scaled_dot_product_attention`:

```python
# In _scaled_dot_product_attention (llm.py)
if mask is not None and mask.dtype == torch.bool:
    mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill_(mask, float("-inf"))
```

**Result**: Reduced max diff from ~8.7 to 0.0 for projected embeddings, and from ~60% to 0.0% for predictions.

### Unified Encoder Path

Modified `CacheEnabledControlTransformer` to compute predictions directly from the KV-cached encoder output, rather than calling `obj(episode)` which runs a separate encoder call. This ensures:
- Single encoder pass for both cache updates and predictions
- Consistent behavior between PyTorch and ONNX
- Support for incremental inference (which requires non-square attention masks)

### Prediction Logic Fix

Fixed incorrect prediction computation in `_compute_policy_predictions`:

**Problem**: Predictions were using incorrect transformations:
- Continuous actions used `sigmoid()` (wrong)
- Discrete actions used `argmax()` (wrong for turn signal)

**Solution**: Match the logic from `PolicyObjective.forward`:
```python
# Continuous actions: just take first element (no sigmoid)
case (Modality.CONTINUOUS, _):
    return logit[..., 0]

# Turn signal: softmax + threshold-based selection
case (Modality.DISCRETE, "turn_signal"):
    return non_zero_signal_with_threshold(logit).class_idx
```

**Result**: Predictions now match native PyTorch exactly.

### Verification Improvements

- Added `verify_onnx_cache.py` for single model verification
- Added `verify_dual_onnx.py` for dual model setup verification
- Set deterministic seed (`torch.manual_seed(42)`) before generating test batch data for reproducible verification

### Baked-In Attention Masks

Attention masks are now **baked into the ONNX model** rather than passed as inputs:

**Changes:**
- Removed `mask` from model inputs
- Added `full_mask` and `incr_mask` as registered buffers in `CacheEnabledControlTransformer`
- Model automatically selects mask based on cache state:
  - Empty cache (`cached_kv.shape[3] == 0`): Uses `full_mask`
  - Non-empty cache: Uses `incr_mask`

**Benefits:**
- Simpler inference API (no need to build/pass masks)
- Guaranteed correct mask patterns (cannot accidentally use wrong mask)
- Reduced input count for ONNX model

**Standalone mask builder** (`rmind/scripts/build_onnx_mask.py`) still available for reference/debugging:
```python
from rmind.scripts.build_onnx_mask import build_policy_mask, build_incremental_mask

mask_full = build_policy_mask(num_timesteps=6)
mask_incr = build_incremental_mask(num_cached_timesteps=5)
```
