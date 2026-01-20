# ONNX Export Guide

This guide explains how to export ControlTransformer models to ONNX format for deployment.

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

```
batch:                      dict - Input batch (full batch or single timestep)
cached_projected_embeddings: [B, S_cached, D] - Cached projected embeddings WITHOUT position
                             embeddings (zeros with S_cached=0 for first call)
cached_kv:                  [L, 2, B, S_cached, D] - Cached KV (zeros with S_cached=0 for first call)
mask:                       [S_new, S_total] - Attention mask where S_total = S_cached + S_new
```

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

# Process all timesteps
predictions, cached_proj_emb, cached_kv = model(full_batch, empty_proj_emb, empty_kv, full_mask)
```

#### Subsequent Inferences (Incremental)
```python
# Process single timestep with cached data
# Mask should be [S_new, S_total] where new positions can attend to all past + self
predictions, cached_proj_emb, cached_kv = model(
    single_timestep_batch,
    cached_proj_emb,      # Previous projected embeddings (without PE)
    cached_kv,            # Previous KV cache
    incremental_mask,     # [S_new, S_cached + S_new]
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

### Building Attention Masks

The attention mask controls which positions can attend to which. For causal (autoregressive) attention:

```python
def build_causal_mask(seq_new: int, seq_total: int) -> np.ndarray:
    """Build causal attention mask for incremental inference.

    Args:
        seq_new: Number of new sequence positions (tokens in new timestep)
        seq_total: Total sequence length (cached + new)

    Returns:
        mask: [seq_new, seq_total] boolean mask where True = masked (cannot attend)
    """
    # New positions start at index (seq_total - seq_new)
    start_idx = seq_total - seq_new

    # Create mask: each new position can attend to all previous + itself
    mask = np.ones((seq_new, seq_total), dtype=bool)
    for i in range(seq_new):
        # Position i (in new tokens) can attend to all positions up to start_idx + i
        mask[i, :start_idx + i + 1] = False

    return mask
```

### First Inference (Full Forward)

```python
def run_first_inference(session, batch: dict, num_timesteps: int):
    """Run first inference with empty cache."""

    seq_len = num_timesteps * tokens_per_timestep

    # Initialize empty caches (sequence dimension is 0, meaning no cached data)
    # Using np.empty since there are no elements to initialize
    empty_proj_emb = np.empty((batch_size, 0, embed_dim), dtype=np.float32)
    empty_kv = np.empty((num_layers, 2, batch_size, 0, embed_dim), dtype=np.float32)

    # Full causal mask [S, S]
    mask = build_causal_mask(seq_len, seq_len)

    # Prepare inputs
    inputs = {
        **flatten_batch(batch),  # Flatten nested batch dict for ONNX
        "cached_projected_embeddings": empty_proj_emb,
        "cached_kv": empty_kv,
        "mask": mask,
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

    # Compute sequence lengths
    seq_cached = cached_proj_emb.shape[1]
    seq_new = tokens_per_timestep  # Single timestep
    seq_total = seq_cached + seq_new

    # Incremental mask: new positions attend to all cached + self
    mask = build_causal_mask(seq_new, seq_total)

    # Prepare inputs
    inputs = {
        **flatten_batch(new_timestep_batch),
        "cached_projected_embeddings": cached_proj_emb,
        "cached_kv": cached_kv,
        "mask": mask,
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
        mask: Tensor,
    ) -> tuple[dict, Tensor, Tensor]:
        # Build episode and get projected embeddings (WITHOUT position embeddings)
        episode = self.episode_builder(batch)
        new_proj_emb = episode.projected_embeddings_packed

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
        predictions = {name: obj(episode) for name, obj in self.objectives.items()}

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
profile_full.set_shape("mask",
    min=(1644, 1644),
    opt=(1644, 1644),
    max=(1644, 1644))
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
profile_incr.set_shape("mask",
    min=(274, 548),         # 1 new + 1 cached
    opt=(274, 1644),        # 1 new + 5 cached
    max=(274, 1644))
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
