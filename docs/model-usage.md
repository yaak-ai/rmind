# Model Usage Guide

This guide explains how to load and use the three variants of the ControlTransformer model:
1. **PyTorch Native** - Full model in PyTorch, suitable for training/experimentation
2. **PyTorch Cache-Enabled** - Optimized for inference with KV cache mechanism
3. **ONNX Models** - Production deployments with TensorRT support

## Quick Reference

| Model | Framework | Caching | Latency | Use Case |
|-------|-----------|---------|---------|----------|
| PyTorch Native | PyTorch | No | Baseline | Development, training |
| Cache-Enabled | PyTorch | Yes | 1.1x faster | Batch inference with history |
| ONNX Full | ONNX Runtime | No | 0.4x | Deployment (TensorRT) |
| ONNX Incremental | ONNX Runtime | Yes | 0.6x | Real-time streaming inference |

---

## 1. PyTorch Native Model

### Loading the Model

```python
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import OmegaConf
from pathlib import Path
import torch

# Initialize Hydra config
config_path = Path(__file__).parents[3] / "config"
with initialize_config_dir(config_dir=str(config_path), version_base=None):
    cfg = compose(config_name="export/onnx_cache")

# Convert config and instantiate model
model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
model = hydra_instantiate(model_cfg).to("cuda:0").eval()

# Optional: Load trained weights from W&B artifact
from rmind.scripts.export_onnx_cache import _load_weights_from_artifact
_load_weights_from_artifact(model, "yaak/cargpt/model-y93ejvgg:v9")
```

### Input Format

Batch dictionary with shape `[B, T, ...]`:

```python
batch = {
    "data": {
        "cam_front_left": torch.uint8 [B, T, H, W, C],           # [1, 6, 324, 576, 3]
        "meta/VehicleMotion/speed": torch.float32 [B, T, 1],     # [1, 6, 1]
        "meta/VehicleMotion/gas_pedal_normalized": torch.float32 [B, T, 1],
        "meta/VehicleMotion/brake_pedal_normalized": torch.float32 [B, T, 1],
        "meta/VehicleMotion/steering_angle_normalized": torch.float32 [B, T, 1],
        "meta/VehicleState/turn_signal": torch.long [B, T, 1],
        "waypoints/xy_normalized": torch.float32 [B, T, 10, 2],
    }
}
```

### Usage

```python
from rmind.components.objectives.base import PredictionKey

with torch.inference_mode():
    # Build episode
    episode = model.episode_builder(batch)

    # Run inference
    policy_objective = model.objectives["policy"]
    predictions = policy_objective.predict(
        episode, keys={PredictionKey.PREDICTION_VALUE}
    )

    # Extract predictions
    pred_obj = predictions[PredictionKey.PREDICTION_VALUE]
    pred_value = pred_obj.value

    brake = float(pred_value["continuous", "brake_pedal"].squeeze().cpu())
    gas = float(pred_value["continuous", "gas_pedal"].squeeze().cpu())
    steering = float(pred_value["continuous", "steering_angle"].squeeze().cpu())
    turn_signal = int(pred_value["discrete", "turn_signal"].squeeze().cpu())
```

### Output

```python
# predictions[PredictionKey.PREDICTION_VALUE].value is a dict with keys:
# ("continuous", "brake_pedal") → tensor [1, 1] ∈ [0, 1]
# ("continuous", "gas_pedal") → tensor [1, 1] ∈ [0, 1]
# ("continuous", "steering_angle") → tensor [1, 1] ∈ [-1, 1]
# ("discrete", "turn_signal") → tensor [1, 1] ∈ {0, 1, 2}
```

### Performance

- **Latency**: ~75ms per 6-timestep batch (baseline)
- **Memory**: ~2GB GPU VRAM
- **Strengths**: Full control, training-compatible
- **Weaknesses**: Slower than optimized variants

---

## 2. PyTorch Cache-Enabled Model

### What is KV Cache?

The cache-enabled model keeps the Key/Value representations from previous timesteps in memory, so only the new timestep needs processing. This is critical for streaming inference.

**Without cache (Full Forward):**
```
Process all 6 timesteps → Full transformer computation
```

**With cache (Incremental):**
```
First batch: Process all 6 timesteps (compute + cache)
Next batch: Process 1 new timestep (use cached KV from previous 5)
```

### Loading the Model

```python
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import OmegaConf
from rmind.scripts.export_onnx_cache import CacheEnabledControlTransformer, _load_weights_from_artifact
from rmind.components.objectives.base import TorchAttentionMaskLegend
from pathlib import Path
import torch

# Load base PyTorch model
config_path = Path(__file__).parents[3] / "config"
with initialize_config_dir(config_dir=str(config_path), version_base=None):
    cfg = compose(config_name="export/onnx_cache")

model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
pytorch_model = hydra_instantiate(model_cfg).to("cuda:0").eval()

# Load trained weights from artifact
_load_weights_from_artifact(pytorch_model, "yaak/cargpt/model-y93ejvgg:v9")

# Build episode to extract mask and position embeddings
episode = pytorch_model.episode_builder(batch)
mask = episode.policy.build_attention_mask(
    episode.index, episode.timestep, legend=TorchAttentionMaskLegend
)
pe = episode.position_embeddings_packed  # [1, T*tokens_per_ts, embed_dim]

# Wrap with cache
cache_model = CacheEnabledControlTransformer(
    pytorch_model,
    mask=mask,
    position_embeddings_packed=pe,
).to("cuda:0").eval()
```

### Usage with Full Forward (6 timesteps)

```python
embed_dim = 384
num_layers = 8
device = torch.device("cuda:0")

# Empty cache for full forward
empty_proj_emb = torch.zeros(1, 0, embed_dim, device=device)
empty_kv = torch.zeros(num_layers, 2, 1, 0, embed_dim, device=device)

with torch.inference_mode():
    # Returns: (brake, gas, steering, turn_signal, proj_emb, kv_cache)
    outputs = cache_model(batch, empty_proj_emb, empty_kv)
    brake, gas, steering, turn_signal, proj_emb_out, kv_out = outputs
```

### Usage with Streaming (Initial + Incremental)

```python
# First batch: Process 6 timesteps
with torch.inference_mode():
    outputs_full = cache_model(batch, empty_proj_emb, empty_kv)
    brake, gas, steering, turn_signal, proj_emb_out, kv_out = outputs_full

    # Cache the first 5 timesteps worth of embeddings
    onnx_tokens_per_ts = 338
    proj_emb_5 = proj_emb_out[:, :5 * onnx_tokens_per_ts, :]
    kv_5 = kv_out[:, :, :, :5 * onnx_tokens_per_ts, :]

# Next batch: Use cache + 1 new timestep
batch_new = load_next_frame_batch()
batch_1 = slice_batch(batch_new, slice(-1, None))  # Only last timestep

with torch.inference_mode():
    outputs_incr = cache_model(batch_1, proj_emb_5, kv_5)
    brake, gas, steering, turn_signal, proj_emb_new, kv_new = outputs_incr

    # Update cache for next iteration
    proj_emb_5 = proj_emb_new[:, :5 * onnx_tokens_per_ts, :]
    kv_5 = kv_new[:, :, :, :5 * onnx_tokens_per_ts, :]
```

### Performance

- **Full forward latency**: ~67ms (1.1x faster than native)
- **Incremental latency**: ~20-30ms per timestep (3-4x faster)
- **Memory**: ~2GB + cache storage (additional ~500MB for KV cache)
- **Strengths**: Streaming capable, faster incremental inference
- **Weaknesses**: More complex API

---

## 3. ONNX Models

### Model Variants

Two ONNX models available in `outputs/latest/`:

1. **ControlTransformer_cache.onnx** - Full forward with 6 timesteps
2. **ControlTransformer_cache_incremental.onnx** - Incremental with 1 timestep

### Loading and Basic Setup

```python
import onnxruntime as ort
import numpy as np

# Load models
full_session = ort.InferenceSession(
    "outputs/latest/ControlTransformer_cache.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
incr_session = ort.InferenceSession(
    "outputs/latest/ControlTransformer_cache_incremental.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Get input/output names
full_inputs = full_session.get_inputs()
full_input_names = [inp.name for inp in full_inputs]
full_outputs = full_session.get_outputs()
```

### Input Preparation

Convert PyTorch batch to ONNX format:

```python
def flatten_batch_to_onnx(batch: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert batch dict to flat ONNX input dict."""
    onnx_inputs = {}

    # Image: [B, T, H, W, C] uint8 → [B*T, H, W, C]
    img = batch["data"]["cam_front_left"]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    B, T, H, W, C = img.shape
    onnx_inputs["batch_data_cam_front_left"] = img.reshape(B*T, H, W, C)

    # Speed: [B, T, 1] → [B*T, 1]
    speed = batch["data"]["meta/VehicleMotion/speed"]
    if isinstance(speed, torch.Tensor):
        speed = speed.cpu().numpy()
    onnx_inputs["batch_data_meta/VehicleMotion/speed"] = speed.reshape(B*T, 1)

    # (Similar for gas_pedal, brake_pedal, steering_angle, turn_signal, waypoints)

    return onnx_inputs
```

### Full Forward Inference

```python
# Prepare inputs
batch_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
onnx_inputs = flatten_batch_to_onnx(batch_cpu)

# Add empty cache
embed_dim = 384
num_layers = 8
onnx_inputs["cached_projected_embeddings"] = np.empty((1, 0, embed_dim), dtype=np.float32)
onnx_inputs["cached_kv"] = np.empty((num_layers, 2, 1, 0, embed_dim), dtype=np.float32)

# Run inference
outputs = full_session.run(None, onnx_inputs)
# outputs = [brake, gas, steering, turn_signal, proj_emb, kv_cache]
```

### Incremental Inference

```python
# First: Get cache from full forward (6 timesteps)
full_outputs = full_session.run(None, onnx_inputs_full)
proj_emb_out, kv_out = full_outputs[-2], full_outputs[-1]

# Extract cache for first 5 timesteps
onnx_tokens_per_ts = 338
proj_emb_5 = proj_emb_out[:, :5 * onnx_tokens_per_ts, :]
kv_5 = kv_out[:, :, :, :5 * onnx_tokens_per_ts, :]

# Next: Run incremental with 1 new timestep
batch_1 = load_next_frame()  # Only 1 timestep
onnx_inputs_incr = flatten_batch_to_onnx(batch_1)
onnx_inputs_incr["cached_projected_embeddings"] = proj_emb_5
onnx_inputs_incr["cached_kv"] = kv_5

# Run
outputs_incr = incr_session.run(None, onnx_inputs_incr)
brake, gas, steering, turn_signal, proj_emb_new, kv_new = outputs_incr
```

### Performance

- **Full forward latency**: ~300ms (reference only, use incremental)
- **Incremental latency**: ~120ms per timestep
- **Memory**: Minimal (~500MB)
- **Strengths**: Production-ready, TensorRT compatible, no dependencies on training code
- **Weaknesses**: Fixed architecture, more verbose input preparation

---

## Comparison & Selection Guide

### When to Use Each Model

**PyTorch Native**
- ✅ Research, experimentation, fine-tuning
- ✅ Need custom loss functions or modifications
- ❌ Production deployment

**PyTorch Cache-Enabled**
- ✅ Real-time inference with streaming data
- ✅ Development of streaming pipelines
- ✅ Benchmarking cache mechanism
- ❌ Production (requires PyTorch runtime)

**ONNX Models**
- ✅ Production deployment
- ✅ Edge devices with TensorRT
- ✅ Fixed inference pipeline
- ✅ Minimal dependencies
- ❌ Custom modifications needed

### Latency Breakdown (6 timestep batch)

```
PyTorch Native:        75 ms  (1.00x)
PyTorch Cache (full):  67 ms  (1.11x faster)
ONNX Full Forward:    325 ms  (0.23x - avoid this)
ONNX Incremental:     117 ms  (0.64x)
```

### Memory Requirements

```
PyTorch Native:        2.0 GB
PyTorch Cache:         2.5 GB (with KV cache)
ONNX Full:             0.5 GB
ONNX Incremental:      0.5 GB
```

---

## Input Specifications

### Image Data

- **Format**: uint8 BGR or RGB
- **Size**: 324×576 pixels (after center crop + resize)
- **Normalization**: ImageNet (baked into ONNX models, manual for PyTorch)
- **Batch**: [B, T, H, W, C]

**PyTorch preprocessing:**
```python
from torchvision.transforms import v2

preprocess = v2.Compose([
    v2.CenterCrop((320, 576)),
    v2.Resize((256, 256)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Continuous Values

- **Speed**: [0, 130] km/h
- **Gas/Brake**: [0, 1] normalized
- **Steering**: [-1, 1] normalized

### Discrete Values

- **Turn Signal**: {0=None, 1=Left, 2=Right}

### Context (Waypoints)

- **Shape**: [B, T, 10, 2]
- **Format**: 10 future waypoints (x, y) in ego-frame, normalized
- **Units**: [-∞, +∞] (egocentric distance in arbitrary units)

---

## GPU Setup (ONNX Runtime)

### Requirements

To enable GPU acceleration for ONNX Runtime, install system-level NVIDIA libraries:

**Required:**
- CUDA Toolkit 12.x
- cuDNN 9.* (critical: version mismatch causes CPU fallback)
- TensorRT (optional but recommended for best performance)

### Installation Steps

1. **Install CUDA 12.x** (if not already installed)
   ```bash
   # Download from NVIDIA
   # https://developer.nvidia.com/cuda-downloads

   # Verify installation
   nvcc --version
   ```

2. **Install cuDNN 9** (required for GPU support)
   ```bash
   # Download from NVIDIA (requires account)
   # https://developer.nvidia.com/cudnn

   # Extract and add to library path
   export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH

   # Verify
   ldconfig -p | grep cudnn
   ```

3. **Install onnxruntime-gpu**
   ```bash
   uv pip install onnxruntime-gpu
   ```

4. **Verify GPU providers are available**
   ```python
   import onnxruntime as ort
   print(ort.get_available_providers())
   # Should show: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
   ```

### Troubleshooting GPU Issues

**Problem:** "Failed to load libcudnn.so.9"
- **Cause:** cuDNN 9 not installed or not in LD_LIBRARY_PATH
- **Solution:**
  ```bash
  export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```

**Problem:** Provider falls back to CPU
- **Check:** `ort.get_available_providers()` should include `CUDAExecutionProvider`
- **Solution:** Ensure all CUDA/cuDNN libraries are properly installed and in PATH

**Problem:** GPU slower than CPU
- **Cause:** Model too small or data transfer overhead
- **Solution:** Use batch processing, keep tensors on GPU

---

## Troubleshooting

### ONNX Runtime Providers

```python
# Check available providers
import onnxruntime as ort
print(ort.get_available_providers())

# Check which provider is actually used
session = ort.InferenceSession("model.onnx")
print(session.get_providers())  # Shows what's actually used

# Force specific provider
session = ort.InferenceSession(
    "model.onnx",
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

### Cache Shape Mismatch

```python
# Error: "input shape mismatch"
# Solution: Ensure cache shapes match token dimensions
onnx_tokens_per_ts = 338
batch_size = 1
embed_dim = 384
num_layers = 8

# For 5 timesteps cached:
proj_emb_5 = np.zeros((batch_size, 5 * onnx_tokens_per_ts, embed_dim))
kv_5 = np.zeros((num_layers, 2, batch_size, 5 * onnx_tokens_per_ts, embed_dim))
```

### Prediction Inconsistency

Ensure:
- ✅ Same attention mask format (boolean for PyTorch, baked into ONNX)
- ✅ Same position embeddings (stored in episode for PyTorch, baked into ONNX)
- ✅ Same image normalization
- ✅ Same batch slicing for incremental (only last timestep)

---

## References

- [ONNX Export Guide](./ONNX_EXPORT.md)
- [Incremental Inference Details](./INCREMENTAL_INFERENCE_REQUEST.md)
- Model config: `config/model/yaak/control_transformer/raw_export.yaml`
- Export config: `config/export/onnx_cache.yaml`
