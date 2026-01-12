# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rmind is a PyTorch Lightning-based framework for training foundation models for spatial intelligence. The codebase uses Hydra for configuration management and focuses on transformer-based control models that process episodic data with multiple modalities (images, continuous/discrete actions, etc.).

## Development Commands

### Environment Setup
```bash
# Using Nix (recommended)
nix develop

# Manual setup
just setup  # installs dependencies and pre-commit hooks
just sync   # sync dependencies from uv.lock
```

### Configuration Generation
**IMPORTANT**: Before running train, predict, or test commands, config files must be generated from templates:
```bash
just generate-config  # generates YAML from templates in config/_templates using ytt
```

This step is automatically run by `just train`, `just predict`, `just test`, and `just export-onnx`.

### Training
```bash
# Standard training
just train experiment=yaak/control_transformer/pretrain [hydra overrides...]

# Debug mode (disables W&B logging)
just train-debug experiment=yaak/control_transformer/pretrain [...]

# Example with overrides
just train experiment=yaak/control_transformer/pretrain trainer.max_epochs=100 model.optimizer.lr=1e-4
```

### Inference
```bash
# Start rerun server first (if using RerunPredictionWriter callback)
just rerun

# Run prediction
just predict inference=yaak/control_transformer/{config} model.artifact=yaak/rmind/model-{run_id}:v{version} [+model.map_location=cuda:0] [+model.strict=false]
```

### Testing
```bash
# Run all tests
just test

# Run specific test file
just test tests/path/to/test.py

# Run specific test function
just test tests/path/to/test.py::test_function_name

# Run with verbose output (default)
just test -v tests/path/to/test.py
```

### Code Quality
```bash
just format              # format code with ruff
just lint                # lint with ruff
just typecheck           # type check with ty (basedpyright wrapper)
just prek                # run all pre-commit hooks on all files
```

### Export
```bash
just export-onnx model=yaak/control_transformer/raw_export input=yaak/control_transformer/dummy +report=true
```

### Cleanup
```bash
just clean  # removes dist, outputs, lightning_logs, wandb, artifacts directories
```

## Architecture Overview

### Core Components

**Models** (`src/rmind/models/`):
- `ControlTransformer`: Main PyTorch Lightning module that orchestrates training
  - Composes: episode_builder, encoder (optional), objectives (ModuleDict)
  - Handles shared encoder: if objectives have `encoder=None`, they share the model's encoder
  - Custom state_dict handling to avoid duplicating shared encoder weights
  - Supports checkpoint loading with hparams_updaters for migration

**Episode System** (`src/rmind/components/episode.py`):
- `EpisodeBuilder`: Transforms raw batch data into structured Episodes
  - Pipeline: input → tokenizers → embeddings → projections → position encodings
  - Builds `Episode` (TensorClass) containing input, tokens, embeddings, indices, and timestep info
  - Dual-mode: returns `Episode` (TensorClass) for training or `EpisodeExport` (dataclass) for export
- `Episode`: Structured representation with:
  - `input`, `input_tokens`, `input_embeddings`: Raw and tokenized inputs
  - `projected_embeddings`, `position_embeddings`: Transformed features
  - `index`: Token indices organized by modality (image, continuous, discrete, context, foresight, summary, utility)
  - `timestep`: Metadata mapping (token_type, modality, name) → position in sequence
  - Property `embeddings_packed`: Flattened sequence of all embeddings ready for transformer

**Objectives** (`src/rmind/components/objectives/`):
- Base class `Objective` with abstract methods:
  - `compute_metrics(episode) -> Metrics`: Computes loss during training
  - `predict(episode, keys, tokenizers) -> TensorDict`: Generates predictions
- Objectives are stored in a ModuleDict and each can have its own encoder or share the model's encoder
- Examples: inverse_dynamics, forward_dynamics, policy, memory_extraction, random_masked_hindsight_control

**Components** (`src/rmind/components/`):
- `ModuleDict`: Custom container with Hydra instantiation support
- Tokenizers, embeddings, projections: Transform inputs to token embeddings
- Position encodings: Support for timestep, observations, actions, special tokens, context
- LLM integration, timm backbone support
- Custom losses, norms, masks

### Configuration System

Uses Hydra with hierarchical configs in `config/`:
- `train.yaml`, `predict.yaml`: Entry points
- `experiment/`: Complete experiment configs (e.g., yaak/control_transformer/pretrain)
- `model/`, `datamodule/`, `trainer/`, `logger/`: Component configs
- `_templates/`: YTT templates that generate final configs with `just generate-config`

Config structure uses `_target_` for Hydra instantiation:
```yaml
model:
  _target_: rmind.models.ControlTransformer
  episode_builder:
    _target_: rmind.components.episode.EpisodeBuilder
    # ...
  objectives:
    _target_: rmind.components.containers.ModuleDict
    # ...
```

### Data Flow

1. **Training**: Raw batch → EpisodeBuilder → Episode → Objectives.compute_metrics() → losses
2. **Prediction**: Raw batch → EpisodeBuilder → Episode → Objectives.predict() → TensorDict
3. **Export**: Same as training but EpisodeBuilder returns EpisodeExport (torch.export compatible)

### Key Design Patterns

- **Pydantic validation**: Models use `@validate_call` and Pydantic models for config validation
- **HydraConfig**: Type-safe wrapper for Hydra configs with `.instantiate()` method
- **TensorDict/TensorClass**: Used throughout for structured tensor data
- **PyTree registration**: Custom classes registered as PyTrees for torch operations
- **Shared encoder optimization**: Encoder can be shared across objectives, state_dict handles deduplication

### Scripts

- `rmind-train`: Main training script with W&B integration and git code logging
- `rmind-predict`: Inference script
- `rmind-export-onnx`: ONNX export script
- All scripts use `multiprocessing.set_forkserver_preload(["rbyte", "polars"])` for stability

## Important Notes

- **Python version**: Requires Python 3.12
- **PyTorch backend**: Uses CUDA 12.8 on Linux, CPU elsewhere (configured in pyproject.toml)
- **Pre-commit hooks**: Run automatically via `prek` (not pre-commit), install with `just setup`
- **Ruff settings**: Aggressive linting/formatting (preview features enabled, unsafe fixes allowed)
- **Type checking**: Uses `ty` (basedpyright wrapper), warnings on unused type ignore comments
- **Environment variables**: Set in justfile (HYDRA_FULL_ERROR, WANDB_DIR, etc.)
- **Config templates**: Must regenerate with `just generate-config` after modifying `_templates/`
- **W&B logging**: Controlled by `WANDB_MODE` env var (set to "disabled" in train-debug)
