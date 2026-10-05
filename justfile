export HOSTNAME := `hostname`
export PYTHONBREAKPOINT := "patdb.debug"
export PATDB_CODE_STYLE := "vim"
export BETTER_EXCEPTIONS := "1"
export LOVELY_TENSORS := "1"
export HYDRA_FULL_ERROR := "1"
export RERUN_STRICT := "1"
export WANDB_DIR := "wandb_logs"
export TORCHDYNAMO_VERBOSE := "1"
export PYTORCH_CUDA_ALLOC_CONF := "expandable_segments:True"

# export PYTHONOPTIMIZE := "1" # incompatible w/ torch.export in 2.10

_default:
    @just --list --unsorted

sync:
    uv sync --all-extras --all-groups --locked

setup: sync
    prek install --overwrite

check:
    uv format --check
    uv run ruff check
    uv check

check-git:
    uv run rmind-check-git

prek *ARGS:
    prek --all-files {{ ARGS }}

# generate config files from templates with ytt
generate-config:
    ytt --file {{ justfile_directory() }}/config/_templates \
        --output-files {{ justfile_directory() }}/config/ \
        --output yaml \
        --ignore-unknown-comments \
        --strict

train *ARGS: generate-config check-git
    uv run \
        --extra train \
        rmind-train \
        --config-path {{ justfile_directory() }}/config \
        --config-name train.yaml \
        {{ ARGS }}

# --- robot-native nero (docs/nero_robot_patch_policy.md, "Environment") ------
# The ONE supported invocation of the nero robot pipeline. Its rbyte ingestion
# (NeroRobotDataFrameBuilder, NeroRobotWindowBuilder, TransformedTensorSource)
# lives on rbyte's feat/nero-arms-depth branch, not in the pinned PyPI rbyte, so
# that checkout (RBYTE_SRC, default ../rbyte) SHADOWS it on PYTHONPATH; the
# `nero` dependency group adds its MCAP deps. Not `uv run --with-editable`: that
# overlay resolves rbyte's deps on its own and pulls in a second torch.
rbyte_src := env("RBYTE_SRC", justfile_directory() / ".." / "rbyte")

# fail fast unless rbyte resolves to RBYTE_SRC and carries the nero ingestion
nero-check-env:
    PYTHONPATH={{ rbyte_src }}/src uv run --extra train --group nero python -c \
        "import pathlib, rbyte, mcap, mcap_protobuf; \
        from rbyte.io import NeroRobotDataFrameBuilder, NeroRobotWindowBuilder, TransformedTensorSource; \
        src = pathlib.Path(rbyte.__file__).resolve(); \
        want = pathlib.Path('{{ rbyte_src }}').resolve(); \
        assert want in src.parents, f'rbyte from {src}, not RBYTE_SRC={want}'; \
        print('rbyte', src)"

# e.g. `just nero-train experiment=yaak/nero_robot/tokenizer relative_mode=hand`
nero-train *ARGS: generate-config nero-check-env
    PYTHONPATH={{ rbyte_src }}/src uv run \
        --extra train --group nero \
        rmind-train \
        --config-path {{ justfile_directory() }}/config \
        --config-name train.yaml \
        {{ ARGS }}

# any nero script, e.g. `just nero-run rmind.scripts.nero_fit_stats --experiment
# yaak/nero_robot/tokenizer --out $NERO_STATS_DIR`
nero-run MODULE *ARGS: generate-config nero-check-env
    PYTHONPATH={{ rbyte_src }}/src uv run \
        --extra train --extra export --group nero \
        python -m {{ MODULE }} {{ ARGS }}

train-debug *ARGS: generate-config
    WANDB_MODE=disabled \
    uv run \
        --extra train \
        rmind-train \
        --config-path {{ justfile_directory() }}/config \
        --config-name train.yaml \
        experiment=yaak/control_transformer/pretrain \
        datamodule=yaak/train_debug \
        ++model.encoder.disable=true \
        {{ ARGS }}

train-action *ARGS: generate-config
    uv run rmind-train \
          --config-path {{ justfile_directory() }}/config \
          --config-name train.yaml \
          experiment=yaak/action_tokenizer/pretrain \
          datamodule=yaak/action_train \
          {{ ARGS }}

predict +ARGS: generate-config
    uv run \
        --extra train --extra predict \
        rmind-predict \
        --config-path {{ justfile_directory() }}/config \
        --config-name predict.yaml \
        {{ ARGS }}

predict-policy-with-permutations +ARGS: generate-config
    uv run \
        --extra train --extra predict \
        rmind-predict \
        --config-path {{ justfile_directory() }}/config \
        --config-name predict.yaml \
        --multirun \
        inference=yaak/control_transformer/policy_with_features_permutation \
        permutation=baseline,speed,cam_front_left,waypoints,all_observations \
        {{ ARGS }}

test *ARGS: generate-config
    uv run \
        --all-extras --group test \
        pytest --capture=no -v {{ ARGS }}

# refresh recorded test snapshots (e.g. training_step_losses.json)
update-snapshots:
    uv run python -m tests.scripts.update_snapshots

export-onnx *ARGS: generate-config
    uv run \
        --extra export \
        rmind-export-onnx \
        --config-path {{ justfile_directory() }}/config \
        --config-name export_onnx.yaml \
        {{ ARGS }}

convert-models model engine:
    trtexec \
        --onnx="{{ model }}" \
        --saveEngine="{{ engine }}" \
        --skipInference \
        --avgTiming=16 \
        --memPoolSize=workspace:24G \
        --builderOptimizationLevel=5

onnxvis *ARGS:
    uvx --python 3.12 --with=ai-edge-model-explorer --from=model-explorer-onnx onnxvis {{ ARGS }}

# start rerun server and viewer
rerun *ARGS:
    uv run rerun --serve-web {{ ARGS }}

clean:
    rm -rf dist outputs lightning_logs wandb artifacts
