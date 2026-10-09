# ruff: noqa: PLR2004, RUF069, ARG001, ARG002, PLR6301, SLF001, S404, PLC0415
"""Bimanual nero patch runs (WP4): split, configs, stats, frame cache, arm selection.

What this pins:

* the generated split lib is exactly the committed split JSON, the JSON is a byte
  copy of nutron-cli's (when a checkout is reachable), and the split is a
  disjoint, class-stratified take split with both one-arm classes in val;
* the bimanual experiments compose, select the bimanual datasets/datamodules
  (old single-arm datasets untouched), and every `_target_` they name imports;
* `nero_fit_stats` pools `hand.{left,right}.*` into one hand block, reports per
  side, fits 26-d stats with side_valid [T, T], and refuses a split without hand
  rows;
* the frame cache is byte-identical to the rbyte decode path and refuses a
  stale or unfinished cache;
* the arm-selection metric (a port of nutron-cli's) and its callback plumbing;
* with the local corpus present: lr_total_steps == len(train_dataloader) x
  max_epochs, and (with stats + tokenizer) a hand_off forward/backward on real
  bimanual windows.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from collections import UserDict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from rmind.data.nero_bimanual import (
    ARM_MOVE_RAD,
    INPUT_ID,
    arm_selection,
    flat_rbyte_batch_with_ids,
    side_excursion,
)
from rmind.datamodules.nero_robot_random import nero_robot_batch
from rmind.scripts import nero_fit_stats, nero_split_lib

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"
CORPUS = Path(
    os.environ.get(
        "NERO_ROBOT_DIR", Path.home() / "data/nero-arms/cube-bimanual/2026-10-07"
    )
)
HAS_CORPUS = (CORPUS / "2026-10-07--14-16-03-494345" / "data.mcap").is_file()
needs_corpus = pytest.mark.skipif(
    not HAS_CORPUS, reason=f"no bimanual corpus at {CORPUS}"
)
EXPERIMENTS = ("bimanual_causal", "bimanual_hand_off", "bimanual_tokenizer")


def _compose(experiment: str, overrides: list[str] | None = None) -> Any:
    from rmind.scripts.nero_steps import compose

    return compose(f"yaak/nero_robot/{experiment}", overrides)


# --------------------------------------------------------------------- split


def test_split_lib_is_generated_from_the_json() -> None:
    assert nero_split_lib.SPLIT_LIB.read_text() == nero_split_lib.render(
        nero_split_lib.SPLIT_JSON
    ), "stale lib: python -m rmind.scripts.nero_split_lib"


def test_split_is_disjoint_and_stratified() -> None:
    split = nero_split_lib.load_split()
    train, val, takes = split["train"], split["val"], split["takes"]
    assert not set(train) & set(val)
    assert (len(train), len(val)) == (76, 9)
    val_classes = [takes[t]["class"] for t in val]
    for c in ("left", "right", "both"):
        assert val_classes.count(c) >= 2, (c, val_classes)
    assert all(t.startswith("2026-10-07--") for t in takes)


def test_split_json_matches_nutron_cli() -> None:
    candidates = [
        os.environ.get("NUTRON_CLI_ROOT"),
        REPO.parent / "nutron-train",  # the workflow lane next to this checkout
        Path.home() / "Code" / "nutron-cli",
    ]
    for root in candidates:
        if root and (source := nero_split_lib.nutron_cli_split(root)) is not None:
            assert source.read_bytes() == nero_split_lib.SPLIT_JSON.read_bytes(), source
            return
    pytest.skip("no nutron-cli checkout with the bimanual split (set NUTRON_CLI_ROOT)")


def test_split_loader_rejects_overlap(tmp_path: Path) -> None:
    split = json.loads(nero_split_lib.SPLIT_JSON.read_text())
    split["val"].append(split["train"][0])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="share takes"):
        nero_split_lib.load_split(bad)


# ------------------------------------------------------------------- configs


@pytest.fixture(scope="module")
def generated_config() -> None:
    if not (CONFIG / "dataset" / "nero" / "robot_bimanual_train.yaml").is_file():
        if shutil.which("ytt") is None:
            pytest.skip(
                "generated dataset configs missing and no ytt (just generate-config)"
            )
        subprocess.run(["just", "generate-config"], cwd=REPO, check=True)  # noqa: S607


def _targets(node: Any) -> set[str]:
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(node, (DictConfig, ListConfig)):
        node = OmegaConf.to_container(node, resolve=False)
    out: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_target_":
                out.add(str(value))
            else:
                out |= _targets(value)
    elif isinstance(node, list):
        for item in node:
            out |= _targets(item)
    return out


def _import(target: str) -> Any:
    module, _, name = target.rpartition(".")
    try:
        return getattr(importlib.import_module(module), name)
    except ModuleNotFoundError:  # a nested attribute (Class.method)
        parent, _, attr = module.rpartition(".")
        return getattr(getattr(importlib.import_module(parent), attr), name)


@pytest.mark.usefixtures("generated_config")
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_bimanual_experiment_composes(experiment: str) -> None:
    cfg = _compose(experiment)
    split = nero_split_lib.load_split()
    dm = cfg.datamodule
    assert list(dm.train.dataset.samples.inputs.input_id) == split["train"]
    assert list(dm.val.dataset.samples.inputs.input_id) == split["val"]
    assert cfg.relative_mode == "all"
    assert cfg.wandb.mode == "disabled"
    assert cfg.trainer.logger._target_ == "pytorch_lightning.loggers.CSVLogger"
    assert "log_model" not in cfg.trainer.logger
    assert dm.val.collate_fn._target_.endswith("flat_rbyte_batch_with_ids")
    targets = _targets(cfg.datamodule) | _targets(cfg.trainer)
    for target in sorted(targets):
        assert _import(target) is not None, target
    if experiment == "bimanual_tokenizer":
        assert dm.train.dataset.streams is None
        return
    source = next(iter(dm.train.dataset.streams["image.base"].sources.values()))
    assert source._target_ == "rmind.data.nero_frame_cache.NeroFrameCacheSource"
    callbacks = [c._target_ for c in cfg.trainer.callbacks]
    assert "rmind.callbacks.nero_arm_selection.NeroArmSelectionLogger" in callbacks
    assert cfg.use_hand_token == (0 if experiment == "bimanual_hand_off" else 1)


@pytest.mark.usefixtures("generated_config")
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_bimanual_stats_dir_defaults_to_the_bimanual_fit(
    experiment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without NERO_STATS_DIR a bimanual run must not pick up the single-arm
    `.nero_stats` (side_valid [T, F]: right side mean 0 / std 1, no error)."""
    monkeypatch.delenv("NERO_STATS_DIR", raising=False)
    cfg = _compose(experiment)
    assert cfg.nero_stats_dir == str(
        Path.home() / "data/nero-arms/cube-bimanual/rmind/stats_v1"
    )
    assert _compose("causal").nero_stats_dir == ".nero_stats"


@pytest.mark.usefixtures("generated_config")
def test_single_arm_datasets_unchanged() -> None:
    cfg = _compose("causal")
    source = next(
        iter(cfg.datamodule.train.dataset.streams["image.base"].sources.values())
    )
    assert source._target_ == "rbyte.streams.transformed.TransformedSource"
    assert next(iter(cfg.datamodule.train.dataset.samples.inputs.input_id)).startswith(
        "2026-10-02"
    )


@pytest.mark.usefixtures("generated_config")
def test_lr_total_steps_consistent_across_runs() -> None:
    on, off = _compose("bimanual_causal"), _compose("bimanual_hand_off")
    assert on.lr_total_steps == off.lr_total_steps
    assert (
        on.batch_size,
        on.episode_length,
        on.episode_stride,
        on.trainer.max_epochs,
    ) == (
        off.batch_size,
        off.episode_length,
        off.episode_stride,
        off.trainer.max_epochs,
    )


@needs_corpus
@pytest.mark.usefixtures("generated_config")
@pytest.mark.parametrize(
    "experiment",
    ["bimanual_hand_off", "bimanual_tokenizer", "bimanual_w1", "bimanual_w1_hand_off"],
)
def test_lr_total_steps_is_the_real_step_count(experiment: str) -> None:
    from rmind.scripts.nero_steps import step_count

    count = step_count(f"yaak/nero_robot/{experiment}")
    assert count["configured"] == count["steps"], count


# --------------------------------------------------------------------- stats


def bimanual_batch(seed: int = 0, **kwargs: Any) -> dict[str, Any]:
    """Two single-arm synthetic batches as one bimanual (rbyte nero-bimanual-26) batch."""
    left = nero_robot_batch(seed=seed, images=False, **kwargs)
    right = nero_robot_batch(seed=seed + 1000, images=False, **kwargs)
    out = {k: v for k, v in left.items() if not k.startswith("hand.")}
    out["state"] = left["state"].clone()
    out["state"][:, :, 1] = right["state"][:, :, 0]
    out["action.chunk"] = left["action.chunk"].clone()
    out["action.chunk"][..., 1, :] = right["action.chunk"][..., 0, :]
    out["action.is_pad"] = left["action.is_pad"] | right["action.is_pad"]
    out["side_valid"] = torch.ones_like(left["side_valid"])
    for side, batch in (("left", left), ("right", right)):
        for key, value in batch.items():
            if key.startswith("hand."):
                out[f"hand.{side}.{key.removeprefix('hand.')}"] = value
    return out


def test_fit_stats_pools_both_hands() -> None:
    batches = [bimanual_batch(seed=i, batch_size=8, num_frames=8) for i in range(4)]
    result = nero_fit_stats.fit(batches, hand_min_rows=50)
    report = result["report"]["hand"]
    assert report["source"] == "train:hand"
    per_side = report["per_side"]
    assert set(per_side) == {"left", "right"}
    assert per_side["left"]["rows"] == per_side["right"]["rows"] == 4 * 8 * 8
    assert report["rows_seen"] == per_side["left"]["rows"] + per_side["right"]["rows"]
    assert report["motor_rows"] == (
        per_side["left"]["motor_ok_rows"] + per_side["right"]["motor_ok_rows"]
    )
    stats = result["absolute_action_stats"]
    assert stats["side_valid"] == [True, True]
    assert len(stats["min"]) == len(stats["max"]) == len(stats["q50"]) == 26
    # the right side is fitted from right-arm data, not left as mean0/std1
    state = result["state"]
    assert not torch.allclose(state.std[1], torch.ones_like(state.std[1]))


def test_fit_stats_hand_pool_is_left_then_right() -> None:
    batch = bimanual_batch(batch_size=4, num_frames=4)
    swapped = dict(batch)
    for key in [k for k in batch if k.startswith("hand.left.")]:
        swapped[key] = batch[key.replace("left", "right")]
        swapped[key.replace("left", "right")] = batch[key]
    a = nero_fit_stats.fit([batch], hand_min_rows=10)["hand"]
    b = nero_fit_stats.fit([swapped], hand_min_rows=10)["hand"]
    # pooled: the side order does not change the fitted affine
    torch.testing.assert_close(a.mean, b.mean)
    torch.testing.assert_close(a.std, b.std)


def test_fit_stats_single_arm_path_unchanged() -> None:
    batches = [
        nero_robot_batch(seed=i, batch_size=8, num_frames=8, images=False)
        for i in range(2)
    ]
    report = nero_fit_stats.fit(batches, hand_min_rows=50)["report"]["hand"]
    assert set(report["per_side"]) == {"unsided"}
    assert report["rows_seen"] == 2 * 8 * 8


def test_fit_stats_refuses_a_split_without_hand_rows() -> None:
    batch = {k: v for k, v in bimanual_batch().items() if not k.startswith("hand.")}
    with pytest.raises(ValueError, match="no hand rows"):
        nero_fit_stats.fit([batch])
    result = nero_fit_stats.fit([batch], require_hand=False)
    assert result["report"]["hand"]["source"] == "physical_prior"


# --------------------------------------------------------------- frame cache


CACHE_TAKE = "2026-10-07--14-16-03-494345"


@needs_corpus
def test_frame_cache_is_byte_identical_to_the_decode_path(tmp_path: Path) -> None:
    from rbyte.streams.transformed import TransformedSource
    from rbyte.streams.video import TorchCodecVideoSource

    from rmind.data.nero_frame_cache import (
        NeroFrameCacheSource,
        build_camera_cache,
        cache_paths,
    )
    from rmind.data.nero_image import NeroImagePreprocess

    video = CORPUS / CACHE_TAKE / "side_right.mp4"
    npy, manifest = cache_paths(tmp_path, CACHE_TAKE, "side_right")
    built = build_camera_cache(video, npy, input_hw=(140, 224), num_threads=4)
    cached = NeroFrameCacheSource(path=npy, input_hw=[140, 224], video=video)
    decoded = TransformedSource(
        source=TorchCodecVideoSource(source=video, num_ffmpeg_threads=4),
        transform=NeroImagePreprocess((140, 224)),
    )
    n = len(cached)
    assert n == built["num_frames"]
    idx = sorted({0, 1, n // 3, n // 2, n - 2, n - 1})
    assert torch.equal(cached[idx], decoded[idx])
    assert torch.equal(cached[idx[2]], decoded[idx[2]])
    assert cached[idx].dtype == torch.uint8

    # stale: another grid, another mp4, or no manifest (unfinished build)
    with pytest.raises(ValueError, match="input_hw"):
        NeroFrameCacheSource(path=npy, input_hw=[126, 224], video=video)
    with pytest.raises(ValueError, match="different"):
        NeroFrameCacheSource(
            path=npy, input_hw=[140, 224], video=CORPUS / CACHE_TAKE / "base.mp4"
        )
    manifest.unlink()
    with pytest.raises(ValueError, match="no cache"):
        NeroFrameCacheSource(path=npy, input_hw=[140, 224])


def test_frame_cache_refuses_other_preprocessing(tmp_path: Path) -> None:
    from rmind.data.nero_frame_cache import MANIFEST_SCHEMA, NeroFrameCacheSource

    npy = tmp_path / "base.npy"
    np.save(npy, np.zeros((3, 3, 140, 224), dtype=np.uint8))
    npy.with_suffix(".json").write_text(
        json.dumps({
            "schema": MANIFEST_SCHEMA,
            "preprocessing_sha256": "0" * 64,
            "input_hw": [140, 224],
            "num_frames": 3,
        })
    )
    with pytest.raises(ValueError, match="preprocessing_sha256"):
        NeroFrameCacheSource(path=npy, input_hw=[140, 224])


# ------------------------------------------------------------- arm selection


def test_side_excursion_uses_valid_steps_and_the_anchor() -> None:
    x = np.zeros((2, 4, 13))
    x[0, 1, 2] = 0.3
    x[0, 3, 0] = 9.0  # beyond the valid steps
    x[1, :, 7] = 5.0  # a finger, not an arm joint
    valid = np.array([[True, True, True, False], [False] * 4])
    exc = side_excursion(x, np.zeros((2, 13)), valid, slice(0, 7))
    assert exc[0] == pytest.approx(0.3)
    assert np.isnan(exc[1])


def test_arm_selection_per_class() -> None:
    # 3 takes: left (policy right), right (policy wrong), both (both move)
    take = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4, dtype=object)
    cls = np.array(["left"] * 4 + ["right"] * 4 + ["both"] * 4, dtype=object)
    demo = np.array([[0.5, 0.01]] * 4 + [[0.01, 0.5]] * 4 + [[0.4, 0.4]] * 4)
    pred = np.array([[0.4, 0.02]] * 4 + [[0.4, 0.02]] * 4 + [[0.3, 0.2]] * 4)
    out = arm_selection("val/", ("left", "right"), pred, demo, take, cls)
    assert out["val/arm_select/left/correct_side_rate"] == 1.0
    assert out["val/arm_select/right/correct_side_rate"] == 0.0
    assert out["val/arm_select/left/take_correct_rate"] == 1.0
    assert out["val/arm_select/right/take_correct_rate"] == 0.0
    assert out["val/arm_select/both/both_move_rate"] == 1.0
    assert out["val/arm_select/correct_side_rate"] == 0.5
    assert out["val/arm_select/left/idle_exc_ratio"] == pytest.approx(2.0)
    assert out["val/arm_select/right/idle_exc_ratio"] == pytest.approx(40.0)
    assert ARM_MOVE_RAD == 0.05


def test_arm_selection_matches_nutron_cli() -> None:
    """Same numbers as the ACT family's implementation, when it is importable."""
    roots = [os.environ.get("NUTRON_CLI_ROOT"), REPO.parent / "nutron-train"]
    path = next(
        (
            Path(r) / "runtime/training/nutron_act/metrics.py"
            for r in roots
            if r and (Path(r) / "runtime/training/nutron_act/metrics.py").is_file()
        ),
        None,
    )
    if path is None:
        pytest.skip("no nutron-cli checkout with nutron_act metrics")
    import ast

    tree = ast.parse(path.read_text())
    keep = {
        "side_excursion",
        "arm_selection",
        "_nanmean",
        "ARM_MOVE_RAD",
        "ARM_DECISIVE_FACTOR",
    }
    body: list[ast.stmt] = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in keep)
        or (
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) in keep for t in n.targets)
        )
    ]
    ns: dict[str, Any] = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    rng = np.random.default_rng(0)
    n = 300
    take = np.array([f"t{i // 20}" for i in range(n)], dtype=object)
    cls = np.array(
        [("left", "right", "both")[(i // 20) % 3] for i in range(n)], dtype=object
    )
    pred, demo = rng.gamma(1.0, 0.1, (n, 2)), rng.gamma(1.0, 0.1, (n, 2))
    pred[rng.random((n, 2)) < 0.05] = np.nan
    ours = arm_selection("val/", ("left", "right"), pred, demo, take, cls)
    theirs = ns["arm_selection"]("val/", ("left", "right"), pred, demo, take, cls)
    assert ours.keys() == theirs.keys()
    for key, value in ours.items():
        assert value == pytest.approx(theirs[key], nan_ok=True), key


class _Meta(UserDict):
    pass


class _Batch:
    def __init__(self, data: dict[str, Any], ids: list[str]) -> None:
        self.data = data
        self.meta = _Meta({INPUT_ID: ids})


def test_val_collate_carries_take_ids() -> None:
    out = flat_rbyte_batch_with_ids(_Batch({"state": torch.zeros(2, 1)}, ["a", "b"]))
    assert out[INPUT_ID] == ["a", "b"]
    assert set(out) == {"state", INPUT_ID}


class _StubPolicy:
    """The four NeroPatchPolicy helpers the callback reads, on a fixed batch."""

    robot = True

    def __init__(self, b: int, t: int, horizon: int = 100) -> None:
        self.b, self.t, self.h = b, t, horizon

    def _features(self, batch: Any) -> torch.Tensor:
        return torch.zeros(self.b, self.t, 4)

    def _robot_rows(
        self, batch: Any, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        n = self.b * self.t * 2
        side = torch.arange(2).repeat(self.b * self.t)
        target = torch.zeros(n, self.h, 13)
        target[side == 0, :, 0] = torch.linspace(0, 1, self.h)  # demo: left moves 1 rad
        logits = torch.zeros(n, 2, 3)
        return {
            "row_valid": torch.ones(n, dtype=torch.bool),
            "side": side,
            "target": target,
            "real": torch.ones(n, self.h, dtype=torch.bool),
            "code_logits": logits,
            "offsets": torch.zeros(n, 1),
        }

    def _decode(self, offsets: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(offsets.shape[0], self.h, 13)
        out[1::2, :, 3] = 0.2  # the policy moves the RIGHT arm
        return out

    def rows_prediction(self, rows: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._decode(rows["offsets"], rows["code_logits"].argmax(dim=-1))

    def _anchor_rows(self, batch: Any, row_valid: torch.Tensor) -> torch.Tensor:
        return torch.zeros(int(row_valid.sum()), 13)

    def _absolute(
        self, chunk: torch.Tensor, anchor: torch.Tensor, side: torch.Tensor
    ) -> torch.Tensor:
        return chunk


def test_arm_selection_callback_rows(tmp_path: Path) -> None:
    from rmind.callbacks.nero_arm_selection import NeroArmSelectionLogger

    split = json.loads(nero_split_lib.SPLIT_JSON.read_text())
    left_take = next(t for t in split["val"] if split["takes"][t]["class"] == "left")
    cb = NeroArmSelectionLogger(exec_steps=50)
    pred, demo = cb.excursions(_StubPolicy(b=2, t=3), {})
    assert pred.shape == demo.shape == (6, 2)
    # executed horizon: 50 of 100 steps of a 0..1 ramp
    np.testing.assert_allclose(demo[:, 0], 49 / 99, rtol=1e-5)
    np.testing.assert_allclose(demo[:, 1], 0.0)
    np.testing.assert_allclose(pred[:, 1], 0.2, rtol=1e-5)
    cb._pred, cb._demo = [pred], [demo]
    cb._take = [np.array([left_take] * 6, dtype=object)]
    out = cb.compute()
    assert out["val/arm_select/left/correct_side_rate"] == 0.0
    assert out["val/arm_select/left/frames"] == 6


class _Trainer:
    def __init__(self, epoch: int, max_epochs: int, *, sanity: bool = False) -> None:
        self.current_epoch, self.max_epochs, self.sanity_checking = (
            epoch,
            max_epochs,
            sanity,
        )


def test_arm_selection_every_n_epochs() -> None:
    from rmind.callbacks.nero_arm_selection import NeroArmSelectionLogger

    cb = NeroArmSelectionLogger(every_n_epochs=3)
    ran = [e for e in range(10) if cb.active(_Trainer(e, 10))]  # ty: ignore[invalid-argument-type]
    assert ran == [2, 5, 8, 9]  # every 3rd, plus the last
    unbounded = [e for e in range(10) if cb.active(_Trainer(e, -1))]  # ty: ignore[invalid-argument-type]
    assert unbounded == [2, 5, 8]
    assert not cb.active(_Trainer(2, 10, sanity=True))  # ty: ignore[invalid-argument-type]
    assert all(
        NeroArmSelectionLogger().active(_Trainer(e, 10))  # ty: ignore[invalid-argument-type]
        for e in range(10)
    )
    with pytest.raises(ValueError, match="every_n_epochs"):
        NeroArmSelectionLogger(every_n_epochs=0)


# ---------------------------------------------------- real-window smoke (GPU)


def _first_batch(loader: Any) -> Any:
    """The loader's first batch, then its torchdata worker/pin/prefetch threads
    stopped and joined. Left running, a thread killed mid-decode at interpreter
    teardown aborts the process ('terminate called without an active
    exception') after pytest has already reported success."""
    import gc

    it = iter(loader)
    try:
        return next(it)
    finally:
        seen: set[int] = set()
        stack: list[Any] = [it, getattr(loader, "_loader", None)]
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            shutdown = getattr(node, "_shutdown", None)
            if callable(shutdown):
                shutdown()
            stack.extend(
                getattr(node, attr, None)
                for attr in ("_it", "root", "source", "_root", "_source", "loader")
            )
        del it
        gc.collect()


@needs_corpus
@pytest.mark.usefixtures("generated_config")
@pytest.mark.skipif(
    not (os.environ.get("NERO_STATS_DIR") and os.environ.get("NERO_TOKENIZER_CKPT")),
    reason="needs NERO_STATS_DIR and NERO_TOKENIZER_CKPT (bimanual stats + tokenizer)",
)
def test_hand_off_forward_backward_on_real_bimanual_windows() -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from rmind.callbacks.nero_arm_selection import NeroArmSelectionLogger

    cfg = _compose(
        "bimanual_hand_off", ["batch_size=2", "datamodule=yaak/nero_robot_bimanual"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    loader = instantiate(cfg.datamodule.val)
    batch = _first_batch(loader)
    takes = batch[INPUT_ID]
    assert len(takes) == 2
    assert all(t in nero_split_lib.load_split()["val"] for t in takes)
    batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    assert bool(batch["side_valid"].all())
    model.train()
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        metrics = model.compute_metrics(batch)
        loss = metrics["policy", "loss"].sum(reduce=True)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    model.eval()
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        pred, demo = NeroArmSelectionLogger().excursions(model, batch)
    assert pred.shape == demo.shape == (2 * cfg.episode_length, 2)
    assert np.isfinite(demo).all()


@needs_corpus
@pytest.mark.usefixtures("generated_config")
@pytest.mark.skipif(
    not (os.environ.get("NERO_STATS_DIR") and os.environ.get("NERO_TOKENIZER_CKPT")),
    reason="needs NERO_STATS_DIR and NERO_TOKENIZER_CKPT (bimanual stats + tokenizer)",
)
def test_hand_on_reads_both_hands_on_real_bimanual_windows() -> None:
    """WP5: bimanual_causal builds two hand tokens per frame from the real
    `hand.{left,right}.*` columns (no no_hand-everywhere fallback), and the
    loss reaches both sides' hand path."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = _compose(
        "bimanual_causal", ["batch_size=2", "datamodule=yaak/nero_robot_bimanual"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    assert model.hand_sides == ("left", "right")
    assert model.tokens_per_frame() == 483
    batch = _first_batch(instantiate(cfg.datamodule.val))
    batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    vec = model.hand_vector(batch)
    assert vec is not None
    assert vec.shape == (2, cfg.episode_length, 2, 14)
    for side in range(2):
        assert float(vec[..., side, -1].mean()) > 0.5  # ~0.9 valid
    model.train()
    norms: dict[str, torch.Tensor] = {}
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        loss = model.compute_metrics(batch, token_norms=norms)["policy", "loss"].sum(
            reduce=True
        )
    assert torch.isfinite(loss)
    loss.backward()
    grad = model.hand_side_embedding.weight.grad
    assert grad is not None
    assert (grad.abs().sum(dim=-1) > 0).all()
    assert {"hand_valid_frac/left", "hand_valid_frac/right"} <= set(norms)


@needs_corpus
@pytest.mark.usefixtures("generated_config")
def test_real_val_batches_carry_take_ids() -> None:
    from hydra.utils import instantiate

    cfg = _compose("bimanual_tokenizer", ["batch_size=8"])
    batch = _first_batch(instantiate(cfg.datamodule.val))
    assert len(batch[INPUT_ID]) == 8
    assert set(batch[INPUT_ID]) <= set(nero_split_lib.load_split()["val"])
    assert bool(batch["side_valid"].all())
    for side in ("left", "right"):
        assert batch[f"hand.{side}.motor_ok"].shape == batch["state"].shape[:2]
