"""The bimanual nero RUN experiments (docs/nero_runs.md): each composes with no
CLI overrides, reads its data through a `nero_data` profile and its tokenizer
through a `nero_tokenizer` pin, and ignores the legacy per-directory NERO_* vars.
Compose-only (no corpus, no GPU)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess  # noqa: S404
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from rmind.scripts.train import check_pins

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"
NAS_DATA = "/nasa/max/nero-cache/cube-bimanual"
NAS_CKPT = "/nasa/max/nero-sweep-ckpts"
SHA256_HEX = 64
DEFAULT_SEED = 1337
RUN_SEED = 2
CLI_SEED = 5

# run -> (data profile files it must resolve to, tokenizer pin or None)
RUNS: dict[str, tuple[str, str, str | None, str | None]] = {
    # run: (takes, stats, frame cache, tokenizer path under NERO_CKPT_ROOT)
    "bimanual_tokenizer": ("2026-10-07", "stats_v1", None, None),
    "bimanual_tokenizer_e300": ("2026-10-07", "stats_v1", None, None),
    "bimanual_tokenizer_v2": ("2026-10-07", "stats_v1", None, None),
    "bimanual_tokenizer_v2_q32": ("2026-10-07", "stats_v1", None, None),
    "bimanual_tokenizer_v2_ew2": ("2026-10-07", "stats_v1", None, None),
    **dict.fromkeys(
        ("bimanual_hand_off_v2", "bimanual_hand_on_v2"),
        (
            "2026-10-07",
            "stats_v1",
            "frame-cache-140x224/2026-10-07",
            "bimanual_tokenizer/version_1/checkpoints/epoch=19-step=1320.ckpt",
        ),
    ),
    **{
        f"bimanual_hand_{side}_tok{tok}": (
            "2026-10-07",
            "stats_v1",
            "frame-cache-140x224/2026-10-07",
            f"{ckpt}/version_0/checkpoints/epoch=149-step=9900.ckpt",
        )
        for side in ("on", "off")
        for tok, ckpt in (
            ("v2", "bimanual_tokenizer_v2"),
            ("q32", "bimanual_tokenizer_v2_q32"),
        )
    },
    **{
        run: (
            "2026-10-07",
            "stats_v1",
            cache,
            "bimanual_tokenizer_v2/version_0/checkpoints/epoch=149-step=9900.ckpt",
        )
        for run, cache in (
            ("sweep_w1_off", "frame-cache-140x224/2026-10-07"),
            ("sweep_w1_off_drop", "frame-cache-140x224/2026-10-07"),
            ("sweep_w1_off_dinoft", "frame-cache-140x224/2026-10-07"),
            ("sweep_w1_off_r18", "frame-cache-416x640/2026-10-07"),
            ("sweep_w1_off_dinoft336", "frame-cache-210x336/2026-10-07"),
            ("sweep_w1_on_drop", "frame-cache-140x224/2026-10-07"),
            ("l1_codes_w1_drop_seed2", "frame-cache-140x224/2026-10-07"),
            ("l1_w1_drop", "frame-cache-140x224/2026-10-07"),
            ("l1_w1_drop_pad", "frame-cache-140x224/2026-10-07"),
            ("codes_w1_drop_pad", "frame-cache-140x224/2026-10-07"),
            ("l1_w1_r18_pad", "frame-cache-416x640/2026-10-07"),
        )
    },
    "paper_tok_c10": ("2026-10-07", "stats_c10", None, None),
    "paper_tok_c10_q4": ("2026-10-07", "stats_c10", None, None),
    "paper_pp_w2_c10_codes": (
        "2026-10-07",
        "stats_c10",
        "frame-cache-224x224-stretch/2026-10-07",
        "paper_tok_c10/version_0/checkpoints/epoch=399-step=32400.ckpt",
    ),
    "paper_pp_w2_c10_l1": (
        "2026-10-07",
        "stats_c10",
        "frame-cache-224x224-stretch/2026-10-07",
        "paper_tok_c10/version_0/checkpoints/epoch=89-step=7290.ckpt",
    ),
    **dict.fromkeys(
        ("paper2_tok_cb64", "paper2_tok_cb256", "paper2_tok_split", "paper2_tok_q4"),
        ("takes-v3", "stats_c10_v3", None, None),
    ),
    **{
        run: ("takes-v3", "stats_c10_v3", "frame-cache-224x224-stretch/v3", ckpt)
        for run, ckpt in (
            (
                "paper2_pp_l1_handon",
                "paper2_tok_cb64/checkpoints/epoch=9-step=1500.ckpt",
            ),
            (
                "paper2_pp_l1_handoff",
                "paper2_tok_cb64/checkpoints/epoch=9-step=1500.ckpt",
            ),
            (
                "paper2_pp_codes_cond",
                "paper2_tok_cb64/checkpoints/epoch=999-step=150000.ckpt",
            ),
            (
                "paper2_pp_codes_nocond",
                "paper2_tok_cb64/checkpoints/epoch=999-step=150000.ckpt",
            ),
            (
                "paper2_pp_codes_q4_nocond",
                "paper2_tok_q4/checkpoints/epoch=999-step=150000.ckpt",
            ),
        )
    },
}
LEGACY = ("NERO_ROBOT_DIR", "NERO_FRAME_CACHE", "NERO_STATS_DIR", "NERO_TOKENIZER_CKPT")
ROOTS = ("NERO_DATA_ROOT", "NERO_FRAME_CACHE_ROOT", "NERO_STATS_ROOT", "NERO_CKPT_ROOT")


@pytest.fixture(scope="module")
def generated_config() -> None:
    if not (CONFIG / "nero_data" / "cube_v1.yaml").is_file():
        if shutil.which("ytt") is None:
            pytest.skip("generated configs missing and no ytt (just generate-config)")
        subprocess.run(["just", "generate-config"], cwd=REPO, check=True)  # noqa: S607


def _compose(experiment: str) -> Any:
    from rmind.scripts.nero_steps import compose  # noqa: PLC0415

    return compose(f"yaak/nero_robot/{experiment}")


@pytest.mark.usefixtures("generated_config")
@pytest.mark.parametrize("run", sorted(RUNS))
def test_run_experiment_pins_its_inputs(
    run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in LEGACY:  # a stale launch shell must not leak in
        monkeypatch.setenv(var, "/legacy/ignored")
    for var in ROOTS:
        monkeypatch.delenv(var, raising=False)
    takes, stats, cache, tok = RUNS[run]
    cfg = _compose(run)
    assert cfg.run_name == run
    assert cfg.nero_robot_dir == f"{NAS_DATA}/{takes}"
    assert cfg.nero_stats_dir == f"{NAS_DATA}/{stats}"
    if cache is not None:
        assert cfg.nero_frame_cache == f"{NAS_DATA}/{cache}"
    if tok is not None:
        assert cfg.tokenizer_ckpt == f"{NAS_CKPT}/{tok}"
        assert len(cfg.tokenizer_ckpt_sha256) == SHA256_HEX
    if takes == "takes-v3":
        assert cfg.nero_split_file == "config/splits/nero_cube_bimanual_v3.json"
        assert (REPO / cfg.nero_split_file).is_file()
    assert "/legacy/ignored" not in OmegaConf.to_yaml(cfg, resolve=True)


@pytest.mark.usefixtures("generated_config")
def test_roots_relocate_the_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NERO_DATA_ROOT", "/data")
    monkeypatch.setenv("NERO_FRAME_CACHE_ROOT", "/fast/cache")
    monkeypatch.setenv("NERO_CKPT_ROOT", "/ckpt")
    monkeypatch.delenv("NERO_STATS_ROOT", raising=False)
    cfg = _compose("paper2_pp_codes_cond")
    assert cfg.nero_robot_dir == "/data/takes-v3"
    assert cfg.nero_stats_dir == "/data/stats_c10_v3"
    assert cfg.nero_frame_cache == "/fast/cache/frame-cache-224x224-stretch/v3"
    assert cfg.tokenizer_ckpt.startswith("/ckpt/paper2_tok_cb64/")


@pytest.mark.usefixtures("generated_config")
def test_experiment_seed() -> None:
    assert _compose("l1_codes_w1_drop_seed2").seed == RUN_SEED
    assert _compose("sweep_w1_off_drop").seed == DEFAULT_SEED


@pytest.mark.usefixtures("generated_config")
def test_cli_seed_still_wins() -> None:
    from rmind.scripts.nero_steps import compose  # noqa: PLC0415

    assert (
        compose("yaak/nero_robot/l1_codes_w1_drop_seed2", [f"seed={CLI_SEED}"]).seed
        == CLI_SEED
    )


def testcheck_pins(tmp_path: Path) -> None:
    ckpt = tmp_path / "tok.ckpt"
    ckpt.write_bytes(b"tokenizer")
    digest = hashlib.sha256(b"tokenizer").hexdigest()
    check_pins(OmegaConf.create({"tokenizer_ckpt": str(ckpt)}))  # no pin: no check
    check_pins(
        OmegaConf.create({"tokenizer_ckpt": str(ckpt), "tokenizer_ckpt_sha256": digest})
    )
    with pytest.raises(ValueError, match="sha256"):
        check_pins(
            OmegaConf.create({
                "tokenizer_ckpt": str(ckpt),
                "tokenizer_ckpt_sha256": "0" * 64,
            })
        )
