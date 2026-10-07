"""Fit the robot-native nero standardizers on the TRAIN split (P5, contract v3).

    python -m rmind.scripts.nero_fit_stats --experiment yaak/nero_robot/tokenizer --out STATS
    python -m rmind.scripts.nero_fit_stats --synthetic --out STATS   # smoke

Iterates `datamodule.train` of the experiment (only the TRAIN split: the split
is by source episode in `config/_templates/dataset/nero/robot_split.lib.yml`, or
`robot_bimanual_split.lib.yml` for the `bimanual_*` experiments)
and writes, all `nutron_standardizer` v1 JSON unless noted:

* `state_standardizer.json`: per-axis over every frame's raw state (measured q +
  hand_prev);
* `action_standardizer_{none,hand,all}.json`: per-axis over every REAL
  (non-padded) chunk step, in that relative mode's space (each frame anchored
  at its own state) -- one per mode, because the spreads differ;
* `event_reference_{none,hand,all}.json`: the per-axis atom of that mode's
  STANDARDIZED real chunk steps (the exact mode when it holds >= 50% of the
  steps, else the median; `EventReference`), pinned to the standardizer's SHA256 -- the
  chunk tokenizer's event-weighting reference. Deterministic: the whole train
  split, never a batch (the old first-batch median moved with the batch);
* `absolute_action_stats.json`: `{min, max, q50}` per contract action axis over
  the ABSOLUTE real chunk steps (serving's range rail / hand seed), plus
  `side_valid`;
* `hand_standardizer.json` (`nero_hand_token_standardizer` v1): the hand
  token's in-graph per-column affine, fitted on the train split's VALID hand
  rows (motor_ok for current/pos_err/pos, tip_ok for tip) when a group has at
  least `--hand-min-rows` of them, else the documented physical band
  (`HAND_PHYSICAL_PRIOR`); `source` in the file and `hand` in the report say
  which. hand_age / hand_valid are never scaled;
* `fit_report.json`: counts and the train episode ids seen.

BIMANUAL takes (rbyte `nero-bimanual-26`) carry the hand blocks per side as
`hand.{left,right}.*`. Both sides' rows are stacked into ONE pooled block (the
hand standardizer is shared by the two side-tagged hand tokens; its API is
unchanged), and the report's `hand.per_side` gives each side's rows and valid
motor rows. A single-arm take's unsided `hand.*` blocks are read as before. A
fit that finds NO hand rows at all fails (`--allow-no-hand` to accept the
physical prior anyway): it means the loader's hand columns were not the ones
read here, and a silently prior-only standardizer would be shipped as fitted.

The degenerate-axis rule (train std < 1e-6 -> std 1, mean = the constant; an
invalid side -> mean 0, std 1) is `AxisStandardizer.fit`'s.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rmind.data.nero_robot import (
    HAND_GROUPS,
    RELATIVE_MODES,
    SIDES,
    AxisStandardizer,
    EventReference,
    HandTokenStandardizer,
    to_relative,
)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def batches_from_experiment(experiment: str, overrides: list[str]) -> Any:
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from hydra.utils import instantiate  # noqa: PLC0415

    import rmind  # noqa: F401, PLC0415  (registers the `eval` resolver)

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        # every train sample exactly once, in a fixed order: a shuffled loader
        # with drop_last drops a RANDOM remainder each pass, so the fit (and the
        # event reference) would move from run to run
        fixed = ["datamodule.train.shuffle=false", "datamodule.train.drop_last=false"]
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", *overrides, *fixed],
        )
    return instantiate(cfg.datamodule).train_dataloader()


def _get(batch: Any, key: str) -> Any:
    try:
        return batch[key]
    except KeyError:
        return None


def hand_prefixes(batch: Any) -> list[str]:
    """The hand block prefixes a batch carries: per side if bimanual, else `hand.`."""
    sided = [
        f"hand.{side}."
        for side in SIDES
        if _get(batch, f"hand.{side}.motor_ok") is not None
    ]
    if sided:
        return sided
    return ["hand."] if _get(batch, "hand.motor_ok") is not None else []


def fit(  # noqa: C901, PLR0914, PLR0915
    batches: Any,
    *,
    max_batches: int | None = None,
    hand_min_rows: int = 1000,
    require_hand: bool = True,
) -> dict[str, Any]:
    states, valids_state = [], []
    hand: dict[str, list[torch.Tensor]] = {
        g: [] for g in (*HAND_GROUPS, "motor_ok", "tip_ok")
    }
    hand_seen: dict[str, dict[str, int]] = {}
    rel: dict[str, list[torch.Tensor]] = {m: [] for m in RELATIVE_MODES}
    valids_chunk = []
    seen = 0
    side_valid_any = None
    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        state = batch["state"].float()  # (b, T, S, A)
        chunk = batch["action.chunk"].float()  # (b, T, H, S, A)
        valid = batch["side_valid"].bool()  # (b, S)
        real = ~batch["action.is_pad"].bool()  # (b, T, H)
        b, t, h, s, a = chunk.shape
        states.append(state.reshape(-1, s, a))
        valids_state.append(valid[:, None].expand(b, t, s).reshape(-1, s))
        for mode in RELATIVE_MODES:
            r = to_relative(chunk, state, mode)[real]  # (m, S, A)
            rel[mode].append(r)
        valids_chunk.append(valid[:, None, None].expand(b, t, h, s)[real])
        # bimanual: left rows then right rows, stacked into one pooled block
        for prefix in hand_prefixes(batch):
            for key, rows in hand.items():
                value = _get(batch, f"{prefix}{key}")
                if value is not None:
                    width = value.shape[-1] if key in HAND_GROUPS else 1
                    rows.append(value.reshape(-1, width))
            ok = _get(batch, f"{prefix}motor_ok").reshape(-1).bool()
            side = prefix.removeprefix("hand.").rstrip(".") or "unsided"
            seen_side = hand_seen.setdefault(side, {"rows": 0, "motor_ok_rows": 0})
            seen_side["rows"] += int(ok.numel())
            seen_side["motor_ok_rows"] += int(ok.sum())
        side_valid_any = (
            valid.any(0) if side_valid_any is None else side_valid_any | valid.any(0)
        )
        seen += b
    if not states:
        msg = "no batches"
        raise ValueError(msg)
    state_all = torch.cat(states)
    state_valid = torch.cat(valids_state)
    chunk_valid = torch.cat(valids_chunk)
    out: dict[str, Any] = {
        "state": AxisStandardizer.fit(state_all, state_valid, source="train:state"),
        "action": {},
        "event_reference": {},
    }
    for mode, rows in rel.items():
        values = torch.cat(rows)
        std = AxisStandardizer.fit(values, chunk_valid, source=f"train:action:{mode}")
        out["action"][mode] = std
        out["event_reference"][mode] = EventReference.fit(
            std(values), chunk_valid, relative_mode=mode, standardizer_sha256=std.digest
        )
    hand_blocks = {g: torch.cat(v) for g, v in hand.items() if v and g in HAND_GROUPS}
    motor_ok = torch.cat(hand["motor_ok"]) if hand["motor_ok"] else torch.zeros(0)
    tip_ok = torch.cat(hand["tip_ok"]) if hand["tip_ok"] else None
    out["hand"], hand_report = HandTokenStandardizer.fit(
        hand_blocks, motor_ok, tip_ok, min_rows=hand_min_rows
    )
    # rows SEEN (refused or not): 0 here means the loader carried no hand
    # stream at all, not "every reading refused"
    hand_report["rows_seen"] = int(motor_ok.numel())
    hand_report["per_side"] = hand_seen
    if require_hand and hand_report["rows_seen"] == 0:
        msg = (
            "no hand rows in the train split (no hand.motor_ok / "
            "hand.{left,right}.motor_ok columns): the hand standardizer would be the "
            "physical prior. Pass --allow-no-hand if that is intended."
        )
        raise ValueError(msg)
    absolute = torch.cat(rel["none"])  # (m, S, A) absolute chunk steps
    flat = absolute.reshape(absolute.shape[0], -1)  # side-major 26
    flat_valid = chunk_valid[:, :, None].expand_as(absolute).reshape(flat.shape)
    stats = {"min": [], "max": [], "q50": []}
    for j in range(flat.shape[1]):
        col = flat[flat_valid[:, j], j]
        if col.numel() == 0:
            stats["min"].append(0.0)
            stats["max"].append(0.0)
            stats["q50"].append(0.0)
            continue
        stats["min"].append(float(col.min()))
        stats["max"].append(float(col.max()))
        stats["q50"].append(float(col.median()))
    out["absolute_action_stats"] = {
        "min": stats["min"],
        "max": stats["max"],
        "q50": stats["q50"],
        "side_valid": [bool(v) for v in side_valid_any.tolist()],
    }
    out["report"] = {
        "samples": seen,
        "state_rows": int(state_all.shape[0]),
        "chunk_steps": int(absolute.shape[0]),
        "hand": hand_report,
    }
    return out


def write(result: dict[str, Any], out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    digests = {
        "state_standardizer.json": result["state"].save(out / "state_standardizer.json")
    }
    digests["hand_standardizer.json"] = result["hand"].save(
        out / "hand_standardizer.json"
    )
    for mode, std in result["action"].items():
        name = f"action_standardizer_{mode}.json"
        digests[name] = std.save(out / name)
    for mode, ref in result["event_reference"].items():
        name = f"event_reference_{mode}.json"
        digests[name] = ref.save(out / name)
    (out / "absolute_action_stats.json").write_text(
        json.dumps(result["absolute_action_stats"], indent=1) + "\n"
    )
    (out / "fit_report.json").write_text(
        json.dumps(result["report"] | {"sha256": digests}, indent=1) + "\n"
    )
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--allow-no-hand",
        action="store_true",
        help="accept a train split without hand rows (prior-only hand standardizer)",
    )
    parser.add_argument(
        "--hand-min-rows",
        type=int,
        default=1000,
        help="valid train hand rows a group needs before it is fitted (else physical band)",
    )
    args = parser.parse_args()
    if args.synthetic:
        from rmind.datamodules.nero_robot_random import (  # noqa: PLC0415
            NeroRobotRandomDataLoader,
        )

        batches = NeroRobotRandomDataLoader(
            num_batches=args.max_batches or 32, batch_size=8, num_frames=8, images=False
        )
    else:
        if args.max_batches is not None:
            # a prefix of the (unshuffled) split: deterministic but NOT the
            # train split's statistics -- smokes only
            print(  # noqa: T201
                f"WARNING: --max-batches {args.max_batches} fits on a PREFIX of the "
                "train split: the standardizers and the event reference are not "
                "the split's. Use it for smokes only."
            )
        batches = batches_from_experiment(args.experiment, args.override)
    digests = write(
        fit(
            batches,
            max_batches=args.max_batches,
            hand_min_rows=args.hand_min_rows,
            require_hand=not args.allow_no_hand,
        ),
        args.out,
    )
    print(json.dumps(digests, indent=1))  # noqa: T201


if __name__ == "__main__":
    main()
