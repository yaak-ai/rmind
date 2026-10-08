"""Robot-native nero action space (P4), relative actions (P5), hand token (P6).

The training-side half of contract v3 (nutron-cli `runtime/jetson/policy_contract.py`
is the serving-side half). Everything a checkpoint needs to agree on with serving
lives here as data: the axis names, the side-major flat layout, the relative mask
and anchor rule, the standardizer JSON format, and the hand-token layout.

Layout
------
Per side `A = 13`: `joint1..joint7` (rad, `robot.command.q` / `robot.measured.q`)
then the six Revo2 fingers `thumb_flex, thumb_aux, index, middle, ring, pinky`
(`robot.hand.command / 1000`). Tensors are `(..., S=2, A=13)`; the contract's
flat 26-d vectors are the side-major flatten `left.* then right.*`
(`flat_names`). The rbyte chunk is `(..., H, S, A)`, so a flat contract chunk
`(H, 26)` is `chunk.flatten(-2, -1)` -- no transpose -- while the per-side
`(S, H, A)` layout the tokenizer consumes needs a `transpose(-3, -2)`.

Relative actions (P5, an ablation flag)
---------------------------------------
`relative_mode` in {none, hand, all}: the masked dims of every chunk step are
expressed relative to the ANCHOR, which is the observation state of the frame
the chunk belongs to (`relative_anchor: observation_state_of_chunk`): measured
q for the arm, `hand_prev` for the fingers. Each frame of a window has its OWN
anchor -- a single window anchor would make every other frame's target wrong.
Each relative mode gets its own standardizer and its own tokenizer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "AXIS_NAMES",
    "FINGER_AXES",
    "HAND_GROUPS",
    "HAND_GROUP_WIDTH",
    "NUM_AXES",
    "RELATIVE_MODES",
    "SIDES",
    "AxisStandardizer",
    "EventReference",
    "HandTokenStandardizer",
    "compose_hand_token",
    "compose_hand_tokens",
    "flat_names",
    "hand_feature_columns",
    "hand_token_columns",
    "hand_token_dim",
    "normalize_hand_groups",
    "normalize_hand_sides",
    "relative_mask",
    "to_absolute",
    "to_relative",
]

SIDES: Final = ("left", "right")
JOINT_NAMES: Final = tuple(f"joint{i}" for i in range(1, 8))
FINGER_NAMES: Final = ("thumb_flex", "thumb_aux", "index", "middle", "ring", "pinky")
AXIS_NAMES: Final = JOINT_NAMES + FINGER_NAMES
NUM_AXES: Final = len(AXIS_NAMES)
ARM_AXES: Final = tuple(range(len(JOINT_NAMES)))
FINGER_AXES: Final = tuple(range(len(JOINT_NAMES), NUM_AXES))

RelativeMode = Literal["none", "hand", "all"]
RELATIVE_MODES: Final = ("none", "hand", "all")
RELATIVE_ANCHOR: Final = "observation_state_of_chunk"

#: hand_features.SELECTABLE_GROUPS, in its canonical order, and the widths of
#: its `token_group_columns`. `test_nero_robot.py` pins these against the
#: vendored builder.
HAND_GROUPS: Final = ("current", "pos_err", "pos", "tip")
HAND_GROUP_WIDTH: Final = {"current": 6, "pos_err": 6, "pos": 6, "tip": 10}
HAND_TIP_NAMES: Final = ("thumb", "index", "middle", "ring", "pinky")

STANDARDIZER_SCHEMA: Final = "nutron_standardizer"
STANDARDIZER_VERSION: Final = 1
#: an axis with train std below this is a constant channel: std 1, mean = constant
DEGENERATE_STD: Final = 1e-6


def flat_names(sides: Sequence[str] = SIDES) -> list[str]:
    """`left.joint1 .. left.pinky, right.joint1 ..` -- `pc.patch_axis_names`."""
    return [f"{side}.{axis}" for side in sides for axis in AXIS_NAMES]


def relative_mask(mode: str) -> Tensor:
    """`(A,)` bool: which per-side dims are relative under `mode`.

    Raises:
        ValueError: on an unknown mode.
    """
    mask = torch.zeros(NUM_AXES, dtype=torch.bool)
    match mode:
        case "none":
            pass
        case "hand":
            mask[list(FINGER_AXES)] = True
        case "all":
            mask[:] = True
        case _:
            msg = f"relative_mode {mode!r} not in {RELATIVE_MODES}"
            raise ValueError(msg)
    return mask


def to_relative(chunk: Tensor, anchor: Tensor, mode: str) -> Tensor:
    """`chunk (..., H, S, A) - anchor (..., S, A)` on the masked dims."""
    mask = relative_mask(mode).to(chunk.device)
    return torch.where(mask, chunk - anchor.unsqueeze(-3), chunk)


def to_absolute(chunk: Tensor, anchor: Tensor, mode: str) -> Tensor:
    """Inverse of `to_relative`."""
    mask = relative_mask(mode).to(chunk.device)
    return torch.where(mask, chunk + anchor.unsqueeze(-3), chunk)


# ------------------------------------------------------------------ hand token


def normalize_hand_groups(groups: Sequence[str] | None) -> tuple[str, ...]:
    """Canonical order, duplicates refused (mirrors `hf.normalize_groups`).

    Raises:
        ValueError: on an unknown or repeated group.
    """
    groups = tuple(groups or ())
    unknown = sorted(set(groups) - set(HAND_GROUPS))
    if unknown or len(set(groups)) != len(groups):
        msg = f"hand groups {groups!r}: selectable are {HAND_GROUPS}, no repeats"
        raise ValueError(msg)
    return tuple(g for g in HAND_GROUPS if g in groups)


def hand_token_columns(groups: Sequence[str]) -> list[str]:
    """`hf.token_columns`, re-derived (pinned against the builder in tests)."""
    out: list[str] = []
    for g in normalize_hand_groups(groups):
        if g == "tip":
            out += [
                f"tip.now.{kind}.{tip}"
                for kind in ("normal", "tangential")
                for tip in HAND_TIP_NAMES
            ]
        else:
            out += [f"{g}.now.{f}" for f in FINGER_NAMES]
    return [*out, "hand_age", "hand_valid"] if out else []


def hand_token_dim(groups: Sequence[str]) -> int:
    return len(hand_token_columns(groups))


def compose_hand_token(
    batch: Mapping[str, Any], groups: Sequence[str], *, prefix: str = "hand."
) -> Tensor:
    """`hf.TokenBlocks.compose` in torch: `(..., dim)`, refused rows all zero.

    Reads `<prefix><group>` blocks, `<prefix>age`, `<prefix>motor_ok` and (when
    tip is selected) `<prefix>tip_ok`, as rbyte's `NeroRobotReader`
    stores them. `hand_valid` is the LAST column.
    """
    sel = normalize_hand_groups(groups)
    ok = batch[f"{prefix}motor_ok"].bool()
    if "tip" in sel:
        ok &= batch[f"{prefix}tip_ok"].bool()
    parts = [batch[f"{prefix}{g}"].float() for g in sel]
    parts += [batch[f"{prefix}age"].float().unsqueeze(-1), ok.float().unsqueeze(-1)]
    token = torch.cat(parts, dim=-1)
    return torch.where(ok.unsqueeze(-1), token, torch.zeros_like(token))


def normalize_hand_sides(sides: Sequence[str] | None) -> tuple[str, ...]:
    """`()` = ONE untagged hand token (the single-arm layout, `hand.*` columns);
    else one side-tagged token per entry, in `SIDES` order (contract v3
    `hand_sides`, side-major like everything else).

    Raises:
        ValueError: on an unknown, repeated or out-of-order side.
    """
    sides = tuple(sides or ())
    if any(s not in SIDES for s in sides) or sides != tuple(
        s for s in SIDES if s in sides
    ):
        msg = f"hand sides {sides!r}: a subset of {SIDES}, in that order, no repeats"
        raise ValueError(msg)
    return sides


def compose_hand_tokens(
    batch: Mapping[str, Any],
    groups: Sequence[str],
    sides: Sequence[str],
    *,
    prefix: str = "hand.",
) -> Tensor:
    """Per-side `compose_hand_token`: `(..., S, dim)`, one row per side in `sides`.

    Side `s` reads rbyte's bimanual `<prefix><s>.<block>` columns
    (`hand.left.current`, ...); every row carries its OWN `hand_valid` (last
    column), so one side's refused reading never touches the other's. This is
    the layout nutron-cli serving feeds as `hand_token (1, S, dim)`.
    """
    return torch.stack(
        [
            compose_hand_token(batch, groups, prefix=f"{prefix}{side}.")
            for side in sides
        ],
        dim=-2,
    )


# ------------------------------------------------------------------ standardizer


class AxisStandardizer(nn.Module):
    """Per-axis `(x - mean) / std` over `(..., S, A)`, stored as a versioned JSON.

    The JSON is nutron-cli's `nutron_standardizer` v1 (`pc.load_standardizer`):
    `{"schema", "version", "names" (26, side-major), "mean", "std"}`. Fit on the
    TRAIN split only; a constant axis (train std < 1e-6) and every axis of an
    invalid side get `std = 1` and `mean = the constant` (0 for an invalid side),
    because the serving loader applies no floor. `digest` is the SHA256 of the
    canonical file bytes and is what a tokenizer checkpoint pins.
    """

    # registered as buffers in __init__
    mean: Tensor
    std: Tensor

    def __init__(
        self,
        *,
        mean: Sequence[Sequence[float]] | Tensor | None = None,
        std: Sequence[Sequence[float]] | Tensor | None = None,
        sides: Sequence[str] = SIDES,
        source: str = "identity",
    ) -> None:
        super().__init__()
        shape = (len(sides), NUM_AXES)
        mean_t = torch.zeros(shape) if mean is None else torch.as_tensor(mean).float()
        std_t = torch.ones(shape) if std is None else torch.as_tensor(std).float()
        if mean_t.shape != shape or std_t.shape != shape:
            msg = f"standardizer mean/std must be {shape}"
            raise ValueError(msg)
        if not bool((std_t > 0).all()):
            msg = "every std must be > 0"
            raise ValueError(msg)
        self.sides = tuple(sides)
        self.source = source
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    # -- math
    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def unstandardize(self, x: Tensor) -> Tensor:
        return x * self.std.to(x.dtype) + self.mean.to(x.dtype)

    # -- fitting
    @classmethod
    def fit(
        cls,
        values: Tensor,
        side_valid: Tensor,
        *,
        mask: Tensor | None = None,
        source: str = "fit",
    ) -> AxisStandardizer:
        """`values (n, S, A)` with `side_valid (n, S)` and an optional `(n,)` row mask."""
        values = values.double()
        n, s, _ = values.shape
        rows = torch.ones(n, dtype=torch.bool) if mask is None else mask.bool()
        mean = torch.zeros(s, NUM_AXES, dtype=torch.float64)
        std = torch.ones(s, NUM_AXES, dtype=torch.float64)
        for side in range(s):
            keep = rows & side_valid[:, side].bool()
            if not bool(keep.any()):
                continue
            v = values[keep, side]
            mean[side] = v.mean(0)
            sd = v.std(0, unbiased=False)
            constant = sd < DEGENERATE_STD
            std[side] = torch.where(constant, torch.ones_like(sd), sd)
        return cls(mean=mean.float(), std=std.float(), source=source)

    # -- artifact
    def payload(self) -> dict[str, Any]:
        return {
            "schema": STANDARDIZER_SCHEMA,
            "version": STANDARDIZER_VERSION,
            "names": flat_names(self.sides),
            "mean": [float(v) for v in self.mean.flatten().tolist()],
            "std": [float(v) for v in self.std.flatten().tolist()],
        }

    def to_bytes(self) -> bytes:
        return (json.dumps(self.payload(), indent=1, sort_keys=True) + "\n").encode()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def save(self, path: str | Path) -> str:
        Path(path).write_bytes(self.to_bytes())
        return self.digest

    @classmethod
    def load(cls, path: str | Path, *, sha256: str | None = None) -> AxisStandardizer:
        """Load a `nutron_standardizer` JSON.

        Raises:
            ValueError: on a schema/version/name mismatch or a stated hash that
                does not match the file.
        """
        raw = Path(path).read_bytes()
        if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
            msg = f"{path}: sha256 does not match the pinned {sha256}"
            raise ValueError(msg)
        payload = json.loads(raw)
        if (payload.get("schema"), payload.get("version")) != (
            STANDARDIZER_SCHEMA,
            STANDARDIZER_VERSION,
        ):
            msg = f"{path}: not a {STANDARDIZER_SCHEMA} v{STANDARDIZER_VERSION}"
            raise ValueError(msg)
        sides = tuple(dict.fromkeys(n.split(".", 1)[0] for n in payload["names"]))
        if payload["names"] != flat_names(sides):
            msg = f"{path}: names are not the side-major robot layout"
            raise ValueError(msg)
        shape = (len(sides), NUM_AXES)
        return cls(
            mean=torch.tensor(payload["mean"]).reshape(shape),
            std=torch.tensor(payload["std"]).reshape(shape),
            sides=sides,
            source=str(path),
        )


# --------------------------------------------------------- hand token standardizer

HAND_STANDARDIZER_SCHEMA: Final = "nero_hand_token_standardizer"
HAND_STANDARDIZER_VERSION: Final = 1
#: the two auto columns `hf.build_token` appends; NEVER scaled (hand_valid is
#: the graph's substitution switch, hand_age is already on a fixed [0, 2] scale)
HAND_AUTO_COLUMNS: Final = ("hand_age", "hand_valid")
#: Documented PHYSICAL bands, in the builder's units (counts / 1000), used until
#: the train split has enough valid hand rows to fit (`nero_fit_stats`). Not
#: derived from any episode; the one tactile episode (val, 2026-10-02--17-33-40,
#: 348 valid frames from 62 motor samples) is only quoted as a sanity check in
#: docs/nero_robot_patch_policy.md:
#:
#: * current: signed, 0-centred; 100 counts (0.1, a tenth of the +-1000 clip) is
#:   the working band of a grasp -- the val episode's per-finger std is
#:   0.03-0.12, p99 up to 0.54. Unscaled, the column sat at ~0.05 next to pos
#:   (~0.2-0.6) and age (~0.5) and contributed a few percent of the first layer.
#: * pos_err (command - measured): signed, 0-centred; 50 counts (0.05) is the
#:   typical command-measured band (val std 0.009-0.077; a blocked finger at
#:   contact reaches a few hundred counts, i.e. several std -- the signal).
#: * pos: the Revo2 range is 0..1000 counts -> [0, 1]; uniform over it is mean
#:   0.5, std 1/sqrt(12) = 0.289.
#: * tip (ablation only): force clipped 0..5000 -> [0, 5]; no measured band yet,
#:   so 0-centred with unit std (= 1000 counts) -- a placeholder, refit first.
HAND_PHYSICAL_PRIOR: Final = {
    "current": (0.0, 0.1),
    "pos_err": (0.0, 0.05),
    "pos": (0.5, 0.2887),
    "tip": (0.0, 1.0),
}
#: a fitted column whose std is below this fraction of its physical band is
#: treated as degenerate (a finger that never moved in train) and keeps the band
HAND_MIN_STD_FRACTION: Final = 0.1


def hand_feature_columns(groups: Sequence[str] = HAND_GROUPS) -> list[str]:
    """`hand_token_columns(groups)` without the auto columns (what gets scaled)."""
    return [c for c in hand_token_columns(groups) if c not in HAND_AUTO_COLUMNS]


def _column_group(column: str) -> str:
    return column.split(".", 1)[0]


class HandTokenStandardizer(nn.Module):
    """Fixed per-column affine `(x - mean) / std` for the hand token (in-graph).

    The hand token's inputs come from `hf.build_token` on fixed `/1000` scales
    that were built for ACT, which standardizes with dataset stats. Without an
    affine here the columns the token exists for (current, pos_err) sit at
    ~0.01-0.1 next to pos (~0.2-0.6) and age (~0.5), and at default Linear init
    contribute a few percent of the first layer's pre-activation. This mirrors
    the state standardizer: data, not a parameter (buffers, no grad, never
    weight-decayed), shipped as a versioned JSON whose SHA256 the contract
    carries.

    The FILE always covers every selectable group's feature columns
    (`hand_feature_columns(HAND_GROUPS)`, 28) so one stats file serves every
    `hand_groups` ablation and its digest does not depend on the selection; the
    module applies the subset for `groups`. `hand_age` and `hand_valid` are
    NOT in the file and pass through unchanged -- in particular `hand_valid`
    (the last column) stays exactly 0/1. Refused rows (all zero) come out
    non-zero on the scaled columns, which is irrelevant: the policy replaces
    them with `no_hand` before they reach the trunk.

    `source` is `"physical_prior"` or `"train:hand"` (fit on the train split's
    valid rows by `nero_fit_stats`).
    """

    # registered as buffers in __init__
    full_mean: Tensor
    full_std: Tensor
    index: Tensor

    def __init__(
        self,
        *,
        groups: Sequence[str] = ("current", "pos_err"),
        mean: Sequence[float] | Tensor | None = None,
        std: Sequence[float] | Tensor | None = None,
        source: str = "physical_prior",
    ) -> None:
        super().__init__()
        names = hand_feature_columns(HAND_GROUPS)
        if mean is None or std is None:
            prior = [HAND_PHYSICAL_PRIOR[_column_group(c)] for c in names]
            mean = [m for m, _ in prior] if mean is None else mean
            std = [s for _, s in prior] if std is None else std
        mean_t = torch.as_tensor(mean, dtype=torch.float64).float()
        std_t = torch.as_tensor(std, dtype=torch.float64).float()
        if mean_t.shape != (len(names),) or std_t.shape != (len(names),):
            msg = f"hand standardizer mean/std must be ({len(names)},)"
            raise ValueError(msg)
        if not bool((std_t > 0).all()):
            msg = "every hand std must be > 0"
            raise ValueError(msg)
        self.groups = normalize_hand_groups(groups)
        self.source = source
        self.register_buffer("full_mean", mean_t)
        self.register_buffer("full_std", std_t)
        index = [names.index(c) for c in hand_feature_columns(self.groups)]
        # selection is config, not data: not part of the state_dict
        self.register_buffer(
            "index", torch.tensor(index, dtype=torch.long), persistent=False
        )
        self.requires_grad_(False)  # ruff: ignore[boolean-positional-value-in-call]

    @property
    def mean(self) -> Tensor:
        return self.full_mean[self.index]

    @property
    def std(self) -> Tensor:
        return self.full_std[self.index]

    def forward(self, x: Tensor) -> Tensor:
        """`(..., hand_token_dim(groups))` -> same shape; the last 2 columns untouched.

        Raises:
            ValueError: when the token width does not match `groups`.
        """
        n = int(self.index.numel())
        if x.shape[-1] != n + len(HAND_AUTO_COLUMNS):
            msg = f"hand token has {x.shape[-1]} columns, expected {n + 2} for {self.groups}"
            raise ValueError(msg)
        feats = (x[..., :n] - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        return torch.cat([feats, x[..., n:]], dim=-1)

    # -- fitting
    @classmethod
    def fit(
        cls,
        blocks: Mapping[str, Tensor],
        motor_ok: Tensor,
        tip_ok: Tensor | None = None,
        *,
        groups: Sequence[str] = ("current", "pos_err"),
        min_rows: int = 1000,
    ) -> tuple[HandTokenStandardizer, dict[str, Any]]:
        """Fit from rbyte's per-group blocks `{group: (n, width)}` on valid rows.

        current/pos_err/pos use the `motor_ok` rows, tip the `tip_ok` rows (a
        missing group, or fewer than `min_rows` valid rows for it, keeps the
        physical prior for its columns). A fitted std below
        `HAND_MIN_STD_FRACTION` of the band keeps the band (a finger that never
        moved in train); its mean is still fitted.

        Returns:
            The standardizer and a report (rows, source, columns that kept the prior).
        """
        names = hand_feature_columns(HAND_GROUPS)
        prior = cls(groups=groups)
        mean = prior.full_mean.double().clone()
        std = prior.full_std.double().clone()
        motor = motor_ok.reshape(-1).bool()
        tip = torch.zeros_like(motor) if tip_ok is None else tip_ok.reshape(-1).bool()
        report: dict[str, Any] = {
            "motor_rows": int(motor.sum()),
            "tip_rows": int(tip.sum()),
            "min_rows": min_rows,
        }
        kept: list[str] = []
        fitted: list[str] = []
        for group in HAND_GROUPS:
            cols = [i for i, c in enumerate(names) if _column_group(c) == group]
            rows = tip if group == "tip" else motor
            block = blocks.get(group)
            if block is None or int(rows.sum()) < min_rows:
                kept += [names[i] for i in cols]
                continue
            values = block.reshape(-1, len(cols)).double()[rows]
            for j, i in enumerate(cols):
                mean[i] = values[:, j].mean()
                sd = values[:, j].std(unbiased=False)
                if sd >= HAND_MIN_STD_FRACTION * std[i]:
                    std[i] = sd
                    fitted.append(names[i])
                else:
                    kept.append(names[i])
        source = "train:hand" if fitted else "physical_prior"
        report |= {"source": source, "fitted": fitted, "kept_prior_std": kept}
        if not fitted:
            report["reason"] = (
                f"{int(motor.sum())} valid motor rows / {int(tip.sum())} tip rows "
                f"< {min_rows}: physical prior"
            )
            return prior, report
        return cls(groups=groups, mean=mean, std=std, source=source), report

    # -- artifact
    def payload(self) -> dict[str, Any]:
        return {
            "schema": HAND_STANDARDIZER_SCHEMA,
            "version": HAND_STANDARDIZER_VERSION,
            "source": self.source,
            "names": hand_feature_columns(HAND_GROUPS),
            "passthrough": list(HAND_AUTO_COLUMNS),
            "mean": [float(v) for v in self.full_mean.tolist()],
            "std": [float(v) for v in self.full_std.tolist()],
        }

    def to_bytes(self) -> bytes:
        return (json.dumps(self.payload(), indent=1, sort_keys=True) + "\n").encode()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def save(self, path: str | Path) -> str:
        Path(path).write_bytes(self.to_bytes())
        return self.digest

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        groups: Sequence[str] = ("current", "pos_err"),
        sha256: str | None = None,
    ) -> HandTokenStandardizer:
        """Load a `nero_hand_token_standardizer` JSON for `groups`.

        Raises:
            ValueError: on a schema/version/name mismatch or a stated hash that
                does not match the file.
        """
        raw = Path(path).read_bytes()
        if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
            msg = f"{path}: sha256 does not match the pinned {sha256}"
            raise ValueError(msg)
        payload = json.loads(raw)
        if (payload.get("schema"), payload.get("version")) != (
            HAND_STANDARDIZER_SCHEMA,
            HAND_STANDARDIZER_VERSION,
        ):
            msg = (
                f"{path}: not a {HAND_STANDARDIZER_SCHEMA} v{HAND_STANDARDIZER_VERSION}"
            )
            raise ValueError(msg)
        if payload.get("names") != hand_feature_columns(HAND_GROUPS) or payload.get(
            "passthrough"
        ) != list(HAND_AUTO_COLUMNS):
            msg = f"{path}: names are not the hand_features token columns"
            raise ValueError(msg)
        return cls(
            groups=groups,
            mean=payload["mean"],
            std=payload["std"],
            source=str(payload.get("source", path)),
        )


EVENT_REFERENCE_SCHEMA: Final = "nero_event_reference"
EVENT_REFERENCE_VERSION: Final = 1
#: the most frequent exact value is THE atom only if it holds at least this share
#: of the steps; below it, the mode would make the "events" the MAJORITY (e.g.
#: middle/ring on the 4 pulled episodes: 14-16% exactly open, the rest spread),
#: which inverts the inverse-frequency weighting -- use the median then.
ATOM_MIN_SHARE: Final = 0.5
#: the tokenizer's default `event_threshold`, for the reported `event_fraction`
EVENT_THRESHOLD: Final = 0.5


class EventReference:
    """Per-axis "atom" of a relative mode's STANDARDIZED chunk, fitted on the TRAIN split.

    The chunk tokenizer boosts the loss on elements more than `event_threshold`
    from this reference (inverse-frequency event weighting, the playbook's D12
    fix). It used to be the median of the FIRST training batch -- batch-dependent:
    a first batch taken mid-grasp puts the reference on the closed plateau and
    the boost on the common open state. Now it is fitted deterministically by
    `nero_fit_stats` over every real (non-pad) chunk step of every valid side of
    the train split:

    * commanded fingers (`counts / 1000`) are quantized, so an atom is an exact
      value: the MODE, used when it holds >= `ATOM_MIN_SHARE` of the steps (the
      D12 fork shape: e.g. pinky, 54% exactly open);
    * otherwise the MEDIAN (no dominant atom; equal to the mode above 50%).

    The file also reports, per axis, the mode, its share, the median and
    `event_fraction` -- the share of steps more than `EVENT_THRESHOLD` from the
    reference, i.e. what the event weighting boosts. Inverse-frequency weighting
    only makes sense where that fraction is small.

    Pooled over valid sides (the tokenizer's rows carry no side id). The JSON
    pins the SHA256 of the action standardizer it was fitted in the units of;
    the tokenizer refuses a mismatch.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        reference: Sequence[float],
        relative_mode: str,
        standardizer_sha256: str,
        method: Sequence[str] | None = None,
        atom_share: Sequence[float] | None = None,
        median: Sequence[float] | None = None,
        mode: Sequence[float] | None = None,
        event_fraction: Sequence[float] | None = None,
        steps: int = 0,
    ) -> None:
        if len(reference) != NUM_AXES:
            msg = f"event reference must have {NUM_AXES} axes"
            raise ValueError(msg)
        self.reference = [float(v) for v in reference]
        self.relative_mode = relative_mode
        self.standardizer_sha256 = standardizer_sha256
        self.method = list(method) if method is not None else ["given"] * NUM_AXES
        self.atom_share = [float(v) for v in atom_share] if atom_share else []
        self.median = [float(v) for v in median] if median else []
        self.mode = [float(v) for v in mode] if mode else []
        self.event_fraction = (
            [float(v) for v in event_fraction] if event_fraction else []
        )
        self.steps = int(steps)

    @classmethod
    def fit(  # noqa: PLR0914
        cls,
        standardized: Tensor,
        side_valid: Tensor,
        *,
        relative_mode: str,
        standardizer_sha256: str,
    ) -> EventReference:
        """`standardized (m, S, A)` real chunk steps, `side_valid (m, S)`."""
        reference, method, share, medians, modes, events = [], [], [], [], [], []
        keep = side_valid.bool()
        for axis in range(NUM_AXES):
            col = standardized[..., axis][keep].float()
            if col.numel() == 0:
                reference.append(0.0)
                method.append("empty")
                share.append(0.0)
                medians.append(0.0)
                modes.append(0.0)
                events.append(0.0)
                continue
            values, counts = torch.unique(col, return_counts=True)
            top = int(counts.argmax())  # ties -> the lowest value (unique is sorted)
            mode_value = float(values[top])
            mode_share = float(counts[top]) / col.numel()
            median = float(col.median())
            atom = mode_share >= ATOM_MIN_SHARE
            ref = mode_value if atom else median
            reference.append(ref)
            method.append("mode" if atom else "median")
            events.append(float(((col - ref).abs() > EVENT_THRESHOLD).float().mean()))
            share.append(mode_share)
            medians.append(median)
            modes.append(mode_value)
        return cls(
            reference=reference,
            relative_mode=relative_mode,
            standardizer_sha256=standardizer_sha256,
            method=method,
            atom_share=share,
            median=medians,
            mode=modes,
            event_fraction=events,
            steps=int(keep.sum()),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": EVENT_REFERENCE_SCHEMA,
            "version": EVENT_REFERENCE_VERSION,
            "relative_mode": self.relative_mode,
            "standardizer_sha256": self.standardizer_sha256,
            "axes": list(AXIS_NAMES),
            "reference": self.reference,
            "method": self.method,
            "atom_share": self.atom_share,
            "median": self.median,
            "mode": self.mode,
            "event_fraction": self.event_fraction,
            "steps": self.steps,
        }

    def to_bytes(self) -> bytes:
        return (json.dumps(self.payload(), indent=1, sort_keys=True) + "\n").encode()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def save(self, path: str | Path) -> str:
        Path(path).write_bytes(self.to_bytes())
        return self.digest

    @classmethod
    def load(cls, path: str | Path) -> EventReference:
        """Load an `event_reference_<mode>.json`.

        Raises:
            ValueError: on a schema/version/axis mismatch.
        """
        payload = json.loads(Path(path).read_bytes())
        if (payload.get("schema"), payload.get("version")) != (
            EVENT_REFERENCE_SCHEMA,
            EVENT_REFERENCE_VERSION,
        ):
            msg = f"{path}: not a {EVENT_REFERENCE_SCHEMA} v{EVENT_REFERENCE_VERSION}"
            raise ValueError(msg)
        if payload.get("axes") != list(AXIS_NAMES):
            msg = f"{path}: axes are not the robot layout {AXIS_NAMES}"
            raise ValueError(msg)
        return cls(
            reference=payload["reference"],
            relative_mode=payload["relative_mode"],
            standardizer_sha256=payload["standardizer_sha256"],
            method=payload.get("method"),
            atom_share=payload.get("atom_share"),
            median=payload.get("median"),
            mode=payload.get("mode"),
            event_fraction=payload.get("event_fraction"),
            steps=payload.get("steps", 0),
        )


def flat_rbyte_batch(batch: Any) -> dict[str, Any]:
    """rbyte `Batch` -> `{column: tensor}` (the model reads flat keys)."""
    data = batch.data if hasattr(batch, "data") else batch["data"]
    return {key: data[key] for key in data.keys()}  # noqa: SIM118
