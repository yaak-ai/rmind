"""rmind.data.nero_relabel: the opt-in relabel rule of the v3 reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rmind.data import nero_relabel as nr

RELABEL = {
    "outcome": "success",
    "source": "relabel:operator-request-2026-10-09",
    "relabeled_from": "rejected (discard, no outcome.json)",
}
DAY = Path.home() / "data/nero-arms/cube-bimanual/2026-10-09"
RELABELLED = DAY / "2026-10-09--11-56-19-864211"
needs_take = pytest.mark.skipif(
    not (RELABELLED / "data.mcap").exists(), reason=f"no {RELABELLED}"
)


def test_relabel_provenance_needs_both_fields() -> None:
    assert nr.relabel_provenance(RELABEL) == RELABEL["source"]
    assert nr.relabel_provenance({**RELABEL, "source": "shared_dashboard"}) is None
    no_from = {k: v for k, v in RELABEL.items() if k != "relabeled_from"}
    assert nr.relabel_provenance(no_from) is None


def _take(tmp_path: Path, outcome: dict) -> Path:
    take = tmp_path / RELABELLED.name
    take.mkdir()
    for f in RELABELLED.iterdir():
        if f.name != "outcome.json":
            (take / f.name).symlink_to(f)
    (take / "outcome.json").write_text(json.dumps(outcome))
    return take / "data.mcap"


@needs_take
def test_relabelled_take_is_read_only_when_opted_in(tmp_path: Path) -> None:
    mcap = _take(tmp_path, RELABEL)
    record = nr.relabel_override(mcap)
    assert record is not None
    assert record["mcap_outcome"] is None
    assert nr.NeroRobotRelabelReader(chunk_size=100)(mcap).height > 0
    with pytest.raises(Exception, match="mcap outcome is None"):
        nr.NeroRobotRelabelReader(chunk_size=100, accept_relabel=False)(mcap)


@needs_take
def test_success_without_provenance_is_still_refused(tmp_path: Path) -> None:
    mcap = _take(tmp_path, {"outcome": "success", "source": "shared_dashboard"})
    assert nr.relabel_override(mcap) is None
    with pytest.raises(Exception, match="mcap outcome is None"):
        nr.NeroRobotRelabelReader(chunk_size=100)(mcap)
