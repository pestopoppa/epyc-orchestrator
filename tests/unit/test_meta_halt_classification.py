"""Unit tests for autopilot._classify_meta_halt (2026-05-31).

A meta-action-loop halt is classified 'converged' (benign — recent metric
trials reproduced an already-kept above-baseline config) vs 'stuck' (genuine
gate-lock / corruption). 'converged' must NOT read as a malfunction or an
instrument-noise problem.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

from autopilot import _classify_meta_halt  # type: ignore[import-not-found]


@dataclass
class _Entry:
    deficiency_category: str = ""
    bug_corrupted_by: str = ""


@dataclass
class _FakeJournal:
    entries: list[_Entry] = field(default_factory=list)

    def recent_hypotheses(self, n=5, exclude_bug_corrupted=False):
        return self.entries[-n:]


def test_converged_when_recent_reproduction_confirmed_and_no_corruption():
    j = _FakeJournal([
        _Entry(deficiency_category="reproduction_confirmed"),
        _Entry(deficiency_category="reproduction_confirmed"),
        _Entry(deficiency_category=""),
    ])
    assert _classify_meta_halt(j) == "converged"


def test_stuck_when_recent_has_corruption():
    j = _FakeJournal([
        _Entry(deficiency_category="reproduction_confirmed"),
        _Entry(bug_corrupted_by="ec9622d"),  # operator/code invalidation present
    ])
    assert _classify_meta_halt(j) == "stuck"


def test_stuck_when_no_reproduction_signal():
    j = _FakeJournal([_Entry(), _Entry(deficiency_category="regression")])
    assert _classify_meta_halt(j) == "stuck"


def test_classification_never_crashes_on_bad_journal():
    class _Boom:
        def recent_hypotheses(self, *a, **k):
            raise RuntimeError("journal unavailable")
    assert _classify_meta_halt(_Boom()) == "stuck"


# ── JEV-2: typed halt record ──────────────────────────────────────────
from autopilot import (  # type: ignore[import-not-found]  # noqa: E402
    MetaHaltStopReason,
    _classify_meta_halt_record,
)


def test_record_reason_is_enum_and_carries_evidence():
    class _J:
        def recent_hypotheses(self, n, exclude_bug_corrupted):
            return [type("E", (), {"deficiency_category": "reproduction_confirmed", "bug_corrupted_by": ""})()]

    r = _classify_meta_halt_record(_J())
    assert r.reason is MetaHaltStopReason.CONVERGED
    assert r.reproduction_confirmed == 1 and r.bug_corrupted == 0
    assert r.to_dict()["reason"] == "converged"


def test_record_exposes_classifier_error():
    class _Boom:
        def recent_hypotheses(self, n, exclude_bug_corrupted):
            raise RuntimeError("journal gone")

    r = _classify_meta_halt_record(_Boom())
    assert r.reason is MetaHaltStopReason.STUCK
    assert r.classifier_error and "journal gone" in r.classifier_error


def test_stop_reason_enum_is_exhaustive():
    assert {m.value for m in MetaHaltStopReason} == {"converged", "stuck"}
