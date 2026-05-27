"""Tests for the ParetoArchive.bt_tiebreak_topk method (AP-38).

AP-38 wires the shared Bradley-Terry module into the stagnation handler by
aggregating top-K Pareto frontier entries via axis-wise pairwise comparison.
These tests verify the aggregation path itself; the actual stagnation-handler
integration in autopilot.py is exercised by existing autopilot integration
tests (the new template field has a fallback path so the prompt build does
not crash if BT data is unavailable).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

pareto_archive = importlib.import_module("pareto_archive")
ParetoArchive = pareto_archive.ParetoArchive
ParetoEntry = pareto_archive.ParetoEntry


def _make_entry(trial_id: int, q: float, sp: float, neg_cost: float, rel: float) -> ParetoEntry:
    return ParetoEntry(
        trial_id=trial_id,
        objectives=(q, sp, neg_cost, rel),
    )


def _archive_with(entries: list[ParetoEntry]) -> ParetoArchive:
    a = ParetoArchive(state_path=Path("/tmp/_test_archive_does_not_exist.json"))
    # Bypass file IO; just inject the frontier directly.
    a._frontier = list(entries)
    a._all_entries = list(entries)
    return a


def test_empty_frontier_returns_safe_default():
    a = ParetoArchive(state_path=Path("/tmp/_test_archive_empty.json"))
    out = a.bt_tiebreak_topk(k=5)
    assert out["ranking"] == []
    assert "skipped" in out["note"]


def test_singleton_frontier_returns_safe_default():
    a = _archive_with([_make_entry(1, 2.0, 50.0, -0.1, 0.9)])
    out = a.bt_tiebreak_topk(k=5)
    assert out["ranking"] == [1]
    assert "skipped" in out["note"]


def test_two_entry_frontier_runs_and_returns_ranking():
    a = _archive_with([
        _make_entry(1, 3.0, 50.0, -0.1, 0.9),
        _make_entry(2, 2.0, 30.0, -0.2, 0.7),
    ])
    out = a.bt_tiebreak_topk(k=5)
    assert set(out["ranking"]) == {1, 2}
    # #1 dominates on all axes, so BT must rank it first.
    assert out["ranking"][0] == 1
    assert out["converged"]


def test_axis_wise_disagreement_surfaced():
    """A candidate that beats peers on more axes (but loses on the
    high-magnitude axis sum) should be picked by BT — that disagreement
    is the value of this method.

    Setup designed so the naive sum-of-axes prefers trial 10 (dominates
    the speed axis whose magnitude ~80 swamps all other axes' sums) but
    BT prefers trial 11 (wins on 3 of 4 axes pairwise).
    """
    a = _archive_with([
        _make_entry(10, 3.0, 80.0, -0.5, 0.5),  # only wins on speed
        _make_entry(11, 4.0, 30.0, -0.1, 0.9),  # wins on quality, cost, rel
        _make_entry(12, 2.0, 40.0, -0.3, 0.6),  # mid everywhere
    ])
    out = a.bt_tiebreak_topk(k=3)
    # Naive scalarization (sum of axes) would put trial 10 on top
    # because speed=80 dominates the sum. BT must instead pick trial 11.
    assert out["top_k_trial_ids"][0] == 10, "test setup invariant: naive-top is trial 10"
    assert out["ranking"][0] == 11, "BT should disagree and pick trial 11"
    assert "disagrees" in out["note"] or "BT picks" in out["note"]


def test_k_caps_at_frontier_size():
    """Asking for k > frontier_size should not raise; uses all available."""
    a = _archive_with([
        _make_entry(1, 3.0, 50.0, -0.1, 0.9),
        _make_entry(2, 2.0, 60.0, -0.2, 0.8),
    ])
    out = a.bt_tiebreak_topk(k=10)
    assert len(out["ranking"]) == 2


def test_uniform_axes_all_ties():
    """Pathological case: all entries equal on every axis → BT must not crash;
    ranking is arbitrary but every log_skill is ~0 and convergence holds."""
    a = _archive_with([
        _make_entry(1, 2.0, 50.0, -0.1, 0.8),
        _make_entry(2, 2.0, 50.0, -0.1, 0.8),
        _make_entry(3, 2.0, 50.0, -0.1, 0.8),
    ])
    out = a.bt_tiebreak_topk(k=3)
    assert out["converged"]
    for ls in out["log_skills"].values():
        assert abs(ls) < 1e-6


def test_log_skill_anchor_is_zero():
    a = _archive_with([
        _make_entry(1, 4.0, 60.0, -0.1, 0.9),
        _make_entry(2, 3.0, 50.0, -0.2, 0.8),
        _make_entry(3, 2.0, 40.0, -0.3, 0.7),
    ])
    out = a.bt_tiebreak_topk(k=3)
    assert min(out["log_skills"].values()) == 0.0


def test_dominance_skew_warning_propagates_through_archive_method():
    """If one frontier entry sweeps the field, the warning must surface."""
    a = _archive_with([
        _make_entry(1, 10.0, 100.0, 0.0, 1.0),  # dominates on every axis
        _make_entry(2, 1.0, 10.0, -0.5, 0.1),
        _make_entry(3, 1.0, 10.0, -0.5, 0.1),
        _make_entry(4, 1.0, 10.0, -0.5, 0.1),
    ])
    out = a.bt_tiebreak_topk(k=4)
    assert out["ranking"][0] == 1
    assert any("dominance skew" in w for w in out["warnings"])
