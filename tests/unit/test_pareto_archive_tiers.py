"""Pareto archive tier semantics.

Tier-0 is a fast-reject sentinel eval with 10-question granularity. It is useful
for quick screening and audit history, but it must not define the production
Pareto frontier or baseline archive-max guard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

from pareto_archive import (  # noqa: E402
    PARETO_STATUS_TIER_EXCLUDED,
    ParetoArchive,
    ParetoEntry,
)


def _entry(
    trial_id: int,
    *,
    tier: int,
    q: float,
    speed: float = 50.0,
    cost: float = -0.5,
    reliability: float = 0.9,
) -> ParetoEntry:
    return ParetoEntry(
        trial_id=trial_id,
        objectives=(q, speed, cost, reliability),
        eval_tier=tier,
    )


def test_t0_entry_is_audit_only_not_frontier(tmp_path: Path) -> None:
    archive = ParetoArchive(state_path=tmp_path / "state.json")

    status = archive.update(_entry(7, tier=0, q=2.4))

    assert status == PARETO_STATUS_TIER_EXCLUDED
    assert len(archive._all_entries) == 1
    assert archive.frontier() == []
    assert archive.hypervolume_trend() == []
    assert archive.summary()["frontier_size"] == 0


def test_t0_quality_does_not_dominate_t1_frontier(tmp_path: Path) -> None:
    archive = ParetoArchive(state_path=tmp_path / "state.json")

    t0_status = archive.update(_entry(7, tier=0, q=2.4, speed=80.0))
    t1_status = archive.update(_entry(118, tier=1, q=1.895, speed=71.0))

    assert t0_status == PARETO_STATUS_TIER_EXCLUDED
    assert t1_status == "frontier"
    assert [e.trial_id for e in archive.frontier()] == [118]
    assert archive.summary()["best_quality"] == 1.895


def test_load_rebuilds_frontier_without_legacy_t0_pollution(tmp_path: Path) -> None:
    """Existing state may have T1 entries in all_entries that were kept off the
    frontier only because saturated T0 entries dominated them. Loading must
    rebuild the frontier from all eligible T1/T2 entries, not merely filter the
    old frontier list.
    """
    state_path = tmp_path / "state.json"
    t0 = _entry(38, tier=0, q=2.4, speed=20.0).to_dict()
    t1 = _entry(118, tier=1, q=1.895, speed=71.0).to_dict()
    state_path.write_text(json.dumps({
        "trial_counter": 569,
        "pareto_archive": {
            "frontier": [t0],
            "all_entries": [t0, t1],
            "hypervolume_history": [[38, 10.0]],
        },
    }))

    archive = ParetoArchive(state_path=state_path)

    assert [e.trial_id for e in archive.frontier()] == [118]
    assert archive.summary()["best_quality"] == 1.895
