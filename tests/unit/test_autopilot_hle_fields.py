"""HLE-4 observe-only metric plumbing tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

from experiment_journal import ExperimentJournal, JournalEntry  # noqa: E402
from safety_gate import EvalResult  # noqa: E402


def test_eval_result_carries_hle_observe_only_fields() -> None:
    result = EvalResult(
        tier=1,
        quality=0.75,
        speed=20.0,
        cost=0.2,
        reliability=0.95,
        metric_schema_version=1,
        harness_metrics={
            "execution_fidelity": {
                "score": 0.8,
                "evidence_event_ids": [101, 102],
                "confidence": 0.7,
            }
        },
        oracle_adequacy={
            "sentinel_python": {
                "oracle_type": "unit_test",
                "deterministic": True,
                "known_blind_spots": ["does_not_check_latency"],
            }
        },
    )

    assert result.metric_schema_version == 1
    assert result.harness_metrics["execution_fidelity"]["score"] == 0.8
    assert result.oracle_adequacy["sentinel_python"]["deterministic"] is True


def test_journal_round_trips_hle_fields_in_jsonl(tmp_path: Path) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    entry = JournalEntry(
        trial_id=7,
        timestamp="2026-05-27T00:00:00Z",
        species="hle-observe",
        action_type="observe_only",
        tier=1,
        quality=0.75,
        speed=20.0,
        cost=0.2,
        reliability=0.95,
        pareto_status="candidate",
        metric_schema_version=1,
        harness_metrics={"planning_stability": {"score": 0.6, "evidence_event_ids": [42]}},
        oracle_adequacy={"coding_sentinel": {"oracle_type": "pytest", "deterministic": True}},
    )

    journal.record(entry)

    raw = json.loads((tmp_path / "autopilot_journal.jsonl").read_text().splitlines()[0])
    assert raw["metric_schema_version"] == 1
    assert raw["harness_metrics"]["planning_stability"]["score"] == 0.6
    assert raw["oracle_adequacy"]["coding_sentinel"]["oracle_type"] == "pytest"

    reloaded = ExperimentJournal(journal_dir=tmp_path)
    loaded = reloaded.all_entries()[0]
    assert loaded.metric_schema_version == 1
    assert loaded.harness_metrics == entry.harness_metrics
    assert loaded.oracle_adequacy == entry.oracle_adequacy


def test_journal_loads_hle_fields_from_legacy_eval_details(tmp_path: Path) -> None:
    (tmp_path / "autopilot_journal.jsonl").write_text(
        json.dumps({
            "trial_id": 3,
            "timestamp": "2026-05-27T00:00:00Z",
            "species": "legacy",
            "action_type": "observe_only",
            "tier": 1,
            "quality": 0.5,
            "speed": 10.0,
            "cost": 0.4,
            "reliability": 0.9,
            "pareto_status": "dominated",
            "eval_details": {
                "metric_schema_version": 1,
                "harness_metrics": {"memory_coherence": {"score": 0.9}},
                "oracle_adequacy": {"qa": {"oracle_type": "exact_match"}},
            },
        })
        + "\n"
    )

    loaded = ExperimentJournal(journal_dir=tmp_path).all_entries()[0]

    assert loaded.metric_schema_version == 1
    assert loaded.harness_metrics == {"memory_coherence": {"score": 0.9}}
    assert loaded.oracle_adequacy == {"qa": {"oracle_type": "exact_match"}}
