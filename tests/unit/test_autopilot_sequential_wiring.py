from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import autopilot  # type: ignore[import-not-found]  # noqa: E402
from experiment_journal import ExperimentJournal, JournalEntry  # noqa: E402


def _entry(
    trial_id: int,
    action: dict,
    *,
    tier: int = 1,
    correct: bool = True,
    seq: dict | None = None,
    corrupt: str = "",
    outcome_status: str = "ok",
) -> JournalEntry:
    return JournalEntry(
        trial_id=trial_id,
        timestamp="2026-06-18T00:00:00Z",
        species="test",
        action_type=str(action.get("type") or "seed_batch"),
        tier=tier,
        quality=3.0 if correct else 0.0,
        speed=10.0,
        cost=0.2,
        reliability=1.0,
        pareto_status="candidate",
        config_snapshot=dict(action),
        eval_details={
            "eval_wall_s": 1800.0,
            "question_results": [{"qid": "q1", "correct": correct}],
        },
        seq=seq or {},
        bug_corrupted_by=corrupt,
        outcome_status=outcome_status,
    )


def test_seq_inputs_use_trusted_same_tier_prior_rows(tmp_path: Path) -> None:
    action = {"type": "seed_batch", "n_questions": 10}
    candidate = autopilot._config_fingerprint(action)
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(1, action, correct=True))
    journal.record(
        _entry(
            2,
            action,
            correct=False,
            seq={
                "candidate": candidate,
                "core_id": "core_v1",
                "z": 0.25,
                "z_rate": 0.1,
                "state": "accumulating",
                "policy_version": "seq-v1",
            },
        )
    )
    journal.record(_entry(3, action, correct=True, corrupt="resource_contention"))
    journal.record(_entry(4, action, correct=True, outcome_status="skipped"))
    journal.record(_entry(5, action, tier=2, correct=True))

    inputs = autopilot._seq_inputs_for_trial(journal=journal, action=action, tier=1)

    assert inputs["candidate"] == candidate
    assert inputs["core_id"] == "core_v1"
    assert inputs["baseline_profile"] == {"q1": 0.5}
    assert inputs["baseline_task_rate"] == pytest.approx(2.0)
    assert inputs["prior_quality_obs"] == [(2, 0.25)]
    assert inputs["prior_rate_obs"] == [(2, 0.1)]
