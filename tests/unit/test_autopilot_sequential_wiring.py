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
    timestamp: str = "2026-06-18T00:00:00Z",
    eval_details_extra: dict | None = None,
) -> JournalEntry:
    eval_details = {
        "eval_wall_s": 1800.0,
        "question_results": [{"qid": "q1", "correct": correct}],
    }
    if eval_details_extra:
        eval_details.update(eval_details_extra)
    return JournalEntry(
        trial_id=trial_id,
        timestamp=timestamp,
        species="test",
        action_type=str(action.get("type") or "seed_batch"),
        tier=tier,
        quality=3.0 if correct else 0.0,
        speed=10.0,
        cost=0.2,
        reliability=1.0,
        pareto_status="candidate",
        config_snapshot=dict(action),
        eval_details=eval_details,
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
    assert inputs["baseline_reference"]["due"] is True


def test_seq_baseline_reference_state_tracks_cadence_and_staleness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(autopilot, "SEQ_BASELINE_REFRESH_CADENCE", 2)
    monkeypatch.setattr(autopilot, "SEQ_BASELINE_REFERENCE_STALE_AFTER_S", 3600.0)
    action = {"type": "seed_batch", "n_questions": 12}
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(
        _entry(
            1,
            action,
            timestamp="2026-06-18T00:00:00Z",
            eval_details_extra={"seq_baseline_reference_draw": True},
        )
    )
    journal.record(_entry(2, action, timestamp="2026-06-18T00:10:00Z"))

    fresh = autopilot._seq_baseline_reference_state(
        journal,
        tier=1,
        now_ts=autopilot._parse_journal_timestamp("2026-06-18T00:20:00Z"),
    )

    assert fresh["due"] is False
    assert fresh["stale_reference"] is False
    assert fresh["trials_since_reference"] == 1

    journal.record(_entry(3, action, timestamp="2026-06-18T00:30:00Z"))
    cadence_due = autopilot._seq_baseline_reference_state(
        journal,
        tier=1,
        now_ts=autopilot._parse_journal_timestamp("2026-06-18T00:40:00Z"),
    )
    assert cadence_due["due"] is True
    assert cadence_due["trials_since_reference"] == 2
    assert "trusted profile trials" in cadence_due["reason"]

    stale = autopilot._seq_baseline_reference_state(
        journal,
        tier=1,
        now_ts=autopilot._parse_journal_timestamp("2026-06-18T02:30:01Z"),
    )
    assert stale["due"] is True
    assert stale["stale_reference"] is True


def test_maybe_force_seq_baseline_draw_marks_rationale_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(autopilot, "SEQ_BASELINE_REFRESH_CADENCE", 10)
    journal = ExperimentJournal(journal_dir=tmp_path)
    action = {"type": "noop"}
    state: dict = {}

    forced, rationale, reference = autopilot._maybe_force_seq_baseline_draw(
        action,
        state=state,
        journal=journal,
        tier=1,
        blacklist=[],
        rationale={"source": "test"},
        trial_counter=8,
        enabled=True,
    )

    assert forced == {"type": "seed_batch", "n_questions": 12}
    assert rationale == {
        "source": "test",
        "seq_baseline_reference_draw": True,
        "seq_baseline_reference_reason": "no marked seq baseline-reference draw",
    }
    assert reference is not None
    assert state["seq_baseline_draw_forced"]["trial_id"] == 8


def test_maybe_force_seq_baseline_draw_respects_blacklist(tmp_path: Path) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    action = {"type": "noop"}
    state: dict = {}

    forced, rationale, reference = autopilot._maybe_force_seq_baseline_draw(
        action,
        state=state,
        journal=journal,
        tier=1,
        blacklist=[
            {"pattern": {"type": "seed_batch", "n_questions": 12}, "reason": "test"}
        ],
        rationale=None,
        trial_counter=9,
        enabled=True,
    )

    assert forced == action
    assert rationale is None
    assert reference is None
    assert state["seq_baseline_draw_blocked"]["reason"] == "test"

    retried, retry_rationale, retry_reference = autopilot._maybe_force_seq_baseline_draw(
        action,
        state=state,
        journal=journal,
        tier=1,
        blacklist=[
            {"pattern": {"type": "seed_batch", "n_questions": 12}, "reason": "test"}
        ],
        rationale=None,
        trial_counter=10,
        enabled=True,
    )
    assert retried == action
    assert retry_rationale is None
    assert retry_reference is None
    assert state["seq_baseline_draw_blocked"]["trial_id"] == 9


def test_seq_promotion_finalization_requires_fresh_eval_fresh_reference_and_e() -> None:
    seq = {"confirmed": True, "E_quality": 120.0, "E_rate_noninf": 110.0}
    reference = {
        "tier": 2,
        "latest_reference_trial_id": 4,
        "latest_reference_age_s": 120.0,
        "trials_since_reference": 1,
        "stale_reference": False,
    }

    finalized = autopilot._annotate_seq_promotion_finalization(
        seq,
        baseline_reference=reference,
        is_fresh_eval=True,
        fresh_eval_context={"candidate": "abc", "source_trial_id": 3},
    )

    assert finalized is True
    assert seq["baseline_promotion_finalized"] is True
    assert seq["baseline_promotion_combined_E"] == pytest.approx(110.0)
    assert seq["baseline_promotion_fresh_eval_for"] == {
        "candidate": "abc",
        "source_trial_id": 3,
    }

    not_fresh = {"confirmed": True, "E_quality": 120.0, "E_rate_noninf": 110.0}
    assert (
        autopilot._annotate_seq_promotion_finalization(
            not_fresh,
            baseline_reference=reference,
            is_fresh_eval=False,
        )
        is False
    )

    stale = {"confirmed": True, "E_quality": 120.0, "E_rate_noninf": 110.0}
    stale_reference = dict(reference, stale_reference=True)
    assert (
        autopilot._annotate_seq_promotion_finalization(
            stale,
            baseline_reference=stale_reference,
            is_fresh_eval=True,
        )
        is False
    )
    assert stale["baseline_reference_state"] == "stale-reference"

    low_e = {"confirmed": True, "E_quality": 120.0, "E_rate_noninf": 99.0}
    assert (
        autopilot._annotate_seq_promotion_finalization(
            low_e,
            baseline_reference=reference,
            is_fresh_eval=True,
        )
        is False
    )


def test_seq_promotion_state_queues_and_forces_one_fresh_eval() -> None:
    state: dict = {}
    action = {"type": "seed_batch", "n_questions": 12}
    eval_result = autopilot.EvalResult(
        tier=2,
        quality=3.0,
        speed=10.0,
        cost=0.1,
        reliability=1.0,
    )

    autopilot._update_seq_promotion_fresh_eval_state(
        state,
        seq={
            "candidate": "candidate-a",
            "confirmed": True,
            "baseline_reference_state": "fresh",
            "baseline_promotion_combined_E": 25.0,
        },
        action=action,
        eval_result=eval_result,
        trial_counter=11,
        is_fresh_eval=False,
        finalized=False,
    )

    assert state["seq_pending_promotion_fresh_eval"]["candidate"] == "candidate-a"
    forced, rationale, context = autopilot._maybe_force_seq_promotion_fresh_eval(
        {"type": "noop"},
        state=state,
        blacklist=[],
        rationale=None,
        trial_counter=12,
        enabled=True,
    )

    assert forced == {"type": "deep_eval", "tier": 2}
    assert rationale == {
        "seq_promotion_fresh_eval": True,
        "seq_promotion_candidate": "candidate-a",
    }
    assert context is not None
    assert context["candidate"] == "candidate-a"
    assert state["seq_pending_promotion_fresh_eval"]["attempts"] == 1


def test_seq_promotion_fresh_eval_blacklist_suppresses_retry() -> None:
    state = {
        "seq_pending_promotion_fresh_eval": {
            "candidate": "candidate-a",
            "source_trial_id": 20,
            "tier": 2,
            "attempts": 0,
        }
    }

    action = {"type": "noop"}
    first, _, context = autopilot._maybe_force_seq_promotion_fresh_eval(
        action,
        state=state,
        blacklist=[{"pattern": {"type": "deep_eval", "tier": 2}, "reason": "test"}],
        rationale=None,
        trial_counter=21,
        enabled=True,
    )

    assert first == action
    assert context is None
    pending = state["seq_pending_promotion_fresh_eval"]
    assert pending["attempts"] == 1
    assert pending["blocked_reason"] == "test"

    second, _, second_context = autopilot._maybe_force_seq_promotion_fresh_eval(
        action,
        state=state,
        blacklist=[{"pattern": {"type": "deep_eval", "tier": 2}, "reason": "test"}],
        rationale=None,
        trial_counter=22,
        enabled=True,
    )

    assert second == action
    assert second_context is None
    assert state["seq_pending_promotion_fresh_eval"]["blocked_at_trial"] == 21
