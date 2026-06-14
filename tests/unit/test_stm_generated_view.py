"""Tests for journal-derived read-only AutoPilot STM preview."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(AUTOPILOT_DIR))

from experiment_journal import ExperimentJournal, JournalEntry  # noqa: E402
from stm_generated_view import render_generated_stm  # noqa: E402


def _entry(
    trial_id: int,
    *,
    hypothesis: str = "Try a narrower prompt",
    optimization_directions: str = "tighten routing; reduce retries",
    failure_analysis: str = "",
    eval_details: dict | None = None,
    bug_corrupted_by: str = "",
    outcome_status: str = "ok",
    pareto_status: str = "dominated",
) -> JournalEntry:
    return JournalEntry(
        trial_id=trial_id,
        timestamp=f"2026-06-14T00:00:{trial_id:02d}+00:00",
        species="seeder",
        action_type="seed_batch",
        tier=1,
        quality=1.2 + trial_id / 100,
        speed=40.0,
        cost=0.4,
        reliability=0.95,
        pareto_status=pareto_status,
        hypothesis=hypothesis,
        optimization_directions=optimization_directions,
        failure_analysis=failure_analysis,
        keep_revert_decision="keep",
        eval_details=eval_details or {},
        bug_corrupted_by=bug_corrupted_by,
        outcome_status=outcome_status,
    )


def test_render_generated_stm_does_not_write_memory_file(tmp_path: Path) -> None:
    memory_file = tmp_path / "short_term_memory.md"

    text = render_generated_stm([_entry(1, pareto_status="frontier")], last_n=5)

    assert not memory_file.exists()
    assert "Journal-derived read-only preview" in text
    assert "[t1] Try a narrower prompt -- confirmed" in text
    assert "[t1] tighten routing" in text
    assert "[t1] reduce retries" in text


def test_render_generated_stm_excludes_superseded_corrupted_rows(tmp_path: Path) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    journal.record(_entry(1, hypothesis="KEEP_ME"))
    journal.record(_entry(2, hypothesis="DROP_ME"))
    journal.append_supersession_event(
        target_trial_ids=[2],
        fields={"bug_corrupted_by": "resource_contention"},
        reason="synthetic contention",
        policy_version="supersession-v1",
        actor="unit-test",
    )

    text = render_generated_stm(journal.entries_with_supersessions())

    assert "KEEP_ME" in text
    assert "DROP_ME" not in text
    assert journal.all_entries()[1].bug_corrupted_by == ""


def test_render_generated_stm_excludes_learning_and_invalid_rows() -> None:
    entries = [
        _entry(1, hypothesis="KEEP_ME"),
        _entry(
            2,
            hypothesis="DROP_LEARNING",
            eval_details={"learning_exclusion": {"by": "reproduction_confirmed"}},
        ),
        _entry(3, hypothesis="DROP_ERROR", outcome_status="error"),
        _entry(4, hypothesis="DROP_CORRUPT", bug_corrupted_by="autopilot_killed_mid_trial"),
    ]

    text = render_generated_stm(entries)

    assert "KEEP_ME" in text
    assert "DROP_LEARNING" not in text
    assert "DROP_ERROR" not in text
    assert "DROP_CORRUPT" not in text


def test_render_generated_stm_uses_sanitized_failure_and_weak_suite_context() -> None:
    entry = _entry(
        5,
        failure_analysis="Quality regression: 0.000 vs baseline 9.900 (-100.0%)",
        eval_details={"per_suite_quality": {"coder": 1.2, "math": 1.8}},
    )

    text = render_generated_stm([entry])

    assert "baseline 9.900" not in text
    assert "legacy-scale failure_analysis omitted" in text
    assert "Weak suites: coder=1.20" in text
