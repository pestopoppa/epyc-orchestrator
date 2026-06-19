from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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

    assert forced == {"type": "seed_batch", "n_questions": 14}
    assert rationale == {
        "source": "test",
        "seq_baseline_reference_draw": True,
        "seq_baseline_reference_reason": "no marked seq baseline-reference draw",
    }
    assert reference is not None
    assert state["seq_baseline_draw_forced"]["trial_id"] == 8


def test_maybe_force_seq_baseline_draw_uses_alternate_when_default_blacklisted(
    tmp_path: Path,
) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    action = {"type": "noop"}
    state: dict = {}

    forced, rationale, reference = autopilot._maybe_force_seq_baseline_draw(
        action,
        state=state,
        journal=journal,
        tier=1,
        blacklist=[
            {"pattern": {"type": "seed_batch", "n_questions": 14}, "reason": "test"}
        ],
        rationale=None,
        trial_counter=9,
        enabled=True,
    )

    assert forced == {"type": "seed_batch", "n_questions": 16}
    assert rationale == {
        "seq_baseline_reference_draw": True,
        "seq_baseline_reference_reason": "no marked seq baseline-reference draw",
    }
    assert reference is not None
    assert state["seq_baseline_draw_forced"]["action"] == forced


def test_maybe_force_seq_baseline_draw_records_block_when_all_fallbacks_blacklisted(
    tmp_path: Path,
) -> None:
    journal = ExperimentJournal(journal_dir=tmp_path)
    action = {"type": "noop"}
    state: dict = {}
    blacklist = [
        {"pattern": candidate, "reason": f"blocked-{idx}"}
        for idx, candidate in enumerate(autopilot._seed_action_candidates())
    ]

    forced, rationale, reference = autopilot._maybe_force_seq_baseline_draw(
        action,
        state=state,
        journal=journal,
        tier=1,
        blacklist=blacklist,
        rationale=None,
        trial_counter=9,
        enabled=True,
    )

    assert forced == action
    assert rationale is None
    assert reference is None
    assert state["seq_baseline_draw_blocked"]["reason"]
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


def test_seq_promotion_failed_fresh_eval_consumes_pending_attempt() -> None:
    state = {
        "seq_pending_promotion_fresh_eval": {
            "candidate": "candidate-a",
            "source_trial_id": 20,
            "tier": 2,
            "attempts": 1,
        }
    }
    eval_result = autopilot.EvalResult(
        tier=2,
        quality=2.5,
        speed=10.0,
        cost=0.1,
        reliability=1.0,
    )

    autopilot._update_seq_promotion_fresh_eval_state(
        state,
        seq={
            "candidate": "candidate-a",
            "confirmed": False,
            "baseline_reference_state": "fresh",
            "baseline_promotion_combined_E": 0.4,
        },
        action={"type": "deep_eval", "tier": 2},
        eval_result=eval_result,
        trial_counter=21,
        is_fresh_eval=True,
        finalized=False,
    )

    assert "seq_pending_promotion_fresh_eval" not in state
    assert state["seq_last_promotion_blocked"] == {
        "trial_id": 21,
        "candidate": "candidate-a",
        "reason": "fresh-eval did not confirm",
        "combined_E": 0.4,
    }


def test_seq_promotion_finalized_requires_baseline_update_acceptance() -> None:
    state = {
        "seq_pending_promotion_fresh_eval": {
            "candidate": "candidate-a",
            "source_trial_id": 20,
            "tier": 2,
            "attempts": 1,
        }
    }
    eval_result = autopilot.EvalResult(
        tier=2,
        quality=2.5,
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
            "baseline_promotion_combined_E": 120.0,
        },
        action={"type": "deep_eval", "tier": 2},
        eval_result=eval_result,
        trial_counter=21,
        is_fresh_eval=True,
        finalized=True,
        baseline_update=SimpleNamespace(
            updated=False,
            reason="not a monotonic same-tier improvement",
        ),
    )

    assert "seq_pending_promotion_fresh_eval" not in state
    assert "seq_last_promotion_finalized" not in state
    assert state["seq_last_promotion_blocked"] == {
        "trial_id": 21,
        "candidate": "candidate-a",
        "reason": "baseline-update-refused",
        "baseline_update_reason": "not a monotonic same-tier improvement",
        "combined_E": 120.0,
    }


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


def _run_loop_inner_seq_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: dict[str, Any],
    verdict_seq: dict[str, Any],
    force_fresh_eval_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[tuple[bool, int]]]:
    baseline_update_calls: list[tuple[bool, int]] = []

    class FakeJournal:
        def __init__(self) -> None:
            self._entries: list[JournalEntry] = []
            self._promotions: list[dict[str, Any]] = []

        def record(self, entry: JournalEntry) -> None:
            self._entries.append(entry)

        def all_entries(self) -> list[JournalEntry]:
            return list(self._entries)

        def entries_with_supersessions(self) -> list[JournalEntry]:
            return list(self._entries)

        def by_species(self, species: str) -> list[JournalEntry]:
            return [entry for entry in self._entries if entry.species == species]

        def species_effectiveness(self, window: int = 50) -> dict[str, float]:
            return {}

        def baseline_promotion_events(self) -> list[dict[str, Any]]:
            return list(self._promotions)

        def append_baseline_promotion_event(self, **payload: Any) -> dict[str, Any]:
            self._promotions.append(payload)
            return payload

        def supersession_events(self) -> list[dict[str, Any]]:
            return []

    class FakeVerdict:
        def __init__(self, seq: dict[str, Any]) -> None:
            self.seq = dict(seq)
            self.passed = True
            self.categories: list[str] = []
            self.violations: list[str] = []

        def __bool__(self) -> bool:
            return self.passed

    class FakeCriticism:
        keep_or_revert = "keep"

        def as_text(self) -> str:
            return "ok"

        def directions_text(self) -> str:
            return ""

    class FakeSafetyGate:
        def __init__(
            self,
            consecutive_failures: int = 0,
            quality_history: list[Any] | None = None,
            quality_history_by_tier: dict[str, Any] | None = None,
            baseline_state: dict[str, Any] | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            self.consecutive_failures = consecutive_failures
            self.quality_history = quality_history if quality_history is not None else []
            self.quality_history_by_tier = (
                quality_history_by_tier if quality_history_by_tier is not None else {}
            )
            self.baseline = SimpleNamespace(
                quality_for_tier=lambda *_args: 0.0,
                to_state_dict=lambda: baseline_state or {},
            )
            self.use_sequential = True

        def check(self, *args: Any, **kwargs: Any) -> FakeVerdict:
            return FakeVerdict(verdict_seq)

        def analyze_failure(self, *args: Any, **kwargs: Any) -> str:
            return ""

        def should_rollback(self) -> bool:
            return False

        def reset_failures(self) -> None:
            self.consecutive_failures = 0

        def update_baseline(
            self,
            *_args: Any,
            seq_confirmed: bool | None = None,
            source_trial_id: int | None = None,
            **kwargs: Any,
        ) -> Any:
            baseline_update_calls.append((bool(seq_confirmed), int(source_trial_id or 0)))
            return SimpleNamespace(
                updated=True,
                reason="",
                tier=2,
                previous_quality=2.0,
                new_quality=2.5,
                proof=None,
            )

    class FakeMetaOptimizer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.budget = autopilot.SpeciesBudget()

        def select_species(self) -> str:
            return "seed_batch"

        def should_rebalance(self, _trial_counter: int) -> bool:
            return False

        def rebalance(self, *args: Any, **kwargs: Any) -> None:
            return None

    class FakeSeeder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get_memory_count(self) -> int:
            return 0

        def restore_state(self, _state: dict[str, Any]) -> None:
            return None

        def export_state(self) -> dict[str, Any]:
            return {"td_errors": []}

        @property
        def is_converged(self) -> bool:
            return False

    class FakeEvalTower:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def capture_recent_traces(self, _limit: int = 50) -> str:
            return ""

    class FakePromptForge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeStructuralLab:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def restore_checkpoint(self) -> None:
            return None

        def checkpoint_state(self, *args: Any, **kwargs: Any) -> None:
            return None

    class FakeEvolutionManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeParetoArchive:
        def frontier_size(self, *_args: Any, **_kwargs: Any) -> int:
            return 1

        def add(self, *args: Any, **kwargs: Any) -> bool:
            return True

        def update(self, *args: Any, **kwargs: Any) -> str:
            return "frontier"

        def hypervolume_slope(self, *_args: Any, **_kwargs: Any) -> float:
            return 0.0

        def hypervolume(self, *_args: Any, **_kwargs: Any) -> float:
            return 0.0

        def get_frontier(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

        def summary(self) -> dict[str, Any]:
            return {}

    class FakeShortTermMemory:
        def __init__(self) -> None:
            pass

        def refresh_from_journal(self, _journal: Any) -> None:
            return None

    class FakeStrategyStore:
        def count(self) -> int:
            return 0

        def store(self, *args: Any, **kwargs: Any) -> None:
            return None

        def close(self) -> None:
            return None

    class FakePhaseTracker:
        def set(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {}

        def clear(self, *args: Any, **kwargs: Any) -> None:
            return None

    class FakeAsyncTaskRunner:
        def reap(self, *args: Any, **kwargs: Any) -> None:
            return None

        def submit_subprocess(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            return None

        def submit(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def shutdown(self) -> None:
            return None

    fake_journal = FakeJournal()

    def fake_check_blacklist(
        action: dict[str, Any], _blacklist: list[dict[str, Any]]
    ) -> None:
        return None

    def fake_replace_blacklisted_seed_fallback(
        action: dict[str, Any],
        blacklist: list[dict[str, Any]],  # noqa: ARG001
        rationale: dict[str, Any] | None,
        reason_label: str = "",  # noqa: ARG001
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return action, rationale

    def fake_replace_blacklisted_autonomous_action(
        action: dict[str, Any], _blacklist: list[dict[str, Any]], rationale: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return action, rationale

    def fake_enforce_experiment_quota(
        action: dict[str, Any],
        _state: dict[str, Any],
        _memory_count: int,
        rationale: dict[str, Any],
        _trial_counter: int,
        _blacklist: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return action, rationale

    def fake_force_metric_action_after_meta(
        action: dict[str, Any],
        _state: dict[str, Any],
        rationale: dict[str, Any],
        _blacklist: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return action, rationale

    def fake_maybe_force_seq_promotion_fresh_eval(
        action: dict[str, Any],
        state: dict[str, Any],  # noqa: ARG001
        blacklist: list[dict[str, Any]],  # noqa: ARG001
        rationale: dict[str, Any] | None,
        trial_counter: int,  # noqa: ARG001
        enabled: bool,  # noqa: ARG001
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        if force_fresh_eval_context is None:
            return action, rationale, None
        return (
            {"type": "deep_eval", "tier": 2},
            (rationale or {}) | {"seq_promotion_fresh_eval": True},
            dict(force_fresh_eval_context),
        )

    def fake_maybe_force_seq_baseline_draw(
        action: dict[str, Any],
        state: dict[str, Any],  # noqa: ARG001
        journal: Any,  # noqa: ARG001
        tier: int,  # noqa: ARG001
        blacklist: list[dict[str, Any]],  # noqa: ARG001
        rationale: dict[str, Any] | None,
        trial_counter: int,  # noqa: ARG001
        enabled: bool,  # noqa: ARG001
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        return action, rationale, None

    def fake_dispatch_action(*args: Any, **kwargs: Any) -> tuple[Any, str]:
        return (
            autopilot.EvalResult(
                tier=2,
                quality=2.5,
                speed=10.0,
                cost=0.1,
                reliability=1.0,
            ),
            "seed_batch",
        )

    class FakeSelfCriticism:
        def __call__(self, *args: Any, **kwargs: Any) -> FakeCriticism:
            return FakeCriticism()

    monkeypatch.setattr(autopilot, "load_state", lambda: state)
    monkeypatch.setattr(autopilot, "save_state", lambda *args: None)
    monkeypatch.setattr(autopilot, "ExperimentJournal", lambda: fake_journal)
    monkeypatch.setattr(autopilot, "ParetoArchive", FakeParetoArchive)
    monkeypatch.setattr(autopilot, "SafetyGate", FakeSafetyGate)
    monkeypatch.setattr(autopilot, "MetaOptimizer", FakeMetaOptimizer)
    monkeypatch.setattr(autopilot, "Seeder", FakeSeeder)
    monkeypatch.setattr(autopilot, "NumericSwarm", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(autopilot, "PromptForge", FakePromptForge)
    monkeypatch.setattr(autopilot, "StructuralLab", FakeStructuralLab)
    monkeypatch.setattr(autopilot, "EvolutionManager", FakeEvolutionManager)
    monkeypatch.setattr(autopilot, "ShortTermMemory", FakeShortTermMemory)
    monkeypatch.setattr(autopilot, "StrategyStore", FakeStrategyStore)
    monkeypatch.setattr(autopilot, "EvalTower", FakeEvalTower)
    monkeypatch.setattr(autopilot, "AsyncTaskRunner", FakeAsyncTaskRunner)
    monkeypatch.setattr(autopilot, "PhaseTracker", FakePhaseTracker)
    monkeypatch.setattr(autopilot, "check_blacklist", fake_check_blacklist)
    monkeypatch.setattr(
        autopilot,
        "_replace_blacklisted_seed_fallback",
        fake_replace_blacklisted_seed_fallback,
    )
    monkeypatch.setattr(
        autopilot,
        "_replace_blacklisted_autonomous_action",
        fake_replace_blacklisted_autonomous_action,
    )
    monkeypatch.setattr(
        autopilot,
        "_enforce_experiment_quota",
        fake_enforce_experiment_quota,
    )
    monkeypatch.setattr(
        autopilot,
        "_force_metric_action_after_meta",
        fake_force_metric_action_after_meta,
    )
    monkeypatch.setattr(autopilot, "_auto_action", lambda *args, **kwargs: {"type": "seed_batch"})
    monkeypatch.setattr(
        autopilot,
        "_maybe_force_seq_promotion_fresh_eval",
        fake_maybe_force_seq_promotion_fresh_eval,
    )
    monkeypatch.setattr(
        autopilot,
        "_maybe_force_seq_baseline_draw",
        fake_maybe_force_seq_baseline_draw,
    )
    monkeypatch.setattr(autopilot, "dispatch_action", fake_dispatch_action)
    monkeypatch.setattr(autopilot, "_journal_archive_payload_for_authority", lambda *args: None)
    monkeypatch.setattr(autopilot, "_sync_startup_archive_from_journal_authority", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        autopilot,
        "_recover_from_in_flight_trial",
        lambda _state, _journal, _archive, trial_counter: trial_counter,
    )
    monkeypatch.setattr(autopilot, "_save_state_with_journal_archive_authority", lambda *args, **kwargs: None)
    monkeypatch.setattr(autopilot, "_append_baseline_promotion_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(autopilot, "health_check", lambda *args, **kwargs: object())
    monkeypatch.setattr(autopilot, "should_generate_today", lambda _state: False)
    monkeypatch.setattr(autopilot.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(autopilot, "_git_tag", lambda *args, **kwargs: None)
    monkeypatch.setattr(autopilot, "generate_self_criticism", FakeSelfCriticism())
    monkeypatch.setattr(
        autopilot,
        "classify_learning_exclusion",
        lambda *args, **kwargs: (None, "", None),
    )
    monkeypatch.setattr(autopilot.peaf, "compute_surprise", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        autopilot.peaf,
        "actual_objectives_from_eval",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(autopilot, "get_preflight_diagnostics", None)

    autopilot._run_loop_inner(
        max_trials=1,
        dry_run=False,
        use_controller=False,
        tui=None,
    )

    return state, baseline_update_calls


def test_run_loop_inner_forwards_finalized_seq_to_gate_and_clears_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "trial_counter": 0,
        "paused": False,
        "td_errors": [],
        "seeder_state": {},
        "consecutive_failures": 0,
        "quality_history": [],
        "quality_history_by_tier": {},
        "baseline_state": {},
    }

    returned_state, baseline_update_calls = _run_loop_inner_seq_harness(
        monkeypatch,
        state=state,
        verdict_seq={
            "candidate": "candidate-a",
            "confirmed": True,
            "E_quality": 120.0,
            "E_rate_noninf": 120.0,
        },
        force_fresh_eval_context={
            "candidate": "candidate-a",
            "source_trial_id": 13,
        },
    )

    assert baseline_update_calls == [(True, 0)]
    assert returned_state["seq_last_promotion_finalized"] == {
        "trial_id": 0,
        "candidate": "candidate-a",
        "combined_E": 120.0,
        "baseline_update_reason": "",
    }
    assert "seq_pending_promotion_fresh_eval" not in returned_state


def test_run_loop_inner_nonfinalized_seq_does_not_promote_and_leaves_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "trial_counter": 0,
        "paused": False,
        "td_errors": [],
        "seeder_state": {},
        "consecutive_failures": 0,
        "quality_history": [],
        "quality_history_by_tier": {},
        "baseline_state": {},
        "seq_pending_promotion_fresh_eval": {
            "candidate": "candidate-a",
            "source_trial_id": 13,
            "tier": 2,
            "attempts": 3,
        },
    }

    returned_state, baseline_update_calls = _run_loop_inner_seq_harness(
        monkeypatch,
        state=state,
        verdict_seq={
            "candidate": "candidate-a",
            "confirmed": True,
            "E_quality": 90.0,
            "E_rate_noninf": 90.0,
        },
        force_fresh_eval_context=None,
    )

    assert baseline_update_calls == [(False, 0)]
    assert "seq_last_promotion_finalized" not in returned_state
    assert "seq_pending_promotion_fresh_eval" in returned_state
