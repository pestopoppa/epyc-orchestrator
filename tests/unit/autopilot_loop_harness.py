"""Run the real ``autopilot._run_loop_inner`` against in-memory fakes.

Not collected (no ``test_`` prefix). Every collaborator that would reach the host is a fake
or is redirected under a temp dir: the state/lock/outbox/breadcrumb/blacklist paths,
subprocesses, git tags, plots, checkpoints and the reproposal-rate hook. The loop's
process-global hooks are captured or undone after each run (B1-15): SIGINT/SIGTERM handlers
are recorded instead of installed, and the promotion-guard provider and eval-leakage observer
are reset. Each run also asserts it left ORCH_ROOT/orchestration, ORCH_ROOT/logs and the
blacklist file untouched.

``run_loop_inner`` returns a :class:`LoopRun` whose ``events`` list records, in call order,
every call the loop made into an instrumented seam, so tests assert behaviour and ordering
instead of reading autopilot.py as text.
"""

from __future__ import annotations

import ast
import copy
import os
import signal
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "autopilot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import autopilot  # type: ignore[import-not-found]  # noqa: E402
import safety_gate  # type: ignore[import-not-found]  # noqa: E402
from config_applicator import HealthCheckResult  # noqa: E402
from experiment_journal import JournalEntry  # noqa: E402
from planner_coordinator import PlannerCritique, PlannerDecision  # noqa: E402

from src.autopilot_core import mutation_guards  # noqa: E402

__all__ = ["PlannerCritique", "PlannerDecision"]

Event = tuple[str, dict[str, Any]]
# Captured at import so repeated runs in one test wrap the real functions, not a prior spy.
_REAL = SimpleNamespace(
    guard_scope=autopilot._install_promotion_guard_scope,
    reconcile=autopilot._reconcile_promotion_with_journal,
    lineage=autopilot.select_lineage_parent,
    monitor=autopilot.EvalLeakageMonitor,
    signal=signal.signal,
    time=autopilot.time,
)
DEFAULT_SEQ: dict[str, Any] = {
    "candidate": "candidate-a",
    "confirmed": False,
    "state": "accumulating",
}
HARNESS_TARGETS = [{"kind": "harness", "id": "t"}]
lb = autopilot._loop_beliefs


def measured_result(*, tier: int = 2, quality: float = 2.5, n: int = 50) -> Any:
    """An EvalResult carrying every live dominance axis, so the loop's ``rate_measured`` holds."""
    return autopilot.EvalResult(
        tier=tier,
        quality=quality,
        speed=10.0,
        cost=0.2,
        reliability=0.98,
        per_suite_quality={"coder": quality},
        n_questions=n,
        eval_wall_s=900.0,
        question_results=[{"qid": f"q{i}", "correct": True} for i in range(n)],
    )


def fresh_state(**extra: Any) -> dict[str, Any]:
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
    state.update(extra)
    return state


def multitier_final_t1_state(candidate_action: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A ratified multitier bundle whose candidate is due its final T1 confirmation."""
    from src.autopilot_core.multitier_decision import (
        MULTITIER_POLICY_VERSION,
        build_tier_baseline_evidence,
    )

    def evidence(tier: int) -> dict[str, Any]:
        rows = [{"qid": f"t{tier}q{i}", "correct": True} for i in range(10)]
        return build_tier_baseline_evidence(
            SimpleNamespace(
                tier=tier,
                quality=3.0,
                reliability=1.0,
                core_id=f"core-t{tier}",
                dataset_content_sha256=f"dataset-t{tier}",
                test_profile=f"profile-t{tier}",
                question_results=rows,
                details={},
            )
        )

    return fresh_state(
        multitier_promotion_policy={"enabled": True, "policy_version": MULTITIER_POLICY_VERSION},
        **{
            autopilot.MULTITIER_BASELINE_STATE_KEY: {
                "policy_version": MULTITIER_POLICY_VERSION,
                "tiers": {"2": evidence(2), "3": evidence(3)},
            },
            autopilot.MULTITIER_PENDING_STATE_KEY: {
                "policy_version": MULTITIER_POLICY_VERSION,
                "status": "pending",
                "candidate": "cand",
                "candidate_action": dict(candidate_action),
                "source_trial_id": 7,
                "next_tier": "final_t1",
                "higher_tier_improvements": [2, 3],
                "final_t1_attempts": 0,
            },
        },
        **extra,
    )


def planner_decision(action: dict[str, Any] | None, **fields: Any) -> PlannerDecision:
    values: dict[str, Any] = {
        "rationale": {"falsifier": "f"},
        "canonical_text": "",
        "draft_text": "",
        "draft_provider": "local",
        "mode": "draft_critique",
    }
    values.update(fields)
    return PlannerDecision(action=action, **values)


def autopilot_calls(name: str) -> list[tuple[str, ast.Call]]:
    """``(enclosing function, call)`` for every call of ``name`` or ``*.name`` in autopilot.py.

    For structural rules that must hold at every call site however the loop is split into
    helpers or reformatted (an AST walk, never a text slice of ``_run_loop_inner``).
    """
    tree = ast.parse(Path(autopilot.__file__).read_text())
    found: list[tuple[str, ast.Call]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            )
            if isinstance(child, ast.Call):
                func = child.func
                callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if callee == name:
                    found.append((scope, child))
            visit(child, inner)

    visit(tree, "<module>")
    return found


@dataclass
class LoopRun:
    state: dict[str, Any]
    events: list[Event]
    baseline_update_calls: list[tuple[bool, int]]
    handlers: dict[int, Any]

    def of(self, name: str) -> list[dict[str, Any]]:
        return [payload for event, payload in self.events if event == name]

    def index(self, name: str, start: int = 0) -> int:
        for i in range(start, len(self.events)):
            if self.events[i][0] == name:
                return i
        raise AssertionError(f"no {name!r} event at or after {start}: {self.names}")

    @property
    def names(self) -> list[str]:
        return [event for event, _ in self.events]

    @property
    def phases(self) -> list[str]:
        return [payload["name"] for payload in self.of("phase")]

    @property
    def journal_entries(self) -> list[JournalEntry]:
        return [payload["entry"] for payload in self.of("record")]

    @property
    def dispatched(self) -> list[dict[str, Any]]:
        return [payload["action"] for payload in self.of("dispatch")]

    def saves(self) -> list[tuple[int, dict[str, Any]]]:
        """``(event index, payload)`` of every state persistence, plain or journal-authority."""
        return [
            (i, p) for i, (n, p) in enumerate(self.events) if n in ("save_state", "save_authority")
        ]


def _scripted(items: Iterable[Any] | None, default: Any) -> Callable[[], Any]:
    """Successive items; the last one repeats. ``None`` -> always ``default``."""
    queue = list(items) if items is not None else [default]
    position = {"i": 0}

    def _next() -> Any:
        item = queue[min(position["i"], len(queue) - 1)]
        position["i"] += 1
        return item

    return _next


def _host_snapshot() -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    for directory in (autopilot.ORCH_ROOT / "orchestration", autopilot.ORCH_ROOT / "logs"):
        if directory.is_dir():
            for entry in os.scandir(directory):
                info = entry.stat(follow_symlinks=False)
                snap[entry.path] = (info.st_mtime_ns, info.st_size)
    blacklist = autopilot.SCRIPT_DIR / "failure_blacklist.yaml"
    if blacklist.exists():
        info = blacklist.stat()
        snap[str(blacklist)] = (info.st_mtime_ns, info.st_size)
    return snap


@contextmanager
def _sandbox(tmp_path: Path | None) -> Iterator[Path]:
    if tmp_path is not None:
        yield tmp_path
        return
    with tempfile.TemporaryDirectory(prefix="autopilot-loop-") as tmp:
        yield Path(tmp)


def _ready_answer(evidence: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema": lb.ANSWER_SCHEMA,
        "consumer": lb.CONSUMER,
        "as_of": "2026-10-21T12:00:00+00:00",
        "status": "ready",
        "reason": None,
        "known_good_path": "",
        "targets": [],
        "ledger_state": {
            "schema": lb.LEDGER_STATE_SCHEMA,
            "ledger_path": "/nonexistent/ledger.jsonl",
            "frame_count": 1,
            "head_seq": 0,
            "head_hash": "sha256:" + "ab" * 32,
            "verified_at": "2026-10-21T12:00:00Z",
            "fresh": True,
            "freshness": {"sync_action": "already-fresh", "was_fresh": True, "frames_emitted": 0},
        },
        "ledger_state_digest": "sha256:" + "cd" * 32,
        "evidence": {
            "status": "omitted",
            "reasons": ["inapplicable_target"],
            "claim_ids": [],
            "contested_claim_ids": [],
            "withdrawn_claim_ids": [],
            "section": "",
            "section_sha256": "",
            **(evidence or {}),
        },
    }


def run_loop_inner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: dict[str, Any],
    verdict_seq: dict[str, Any] | None = DEFAULT_SEQ,
    verdict_passed: bool = True,
    should_rollback: bool = False,
    baseline_updated: bool = True,
    force_fresh_eval_context: dict[str, Any] | None = None,
    seq_gate_preflight_payload: dict[str, Any] | None = None,
    learning_exclusion: tuple[str | None, str, Any] = (None, "", None),
    use_controller: bool = False,
    planner_should_not_run: bool = False,
    planner_decisions: Iterable[PlannerDecision] | None = None,
    belief_refused: str = "",
    belief_evidence: dict[str, Any] | None = None,
    auto_actions: Iterable[dict[str, Any]] | None = None,
    blacklisted: str = "",
    higher_tier_probe: bool = True,
    dispatch_results: Iterable[Any] | None = None,
    health: Iterable[bool] | None = None,
    stop_after_sleeps: int | None = None,
    multitier: bool = False,
    max_trials: int | None = 1,
    journal_entries_out: list[JournalEntry] | None = None,
    dispatch_actions_out: list[dict[str, Any]] | None = None,
    gate_check_raises_on_seq: bool = False,
    tmp_path: Path | None = None,
    real_state_io: bool = False,
    planner_hook: Callable[[PlannerDecision, dict[int, Any]], None] | None = None,
    selection_hook: Callable[[dict[str, Any], dict[int, Any]], None] | None = None,
    breadcrumb: Any | None = None,
) -> LoopRun:
    """Drive ``_run_loop_inner`` with scripted collaborators.

    ``dispatch_results`` items are an outcome (EvalResult | SkipOutcome | None), an
    ``(outcome, species)`` tuple, or ``callable(action, state)`` returning either.
    ``auto_actions`` (autonomous mode), ``planner_decisions`` (controller mode), ``health``
    and ``dispatch_results`` are scripts whose last item repeats. ``stop_after_sleeps``
    delivers SIGTERM to the loop's own handler on that sleep. ``higher_tier_probe=False``
    turns off the forcer that rewrites any action into a T2 deep_eval on an empty journal.
    """
    events: list[Event] = []
    baseline_update_calls: list[tuple[bool, int]] = []
    installed: list[Any] = []
    handlers: dict[int, Any] = {}
    sleeps: list[float] = []

    def emit(event: str, /, **payload: Any) -> None:
        events.append((event, payload))

    def marker() -> Any:
        flight = state.get("in_flight_trial")
        return flight.get("trial_id") if isinstance(flight, dict) else flight

    class FakeJournal:
        def __init__(self) -> None:
            self._entries: list[JournalEntry] = []
            self._promotions: list[dict[str, Any]] = []

        def record(self, entry: JournalEntry) -> None:
            self._entries.append(entry)
            if journal_entries_out is not None:
                journal_entries_out.append(entry)
            emit(
                "record",
                entry=entry,
                eval_details=copy.deepcopy(entry.eval_details),
                belief_scope=copy.deepcopy(entry.belief_scope),
            )

        def all_entries(self) -> list[JournalEntry]:
            return list(self._entries)

        def entries_with_supersessions(self) -> list[JournalEntry]:
            return list(self._entries)

        def by_species(self, species: str) -> list[JournalEntry]:
            return [entry for entry in self._entries if entry.species == species]

        def recent(self, n: int = 10) -> list[JournalEntry]:
            return list(self._entries[-n:])

        def species_effectiveness(self, window: int = 50) -> dict[str, float]:
            return {}

        def summary_text(self, *args: Any, **kwargs: Any) -> str:
            return ""

        def suite_quality_trend(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {}

        def baseline_promotion_events(self) -> list[dict[str, Any]]:
            return list(self._promotions)

        def baseline_ledger_events(self) -> list[dict[str, Any]]:
            return list(self._promotions)

        def append_baseline_promotion_event(self, **payload: Any) -> dict[str, Any]:
            self._promotions.append(payload)
            return payload

        def supersession_events(self) -> list[dict[str, Any]]:
            return []

    class FakeVerdict:
        def __init__(self, seq: dict[str, Any] | None) -> None:
            self.seq = dict(seq) if seq is not None else None
            self.passed = verdict_passed
            self.categories: list[str] = []
            self.violations: list[str] = [] if verdict_passed else ["harness: verdict failed"]
            self.warnings: list[str] = []
            # Mirror safety_gate.SafetyVerdict (ETR-5 0399c3fef: reliability_blocked /
            # quality_unmeasured; autopilot.py reads `retry_not_revert` since 15e5b52ec).
            self.reliability_blocked = False
            self.quality_unmeasured = False

        @property
        def retry_not_revert(self) -> bool:
            return self.reliability_blocked or self.quality_unmeasured

        def __bool__(self) -> bool:
            return self.passed

    class FakeCriticism:
        keep_or_revert = "keep"

        def as_text(self) -> str:
            return "ok"

        def directions_text(self) -> str:
            return ""

    class FakeBaseline:
        def __init__(self, state_dict: dict[str, Any] | None) -> None:
            self.state_dict = dict(state_dict or {})

        def quality_for_tier(self, *args: Any, **kwargs: Any) -> float:
            return 0.0

        def to_state_dict(self) -> dict[str, Any]:
            return dict(self.state_dict)

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
            emit("safety_gate")
            self.consecutive_failures = consecutive_failures
            self.quality_history = quality_history if quality_history is not None else []
            # A copy, so state only reflects the gate when the loop writes it back.
            self.quality_history_by_tier = copy.deepcopy(quality_history_by_tier or {})
            # Defect #4: real SafetyGate now exposes the provenance-bearing window that
            # _run_loop_inner persists to state["quality_history_provenance_by_tier"].
            self.quality_history_provenance_by_tier = (
                kwargs.get("quality_history_provenance_by_tier") or {}
            )
            self.baseline = FakeBaseline(baseline_state)
            self.use_sequential = True

        def check(self, eval_result: Any, *args: Any, **kwargs: Any) -> FakeVerdict:
            # SEQ-3b: when opted in, raise on the SEQUENTIAL check (the call that
            # threads baseline_profile) but succeed on the bare legacy fallback call
            # (`gate.check(eval_result)`, no kwargs) so the main loop must catch and
            # fall back instead of crashing.
            if gate_check_raises_on_seq and "baseline_profile" in kwargs:
                raise RuntimeError("corrupt sequential inputs (SEQ-3b test)")
            if not verdict_passed:
                self.consecutive_failures += 1
            self.quality_history_by_tier.setdefault(str(eval_result.tier), []).append(
                eval_result.quality
            )
            return FakeVerdict(verdict_seq)

        def analyze_failure(self, *args: Any, **kwargs: Any) -> str:
            return ""

        def should_rollback(self) -> bool:
            return should_rollback

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
            emit(
                "update_baseline",
                source_trial_id=source_trial_id,
                seq_confirmed=seq_confirmed,
                **kwargs,
            )
            if baseline_updated:  # the decision moves the in-memory baseline, as the real one does
                self.baseline.state_dict["promoted_by_trial"] = source_trial_id
            return safety_gate.BaselineUpdateResult(
                updated=baseline_updated, reason="", tier=2, previous_quality=2.0, new_quality=2.5
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

        def convergence_status(self) -> dict[str, Any]:
            return {}

        @property
        def is_converged(self) -> bool:
            return False

    class FakeEvalTower:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def capture_recent_traces(self, _limit: int = 50) -> str:
            return ""

    class FakeStructuralLab:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def restore_checkpoint(self) -> None:
            emit("restore_checkpoint")

        def checkpoint_state(self, *args: Any, **kwargs: Any) -> None:
            emit("checkpoint", **kwargs)

        def current_flags(self) -> dict[str, Any]:
            return {}

    class FakeParetoArchive:
        def frontier_size(self, *_args: Any, **_kwargs: Any) -> int:
            return 1

        def add(self, *args: Any, **kwargs: Any) -> bool:
            return True

        def update(self, *args: Any, **kwargs: Any) -> str:
            return "frontier"

        def upsert_representative(self, *args: Any, **kwargs: Any) -> tuple[str, list[float]]:
            return "frontier", [0.0, 0.0, 0.0, 0.0]

        def reproduction_count(self, *args: Any, **kwargs: Any) -> int:
            return 1

        def hypervolume_slope(self, *_args: Any, **_kwargs: Any) -> float:
            return 0.0

        def hypervolume(self, *_args: Any, **_kwargs: Any) -> float:
            return 0.0

        def get_frontier(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

        def summary(self) -> dict[str, Any]:
            return {}

        def summary_text(self, *args: Any, **kwargs: Any) -> str:
            return ""

    class FakeShortTermMemory:
        def refresh_from_journal(self, _journal: Any) -> str:
            return ""

    class FakeStrategyStore:
        def count(self) -> int:
            return 0

        def store(self, *args: Any, **kwargs: Any) -> None:
            return None

        def close(self) -> None:
            return None

    class FakePhaseTracker:
        def set(self, phase: str, **fields: Any) -> dict[str, Any]:
            emit("phase", name=phase, **fields)
            return {}

        def clear(self, *args: Any, **kwargs: Any) -> None:
            return None

    class FakeAsyncTaskRunner:
        def reap(self, *args: Any, **kwargs: Any) -> None:
            return None

        def submit_subprocess(self, name: str, *args: Any, **kwargs: Any) -> None:
            emit("subprocess", task=name)

        def submit(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def shutdown(self) -> None:
            return None

    class SpyLeakageMonitor(_REAL.monitor):
        def install(self) -> Any:
            emit("leakage_install")
            installed.append(self)
            return super().install()

        def preflight(self) -> bool:
            emit("leakage_preflight")
            return super().preflight()

    fake_journal = FakeJournal()
    next_health = _scripted(health, True)
    next_auto_action = _scripted(auto_actions, {"type": "seed_batch"})
    next_decision = _scripted(planner_decisions, None)
    next_result = _scripted(
        dispatch_results,
        autopilot.EvalResult(tier=2, quality=2.5, speed=10.0, cost=0.1, reliability=1.0),
    )

    def fake_save_state(*args: Any, **kwargs: Any) -> None:
        emit(
            "save_state",
            in_flight=marker(),
            paused=bool(state.get("paused")),
            merge_control=bool(kwargs.get("merge_control")),
        )

    def fake_save_authority(*args: Any, context: str = "", **kwargs: Any) -> None:
        emit(
            "save_authority", context=context, in_flight=marker(), paused=bool(state.get("paused"))
        )

    def fake_dispatch_action(action: dict[str, Any], *args: Any, **kwargs: Any) -> tuple[Any, str]:
        if dispatch_actions_out is not None:
            dispatch_actions_out.append(dict(action))
        flight = state.get("in_flight_trial")
        emit(
            "dispatch",
            action=dict(action),
            in_flight=marker(),
            run_manifest=copy.deepcopy(flight.get("run_manifest"))
            if isinstance(flight, dict)
            else None,
            served_content_present=autopilot.SERVED_CONTENT_STATE_KEY in state,
        )
        result = next_result()
        if callable(result):
            result = result(action, state)
        if isinstance(result, tuple):
            return result
        return result, "seed_batch"

    def fake_health_check(*args: Any, **kwargs: Any) -> Any:
        healthy = bool(next_health())
        return HealthCheckResult(ok=healthy, failure_reason="" if healthy else "connect_error")

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        emit("sleep", seconds=seconds)
        if stop_after_sleeps is not None and len(sleeps) >= stop_after_sleeps:
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    def capture_signal(signum: int, handler: Any) -> Any:
        if signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = handler
            return signal.getsignal(signum)
        return _REAL.signal(signum, handler)

    def fake_plan_with_providers(*args: Any, **kwargs: Any) -> PlannerDecision:
        if planner_should_not_run:
            raise AssertionError("planner should be bypassed for forced seq action")
        decision = next_decision()
        if decision is None:
            raise AssertionError("controller turn reached the planner without planner_decisions")
        emit("plan", belief_context=kwargs.get("belief_context"))
        if planner_hook is not None:
            planner_hook(copy.deepcopy(decision), handlers)
        return copy.deepcopy(decision)

    def fake_prepare(settings: Any, targets: Any = None, **kwargs: Any) -> Any:
        emit("belief_prepare", targets=targets)
        if belief_refused:
            return lb.PrepareOutcome("refused", None, belief_refused, "harness refusal")
        answer = _ready_answer(belief_evidence)
        return lb.PrepareOutcome("ready", lb.validate_ready(copy.deepcopy(answer)), answer=answer)

    def fake_derive_targets(**kwargs: Any) -> list[dict[str, Any]]:
        emit("derive_targets", kwargs=sorted(kwargs))
        return list(HARNESS_TARGETS)

    def spy_guard_scope(journal: Any, live_state: Any) -> None:
        emit("guard_scope")
        _REAL.guard_scope(journal, live_state)

    def spy_reconcile(gate: Any, update: Any, *args: Any) -> Any:
        result = _REAL.reconcile(gate, update, *args)
        emit("reconcile", rolled_back=result is not update)
        return result

    def spy_lineage(journal: Any, species: str, *, before_trial_id: int) -> Any:
        lineage = _REAL.lineage(journal, species, before_trial_id=before_trial_id)
        emit("lineage", before_trial_id=before_trial_id, lineage=copy.deepcopy(lineage))
        return lineage

    def spy_promotion_event(*, baseline_update: Any, source_trial_id: int, **kwargs: Any) -> None:
        emit(
            "promotion_event",
            source_trial_id=source_trial_id,
            updated=bool(getattr(baseline_update, "updated", False)),
        )

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
        forced_action = {"type": "deep_eval", "tier": 2}
        if selection_hook is not None:
            selection_hook(copy.deepcopy(forced_action), handlers)
        return (
            forced_action,
            (rationale or {}) | {"seq_promotion_fresh_eval": True},
            dict(force_fresh_eval_context),
        )

    def fake_maybe_defer_seq_unreachable_candidate_action(
        action: dict[str, Any], *, state: dict[str, Any], rationale: Any, **_kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        payload = dict(seq_gate_preflight_payload or {})
        payload.setdefault("original_action", dict(action))
        state["seq_gate_reachability_preflight"] = payload
        return action, rationale, payload

    def no_seq_context(action: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return action, kwargs.get("rationale"), None

    m = monkeypatch.setattr
    if not real_state_io:
        m(autopilot, "load_state", lambda: state)
        m(autopilot, "save_state", fake_save_state)
    m(autopilot, "_save_state_with_journal_archive_authority", fake_save_authority)
    m(autopilot, "ExperimentJournal", lambda: fake_journal)
    m(autopilot, "ParetoArchive", FakeParetoArchive)
    m(autopilot, "SafetyGate", FakeSafetyGate)
    m(autopilot, "MetaOptimizer", FakeMetaOptimizer)
    m(autopilot, "Seeder", FakeSeeder)
    m(autopilot, "NumericSwarm", lambda *_args, **_kwargs: [])
    m(autopilot, "PromptForge", lambda *_args, **_kwargs: SimpleNamespace())
    m(autopilot, "StructuralLab", FakeStructuralLab)
    m(autopilot, "EvolutionManager", lambda *_args, **_kwargs: SimpleNamespace())
    m(autopilot, "ShortTermMemory", FakeShortTermMemory)
    m(autopilot, "StrategyStore", FakeStrategyStore)
    m(autopilot, "EvalTower", FakeEvalTower)
    m(autopilot, "AsyncTaskRunner", FakeAsyncTaskRunner)
    m(autopilot, "PhaseTracker", FakePhaseTracker)
    m(autopilot, "EvalLeakageMonitor", SpyLeakageMonitor)
    m(autopilot, "_install_promotion_guard_scope", spy_guard_scope)
    m(autopilot, "_reconcile_promotion_with_journal", spy_reconcile)
    m(autopilot, "select_lineage_parent", spy_lineage)
    m(autopilot, "_append_baseline_promotion_event", spy_promotion_event)
    m(autopilot, "_record_reproposal_rate_windows", lambda _journal: emit("reproposal_rate"))
    m(autopilot, "append_blacklist", lambda *a, **kw: emit("append_blacklist", **kw))
    m(autopilot, "_append_operator_outbox_item", lambda *a, **kw: False)
    m(autopilot, "check_blacklist", lambda action, blacklist: blacklisted or None)
    m(
        autopilot,
        "_replace_blacklisted_seed_fallback",
        lambda action, blacklist, rationale, reason_label="": (action, rationale),
    )
    m(
        autopilot,
        "_replace_blacklisted_autonomous_action",
        lambda action, blacklist, rationale: (action, rationale),
    )
    m(
        autopilot,
        "_enforce_experiment_quota",
        lambda action, state, memory_count, rationale, trial, blacklist: (action, rationale),
    )
    m(
        autopilot,
        "_force_metric_action_after_meta",
        lambda action, state, rationale, blacklist: (action, rationale),
    )

    def fake_auto_action(*args: Any, **kwargs: Any) -> dict[str, Any]:
        emit("auto_action")
        action = dict(next_auto_action())
        if selection_hook is not None:
            selection_hook(copy.deepcopy(action), handlers)
        return action

    m(autopilot, "_auto_action", fake_auto_action)
    m(autopilot, "_maybe_force_seq_promotion_fresh_eval", fake_maybe_force_seq_promotion_fresh_eval)
    m(autopilot, "_maybe_force_seq_baseline_draw", no_seq_context)
    if seq_gate_preflight_payload is not None:
        m(
            autopilot,
            "_maybe_defer_seq_unreachable_candidate_action",
            fake_maybe_defer_seq_unreachable_candidate_action,
        )
    if not higher_tier_probe:
        m(autopilot, "HIGHER_TIER_PROBE_GUARD", False)
    m(autopilot, "dispatch_action", fake_dispatch_action)
    m(autopilot, "plan_with_providers", fake_plan_with_providers)
    m(autopilot, "_journal_archive_payload_for_authority", lambda *args, **kwargs: None)
    m(autopilot, "_sync_startup_archive_from_journal_authority", lambda *args, **kwargs: False)
    m(
        autopilot,
        "_recover_from_in_flight_trial",
        lambda _state, _journal, _archive, trial_counter: trial_counter,
    )
    m(autopilot, "health_check", fake_health_check)
    m(autopilot, "should_generate_today", lambda _state: False)

    def fake_subprocess_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        emit("subprocess_run", cmd=list(cmd))
        # subprocess is a shared module object, so returning None here also breaks
        # newly-added read-only planner bridges that consume CompletedProcess output.
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="subprocesses disabled by the loop harness",
        )

    m(autopilot.subprocess, "run", fake_subprocess_run)
    m(autopilot, "_git_tag", lambda *args, **kwargs: None)
    m(autopilot, "generate_self_criticism", lambda *args, **kwargs: FakeCriticism())
    m(autopilot, "classify_learning_exclusion", lambda *args, **kwargs: learning_exclusion)
    m(autopilot.peaf, "compute_surprise", lambda *args, **kwargs: 0.0)
    m(autopilot.peaf, "actual_objectives_from_eval", lambda *args, **kwargs: {})
    m(autopilot, "get_preflight_diagnostics", None)
    m(autopilot, "time", SimpleNamespace(**{**vars(_REAL.time), "sleep": fake_sleep}))
    m(signal, "signal", capture_signal)
    m(autopilot, "MULTITIER_PROMOTION_ENABLED", multitier)
    m(lb, "prepare", fake_prepare)
    m(lb, "derive_targets", fake_derive_targets)
    m(autopilot, "load_model_signatures", lambda: {})

    before = _host_snapshot()
    try:
        with _sandbox(tmp_path) as sandbox:
            for name, filename in (
                ("STATE_PATH", "autopilot_state.json"),
                ("LOCK_PATH", ".autopilot.lock"),
                ("OPERATOR_OUTBOX_PATH", "autopilot_operator_outbox.jsonl"),
                ("EXIT_BREADCRUMB_PATH", "autopilot_exit_breadcrumb.jsonl"),
                ("BLACKLIST_PATH", "failure_blacklist.yaml"),
            ):
                m(autopilot, name, sandbox / filename)
            if real_state_io:
                autopilot.save_state(state)
            belief_settings = lb.BeliefSettings(
                root_repo=sandbox / "root",
                degraded_log=sandbox / "degraded.jsonl",
                review_flags=sandbox / "review.jsonl",
            )
            autopilot._run_loop_inner(
                max_trials, False, use_controller, None, breadcrumb, belief_settings
            )
    finally:
        for monitor in installed:
            monitor.uninstall()
        safety_gate.configure_promotion_guard_archive(None)
    assert _host_snapshot() == before, "the loop harness wrote to the live tree"
    assert mutation_guards._EVAL_LEAKAGE_OBSERVER is None
    return LoopRun(state, events, baseline_update_calls, handlers)


def _run_loop_inner_seq_harness(
    monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[dict[str, Any], list[tuple[bool, int]]]:
    """The original callers' shape: ``(state, [(seq_confirmed, source_trial_id), ...])``."""
    run = run_loop_inner(monkeypatch, **kwargs)
    return run.state, run.baseline_update_calls
