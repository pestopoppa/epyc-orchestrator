"""Tests for the extracted autopilot.actions module dispatcher."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

actions = importlib.import_module("actions")
autopilot = importlib.import_module("autopilot")


def _ctx(**overrides):
    """Build a minimal _ActionContext with all the fields zeroed/None unless overridden."""
    defaults = dict(
        seeder=None, swarm=None, forge=None, lab=None, tower=None,
        gate=None, archive=None, journal=None, state={}, strategy_store=None, evo=None,
    )
    defaults.update(overrides)
    return actions._ActionContext(**defaults)


# ----- dispatcher routing -----


def test_dispatcher_rejects_ap9_scope_violation(caplog) -> None:
    # numeric_trial with 2 explicit params violates AP-9. The dispatcher now
    # returns a structured SkipOutcome (not bare None) so the main loop can
    # journal/count/feed-back the reason.
    result, species = actions.dispatch_action(
        {"type": "numeric_trial", "params": {"a": 1, "b": 2}},
        seeder=None, swarm=None, forge=None, lab=None, tower=None,
        gate=None, archive=None, journal=None, state={},
    )
    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "skipped"
    assert "AP-9" in result.reason
    assert species == "numeric_trial"


def test_dispatcher_rejects_deep_eval_sampling_knobs(monkeypatch) -> None:
    def fail_handler(action, ctx):  # noqa: ANN001, ARG001
        raise AssertionError("deep_eval handler should not run for invalid schema")

    monkeypatch.setitem(actions._ACTION_HANDLERS, "deep_eval", fail_handler)

    result, species = actions.dispatch_action(
        {"type": "deep_eval", "tier": 2, "n_questions": 7, "seed": 999},
        seeder=None, swarm=None, forge=None, lab=None, tower=None,
        gate=None, archive=None, journal=None, state={},
    )

    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "skipped"
    assert "AP-9" in result.reason
    assert "unsupported keys" in result.reason
    assert species == "deep_eval"


def test_dispatcher_unknown_action_type() -> None:
    result, species = actions.dispatch_action(
        {"type": "nonexistent_action"},
        seeder=None, swarm=None, forge=None, lab=None, tower=None,
        gate=None, archive=None, journal=None, state={},
    )
    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "skipped"
    assert "unknown action type" in result.reason
    assert species == "unknown"


def test_blacklisted_action_becomes_invalid_skip() -> None:
    result = autopilot._blacklisted_action_skip(
        {"type": "seed_batch", "n_questions": 10},
        "Auto-blacklisted: 3 consecutive failures",
    )
    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "invalid"
    assert result.reason == "action blacklisted: Auto-blacklisted: 3 consecutive failures"
    assert result.action_type == "seed_batch"


def test_blacklist_prompt_includes_older_enforced_patterns(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "BLACKLIST_RENDER_CAP", 2)

    text = autopilot._format_blacklist_for_prompt([
        {
            "pattern": {
                "type": "structural_experiment",
                "flags": {"skillbank": True},
            },
            "reason": "old hidden reason",
            "source_trial": 505,
        },
        {
            "pattern": {"type": "numeric_trial", "surface": "monitor"},
            "reason": "recent monitor reason",
            "source_trial": 747,
        },
        {
            "pattern": {"type": "train_routing_models"},
            "reason": "recent routing reason",
            "source_trial": 781,
        },
    ])

    assert "recent monitor reason" in text
    assert "recent routing reason" in text
    assert "Older enforced patterns" in text
    assert '{"flags":{"skillbank":true},"type":"structural_experiment"}' in text
    assert "source_trial=505" in text


def test_structural_experiment_invalid_flags_returns_skip_outcome() -> None:
    """Invalid flag dependency surfaces the validator reason as a SkipOutcome,
    not a bare None — this is the graph_router-deadlock fix."""
    class FakeLab:
        def propose_flag_experiment(self, flags):
            return {
                "status": "invalid",
                "errors": ["graph_router feature requires specialist_routing feature"],
                "proposed_flags": flags,
            }

    result, species = actions._action_structural_experiment(
        {"type": "structural_experiment", "flags": {"graph_router": True}},
        _ctx(lab=FakeLab()),
    )
    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "invalid"
    assert "specialist_routing" in result.reason
    assert species == "structural_lab"


def test_structural_experiment_error_status_is_skipped_not_invalid() -> None:
    """A transient validator 'error' (e.g. orchestrator unreachable) maps to a
    non-blacklisting 'skipped' SkipOutcome, never 'invalid' — so a blip cannot
    permanently blacklist a valid flag."""
    class FakeLab:
        def propose_flag_experiment(self, flags):
            return {"status": "error", "error": "live flag state unavailable"}

    result, species = actions._action_structural_experiment(
        {"type": "structural_experiment", "flags": {"graph_router": True}},
        _ctx(lab=FakeLab()),
    )
    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "skipped"
    assert "unavailable" in result.reason
    assert species == "structural_lab"


def test_dispatcher_routes_to_correct_handler(monkeypatch) -> None:
    """Smoke test that each registered handler is callable from dispatch_action."""
    captured = {}

    def fake_handler(action, ctx):
        captured["action"] = action
        captured["ctx_type"] = type(ctx).__name__
        return ("EVAL_SENTINEL", "test_species")

    monkeypatch.setitem(actions._ACTION_HANDLERS, "seed_batch", fake_handler)

    result, species = actions.dispatch_action(
        {"type": "seed_batch", "n_questions": 5},
        seeder="seeder_obj", swarm="swarm_obj", forge=None, lab=None, tower=None,
        gate=None, archive=None, journal=None, state={},
    )
    assert result == "EVAL_SENTINEL"
    assert species == "test_species"
    assert captured["action"]["n_questions"] == 5
    assert captured["ctx_type"] == "_ActionContext"


def test_dispatcher_sets_tower_trial_context(monkeypatch) -> None:
    captured = {}

    class FakeTower:
        def set_trial_context(self, trial_id):
            captured["trial_id"] = trial_id

    def fake_handler(action, ctx):
        captured["handler_saw_trial_id"] = captured.get("trial_id")
        return ("EVAL_SENTINEL", "test_species")

    monkeypatch.setitem(actions._ACTION_HANDLERS, "seed_batch", fake_handler)

    result, species = actions.dispatch_action(
        {"type": "seed_batch", "n_questions": 5},
        seeder="seeder_obj", swarm="swarm_obj", forge=None, lab=None, tower=FakeTower(),
        gate=None, archive=None, journal=None, state={"trial_counter": 817},
    )

    assert result == "EVAL_SENTINEL"
    assert species == "test_species"
    assert captured["trial_id"] == 817
    assert captured["handler_saw_trial_id"] == 817


def test_action_handlers_registered_for_all_known_types() -> None:
    """Sanity check: every documented action type has a handler."""
    expected = {
        "seed_batch", "numeric_trial", "prompt_mutation", "gepa_optimize",
        "code_mutation", "structural_experiment", "structural_prune",
        "train_routing_models", "distill_skillbank", "reset_memories",
        "deep_eval", "rollback", "distill_knowledge", "slot_compact",
    }
    assert expected == set(actions._ACTION_HANDLERS.keys())


def test_bsv2_payload_uses_avg_prompt_tokens_not_instruction_tokens() -> None:
    result = actions.EvalResult(
        tier=1,
        quality=2.1,
        speed=50.0,
        cost=0.5,
        reliability=1.0,
        avg_prompt_tokens=321,
        instruction_token_count=9999,
    )

    payload = actions._bsv2_eval_payload(
        result,
        label="candidate",
        artifact_kind="prompt",
        target="frontdoor.md",
        mutation_type="compress",
    )

    assert payload["avg_prompt_tokens"] == 321


# ----- individual action handler unit (seed_batch is the simplest) -----


def test_seed_batch_handler_runs_eval_after_seed() -> None:
    """seed_batch must call seeder.run_batch then tower.hybrid_eval."""
    calls = []

    class FakeSeeder:
        def run_batch(self, *, n_questions, suites, watcher=None):
            calls.append(("seed", n_questions, suites))
            return None

    class FakeTower:
        def hybrid_eval(self):
            calls.append(("eval",))
            return "EVAL_RESULT"

    ctx = _ctx(seeder=FakeSeeder(), tower=FakeTower())
    result, species = actions._action_seed_batch(
        {"type": "seed_batch", "n_questions": 12, "suites": ["math"]}, ctx
    )
    assert result == "EVAL_RESULT"
    assert species == "seeder"
    assert calls == [("seed", 12, ["math"]), ("eval",)]


def test_deep_eval_handler_calls_tower_evaluate_with_tier() -> None:
    class FakeTower:
        def __init__(self):
            self.calls = []

        def evaluate(self, **kwargs):
            self.calls.append(kwargs)
            return "DEEP_EVAL"

    tower = FakeTower()
    result, species = actions._action_deep_eval(
        {"type": "deep_eval", "tier": 2, "n": 7, "n_questions": 7, "seed": 999},
        _ctx(tower=tower),
    )
    assert result == "DEEP_EVAL"
    assert species == "seeder"
    assert tower.calls == [{"tier": 2}]


def _eval_result(
    *,
    quality: float = 1.0,
    per_suite_quality: dict[str, float] | None = None,
    question_results: list[dict] | None = None,
) -> actions.EvalResult:
    return actions.EvalResult(
        tier=1,
        quality=quality,
        speed=10.0,
        cost=0.0,
        reliability=1.0,
        per_suite_quality=per_suite_quality or {},
        question_results=question_results or [],
    )


class _AlwaysPassGate:
    def check(self, result):
        return True


class _FakeJournal:
    def recent_failures(self, *, species, n):
        return []

    def insights_text(self, n):
        return "(no insights yet)"

    def all_entries(self):
        return []

    def recent(self, n):
        return []


class _RecordingGate:
    def __init__(self, *, use_sequential: bool):
        self.use_sequential = use_sequential
        self.calls: list[dict] = []

    def check(self, result, **kwargs):
        self.calls.append({"result": result, "kwargs": kwargs})
        return SimpleNamespace(seq={"state": "confirmed"} if kwargs else None)


def test_action_gate_check_default_off_uses_legacy_call() -> None:
    gate = _RecordingGate(use_sequential=False)
    result = _eval_result()

    verdict = actions._action_gate_check(
        {"type": "prompt_mutation"},
        _ctx(gate=gate),
        result,
    )

    assert verdict.seq is None
    assert len(gate.calls) == 1
    assert gate.calls[0]["kwargs"] == {}
    assert "seq_action_gate_check" not in result.details


def test_action_gate_check_threads_seq_inputs(monkeypatch) -> None:
    gate = _RecordingGate(use_sequential=True)
    result = _eval_result()
    result.question_results = [{"qid": 1, "ok": True}]

    monkeypatch.setattr(
        autopilot,
        "_seq_inputs_for_trial",
        lambda **kwargs: {
            "baseline_profile": {1: True},
            "baseline_task_rate": 12.5,
            "prior_quality_obs": [(7, 1.2)],
            "prior_rate_obs": [(8, 0.3)],
            "candidate": "candidate-x",
            "core_id": "core-v",
        },
    )
    monkeypatch.setattr(autopilot, "task_rate_qph_from", lambda _result: 42.0)

    verdict = actions._action_gate_check(
        {"type": "prompt_mutation"},
        _ctx(gate=gate, journal=_FakeJournal()),
        result,
    )

    assert verdict.seq == {"state": "confirmed"}
    kwargs = gate.calls[0]["kwargs"]
    assert kwargs["question_results"] == [{"qid": 1, "ok": True}]
    assert kwargs["task_rate"] == 42.0
    assert kwargs["baseline_profile"] == {1: True}
    assert kwargs["baseline_task_rate"] == 12.5
    assert kwargs["prior_quality_obs"] == [(7, 1.2)]
    assert kwargs["prior_rate_obs"] == [(8, 0.3)]
    assert kwargs["candidate"] == "candidate-x"
    assert kwargs["core_id"] == "core-v"
    assert result.details["seq_action_gate_check"] == {
        "enabled": True,
        "applied": True,
        "reason": "",
        "candidate": "candidate-x",
        "core_id": "core-v",
    }


class _FakeSwarm:
    def __init__(self):
        self.epochs: list[str] = []

    def mark_epoch(self, epoch):
        self.epochs.append(epoch)


class _QueuedTower:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def hybrid_eval(self):
        self.calls += 1
        return self.results.pop(0)


def test_prompt_mutation_skill_gate_default_off_single_eval(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_SKILL_EFFICACY_GATE", raising=False)
    monkeypatch.delenv("AUTOPILOT_BSV2_ACCEPT_GATE", raising=False)

    class FakeForge:
        def __init__(self):
            self.applied = 0
            self.reverted = 0

        def propose_mutation(self, **kwargs):
            return SimpleNamespace(
                file=kwargs["target_file"],
                mutation_type=kwargs["mutation_type"],
                description="test",
                original_content="old",
                mutated_content="new",
            )

        def apply_mutation(self, mutation):
            self.applied += 1

        def revert_mutation(self, mutation):
            self.reverted += 1

    tower = _QueuedTower([_eval_result(per_suite_quality={"math": 1.1})])
    forge = FakeForge()
    swarm = _FakeSwarm()
    result, species = actions._action_prompt_mutation(
        {"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix"},
        _ctx(forge=forge, tower=tower, gate=_AlwaysPassGate(), swarm=swarm, journal=_FakeJournal()),
    )

    assert species == "prompt_forge"
    assert result.details.get("skill_efficacy") is None
    assert result.details.get("bsv2_accept_gate") is None
    assert tower.calls == 1
    assert forge.applied == 1
    assert forge.reverted == 0
    assert swarm.epochs == ["prompt_mutation:frontdoor.md/targeted_fix"]


def test_prompt_mutation_bsv2_gate_rejects_behavior_regression(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_SKILL_EFFICACY_GATE", raising=False)
    monkeypatch.setenv("AUTOPILOT_BSV2_ACCEPT_GATE", "1")
    monkeypatch.setenv("AUTOPILOT_BSV2_MIN_SHARED_QIDS", "2")

    class FakeForge:
        def __init__(self):
            self.applied = 0
            self.reverted = 0

        def propose_mutation(self, **kwargs):
            return SimpleNamespace(
                file=kwargs["target_file"],
                mutation_type=kwargs["mutation_type"],
                description="test",
                original_content="old",
                mutated_content="new",
            )

        def apply_mutation(self, mutation):
            self.applied += 1

        def revert_mutation(self, mutation):
            self.reverted += 1

    baseline = _eval_result(
        question_results=[
            {"qid": "q1", "suite": "math", "correct": True},
            {"qid": "q2", "suite": "math", "correct": True},
        ]
    )
    candidate = _eval_result(
        question_results=[
            {"qid": "q1", "suite": "math", "correct": False},
            {"qid": "q2", "suite": "math", "correct": True},
        ]
    )
    tower = _QueuedTower([baseline, candidate])
    forge = FakeForge()
    swarm = _FakeSwarm()

    result, species = actions._action_prompt_mutation(
        {"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix"},
        _ctx(forge=forge, tower=tower, gate=_AlwaysPassGate(), swarm=swarm, journal=_FakeJournal()),
    )

    assert species == "prompt_forge"
    assert tower.calls == 2
    assert forge.applied == 1
    assert forge.reverted == 1
    assert swarm.epochs == []
    detail = result.details["bsv2_accept_gate"]
    assert detail["accept"] is False
    assert detail["gate_decision"] == "block"
    assert detail["artifact_kind"] == "prompt"
    assert detail["paired_stats"]["shared_qids"] == 2
    assert detail["paired_stats"]["delta_b_minus_a"] == pytest.approx(-0.5)
    assert detail["signature_diff"]["severity"] == "blocking"


def test_prompt_mutation_transfer_safety_skip_stops_before_apply_or_eval() -> None:
    class FakeForge:
        def __init__(self):
            self.applied = 0

        def propose_mutation(self, **kwargs):
            return SimpleNamespace(
                file=kwargs["target_file"],
                mutation_type=kwargs["mutation_type"],
                description="test",
                original_content="old",
                mutated_content="old",
                safety_valid=False,
                safety_reason="domain_mismatched_anchoring",
            )

        def apply_mutation(self, mutation):
            self.applied += 1

    tower = _QueuedTower([])
    forge = FakeForge()
    swarm = _FakeSwarm()
    result, species = actions._action_prompt_mutation(
        {"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix"},
        _ctx(forge=forge, tower=tower, gate=_AlwaysPassGate(), swarm=swarm, journal=_FakeJournal()),
    )

    assert result is None
    assert species == "prompt_forge"
    assert forge.applied == 0
    assert tower.calls == 0
    assert swarm.epochs == []


def test_prompt_mutation_skill_gate_rejects_per_suite_regression(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_SKILL_EFFICACY_GATE", "1")

    class FakeForge:
        def __init__(self):
            self.reverted = 0

        def propose_mutation(self, **kwargs):
            return SimpleNamespace(
                file=kwargs["target_file"],
                mutation_type=kwargs["mutation_type"],
                description="test",
                original_content="old",
                mutated_content="new",
            )

        def apply_mutation(self, mutation):
            pass

        def revert_mutation(self, mutation):
            self.reverted += 1

    tower = _QueuedTower(
        [
            _eval_result(per_suite_quality={"math": 1.0, "web": 1.0}),
            _eval_result(per_suite_quality={"math": 2.5, "web": 0.0}),
        ]
    )
    forge = FakeForge()
    swarm = _FakeSwarm()
    result, species = actions._action_prompt_mutation(
        {"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix"},
        _ctx(forge=forge, tower=tower, gate=_AlwaysPassGate(), swarm=swarm, journal=_FakeJournal()),
    )

    assert species == "prompt_forge"
    assert tower.calls == 2
    assert forge.reverted == 1
    assert swarm.epochs == []
    detail = result.details["skill_efficacy"]
    assert detail["accept"] is False
    assert detail["artifact_kind"] == "prompt"
    assert detail["regressed_suites"] == [("web", -1.0)]


def test_code_mutation_skill_gate_accepts_clean_gain(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_SKILL_EFFICACY_GATE", "true")
    monkeypatch.delenv("AUTOPILOT_BSV2_ACCEPT_GATE", raising=False)

    class FakeForge:
        def __init__(self):
            self.reverted = 0

        def propose_code_mutation(self, **kwargs):
            return SimpleNamespace(
                file=kwargs["target_file"],
                mutation_type=kwargs["mutation_type"],
                description="test",
                original_content="old",
                mutated_content="new",
                syntax_valid=True,
            )

        def apply_code_mutation(self, mutation):
            pass

        def revert_code_mutation(self, mutation):
            self.reverted += 1

    tower = _QueuedTower(
        [
            _eval_result(per_suite_quality={"math": 1.0, "web": 1.0}),
            _eval_result(per_suite_quality={"math": 1.2, "web": 1.1}),
        ]
    )
    forge = FakeForge()
    swarm = _FakeSwarm()
    result, species = actions._action_code_mutation(
        {"type": "code_mutation", "file": "src/escalation.py", "mutation": "targeted_fix"},
        _ctx(forge=forge, tower=tower, gate=_AlwaysPassGate(), swarm=swarm, journal=_FakeJournal()),
    )

    assert species == "prompt_forge"
    assert tower.calls == 2
    assert forge.reverted == 0
    assert swarm.epochs == ["code_mutation:src/escalation.py/targeted_fix"]
    detail = result.details["skill_efficacy"]
    assert detail["accept"] is True
    assert detail["artifact_kind"] == "code"
    assert detail["aggregate_delta"] == pytest.approx(0.15)


def test_code_mutation_bsv2_gate_accepts_watch_behavior(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_SKILL_EFFICACY_GATE", raising=False)
    monkeypatch.setenv("AUTOPILOT_BSV2_ACCEPT_GATE", "true")
    monkeypatch.setenv("AUTOPILOT_BSV2_MIN_SHARED_QIDS", "2")

    class FakeForge:
        def __init__(self):
            self.reverted = 0

        def propose_code_mutation(self, **kwargs):
            return SimpleNamespace(
                file=kwargs["target_file"],
                mutation_type=kwargs["mutation_type"],
                description="test",
                original_content="old",
                mutated_content="new",
                syntax_valid=True,
                safety_valid=True,
            )

        def apply_code_mutation(self, mutation):
            pass

        def revert_code_mutation(self, mutation):
            self.reverted += 1

    shared_vector = [
        {"qid": "q1", "suite": "math", "correct": True},
        {"qid": "q2", "suite": "math", "correct": False},
    ]
    tower = _QueuedTower([
        _eval_result(question_results=shared_vector),
        _eval_result(question_results=shared_vector),
    ])
    forge = FakeForge()
    swarm = _FakeSwarm()

    result, species = actions._action_code_mutation(
        {"type": "code_mutation", "file": "src/escalation.py", "mutation": "targeted_fix"},
        _ctx(forge=forge, tower=tower, gate=_AlwaysPassGate(), swarm=swarm, journal=_FakeJournal()),
    )

    assert species == "prompt_forge"
    assert tower.calls == 2
    assert forge.reverted == 0
    assert swarm.epochs == ["code_mutation:src/escalation.py/targeted_fix"]
    detail = result.details["bsv2_accept_gate"]
    assert detail["accept"] is True
    assert detail["gate_decision"] == "pass"
    assert detail["artifact_kind"] == "code"
    assert detail["signature_diff"]["severity"] == "watch"


def test_code_mutation_transfer_safety_skip_stops_before_apply_or_eval() -> None:
    class FakeForge:
        def __init__(self):
            self.applied = 0

        def propose_code_mutation(self, **kwargs):
            return SimpleNamespace(
                file=kwargs["target_file"],
                mutation_type=kwargs["mutation_type"],
                description="test",
                original_content="old",
                mutated_content="old",
                syntax_valid=True,
                safety_valid=False,
                safety_reason="misapplied_best_practice",
            )

        def apply_code_mutation(self, mutation):
            self.applied += 1

    tower = _QueuedTower([])
    forge = FakeForge()
    swarm = _FakeSwarm()
    result, species = actions._action_code_mutation(
        {"type": "code_mutation", "file": "src/escalation.py", "mutation": "targeted_fix"},
        _ctx(forge=forge, tower=tower, gate=_AlwaysPassGate(), swarm=swarm, journal=_FakeJournal()),
    )

    assert result is None
    assert species == "prompt_forge"
    assert forge.applied == 0
    assert tower.calls == 0
    assert swarm.epochs == []


def test_code_mutation_noop_skips_eval() -> None:
    class FakeForge:
        def __init__(self):
            self.applied = 0

        def propose_code_mutation(self, **kwargs):
            return SimpleNamespace(
                file=kwargs["target_file"],
                mutation_type=kwargs["mutation_type"],
                description="test",
                original_content="same",
                mutated_content="same",
                syntax_valid=True,
                safety_valid=True,
            )

        def apply_code_mutation(self, mutation):
            self.applied += 1

    tower = _QueuedTower([])
    forge = FakeForge()
    swarm = _FakeSwarm()
    result, species = actions._action_code_mutation(
        {"type": "code_mutation", "file": "src/escalation.py", "mutation": "targeted_fix"},
        _ctx(forge=forge, tower=tower, gate=_AlwaysPassGate(), swarm=swarm, journal=_FakeJournal()),
    )

    assert species == "prompt_forge"
    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "skipped"
    assert result.reason == "code_mutation produced no file changes"
    assert forge.applied == 0
    assert tower.calls == 0
    assert swarm.epochs == []


def test_distill_knowledge_returns_evolution_manager_species() -> None:
    """Without evo/strategy_store, distill_knowledge is a journalable invalid outcome."""
    result, species = actions._action_distill_knowledge(
        {"type": "distill_knowledge"}, _ctx(evo=None, strategy_store=None),
    )
    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "invalid"
    assert "missing evo or strategy_store" in result.reason
    assert species == "evolution_manager"


def test_distill_knowledge_uses_superseded_journal_view() -> None:
    class FakeJournal(_FakeJournal):
        def all_entries(self):
            return ["raw"]

        def entries_with_supersessions(self):
            return ["folded"]

    class FakeEvolutionManager:
        def __init__(self):
            self.journal_entries = None

        def distill(self, *, journal_entries, strategy_store, last_n, trial_id):
            self.journal_entries = journal_entries
            return {
                "status": "success",
                "strategy_store": strategy_store,
                "last_n": last_n,
                "trial_id": trial_id,
            }

    evo = FakeEvolutionManager()

    result, species = actions._action_distill_knowledge(
        {"type": "distill_knowledge", "last_n": 7},
        _ctx(
            evo=evo,
            strategy_store=object(),
            journal=FakeJournal(),
            state={"trial_counter": 42},
        ),
    )

    assert species == "evolution_manager"
    assert result is None
    assert evo.journal_entries == ["folded"]


def test_distill_knowledge_failed_result_returns_invalid_skip() -> None:
    class FakeEvolutionManager:
        def distill(self, *, journal_entries, strategy_store, last_n, trial_id):
            return {"status": "failed", "reason": "LLM invocation failed"}

    result, species = actions._action_distill_knowledge(
        {"type": "distill_knowledge", "last_n": 7},
        _ctx(
            evo=FakeEvolutionManager(),
            strategy_store=object(),
            journal=_FakeJournal(),
            state={"trial_counter": 42},
        ),
    )

    assert species == "evolution_manager"
    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "invalid"
    assert result.reason == "distill_knowledge failed: LLM invocation failed"


def test_distill_skillbank_unavailable_skips_eval() -> None:
    class FakeLab:
        def __init__(self):
            self.checkpoints = []

        def checkpoint_state(self, **kwargs):
            self.checkpoints.append(kwargs)

        def distill_skillbank(self, *, teacher, categories):
            return {"status": "not_available"}

    class FakeTower:
        def hybrid_eval(self):
            raise AssertionError("unavailable distill_skillbank must not run eval")

    result, species = actions._action_distill_skillbank(
        {"type": "distill_skillbank", "teacher": "claude", "categories": ["routing"]},
        _ctx(lab=FakeLab(), tower=FakeTower(), state={"trial_counter": 42}),
    )

    assert species == "structural_lab"
    assert isinstance(result, actions.SkipOutcome)
    assert result.status == "skipped"
    assert "DistillationPipeline not available" in result.reason


def test_distill_skillbank_available_runs_eval() -> None:
    class FakeLab:
        def checkpoint_state(self, **kwargs):
            pass

        def distill_skillbank(self, *, teacher, categories):
            return {"status": "success"}

    class FakeTower:
        def hybrid_eval(self):
            return "EVAL"

    result, species = actions._action_distill_skillbank(
        {"type": "distill_skillbank", "teacher": "claude", "categories": ["routing"]},
        _ctx(lab=FakeLab(), tower=FakeTower(), state={"trial_counter": 42}),
    )

    assert species == "structural_lab"
    assert result == "EVAL"


def test_mutation_context_filters_strategy_trials_superseded_by_journal() -> None:
    class FakeJournal(_FakeJournal):
        def entries_with_supersessions(self):
            return [
                SimpleNamespace(trial_id=1, bug_corrupted_by=""),
                SimpleNamespace(trial_id=2, bug_corrupted_by="resource_contention"),
                SimpleNamespace(trial_id=3, bug_corrupted_by="", outcome_status="error"),
                SimpleNamespace(
                    trial_id=4,
                    bug_corrupted_by="",
                    outcome_status="ok",
                    keep_revert_decision="excluded",
                ),
                SimpleNamespace(
                    trial_id=5,
                    bug_corrupted_by="",
                    outcome_status="ok",
                    keep_revert_decision="",
                    eval_details={"learning_exclusion": {"by": "mad_noise"}},
                ),
            ]

    class FakeStrategyStore:
        def __init__(self):
            self.journal = None
            self.query = ""
            self.k = 0

        def retrieve_for_journal(self, query, *, journal, k):
            self.query = query
            self.journal = journal
            self.k = k
            return []

    store = FakeStrategyStore()

    actions._build_mutation_context(
        {
            "file": "src/example.py",
            "mutation": "targeted_fix",
            "description": "example",
        },
        _ctx(journal=FakeJournal(), strategy_store=store, state={}),
    )

    assert store.journal is not None
    assert store.k == 3
    assert "src/example.py" in store.query


def test_mutation_context_skips_legacy_strategy_store_without_journal_view(caplog) -> None:
    class LegacyStrategyStore:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("raw retrieve fallback should not be called")

    failure_context, last_per_suite = actions._build_mutation_context(
        {
            "file": "src/example.py",
            "mutation": "targeted_fix",
            "description": "example",
        },
        _ctx(journal=_FakeJournal(), strategy_store=LegacyStrategyStore(), state={}),
    )

    assert last_per_suite is None
    assert "Past Strategy Insights" not in failure_context
    assert "retrieve_for_journal" in caplog.text


def test_reset_memories_returns_none_eval() -> None:
    class FakeLab:
        def reset_and_reseed(self, **kw):
            return {"reset": True}

    result, species = actions._action_reset_memories(
        {"type": "reset_memories"}, _ctx(lab=FakeLab(), state={"trial_counter": 5}),
    )
    assert result is None
    assert species == "structural_lab"


def test_repeated_meta_action_forces_metric_seed_batch() -> None:
    action, rationale = autopilot._force_metric_action_after_meta(
        {"type": "distill_knowledge", "last_n": 10},
        {"consecutive_meta_actions": 1},
        {"falsifier": "noop"},
    )
    assert action == {
        "type": "seed_batch",
        "n_questions": autopilot.SAFE_FALLBACK_SEED_N,
    }
    assert rationale == {
        "falsifier": "noop",
        "meta_action_forced_metric_trial": True,
    }


def test_repeated_meta_action_avoids_blacklisted_metric_seed_batch() -> None:
    action, rationale = autopilot._force_metric_action_after_meta(
        {"type": "distill_knowledge", "last_n": 10},
        {"consecutive_meta_actions": 1},
        {"falsifier": "noop"},
        [
            {
                "pattern": {
                    "type": "seed_batch",
                    "n_questions": autopilot.SAFE_FALLBACK_SEED_N,
                },
                "reason": "blocked",
            }
        ],
    )

    assert action == {"type": "seed_batch", "n_questions": 16}
    assert rationale["meta_action_forced_metric_trial"] is True
    assert rationale["fallback_seed_reselected"] is True
    assert rationale["fallback_seed_reselected_reason"] == "blocked"


def test_pre_dispatch_seed_fallback_reselects_blacklisted_action() -> None:
    action, rationale = autopilot._replace_blacklisted_seed_fallback(
        {
            "type": "seed_batch",
            "n_questions": autopilot.SAFE_FALLBACK_SEED_N,
            "suites": ["coder", "math"],
        },
        [
            {
                "pattern": {
                    "type": "seed_batch",
                    "n_questions": autopilot.SAFE_FALLBACK_SEED_N,
                    "suites": ["coder", "math"],
                },
                "reason": "blocked suite draw",
            }
        ],
        {"falsifier": "noop"},
        reason_label="test",
    )

    assert action == {"type": "seed_batch", "n_questions": autopilot.SAFE_FALLBACK_SEED_N}
    assert rationale["fallback_seed_reselected"] is True
    assert rationale["fallback_seed_reselected_context"] == "test"


def test_autonomous_blacklisted_action_reselects_seed_fallback() -> None:
    action, rationale = autopilot._replace_blacklisted_autonomous_action(
        {"type": "gepa_optimize", "file": "frontdoor.md", "max_evals": 50},
        [
            {
                "pattern": {"type": "gepa_optimize", "file": "frontdoor.md"},
                "reason": "manual prompt freeze",
            }
        ],
        {"falsifier": "noop"},
    )

    assert action == {
        "type": "seed_batch",
        "n_questions": autopilot.SAFE_FALLBACK_SEED_N,
    }
    assert rationale["autonomous_blacklisted_replaced"] is True
    assert rationale["autonomous_blacklisted_reason"] == "manual prompt freeze"
    assert rationale["autonomous_blacklisted_from"] == {
        "type": "gepa_optimize",
        "file": "frontdoor.md",
        "max_evals": 50,
    }


def test_autonomous_meta_action_reselects_seed_fallback() -> None:
    action, rationale = autopilot._replace_blacklisted_autonomous_action(
        {"type": "distill_knowledge", "last_n": 10},
        [],
        {"falsifier": "noop"},
    )

    assert action == {
        "type": "seed_batch",
        "n_questions": autopilot.SAFE_FALLBACK_SEED_N,
    }
    assert rationale["autonomous_blacklisted_replaced"] is True
    assert rationale["autonomous_blacklisted_reason"] == (
        "autonomous meta action does not collect metrics"
    )
    assert rationale["autonomous_blacklisted_from"] == {
        "type": "distill_knowledge",
        "last_n": 10,
    }


def test_autonomous_blacklisted_action_stays_when_seed_fallbacks_exhausted() -> None:
    requested = {"type": "gepa_optimize", "file": "frontdoor.md", "max_evals": 50}
    blacklist = [
        {
            "pattern": {"type": "gepa_optimize", "file": "frontdoor.md"},
            "reason": "manual prompt freeze",
        },
        *[
            {
                "pattern": {"type": "seed_batch", "n_questions": n_questions},
                "reason": f"blocked {n_questions}",
            }
            for n_questions in autopilot.FALLBACK_SEED_CANDIDATES
        ],
    ]

    action, rationale = autopilot._replace_blacklisted_autonomous_action(
        requested,
        blacklist,
        {"falsifier": "noop"},
    )

    assert action == requested
    assert rationale == {"falsifier": "noop"}


def test_first_unblacklisted_seed_action_reports_exhaustion() -> None:
    blacklist = [
        {
            "pattern": {"type": "seed_batch", "n_questions": n_questions},
            "reason": f"blocked {n_questions}",
        }
        for n_questions in autopilot.FALLBACK_SEED_CANDIDATES
    ]

    action, reason = autopilot._first_unblacklisted_seed_action(blacklist)

    assert action is None
    assert reason == f"blocked {autopilot.FALLBACK_SEED_CANDIDATES[-1]}"


def test_critic_fallback_seed_skip_when_seed_candidates_exhausted() -> None:
    skip = autopilot._critic_fallback_seed_skip(
        {"type": "seed_batch", "n_questions": autopilot.SAFE_FALLBACK_SEED_N},
        [
            {
                "pattern": {"type": "seed_batch", "n_questions": n_questions},
                "reason": f"blocked {n_questions}",
            }
            for n_questions in autopilot.FALLBACK_SEED_CANDIDATES
        ],
    )

    assert skip is not None
    assert skip.status == "skipped"
    assert skip.action_type == "planner_coordinator"
    assert "critic fallback seed_batch unavailable" in skip.reason


def test_first_meta_action_is_allowed() -> None:
    action, rationale = autopilot._force_metric_action_after_meta(
        {"type": "distill_knowledge", "last_n": 10},
        {"consecutive_meta_actions": 0},
        {"falsifier": "noop"},
    )
    assert action == {"type": "distill_knowledge", "last_n": 10}
    assert rationale == {"falsifier": "noop"}


# ----- experiment quota (passive-action ceiling) -----


def test_quota_passes_passive_below_memory_threshold() -> None:
    state = {"consecutive_passive_actions": 99}
    action, _ = autopilot._enforce_experiment_quota(
        {"type": "seed_batch", "n_questions": 10}, state,
        memory_count=10, rationale=None, trial_counter=1,
    )
    # Below threshold seeding is legitimate — never overridden.
    assert action["type"] == "seed_batch"
    assert state["consecutive_passive_actions"] == 100


def test_quota_forces_experiment_after_consecutive_passive_when_memory_large() -> None:
    state = {"consecutive_passive_actions": autopilot.MAX_CONSECUTIVE_PASSIVE}
    action, rationale = autopilot._enforce_experiment_quota(
        {"type": "seed_batch", "n_questions": 10}, state,
        memory_count=autopilot.QUOTA_MEMORY_THRESHOLD + 1,
        rationale={"falsifier": "x"}, trial_counter=0,
    )
    assert action["type"] == "numeric_trial"
    assert action["params"] == {}
    assert rationale["experiment_quota_forced"] is True
    # Counter resets after forcing an experiment.
    assert state["consecutive_passive_actions"] == 0


def test_quota_numeric_surfaces_follow_numeric_swarm_registry() -> None:
    from species.numeric_swarm import SURFACES

    assert tuple(SURFACES) == autopilot._QUOTA_NUMERIC_SURFACES
    assert "chat_pipeline" in autopilot._QUOTA_NUMERIC_SURFACES
    assert "repl_budget" in autopilot._QUOTA_NUMERIC_SURFACES
    assert "kv_compaction" in autopilot._QUOTA_NUMERIC_SURFACES


def test_quota_skips_blacklisted_numeric_surface() -> None:
    state = {"consecutive_passive_actions": autopilot.MAX_CONSECUTIVE_PASSIVE}
    blocked_surface = autopilot._QUOTA_NUMERIC_SURFACES[0]
    expected_surface = autopilot._QUOTA_NUMERIC_SURFACES[1]
    action, rationale = autopilot._enforce_experiment_quota(
        {"type": "seed_batch", "n_questions": 10},
        state,
        memory_count=autopilot.QUOTA_MEMORY_THRESHOLD + 1,
        rationale={"falsifier": "x"},
        trial_counter=0,
        blacklist=[
            {
                "pattern": {
                    "type": "numeric_trial",
                    "surface": blocked_surface,
                    "params": {},
                },
                "reason": f"blocked {blocked_surface}",
            }
        ],
    )

    assert action == {"type": "numeric_trial", "surface": expected_surface, "params": {}}
    assert rationale["experiment_quota_forced"] is True
    assert state["consecutive_passive_actions"] == 0


def test_quota_records_block_when_all_numeric_surfaces_blacklisted() -> None:
    state = {"consecutive_passive_actions": autopilot.MAX_CONSECUTIVE_PASSIVE}
    requested = {"type": "seed_batch", "n_questions": 10}
    action, rationale = autopilot._enforce_experiment_quota(
        requested,
        state,
        memory_count=autopilot.QUOTA_MEMORY_THRESHOLD + 1,
        rationale={"falsifier": "x"},
        trial_counter=2,
        blacklist=[
            {
                "pattern": {
                    "type": "numeric_trial",
                    "surface": surface,
                    "params": {},
                },
                "reason": f"blocked {surface}",
            }
            for surface in autopilot._QUOTA_NUMERIC_SURFACES
        ],
    )

    assert action == requested
    assert rationale == {"falsifier": "x"}
    assert state["consecutive_passive_actions"] == autopilot.MAX_CONSECUTIVE_PASSIVE + 1
    assert state["experiment_quota_blocked"]["trial_id"] == 2


def test_quota_resets_counter_on_nonpassive_action() -> None:
    state = {"consecutive_passive_actions": 5}
    action, _ = autopilot._enforce_experiment_quota(
        {"type": "prompt_mutation", "file": "frontdoor.md"}, state,
        memory_count=99999, rationale=None, trial_counter=0,
    )
    assert action["type"] == "prompt_mutation"
    assert state["consecutive_passive_actions"] == 0


def test_frontier_rerun_forces_numeric_trial() -> None:
    state = {
        "frontier_rerun_required": {
            "required": True,
            "reason": "v6 kernel era opened",
        }
    }
    action, rationale = autopilot._maybe_force_frontier_rerun_action(
        {"type": "structural_prune", "file": "rules.md"},
        state,
        rationale={"falsifier": "x"},
        trial_counter=0,
    )

    assert action == {
        "type": "numeric_trial",
        "surface": autopilot._QUOTA_NUMERIC_SURFACES[0],
        "params": {},
    }
    assert rationale["frontier_rerun_forced"] is True
    assert rationale["frontier_rerun_reason"] == "v6 kernel era opened"
    assert state["frontier_rerun_forced"]["original_action"]["type"] == "structural_prune"


def test_frontier_rerun_accepts_selected_numeric_trial() -> None:
    state = {
        "frontier_rerun_required": {
            "required": True,
            "reason": "v6 kernel era opened",
        }
    }
    selected = {"type": "numeric_trial", "surface": "monitor", "params": {}}
    action, rationale = autopilot._maybe_force_frontier_rerun_action(
        selected,
        state,
        rationale={},
        trial_counter=1,
    )

    assert action == selected
    assert rationale["frontier_rerun_satisfied_by_selected_action"] is True
    assert state["frontier_rerun_forced"] is None
    assert state["frontier_rerun_pending_clear"] == {
        "trial_id": 1,
        "action": selected,
        "reason": "v6 kernel era opened",
    }


def test_frontier_rerun_pending_clear_does_not_force_again() -> None:
    state = {
        "frontier_rerun_required": {
            "required": True,
            "reason": "v6 kernel era opened",
        },
        "frontier_rerun_pending_clear": {
            "trial_id": 7,
            "action": {"type": "numeric_trial", "surface": "monitor", "params": {}},
            "reason": "v6 kernel era opened",
        },
    }
    requested = {"type": "structural_prune", "file": "rules.md"}
    action, rationale = autopilot._maybe_force_frontier_rerun_action(
        requested,
        state,
        rationale={"falsifier": "x"},
        trial_counter=8,
    )

    assert action == requested
    assert rationale["frontier_rerun_pending_clear"] is True
    assert rationale["frontier_rerun_pending_trial_id"] == 7


def test_frontier_rerun_pending_clear_keeps_forcing_until_min_trials() -> None:
    state = {
        "frontier_rerun_required": {
            "required": True,
            "reason": "v6 kernel era opened",
            "opened_at": "2026-06-28T01:40:00Z",
            "min_numeric_trials": 4,
        },
        "frontier_rerun_pending_clear": {
            "trial_id": 7,
            "action": {"type": "numeric_trial", "surface": "monitor", "params": {}},
            "reason": "v6 kernel era opened",
        },
    }
    journal = SimpleNamespace(
        entries_with_supersessions=lambda: [
            SimpleNamespace(
                bug_corrupted_by="",
                action_type="numeric_trial",
                tier=1,
                timestamp="2026-06-28T01:41:00Z",
            )
        ]
    )

    action, rationale = autopilot._maybe_force_frontier_rerun_action(
        {"type": "structural_prune", "file": "rules.md"},
        state,
        journal=journal,
        rationale={},
        trial_counter=8,
    )

    assert action["type"] == "numeric_trial"
    assert rationale["frontier_rerun_forced"] is True
    assert rationale["frontier_rerun_completed_numeric_trials"] == 1
    assert rationale["frontier_rerun_min_numeric_trials"] == 4


def test_frontier_rerun_pending_clear_stops_after_min_trials() -> None:
    state = {
        "frontier_rerun_required": {
            "required": True,
            "reason": "v6 kernel era opened",
            "opened_at": "2026-06-28T01:40:00Z",
            "min_numeric_trials": 2,
        },
        "frontier_rerun_pending_clear": {
            "trial_id": 7,
            "action": {"type": "numeric_trial", "surface": "monitor", "params": {}},
            "reason": "v6 kernel era opened",
        },
    }
    journal = SimpleNamespace(
        entries_with_supersessions=lambda: [
            SimpleNamespace(
                bug_corrupted_by="",
                action_type="numeric_trial",
                tier=1,
                timestamp="2026-06-28T01:41:00Z",
            ),
            SimpleNamespace(
                bug_corrupted_by="",
                action_type="numeric_trial",
                tier=1,
                timestamp="2026-06-28T01:42:00Z",
            ),
        ]
    )
    requested = {"type": "structural_prune", "file": "rules.md"}

    action, rationale = autopilot._maybe_force_frontier_rerun_action(
        requested,
        state,
        journal=journal,
        rationale={},
        trial_counter=9,
    )

    assert action == requested
    assert rationale["frontier_rerun_cleared"] is True
    assert rationale["frontier_rerun_completed_numeric_trials"] == 2
    assert rationale["frontier_rerun_min_numeric_trials"] == 2
    assert state["frontier_rerun_required"]["required"] is False
    assert state["frontier_rerun_required"]["cleared_after_trial_id"] == 7
    assert state["frontier_rerun_required"]["completed_numeric_trials"] == 2
    assert state["frontier_rerun_required"]["min_numeric_trials"] == 2
    assert "frontier rerun satisfied" in state["frontier_rerun_required"]["reason"]
    assert state["frontier_rerun_forced"] is None
    assert "frontier_rerun_pending_clear" not in state


def test_frontier_rerun_records_block_when_all_numeric_surfaces_blacklisted() -> None:
    state = {
        "frontier_rerun_required": {
            "required": True,
            "reason": "v6 kernel era opened",
        }
    }
    requested = {"type": "structural_prune", "file": "rules.md"}
    action, rationale = autopilot._maybe_force_frontier_rerun_action(
        requested,
        state,
        blacklist=[
            {
                "pattern": {
                    "type": "numeric_trial",
                    "surface": surface,
                    "params": {},
                },
                "reason": f"blocked {surface}",
            }
            for surface in autopilot._QUOTA_NUMERIC_SURFACES
        ],
        rationale={"falsifier": "x"},
        trial_counter=3,
    )

    assert action == requested
    assert rationale == {"falsifier": "x"}
    assert state["frontier_rerun_blocked"]["trial_id"] == 3


def test_frontier_rerun_summary_reports_live_progress() -> None:
    state = {
        "frontier_rerun_required": {
            "required": True,
            "reason": "v6 kernel era opened",
            "opened_at": "2026-06-28T01:40:00Z",
            "min_numeric_trials": 4,
        },
        "frontier_rerun_pending_clear": {
            "trial_id": 8,
            "action": {"type": "numeric_trial", "surface": "think_harder"},
        },
    }
    journal = SimpleNamespace(
        entries_with_supersessions=lambda: [
            SimpleNamespace(
                bug_corrupted_by="",
                action_type="numeric_trial",
                tier=1,
                timestamp="2026-06-28T01:41:00Z",
            ),
            SimpleNamespace(
                bug_corrupted_by="autopilot_killed_mid_trial",
                action_type="numeric_trial",
                tier=1,
                timestamp="2026-06-28T01:42:00Z",
            ),
        ]
    )

    lines = autopilot._frontier_rerun_summary_lines(state, journal)

    assert "Frontier rerun: required (1/4 numeric trials complete)" in lines
    assert "Frontier rerun reason: v6 kernel era opened" in lines
    assert "Frontier rerun opened: 2026-06-28T01:40:00Z" in lines
    assert "Frontier rerun pending: trial #8 numeric_trial/think_harder" in lines


def test_frontier_rerun_summary_reports_inactive_marker() -> None:
    journal = SimpleNamespace(entries_with_supersessions=lambda: [])

    assert autopilot._frontier_rerun_summary_lines({}, journal) == [
        "Frontier rerun: not required"
    ]


def test_numeric_swarm_epoch_label_prefers_autopilot_speed_era() -> None:
    state = {
        "active_instrument_eras": {
            "cpu_bench": "E5-cpu-kernel",
            "autopilot_speed": "E5-autopilot-speed",
        }
    }

    assert autopilot._numeric_swarm_epoch_label_from_state(state) == "E5-autopilot-speed"


def test_numeric_swarm_epoch_label_absent_without_speed_era() -> None:
    assert autopilot._numeric_swarm_epoch_label_from_state({}) is None
    assert autopilot._numeric_swarm_epoch_label_from_state(
        {"active_instrument_eras": {"cpu_bench": "E5-cpu-kernel"}}
    ) is None


def test_numeric_swarm_epoch_label_required_for_active_frontier_rerun() -> None:
    with pytest.raises(
        ValueError,
        match="frontier rerun requires active_instrument_eras.autopilot_speed",
    ):
        autopilot._numeric_swarm_epoch_label_from_state(
            {"frontier_rerun_required": {"required": True}}
        )


def test_archive_epoch_params_require_timestamps_for_active_speed_era() -> None:
    with pytest.raises(
        ValueError,
        match="active speed era requires pareto_epoch_ts and pareto_exclude_before_ts",
    ):
        autopilot._archive_epoch_params_from_state(
            {"active_instrument_eras": {"autopilot_speed": "E5-autopilot-speed"}}
        )


def test_archive_epoch_params_reject_malformed_epoch_for_active_speed_era() -> None:
    state = {
        "active_instrument_eras": {"autopilot_speed": "E5-autopilot-speed"},
        "pareto_epoch_ts": "not-a-timestamp",
        "pareto_exclude_before_ts": 1782511631.0,
    }

    with pytest.raises(ValueError, match="invalid pareto_epoch_ts"):
        autopilot._archive_epoch_params_from_state(state)


def test_archive_epoch_params_keep_legacy_tolerance_without_speed_era() -> None:
    assert autopilot._archive_epoch_params_from_state(
        {
            "pareto_epoch_ts": "not-a-timestamp",
            "pareto_pre_epoch_speed_factor": "bad-factor",
            "pareto_exclude_before_ts": "bad-exclude",
        }
    ) == (None, 1.0, None)


# ----- non-executing-action residue feedback -----


def test_last_invalid_feedback_none_when_clean() -> None:
    assert "none" in autopilot._build_last_invalid_feedback({}).lower()


def test_last_invalid_feedback_surfaces_reason_and_count() -> None:
    act = {"type": "structural_experiment", "flags": {"graph_router": True}}
    sig = autopilot._action_signature(act)
    state = {
        "last_invalid_action": act,
        "last_invalid_reason": "graph_router feature requires specialist_routing feature",
        "last_invalid_status": "invalid",
        "invalid_signature_counts": {sig: 3},
    }
    text = autopilot._build_last_invalid_feedback(state)
    assert "specialist_routing" in text
    assert "3×" in text
    assert "DO NOT repeat" in text


def test_last_invalid_feedback_surfaces_repeats_after_clear() -> None:
    """A repeated signature must still surface even when last_invalid_action was
    cleared by a successful (substituted) trial — the persistent counter drives
    the feedback so a critic-rejected draft cannot silently vanish."""
    state = {
        "last_invalid_action": None,
        "invalid_signature_counts": {'{"type": "x"}': 4},
    }
    text = autopilot._build_last_invalid_feedback(state)
    assert "Repeatedly non-executing" in text
    assert "4×" in text


def test_prior_planner_decision_digest_summarizes_bounded_archive(tmp_path) -> None:
    archive = tmp_path / "planner_archive.jsonl"
    archive.write_text(
        "\n".join(
            [
                '{"type":"planner_coordinator","action_type":"seed_batch",'
                '"draft_provider":"claude","critic_provider":"codex",'
                '"critique_decision":"approve","degraded":false}',
                '{"type":"planner_provider_call","provider":"claude","role":"draft",'
                '"status":"timeout","ok":false,"error":"timeout after 600s"}',
                '{"type":"planner_coordinator","action_type":"numeric_trial",'
                '"draft_provider":"claude","critic_provider":"codex",'
                '"critique_decision":"unavailable","degraded":true,'
                '"fallback_reason":"critic timeout",'
                '"critique_issues":["review unavailable"]}',
            ]
        )
        + "\n"
    )

    text = autopilot._build_prior_planner_decision_digest(archive, limit=2)

    assert "seed_batch" not in text
    assert "timeout after 600s" in text
    assert "numeric_trial" in text
    assert "fallback=critic timeout" in text


def test_prior_planner_decision_digest_handles_missing_archive(tmp_path) -> None:
    text = autopilot._build_prior_planner_decision_digest(
        tmp_path / "missing.jsonl",
        limit=2,
    )

    assert "none yet" in text


# ----- draft_critique: rejected-draft feedback (req #3) -----


class _FakeCritique:
    def __init__(self, decision="reject", issues=None):
        self.decision = decision
        self.issues = issues or ["unsafe"]


def test_record_rejected_draft_counts_and_sets_feedback() -> None:
    state = {}
    draft = {"type": "structural_experiment", "flags": {"graph_router": True}}
    blacklisted = autopilot._record_rejected_draft(state, draft, _FakeCritique(), trial_id=10)
    assert blacklisted is False  # first occurrence
    sig = autopilot._action_signature(draft)
    assert state["invalid_signature_counts"][sig] == 1
    assert state["last_invalid_status"] == "critic_rejected"
    assert state["last_invalid_action"] == draft
    assert "critic rejected" in state["last_invalid_reason"]
    assert state["consecutive_rejected_drafts"] == 1


def test_record_rejected_draft_blacklists_on_repeat(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(autopilot, "append_blacklist",
                        lambda action, tid, reason: calls.append((action, reason)))
    state = {}
    draft = {"type": "structural_experiment", "flags": {"graph_router": True}}
    autopilot._record_rejected_draft(state, draft, _FakeCritique(), trial_id=1)
    blacklisted = autopilot._record_rejected_draft(state, draft, _FakeCritique(), trial_id=2)
    assert blacklisted is True
    assert len(calls) == 1  # blacklisted exactly once, at the threshold (2x)
    assert state["consecutive_rejected_drafts"] == 2


# ----- ActionContext bundle -----


def test_action_context_is_dataclass() -> None:
    ctx = actions._ActionContext(
        seeder="s", swarm="sw", forge="f", lab="l", tower="t",
        gate="g", archive="a", journal="j", state={},
    )
    assert ctx.seeder == "s"
    assert ctx.strategy_store is None
    assert ctx.evo is None
