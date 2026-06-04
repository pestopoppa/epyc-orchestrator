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


def test_action_handlers_registered_for_all_known_types() -> None:
    """Sanity check: every documented action type has a handler."""
    expected = {
        "seed_batch", "numeric_trial", "prompt_mutation", "gepa_optimize",
        "code_mutation", "structural_experiment", "structural_prune",
        "train_routing_models", "distill_skillbank", "reset_memories",
        "deep_eval", "rollback", "distill_knowledge", "slot_compact",
    }
    assert expected == set(actions._ACTION_HANDLERS.keys())


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

        def evaluate(self, *, tier):
            self.calls.append(tier)
            return "DEEP_EVAL"

    tower = FakeTower()
    result, species = actions._action_deep_eval(
        {"type": "deep_eval", "tier": 3}, _ctx(tower=tower),
    )
    assert result == "DEEP_EVAL"
    assert species == "seeder"
    assert tower.calls == [3]


def test_distill_knowledge_returns_evolution_manager_species() -> None:
    """Without evo/strategy_store, distill_knowledge logs warning and returns None."""
    result, species = actions._action_distill_knowledge(
        {"type": "distill_knowledge"}, _ctx(evo=None, strategy_store=None),
    )
    assert result is None
    assert species == "evolution_manager"


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
    assert action == {"type": "seed_batch", "n_questions": 10}
    assert rationale == {
        "falsifier": "noop",
        "meta_action_forced_metric_trial": True,
    }


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


def test_quota_resets_counter_on_nonpassive_action() -> None:
    state = {"consecutive_passive_actions": 5}
    action, _ = autopilot._enforce_experiment_quota(
        {"type": "prompt_mutation", "file": "frontdoor.md"}, state,
        memory_count=99999, rationale=None, trial_counter=0,
    )
    assert action["type"] == "prompt_mutation"
    assert state["consecutive_passive_actions"] == 0


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
