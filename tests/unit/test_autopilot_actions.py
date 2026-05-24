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
    # numeric_trial with 2 explicit params violates AP-9
    result, species = actions.dispatch_action(
        {"type": "numeric_trial", "params": {"a": 1, "b": 2}},
        seeder=None, swarm=None, forge=None, lab=None, tower=None,
        gate=None, archive=None, journal=None, state={},
    )
    assert result is None
    assert species == "numeric_trial"


def test_dispatcher_unknown_action_type() -> None:
    result, species = actions.dispatch_action(
        {"type": "nonexistent_action"},
        seeder=None, swarm=None, forge=None, lab=None, tower=None,
        gate=None, archive=None, journal=None, state={},
    )
    assert result is None
    assert species == "unknown"


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


# ----- ActionContext bundle -----


def test_action_context_is_dataclass() -> None:
    ctx = actions._ActionContext(
        seeder="s", swarm="sw", forge="f", lab="l", tower="t",
        gate="g", archive="a", journal="j", state={},
    )
    assert ctx.seeder == "s"
    assert ctx.strategy_store is None
    assert ctx.evo is None
