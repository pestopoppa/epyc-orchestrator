"""Task-wide cap on failed-turn code regenerations (REPL hardening #2)."""

from types import SimpleNamespace

from src.escalation import ErrorCategory
from src.graph.decision_gates import _regeneration_cap_reached, _should_escalate, _should_retry
from src.graph.state import GraphConfig, TaskState
from src.graph.think_harder import _should_think_harder
from src.roles import Role


def _ctx(total, cap=3, consecutive=0, **cfg):
    config = GraphConfig(max_retries=5, max_regenerations=cap, **cfg)
    state = TaskState(prompt="p")
    state.total_repl_failures = total
    state.consecutive_failures = consecutive
    return SimpleNamespace(state=state, deps=SimpleNamespace(config=config))


def test_default_is_conservative_and_env_override(monkeypatch):
    assert GraphConfig().max_regenerations == 8
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_REGENERATIONS", "2")
    assert GraphConfig().max_regenerations == 2
    monkeypatch.setenv("ORCHESTRATOR_REPL_MAX_REGENERATIONS", "junk")
    assert GraphConfig().max_regenerations == 8


def test_retry_allowed_under_cap_blocked_at_cap():
    assert _should_retry(_ctx(2), ErrorCategory.CODE) is True
    assert _should_retry(_ctx(3), ErrorCategory.CODE) is False


def test_escalation_and_think_harder_blocked_at_cap():
    assert _should_escalate(_ctx(3, consecutive=5), ErrorCategory.CODE, Role.CODER_ESCALATION) is False
    assert _should_think_harder(_ctx(3, consecutive=4), ErrorCategory.CODE) is False


def test_cap_zero_disables():
    ctx = _ctx(100, cap=0)
    assert _regeneration_cap_reached(ctx) is False
    assert _should_retry(ctx, ErrorCategory.CODE) is True


def test_counter_survives_escalation_reset():
    # consecutive_failures resets on escalation; the task-wide counter must not.
    ctx = _ctx(3, consecutive=0)
    assert _should_retry(ctx, ErrorCategory.CODE) is False
