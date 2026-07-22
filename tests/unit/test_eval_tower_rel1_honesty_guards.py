"""REL-1 eval-honesty guards (2026-07-21 EV-11c circuit-open incident).

Fixture-driven, INFERENCE-FREE. Covers the three eval-honesty guards added to
``eval_tower._eval_question`` and ``seeding_orchestrator.call_orchestrator_forced``:

  1. In-band error detection: a "[ERROR: Backend unavailable (circuit open):
     ...]" answer with error=None becomes an ERROR row (excluded from the
     quality denominator, counted against reliability) — never scored wrong.
  2. Forced-role integrity: a forced-role question served by a DIFFERENT role
     (silent circuit_open fallback) becomes an ERROR row, even when the answer
     text would otherwise score correct.
  3. Deadline-starvation floor: an eval llama call whose remaining budget is
     below the floor is FAILED pre-send (no HTTP request), so it never trips
     the production circuit breaker.

Run: .venv/bin/python -m pytest tests/unit/test_eval_tower_rel1_honesty_guards.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    REPO_ROOT / "scripts" / "autopilot",
    REPO_ROOT / "scripts" / "benchmark",
    REPO_ROOT,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_tower  # noqa: E402
import seeding_orchestrator  # noqa: E402
from eval_tower import EvalTower  # noqa: E402
from seeding_scoring import _classify_error  # noqa: E402

_CIRCUIT_OPEN_ANSWER = (
    "[ERROR: Backend unavailable (circuit open): http://localhost:8082]"
)


# ── Helper-level anchoring (start-of-answer, not a loose substring) ───────────


def test_inband_error_text_anchors_to_start_of_answer() -> None:
    assert eval_tower._inband_error_text(_CIRCUIT_OPEN_ANSWER) == _CIRCUIT_OPEN_ANSWER
    # Leading whitespace is tolerated (lstrip), still start-of-answer.
    assert (
        eval_tower._inband_error_text("  \n" + _CIRCUIT_OPEN_ANSWER)
        == _CIRCUIT_OPEN_ANSWER
    )
    # A mid-answer mention of the marker is NOT an in-band error.
    assert eval_tower._inband_error_text("The answer is [ERROR: foo]") is None
    assert eval_tower._inband_error_text("42") is None
    assert eval_tower._inband_error_text("") is None
    assert eval_tower._inband_error_text(None) is None


def test_forced_role_serving_mismatch_semantics() -> None:
    # Empty force_role → never a mismatch (routing was free to choose).
    assert (
        eval_tower._forced_role_serving_mismatch("", {"routed_to": "worker_general"})
        is None
    )
    # Matching serving role → no mismatch.
    assert (
        eval_tower._forced_role_serving_mismatch(
            "worker_math", {"routed_to": "worker_math"}
        )
        is None
    )
    # routed_to differs → returns the serving role.
    assert (
        eval_tower._forced_role_serving_mismatch(
            "worker_math", {"routed_to": "worker_general"}
        )
        == "worker_general"
    )
    # routed_to absent → fall back to the terminal role_history entry.
    assert (
        eval_tower._forced_role_serving_mismatch(
            "worker_math", {"role_history": ["worker_math", "worker_general"]}
        )
        == "worker_general"
    )
    # Cannot determine serving role → no false positive.
    assert eval_tower._forced_role_serving_mismatch("worker_math", {}) is None


# ── Guard 1: in-band error answer → error row, not a wrong answer ─────────────


def test_guard1_inband_error_answer_becomes_error_row(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": _CIRCUIT_OPEN_ANSWER,
            "error": None,  # the incident: in-band error with error=None
            "tokens_generated": 5,
            "routed_to": "worker_math",
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "math-circuit-open",
                "suite": "math",
                "prompt": "What is 2+2?",
                "expected": "4",
                "scoring_method": "exact_match",
                "force_role": "worker_math",
            },
            client,
        )

    # REL-1: excluded row, not a wrong answer.
    assert result.error is not None
    assert result.error.startswith("[ERROR:")
    assert result.correct is False
    # Classifies as infrastructure (not a model task_failure).
    assert _classify_error(result.error) == "infrastructure"


# ── Guard 2: forced-role mismatch → error row (even for a correct-looking answer)


def test_guard2_forced_role_mismatch_becomes_error_row(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        # A plausibly-CORRECT answer, but served by the WRONG role.
        return {
            "answer": "4",
            "error": None,
            "tokens_generated": 3,
            "routed_to": "worker_general",  # forced worker_math
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "math-forced-fallback",
                "suite": "math",
                "prompt": "What is 2+2?",
                "expected": "4",
                "scoring_method": "exact_match",
                "force_role": "worker_math",
            },
            client,
        )

    assert result.error is not None
    assert "forced_role_fallback" in result.error
    assert "forced=worker_math" in result.error
    assert "served_by=worker_general" in result.error
    # NOT scored, despite the answer text matching the expected value.
    assert result.correct is False
    assert result.route_used == "worker_general"
    assert _classify_error(result.error) == "infrastructure"


def test_guard2_uses_role_history_when_routed_to_absent(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": "4",
            "error": None,
            "tokens_generated": 3,
            "role_history": ["worker_math", "worker_general"],
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "math-history-fallback",
                "suite": "math",
                "prompt": "What is 2+2?",
                "expected": "4",
                "scoring_method": "exact_match",
                "force_role": "worker_math",
            },
            client,
        )

    assert result.error is not None
    assert "served_by=worker_general" in result.error
    assert result.correct is False


# ── Guard 3: deadline-starvation floor → pre-send error, NO HTTP call ─────────


def test_guard3_starved_deadline_refuses_pre_send_no_http(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_EVAL_MIN_LLAMA_BUDGET_S", raising=False)
    sent: list[object] = []

    def _handler(request):  # noqa: ANN001
        sent.append(request)
        raise AssertionError(
            "no HTTP request may be sent for a deadline-starved eval call"
        )

    transport = eval_tower.httpx.MockTransport(_handler)
    with eval_tower.httpx.Client(transport=transport) as client:
        resp = seeding_orchestrator.call_orchestrator_forced(
            prompt="a hard MATH-tail question",
            force_role="worker_math",
            timeout=1,  # below the 30s floor
            client=client,
            workload_class="eval_batch",
        )

    assert sent == []  # transport never touched
    assert resp["answer"] == ""
    assert resp["failure_reason"] == "deadline_starved"
    assert "deadline_starved" in resp["error"]
    assert _classify_error(resp["error"]) == "infrastructure"


def test_guard3_normal_budget_sends_request(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_EVAL_MIN_LLAMA_BUDGET_S", raising=False)
    seen: list[object] = []

    def _handler(request):  # noqa: ANN001
        seen.append(request)
        return eval_tower.httpx.Response(
            200,
            json={"answer": "ok", "tokens_generated": 2, "routed_to": "worker_math"},
        )

    transport = eval_tower.httpx.MockTransport(_handler)
    with eval_tower.httpx.Client(transport=transport) as client:
        resp = seeding_orchestrator.call_orchestrator_forced(
            prompt="q",
            force_role="worker_math",
            timeout=120,  # healthy budget → normal path unchanged
            client=client,
            workload_class="eval_batch",
        )

    assert len(seen) == 1
    assert resp["answer"] == "ok"


def test_guard3_floor_scoped_to_eval_traffic(monkeypatch) -> None:
    # Non-eval callers keep the EXACT legacy path even with a tiny timeout.
    monkeypatch.delenv("AUTOPILOT_EVAL_MIN_LLAMA_BUDGET_S", raising=False)
    seen: list[object] = []

    def _handler(request):  # noqa: ANN001
        seen.append(request)
        return eval_tower.httpx.Response(200, json={"answer": "ok"})

    transport = eval_tower.httpx.MockTransport(_handler)
    with eval_tower.httpx.Client(transport=transport) as client:
        resp = seeding_orchestrator.call_orchestrator_forced(
            prompt="q",
            force_role="frontdoor",
            timeout=1,  # tiny, but NOT eval traffic (no workload_class)
            client=client,
        )

    assert len(seen) == 1  # request WAS sent → legacy behavior preserved
    assert resp["answer"] == "ok"


# ── Normal path: guards dormant, question scored normally ─────────────────────


def test_guards_dormant_on_normal_answer(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": "4",
            "error": None,
            "tokens_generated": 3,
            "routed_to": "worker_math",  # matches force_role
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "math-normal",
                "suite": "math",
                "prompt": "What is 2+2?",
                "expected": "4",
                "scoring_method": "exact_match",
                "force_role": "worker_math",
            },
            client,
        )

    assert result.error is None
    assert result.correct is True
    assert result.route_used == "worker_math"


# ── Guard 3 env override: raise the per-question eval budget ──────────────────


def test_env_override_raises_eval_request_timeout(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_REQUEST_TIMEOUT_S", "1800")
    # Bypasses the 600s cap for rebaseline-class runs.
    assert eval_tower._default_eval_timeout() == 1800

    monkeypatch.delenv("AUTOPILOT_EVAL_REQUEST_TIMEOUT_S", raising=False)
    # Unset → current behavior (registry-derived, capped at 600).
    assert eval_tower._default_eval_timeout() <= 600
