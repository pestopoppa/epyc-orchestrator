"""Unit tests for gate-3 telemetry-contract helpers (no inference).

Validates the HARD get_eval_secret contract, the request-local isolation check,
and the SOFT web_research classifier (incl. the structured-unwrap guard) against
synthetic /chat response dicts — so the gate logic is verified before deploy.
"""
from __future__ import annotations

import sys
from pathlib import Path

_AP = Path(__file__).resolve().parents[2] / "scripts" / "autopilot"
if str(_AP) not in sys.path:
    sys.path.insert(0, str(_AP))

from gate3_tool_telemetry import (  # noqa: E402
    _gate_request_timeout,
    _gate_skip_soft,
    check_get_eval_secret_contract,
    check_isolation,
    classify_web_research,
    tool_name_counts,
)


def _ges(success: bool = True) -> dict:
    return {
        "tools_called": ["get_eval_secret"],
        "tools_used": 1,
        "tool_timings": [{"tool_name": "get_eval_secret", "elapsed_ms": 1.0, "success": success}],
    }


def test_get_eval_secret_contract_passes_on_clean_batch():
    ok, lines = check_get_eval_secret_contract([_ges(), _ges(), _ges()])
    assert ok, lines
    assert tool_name_counts([_ges(), _ges(), _ges()])["get_eval_secret"] == 3


def test_get_eval_secret_contract_fails_under_min():
    ok, _ = check_get_eval_secret_contract([_ges(), _ges()])
    assert not ok  # only 2 < 3


def test_get_eval_secret_contract_fails_on_any_failed_timing_row():
    ok, _ = check_get_eval_secret_contract([_ges(True), _ges(True), _ges(False)])
    assert not ok  # one row success is not True


def test_isolation_pass_and_fail():
    assert check_isolation({"tools_called": [], "tools_used": 0})[0] is True
    assert check_isolation({"tools_called": ["foo"], "tools_used": 1})[0] is False


def test_web_research_pass():
    status, _ = classify_web_research({
        "tools_called": ["web_research"],
        "tool_timings": [{"tool_name": "web_research", "success": True}],
        "web_research_results": [{"query": "q", "pages_fetched": 2}],
    })
    assert status == "PASS"


def test_web_research_success_with_response_error_still_passes_soft_telemetry():
    status, lines = classify_web_research({
        "error_code": 500,
        "tools_called": ["web_research"],
        "tool_timings": [{"tool_name": "web_research", "success": True}],
        "web_research_results": [{"query": "q", "pages_fetched": 1}],
    })
    assert status == "PASS"
    assert "post-tool error" in " ".join(lines)


def test_web_research_success_but_empty_results_is_infra_not_pass():
    """Guards the structured-unwrap regression: success rows but empty results."""
    status, lines = classify_web_research({
        "tools_called": ["web_research"],
        "tool_timings": [{"tool_name": "web_research", "success": True}],
        "web_research_results": [],
    })
    assert status == "INFRA_FAIL"
    assert "EMPTY" in " ".join(lines)


def test_web_research_tool_failure_is_infra():
    status, _ = classify_web_research({
        "tools_called": ["web_research"],
        "tool_timings": [{"tool_name": "web_research", "success": False}],
        "web_research_results": [],
    })
    assert status == "INFRA_FAIL"


def test_web_research_not_routed_is_inconclusive():
    status, _ = classify_web_research({"tools_called": [], "tool_timings": []})
    assert status == "INCONCLUSIVE"


def test_web_research_request_error_is_infra():
    status, _ = classify_web_research({"error": "500 Server Error"})
    assert status == "INFRA_FAIL"


def test_gate_request_timeout_defaults_and_clamps(monkeypatch):
    monkeypatch.delenv("AUTOPILOT_GATE3_REQUEST_TIMEOUT_S", raising=False)
    assert _gate_request_timeout(270) == 270

    monkeypatch.setenv("AUTOPILOT_GATE3_REQUEST_TIMEOUT_S", "120")
    assert _gate_request_timeout(270) == 120

    monkeypatch.setenv("AUTOPILOT_GATE3_REQUEST_TIMEOUT_S", "999")
    assert _gate_request_timeout(270) == 270

    monkeypatch.setenv("AUTOPILOT_GATE3_REQUEST_TIMEOUT_S", "bad")
    assert _gate_request_timeout(270) == 270


def test_gate_skip_soft_parses_truthy_values(monkeypatch):
    monkeypatch.delenv("AUTOPILOT_GATE3_SKIP_SOFT", raising=False)
    assert _gate_skip_soft() is False

    monkeypatch.setenv("AUTOPILOT_GATE3_SKIP_SOFT", "1")
    assert _gate_skip_soft() is True

    monkeypatch.setenv("AUTOPILOT_GATE3_SKIP_SOFT", "yes")
    assert _gate_skip_soft() is True

    monkeypatch.setenv("AUTOPILOT_GATE3_SKIP_SOFT", "false")
    assert _gate_skip_soft() is False
