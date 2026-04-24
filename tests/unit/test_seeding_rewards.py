"""Unit tests for benchmark seeding_rewards helper module."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from unittest.mock import Mock, patch


_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
sys.path.insert(0, str(_ROOT))
_SPEC = importlib.util.spec_from_file_location("seeding_rewards_test", _ROOT / "seeding_rewards.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["seeding_rewards_test"] = _MOD
_SPEC.loader.exec_module(_MOD)


def _rr(**overrides):
    base = dict(
        role="frontdoor",
        mode="direct",
        answer="answer",
        passed=False,
        elapsed_seconds=1.0,
    )
    base.update(overrides)
    return _MOD.RoleResult(**base)


def test_compute_comparative_rewards_without_baseline_defaults_to_binary_success():
    rewards = _MOD.compute_comparative_rewards(
        {
            "worker_explore:direct": _rr(role="worker_explore", passed=True),
            "worker_math:direct": _rr(role="worker_math", passed=False),
        },
        baseline_key="missing:key",
    )
    assert rewards == {
        "worker_explore:direct": 1.0,
        "worker_math:direct": 0.0,
    }


def test_compute_comparative_rewards_all_main_branches():
    baseline = _rr(role="frontdoor", passed=True)
    specialist_better = _rr(role="worker_explore", passed=True, generation_ms=20000, tokens_generated=200, elapsed_seconds=20.0)
    specialist_worse = _rr(role="worker_math", passed=False)
    both_correct_no_timing = _rr(role="unknown_role", passed=True, tokens_generated=0, generation_ms=0)
    both_wrong = _rr(role="worker_vision", passed=False)

    rewards = _MOD.compute_comparative_rewards(
        {
            "frontdoor:direct": baseline,
            "worker_explore:direct": specialist_better,
            "worker_math:direct": specialist_worse,
            "unknown:direct": both_correct_no_timing,
            "worker_vision:direct": both_wrong,
        }
    )

    assert rewards["frontdoor:direct"] == 1.0
    # both correct branch with cost penalty
    assert 0.1 <= rewards["worker_explore:direct"] < 0.5
    assert rewards["worker_math:direct"] == -0.5
    # both correct fallback without timing/role TPS
    assert rewards["unknown:direct"] == 0.3
    assert rewards["worker_vision:direct"] == -0.5


def test_compute_comparative_rewards_specialist_win_when_baseline_fails():
    rewards = _MOD.compute_comparative_rewards(
        {
            "frontdoor:direct": _rr(role="frontdoor", passed=False),
            "worker_math:direct": _rr(role="worker_math", passed=True),
        }
    )
    assert rewards["worker_math:direct"] == 1.0


def test_compute_comparative_rewards_both_wrong_branch():
    rewards = _MOD.compute_comparative_rewards(
        {
            "frontdoor:direct": _rr(role="frontdoor", passed=False),
            "worker_explore:direct": _rr(role="worker_explore", passed=False),
        }
    )
    assert rewards["worker_explore:direct"] == -0.3


def test_detect_escalation_chains_prefers_cheapest_passing_expensive_target():
    chains = _MOD.detect_escalation_chains(
        {
            "worker_explore:direct": _rr(role="worker_explore", mode="direct", passed=False, error=None),
            "coder_escalation:direct": _rr(role="coder_escalation", mode="direct", passed=True),
            "architect_general:direct": _rr(role="architect_general", mode="direct", passed=True),
            "worker_math:direct": _rr(role="worker_math", mode="direct", passed=False, error="backend down"),
        }
    )
    assert len(chains) == 1
    chain = chains[0]
    assert chain["from_role"] == "worker_explore"
    assert chain["to_role"] == "coder_escalation"
    assert chain["reward"] == _MOD.ESCALATION_REWARD


def test_reward_injection_helpers_handle_success_non_200_and_exception():
    comp = _MOD.ComparativeResult(
        suite="math",
        question_id="q1",
        prompt="prompt",
        expected="42",
        rewards={"frontdoor:direct": 1.0, "worker_math:direct": 0.0},
    )
    client = Mock()
    client.post.side_effect = [
        Mock(status_code=200),
        Mock(status_code=503),
        RuntimeError("boom"),
    ]

    injected_rewards = _MOD._inject_rewards_http(comp, "http://localhost:8000", client)
    assert injected_rewards == 1

    chains = [
        {"action": "escalate:worker->coder", "reward": 0.8, "from_role": "worker_explore", "to_role": "coder_escalation"}
    ]
    injected_chains = _MOD._inject_escalation_chains_http(comp, chains, "http://localhost:8000", client)
    assert injected_chains == 0


def test_success_reward_and_compute_3way_rewards_mapping():
    assert _MOD.success_reward(True) == 1.0
    assert _MOD.success_reward(False) == 0.0

    rewards = _MOD.compute_3way_rewards(
        {
            "frontdoor:direct": _rr(role="frontdoor", mode="direct", passed=True),
            "frontdoor:repl": _rr(role="frontdoor", mode="repl", passed=False),
            "architect_general:delegated": _rr(role="architect_general", mode="delegated", passed=True),
            "architect_coding:delegated": _rr(role="architect_coding", mode="delegated", passed=False),
        }
    )
    assert rewards[_MOD.ACTION_SELF_DIRECT] == 1.0
    assert rewards[_MOD.ACTION_SELF_REPL] == 0.0
    assert rewards[_MOD.ACTION_ARCHITECT] == 1.0

    # Legacy worker_vision:react path for SELF:repl
    legacy = _MOD.compute_3way_rewards(
        {
            "worker_vision:direct": _rr(role="worker_vision", mode="direct", passed=False),
            "worker_vision:react": _rr(role="worker_vision", mode="react", passed=True),
        }
    )
    assert legacy[_MOD.ACTION_SELF_REPL] == 1.0


def test_has_delegation_and_score_delegation_chain_paths():
    infra = _rr(error_type="infrastructure", delegation_events=[{"x": 1}])
    assert _MOD._has_delegation(infra) is False
    assert _MOD._has_delegation(_rr(delegation_events=[{"x": 1}])) is True
    assert _MOD._has_delegation(_rr(tools_called=["delegate_to_worker"])) is True
    assert _MOD._has_delegation(_rr(role_history=["frontdoor", "worker_math"])) is True
    assert _MOD._has_delegation(_rr()) is False

    repl_rr = _rr(
        role="frontdoor",
        mode="repl",
        passed=False,
        delegation_events=[{"event": "delegated"}],
        delegation_success=True,
    )
    arch_rr = _rr(
        role="architect_general",
        mode="delegated",
        passed=True,
        delegation_events=[{"event": "delegated"}],
        delegation_success=None,  # falls back to passed
    )
    rewards = _MOD.score_delegation_chain(
        {
            "frontdoor:repl": repl_rr,
            "architect_general:delegated": arch_rr,
        }
    )
    assert rewards[_MOD.ACTION_WORKER] == 1.0


def test_compute_tool_value():
    assert _MOD.compute_tool_value(False, True)["tools_helped"] is True
    assert _MOD.compute_tool_value(True, True)["tools_neutral"] is True
    assert _MOD.compute_tool_value(True, False)["tools_hurt"] is True
    assert _MOD.compute_tool_value(False, True)["tool_advantage"] == 1


def test_extract_web_research_telemetry_and_rewards():
    telemetry = _MOD.extract_web_research_telemetry(
        [
            {
                "pages_fetched": 4,
                "pages_synthesized": 2,
                "total_elapsed_ms": 120.5,
                "query": "q1",
                "sources": [{"url": "https://example.com/a"}, {"url": "https://docs.example.com/b"}],
            },
            {"query": "q2", "sources": [{"url": "not a real url"}]},
            "not-a-dict",
        ]
    )
    # call_count uses raw tool_results length by design
    assert telemetry.call_count == 3
    assert telemetry.total_pages_fetched == 4
    assert telemetry.total_pages_synthesized == 2
    assert math.isclose(telemetry.total_elapsed_ms, 120.5)
    assert telemetry.unique_domains >= 2
    assert telemetry.queries == ["q1", "q2"]

    no_calls = _MOD.compute_web_research_rewards(_MOD.WebResearchTelemetry(), passed=False)
    assert no_calls == {}

    rewards = _MOD.compute_web_research_rewards(telemetry, passed=True, f1_score=2.5)
    assert rewards["wr_accuracy"] == 1.0
    assert rewards["wr_completeness"] == 1.0  # clamped
    assert 0.0 <= rewards["wr_source_diversity"] <= 1.0
    assert 0.0 < rewards["wr_efficiency"] <= 1.0


def test_aggregate_web_research_reward_and_strategy_scoring():
    assert _MOD.aggregate_web_research_reward({}) == 0.0
    assert _MOD.aggregate_web_research_reward({"x": 1.0}, weights={"x": 0.0}) == 0.0

    agg = _MOD.aggregate_web_research_reward(
        {
            "wr_accuracy": 1.0,
            "wr_completeness": 0.5,
            "wr_source_diversity": 0.5,
            "wr_efficiency": 0.5,
        }
    )
    assert 0.0 <= agg <= 1.0

    strategy = _MOD.score_query_strategy(
        [
            {
                "query": "capital of france",
                "sources": [{"url": "https://example.com/a"}],
            },
            {
                "query": "population of paris",
                "sources": [{"url": "https://wikipedia.org/wiki/Paris"}],
            },
        ]
    )
    assert strategy["query_count"] == 2.0
    assert strategy["query_diversity"] >= 0.0
    assert strategy["source_yield"] >= 1.0

    assert _MOD.score_query_strategy([]) == {}


def test_compute_scratchpad_rewards_branches():
    assert _MOD.compute_scratchpad_rewards([], [], "answer", True) == {}

    web_only = _MOD.compute_scratchpad_rewards([], [{"query": "q"}], "answer", True)
    assert web_only["sp_insight_count"] == 0.0
    assert web_only["sp_web_insight_ratio"] == 0.0

    insights = [
        {"insight": "Search found source evidence about protein folding."},
        {"insight": "Final value is 42 from article data."},
    ]
    rewards = _MOD.compute_scratchpad_rewards(insights, [{"query": "q"}], "The final value is 42.", True)
    assert rewards["sp_insight_count"] == 2.0
    assert 0.0 <= rewards["sp_web_insight_ratio"] <= 1.0
    assert 0.0 <= rewards["sp_answer_containment"] <= 1.0

    # insights present but empty answer => containment forced to 0
    rewards_no_answer = _MOD.compute_scratchpad_rewards(insights, [], "", False)
    assert rewards_no_answer["sp_answer_containment"] == 0.0


def test_reward_injection_additional_success_and_exception_edges():
    comp = _MOD.ComparativeResult(
        suite="math",
        question_id="q2",
        prompt="prompt",
        expected="42",
        rewards={"frontdoor:direct": 1.0},
    )

    client_success = Mock()
    client_success.post.return_value = Mock(status_code=200)
    injected_chains = _MOD._inject_escalation_chains_http(
        comp,
        [{"action": "escalate:a->b", "reward": 0.7, "from_role": "a", "to_role": "b"}],
        "http://localhost:8000",
        client_success,
    )
    assert injected_chains == 1

    client_fail = Mock()
    client_fail.post.side_effect = RuntimeError("down")
    injected_rewards = _MOD._inject_rewards_http(comp, "http://localhost:8000", client_fail)
    assert injected_rewards == 0


def test_score_delegation_chain_infra_skip_and_architect_only_path():
    rewards = _MOD.score_delegation_chain(
        {
            "frontdoor:repl": _rr(
                role="frontdoor",
                mode="repl",
                passed=True,
                error_type="infrastructure",
                delegation_events=[{"event": "delegated"}],
            ),
            "architect_general:delegated": _rr(
                role="architect_general",
                mode="delegated",
                passed=False,
                delegation_events=[{"event": "delegated"}],
                delegation_success=None,
            ),
            "architect_coding:delegated": _rr(
                role="architect_coding",
                mode="delegated",
                passed=True,
                error_type="infrastructure",
                delegation_events=[{"event": "delegated"}],
            ),
        }
    )
    assert rewards[_MOD.ACTION_WORKER] == 0.0


def test_score_delegation_chain_repl_uses_passed_when_delegation_success_missing():
    rewards = _MOD.score_delegation_chain(
        {
            "frontdoor:repl": _rr(
                role="frontdoor",
                mode="repl",
                passed=True,
                delegation_events=[{"event": "delegated"}],
                delegation_success=None,
            ),
        }
    )
    assert rewards[_MOD.ACTION_WORKER] == 1.0


def test_web_research_telemetry_strategy_and_scratchpad_residual_branches():
    empty = _MOD.extract_web_research_telemetry([])
    assert empty.call_count == 0

    with patch.object(_MOD, "urlparse", side_effect=ValueError("bad url")):
        telemetry = _MOD.extract_web_research_telemetry(
            [{"sources": [{"url": "https://example.com"}]}]
        )
    assert telemetry.unique_domains == 0

    rewards = _MOD.compute_web_research_rewards(
        _MOD.WebResearchTelemetry(call_count=1, total_pages_fetched=0, unique_domains=9),
        passed=True,
    )
    assert rewards["wr_source_diversity"] == 0.0
    assert rewards["wr_efficiency"] == 0.0

    rewards_fail = _MOD.compute_web_research_rewards(
        _MOD.WebResearchTelemetry(call_count=1, total_pages_fetched=2, unique_domains=1),
        passed=False,
    )
    assert rewards_fail["wr_efficiency"] == 0.0

    with patch.object(_MOD, "urlparse", side_effect=ValueError("bad parse")):
        strategy = _MOD.score_query_strategy(
            [
                {"query": "single query", "sources": [{"url": "https://bad"}]},
                "not-a-dict",
            ]
        )
    assert strategy["query_count"] == 2.0
    assert strategy["query_diversity"] == 0.0
    assert strategy["source_yield"] == 0.0

    sp = _MOD.compute_scratchpad_rewards(
        [{"insight": "a an to in"}],
        [],
        "answer text",
        passed=False,
    )
    assert sp["sp_answer_containment"] == 0.0
