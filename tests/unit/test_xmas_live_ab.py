from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "xmas_live_ab.py"
SPEC = importlib.util.spec_from_file_location("xmas_live_ab", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
xmas_live_ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(xmas_live_ab)


def test_arm_sequence_uses_abba_order() -> None:
    assert xmas_live_ab.arm_sequence(1) == ["baseline", "xmas"]
    assert xmas_live_ab.arm_sequence(2) == ["baseline", "xmas", "xmas", "baseline"]


def test_reload_env_sets_launch_time_xmas_flags(tmp_path: Path) -> None:
    table = tmp_path / "xmas_winner_table.yaml"

    baseline = xmas_live_ab.reload_env("baseline", table)
    assert baseline["ORCHESTRATOR_XMAS_ROUTING_MODE"] == "off"
    assert baseline["ORCHESTRATOR_XMAS_WINNER_TABLE_PATH"] == ""

    candidate = xmas_live_ab.reload_env("xmas", table)
    assert candidate["ORCHESTRATOR_XMAS_ROUTING_MODE"] == "enforce"
    assert candidate["ORCHESTRATOR_XMAS_WINNER_TABLE_PATH"] == str(table)


def test_load_prompts_accepts_builtin_json_and_jsonl(tmp_path: Path) -> None:
    builtin = xmas_live_ab.load_prompts(None)
    assert builtin
    assert {item["domain"] for item in builtin} >= {"math", "code", "reasoning"}

    json_path = tmp_path / "prompts.json"
    json_path.write_text('{"prompts": [{"id": "a", "prompt": "A"}]}', encoding="utf-8")
    assert xmas_live_ab.load_prompts(json_path) == [{"id": "a", "prompt": "A"}]

    jsonl_path = tmp_path / "prompts.jsonl"
    jsonl_path.write_text('{"id": "a", "prompt": "A"}\n{"id": "b", "prompt": "B"}\n', encoding="utf-8")
    assert [item["id"] for item in xmas_live_ab.load_prompts(jsonl_path)] == ["a", "b"]


def test_score_answer_supports_common_methods() -> None:
    assert xmas_live_ab.score_answer("The answer is 36.", {"expected": "36"}) is True
    assert xmas_live_ab.score_answer("36", {"expected": "36", "scoring": "exact_match"}) is True
    assert xmas_live_ab.score_answer("<answer>valid</answer>", {"expected": "valid", "scoring": "exact_match"}) is True
    assert xmas_live_ab.score_answer("<answer>42</answer>", {"expected": "42"}) is True
    assert xmas_live_ab.score_answer("Answer: A", {"expected": "A", "scoring": "multiple_choice"}) is True
    assert xmas_live_ab.score_answer("<answer>A</answer>", {"expected": "A", "scoring": "multiple_choice"}) is True
    assert xmas_live_ab.score_answer("Answer: B", {"expected": "A", "scoring": "multiple_choice"}) is False
    assert xmas_live_ab.score_answer("anything", {}) is None


def test_summarize_reports_routes_scores_and_xmas_apply_count() -> None:
    rows = [
        {
            "arm": "baseline",
            "score": True,
            "elapsed_s": 10.0,
            "routing_strategy": "rules",
            "routed_to": "frontdoor",
        },
        {
            "arm": "baseline",
            "score": False,
            "elapsed_s": 20.0,
            "routing_strategy": "rules",
            "routed_to": "frontdoor",
        },
        {
            "arm": "xmas",
            "score": True,
            "elapsed_s": 7.0,
            "routing_strategy": "xmas_enforce:rules",
            "routed_to": "worker_general",
        },
        {
            "arm": "xmas",
            "score": True,
            "elapsed_s": 9.0,
            "routing_strategy": "rules",
            "routed_to": "frontdoor",
        },
    ]

    summary = xmas_live_ab.summarize(rows)

    assert summary["arms"]["baseline"]["score_rate"] == 0.5
    assert summary["arms"]["xmas"]["score_rate"] == 1.0
    assert summary["arms"]["xmas"]["xmas_applied_n"] == 1
    assert summary["arms"]["xmas"]["routed_to_counts"] == {
        "frontdoor": 1,
        "worker_general": 1,
    }
    assert summary["score_delta_xmas_minus_baseline"] == 0.5
