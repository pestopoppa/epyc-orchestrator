from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_validate_table_requires_function_axis(monkeypatch, tmp_path: Path) -> None:
    table = tmp_path / "xmas_winner_table.yaml"
    table.write_text("version: test\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(cmd, *, cwd, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(xmas_live_ab.subprocess, "run", fake_run)

    xmas_live_ab.validate_table(table)

    assert "--require-function-axis" in captured["cmd"]
    assert captured["cwd"] == xmas_live_ab.ORCH


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


def test_real_run_requires_explicit_prompt_manifest(tmp_path: Path) -> None:
    args = SimpleNamespace(
        table=tmp_path / "xmas_winner_table.yaml",
        prompts=None,
        dry_run=False,
    )

    try:
        xmas_live_ab.run(args)
    except SystemExit as exc:
        assert "pass --prompts with a held-out prompt manifest" in str(exc)
    else:
        raise AssertionError("real X-MAS A/B must reject built-in smoke prompts")


def test_dry_run_can_use_builtin_smoke_set_without_inference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    called = {"chat": False, "restart": False}

    def fail_chat(*args, **kwargs):
        called["chat"] = True
        raise AssertionError("dry-run must not call chat")

    def fail_restart(*args, **kwargs):
        called["restart"] = True
        raise AssertionError("dry-run must not restart orchestrator")

    monkeypatch.setattr(xmas_live_ab, "validate_table", lambda _table: None)
    monkeypatch.setattr(xmas_live_ab, "chat", fail_chat)
    monkeypatch.setattr(xmas_live_ab, "restart_orchestrator", fail_restart)
    output = tmp_path / "dryrun"
    args = SimpleNamespace(
        table=tmp_path / "xmas_winner_table.yaml",
        prompts=None,
        sample_size=1,
        reps=1,
        output=output,
        max_turns=1,
        dry_run=True,
        host_quiet_confirmed=False,
        timeout_s=1.0,
        restore_baseline=True,
    )

    assert xmas_live_ab.run(args) == 0
    assert called == {"chat": False, "restart": False}
    assert json.loads((output / "meta.json").read_text())["prompt_manifest"] == (
        "builtin_smoke"
    )
    assert json.loads((output / "summary.json").read_text())["dry_run"] is True


def test_score_answer_supports_common_methods() -> None:
    assert xmas_live_ab.score_answer("The answer is 36.", {"expected": "36"}) is True
    assert xmas_live_ab.score_answer("36", {"expected": "36", "scoring": "exact_match"}) is True
    assert xmas_live_ab.score_answer("<answer>valid</answer>", {"expected": "valid", "scoring": "exact_match"}) is True
    assert xmas_live_ab.score_answer("<answer>42</answer>", {"expected": "42"}) is True
    assert xmas_live_ab.score_answer("Answer: A", {"expected": "A", "scoring": "multiple_choice"}) is True
    assert xmas_live_ab.score_answer("<answer>A</answer>", {"expected": "A", "scoring": "multiple_choice"}) is True
    assert xmas_live_ab.score_answer("Answer: B", {"expected": "A", "scoring": "multiple_choice"}) is False
    assert xmas_live_ab.score_answer("anything", {}) is None


def test_chat_records_http_errors(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, json: dict):
            raise xmas_live_ab.httpx.ReadTimeout("timed out")

    monkeypatch.setattr(xmas_live_ab.httpx, "Client", FailingClient)

    result = xmas_live_ab.chat("prompt", timeout_s=1.0, session_id="s", max_turns=1)

    assert result["status"] == 0
    assert result["body"]["answer"] == ""
    assert result["body"]["error_code"] == "ReadTimeout"


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
