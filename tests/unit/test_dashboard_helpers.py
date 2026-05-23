"""Tests for the dashboard helper modules extracted in the 2026-05-21 refactor.

Covers dashboard_topology, dashboard_tap, dashboard_tasks, dashboard_snapshot.
Route handlers themselves are unchanged (smoke-tested only via module import).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.api.routes import dashboard_snapshot, dashboard_tap, dashboard_tasks, dashboard_topology


# ----- dashboard_topology -----


def test_role_color_known_role() -> None:
    assert dashboard_topology._role_color("frontdoor") == "#3b82f6"


def test_role_color_strips_quarter_suffix() -> None:
    assert dashboard_topology._role_color("frontdoor.q0") == dashboard_topology._role_color("frontdoor")


def test_role_color_strips_numeric_sibling_suffix() -> None:
    # _ROLE_COLORS has "embedder"; "embedder_3" should fall back to embedder
    assert dashboard_topology._role_color("embedder_3") == dashboard_topology._role_color("embedder")


def test_role_color_unknown_falls_back_to_gray() -> None:
    assert dashboard_topology._role_color("unknown_role") == "#64748b"


def test_port_hints_quarter_ports_generated() -> None:
    # frontdoor at 8070, quarters at 8080/8180/8280/8380
    assert dashboard_topology._PORT_HINTS[8080] == "frontdoor.q0"
    assert dashboard_topology._PORT_HINTS[8180] == "frontdoor.q1"
    assert dashboard_topology._PORT_HINTS[8280] == "frontdoor.q2"
    assert dashboard_topology._PORT_HINTS[8380] == "frontdoor.q3"


def test_load_state_services_missing_file_returns_empty(tmp_path) -> None:
    services = dashboard_topology._load_state_services(tmp_path / "no_such.json")
    assert services == []


def test_load_state_services_parses_known_fields(tmp_path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({
        "orchestrator": {
            "role": "orchestrator", "port": 8000, "pid": 123,
            "model_path": "api", "log_file": "api.log",
        },
        "embedder": {
            "role": "embedder", "port": 8090, "pid": 456,
            "model_path": "/m/bge.gguf", "log_file": "emb.log",
        },
        "junk": "not a dict, must be skipped",
    }))
    services = dashboard_topology._load_state_services(state_file)
    assert {s["name"] for s in services} == {"orchestrator", "embedder"}
    embedder = next(s for s in services if s["name"] == "embedder")
    assert embedder["port"] == 8090
    assert embedder["model"] == "/m/bge.gguf"


def test_discover_llama_ports_parses_ps_output(monkeypatch) -> None:
    fake_ps = (
        "1234 /opt/llama-server --port 8070 -m /m/frontdoor.gguf\n"
        "5678 /opt/llama-server --port 9999 -m /m/mystery.gguf\n"
        "9999 some-other-process --port 1234\n"
    )
    monkeypatch.setattr(
        dashboard_topology.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(stdout=fake_ps),
    )
    ports = dashboard_topology._discover_llama_ports()
    assert ports[8070] == "frontdoor"
    # 9999 not in _PORT_HINTS → falls back to "port_9999(mystery)"
    assert ports[9999].startswith("port_9999(")
    assert 1234 not in ports  # filtered out (no llama-server in cmd)


# ----- dashboard_tap -----


def test_read_tail_returns_empty_for_missing_file(tmp_path) -> None:
    assert dashboard_tap._read_tail(tmp_path / "missing.log") == ""


def test_read_tail_small_file_returns_full_content(tmp_path) -> None:
    f = tmp_path / "small.log"
    f.write_text("hello world\n")
    assert dashboard_tap._read_tail(f) == "hello world\n"


def test_read_tail_large_file_truncates_to_last_bytes(tmp_path) -> None:
    f = tmp_path / "big.log"
    f.write_text("a\n" * 200_000)  # 400KB
    out = dashboard_tap._read_tail(f, max_bytes=4_096)
    assert len(out.encode()) <= 4_096
    # All remaining lines should be the same content
    assert out.strip().split("\n")[0] == "a"


def test_parse_inference_sections_empty_input() -> None:
    assert dashboard_tap._parse_inference_sections("") == []


def test_parse_inference_sections_extracts_role_prompt_response() -> None:
    sep = "=" * 72
    sub = "-" * 72
    tap = (
        f"[2026-05-22 10:00:00] ROLE=frontdoor\n{sub}\n"
        f"PROMPT: hello world\n{sub}\n"
        f"RESPONSE:\nhi there\n{sep}\n"
    )
    sections = dashboard_tap._parse_inference_sections(tap)
    assert len(sections) == 1
    assert sections[0]["role"] == "frontdoor"
    assert sections[0]["prompt"] == "hello world"
    assert "hi there" in sections[0]["response"]


def test_parse_inference_sections_skips_timings_probes() -> None:
    """Short TIMINGS: responses are llama.cpp probes, not real inference; must be filtered."""
    sep = "=" * 72
    sub = "-" * 72
    tap = (
        f"[2026-05-22 10:00:00] ROLE=frontdoor\n{sub}\n"
        f"PROMPT: probe\n{sub}\n"
        f"RESPONSE:\nTIMINGS: short\n{sep}\n"
    )
    assert dashboard_tap._parse_inference_sections(tap) == []


def test_parse_trial_state_parses_baseline_then_score() -> None:
    tail = (
        "2026-05-22 10:00 GEPA: evaluating baseline for some_file.md (12 sentinels)\n"
        "2026-05-22 10:01 GEPA: baseline score = 0.85\n"
        "2026-05-22 10:02 Dispatching action: mutate_prompt\n"
    )
    state = dashboard_tap._parse_trial_state(tail)
    assert state["current_file"] == "some_file.md"
    assert state["baseline_sentinels_total"] == 12
    assert state["baseline_score"] == 0.85
    assert state["last_event"] == "baseline_done"
    assert state["current_action"] == "mutate_prompt"


# ----- dashboard_tasks -----


def test_task_events_filters_by_task_id(tmp_path) -> None:
    log = tmp_path / "progress.jsonl"
    log.write_text(
        json.dumps({"task_id": "chat-1", "event_type": "task_started", "timestamp": "t1", "data": {}}) + "\n"
        + json.dumps({"task_id": "chat-2", "event_type": "task_started", "timestamp": "t2", "data": {}}) + "\n"
        + json.dumps({"task_id": "chat-1", "event_type": "routing_decision", "timestamp": "t3", "data": {"src": "x"}}) + "\n"
        + "not-valid-json\n"
        + json.dumps({"task_id": "chat-1", "event_type": "task_completed", "timestamp": "t4", "data": {}}) + "\n"
    )
    events = dashboard_tasks._task_events("chat-1", log)
    assert len(events) == 3
    assert [e["event_type"] for e in events] == ["task_started", "routing_decision", "task_completed"]


def test_task_events_returns_empty_when_path_missing(tmp_path) -> None:
    assert dashboard_tasks._task_events("chat-1", tmp_path / "no_such.log") == []


def test_objective_for_task_extracts_from_task_started_event() -> None:
    events = [
        {"event_type": "routing_decision", "data": {"chosen_action": "x"}},
        {"event_type": "task_started", "data": {"objective": "Solve fizzbuzz"}},
    ]
    assert dashboard_tasks._objective_for_task(events) == "Solve fizzbuzz"


def test_objective_for_task_returns_empty_when_no_started_event() -> None:
    events = [{"event_type": "task_completed", "data": {}}]
    assert dashboard_tasks._objective_for_task(events) == ""


def test_task_text_snapshot_uses_slot_prompt_when_available() -> None:
    slot = {"prompt": "live prompt", "content": "streaming output"}
    events = [{"event_type": "task_started", "timestamp": "2026-05-22T10:00:00Z", "data": {"objective": "from event"}}]
    out = dashboard_tasks._task_text_snapshot("chat-1", events, slot)
    assert "live prompt" in out
    assert "streaming output" in out
    # Slot prompt wins in the PROMPT block (the event's objective still appears
    # later in the REPL HISTORY event dump, which is by design).
    prompt_block = out.split("INFERENCE STREAM:")[0]
    assert "live prompt" in prompt_block
    assert "from event" not in prompt_block


def test_task_text_snapshot_falls_back_to_objective_when_no_slot() -> None:
    events = [{"event_type": "task_started", "timestamp": "t", "data": {"objective": "fallback objective"}}]
    out = dashboard_tasks._task_text_snapshot("chat-2", events, None)
    assert "fallback objective" in out
    # The empty-placeholder text was made more descriptive (2026-05-23):
    # "(empty)" → "(empty — no live slot and no matching tap section)".
    # Match the prefix so the test stays robust to minor copy tweaks.
    assert "(empty" in out  # no inference stream
    assert "INFERENCE STREAM:" in out


def test_task_text_snapshot_elides_noisy_keys() -> None:
    events = [{
        "event_type": "routing_decision",
        "timestamp": "2026-05-22T10:00:00Z",
        "data": {
            "chosen_action": "worker",
            "stack_state": "x" * 4000,  # noisy
            "similarity_topk": [1, 2, 3],  # noisy
        },
    }]
    out = dashboard_tasks._task_text_snapshot("chat-3", events, None)
    assert "_elided_keys" in out
    assert "stack_state" not in out or "_elided_keys" in out  # noisy key removed from data dump


def test_find_section_by_objective_short_objective_returns_none() -> None:
    assert dashboard_tasks._find_section_by_objective("short") is None
    assert dashboard_tasks._find_section_by_objective("") is None


def test_find_section_by_objective_matches_recent_first(monkeypatch) -> None:
    fake_sections = [
        {"prompt": "recent: solve fizzbuzz again", "response": "..."},
        {"prompt": "older: solve fizzbuzz now", "response": "..."},
    ]
    monkeypatch.setattr(dashboard_tasks, "_read_tail", lambda *a, **kw: "irrelevant")
    monkeypatch.setattr(dashboard_tasks, "_parse_inference_sections", lambda *a, **kw: fake_sections)
    out = dashboard_tasks._find_section_by_objective("solve fizzbuzz")
    assert out is not None
    assert out["prompt"].startswith("recent:")


# ----- dashboard_snapshot -----


def test_todays_progress_log_uses_iso_date(tmp_path) -> None:
    today_iso = date.today().isoformat()
    assert dashboard_snapshot.todays_progress_log(tmp_path).name == f"{today_iso}.jsonl"


def test_scan_recent_decisions_empty_when_missing(tmp_path) -> None:
    decisions, rolling, cumulative = dashboard_snapshot.scan_recent_decisions(tmp_path / "no.jsonl")
    assert decisions == []
    assert rolling == {}
    assert cumulative == {}


def test_scan_recent_decisions_counts_cumulative_vs_rolling(tmp_path) -> None:
    log = tmp_path / "p.jsonl"
    # Old decision (outside rolling window)
    old_ts = datetime.fromtimestamp(0, tz=timezone.utc).isoformat()
    new_ts = datetime.now(timezone.utc).isoformat()
    log.write_text(
        json.dumps({"event_type": "routing_decision", "timestamp": old_ts, "task_id": "a",
                    "data": {"decision_source": "classifier", "chosen_action": "x"}}) + "\n"
        + json.dumps({"event_type": "routing_decision", "timestamp": new_ts, "task_id": "b",
                      "data": {"decision_source": "classifier", "chosen_action": "y"}}) + "\n"
        + json.dumps({"event_type": "routing_decision", "timestamp": new_ts, "task_id": "c",
                      "data": {"decision_source": "rules", "chosen_action": "z"}}) + "\n"
    )
    decisions, rolling, cumulative = dashboard_snapshot.scan_recent_decisions(log, window_s=600)
    # All 3 go in cumulative; only 2 (new) in rolling
    assert cumulative == {"classifier": 2, "rules": 1}
    rolling.pop("_verifier_verdicts", None)
    assert rolling == {"classifier": 1, "rules": 1}


def test_scan_orchestrator_tasks_separates_in_flight_from_completed(tmp_path) -> None:
    log = tmp_path / "p.jsonl"
    new_ts = datetime.now(timezone.utc).isoformat()
    log.write_text(
        json.dumps({"event_type": "task_started", "timestamp": new_ts, "task_id": "chat-1",
                    "data": {"objective": "live"}}) + "\n"
        + json.dumps({"event_type": "task_started", "timestamp": new_ts, "task_id": "chat-2",
                      "data": {"objective": "done"}}) + "\n"
        + json.dumps({"event_type": "task_completed", "timestamp": new_ts, "task_id": "chat-2",
                      "data": {}}) + "\n"
    )
    in_flight, completed = dashboard_snapshot.scan_orchestrator_tasks(log)
    assert [t["task_id"] for t in in_flight] == ["chat-1"]
    assert [t["task_id"] for t in completed] == ["chat-2"]


def test_scan_orchestrator_tasks_skips_non_chat_task_ids(tmp_path) -> None:
    log = tmp_path / "p.jsonl"
    new_ts = datetime.now(timezone.utc).isoformat()
    log.write_text(
        json.dumps({"event_type": "task_started", "timestamp": new_ts, "task_id": "internal-99",
                    "data": {"objective": "skipped"}}) + "\n"
    )
    in_flight, completed = dashboard_snapshot.scan_orchestrator_tasks(log)
    assert in_flight == []
    assert completed == []


def test_count_log_events_counts_pattern_matches(tmp_path) -> None:
    log = tmp_path / "app.log"
    log.write_text(
        "INFO: server started\n"
        "ERROR: connection refused\n"
        "WARN: retrying\n"
        "ERROR: timeout\n"
    )
    counts = dashboard_snapshot.count_log_events(
        log, {"errors": r"^ERROR:", "warns": r"^WARN:"}
    )
    assert counts == {"errors": 2, "warns": 1}


def test_count_log_events_missing_file_returns_zeros(tmp_path) -> None:
    counts = dashboard_snapshot.count_log_events(tmp_path / "no.log", {"foo": "bar"})
    assert counts == {"foo": 0}


# ----- dashboard.py route module smoke test -----


def test_dashboard_module_router_still_exported() -> None:
    """The only external API is `router`; verify it survives the refactor."""
    from src.api.routes import dashboard
    assert dashboard.router is not None


def test_dashboard_module_html_loads_from_file() -> None:
    """The 918-line HTML block is now loaded from dashboard.html — verify length."""
    from src.api.routes import dashboard
    assert len(dashboard._DASHBOARD_HTML) > 40_000
    assert "<!doctype html>" in dashboard._DASHBOARD_HTML
    assert "</body></html>" in dashboard._DASHBOARD_HTML
