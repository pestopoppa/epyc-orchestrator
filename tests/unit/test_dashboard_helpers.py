"""Tests for the dashboard helper modules extracted in the 2026-05-21 refactor.

Covers dashboard_topology, dashboard_tap, dashboard_tasks, dashboard_snapshot.
Route handlers themselves are unchanged (smoke-tested only via module import).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from src.api.routes import (
    dashboard,
    dashboard_snapshot,
    dashboard_tap,
    dashboard_tasks,
    dashboard_topology,
)


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


def test_expected_stack_services_include_embedder_fleet() -> None:
    services = dashboard_topology.expected_stack_services()
    embedders = [s for s in services if s.get("embedding")]

    assert [s["port"] for s in embedders] == [
        8090, 8091, 8092, 8093, 8094, 8095, 8096, 8097, 8098,
    ]
    assert [s["role"] for s in embedders] == [
        "embedder",
        "embedder_1",
        "embedder_2",
        "embedder_3",
        "embedder_4",
        "embedder_5",
        "embedder_granite_97m_r2",
        "embedder_multilingual_e5_base",
        "embedder_bge_m3",
    ]


def test_topology_emits_expected_unloaded_stack_servers(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "_discover_llama_ports", lambda: {})
    monkeypatch.setattr(dashboard, "_discover_llama_models", lambda: {})
    monkeypatch.setattr(dashboard, "_load_state_services", lambda: [])

    response = asyncio.run(dashboard.topology())
    data = json.loads(response.body)
    by_port = {node["port"]: node for node in data["nodes"]}

    embedder = by_port[8090]
    assert embedder["role"] == "embedder"
    assert embedder["kind"] == "expected-stack-server"
    assert embedder["expected"] is True
    assert embedder["running"] is False


def test_topology_activity_initializes_expected_embedder_bucket(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dashboard, "_read_tail", lambda *a, **kw: "")
    monkeypatch.setattr(dashboard, "_parse_inference_sections", lambda *a, **kw: [])
    monkeypatch.setattr(dashboard, "_todays_progress_log", lambda: tmp_path / "missing.jsonl")

    response = asyncio.run(dashboard.topology_activity())
    data = json.loads(response.body)

    embedder = data["per_role"]["embedder"]
    assert embedder["expected"] is True
    assert embedder["n_recent"] == 0
    assert embedder["n_completed"] == 0


def test_service_port_hints_are_auxiliary_only() -> None:
    hints = dashboard_topology._service_port_hints()
    assert hints[8000] == "orchestrator"
    assert hints[8090] == "embedder"
    assert "worker_fast" not in hints.values()


def test_stack_prior_port_hints_skip_alias_records(tmp_path) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        json.dumps(
            {
                "roles": {
                    "frontdoor": {
                        "deployment_status": "live_stack",
                        "serving": {
                            "ports": [8070, 8080, 8180],
                            "launch": {
                                "primary_roles": ["frontdoor"],
                                "entries": [
                                    {
                                        "port": 8070,
                                        "primary_role": "frontdoor",
                                        "alias": False,
                                        "numa_instance": 0,
                                    },
                                    {
                                        "port": 8080,
                                        "primary_role": "frontdoor",
                                        "alias": False,
                                        "numa_instance": 1,
                                    },
                                    {
                                        "port": 8180,
                                        "primary_role": "frontdoor",
                                        "alias": False,
                                        "numa_instance": 2,
                                    },
                                ],
                            },
                        },
                    },
                    "coder_escalation": {
                        "deployment_status": "live_stack",
                        "serving": {
                            "ports": [8070],
                            "launch": {
                                "primary_roles": ["frontdoor"],
                                "entries": [
                                    {
                                        "port": 8070,
                                        "primary_role": "frontdoor",
                                        "alias": True,
                                        "numa_instance": 0,
                                    },
                                ],
                            },
                        },
                    },
                    "candidate_only": {
                        "deployment_status": "benchmark_or_candidate",
                        "serving": {
                            "ports": [9999],
                            "launch": {"primary_roles": [], "entries": []},
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    hints = dashboard_topology._stack_prior_port_hints(priors)
    assert hints[8070] == "frontdoor"
    assert hints[8080] == "frontdoor.q0"
    assert hints[8180] == "frontdoor.q1"
    assert 9999 not in hints
    assert "coder_escalation" not in hints.values()


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
            "role": "embedder", "port": 8090, "pid": 999999999,
            "model_path": "/m/bge.gguf", "log_file": "emb.log",
        },
        "junk": "not a dict, must be skipped",
    }))
    services = dashboard_topology._load_state_services(state_file)
    assert {s["name"] for s in services} == {"orchestrator", "embedder"}
    embedder = next(s for s in services if s["name"] == "embedder")
    assert embedder["port"] == 8090
    assert embedder["model"] == "/m/bge.gguf"
    assert embedder["running"] is False


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


def test_parse_structured_tap_requests_groups_events_by_request() -> None:
    def event(kind: str, request_id: str, **fields):
        payload = {
            "event": kind,
            "request_id": request_id,
            "role": "frontdoor",
            "ts": f"2026-05-22T10:00:0{len(lines)}+00:00",
            "ts_epoch": float(len(lines)),
            **fields,
        }
        lines.append(json.dumps(payload))

    lines: list[str] = []
    event("start", "req-1", prompt="hello", task_id="task-1", trial_id=3, batch_id="b1")
    event("metadata", "req-1", instance_idx=2, instance_shape="q1", port=8082, topology_hash="abc123")
    event("chunk", "req-1", text="hi ")
    event("chunk", "req-1", text="there")
    event("timings", "req-1", tokens=2, prompt_ms=10.0, gen_ms=20.0, tps=100.0, total_s=0.03)
    event("end", "req-1")

    parsed = dashboard_tap._parse_structured_tap_requests("\n".join(lines))

    assert len(parsed) == 1
    req = parsed[0]
    assert req["request_id"] == "req-1"
    assert req["task_id"] == "task-1"
    assert req["trial_id"] == 3
    assert req["batch_id"] == "b1"
    assert req["instance_idx"] == 2
    assert req["instance_shape"] == "q1"
    assert req["port"] == 8082
    assert req["topology_hash"] == "abc123"
    assert req["prompt"] == "hello"
    assert req["response"] == "hi there"
    assert req["status"] == "complete"
    assert req["is_live"] is False
    assert req["chunk_count"] == 2
    assert "2 tokens" in req["timings"]


def test_parse_structured_tap_requests_marks_quiet_open_request() -> None:
    line = json.dumps({
        "event": "start",
        "request_id": "req-quiet",
        "role": "coder_escalation",
        "topology_role": "frontdoor",
        "lock_role": "frontdoor",
        "ts": "2026-05-22T10:00:00+00:00",
        "ts_epoch": 100.0,
        "prompt": "hello",
    })

    parsed = dashboard_tap._parse_structured_tap_requests(
        line,
        now_epoch=130.0,
        quiet_after_s=10.0,
    )

    assert len(parsed) == 1
    req = parsed[0]
    assert req["status"] == "quiet"
    assert req["is_live"] is True
    assert req["age_s"] == 30.0
    assert req["quiet_s"] == 30.0
    assert req["topology_role"] == "frontdoor"
    assert "no tap output" in req["status_reason"]


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


def test_read_autopilot_phase_returns_dict(tmp_path, monkeypatch) -> None:
    phase_path = tmp_path / "phase.json"
    phase_path.write_text(json.dumps({"phase": "dispatch_action", "trial_id": 12}))
    monkeypatch.setattr(dashboard, "AUTOPILOT_PHASE_PATH", phase_path)

    phase = dashboard._read_autopilot_phase()

    assert phase["phase"] == "dispatch_action"
    assert phase["trial_id"] == 12


def test_read_autopilot_phase_invalid_returns_empty(tmp_path, monkeypatch) -> None:
    phase_path = tmp_path / "phase.json"
    phase_path.write_text("not json")
    monkeypatch.setattr(dashboard, "AUTOPILOT_PHASE_PATH", phase_path)

    assert dashboard._read_autopilot_phase() == {}


def test_autopilot_phase_health_reports_active(tmp_path, monkeypatch) -> None:
    phase_path = tmp_path / "phase.json"
    phase_path.write_text(json.dumps({
        "phase": "dispatch_action",
        "trial_id": 12,
        "action_type": "numeric_trial",
        "pid": os.getpid(),
        "updated_at": time.time(),
    }))
    monkeypatch.setattr(dashboard, "AUTOPILOT_PHASE_PATH", phase_path)

    health = dashboard._autopilot_phase_health()

    assert health["ok"] is True
    assert health["status"] == "active"
    assert health["trial_id"] == 12
    assert health["pid_alive"] is True


def test_autopilot_phase_health_reports_stale(tmp_path, monkeypatch) -> None:
    phase_path = tmp_path / "phase.json"
    phase_path.write_text(json.dumps({
        "phase": "dispatch_action",
        "trial_id": 12,
        "action_type": "deep_eval",
        "pid": os.getpid(),
        "updated_at": time.time() - 3600,
    }))
    monkeypatch.setattr(dashboard, "AUTOPILOT_PHASE_PATH", phase_path)

    health = dashboard._autopilot_phase_health()

    assert health["ok"] is False
    assert health["status"] == "stale"
    assert health["blockers"]


def test_process_status_includes_autopilot_phase_health(tmp_path, monkeypatch) -> None:
    phase_path = tmp_path / "phase.json"
    phase_path.write_text(json.dumps({
        "phase": "dispatch_action",
        "trial_id": 12,
        "action_type": "numeric_trial",
        "pid": os.getpid(),
        "updated_at": time.time(),
    }))
    monkeypatch.setattr(dashboard, "AUTOPILOT_PHASE_PATH", phase_path)
    monkeypatch.setattr(dashboard, "AUTOPILOT_LOG", tmp_path / "missing.log")
    monkeypatch.setattr(
        dashboard,
        "_process_info_by_match",
        lambda _match: {"running": True, "pid": os.getpid()},
    )

    response = asyncio.run(dashboard.process_status())
    payload = json.loads(response.body)

    assert payload["autopilot_phase"]["trial_id"] == 12
    assert payload["autopilot_phase_health"]["status"] == "active"
    assert payload["autopilot_phase_age_s"] == payload["autopilot_phase_health"]["heartbeat_age_s"]


def test_repo_readiness_summary_loads_latest_advisory_queue(tmp_path) -> None:
    data_dir = tmp_path / "data"
    progress_dir = tmp_path / "progress"
    data_dir.mkdir()
    progress_dir.mkdir()
    (data_dir / "repo_readiness_2026-06-19.json").write_text(
        json.dumps({"generated_at": "old", "repos": {}}),
        encoding="utf-8",
    )
    report = data_dir / "repo_readiness_2026-06-20.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-20T00:00:00Z",
                "portfolio": {"maturity": {"achieved_level": 2}},
                "repos": {
                    "epyc-root": {"maturity": {"achieved_level": 4}},
                },
            }
        ),
        encoding="utf-8",
    )
    queue = data_dir / "repo_readiness_remediation_queue_2026-06-20.json"
    queue.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-20T00:00:01Z",
                "version": 1,
                "item_count": 2,
                "items": [
                    {"priority": "P1", "repo": "epyc-root", "criterion_id": "L5.auto_eval"},
                    {"priority": "P0", "repo": "epyc-orchestrator", "criterion_id": "L3.security"},
                ],
            }
        ),
        encoding="utf-8",
    )
    markdown = progress_dir / "repo-readiness-remediation-2026-06-20.md"
    markdown.write_text("# queue\n", encoding="utf-8")

    summary = dashboard._repo_readiness_summary(
        data_dir=data_dir,
        progress_dir=progress_dir,
        top_n=1,
    )

    assert summary["available"] is True
    assert summary["authority"] == "advisory"
    assert summary["autopilot_gate"] is False
    assert summary["report_path"] == str(report)
    assert summary["queue_path"] == str(queue)
    assert summary["markdown_path"] == str(markdown)
    assert summary["generated_at"] == "2026-06-20T00:00:01Z"
    assert summary["portfolio_level"] == {"achieved_level": 2}
    assert summary["repo_levels"]["epyc-root"] == {"achieved_level": 4}
    assert summary["priority_counts"] == {"P1": 1, "P0": 1}
    assert summary["top_items"] == [
        {"priority": "P0", "repo": "epyc-orchestrator", "criterion_id": "L3.security"}
    ]


def test_repo_readiness_route_is_advisory_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "REPO_READINESS_DIR", tmp_path / "missing-data")
    monkeypatch.setattr(dashboard, "REPO_READINESS_PROGRESS_DIR", tmp_path / "missing-progress")

    response = asyncio.run(dashboard.repo_readiness())
    payload = json.loads(response.body)

    assert payload["available"] is False
    assert payload["authority"] == "advisory"
    assert payload["autopilot_gate"] is False
    assert payload["item_count"] == 0
    assert payload["top_items"] == []


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


def test_pareto_from_journal_excludes_tier0(tmp_path, monkeypatch) -> None:
    """T0 sentinel rows (10q, quality saturates ~2.4=8/10) must not enter the
    dashboard-reconstructed frontier/hypervolume — mirrors ec9622d's archive
    exclusion so the operator panel shows the real T1/T2 frontier, not a phantom 2.4."""
    journal = tmp_path / "autopilot_journal.jsonl"
    rows = [
        # T0 sentinel: saturated quality that WOULD dominate if admitted
        {"trial_id": 10, "tier": 0, "quality": 2.4, "speed": 60.0, "cost": 0.5,
         "reliability": 0.9, "timestamp": "2026-05-31T10:00:00+00:00"},
        # honest T1 entry — the real frontier point
        {"trial_id": 11, "tier": 1, "quality": 1.9, "speed": 70.0, "cost": 0.5,
         "reliability": 0.9, "timestamp": "2026-05-31T10:01:00+00:00"},
    ]
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal)

    archive = dashboard._pareto_from_journal(None, current_run_only=False)
    assert archive is not None
    frontier_ids = {e["trial_id"] for e in archive["frontier"]}
    assert frontier_ids == {11}, "only the honest T1 entry belongs on the frontier"
    assert all(e["trial_id"] != 10 for e in archive["all_entries"]), "T0 excluded entirely"


def test_pareto_from_journal_segregates_tiers(tmp_path, monkeypatch) -> None:
    """T2 validation quality must not dominate the canonical T1 dashboard frontier."""
    journal = tmp_path / "autopilot_journal.jsonl"
    rows = [
        {"trial_id": 20, "tier": 1, "quality": 1.5, "speed": 30.0, "cost": 0.5,
         "reliability": 0.9, "timestamp": "2026-05-31T10:00:00+00:00"},
        {"trial_id": 21, "tier": 2, "quality": 2.4, "speed": 40.0, "cost": 0.5,
         "reliability": 0.9, "timestamp": "2026-05-31T10:01:00+00:00"},
    ]
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal)

    archive = dashboard._pareto_from_journal(None, current_run_only=False)
    assert archive is not None
    assert [e["trial_id"] for e in archive["frontier"]] == [20]
    assert [e["trial_id"] for e in archive["frontiers_by_tier"]["2"]] == [21]


def test_task_text_snapshot_falls_back_to_objective_when_no_slot() -> None:
    events = [{"event_type": "task_started", "timestamp": "t", "data": {"objective": "fallback objective"}}]
    out = dashboard_tasks._task_text_snapshot("chat-2", events, None)
    assert "fallback objective" in out
    # The empty-placeholder text was made more descriptive (2026-05-23):
    # "(empty)" → "(empty — no live slot and no matching tap section)".
    # Match the prefix so the test stays robust to minor copy tweaks.
    assert "(empty" in out  # no inference stream
    assert "INFERENCE STREAM:" in out


def test_task_text_snapshot_uses_structured_tap_section() -> None:
    tap_section = {
        "source": "structured_tap",
        "request_id": "req-1",
        "started_at": "2026-05-22T10:00:00+00:00",
        "role": "frontdoor",
        "prompt": "structured prompt",
        "response": "structured response",
    }
    out = dashboard_tasks._task_text_snapshot(
        "tap_req-1", [], None, tap_section=tap_section
    )
    assert "structured prompt" in out
    assert "structured response" in out
    assert "inference_tap_events.jsonl request req-1" in out


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


def test_find_structured_request_by_id_strips_tap_prefix(monkeypatch) -> None:
    lines = [
        json.dumps({
            "event": "start",
            "request_id": "req-1",
            "role": "frontdoor",
            "ts": "2026-05-22T10:00:00+00:00",
            "ts_epoch": 1.0,
            "prompt": "hello",
        }),
        json.dumps({
            "event": "response",
            "request_id": "req-1",
            "role": "frontdoor",
            "ts": "2026-05-22T10:00:01+00:00",
            "ts_epoch": 2.0,
            "text": "world",
        }),
    ]
    monkeypatch.setattr(
        dashboard_tasks, "_grep_lines_reverse", lambda *a, **kw: "\n".join(lines)
    )
    monkeypatch.setattr(
        dashboard_tasks, "_read_tail", lambda *args, **kwargs: "\n".join(lines)
    )

    out = dashboard_tasks._find_structured_request_by_id("tap_req-1")

    assert out is not None
    assert out["source"] == "structured_tap"
    assert out["request_id"] == "req-1"
    assert out["prompt"] == "hello"
    assert out["response"] == "world"


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


def test_find_section_by_objective_role_filtered_no_global_fallback(monkeypatch) -> None:
    # Caller knows the task ran under frontdoor; the global pass would
    # otherwise return a syntactically-valid but cross-contaminated section
    # written by worker_explore (regression guard for the 2026-05-30
    # chat-83123001/chat-c7bf9580 interleaved-write incident).
    fake_sections = [
        {"role": "worker_explore", "prompt": "solve fizzbuzz now", "response": "wrong"},
    ]
    monkeypatch.setattr(dashboard_tasks, "_read_tail", lambda *a, **kw: "irrelevant")
    monkeypatch.setattr(dashboard_tasks, "_parse_inference_sections", lambda *a, **kw: fake_sections)
    out = dashboard_tasks._find_section_by_objective(
        "solve fizzbuzz", expected_role="frontdoor"
    )
    assert out is None


def test_find_section_by_objective_role_alias_canonicalizes_worker_explore(monkeypatch) -> None:
    fake_sections = [
        {"role": "worker_general", "prompt": "solve fizzbuzz now", "response": "ok"},
    ]
    monkeypatch.setattr(dashboard_tasks, "_read_tail", lambda *a, **kw: "irrelevant")
    monkeypatch.setattr(dashboard_tasks, "_parse_inference_sections", lambda *a, **kw: fake_sections)
    out = dashboard_tasks._find_section_by_objective(
        "solve fizzbuzz", expected_role="worker_explore"
    )
    assert out is not None
    assert out["role"] == "worker_general"


def test_find_structured_request_by_task_id_matches_chat_id(monkeypatch) -> None:
    # Each chat task has its own derived request_id; resolving by task_id
    # gives a deterministic mapping for chat-* dashboard task ids.
    lines = [
        json.dumps({
            "event": "start",
            "request_id": "chat-83123001:b763498c",
            "task_id": "chat-83123001",
            "role": "frontdoor",
            "ts": "2026-05-30T22:16:24+00:00",
            "ts_epoch": 1.0,
            "prompt": "count monkey collisions",
        }),
        json.dumps({
            "event": "response",
            "request_id": "chat-83123001:b763498c",
            "task_id": "chat-83123001",
            "role": "frontdoor",
            "ts": "2026-05-30T22:16:53+00:00",
            "ts_epoch": 2.0,
            "text": "the actual frontdoor answer",
        }),
        # Interleaved worker_explore request that previously confused the
        # plaintext fallback — verify the task_id lookup does not return it.
        json.dumps({
            "event": "start",
            "request_id": "chat-c7bf9580:aaaa1111",
            "task_id": "chat-c7bf9580",
            "role": "worker_explore",
            "ts": "2026-05-30T22:16:25+00:00",
            "ts_epoch": 1.5,
            "prompt": "count spaces in pirate's speak",
        }),
        json.dumps({
            "event": "response",
            "request_id": "chat-c7bf9580:aaaa1111",
            "task_id": "chat-c7bf9580",
            "role": "worker_explore",
            "ts": "2026-05-30T22:16:28+00:00",
            "ts_epoch": 1.6,
            "text": "5",
        }),
    ]
    # The resolver reverse-greps the tap file first (recovers a request of any age
    # from a multi-GB tap), then falls back to the small live tail. Stub both seams
    # so the matching logic is exercised against the fixture, not the real file.
    monkeypatch.setattr(
        dashboard_tasks, "_grep_lines_reverse", lambda *a, **kw: "\n".join(lines)
    )
    monkeypatch.setattr(
        dashboard_tasks, "_read_tail", lambda *args, **kwargs: "\n".join(lines)
    )

    out = dashboard_tasks._find_structured_request_by_task_id("chat-83123001")

    assert out is not None
    assert out["source"] == "structured_tap"
    assert out["task_id"] == "chat-83123001"
    assert out["role"] == "frontdoor"
    assert out["prompt"] == "count monkey collisions"
    assert out["response"] == "the actual frontdoor answer"


def test_find_structured_request_by_task_id_missing_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(dashboard_tasks, "_grep_lines_reverse", lambda *a, **kw: "")
    monkeypatch.setattr(dashboard_tasks, "_read_tail", lambda *a, **kw: "")
    assert dashboard_tasks._find_structured_request_by_task_id("chat-83123001") is None
    assert dashboard_tasks._find_structured_request_by_task_id("") is None


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
    rolling_public = {k: v for k, v in rolling.items() if not k.startswith("_")}
    assert rolling_public == {"classifier": 1, "rules": 1}


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


def _aged_ts(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def test_base_role_normalization() -> None:
    assert dashboard_topology.base_role("frontdoor.q2") == "frontdoor"
    assert dashboard_topology.base_role("embedder_3") == "embedder"
    assert dashboard_topology.base_role("worker_general.q0") == "worker_general"
    # Multi-word roles without a numeric instance suffix are left intact.
    assert dashboard_topology.base_role("architect_general") == "architect_general"
    assert dashboard_topology.base_role("ingest_long_context") == "ingest_long_context"
    assert dashboard_topology.base_role("") == ""


def test_role_color_canonicalizes_worker_aliases() -> None:
    assert dashboard_topology._role_color("worker_general") == "#10b981"
    assert dashboard_topology._role_color("worker_explore") == "#10b981"
    assert dashboard_topology._role_color("worker_explore.q1") == "#10b981"


def test_scan_orchestrator_tasks_role_aware_inflight_cutoff(tmp_path) -> None:
    """Slow roles keep a wider in-flight ceiling; default roles are truncated."""
    log = tmp_path / "p.jsonl"
    old = _aged_ts(600)  # past the 300s default, inside the 900s architect ceiling
    log.write_text(
        json.dumps({"event_type": "task_started", "timestamp": old, "task_id": "chat-slow",
                    "data": {"objective": "long reasoning"}}) + "\n"
        + json.dumps({"event_type": "routing_decision", "timestamp": old, "task_id": "chat-slow",
                      "data": {"chosen_action": "architect_general"}}) + "\n"
        + json.dumps({"event_type": "task_started", "timestamp": old, "task_id": "chat-fd",
                      "data": {"objective": "chat"}}) + "\n"
        + json.dumps({"event_type": "routing_decision", "timestamp": old, "task_id": "chat-fd",
                      "data": {"chosen_action": "frontdoor"}}) + "\n"
    )
    in_flight, _ = dashboard_snapshot.scan_orchestrator_tasks(log)
    ids = {t["task_id"] for t in in_flight}
    assert "chat-slow" in ids  # architect_general ceiling is 900s
    assert "chat-fd" not in ids  # frontdoor uses the 300s default


def test_scan_orchestrator_tasks_stamps_canonical_role(tmp_path) -> None:
    """In-flight tasks carry a base-normalised `role` for consistent grouping."""
    log = tmp_path / "p.jsonl"
    now = _aged_ts(2)
    log.write_text(
        json.dumps({"event_type": "task_started", "timestamp": now, "task_id": "chat-q",
                    "data": {"objective": "q"}}) + "\n"
        + json.dumps({"event_type": "routing_decision", "timestamp": now, "task_id": "chat-q",
                      "data": {"chosen_action": "frontdoor.q2"}}) + "\n"
    )
    in_flight, _ = dashboard_snapshot.scan_orchestrator_tasks(log)
    assert in_flight and in_flight[0]["role"] == "frontdoor"


def test_gate_inflight_drops_idle_orphans() -> None:
    """An old started-but-unterminated task with no busy slot is a restart-orphan."""
    orphan = [{"task_id": "chat-x", "age_s": 500.0, "role": "frontdoor"}]
    assert dashboard._gate_inflight_by_live_slots(orphan, {}) == []
    assert dashboard._gate_inflight_by_live_slots(orphan, {"frontdoor": 0}) == []


def test_gate_inflight_keeps_fresh_without_busy_slot() -> None:
    """A just-started task is kept even before its slot flips to processing."""
    fresh = [{"task_id": "chat-y", "age_s": 3.0, "role": "frontdoor"}]
    gated = dashboard._gate_inflight_by_live_slots(fresh, {})
    assert len(gated) == 1
    assert gated[0]["live_state"] == "pending"


def test_gate_inflight_keeps_live_long_task() -> None:
    """An old task is kept while its role's server reports a busy slot."""
    long_task = [{"task_id": "chat-z", "age_s": 500.0, "role": "ingest_long_context"}]
    gated = dashboard._gate_inflight_by_live_slots(long_task, {"ingest_long_context": 1})
    assert len(gated) == 1
    assert gated[0]["live_state"] == "decoding"


def test_gate_inflight_maps_alias_roles_to_topology_busy_slots() -> None:
    """Logical aliases sharing a physical pool should track that pool's live slots."""
    task = [{"task_id": "chat-coder", "age_s": 45.0, "role": "coder_escalation"}]
    gated = dashboard._gate_inflight_by_live_slots(
        task,
        {"frontdoor": 1},
        alias_to_topology_role={"coder_escalation": "frontdoor"},
    )

    assert len(gated) == 1
    assert gated[0]["topology_role"] == "frontdoor"
    assert gated[0]["live_state"] == "decoding"


def test_gate_inflight_canonicalizes_worker_explore_alias_to_general() -> None:
    task = [{"task_id": "chat-worker", "age_s": 45.0, "role": "worker_explore"}]
    gated = dashboard._gate_inflight_by_live_slots(task, {"worker_general": 1})

    assert len(gated) == 1
    assert gated[0]["topology_role"] == "worker_general"
    assert gated[0]["live_state"] == "decoding"


def test_gate_inflight_strips_route_modifier_before_slot_match() -> None:
    task = [{"task_id": "chat-frontdoor", "age_s": 45.0, "role": "frontdoor:direct"}]
    gated = dashboard._gate_inflight_by_live_slots(task, {"frontdoor": 1})

    assert len(gated) == 1
    assert gated[0]["topology_role"] == "frontdoor"
    assert gated[0]["live_state"] == "decoding"


def test_gate_inflight_caps_to_busy_slots() -> None:
    """With one busy slot, only the newest non-fresh candidate is shown."""
    many = [{"task_id": f"chat-{i}", "age_s": 100.0 + i * 10, "role": "frontdoor"} for i in range(3)]
    gated = dashboard._gate_inflight_by_live_slots(many, {"frontdoor": 1})
    assert [t["task_id"] for t in gated] == ["chat-0"]


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


def test_dashboard_html_surfaces_autopilot_phase_health() -> None:
    from src.api.routes import dashboard

    html = dashboard._DASHBOARD_HTML
    assert "autopilot_phase_health" in html
    assert "phase health" in html


def test_dashboard_effective_journal_rows_fold_supersession_events() -> None:
    rows = [
        {
            "trial_id": 1,
            "timestamp": "2026-06-13T00:00:00+00:00",
            "bug_corrupted_by": "",
        },
        {
            "trial_id": 2,
            "timestamp": "2026-06-13T00:01:00+00:00",
            "bug_corrupted_by": "",
        },
        {
            "type": "supersession",
            "timestamp": "2026-06-13T00:02:00+00:00",
            "target_trial_ids": [2],
            "fields": {"bug_corrupted_by": "resource_contention"},
        },
    ]

    effective = dashboard._effective_journal_trial_rows(rows)

    assert [row["trial_id"] for row in effective] == [1, 2]
    assert effective[1]["bug_corrupted_by"] == "resource_contention"
    assert rows[1]["bug_corrupted_by"] == ""


def test_dashboard_baseline_promotion_summary_scopes_to_current_run() -> None:
    rows = [
        {
            "trial_id": 50,
            "timestamp": "2026-05-28T08:59:00+00:00",
        },
        {
            "type": "baseline_promotion",
            "source_trial_id": 50,
            "tier": 1,
            "previous_quality": 1.0,
            "new_quality": 1.1,
            "timestamp": "2026-05-28T08:59:30+00:00",
            "reason": "old run",
        },
        {
            "trial_id": 0,
            "timestamp": "2026-05-28T09:00:00+00:00",
        },
        {
            "trial_id": 1,
            "timestamp": "2026-05-28T09:01:00+00:00",
        },
        {
            "type": "baseline_promotion",
            "source_trial_id": 1,
            "tier": 1,
            "previous_quality": 1.1,
            "new_quality": 1.4,
            "timestamp": "2026-05-28T09:01:30+00:00",
            "reason": "current run",
            "proof": {
                "matrix_status": "ok",
                "speed_metric_mode": "aggregate_batch_tps",
            },
            "result_metrics": {
                "quality": 1.4,
                "speed": 42.0,
                "pareto_status": "frontier",
            },
        },
    ]

    summary = dashboard._baseline_promotion_summary(rows, current_run_only=True)

    assert summary["count"] == 1
    event = summary["recent"][0]
    assert event["source_trial_id"] == 1
    assert round(event["quality_delta"], 3) == 0.3
    assert event["matrix_status"] == "ok"
    assert event["result_speed"] == 42.0


def test_autopilot_progress_uses_superseded_journal_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    started_at = datetime.now(timezone.utc).timestamp() - 45
    state_path.write_text(json.dumps({
        "in_flight_trial": {
            "trial_id": 4,
            "started_at": started_at,
            "action": {"type": "seed_batch"},
        }
    }))
    rows = [
        {
            "trial_id": 0,
            "timestamp": "2026-06-13T00:00:00+00:00",
            "action_type": "seed_batch",
        },
        {
            "trial_id": 1,
            "timestamp": "2026-06-13T00:01:00+00:00",
            "action_type": "seed_batch",
        },
        {
            "trial_id": 2,
            "timestamp": "2026-06-13T00:02:00+00:00",
            "action_type": "seed_batch",
        },
        {
            "trial_id": 3,
            "timestamp": "2026-06-13T00:03:00+00:00",
            "action_type": "seed_batch",
        },
        {
            "type": "supersession",
            "timestamp": "2026-06-13T00:04:00+00:00",
            "target_trial_ids": [2],
            "fields": {"bug_corrupted_by": "resource_contention"},
        },
    ]
    journal_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_STATE_PATH", state_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)

    response = asyncio.run(dashboard.autopilot_progress())
    payload = json.loads(response.body)

    assert payload["percent_source"] == "action_p50"
    assert payload["n_action_type_samples"] == 2


def test_gepa_status_uses_superseded_journal_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "autopilot.log"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    log_path.write_text("2026-06-13 00:00:00,000 GEPA: Trial active\n")
    rows = [
        {
            "trial_id": 1,
            "timestamp": "2026-06-13T00:00:00+00:00",
            "species": "prompt_forge",
            "quality": 1.0,
            "speed": 10.0,
            "cost": 0.5,
            "reliability": 1.0,
            "pareto_status": "frontier",
        },
        {
            "trial_id": 2,
            "timestamp": "2026-06-13T00:01:00+00:00",
            "species": "prompt_forge",
            "quality": 9.0,
            "speed": 99.0,
            "cost": 0.5,
            "reliability": 1.0,
            "pareto_status": "frontier",
        },
        {
            "type": "supersession",
            "timestamp": "2026-06-13T00:02:00+00:00",
            "target_trial_ids": [2],
            "fields": {"bug_corrupted_by": "resource_contention"},
        },
    ]
    journal_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(dashboard, "AUTOPILOT_LOG", log_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL", journal_path)

    response = asyncio.run(dashboard.gepa_status())
    payload = json.loads(response.body)

    assert [trial["trial_id"] for trial in payload["recent_trials"]] == [1]


def test_pareto_endpoint_prefers_current_journal_run_over_old_rows_and_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Stale pre-reset journal rows and stale state should not pollute plots."""
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    session_start = datetime(2026, 5, 28, 9, 0, tzinfo=timezone.utc)
    state_path.write_text(json.dumps({
        "autopilot_fleet_started_at": session_start.timestamp(),
        "trial_counter": 12,
        "pareto_archive": {
            "frontier": [{"trial_id": 99, "objectives": [9.0, 9.0, 0.0, 0.0]}],
            "all_entries": [{"trial_id": 99, "objectives": [9.0, 9.0, 0.0, 0.0]}],
            "hypervolume_history": [[99, 999.0]],
        },
    }))
    rows = [
        {
            "trial_id": 568,
            "timestamp": "2026-05-28T08:59:00+00:00",
            "species": "old",
            "quality": 10.0,
            "speed": 100.0,
            "cost": 0.5,
            "reliability": 1.0,
        },
        {
            "trial_id": 0,
            "timestamp": "2026-05-28T09:00:00+00:00",
            "species": "reset",
            "quality": 0.8,
            "speed": 7.0,
            "cost": 0.5,
            "reliability": 0.7,
        },
        {
            "trial_id": 10,
            "timestamp": "2026-05-28T09:01:00+00:00",
            "species": "seeder",
            "quality": 1.0,
            "speed": 10.0,
            "cost": 0.5,
            "reliability": 0.8,
        },
        {
            "trial_id": 11,
            "timestamp": "2026-05-28T09:02:00+00:00",
            "species": "seeder",
            "quality": 0.5,
            "speed": 5.0,
            "cost": 0.5,
            "reliability": 0.7,
        },
        {
            "trial_id": 12,
            "timestamp": "2026-05-28T09:03:00+00:00",
            "species": "numeric_swarm",
            "quality": 1.2,
            "speed": 9.0,
            "cost": 0.5,
            "reliability": 0.9,
        },
        {
            "trial_id": 13,
            "timestamp": "2026-05-28T09:04:00+00:00",
            "species": "corrupt",
            "quality": 20.0,
            "speed": 200.0,
            "cost": 0.5,
            "reliability": 1.0,
            "bug_corrupted_by": "test",
        },
        {
            "type": "baseline_promotion",
            "source_trial_id": 12,
            "tier": 1,
            "previous_quality": 1.0,
            "new_quality": 1.2,
            "timestamp": "2026-05-28T09:04:30+00:00",
            "reason": "accepted",
            "proof": {"matrix_status": "ok"},
            "result_metrics": {
                "quality": 1.2,
                "speed": 9.0,
                "pareto_status": "frontier",
            },
        },
    ]
    journal_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_STATE_PATH", state_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)

    response = asyncio.run(dashboard.pareto())
    payload = json.loads(response.body)

    assert payload["source"] == "journal_current_run"
    assert payload["legacy_state_archive_warning"] is None
    assert payload["archive_authority"] == {
        "source": "journal_current_run",
        "journal_rows_available": 7,
        "state_archive_present": True,
        "state_error": None,
        "using_legacy_state_archive": False,
    }
    assert payload["totals"] == {
        "frontier_size": 2,
        "all_entries": 4,
        "hv_points": 4,
    }
    assert {point["trial_id"] for point in payload["frontier"]} == {10, 12}
    assert [point["trial_id"] for point in payload["dominated"]] == [11, 0]
    assert payload["hypervolume_history"][-1][0] == 12
    assert payload["journal_run_start_trial_id"] == 0
    assert payload["baseline_promotions"]["count"] == 1
    assert payload["baseline_promotions"]["recent"][0]["source_trial_id"] == 12


def test_pareto_endpoint_uses_current_run_when_restart_marker_is_newer(
    tmp_path: Path, monkeypatch
) -> None:
    """A new run marker just after the last row must not collapse the plots."""
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    session_start = datetime(2026, 5, 28, 9, 43, 33, 596323, tzinfo=timezone.utc)
    state_path.write_text(json.dumps({
        "autopilot_fleet_started_at": session_start.timestamp(),
        "trial_counter": 73,
        "pareto_archive": {
            "frontier": [{"trial_id": 99, "objectives": [9.0, 9.0, 0.0, 0.0]}],
            "all_entries": [{"trial_id": 99, "objectives": [9.0, 9.0, 0.0, 0.0]}],
            "hypervolume_history": [[99, 999.0]],
        },
    }))
    rows = [
        {
            "trial_id": 72,
            "timestamp": "2026-05-28T09:05:40.024205+00:00",
            "species": "seeder",
            "quality": 1.2,
            "speed": 28.0,
            "cost": 0.5,
            "reliability": 0.8,
        },
        {
            "trial_id": 73,
            "timestamp": "2026-05-28T09:43:33.596165+00:00",
            "species": "(killed)",
            "quality": 0.0,
            "speed": 0.0,
            "cost": 0.0,
            "reliability": 0.0,
            "bug_corrupted_by": "autopilot_killed_mid_trial",
        },
    ]
    journal_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_STATE_PATH", state_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)

    response = asyncio.run(dashboard.pareto())
    payload = json.loads(response.body)

    assert payload["source"] == "journal_current_run"
    assert payload["totals"] == {
        "frontier_size": 1,
        "all_entries": 1,
        "hv_points": 1,
    }
    assert payload["frontier"][0]["trial_id"] == 72
    assert payload["hypervolume_history"][-1][0] == 72


def test_pareto_endpoint_marks_legacy_state_archive_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    """State-cache fallback should be visible enough to block decision use."""
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    state_path.write_text(json.dumps({
        "trial_counter": 5,
        "pareto_archive": {
            "frontier": [{"trial_id": 5, "objectives": [1.0, 2.0, -0.1, 0.9]}],
            "all_entries": [{"trial_id": 5, "objectives": [1.0, 2.0, -0.1, 0.9]}],
            "hypervolume_history": [[5, 2.0]],
        },
    }))
    journal_path.write_text("")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_STATE_PATH", state_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)

    response = asyncio.run(dashboard.pareto())
    payload = json.loads(response.body)

    assert payload["source"] == "state_archive"
    assert payload["legacy_state_archive_warning"] == {
        "state_archive_present": True,
        "journal_rows_available": 0,
        "detail": (
            "dashboard fell back to autopilot_state.json:pareto_archive; "
            "treat this as a legacy state-cache view and run strict archive "
            "authority validation before using it for decisions"
        ),
    }
    assert payload["archive_authority"] == {
        "source": "state_archive",
        "journal_rows_available": 0,
        "state_archive_present": True,
        "state_error": None,
        "using_legacy_state_archive": True,
    }
