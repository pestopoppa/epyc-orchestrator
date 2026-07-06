"""Tests for the dashboard helper modules extracted in the 2026-05-21 refactor.

Covers dashboard_topology, dashboard_tap, dashboard_tasks, dashboard_snapshot.
Route handlers themselves are unchanged (smoke-tested only via module import).
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
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
    monkeypatch.setattr(dashboard, "_read_tap_events_tail", lambda *a, **kw: "")
    monkeypatch.setattr(dashboard, "_parse_inference_sections", lambda *a, **kw: [])
    monkeypatch.setattr(dashboard, "_todays_progress_log", lambda: tmp_path / "missing.jsonl")
    monkeypatch.setattr(
        dashboard,
        "_discover_llama_ports",
        lambda: {
            8090: "embedder",
            8091: "embedder_1",
            8092: "embedder_2",
            8093: "embedder_3",
            8094: "embedder_4",
            8095: "embedder_5",
        },
    )

    response = asyncio.run(dashboard.topology_activity())
    data = json.loads(response.body)

    embedder = data["per_role"]["embedder"]
    assert embedder["expected"] is True
    assert embedder["running"] is True
    assert embedder["expected_ports"] == [8090, 8091, 8092, 8093, 8094, 8095]
    assert embedder["running_ports"] == [8090, 8091, 8092, 8093, 8094, 8095]
    assert embedder["expected_instance_count"] == 6
    assert embedder["running_instance_count"] == 6
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
        "4242 /mnt/raid0/llm/llama.cpp-mi210-hip/build-hip/bin/llama-server --port 8802 -m /m/Qwen.gguf\n"
        "9999 some-other-process --port 1234\n"
    )
    monkeypatch.setattr(
        dashboard_topology.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(stdout=fake_ps),
    )
    ports = dashboard_topology._discover_llama_ports()
    assert ports[8070] == "frontdoor"
    # Unmapped ports get an honest role, never a model-mangled one — the old
    # "port_9999(mystery)" fallback leaked garbled keys into live_busy_by_role.
    assert ports[9999] == "extern_9999"
    # MI210 HIP builds are the GPU testbed (operator-decided first-class role).
    assert ports[8802] == "mi210_gpu"
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


def test_structured_tap_active_ignores_stale_quiet_history() -> None:
    assert dashboard._structured_tap_active(
        [
            {"status": "quiet", "quiet_s": 300.0},
            {"status": "complete", "quiet_s": 0.0},
        ]
    ) is False
    assert dashboard._structured_tap_active(
        [{"status": "running", "quiet_s": 15.0}]
    ) is False
    assert dashboard._structured_tap_active(
        [{"status": "running", "quiet_s": 2.0}]
    ) is True


def test_inference_tap_snapshot_marks_stale_sentinel_inactive(
    monkeypatch,
    tmp_path,
) -> None:
    sentinel = tmp_path / "inference_tap_active"
    sentinel.touch()
    event_line = json.dumps(
        {
            "event": "start",
            "request_id": "req-stale",
            "role": "ingest_long_context",
            "ts": "2026-05-22T10:00:00+00:00",
            "ts_epoch": 100.0,
            "prompt": "planner critique",
        }
    )

    monkeypatch.setattr(dashboard, "_TAP_SENTINEL_PATH", sentinel)
    monkeypatch.setattr(dashboard.time, "time", lambda: 130.0)
    monkeypatch.setattr(dashboard, "_read_tail", lambda *a, **kw: "")
    monkeypatch.setattr(dashboard, "_read_tap_events_tail", lambda *a, **kw: event_line)
    monkeypatch.setattr(dashboard, "_latest_tap_events_mtime", lambda: 100.0)

    response = asyncio.run(dashboard.inference_tap_snapshot())
    payload = json.loads(response.body)

    assert payload["tap_sentinel_active"] is True
    assert payload["tap_active"] is False
    assert payload["structured_requests"][0]["status"] == "quiet"


def test_structured_tap_requests_for_dashboard_uses_shared_region_lock_frame(monkeypatch) -> None:
    now = 100.0
    event_line = json.dumps(
        {
            "event": "start",
            "request_id": "chat-coder:abc",
            "role": "coder_escalation",
            "port": 8070,
            "ts": "2026-05-22T10:00:00+00:00",
            "ts_epoch": now - 1,
            "prompt": "fix code",
        }
    )
    region_locks = {
        "by_role": {
            "frontdoor": {
                "instances": [
                    {"idx": 0, "shape": "half0", "regions": ["q0", "q1"]},
                    {"idx": 3, "shape": "q2", "regions": ["q2"]},
                ],
            },
        },
    }

    monkeypatch.setattr(dashboard, "_read_tap_events_tail", lambda *a, **kw: event_line)
    enriched = dashboard._structured_tap_requests_for_dashboard(
        max_requests=20,
        now_epoch=now,
        region_locks=region_locks,
        port_roles={8070: "frontdoor"},
    )

    assert len(enriched) == 1
    req = enriched[0]
    assert req["role"] == "coder_escalation"
    assert req["topology_role"] == "frontdoor"
    assert req["lock_role"] == "frontdoor"
    assert req["instance_idx"] == 0
    assert req["instance_shape"] == "half0"
    assert req["instance_regions"] == ["q0", "q1"]


def test_topology_activity_uses_structured_tap_not_legacy_sections(monkeypatch) -> None:
    now = 1_000.0
    structured_lines = [
        json.dumps({
            "event": "start",
            "request_id": "req-worker",
            "role": "worker_general",
            "topology_role": "worker_general",
            "port": 8072,
            "ts": "2026-07-02T20:00:00+00:00",
            "ts_epoch": now - 5,
            "prompt": "worker prompt",
        }),
        json.dumps({
            "event": "timings",
            "request_id": "req-worker",
            "role": "worker_general",
            "topology_role": "worker_general",
            "port": 8072,
            "ts": "2026-07-02T20:00:01+00:00",
            "ts_epoch": now - 4,
            "tokens": 10,
            "prompt_ms": 0,
            "gen_ms": 1000,
            "tps": 42.0,
            "total_s": 1.0,
        }),
    ]
    legacy_tap = (
        f"[2026-07-02 20:00:00] ROLE=frontdoor\n{'-' * 72}\n"
        f"PROMPT: stale legacy prompt\n{'-' * 72}\n"
        f"RESPONSE:\nTIMINGS: 1 tokens in 1.00s (prompt=0ms, gen=1000ms, 9.0 t/s)\n"
        f"{'=' * 72}\n"
    )

    def fake_read_tail(path, *args, **kwargs):
        if str(path).endswith("inference_tap_events.jsonl"):
            return "\n".join(structured_lines)
        if str(path).endswith("inference_tap.log"):
            return legacy_tap
        return ""

    monkeypatch.setattr(dashboard, "_read_tail", fake_read_tail)
    monkeypatch.setattr(dashboard, "_read_tap_events_tail", fake_read_tail)
    monkeypatch.setattr(
        dashboard,
        "_discover_llama_ports",
        lambda: {8070: "frontdoor", 8072: "worker_general"},
    )
    monkeypatch.setattr(dashboard, "expected_stack_services", lambda: [
        {"role": "frontdoor", "port": 8070},
        {"role": "worker_general", "port": 8072},
    ])
    monkeypatch.setattr(dashboard, "_todays_progress_log", lambda: Path("/does/not/exist"))
    monkeypatch.setattr(dashboard, "_scan_recent_decisions", lambda _path: ([], [], []))
    monkeypatch.setattr(dashboard, "_scan_orchestrator_tasks", lambda *a, **kw: ([], []))
    monkeypatch.setattr(dashboard.time, "time", lambda: now)

    response = asyncio.run(dashboard.topology_activity(window_s=60.0))
    payload = json.loads(response.body)

    assert payload["per_role"]["worker_general"]["n_recent"] == 1
    assert payload["per_role"]["worker_general"]["avg_tps_recent"] == 42.0
    assert payload["per_role"]["frontdoor"]["n_recent"] == 0


def test_enrich_structured_tap_requests_recovers_alias_port_lock_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "role_aliases",
        lambda role: ["coder_escalation", "worker_summarize"] if role == "frontdoor" else [],
    )
    region_locks = {
        "by_role": {
            "frontdoor": {
                "instances": [
                    {"idx": 0, "shape": "half0", "regions": ["q0", "q1"]},
                    {"idx": 3, "shape": "q2", "regions": ["q2"]},
                ],
            },
            "worker_general": {
                "instances": [
                    {"idx": 0, "shape": "full", "regions": ["q0", "q1", "q2", "q3"]},
                    {"idx": 3, "shape": "q2", "regions": ["q2"]},
                ],
            },
        },
    }

    enriched = dashboard._enrich_structured_tap_requests(
        [
            {"request_id": "coder", "role": "coder_escalation", "port": 8070},
            {"request_id": "worker", "role": "worker_general", "port": 8072},
        ],
        port_roles={8070: "frontdoor", 8072: "worker_general"},
        region_locks=region_locks,
    )

    coder, worker = enriched
    assert coder["role"] == "coder_escalation"
    assert coder["topology_role"] == "frontdoor"
    assert coder["lock_role"] == "frontdoor"
    assert coder["instance_idx"] == 0
    assert coder["instance_shape"] == "half0"
    assert coder["instance_regions"] == ["q0", "q1"]
    assert worker["topology_role"] == "worker_general"
    assert worker["lock_role"] == "worker_general"
    assert worker["instance_shape"] == "full"
    assert worker["instance_regions"] == ["q0", "q1", "q2", "q3"]


def test_enrich_structured_tap_requests_uses_quarter_port_shape(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "role_aliases", lambda role: [])
    region_locks = {
        "by_role": {
            "worker_general": {
                "instances": [
                    {"idx": 0, "shape": "full", "regions": ["q0", "q1", "q2", "q3"]},
                    {"idx": 3, "shape": "q2", "regions": ["q2"]},
                ],
            },
        },
    }

    [req] = dashboard._enrich_structured_tap_requests(
        [{"request_id": "quarter", "role": "worker_general", "port": 8282}],
        port_roles={8282: "worker_general.q2"},
        region_locks=region_locks,
    )

    assert req["topology_role"] == "worker_general"
    assert req["instance_idx"] == 3
    assert req["instance_shape"] == "q2"
    assert req["instance_regions"] == ["q2"]


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


def test_autopilot_current_code_health_includes_restart_advice(monkeypatch) -> None:
    def fake_phase_health(**_kwargs):
        return {
            "ok": False,
            "status": "code_stale",
            "phase": "dispatch_action",
            "pid": os.getpid(),
            "pid_alive": True,
            "trial_id": 1207,
            "action_type": "seed_batch",
            "idle_reason": "evaluating question",
            "code_stale": True,
            "blockers": ["autopilot process predates runtime source changes: autopilot.py"],
        }

    monkeypatch.setattr(dashboard, "build_phase_health_report", fake_phase_health)

    health = dashboard._autopilot_current_code_health()

    assert health is not None
    assert health["restart_advice"]["restart_needed"] is True
    assert health["restart_advice"]["safe_to_restart_now"] is False
    assert health["restart_advice"]["status"] == "wait_for_boundary"


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
    """Validation/stress quality must not dominate the canonical T1 dashboard frontier."""
    journal = tmp_path / "autopilot_journal.jsonl"
    rows = [
        {"trial_id": 20, "tier": 1, "quality": 1.5, "speed": 30.0, "cost": 0.5,
         "reliability": 0.9, "timestamp": "2026-05-31T10:00:00+00:00"},
        {"trial_id": 21, "tier": 2, "quality": 2.4, "speed": 40.0, "cost": 0.5,
         "reliability": 0.9, "timestamp": "2026-05-31T10:01:00+00:00"},
        {"trial_id": 22, "tier": 3, "quality": 1.2, "speed": 45.0, "cost": 0.5,
         "reliability": 0.9, "timestamp": "2026-05-31T10:02:00+00:00"},
    ]
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal)

    archive = dashboard._pareto_from_journal(None, current_run_only=False)
    assert archive is not None
    assert [e["trial_id"] for e in archive["frontier"]] == [20]
    assert [e["trial_id"] for e in archive["frontiers_by_tier"]["2"]] == [21]
    assert [e["trial_id"] for e in archive["frontiers_by_tier"]["3"]] == [22]


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
    assert out["source"] == "legacy_inference_tap"
    assert out["legacy_source"] is True
    assert out["non_authoritative"] is True


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
    assert out["legacy_source"] is True


def test_task_text_snapshot_marks_legacy_tap_source_non_authoritative() -> None:
    out = dashboard_tasks._task_text_snapshot(
        "chat-legacy",
        [],
        None,
        tap_section={
            "source": "legacy_inference_tap",
            "legacy_source": True,
            "non_authoritative": True,
            "timestamp": "2026-05-22T10:00:00+00:00",
            "role": "frontdoor",
            "prompt": "hello",
            "response": "world",
        },
    )

    assert "legacy best-effort/non-authoritative" in out


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


def test_read_strategy_store_rows_reads_read_only_sqlite(tmp_path, monkeypatch) -> None:
    store_dir = tmp_path / "strategies"
    store_dir.mkdir()
    db_path = store_dir / "strategies.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE strategies (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            insight TEXT NOT NULL,
            source_trial_id INTEGER,
            species TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}',
            entry_type TEXT DEFAULT 'raw',
            evidence_trial_ids TEXT DEFAULT '[]'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO strategies (
            id, description, insight, source_trial_id, species, created_at,
            metadata_json, entry_type, evidence_trial_ids
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "row-1",
            "operator handoff row",
            "keep prompt graphs compact",
            12,
            "prompt_forge",
            "2026-06-28T00:00:00+00:00",
            json.dumps({"bind_status": "future", "seed_campaign": "graph-panel"}),
            "pattern",
            json.dumps([12]),
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(dashboard, "_STRATEGY_STORE_PATH", store_dir)

    rows = dashboard._read_strategy_store_rows()

    assert rows is not None
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "row-1"
    assert row["entry_type"] == "pattern"
    assert row["metadata"]["bind_status"] == "future"
    assert row["evidence_trial_ids"] == [12]


def test_read_planner_hint_seed_rows_reads_yaml(tmp_path, monkeypatch) -> None:
    seed_path = tmp_path / "operator_seed_strategies.yaml"
    seed_path.write_text(
        """
- slug: graph-panel
  tranche: green
  species: prompt_forge
  entry_type: pattern
  title: Graph panel
  description: keep the dashboard graph small
  insight: Keep the dashboard panel compact.
  evidence_trial_ids: [12]
  source_handoff: dashboard-graph-panel
  seeded_reason: Keep the dashboard panel compact
  confidence: medium
  bind_status: future
  bind_identifiers: [graph_panel]
""".strip()
    )

    monkeypatch.setattr(dashboard, "_PLANNER_HINT_SEEDS_PATH", seed_path)

    rows = dashboard._read_planner_hint_seed_rows()

    assert rows is not None
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "seed:graph-panel"
    assert row["planner_hint"] is True
    assert row["metadata"]["source_handoff"] == "dashboard-graph-panel"
    assert row["metadata"]["bind_identifiers"] == ["graph_panel"]
    assert row["evidence_trial_ids"] == [12]


def test_insight_graph_endpoint_merges_strategy_and_journal_rows(monkeypatch) -> None:
    journal_rows = [
        {
            "trial_id": 12,
            "timestamp": "2026-06-28T00:01:00+00:00",
            "species": "prompt_forge",
            "action_type": "prompt_mutation",
            "pareto_status": "frontier",
            "quality": 1.5,
            "speed": 42.0,
            "hypothesis": "Keep prompt graphs compact",
            "reasoning": "The operator hint panel should stay small.",
        },
        {
            "trial_id": 13,
            "timestamp": "2026-06-28T00:02:00+00:00",
            "species": "prompt_forge",
            "action_type": "prompt_mutation",
            "pareto_status": "dominated",
            "quality": 1.2,
            "speed": 39.0,
            "parent_trial": 12,
            "reasoning": "A longer prompt was slower.",
        },
    ]
    strategy_rows = [
        {
            "id": "journal-frontier-trial-12",
            "description": "prompt mutation",
            "insight": "q=1.5 s=42.0 mechanism=compact graph",
            "source_trial_id": 12,
            "species": "prompt_forge",
            "created_at": "2026-06-28T00:01:05+00:00",
            "metadata": {
                "generated_from": "journal_frontier",
                "journal_trial_id": 12,
            },
            "entry_type": "raw",
            "evidence_trial_ids": [12],
        },
        {
            "id": "opseed-graph-panel",
            "description": "Graph panel seed",
            "insight": "Show compact insight graphs in the dashboard",
            "source_trial_id": 12,
            "species": "prompt_forge",
            "created_at": "2026-06-28T00:03:00+00:00",
            "metadata": {
                "seed_campaign": "graph-panel",
                "source_handoff": "dashboard-graph-panel",
                "bind_status": "future",
                "seeded_by": "operator",
                "seeded_reason": "Keep the dashboard panel compact",
            },
            "entry_type": "pattern",
            "evidence_trial_ids": [12],
        },
        {
            "id": "opseed-graph-panel-live",
            "description": "Graph panel seed live",
            "insight": "Applied insight path",
            "source_trial_id": 13,
            "species": "prompt_forge",
            "created_at": "2026-06-28T00:04:00+00:00",
            "metadata": {
                "seed_campaign": "graph-panel",
                "source_handoff": "dashboard-graph-panel",
                "bind_status": "live",
                "seeded_by": "operator",
            },
            "entry_type": "convention",
            "evidence_trial_ids": [13],
        },
    ]
    monkeypatch.setattr(dashboard, "_read_autopilot_journal_rows", lambda path=None: journal_rows)
    monkeypatch.setattr(dashboard, "_read_strategy_store_rows", lambda path=None: strategy_rows)

    response = asyncio.run(dashboard.insight_graph(focus="campaign:graph-panel", depth=2))
    data = json.loads(response.body)

    assert data["available"] is True
    assert data["read_only"] is True
    assert data["focus"]["focus_kind"] == "campaign"
    assert data["focus"]["reason"] == "matched focus query"
    assert data["summary"]["state_counts"]["applied"] >= 1
    assert data["summary"]["state_counts"]["pending"] >= 1

    kinds = {node["kind"] for node in data["nodes"]}
    assert {"journal", "strategy", "campaign", "handoff"} <= kinds

    edge_kinds = {edge["kind"] for edge in data["edges"]}
    assert {"projection", "campaign", "handoff", "parent"} <= edge_kinds


def test_insight_graph_endpoint_falls_back_to_planner_hints(monkeypatch, tmp_path) -> None:
    seed_path = tmp_path / "operator_seed_strategies.yaml"
    seed_path.write_text(
        """
- slug: graph-panel
  tranche: green
  species: prompt_forge
  entry_type: pattern
  title: Graph panel
  description: keep the dashboard graph small
  insight: Keep the dashboard panel compact.
  evidence_trial_ids: []
  source_handoff: dashboard-graph-panel
  seeded_reason: Keep the dashboard panel compact
  confidence: medium
  bind_status: future
  bind_identifiers: [graph_panel]
""".strip()
    )

    monkeypatch.setattr(dashboard, "_PLANNER_HINT_SEEDS_PATH", seed_path)
    monkeypatch.setattr(dashboard, "_read_strategy_store_rows", lambda path=None: None)
    monkeypatch.setattr(dashboard, "_read_autopilot_journal_rows", lambda path=None: None)

    response = asyncio.run(dashboard.insight_graph(focus="dashboard-graph-panel", depth=1))
    data = json.loads(response.body)

    assert data["available"] is True
    assert data["source"]["graph_source"] == "planner_hint_seed"
    assert data["summary"]["strategy_rows"] == 0
    assert data["summary"]["planner_hint_rows"] == 1
    kinds = {node["kind"] for node in data["nodes"]}
    assert "planner_hint" in kinds


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


def test_dashboard_html_repaints_topology_after_region_lock_refresh() -> None:
    """Live inference/topology/CPU-lock panels should share one lock-cache frame."""
    from src.api.routes import dashboard

    html = dashboard._DASHBOARD_HTML
    assert "const overlayInflight = snapshotSeq != null" in html
    assert "updateTopologyInflight(overlayInflight, snapshotSeq);" in html
    assert "same lock cache" in html


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
    assert summary["latest_trial_id"] == 1
    assert summary["latest_promotion_trial_id"] == 1
    assert summary["trials_since_promotion"] == 0
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


def test_autopilot_progress_surfaces_eval_label_from_log_tail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "autopilot_state.json"
    log_dir = tmp_path / "logs"
    log_path = log_dir / "autopilot_restart_1.log"
    started_at = datetime.now(timezone.utc).timestamp() - 90
    state_path.write_text(json.dumps({
        "in_flight_trial": {
            "trial_id": 9,
            "started_at": started_at,
            "action": {"type": "deep_eval"},
        }
    }))
    log_dir.mkdir()
    log_path.write_text("\n".join([
        "2026-07-04 19:14:09 [autopilot] INFO: Trial 9: {\"type\": \"deep_eval\"}",
        "2026-07-04 21:43:41 [autopilot.eval] INFO: T3 progress: 40/160 (58% correct)",
    ]) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_STATE_PATH", state_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_LOG_DIR", log_dir)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", tmp_path / "missing.jsonl")

    response = asyncio.run(dashboard.autopilot_progress())
    payload = json.loads(response.body)

    assert payload["percent_source"] == "log_tail"
    assert payload["eval_label"] == "T3"
    assert payload["log_tail_progress"] == {"completed": 40, "total": 160}


def test_autopilot_progress_surfaces_baseline_promotion_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    journal_shard_path = tmp_path / "autopilot_journal_1.jsonl"
    started_at = datetime.now(timezone.utc).timestamp() - 90
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
            "timestamp": "2026-07-05T00:00:00+00:00",
            "action_type": "seed_batch",
        },
        {
            "trial_id": 1,
            "timestamp": "2026-07-05T00:01:00+00:00",
            "action_type": "seed_batch",
        },
        {
            "type": "baseline_promotion",
            "source_trial_id": 1,
            "tier": 1,
            "previous_quality": 1.0,
            "new_quality": 1.2,
            "timestamp": "2026-07-05T00:01:30+00:00",
            "reason": "kept",
        },
        {
            "trial_id": 2,
            "timestamp": "2026-07-05T00:02:00+00:00",
            "action_type": "seed_batch",
        },
    ]
    shard_rows = [
        {
            "trial_id": 4,
            "timestamp": "2026-07-05T00:03:00+00:00",
            "action_type": "seed_batch",
        },
    ]
    journal_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    journal_shard_path.write_text("\n".join(json.dumps(row) for row in shard_rows) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_STATE_PATH", state_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)

    response = asyncio.run(dashboard.autopilot_progress())
    payload = json.loads(response.body)

    assert payload["baseline_promotions"]["count"] == 1
    assert payload["baseline_promotions"]["latest_trial_id"] == 4
    assert payload["baseline_promotions"]["latest_promotion_trial_id"] == 1
    assert payload["baseline_promotions"]["trials_since_promotion"] == 3


def test_autopilot_progress_surfaces_outcome_kpis_and_current_code_health(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    started_at = datetime.now(timezone.utc).timestamp() - 90
    state_path.write_text(json.dumps({
        "in_flight_trial": {
            "trial_id": 4,
            "started_at": started_at,
            "action": {"type": "seed_batch"},
        }
    }))
    rows = [
        {
            "trial_id": 1,
            "timestamp": "2026-07-05T00:00:00+00:00",
            "action_type": "seed_batch",
            "keep_revert_decision": "keep",
        },
        {
            "trial_id": 2,
            "timestamp": "2026-07-05T00:01:00+00:00",
            "action_type": "seed_batch",
            "keep_revert_decision": "revert",
        },
        {
            "trial_id": 3,
            "timestamp": "2026-07-05T00:02:00+00:00",
            "action_type": "seed_batch",
            "keep_revert_decision": "excluded",
            "eval_details": {
                "learning_exclusion": {"by": "mad_noise"},
            },
        },
        {
            "type": "baseline_promotion",
            "source_trial_id": 1,
            "tier": 1,
            "previous_quality": 1.0,
            "new_quality": 1.2,
            "timestamp": "2026-07-05T00:02:30+00:00",
            "reason": "kept",
        },
        {
            "trial_id": 4,
            "timestamp": "2026-07-05T00:03:00+00:00",
            "action_type": "seed_batch",
        },
    ]
    journal_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_STATE_PATH", state_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)
    monkeypatch.setattr(
        dashboard,
        "_autopilot_current_code_health",
        lambda: {
            "ok": True,
            "status": "active",
            "code_stale": False,
            "require_current_code": True,
        },
    )

    response = asyncio.run(dashboard.autopilot_progress())
    payload = json.loads(response.body)

    assert payload["baseline_promotions"]["count"] == 1
    assert payload["outcome_kpis"]["keepable_rate"] == {
        "count": 1,
        "total": 3,
        "rate": 0.333,
    }
    assert payload["outcome_kpis"]["wasted_eval_rate"] == {
        "count": 1,
        "total": 3,
        "rate": 0.333,
    }
    assert payload["outcome_kpis"]["learning_excluded_rate"] == {
        "count": 1,
        "total": 3,
        "rate": 0.333,
    }
    assert payload["current_code_health"] == {
        "ok": True,
        "status": "active",
        "code_stale": False,
        "require_current_code": True,
    }


def test_autopilot_control_pause_and_resume_updates_state_latch(tmp_path: Path) -> None:
    state_path = tmp_path / "autopilot_state.json"
    audit_path = tmp_path / "autopilot_operator_control.jsonl"
    state_path.write_text(
        json.dumps(
            {
                "paused": False,
                "trial_counter": 12,
                "_dispatch_deficiency": "skip_action_loop",
                "consecutive_skip_actions": 3,
                "last_invalid_action": {"type": "skip"},
                "last_invalid_reason": "loop",
                "last_invalid_status": "rejected",
                "consecutive_meta_actions": 2,
                "_meta_halt_reason": "operator review",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paused = dashboard._apply_autopilot_control_action(
        action="pause",
        note="operator pause",
        state_path=state_path,
        audit_path=audit_path,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert paused["status"] == "ok"
    assert paused["paused_pre"] is False
    assert paused["paused"] is True
    assert state["paused"] is True
    assert state["pause_reason"] == "operator pause"

    resumed = dashboard._apply_autopilot_control_action(
        action="resume",
        note="operator resume",
        state_path=state_path,
        audit_path=audit_path,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert resumed["status"] == "ok"
    assert resumed["paused_pre"] is True
    assert resumed["paused"] is False
    assert state["paused"] is False
    assert "pause_reason" not in state
    assert "_dispatch_deficiency" not in state
    assert "_meta_halt_reason" not in state
    assert state["consecutive_skip_actions"] == 0
    assert state["last_invalid_action"] is None
    assert state["last_invalid_reason"] is None
    assert state["last_invalid_status"] is None
    assert state["consecutive_meta_actions"] == 0

    audit_rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["action"] for row in audit_rows] == ["pause", "resume"]
    assert [row["paused_post"] for row in audit_rows] == [True, False]


def test_autopilot_progress_leaves_outcome_kpis_unknown_without_source_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    started_at = datetime.now(timezone.utc).timestamp() - 45
    state_path.write_text(json.dumps({
        "in_flight_trial": {
            "trial_id": 8,
            "started_at": started_at,
            "action": {"type": "seed_batch"},
        }
    }))
    journal_path.write_text("\n".join(json.dumps({
        "trial_id": trial_id,
        "timestamp": f"2026-07-05T00:0{trial_id}:00+00:00",
        "action_type": "seed_batch",
    }) for trial_id in [1, 2, 3]) + "\n")
    monkeypatch.setattr(dashboard, "_AUTOPILOT_STATE_PATH", state_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)
    monkeypatch.setattr(dashboard, "_autopilot_current_code_health", lambda: None)

    response = asyncio.run(dashboard.autopilot_progress())
    payload = json.loads(response.body)

    assert payload["outcome_kpis"]["keepable_rate"] == {
        "count": 0,
        "total": 0,
        "rate": None,
    }
    assert payload["outcome_kpis"]["wasted_eval_rate"] == {
        "count": 0,
        "total": 0,
        "rate": None,
    }
    assert payload["outcome_kpis"]["learning_excluded_rate"] == {
        "count": 0,
        "total": 0,
        "rate": None,
    }
    assert payload["current_code_health"] is None


def test_autopilot_progress_prefers_active_autopilot_log_over_stale_restart_log(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "autopilot_state.json"
    log_dir = tmp_path / "logs"
    restart_log = log_dir / "autopilot_restart_1.log"
    active_log = log_dir / "autopilot.log"
    started_at = datetime.now(timezone.utc).timestamp() - 90
    state_path.write_text(json.dumps({
        "in_flight_trial": {
            "trial_id": 1156,
            "started_at": started_at,
            "action": {"type": "deep_eval", "tier": 3},
        }
    }))
    log_dir.mkdir()
    restart_log.write_text("\n".join([
        "2026-07-05 06:24:34 [autopilot.eval] INFO: T1 progress: 30/38 (67% correct)",
    ]) + "\n")
    active_log.write_text("\n".join([
        "2026-07-05 06:45:34 [autopilot] INFO: Trial 1156: {\"type\": \"deep_eval\", \"tier\": 3}",
        "2026-07-05 07:25:24 [autopilot.eval] INFO: T3 progress: 100/160 (57% correct)",
    ]) + "\n")
    now = datetime.now(timezone.utc).timestamp()
    os.utime(restart_log, (now - 100, now - 100))
    os.utime(active_log, (now, now))
    monkeypatch.setattr(dashboard, "_AUTOPILOT_STATE_PATH", state_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_LOG_DIR", log_dir)
    monkeypatch.setattr(dashboard, "AUTOPILOT_LOG", active_log)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", tmp_path / "missing.jsonl")

    response = asyncio.run(dashboard.autopilot_progress())
    payload = json.loads(response.body)

    assert payload["percent_source"] == "log_tail"
    assert payload["eval_label"] == "T3"
    assert payload["log_tail_progress"] == {"completed": 100, "total": 160}


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
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)

    response = asyncio.run(dashboard.gepa_status())
    payload = json.loads(response.body)

    # Superseded/corrupted trials are no longer dropped from the trajectory list —
    # they ride along tagged so a mid-trial kill stays visible. The supersession
    # fold must still be APPLIED, so trial 2 carries the corruption tag while the
    # clean trial 1 does not.
    by_trial = {t["trial_id"]: t for t in payload["recent_trials"]}
    assert set(by_trial) == {1, 2}
    assert by_trial[1]["bug_corrupted_by"] is None
    assert by_trial[2]["bug_corrupted_by"] == "resource_contention"


def test_gepa_status_recent_trials_carry_tier(tmp_path: Path, monkeypatch) -> None:
    """Trajectory rows must expose `tier` so the dashboard can label per-tier
    quality (a by-design-low T3 row must not read as a T1 regression)."""
    log_path = tmp_path / "autopilot.log"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    log_path.write_text("2026-07-04 00:00:00,000 GEPA: Trial active\n")
    rows = [
        {
            "trial_id": 30, "tier": 1, "timestamp": "2026-07-04T00:00:00+00:00",
            "species": "prompt_forge", "quality": 1.8, "speed": 30.0,
            "cost": 0.5, "reliability": 1.0, "pareto_status": "frontier",
        },
        {
            "trial_id": 31, "tier": 3, "timestamp": "2026-07-04T00:01:00+00:00",
            "species": "prompt_forge", "quality": 1.2, "speed": 45.0,
            "cost": 0.5, "reliability": 1.0, "pareto_status": "frontier",
        },
    ]
    journal_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(dashboard, "AUTOPILOT_LOG", log_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)

    response = asyncio.run(dashboard.gepa_status())
    payload = json.loads(response.body)

    tier_by_trial = {t["trial_id"]: t["tier"] for t in payload["recent_trials"]}
    assert tier_by_trial == {30: 1, 31: 3}


def test_gepa_status_recent_trials_carry_real_suite_metric(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Real-suite rows should be visible in the GEPA trajectory panel."""
    log_path = tmp_path / "autopilot.log"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    log_path.write_text("2026-07-05 00:00:00,000 Trial 40 complete\n")
    rows = [
        {
            "trial_id": 40,
            "tier": 3,
            "timestamp": "2026-07-05T00:00:00+00:00",
            "species": "structural_lab",
            "quality": 1.2,
            "speed": 45.0,
            "cost": 0.5,
            "reliability": 1.0,
            "pareto_status": "dominated",
            "eval_details": {
                "per_suite_quality": {"real_suite_v1": 1.5},
                "details": {"per_suite_counts": {"real_suite_v1": 4}},
                "question_results": [
                    {"suite": "real_suite_v1", "qid": "a", "correct": True},
                    {"suite": "real_suite_v1", "qid": "b", "correct": False},
                ],
            },
        }
    ]
    journal_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(dashboard, "AUTOPILOT_LOG", log_path)
    monkeypatch.setattr(dashboard, "_AUTOPILOT_JOURNAL_PATH", journal_path)

    response = asyncio.run(dashboard.gepa_status())
    payload = json.loads(response.body)

    metric = payload["recent_trials"][0]["real_suite_v1"]
    assert metric == {
        "suite": "real_suite_v1",
        "quality": 1.5,
        "count": 4,
        "correct": 1,
    }


def test_shape_pareto_entry_carries_real_suite_metric() -> None:
    shaped = dashboard._shape_pareto_entry(
        {
            "trial_id": 50,
            "objectives": [1.0, 20.0, -0.2, 1.0],
            "eval_details": {
                "details": {"per_suite_counts": {"real_suite_v1": 2}},
                "question_results": [
                    {"suite": "real_suite_v1", "correct": True},
                    {"suite": "real_suite_v1", "correct": True},
                ],
            },
        }
    )

    assert shaped["real_suite_v1"] == {
        "suite": "real_suite_v1",
        "quality": 3.0,
        "count": 2,
        "correct": 2,
    }


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


# ----- _poll_all_slots fan-out deadline -----


def test_poll_all_slots_deadline_bounds_hung_ports(monkeypatch) -> None:
    """A hung /slots endpoint must cost at most the fan-out deadline, not its
    own per-request budget times the port count — the snapshot serve path
    feeds the topology/region-locks/live-tap panels."""
    monkeypatch.setattr(
        dashboard, "_discover_llama_ports",
        lambda: {8070: "frontdoor", 8071: "hung_a", 8072: "hung_b"},
    )

    async def fake_poll(client, port):
        if port == 8070:
            return [{"id": 0, "is_processing": False}]
        await asyncio.sleep(30)
        return [{"id": 99}]

    monkeypatch.setattr(dashboard, "_poll_slot", fake_poll)
    monkeypatch.setattr(dashboard, "_SLOTS_FANOUT_DEADLINE_S", 0.3)

    started = time.time()
    slots_by_port, meta = asyncio.run(dashboard._poll_all_slots())
    elapsed = time.time() - started

    assert elapsed < 2.0
    assert slots_by_port[8070] == [{"id": 0, "is_processing": False}]
    assert slots_by_port[8071] == []
    assert slots_by_port[8072] == []
    assert meta["ports"] == 3
    assert meta["answered"] == 1
    assert meta["timed_out"] == 2
    assert meta["duration_s"] >= 0.3


def test_poll_all_slots_empty_ports_meta() -> None:
    from unittest.mock import patch

    with patch.object(dashboard, "_discover_llama_ports", lambda: {}):
        slots_by_port, meta = asyncio.run(dashboard._poll_all_slots())
    assert slots_by_port == {}
    assert meta == {"ports": 0, "answered": 0, "timed_out": 0, "duration_s": 0.0}


# ----- region-locks cache + tap-enrich fail-open (serve-path decoupling) -----


def test_region_locks_cached_ttl_and_fail_open(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_payload():
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RuntimeError("proc scan exploded")
        return {"entries": [{"role": "frontdoor"}], "by_role": {"frontdoor": {}}}

    monkeypatch.setattr(dashboard, "_region_locks_payload", fake_payload)
    monkeypatch.setitem(dashboard._REGION_LOCKS_CACHE, "ts", 0.0)
    monkeypatch.setitem(dashboard._REGION_LOCKS_CACHE, "payload", None)

    first = dashboard._region_locks_cached()
    second = dashboard._region_locks_cached()  # inside TTL → no rebuild
    assert calls["n"] == 1
    assert first is second

    # Expire the TTL twice: second rebuild succeeds, third raises → the cached
    # payload is served marked stale instead of raising into the serve path.
    dashboard._REGION_LOCKS_CACHE["ts"] = 0.0
    dashboard._region_locks_cached()
    assert calls["n"] == 2
    dashboard._REGION_LOCKS_CACHE["ts"] = 0.0
    failed = dashboard._region_locks_cached()
    assert calls["n"] == 3
    assert failed["stale_cache"] is True
    assert "proc scan exploded" in failed["error"]
    assert failed["entries"] == [{"role": "frontdoor"}]


def test_snapshot_uses_fresh_region_lock_scan(monkeypatch) -> None:
    fresh = {"generated_at": 123.0, "entries": [{"role": "fresh"}], "by_role": {"fresh": {}}}

    async def fake_poll_all_slots():
        return {}, {"ports": 0, "answered": 0, "timed_out": 0, "duration_s": 0.0}

    def fake_cached():
        raise AssertionError("snapshot must not use the cached region-lock payload")

    monkeypatch.setattr(dashboard, "_poll_all_slots", fake_poll_all_slots)
    monkeypatch.setattr(dashboard, "_todays_progress_log", lambda: Path("/does/not/exist"))
    monkeypatch.setattr(dashboard, "_scan_recent_decisions", lambda _path: ([], {}, {}))
    monkeypatch.setattr(dashboard, "_count_log_events", lambda *_a, **_k: {})
    monkeypatch.setattr(dashboard, "_discover_llama_ports", lambda: {})
    monkeypatch.setattr(dashboard, "_gate_inflight_by_live_slots", lambda in_flight, *_a, **_k: in_flight)
    monkeypatch.setattr(dashboard, "_region_locks_payload", lambda: fresh)
    monkeypatch.setattr(dashboard, "_region_locks_cached", fake_cached)
    monkeypatch.setattr(dashboard, "_structured_tap_requests_for_dashboard", lambda **_k: [])
    monkeypatch.setattr(dashboard, "_topology_nodes_cached", lambda: [])

    response = asyncio.run(dashboard._snapshot_impl())
    payload = json.loads(response.body)

    assert payload["region_locks"] == fresh


def test_enrich_structured_tap_requests_fails_open(monkeypatch) -> None:
    """Tap content must render even when the locks/topology domain raises."""
    def boom():
        raise RuntimeError("locks domain down")

    monkeypatch.setattr(dashboard, "_port_roles_cached", boom)
    monkeypatch.setattr(dashboard, "_region_locks_cached", boom)
    reqs = [{"request_id": "chat-1:aa", "role": "frontdoor", "port": 8070}]
    out = dashboard._enrich_structured_tap_requests(reqs)
    assert out == reqs


def test_read_tap_events_tail_stitches_rotation_window(tmp_path) -> None:
    """Right after a rotation the base is missing/tiny; the tail must include
    the rotated shard so parsers keep a full window (empty-panel bug)."""
    base = tmp_path / "inference_tap_events.jsonl"
    rotated = tmp_path / "inference_tap_events.jsonl.1"
    rotated.write_text('{"seq":1}\n{"seq":2}\n')

    # Base absent entirely (rotation happened, no append yet).
    out = dashboard_tap._read_tap_events_tail(base, max_bytes=1024)
    assert '{"seq":2}' in out

    # Base tiny: stitched = rotated tail + base, in order.
    base.write_text('{"seq":3}\n')
    out = dashboard_tap._read_tap_events_tail(base, max_bytes=1024)
    assert out.index('{"seq":2}') < out.index('{"seq":3}')

    # Base large enough (>= half budget): no stitching, base only.
    base.write_text('{"seq":4}\n' * 200)
    out = dashboard_tap._read_tap_events_tail(base, max_bytes=1024)
    assert '{"seq":2}' not in out


# ----- health serve-path coverage (hang/crash visibility) ---------------------


def _reset_snapshot_stats(monkeypatch, **overrides) -> None:
    fresh = {
        "last_attempt_ts": None, "last_success_ts": None, "last_duration_s": None,
        "last_error": None, "last_error_ts": None, "build_count": 0,
    }
    fresh.update(overrides)
    for k, v in fresh.items():
        monkeypatch.setitem(dashboard._SNAPSHOT_BUILD_STATS, k, v)


def test_serve_path_health_idle_worker_is_fresh(monkeypatch) -> None:
    now = 1_000_000.0
    _reset_snapshot_stats(monkeypatch)  # never built: idle, not degraded
    assert dashboard._serve_path_health(now)["staleness_class"] == "fresh"
    _reset_snapshot_stats(  # built long ago, no demand since: still fresh
        monkeypatch, last_attempt_ts=now - 5000, last_success_ts=now - 5000,
        last_duration_s=0.9, build_count=42,
    )
    assert dashboard._serve_path_health(now)["staleness_class"] == "fresh"


def test_serve_path_health_flags_hang_and_crash(monkeypatch) -> None:
    now = 1_000_000.0
    # Hang: attempt outstanding past the stall threshold, no success since.
    _reset_snapshot_stats(
        monkeypatch, last_attempt_ts=now - 45, last_success_ts=now - 300,
        build_count=10,
    )
    out = dashboard._serve_path_health(now)
    assert out["staleness_class"] == "stale"
    assert "stalled" in out["reason"]
    # In-flight but young: not yet a hang.
    _reset_snapshot_stats(
        monkeypatch, last_attempt_ts=now - 5, last_success_ts=now - 300,
        build_count=10,
    )
    assert dashboard._serve_path_health(now)["staleness_class"] == "fresh"
    # Crash loop: newest attempt errored.
    _reset_snapshot_stats(
        monkeypatch, last_attempt_ts=now - 1, last_success_ts=now - 60,
        last_error="boom", last_error_ts=now - 1, build_count=10,
    )
    out = dashboard._serve_path_health(now)
    assert out["staleness_class"] == "stale"
    assert "erroring" in out["reason"]
    assert "boom" in out["reason"]


def test_health_folds_serve_path_and_probe(monkeypatch) -> None:
    now_ref = time.time()
    # Healthy stats → health ok, serve_path present.
    _reset_snapshot_stats(
        monkeypatch, last_attempt_ts=now_ref, last_success_ts=now_ref,
        last_duration_s=0.5, build_count=3,
    )
    resp = asyncio.run(dashboard.dashboard_health())
    payload = json.loads(resp.body)
    assert payload["serve_path"]["staleness_class"] == "fresh"
    assert "probe" not in payload

    # Stalled stats must degrade the folded verdict.
    _reset_snapshot_stats(
        monkeypatch, last_attempt_ts=now_ref - 120, last_success_ts=now_ref - 600,
        build_count=3,
    )
    resp = asyncio.run(dashboard.dashboard_health())
    payload = json.loads(resp.body)
    assert payload["serve_path"]["staleness_class"] == "stale"
    assert payload["status"] == "degraded"


def test_health_probe_reports_timeout_and_error(monkeypatch) -> None:
    async def timing_out_snapshot():
        raise asyncio.TimeoutError()

    monkeypatch.setattr(dashboard, "snapshot", timing_out_snapshot)
    resp = asyncio.run(dashboard.dashboard_health(probe="snapshot"))
    payload = json.loads(resp.body)
    assert payload["probe"]["ok"] is False
    assert "timeout_s" in payload["probe"]
    assert payload["status"] == "degraded"

    async def crashing_snapshot():
        raise RuntimeError("serve path exploded")

    monkeypatch.setattr(dashboard, "snapshot", crashing_snapshot)
    resp = asyncio.run(dashboard.dashboard_health(probe="snapshot"))
    payload = json.loads(resp.body)
    assert payload["probe"]["ok"] is False
    assert "serve path exploded" in payload["probe"]["error"]
    assert payload["status"] == "degraded"
