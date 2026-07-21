#!/usr/bin/env python3
"""Unit tests for scripts/autopilot/migration_probe.py (ROUTE-A3 J2/J3 probe).

Coverage is entirely INFERENCE-FREE and asserts concrete expected values on the
NON-inference logic:
  * oscillation parsing + role->model/quant resolution,
  * the WP-3/WP-4 expected-migration state machine over synthetic load profiles
    (exact forward/reverse/skip counts, event times, per-step primary location,
    cooldown + per-session-cap skip reasons),
  * plan construction (request fan-out, model/quant indexing, single-worker +
    oscillation validation warnings, placement-queue transport, never /chat),
  * migration-event analysis on synthetic OBSERVED outcomes (direction totals,
    thrash-skip totals, aborts, per-session cap violations, J2/J3 verdict),
  * the double gate (default dry-run; --execute still env-gated -> mocked bridge).
The execution bridge is never called with a real client / orchestrator here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

# ── load the runner module by path (robust; no scripts.* package needed) ──────
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "autopilot" / "migration_probe.py"
)
_SPEC = importlib.util.spec_from_file_location("migration_probe", _MODULE_PATH)
mp = importlib.util.module_from_spec(_SPEC)
sys.modules["migration_probe"] = mp  # register before exec (dataclasses)
_SPEC.loader.exec_module(mp)


# --------------------------------------------------------------------------- #
# Oscillation parsing
# --------------------------------------------------------------------------- #
def test_parse_oscillation_forms():
    assert mp.parse_oscillation("1,4,4,1") == [1, 4, 4, 1]
    assert mp.parse_oscillation(" 2 3 ") == [2, 3]          # whitespace-separated
    assert mp.parse_oscillation("1, ,4") == [1, 4]          # blank tokens dropped
    assert mp.parse_oscillation([0, 2, -5]) == [1, 2, 1]    # coerced to >= 1
    assert mp.parse_oscillation(None) == list(mp.DEFAULT_OSCILLATION)


# --------------------------------------------------------------------------- #
# role -> model/quant resolution (pure; operates on a loaded registry dict)
# --------------------------------------------------------------------------- #
def test_resolve_model_quant_for_role():
    reg = {"roles": {"frontdoor": {"model": {"name": "Qwen3.6-35B-A3B-MTP-Q8_0",
                                             "quant": "Q8_0"}}}}
    assert mp.resolve_model_quant_for_role(reg, "frontdoor") == (
        "Qwen3.6-35B-A3B-MTP-Q8_0", "Q8_0"
    )
    assert mp.resolve_model_quant_for_role(reg, "ghost") == (None, None)
    assert mp.resolve_model_quant_for_role({}, "frontdoor") == (None, None)


# --------------------------------------------------------------------------- #
# Expected-migration state machine (WP-3 forward / WP-4 reverse)
# --------------------------------------------------------------------------- #
def test_schedule_forward_then_reverse_with_cooldown_skip():
    # Canonical probe profile: warm -> burst(4>3) -> hold -> drop-to-1 x3.
    s = mp.expected_migration_schedule(
        [1, 4, 4, 1, 1, 1], safe_slots=3, cooldown_ms=2000, window_ms=30000,
        per_session_cap=5, dwell_ms=1500, session_id="P",
    )
    assert s["expected_forward"] == 1
    assert s["expected_reverse"] == 1
    assert s["expected_skips"] == 1
    assert s["session_migrations"] == 2

    # forward fires at the first burst step (step 1, t=1500); reverse after the
    # cooldown elapses (step 4, t=6000).
    assert [(e["step"], e["t_ms"]) for e in s["forward_events"]] == [(1, 1500)]
    assert [(e["step"], e["t_ms"]) for e in s["reverse_events"]] == [(4, 6000)]
    # the first drop-to-1 (step 3, t=4500) is skipped: full idle 1500ms < 2000ms.
    assert [(k["step"], k["reason"]) for k in s["reverse_skips"]] == [
        (3, mp.SKIP_COOLDOWN)
    ]
    # per-step primary residency: full, then quartered across the burst+cooldown,
    # then back on full once the reverse commits.
    assert [st["primary_location_after"] for st in s["steps"]] == [
        "full", "quarter", "quarter", "quarter", "full", "full"
    ]
    # forward event carries the reused session id + the from/to instances.
    fwd = s["forward_events"][0]
    assert fwd["session_id"] == "P"
    assert (fwd["from_instance"], fwd["to_instance"]) == ("full", "quarter")


def test_schedule_never_forward_when_load_below_safe_slots():
    s = mp.expected_migration_schedule([1, 2, 3, 1], safe_slots=3, dwell_ms=1500)
    assert s["expected_forward"] == 0
    assert s["expected_reverse"] == 0
    assert s["forward_events"] == []


def test_schedule_per_session_cap_blocks_reverse():
    # cap=1 lets exactly one migration (the forward) through; every later reverse
    # is skipped with a session_cap reason (after the per-step cooldown skip).
    s = mp.expected_migration_schedule(
        [1, 4, 4, 1, 1, 4, 4, 1, 1], safe_slots=3, cooldown_ms=2000,
        per_session_cap=1, dwell_ms=1500, session_id="P",
    )
    assert s["expected_forward"] == 1
    assert s["expected_reverse"] == 0
    assert s["session_migrations"] == 1
    reasons = [(k["step"], k["reason"]) for k in s["reverse_skips"]]
    assert reasons == [
        (3, mp.SKIP_COOLDOWN),
        (4, mp.SKIP_SESSION_CAP),
        (7, mp.SKIP_COOLDOWN),
        (8, mp.SKIP_SESSION_CAP),
    ]


def test_schedule_stale_window_skip():
    # A long dwell pushes the drop-to-1 step outside the recency window, so the
    # reverse is skipped as stale (not cooldown): dwell 40s > window 30s.
    s = mp.expected_migration_schedule(
        [1, 4, 1], safe_slots=3, cooldown_ms=2000, window_ms=30000,
        per_session_cap=5, dwell_ms=40000, session_id="P",
    )
    assert s["expected_forward"] == 1
    assert s["expected_reverse"] == 0
    assert [(k["step"], k["reason"]) for k in s["reverse_skips"]] == [
        (2, mp.SKIP_STALE_WINDOW)
    ]


# --------------------------------------------------------------------------- #
# Plan construction
# --------------------------------------------------------------------------- #
def test_plan_request_fanout_and_primary_count():
    p = mp.plan_migration_probe(
        role="frontdoor", model="Qwen3.6-35B-A3B-MTP-Q8_0", quant="Q8_0",
        oscillation=[1, 4, 4, 1, 1, 1], safe_slots=3,
    )
    # one request per unit of concurrency across all steps.
    assert len(p.requests) == sum([1, 4, 4, 1, 1, 1]) == 12
    # exactly one reused primary per step (6 steps).
    primaries = [r for r in p.requests if r.is_primary]
    assert len(primaries) == 6
    assert all(r.session_id == "migprobe-primary" for r in primaries)
    # interferers carry distinct, per-step session ids.
    interferers = [r for r in p.requests if not r.is_primary]
    assert len(interferers) == 6
    assert len({r.session_id for r in interferers}) == 6


def test_plan_is_model_quant_indexed_never_role_indexed():
    p = mp.plan_migration_probe(
        role="frontdoor", model="Qwen3.6-35B-A3B-MTP-Q8_0", quant="Q8_0",
        oscillation=[1, 4, 1],
    )
    idx = p.model_index()
    assert idx == {
        "model": "Qwen3.6-35B-A3B-MTP-Q8_0",
        "quant": "Q8_0",
        "model_quant_key": "Qwen3.6-35B-A3B-MTP-Q8_0::Q8_0",
    }
    # the result identity is keyed on model/quant, NOT on role.
    assert "role" not in idx
    d = p.to_dict()
    assert d["model_index"] == idx
    assert d["placement_role"] == "frontdoor"   # role kept only as placement detail
    # unknown model/quant degrade to "unknown", never to the role name.
    p2 = mp.plan_migration_probe(role="frontdoor", model=None, quant=None,
                                 oscillation=[1, 4, 1])
    assert p2.model_index()["model_quant_key"] == "unknown::unknown"


def test_plan_transport_is_placement_queue_never_chat():
    p = mp.plan_migration_probe(role="frontdoor", model="m", quant="q",
                                oscillation=[1, 4, 4, 1, 1, 1])
    d = p.to_dict()
    assert d["transport"] == {
        "transport": "placement_queue",
        "request_priority": "background",
        "workload_class": "eval_batch",
        "uses_chat_endpoint": False,
    }
    # transport-bearing surfaces (requests + transport) never name a /chat endpoint
    # (human-readable notes legitimately say "NEVER a foreground /chat request").
    transport_blob = json.dumps({"requests": d["requests"], "transport": d["transport"]})
    assert "/chat" not in transport_blob
    for r in p.requests:
        rd = r.to_dict()
        assert rd["transport"] == "placement_queue"
        assert rd["request_priority"] == "background"
        assert rd["workload_class"] == "eval_batch"
        assert rd["force_bindings"] == {"force_role": "frontdoor", "allow_delegation": False}


def test_plan_multiworker_and_no_burst_warnings():
    # multi-worker confound warning.
    p = mp.plan_migration_probe(role="frontdoor", model="m", quant="q",
                                oscillation=[1, 4, 4, 1, 1, 1], worker_count=6)
    assert any("multi-worker" in w and "workers=6" in w for w in p.validation_warnings)

    # oscillation that never bursts -> J2 warning; single worker -> no worker warn.
    p2 = mp.plan_migration_probe(role="frontdoor", model="m", quant="q",
                                 oscillation=[1, 2, 3, 1], safe_slots=3,
                                 worker_count=1)
    assert any("never triggers J2 forward migration" in w for w in p2.validation_warnings)
    assert not any("multi-worker" in w for w in p2.validation_warnings)

    # canonical single-worker profile -> no warnings at all.
    p3 = mp.plan_migration_probe(role="frontdoor", model="m", quant="q",
                                 oscillation=[1, 4, 4, 1, 1, 1], safe_slots=3,
                                 worker_count=1)
    assert p3.validation_warnings == []


def test_plan_matches_schedule_expectations():
    p = mp.plan_migration_probe(role="frontdoor", model="m", quant="q",
                                oscillation=[1, 4, 4, 1, 1, 1], safe_slots=3)
    d = p.to_dict()
    assert d["expected_forward"] == 1
    assert d["expected_reverse"] == 1
    assert d["expected_skips"] == 1
    assert d["requires_single_worker"] is True
    assert d["route"] == "ROUTE-A3-j2j3-single-worker"


def test_execution_orders_burst_handover_interferer_before_primary():
    p = mp.plan_migration_probe(
        role="frontdoor", model="m", quant="q",
        oscillation=[1, 4], safe_slots=3,
    )
    step_specs = [r for r in p.requests if r.step == 1]
    ordered, stagger = mp._execution_order_for_step(step_specs, safe_slots=3)
    assert stagger is True
    assert ordered[0].is_primary is False
    assert any(spec.is_primary for spec in ordered[1:])


def test_execute_probe_runs_same_step_requests_concurrently(tmp_path):
    plan = mp.plan_migration_probe(
        role="frontdoor", model="m", quant="q",
        oscillation=[1, 4], safe_slots=3, dwell_ms=1,
    )
    active = 0
    max_active = 0
    step1_started = 0
    lock = threading.Lock()
    release = threading.Event()

    def _request(spec):
        nonlocal active, max_active, step1_started
        if spec.step != 1:
            return {"ok": True}
        with lock:
            active += 1
            step1_started += 1
            max_active = max(max_active, active)
            if step1_started == 4:
                release.set()
        assert release.wait(2), "burst step was serialized; not all requests overlapped"
        time.sleep(0.01)
        with lock:
            active -= 1
        return {"ok": True}

    def _counter(_url):
        return {"forward": 0, "reverse": 0, "thrash": {}}

    mp.execute_migration_probe(
        plan,
        url="http://unused",
        request_fn=_request,
        counter_probe=_counter,
        sleep_fn=lambda _seconds: None,
        output_path=tmp_path / "probe.jsonl",
    )
    assert max_active >= 2
    assert step1_started == 4


# --------------------------------------------------------------------------- #
# Migration-event analysis on synthetic OBSERVED outcomes
# --------------------------------------------------------------------------- #
def test_analyze_pass_verdict_forward_and_reverse():
    observed = [
        {"direction": "forward", "session_id": "P", "committed": True},
        {"skipped": mp.SKIP_COOLDOWN, "session_id": "P"},
        {"direction": "reverse", "session_id": "P", "committed": True},
    ]
    a = mp.analyze_migration_events(observed, per_session_cap=5)
    assert a["direction_total"] == {"forward": 1, "reverse": 1}
    assert a["thrash_skipped_total"] == {mp.SKIP_COOLDOWN: 1}
    assert a["n_committed"] == 2
    assert a["n_aborted"] == 0
    assert a["j2_forward_observed"] is True
    assert a["j3_reverse_observed"] is True
    assert a["verdict"] == "PASS"
    assert a["verdict_reasons"] == []
    assert a["observation_only"] is True
    assert a["per_session"]["P"] == {"forward": 1, "reverse": 1, "total": 2, "aborted": 0}


def test_analyze_inconclusive_when_no_reverse():
    observed = [{"direction": "forward", "session_id": "P", "committed": True}]
    a = mp.analyze_migration_events(observed)
    assert a["j2_forward_observed"] is True
    assert a["j3_reverse_observed"] is False
    assert a["verdict"] == "INCONCLUSIVE"
    assert "no reverse (J3) migration observed" in a["verdict_reasons"]


def test_analyze_aborted_not_counted_as_committed():
    observed = [
        {"direction": "forward", "session_id": "P", "committed": True},
        {"direction": "reverse", "session_id": "P", "committed": False},  # aborted
    ]
    a = mp.analyze_migration_events(observed)
    assert a["direction_total"] == {"forward": 1, "reverse": 0}
    assert a["n_aborted"] == 1
    assert a["per_session"]["P"] == {"forward": 1, "reverse": 0, "total": 1, "aborted": 1}
    assert a["verdict"] == "INCONCLUSIVE"
    assert any("aborted" in r for r in a["verdict_reasons"])


def test_analyze_sessions_over_cap():
    observed = [
        {"direction": "forward", "session_id": "P", "committed": True},
        {"direction": "reverse", "session_id": "P", "committed": True},
        {"direction": "forward", "session_id": "P", "committed": True},
    ]
    a = mp.analyze_migration_events(observed, per_session_cap=2)
    assert a["per_session"]["P"]["total"] == 3
    assert a["sessions_over_cap"] == ["P"]
    assert any("exceeded per-session cap" in r for r in a["verdict_reasons"])


def test_analyze_expected_match():
    expected = [
        {"direction": "forward"}, {"direction": "reverse"},
    ]
    observed = [
        {"direction": "forward", "session_id": "P", "committed": True},
        {"direction": "reverse", "session_id": "P", "committed": True},
    ]
    a = mp.analyze_migration_events(observed, expected=expected)
    assert a["expected_match"] == {
        "expected_forward": 1, "expected_reverse": 1,
        "forward_matches": True, "reverse_matches": True,
    }


def test_default_counter_probe_reads_contention_migration_counters(monkeypatch):
    class Resp:
        def json(self):
            return {
                "migration_counters": {
                    "kv_migration_direction_total": {"forward": 2, "reverse": 1},
                    "kv_migration_thrash_skipped_total": {"cooldown": 3},
                }
            }

    class FakeHttpx:
        @staticmethod
        def get(url, timeout):
            assert url == "http://api/dashboard/api/contention"
            assert timeout == 10
            return Resp()

    monkeypatch.setitem(sys.modules, "httpx", FakeHttpx)
    assert mp._default_counter_probe("http://api", role="frontdoor") == {
        "forward": 2,
        "reverse": 1,
        "thrash": {"cooldown": 3},
    }


def test_default_counter_probe_falls_back_to_per_role_scheduling(monkeypatch):
    class Resp:
        def json(self):
            return {
                "per_role_scheduling": {
                    "frontdoor": {
                        "migrations_started": 4,
                        "reverse_migrations": 2,
                    },
                    "worker_general": {
                        "migrations_started": 99,
                        "reverse_migrations": 99,
                    },
                }
            }

    class FakeHttpx:
        @staticmethod
        def get(_url, timeout):
            return Resp()

    monkeypatch.setitem(sys.modules, "httpx", FakeHttpx)
    assert mp._default_counter_probe("http://api", role="frontdoor") == {
        "forward": 4,
        "reverse": 2,
        "thrash": {},
    }


# --------------------------------------------------------------------------- #
# Double gate: default dry-run; --execute still env-gated -> mocked bridge
# --------------------------------------------------------------------------- #
def test_run_probe_env_flag_off_returns_dry_run(monkeypatch):
    monkeypatch.delenv(mp.MIGRATION_PROBE_INFERENCE_ENV, raising=False)

    def _boom(*a, **k):  # execution bridge must NOT run with the flag off
        raise AssertionError("execute_migration_probe called with inference flag OFF")

    monkeypatch.setattr(mp, "execute_migration_probe", _boom)

    plan = mp.plan_migration_probe(role="frontdoor", model="m", quant="q",
                                   oscillation=[1, 4, 4, 1, 1, 1])
    out = mp.run_migration_probe(plan)
    assert out["mode"] == "dry_run"
    assert out["inference_ran"] is False
    assert out["n_requests"] == 12
    assert out["plan"]["kind"] == "migration_probe_plan"


def test_run_probe_env_flag_on_routes_to_execute_bridge(monkeypatch):
    monkeypatch.setenv(mp.MIGRATION_PROBE_INFERENCE_ENV, "1")

    captured = {}

    def _fake_execute(plan, **kwargs):
        captured["plan"] = plan
        captured["kwargs"] = kwargs
        return {"model": "m", "quant": "q", "analysis": {"verdict": "PASS"}}

    monkeypatch.setattr(mp, "execute_migration_probe", _fake_execute)

    plan = mp.plan_migration_probe(role="frontdoor", model="m", quant="q",
                                   oscillation=[1, 4, 1])
    out = mp.run_migration_probe(plan, output_path=Path("/tmp/does-not-matter.jsonl"))
    assert out["mode"] == "execute"
    assert out["inference_ran"] is True
    assert out["result"]["analysis"]["verdict"] == "PASS"
    assert isinstance(captured["plan"], mp.MigrationProbePlan)


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("", False), ("no", False),
])
def test_env_flag_semantics(monkeypatch, val, expected):
    monkeypatch.setenv(mp.MIGRATION_PROBE_INFERENCE_ENV, val)
    assert mp._env_flag_enabled(mp.MIGRATION_PROBE_INFERENCE_ENV) is expected


# --------------------------------------------------------------------------- #
# CLI __main__: default dry-run is pure; --execute without env is still dry-run
# --------------------------------------------------------------------------- #
def test_main_default_is_pure_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(mp.MIGRATION_PROBE_INFERENCE_ENV, raising=False)
    out_path = tmp_path / "results.jsonl"
    code = mp.main([
        "--role", "frontdoor",
        "--model", "Qwen3.6-35B-A3B-MTP-Q8_0", "--quant", "Q8_0",
        "--oscillation", "1,4,4,1,1,1",
        "--safe-slots", "3",
        "--output", str(out_path),
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "migration_probe_plan"
    assert payload["transport"]["uses_chat_endpoint"] is False
    assert payload["model_index"]["model_quant_key"] == "Qwen3.6-35B-A3B-MTP-Q8_0::Q8_0"
    assert payload["n_requests"] == 12
    assert payload["expected_forward"] == 1
    assert payload["expected_reverse"] == 1
    # default dry-run writes NO results file.
    assert not out_path.exists()


def test_main_execute_without_env_flag_falls_back_to_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(mp.MIGRATION_PROBE_INFERENCE_ENV, raising=False)

    def _boom(*a, **k):  # double gate: --execute alone must not run inference
        raise AssertionError("execute bridge ran without the env flag")

    monkeypatch.setattr(mp, "execute_migration_probe", _boom)

    code = mp.main([
        "--role", "frontdoor", "--model", "m", "--quant", "q",
        "--oscillation", "1,4,1", "--execute",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry_run"
    assert payload["inference_ran"] is False
