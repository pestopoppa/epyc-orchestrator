from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autopilot" / "start_fable_authority_daemon.py"
spec = importlib.util.spec_from_file_location("start_fable_authority_daemon", SCRIPT)
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_authority_env_forces_required_flags() -> None:
    env = launcher.authority_env(
        {
            "AUTOPILOT_TOOL_SENTINELS": "0",
            "AUTOPILOT_PLANNER_HINTS": "0",
            "KEEP": "value",
        }
    )

    assert env["KEEP"] == "value"
    assert env["AUTOPILOT_TOOL_SENTINELS"] == "1"
    assert env["AUTOPILOT_PLANNER_HINTS"] == "1"
    assert env["AUTOPILOT_SEQ_VERDICT"] == "1"
    assert env["AUTOPILOT_W6_AUDIT_BLOCK"] == "1"
    assert env["AUTOPILOT_PLANNER_TIMEOUT"] == "600"
    assert env["AUTOPILOT_PLANNER_SPEND_BREAKER"] == "1"


def test_authority_env_defaults_to_frontdoor_local_planner_without_overriding() -> None:
    env = launcher.authority_env(
        {
            "AUTOPILOT_PLANNER_PRIMARY": "claude",
            "AUTOPILOT_LOCAL_PLANNER_MAX_TOKENS": "4096",
        }
    )

    assert env["AUTOPILOT_PLANNER_PRIMARY"] == "claude"
    assert env["AUTOPILOT_PLANNER_CRITIC"] == "local_ingest"
    assert env["AUTOPILOT_PLANNER_CRITIC_FALLBACK"] == "claude"
    assert env["AUTOPILOT_LOCAL_PLANNER_ROLE"] == "frontdoor"
    assert env["AUTOPILOT_LOCAL_PLANNER_MODEL"] == "frontdoor"
    assert env["AUTOPILOT_LOCAL_PLANNER_TEMPERATURE"] == "0"
    assert env["AUTOPILOT_LOCAL_PLANNER_MAX_TOKENS"] == "4096"

    default_env = launcher.authority_env({})
    assert default_env["AUTOPILOT_PLANNER_PRIMARY"] == "local_frontdoor"
    assert default_env["AUTOPILOT_PLANNER_CRITIC"] == "local_ingest"


def test_authority_env_sets_latest_repo_readiness_pickup(
    monkeypatch,
    tmp_path,
) -> None:
    older = tmp_path / "repo_readiness_autopilot_pickup_2026-07-03.json"
    newer = tmp_path / "repo_readiness_autopilot_pickup_2026-07-05.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(launcher, "DEFAULT_REPO_READINESS_DIRS", (tmp_path,))

    env = launcher.authority_env({})

    assert env["AUTOPILOT_REPO_READINESS_PICKUP"] == str(newer)


def test_authority_env_preserves_explicit_repo_readiness_pickup(
    monkeypatch,
    tmp_path,
) -> None:
    generated = tmp_path / "repo_readiness_autopilot_pickup_2026-07-05.json"
    generated.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(launcher, "DEFAULT_REPO_READINESS_DIRS", (tmp_path,))

    env = launcher.authority_env({"AUTOPILOT_REPO_READINESS_PICKUP": "/custom/pickup.json"})

    assert env["AUTOPILOT_REPO_READINESS_PICKUP"] == "/custom/pickup.json"


def test_build_command_uses_autopilot_start_and_default_trials(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "python_executable", lambda: "/venv/bin/python3")

    command = launcher.build_command(3000)

    assert command == [
        "/venv/bin/python3",
        "scripts/autopilot/autopilot.py",
        "start",
        "--max-trials",
        "3000",
    ]


def test_dry_run_prints_authority_payload(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(launcher, "python_executable", lambda: "/venv/bin/python3")
    monkeypatch.setattr(launcher, "live_autopilot_processes", lambda: ["123 live"])

    rc = launcher.main(["--dry-run", "--log-dir", str(tmp_path), "--max-trials", "1234"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == [
        "/venv/bin/python3",
        "scripts/autopilot/autopilot.py",
        "start",
        "--max-trials",
        "1234",
    ]
    assert payload["env"]["AUTOPILOT_TOOL_SENTINELS"] == "1"
    assert payload["env"]["AUTOPILOT_SEQ_VERDICT"] == "1"
    assert payload["pid"] is None


def test_preflight_prints_restart_advice_without_starting(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        launcher,
        "build_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not build")),
    )

    class FakeAdvisor:
        @staticmethod
        def build_restart_advice(report, *, max_trials):
            return {
                "advisor_version": "autopilot_restart_advisor.v1",
                "ok": True,
                "status": "restart_recommended",
                "restart_needed": True,
                "safe_to_restart_now": True,
                "reason": "unit",
                "blockers": [],
                "phase": report["phase"],
                "max_trials": max_trials,
            }

    class FakePhase:
        @staticmethod
        def build_phase_health_report(**kwargs):
            return {"phase": "loop_start", "require_current_code": kwargs["require_current_code"]}

    monkeypatch.setitem(sys.modules, "autopilot_restart_advisor", FakeAdvisor)
    monkeypatch.setitem(sys.modules, "phase_status", FakePhase)

    rc = launcher.main(["--preflight", "--log-dir", str(tmp_path), "--max-trials", "1234"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "restart_recommended"
    assert payload["phase"] == "loop_start"
    assert payload["max_trials"] == 1234
