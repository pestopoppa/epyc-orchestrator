from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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

    command = launcher.build_command(2000)

    assert command == [
        "/venv/bin/python3",
        "scripts/autopilot/autopilot.py",
        "start",
        "--max-trials",
        "2000",
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
