from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import autopilot  # type: ignore[import-not-found]  # noqa: E402


def test_startup_attestation_reports_gate_env_and_config_hash(monkeypatch) -> None:
    for key, value in autopilot.AUTOPILOT_REQUIRED_GATE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("AUTOPILOT_PLANNER_PRIMARY", "local_ingest")
    monkeypatch.setenv("AUTOPILOT_PLANNER_CRITIC", "local_frontdoor")
    monkeypatch.setenv("AUTOPILOT_PLANNER_SPEND_BREAKER", "0")

    payload = autopilot._startup_attestation_payload()

    assert payload["missing_or_mismatch"] == {}
    assert len(payload["config_hash"]) == 64
    assert payload["gate_env"]["AUTOPILOT_TOOL_SENTINELS"] == "1"
    assert payload["gate_env"]["AUTOPILOT_PLANNER_PRIMARY"] == "local_ingest"
    assert payload["gate_env"]["AUTOPILOT_PLANNER_SPEND_BREAKER"] == "0"


def test_startup_attestation_marks_bare_start_gate_gaps(monkeypatch) -> None:
    for key in autopilot.AUTOPILOT_REQUIRED_GATE_ENV:
        monkeypatch.delenv(key, raising=False)

    payload = autopilot._startup_attestation_payload()

    assert set(payload["missing_or_mismatch"]) == set(autopilot.AUTOPILOT_REQUIRED_GATE_ENV)
