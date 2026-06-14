"""Tests for AutoPilot preflight diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.autopilot_core.journal_reconstruction import reconstruct_archive_from_journal_rows
from scripts.autopilot import preflight_audit as _MOD


def _journal_row(
    trial_id: int,
    *,
    quality: float = 1.0,
    speed: float = 40.0,
) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "timestamp": f"2026-06-14T00:00:0{trial_id}Z",
        "species": "unit",
        "action_type": "seed_batch",
        "tier": 1,
        "quality": quality,
        "speed": speed,
        "cost": 0.2,
        "reliability": 0.9,
        "pareto_status": "frontier",
    }


def _archive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    archive = reconstruct_archive_from_journal_rows(rows, None, current_run_only=False)
    assert archive is not None
    return archive


def test_model_server_targets_derive_from_stack_priors(tmp_path: Path) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
  coder_escalation:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
  worker_vision:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8086
  reap_25b_frontdoor:
    deployment_status: benchmark_or_candidate
    serving:
      endpoint: http://localhost:8196
""",
        encoding="utf-8",
    )

    targets = set(_MOD._model_server_targets(priors, "http://orchestrator.local:8001"))

    assert ("API", "http://orchestrator.local:8001/health") in targets
    assert ("coder_escalation/frontdoor", "http://localhost:8070/health") in targets
    assert ("worker_vision", "http://localhost:8086/health") in targets
    assert all("8196" not in health_url for _, health_url in targets)


def test_model_server_targets_fallback_is_current(tmp_path: Path) -> None:
    targets = _MOD._model_server_targets(tmp_path / "missing.yaml", "http://localhost:8002")
    health_urls = {health_url for _, health_url in targets}

    assert ("API", "http://localhost:8002/health") in targets
    assert "http://localhost:8071/health" not in health_urls
    assert "http://localhost:8086/health" in health_urls
    assert "http://localhost:8087/health" in health_urls
    assert "http://localhost:8090/health" not in health_urls


def test_model_server_targets_fallback_follows_manifest_without_literal_port_list(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        _MOD,
        "PORT_MAP",
        {
            "frontdoor": 9001,
            "coder_escalation": 9001,
            "worker_general": 9002,
            "embedder": 9090,
        },
    )
    monkeypatch.setattr(
        _MOD,
        "HOT_ROLES",
        {"frontdoor", "coder_escalation", "worker_general", "embedder"},
    )

    targets = _MOD._model_server_targets(tmp_path / "missing.yaml", "http://localhost:8002")

    assert targets == [
        ("API", "http://localhost:8002/health"),
        ("coder_escalation/frontdoor", "http://localhost:9001/health"),
        ("worker_general", "http://localhost:9002/health"),
    ]


def test_audit_stack_change_gate_runs_canonical_command(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="summary: ok\n", stderr="")

    monkeypatch.setattr(_MOD.subprocess, "run", fake_run)

    assert _MOD.audit_stack_change_gate() is True
    assert calls == [
        (
            _MOD.STACK_CHANGE_GATE_COMMAND,
            {
                "cwd": _MOD.REPO_ROOT,
                "capture_output": True,
                "text": True,
                "timeout": _MOD.STACK_CHANGE_GATE_TIMEOUT_S,
            },
        )
    ]


def test_audit_stack_change_gate_blocks_on_failure(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="summary: failed\n",
            stderr="promotion gate rejected stack change\n",
        )

    monkeypatch.setattr(_MOD.subprocess, "run", fake_run)

    assert _MOD.audit_stack_change_gate() is False


def test_audit_stack_change_gate_fails_closed_on_timeout(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise _MOD.subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(_MOD.subprocess, "run", fake_run)

    assert _MOD.audit_stack_change_gate() is False


def test_audit_blacklist_detects_superseded_corrupted_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script_dir = tmp_path / "scripts" / "autopilot"
    orchestration_dir = tmp_path / "orchestration"
    script_dir.mkdir(parents=True)
    orchestration_dir.mkdir()
    (script_dir / "failure_blacklist.yaml").write_text(
        """
blacklist:
  - reason: manual
    source_trial: -1
  - reason: contaminated
    source_trial: 2
""",
        encoding="utf-8",
    )
    rows = [
        {"trial_id": 2, "bug_corrupted_by": ""},
        {
            "type": "supersession",
            "target_trial_ids": [2],
            "fields": {"bug_corrupted_by": "resource_contention"},
        },
    ]
    (orchestration_dir / "autopilot_journal.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_MOD, "SCRIPT_DIR", script_dir)

    assert _MOD.audit_blacklist() is False


def test_archive_authority_diagnostic_matches_reconstructed_state() -> None:
    rows = [_journal_row(1, quality=1.2)]
    archive = _archive(rows)
    state = {"trial_counter": 2, "pareto_archive": archive}

    diagnostic = _MOD.archive_authority_diagnostic(state, rows)

    assert diagnostic["status"] == "match"
    assert diagnostic["state_entry_count"] == 1
    assert diagnostic["journal_entry_count"] == 1
    assert diagnostic["state_frontier_count"] == 1
    assert diagnostic["journal_frontier_count"] == 1


def test_archive_authority_diagnostic_ignores_nonsemantic_entry_metadata() -> None:
    rows = [_journal_row(1, quality=1.2)]
    archive = json.loads(json.dumps(_archive(rows)))
    for entries in (
        archive["all_entries"],
        archive["frontier"],
        archive["frontiers_by_tier"]["1"],
    ):
        entries[0].update(
            {
                "config_fingerprint": "",
                "git_tag": "",
                "n_reproductions": 1,
                "species": "runtime-only-label",
                "timestamp": "2026-06-14T00:00:01.001Z",
            }
        )
    state = {"trial_counter": 2, "pareto_archive": archive}

    diagnostic = _MOD.archive_authority_diagnostic(state, rows)

    assert diagnostic["status"] == "match"


def test_archive_authority_diagnostic_detects_archive_drift() -> None:
    rows = [_journal_row(1, quality=1.2)]
    archive = json.loads(json.dumps(_archive(rows)))
    archive["frontier"][0]["objectives"][0] = 0.5
    state = {"trial_counter": 2, "pareto_archive": archive}

    diagnostic = _MOD.archive_authority_diagnostic(state, rows)

    assert diagnostic["status"] == "drift"
    assert diagnostic["state_entry_count"] == 1
    assert diagnostic["journal_entry_count"] == 1


def test_archive_authority_diagnostic_flags_journal_ahead_of_state() -> None:
    rows = [_journal_row(1)]
    state = {"trial_counter": 1, "pareto_archive": _archive(rows)}

    diagnostic = _MOD.archive_authority_diagnostic(state, rows)

    assert diagnostic["status"] == "drift"
    assert diagnostic["warnings"] == [
        "journal max trial 1 is not below state trial_counter 1"
    ]


def test_audit_archive_authority_uses_state_and_journal_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = [_journal_row(1)]
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    state_path.write_text(
        json.dumps({"trial_counter": 2, "pareto_archive": _archive(rows)}),
        encoding="utf-8",
    )
    journal_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_MOD, "STATE_PATH", state_path)
    monkeypatch.setattr(_MOD, "JOURNAL_PATH", journal_path)

    assert _MOD.audit_archive_authority() is True

    state_path.write_text(
        json.dumps(
            {
                "trial_counter": 2,
                "pareto_archive": {"frontier": [], "all_entries": []},
            }
        ),
        encoding="utf-8",
    )

    assert _MOD.audit_archive_authority() is False
