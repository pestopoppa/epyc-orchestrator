from __future__ import annotations

from pathlib import Path

import vidya_planner_bridge as bridge


def test_empty_resolution_ledger_skips_subprocess(tmp_path, monkeypatch):
    ledger = tmp_path / "resolutions.jsonl"
    ledger.write_text("")

    def forbidden(*args, **kwargs):  # pragma: no cover - assertion explains the contract
        raise AssertionError("subprocess must not run for an empty ledger")

    monkeypatch.setattr(bridge.subprocess, "run", forbidden)
    assert "none" in bridge.build_settled_ground_block(resolutions_path=ledger)


def test_lookup_output_is_forwarded(tmp_path, monkeypatch):
    ledger = tmp_path / "resolutions.jsonl"
    ledger.write_text('{"hypothesis_id":"h1"}\n')
    lookup = tmp_path / "lookup.py"
    lookup.write_text("# fixture")

    class Result:
        returncode = 0
        stdout = "  Read-only Vidya check\n  - [h1] standing=sealed\n"
        stderr = ""

    monkeypatch.setattr(bridge.subprocess, "run", lambda *args, **kwargs: Result())
    text = bridge.build_settled_ground_block(
        resolutions_path=ledger, lookup_path=lookup
    )
    assert "standing=sealed" in text


def test_lookup_failure_is_not_misreported_as_empty(tmp_path, monkeypatch):
    ledger = tmp_path / "resolutions.jsonl"
    ledger.write_text('{"hypothesis_id":"h1"}\n')
    lookup = tmp_path / "lookup.py"
    lookup.write_text("# fixture")

    class Result:
        returncode = 2
        stdout = ""
        stderr = "broken fold"

    monkeypatch.setattr(bridge.subprocess, "run", lambda *args, **kwargs: Result())
    text = bridge.build_settled_ground_block(
        resolutions_path=ledger, lookup_path=lookup
    )
    assert "UNAVAILABLE" in text
    assert "broken fold" in text
