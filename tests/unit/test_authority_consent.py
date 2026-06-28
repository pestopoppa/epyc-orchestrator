"""Unit tests for the fail-closed operator authority-consent gate."""

from __future__ import annotations

import json

from src.autopilot_core.authority_consent import authority_consent
from src.autopilot_core.baseline_ledger import baseline_ledger_authority_enabled


def test_consent_absent_is_denied(tmp_path):
    assert authority_consent("baseline_ledger", path=tmp_path / "nope.json") is False


def test_consent_allow_grants(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"baseline_ledger": "allow"}))
    assert authority_consent("baseline_ledger", path=p) is True


def test_consent_other_value_denied(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"baseline_ledger": "deny"}))
    assert authority_consent("baseline_ledger", path=p) is False
    p.write_text(json.dumps({"baseline_ledger": True}))  # not the literal "allow"
    assert authority_consent("baseline_ledger", path=p) is False


def test_consent_malformed_is_denied(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ not json")
    assert authority_consent("baseline_ledger", path=p) is False
    p.write_text(json.dumps(["not", "a", "dict"]))
    assert authority_consent("baseline_ledger", path=p) is False


def test_baseline_authority_requires_both_flag_and_consent(tmp_path, monkeypatch):
    grant = tmp_path / "c.json"
    grant.write_text(json.dumps({"baseline_ledger": "allow"}))
    monkeypatch.setenv("AUTOPILOT_AUTHORITY_CONSENT_PATH", str(grant))

    # flag off -> denied regardless of consent
    assert baseline_ledger_authority_enabled({}) is False
    # flag on + consent allow -> enabled
    assert baseline_ledger_authority_enabled(
        {"baseline_ledger_authority_enabled": True}
    ) is True

    # flag on but consent revoked -> fail-closed off (the operator's lever)
    monkeypatch.setenv("AUTOPILOT_AUTHORITY_CONSENT_PATH", str(tmp_path / "gone.json"))
    assert baseline_ledger_authority_enabled(
        {"baseline_ledger_authority_enabled": True}
    ) is False
