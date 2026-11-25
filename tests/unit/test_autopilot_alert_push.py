"""autopilot_alert_push: dedupe, rate limit, escalation, auto-ack, flag-off no-op, real bus validation."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "autopilot"))
import alert_push  # noqa: E402

LIVE_BUS = Path("/workspace/coordination/session-bus")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    alert_push.reset()
    monkeypatch.setenv("ORCHESTRATOR_ALERTS_TARGET_SESSION", "pipeline-mgmt")
    yield
    alert_push.reset()


def _on(monkeypatch, on=True):
    monkeypatch.setattr(alert_push, "_enabled", lambda: on)


def test_degraded_emits_once_and_repeat_deduped(monkeypatch):
    _on(monkeypatch)
    sent = []
    tr = lambda tgt, line: sent.append((tgt, line))  # noqa: E731
    assert alert_push.alert_degraded("conn refused", 5, transport=tr, now=1000.0)
    assert not alert_push.alert_degraded("conn refused", 6, transport=tr, now=1100.0)
    assert not alert_push.alert_degraded("conn refused", 7, transport=tr, now=5000.0)  # still active: no repeat
    assert len(sent) == 1 and sent[0][0] == "pipeline-mgmt" and "health-degraded" in sent[0][1]


def test_rate_limit_across_kinds(monkeypatch):
    _on(monkeypatch)
    sent = []
    tr = lambda tgt, line: sent.append(line)  # noqa: E731
    assert alert_push.alert_degraded("x", transport=tr, now=0.0)
    assert not alert_push.alert_smoke_failed("s1", "boom", transport=tr, now=10.0)
    assert alert_push.alert_smoke_failed("s1", "boom", transport=tr, now=100.0)
    assert len(sent) == 2


def test_escalation_after_900s_once(monkeypatch):
    _on(monkeypatch)
    sent = []
    tr = lambda tgt, line: sent.append(line)  # noqa: E731
    alert_push.alert_degraded("x", transport=tr, now=0.0)
    assert alert_push.check_escalations(transport=tr, now=899.0) == 0
    assert alert_push.check_escalations(transport=tr, now=901.0) == 1
    assert "escalated" in sent[-1] and len(sent) == 2
    assert alert_push.check_escalations(transport=tr, now=5000.0) == 0  # once only
    assert len(sent) == 2


def test_auto_ack_clears_state_and_blocks_escalation(monkeypatch):
    _on(monkeypatch)
    sent = []
    tr = lambda tgt, line: sent.append(line)  # noqa: E731
    alert_push.alert_degraded("x", transport=tr, now=0.0)
    assert alert_push.clear_degraded()
    assert alert_push.check_escalations(transport=tr, now=2000.0) == 0
    assert alert_push.alert_degraded("x", transport=tr, now=2100.0)  # recurrence alerts afresh
    assert len(sent) == 2


def test_consecutive_failures_threshold_ack_and_escalation(monkeypatch):
    _on(monkeypatch)
    sent = []
    tr = lambda tgt, line: sent.append(line)  # noqa: E731
    assert not alert_push.alert_consecutive_failures(1, transport=tr, now=0.0)
    assert alert_push.alert_consecutive_failures(3, transport=tr, now=0.0)
    assert not alert_push.alert_consecutive_failures(4, transport=tr, now=1000.0)  # escalation path
    assert len(sent) == 2 and "escalated" in sent[1]
    assert not alert_push.alert_consecutive_failures(0, transport=tr, now=1100.0)  # reset = ack
    assert alert_push.alert_consecutive_failures(3, transport=tr, now=1200.0)
    assert len(sent) == 3


def test_flag_off_emits_nothing(monkeypatch):
    _on(monkeypatch, False)
    sent = []
    tr = lambda t, l: sent.append(l)  # noqa: E731
    assert not alert_push.alert_degraded("x", transport=tr, now=0.0)
    assert alert_push.check_escalations(transport=tr, now=10_000.0) == 0
    assert sent == []


def test_empty_target_is_noop(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("ORCHESTRATOR_ALERTS_TARGET_SESSION", "")
    sent = []
    assert not alert_push.alert_degraded("x", transport=lambda t, l: sent.append(l), now=0.0)
    assert sent == []


def test_flag_default_off_real_features(monkeypatch):
    from src.features import features
    monkeypatch.delenv("ORCHESTRATOR_AUTOPILOT_ALERT_PUSH", raising=False)
    assert features().autopilot_alert_push is False
    sent = []
    assert not alert_push.alert_degraded("x", transport=lambda t, l: sent.append(l), now=0.0)
    assert sent == []


def test_row_accepted_by_real_session_bus_validator(tmp_path):
    """Run the REAL session_bus.py append against a temp bus root (copy of live config+schema); live bus untouched."""
    if not (LIVE_BUS / "config.yaml").exists() or not Path(alert_push.SESSION_BUS).exists():
        pytest.skip("live bus definition not available")
    bus = tmp_path / "bus"
    (bus / "outbox").mkdir(parents=True)
    for name in ("config.yaml", "session_bus.schema.json"):
        shutil.copy(LIVE_BUS / name, bus / name)
    row = alert_push.build_row("pipeline-mgmt", "autopilot health-degraded key=health cause=x consecutive=5")
    r = subprocess.run(alert_push.bus_append_cmd(row, str(bus)), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr + r.stdout
    rows = [json.loads(x) for x in (bus / "outbox" / f"{alert_push.SENDER}.jsonl").read_text().splitlines()]
    assert rows[0]["kind"] == "nudge" and rows[0]["to"] == "pipeline-mgmt" and rows[0]["from"] == alert_push.SENDER
