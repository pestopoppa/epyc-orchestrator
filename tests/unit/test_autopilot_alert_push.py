"""autopilot_alert_push: degraded emits once, repeats dedupe, flag off / no target = no-op."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "autopilot"))
import alert_push  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    alert_push.reset()
    monkeypatch.setenv("ORCHESTRATOR_ALERTS_TARGET_SESSION", "ops-main")
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
    assert len(sent) == 1 and sent[0][0] == "ops-main" and "health-degraded" in sent[0][1]
    assert alert_push.alert_degraded("conn refused", 7, transport=tr, now=1000.0 + alert_push.DEDUPE_WINDOW_S + 1)
    assert len(sent) == 2


def test_rate_limit_across_kinds(monkeypatch):
    _on(monkeypatch)
    sent = []
    tr = lambda tgt, line: sent.append(line)  # noqa: E731
    assert alert_push.alert_degraded("x", transport=tr, now=0.0)
    assert not alert_push.alert_smoke_failed("s1", "boom", transport=tr, now=10.0)
    assert alert_push.alert_smoke_failed("s1", "boom", transport=tr, now=100.0)
    assert len(sent) == 2


def test_consecutive_failures_threshold(monkeypatch):
    _on(monkeypatch)
    sent = []
    tr = lambda tgt, line: sent.append(line)  # noqa: E731
    assert not alert_push.alert_consecutive_failures(1, transport=tr, now=0.0)
    assert alert_push.alert_consecutive_failures(3, transport=tr, now=0.0)
    assert len(sent) == 1


def test_flag_off_emits_nothing(monkeypatch):
    _on(monkeypatch, False)
    sent = []
    assert not alert_push.alert_degraded("x", transport=lambda t, l: sent.append(l), now=0.0)
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
