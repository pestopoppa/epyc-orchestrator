"""Focused regressions for independently repairable E5 defects."""

import os
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import autopilot_tui  # noqa: E402
import eval_tower  # noqa: E402


def test_suite_health_skips_file_removed_during_stat(monkeypatch, tmp_path):
    older = tmp_path / "item_analytics_old.json"
    newer = tmp_path / "item_analytics_new.json"
    vanished = tmp_path / "item_analytics_gone.json"
    older.touch()
    newer.touch()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    monkeypatch.delenv("AUTOPILOT_SEQ_PROMOTION_SUITE_HEALTH_PATH", raising=False)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter([older, vanished, newer]))
    assert eval_tower._latest_promotion_suite_health_path() == newer



def test_suite_health_all_candidates_disappear(monkeypatch, tmp_path):
    missing = tmp_path / "item_analytics_gone.json"
    monkeypatch.delenv("AUTOPILOT_SEQ_PROMOTION_SUITE_HEALTH_PATH", raising=False)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter([missing]))
    assert eval_tower._latest_promotion_suite_health_path() is None


def test_suite_health_override_remains_authoritative(monkeypatch, tmp_path):
    override = tmp_path / "health.json"
    override.touch()
    monkeypatch.setenv("AUTOPILOT_SEQ_PROMOTION_SUITE_HEALTH_PATH", str(override))
    monkeypatch.setattr(Path, "glob", lambda self, pattern: (_ for _ in ()).throw(AssertionError()))
    assert eval_tower._latest_promotion_suite_health_path() == override
    override.unlink()
    assert eval_tower._latest_promotion_suite_health_path() is None


def test_prompt_tap_writes_are_serialized(monkeypatch):
    entered_write = threading.Event()
    release_write = threading.Event()

    def write_text(_path, _prompt):
        entered_write.set()
        assert release_write.wait(2)

    monkeypatch.setattr(Path, "write_text", write_text)
    tui = autopilot_tui.AutoPilotTUI()
    worker = threading.Thread(target=tui.set_prompt, args=("first",))
    worker.start()
    acquired = False
    try:
        assert entered_write.wait(2)
        acquired = tui._lock.acquire(blocking=False)
        assert not acquired, "tap write must hold the prompt lock"
    finally:
        if acquired:
            tui._lock.release()
        release_write.set()
        worker.join(2)
    assert not worker.is_alive()
    assert tui._current_prompt == "first"
