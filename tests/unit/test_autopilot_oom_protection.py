"""cmd_start sets oom_score_adj=-1000 on the autopilot process itself.

Durable earlyoom control-plane protection (mirrors orchestrator_stack.start_orchestrator):
the autopilot is comm=python and cannot be earlyoom --ignore'd by name, so it self-protects
via oom_score_adj=-1000 once it holds the singleton lock. Only its own pid is protected —
transient GEPA/planner subprocesses must stay killable. See epyc-root
handoffs/active/earlyoom-oom-protection.md.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

autopilot = importlib.import_module("autopilot")


def test_cmd_start_self_protects_via_oom_score_adj(monkeypatch, tmp_path) -> None:
    calls: list[list[int]] = []

    def _spy(pids, **_kwargs):
        captured = [int(p) for p in pids]
        calls.append(captured)
        return len(captured)

    # Don't touch the real lock or run the loop; just exercise cmd_start's wiring.
    monkeypatch.setattr(autopilot, "LOCK_PATH", tmp_path / ".autopilot.lock")
    monkeypatch.setattr(autopilot.fcntl, "flock", lambda *_a, **_k: None)
    monkeypatch.setattr(autopilot, "run_loop", lambda **_k: None)
    monkeypatch.setattr(autopilot, "_set_oom_score_adj", _spy)

    args = argparse.Namespace(max_trials=1, dry_run=True, no_controller=True, tui=False)
    autopilot.cmd_start(args)

    # Exactly this process protected — and only this pid (no transient children).
    assert calls == [[os.getpid()]]
