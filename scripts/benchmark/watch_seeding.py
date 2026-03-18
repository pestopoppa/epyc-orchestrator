#!/usr/bin/env python3
"""Standalone TUI observer for an actively running seeding session.

Tails the inference tap and REPL tap files written by another process
(e.g. seed_specialist_routing.py --tui) and renders a live Rich display.

Unlike seeding_tui.py (which is in-process), this is an external observer:
- Writes the sentinel file to activate tap output in the running orchestrator
- Cleans up sentinel on exit (tap output stops, tap file preserved)
- Does NOT capture log handlers or interfere with the seeding loop
- Optionally tails the checkpoint JSONL for live pass/fail stats

Usage:
    python scripts/benchmark/watch_seeding.py
    python scripts/benchmark/watch_seeding.py --checkpoint 3way_20260304_1400
    python scripts/benchmark/watch_seeding.py --tap /path/to/tap.log

Press Ctrl-C to exit (no cleanup side effects).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Import reusable components from the in-process TUI
sys.path.insert(0, str(Path(__file__).parent))
from seeding_tui import (
    TapTailer,
    _sanitize_display,
    _style_repl_lines,
    _style_stream_lines,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_TAP = "/mnt/raid0/llm/tmp/inference_tap.log"
_DEFAULT_REPL_TAP = "/mnt/raid0/llm/tmp/repl_tap.log"
_SENTINEL = "/mnt/raid0/llm/tmp/.inference_tap_active"
_EVAL_DIR = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval")


# ---------------------------------------------------------------------------
# Checkpoint tailer — reads JSONL for live stats
# ---------------------------------------------------------------------------


class CheckpointTailer:
    """Daemon thread that tails a checkpoint JSONL for pass/fail stats."""

    def __init__(self, path: Path, poll: float = 2.0) -> None:
        self._path = path
        self._poll = poll
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._total = 0
        self._passed = 0
        self._by_suite: dict[str, tuple[int, int]] = {}  # suite -> (passed, total)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._path.exists():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="ckpt-tail")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def stats(self) -> tuple[int, int, dict[str, tuple[int, int]]]:
        """Return (passed, total, by_suite)."""
        with self._lock:
            return self._passed, self._total, dict(self._by_suite)

    def _run(self) -> None:
        offset = 0
        while not self._stop.is_set():
            try:
                size = self._path.stat().st_size
            except OSError:
                self._stop.wait(self._poll)
                continue
            if size > offset:
                with open(self._path) as f:
                    f.seek(offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self._ingest(rec)
                    offset = f.tell()
            self._stop.wait(self._poll)

    def _ingest(self, rec: dict) -> None:
        passed = rec.get("passed", False)
        suite = rec.get("suite", "?")
        with self._lock:
            self._total += 1
            if passed:
                self._passed += 1
            sp, st = self._by_suite.get(suite, (0, 0))
            self._by_suite[suite] = (sp + (1 if passed else 0), st + 1)


# ---------------------------------------------------------------------------
# Tap discovery — find the active tap file
# ---------------------------------------------------------------------------


def _discover_tap() -> str:
    """Find the inference tap path from sentinel or default."""
    try:
        with open(_SENTINEL) as f:
            path = f.read().strip()
        if path and os.path.exists(path):
            return path
    except (FileNotFoundError, OSError):
        pass
    if os.path.exists(_DEFAULT_TAP):
        return _DEFAULT_TAP
    return _DEFAULT_TAP  # will wait for it to appear


def _discover_checkpoint() -> Path | None:
    """Find the most recently modified checkpoint JSONL."""
    if not _EVAL_DIR.exists():
        return None
    candidates = sorted(_EVAL_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.name == "seen_questions.jsonl":
            continue
        return c
    return None


# ---------------------------------------------------------------------------
# WatchTUI
# ---------------------------------------------------------------------------


class WatchTUI:
    """Observer TUI — activates inference tap via sentinel, tails tap files.

    Writes the sentinel file so the running orchestrator API starts writing
    to the tap. Cleans up the sentinel on exit so tap output stops.
    The tap file itself is NOT deleted (the running session may want it).
    """

    def __init__(
        self,
        tap_path: str,
        repl_tap_path: str = _DEFAULT_REPL_TAP,
        checkpoint_path: Path | None = None,
        refresh: int = 4,
    ) -> None:
        self._console = Console()
        self._tailer = TapTailer(tap_path, name="watch-inference")
        self._repl_tailer = TapTailer(repl_tap_path, name="watch-repl", section_aware=False)
        self._ckpt = CheckpointTailer(checkpoint_path) if checkpoint_path else None
        self._refresh = refresh
        self._start_time = time.monotonic()
        self._tap_path = tap_path
        self._owns_sentinel = False

    def _activate_tap(self) -> None:
        """Write sentinel so the orchestrator API starts writing tap output."""
        # Don't clobber if sentinel already exists (another TUI owns it)
        if os.path.exists(_SENTINEL):
            return
        try:
            Path(_SENTINEL).parent.mkdir(parents=True, exist_ok=True)
            # Create/truncate the tap file so the tailer has something to open
            Path(self._tap_path).parent.mkdir(parents=True, exist_ok=True)
            if not os.path.exists(self._tap_path):
                Path(self._tap_path).touch()
            with open(_SENTINEL, "w") as f:
                f.write(self._tap_path)
            self._owns_sentinel = True
        except OSError:
            pass

    def _deactivate_tap(self) -> None:
        """Remove sentinel on exit so tap output stops."""
        if not self._owns_sentinel:
            return
        try:
            os.unlink(_SENTINEL)
        except (FileNotFoundError, OSError):
            pass

    def run(self) -> None:
        self._activate_tap()
        self._tailer.start()
        self._repl_tailer.start()
        if self._ckpt:
            self._ckpt.start()

        try:
            with Live(
                console=self._console,
                screen=True,
                refresh_per_second=self._refresh,
                get_renderable=self._make_layout,
            ):
                # Block until Ctrl-C
                stop = threading.Event()
                signal.signal(signal.SIGINT, lambda *_: stop.set())
                stop.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self._tailer.stop()
            self._repl_tailer.stop()
            if self._ckpt:
                self._ckpt.stop()
            self._deactivate_tap()

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body"),
        )

        # -- Header --
        elapsed = time.monotonic() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        parts = [" WATCH MODE"]
        role_chain = self._tailer.get_role_chain()
        if role_chain:
            parts.append(" \u2192 ".join(role_chain))

        if self._ckpt:
            passed, total, _ = self._ckpt.stats()
            if total > 0:
                rate = passed / total * 100
                parts.append(f"{passed}/{total} passed ({rate:.0f}%)")

        sentinel_exists = os.path.exists(_SENTINEL)
        tap_has_data = False
        try:
            tap_has_data = os.path.exists(self._tap_path) and os.path.getsize(self._tap_path) > 0
        except OSError:
            pass
        if tap_has_data:
            parts.append("\u2022 tap streaming")
        elif sentinel_exists:
            parts.append("\u2022 tap activated, waiting for data...")
        else:
            parts.append("\u2022 tap inactive")
        parts.append(f"{mins}m{secs:02d}s")

        layout["header"].update(Text(" | ".join(parts), style="bold"))

        # -- Body: inference (large) + sidebar --
        layout["body"].split_row(
            Layout(name="main", ratio=7),
            Layout(name="side", ratio=3),
        )

        # Main: inference stream
        layout["main"].split_column(
            Layout(name="stream", ratio=7),
            Layout(name="repl", ratio=3),
        )

        try:
            total_h = self._console.height - 1
            stream_h = max(8, total_h * 7 // 10 - 2)
            repl_h = max(3, total_h * 3 // 10 - 2)
            side_w = max(20, self._console.width * 3 // 10 - 4)
        except Exception:
            stream_h, repl_h, side_w = 20, 8, 30

        # Inference panel
        raw_section = self._tailer.get_current_section()
        filtered: list[str] = []
        in_prompt = False
        for line in raw_section:
            if line.startswith("PROMPT:") or line == "PROMPT:":
                in_prompt = True
                filtered.append("PROMPT: [...]")
                continue
            if in_prompt:
                if line.startswith("-" * 20) or line.startswith("=" * 20):
                    in_prompt = False
                else:
                    continue
            if line.startswith("RESPONSE:"):
                filtered.append("")
            filtered.append(line)

        stream_vis = max(5, stream_h - 2)
        hidden = max(0, len(filtered) - stream_vis)
        in_code_init = False
        if hidden > 0:
            for h in filtered[:hidden]:
                if h.lstrip().startswith("```"):
                    in_code_init = not in_code_init
        display = [_sanitize_display(l) for l in filtered[-stream_vis:]]
        stream_text = _style_stream_lines(display, in_code_init) if display else Text("(waiting for inference tap...)")

        stream_title = f"Inference ({' \u2192 '.join(role_chain)})" if role_chain else "Inference Stream"
        layout["stream"].update(Panel(stream_text, title=stream_title, border_style="cyan"))

        # REPL panel
        repl_vis = max(3, repl_h - 2)
        repl_lines = self._repl_tailer.get_current_section()
        repl_display = [_sanitize_display(l) for l in repl_lines[-repl_vis:]]
        repl_text = _style_repl_lines(repl_display) if repl_display else Text("(no REPL activity)")
        layout["repl"].update(Panel(repl_text, title="REPL Execution", border_style="magenta"))

        # Sidebar: checkpoint stats
        if self._ckpt:
            passed, total, by_suite = self._ckpt.stats()
            lines: list[str] = []
            if total > 0:
                rate = passed / total * 100
                lines.append(f"Overall: {passed}/{total} ({rate:.1f}%)")
                lines.append("")
                # Sort suites by total descending
                for suite, (sp, st) in sorted(by_suite.items(), key=lambda x: -x[1][1]):
                    sr = sp / st * 100 if st else 0
                    bar_len = 12
                    filled = int(sr / 100 * bar_len)
                    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                    lines.append(f"{suite[:15]:<15}")
                    lines.append(f"  {bar} {sp}/{st} ({sr:.0f}%)")
            else:
                lines.append("(no results yet)")
            stats_text = Text("\n".join(lines))
            layout["side"].update(Panel(stats_text, title="Pass Rates", border_style="green"))
        else:
            layout["side"].update(Panel(
                Text("No checkpoint.\nUse --checkpoint <session_id>\nto see live stats."),
                title="Stats",
                border_style="dim",
            ))

        return layout


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch an active seeding session (passive observer TUI)"
    )
    parser.add_argument(
        "--tap", default=None,
        help=f"Inference tap path (default: auto-discover from sentinel or {_DEFAULT_TAP})"
    )
    parser.add_argument(
        "--repl-tap", default=_DEFAULT_REPL_TAP,
        help=f"REPL tap path (default: {_DEFAULT_REPL_TAP})"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Session ID or path to checkpoint JSONL for live stats (default: most recent)"
    )
    parser.add_argument(
        "--refresh", type=int, default=4,
        help="Refresh rate in Hz (default: 4)"
    )
    args = parser.parse_args()

    # Resolve tap path
    tap = args.tap or _discover_tap()

    # Resolve checkpoint
    ckpt_path: Path | None = None
    if args.checkpoint:
        p = Path(args.checkpoint)
        if p.exists():
            ckpt_path = p
        else:
            # Treat as session ID
            ckpt_path = _EVAL_DIR / f"{args.checkpoint}.jsonl"
            if not ckpt_path.exists():
                print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
                ckpt_path = None
    else:
        ckpt_path = _discover_checkpoint()

    print(f"Tap:        {tap}")
    print(f"REPL tap:   {args.repl_tap}")
    print(f"Checkpoint: {ckpt_path or '(none)'}")
    print("Starting watch TUI... (Ctrl-C to exit)\n")
    time.sleep(0.5)

    tui = WatchTUI(
        tap_path=tap,
        repl_tap_path=args.repl_tap,
        checkpoint_path=ckpt_path,
        refresh=args.refresh,
    )
    tui.run()


if __name__ == "__main__":
    main()
