"""Rich live TUI for monitoring AutoPilot inference.

Four-panel layout matching the seeding TUI:
- Upper left:  AutoPilot log (tail autopilot.log)
- Lower left:  Current prompt being tested
- Upper right: Live inference stream (tap file)
- Lower right: REPL execution log (tap file)

Primary purpose: see inference live to detect hangs.

Usage:
    python autopilot_tui.py [--log PATH] [--tap PATH] [--repl-tap PATH]

    Or import and use as context manager alongside autopilot:
        with AutoPilotTUI() as tui:
            tui.set_prompt("What is 2+2?")
            # ... autopilot loop runs ...
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import textwrap
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Reuse TapTailer and styling from seeding TUI
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent / "benchmark"

import sys
sys.path.insert(0, str(BENCHMARK_DIR))

from seeding_tui import TapTailer, _style_stream_lines, _style_repl_lines, _sanitize_display

# Defaults
ORCH_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_LOG = ORCH_ROOT / "logs" / "autopilot.log"
DEFAULT_TAP = "/mnt/raid0/llm/tmp/inference_tap.log"
DEFAULT_REPL_TAP = "/mnt/raid0/llm/tmp/repl_tap.log"
SENTINEL_PATH = "/mnt/raid0/llm/tmp/.inference_tap_active"


class LogTailer:
    """Tails a log file (like autopilot.log), keeping last N lines."""

    def __init__(self, path: str, max_lines: int = 200, poll_interval: float = 0.25):
        self._path = path
        self._max = max_lines
        self._poll = poll_interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._lines: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="log-tailer")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_lines(self, tail: int = 0) -> list[str]:
        with self._lock:
            items = list(self._lines)
            return items[-tail:] if tail > 0 else items

    def _run(self) -> None:
        while not self._stop.is_set():
            if os.path.exists(self._path):
                break
            self._stop.wait(0.5)
        if self._stop.is_set():
            return

        with open(self._path, "rb") as fh:
            fh.seek(0, 2)  # seek to end
            while not self._stop.is_set():
                pos = fh.tell()
                fh.seek(pos)
                raw = fh.read(16384)
                if raw:
                    text = raw.decode("utf-8", errors="replace")
                    with self._lock:
                        for line in text.split("\n"):
                            if line.strip():
                                self._lines.append(line)
                else:
                    self._stop.wait(self._poll)


class AutoPilotTUI:
    """Context manager running the Rich live TUI for autopilot monitoring.

    Layout::

        ┌──────────────────┬───────────────────────────┐
        │ AutoPilot Log    │                           │
        │ (tail            │   Inference Stream        │
        │  autopilot.log)  │   (live model output)     │
        ├──────────────────┤                           │
        │ Current Prompt   ├───────────────────────────┤
        │ (being tested)   │   REPL Execution          │
        │                  │   (code + output)         │
        └──────────────────┴───────────────────────────┘
    """

    def __init__(
        self,
        log_path: str = str(DEFAULT_LOG),
        tap_path: str = DEFAULT_TAP,
        repl_tap_path: str = DEFAULT_REPL_TAP,
        refresh_per_second: int = 4,
    ):
        self._log_path = log_path
        self._tap_path = tap_path
        self._repl_tap_path = repl_tap_path
        self._refresh = refresh_per_second

        self._console = Console()
        self._log_tailer = LogTailer(log_path)
        self._inference_tailer = TapTailer(tap_path, name="tap-inference")
        self._repl_tailer = TapTailer(
            repl_tap_path, name="tap-repl", section_aware=False,
        )

        self._lock = threading.Lock()
        self._current_prompt: str = ""
        self._current_trial: int = 0
        self._current_species: str = ""
        self._start_time: float = time.monotonic()
        self._live: Live | None = None

    # -- public API (called from autopilot loop or externally) --

    def set_prompt(self, prompt: str) -> None:
        with self._lock:
            self._current_prompt = prompt

    def set_trial(self, trial_id: int, species: str = "") -> None:
        with self._lock:
            self._current_trial = trial_id
            self._current_species = species

    # -- context manager --

    def __enter__(self) -> "AutoPilotTUI":
        # Write sentinel so API discovers tap path
        try:
            Path(SENTINEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            Path(SENTINEL_PATH).write_text(self._tap_path)
        except OSError:
            pass

        self._log_tailer.start()
        self._inference_tailer.start()
        self._repl_tailer.start()

        self._live = Live(
            console=self._console,
            screen=True,
            refresh_per_second=self._refresh,
            get_renderable=self._make_layout,
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
        self._log_tailer.stop()
        self._inference_tailer.stop()
        self._repl_tailer.stop()
        try:
            Path(SENTINEL_PATH).unlink(missing_ok=True)
        except OSError:
            pass
        return False

    # -- layout --

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body"),
        )

        # Header
        elapsed = time.monotonic() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        with self._lock:
            trial = self._current_trial
            species = self._current_species

        role_chain = self._inference_tailer.get_role_chain()
        roles_str = " → ".join(role_chain) if role_chain else "idle"

        header_parts = [
            f" AutoPilot",
            f"trial #{trial}",
        ]
        if species:
            header_parts.append(species)
        header_parts.append(roles_str)
        header_parts.append(f"{hours}h{mins:02d}m{secs:02d}s")
        layout["header"].update(Text(" | ".join(header_parts), style="bold"))

        # Body
        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=5),
        )
        layout["left"].split_column(
            Layout(name="log", ratio=6),
            Layout(name="prompt", ratio=4),
        )
        layout["right"].split_column(
            Layout(name="stream", ratio=7),
            Layout(name="repl", ratio=3),
        )

        try:
            total_height = self._console.height - 1
            left_width = max(20, self._console.width * 3 // 8 - 6)
            log_height = max(5, total_height * 6 // 10 - 2)
            prompt_height = max(3, total_height * 4 // 10 - 2)
            stream_height = max(8, total_height * 7 // 10 - 2)
            repl_height = max(3, total_height * 3 // 10 - 2)
        except Exception:
            left_width = 50
            log_height, prompt_height = 15, 8
            stream_height, repl_height = 20, 8

        # Upper left: AutoPilot log
        log_vis = max(3, log_height - 2)
        log_wrap_w = max(20, left_width - 4)
        raw_log = self._log_tailer.get_lines()
        wrapped: list[str] = []
        for line in raw_log:
            line = _sanitize_display(line)
            if len(line) <= log_wrap_w:
                wrapped.append(line)
            else:
                wrapped.extend(textwrap.wrap(line, width=log_wrap_w))
        log_lines = wrapped[-log_vis:]
        log_text = self._style_log(log_lines) if log_lines else Text("(waiting for autopilot.log...)")
        layout["log"].update(Panel(
            log_text,
            title=f"AutoPilot Log ({len(raw_log)})",
            border_style="green",
        ))

        # Lower left: Current prompt
        p_vis = max(3, prompt_height - 2)
        with self._lock:
            prompt_raw = self._current_prompt
        if prompt_raw:
            prompt_raw = _sanitize_display(prompt_raw)
            p_wrap_w = max(20, left_width - 4)
            p_wrapped: list[str] = []
            for para in prompt_raw.split("\n"):
                if para.strip():
                    p_wrapped.extend(textwrap.wrap(para, width=p_wrap_w))
                else:
                    p_wrapped.append("")
            p_lines = p_wrapped[-p_vis:]
            p_display = "\n".join(p_lines)
        else:
            p_display = "(no prompt active)"
        layout["prompt"].update(Panel(
            Text(p_display, overflow="crop"),
            title="Current Prompt",
            border_style="yellow",
        ))

        # Upper right: Inference stream
        raw_section = self._inference_tailer.get_current_section()
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

        stream_vis = max(5, stream_height - 2)
        visible_stream = filtered[-stream_vis:]
        stream_styled = _style_stream_lines(visible_stream)
        role_title = " → ".join(role_chain[-2:]) if role_chain else "Inference"
        layout["stream"].update(Panel(
            stream_styled,
            title=role_title,
            border_style="cyan",
        ))

        # Lower right: REPL
        repl_lines = self._repl_tailer.get_current_section()
        repl_vis = max(3, repl_height - 2)
        visible_repl = repl_lines[-repl_vis:]
        repl_styled = _style_repl_lines(visible_repl)
        layout["repl"].update(Panel(
            repl_styled,
            title="REPL",
            border_style="magenta",
        ))

        return layout

    @staticmethod
    def _style_log(lines: list[str]) -> Text:
        """Highlight autopilot log lines."""
        styled = Text(overflow="crop")
        for i, line in enumerate(lines):
            if i > 0:
                styled.append("\n")
            if "ERROR" in line or "FAIL" in line:
                styled.append(line, style="bold red")
            elif "WARNING" in line:
                styled.append(line, style="yellow")
            elif "Trial" in line and "complete" in line:
                styled.append(line, style="bold green")
            elif "frontier" in line:
                styled.append(line, style="bold cyan")
            elif "reverting" in line or "rollback" in line:
                styled.append(line, style="bold magenta")
            else:
                styled.append(line)
        return styled


# -- Standalone mode: just monitor, don't run autopilot --

def main():
    parser = argparse.ArgumentParser(description="AutoPilot live inference monitor")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="Path to autopilot.log")
    parser.add_argument("--tap", default=DEFAULT_TAP, help="Inference tap file")
    parser.add_argument("--repl-tap", default=DEFAULT_REPL_TAP, help="REPL tap file")
    args = parser.parse_args()

    print("Starting AutoPilot TUI monitor...")
    print("  Log:      ", args.log)
    print("  Tap:      ", args.tap)
    print("  REPL tap: ", args.repl_tap)
    print("Press Ctrl+C to exit.\n")

    with AutoPilotTUI(
        log_path=args.log,
        tap_path=args.tap,
        repl_tap_path=args.repl_tap,
    ) as tui:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
