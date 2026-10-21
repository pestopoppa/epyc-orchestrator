"""Wall-clock reads that reach a prompt, as a recorded tape (HIL-16b).

The session log stamps ``HH:MM:SS`` into turn records and consolidated segments stamp ``time.time()``;
both can surface in later prompts (directly, or through the session-log file the model may ``peek``).
A replay on another second cannot reproduce them, so every such read goes through ``time()`` /
``now_hms()`` here. With no tape active they return exactly what the original expressions did, so live
bytes are unchanged. While recording, each read is appended to the tape; while replaying, reads are
served from the recorded tape in order. A tape that runs dry falls back to the real clock and sets
``exhausted`` (the byte comparison then reports the drift rather than hiding it).
"""
from __future__ import annotations

import time as _time
from contextvars import ContextVar
from datetime import datetime, timezone


class ClockTape:
    def __init__(self, replay: list[float] | None = None) -> None:
        self.replaying = replay is not None
        self.reads: list[float] = list(replay or [])
        self._i = 0
        self.exhausted = False

    def read(self) -> float:
        if not self.replaying:
            t = _time.time()
            self.reads.append(t)
            return t
        if self._i < len(self.reads):
            t = self.reads[self._i]
            self._i += 1
            return t
        self.exhausted = True
        return _time.time()


_TAPE: ContextVar[ClockTape | None] = ContextVar("replay_clock_tape", default=None)


def activate(tape: ClockTape | None):
    return _TAPE.set(tape)


def time() -> float:
    tape = _TAPE.get()
    return _time.time() if tape is None else tape.read()


def now_hms() -> str:
    tape = _TAPE.get()
    if tape is None:
        return datetime.now(timezone.utc).strftime("%H:%M:%S")
    return datetime.fromtimestamp(tape.read(), timezone.utc).strftime("%H:%M:%S")
