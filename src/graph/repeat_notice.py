"""TOC-SP-9 DRAFT: result-equality repeat-notice guard (default off, NOT wired).

Separable from ``ORCHESTRATOR_REPL_LOOP_GUARD`` (which counts turns without a file write
or FINAL). The evidence key is (action sha256, observation/exit+output sha256, file
revision, range): the same call with a different result, or the same result against a
different file revision, is a different key. One factual notice per key, only after a
quiet period; the text does not claim stalled progress. Terminate/reminder thresholds
(5/8 identical failing turns) are a published hypothesis with no efficacy evidence
(intake-1991#record). The notice is measured apart from view shortening (intake-2153#02).
No caller exists yet; activation belongs to a later A/B.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

QUIET_THROUGH_TURN = 10
REMINDER_STREAK = 5
TERMINATE_STREAK = 8


def repeat_notice_enabled() -> bool:
    return os.environ.get("ORCHESTRATOR_REPL_REPEAT_NOTICE") == "1"


def _h(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()


@dataclass(frozen=True)
class Decision:
    notice: str | None = None
    reminder: bool = False
    terminate: bool = False


@dataclass
class RepeatNoticeGuard:
    quiet_through_turn: int = QUIET_THROUGH_TURN
    reminder_streak: int = REMINDER_STREAK
    terminate_streak: int = TERMINATE_STREAK
    _seen: dict = field(default_factory=dict)
    _noticed: set = field(default_factory=set)
    _streak_key: tuple | None = None
    _streak: int = 0

    @staticmethod
    def key(action: str, observation: str, exit_code: int | None = None,
            file_rev: str | None = None, file_range: str | None = None) -> tuple:
        result = f"{exit_code}\0{observation}" if exit_code is not None else observation
        return (_h(action), _h(result), file_rev, file_range)

    def observe(self, turn: int, action: str, observation: str, *, failed: bool = False,
                exit_code: int | None = None, file_rev: str | None = None,
                file_range: str | None = None) -> Decision:
        k = self.key(action, observation, exit_code, file_rev, file_range)
        count = self._seen.get(k, 0) + 1
        self._seen[k] = count
        if failed and k == self._streak_key:
            self._streak += 1
        else:
            self._streak_key, self._streak = (k if failed else None), (1 if failed else 0)
        if failed and self._streak >= self.terminate_streak:
            return Decision(terminate=True)
        notice = None
        if count >= 2 and turn > self.quiet_through_turn and k not in self._noticed:
            self._noticed.add(k)
            notice = (f"Repeat notice: this action returned a result identical to {count - 1} "
                      "earlier occurrence(s) (same file revision and range). This does not "
                      "establish that progress has stalled.")
        return Decision(notice=notice, reminder=failed and self._streak == self.reminder_streak)
