"""HIL-2 item 21: append-only attempt ledger on the LOOP's ExperimentJournal.

Each opened candidate-vs-incumbent comparison is one attempt. Its alpha is RESERVED when it is
opened, before any observation; an attempt that never closes (crash, interrupt) still consumes
its alpha; a restart resumes the attempt index from the journal. Rows are the journal's own
durable ledger events (flock + fsync + torn-tail repair), so the ledger is outside the mutable
surface: the loop can append to it but has no API to rewrite it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

OPEN = "hil_attempt_open"
CLOSE = "hil_attempt_close"
ATTEMPT_OUTCOMES = frozenset({"keep", "keep_base", "refused", "error"})


class AlphaBudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Attempt:
    index: int
    candidate_id: str
    alpha_reserved: float
    outcome: str | None  # None == opened, never closed (interrupted)


class AttemptLedger:
    def __init__(self, journal: Any, *, alpha_budget: float) -> None:
        if not (math.isfinite(alpha_budget) and 0.0 < alpha_budget <= 1.0):
            raise ValueError(f"alpha_budget must be in (0, 1], got {alpha_budget!r}")
        self._journal = journal
        self.alpha_budget = float(alpha_budget)

    def attempts(self) -> list[Attempt]:
        closes = {int(e["attempt_index"]): str(e.get("outcome"))
                  for e in self._journal.ledger_events(CLOSE)}
        return [
            Attempt(int(e["attempt_index"]), str(e.get("candidate_id", "")),
                    float(e["alpha_reserved"]), closes.get(int(e["attempt_index"])))
            for e in self._journal.ledger_events(OPEN)
        ]

    def alpha_spent(self) -> float:
        return float(sum(a.alpha_reserved for a in self.attempts()))

    def next_index(self) -> int:
        return len(self._journal.ledger_events(OPEN))

    def open(self, candidate_id: str, alpha: float, **context: Any) -> int:
        if not (math.isfinite(alpha) and alpha > 0.0):
            raise ValueError(f"alpha must be positive, got {alpha!r}")
        if self.alpha_spent() + alpha > self.alpha_budget + 1e-12:
            raise AlphaBudgetExhausted(
                f"alpha {alpha:g} would exceed budget {self.alpha_budget:g} "
                f"(spent {self.alpha_spent():g})"
            )
        index = self.next_index()
        self._journal.append_ledger_event({
            "type": OPEN, "attempt_index": index, "candidate_id": candidate_id,
            "alpha_reserved": float(alpha), **context,
        })
        return index

    def close(self, index: int, outcome: str, **detail: Any) -> None:
        if outcome not in ATTEMPT_OUTCOMES:
            raise ValueError(f"outcome {outcome!r} not in {sorted(ATTEMPT_OUTCOMES)}")
        opened = {a.index: a for a in self.attempts()}
        if index not in opened:
            raise KeyError(f"attempt {index} was never opened")
        if opened[index].outcome is not None:
            raise ValueError(f"attempt {index} already closed ({opened[index].outcome})")
        self._journal.append_ledger_event({
            "type": CLOSE, "attempt_index": int(index), "outcome": outcome, **detail,
        })
