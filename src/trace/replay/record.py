"""Per-turn model-input recording for the replay tier (HIL-16).

One ``model_turn`` row per root-LM turn: the exact prompt string and raw model output (both in the
CAS), the call kwargs, ``step_mark`` (ordinal of the next ``tool_step`` event, so the steps a turn's
code executed are ``[step_mark, next_turn.step_mark)``), and a hash ``chain`` over every prompt and
output so far. Tool outputs are recorded by ``StepEventLog`` (``output_cas`` / ``args_cas`` fields)
when it is given a CAS. Default OFF behind ``replay_recording``; with it off nothing here is built.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from src.trace.replay.cas import CAS

CATEGORY = "model_turn"
SCHEMA = "model_turn/v1"
GENESIS = "sha256:" + "0" * 64


def chain_next(prev: str, prompt_cas: str, output_cas: str) -> str:
    return "sha256:" + hashlib.sha256(f"{prev}\x1f{prompt_cas}\x1f{output_cas}".encode()).hexdigest()


def _jsonable(kwargs: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(kwargs, sort_keys=True, default=str))


@dataclass
class ReplayRecorder:
    run_id: str
    cas: CAS
    sink: Callable[[dict], None] | None = None
    role: str | None = None
    turn: int = 0
    chain: str = GENESIS
    records: list[dict] = field(default_factory=list)

    def record_turn(self, *, role: str | None, prompt: str, output: str, kwargs: dict[str, Any] | None,
                    step_mark: int) -> dict | None:
        try:
            p, o = self.cas.put(prompt), self.cas.put(output)
            self.chain = chain_next(self.chain, p, o)
            rec = {"schema": SCHEMA, "run_id": self.run_id, "turn": self.turn, "role": role or self.role,
                   "prompt_cas": p, "output_cas": o, "kwargs": _jsonable(kwargs or {}),
                   "step_mark": step_mark, "chain": self.chain}
            self.turn += 1
            self.records.append(rec)
            if self.sink is not None:
                self.sink(rec)
            return rec
        except Exception:   # recording must never fail a turn
            return None


def trace_store_sink(db_path: Any = None) -> Callable[[dict], None]:
    from src.trace.emit import emit, synthetic_source_path
    from src.trace.store import DEFAULT_DB_PATH, Event, EventSource

    def _sink(rec: dict) -> None:
        emit(Event(
            ts_utc="", source=EventSource.REVIEW_PLANE,
            source_path=synthetic_source_path("model_turn", rec["run_id"], rec["turn"]),
            session_id=rec["run_id"], role=rec.get("role"), category=CATEGORY, status="ok",
            turn_ordinal=rec["turn"], summary=f"model_turn {rec['turn']}",
            detail_json=json.dumps(rec, sort_keys=True),
        ), db_path=db_path or DEFAULT_DB_PATH)

    return _sink
