"""Load a recorded run (``model_turn`` + ``tool_step`` rows + CAS) and check its integrity."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.trace.replay.cas import CAS, CASIntegrityError
from src.trace.replay.record import GENESIS, chain_next


@dataclass
class StepRec:
    idx: int
    kind: str
    tool: str | None
    args_hash: str | None
    args_cas: str | None
    output_cas: str | None
    ok: bool
    error_class: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # HIL-16b: output_kind / envelope / exc


@dataclass
class TurnRec:
    turn: int
    role: str | None
    prompt_cas: str
    output_cas: str
    kwargs: dict[str, Any]
    step_mark: int
    chain: str
    steps: list[StepRec] = field(default_factory=list)
    aux: list[dict[str, Any]] = field(default_factory=list)   # HIL-16b: other LLM calls made before the main call
    clock: list[float] = field(default_factory=list)          # HIL-16b: wall-clock reads since the previous row
    meta: dict[str, Any] = field(default_factory=dict)        # HIL-16b: the call's inference meta (token counts feed budgets)
    force_answer: bool | None = None                          # HIL-16b: the (wall-time derived) forced-answer decision


@dataclass
class RecordedRun:
    run_id: str
    turns: list[TurnRec]
    seed: dict[str, Any] = field(default_factory=dict)   # HIL-16b: initial state/env of the recorded run


def _rows(conn: sqlite3.Connection, run_id: str, category: str) -> list[dict]:
    cur = conn.execute("select detail_json from event where category=? and session_id=? order by id", (category, run_id))
    return [json.loads(d) for (d,) in cur]


def load_run(db_path: Path | str, run_id: str) -> RecordedRun:
    conn = sqlite3.connect(str(db_path))
    try:
        turns_raw = sorted(_rows(conn, run_id, "model_turn"), key=lambda r: r["turn"])
        steps_raw = sorted(_rows(conn, run_id, "tool_step"), key=lambda r: r["idx"])
    finally:
        conn.close()
    if not turns_raw:
        raise ValueError(f"no model_turn rows for run {run_id!r} (was it recorded with replay_recording on?)")
    turns = [TurnRec(r["turn"], r.get("role"), r["prompt_cas"], r["output_cas"], r.get("kwargs") or {},
                     r["step_mark"], r["chain"],
                     aux=list(r.get("aux") or []), clock=list(r.get("clock") or []),
                     force_answer=r.get("force_answer"), meta=dict(r.get("meta") or {})) for r in turns_raw]
    for i, t in enumerate(turns):
        hi = turns[i + 1].step_mark if i + 1 < len(turns) else None
        for s in steps_raw:
            if s["idx"] >= t.step_mark and (hi is None or s["idx"] < hi):
                t.steps.append(StepRec(s["idx"], s["kind"], s.get("tool"), s.get("args_hash"), s.get("args_cas"),
                                       s.get("output_cas"), bool(s.get("ok")), s.get("error_class"),
                                       {k: s[k] for k in ("output_kind", "envelope", "exc") if k in s}))
    return RecordedRun(run_id, turns, seed=dict(turns_raw[0].get("seed") or {}))


def verify_integrity(run: RecordedRun, cas: CAS) -> list[str]:
    """Empty list == every blob re-hashes to its address and the prompt/output chain recomputes."""
    problems: list[str] = []
    prev = GENESIS
    for t in run.turns:
        for what, addr in (("prompt", t.prompt_cas), ("output", t.output_cas)):
            try:
                cas.get(addr)
            except (CASIntegrityError, FileNotFoundError) as e:
                problems.append(f"turn {t.turn} {what}: {type(e).__name__}")
        prev = chain_next(prev, t.prompt_cas, t.output_cas)
        if prev != t.chain:
            problems.append(f"turn {t.turn}: chain mismatch")
        for s in t.steps:
            for what, addr in (("args", s.args_cas), ("output", s.output_cas)):
                if s.kind == "tool_call" and addr is None:
                    problems.append(f"step {s.idx} {what}: not recorded")
                elif addr is not None:
                    try:
                        cas.get(addr)
                    except (CASIntegrityError, FileNotFoundError) as e:
                        problems.append(f"step {s.idx} {what}: {type(e).__name__}")
    return problems
