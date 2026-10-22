"""Per-turn model-input recording for the replay tier (HIL-16).

One ``model_turn`` row per root-LM turn: the exact prompt string and raw model output (both in the
CAS), the call kwargs, ``step_mark`` (ordinal of the next ``tool_step`` event, so the steps a turn's
code executed are ``[step_mark, next_turn.step_mark)``), and a hash ``chain`` over every prompt and
output so far. Tool outputs are recorded by ``StepEventLog`` (``output_cas`` / ``args_cas`` fields)
when it is given a CAS. Default OFF behind ``replay_recording``; with it off nothing here is built.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

from src.trace.replay.cas import CAS
from src.trace.replay.clock import ClockTape, activate as _activate_clock

CATEGORY = "model_turn"
SCHEMA = "model_turn/v1"
GENESIS = "sha256:" + "0" * 64


def chain_next(prev: str, prompt_cas: str, output_cas: str) -> str:
    return "sha256:" + hashlib.sha256(f"{prev}\x1f{prompt_cas}\x1f{output_cas}".encode()).hexdigest()


def _jsonable(kwargs: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(kwargs, sort_keys=True, default=str))


_ACTIVE: ContextVar["ReplayRecorder | None"] = ContextVar("replay_active_recorder", default=None)
_SECRET_RE = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_ENV_EXACT = ("TOOL_DEFINITION_VARIANT", "PYTHONHASHSEED")


def prompt_env_snapshot() -> dict[str, str]:
    """Environment that can change prompt bytes: ORCHESTRATOR_* plus the few non-prefixed knobs; never secrets."""
    return {k: v for k, v in sorted(os.environ.items())
            if (k.startswith("ORCHESTRATOR_") or k in _ENV_EXACT) and not any(w in k.upper() for w in _SECRET_RE)}


def _wrap_llm_call(primitives: Any) -> None:
    """Wrap ``primitives.llm_call`` ONCE so every model call made while a recorder is active (in this
    contextvar scope) is noted. Recorder-less callers pass straight through. Flag-on only."""
    orig = primitives.llm_call
    if getattr(orig, "_replay_wrapped", False):
        return

    @functools.wraps(orig)
    def _wrapped(*a: Any, **kw: Any) -> Any:
        out = orig(*a, **kw)
        # contextvar first; executors that do not copy context (run_in_executor) fall back to the latest
        # recorder begun on these primitives (ambiguous only if two runs share one primitives object)
        rec = _ACTIVE.get() or getattr(primitives, "_replay_last_recorder", None)
        if rec is not None:
            rec.note_llm_call(a[0] if a else kw.get("prompt", ""), kw.get("role"), {k: v for k, v in kw.items() if k != "role"}, out)
        return out

    _wrapped._replay_wrapped = True   # type: ignore[attr-defined]
    primitives.llm_call = _wrapped


@dataclass
class ReplayRecorder:
    run_id: str
    cas: CAS
    sink: Callable[[dict], None] | None = None
    role: str | None = None
    turn: int = 0
    chain: str = GENESIS
    records: list[dict] = field(default_factory=list)
    tape: ClockTape = field(default_factory=ClockTape)
    _tape_mark: int = 0
    _calls: list[dict] = field(default_factory=list)
    _seed: dict | None = None
    step_mark: int = 0

    # -- HIL-16b: per-turn context (all flag-on only) ---------------------------------------------
    def begin_turn(self, *, primitives: Any, step_mark: int, seed_fn: Callable[[], dict] | None = None) -> None:
        """Called at the top of ``_execute_turn``: activate this run's clock tape and LLM-call tap, take the
        step mark BEFORE any pre-prompt tool call (auto-gather peeks), and capture the seed on the first turn."""
        try:
            # ``_calls`` is NOT cleared here (HIL-16c): calls noted since the previous row -- REPL-code
            # ``llm_call`` sub-calls and a reasoning retry, made after that row's main call -- become the head
            # of this row's ``aux`` list, ahead of this turn's own pre-prompt calls. ``record_turn`` clears it.
            self.step_mark = step_mark
            _ACTIVE.set(self)
            _activate_clock(self.tape)
            if primitives is not None and callable(getattr(primitives, "llm_call", None)):
                _wrap_llm_call(primitives)
                primitives._replay_last_recorder = self
            if self.turn == 0 and self._seed is None and seed_fn is not None:
                self._seed = seed_fn()
        except Exception:   # recording must never fail a turn
            pass

    def note_llm_call(self, prompt: str, role: str | None, kwargs: dict[str, Any], output: Any) -> None:
        try:
            self._calls.append({"prompt_cas": self.cas.put(prompt), "output_cas": self.cas.put(output if isinstance(output, str) else str(output)),
                                "role": role, "kwargs": _jsonable(kwargs)})
        except Exception:
            pass

    def record_turn(self, *, role: str | None, prompt: str, output: str, kwargs: dict[str, Any] | None,
                    step_mark: int | None = None, force_answer: bool | None = None,
                    meta: dict[str, Any] | None = None) -> dict | None:
        try:
            p, o = self.cas.put(prompt), self.cas.put(output)
            self.chain = chain_next(self.chain, p, o)
            rec = {"schema": SCHEMA, "run_id": self.run_id, "turn": self.turn, "role": role or self.role,
                   "prompt_cas": p, "output_cas": o, "kwargs": _jsonable(kwargs or {}),
                   "step_mark": self.step_mark if step_mark is None else step_mark, "chain": self.chain}
            # every model call since the previous row before this main one: the previous turn's post-main calls
            # (REPL llm_call, reasoning retry; HIL-16c), then this turn's session summary / consolidation / compaction
            main = next((i for i in range(len(self._calls) - 1, -1, -1) if self._calls[i]["prompt_cas"] == p), None)
            aux = self._calls[:main] if main is not None else list(self._calls)
            if aux:
                rec["aux"] = aux
            reads = self.tape.reads[self._tape_mark:]
            self._tape_mark = len(self.tape.reads)
            if reads:
                rec["clock"] = reads
            if meta:
                rec["meta"] = _jsonable(meta)
            if force_answer is not None:
                rec["force_answer"] = bool(force_answer)
            if self.turn == 0 and self._seed is not None:
                rec["seed"] = self._seed
            self._calls = []
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
