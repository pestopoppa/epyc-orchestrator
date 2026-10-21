"""Prefix-preserving replay driver (HIL-16).

``ReplayDriver`` restarts a recorded run at turn ``k``. Turns ``< k`` are replayed from the
recording: the harness rebuilds each prompt from its OWN state and the driver compares those bytes
to the recorded prompt; the recorded model output is fed back and every tool call is served from
the recorded (content-addressed) output, with tool name and argument hash checked. Any difference
is a ``PrefixDivergence`` and NO model call is made on a diverged prefix. From turn ``k`` the live
model and live tools take over, with an optional ``Patch`` appended to the turn-k prompt.

The prefix check needs no inference (``check_prefix``); ``run_from`` is the inference path.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.trace.replay.cas import CAS
from src.trace.replay.model import RecordedRun, TurnRec
from src.trace.step_events import args_hash as _args_hash


class PrefixDivergence(Exception):
    def __init__(self, turn: int, what: str, detail: str = "") -> None:
        super().__init__(f"turn {turn}: {what} {detail}".strip())
        self.turn, self.what, self.detail = turn, what, detail


@dataclass
class ToolCall:
    tool: str
    args_hash: str
    output: str


@dataclass
class Outcome:
    is_final: bool = False
    final_answer: str | None = None
    calls: list[ToolCall] = field(default_factory=list)


class ToolServer:
    """Serves one turn's tool calls: from the recording (replay) or from ``live`` (after the restart)."""

    def __init__(self, turn: TurnRec | None, cas: CAS, live: bool) -> None:
        self._steps = [s for s in (turn.steps if turn else []) if s.kind == "tool_call"]
        self._cas, self._live, self._i = cas, live, 0
        self._turn = turn.turn if turn else -1

    def call(self, tool: str, args: dict[str, Any] | None, live_fn: Callable[[], str] | None = None) -> str:
        ah = _args_hash(args)
        if self._live:
            if live_fn is None:
                raise RuntimeError("live tool call without a live_fn")
            return live_fn()
        if self._i >= len(self._steps):
            raise PrefixDivergence(self._turn, "extra_tool_call", f"{tool}")
        rec = self._steps[self._i]
        self._i += 1
        if rec.tool != tool or rec.args_hash != ah:
            raise PrefixDivergence(self._turn, "tool_call_mismatch",
                                   f"recorded {rec.tool}/{rec.args_hash} vs {tool}/{ah}")
        return self._cas.get_text(rec.output_cas)

    def finish(self) -> None:
        if not self._live and self._i != len(self._steps):
            raise PrefixDivergence(self._turn, "unconsumed_tool_calls", f"{len(self._steps) - self._i} left")


class Harness(Protocol):
    def reset(self, run: RecordedRun) -> None: ...
    def build_prompt(self) -> str: ...
    def execute(self, model_output: str, tools: ToolServer) -> Outcome: ...


@dataclass(frozen=True)
class Patch:
    text: str

    def apply(self, prompt: str) -> str:
        return prompt + "\n\n" + self.text


@dataclass(frozen=True)
class Arm:
    name: str                 # targeted | no_hint | placebo | wrong_step
    restart_turn: int
    patch: Patch | None


@dataclass
class PrefixResult:
    ok: bool
    turns_checked: int
    digest: str
    divergence: dict | None = None


@dataclass
class ReplayRunResult:
    prefix: PrefixResult
    final_answer: str | None = None
    is_final: bool = False
    turns_run: int = 0
    transcript: list[dict] = field(default_factory=list)


LLM = Callable[..., str]     # llm(prompt, role=..., **kwargs) -> raw model output


class ReplayDriver:
    def __init__(self, run: RecordedRun, cas: CAS, harness: Harness) -> None:
        self.run, self.cas, self.harness = run, cas, harness

    def _replay_prefix(self, k: int) -> PrefixResult:
        if not 0 <= k <= len(self.run.turns):
            raise ValueError(f"restart turn {k} outside 0..{len(self.run.turns)}")
        self.harness.reset(self.run)
        h = hashlib.sha256()
        for t in self.run.turns[:k]:
            try:
                want = self.cas.get(t.prompt_cas)
                got = self.harness.build_prompt().encode("utf-8")
                if not hmac.compare_digest(want, got):
                    raise PrefixDivergence(t.turn, "prompt_bytes_differ", f"{len(want)} vs {len(got)} bytes")
                out = self.cas.get_text(t.output_cas)
                tools = ToolServer(t, self.cas, live=False)
                outcome = self.harness.execute(out, tools)
                tools.finish()
                for s in (x for x in t.steps if x.kind == "tool_call"):
                    h.update(self.cas.get(s.output_cas))
                h.update(want); h.update(out.encode("utf-8"))
                if outcome.is_final and t.turn < k - 1:
                    raise PrefixDivergence(t.turn, "final_before_restart")
            except PrefixDivergence as e:
                return PrefixResult(False, t.turn, h.hexdigest(), {"turn": e.turn, "what": e.what, "detail": e.detail})
        return PrefixResult(True, k, "sha256:" + h.hexdigest())

    def check_prefix(self, k: int) -> PrefixResult:
        """No inference: replay turns < k and compare bytes."""
        return self._replay_prefix(k)

    def run_from(self, k: int, llm: LLM, patch: Patch | None = None, *, max_turns: int = 30,
                 llm_kwargs: dict[str, Any] | None = None) -> ReplayRunResult:
        prefix = self._replay_prefix(k)
        res = ReplayRunResult(prefix=prefix)
        if not prefix.ok:
            return res           # never spend inference on a diverged prefix
        recorded = self.run.turns
        for turn in range(k, max_turns):
            base = recorded[turn] if turn < len(recorded) else recorded[-1]
            prompt = self.harness.build_prompt()
            if turn == k and patch is not None:
                prompt = patch.apply(prompt)
            kwargs = {**base.kwargs, **(llm_kwargs or {})}
            out = llm(prompt, role=base.role, **kwargs)
            outcome = self.harness.execute(out, ToolServer(None, self.cas, live=True))
            res.transcript.append({"turn": turn, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                                   "output": out, "calls": [c.tool for c in outcome.calls]})
            res.turns_run += 1
            if outcome.is_final:
                res.is_final, res.final_answer = True, outcome.final_answer
                break
        return res


# --------------------------------------------------------------------------- arms
PLACEBO_BASE = ("Take your time and work carefully. Keep your reasoning organised and check each step "
                "before you continue. ")


def placebo_text(hint: str) -> str:
    """Neutral text of exactly ``len(hint)`` characters, carrying no task information."""
    reps = len(hint) // len(PLACEBO_BASE) + 1
    return (PLACEBO_BASE * reps)[: len(hint)]


def choose_wrong_turn(k: int, n_turns: int, salt: str = "") -> int:
    """A restart turn j != k, deterministic in (k, n_turns, salt); prefers |j-k| >= 2 when one exists."""
    cands = [j for j in range(n_turns) if j != k]
    if not cands:
        raise ValueError("a run with one turn has no wrong-step control")
    far = [j for j in cands if abs(j - k) >= 2] or cands
    h = int(hashlib.sha256(f"{k}:{n_turns}:{salt}".encode()).hexdigest(), 16)
    return far[h % len(far)]


def build_arms(run: RecordedRun, k: int, hint: str, *, wrong_turn: int | None = None) -> list[Arm]:
    wj = choose_wrong_turn(k, len(run.turns), run.run_id) if wrong_turn is None else wrong_turn
    if wj == k:
        raise ValueError("wrong-step turn must differ from the targeted turn")
    return [
        Arm("targeted", k, Patch(hint)),
        Arm("no_hint", k, None),
        Arm("placebo", k, Patch(placebo_text(hint))),
        Arm("wrong_step", wj, Patch(hint)),
    ]


Scorer = Callable[[str | None], bool]


def run_arm(arm: Arm, n: int, *, run: RecordedRun, cas: CAS, harness_factory: Callable[[], Harness],
            llm_factory: Callable[[int], LLM], scorer: Scorer, base_seed: int = 1000,
            temperature: float = 0.7, max_turns: int = 30) -> list[dict]:
    """n samples of one arm. Every sample started is returned, including errors and diverged prefixes."""
    out: list[dict] = []
    for i in range(n):
        seed = base_seed + i
        try:
            d = ReplayDriver(run, cas, harness_factory())
            r = d.run_from(arm.restart_turn, llm_factory(seed), arm.patch, max_turns=max_turns,
                           llm_kwargs={"temperature": temperature, "seed": seed})
            if not r.prefix.ok:
                out.append({"seed": seed, "result": "error", "reason": "prefix_diverged", "detail": r.prefix.divergence})
            elif not r.is_final:
                out.append({"seed": seed, "result": "fail", "reason": "no_final", "turns": r.turns_run})
            else:
                out.append({"seed": seed, "result": "pass" if scorer(r.final_answer) else "fail", "turns": r.turns_run})
        except Exception as e:   # counted in the denominator, never dropped
            out.append({"seed": seed, "result": "error", "reason": type(e).__name__})
    return out
