"""RA-9 — gold-sanity gate for machine-generated tests and annotations (intake-983#record).

Procedure, in this order, and **the order is the finding**:

1. inject the generated test into the REAL project test file,
2. apply the GOLD solution (the code state known to be correct),
3. run the project's NATIVE runner,
4. on gold failure: discard and regenerate at a HIGHER temperature,
5. only after a native pass: consult a 3-sample self-consistency judge.

The judge reasons about generated test code that may not itself run; in the
source study it endorsed all six named invalid cases, which is why it may never
be consulted before the native runner has passed on gold. The measured
per-augmentation defect rate the gate exists for is 61.9% (n=105).

Ablation preserved by construction: retry PRESENCE is load-bearing (3/11 ->
9/11) and retry STYLE is not (a neutral prompt also reaches 9/11). So a
configuration with no retry is refused, and ``generate`` receives only a
temperature: there is deliberately no retry-prompt parameter to design.

This module executes nothing itself. The caller supplies the hooks: the
native runner is a subprocess the caller owns, and ``generate`` / ``judge`` are
model calls the caller owns. Tests use fakes. ``validate_record`` re-checks the
ordering invariants on a stored record, so a record assembled by hand (or by a
different harness) cannot claim admission it did not earn.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

PROCEDURE_VERSION = "gold-sanity-v1"
STAGES: tuple[str, ...] = ("inject", "apply_gold", "native_run", "judge")
DEFAULT_TEMPERATURES: tuple[float, ...] = (0.2, 0.6, 1.0)
JUDGE_SAMPLES = 3

ADMITTED = "admitted"
DISCARDED_GOLD_FAILURE = "discarded_gold_failure"
DISCARDED_JUDGE = "discarded_judge"


class GoldSanityConfigError(ValueError):
    """The gate was configured in a way that removes a load-bearing property."""


@dataclass(frozen=True)
class GateHooks:
    """Caller-owned side effects. Each stage hook returns True on success.

    ``generate(temperature)`` -> the generated artifact (test source text).
    ``inject(artifact)``      -> True if the artifact was placed in the real test file.
    ``apply_gold(artifact)``  -> True if the gold solution was applied.
    ``run_native(artifact)``  -> True if the project's native runner passed.
    ``judge(artifact)``       -> one self-consistency sample: True = endorse.
    ``hash_artifact``         -> optional content hash for the record.
    """

    generate: Callable[[float], str]
    inject: Callable[[str], bool]
    apply_gold: Callable[[str], bool]
    run_native: Callable[[str], bool]
    judge: Callable[[str], bool]
    hash_artifact: Callable[[str], str] | None = None


@dataclass
class _Attempt:
    attempt: int
    temperature: float
    artifact_hash: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    judge_votes: list[bool] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "attempt": self.attempt,
            "temperature": self.temperature,
            "stages": self.stages,
        }
        if self.artifact_hash is not None:
            out["artifact_hash"] = self.artifact_hash
        if self.judge_votes is not None:
            out["judge_votes"] = self.judge_votes
        return out


@dataclass(frozen=True)
class GateResult:
    outcome: str
    artifact: str | None
    record: dict[str, Any]

    @property
    def admitted(self) -> bool:
        return self.outcome == ADMITTED


def _check_config(temperatures: Sequence[float], judge_samples: int) -> None:
    if len(temperatures) < 2:
        raise GoldSanityConfigError(
            "retry presence is load-bearing (intake-983: 3/11 -> 9/11); configure >= 2 temperatures"
        )
    if any(b <= a for a, b in zip(temperatures, temperatures[1:])):
        raise GoldSanityConfigError("retry temperatures must be strictly increasing")
    if any(t < 0 for t in temperatures):
        raise GoldSanityConfigError("temperatures must be non-negative")
    if judge_samples < JUDGE_SAMPLES or judge_samples % 2 == 0:
        raise GoldSanityConfigError("judge needs an odd sample count >= 3 (self-consistency majority)")


def _run_stage(attempt: _Attempt, stage: str, hook: Callable[[str], bool], artifact: str) -> bool:
    try:
        ok = bool(hook(artifact))
        detail = None
    except Exception as exc:  # a crashed stage is a failed stage, never a pass
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    entry: dict[str, Any] = {"stage": stage, "ok": ok}
    if detail:
        entry["detail"] = detail
    attempt.stages.append(entry)
    return ok


def run_gate(
    hooks: GateHooks,
    *,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    judge_samples: int = JUDGE_SAMPLES,
) -> GateResult:
    """Run the gate. The judge is consulted at most once, and only after a native pass on gold."""
    _check_config(temperatures, judge_samples)
    attempts: list[_Attempt] = []
    for i, temperature in enumerate(temperatures, start=1):
        artifact = hooks.generate(temperature)
        att = _Attempt(
            attempt=i,
            temperature=float(temperature),
            artifact_hash=hooks.hash_artifact(artifact) if hooks.hash_artifact else None,
        )
        attempts.append(att)
        gold_ok = (
            _run_stage(att, "inject", hooks.inject, artifact)
            and _run_stage(att, "apply_gold", hooks.apply_gold, artifact)
            and _run_stage(att, "native_run", hooks.run_native, artifact)
        )
        if not gold_ok:
            continue  # discard; regenerate at the next (higher) temperature
        votes: list[bool] = []
        for _ in range(judge_samples):
            try:
                votes.append(bool(hooks.judge(artifact)))
            except Exception:  # an errored sample is not an endorsement
                votes.append(False)
        att.judge_votes = votes
        endorsed = sum(votes) * 2 > len(votes)
        att.stages.append({"stage": "judge", "ok": endorsed})
        outcome = ADMITTED if endorsed else DISCARDED_JUDGE
        return GateResult(outcome, artifact if endorsed else None, _record(attempts, outcome))
    return GateResult(DISCARDED_GOLD_FAILURE, None, _record(attempts, DISCARDED_GOLD_FAILURE))


def _record(attempts: Sequence[_Attempt], outcome: str) -> dict[str, Any]:
    return {
        "procedure_version": PROCEDURE_VERSION,
        "outcome": outcome,
        "attempts": [a.as_dict() for a in attempts],
    }


def validate_record(record: Mapping[str, Any]) -> list[str]:
    """Ordering invariants a stored gold-sanity record must satisfy (empty == valid).

    Checked: attempts numbered 1..n; strictly increasing temperatures; each
    attempt's stages are a prefix of inject -> apply_gold -> native_run -> judge
    that stops at the first failure; judge appears only after a passing
    native_run, only on the final attempt, with an odd vote count >= 3 whose
    majority matches the judge stage; every non-final attempt failed on gold
    (a gold pass is never retried); and the outcome follows from the final attempt.
    """
    errs: list[str] = []
    attempts = list(record.get("attempts") or [])
    if not attempts:
        return ["no attempts recorded"]
    if [a.get("attempt") for a in attempts] != list(range(1, len(attempts) + 1)):
        errs.append("attempts must be numbered 1..n in order")
    temps = [a.get("temperature") for a in attempts]
    if any(not isinstance(t, (int, float)) for t in temps):
        errs.append("every attempt needs a numeric temperature")
    elif any(b <= a for a, b in zip(temps, temps[1:])):
        errs.append("retry temperatures must strictly increase")

    final_state = None
    for idx, att in enumerate(attempts):
        n = att.get("attempt")
        stages = list(att.get("stages") or [])
        names = [s.get("stage") for s in stages]
        if names != list(STAGES[: len(names)]) or not names:
            errs.append(f"attempt {n}: stages {names} are not an in-order prefix of {list(STAGES)}")
            continue
        for s in stages[:-1]:
            if not s.get("ok"):
                errs.append(f"attempt {n}: stage {s.get('stage')} failed but later stages ran")
        last = stages[-1]
        is_final = idx == len(attempts) - 1
        if last["stage"] == "judge":
            if not is_final:
                errs.append(f"attempt {n}: judge consulted on a non-final attempt")
            votes = att.get("judge_votes")
            if not isinstance(votes, list) or len(votes) < JUDGE_SAMPLES or len(votes) % 2 == 0:
                errs.append(f"attempt {n}: judge needs an odd vote count >= {JUDGE_SAMPLES}")
            elif (sum(bool(v) for v in votes) * 2 > len(votes)) != bool(last.get("ok")):
                errs.append(f"attempt {n}: judge stage disagrees with its votes")
            final_state = "endorsed" if last.get("ok") else "judge_rejected"
        else:
            if att.get("judge_votes") is not None:
                errs.append(f"attempt {n}: judge votes recorded without a passing native_run")
            if last["stage"] == "native_run" and last.get("ok"):
                errs.append(f"attempt {n}: native_run passed on gold but the judge was never consulted")
            elif last.get("ok"):
                errs.append(f"attempt {n}: attempt stopped after a passing {last['stage']}")
            final_state = "gold_failure"

    expected = {
        "endorsed": ADMITTED,
        "judge_rejected": DISCARDED_JUDGE,
        "gold_failure": DISCARDED_GOLD_FAILURE,
    }.get(final_state or "")
    if expected is not None and record.get("outcome") != expected:
        errs.append(f"outcome {record.get('outcome')!r} does not follow from the attempts (expected {expected!r})")
    if record.get("outcome") == DISCARDED_GOLD_FAILURE and len(attempts) < 2:
        errs.append("a gold failure was discarded without any retry")
    return errs
