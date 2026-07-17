"""RD-4 — verifier-request adapter (tiered cheap-first, three-valued outcomes).

A ReviewDecision may carry ``verifier_requests[]`` — the objective checks a
reviewer wants run before it will conclude (the ``request_evidence`` payload).
This adapter maps each request onto either:

  * a **gate** (``src/gate_runner.py``: lint / typecheck / unit / format / build),
    executed cheap-first through the codified ``config/gates.yaml``; or
  * a **formalizer** — a pluggable per-domain checker implementing
    ``Formalizer.check(...) -> CheckResult`` — for scorer / retrieval / math /
    constraint checks.

Every result is THREE-VALUED (``pass|fail|inconclusive``). Verifier precedence
over the reviewer (RD-3) applies only to *conclusive* (pass/fail) outcomes:
formalization is incomplete (~15% false-positive tax — intake-843/Sistla), so a
checker that cannot decide returns ``inconclusive`` and hands control back to the
reviewer rather than fabricating a verdict. A ``fail`` MUST carry a
``certificate`` (failing assertion / counterexample / constraint violation) — that
certificate is the evidence handed back to the author.

Tiering
-------
Tier 1 (implemented, local stdlib + installed deps only): jsonschema validation,
invariant-assertion predicates, regex/constraint checks (instruction precision),
numeric-answer checking (math), retrieval-grounding span-containment.

Tier 2 (registered, NOT implemented): Hypothesis property tests, Soufflé/Datalog,
Z3. They register as named formalizers that degrade gracefully — returning
``inconclusive`` with reason ``not_installed`` (dependency absent) or
``not_implemented`` (present but wiring is future work).

The output is a schema-valid ``VerificationReport`` dict
(``orchestration/verification_report.schema.json``).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

# orchestration/ lives at the repo root: .../src/proactive_delegation/<this> -> parents[2]
_ORCH_DIR = Path(__file__).resolve().parents[2] / "orchestration"
_VERIFICATION_REPORT_SCHEMA = _ORCH_DIR / "verification_report.schema.json"

REPORT_SCHEMA_VERSION = "1.0.0"

# Gate-family request kinds route to gate_runner; the rest route to formalizers.
_GATE_KINDS = {"gate", "test", "lint", "typecheck", "format", "build"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── CheckResult + Formalizer interface ─────────────────────────────────


@dataclass
class CheckResult:
    """One formalizer's normalized output (maps 1:1 onto a report ``check``).

    ``outcome`` is three-valued. On ``fail`` a ``certificate`` is required (the
    request_evidence payload); on ``inconclusive`` an ``inconclusive_reason`` is
    required. ``instrument`` is the {name, version} attestation.
    """

    outcome: str  # "pass" | "fail" | "inconclusive"
    instrument: dict[str, str]
    kind: str = "constraint_check"
    certificate: dict[str, Any] | None = None
    inconclusive_reason: str | None = None
    output: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    required: bool = True


@runtime_checkable
class Formalizer(Protocol):
    """Pluggable per-domain objective checker.

    ``name`` is the registry key (a ``verifier_request.verifier``); ``kind`` is the
    check family (a ``verifier_request.kind``); ``version`` feeds the instrument
    attestation. ``check`` inspects a candidate (+ request params) and returns a
    three-valued :class:`CheckResult`.
    """

    name: str
    version: str
    kind: str

    def check(self, request: dict[str, Any], candidate: Any, domain: str) -> CheckResult: ...


def _instr(f: Formalizer) -> dict[str, str]:
    return {"name": f.name, "version": f.version}


def _cert(cert_type: str, payload: Any, location: str | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {"type": cert_type, "payload": payload}
    if location:
        c["location"] = location
    return c


# ── Tier 1 formalizers (implemented) ───────────────────────────────────


@dataclass
class JsonSchemaFormalizer:
    """Validate the candidate (or ``request['schema']``-named target) against a
    JSON Schema. Fail → constraint_violation certificate listing every violation.

    The schema is taken from ``request['schema']`` (inline) or, absent that,
    ``inconclusive`` (nothing to check against — never a silent pass)."""

    name: str = "jsonschema"
    version: str = "1.0.0"
    kind: str = "constraint_check"

    def check(self, request: dict[str, Any], candidate: Any, domain: str) -> CheckResult:
        schema = request.get("schema")
        if schema is None:
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason="no schema supplied in request['schema']",
            )
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason="not_installed: jsonschema",
            )
        target = request.get("target_value", candidate)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(target),
            key=lambda e: list(e.absolute_path),
        )
        if not errors:
            return CheckResult(outcome="pass", instrument=_instr(self), kind=self.kind)
        msgs = [
            f"{'$' + ''.join(f'.{p}' for p in e.absolute_path)}: {e.message}" for e in errors[:20]
        ]
        return CheckResult(
            outcome="fail",
            instrument=_instr(self),
            kind=self.kind,
            certificate=_cert("constraint_violation", msgs),
            errors=msgs,
        )


# An invariant predicate: (candidate, request, domain) -> (ok, detail).
InvariantPredicate = Callable[[Any, dict[str, Any], str], "tuple[bool, str]"]


@dataclass
class InvariantAssertionFormalizer:
    """Run a NAMED, configurable invariant predicate against the candidate.

    Predicates are registered by name; the request selects one via
    ``request['invariant']`` (falling back to ``request['verifier']``). Unknown
    predicate → inconclusive (not a fail). Failing predicate → failing_assertion
    certificate."""

    predicates: dict[str, InvariantPredicate] = field(default_factory=dict)
    name: str = "invariant"
    version: str = "1.0.0"
    kind: str = "constraint_check"

    def register(self, key: str, predicate: InvariantPredicate) -> None:
        self.predicates[key] = predicate

    def check(self, request: dict[str, Any], candidate: Any, domain: str) -> CheckResult:
        key = request.get("invariant") or request.get("verifier")
        predicate = self.predicates.get(key) if key else None
        if predicate is None:
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason=f"no invariant predicate registered for {key!r}",
            )
        try:
            ok, detail = predicate(candidate, request, domain)
        except Exception as exc:  # a predicate that raises is inconclusive, not a fail
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason=f"predicate raised: {type(exc).__name__}: {exc}",
            )
        if ok:
            return CheckResult(outcome="pass", instrument=_instr(self), kind=self.kind)
        return CheckResult(
            outcome="fail",
            instrument=_instr(self),
            kind=self.kind,
            certificate=_cert("failing_assertion", detail or f"invariant {key} violated"),
            errors=[detail] if detail else [],
        )


@dataclass
class RegexConstraintFormalizer:
    """Instruction-precision constraint checker.

    Supports the constraint kinds an instruction-following task needs, taken from
    ``request``:
      * ``must_match``: candidate text must match this regex.
      * ``must_not_match``: candidate text must NOT match this regex.
      * ``must_contain`` / ``must_not_contain``: literal substring presence.
      * ``max_words`` / ``min_words``: length bounds.
    Multiple constraints in one request are ALL required (AND). Any violation →
    constraint_violation certificate enumerating which constraints failed."""

    name: str = "regex_constraint"
    version: str = "1.0.0"
    kind: str = "constraint_check"

    def check(self, request: dict[str, Any], candidate: Any, domain: str) -> CheckResult:
        text = candidate if isinstance(candidate, str) else str(candidate)
        constraints = self._active_constraints(request)
        if not constraints:
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason="no constraint keys present in request",
            )
        violations: list[str] = []
        for kind, value in constraints:
            v = self._eval(kind, value, text)
            if v is not None:
                violations.append(v)
        if not violations:
            return CheckResult(outcome="pass", instrument=_instr(self), kind=self.kind)
        return CheckResult(
            outcome="fail",
            instrument=_instr(self),
            kind=self.kind,
            certificate=_cert("constraint_violation", violations),
            errors=violations,
        )

    @staticmethod
    def _active_constraints(request: dict[str, Any]) -> list[tuple[str, Any]]:
        keys = (
            "must_match",
            "must_not_match",
            "must_contain",
            "must_not_contain",
            "max_words",
            "min_words",
        )
        return [(k, request[k]) for k in keys if k in request]

    @staticmethod
    def _eval(kind: str, value: Any, text: str) -> str | None:
        if kind == "must_match":
            return None if re.search(str(value), text) else f"must_match /{value}/ did not match"
        if kind == "must_not_match":
            return f"must_not_match /{value}/ matched" if re.search(str(value), text) else None
        if kind == "must_contain":
            return None if str(value) in text else f"must_contain {value!r} absent"
        if kind == "must_not_contain":
            return f"must_not_contain {value!r} present" if str(value) in text else None
        if kind == "max_words":
            n = len(text.split())
            return f"max_words {value} exceeded ({n})" if n > int(value) else None
        if kind == "min_words":
            n = len(text.split())
            return f"min_words {value} not met ({n})" if n < int(value) else None
        return None


@dataclass
class NumericAnswerFormalizer:
    """Math-domain numeric-answer checker.

    Extracts the candidate's final numeric answer and compares it to
    ``request['expected']`` within ``request['tol']`` (default 1e-6, relative+abs).
    Mismatch → counterexample certificate (expected vs got). No expected value or
    no extractable number → inconclusive."""

    name: str = "numeric_answer"
    version: str = "1.0.0"
    kind: str = "math_check"

    _NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")

    def check(self, request: dict[str, Any], candidate: Any, domain: str) -> CheckResult:
        if "expected" not in request:
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason="no request['expected']",
            )
        try:
            expected = float(request["expected"])
        except (ValueError, TypeError):
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason="request['expected'] is not numeric",
            )
        got = self._extract_number(candidate)
        if got is None:
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason="no numeric answer extractable from candidate",
            )
        tol = float(request.get("tol", 1e-6))
        if abs(got - expected) <= tol + tol * abs(expected):
            return CheckResult(outcome="pass", instrument=_instr(self), kind=self.kind)
        return CheckResult(
            outcome="fail",
            instrument=_instr(self),
            kind=self.kind,
            certificate=_cert(
                "counterexample", {"expected": expected, "got": got, "tol": tol}
            ),
            errors=[f"expected {expected}, got {got}"],
        )

    def _extract_number(self, candidate: Any) -> float | None:
        text = candidate if isinstance(candidate, str) else str(candidate)
        matches = self._NUM.findall(text)
        if not matches:
            return None
        try:
            return float(matches[-1].replace(",", ""))  # last number = the final answer
        except ValueError:
            return None


@dataclass
class RetrievalGroundingFormalizer:
    """Retrieval-grounding span-containment check (Tier-1 stub, functional).

    Verifies each cited span in ``request['spans']`` (or candidate-embedded quotes)
    is literally contained in the supplied source(s) ``request['sources']`` (list of
    strings). Ungrounded spans → constraint_violation certificate. This is a
    containment stub — semantic-similarity grounding is future work; absent
    sources → inconclusive (cannot verify)."""

    name: str = "retrieval_grounding"
    version: str = "0.1.0"  # stub: span-containment only
    kind: str = "retrieval_check"

    def check(self, request: dict[str, Any], candidate: Any, domain: str) -> CheckResult:
        sources = request.get("sources")
        if not sources:
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason="no request['sources'] to ground against",
            )
        haystack = "\n".join(str(s) for s in sources)
        spans = request.get("spans")
        if not spans:
            spans = self._quoted_spans(candidate)
        if not spans:
            return CheckResult(
                outcome="inconclusive",
                instrument=_instr(self),
                kind=self.kind,
                inconclusive_reason="no cited spans to check (none supplied/extractable)",
            )
        ungrounded = [s for s in spans if str(s).strip() and str(s) not in haystack]
        if not ungrounded:
            return CheckResult(outcome="pass", instrument=_instr(self), kind=self.kind)
        return CheckResult(
            outcome="fail",
            instrument=_instr(self),
            kind=self.kind,
            certificate=_cert("constraint_violation", {"ungrounded_spans": ungrounded}),
            errors=[f"{len(ungrounded)} ungrounded span(s)"],
        )

    @staticmethod
    def _quoted_spans(candidate: Any) -> list[str]:
        text = candidate if isinstance(candidate, str) else str(candidate)
        return re.findall(r'"([^"]{4,})"', text)


# ── Tier 2 formalizers (registered, not implemented → graceful inconclusive) ──


@dataclass
class _Tier2Stub:
    """A Tier-2 formalizer that is registered but not wired.

    ``check`` returns ``inconclusive`` with reason ``not_installed`` when the
    backing dependency/binary is absent, else ``not_implemented`` (present but the
    formalization is future work). Never fabricates pass/fail."""

    name: str
    kind: str
    probe: Callable[[], bool]  # True = dependency present
    version: str = "0.0.0"

    def check(self, request: dict[str, Any], candidate: Any, domain: str) -> CheckResult:
        present = False
        try:
            present = bool(self.probe())
        except Exception:
            present = False
        reason = "not_implemented" if present else "not_installed"
        return CheckResult(
            outcome="inconclusive",
            instrument=_instr(self),
            kind=self.kind,
            inconclusive_reason=f"{reason}: {self.name}",
        )


def _import_present(module: str) -> Callable[[], bool]:
    def _probe() -> bool:
        import importlib.util

        return importlib.util.find_spec(module) is not None

    return _probe


def _binary_present(binary: str) -> Callable[[], bool]:
    def _probe() -> bool:
        import shutil

        return shutil.which(binary) is not None

    return _probe


def hypothesis_formalizer() -> _Tier2Stub:
    return _Tier2Stub(name="hypothesis", kind="test", probe=_import_present("hypothesis"))


def souffle_formalizer() -> _Tier2Stub:
    return _Tier2Stub(name="souffle", kind="constraint_check", probe=_binary_present("souffle"))


def z3_formalizer() -> _Tier2Stub:
    return _Tier2Stub(name="z3", kind="constraint_check", probe=_import_present("z3"))


# ── registry ───────────────────────────────────────────────────────────


class FormalizerRegistry:
    """Name- and kind-indexed registry of formalizers (cheap-first ordering).

    ``for_request`` resolves a request to a formalizer: exact ``verifier`` name
    first, then the first formalizer registered for the request ``kind``."""

    def __init__(self) -> None:
        self._by_name: dict[str, Formalizer] = {}
        self._by_kind: dict[str, list[str]] = {}

    def register(self, formalizer: Formalizer) -> None:
        self._by_name[formalizer.name] = formalizer
        self._by_kind.setdefault(formalizer.kind, []).append(formalizer.name)

    def get(self, name: str) -> Formalizer | None:
        return self._by_name.get(name)

    def by_kind(self, kind: str) -> list[Formalizer]:
        return [self._by_name[n] for n in self._by_kind.get(kind, [])]

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def for_request(self, request: dict[str, Any]) -> Formalizer | None:
        verifier = request.get("verifier")
        if verifier and verifier in self._by_name:
            return self._by_name[verifier]
        kind = request.get("kind")
        by_kind = self._by_kind.get(kind or "", [])
        return self._by_name[by_kind[0]] if by_kind else None


def default_registry(
    invariant_predicates: dict[str, InvariantPredicate] | None = None,
) -> FormalizerRegistry:
    """Registry seeded with all Tier-1 formalizers + Tier-2 graceful stubs.

    Tier 1 (implemented): jsonschema, invariant, regex_constraint, numeric_answer,
    retrieval_grounding. Tier 2 (inconclusive stubs): hypothesis, souffle, z3.
    """
    reg = FormalizerRegistry()
    reg.register(JsonSchemaFormalizer())
    reg.register(InvariantAssertionFormalizer(predicates=dict(invariant_predicates or {})))
    reg.register(RegexConstraintFormalizer())
    reg.register(NumericAnswerFormalizer())
    reg.register(RetrievalGroundingFormalizer())
    reg.register(hypothesis_formalizer())
    reg.register(souffle_formalizer())
    reg.register(z3_formalizer())
    return reg


# ── gate_runner bridge ─────────────────────────────────────────────────


def _gate_result_to_check(gate_name: str, result: Any) -> dict[str, Any]:
    """Normalize a gate_runner.GateResult into a verification-report ``check``."""
    instrument = {"name": "gate_runner", "version": "1.0.0"}
    errors = list(getattr(result, "errors", []) or [])
    output = getattr(result, "output", "") or ""
    exit_code = getattr(result, "exit_code", 0)
    elapsed = float(getattr(result, "elapsed_seconds", 0.0) or 0.0)
    required = bool(getattr(result, "required", True))

    check: dict[str, Any] = {
        "check_id": gate_name,
        "kind": "gate",
        "instrument": instrument,
        "required": required,
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "output": output[:4000],
        "errors": errors[:20],
        "warnings": list(getattr(result, "warnings", []) or [])[:20],
    }

    # Unknown gate / execution error / timeout can't conclude -> inconclusive.
    unknown = any(str(e).lower().startswith("unknown gate") for e in errors)
    ran = exit_code not in (-1,) and not unknown
    if unknown:
        check["outcome"] = "inconclusive"
        check["inconclusive_reason"] = f"unknown gate: {gate_name}"
        return check
    if not ran:
        check["outcome"] = "inconclusive"
        check["inconclusive_reason"] = output[:200] or "gate could not execute (exit -1)"
        return check

    if getattr(result, "passed", False):
        check["outcome"] = "pass"
    else:
        check["outcome"] = "fail"
        payload = errors or [output[:1000]]
        check["certificate"] = {"type": "failing_assertion", "payload": payload}
    return check


def _check_result_to_check(check_id: str, kind: str, cr: CheckResult) -> dict[str, Any]:
    """Normalize a formalizer CheckResult into a verification-report ``check``."""
    check: dict[str, Any] = {
        "check_id": check_id,
        "kind": cr.kind or kind,
        "outcome": cr.outcome,
        "instrument": dict(cr.instrument),
        "required": cr.required,
        "elapsed_seconds": cr.elapsed_seconds,
    }
    if cr.output:
        check["output"] = cr.output[:4000]
    if cr.errors:
        check["errors"] = cr.errors[:20]
    if cr.warnings:
        check["warnings"] = cr.warnings[:20]
    if cr.outcome == "fail":
        check["certificate"] = cr.certificate or {
            "type": "failing_assertion",
            "payload": cr.errors or "check failed without detail",
        }
    elif cr.outcome == "inconclusive":
        check["inconclusive_reason"] = cr.inconclusive_reason or "unspecified"
    return check


# ── the adapter entrypoint ─────────────────────────────────────────────


def run_verifier_requests(
    requests: Iterable[dict[str, Any]],
    candidate: Any,
    domain: str,
    *,
    registry: FormalizerRegistry | None = None,
    gate_runner: Any | None = None,
    candidate_ref: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Run ``verifier_requests[]`` and return a schema-valid VerificationReport dict.

    Gate-family requests (kind in gate/test/lint/typecheck/format/build) route to
    ``gate_runner`` (a ``src.gate_runner.GateRunner``; lazily constructed from the
    codified ``config/gates.yaml`` if not supplied); every other kind routes to a
    formalizer from ``registry`` (default: all Tier-1 + Tier-2 stubs).

    Aggregate ``conclusive_verdict`` (over *required* checks): a conclusive FAIL is
    decisive (``fail``); else any inconclusive → ``inconclusive``; else ``pass``.
    The per-check outcomes are what RD-3 precedence actually consumes — the
    aggregate is a convenience rollup.
    """
    registry = registry if registry is not None else default_registry()
    requests = list(requests)

    checks: list[dict[str, Any]] = []
    for req in requests:
        kind = req.get("kind", "")
        if kind in _GATE_KINDS:
            checks.append(_run_gate_request(req, gate_runner))
        else:
            checks.append(_run_formalizer_request(req, candidate, domain, registry))

    if not checks:
        # schema requires >=1 check; record an explicit no-op inconclusive.
        checks.append(
            {
                "check_id": "no_requests",
                "kind": "constraint_check",
                "outcome": "inconclusive",
                "instrument": {"name": "verifier_adapter", "version": "1.0.0"},
                # required so the rollup reflects "no objective signal" as
                # inconclusive (never a fabricated pass on an empty request set).
                "required": True,
                "inconclusive_reason": "no verifier_requests supplied",
            }
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": str(uuid.uuid4()),
        "created_at": _now_utc(),
        "summary": _summarize(checks),
        "checks": checks,
        "provenance": {"runner": "verifier_adapter"},
    }
    if candidate_ref:
        report["candidate_ref"] = candidate_ref

    if validate:
        _validate_report(report)
    return report


# gate_runner is constructed once per adapter call and reused across gate requests.
def _run_gate_request(req: dict[str, Any], gate_runner: Any | None) -> dict[str, Any]:
    verifier = req.get("verifier", "")
    runner = gate_runner if gate_runner is not None else _default_gate_runner()
    if runner is None:
        return {
            "check_id": verifier or "gate",
            "kind": "gate",
            "outcome": "inconclusive",
            "instrument": {"name": "gate_runner", "version": "1.0.0"},
            "required": True,
            "inconclusive_reason": "gate_runner unavailable",
        }
    results = runner.run_gates_by_name([verifier])
    result = results[0] if results else None
    if result is None:
        return {
            "check_id": verifier or "gate",
            "kind": "gate",
            "outcome": "inconclusive",
            "instrument": {"name": "gate_runner", "version": "1.0.0"},
            "required": True,
            "inconclusive_reason": "gate produced no result",
        }
    return _gate_result_to_check(verifier, result)


def _run_formalizer_request(
    req: dict[str, Any], candidate: Any, domain: str, registry: FormalizerRegistry
) -> dict[str, Any]:
    verifier = req.get("verifier", "")
    kind = req.get("kind", "constraint_check")
    formalizer = registry.for_request(req)
    if formalizer is None:
        return {
            "check_id": verifier or kind,
            "kind": kind,
            "outcome": "inconclusive",
            "instrument": {"name": "verifier_adapter", "version": "1.0.0"},
            "required": True,
            "inconclusive_reason": f"no formalizer for verifier={verifier!r} kind={kind!r}",
        }
    cr = formalizer.check(req, candidate, domain)
    return _check_result_to_check(verifier or formalizer.name, kind, cr)


def _default_gate_runner() -> Any | None:
    try:
        from src.gate_runner import GateRunner

        return GateRunner()
    except Exception:
        return None


def _summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for c in checks if c["outcome"] == "pass")
    failed = sum(1 for c in checks if c["outcome"] == "fail")
    inconclusive = sum(1 for c in checks if c["outcome"] == "inconclusive")
    required = [c for c in checks if c.get("required", True)]
    if any(c["outcome"] == "fail" for c in required):
        verdict = "fail"
    elif any(c["outcome"] == "inconclusive" for c in required):
        verdict = "inconclusive"
    else:
        verdict = "pass"
    return {
        "passed": passed,
        "failed": failed,
        "inconclusive": inconclusive,
        "conclusive_verdict": verdict,
    }


def _validate_report(report: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    schema = _load_report_schema()
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report), key=lambda e: list(e.absolute_path)
    )
    if errors:
        msgs = [
            f"{'$' + ''.join(f'.{p}' for p in e.absolute_path)}: {e.message}" for e in errors[:10]
        ]
        raise ValueError("VerificationReport failed schema validation: " + "; ".join(msgs))


def _load_report_schema() -> dict[str, Any]:
    import json

    return json.loads(_VERIFICATION_REPORT_SCHEMA.read_text(encoding="utf-8"))
