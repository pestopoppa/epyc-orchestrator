"""UTM-V1 / V2 / V3-structural: typed memory-operation envelope, state-transition engine,
deterministic structural verifier and a SHADOW-ONLY scoped write proposer.

Scope and honesty
-----------------
* Nothing here writes ``MemoryActionStore`` or any live store. ``MemoryEngine`` keeps its own
  append-only transition log (optionally mirrored to a JSONL path the caller names) over
  immutable source-event ids that a caller-supplied ``source_exists`` resolver checks. No live
  path imports this module; adoption is gated by M-12k/M-12 (memory-on >= memory-off).
* The structural verifier is deterministic host code (schema, scope, provenance, version,
  tombstone, rollback, source existence, transition legality). It is the STRUCTURAL half of
  UTM-V3 only. The semantic judge, human-labelled calibration and the VerMem comparison are
  not built here, and ``calibration_report`` only tabulates verdicts against labels someone
  else supplies; it fabricates none.
* Raw traces are never overwritten: UPDATE appends a version, soft DELETE appends a tombstone
  version, rollback moves the head pointer back and records the op ``rolled_back``. A
  tombstone is NOT erasure: physical privacy erasure is a separate explicit item state
  (``erasure_requested`` -> ``erased`` with a receipt) and the engine never claims ``erased``
  from a tombstone or on its own.
* The proposer (V2) is not an authoritative writer: it returns ``proposed`` operations only,
  records actor versus subject, and is deterministic by default (an injected ``decide_fn`` may
  replace the 4-way mutation decision).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

OPS = ("ADD", "UPDATE", "DELETE", "RETRIEVE", "FILTER", "SELECT_EPISODE", "SUMMARIZE", "NOOP")
MUTATING = frozenset({"ADD", "UPDATE", "DELETE"})
STATES = ("proposed", "validated", "committed", "rejected", "rolled_back")
TRANSITIONS = {
    "proposed": frozenset({"validated", "rejected"}),
    "validated": frozenset({"committed", "rejected"}),
    "committed": frozenset({"rolled_back"}),
    "rejected": frozenset(),
    "rolled_back": frozenset(),
}
SCOPES = ("user", "agent", "run")
DELETE_MODES = ("tombstone", "erase")
TOMBSTONE_HASH = hashlib.sha256(b"\x00tombstone").hexdigest()
ENVELOPE_VERSION = 1


def value_hash(value: str | None) -> str:
    return TOMBSTONE_HASH if value is None else hashlib.sha256(value.encode("utf-8")).hexdigest()


class MemoryOperationError(ValueError):
    pass


@dataclass(frozen=True)
class MemoryOperation:
    op_id: str
    session_id: str
    actor_id: str
    op: str
    subject_id: str  # whose memory this is; may differ from actor_id (inference vs fact)
    key: str = ""  # memory item id; required for mutating ops
    source_event_ids: tuple[str, ...] = ()
    before_hash: str | None = None
    after_hash: str | None = None
    value: str | None = None  # ADD/UPDATE payload; None for DELETE and non-mutating
    version: int | None = None  # the NEW item version a mutating op creates
    supersedes: str | None = None  # op_id of the head version this UPDATE/DELETE replaces
    applicability: Mapping[str, Any] = field(default_factory=dict)
    proposer_version: str = ""
    verifier_version: str = ""
    state: str = "proposed"
    error: str = ""
    delete_mode: str = "tombstone"
    rollback_of: str | None = None
    envelope_version: int = ENVELOPE_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["source_event_ids"] = list(self.source_event_ids)
        d["applicability"] = dict(self.applicability)
        return d


# ----------------------------------------------------------------------------- structural verifier


def _schema_errors(op: MemoryOperation) -> list[str]:
    errs: list[str] = []
    for name in ("op_id", "session_id", "actor_id", "subject_id"):
        if not isinstance(getattr(op, name), str) or not getattr(op, name).strip():
            errs.append(f"schema:{name}_missing")
    if op.op not in OPS:
        errs.append(f"schema:unknown_op:{op.op!r}")
        return errs
    if op.state not in STATES:
        errs.append(f"schema:unknown_state:{op.state!r}")
    if op.envelope_version != ENVELOPE_VERSION:
        errs.append(f"schema:envelope_version:{op.envelope_version!r}")
    if not isinstance(op.source_event_ids, tuple) or any(
        not isinstance(s, str) or not s.strip() for s in op.source_event_ids
    ):
        errs.append("schema:source_event_ids_malformed")
    elif len(set(op.source_event_ids)) != len(op.source_event_ids):
        errs.append("schema:source_event_ids_duplicate")
    if op.op in MUTATING:
        if not op.key.strip():
            errs.append("schema:key_missing")
        if not isinstance(op.version, int) or isinstance(op.version, bool) or op.version < 1:
            errs.append("schema:version_missing")
        if op.op in ("ADD", "UPDATE"):
            if not isinstance(op.value, str) or not op.value.strip():
                errs.append("schema:value_missing")
            elif op.after_hash != value_hash(op.value):
                errs.append("schema:after_hash_mismatch")
        else:
            if op.value is not None:
                errs.append("schema:delete_carries_value")
            if op.delete_mode not in DELETE_MODES:
                errs.append(f"schema:delete_mode:{op.delete_mode!r}")
            if op.after_hash != TOMBSTONE_HASH:
                errs.append("schema:after_hash_mismatch")
    else:
        if op.version is not None or op.value is not None or op.after_hash is not None:
            errs.append("schema:nonmutating_carries_mutation_fields")
        if op.op == "SUMMARIZE" and not op.source_event_ids:
            errs.append("provenance:summary_without_sources")
    return errs


def _scope_errors(op: MemoryOperation) -> list[str]:
    app = op.applicability
    if not isinstance(app, Mapping) or app.get("scope") not in SCOPES:
        return ["scope:applicability_scope_missing_or_unknown"]
    scope = app["scope"]
    want = {"user": ("user_id", op.subject_id), "agent": ("agent_id", None), "run": ("run_id", op.session_id)}[scope]
    got = app.get(want[0])
    if not isinstance(got, str) or not got.strip():
        return [f"scope:{want[0]}_missing_for_{scope}"]
    if want[1] is not None and got != want[1]:
        return [f"scope:{want[0]}_mismatch_for_{scope}"]
    return []


@dataclass(frozen=True)
class ItemVersion:
    version: int
    value: str | None  # None = tombstone
    value_hash: str
    op_id: str
    delete_mode: str = ""


@dataclass
class Item:
    versions: list[ItemVersion] = field(default_factory=list)  # append-only, never mutated
    rolled_back: set[int] = field(default_factory=set)
    erasure: str = ""  # "" | "requested" | "erased"

    @property
    def max_version(self) -> int:
        return max((v.version for v in self.versions), default=0)

    def head(self) -> ItemVersion | None:
        live = [v for v in self.versions if v.version not in self.rolled_back]
        return live[-1] if live else None


def _item_errors(op: MemoryOperation, item: Item | None) -> list[str]:
    if op.op not in MUTATING:
        return []
    errs: list[str] = []
    item = item or Item()
    head = item.head()
    if item.erasure:
        errs.append(f"tombstone:item_erasure_{item.erasure}")
        return errs
    if op.version is not None and op.version != item.max_version + 1:
        errs.append(f"version:expected_{item.max_version + 1}_got_{op.version}")
    head_hash = None if head is None else head.value_hash
    tomb = head is not None and head.value is None
    if op.op == "ADD":
        if head is not None and not tomb:
            errs.append("version:item_exists")
        if op.before_hash is not None and op.before_hash != head_hash:
            errs.append("stale:before_hash")
        if op.supersedes is not None and (head is None or op.supersedes != head.op_id):
            errs.append("version:supersedes_not_head")
    else:
        if head is None:
            errs.append("version:no_head_to_" + op.op.lower())
        elif tomb:
            errs.append("tombstone:" + ("update_on_tombstone" if op.op == "UPDATE" else "delete_on_tombstone"))
        else:
            if op.before_hash != head_hash:
                errs.append("stale:before_hash")
            if op.supersedes != head.op_id:
                errs.append("version:supersedes_not_head")
    return errs


def structural_check(
    op: MemoryOperation,
    item: Item | None = None,
    *,
    source_exists: Callable[[str], bool] | None = None,
) -> list[str]:
    """Deterministic reasons ([] = structurally valid). Does NOT judge truth, privacy or
    manipulation: those are semantic and unassessed here."""
    errs = _schema_errors(op)
    if any(e.startswith("schema:unknown_op") for e in errs):
        return errs
    errs += _scope_errors(op)
    if op.op in MUTATING and not op.source_event_ids:
        errs.append("provenance:mutation_without_sources")
    if source_exists is not None:
        errs += [f"provenance:missing_source:{s}" for s in op.source_event_ids if not source_exists(s)]
    errs += _item_errors(op, item)
    return errs


# ----------------------------------------------------------------------------- engine


class MemoryEngine:
    """Append-only transition log + item store. Every state change is a log record; items are
    version lists. The engine never deletes a version and never overwrites a source event."""

    def __init__(
        self,
        *,
        source_exists: Callable[[str], bool] | None = None,
        log_path: Path | str | None = None,
        verifier_version: str = "structural.v1",
    ) -> None:
        self._source_exists = source_exists
        self._verifier_version = verifier_version
        self._log_path = Path(log_path) if log_path else None
        self.log: list[dict[str, Any]] = []
        self.ops: dict[str, MemoryOperation] = {}
        self.items: dict[str, Item] = {}

    # -- log
    def _record(self, op: MemoryOperation, frm: str | None, to: str, note: str = "") -> None:
        rec = {"seq": len(self.log) + 1, "op_id": op.op_id, "op": op.op, "from": frm, "to": to,
               "note": note, "verifier_version": self._verifier_version}
        self.log.append(rec)
        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")

    def _move(self, op_id: str, to: str, error: str = "") -> MemoryOperation:
        op = self.ops[op_id]
        if to not in TRANSITIONS[op.state]:
            raise MemoryOperationError(f"illegal transition {op.state} -> {to} for {op_id}")
        new = replace(op, state=to, error=error or op.error, verifier_version=self._verifier_version)
        self.ops[op_id] = new
        self._record(new, op.state, to, error)
        return new

    # -- API
    def submit(self, op: MemoryOperation) -> MemoryOperation:
        """Register a PROPOSED operation (any other incoming state is refused: callers cannot
        smuggle in a pre-validated or pre-committed op)."""
        if op.state != "proposed":
            raise MemoryOperationError("only proposed operations may be submitted")
        if op.op_id in self.ops:
            raise MemoryOperationError(f"duplicate op_id {op.op_id}")
        self.ops[op.op_id] = op
        self._record(op, None, "proposed")
        return op

    def validate(self, op_id: str) -> MemoryOperation:
        op = self.ops[op_id]
        errs = structural_check(op, self.items.get(op.key), source_exists=self._source_exists)
        if errs:
            return self._move(op_id, "rejected", "; ".join(errs))
        return self._move(op_id, "validated")

    def commit(self, op_id: str) -> MemoryOperation:
        """Re-check at commit time (stale fencing): a head that moved since validation rejects."""
        op = self.ops[op_id]
        if op.state != "validated":
            raise MemoryOperationError(f"commit requires validated, op is {op.state}")
        errs = structural_check(op, self.items.get(op.key), source_exists=self._source_exists)
        if errs:
            return self._move(op_id, "rejected", "commit_recheck: " + "; ".join(errs))
        if op.op in MUTATING:
            item = self.items.setdefault(op.key, Item())
            item.versions.append(ItemVersion(
                op.version, None if op.op == "DELETE" else op.value, op.after_hash, op.op_id,
                op.delete_mode if op.op == "DELETE" else "",
            ))
            if op.op == "DELETE" and op.delete_mode == "erase":
                item.erasure = "requested"  # a REQUEST: never `erased` without a receipt
        return self._move(op_id, "committed")

    def reject(self, op_id: str, reason: str) -> MemoryOperation:
        return self._move(op_id, "rejected", reason)

    def rollback(self, op_id: str, *, reason: str = "") -> MemoryOperation:
        """Roll back a committed mutating op that is its item's current head. Versions are kept;
        the head pointer moves back. A non-head op is rejected, not silently reordered."""
        op = self.ops[op_id]
        if op.state != "committed":
            raise MemoryOperationError(f"rollback requires committed, op is {op.state}")
        if op.op in MUTATING:
            item = self.items[op.key]
            head = item.head()
            if head is None or head.op_id != op_id:
                raise MemoryOperationError("rollback_not_head: later versions depend on this op")
            if item.erasure == "erased":
                raise MemoryOperationError("rollback_after_erasure")
            item.rolled_back.add(head.version)
            if op.delete_mode == "erase" and op.op == "DELETE":
                item.erasure = ""
        return self._move(op_id, "rolled_back", reason or "rollback")

    def confirm_erasure(self, key: str, receipt: str) -> None:
        """Mark physical erasure ONLY with an explicit receipt string from the erasing system."""
        item = self.items.get(key)
        if item is None or item.erasure != "requested" or not str(receipt).strip():
            raise MemoryOperationError("confirm_erasure needs a requested erasure and a receipt")
        item.erasure = "erased"
        self.log.append({"seq": len(self.log) + 1, "key": key, "erasure": "erased", "receipt": receipt})

    def head_value(self, key: str) -> str | None:
        item = self.items.get(key)
        head = item.head() if item else None
        return None if head is None else head.value

    def restore_candidates(self, key: str) -> list[ItemVersion]:
        """Every kept version, newest first (history is never lost)."""
        item = self.items.get(key)
        return list(reversed(item.versions)) if item else []


# ----------------------------------------------------------------------------- V2 shadow proposer

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOK.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


@dataclass(frozen=True)
class WriteContext:
    session_id: str
    actor_id: str  # who inferred/wrote it
    subject_id: str  # whose memory it is
    scope: str  # "user" | "agent" | "run"
    scope_id: str
    recent_messages: tuple[str, ...] = ()


Decide = Callable[[str, Sequence[tuple[str, str, float]]], tuple[str, str | None]]


def default_decide(candidate: str, neighbors: Sequence[tuple[str, str, float]]) -> tuple[str, str | None]:
    """Deterministic 4-way: exact duplicate -> NOOP; overlap >= 0.6 with a different value ->
    UPDATE that neighbour; otherwise ADD. DELETE is never proposed by default (a negation needs
    the semantic judge / human labels)."""
    for key, value, score in neighbors:
        if value.strip().lower() == candidate.strip().lower():
            return "NOOP", key
    if neighbors and neighbors[0][2] >= 0.6:
        return "UPDATE", neighbors[0][0]
    return "ADD", None


class ScopedWriteProposer:
    """SHADOW-ONLY: emits ``proposed`` operations from candidate facts; never commits."""

    def __init__(self, engine: MemoryEngine, *, top_k: int = 5, decide_fn: Decide = default_decide,
                 proposer_version: str = "utm-v2-shadow.v1") -> None:
        self._engine, self._k, self._decide, self._ver = engine, top_k, decide_fn, proposer_version

    def _neighbors(self, fact: str, ctx: WriteContext) -> list[tuple[str, str, float]]:
        out = []
        for key, item in self._engine.items.items():
            head = item.head()
            if head is None or head.value is None:
                continue
            first = self._engine.ops.get(item.versions[0].op_id)
            if first is None or first.applicability.get("scope") != ctx.scope:
                continue  # conflict retrieval stays inside the declared scope
            if first.subject_id != ctx.subject_id:
                continue
            out.append((key, head.value, jaccard(fact, head.value)))
        out.sort(key=lambda t: (-t[2], t[0]))
        return out[: self._k]

    def propose(self, fact: str, source_event_ids: Iterable[str], ctx: WriteContext, *,
                op_id: str, key_hint: str = "") -> MemoryOperation:
        app = {"scope": ctx.scope, {"user": "user_id", "agent": "agent_id", "run": "run_id"}[ctx.scope]: ctx.scope_id}
        neighbors = self._neighbors(fact, ctx)
        kind, target = self._decide(fact, neighbors)
        base = dict(op_id=op_id, session_id=ctx.session_id, actor_id=ctx.actor_id, subject_id=ctx.subject_id,
                    source_event_ids=tuple(source_event_ids), applicability=app, proposer_version=self._ver)
        if kind == "NOOP":
            return MemoryOperation(op="NOOP", **base)
        item = self._engine.items.get(target) if target else None
        head = item.head() if item else None
        if kind == "ADD":
            key = key_hint or f"mem-{hashlib.sha256(fact.encode()).hexdigest()[:12]}"
            existing = self._engine.items.get(key) or Item()
            return MemoryOperation(op="ADD", key=key, value=fact, after_hash=value_hash(fact),
                                   version=existing.max_version + 1, **base)
        if kind == "UPDATE" and head is not None:
            return MemoryOperation(op="UPDATE", key=target, value=fact, after_hash=value_hash(fact),
                                   before_hash=head.value_hash, supersedes=head.op_id,
                                   version=item.max_version + 1, **base)
        if kind == "DELETE" and head is not None:
            return MemoryOperation(op="DELETE", key=target, after_hash=TOMBSTONE_HASH,
                                   before_hash=head.value_hash, supersedes=head.op_id,
                                   version=item.max_version + 1, **base)
        return MemoryOperation(op="NOOP", **base)  # an undecidable proposal never becomes a write


def agreement_report(proposed: Sequence[MemoryOperation], labels: Mapping[str, str]) -> dict[str, Any]:
    """Shadow comparison against human labels (op type per op_id). Unlabelled proposals are
    counted as unlabelled, never as agreement."""
    agree = disagree = unlabelled = 0
    confusion: dict[str, int] = {}
    for p in proposed:
        lab = labels.get(p.op_id)
        if lab is None:
            unlabelled += 1
            continue
        if lab == p.op:
            agree += 1
        else:
            disagree += 1
            confusion[f"{lab}->{p.op}"] = confusion.get(f"{lab}->{p.op}", 0) + 1
    return {"agree": agree, "disagree": disagree, "unlabelled": unlabelled,
            "confusion": dict(sorted(confusion.items()))}


def calibration_report(cases: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-class false-pass / false-fail tabulation for the STRUCTURAL verifier over cases
    ``{"class": str, "label_valid": bool, "verdict_valid": bool}``. Labels are supplied by a
    human author; classes with no cases are absent (absence is not a 0% rate)."""
    out: dict[str, dict[str, Any]] = {}
    for c in cases:
        row = out.setdefault(str(c["class"]), {"n": 0, "false_pass": 0, "false_fail": 0})
        row["n"] += 1
        if c["verdict_valid"] and not c["label_valid"]:
            row["false_pass"] += 1
        if (not c["verdict_valid"]) and c["label_valid"]:
            row["false_fail"] += 1
    return out
