"""HIL-15: observed dependency logging at the tool-call layer.

For each tool call the harness sees, record the resource ids it READS (file path, url, artifact id,
retrieved document id) and WRITES, and for each read the id of the step that last wrote that
resource. Grade: ``observed`` -- the harness saw the call; this is NOT an intervention-validated
edge (EDGE/GRADE distinction, intake-2329#record). Default OFF behind the ``observed_dependency_logging``
feature flag; with the flag off nothing here is constructed or called, so behaviour is
byte-identical.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

GRADE_OBSERVED = "observed"
CATEGORY = "tool_dependency"

_PATH_KEYS = ("path", "file_path", "filepath", "filename", "file")
_URL_KEYS = ("url", "uri")
_ID_KEYS = {"artifact_id": "artifact", "doc_id": "doc", "document_id": "doc"}
_WRITE_NAME = re.compile(r"write|edit|patch|create|append|replace|insert|delete|save|log_append|str_replace", re.I)
_DOC_RESULT_TOOLS = re.compile(r"search|retriev|rag|lookup|query", re.I)
_MAX_RESULT_IDS = 20
# These are REPL wrappers, not registry calls. Their inner invocations are recorded
# individually at the native registry boundary in _registry_invoke_observed.
_COMPOSITE_BUILTINS = frozenset({"batch_web_search", "search_and_verify"})


def _resources_from_args(args: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k in _PATH_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v:
            out.append("file:" + os.path.normpath(v))
    for k in _URL_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v:
            out.append("url:" + v)
    for k, kind in _ID_KEYS.items():
        v = args.get(k)
        if isinstance(v, (str, int)) and str(v):
            out.append(f"{kind}:{v}")
    return out


def _doc_ids_from_result(result: Any) -> list[str]:
    ids: list[str] = []

    def take(d: Any) -> None:
        if isinstance(d, dict):
            for k in ("doc_id", "document_id"):
                if d.get(k) not in (None, ""):
                    ids.append(f"doc:{d[k]}")

    if isinstance(result, dict):
        take(result)
        for v in result.values():
            if isinstance(v, list):
                for it in v[:_MAX_RESULT_IDS]:
                    take(it)
    elif isinstance(result, list):
        for it in result[:_MAX_RESULT_IDS]:
            take(it)
    return list(dict.fromkeys(ids))[:_MAX_RESULT_IDS]


def classify(name: str, args: dict[str, Any] | None, result: Any = None) -> tuple[list[str], list[str]]:
    """(reads, writes) resource ids for one call. Argument-named resources are reads unless the
    tool name marks it as a mutation; ids found in a retrieval tool's result are reads."""
    # Do not turn an aggregate wrapper result into a purported native dependency.
    # The actual registry calls are observed separately at their call boundary.
    if name in _COMPOSITE_BUILTINS:
        return [], []
    args = args or {}
    res = _resources_from_args(args)
    reads: list[str] = []
    writes: list[str] = []
    if _WRITE_NAME.search(name or ""):
        writes = res
    else:
        reads = res
    if _DOC_RESULT_TOOLS.search(name or ""):
        reads = list(dict.fromkeys(reads + _doc_ids_from_result(result)))
    return reads, writes


@dataclass
class DependencyLog:
    """Per-run (request-scoped) tracker. ``sink`` receives one dict per call that touched resources."""

    run_id: str
    sink: Callable[[dict], None] | None = None
    ordinal: int = 0
    last_writer: dict[str, str] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)

    def observe(self, name: str, args: dict[str, Any] | None, result: Any = None, ok: bool = True) -> dict | None:
        self.ordinal += 1
        step_id = f"{self.run_id}:{self.ordinal}"
        reads, writes = classify(name, args, result)
        if not reads and not writes:
            return None
        rec = {
            "grade": GRADE_OBSERVED, "run_id": self.run_id, "step_id": step_id, "tool": name, "ok": bool(ok),
            "reads": [{"resource": r, "last_writer": self.last_writer.get(r)} for r in reads],
            "writes": list(writes),
        }
        if ok:   # a failed write did not change the resource
            for w in writes:
                self.last_writer[w] = step_id
        self.records.append(rec)
        if self.sink is not None:
            try:
                self.sink(rec)
            except Exception:   # telemetry must never fail a tool call
                pass
        return rec


def trace_store_sink(db_path: Any = None) -> Callable[[dict], None]:
    """Sink that writes one ``tool_dependency`` event per record to the unified trace store."""
    import json

    from src.trace.emit import emit, synthetic_source_path
    from src.trace.store import DEFAULT_DB_PATH, Event, EventSource

    def _sink(rec: dict) -> None:
        emit(Event(
            ts_utc="", source=EventSource.REVIEW_PLANE,
            source_path=synthetic_source_path("tool_dependency", rec["step_id"]),
            session_id=rec["run_id"], category=CATEGORY, status="ok" if rec["ok"] else "error",
            summary=f"{rec['tool']} reads={len(rec['reads'])} writes={len(rec['writes'])}",
            detail_json=json.dumps(rec, sort_keys=True),
        ), db_path=db_path or DEFAULT_DB_PATH)

    return _sink
