"""REPL-EMB-2.8a: passive, hash-and-span-only capture of REPL context-search traces.

REPL-EMB-2.8 (re-check the 2.2 verdict) needs >= 75 real traces and none were persisted. This
module is a zero-stack-change tap on ``ContextBundle.search`` / ``ContextBundle.get`` and on
``peek`` / ``grep`` follows of files created by ``REPLEnvironment._spill_output``.

OFF BY DEFAULT. ``EPYC_REPL_TRACE_CAPTURE`` unset / "" / "0" => every function here returns
immediately: no file, no directory, no state. Set it to ``1`` to enable.

Records (JSONL, one per line, NO raw text):
  search: {v, kind:"search", provenance:"trace", session, ts, query_digest (sha256 of the query),
           k, section, mode, hits: [[section, start, end], ...]}      (char offsets, as returned)
  open:   {v, kind:"open", provenance:"trace", session, ts, section, start, end}
A search hit counts as OPENED when a later ``open`` in the same session overlaps its span
(the REPL reads a pointer with ``context.get(section, offset=start, max_chars=end-start)``).
Every record passes :func:`pii_scan` before it is written; a record with any string outside the
closed shapes (sha256 hex, safe identifier, ISO timestamp, fixed vocab) is DROPPED, never
redacted into something that looks real. A section name that is not a safe identifier is
replaced by ``h:<12 hex of its sha256>``.

Output: ``$EPYC_REPL_TRACE_DIR`` or ``<repo>/logs/repl_trace_capture/`` (logs/ is untracked),
file ``traces-YYYYMMDD.jsonl``. Session id = random per-bundle id (no request content).

OPERATOR STEP (one session): export EPYC_REPL_TRACE_CAPTURE=1 in the orchestrator API
environment, ``orchestrator_stack.py reload orchestrator`` (API only), then run REPL sessions
with ``context_bundle`` + ``context_search`` enabled. Load with :func:`load_real_trace_rows`.
The digest cannot be inverted: rows carry ``text=None`` and ``query_digest``; pass a
``text_resolver(digest)->str|None`` (re-supplied from the session's own prompts) to fill text,
otherwise the rows are flagged ``needs_text`` for the harness.

Failure isolation: every public entry point swallows all exceptions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

ENV_FLAG = "EPYC_REPL_TRACE_CAPTURE"
ENV_DIR = "EPYC_REPL_TRACE_DIR"
SCHEMA_VERSION = 1
STRATUM = "real_trace"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,47}$")
_HASHED_ID = re.compile(r"^h:[0-9a-f]{12}$")
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$")
_MODE = re.compile(r"^(hybrid|lexical|grep|lexical_fallback:[a-z_]{1,32})$")
_PII = re.compile(
    r"(@[A-Za-z0-9-]+\.[A-Za-z]{2,}|\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b|\b(?:sk|ghp|hf|xox[a-z])[-_][A-Za-z0-9]{8,}"
    r"|\b\d{13,19}\b|\b\d{1,3}(?:\.\d{1,3}){3}\b|-----BEGIN)"
)
_KEYS = {
    "search": {"v", "kind", "provenance", "session", "ts", "query_digest", "k", "section", "mode", "hits"},
    "open": {"v", "kind", "provenance", "session", "ts", "section", "start", "end"},
}


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip() not in ("", "0", "false", "False")


def trace_dir() -> Path:
    env = os.environ.get(ENV_DIR, "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "logs" / "repl_trace_capture"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _safe_section(name: Any) -> str | None:
    if name is None:
        return None
    s = str(name)
    if _SAFE_ID.match(s) and not _PII.search(s):
        return s
    return "h:" + digest(s)[:12]


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 10**12


def _check_str(value: Any, pattern: re.Pattern, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return isinstance(value, str) and bool(pattern.match(value)) and not _PII.search(value)


def pii_scan(rec: Mapping[str, Any]) -> list[str]:
    """Problems that make a record unwritable (empty list = clean). Closed-shape check."""
    kind = rec.get("kind")
    if kind not in _KEYS:
        return ["unknown kind"]
    problems: list[str] = []
    if set(rec) != _KEYS[kind]:
        problems.append("unexpected or missing keys")
    if rec.get("v") != SCHEMA_VERSION or rec.get("provenance") != "trace":
        problems.append("bad v/provenance")
    if not _check_str(rec.get("session"), re.compile(r"^[0-9a-f]{16}$")):
        problems.append("session shape")
    if not _check_str(rec.get("ts"), _TS):
        problems.append("ts shape")
    sec = rec.get("section")
    if not (_check_str(sec, _SAFE_ID, True) or _check_str(sec, _HASHED_ID)):
        problems.append("section shape")
    if kind == "search":
        if not _check_str(rec.get("query_digest"), _HEX64):
            problems.append("query_digest shape")
        if not _is_int(rec.get("k")):
            problems.append("k")
        if not _check_str(rec.get("mode"), _MODE):
            problems.append("mode shape")
        hits = rec.get("hits")
        if not isinstance(hits, list) or len(hits) > 64:
            problems.append("hits")
        else:
            for h in hits:
                if not (isinstance(h, list) and len(h) == 3 and _is_int(h[1]) and _is_int(h[2])
                        and (_check_str(h[0], _SAFE_ID) or _check_str(h[0], _HASHED_ID))):
                    problems.append("hit shape")
                    break
    else:
        if not (_is_int(rec.get("start")) and _is_int(rec.get("end"))):
            problems.append("span")
    return problems


def _session_id(bundle: Any) -> str:
    sid = getattr(bundle, "_trace_capture_session", None)
    if sid is None:
        sid = uuid.uuid4().hex[:16]
        try:
            bundle._trace_capture_session = sid
        except Exception:
            pass
    return sid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(rec: dict) -> bool:
    if pii_scan(rec):
        return False
    d = trace_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"traces-{time.strftime('%Y%m%d', time.gmtime())}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    return True


def record_search(bundle: Any, query: str, k: int, section: str | None, mode: str, hits: Iterable[Mapping]) -> None:
    """Tap for ``ContextBundle.search``. Never raises; no-op when the flag is off."""
    try:
        if not enabled():
            return
        _write({
            "v": SCHEMA_VERSION, "kind": "search", "provenance": "trace",
            "session": _session_id(bundle), "ts": _now(),
            "query_digest": digest(query), "k": int(k), "section": _safe_section(section),
            "mode": str(mode),
            "hits": [[_safe_section(h["section"]), int(h["start"]), int(h["end"])] for h in hits],
        })
    except Exception:
        return


def record_open(bundle: Any, section: str, start: int, end: int) -> None:
    """Tap for a span read (``ContextBundle.get`` of a whole section page). Never raises."""
    try:
        if not enabled():
            return
        _write({
            "v": SCHEMA_VERSION, "kind": "open", "provenance": "trace",
            "session": _session_id(bundle), "ts": _now(),
            "section": _safe_section(section), "start": int(start), "end": int(end),
        })
    except Exception:
        return


def record_spill_search(env: Any, path: str, query: str, k: int, hits: Iterable[Mapping]) -> None:
    """Capture a grep query over a spill file using the same hash-and-span schema."""
    record_search(env, query, k, path, "grep", hits)


def load_records(path_or_dir: str | Path | None = None) -> list[dict]:
    """All well-formed records (malformed / PII-failing lines are skipped)."""
    p = Path(path_or_dir) if path_or_dir else trace_dir()
    files = sorted(p.glob("traces-*.jsonl")) if p.is_dir() else [p]
    out = []
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and not pii_scan(rec):
                out.append(rec)
    return out


def load_real_trace_rows(
    path_or_dir: str | Path | None = None,
    text_resolver: Callable[[str], str | None] | None = None,
    bundle_resolver: Callable[[str], str | None] | None = None,
) -> list[dict]:
    """Records -> harness query rows with ``stratum='real_trace'`` (one per distinct
    (session, query_digest) search that returned hits). ``gold`` = the opened hit spans
    ({path, ranges, unit:'chars'}); rows with no opened hit are kept with ``opened=False`` and
    ``gold=None`` (not-opened is a recorded outcome, not a drop)."""
    recs = load_records(path_or_dir)
    opens: dict[str, list[tuple[str, int, int]]] = {}
    for r in recs:
        if r["kind"] == "open":
            opens.setdefault(r["session"], []).append((r["section"], r["start"], r["end"]))
    rows, seen = [], set()
    for r in recs:
        if r["kind"] != "search" or not r["hits"]:
            continue
        key = (r["session"], r["query_digest"])
        if key in seen:
            continue
        seen.add(key)
        sess_opens = opens.get(r["session"], [])
        opened = [h for h in r["hits"]
                  if any(o[0] == h[0] and o[1] < h[2] and h[1] < o[2] for o in sess_opens)]
        text = text_resolver(r["query_digest"]) if text_resolver else None
        bundle = bundle_resolver(r["session"]) if bundle_resolver else None
        gold = None
        if opened:
            path = opened[0][0]
            gold = {"path": path, "unit": "chars",
                    "ranges": [[h[1], h[2]] for h in opened if h[0] == path]}
        rows.append({
            "id": f"real_trace-{r['session']}-{r['query_digest'][:12]}",
            "stratum": STRATUM, "provenance": "trace",
            "text": text, "needs_text": text is None, "query_digest": r["query_digest"],
            "code_intent": None, "gold": gold, "opened": bool(opened),
            "bundle": bundle, "session": r["session"], "ts": r["ts"], "mode": r["mode"],
            "hits": r["hits"],
        })
    return rows


def write_real_trace_queries(
    output: str | Path,
    path_or_dir: str | Path | None = None,
    *,
    text_resolver: Callable[[str], str | None],
    bundle_resolver: Callable[[str], str | None],
) -> int:
    """Write rows accepted by ``repl_emb_eval.py freeze --trace-queries``.

    Query text and the corresponding bundle id are supplied by the evaluator; neither is
    reconstructed from the hash-only trace. Rows with unresolved query text or bundle ids are
    refused so they cannot silently enter the frozen query set as unusable evidence.
    """
    rows = load_real_trace_rows(path_or_dir, text_resolver, bundle_resolver)
    unresolved = [r["id"] for r in rows if not r["text"] or not r["bundle"]]
    if unresolved:
        raise ValueError(f"real-trace rows need text and bundle resolution: {unresolved[:5]}")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return len(rows)
