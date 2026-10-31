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
v1 ContextBundle rows have no corpus identity and remain diagnostic only. Prospective v2
spill search/open records add only content_digest (SHA-256) and content_chars. No raw text.
A v2 search hit counts as OPENED when a later same-content ``open`` in the same session overlaps its span
(the REPL reads a pointer with ``context.get(section, offset=start, max_chars=end-start)``).
Every record passes :func:`pii_scan` before it is written; a record with any string outside the
closed shapes (sha256 hex, safe identifier, ISO timestamp, fixed vocab) is DROPPED, never
redacted into something that looks real. A section name that is not a safe identifier is
replaced by ``h:<12 hex of its sha256>``.

Output: ``$EPYC_REPL_TRACE_DIR`` or ``<repo>/logs/repl_trace_capture/`` (logs/ is untracked),
file ``traces-YYYYMMDD.jsonl``. Session id = random per-bundle id (no request content).

LIVE ACTIVATION: the session owning the API enables EPYC_REPL_TRACE_CAPTURE=1 and reloads
only the orchestrator at its own boundary. Ordinary spill -> grep -> peek use then writes v2.
Capture JSONL never stores query/corpus text. Separately retain authorized exact originals;
text_resolver verifies the query SHA and bundle_resolver supplies
{bundle: id, sections: {original_path: exact_text}} for corpus SHA verification. Old v1 records
cannot be retroactively given corpus custody. Exported gold uses the harness's inclusive line
units, and freeze rechecks the actual bundle text against the native content digest.

Failure isolation: runtime capture entry points swallow exceptions; offline resolvers fail closed.
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
SCHEMA_VERSION = 1  # Existing ContextBundle records retain their closed v1 schema.
SPILL_SCHEMA_VERSION = 2
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
    version = rec.get("v")
    expected = _KEYS[kind] | ({"content_digest", "content_chars"} if version == SPILL_SCHEMA_VERSION else set())
    if set(rec) != expected:
        problems.append("unexpected or missing keys")
    if version not in (SCHEMA_VERSION, SPILL_SCHEMA_VERSION) or rec.get("provenance") != "trace":
        problems.append("bad v/provenance")
    if not _check_str(rec.get("session"), re.compile(r"^[0-9a-f]{16}$")):
        problems.append("session shape")
    if not _check_str(rec.get("ts"), _TS):
        problems.append("ts shape")
    if version == SPILL_SCHEMA_VERSION:
        if not _check_str(rec.get("content_digest"), _HEX64) or not _is_int(rec.get("content_chars")):
            problems.append("content identity")
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
        hit_limit = max(64, rec["k"]) if version == SPILL_SCHEMA_VERSION and _is_int(rec.get("k")) else 64
        if not isinstance(hits, list) or len(hits) > hit_limit:
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
    if version == SPILL_SCHEMA_VERSION and not problems:
        if sec is None:
            problems.append("spill section missing")
        if kind == "search":
            if rec["mode"] != "grep" or any(h[0] != sec or not 0 <= h[1] <= h[2] <= rec["content_chars"] for h in rec["hits"]):
                problems.append("spill hit identity/span")
        elif not 0 <= rec["start"] <= rec["end"] <= rec["content_chars"]:
            problems.append("spill open span")
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


def register_spill(env: Any, path: str, text: str) -> None:
    """Prospective write-side identity. No state is allocated while disabled."""
    try:
        if not enabled():
            return
        registry = getattr(env, "_repl_trace_spills", None)
        if registry is None:
            registry = env._repl_trace_spills = {}
        registry[os.path.abspath(path)] = digest(text)
    except Exception:
        return


def is_registered_spill(env: Any, path: str) -> bool:
    if not enabled():
        return False
    registry = getattr(env, "_repl_trace_spills", None)
    return registry is not None and os.path.abspath(path) in registry


def _spill_record(env: Any, path: str, text: str, kind: str, **fields: Any) -> None:
    try:
        if not is_registered_spill(env, path):
            return
        content_digest = digest(text)
        if env._repl_trace_spills[os.path.abspath(path)] != content_digest:
            return  # The path was overwritten outside this environment's spill writer.
        _write({
            "v": SPILL_SCHEMA_VERSION, "kind": kind, "provenance": "trace",
            "session": _session_id(env), "ts": _now(), "section": _safe_section(os.path.abspath(path)),
            "content_digest": content_digest, "content_chars": len(text), **fields,
        })
    except Exception:
        return


def record_spill_search(env: Any, path: str, text: str, query: str, k: int, hits: Iterable[Mapping]) -> None:
    """v2 binds hit offsets and query hash to the original registered spill text."""
    try:
        if not enabled():
            return
        _spill_record(env, path, text, "search", query_digest=digest(query), k=int(k), mode="grep",
                      hits=[[_safe_section(os.path.abspath(path)), int(h["start"]), int(h["end"])] for h in hits])
    except Exception:
        return


def record_spill_open(env: Any, path: str, text: str, start: int, end: int) -> None:
    """v2 open from the same read that produced the returned page, including negative offsets."""
    _spill_record(env, path, text, "open", start=start, end=end)


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
    bundle_resolver: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> list[dict]:
    """Load diagnostic rows, never inventing gold for custody-unknown v1 records.

    A resolver supplies {bundle: id, sections: {original_section_name: exact_text}}.
    Query SHA and v2 source SHA are checked before line-unit gold is constructed. Only
    later opens of the SAME content version count; a v1 overlap cannot prove custody.
    This is preparation, not the >=75-real-trace 2.8 gate or measurement warrant.
    """
    recs = load_records(path_or_dir)
    rows, seen = [], set()
    for index, r in enumerate(recs):
        if r["kind"] != "search" or not r["hits"]:
            continue
        identity = (r["session"], r["query_digest"], r.get("content_digest"), r["section"])
        if identity in seen:
            continue
        seen.add(identity)
        versioned = r["v"] == SPILL_SCHEMA_VERSION
        later_opens = [o for o in recs[index + 1:]
                       if o["kind"] == "open" and o["session"] == r["session"]
                       and versioned and o["v"] == SPILL_SCHEMA_VERSION
                       and o["content_digest"] == r["content_digest"]
                       and o["content_chars"] == r["content_chars"]]
        opened = [h for h in r["hits"] if h[0] == r["section"] and any(
            o["section"] == h[0] and o["start"] < h[2] and h[1] < o["end"] for o in later_opens)]
        text = text_resolver(r["query_digest"]) if text_resolver else None
        if text is not None and (not isinstance(text, str) or digest(text) != r["query_digest"]):
            raise ValueError("query resolver digest mismatch")
        resolved = bundle_resolver(r["session"]) if bundle_resolver else None
        bundle, source_path, source_text = None, None, None
        if resolved is not None:
            if not isinstance(resolved, Mapping) or not isinstance(resolved.get("sections"), Mapping):
                raise ValueError("bundle resolver must supply bundle id and original section text")
            bundle = resolved.get("bundle")
            candidates = [(path, body) for path, body in resolved["sections"].items()
                          if isinstance(path, str) and _safe_section(path) == r["section"]]
            if versioned:
                if len(candidates) != 1:
                    raise ValueError("missing or ambiguous original corpus section")
                source_path, source_text = candidates[0]
                if (not isinstance(source_text, str) or digest(source_text) != r["content_digest"]
                        or len(source_text) != r["content_chars"]):
                    raise ValueError("corpus resolver digest mismatch")
        gold = None
        if versioned and source_text is not None and opened:
            ranges = []
            for _, start, end in opened:
                if not 0 <= start < end <= len(source_text):
                    raise ValueError("hit outside original corpus")
                # Harness chunk pointers use 1-based inclusive lines, never char offsets.
                ranges.append([source_text.count("\n", 0, start) + 1,
                               source_text.count("\n", 0, end - 1) + 1])
            gold = {"path": source_path, "unit": "lines", "ranges": ranges}
        row = {
            "id": f"real_trace-{r['session']}-{r['query_digest'][:12]}-{digest(str(identity))[:12]}",
            "stratum": STRATUM, "provenance": "trace",
            "text": text, "needs_text": text is None, "query_digest": r["query_digest"],
            "code_intent": None, "gold": gold, "opened": bool(opened) if versioned else None,
            "bundle": bundle, "session": r["session"], "ts": r["ts"], "mode": r["mode"],
            "hits": r["hits"], "custody": "verified" if source_text is not None else "unknown",
            "custody_reason": ("verified_original_content_digest" if source_text is not None else
                               "original_corpus_not_supplied" if versioned else "legacy_v1_missing_content_digest"),
        }
        if versioned:
            row.update(content_digest=r["content_digest"], content_chars=r["content_chars"],
                       source_section=source_path)
        rows.append(row)
    return rows


def write_real_trace_queries(
    output: str | Path,
    path_or_dir: str | Path | None = None,
    *,
    text_resolver: Callable[[str], str | None],
    bundle_resolver: Callable[[str], Mapping[str, Any] | None],
) -> int:
    """Export only hash-verified v2 rows for the harness's real_trace stratum.

    Source texts stay outside capture JSONL. The separately authorized resolver must provide
    exact originals, not a guessed bundle label; legacy v1 custody is unknown and refused.
    Exported source_section/content_digest are rechecked against actual bundles at freeze.
    Export is all-or-nothing: mixed v1/v2 logs refuse if any row has unknown custody. Callers
    must explicitly scope the input file to a qualified v2 spill session, retaining the native
    source log and selection provenance; the exporter never silently drops unknown rows.
    """
    rows = load_real_trace_rows(path_or_dir, text_resolver, bundle_resolver)
    unresolved = [(r["id"], r["custody_reason"]) for r in rows if not r["text"] or not r["bundle"] or r["custody"] != "verified"]
    if unresolved:
        raise ValueError(f"real-trace rows need verified text and corpus custody: {unresolved[:5]}")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return len(rows)
