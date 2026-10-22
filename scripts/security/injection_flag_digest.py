#!/usr/bin/env python3
"""CT-19.1: human-readable digest of the CT-19b injection-flag ledger(s).

CT-19b (``src/security/skill_scan.py``, flag ``skill_injection_scan``, default off) appends one
warn-only JSONL line per flagged model-authored text to ``injection_flags.jsonl``; until this
script nothing read it. This is the wrap-up digest surface (no dashboard panel: a hub panel would
need a registry entry, a health probe and a freshness envelope under the dashboard plane rule).

Ledgers read, in order, de-duplicated by resolved path:
  1. ``skill_scan.default_ledger()`` (``ORCHESTRATOR_INJECTION_FLAG_LEDGER`` or ``logs/injection_flags.jsonl``);
  2. the default SkillBank directory's ledger (``scan_skill(..., bank_dir=<skills.db parent>)``),
     ``orchestration/repl_memory/sessions/injection_flags.jsonl``;
  3. every ``--ledger PATH`` given.

Read-only. Prints counts by source, flagged field and threat, freshness (age of the newest line)
and the newest N entries (time, source, subject, field -> threat names; the ledger holds no
scanned text, so none is printed). A missing ledger is reported as such, never as "clean".

Usage:
    python scripts/security/injection_flag_digest.py [--ledger PATH ...] [--newest N]
        [--since-hours H] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

SCHEMA = "epyc.orchestrator.injection_flag_digest.v1"
BANK_LEDGER = _REPO / "orchestration" / "repl_memory" / "sessions" / "injection_flags.jsonl"


def candidate_ledgers(extra: Iterable[str | Path] = ()) -> list[Path]:
    from src.security.skill_scan import default_ledger

    out: list[Path] = []
    seen: set[str] = set()
    for p in [default_ledger(), BANK_LEDGER, *[Path(x) for x in extra]]:
        key = str(Path(p).expanduser().resolve())
        if key not in seen:
            seen.add(key)
            out.append(Path(p).expanduser())
    return out


def _iso(ts: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def read_ledger(path: Path, *, since_ts: float | None = None) -> dict[str, Any]:
    """Parse one ledger. Malformed lines are counted, never silently dropped."""
    info: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "entries": [],
                            "malformed_lines": 0, "lines": 0}
    if not info["exists"]:
        return info
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        info["lines"] += 1
        try:
            rec = json.loads(raw)
            if not isinstance(rec, dict) or not isinstance(rec.get("flags"), dict):
                raise ValueError("not a flag record")
            ts = float(rec.get("ts"))
        except (ValueError, TypeError):
            info["malformed_lines"] += 1
            continue
        if since_ts is not None and ts < since_ts:
            continue
        info["entries"].append({"ts": ts, "source": str(rec.get("source", "")),
                                "subject": str(rec.get("subject", "")),
                                "flags": {str(k): [str(t) for t in (v or [])] for k, v in rec["flags"].items()},
                                "action": str(rec.get("action", ""))})
    return info


def digest(ledgers: Sequence[Path], *, newest: int = 10, since_hours: float | None = None,
           now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    since_ts = now - since_hours * 3600 if since_hours is not None else None
    per = [read_ledger(p, since_ts=since_ts) for p in ledgers]
    entries = sorted((dict(e, ledger=i["path"]) for i in per for e in i["entries"]),
                     key=lambda e: e["ts"], reverse=True)
    by_source = Counter(e["source"] for e in entries)
    by_field = Counter(f for e in entries for f in e["flags"])
    by_threat = Counter(t for e in entries for v in e["flags"].values() for t in v)
    newest_ts = entries[0]["ts"] if entries else None
    return {
        "schema": SCHEMA,
        "generated_at": _iso(now),
        "window_hours": since_hours,
        "ledgers": [{k: i[k] for k in ("path", "exists", "lines", "malformed_lines")}
                    | {"entries_in_window": len(i["entries"])} for i in per],
        "any_ledger_exists": any(i["exists"] for i in per),
        "total_flags": len(entries),
        "newest_at": _iso(newest_ts) if newest_ts is not None else None,
        "newest_age_hours": round((now - newest_ts) / 3600, 2) if newest_ts is not None else None,
        "by_source": dict(by_source.most_common()),
        "by_field": dict(by_field.most_common()),
        "by_threat": dict(by_threat.most_common()),
        "newest": [{"at": _iso(e["ts"]), "source": e["source"], "subject": e["subject"],
                    "flags": e["flags"], "ledger": e["ledger"]} for e in entries[:max(0, newest)]],
    }


def render(d: dict[str, Any]) -> str:
    lines = ["Injection-flag ledger digest (CT-19b, warn-only; flag skill_injection_scan)",
             f"generated {d['generated_at']}" + (f", window {d['window_hours']} h" if d["window_hours"] else "")]
    for led in d["ledgers"]:
        state = (f"{led['lines']} lines, {led['entries_in_window']} in window"
                 + (f", {led['malformed_lines']} MALFORMED" if led["malformed_lines"] else "")
                 if led["exists"] else "absent")
        lines.append(f"  ledger {led['path']}: {state}")
    if not d["any_ledger_exists"]:
        lines.append("No ledger exists: either the scan flag is off or nothing was ever flagged "
                     "(this digest cannot tell which; check `skill_injection_scan`).")
        return "\n".join(lines)
    lines.append(f"Total flags: {d['total_flags']}"
                 + (f"; newest {d['newest_at']} ({d['newest_age_hours']} h ago)" if d["newest_at"] else ""))
    for title, key in (("By source", "by_source"), ("By field", "by_field"), ("By threat", "by_threat")):
        if d[key]:
            lines.append(f"{title}: " + ", ".join(f"{k}={v}" for k, v in d[key].items()))
    if d["newest"]:
        lines.append(f"Newest {len(d['newest'])}:")
        for e in d["newest"]:
            flags = "; ".join(f"{f}: {','.join(t)}" for f, t in e["flags"].items())
            lines.append(f"  {e['at']}  {e['source']}  {e['subject'] or '-'}  [{flags}]")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", action="append", default=[], help="extra ledger path (repeatable)")
    ap.add_argument("--newest", type=int, default=10)
    ap.add_argument("--since-hours", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    d = digest(candidate_ledgers(a.ledger), newest=a.newest, since_hours=a.since_hours)
    print(json.dumps(d, indent=2, sort_keys=True) if a.json else render(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
