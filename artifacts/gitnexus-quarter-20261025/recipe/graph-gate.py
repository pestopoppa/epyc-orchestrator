#!/usr/bin/python3
"""No HIGH/CRITICAL or ambiguous graph can silently authorize source apply."""
import hashlib
import json
from pathlib import Path
import sys

output, approval_file, callers_file = map(Path, sys.argv[1:])
approval = json.loads(approval_file.read_text())
unknown = []
high = []


def severe(value):
    if isinstance(value, str):
        return value.upper() in {"HIGH", "CRITICAL"}
    if isinstance(value, dict):
        return any(severe(v) for v in value.values())
    if isinstance(value, list):
        return any(severe(v) for v in value)
    return False


for label in ("guard-admit", "guard-valid-payload", "writer-guard", "writer-main"):
    try:
        value = json.loads((output / (label + ".json")).read_text())
    except (OSError, ValueError):
        unknown.append(label + ": no JSON result")
        continue
    if severe(value):
        high.append(label)
    elif not isinstance(value, dict) or value.get("error") or not value.get("target", {}).get("id"):
        unknown.append(label + ": unresolved symbol")
status = (output / "private-status.txt").read_text()
if "stale" in status.lower() or "up to date" not in status.lower():
    unknown.append("private snapshot is stale or freshness is unproven")
if high:
    decision = "STOP_HIGH_CRITICAL"
elif unknown and not (approval.get("unknown_graph_fallback_approved") is True and
        approval.get("static_callers_sha256") == hashlib.sha256(callers_file.read_bytes()).hexdigest()):
    decision = "STOP_UNKNOWN_REQUIRES_MAIN_REVIEW"
else:
    decision = "ALLOW_PRIVATE_GUARD_FIXTURE_ONLY"
(output / "GRAPH-GATE.json").write_text(json.dumps({"status": decision, "high": high,
    "unknown": unknown, "scope": "guard proposal only; no USAGE validation or index"}, indent=2))
raise SystemExit(0 if decision == "ALLOW_PRIVATE_GUARD_FIXTURE_ONLY" else 69)
