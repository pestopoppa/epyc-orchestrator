#!/usr/bin/env python3
"""Fail-closed hygiene preflight for raw trace export/publish candidates.

This is a local, no-inference scanner for F3 W-aux. It does not authorize
publishing or training; it only reports whether candidate JSONL/text exports
contain credential-like content, high-entropy token shapes, or sensitive hits
inside reasoning/trace fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.repl_environment.redaction import redact_credentials

SCHEMA_VERSION = "raw_trace_publish_preflight.v1"
REASONING_FIELD_RE = re.compile(r"(reason|thinking|thought|trace|chain)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9_+/=-]{32,}")
HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
FALSE_POSITIVE_KEYS = {
    "sha",
    "sha256",
    "hash",
    "prompt_hash",
    "answer_sha256",
    "expected_sha256",
    "commit",
    "git_sha",
}


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _iter_strings(value: Any, *, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, child in value.items():
            out.extend(_iter_strings(child, path=f"{path}.{key}"))
        return out
    if isinstance(value, list):
        out = []
        for idx, child in enumerate(value):
            out.extend(_iter_strings(child, path=f"{path}[{idx}]"))
        return out
    return []


def _path_leaf(field_path: str) -> str:
    return re.split(r"[.\[]", field_path)[-1].rstrip("]").lower()


def _looks_like_known_hash(token: str, field_path: str) -> bool:
    leaf = _path_leaf(field_path)
    if leaf in FALSE_POSITIVE_KEYS:
        return True
    return bool(HEX_RE.fullmatch(token) and len(token) in {40, 64})


def _scan_text(text: str, *, field_path: str, line_no: int | None) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    redacted = redact_credentials(text)
    if redacted.redacted_count:
        hits.append({
            "line": line_no,
            "field_path": field_path,
            "kind": "credential_pattern",
            "categories": sorted(redacted.categories),
            "count": redacted.redacted_count,
        })
    for token in TOKEN_RE.findall(text):
        if _looks_like_known_hash(token, field_path):
            continue
        entropy = _entropy(token)
        if entropy >= 4.0 and len(set(token)) >= 12:
            hits.append({
                "line": line_no,
                "field_path": field_path,
                "kind": "high_entropy_token",
                "length": len(token),
                "entropy": round(entropy, 3),
            })
            break
    return hits


def scan_path(path: Path) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    rows_scanned = 0
    reasoning_fields_scanned = 0
    jsonl_rows = 0
    parse_errors = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows_scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                hits.extend(_scan_text(line, field_path="$", line_no=line_no))
                continue
            jsonl_rows += 1
            for field_path, text in _iter_strings(row):
                if REASONING_FIELD_RE.search(field_path):
                    reasoning_fields_scanned += 1
                hits.extend(_scan_text(text, field_path=field_path, line_no=line_no))

    reasoning_hits = [
        hit for hit in hits
        if REASONING_FIELD_RE.search(str(hit.get("field_path", "")))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "ok": not hits and parse_errors == 0,
        "rows_scanned": rows_scanned,
        "jsonl_rows": jsonl_rows,
        "parse_errors": parse_errors,
        "reasoning_fields_scanned": reasoning_fields_scanned,
        "hit_count": len(hits),
        "reasoning_hit_count": len(reasoning_hits),
        "hits": hits[:50],
        "truncated_hits": max(0, len(hits) - 50),
    }


def run(paths: list[Path]) -> dict[str, Any]:
    reports = [scan_path(path) for path in paths]
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": all(report["ok"] for report in reports),
        "paths": reports,
        "summary": {
            "n_paths": len(reports),
            "n_blocked_paths": sum(not report["ok"] for report in reports),
            "hit_count": sum(report["hit_count"] for report in reports),
            "parse_errors": sum(report["parse_errors"] for report in reports),
            "reasoning_hit_count": sum(report["reasoning_hit_count"] for report in reports),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Candidate JSONL/text export path(s)")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args.paths)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
