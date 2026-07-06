#!/usr/bin/env python3
"""Read-only consistency report for the Fable quiet-window queue."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/mnt/raid0/llm/epyc-root")
QUEUE_FILES = (
    "handoffs/active/master-handoff-index.md",
    "handoffs/active/bulk-inference-campaign.md",
    "handoffs/active/routing-and-optimization-index.md",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _line_with(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line:
            return line.strip()
    return ""


def _finding(severity: str, rel_path: str, issue: str, evidence: str) -> dict[str, str]:
    return {
        "severity": severity,
        "file": rel_path,
        "issue": issue,
        "evidence": evidence,
    }


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def build_report(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = root.expanduser().resolve()
    files = {rel: _read(root / rel) for rel in QUEUE_FILES}
    findings: list[dict[str, str]] = []

    required_concepts = {
        "E1 dense-control": {
            "terms": ("E1 dense-control", "server_np_sweep.py"),
            "issue": "remaining E1 quiet-window sweep is missing",
        },
        "real_suite_v1": {
            "terms": ("real_suite_v1",),
            "issue": "real_suite_v1 clean-window ledger run is missing",
        },
        "W8/Fable readiness": {
            "terms": ("W8/Fable readiness", "W8 trajectory"),
            "issue": "W8/Fable readiness refresh is missing",
        },
    }
    for rel, text in files.items():
        if not text:
            findings.append(_finding("high", rel, "queue file missing or unreadable", ""))
            continue
        for concept, spec in required_concepts.items():
            terms = tuple(spec["terms"])
            if not _contains_any(text, terms):
                findings.append(
                    _finding(
                        "medium",
                        rel,
                        str(spec["issue"]),
                        f"missing concept {concept}; accepted terms: {', '.join(terms)}",
                    )
                )

    master = files.get("handoffs/active/master-handoff-index.md", "")
    bulk = files.get("handoffs/active/bulk-inference-campaign.md", "")
    routing = files.get("handoffs/active/routing-and-optimization-index.md", "")

    active_table_text = "\n".join(
        line
        for line in (master + "\n" + bulk).splitlines()
        if line.startswith("|") or "active batch" in line or "active next-run" in line
    )
    stale_active_terms = {
        "DS-E1 KV measurement": "DS-E1 appears as an active quiet-window task",
        "E2/E1 batched-decode measurement": "closed E2/E1 combined row appears active",
    }
    for term, issue in stale_active_terms.items():
        if term in active_table_text:
            findings.append(
                _finding(
                    "high",
                    "handoffs/active/{master-handoff-index.md,bulk-inference-campaign.md}",
                    issue,
                    _line_with(active_table_text, term),
                )
            )

    has_routing_guard = (
        "E2 activation/rollback is closed" in routing
        and "DS-E1 is decision-ready" in routing
        and "scheduled as next-run work" in routing
    )
    if not has_routing_guard:
        findings.append(
            _finding(
                "medium",
                "handoffs/active/routing-and-optimization-index.md",
                "routing index does not explicitly keep closed E2 and decision-ready DS-E1 out of next-run work",
                _line_with(routing, "DS-E1") or _line_with(routing, "E2"),
            )
        )

    status = "ok" if not findings else "attention"
    return {
        "schema_version": "quiet_window_queue_report.v1",
        "generated_at": utc_now(),
        "ok": not findings,
        "status": status,
        "blockers": [row["issue"] for row in findings if row["severity"] == "high"],
        "findings": findings,
        "checked_files": list(QUEUE_FILES),
        "active_queue": {
            "requires": sorted(required_concepts),
            "forbidden_active_terms": sorted(stale_active_terms),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--json", action="store_true", help="Accepted for consistency; output is always JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(Path(args.root))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
