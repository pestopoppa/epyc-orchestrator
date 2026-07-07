#!/usr/bin/env python3
"""Report F3 intake-triage reviewed-label readiness without creating labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datasets._common import load_jsonl, utc_now


DEFAULT_QUEUE = Path("orchestration/datasets/intake_triage_review_queue.jsonl")
DEFAULT_REVIEWED_LABELS = Path("orchestration/datasets/intake_triage_reviewed.jsonl")
DEFAULT_REPORT = Path("orchestration/reports/intake_triage_review_status.json")
DEFAULT_REVIEW_PACKET = Path("orchestration/reports/intake_triage_review_packet.md")
DEFAULT_BATCH_TEMPLATE = Path("orchestration/datasets/intake_triage_review_batch_template.jsonl")
REPORT_VERSION = "intake_triage_review_status.v1"
DEFAULT_TRUSTED_LABEL_SOURCES = ("operator",)


def _load_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _intake_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("intake_id") or "") for row in rows if row.get("intake_id")}


def _effective_label_sources(raw_sources: list[str] | tuple[str, ...] | None) -> set[str]:
    sources = {source for source in (raw_sources or DEFAULT_TRUSTED_LABEL_SOURCES) if source}
    return sources or set(DEFAULT_TRUSTED_LABEL_SOURCES)


def _label_source(row: dict[str, Any]) -> str:
    return str(row.get("label_source") or "operator")


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = _label_source(row)
        counts[source] = counts.get(source, 0) + 1
    return counts


def _pending_sample(
    queue_rows: list[dict[str, Any]],
    *,
    reviewed_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    sample: list[dict[str, Any]] = []
    for row in queue_rows:
        intake_id = str(row.get("intake_id") or "")
        if not intake_id or intake_id in reviewed_ids:
            continue
        sample.append(
            {
                "intake_id": intake_id,
                "title": row.get("title") or "",
                "url": row.get("url") or "",
                "source_type": row.get("source_type") or "",
                "categories": row.get("categories") or [],
                "novelty": row.get("novelty") or "",
                "relevance": row.get("relevance") or "",
                "current_verdict": row.get("current_verdict") or "",
                "destination_handoff": row.get("destination_handoff") or "",
                "destination_index": row.get("destination_index") or "",
                "record_command": row.get("record_command") or "",
                "source_text_excluded": row.get("source_text_excluded") is True,
            }
        )
        if len(sample) >= limit:
            break
    return sample


def _batch_template_rows(pending_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in pending_items:
        rows.append(
            {
                "schema_version": "intake_triage_review_batch.v1",
                "intake_id": item["intake_id"],
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "source_type": item.get("source_type") or "",
                "categories": item.get("categories") or [],
                "suggested_verdict": item.get("current_verdict") or "",
                "verdict": "",
                "destination_handoff": item.get("destination_handoff") or "",
                "destination_index": item.get("destination_index") or "",
                "label_source": "operator",
                "reviewer": "operator",
                "notes": "",
                "source_text_excluded": True,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Intake Triage Review Packet",
        "",
        f"- generated_at: `{report.get('generated_at', '')}`",
        f"- status: `{report.get('status', '')}`",
        f"- queue_rows: `{report.get('queue_rows', 0)}`",
        f"- trusted_reviewed_unique_intake_ids: `{report.get('trusted_reviewed_unique_intake_ids', 0)}`",
        f"- labels_needed: `{report.get('labels_needed', 0)}`",
        f"- remaining_queue_items: `{report.get('remaining_queue_items', 0)}`",
        f"- trusted_label_sources: `{', '.join(report.get('trusted_label_sources') or [])}`",
        "",
        "Edit the batch template rows by filling `verdict`, `reviewer`, and `notes`, then validate with:",
        "",
        "```bash",
        "uv run python scripts/datasets/apply_intake_triage_review_batch.py --batch <filled-template.jsonl>",
        "```",
        "",
        "Apply only after review:",
        "",
        "```bash",
        "uv run python scripts/datasets/apply_intake_triage_review_batch.py --batch <filled-template.jsonl> --apply",
        "```",
        "",
    ]
    items = report.get("next_review_items") or []
    if items:
        lines.extend(
            [
                "## Pending Items",
                "",
                "| intake_id | verdict | relevance | novelty | destination | title |",
                "|---|---|---|---|---|---|",
            ]
        )
        for item in items:
            title = str(item.get("title") or "").replace("|", "\\|")
            destination = str(item.get("destination_handoff") or item.get("destination_index") or "")
            lines.append(
                "| {intake_id} | {verdict} | {relevance} | {novelty} | `{destination}` | {title} |".format(
                    intake_id=item.get("intake_id", ""),
                    verdict=item.get("current_verdict", ""),
                    relevance=item.get("relevance", ""),
                    novelty=item.get("novelty", ""),
                    destination=destination,
                    title=title,
                )
            )
        lines.extend(["", "## Batch Template", "", "```jsonl"])
        for row in report.get("review_batch_template", []) or []:
            lines.append(json.dumps(row, sort_keys=True))
        lines.extend(["```", ""])
    else:
        lines.extend(["No pending intake-triage review items were included in this packet.", ""])
    return "\n".join(lines).rstrip() + "\n"


def summarize(
    *,
    queue_path: Path,
    reviewed_labels_path: Path,
    min_reviewed_labels: int,
    trusted_label_sources: set[str] | None = None,
    pending_sample_limit: int = 0,
) -> dict[str, Any]:
    queue_rows = _load_jsonl_if_present(queue_path)
    reviewed_rows = _load_jsonl_if_present(reviewed_labels_path)
    trusted_sources = trusted_label_sources or set(DEFAULT_TRUSTED_LABEL_SOURCES)
    trusted_reviewed_rows = [
        row for row in reviewed_rows if _label_source(row) in trusted_sources
    ]
    queue_ids = _intake_ids(queue_rows)
    reviewed_ids = _intake_ids(trusted_reviewed_rows)
    all_reviewed_ids = _intake_ids(reviewed_rows)
    reviewed_queue_ids = queue_ids & reviewed_ids
    remaining_queue_ids = queue_ids - reviewed_ids
    labels_needed = max(0, min_reviewed_labels - len(reviewed_ids))

    if len(reviewed_ids) >= min_reviewed_labels:
        status = "ready_for_baseline"
    elif not queue_ids:
        status = "missing_review_queue"
    elif len(reviewed_ids) + len(remaining_queue_ids) < min_reviewed_labels:
        status = "queue_exhausted_below_gate"
    else:
        status = "needs_reviewed_labels"

    next_review_items = _pending_sample(
        queue_rows,
        reviewed_ids=reviewed_ids,
        limit=max(0, pending_sample_limit),
    )
    review_batch_template = _batch_template_rows(next_review_items)
    return {
        "schema_version": REPORT_VERSION,
        "generated_at": utc_now(),
        "queue_path": str(queue_path),
        "reviewed_labels_path": str(reviewed_labels_path),
        "min_reviewed_labels": min_reviewed_labels,
        "trusted_label_sources": sorted(trusted_sources),
        "status": status,
        "queue_rows": len(queue_rows),
        "queue_unique_intake_ids": len(queue_ids),
        "reviewed_rows": len(reviewed_rows),
        "reviewed_unique_intake_ids": len(all_reviewed_ids),
        "trusted_reviewed_rows": len(trusted_reviewed_rows),
        "trusted_reviewed_unique_intake_ids": len(reviewed_ids),
        "reviewed_label_sources": _source_counts(reviewed_rows),
        "reviewed_queue_items": len(reviewed_queue_ids),
        "remaining_queue_items": len(remaining_queue_ids),
        "labels_needed": labels_needed,
        "next_review_items": next_review_items,
        "review_batch_template": review_batch_template,
        "review_batch_template_rows": len(review_batch_template),
        "ready_for_baseline": status == "ready_for_baseline",
        "privacy": {
            "raw_text_in_report": False,
            "reported_fields": (
                "aggregate counts plus sanitized pending review sample"
                if next_review_items
                else "aggregate counts only"
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    queue_path = Path(args.queue).expanduser()
    reviewed_labels_path = Path(args.reviewed_labels).expanduser()
    report = summarize(
        queue_path=queue_path,
        reviewed_labels_path=reviewed_labels_path,
        min_reviewed_labels=args.min_reviewed_labels,
        trusted_label_sources=_effective_label_sources(
            getattr(args, "trusted_label_source", [])
        ),
        pending_sample_limit=max(0, args.pending_sample),
    )
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["report_path"] = str(report_path)
    if args.batch_template:
        batch_template_path = Path(args.batch_template).expanduser()
        written = _write_jsonl(batch_template_path, report["review_batch_template"])
        report["batch_template_path"] = str(batch_template_path)
        report["batch_template_written"] = written
    if args.output_md:
        output_md_path = Path(args.output_md).expanduser()
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(render_markdown(report), encoding="utf-8")
        report["output_md_path"] = str(output_md_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--reviewed-labels", default=str(DEFAULT_REVIEWED_LABELS))
    parser.add_argument("--min-reviewed-labels", type=int, default=100)
    parser.add_argument(
        "--pending-sample",
        type=int,
        default=0,
        help="Include this many sanitized pending review rows and recorder commands.",
    )
    parser.add_argument(
        "--trusted-label-source",
        action="append",
        default=[],
        help="Reviewed label source to count toward baseline readiness; defaults to operator.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Accepted for CLI consistency; output is always JSON.",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const=str(DEFAULT_REPORT),
        default="",
        help="Optionally write a JSON report; defaults to the standard report path when no value is supplied.",
    )
    parser.add_argument(
        "--batch-template",
        nargs="?",
        const=str(DEFAULT_BATCH_TEMPLATE),
        default="",
        help="Optionally write an operator-fillable JSONL batch template from the pending sample.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit a markdown review packet instead of JSON.",
    )
    parser.add_argument(
        "--output-md",
        nargs="?",
        const=str(DEFAULT_REVIEW_PACKET),
        default="",
        help="Optionally write a markdown review packet; defaults to the standard report path when no value is supplied.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    if args.markdown:
        print(render_markdown(result), end="")
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
