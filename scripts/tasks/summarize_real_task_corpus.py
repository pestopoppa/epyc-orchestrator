#!/usr/bin/env python3
"""Summarize mixed real-task corpus evidence without publishing raw prompts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_MAX_WEIGHTED_SOURCE_FAMILY_SHARE = 0.60


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _count_payload(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _manifest_counts(manifest: dict[str, Any]) -> dict[str, Any]:
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        return counts
    return {}


def summarize_source(spec: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(spec["manifest"]).expanduser()
    rows_path = Path(spec["rows"]).expanduser() if spec.get("rows") else None
    manifest = load_json(manifest_path)
    counts = _manifest_counts(manifest)
    rows = load_jsonl(rows_path) if rows_path else []
    written = int(counts.get("written") or len(rows))
    source_family = str(spec["source_family"])
    source_weight = float(spec["weight"])
    by_class = {str(k): int(v) for k, v in dict(counts.get("by_class") or {}).items()}
    by_outcome = {str(k): int(v) for k, v in dict(counts.get("by_outcome") or {}).items()}

    token_payloads = sum(1 for row in rows if _count_payload(row, "tokens"))
    wall_time_rows = sum(1 for row in rows if _count_payload(row, "wall_s"))
    prompt_text_rows = sum(1 for row in rows if _count_payload(row, "prompt"))
    prompt_ref_rows = sum(1 for row in rows if _count_payload(row, "prompt_ref"))

    return {
        "label": str(spec["label"]),
        "source_family": source_family,
        "evidence_role": str(spec["evidence_role"]),
        "weight": source_weight,
        "weighted_records": written * source_weight,
        "manifest_path": str(manifest_path),
        "rows_path": str(rows_path) if rows_path else None,
        "written": written,
        "by_class": by_class,
        "by_outcome": by_outcome,
        "rows_inspected": len(rows),
        "token_payloads": token_payloads,
        "wall_time_rows": wall_time_rows,
        "prompt_text_rows": prompt_text_rows,
        "prompt_ref_rows": prompt_ref_rows,
        "manifest_counts": counts,
    }


def _weighted_counter(sources: list[dict[str, Any]], key: str) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for source in sources:
        weight = float(source["weight"])
        for item, count in dict(source.get(key) or {}).items():
            totals[str(item)] += float(count) * weight
    return {k: round(v, 6) for k, v in sorted(totals.items())}


def _raw_counter(sources: list[dict[str, Any]], key: str) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for source in sources:
        for item, count in dict(source.get(key) or {}).items():
            totals[str(item)] += int(count)
    return dict(sorted(totals.items()))


def _sum_by_source_field(
    sources: list[dict[str, Any]], field: str, value_field: str, *, as_float: bool
) -> dict[str, float] | dict[str, int]:
    totals: Counter[str] = Counter()
    for source in sources:
        key = str(source[field])
        value = source[value_field]
        totals[key] += float(value) if as_float else int(value)
    if as_float:
        return {k: round(float(v), 6) for k, v in sorted(totals.items())}
    return {k: int(v) for k, v in sorted(totals.items())}


def _source_family_share_readout(
    source_family_weighted: dict[str, float], weighted_records: float
) -> dict[str, Any]:
    if weighted_records <= 0 or not source_family_weighted:
        return {
            "by_source_family_weighted_share": {},
            "dominant_source_family": None,
            "max_source_family_weighted_share": 0.0,
        }
    shares = {
        family: round(float(weighted) / weighted_records, 6)
        for family, weighted in sorted(source_family_weighted.items())
    }
    dominant_family, max_share = max(shares.items(), key=lambda item: item[1])
    return {
        "by_source_family_weighted_share": shares,
        "dominant_source_family": dominant_family,
        "max_source_family_weighted_share": max_share,
    }


def build_summary(source_specs: list[dict[str, Any]], *, generated_at: str | None = None) -> dict[str, Any]:
    sources = [summarize_source(spec) for spec in source_specs]
    raw_records = sum(int(source["written"]) for source in sources)
    weighted_records = sum(float(source["weighted_records"]) for source in sources)
    source_family_raw = _sum_by_source_field(sources, "source_family", "written", as_float=False)
    source_family_weighted = _sum_by_source_field(
        sources, "source_family", "weighted_records", as_float=True
    )
    evidence_role_raw = _sum_by_source_field(sources, "evidence_role", "written", as_float=False)
    source_family_share = _source_family_share_readout(source_family_weighted, weighted_records)
    rows_inspected = sum(int(source["rows_inspected"]) for source in sources)
    token_payloads = sum(int(source["token_payloads"]) for source in sources)
    wall_time_rows = sum(int(source["wall_time_rows"]) for source in sources)
    prompt_text_rows = sum(int(source["prompt_text_rows"]) for source in sources)
    prompt_ref_rows = sum(int(source["prompt_ref_rows"]) for source in sources)
    source_weight_dominance_ok = (
        source_family_share["max_source_family_weighted_share"]
        <= DEFAULT_MAX_WEIGHTED_SOURCE_FAMILY_SHARE
    )

    return {
        "schema_version": "mixed_real_task_corpus_summary.v1",
        "generated_at": generated_at or utc_now(),
        "sources": sources,
        "totals": {
            "raw_records": raw_records,
            "weighted_records": round(weighted_records, 6),
            "source_family_count": len({source["source_family"] for source in sources}),
            "by_source_family_raw": source_family_raw,
            "by_source_family_weighted": source_family_weighted,
            **source_family_share,
            "max_weighted_source_family_share_allowed": DEFAULT_MAX_WEIGHTED_SOURCE_FAMILY_SHARE,
            "by_evidence_role_raw": evidence_role_raw,
            "by_class_raw": _raw_counter(sources, "by_class"),
            "by_class_weighted": _weighted_counter(sources, "by_class"),
            "by_outcome_raw": _raw_counter(sources, "by_outcome"),
            "rows_inspected": rows_inspected,
            "token_payloads": token_payloads,
            "wall_time_rows": wall_time_rows,
            "prompt_text_rows": prompt_text_rows,
            "prompt_ref_rows": prompt_ref_rows,
        },
        "gate_readout": {
            "class_outcome_count_gate": raw_records >= 100,
            "multiple_source_families": len({source["source_family"] for source in sources}) >= 2,
            "token_payload_coverage": token_payloads > 0,
            "source_weight_dominance_ok": source_weight_dominance_ok,
            "privacy_prompt_text_free": prompt_text_rows == 0,
            "status": "summary_checkpoint_not_final_w2",
            "notes": [
                "Benchmark/eval rows remain valid high-volume AutoPilot RL/calibration fuel.",
                "Historical operator conversations are tracked as a separate demand-distribution stratum.",
                "Weighted source-family shares must keep any single source family from defining the whole distribution.",
                "This summary is safe to commit because it contains aggregate counts and paths only, not raw transcript text.",
            ],
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    lines = [
        "# Mixed Real-Task Corpus Summary",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Raw records represented: {totals['raw_records']}",
        f"- Weighted records represented: {totals['weighted_records']}",
        f"- Source families: {totals['source_family_count']}",
        f"- Rows inspected for privacy/token fields: {totals['rows_inspected']}",
        f"- Token payload rows: {totals['token_payloads']}",
        f"- Prompt text rows: {totals['prompt_text_rows']}",
        f"- Prompt ref rows: {totals['prompt_ref_rows']}",
        "",
        "## Gate Readout",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for key, value in summary["gate_readout"].items():
        if key == "notes":
            continue
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## Source Families", "", "| Source family | Raw records | Weighted records |"])
    lines.append("|---|---:|---:|")
    weighted = totals["by_source_family_weighted"]
    for family, raw_count in totals["by_source_family_raw"].items():
        lines.append(f"| {family} | {raw_count} | {weighted.get(family, 0)} |")

    lines.extend(["", "## Source Weight Shares", "", "| Source family | Weighted share |"])
    lines.append("|---|---:|")
    for family, share in totals["by_source_family_weighted_share"].items():
        lines.append(f"| {family} | {share} |")
    lines.extend(
        [
            "",
            f"- Dominant source family: `{totals['dominant_source_family']}`",
            f"- Max weighted source-family share: {totals['max_source_family_weighted_share']}",
            "- Max allowed weighted source-family share: "
            f"{totals['max_weighted_source_family_share_allowed']}",
        ]
    )

    lines.extend(["", "## Classes", "", "| Class | Raw records | Weighted records |"])
    lines.append("|---|---:|---:|")
    weighted_classes = totals["by_class_weighted"]
    for task_class, raw_count in totals["by_class_raw"].items():
        lines.append(f"| {task_class} | {raw_count} | {weighted_classes.get(task_class, 0)} |")

    lines.extend(["", "## Sources", "", "| Label | Family | Evidence role | Weight | Records | Manifest | Rows |"])
    lines.append("|---|---|---|---:|---:|---|---|")
    for source in summary["sources"]:
        lines.append(
            "| {label} | {family} | {role} | {weight} | {records} | `{manifest}` | `{rows}` |".format(
                label=source["label"],
                family=source["source_family"],
                role=source["evidence_role"],
                weight=source["weight"],
                records=source["written"],
                manifest=source["manifest_path"],
                rows=source["rows_path"],
            )
        )

    lines.extend(["", "## Notes", ""])
    for note in summary["gate_readout"]["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def parse_source(value: str) -> dict[str, Any]:
    parts = value.split("|")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "--source must be 'label|source_family|evidence_role|weight|manifest_path|rows_path_or_none'"
        )
    label, source_family, evidence_role, weight, manifest, rows = parts
    try:
        parsed_weight = float(weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid source weight: {weight}") from exc
    return {
        "label": label,
        "source_family": source_family,
        "evidence_role": evidence_role,
        "weight": parsed_weight,
        "manifest": manifest,
        "rows": None if rows in {"", "-"} else rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--generated-at", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_summary(args.source, generated_at=args.generated_at)
    output_json = Path(args.output_json).expanduser()
    output_md = Path(args.output_md).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, sort_keys=True))


if __name__ == "__main__":
    main()
