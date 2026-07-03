#!/usr/bin/env python3
"""Report live MTP/NEXTN acceptance evidence from local llama-server logs.

This is a zero-inference diagnostic. It reads the current serving attestation
for role/port inventory, scans existing server logs for llama.cpp MTP
acceptance counters, and fails loudly when a role configured for draft-mtp has
no acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTESTATION = PROJECT_ROOT / "orchestration" / "attestation" / "latest.json"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"

TASK_ACCEPTANCE_RE = re.compile(
    r"task\s+(?P<task_id>\d+)\s+\|\s+draft acceptance\s+=\s+"
    r"(?P<rate>[0-9.]+)\s+\(\s*(?P<accepted>\d+)\s+accepted\s*/\s*"
    r"(?P<generated>\d+)\s+generated\),\s+mean acceptance length\s+=\s+"
    r"(?P<mean>[0-9.]+),\s+acceptance rate per position\s+=\s+\((?P<per_pos>[^)]*)\)"
)
CUMULATIVE_STATS_RE = re.compile(
    r"statistics\s+(?P<spec_type>[^:]+):.*?#gen drafts\s*=\s*(?P<gen_drafts>\d+),\s*"
    r"#acc drafts\s*=\s*(?P<acc_drafts>\d+),\s*#gen tokens\s*=\s*(?P<gen_tokens>\d+),\s*"
    r"#acc tokens\s*=\s*(?P<acc_tokens>\d+),\s*#mean acc len\s*=\s*(?P<mean>[0-9.]+),\s*"
    r"#acc rate/pos\s*=\s*\((?P<per_pos>[^)]*)\)"
)
NO_SPEC_IMPL_RE = re.compile(r"no implementations specified for speculative decoding", re.IGNORECASE)


@dataclass(frozen=True)
class PortInventory:
    """Current serving inventory for one port."""

    port: int
    primary_role: str
    registry_roles: list[str]
    pid: int | None
    spec_type: str | None
    draft_model_path: str | None
    model_path: str | None
    cpu_intent: str | None

    @property
    def mtp_configured(self) -> bool:
        return self.spec_type == "draft-mtp" or bool(self.draft_model_path)


@dataclass
class TaskAcceptance:
    """One per-task acceptance line."""

    task_id: int
    accepted_tokens: int
    generated_tokens: int
    acceptance_rate: float
    mean_acceptance_length: float
    per_position_rates: list[float]
    line_number: int


@dataclass
class CumulativeStats:
    """Latest cumulative llama.cpp draft-mtp statistics line for a process."""

    spec_type: str
    generated_drafts: int
    accepted_drafts: int
    generated_tokens: int
    accepted_tokens: int
    mean_acceptance_length: float
    per_position_rates: list[float]
    line_number: int

    @property
    def token_acceptance_rate(self) -> float | None:
        if self.generated_tokens <= 0:
            return None
        return self.accepted_tokens / self.generated_tokens

    @property
    def draft_acceptance_rate(self) -> float | None:
        if self.generated_drafts <= 0:
            return None
        return self.accepted_drafts / self.generated_drafts


@dataclass
class LogEvidence:
    """Acceptance evidence parsed from one log file."""

    path: str
    exists: bool
    size_bytes: int | None = None
    mtime_utc: str | None = None
    line_count: int = 0
    task_line_count: int = 0
    cumulative_line_count: int = 0
    task_generated_tokens: int = 0
    task_accepted_tokens: int = 0
    no_spec_implementation: bool = False
    latest_task: TaskAcceptance | None = None
    latest_cumulative: CumulativeStats | None = None

    @property
    def has_acceptance_evidence(self) -> bool:
        return self.task_line_count > 0 or self.cumulative_line_count > 0

    @property
    def task_token_acceptance_rate(self) -> float | None:
        if self.task_generated_tokens <= 0:
            return None
        return self.task_accepted_tokens / self.task_generated_tokens


@dataclass
class PortReport:
    """Port-level acceptance summary."""

    port: int
    primary_role: str
    registry_roles: list[str]
    pid: int | None
    spec_type: str | None
    mtp_configured: bool
    model_path: str | None
    draft_model_path: str | None
    cpu_intent: str | None
    log: LogEvidence
    status: str
    token_acceptance_rate: float | None
    draft_acceptance_rate: float | None
    generated_tokens: int
    accepted_tokens: int
    generated_drafts: int | None
    accepted_drafts: int | None
    mean_acceptance_length: float | None
    per_position_rates: list[float]
    evidence_source: str


@dataclass
class RoleReport:
    """Primary-role aggregate over one or more serving ports."""

    role: str
    ports: list[int]
    registry_roles: list[str]
    mtp_configured_ports: list[int]
    evidence_ports: list[int]
    missing_evidence_ports: list[int]
    acceptance_line_count: int
    cumulative_line_count: int
    generated_tokens: int
    accepted_tokens: int
    generated_drafts: int
    accepted_drafts: int
    token_acceptance_rate: float | None
    draft_acceptance_rate: float | None
    status: str


@dataclass
class AcceptanceReport:
    """Full report payload."""

    generated_at: str
    attestation_path: str
    logs_dir: str
    min_lines_per_mtp_role: int
    summary: dict[str, Any]
    roles: list[RoleReport] = field(default_factory=list)
    ports: list[PortReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    return values


def parse_task_acceptance_line(line: str, line_number: int = 0) -> TaskAcceptance | None:
    """Parse a llama.cpp per-task draft acceptance timing line."""
    match = TASK_ACCEPTANCE_RE.search(line)
    if not match:
        return None
    return TaskAcceptance(
        task_id=int(match.group("task_id")),
        accepted_tokens=int(match.group("accepted")),
        generated_tokens=int(match.group("generated")),
        acceptance_rate=float(match.group("rate")),
        mean_acceptance_length=float(match.group("mean")),
        per_position_rates=_parse_float_list(match.group("per_pos")),
        line_number=line_number,
    )


def parse_cumulative_stats_line(line: str, line_number: int = 0) -> CumulativeStats | None:
    """Parse a llama.cpp cumulative speculative decoding statistics line."""
    match = CUMULATIVE_STATS_RE.search(line)
    if not match:
        return None
    return CumulativeStats(
        spec_type=match.group("spec_type").strip(),
        generated_drafts=int(match.group("gen_drafts")),
        accepted_drafts=int(match.group("acc_drafts")),
        generated_tokens=int(match.group("gen_tokens")),
        accepted_tokens=int(match.group("acc_tokens")),
        mean_acceptance_length=float(match.group("mean")),
        per_position_rates=_parse_float_list(match.group("per_pos")),
        line_number=line_number,
    )


def parse_log(path: Path) -> LogEvidence:
    """Parse one server log for MTP acceptance evidence."""
    evidence = LogEvidence(path=str(path), exists=path.exists())
    if not path.exists():
        return evidence

    stat = path.stat()
    evidence.size_bytes = stat.st_size
    evidence.mtime_utc = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            evidence.line_count = line_number
            if NO_SPEC_IMPL_RE.search(line):
                evidence.no_spec_implementation = True

            task = parse_task_acceptance_line(line, line_number)
            if task is not None:
                evidence.task_line_count += 1
                evidence.task_generated_tokens += task.generated_tokens
                evidence.task_accepted_tokens += task.accepted_tokens
                evidence.latest_task = task
                continue

            cumulative = parse_cumulative_stats_line(line, line_number)
            if cumulative is not None:
                evidence.cumulative_line_count += 1
                evidence.latest_cumulative = cumulative

    return evidence


def _coerce_port(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _registry_roles(row: dict[str, Any]) -> list[str]:
    roles = {
        str(match.get("role"))
        for match in row.get("registry_matches", [])
        if isinstance(match, dict) and match.get("role") and match.get("role") != "server_defaults"
    }
    primary = (row.get("numa_intent") or {}).get("role")
    if primary:
        roles.add(str(primary))
    return sorted(roles)


def load_port_inventory(attestation_path: Path) -> list[PortInventory]:
    """Load current live port inventory from the serving attestation."""
    data = json.loads(attestation_path.read_text(encoding="utf-8"))
    rows = ((data.get("sections") or {}).get("serving_config") or [])
    inventory: list[PortInventory] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        port = _coerce_port(row.get("port"))
        if port is None:
            continue
        numa_intent = row.get("numa_intent") or {}
        primary_role = str(numa_intent.get("role") or "")
        roles = _registry_roles(row)
        if not primary_role:
            primary_role = roles[0] if roles else f"port_{port}"
        inventory.append(
            PortInventory(
                port=port,
                primary_role=primary_role,
                registry_roles=roles,
                pid=_coerce_port(row.get("pid")),
                spec_type=row.get("spec_type"),
                draft_model_path=row.get("draft_model_path"),
                model_path=row.get("model_path"),
                cpu_intent=numa_intent.get("cpu_list"),
            )
        )

    return sorted(inventory, key=lambda item: (item.primary_role, item.port))


def _log_path_for_port(logs_dir: Path, port: int) -> Path:
    candidates = (
        logs_dir / f"worker-explore-{port}.log",
        logs_dir / f"llama-server-{port}.log",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _build_port_report(inventory: PortInventory, logs_dir: Path) -> PortReport:
    log = parse_log(_log_path_for_port(logs_dir, inventory.port))
    cumulative = log.latest_cumulative

    if cumulative is not None:
        token_rate = cumulative.token_acceptance_rate
        draft_rate = cumulative.draft_acceptance_rate
        generated_tokens = cumulative.generated_tokens
        accepted_tokens = cumulative.accepted_tokens
        generated_drafts = cumulative.generated_drafts
        accepted_drafts = cumulative.accepted_drafts
        mean_acceptance_length = cumulative.mean_acceptance_length
        per_position_rates = cumulative.per_position_rates
        evidence_source = "latest_cumulative_stats"
    elif log.task_line_count > 0:
        token_rate = log.task_token_acceptance_rate
        draft_rate = None
        generated_tokens = log.task_generated_tokens
        accepted_tokens = log.task_accepted_tokens
        generated_drafts = None
        accepted_drafts = None
        mean_acceptance_length = log.latest_task.mean_acceptance_length if log.latest_task else None
        per_position_rates = log.latest_task.per_position_rates if log.latest_task else []
        evidence_source = "summed_task_lines"
    else:
        token_rate = None
        draft_rate = None
        generated_tokens = 0
        accepted_tokens = 0
        generated_drafts = None
        accepted_drafts = None
        mean_acceptance_length = None
        per_position_rates = []
        evidence_source = "none"

    if not inventory.mtp_configured:
        status = "not_mtp_configured"
    elif log.has_acceptance_evidence:
        status = "ok"
    elif log.no_spec_implementation:
        status = "spec_no_implementation"
    elif not log.exists:
        status = "missing_log"
    else:
        status = "missing_acceptance_evidence"

    return PortReport(
        port=inventory.port,
        primary_role=inventory.primary_role,
        registry_roles=inventory.registry_roles,
        pid=inventory.pid,
        spec_type=inventory.spec_type,
        mtp_configured=inventory.mtp_configured,
        model_path=inventory.model_path,
        draft_model_path=inventory.draft_model_path,
        cpu_intent=inventory.cpu_intent,
        log=log,
        status=status,
        token_acceptance_rate=token_rate,
        draft_acceptance_rate=draft_rate,
        generated_tokens=generated_tokens,
        accepted_tokens=accepted_tokens,
        generated_drafts=generated_drafts,
        accepted_drafts=accepted_drafts,
        mean_acceptance_length=mean_acceptance_length,
        per_position_rates=per_position_rates,
        evidence_source=evidence_source,
    )


def _role_report(role: str, ports: list[PortReport], min_lines: int) -> RoleReport:
    mtp_ports = [port.port for port in ports if port.mtp_configured]
    evidence_ports = [port.port for port in ports if port.mtp_configured and port.log.has_acceptance_evidence]
    missing_ports = [port.port for port in ports if port.mtp_configured and not port.log.has_acceptance_evidence]
    generated_tokens = sum(port.generated_tokens for port in ports if port.mtp_configured)
    accepted_tokens = sum(port.accepted_tokens for port in ports if port.mtp_configured)
    generated_drafts = sum(port.generated_drafts or 0 for port in ports if port.mtp_configured)
    accepted_drafts = sum(port.accepted_drafts or 0 for port in ports if port.mtp_configured)
    acceptance_lines = sum(port.log.task_line_count for port in ports if port.mtp_configured)
    cumulative_lines = sum(port.log.cumulative_line_count for port in ports if port.mtp_configured)
    registry_roles = sorted({alias for port in ports for alias in port.registry_roles})

    token_rate = accepted_tokens / generated_tokens if generated_tokens > 0 else None
    draft_rate = accepted_drafts / generated_drafts if generated_drafts > 0 else None

    if not mtp_ports:
        status = "not_mtp_configured"
    elif acceptance_lines + cumulative_lines < min_lines:
        status = "missing_acceptance_evidence"
    elif missing_ports:
        status = "ok_partial_port_traffic"
    else:
        status = "ok"

    return RoleReport(
        role=role,
        ports=[port.port for port in ports],
        registry_roles=registry_roles,
        mtp_configured_ports=mtp_ports,
        evidence_ports=evidence_ports,
        missing_evidence_ports=missing_ports,
        acceptance_line_count=acceptance_lines,
        cumulative_line_count=cumulative_lines,
        generated_tokens=generated_tokens,
        accepted_tokens=accepted_tokens,
        generated_drafts=generated_drafts,
        accepted_drafts=accepted_drafts,
        token_acceptance_rate=token_rate,
        draft_acceptance_rate=draft_rate,
        status=status,
    )


def build_report(
    *,
    attestation_path: Path = DEFAULT_ATTESTATION,
    logs_dir: Path = DEFAULT_LOGS_DIR,
    min_lines_per_mtp_role: int = 1,
) -> AcceptanceReport:
    """Build an MTP acceptance report from current local artifacts."""
    inventory = load_port_inventory(attestation_path)
    port_reports = [_build_port_report(item, logs_dir) for item in inventory]

    by_role: dict[str, list[PortReport]] = defaultdict(list)
    for port in port_reports:
        by_role[port.primary_role].append(port)
    role_reports = [
        _role_report(role, sorted(ports, key=lambda port: port.port), min_lines_per_mtp_role)
        for role, ports in sorted(by_role.items())
    ]

    mtp_roles = [role for role in role_reports if role.mtp_configured_ports]
    failed_roles = [role.role for role in mtp_roles if role.status == "missing_acceptance_evidence"]
    evidence_roles = [role.role for role in mtp_roles if role.evidence_ports]
    generated_tokens = sum(role.generated_tokens for role in mtp_roles)
    accepted_tokens = sum(role.accepted_tokens for role in mtp_roles)

    notes = [
        "Rates are process/log aggregates. Shared ports such as 8070 cannot be split by alias role "
        "unless server acceptance lines are joined to request-level role telemetry.",
        "Roles without draft-mtp in the current serving attestation are reported as not_mtp_configured.",
    ]
    summary = {
        "port_count": len(port_reports),
        "role_count": len(role_reports),
        "mtp_configured_role_count": len(mtp_roles),
        "roles_with_acceptance_evidence": evidence_roles,
        "failed_mtp_roles": failed_roles,
        "generated_tokens": generated_tokens,
        "accepted_tokens": accepted_tokens,
        "token_acceptance_rate": accepted_tokens / generated_tokens if generated_tokens > 0 else None,
    }

    return AcceptanceReport(
        generated_at=datetime.now(UTC).isoformat(),
        attestation_path=str(attestation_path),
        logs_dir=str(logs_dir),
        min_lines_per_mtp_role=min_lines_per_mtp_role,
        summary=summary,
        roles=role_reports,
        ports=port_reports,
        notes=notes,
    )


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def render_markdown(report: AcceptanceReport) -> str:
    """Render a concise Markdown report."""
    lines = [
        "# Live MTP Acceptance Report",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Attestation: `{report.attestation_path}`",
        f"- Logs: `{report.logs_dir}`",
        f"- Minimum evidence lines per MTP role: `{report.min_lines_per_mtp_role}`",
        "",
        "## Summary",
        "",
        f"- MTP-configured roles with evidence: {', '.join(report.summary['roles_with_acceptance_evidence']) or 'none'}",
        f"- Failed MTP roles: {', '.join(report.summary['failed_mtp_roles']) or 'none'}",
        f"- Aggregate token acceptance: `{_fmt_rate(report.summary['token_acceptance_rate'])}` "
        f"({report.summary['accepted_tokens']} accepted / {report.summary['generated_tokens']} generated)",
        "",
        "## Role Aggregates",
        "",
        "| Role | Status | Ports | Evidence ports | Token alpha | Draft alpha | Lines | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in report.roles:
        token_pair = f"{role.accepted_tokens}/{role.generated_tokens}" if role.generated_tokens else "-"
        lines.append(
            "| "
            f"{role.role} | {role.status} | {','.join(map(str, role.ports)) or '-'} | "
            f"{','.join(map(str, role.evidence_ports)) or '-'} | {_fmt_rate(role.token_acceptance_rate)} | "
            f"{_fmt_rate(role.draft_acceptance_rate)} | {role.acceptance_line_count + role.cumulative_line_count} | "
            f"{token_pair} |"
        )

    lines.extend(
        [
            "",
            "## Port Details",
            "",
            "| Port | Primary role | Registry roles | Status | Token alpha | Source | Log |",
            "|---:|---|---|---|---:|---|---|",
        ]
    )
    for port in report.ports:
        lines.append(
            "| "
            f"{port.port} | {port.primary_role} | {','.join(port.registry_roles) or '-'} | {port.status} | "
            f"{_fmt_rate(port.token_acceptance_rate)} | {port.evidence_source} | {Path(port.log.path).name} |"
        )

    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    lines.append("")
    return "\n".join(lines)


def _default_report_path(suffix: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "orchestration" / "reports" / f"mtp_acceptance_report_{stamp}.{suffix}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--min-lines-per-role", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--no-write-defaults", action="store_true", help="Print only unless explicit output paths are given.")
    parser.add_argument("--no-strict", action="store_true", help="Return success even when configured MTP roles lack evidence.")
    args = parser.parse_args(argv)

    report = build_report(
        attestation_path=args.attestation,
        logs_dir=args.logs_dir,
        min_lines_per_mtp_role=args.min_lines_per_role,
    )
    json_payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    markdown_payload = render_markdown(report)

    output_json = args.output_json
    output_md = args.output_md
    if not args.no_write_defaults:
        output_json = output_json or _default_report_path("json")
        output_md = output_md or Path(str(output_json).removesuffix(".json") + ".md")

    if output_json is not None:
        _write_text(output_json, json_payload + "\n")
    if output_md is not None:
        _write_text(output_md, markdown_payload)

    failed_roles = report.summary["failed_mtp_roles"]
    print(
        json.dumps(
            {
                "generated_at": report.generated_at,
                "roles_with_acceptance_evidence": report.summary["roles_with_acceptance_evidence"],
                "failed_mtp_roles": failed_roles,
                "token_acceptance_rate": report.summary["token_acceptance_rate"],
                "output_json": str(output_json) if output_json else None,
                "output_md": str(output_md) if output_md else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failed_roles and not args.no_strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
