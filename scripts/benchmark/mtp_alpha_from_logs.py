#!/usr/bin/env python3
"""Deterministic per-role MTP/spec-dec acceptance (alpha) from existing server logs.

Zero inference. This tool reads *already written* production llama-server logs
and computes, per serving role, the draft acceptance rate

    alpha = accepted_draft_tokens / proposed_draft_tokens

directly from the acceptance counters llama.cpp emits. It never launches a
server, a benchmark, or any inference; it only reads files on disk.

Two log-line shapes are parsed (both observed in production / research logs):

  Per-task slot line (headline alpha source):
    "<ts> I slot print_timing: id  0 | task 35853 | draft acceptance = 0.91000 "
    "(   91 accepted /   100 generated), mean acceptance length =  4.64, "
    "acceptance rate per position = (1.000, 0.960, 0.880, 0.800)"
  older/alt variant:
    "draft acceptance rate = 0.71901 (  174 accepted /   242 generated)"

  Cumulative per-spec-type statistics line (secondary breakdown):
    "<ts> I statistics        draft-mtp: #calls(b,g,a) = 2 56 56, #gen drafts = 56, "
    "#acc drafts = 55, #gen tokens = 224, #acc tokens = 190, #mean acc len = 4.39, "
    "#acc rate/pos = (...), dur(b,g,a) = ..."
  older/alt variant (no #mean acc len / #acc rate/pos):
    "statistics tree: #calls(b,g,a) = 1 81 66, #gen drafts = 81, #acc drafts = 66, "
    "#gen tokens = 243, #acc tokens = 174, dur(b,g,a) = ..."

Headline per-role alpha = sum(accepted_tokens) / sum(generated_tokens) over the
per-task lines found in that role's log(s). n = number of per-task lines. The
per-task "draft acceptance" value is the slot-level acceptance combined over all
speculative implementations active for the task, which is exactly what the role
achieves in production. A per-spec-type breakdown (draft-mtp vs ngram-mod vs ...)
is reported alongside from the cumulative statistics lines.

Role attribution: each log file name carries a port (llama-server-<port>.log,
worker-explore-<port>.log); the serving attestation maps port -> role. An
explicit "role=path" mapping can be supplied to bypass the attestation.

LOUD FAIL contract (handoff requirement): if a requested role has zero matching
acceptance lines -- or, when no role is requested, if the discovered logs have
zero acceptance lines anywhere -- the tool prints a clear error and exits
non-zero. It never silently emits alpha=0 / an empty report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTESTATION = PROJECT_ROOT / "orchestration" / "attestation" / "latest.json"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"

# Exit codes.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NO_EVIDENCE = 3  # loud-fail: zero acceptance lines for a requested / all roles.

# Per-task slot acceptance line. Matches both "draft acceptance =" (v6 slot
# print_timing) and the older "draft acceptance rate =" variant.
PER_TASK_RE = re.compile(
    r"draft acceptance(?:\s+rate)?\s*=\s*(?P<rate>[0-9.]+)\s*"
    r"\(\s*(?P<accepted>\d+)\s+accepted\s*/\s*(?P<generated>\d+)\s+generated\s*\)"
)
# Optional per-task extras (present in the v6 format only).
_MEAN_ACC_LEN_RE = re.compile(r"mean acceptance length\s*=\s*(?P<mean>[0-9.]+)")
# Task id, when present on the same line.
_TASK_ID_RE = re.compile(r"\btask\s+(?P<task_id>\d+)\b")

# Cumulative per-spec-type statistics line. The spec label sits between
# "statistics" and the first colon; counter fields are pulled individually so
# the older variant (missing #mean acc len / #acc rate/pos) still parses.
_STATS_LABEL_RE = re.compile(r"\bstatistics\s+(?P<spec>[A-Za-z0-9_.\-]+)\s*:")
_GEN_DRAFTS_RE = re.compile(r"#gen drafts\s*=\s*(?P<v>\d+)")
_ACC_DRAFTS_RE = re.compile(r"#acc drafts\s*=\s*(?P<v>\d+)")
_GEN_TOKENS_RE = re.compile(r"#gen tokens\s*=\s*(?P<v>\d+)")
_ACC_TOKENS_RE = re.compile(r"#acc tokens\s*=\s*(?P<v>\d+)")
_STATS_MEAN_RE = re.compile(r"#mean acc len\s*=\s*(?P<v>[0-9.]+)")

# Log file names that carry a serving port.
_PORT_FROM_NAME_RE = re.compile(r"(?:llama-server|worker-explore|worker)[-_](?P<port>\d+)\.log$")


class NoAcceptanceEvidenceError(RuntimeError):
    """Raised when a requested role (or, globally, all logs) has no alpha lines."""


@dataclass
class TaskAccept:
    """One per-task slot acceptance record."""

    task_id: int | None
    accepted_tokens: int
    generated_tokens: int
    rate: float
    mean_acceptance_length: float | None
    line_number: int


@dataclass
class CumulativeStat:
    """One cumulative per-spec-type statistics record (a running total)."""

    spec_type: str
    generated_drafts: int
    accepted_drafts: int
    generated_tokens: int
    accepted_tokens: int
    mean_acceptance_length: float | None
    line_number: int


@dataclass
class LogParse:
    """Acceptance evidence parsed from a single log file."""

    path: str
    exists: bool
    size_bytes: int | None = None
    mtime_utc: str | None = None
    task_line_count: int = 0
    accepted_tokens: int = 0
    generated_tokens: int = 0
    # Best (max #gen tokens) cumulative running-total per spec type.
    spec_best: dict[str, CumulativeStat] = field(default_factory=dict)

    @property
    def alpha(self) -> float | None:
        if self.generated_tokens <= 0:
            return None
        return self.accepted_tokens / self.generated_tokens


@dataclass
class RoleAlpha:
    """Per-role aggregate alpha over one or more logs."""

    role: str
    ports: list[int]
    logs: list[str]
    task_line_count: int
    accepted_tokens: int
    generated_tokens: int
    spec_breakdown: dict[str, dict[str, Any]]

    @property
    def alpha(self) -> float | None:
        if self.generated_tokens <= 0:
            return None
        return self.accepted_tokens / self.generated_tokens


# --------------------------------------------------------------------------- #
# Line parsing
# --------------------------------------------------------------------------- #
def parse_task_line(line: str, line_number: int = 0) -> TaskAccept | None:
    """Parse a per-task slot 'draft acceptance' line. None if it does not match."""
    match = PER_TASK_RE.search(line)
    if not match:
        return None
    mean_match = _MEAN_ACC_LEN_RE.search(line)
    task_match = _TASK_ID_RE.search(line)
    return TaskAccept(
        task_id=int(task_match.group("task_id")) if task_match else None,
        accepted_tokens=int(match.group("accepted")),
        generated_tokens=int(match.group("generated")),
        rate=float(match.group("rate")),
        mean_acceptance_length=float(mean_match.group("mean")) if mean_match else None,
        line_number=line_number,
    )


def parse_cumulative_line(line: str, line_number: int = 0) -> CumulativeStat | None:
    """Parse a cumulative 'statistics <spec>: ...' line. None if it does not match."""
    label = _STATS_LABEL_RE.search(line)
    gen_tok = _GEN_TOKENS_RE.search(line)
    acc_tok = _ACC_TOKENS_RE.search(line)
    if not (label and gen_tok and acc_tok):
        return None
    gen_drafts = _GEN_DRAFTS_RE.search(line)
    acc_drafts = _ACC_DRAFTS_RE.search(line)
    mean = _STATS_MEAN_RE.search(line)
    return CumulativeStat(
        spec_type=label.group("spec"),
        generated_drafts=int(gen_drafts.group("v")) if gen_drafts else 0,
        accepted_drafts=int(acc_drafts.group("v")) if acc_drafts else 0,
        generated_tokens=int(gen_tok.group("v")),
        accepted_tokens=int(acc_tok.group("v")),
        mean_acceptance_length=float(mean.group("v")) if mean else None,
        line_number=line_number,
    )


def parse_log(path: Path) -> LogParse:
    """Parse a single log file for acceptance evidence (no inference)."""
    result = LogParse(path=str(path), exists=path.exists())
    if not path.exists():
        return result

    stat = path.stat()
    result.size_bytes = stat.st_size
    result.mtime_utc = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            task = parse_task_line(line, line_number)
            if task is not None:
                result.task_line_count += 1
                result.accepted_tokens += task.accepted_tokens
                result.generated_tokens += task.generated_tokens
                continue
            cumulative = parse_cumulative_line(line, line_number)
            if cumulative is not None:
                prior = result.spec_best.get(cumulative.spec_type)
                # Keep the furthest-along running total per spec type (max #gen
                # tokens) so we do not double-count the repeated running totals.
                if prior is None or cumulative.generated_tokens >= prior.generated_tokens:
                    result.spec_best[cumulative.spec_type] = cumulative

    return result


# --------------------------------------------------------------------------- #
# Role attribution
# --------------------------------------------------------------------------- #
def port_from_log_name(path: Path) -> int | None:
    """Extract the serving port encoded in a log file name, if any."""
    match = _PORT_FROM_NAME_RE.search(path.name)
    return int(match.group("port")) if match else None


def load_port_roles(attestation_path: Path) -> dict[int, str]:
    """Map serving port -> role from the serving attestation. {} if unreadable."""
    if not attestation_path.exists():
        return {}
    try:
        data = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = ((data.get("sections") or {}).get("serving_config") or [])
    mapping: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            port = int(row.get("port"))
        except (TypeError, ValueError):
            continue
        role = str((row.get("numa_intent") or {}).get("role") or "").strip()
        if role:
            mapping[port] = role
    return mapping


def resolve_role(path: Path, port_roles: dict[int, str], explicit: str | None) -> tuple[str, int | None]:
    """Resolve (role, port) for a log file. explicit role wins over attestation."""
    port = port_from_log_name(path)
    if explicit:
        return explicit, port
    if port is not None and port in port_roles:
        return port_roles[port], port
    if port is not None:
        return f"port_{port}", port
    return path.stem, None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def discover_logs(logs_dir: Path) -> list[Path]:
    """Discover candidate server logs in a directory (sorted, deterministic)."""
    if not logs_dir.is_dir():
        return []
    found: set[Path] = set()
    for pattern in ("llama-server-*.log", "worker-explore-*.log", "worker-*.log"):
        found.update(logs_dir.glob(pattern))
    return sorted(found)


def aggregate_roles(
    parsed: list[tuple[str, int | None, LogParse]],
) -> dict[str, RoleAlpha]:
    """Aggregate parsed logs into per-role alpha records."""
    roles: dict[str, RoleAlpha] = {}
    for role, port, log in parsed:
        entry = roles.get(role)
        if entry is None:
            entry = RoleAlpha(
                role=role,
                ports=[],
                logs=[],
                task_line_count=0,
                accepted_tokens=0,
                generated_tokens=0,
                spec_breakdown={},
            )
            roles[role] = entry
        if port is not None and port not in entry.ports:
            entry.ports.append(port)
        entry.logs.append(log.path)
        entry.task_line_count += log.task_line_count
        entry.accepted_tokens += log.accepted_tokens
        entry.generated_tokens += log.generated_tokens
        for spec, stat in log.spec_best.items():
            bucket = entry.spec_breakdown.setdefault(
                spec,
                {"generated_tokens": 0, "accepted_tokens": 0,
                 "generated_drafts": 0, "accepted_drafts": 0},
            )
            bucket["generated_tokens"] += stat.generated_tokens
            bucket["accepted_tokens"] += stat.accepted_tokens
            bucket["generated_drafts"] += stat.generated_drafts
            bucket["accepted_drafts"] += stat.accepted_drafts
    # Finalize spec-type alpha.
    for entry in roles.values():
        for spec, bucket in entry.spec_breakdown.items():
            gen = bucket["generated_tokens"]
            gen_d = bucket["generated_drafts"]
            bucket["token_alpha"] = (bucket["accepted_tokens"] / gen) if gen > 0 else None
            bucket["draft_alpha"] = (bucket["accepted_drafts"] / gen_d) if gen_d > 0 else None
    for entry in roles.values():
        entry.ports.sort()
    return roles


def build_report(
    logs_dir: Path,
    attestation_path: Path,
    requested_roles: list[str],
    explicit_logs: list[tuple[str | None, Path]],
    min_lines: int,
) -> dict[str, Any]:
    """Parse logs and build the per-role alpha report.

    Raises NoAcceptanceEvidenceError when the loud-fail contract is violated.
    """
    port_roles = load_port_roles(attestation_path)

    log_specs: list[tuple[str | None, Path]] = list(explicit_logs)
    if not log_specs:
        log_specs = [(None, p) for p in discover_logs(logs_dir)]

    parsed: list[tuple[str, int | None, LogParse]] = []
    for explicit_role, path in log_specs:
        role, port = resolve_role(path, port_roles, explicit_role)
        parsed.append((role, port, parse_log(path)))

    roles = aggregate_roles(parsed)

    # --- LOUD FAIL contract ------------------------------------------------ #
    if requested_roles:
        missing: list[str] = []
        for role in requested_roles:
            entry = roles.get(role)
            if entry is None or entry.task_line_count < min_lines:
                have = 0 if entry is None else entry.task_line_count
                missing.append(f"{role} (found {have} acceptance line(s), need >= {min_lines})")
        if missing:
            with_evidence = sorted(r for r, e in roles.items() if e.task_line_count > 0)
            known = ", ".join(with_evidence) or "<none>"
            raise NoAcceptanceEvidenceError(
                "No MTP/spec-dec acceptance evidence for requested role(s): "
                + "; ".join(missing)
                + f". Roles with acceptance evidence in scanned logs: {known}."
            )
        selected = {r: roles[r] for r in requested_roles}
    else:
        total_lines = sum(e.task_line_count for e in roles.values())
        if total_lines < max(1, min_lines):
            scanned = len(log_specs)
            raise NoAcceptanceEvidenceError(
                f"No MTP/spec-dec acceptance lines found in any of {scanned} scanned log(s) "
                f"under {logs_dir}. Refusing to emit an empty/zero-alpha report."
            )
        selected = roles

    role_payload = []
    for role in sorted(selected):
        entry = selected[role]
        role_payload.append(
            {
                "role": role,
                "alpha": entry.alpha,
                "n_task_lines": entry.task_line_count,
                "accepted_tokens": entry.accepted_tokens,
                "generated_tokens": entry.generated_tokens,
                "ports": entry.ports,
                "logs": entry.logs,
                "spec_breakdown": entry.spec_breakdown,
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "tool": "mtp_alpha_from_logs",
        "zero_inference": True,
        "logs_dir": str(logs_dir),
        "attestation_path": str(attestation_path),
        "attestation_used": bool(port_roles),
        "min_lines_per_role": min_lines,
        "requested_roles": requested_roles,
        "scanned_logs": [str(p) for _, p in log_specs],
        "roles": role_payload,
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_table(report: dict[str, Any]) -> str:
    """Render a compact per-role alpha table."""
    rows = report["roles"]
    header = f"{'role':<24} {'alpha':>8} {'n':>6} {'acc_tok':>10} {'gen_tok':>10}  spec-breakdown"
    lines = [header, "-" * len(header)]
    for row in rows:
        alpha = row["alpha"]
        alpha_s = f"{alpha:.4f}" if alpha is not None else "  n/a "
        specs = row["spec_breakdown"]
        spec_bits = []
        for spec in sorted(specs):
            ta = specs[spec].get("token_alpha")
            spec_bits.append(f"{spec}={ta:.3f}" if ta is not None else f"{spec}=n/a")
        lines.append(
            f"{row['role']:<24} {alpha_s:>8} {row['n_task_lines']:>6} "
            f"{row['accepted_tokens']:>10} {row['generated_tokens']:>10}  "
            + ", ".join(spec_bits)
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_explicit_log(value: str) -> tuple[str | None, Path]:
    """Parse a --log value of the form 'role=path' or 'path'."""
    if "=" in value:
        role, _, path = value.partition("=")
        role = role.strip()
        return (role or None), Path(path.strip())
    return None, Path(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic per-role MTP/spec-dec acceptance (alpha) from existing "
        "llama-server logs. Zero inference: reads files only.",
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"Directory of server logs to scan (default: {DEFAULT_LOGS_DIR}).",
    )
    parser.add_argument(
        "--attestation", type=Path, default=DEFAULT_ATTESTATION,
        help=f"Serving attestation for port->role mapping (default: {DEFAULT_ATTESTATION}).",
    )
    parser.add_argument(
        "--role", action="append", default=[], dest="roles", metavar="ROLE",
        help="Restrict to this role and LOUD-FAIL if it has no acceptance lines. Repeatable.",
    )
    parser.add_argument(
        "--log", action="append", default=[], dest="logs", metavar="[ROLE=]PATH",
        help="Explicit log file (optionally 'role=path'). Repeatable; bypasses --logs-dir discovery.",
    )
    parser.add_argument(
        "--min-lines", type=int, default=1,
        help="Minimum per-task acceptance lines required per role (default: 1).",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Write the full JSON report to this path.",
    )
    parser.add_argument(
        "--stdout-json", action="store_true",
        help="Emit the full JSON report to stdout instead of the table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    explicit_logs = [_parse_explicit_log(v) for v in args.logs]

    if args.logs:
        missing_paths = [str(p) for _, p in explicit_logs if not p.exists()]
        if missing_paths:
            print(f"error: log file(s) not found: {', '.join(missing_paths)}", file=sys.stderr)
            return EXIT_USAGE
    elif not args.logs_dir.is_dir():
        print(f"error: logs dir not found: {args.logs_dir}", file=sys.stderr)
        return EXIT_USAGE

    try:
        report = build_report(
            logs_dir=args.logs_dir,
            attestation_path=args.attestation,
            requested_roles=args.roles,
            explicit_logs=explicit_logs,
            min_lines=args.min_lines,
        )
    except NoAcceptanceEvidenceError as exc:
        print(f"LOUD-FAIL: {exc}", file=sys.stderr)
        return EXIT_NO_EVIDENCE

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.stdout_json:
        print(json.dumps(report, indent=2))
    else:
        print(render_table(report))
        if args.json_out is not None:
            print(f"\nJSON report written to {args.json_out}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
