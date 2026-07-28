#!/usr/bin/env python3
"""Guarded preflight probe for the GPU-resident shadow lane (plan-only default).

Pattern: scripts/benchmark/eval_batch_serving_probe.py (the E2 activation-probe
precedent) applied to the gpu_shadow_lane scaffold. When the operator runs it,
the probe verifies — WITHOUT touching any production process, launching any
server, or sending any inference:

  - VRAM budget on ROCm0 (rocm-smi meminfo) covers the planned tenant
    residency + KV estimate from orchestration/gpu_shadow_lane_np_ceiling.yaml
  - no foreign GPU compute PIDs (rocm-smi showpids; the operator-owned
    external process precedent means this is a hard blocker unless explicitly
    acknowledged)
  - KFD state (/dev/kfd + /sys/class/kfd topology)
  - production HIP binary reports version 10107 (67a433bf4) — the v8 freeze
  - the planned host cpuset (184-191, the GPU host-thread SMT rule) does not
    overlap any LIVE production llama-server's affinity (refuses if a
    production CPU lane would be disturbed; static NUMA_CONFIG co-tenants of
    the SMT slice are reported as warnings)
  - the (np, per-slot context) plan is within the validated np_ceiling policy

Mutation scope: NONE in plan mode beyond writing the report directory.
``--apply`` adds exactly one artifact — ``preflight_attestation.json`` inside
the report directory, consumed by the activation choreography
(docs/gpu-shadow-lane.md) — and is refused while any blocker stands. The probe
never starts, stops, signals, or renices a process.

Exit codes: 0 = clean; 75 = blocked (matches the eval_batch probe convention);
argparse's 2 for usage errors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from scripts.server.gpu_shadow_lane import (
    LANE_BINARY,
    LANE_BINARY_COMMIT,
    LANE_BINARY_VERSION,
    LANE_DEVICE,
    LANE_HOST_CPUSET,
    LANE_PORT,
    NpCeilingPolicy,
    TENANT_CANDIDATE_ID,
    build_tenant_launch_plan,
    estimated_dynamic_gib,
    lane_enabled,
    load_np_ceiling_policy,
    np_ceiling,
)
from scripts.server.stack_numa import NUMA_CONFIG
from src.features import Features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "orchestration" / "reports"
KFD_DEV = Path("/dev/kfd")
KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ── Pure helpers (unit-tested; no subprocess, no filesystem) ─────────────────


def parse_cpu_list(spec: str) -> set[int]:
    """Parse a taskset-style cpu list ("72-95,168-191") into a set of ints."""
    cpus: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(chunk))
    return cpus


def cpuset_overlap(a: str, b: str) -> set[int]:
    return parse_cpu_list(a) & parse_cpu_list(b)


def static_smt_overlap_roles(host_cpuset: str = LANE_HOST_CPUSET) -> dict[str, list[int]]:
    """Production roles whose NUMA_CONFIG cpusets share CPUs with the lane's
    host cpuset (static topology fact — informational, not a live blocker)."""
    lane_cpus = parse_cpu_list(host_cpuset)
    overlaps: dict[str, list[int]] = {}
    for role, cfg in NUMA_CONFIG.items():
        ports = []
        for cpu_list, port, _threads in cfg.get("instances", []):
            if parse_cpu_list(cpu_list) & lane_cpus:
                ports.append(port)
        if ports:
            overlaps[role] = ports
    return overlaps


def parse_rocm_meminfo(payload: Any) -> tuple[float, float] | None:
    """Extract (total_gib, used_gib) from `rocm-smi --showmeminfo vram --json`."""
    if not isinstance(payload, dict):
        return None
    for record in payload.values():
        if not isinstance(record, dict):
            continue
        total = used = None
        for key, value in record.items():
            key_l = str(key).lower()
            if "vram total memory" in key_l:
                total = value
            elif "used" in key_l and "vram" in key_l:
                used = value
        if total is not None and used is not None:
            try:
                return float(total) / float(1 << 30), float(used) / float(1 << 30)
            except (TypeError, ValueError):
                return None
    return None


_PID_KEY_RE = re.compile(r"pid\D*?(\d+)", re.IGNORECASE)


def parse_rocm_pids(payload: Any) -> list[int]:
    """Extract GPU compute PIDs from `rocm-smi --showpids --json` (tolerant).

    rocm-smi encodes the PID in the mapping KEY ("PID 12345": [...]); some
    variants use {"pid": N} records. Both are accepted.
    """
    pids: set[int] = set()

    def _walk(node: Any, key_hint: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key)
                match = _PID_KEY_RE.search(key_text)
                if match:
                    pids.add(int(match.group(1)))
                _walk(value, key_text)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item, key_hint)
        else:
            if key_hint.lower() == "pid":
                try:
                    pid = int(str(node).strip())
                except (TypeError, ValueError):
                    return
                if pid > 0:
                    pids.add(pid)

    _walk(payload)
    return sorted(pid for pid in pids if pid > 0)


@dataclass
class LanePlan:
    tenant_id: str
    model_path: str
    model_vram_gib: float
    np_slots: int
    slot_context_tokens: int
    port: int
    host_cpuset: str
    dynamic_budget_gib: float
    np_ceiling: int | None
    estimated_dynamic_gib: float | None
    launch_argv: list[str] = field(default_factory=list)


@dataclass
class PreflightFacts:
    flag_enabled: bool
    binary_exists: bool
    binary_version_output: str | None
    kfd_dev_present: bool
    kfd_topology_present: bool
    vram_total_gib: float | None
    vram_used_gib: float | None
    gpu_compute_pids: list[int]
    live_llama_processes: list[dict[str, Any]]  # {pid, port?, cpus_allowed_list, overlap}
    model_file_exists: bool
    static_smt_overlaps: dict[str, list[int]]
    errors: list[str] = field(default_factory=list)


def evaluate_preflight(
    facts: PreflightFacts,
    plan: LanePlan,
    *,
    require_enabled: bool = False,
    allow_existing_gpu_pids: bool = False,
    allow_smt_overlap: bool = False,
) -> tuple[list[str], list[str]]:
    """Compute (blockers, warnings). Pure — no I/O."""
    blockers: list[str] = []
    warnings: list[str] = []

    if require_enabled and not facts.flag_enabled:
        blockers.append("gpu_shadow_lane feature flag is not enabled on this environment")
    elif not facts.flag_enabled:
        warnings.append("gpu_shadow_lane feature flag is off (expected pre-activation)")

    if not facts.binary_exists:
        blockers.append(f"production HIP binary missing: {LANE_BINARY}")
    else:
        version_output = facts.binary_version_output or ""
        if LANE_BINARY_VERSION not in version_output or LANE_BINARY_COMMIT not in version_output:
            blockers.append(
                "HIP binary version mismatch: expected "
                f"{LANE_BINARY_VERSION} ({LANE_BINARY_COMMIT}), got: "
                f"{version_output.strip()[:200] or '<no output>'}"
            )

    if not facts.kfd_dev_present or not facts.kfd_topology_present:
        blockers.append("KFD state not healthy (/dev/kfd or kfd topology missing)")

    if facts.vram_total_gib is None or facts.vram_used_gib is None:
        blockers.append("could not attest VRAM via rocm-smi meminfo")
    else:
        free_gib = facts.vram_total_gib - facts.vram_used_gib
        required = plan.estimated_dynamic_gib
        if required is None:
            warnings.append(
                "no KV arithmetic model for tenant; VRAM check limited to model residency"
            )
            required = 0.0
        # model weights + dynamic estimate must fit in currently-free VRAM
        if free_gib < plan.model_vram_gib + required:
            blockers.append(
                f"VRAM budget short: free {free_gib:.1f} GiB < model "
                f"{plan.model_vram_gib:.1f} GiB + dynamic estimate {required:.1f} GiB"
            )

    if facts.gpu_compute_pids:
        message = (
            f"foreign GPU compute PIDs present: {facts.gpu_compute_pids} "
            "(operator-owned processes are never killed; wait for them to vacate)"
        )
        if allow_existing_gpu_pids:
            warnings.append(message)
        else:
            blockers.append(message)

    disturbed = [
        proc for proc in facts.live_llama_processes if proc.get("overlap")
    ]
    if disturbed:
        message = (
            "planned host cpuset "
            f"{plan.host_cpuset} overlaps LIVE production llama-server affinity: "
            + ", ".join(
                f"pid {proc.get('pid')} (cpus {proc.get('cpus_allowed_list')}, "
                f"overlap {sorted(proc.get('overlap', []))})"
                for proc in disturbed
            )
        )
        if allow_smt_overlap:
            warnings.append(message)
        else:
            blockers.append(message)

    if facts.static_smt_overlaps:
        warnings.append(
            "static NUMA_CONFIG co-tenants of the lane SMT slice (contention "
            f"recert required at activation Step 4): {facts.static_smt_overlaps}"
        )

    if not facts.model_file_exists:
        blockers.append(f"tenant model file missing: {plan.model_path}")

    if plan.np_ceiling is None:
        blockers.append(
            f"np_ceiling policy has no validated operating point for tenant "
            f"{plan.tenant_id} at slot_context={plan.slot_context_tokens} within "
            f"budget {plan.dynamic_budget_gib} GiB — refuse, never extrapolate"
        )
    elif plan.np_slots > plan.np_ceiling:
        blockers.append(
            f"planned -np {plan.np_slots} exceeds validated ceiling {plan.np_ceiling} "
            f"for slot_context={plan.slot_context_tokens}"
        )

    blockers.extend(facts.errors)
    return blockers, warnings


# ── Gather (subprocess/filesystem; not exercised by unit tests) ──────────────


def _run_json(cmd: list[str], timeout_s: float = 10.0) -> tuple[Any, str | None]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{cmd[0]}: {exc}"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        return None, f"{cmd[0]}: non-JSON output ({proc.stdout[:120]!r})"


def _binary_version_output() -> str | None:
    if not LANE_BINARY.exists():
        return None
    try:
        proc = subprocess.run(
            [str(LANE_BINARY), "--version"],
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def _live_llama_processes(host_cpuset: str) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "llama-server"], capture_output=True, text=True, check=False
        )
    except OSError:
        return []
    lane_cpus = parse_cpu_list(host_cpuset)
    records: list[dict[str, Any]] = []
    for token in proc.stdout.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        status_path = Path(f"/proc/{pid}/status")
        cpus_allowed = None
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    cpus_allowed = line.split(":", 1)[1].strip()
                    break
        except OSError:
            continue
        overlap = sorted(parse_cpu_list(cpus_allowed) & lane_cpus) if cpus_allowed else []
        records.append(
            {"pid": pid, "cpus_allowed_list": cpus_allowed, "overlap": overlap}
        )
    return records


def gather_facts(plan: LanePlan) -> PreflightFacts:
    errors: list[str] = []
    meminfo_payload, meminfo_error = _run_json(
        ["rocm-smi", "--showmeminfo", "vram", "--json"]
    )
    if meminfo_error:
        errors.append(meminfo_error)
    meminfo = parse_rocm_meminfo(meminfo_payload)
    pids_payload, pids_error = _run_json(["rocm-smi", "--showpids", "--json"])
    if pids_error:
        errors.append(pids_error)
    return PreflightFacts(
        flag_enabled=lane_enabled(),
        binary_exists=LANE_BINARY.exists(),
        binary_version_output=_binary_version_output(),
        kfd_dev_present=KFD_DEV.exists(),
        kfd_topology_present=KFD_TOPOLOGY.exists(),
        vram_total_gib=meminfo[0] if meminfo else None,
        vram_used_gib=meminfo[1] if meminfo else None,
        gpu_compute_pids=parse_rocm_pids(pids_payload),
        live_llama_processes=_live_llama_processes(plan.host_cpuset),
        model_file_exists=Path(plan.model_path).exists(),
        static_smt_overlaps=static_smt_overlap_roles(plan.host_cpuset),
        errors=errors,
    )


# ── Plan construction + reporting ────────────────────────────────────────────


def build_plan(policy: NpCeilingPolicy, args: argparse.Namespace) -> LanePlan:
    tenant = policy.tenants[args.tenant]
    rows = {row.name: row for row in tenant.budgets}
    if args.budget_profile not in rows:
        raise KeyError(
            f"budget profile {args.budget_profile!r} not in policy for {args.tenant!r} "
            f"(have: {sorted(rows)})"
        )
    budget = rows[args.budget_profile].dynamic_budget_gib
    ceiling = np_ceiling(
        policy,
        args.tenant,
        dynamic_budget_gib=budget,
        slot_context_tokens=args.slot_context,
    )
    return LanePlan(
        tenant_id=args.tenant,
        model_path=tenant.model_path,
        model_vram_gib=tenant.model_vram_gib,
        np_slots=args.np,
        slot_context_tokens=args.slot_context,
        port=args.port,
        host_cpuset=LANE_HOST_CPUSET,
        dynamic_budget_gib=budget,
        np_ceiling=ceiling,
        estimated_dynamic_gib=estimated_dynamic_gib(
            tenant, np_slots=args.np, slot_context_tokens=args.slot_context
        ),
        launch_argv=build_tenant_launch_plan(
            model_path=tenant.model_path,
            np_slots=args.np,
            slot_context_tokens=args.slot_context,
            port=args.port,
        ),
    )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    plan = report["plan"]
    lines = [
        "# GPU Shadow Lane Preflight",
        "",
        f"- status: `{report['status']}`",
        f"- mode: `{report['mode']}`",
        f"- tenant: `{plan['tenant_id']}`",
        f"- plan: `-np {plan['np_slots']}` x `{plan['slot_context_tokens']}` ctx on port `{plan['port']}` ({LANE_DEVICE})",
        f"- np_ceiling: `{plan['np_ceiling']}` (budget `{plan['dynamic_budget_gib']}` GiB)",
        f"- host cpuset: `{plan['host_cpuset']}`",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(["", "## Planned launch argv (NOT executed by this probe)", ""])
    lines.append("```bash\n" + " ".join(plan["launch_argv"]) + "\n```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    return json_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default=TENANT_CANDIDATE_ID)
    parser.add_argument(
        "--budget-profile",
        default="phase2_resident_set",
        help="np_ceiling budget row to plan against (e.g. solo_resident, phase2_resident_set)",
    )
    parser.add_argument("--np", type=int, default=8)
    parser.add_argument("--slot-context", type=int, default=8192)
    parser.add_argument("--port", type=int, default=LANE_PORT)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write preflight_attestation.json for the activation choreography. "
            "Refused while any blocker stands. This is the probe's ONLY mutation."
        ),
    )
    parser.add_argument("--require-enabled", action="store_true")
    parser.add_argument(
        "--allow-existing-gpu-pids",
        action="store_true",
        help="Downgrade foreign-GPU-PID blocker to a warning (explicit operator call).",
    )
    parser.add_argument(
        "--allow-smt-overlap",
        action="store_true",
        help="Downgrade live host-cpuset overlap blocker to a warning (explicit operator call).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = utc_stamp()
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"gpu_shadow_lane_preflight_{stamp}")
    # The probe is itself the guarded tool: it reads the policy with an explicit
    # override so it can run BEFORE the flag is flipped, while the report always
    # records the environment's real flag state (facts.flag_enabled).
    policy = load_np_ceiling_policy(
        args.policy_path, feats=Features(gpu_shadow_lane=True)
    )
    plan = build_plan(policy, args)
    facts = gather_facts(plan)
    blockers, warnings = evaluate_preflight(
        facts,
        plan,
        require_enabled=args.require_enabled,
        allow_existing_gpu_pids=args.allow_existing_gpu_pids,
        allow_smt_overlap=args.allow_smt_overlap,
    )

    attested = False
    if args.apply and not blockers:
        attested = True
    status = "blocked" if blockers else ("attested" if attested else "plan_only_ok")
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "mode": "apply" if args.apply else "plan-only",
        "blockers": blockers,
        "warnings": warnings,
        "plan": asdict(plan),
        "facts": asdict(facts),
    }
    json_path, md_path = write_report(report, output_dir)
    if attested:
        attestation_path = output_dir / "preflight_attestation.json"
        attestation_path.write_text(
            json.dumps(
                {
                    "lane": "gpu_shadow_lane",
                    "attested_at": report["generated_at"],
                    "plan": report["plan"],
                    "facts_summary": {
                        "vram_free_gib": (
                            None
                            if facts.vram_total_gib is None or facts.vram_used_gib is None
                            else round(facts.vram_total_gib - facts.vram_used_gib, 2)
                        ),
                        "binary_version_ok": True,
                        "kfd_ok": True,
                        "gpu_compute_pids": facts.gpu_compute_pids,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if not args.summary_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {json_path}")
        print(f"wrote {md_path}")
    if blockers:
        return 75
    return 0


if __name__ == "__main__":
    sys.exit(main())
