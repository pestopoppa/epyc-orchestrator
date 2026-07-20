#!/usr/bin/env python3
"""B4 — Inference-batch preflight gate.

Composes the existing read-only host/registry probes into ONE attestation the
batch loop trusts before it fires a topology-gated entry. This tool never
mutates the stack, flags, or registry — it only reads and emits an attestation
artifact.

Composition (which existing probe is imported vs subprocessed, and why):

  IMPORTED (in-process, side-effect-free):
    * scripts.server.stack_numa.NUMA_CONFIG      — configured server ports, so the
                                                   optional --require-servers health
                                                   probe knows what to poll.
    * scripts.server.stack_health.wait_for_health — per-port /health probe.
    * src.scheduling.contention topology hash    — topology_hash.
    * hashlib over the live registry YAML        — registry_hash.

  SUBPROCESSED (standalone CLIs whose exit codes / artifacts ARE the contract;
  re-running them in-process would fight their argparse/sys.path and hide the
  exit-code semantics the loop relies on):
    * scripts/server/affinity_preflight.py       — writes the live-affinity
      artifact and returns live_affinity_verified.
    * scripts/validate/check_contention_matrix_fresh.py — exit 0 fresh / 2 stale
      or missing / 3 invalid. This is a hard preflight gate.
    * epyc-root scripts/session/health_check.sh --profile batch — structural
      batch health (exit 0).

Hash mapping:
    topology_hash  = 16-char live NUMA/contention-matrix topology fingerprint
                     [== inference-batch required_topology_hash]
    registry_hash  = sha256(live/lean orchestrator model_registry.yaml)

Attestation JSON shape (written to coordination/inference-batch/attestations/<ts>.json):
    {ts, topology_hash, registry_hash, expected_topology_hash,
     live_affinity_verified, contention_matrix_fresh, health_ok,
     checks: {live_affinity, health, topology, contention_matrix},
     overall: "PASS"|"FAIL", fail_reasons: [...]}

CLI:  python3 scripts/server/preflight_gate.py [--expected-topology-hash H]
                 [--require-servers] [--roles ...] [--max-age-days N]
                 [--output-dir DIR] [--no-write] [--json]
Importable:  from scripts.server.preflight_gate import attest
             att = attest(expected_topology_hash=..., require_servers=...)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ORCH = Path(__file__).resolve().parents[2]  # /mnt/raid0/llm/epyc-orchestrator

# Registry files (metadata hash sources).
LIVE_REGISTRY = ORCH / "orchestration" / "model_registry.yaml"
CONTENTION_MATRIX = ORCH / "orchestration" / "contention_matrix.yaml"

# Subprocessed probe scripts.
AFFINITY_SCRIPT = ORCH / "scripts" / "server" / "affinity_preflight.py"
CONTENTION_FRESH_SCRIPT = ORCH / "scripts" / "validate" / "check_contention_matrix_fresh.py"
ROOT_HEALTH_CHECK_SCRIPT = Path("/mnt/raid0/llm/epyc-root/scripts/session/health_check.sh")
HEALTH_CHECK_SCRIPT = (
    ROOT_HEALTH_CHECK_SCRIPT
    if ROOT_HEALTH_CHECK_SCRIPT.exists()
    else ORCH / "scripts" / "session" / "health_check.sh"
)

# Attestation output lives with the inference-batch coordination surface (epyc-root).
ATTEST_DIR = Path(
    "/mnt/raid0/llm/epyc-root/coordination/inference-batch/attestations"
)
PROJECT_VENV_PY = ORCH / ".venv/bin/python"


def _reexec_under_project_venv() -> None:
    """Run CLI invocations under the project venv so probe imports are stable."""

    if (
        PROJECT_VENV_PY.exists()
        and Path(sys.executable).resolve() != PROJECT_VENV_PY.resolve()
        and os.environ.get("ORCHESTRATOR_PREFLIGHT_REEXEC") != "1"
    ):
        os.environ["ORCHESTRATOR_PREFLIGHT_REEXEC"] = "1"
        os.execv(
            str(PROJECT_VENV_PY),
            [str(PROJECT_VENV_PY), __file__, *sys.argv[1:]],
        )


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], cwd: Path | None = None,
         timeout: float = 180.0) -> subprocess.CompletedProcess | None:
    """Run a read-only probe subprocess. None ⇒ tool missing / spawn error."""
    try:
        return subprocess.run(
            cmd, cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live_topology_hash(matrix_path: Path = CONTENTION_MATRIX) -> str | None:
    """Return the same 16-char topology hash used by the contention matrix gate."""
    try:
        from scripts.server.stack_numa import NUMA_CONFIG
        from src.scheduling.contention import (
            load_contention_matrix,
            topology_fingerprint,
            topology_fingerprint_for_matrix,
        )
    except Exception:
        return None
    matrix = None
    try:
        matrix = load_contention_matrix(matrix_path)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return (
        topology_fingerprint_for_matrix(NUMA_CONFIG, matrix)
        if matrix is not None
        else topology_fingerprint(NUMA_CONFIG)
    )


def _tail(text: str | None, n: int = 2) -> list[str]:
    if not text:
        return []
    return text.strip().splitlines()[-n:]


def _pid_on_port(port: int) -> str | None:
    res = _run(
        [
            "bash",
            "-c",
            f"ps -eo pid,args | grep -E 'llama-server|ik_llama' | "
            f"grep -- '--port {port}' | grep -v grep | awk '{{print $1}}' | head -1",
        ],
        timeout=5.0,
    )
    if res is None or res.returncode not in (0, 1):
        return None
    return res.stdout.strip() or None


# --------------------------------------------------------------------------- #
# Individual checks — each returns a detail dict with a boolean "ok"
# --------------------------------------------------------------------------- #
def check_live_affinity(roles: list[str] | None = None,
                        timeout: float = 180.0,
                        script: Path = AFFINITY_SCRIPT,
                        live_only: bool = False) -> dict:
    """Subprocess affinity_preflight.py; read live_affinity_verified from its artifact.

    Conservative: any failure to run or parse ⇒ verified False ⇒ check FAIL. A
    live-affinity gate that cannot prove pinning must not pass.
    """
    detail: dict = {"probe": "affinity_preflight.py", "method": "subprocess"}
    if not Path(script).exists():
        return {"ok": False, "live_affinity_verified": False,
                "error": f"affinity_preflight missing: {script}", **detail}
    fd, tmp_name = tempfile.mkstemp(suffix=".json", prefix="affinity_preflight_")
    import os as _os

    _os.close(fd)
    tmp = Path(tmp_name)
    cmd = [sys.executable, str(script), "--output", str(tmp)]
    if roles:
        cmd += ["--roles", *roles]
    res = _run(cmd, cwd=ORCH, timeout=timeout)
    verified = False
    artifact = None
    try:
        if tmp.exists() and tmp.stat().st_size > 0:
            artifact = json.loads(tmp.read_text())
            verified = bool(artifact.get("live_affinity_verified"))
    except (ValueError, OSError):
        artifact = None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    summary = None
    if isinstance(artifact, dict):
        instances = artifact.get("instances", [])
        live_instances = [
            e for e in instances if isinstance(e, dict) and e.get("pid")
        ]
        live_verified = bool(live_instances) and all(
            bool(e.get("match")) for e in live_instances
        )
        memory_required = bool(artifact.get("memory_locality_required", False))
        memory_verified = (
            not memory_required
            or bool(artifact.get("live_memory_placement_verified", True))
        )
        if live_only:
            verified = live_verified and memory_verified
        summary = {
            "instances": len(instances),
            "matched": sum(1 for e in instances if e.get("match")),
            "live_instances": len(live_instances),
            "live_matched": sum(1 for e in live_instances if e.get("match")),
            "live_memory_placement_verified": artifact.get("live_memory_placement_verified"),
        }
    return {
        "ok": verified,
        "live_affinity_verified": verified,
        "configured_affinity_verified": bool(
            artifact.get("live_affinity_verified")
        ) if isinstance(artifact, dict) else False,
        "live_only": live_only,
        "returncode": (res.returncode if res else None),
        "artifact_summary": summary,
        "memory_locality_required": bool(
            artifact.get("memory_locality_required", False)
        ) if isinstance(artifact, dict) else False,
        "error": None if res else "affinity_preflight subprocess failed to run",
        **detail,
    }


def _probe_ports(ports: list[int] | None, timeout: float) -> dict:
    """Per-port /health probe via the imported stack_health.wait_for_health."""
    if sys.path and str(ORCH) not in sys.path:
        sys.path.insert(0, str(ORCH))
    try:
        from scripts.server.stack_health import wait_for_health
    except Exception as exc:  # noqa: BLE001
        return {"error": f"stack_health import failed: {exc}", "ports": {}}
    if ports is None:
        try:
            from scripts.server.stack_numa import NUMA_CONFIG

            discovered: list[int] = []
            for cfg in NUMA_CONFIG.values():
                for inst in cfg.get("instances", []):
                    port = inst[1]
                    if port not in discovered:
                        discovered.append(port)
            ports = discovered
        except Exception as exc:  # noqa: BLE001
            return {"error": f"stack_numa import failed: {exc}", "ports": {}}
    per_port = {str(p): bool(wait_for_health(p, timeout=int(max(1, timeout)))) for p in ports}
    return {"ports": per_port, "error": None}


def ports_for_roles(roles: list[str] | None, *, live_only: bool = False) -> list[int] | None:
    """Return NUMA-configured health ports for a role-scoped server gate."""

    if not roles:
        return None
    try:
        from scripts.server.stack_numa import NUMA_CONFIG

        ports: list[int] = []
        for role in roles:
            cfg = NUMA_CONFIG.get(role)
            if not cfg:
                continue
            for inst in cfg.get("instances", []):
                port = inst[1]
                if live_only and not _pid_on_port(port):
                    continue
                if port not in ports:
                    ports.append(port)
        if ports:
            return ports
    except Exception:  # noqa: BLE001
        pass

    try:
        from scripts.server.stack_manifest import PORT_MAP
    except Exception:  # noqa: BLE001
        return None
    ports: list[int] = []
    for role in roles:
        port = PORT_MAP.get(role)
        if port is not None and port not in ports:
            ports.append(port)
    return ports


def check_health(require_servers: bool = False,
                 ports: list[int] | None = None,
                 port_timeout: float = 3.0,
                 script: Path = HEALTH_CHECK_SCRIPT,
                 server_health_only: bool = False,
                 health_profile: str | None = "batch") -> dict:
    """Structural stack health (health_check.sh) + optional live server-port probe.

    health_ok = structural_ok, unless --require-servers is set, in which case all
    configured server ports must also answer /health.
    """
    detail: dict = {
        "probe": "health_check.sh (+ stack_health.wait_for_health)",
        "health_profile": health_profile,
    }
    structural_ok = None
    rc = None
    if server_health_only:
        detail["structural_skipped"] = True
    elif Path(script).exists():
        cmd = ["bash", str(script)]
        if health_profile:
            cmd += ["--profile", health_profile]
        res = _run(cmd, cwd=ORCH, timeout=180.0)
        if res is not None:
            rc = res.returncode
            structural_ok = res.returncode == 0
    else:
        detail["structural_note"] = f"health_check.sh missing: {script}"

    port_detail = {"ports": {}, "error": "not probed (require_servers=False)"}
    ports_ok = None
    if require_servers:
        port_detail = _probe_ports(ports, port_timeout)
        probed = port_detail.get("ports", {})
        ports_ok = bool(probed) and all(probed.values())

    structural_pass = True if server_health_only else bool(structural_ok)
    ok = structural_pass and (ports_ok if require_servers else True)
    return {
        "ok": ok,
        "health_ok": ok,
        "structural_ok": structural_ok,
        "health_check_rc": rc,
        "require_servers": require_servers,
        "server_ports": port_detail,
        **detail,
    }


def check_topology_hashes(expected_topology_hash: str | None = None,
                          live_registry: Path = LIVE_REGISTRY,
                          matrix_path: Path = CONTENTION_MATRIX) -> dict:
    """Compute topology_hash (live NUMA topology) + registry_hash.

    Gate semantics: if expected_topology_hash is provided, ok requires an exact
    match (drift ⇒ FAIL). If not provided, the topology hash is recorded as an
    observation and ok is True as long as it is computable.
    """
    topo = _live_topology_hash(Path(matrix_path))
    reg = _sha256(Path(live_registry))
    result: dict = {
        "topology_hash": topo,
        "registry_hash": reg,
        "topology_source": str(matrix_path),
        "registry_source": str(live_registry),
        "expected_topology_hash": expected_topology_hash,
    }
    if topo is None:
        result["ok"] = False
        result["topology_match"] = None
        result["error"] = f"live topology hash unavailable for matrix: {matrix_path}"
    elif expected_topology_hash is not None:
        match = topo == expected_topology_hash
        result["ok"] = match
        result["topology_match"] = match
        result["error"] = None if match else "topology hash drift vs expected"
    else:
        result["ok"] = True
        result["topology_match"] = None
        result["error"] = None
    return result


def check_contention_matrix_fresh(max_age_days: int = 30,
                                  script: Path = CONTENTION_FRESH_SCRIPT,
                                  observation_only: bool = False) -> dict:
    """Subprocess the freshness validator: exit 0 fresh / 2 stale|missing / 3 invalid."""
    detail: dict = {"probe": "check_contention_matrix_fresh.py", "method": "subprocess"}
    if not Path(script).exists():
        return {"ok": False, "contention_matrix_fresh": False,
                "error": f"freshness checker missing: {script}", **detail}
    res = _run(
        [sys.executable, str(script), "--max-age-days", str(max_age_days)],
        cwd=ORCH, timeout=120.0,
    )
    if res is None:
        return {"ok": False, "contention_matrix_fresh": False,
                "error": "freshness checker subprocess failed to run", **detail}
    fresh = res.returncode == 0
    warning = None if fresh else f"contention matrix not fresh (rc={res.returncode})"
    return {
        "ok": fresh or observation_only,
        "contention_matrix_fresh": fresh,
        "returncode": res.returncode,
        "message_tail": _tail(res.stdout) + _tail(res.stderr),
        "error": None if fresh or observation_only else warning,
        "warning": warning,
        "observation_only": observation_only,
        **detail,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
_GATES = [
    ("live_affinity", "live_affinity_verified is false"),
    ("health", "health_ok is false"),
    ("topology", "topology hash gate failed"),
    ("contention_matrix", "contention matrix stale/missing/invalid"),
]


def attest(*, expected_topology_hash: str | None = None,
           roles: list[str] | None = None,
           require_servers: bool = False,
           max_age_days: int = 30,
           ports: list[int] | None = None,
           affinity_live_only: bool = False,
           server_health_only: bool = False,
           health_profile: str | None = "batch",
           checks: dict | None = None) -> dict:
    """Compose the four probes into a single attestation dict (no file write).

    Pass `checks` to inject pre-computed check dicts (tests); otherwise the real
    read-only probes run. overall is PASS iff every check's "ok" is truthy.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    if require_servers and ports is None:
        ports = ports_for_roles(roles, live_only=affinity_live_only)
    if checks is None:
        checks = {
            "live_affinity": check_live_affinity(
                roles=roles,
                live_only=affinity_live_only,
            ),
            "health": check_health(
                require_servers=require_servers,
                ports=ports,
                server_health_only=server_health_only,
                health_profile=health_profile,
            ),
            "topology": check_topology_hashes(expected_topology_hash=expected_topology_hash),
            "contention_matrix": check_contention_matrix_fresh(
                max_age_days=max_age_days,
            ),
        }

    fail_reasons: list[str] = []
    for key, default_msg in _GATES:
        check = checks.get(key, {})
        if not check.get("ok"):
            fail_reasons.append(f"{key}: {check.get('error') or default_msg}")

    topology = checks.get("topology", {})
    return {
        "ts": ts,
        "topology_hash": topology.get("topology_hash"),
        "registry_hash": topology.get("registry_hash"),
        "expected_topology_hash": expected_topology_hash,
        "live_affinity_verified": bool(checks.get("live_affinity", {}).get("live_affinity_verified")),
        "contention_matrix_fresh": bool(checks.get("contention_matrix", {}).get("contention_matrix_fresh")),
        "health_ok": bool(checks.get("health", {}).get("health_ok")),
        "checks": checks,
        "overall": "PASS" if not fail_reasons else "FAIL",
        "fail_reasons": fail_reasons,
    }


def write_attestation(att: dict, output_dir: Path = ATTEST_DIR) -> Path:
    """Persist an attestation to <output_dir>/<ts>.json (read-only wrt the stack)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = att["ts"].replace(":", "").replace("-", "")
    path = output_dir / f"{safe_ts}.json"
    if path.exists():
        path = output_dir / f"{safe_ts}_{int(time.time() * 1000) % 1000:03d}.json"
    path.write_text(json.dumps(att, indent=2))
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    if argv is None:
        _reexec_under_project_venv()
    ap = argparse.ArgumentParser(description="B4 inference-batch preflight gate (read-only).")
    ap.add_argument("--expected-topology-hash", default=None,
                    help="fail if the live topology hash does not match this value")
    ap.add_argument("--roles", nargs="*", default=None, help="affinity roles subset")
    ap.add_argument("--require-servers", action="store_true",
                    help="also require all configured server ports to answer /health")
    ap.add_argument("--affinity-live-only", action="store_true",
                    help="pass affinity if every live scoped instance matches, while recording dropped configured replicas")
    ap.add_argument("--server-health-only", action="store_true",
                    help="skip structural health_check.sh and require only scoped live server ports")
    ap.add_argument("--health-profile", default="batch",
                    help="health_check.sh profile to use for structural health (default: batch)")
    ap.add_argument("--max-age-days", type=int, default=30,
                    help="contention-matrix staleness window (default 30)")
    ap.add_argument("--output-dir", default=str(ATTEST_DIR))
    ap.add_argument("--no-write", action="store_true", help="do not persist the attestation")
    ap.add_argument("--json", action="store_true", help="print the full attestation JSON")
    args = ap.parse_args(argv)

    att = attest(
        expected_topology_hash=args.expected_topology_hash,
        roles=args.roles,
        require_servers=args.require_servers,
        max_age_days=args.max_age_days,
        affinity_live_only=args.affinity_live_only,
        server_health_only=args.server_health_only,
        health_profile=args.health_profile,
    )

    path = None
    if not args.no_write:
        path = write_attestation(att, args.output_dir)
        att["_written_to"] = str(path)

    if args.json:
        print(json.dumps(att, indent=2))
    else:
        print(f"preflight overall = {att['overall']}")
        print(f"  topology_hash          = {att['topology_hash']}")
        print(f"  registry_hash          = {att['registry_hash']}")
        print(f"  live_affinity_verified = {att['live_affinity_verified']}")
        print(f"  contention_matrix_fresh= {att['contention_matrix_fresh']}")
        print(f"  health_ok              = {att['health_ok']}")
        for reason in att["fail_reasons"]:
            print(f"  FAIL: {reason}")
        if path:
            print(f"  attestation → {path}")

    return 0 if att["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
