"""Canonical cross-role contention matrix re-bench tool.

Phase F of `handoffs/active/cross-role-bw-aware-routing.md`. Replaces the
ad-hoc `/workspace/tmp/contention_matrix*.sh` scripts.

Reads NUMA_CONFIG to enumerate relevant role pairs, runs solo + parallel
HTTP benches against the live llama-server ports, writes a YAML matrix that
the gate (Phase A's `src/scheduling/contention.py`) consumes.

Smart-prune rule: once a pair is measured as catastrophic (ratio < 0.65),
do NOT include triples or larger N-way combinations that contain it — they'd
just re-confirm the catastrophic floor at high time cost. Skipped pairs are
recorded explicitly as `unknown_pairs` with `reason: skipped_due_to_pair_X`
so the gate's unknown-pair policy still applies.

Usage:
    # Re-bench the live stack (assumes stack is up + roles healthy)
    python scripts/server/contention_matrix.py

    # Subset
    python scripts/server/contention_matrix.py --roles frontdoor worker_general

    # Dry-run (show what would be measured, don't bench)
    python scripts/server/contention_matrix.py --dry-run

    # Validate the existing matrix without re-running
    python scripts/server/contention_matrix.py --validate-only

Output: writes `orchestration/contention_matrix.yaml` with full metadata
(topology hash, binary commit, host, measured_at). The gate uses the
topology hash to detect when NUMA_CONFIG has drifted from the measured state.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Resolve repo paths regardless of cwd
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "server"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [matrix] %(levelname)s %(message)s")
log = logging.getLogger("contention_matrix")

PROMPT = (
    "Write a 100-word essay analyzing the philosophical implications of autonomous "
    "decision-making in artificial intelligence systems."
)
N_PREDICT = 100
DEFAULT_FLOOR = 0.85
CATASTROPHIC_FLOOR = 0.65  # below this, skip triples/quads containing the pair
TIMEOUT_PER_REQUEST_S = 180
DEFAULT_OUTPUT = REPO_ROOT / "orchestration" / "contention_matrix.yaml"


@dataclass
class BenchResult:
    port: int
    role: str
    tps: float
    elapsed_s: float


@dataclass
class PairBench:
    roles: tuple[str, str]
    instance_a: dict[str, Any]
    instance_b: dict[str, Any]
    solo_a: BenchResult
    solo_b: BenchResult
    parallel_a: BenchResult
    parallel_b: BenchResult
    seq_aggregate_tps: float
    parallel_aggregate_tps: float
    ratio: float

    def verdict(self, floor: float = DEFAULT_FLOOR) -> str:
        if self.ratio >= 1.0:
            return "allow"
        if self.ratio >= floor:
            return "borderline"
        return "block"

    def to_yaml_dict(self, floor: float = DEFAULT_FLOOR) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "instance_a": self.instance_a,
            "instance_b": self.instance_b,
            "seq_aggregate_tps": round(self.seq_aggregate_tps, 2),
            "parallel_aggregate_tps": round(self.parallel_aggregate_tps, 2),
            "ratio": round(self.ratio, 2),
            "samples": 1,
            "verdict": self.verdict(floor),
        }


# ── HTTP bench primitive ────────────────────────────────────────────


def _http_bench(port: int, n_predict: int = N_PREDICT) -> tuple[float, float]:
    """Single /completion call → (tps, elapsed_s). 0.0 on error."""
    try:
        import httpx
    except ImportError:
        log.error("httpx not available")
        return (0.0, 0.0)

    payload = {
        "prompt": PROMPT,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
    }
    try:
        with httpx.Client(timeout=TIMEOUT_PER_REQUEST_S) as client:
            resp = client.post(f"http://localhost:{port}/completion", json=payload)
            resp.raise_for_status()
            data = resp.json()
        timings = data.get("timings", {})
        tps = float(timings.get("predicted_per_second", 0.0))
        ms = float(timings.get("predicted_ms", 0.0))
        return (tps, ms / 1000.0 if ms else 0.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("bench port=%d failed: %s", port, exc)
        return (0.0, 0.0)


def _bench_pair(role_a: str, port_a: int, role_b: str, port_b: int) -> PairBench:
    """Measure solo + parallel for one role pair. Returns a PairBench."""
    log.info("bench pair: %s (port %d) + %s (port %d)", role_a, port_a, role_b, port_b)

    # Solo
    log.info("  solo %s:%d ...", role_a, port_a)
    solo_a_tps, solo_a_el = _http_bench(port_a)
    log.info("  solo %s:%d ...", role_b, port_b)
    solo_b_tps, solo_b_el = _http_bench(port_b)

    # Parallel
    log.info("  parallel %s:%d || %s:%d ...", role_a, port_a, role_b, port_b)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(_http_bench, port_a)
        fut_b = ex.submit(_http_bench, port_b)
        par_a_tps, par_a_el = fut_a.result()
        par_b_tps, par_b_el = fut_b.result()

    total_tokens = N_PREDICT * 2
    seq_time = (solo_a_el + solo_b_el) or 0.001
    par_time = max(par_a_el, par_b_el) or 0.001
    seq_agg = total_tokens / seq_time
    par_agg = total_tokens / par_time
    ratio = par_agg / seq_agg if seq_agg > 0 else 0.0

    log.info(
        "  → seq=%.2f t/s par=%.2f t/s ratio=%.2f",
        seq_agg, par_agg, ratio,
    )

    return PairBench(
        roles=tuple(sorted([role_a, role_b])),  # type: ignore[arg-type]
        instance_a={"port": port_a},
        instance_b={"port": port_b},
        solo_a=BenchResult(port=port_a, role=role_a, tps=solo_a_tps, elapsed_s=solo_a_el),
        solo_b=BenchResult(port=port_b, role=role_b, tps=solo_b_tps, elapsed_s=solo_b_el),
        parallel_a=BenchResult(port=port_a, role=role_a, tps=par_a_tps, elapsed_s=par_a_el),
        parallel_b=BenchResult(port=port_b, role=role_b, tps=par_b_tps, elapsed_s=par_b_el),
        seq_aggregate_tps=seq_agg,
        parallel_aggregate_tps=par_agg,
        ratio=ratio,
    )


# ── Enumeration + smart-prune ───────────────────────────────────────


def _enumerate_full_pairs(numa_config: dict, role_filter: set[str] | None = None) -> list[tuple[str, str]]:
    """Enumerate (role_a, role_b) pairs of full instances. Sorted + deduped."""
    roles = sorted(numa_config.keys())
    if role_filter:
        roles = [r for r in roles if r in role_filter]
    out: list[tuple[str, str]] = []
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            if not numa_config[a].get("instances") or not numa_config[b].get("instances"):
                continue
            out.append((a, b))
    return out


def _full_port(numa_config: dict, role: str) -> int | None:
    instances = numa_config.get(role, {}).get("instances", [])
    if not instances:
        return None
    return int(instances[0][1])


# ── Environment metadata ────────────────────────────────────────────


def _binary_metadata(binary_path: Path) -> dict[str, str]:
    """llama-server binary path + git commit (extracted from --version output)."""
    info = {"llama_server_path": str(binary_path), "git_commit": ""}
    if not binary_path.exists():
        return info
    try:
        out = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True, text=True, timeout=10,
        )
        for line in (out.stdout + out.stderr).splitlines():
            if "build:" in line.lower():
                # Format: "build: 2ffbdbbba (8957)"
                parts = line.split()
                for token in parts:
                    if token.startswith("(") and token.endswith(")"):
                        continue
                    if len(token) >= 7 and all(c in "0123456789abcdef" for c in token):
                        info["git_commit"] = token
                        break
                break
    except Exception:
        pass
    return info


def _host_metadata() -> dict[str, str]:
    try:
        uptime = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        uptime = ""
    return {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "uptime": uptime,
    }


# ── YAML emission ────────────────────────────────────────────────────


def _emit_yaml(
    pairs: list[PairBench],
    *,
    same_role_verdicts: dict[str, str] | None = None,
    unknown_pairs: list[tuple[str, str, str]] | None = None,
    topology_hash: str = "",
    binary: dict[str, str] | None = None,
    host: str = "",
    floor: float = DEFAULT_FLOOR,
) -> str:
    """Render the matrix as YAML (no PyYAML dump dependency — handle ourselves)."""
    lines: list[str] = []
    lines.append(f"# Auto-generated by scripts/server/contention_matrix.py")
    lines.append(f"# Generated {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("version: 1")
    lines.append(f'measured_at: "{datetime.now(timezone.utc).isoformat()}"')
    lines.append(f'host: "{host}"')
    lines.append("binary:")
    if binary:
        for k, v in binary.items():
            lines.append(f'  {k}: "{v}"')
    lines.append(f'topology_hash: "{topology_hash}"')
    lines.append(f"default_floor: {floor}")

    lines.append("")
    lines.append("pairs:")
    for p in sorted(pairs, key=lambda x: x.ratio):  # block first, allow last
        d = p.to_yaml_dict(floor)
        lines.append(f'  - roles: [{", ".join(repr(r) for r in d["roles"])}]')
        lines.append(f"    instance_a: {{port: {d['instance_a']['port']}}}")
        lines.append(f"    instance_b: {{port: {d['instance_b']['port']}}}")
        lines.append(f"    seq_aggregate_tps: {d['seq_aggregate_tps']}")
        lines.append(f"    parallel_aggregate_tps: {d['parallel_aggregate_tps']}")
        lines.append(f"    ratio: {d['ratio']}")
        lines.append(f"    samples: {d['samples']}")
        lines.append(f'    verdict: "{d["verdict"]}"')

    if same_role_verdicts:
        lines.append("")
        lines.append("same_role:")
        for role in sorted(same_role_verdicts.keys()):
            lines.append(f'  - role: "{role}"')
            lines.append(f'    verdict: "{same_role_verdicts[role]}"')

    if unknown_pairs:
        lines.append("")
        lines.append("unknown_pairs:")
        for a, b, reason in sorted(unknown_pairs):
            lines.append(f"  - roles: [{repr(a)}, {repr(b)}]")
            lines.append(f'    reason: "{reason}"')

    lines.append("")
    return "\n".join(lines)


# ── J4a: N-way candidate enumeration (no inference) ──────────────────
#
# Cross-role N-way closure (handoffs/active/cross-role-nway-contention-matrix.md).
# Reads the live role topology + the existing pairwise matrix and emits a
# deterministic, topology-stamped manifest of:
#   * candidate_sets   — non-trivial N-way active sets whose every constituent
#                        pair is bulk-allowed (ratio >= floor, measured); these
#                        REQUIRE J4b measurement before any cross-role launch.
#   * excluded_sets    — every other size>=3 set, each with the concrete
#                        lower-order evidence (block / below-floor / unknown
#                        pair, or a measured failed triple) that pruned it.
# Pairwise-allowed is a precondition only; nothing here is launch-certified.


def _pair_key_str(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))


def _read_matrix_triples(matrix_path: Path) -> list[dict[str, Any]]:
    """Read the informational `triples:` block (not parsed by the dataclass)."""
    try:
        import yaml

        data = yaml.safe_load(matrix_path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read triples from %s: %s", matrix_path, exc)
        return []
    out: list[dict[str, Any]] = []
    for entry in data.get("triples", []) or []:
        roles = entry.get("roles")
        if isinstance(roles, list) and len(roles) >= 3:
            out.append(
                {
                    "roles": tuple(sorted(roles)),
                    "ratio": float(entry.get("ratio", 0.0)),
                    "note": str(entry.get("note", "")),
                }
            )
    return out


def _classify_pair(matrix, a: str, b: str, floor: float) -> dict[str, Any]:
    """Bulk/background classification of a cross-role pair.

    bulk_allowed iff measured AND ratio >= floor. block / below-floor /
    unknown / missing are all NOT bulk-allowed (they queue or are unmeasured).
    """
    pair = matrix.get_pair(a, b)
    if pair is None:
        kind = "unknown" if matrix.is_unknown_pair(a, b) else "missing"
        return {"ratio": None, "verdict": kind, "bulk_allowed": False, "kind": kind}
    below = pair.ratio < floor
    return {
        "ratio": round(pair.ratio, 4),
        "verdict": pair.verdict,
        "bulk_allowed": (not below),
        "kind": "below_floor" if below else "allowed",
    }


def enumerate_n_way(
    numa_config: dict,
    matrix,
    *,
    floor: float,
    matrix_triples: list[dict[str, Any]] | None = None,
    max_size: int | None = None,
) -> dict[str, Any]:
    """Pure enumeration core. Returns the manifest body (no I/O, no timestamp)."""
    roles = sorted(r for r in numa_config if (numa_config[r].get("instances")))
    matrix_triples = matrix_triples or []
    failed_triples = {t["roles"] for t in matrix_triples if t["ratio"] < floor}

    # Lower-order pair evidence over every cross-role pair (deterministic order).
    pair_evidence: dict[str, dict[str, Any]] = {}
    for a, b in itertools.combinations(roles, 2):
        pair_evidence[_pair_key_str(a, b)] = _classify_pair(matrix, a, b, floor)

    hi = max_size or len(roles)
    candidate_sets: list[dict[str, Any]] = []
    excluded_sets: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []

    for size in range(3, hi + 1):
        for combo in itertools.combinations(roles, size):
            combo_sorted = tuple(sorted(combo))
            offending: list[str] = []
            min_ratio = None
            for a, b in itertools.combinations(combo_sorted, 2):
                ev = pair_evidence[_pair_key_str(a, b)]
                if not ev["bulk_allowed"]:
                    if ev["kind"] in ("unknown", "missing"):
                        offending.append(f"{a}+{b} {ev['kind']} pair (not measured)")
                    else:
                        offending.append(
                            f"{a}+{b} ratio {ev['ratio']} < floor {floor} ({ev['verdict']})"
                        )
                else:
                    r = ev["ratio"]
                    min_ratio = r if min_ratio is None else min(min_ratio, r)
            # measured-failed-triple pruning (superset of a known-bad triple)
            bad_sub = [
                "+".join(ft) for ft in failed_triples if set(ft).issubset(set(combo_sorted))
            ]

            if not offending and not bad_sub:
                candidate_sets.append(
                    {
                        "roles": list(combo_sorted),
                        "size": size,
                        "min_pair_ratio": min_ratio,
                        "constituent_pairs": [
                            _pair_key_str(a, b)
                            for a, b in itertools.combinations(combo_sorted, 2)
                        ],
                    }
                )
            else:
                reason = (
                    "contains_failed_triple"
                    if bad_sub and not offending
                    else "contains_below_floor_or_unknown_pair"
                )
                evidence = offending + [f"contains failed triple {s}" for s in bad_sub]
                excluded_sets.append(
                    {"roles": list(combo_sorted), "size": size, "reason": reason, "evidence": evidence}
                )

    # Discrepancy flags: an excluded set that the matrix actually measured >= floor
    # as an informational triple (pairwise-conservative vs N-way reality — the
    # exact phenomenon this closure exists to catch).
    excluded_role_sets = {tuple(e["roles"]) for e in excluded_sets}
    for t in matrix_triples:
        if t["roles"] in excluded_role_sets and t["ratio"] >= floor:
            flags.append(
                {
                    "roles": list(t["roles"]),
                    "issue": "excluded_by_pairwise_floor_but_measured_positive_as_triple",
                    "measured_triple_ratio": t["ratio"],
                    "recommendation": (
                        "J4b should measure this set explicitly; a constituent pair is "
                        "below the background floor while the measured triple is positive. "
                        "Reconsider for foreground-only allow or revisit the pair floor."
                    ),
                }
            )

    return {
        "floor": floor,
        "roles": roles,
        "pair_evidence": pair_evidence,
        "candidate_sets": sorted(candidate_sets, key=lambda x: (x["size"], x["roles"])),
        "excluded_sets": sorted(excluded_sets, key=lambda x: (x["size"], x["roles"])),
        "flags": flags,
        "summary": {
            "n_roles": len(roles),
            "n_candidates": len(candidate_sets),
            "n_excluded": len(excluded_sets),
            "max_size_enumerated": hi,
        },
    }


def cmd_enumerate(args: argparse.Namespace) -> int:
    from stack_numa import NUMA_CONFIG
    from src.scheduling.contention import (
        load_contention_matrix,
        matrix_status,
        topology_fingerprint,
        MatrixStatus,
    )

    matrix_path = Path(args.matrix) if args.matrix else DEFAULT_OUTPUT
    live_hash = topology_fingerprint(NUMA_CONFIG)
    status = matrix_status(matrix_path, current_topology_hash=live_hash)
    if status != MatrixStatus.OK:
        log.error(
            "matrix status %s (live hash %s) — enumeration requires a fresh, "
            "topology-matching matrix; refusing to emit a manifest against stale evidence.",
            status.value, live_hash,
        )
        return 2

    matrix = load_contention_matrix(matrix_path)
    floor = args.floor if args.floor is not None else matrix.default_floor
    body = enumerate_n_way(
        NUMA_CONFIG,
        matrix,
        floor=floor,
        matrix_triples=_read_matrix_triples(matrix_path),
        max_size=args.max_size,
    )

    # Deterministic content hash (excludes timestamp/run_id) for the J4a
    # "deterministic across two dry runs" closure gate.
    content = json.dumps(body, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    src_bytes = matrix_path.read_bytes()
    git_sha = ""
    try:
        git_sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        pass

    manifest = {
        "task_id": "J4a",
        "run_id": args.run_id or f"j4a-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topology_hash": live_hash,
        "content_hash": content_hash,
        "matrix_source": {
            "path": str(matrix_path),
            "sha256": hashlib.sha256(src_bytes).hexdigest(),
            "git_sha": git_sha,
            "topology_hash": matrix.topology_hash,
            "measured_at": matrix.measured_at,
        },
        **body,
    }

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "j4a_candidate_manifest.json"
        out_path.write_text(json.dumps(manifest, indent=2) + "\n")
        log.info("wrote manifest → %s", out_path)
    else:
        print(json.dumps(manifest, indent=2))

    log.info(
        "J4a enumeration: %d candidates, %d excluded, %d flags (content_hash=%s)",
        manifest["summary"]["n_candidates"],
        manifest["summary"]["n_excluded"],
        len(manifest["flags"]),
        content_hash,
    )
    for c in manifest["candidate_sets"]:
        log.info("  CANDIDATE %s (min_pair_ratio=%s)", c["roles"], c["min_pair_ratio"])
    for f in manifest["flags"]:
        log.warning("  FLAG %s: %s (triple=%.2f)", f["roles"], f["issue"], f["measured_triple_ratio"])
    return 0


# ── CLI ──────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> int:
    from stack_numa import NUMA_CONFIG
    from src.scheduling.contention import topology_fingerprint

    role_filter = set(args.roles) if args.roles else None
    pairs_to_bench = _enumerate_full_pairs(NUMA_CONFIG, role_filter)
    log.info("enumerated %d full-pair combinations", len(pairs_to_bench))

    if args.dry_run:
        log.info("DRY RUN — would measure:")
        for a, b in pairs_to_bench:
            port_a = _full_port(NUMA_CONFIG, a)
            port_b = _full_port(NUMA_CONFIG, b)
            log.info("  %s:%s + %s:%s", a, port_a, b, port_b)
        return 0

    measured: list[PairBench] = []
    blocked_pairs: list[tuple[str, str]] = []  # known catastrophic pairs
    skipped: list[tuple[str, str, str]] = []

    for a, b in pairs_to_bench:
        port_a = _full_port(NUMA_CONFIG, a)
        port_b = _full_port(NUMA_CONFIG, b)
        if port_a is None or port_b is None:
            skipped.append((a, b, "no_port"))
            continue
        pb = _bench_pair(a, port_a, b, port_b)
        measured.append(pb)
        if pb.ratio < CATASTROPHIC_FLOOR:
            blocked_pairs.append(pb.roles)

    binary_path = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-server")
    bin_meta = _binary_metadata(binary_path)
    host_meta = _host_metadata()
    topo_hash = topology_fingerprint(NUMA_CONFIG)

    yaml_str = _emit_yaml(
        measured,
        unknown_pairs=skipped,
        topology_hash=topo_hash,
        binary=bin_meta,
        host=host_meta["hostname"],
        floor=DEFAULT_FLOOR,
    )

    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    out_path.write_text(yaml_str)
    log.info("wrote %s (%d pairs, topology_hash=%s)", out_path, len(measured), topo_hash[:8])
    log.info("catastrophic pairs (ratio < %.2f): %s", CATASTROPHIC_FLOOR, blocked_pairs)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate the existing matrix against live NUMA_CONFIG."""
    from stack_numa import NUMA_CONFIG
    from src.scheduling.contention import (
        load_contention_matrix,
        matrix_status,
        topology_fingerprint,
        MatrixStatus,
    )

    path = Path(args.output) if args.output else DEFAULT_OUTPUT
    current_hash = topology_fingerprint(NUMA_CONFIG)
    status = matrix_status(path, current_topology_hash=current_hash)
    log.info("matrix status: %s (path=%s)", status.value, path)
    log.info("live topology hash: %s", current_hash)

    if status == MatrixStatus.OK:
        m = load_contention_matrix(path)
        log.info("matrix loaded OK: %d pairs, %d same_role, %d unknown_pairs",
                 len(m.pairs), len(m.same_role), len(m.unknown_pairs))
        return 0
    if status == MatrixStatus.MISSING:
        log.error("matrix MISSING — run: python scripts/server/contention_matrix.py")
        return 2
    if status == MatrixStatus.STALE:
        try:
            m = load_contention_matrix(path)
            log.warning("matrix STALE — stored hash %s, live %s (or file > 30d)",
                        m.topology_hash, current_hash)
        except Exception as exc:
            log.warning("matrix STALE — additionally failed to load: %s", exc)
        log.warning("re-run: python scripts/server/contention_matrix.py")
        return 2
    if status == MatrixStatus.INVALID:
        log.error("matrix INVALID — file present but unparseable")
        return 3
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="re-bench + write matrix (default)")
    p_run.add_argument("--roles", nargs="+", help="restrict to these roles")
    p_run.add_argument("--dry-run", action="store_true", help="enumerate without benching")
    p_run.add_argument("--output", help="output YAML path (default: orchestration/contention_matrix.yaml)")
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", help="check freshness against live NUMA_CONFIG")
    p_val.add_argument("--output", help="matrix YAML path to validate")
    p_val.set_defaults(func=cmd_validate)

    p_enum = sub.add_parser(
        "enumerate",
        help="J4a: emit N-way candidate/exclusion manifest (no inference)",
    )
    p_enum.add_argument("--matrix", help="input matrix YAML (default: orchestration/contention_matrix.yaml)")
    p_enum.add_argument("--output", help="output dir for j4a_candidate_manifest.json (default: stdout)")
    p_enum.add_argument("--floor", type=float, default=None, help="bulk/background floor (default: matrix default_floor)")
    p_enum.add_argument("--max-size", type=int, default=None, dest="max_size", help="max active-set size to enumerate (default: #roles)")
    p_enum.add_argument("--run-id", default=None, dest="run_id", help="stable run id for the manifest")
    p_enum.set_defaults(func=cmd_enumerate)

    args = p.parse_args()
    if args.cmd is None:
        args.cmd = "run"
        args.roles = None
        args.dry_run = False
        args.output = None
        args.func = cmd_run
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
