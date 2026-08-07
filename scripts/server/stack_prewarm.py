"""Page-cache prewarm for shared GGUF files.

Runs between [0.5] Validating model paths and [2] Checking target ports.
Reads each unique GGUF (deduped by inode) under `numactl --interleave=all` so
that subsequent mlock from per-instance launches finds pages already
distributed across all NUMA nodes. Without this step, after a cold cache /
container rebuild / drop_caches, sequential mlock pins every page of a shared
model onto whichever NUMA node the first launcher happened to bind to —
collapsing throughput by 50-65% for the quarters whose CPU set lives elsewhere.

See handoffs/active/numa-page-cache-prewarm.md for context. The handoff's
P1-P4 sit here; the wire-up is in stack_commands.cmd_start.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

_FLAG_NAMES = ("-m", "-md", "--mmproj")

# Bash bridge environment override mirroring --skip-page-cache-prewarm.
SKIP_ENV_VAR = "ORCHESTRATOR_SKIP_PAGE_CACHE_PREWARM"


def _extract_paths_from_cmd(cmd: list[str]) -> list[str]:
    """Return every -m/-md/--mmproj argument value in `cmd`.

    Each flag is matched once (its first occurrence). Repeated occurrences
    are not currently produced by build_server_command, so we don't try to
    catch them — keeps the scan O(len(cmd) * 3) and obvious to read.
    """
    out: list[str] = []
    for flag in _FLAG_NAMES:
        try:
            idx = cmd.index(flag)
        except ValueError:
            continue
        if idx + 1 < len(cmd):
            out.append(cmd[idx + 1])
    return out


def collect_targets(
    servers: list[dict[str, Any]],
    build_command: Callable[..., list[str]],
    registry: Any,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Build each server's launch command, extract GGUF paths, dedupe by inode.

    Returns a dict keyed by (st_dev, st_ino) with values:
      {"path": Path, "size_bytes": int, "ports": [int...], "roles": set[str]}

    Inode dedupe (not path dedupe) is deliberate: symlinks, bind mounts, and
    alias paths that resolve to the same physical GGUF must be warmed once.
    """
    by_inode: dict[tuple[int, int], dict[str, Any]] = {}
    for server in servers:
        port = server["port"]
        roles = list(server.get("roles") or [])
        if server.get("gpu_shadow_lane"):
            # gpu-serving-tie-in P2-6 (P2-4): the GPU shadow lane tenant loads
            # into VRAM — a NUMA-interleaved page-cache prewarm of its ~28 GiB
            # GGUF costs minutes and pins CPU page cache for no serving benefit.
            # Inert today: no server entry carries this flag until the lane
            # proposal is applied.
            print(f"  [prewarm] skip port {port} (gpu_shadow_lane: VRAM-resident tenant)")
            continue
        try:
            role_config = registry.get_role(roles[0]) if roles else None
        except Exception:
            role_config = None
        try:
            cmd = build_command(
                role_config,
                port,
                dev_mode=server.get("dev", False),
                embedding_mode=server.get("embedding", False),
                worker_pool_mode=server.get("worker_pool", False),
                worker_type=server.get("worker_type"),
                vision_mode=server.get("vision", False),
                vision_type=server.get("vision_type"),
                eval_batch_frontdoor_mode=server.get("eval_batch_frontdoor", False),
                numa_instance=server.get("numa_instance", 0),
            )
        except Exception as exc:
            print(
                f"  [prewarm] skip port {port} (roles={','.join(roles) or '-'}): "
                f"build_command failed: {exc}"
            )
            continue
        for raw in _extract_paths_from_cmd(cmd):
            path = Path(raw)
            try:
                resolved = path.resolve(strict=True)
                st = resolved.stat()
            except OSError as exc:
                print(f"  [prewarm] skip unreadable {path}: {exc}")
                continue
            key = (st.st_dev, st.st_ino)
            entry = by_inode.get(key)
            if entry is None:
                entry = {
                    "path": resolved,
                    "size_bytes": st.st_size,
                    "ports": [],
                    "roles": set(),
                }
                by_inode[key] = entry
            entry["ports"].append(port)
            entry["roles"].update(roles)
    return by_inode


def prewarm_file(path: Path) -> tuple[bool, float, str]:
    """Read `path` once under `numactl --interleave=all` so the page cache
    populates with an interleaved policy. Returns (ok, elapsed_s, message).

    `numactl --interleave=all cat <path> > /dev/null` is the canonical recipe
    (per feedback_drop_caches_numa_eviction). We use `cat` rather than `mmap`
    because cat's sequential read causes uniform first-touch under the
    interleave policy on EVERY page, which mmap+mlock will then find already
    placed when it pins them.
    """
    numactl = shutil.which("numactl")
    if numactl is None:
        return False, 0.0, "numactl binary not found on PATH"
    cat = shutil.which("cat") or "/bin/cat"
    t0 = time.monotonic()
    try:
        subprocess.run(
            [numactl, "--interleave=all", cat, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()[:200]
        return False, time.monotonic() - t0, f"non-zero exit ({exc.returncode}): {stderr}"
    except OSError as exc:
        return False, time.monotonic() - t0, f"OSError: {exc}"
    return True, time.monotonic() - t0, "ok"


def _skip_from_args_or_env(args: Any) -> bool:
    """Resolve the user's skip choice from CLI flag or env override."""
    if getattr(args, "skip_page_cache_prewarm", False):
        return True
    return os.environ.get(SKIP_ENV_VAR) == "1"


def prewarm_all(
    servers: list[dict[str, Any]],
    build_command: Callable[..., list[str]],
    registry: Any,
    *,
    args: Any = None,
    skip: bool | None = None,
) -> int:
    """Top-level entry called from cmd_start.

    Returns:
      0 — prewarm phase succeeded, was intentionally skipped, or had nothing
          to warm.
      1 — at least one warm attempt failed. Caller decides whether to abort
          — the default callsite continues startup, because warm-cache
          reads may still happen organically once mlock fires.

    Parameters
    ----------
    skip : explicit override for tests; when None, derived from `args` or
           ORCHESTRATOR_SKIP_PAGE_CACHE_PREWARM.
    """
    if skip is None:
        skip = _skip_from_args_or_env(args)
    if skip:
        print("[1.5] Page-cache prewarm SKIPPED (--skip-page-cache-prewarm)")
        print(
            "  [!] NUMA page-cache placement is not enforced; shared-GGUF roles "
            "may see degraded throughput after a cold cache / container rebuild. "
            "Recovery: stop stack, `sync && drop_caches`, "
            "`numactl --interleave=all cat <gguf>`, restart."
        )
        return 0
    print("[1.5] Page-cache prewarm (numactl --interleave=all)")
    targets = collect_targets(servers, build_command, registry)
    if not targets:
        print("  [prewarm] no GGUF targets resolved; skipping")
        return 0
    total_ports = sum(len(v["ports"]) for v in targets.values())
    total_gib = sum(v["size_bytes"] for v in targets.values()) / (1024**3)
    print(
        f"  [prewarm] {len(targets)} unique GGUF(s), {total_gib:.1f} GiB total, "
        f"across {total_ports} server instance(s)"
    )
    any_fail = False
    # Warm largest first so a small file's warm doesn't get a wrong cache-state
    # readout because a big file is still streaming. Order matters less for
    # correctness than for log readability.
    for key in sorted(targets, key=lambda k: -targets[k]["size_bytes"]):
        entry = targets[key]
        size_gib = entry["size_bytes"] / (1024**3)
        ports_label = ",".join(str(p) for p in sorted(entry["ports"]))
        ok, elapsed, msg = prewarm_file(entry["path"])
        status = f"OK in {elapsed:.1f}s" if ok else f"FAIL ({msg})"
        print(f"  [{size_gib:5.1f} GiB] {entry['path'].name} → ports [{ports_label}]: {status}")
        if not ok:
            any_fail = True
    return 1 if any_fail else 0
