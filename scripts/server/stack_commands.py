"""CLI command handlers for orchestrator_stack — start/stop/reload/status.

Extracted from orchestrator_stack.py during the 2026-05-22 Tranche-7 refactor.
These functions are the public CLI surface invoked by `orchestrator_stack.py main()`;
they live here so the CLI shim stays focused on argument parsing + dispatch.

Imports from orchestrator_stack (and helper modules) at module load. This is safe
because orchestrator_stack.py loads stack_commands lazily inside main() — no
circular import at module-import time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Helper modules (extracted earlier in the refactor)
from scripts.server import stack_processes as _stack_processes
from scripts.server.stack_docker import (
    _docker_available,
    docker_container_running,
    start_docker_container,
    stop_docker_container,
)
from scripts.server.stack_health import wait_for_health as _wait_for_health
from scripts.server.stack_host import apply_host_prerequisites
from scripts.server.stack_manifest import (
    DOCKER_SERVICES,
    EMBEDDER_PORTS,
    HOT_ROLES,
    HOT_SERVERS,
    NUMA_REPLICA_PORTS,
    ORCHESTRATOR_PROFILES,
    PORT_MAP,
    ROLE_LAUNCH_META,
    WARM_SERVERS,
    _build_servers_from_classification,
    _filter_by_numa_mode,
    validate_against_registry,
    validate_model_paths,
)
from scripts.server.stack_numa import MLOCK_ROLES, NUMA_CONFIG, _numa_prefix
from scripts.server.stack_paths import (
    _HEALTH_SERVER_STARTUP,
    _HEALTH_VISION_SERVER,
    _HEALTH_WORKER_SERVER,
    LLAMA_MATH_TOOLS,
    _PATHS,
    LLAMA_SERVER,
    LOG_DIR,
    SLOT_SAVE_DIR,
    STATE_FILE,
)
from scripts.server.stack_prewarm import prewarm_all as _prewarm_all
from scripts.server.stack_state import ProcessInfo
from src.registry_loader import RegistryLoader


def wait_for_health(
    port: int, timeout: int = _HEALTH_SERVER_STARTUP, path: str = "/health"
) -> bool:
    """Preserve orchestrator_stack.wait_for_health default-timeout semantics."""
    return _wait_for_health(port, timeout, path)


def _orchestrator_stack():
    """Lazy import of orchestrator_stack to access functions defined there
    (start_server, start_orchestrator, start_*, init_memrl_and_tools, the
    thin process/state wrappers). orchestrator_stack imports this module
    lazily inside main() so no module-load cycle exists.
    """
    from scripts.server import orchestrator_stack

    return orchestrator_stack


# Convenience accessors via the lazy proxy — keep cmd_* bodies readable.
def start_server(*a, **kw):
    return _orchestrator_stack().start_server(*a, **kw)


def start_orchestrator(*a, **kw):
    return _orchestrator_stack().start_orchestrator(*a, **kw)


def start_document_formalizer(*a, **kw):
    return _orchestrator_stack().start_document_formalizer(*a, **kw)


def start_sd_server(*a, **kw):
    return _orchestrator_stack().start_sd_server(*a, **kw)


def start_whisper(*a, **kw):
    return _orchestrator_stack().start_whisper(*a, **kw)


def init_memrl_and_tools(*a, **kw):
    return _orchestrator_stack().init_memrl_and_tools(*a, **kw)


def load_state(*a, **kw):
    return _orchestrator_stack().load_state(*a, **kw)


def save_state(*a, **kw):
    return _orchestrator_stack().save_state(*a, **kw)


def kill_process(*a, **kw):
    return _orchestrator_stack().kill_process(*a, **kw)


def is_port_in_use(*a, **kw):
    return _orchestrator_stack().is_port_in_use(*a, **kw)


def _pids_on_port(*a, **kw):
    return _orchestrator_stack()._pids_on_port(*a, **kw)


def _collect_descendants(*a, **kw):
    return _orchestrator_stack()._collect_descendants(*a, **kw)


def _renice_all_threads(*a, **kw):
    return _orchestrator_stack()._renice_all_threads(*a, **kw)


def check_free_memory(*a, **kw):
    return _orchestrator_stack().check_free_memory(*a, **kw)


def _apply_orchestrator_profile(*a, **kw):
    return _orchestrator_stack()._apply_orchestrator_profile(*a, **kw)


def _find_pids_on_port(port: int) -> list[int]:
    """Find PIDs listening on a port via lsof (fallback for stale state)."""
    return _stack_processes.pids_on_port(port, timeout=5)


def _scan_known_ports() -> dict[int, list[int]]:
    """Scan all known orchestrator ports for running processes."""
    managed_server_ports = {s["port"] for s in HOT_SERVERS + WARM_SERVERS}
    docker_ports = {int(svc["port"]) for svc in DOCKER_SERVICES if "port" in svc}
    native_aux_ports = {8190, 9000, 9001}
    known_ports = sorted(
        managed_server_ports | NUMA_REPLICA_PORTS | docker_ports | native_aux_ports | {8000}
    )
    return _stack_processes.scan_known_ports(known_ports)


def _preserved_process_info(
    role: str,
    port: int,
    model_path: str,
    log_file: str = "",
) -> ProcessInfo | None:
    """Build a durable state record for an already-healthy listener."""
    pids = _pids_on_port(port)
    if not pids:
        print(f"  [WARN] {role} on port {port} is healthy but listener PID was not found")
        return None
    return ProcessInfo(
        role=role,
        pid=pids[0],
        port=port,
        started_at=datetime.now().isoformat(),
        model_path=model_path,
        log_file=log_file,
    )


def cmd_start(args: argparse.Namespace) -> int:
    """Start the orchestrator stack."""
    _registry_yaml = Path(__file__).parent.parent.parent / "orchestration" / "model_registry.yaml"
    _master_registry = Path(
        "/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml"
    )
    _cache_key_path = _registry_yaml.parent / ".lean_cache_key"

    # 2026-05-09: lean-registry compile (opt-in via --compile-registry).
    # When enabled, regenerates orchestration/model_registry.yaml from
    # the master at epyc-inference-research/orchestration/model_registry.yaml,
    # filtered to active roles per ROLE_LAUNCH_META + their transitive
    # draft/alias deps. Cache-keyed by SHA-256 of (master content + active
    # role names) — re-runs without changes are no-ops.
    #
    # Default OFF until the master + orchestrator are reconciled (today the
    # master itself has an internal acceleration disagreement on
    # architect_general — `roles.X.acceleration.type=speculative_decoding`
    # vs `server_mode.X.acceleration.type=moe_expert_reduction`). Fix the
    # master first, then enable this flag in the start command.
    #
    # Escape hatch (when ON): set ORCHESTRATOR_REGISTRY_NO_COMPILE=1 to bypass.
    if getattr(args, "compile_registry", False):
        try:
            from src.registry.registry_compiler import load_or_compile

            active_roles = set(ROLE_LAUNCH_META.keys())
            print(f"[registry-compile] master={_master_registry}")
            print(f"[registry-compile] active_roles={sorted(active_roles)}")
            load_or_compile(
                master_path=_master_registry,
                active_roles=active_roles,
                output_path=_registry_yaml,
                cache_key_path=_cache_key_path,
            )
            print(f"[registry-compile] OK — wrote {_registry_yaml}")
        except Exception as exc:  # noqa: BLE001
            print(f"[registry-compile] FATAL: {exc}")
            return 2

    # 2026-05-09: registry consistency gate — runs first, fails fast on cross-section
    # acceleration disagreements, GGUF/port inconsistencies, or duplicate YAML keys.
    # Catches the failure modes that wasted ~2 hours debugging architect_general on
    # 2026-05-09 (server_mode.X.acceleration silently overridden by roles.X.acceleration).
    try:
        from src.registry.registry_validator import validate_or_raise, RegistryValidationError

        try:
            validate_or_raise(_registry_yaml)
        except RegistryValidationError as exc:
            print(f"[registry-validator] FATAL — refusing to start a stack on a broken registry:")
            print(f"  {exc}")
            print(f"  Fix the registry, then re-run.")
            return 2
    except ImportError:
        # Validator module not present (older deployment) — proceed without gate.
        # Once landed everywhere, drop this fallback.
        pass

    # DS-7 / NIB2-19: --migrate-to handler (runs before any start path)
    migrate_to = getattr(args, "migrate_to", None)
    if migrate_to:
        dry_run = getattr(args, "dry_run", False)
        try:
            from src.config.stack_migration import migrate_to_template
        except Exception as exc:  # noqa: BLE001
            print(f"[DS-7] Migration module unavailable: {exc}")
            return 1
        print(
            f"[DS-7] Migrating stack → template '{migrate_to}' ({'DRY-RUN' if dry_run else 'LIVE'})"
        )
        registry_path = (
            Path(_PATHS.get("model_registry", "")) if _PATHS.get("model_registry") else None
        )
        result = migrate_to_template(migrate_to, dry_run=dry_run, registry_path=registry_path)
        print(result.summary())
        return 0 if result.ok else 1

    # DS-7: Stack template validation (before any other work)
    stack_profile = getattr(args, "stack_profile", None)
    validate_only = getattr(args, "validate_only", False)
    if stack_profile:
        try:
            from src.config.stack_templates import (
                load_template,
                validate_template,
                _TEMPLATES_DIR,
            )

            print(f"[DS-7] Loading stack template: {stack_profile}")
            template = load_template(stack_profile)
            print(f"  Name: {template.name}")
            print(f"  Description: {template.description}")
            print(f"  Roles: {len(template.roles)} ({', '.join(template.role_names())})")
            print(f"  Instances: {template.total_instances}")
            print(f"  RAM: {template.total_ram_gb:.0f} GB")
            print()

            registry_path = (
                Path(_PATHS.get("model_registry", "")) if _PATHS.get("model_registry") else None
            )
            result = validate_template(template, registry_path)
            if result.errors:
                print(f"  [FAIL] {len(result.errors)} validation errors:")
                for err in result.errors:
                    print(f"    ERROR: {err}")
            if result.warnings:
                for warn in result.warnings:
                    print(f"    WARN: {warn}")
            if result.valid:
                print("  [OK] Template valid")
            else:
                print("\n  Template validation failed. Fix errors and retry.")
                return 1

            if validate_only:
                print("\n--validate-only: exiting after validation.")
                return 0

            print(
                f"  (Template loaded but not yet used for server launch — "
                f"integration pending DS-7 Phase 2)"
            )
            print()
        except FileNotFoundError as exc:
            print(f"[DS-7] ERROR: {exc}")
            return 1
        except Exception as exc:
            print(f"[DS-7] Template load error: {exc}")
            return 1

    print("=" * 60)
    print("ORCHESTRATOR STACK STARTUP")
    print("=" * 60)
    print()

    # Host prerequisites — applied before any llama-server launch.
    # See cpu-kernel-env-flags-inventory.md §211 + model-registry-v5-deployment-draft.yaml.
    skip_host_prereqs = getattr(args, "skip_host_prereqs", False)
    if skip_host_prereqs:
        print("[host_prereq] SKIPPED (--skip-host-prereqs). Canonical state NOT enforced.")
    else:
        if not apply_host_prerequisites(auto_fix=True):
            print("[!] Host prerequisites could not be applied. Refusing to launch.")
            print("    Override with --skip-host-prereqs (NOT recommended for benchmarks).")
            return 1
    print()

    # Check memory
    free_gb = check_free_memory()
    print(f"[i] Free memory: {free_gb} GB")
    if free_gb < 100 and not args.dev:
        print("[!] WARNING: Less than 100GB free. Consider --dev mode.")
        if input("Continue? (y/N) ").lower() != "y":
            return 1
    print()

    # Load registry
    registry = RegistryLoader()
    # In --only mode we start a SUBSET of roles, so seed `state` from the
    # existing on-disk state and merge — otherwise save_state() at the end
    # clobbers every other still-running role's entry, leaving live processes
    # untracked (stop/reload can no longer find them). A full start (no --only)
    # rebuilds from scratch, which is correct since it (re)launches everything.
    state: dict[str, ProcessInfo] = load_state() if getattr(args, "only", None) else {}

    # Validate model paths (prevents hallucinations about missing models)
    if not args.dev:
        print("[0.5] Validating model paths...")
        errors = validate_model_paths()
        if errors:
            print("[!] MODEL VALIDATION FAILED:")
            for err in errors:
                print(f"    - {err}")
            print("\nFix missing models or update paths in orchestrator_stack.py")
            print(f"Check {_PATHS['models_dir']} and {_PATHS['model_base']}")
            return 1
        print("  [OK] All model paths validated")
        print()

    # Cross-check launcher classification vs registry process_layout / server_mode.
    # Non-fatal: prints warnings but does not abort. Useful for catching drift
    # between the launcher's ROLE_LAUNCH_META and the registry's source-of-truth
    # process_layout section.
    if not args.dev:
        registry_warnings = validate_against_registry()
        if registry_warnings:
            print("[0] Registry classification warnings:")
            for w in registry_warnings:
                print(f"  ⚠ {w}")
            print()

    # [0.7] Episodic embedding health check (A3, 2026-05-21).
    # Detects the orphan-FAISS condition where the live episodic.db has many more
    # routing memories than FAISS vectors — e.g. after a FAISS reset or a BGE
    # outage during write. KNN fallback (DAR-2 contrastive sharpening, low-confidence
    # routing fallback) silently degrades in that state.
    #
    # Read-only by default — does not block startup. To repair, run:
    #   python3 scripts/maintenance/repair_episodic_embeddings.py --repair
    # Or pass --repair-embeddings to this command for auto-repair before launch.
    if not args.dev:
        try:
            from scripts.maintenance.repair_episodic_embeddings import (
                diagnose as _diagnose_embeddings,
                print_report as _print_embedding_report,
                run_repair as _run_embedding_repair,
                DEFAULT_DB_PATH,
                DEFAULT_FAISS_PATH,
                DEFAULT_ID_MAP_PATH,
                DEFAULT_REEMBEDDED_PATH,
            )

            print("[0.7] Episodic embedding health check...")
            _report = _diagnose_embeddings(
                DEFAULT_DB_PATH,
                DEFAULT_FAISS_PATH,
                DEFAULT_REEMBEDDED_PATH,
            )
            _print_embedding_report(_report)
            if not _report.healthy:
                if getattr(args, "repair_embeddings", False):
                    print("\n[0.7] --repair-embeddings: starting bulk re-embed + FAISS rebuild...")
                    print("      (launches 8 BGE servers; expected 5-15 min)")
                    _run_embedding_repair(
                        db_path=DEFAULT_DB_PATH,
                        faiss_path=DEFAULT_FAISS_PATH,
                        id_map_path=DEFAULT_ID_MAP_PATH,
                        reembedded_path=DEFAULT_REEMBEDDED_PATH,
                    )
                    print("[0.7] Re-running diagnostic post-repair:")
                    _report2 = _diagnose_embeddings(
                        DEFAULT_DB_PATH,
                        DEFAULT_FAISS_PATH,
                        DEFAULT_REEMBEDDED_PATH,
                    )
                    _print_embedding_report(_report2)
                    if not _report2.healthy:
                        print("[!] WARNING: repair did not restore health — proceeding anyway.")
                else:
                    print(
                        "[!] Episodic store is ORPHANED. KNN fallback path will silently degrade."
                    )
                    print(
                        "    Repair: python3 scripts/maintenance/repair_episodic_embeddings.py --repair"
                    )
                    print("    Or re-run with --repair-embeddings to auto-repair.")
            print()
        except ImportError as exc:
            # Maintenance script not present (older deployment) — proceed without check.
            print(f"[0.7] Skipping embedding health check (maintenance module unavailable: {exc})")
            print()
        except Exception as exc:
            # Read-only diagnostic should never raise. If it does, log and continue.
            print(f"[0.7] Embedding health check failed: {exc} (proceeding anyway)")
            print()

    # Determine which servers to start
    servers_to_start = []

    if args.dev:
        print("[1] Starting in DEV mode (single 0.5B model)...")
        servers_to_start = [{"port": 8080, "roles": ["dev"]}]
    elif args.only:
        # --only: start ONLY the specified roles, nothing else
        requested = set(args.only)
        print(f"[1] Selective start: {', '.join(sorted(requested))}")
        for server in HOT_SERVERS + WARM_SERVERS:
            if requested & set(server["roles"]):
                servers_to_start.append(server)
                print(f"  Including: port {server['port']} ({', '.join(server['roles'])})")
        if not servers_to_start:
            print(f"  [!] No servers matched roles: {', '.join(sorted(requested))}")
            print(
                f"  Available roles: {', '.join(sorted({r for s in HOT_SERVERS + WARM_SERVERS for r in s['roles']}))}"
            )
            return 1
    else:
        print("[1] Starting HOT servers...")
        servers_to_start = HOT_SERVERS.copy()

        # Add warm servers if requested
        if args.include_warm:
            for warm_server in WARM_SERVERS:
                for role in warm_server["roles"]:
                    if role in args.include_warm:
                        servers_to_start.append(warm_server)
                        print(f"  Including WARM server: port {warm_server['port']} ({role})")
                        break

    # Apply --numa-mode filter (default 'both' for back-compat — pre-2026-05-08 default).
    # Picks full XOR quarters for any role with full_instance_idx + multiple instances
    # (currently frontdoor + coder_escalation + worker_general); single-instance roles
    # pass through. See launcher-numa-mode-gating handoff.
    numa_mode = getattr(args, "numa_mode", "both")
    if numa_mode == "both":
        # Light advisory only — 'both' has been working for frontdoor/coder_escalation since
        # 2026-03 (Qwen3.6-35B Q8 quarters tuned to coexist with the full instance). The
        # gemma4-MTP exception is the one that needs --numa-mode full per role. We don't
        # spam at every start since most roles are fine.
        if any("worker_general" in s.get("roles", []) for s in servers_to_start):
            print(
                f"  [advisory] worker_general (gemma4-MTP) runs at -t 96; if its full + 4 quarters "
                f"are all kept (default 'both'), expect 1.5× CPU oversubscription. "
                f"Use '--numa-mode full' (single instance) or '--numa-mode quarter' (4 concurrent) "
                f"for that role specifically. See launcher-numa-mode-gating.md."
            )
    pre_filter_count = len(servers_to_start)
    servers_to_start = _filter_by_numa_mode(servers_to_start, numa_mode)
    if numa_mode != "both" and len(servers_to_start) != pre_filter_count:
        dropped = pre_filter_count - len(servers_to_start)
        print(
            f"  [--numa-mode={numa_mode}] dropped {dropped} overlapping instance(s); "
            f"{len(servers_to_start)} server(s) to start"
        )

    print()

    # [1.5] Page-cache prewarm — distribute shared-GGUF pages across NUMA nodes
    # under `numactl --interleave=all` BEFORE the per-instance launches mlock
    # them. Otherwise sequential mlock pins every page of a shared model onto
    # whichever node the first launcher's --membind targeted, and remote-node
    # quarters cross-socket-fetch for the whole stack lifetime (~50-65% t/s drop
    # observed 2026-05-28 after a container rebuild evicted the page cache).
    # See handoffs/active/numa-page-cache-prewarm.md.
    _prewarm_all(
        servers_to_start,
        _orchestrator_stack().build_server_command,
        registry,
        args=args,
    )
    print()

    # Check target ports — skip healthy, clean up unhealthy
    target_ports = {s["port"] for s in servers_to_start}
    print("[2] Checking target ports...")
    already_healthy_ports: set[int] = set()
    for server in servers_to_start:
        port = server["port"]
        if is_port_in_use(port):
            if wait_for_health(port, timeout=3):
                print(f"  Port {port} already healthy, skipping")
                already_healthy_ports.add(port)
                continue
            print(f"  Port {port} in use but unhealthy, cleaning up...")
            try:
                for pid in _pids_on_port(port):
                    kill_process(pid)
            except Exception as e:
                print(f"  [!] Error cleaning port {port}: {e}")
    if already_healthy_ports:
        print(f"  Preserved {len(already_healthy_ports)} healthy server(s)")

    print()

    # Start servers sequentially (skip already-healthy ports)
    print("[3] Starting llama-servers...")
    for i, server in enumerate(servers_to_start):
        port = server["port"]
        roles = server["roles"]

        if port in already_healthy_ports:
            role_label = roles[0] if roles else str(port)
            print(f"  Skipping port {port}: {role_label} (already healthy)")
            # Record existing server in state so status reporting works
            preserved = _preserved_process_info(
                role_label,
                port,
                f"preserved:{role_label}",
                str(LOG_DIR / f"llama-server-{port}.log"),
            )
            if preserved:
                state[f"server_{port}"] = preserved
                for role in roles:
                    if role not in state:
                        state[role] = preserved
            continue

        embedding_mode = server.get("embedding", False)
        worker_pool_mode = server.get("worker_pool", False)
        worker_type = server.get("worker_type")
        vision_mode = server.get("vision", False)
        vision_type = server.get("vision_type")
        numa_instance = server.get("numa_instance", 0)

        info = start_server(
            port,
            roles,
            registry,
            args.dev,
            embedding_mode=embedding_mode,
            worker_pool_mode=worker_pool_mode,
            worker_type=worker_type,
            vision_mode=vision_mode,
            vision_type=vision_type,
            numa_instance=numa_instance,
        )
        if info:
            state[f"server_{port}"] = info
            # Also map all roles to this server
            for role in roles:
                if role not in state:
                    state[role] = info
        else:
            print(f"  [!] Failed to start server on port {port}")
            # Embedding/worker_pool/vision server failure is non-fatal (fallback available)
            is_optional = embedding_mode or worker_pool_mode or vision_mode
            if not args.dev and not is_optional:
                return 1

        # Sequential loading: wait for this server to be healthy before launching
        # the next one. Concurrent mlock on large models causes crashes even when
        # total RAM is sufficient (race condition during page fault + lock).
        is_small_model = (
            embedding_mode
            or (worker_pool_mode and worker_type == "fast")
            or (vision_mode and vision_type != "escalation")
        )
        if i < len(servers_to_start) - 1 and not args.dev and not is_small_model:
            if not wait_for_health(port, timeout=300):
                print(f"  [!] Server on port {port} did not become healthy within 300s")
            else:
                print(f"  Server on port {port} healthy, proceeding to next")

    print()

    # Start orchestrator (skip if already healthy, or if --only was used for model servers)
    if args.only:
        print("[4] Skipping orchestrator API (--only mode)")
        if wait_for_health(8000, timeout=2):
            print("  Orchestrator already healthy")
            preserved = _preserved_process_info(
                "orchestrator", 8000, "uvicorn", str(LOG_DIR / "orchestrator.log")
            )
            if preserved:
                state["orchestrator"] = preserved
        else:
            print("  [i] Orchestrator not running — start separately if needed")
    elif 8000 in already_healthy_ports:
        print("[4] Starting orchestrator API...")
        print("  Orchestrator already healthy, skipping")
        preserved = _preserved_process_info(
            "orchestrator", 8000, "uvicorn", str(LOG_DIR / "orchestrator.log")
        )
        if preserved:
            state["orchestrator"] = preserved
    else:
        info = start_orchestrator(getattr(args, "profile", None))
        if info:
            state["orchestrator"] = info
        else:
            print("  [!] Failed to start orchestrator")
            return 1

    print()

    # Start document formalizer (optional, non-fatal)
    if not args.dev and not args.only:
        print("[5] Starting document formalizer (LightOnOCR-2)...")
        info = None
        if is_port_in_use(9001) and wait_for_health(9001, timeout=3):
            print("  Already healthy, skipping")
            info = _preserved_process_info(
                "document_formalizer",
                9001,
                "LightOnOCR-2",
                str(LOG_DIR / "document_formalizer.log"),
            )
        else:
            info = start_document_formalizer()
        if info:
            state["document_formalizer"] = info
        else:
            print("  [!] Document formalizer failed (non-fatal, continuing)")

        print()

        # Start sd-server diffusion service (optional, non-fatal)
        # ERNIE-Image-Turbo Q8 GGUF + Mistral3 + flux2 VAE via stable-diffusion.cpp.
        # Replaced ComfyUI 2026-05-07 — see start_sd_server() for context.
        if is_port_in_use(8190) and wait_for_health(8190, timeout=3, path="/sdapi/v1/samplers"):
            print("[5a] Starting sd-server (ggml native diffusion)...")
            print("  Already healthy, skipping")
            preserved = _preserved_process_info(
                "sd_server",
                8190,
                "ernie-image-turbo-Q8_0.gguf + ministral-3-3b + flux2-vae",
                str(LOG_DIR / "sd_server.log"),
            )
            if preserved:
                state["sd_server"] = preserved
        else:
            print("[5a] Starting sd-server (ggml native diffusion)...")
            info = start_sd_server()
            if info:
                state["sd_server"] = info
            else:
                print("  [!] sd-server failed (non-fatal, image generation unavailable)")

        print()

        # Start Whisper STT service (optional, non-fatal)
        # Promoted from sidecar 2026-05-06.
        if is_port_in_use(9000) and wait_for_health(9000, timeout=3, path="/health"):
            print("[5b] Starting Whisper STT server...")
            print("  Already healthy, skipping")
            preserved = _preserved_process_info(
                "whisper",
                9000,
                "faster-whisper-large-v3-turbo-int8",
                str(LOG_DIR / "whisper.log"),
            )
            if preserved:
                state["whisper"] = preserved
        else:
            print("[5b] Starting Whisper STT server...")
            info = start_whisper()
            if info:
                state["whisper"] = info
            else:
                print("  [!] Whisper failed (non-fatal, STT unavailable)")

        print()

        # Start Docker services (NextPLAID retrieval + SearXNG metasearch)
        if _docker_available():
            print("[5.5] Starting Docker services (NextPLAID retrieval + SearXNG metasearch)...")
            for service in DOCKER_SERVICES:
                info = start_docker_container(service)
                if info:
                    state[service["name"]] = info
                else:
                    svc_name = service["name"]
                    if svc_name == "searxng":
                        print(
                            f"  [!] {svc_name} failed (non-fatal, web_search falls back to DDG HTML scraping)"
                        )
                    else:
                        print(
                            f"  [!] {svc_name} failed (non-fatal, code_search degrades gracefully)"
                        )
            print()
        else:
            print("[5.5] Docker not available, skipping Docker containers")
            print("  code_search/doc_search will be unavailable")
            print("  web_search will use DDG HTML scraping fallback")
            print()

        # Initialize MemRL databases and tool registry
        init_memrl_and_tools()

        print()

    # Save state
    save_state(state)
    print(f"[i] State saved to {STATE_FILE}")
    print()

    # Final status
    print("=" * 60)
    print("STACK READY")
    print("=" * 60)
    cmd_status(args)

    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop components."""
    state = load_state()

    if not state and args.all:
        # State file empty — fall back to port scanning
        found = _scan_known_ports()
        if not found:
            print("No running components found")
            return 0

        print(f"State file empty but found processes on {len(found)} ports (port scan fallback)")
        killed = 0
        for port, pids in sorted(found.items()):
            for pid in pids:
                print(f"  Stopping PID {pid} on port {port}...")
                if kill_process(pid):
                    print(f"    [OK] Stopped")
                    killed += 1
                else:
                    print(f"    [!] Failed to stop")
        print(f"Stopped {killed} orphaned processes")
        save_state({})
        return 0

    if not state:
        print("No running components found")
        return 0

    targets = []
    if args.all:
        targets = list(state.keys())
    elif args.components:
        targets = args.components
    else:
        print("Specify --all or component names")
        return 1

    for name in targets:
        if name in state:
            info = state[name]
            if info.pid == -1:
                # Docker-managed container
                print(f"Stopping Docker container {name}...")
                if stop_docker_container(info.role):
                    del state[name]
                    print(f"  [OK] Stopped")
                else:
                    print(f"  [!] Failed to stop container {name}")
            else:
                print(f"Stopping {name} (PID {info.pid})...")
                if kill_process(info.pid):
                    del state[name]
                    print(f"  [OK] Stopped")
                else:
                    print(f"  [!] Failed to stop")
        else:
            print(f"  [?] {name} not found in state")

    save_state(state)

    # After state-based stop, scan for orphans that survived
    if args.all:
        orphans = _scan_known_ports()
        if orphans:
            print(
                f"\nFound {sum(len(p) for p in orphans.values())} orphaned processes on {len(orphans)} ports"
            )
            for port, pids in sorted(orphans.items()):
                for pid in pids:
                    print(f"  Stopping orphan PID {pid} on port {port}...")
                    if kill_process(pid):
                        print(f"    [OK] Stopped")
                    else:
                        print(f"    [!] Failed to stop")

    return 0


def cmd_reload(args: argparse.Namespace) -> int:
    """Reload components."""
    state = load_state()
    registry = RegistryLoader()

    for component in args.components:
        print(f"Reloading {component}...")

        # Special case: reload all embedders at once
        if component == "embedders":
            print("  Reloading all 6 BGE embedder instances...")

            # Kill by state file entries
            for port in EMBEDDER_PORTS:
                key = f"server_{port}"
                role = "embedder" if port == 8090 else f"embedder_{port - 8090}"
                if key in state:
                    kill_process(state[key].pid)
                    del state[key]
                if role in state:
                    del state[role]

            # Also kill by port (in case state is stale)
            for port in EMBEDDER_PORTS:
                if is_port_in_use(port):
                    try:
                        for pid in _pids_on_port(port):
                            kill_process(pid)
                            print(f"    Killed stale process on port {port}")
                    except (subprocess.TimeoutExpired, OSError, ValueError):
                        pass  # Best-effort stale process cleanup

            time.sleep(2)  # Wait for ports to free

            # Start all embedders
            success_count = 0
            for port in EMBEDDER_PORTS:
                role = "embedder" if port == 8090 else f"embedder_{port - 8090}"
                info = start_server(
                    port,
                    [role],
                    registry,
                    dev_mode=False,
                    embedding_mode=True,
                )
                if info:
                    state[f"server_{port}"] = info
                    state[role] = info
                    success_count += 1

            print(f"  [OK] {success_count}/{len(EMBEDDER_PORTS)} embedders restarted")
            if success_count == 0:
                return 1
            continue

        elif component == "orchestrator":
            # Stop by authoritative listener port only.
            # State-file PIDs can go stale and be reused by unrelated processes.
            for pid in _pids_on_port(8000):
                kill_process(pid)
            time.sleep(1)

            # Start new
            info = start_orchestrator(getattr(args, "profile", None))
            if info:
                state["orchestrator"] = info
            else:
                print(f"  [!] Failed to restart orchestrator")
                return 1

        elif component in PORT_MAP:
            port = PORT_MAP[component]
            key = f"server_{port}"

            # Find roles and config for this port
            roles = [component]
            worker_pool_mode = False
            worker_type = None
            embedding_mode = False
            vision_mode = False
            vision_type = None

            numa_instance = 0
            for server in HOT_SERVERS + WARM_SERVERS:
                if server["port"] == port:
                    roles = server["roles"]
                    worker_pool_mode = server.get("worker_pool", False)
                    worker_type = server.get("worker_type")
                    embedding_mode = server.get("embedding", False)
                    vision_mode = server.get("vision", False)
                    vision_type = server.get("vision_type")
                    numa_instance = server.get(
                        "numa_instance", 0
                    )  # fix: reload must preserve per-quarter -t
                    break

            # Stop existing
            # Stop by authoritative listener port only.
            # State-file PIDs can go stale and be reused by unrelated processes.
            for pid in _pids_on_port(port):
                kill_process(pid)
            time.sleep(1)

            # Start new
            info = start_server(
                port,
                roles,
                registry,
                dev_mode=False,
                embedding_mode=embedding_mode,
                worker_pool_mode=worker_pool_mode,
                worker_type=worker_type,
                vision_mode=vision_mode,
                vision_type=vision_type,
                numa_instance=numa_instance,
            )
            if info:
                state[key] = info
                for role in roles:
                    state[role] = info
            else:
                print(f"  [!] Failed to restart {component}")
                return 1

        else:
            # Check if it's a Docker service
            docker_service = None
            for svc in DOCKER_SERVICES:
                if component == svc["name"]:
                    docker_service = svc
                    break

            if docker_service:
                print(f"  Reloading Docker service {component}...")
                stop_docker_container(component)
                time.sleep(2)
                info = start_docker_container(docker_service)
                if info:
                    state[component] = info
                else:
                    print(f"  [!] Failed to restart {component}")
                    return 1
            else:
                print(f"  [?] Unknown component: {component}")

    save_state(state)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of all components."""
    state = load_state()

    if not state:
        print("No components running")
        return 0

    print()
    print(f"{'COMPONENT':<25} {'PORT':<8} {'PID':<10} {'STATUS':<10} {'MODEL'}")
    print("-" * 80)

    seen_pids = set()
    for name, info in sorted(state.items()):
        if info.pid != -1 and info.pid in seen_pids:
            continue  # Skip duplicates (roles sharing servers)
        seen_pids.add(info.pid)

        if info.pid == -1:
            # Docker-managed container
            alive = docker_container_running(info.role)
            # Look up health_path for this service (SearXNG uses /, others use /health)
            health_path = "/health"
            for svc in DOCKER_SERVICES:
                if svc["name"] == info.role:
                    health_path = svc.get("health_path", "/health")
                    break
            healthy = wait_for_health(info.port, timeout=3, path=health_path) if alive else False
            status = "healthy" if healthy else ("running" if alive else "stopped")
            pid_str = "docker"
        else:
            # Native process
            try:
                os.kill(info.pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            healthy = wait_for_health(info.port, timeout=3) if alive else False
            if not alive and is_port_in_use(info.port):
                # PID drift can happen if the original launcher PID exits while
                # a listener remains healthy on the same port.
                replacement_pids = _pids_on_port(info.port)
                if replacement_pids:
                    replacement_pid = replacement_pids[0]
                    info.pid = replacement_pid
                    state[name] = info
                    alive = True
                    healthy = wait_for_health(info.port, timeout=3)
            status = "healthy" if healthy else ("running" if alive else "dead")
            pid_str = str(info.pid)

        model = Path(info.model_path).stem if info.model_path != "uvicorn" else "uvicorn"

        print(f"{name:<25} {info.port:<8} {pid_str:<10} {status:<10} {model[:30]}")

    print()
    print(f"State file: {STATE_FILE}")
    save_state(state)
    return 0


# =============================================================================
# MemRL and Tool Registry Initialization
# =============================================================================


def init_memrl_and_tools() -> bool:
    """Initialize MemRL databases and tool registry for the session.

    This ensures all deterministic tools (41 total) are ready and
    the REPL memory system is initialized with seed examples.
    """
    success = True

    # [6] REPL Memory Initialization
    print("[6] Initializing MemRL databases...")

    # Initialize REPL seed examples
    seed_loader_path = _PATHS["project_root"] / "orchestration/repl_memory/seed_loader.py"
    if seed_loader_path.exists():
        result = subprocess.run(
            [sys.executable, str(seed_loader_path), "--init"],
            capture_output=True,
            text=True,
            cwd=str(_PATHS["project_root"]),
        )
        if result.returncode == 0:
            print("  [OK] REPL seed examples loaded")
        else:
            print(
                f"  [WARN] Seed loader failed: {result.stderr[:100] if result.stderr else 'no output'}"
            )

    # Warm up all embedding servers with test query
    try:
        import urllib.request
        import urllib.error

        test_payload = json.dumps({"content": "test embedding warmup"}).encode()
        healthy_count = 0
        for port in EMBEDDER_PORTS:
            try:
                req = urllib.request.Request(
                    f"http://localhost:{port}/embedding",
                    data=test_payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        healthy_count += 1
            except (urllib.error.URLError, TimeoutError, OSError):
                pass  # Expected during warmup — server may still be starting
        print(f"  [OK] Embedding servers warmed up: {healthy_count}/{len(EMBEDDER_PORTS)} healthy")
    except Exception as e:
        print(f"  [WARN] Embedding warmup failed: {e}")

    # [7] Tool Registry Initialization
    print("[7] Initializing deterministic tool registry...")

    # Validate tool registry exists
    tool_registry_path = _PATHS["project_root"] / "orchestration/tool_registry.yaml"
    if not tool_registry_path.exists():
        print(f"  [!] Tool registry not found: {tool_registry_path}")
        success = False
    else:
        # Load and validate tool executor
        try:
            # Add src to path for imports
            import sys as _sys

            _sys.path.insert(0, str(_PATHS["project_root"]))
            from orchestration.tools.executor import get_executor

            executor = get_executor()
            tools = executor.list_tools()
            print(f"  [OK] Tool registry loaded: {len(tools)} tools")

            # Categorize tools
            categories: dict[str, int] = {}
            for t in tools:
                cat = t.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1
            for cat, count in sorted(categories.items()):
                print(f"      {cat}: {count}")
        except Exception as e:
            print(f"  [WARN] Tool executor init failed: {e}")

    # Verify C++ math tools binary
    cpp_binary = LLAMA_MATH_TOOLS
    if cpp_binary.exists():
        print(f"  [OK] C++ math tools binary found: {cpp_binary}")
    else:
        print(f"  [WARN] C++ math tools not built: {cpp_binary}")
        print(
            f"        Run: cd {_PATHS['llm_root']}/llama.cpp/tools/math-tools "
            "&& cmake -B build && cmake --build build"
        )

    return success


# =============================================================================
# Checkpoint Hooks for Self-Management Procedures
# =============================================================================

CHECKPOINT_DIR = _PATHS["project_root"] / "orchestration/checkpoints"
_REGISTRY_PATH_FOR_CHECKPOINT = _PATHS["project_root"] / "orchestration/model_registry.yaml"


def checkpoint_create(name: str, include_state: bool = True) -> dict[str, Any]:
    """Create a checkpoint (wrapper supplying CHECKPOINT_DIR + STATE_FILE)."""
    return _stack_checkpoint.checkpoint_create(
        name,
        CHECKPOINT_DIR,
        STATE_FILE,
        include_state=include_state,
        registry_path=_REGISTRY_PATH_FOR_CHECKPOINT,
    )


def checkpoint_restore(checkpoint_id: str) -> dict[str, Any]:
    """Restore from a checkpoint (wrapper supplying CHECKPOINT_DIR + STATE_FILE)."""
    return _stack_checkpoint.checkpoint_restore(checkpoint_id, CHECKPOINT_DIR, STATE_FILE)


def checkpoint_list(limit: int = 10) -> list[dict[str, Any]]:
    """List available checkpoints (wrapper supplying CHECKPOINT_DIR)."""
    return _stack_checkpoint.checkpoint_list(CHECKPOINT_DIR, limit=limit)


def checkpoint_delete(checkpoint_id: str) -> bool:
    """Delete a checkpoint (wrapper supplying CHECKPOINT_DIR)."""
    return _stack_checkpoint.checkpoint_delete(checkpoint_id, CHECKPOINT_DIR)


# Export hooks for use by procedure_registry
__checkpoint_hooks__ = {
    "create": checkpoint_create,
    "restore": checkpoint_restore,
    "list": checkpoint_list,
    "delete": checkpoint_delete,
}
