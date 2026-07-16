#!/usr/bin/env python3
"""Unified orchestrator stack launcher.

Launches all models + orchestrator with granular reload support.

Usage:
    orchestrator_stack.py start [--hot-only] [--include-warm ROLE...] [--only ROLE...] [--dev]
    orchestrator_stack.py stop [--all | COMPONENT...]
    orchestrator_stack.py reload COMPONENT...
    orchestrator_stack.py status

Examples:
    # Start ONLY specific roles (skip everything else, preserve what's running)
    ./orchestrator_stack.py start --only worker_vision vision_escalation

    # Start HOT models only
    ./orchestrator_stack.py start --hot-only

    # Start with warm architect
    ./orchestrator_stack.py start --include-warm architect_general

    # Dev mode (single 0.5B model)
    ./orchestrator_stack.py start --dev

    # Reload orchestrator API after code changes
    ./orchestrator_stack.py reload orchestrator

    # Check status
    ./orchestrator_stack.py status
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, MutableMapping

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Re-exec under the project venv so heavy deps (yaml, faiss, numpy, ...)
# resolve regardless of the caller's PATH or active interpreter. System
# Python in a fresh devcontainer typically lacks PyYAML — this guard
# means `python3 scripts/server/orchestrator_stack.py ...` always works.
_PROJECT_VENV_PY = Path(__file__).resolve().parents[2] / ".venv/bin/python"
if (
    _PROJECT_VENV_PY.exists()
    and Path(sys.executable).resolve() != _PROJECT_VENV_PY.resolve()
    and os.environ.get("ORCHESTRATOR_STACK_REEXEC") != "1"
):
    os.environ["ORCHESTRATOR_STACK_REEXEC"] = "1"
    os.execv(str(_PROJECT_VENV_PY), [str(_PROJECT_VENV_PY), __file__, *sys.argv[1:]])

from scripts.server import stack_processes as _stack_processes
from scripts.server.stack_env import (
    build_launch_env,
)
from scripts.server.stack_health import wait_for_health as _wait_for_health
from scripts.server.fleet_markers import (
    LAUNCH_SOURCE_STACK_COMMANDS as _FLEET_SRC_STACK,
    write_llama_marker as _write_llama_marker,
    write_orchestrator_marker as _write_orchestrator_marker,
)
from scripts.server.stack_runtime import (
    runtime_requirements_for_role as _runtime_requirements_for_role_impl,
)
from scripts.server.stack_manifest import (
    DEV_MODEL,
    DEV_MODEL_PATH,
    DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
    DEFAULT_UBATCH_TOKENS,
    EMBEDDER_PORTS,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_SERVER_RECIPES,
    EXPLORE_DRAFT_MODEL,
    LAUNCH_CONTEXT_TOKENS,
    LAUNCH_KV_QUANT_CONFIGS,
    NO_SPEC_DECODE_ROLES,
    ORCHESTRATOR_PROFILES,
    SERIAL_ROLES,
    VISION_ESCALATION_MMPROJ,
    VISION_ESCALATION_MODEL,
    VISION_WORKER_MMPROJ,
    VISION_WORKER_MODEL,
    WORKER_POOL_MODELS,
)
from scripts.server.stack_paths import (
    _HEALTH_SERVER_STARTUP,
    _HEALTH_VISION_SERVER,
    _HEALTH_WORKER_SERVER,
    _PATHS,
    _V2_ROLES,
    LLAMA_SERVER,
    LLAMA_SERVER_V2,
    LOG_DIR,
    SLOT_SAVE_DIR,
    STATE_FILE,
)
from scripts.server.stack_numa import (
    MLOCK_ROLES,
    NUMA_CONFIG,
    _numa_prefix,
)
from scripts.server.stack_state import (
    ProcessInfo,
    load_state_file as _load_state_file,
    save_state_file as _save_state_file,
)
from src.config import _registry_timeout
from src.registry_loader import RegistryLoader

# =============================================================================

STACK_PRIORS_PATH = _PATHS["project_root"] / "orchestration/derived/stack_priors.yaml"
_WORKER_GENERAL_DEGRADED_FALLBACK = {
    "spec_type": "draft-mtp",
    "draft_max": 2,
    "draft_p_min": 0.0,
    "threads_draft": 16,
    "ubatch": 512,
    "kv_type_k": "q8_0",
    "kv_type_v": "q8_0",
    "context_tokens": 16384,
}

_CPU_ONLY_DEVICE_FLAGS = ("--device", "-dev")
_CPU_ONLY_DRAFT_DEVICE_FLAGS = ("--device-draft", "-devd")


def _has_any_flag(cmd: list[str], flags: tuple[str, ...]) -> bool:
    return any(flag in cmd for flag in flags)


def _append_cpu_only_device_args(cmd: list[str]) -> None:
    """Pin stack-launched llama-server roles to CPU devices.

    The production stack's text roles are CPU roles. A HIP-capable v7 binary will
    otherwise auto-select ROCm0 for host op offload / draft sampling and regress
    worker_general ngram+MTP throughput on CPU-only launches.
    """
    if not _has_any_flag(cmd, _CPU_ONLY_DEVICE_FLAGS):
        cmd.extend(["--device", "none"])
    if (
        ("--spec-type" in cmd or "-md" in cmd)
        and not _has_any_flag(cmd, _CPU_ONLY_DRAFT_DEVICE_FLAGS)
    ):
        cmd.extend(["--device-draft", "none"])


def _repo_short_sha(path: Path | None = None) -> str | None:
    repo = path or _PATHS["project_root"]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _stack_prior_launch(role_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return generated launch requirements/runtime for a live role, if usable."""
    from src.registry.stack_priors import live_stack_role_records, stack_prior_serving

    record = live_stack_role_records(STACK_PRIORS_PATH).get(role_name)
    if not isinstance(record, dict):
        return {}, {}
    serving = stack_prior_serving(record)
    launch = serving.get("launch")
    if not isinstance(launch, dict):
        return {}, {}
    requirements = launch.get("requirements")
    runtime = launch.get("runtime")
    return (
        requirements if isinstance(requirements, dict) else {},
        runtime if isinstance(runtime, dict) else {},
    )


def _runtime_cache(runtime: dict[str, Any]) -> dict[str, Any]:
    cache = runtime.get("cache")
    return cache if isinstance(cache, dict) else {}


def _runtime_flags(runtime: dict[str, Any]) -> dict[str, Any]:
    flags = runtime.get("flags")
    return flags if isinstance(flags, dict) else {}


def _runtime_positive_int(
    container: dict[str, Any],
    key: str,
    fallback: int | str,
) -> str:
    value = container.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return str(value)
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return value
    return str(fallback)


def _runtime_number_string(
    container: dict[str, Any],
    key: str,
    fallback: int | float | str,
) -> str:
    value = container.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and value:
        try:
            float(value)
        except ValueError:
            return str(fallback)
        return value
    return str(fallback)


def _runtime_string(container: dict[str, Any], key: str, fallback: str) -> str:
    value = container.get(key)
    return value if isinstance(value, str) and value else fallback


def _same_real_model_path(left: str, right: str) -> bool:
    return os.path.realpath(left) == os.path.realpath(right)


def _append_spec_decode_args(
    cmd: list[str],
    *,
    model_path: str,
    draft_model_path: str | None,
    spec_type: str | None,
    draft_max: str | None,
    draft_p_min: str | None = None,
    threads_draft: str | None = None,
) -> None:
    if draft_model_path and not _same_real_model_path(model_path, draft_model_path):
        cmd.extend(["-md", draft_model_path])
    if spec_type:
        cmd.extend(["--spec-type", spec_type])
    if draft_max:
        cmd.extend(["--spec-draft-n-max", draft_max])
    if draft_p_min is not None:
        cmd.extend(["--draft-p-min", draft_p_min])
    if threads_draft:
        cmd.extend(["--threads-draft", threads_draft])


def _append_runtime_kv_args(cmd: list[str], cache: dict[str, Any]) -> None:
    kv_type_k = cache.get("kv_type_k")
    kv_type_v = cache.get("kv_type_v")
    if isinstance(kv_type_k, str) and isinstance(kv_type_v, str):
        cmd.extend(["-ctk", kv_type_k, "-ctv", kv_type_v])
    if cache.get("kv_hadamard") is True:
        # 2026-06-26 v6 cutover: --kv-hadamard removed in v6 (no role sets kv_hadamard true today)
        # cmd.append("--kv-hadamard")
        pass


def _append_runtime_spec_args(cmd: list[str], runtime: dict[str, Any], model_path: str) -> None:
    spec = _runtime_flags(runtime).get("spec")
    if not isinstance(spec, dict) or spec.get("enabled") is not True:
        return
    draft_model_path = spec.get("draft_model_path")
    if not isinstance(draft_model_path, str) or not draft_model_path:
        return
    spec_type = spec.get("type")
    draft_max = spec.get("draft_max")
    draft_p_min = spec.get("draft_p_min")
    threads_draft = spec.get("threads_draft")
    _append_spec_decode_args(
        cmd,
        model_path=model_path,
        draft_model_path=draft_model_path,
        spec_type=spec_type if isinstance(spec_type, str) and spec_type else None,
        # 2026-06-26 v6 cutover: --draft-max renamed to --spec-draft-n-max (same value)
        draft_max=(
            str(draft_max)
            if isinstance(draft_max, int) and not isinstance(draft_max, bool) and draft_max > 0
            else None
        ),
        draft_p_min=(
            str(float(draft_p_min))
            if isinstance(draft_p_min, (int, float)) and not isinstance(draft_p_min, bool)
            else None
        ),
        threads_draft=(
            str(threads_draft)
            if isinstance(threads_draft, int)
            and not isinstance(threads_draft, bool)
            and threads_draft > 0
            else None
        ),
    )


# Path/binary constants moved to scripts/server/stack_paths.py (2026-05-22).
# Manifest (PORT_MAP, ROLE_LAUNCH_META, model paths, classification helpers,
# validate_model_paths, validate_against_registry, etc.) moved to
# scripts/server/stack_manifest.py. All names are re-exported via the imports
# at the top of this file so `from orchestrator_stack import X` works for the
# registry compiler fallback path (src/registry/registry_compiler.py:266).
#
# Port topology note: this launcher intentionally no longer owns the static
# port tables. Full/primary role ports live in stack_manifest.PORT_MAP; NUMA
# quarter/replica ports and their CPU pinning live in stack_numa.NUMA_CONFIG.
# Avoid documenting fixed 808x/818x ranges here because the current topology is
# role-specific (for example 8070 full + 8080/8180/8280/8380 quarters for
# frontdoor, 8072 full + 8082/8182/8282/8382 quarters for worker_general, and
# no 8084 architect_coding server).
# =============================================================================


# =============================================================================
# State Management
# =============================================================================


def load_state() -> dict[str, ProcessInfo]:
    """Load state from file."""
    return _load_state_file(STATE_FILE)


def save_state(state: dict[str, ProcessInfo]) -> None:
    """Save state to file."""
    _save_state_file(STATE_FILE, state)


# =============================================================================
# Process Management
# =============================================================================


def check_free_memory() -> int:
    """Return free memory in GB (delegates to stack_processes.free_memory_gb)."""
    return _stack_processes.free_memory_gb()


# =============================================================================
# Host prerequisites moved to scripts/server/stack_host.py (2026-05-21 refactor).
# Per-role env blocks moved to scripts/server/stack_env.py (same session).
# Re-exported above via the module-level imports so existing call sites
# (start_server, cmd_start etc.) keep resolving these unqualified names.
# =============================================================================


def _runtime_requirements_for_role(
    registry: "RegistryLoader", role_name: str
) -> tuple[str | None, list[str] | None]:
    """Return (binary_dir, ld_library_paths) for a role (delegates to stack_runtime)."""
    return _runtime_requirements_for_role_impl(registry, role_name)


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    return _stack_processes.is_port_in_use(port)


def _pids_on_port(port: int) -> list[int]:
    """Best-effort discovery of LISTEN pids on a TCP port.

    NB: WITHOUT the `-sTCP:LISTEN` filter, `lsof -i :PORT` also returns the
    PIDs of any client process with an ESTABLISHED connection to that port —
    e.g. a Firefox tab open to localhost:8000/dashboard, or autopilot mid-
    HTTP request. Killing those clients was the destructive bug observed
    2026-05-21 where `reload orchestrator` would kill Firefox + autopilot.
    The filter ensures we only target actual listeners.
    """
    return _stack_processes.pids_on_port(port)


def _pid_alive(pid: int) -> bool:
    """Return True when a pid currently exists."""
    return _stack_processes.pid_alive(pid)


def _child_pids(pid: int) -> list[int]:
    """Return direct child pids for a process."""
    return _stack_processes.child_pids(pid)


def _collect_descendants(root_pid: int) -> list[int]:
    """Collect all descendants of root_pid (breadth-first)."""
    return _stack_processes.collect_descendants(root_pid)


def _renice_all_threads(pid: int, nice: int) -> None:
    """Renice every thread of `pid` to `nice`."""
    _stack_processes.renice_all_threads(pid, nice)


def _set_oom_protection(pids: list[int], adj: int = -1000) -> int:
    """Protect control-plane pids from earlyoom via oom_score_adj=-1000.

    See stack_processes.set_oom_score_adj — best-effort (`sudo -n`); the durable
    replacement for the manual one-shot `choom` that did not survive an API restart.
    """
    return _stack_processes.set_oom_score_adj(pids, adj)


def kill_process(pid: int, timeout: int = 5) -> bool:
    """Kill a process tree gracefully, then forcefully."""
    return _stack_processes.kill_process_tree(pid, timeout=timeout)


# =============================================================================
# Docker container management moved to scripts/server/stack_docker.py
# (2026-05-21 refactor). Re-exported via the module-level import above so
# existing call sites (cmd_start, cmd_stop, cmd_reload, cmd_status) keep
# resolving these names unqualified.
# =============================================================================


def wait_for_health(
    port: int, timeout: int = _HEALTH_SERVER_STARTUP, path: str = "/health"
) -> bool:
    """Wait for server health endpoint (delegates to stack_health.wait_for_health).

    Wrapper preserves the registry-driven _HEALTH_SERVER_STARTUP default for
    callers that don't pass timeout explicitly. (Currently every production
    caller does pass one — see the 14 call sites in this file — but the
    default is part of the public API.)
    """
    return _wait_for_health(port, timeout, path)


# =============================================================================
# Server Launching
# =============================================================================


# -----------------------------------------------------------------------------
# Mode-specific command builders (called by build_server_command dispatcher).
# Each returns a fully formed llama-server argv list for one launch shape.
# Kept private (_build_*) because the public API is build_server_command.
# -----------------------------------------------------------------------------


def _build_vision_command(port: int, vision_type: str | None, numa_instance: int = 0) -> list[str]:
    """VL launch: Qwen3-VL-30B MoE (escalation) or Qwen2.5-VL-7B (worker).

    Thread count comes from NUMA_CONFIG per (role, numa_instance) — added
    2026-05-24 along with the per-instance fix for `_build_role_command`, so
    that the newly-quartered vision roles get the correct -t per instance.
    Pre-fix: vision_escalation = hardcoded 96, worker_vision = hardcoded 24.
    """
    if vision_type == "escalation":
        role_name = "vision_escalation"
        requirements, runtime = _stack_prior_launch(role_name)
        cache = _runtime_cache(runtime)
        flags = _runtime_flags(runtime)
        # Qwen3-VL-30B MoE - larger model, expert reduction
        thread_count = _resolve_thread_count(role_name, numa_instance)
        cmd = [
            _runtime_string(runtime, "binary_path", str(LLAMA_SERVER)),
            "-m",
            _runtime_string(requirements, "model_path", VISION_ESCALATION_MODEL),
            "--mmproj",
            _runtime_string(requirements, "mmproj_path", VISION_ESCALATION_MMPROJ),
        ]
        for override in flags.get("override_kv") or ["qwen3vlmoe.expert_used_count=int:4"]:
            if isinstance(override, str) and override:
                cmd.extend(["--override-kv", override])
        cmd.extend(
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-np",
                _runtime_positive_int(cache, "slots", 1),
                "-c",
                _runtime_positive_int(cache, "context_tokens", LAUNCH_CONTEXT_TOKENS[role_name]),
                "-t",
                thread_count,
            ]
        )
        if flags.get("flash_attn", True) is True:
            cmd.extend(["--flash-attn", "on"])
        if cache.get("no_mmap", False) is True:
            cmd.append("--no-mmap")
        return cmd
    # Qwen2.5-VL-7B - smaller worker model
    role_name = "worker_vision"
    requirements, runtime = _stack_prior_launch(role_name)
    cache = _runtime_cache(runtime)
    flags = _runtime_flags(runtime)
    thread_count = _resolve_thread_count(role_name, numa_instance)
    cmd = [
        _runtime_string(runtime, "binary_path", str(LLAMA_SERVER)),
        "-m",
        _runtime_string(requirements, "model_path", VISION_WORKER_MODEL),
        "--mmproj",
        _runtime_string(requirements, "mmproj_path", VISION_WORKER_MMPROJ),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        _runtime_positive_int(cache, "slots", 2),
        "-c",
        _runtime_positive_int(cache, "context_tokens", LAUNCH_CONTEXT_TOKENS[role_name]),
        "-t",
        thread_count,
    ]
    if flags.get("flash_attn", True) is True:
        cmd.extend(["--flash-attn", "on"])
    if cache.get("no_mmap", False) is True:
        cmd.append("--no-mmap")
    return cmd


def _build_embedding_command(port: int) -> list[str]:
    """Embedding server command for production BGE ports or warm eval candidates."""
    recipe = EMBEDDING_SERVER_RECIPES.get(port)
    if recipe is None:
        recipe = {
            "model_path": EMBEDDING_MODEL_PATH,
            "context_tokens": 512,
            "threads": 4,
            "slots": 4,
            "pooling": "cls",
            "flash_attn": True,
        }

    cmd = [
        str(LLAMA_SERVER),
        "-m",
        str(recipe["model_path"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        str(recipe["slots"]),
        "-c",
        str(recipe["context_tokens"]),
        "-t",
        str(recipe["threads"]),
        "--embeddings",
        "--pooling",
        str(recipe["pooling"]),
    ]
    if recipe.get("flash_attn"):
        cmd.extend(["--flash-attn", "on"])
    return cmd


def _build_worker_fast_command(port: int, model_path: str) -> list[str]:
    """Fast worker: 1.5B model, 4 slots for parallel burst capacity."""
    return [
        str(LLAMA_SERVER),
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        "4",  # 4 parallel slots (consolidated from 2×2)
        "-c",
        str(LAUNCH_CONTEXT_TOKENS["worker_fast"]),  # 4K per slot
        "-t",
        "16",  # 16 threads for small model
        "--flash-attn",
        "on",
    ]


def _build_worker_general_command(
    port: int, model_path: str, binary_override: str | None, numa_instance: int = 0
) -> list[str]:
    """Worker-general MTP path: gemma4-26B-A4B Q4_K_M via ik_llama.cpp PR #1744.

    Swapped 2026-05-08 from Qwen3-Coder-30B-A3B Q4_K_M. Tool_compliance 96% vs 78%
    prior, +36% tps. `binary_override` comes from `server_mode.worker.runtime_requirements.binary_dir`
    in the registry — required because gemma4 MTP needs ik_llama.cpp PR #1744 build,
    not the production llama.cpp build.
    """
    requirements, runtime = _stack_prior_launch("worker_general")
    cache = _runtime_cache(runtime)
    flags = _runtime_flags(runtime)
    spec = flags.get("spec") if isinstance(flags.get("spec"), dict) else {}
    binary = (
        binary_override
        if binary_override
        else _runtime_string(runtime, "binary_path", str(LLAMA_SERVER))
    )
    model_path = _runtime_string(requirements, "model_path", model_path)
    draft_model_path = _runtime_string(
        spec,
        "draft_model_path",
        _runtime_string(requirements, "draft_model_path", EXPLORE_DRAFT_MODEL),
    )
    # 2026-05-24: now uses generic _resolve_thread_count(role, numa_instance).
    # Pre-2026-05-24 used a port-matching workaround here because the generic
    # _resolve_thread_count ignored numa_instance — that bug is now fixed at
    # the source, so the workaround is no longer needed.
    numa_thread_count = int(_resolve_thread_count("worker_general", numa_instance))
    cmd = [
        binary,
        "-m",
        model_path,
    ]
    _append_spec_decode_args(
        cmd,
        model_path=model_path,
        draft_model_path=draft_model_path,
        spec_type=_runtime_string(
            spec, "type", str(_WORKER_GENERAL_DEGRADED_FALLBACK["spec_type"])
        ),
        draft_max=_runtime_positive_int(
            spec,
            "draft_max",
            _WORKER_GENERAL_DEGRADED_FALLBACK["draft_max"],
        ),
        draft_p_min=_runtime_number_string(
            spec,
            "draft_p_min",
            _WORKER_GENERAL_DEGRADED_FALLBACK["draft_p_min"],
        ),
        threads_draft=_runtime_positive_int(
            spec,
            "threads_draft",
            _WORKER_GENERAL_DEGRADED_FALLBACK["threads_draft"],
        ),
    )
    cmd.extend(
        [
            "-ub",
            _runtime_positive_int(cache, "ubatch", _WORKER_GENERAL_DEGRADED_FALLBACK["ubatch"]),
            *(
                ["--no-mmap"] if cache.get("no_mmap", True) is True else []
            ),  # canonical recipe: bulk-read on EPYC NUMA cold-cache decode
            "--reasoning",
            str(
                flags.get("reasoning") or "off"
            ),  # disable gemma4 thinking-channel (output otherwise lands in
            # reasoning_content not content; registry: gemma4_26b reasoning=off)
            *(
                ["--jinja"] if flags.get("jinja", True) is True else []
            ),  # gemma4 ships a custom chat template embedded in the gguf;
            # without --jinja, llama.cpp rejects /v1/chat/completions
            # with "this custom template is not supported"
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            # -np 1 (single slot): MTP shares state with the target across slots in a
            # way that the ik_llama.cpp PR #1744 build asserts on with -np 2 ("tensor
            # buffer not set" at ggml-backend.cpp:236 during inference). Single slot
            # matches the working benchmark recipe. Pre-gemma4 worker_general used
            # -np 2 because external-draft spec decode (Qwen3-Coder + 0.75B draft)
            # had per-slot draft state; MTP fuses draft + target, hence -np 1.
            "-np",
            _runtime_positive_int(cache, "slots", 1),
            "-c",
            _runtime_positive_int(
                cache, "context_tokens", _WORKER_GENERAL_DEGRADED_FALLBACK["context_tokens"]
            ),
            # Per-instance thread count (full=96, quarters=48). Pre-2026-05-08 was
            # hardcoded -t 24 (Qwen3-Coder tolerated it); gemma4 + MTP under
            # ik_llama.cpp PR #1744 must match the bench recipe to avoid the
            # "tensor buffer not set" MTP assertion.
            "-t",
            str(numa_thread_count),
            # KV cache q8_0/q8_0 — registry-declared and required for stable MTP buffer
            # allocation. f16 default left some MTP tensor buffers uninitialized.
            "-ctk",
            _runtime_string(
                cache, "kv_type_k", str(_WORKER_GENERAL_DEGRADED_FALLBACK["kv_type_k"])
            ),
            "-ctv",
            _runtime_string(
                cache, "kv_type_v", str(_WORKER_GENERAL_DEGRADED_FALLBACK["kv_type_v"])
            ),
            *(["--flash-attn", "on"] if flags.get("flash_attn", True) is True else []),
        ]
    )
    return cmd


def _build_worker_explore_command(
    port: int, model_path: str, binary_override: str | None, numa_instance: int = 0
) -> list[str]:
    """Compatibility wrapper for the retired worker_explore name."""
    return _build_worker_general_command(port, model_path, binary_override, numa_instance)


def _build_dev_command(port: int) -> list[str]:
    """Dev mode: single 0.5B Qwen2.5-Coder model for fast iteration."""
    return [
        str(LLAMA_SERVER),
        "-m",
        DEV_MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        "4",
        "-c",
        "4096",
        "-t",
        "16",
        "--flash-attn",
        "on",
    ]


def _build_eval_batch_frontdoor_command(port: int, numa_instance: int = 0) -> list[str]:
    """Warm eval-batch frontdoor lane measured by P-BENCH-3/E2.

    This intentionally reuses the frontdoor model/runtime priors but overrides
    the serving shape to a single `-np 8` process on a dedicated high port. The
    role is launcher-only; normal routing still speaks in frontdoor aliases.
    """
    source_role = "frontdoor"
    requirements, runtime = _stack_prior_launch(source_role)
    cache = _runtime_cache(runtime)
    flags = _runtime_flags(runtime)
    model_path = _runtime_string(requirements, "model_path", "")
    if not model_path:
        source_config = RegistryLoader().get_role(source_role)
        if source_config is not None:
            model_path = str(source_config.model.full_path)
    binary = _runtime_string(runtime, "binary_path", str(_resolve_binary_for_role(source_role)))
    cmd = [
        binary,
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        "8",
        "-c",
        _runtime_positive_int(cache, "context_tokens", 32768),
        "-t",
        _resolve_thread_count("eval_batch_frontdoor", numa_instance),
        "-ub",
        _runtime_positive_int(cache, "ubatch", 8192),
        "-ctk",
        _runtime_string(cache, "kv_type_k", "q8_0"),
        "-ctv",
        _runtime_string(cache, "kv_type_v", "q8_0"),
        "--log-colors",
        "off",
    ]
    if flags.get("flash_attn", True) is True:
        cmd.extend(["--flash-attn", "on"])
    if flags.get("jinja", True) is True:
        cmd.append("--jinja")
    if cache.get("mlock", True) is True:
        cmd.append("--mlock")
    if cache.get("no_mmap", False) is True:
        cmd.append("--no-mmap")
    _append_runtime_spec_args(cmd, runtime, model_path)
    reasoning = flags.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        cmd.extend(["--reasoning", reasoning])
    return cmd


# -----------------------------------------------------------------------------
# Default-role builder sub-helpers (called by _build_role_command).
# -----------------------------------------------------------------------------

# Use v2 binary for roles with v3 spec decode bug (Qwen2.5 architecture).
# Currently empty (was {"coder_escalation"}); kept here for clarity of intent.
_NO_SPEC_DECODE = NO_SPEC_DECODE_ROLES

# Role-specific KV cache budgets and quantization.
# Phase 0 benchmarks (2026-03-25): generation speed neutral, memory savings significant at 65K+.
# CRITICAL (2026-03-28): V=q4_0 causes 71% prefill regression on pure-attention models.
# V=f16 has ZERO prefill regression (actually 1% faster due to K bandwidth savings).
# q4_0 K / f16 V = quality-neutral (PPL +0.017 with Hadamard), 37% KV savings, zero speed cost.
# q4_0 / q4_0 = 71% KV savings but 71% prefill regression on pure-attn. OK for hybrid (SSM amortizes).
# --kv-hadamard: production binary rebuilt with Hadamard support (commit b51c905ec, 2026-03-28).
_KV_CONTEXT_SIZES = {
    "architect_general": str(
        LAUNCH_CONTEXT_TOKENS["architect_general"]
    ),  # 122B MoE hybrid → ~16GB KV
    "ingest_long_context": str(
        LAUNCH_CONTEXT_TOKENS["ingest_long_context"]
    ),  # 80B SSM, needs long context (Stage 1 of three_stage_summarization)
}
_KV_QUANT_CONFIGS = LAUNCH_KV_QUANT_CONFIGS


def _resolve_thread_count(role_name: str, numa_instance: int = 0) -> str:
    """NUMA-aware thread count for the given role + instance index (fallback: 96).

    Pre-2026-05-24 this function always returned `instances[0][2]` regardless of
    which instance was being launched. The launcher therefore always passed
    `-t 96` to every frontdoor quarter (which intends `-t 48`), `-t 48` to
    every worker_general quarter (which intends `-t 48` correctly only because
    worker_general had a manual workaround in the worker-general MTP helper),
    and so on. Threading `numa_instance` through lets each instance get the
    thread count its NUMA_CONFIG entry actually specifies.
    """
    numa_cfg = NUMA_CONFIG.get(role_name)
    if numa_cfg and numa_cfg["instances"]:
        # Defensive: out-of-range instance falls back to the first instance's
        # thread count rather than crashing — the wrong number is better than
        # a launcher abort during stack startup.
        idx = numa_instance if 0 <= numa_instance < len(numa_cfg["instances"]) else 0
        return str(numa_cfg["instances"][idx][2])
    return "96"


def _resolve_binary_for_role(role_name: str) -> Path:
    """Pick LLAMA_SERVER_V2 if the role is on the v2 binary allow-list and the binary exists."""
    if role_name in _V2_ROLES and LLAMA_SERVER_V2.exists():
        return LLAMA_SERVER_V2
    return LLAMA_SERVER


def _append_kv_quant_args(cmd: list[str], role_name: str) -> None:
    """Emit -ctk/-ctv (and --kv-hadamard for v2 binary) for roles with a KV quant config."""
    kv_quant = _KV_QUANT_CONFIGS.get(role_name)
    if not kv_quant:
        return
    cmd.extend(["-ctk", kv_quant[0], "-ctv", kv_quant[1]])
    # --kv-hadamard: v3 auto-enables (upstream #21038), v2 needs explicit flag
    if role_name in _V2_ROLES and LLAMA_SERVER_V2.exists():
        # 2026-06-26 v6 cutover: --kv-hadamard removed in v6 (would crash); no role on v2 today
        # cmd.append("--kv-hadamard")
        pass


def _append_acceleration_args(cmd: list[str], role_name: str, accel: Any, model_path: str) -> None:
    """Emit acceleration-mode-specific args (MoE expert reduction / spec decode / self-spec).

    2026-05-09: architect_general is gated out of speculative_decoding because
    Qwen3.5-122B M-RoPE refuses position rollback when speculative draft tokens
    are rejected — see `_NO_SPEC_DECODE` and the comment block above its
    definition for the full incident trace.
    """
    if accel.type == "moe_expert_reduction" and accel.experts:
        cmd.extend(["--override-kv", f"{accel.override_key}=int:{accel.experts}"])
    elif (
        accel.type == "speculative_decoding"
        and accel.draft_role
        and role_name not in _NO_SPEC_DECODE
    ):
        registry = RegistryLoader()
        draft_config = registry.get_role(accel.draft_role)
        if draft_config:
            cmd.extend(
                [
                    "-md",
                    draft_config.model.full_path,
                    "--spec-draft-n-max",  # 2026-06-26 v6 cutover: renamed from --draft-max
                    str(accel.k or 16),
                ]
            )

    # MoE + spec decode combo (e.g., 480B with jukofyork draft + expert reduction)
    if (
        accel.type == "moe_expert_reduction"
        and accel.draft_role
        and role_name not in _NO_SPEC_DECODE
    ):
        registry = RegistryLoader()
        draft_config = registry.get_role(accel.draft_role)
        if draft_config:
            cmd.extend(
                [
                    "-md",
                    draft_config.model.full_path,
                    "--spec-draft-n-max",  # 2026-06-26 v6 cutover: renamed from --draft-max
                    str(accel.k or 16),
                ]
            )

    # Self-speculation: same model as target and draft, draft exits early
    elif accel.type == "self_speculation" and accel.n_layer_exit_draft:
        cmd.extend(
            [
                "--n-layer-exit-draft",
                str(accel.n_layer_exit_draft),
                "--spec-draft-n-max",  # 2026-06-26 v6 cutover: renamed from --draft-max
                str(accel.k or 16),
            ]
        )

    # Hierarchical speculation: self-spec with intermediate verification
    elif accel.type == "hierarchical_speculation":
        cmd.extend(
            [
                "--n-layer-exit-draft",
                str(accel.n_layer_exit_draft or 0),
                "--hierarchical-spec",
                "--spec-draft-n-max",  # 2026-06-26 v6 cutover: renamed from --draft-max
                str(accel.k or 16),
            ]
        )
        if accel.n_layer_exit_intermediate:
            cmd.extend(["--n-layer-exit-intermediate", str(accel.n_layer_exit_intermediate)])


def _apply_numa_spec_overrides(cmd: list[str], numa_cfg: dict | None) -> None:
    """In-place rewrite of --draft-max based on NUMA spec_overrides.

    When NUMA thread count differs from 192t, the optimal draft_max may differ.
    Overrides come from bench_sweep_spec_params.sh results stored in NUMA_CONFIG.
    """
    if not numa_cfg or "spec_overrides" not in numa_cfg:
        return
    overrides = numa_cfg["spec_overrides"]
    if "draft_max" in overrides:
        for i, arg in enumerate(cmd):
            # 2026-06-26 v6 cutover: match renamed --spec-draft-n-max (was --draft-max)
            if arg == "--spec-draft-n-max" and i + 1 < len(cmd):
                cmd[i + 1] = str(overrides["draft_max"])
                break
    # Tree-spec (--draft-p-split) and bare --lookup flag both stripped in v5 binary;
    # see git history for re-enable conditions if/when a future binary restores them.


def _build_role_command(role_config: Any, port: int, numa_instance: int = 0) -> list[str]:
    """Build llama-server command for a registry-backed role (default path).

    `numa_instance` selects which entry in NUMA_CONFIG[role]["instances"] this
    invocation refers to. 0 = the primary/full instance; 1..N = quarter
    instances. The thread count comes from that specific instance's tuple so
    quarters get `-t 48` and the full instance gets its declared `-t 96`.
    """
    model_path = role_config.model.full_path
    accel = role_config.acceleration
    role_name = role_config.name
    _requirements, runtime = _stack_prior_launch(role_name)
    cache = _runtime_cache(runtime)
    flags = _runtime_flags(runtime)
    fallback_slots = 1 if role_name in SERIAL_ROLES else 2
    parallel_slots = _runtime_positive_int(cache, "slots", fallback_slots)
    thread_count = _resolve_thread_count(role_name, numa_instance)
    context_size = _runtime_positive_int(
        cache,
        "context_tokens",
        _KV_CONTEXT_SIZES.get(role_name, str(DEFAULT_EFFECTIVE_CONTEXT_TOKENS)),
    )
    binary = _runtime_string(runtime, "binary_path", str(_resolve_binary_for_role(role_name)))
    ubatch = _runtime_positive_int(cache, "ubatch", DEFAULT_UBATCH_TOKENS)

    # -ub 8192: matches the canonical-bench single-instance recipe
    # (scripts/benchmark/run_qwen36_retest.py uses `-ub 8192` + `-c 8192` + `--parallel 1`).
    # Without this, frontdoor's decode throughput was 12.66 t/s vs 25-27 t/s in the bench CSV.
    cmd = [
        binary,
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        parallel_slots,
        "-c",
        context_size,
        "-t",
        thread_count,
    ]
    cmd.extend(["-ub", ubatch])
    if flags.get("flash_attn", True) is True:
        cmd.extend(["--flash-attn", "on"])

    # --jinja: model's native chat template (enables thinking on Qwen3/3.5).
    # SKIP for architect_general — Qwen3.5 hybrids enter infinite <think> loops.
    # --reasoning off is insufficient: the jinja template itself primes the model
    # into think mode. Without --jinja, llama-server falls back to generic ChatML
    # which has no thinking scaffolding.
    if flags.get("jinja", role_name != "architect_general") is True:
        cmd.append("--jinja")

    if runtime:
        _append_runtime_kv_args(cmd, cache)
    else:
        _append_kv_quant_args(cmd, role_name)

    # mlock: lock model weights in RAM to prevent page cache eviction.
    # Validated in S2: 30x latency improvement under memory pressure.
    if cache.get("mlock", role_name in MLOCK_ROLES) is True:
        cmd.append("--mlock")

    # 2026-06-26 v6 cutover: honor the per-role no_mmap prior (enables N12 topology).
    # Mirrors _build_worker_general_command's cache.get("no_mmap", ...) emit. Default
    # False here so generic-role behavior is unchanged when no_mmap is absent/False.
    if cache.get("no_mmap", False) is True:
        cmd.append("--no-mmap")

    if runtime:
        for override in flags.get("override_kv") or []:
            if isinstance(override, str) and override:
                cmd.extend(["--override-kv", override])
        _append_runtime_spec_args(cmd, runtime, model_path)
        reasoning = flags.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            cmd.extend(["--reasoning", reasoning])
    else:
        _append_acceleration_args(cmd, role_name, accel, model_path)
    _apply_numa_spec_overrides(cmd, NUMA_CONFIG.get(role_name))

    # DS-3: KV state save/restore — per-role subdir to avoid slot ID collisions.
    slot_save_path = cache.get("slot_save_path")
    slot_dir = (
        Path(slot_save_path)
        if isinstance(slot_save_path, str) and slot_save_path
        else SLOT_SAVE_DIR / role_name
    )
    slot_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--slot-save-path", str(slot_dir)])

    return cmd


def build_server_command(
    role_config: Any,
    port: int,
    dev_mode: bool = False,
    embedding_mode: bool = False,
    worker_pool_mode: bool = False,
    worker_type: str = None,
    vision_mode: bool = False,
    vision_type: str = None,
    eval_batch_frontdoor_mode: bool = False,
    binary_override: str | None = None,
    numa_instance: int = 0,
) -> list[str]:
    """Dispatch to the per-mode command builder.

    `binary_override`: when set, replaces `LLAMA_SERVER` for the worker_pool
    general branch (used by worker_general / gemma4 MTP to launch ik_llama.cpp
    PR #1744). Other branches ignore this argument today.

    `numa_instance`: which instance in NUMA_CONFIG[role]["instances"] is being
    launched (0 = full/primary, 1..N = quarters). Used by `_build_role_command`
    to pick per-instance thread count. Defaults to 0 so callers that don't
    care about quarters (vision, embedding, dev, worker_pool) are unaffected.
    """
    if vision_mode:
        cmd = _build_vision_command(port, vision_type, numa_instance)
    elif embedding_mode:
        cmd = _build_embedding_command(port)
    elif eval_batch_frontdoor_mode:
        cmd = _build_eval_batch_frontdoor_command(port, numa_instance)
    elif worker_pool_mode and worker_type:
        model_path = WORKER_POOL_MODELS.get(worker_type)
        if not model_path:
            raise ValueError(f"Unknown worker type: {worker_type}")
        if worker_type == "fast":
            cmd = _build_worker_fast_command(port, model_path)
        else:
            cmd = _build_worker_general_command(port, model_path, binary_override, numa_instance)
    elif dev_mode:
        cmd = _build_dev_command(port)
    else:
        cmd = _build_role_command(role_config, port, numa_instance)
    _append_cpu_only_device_args(cmd)
    return cmd


def start_server(
    port: int,
    roles: list[str],
    registry: RegistryLoader,
    dev_mode: bool = False,
    embedding_mode: bool = False,
    worker_pool_mode: bool = False,
    worker_type: str = None,
    vision_mode: bool = False,
    vision_type: str = None,
    eval_batch_frontdoor_mode: bool = False,
    numa_instance: int = 0,
) -> ProcessInfo | None:
    """Start a llama-server for the given roles."""
    detached_stdio = {
        "stdin": subprocess.DEVNULL,
        "start_new_session": True,
        "close_fds": True,
    }

    # P-BENCH-3/A7 warm eval-batch lane: dedicated frontdoor-model server
    # used only when explicitly started and when EVAL_BATCH_SERVING routes
    # eval-batch traffic to it.
    if eval_batch_frontdoor_mode:
        primary_role = roles[0]
        source_role = "frontdoor"
        log_file = LOG_DIR / f"eval-batch-frontdoor-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        cmd = build_server_command(
            None,
            port,
            eval_batch_frontdoor_mode=True,
            numa_instance=numa_instance,
        )
        requirements, _runtime = _stack_prior_launch(source_role)
        model_path = _runtime_string(requirements, "model_path", "")
        model_name = Path(model_path).name if model_path else "frontdoor model"

        print(f"  Starting eval-batch frontdoor on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        try:
            _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
        except Exception as exc:
            print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")

        with open(log_file, "w") as log:
            env = build_launch_env(source_role, os.environ.copy())
            proc = subprocess.Popen(
                _numa_prefix(primary_role, numa_instance) + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                **detached_stdio,
            )

        print(f"    PID: {proc.pid}")
        print("    Waiting for health...")

        if wait_for_health(port, timeout=max(_HEALTH_SERVER_STARTUP, 180)):
            print("    [OK] Eval-batch frontdoor ready")
            return ProcessInfo(
                role=primary_role,
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )

        print("    [FAIL] Eval-batch frontdoor did not become healthy")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None

    # Vision mode - VL models with multimodal projector
    if vision_mode:
        log_file = LOG_DIR / f"vision-{vision_type or 'worker'}-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        if vision_type == "escalation":
            model_path = VISION_ESCALATION_MODEL
            model_name = "Qwen3-VL-30B-A3B (vision escalation)"
        else:
            model_path = VISION_WORKER_MODEL
            model_name = "Qwen2.5-VL-7B (vision worker)"

        cmd = build_server_command(
            None,
            port,
            vision_mode=True,
            vision_type=vision_type,
            numa_instance=numa_instance,  # fix: forward so quarters get NUMA_CONFIG -t (was always -t 96)
        )

        print(f"  Starting vision server [{vision_type or 'worker'}] on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        # Fleet marker: written BEFORE Popen so subsequent watcher polls
        # see the new startup timestamp immediately + can resolve role→port.
        try:
            _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
        except Exception as exc:
            print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")

        with open(log_file, "w") as log:
            env = build_launch_env(roles[0], os.environ.copy())
            proc = subprocess.Popen(
                _numa_prefix(roles[0], numa_instance) + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                **detached_stdio,
            )

        print(f"    PID: {proc.pid}")
        print("    Waiting for health...")

        # VL models take longer to load (mmproj + main model)
        timeout = _HEALTH_VISION_SERVER if vision_type == "escalation" else _HEALTH_WORKER_SERVER
        if wait_for_health(port, timeout=timeout):
            print(f"    [OK] Vision server {vision_type or 'worker'} ready")
            return ProcessInfo(
                role=roles[0],
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Vision server {vision_type or 'worker'} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Embedding mode uses dedicated config, no registry lookup needed
    if embedding_mode:
        log_file = LOG_DIR / f"embedder-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        cmd = build_server_command(None, port, dev_mode=False, embedding_mode=True)
        recipe = EMBEDDING_SERVER_RECIPES.get(port, {})
        model_name = str(recipe.get("model_name", "embedding model"))
        instance_idx = port - 8090 if port in EMBEDDER_PORTS else port

        print(f"  Starting embedder #{instance_idx} on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        # Fleet marker: written BEFORE Popen so the watcher can resolve
        # role→port and detect operator-initiated reloads.
        try:
            _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
        except Exception as exc:
            print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")

        with open(log_file, "w") as log:
            env = build_launch_env(roles[0], os.environ.copy())
            # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
            proc = subprocess.Popen(
                _numa_prefix(roles[0], numa_instance) + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                **detached_stdio,
            )

        print(f"    PID: {proc.pid}")
        print("    Waiting for health...")

        if wait_for_health(port, timeout=60):  # Faster timeout for small model
            print(f"    [OK] Embedder #{instance_idx} ready")
            return ProcessInfo(
                role=roles[0],  # Use actual role name (embedder, embedder_1, etc.)
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=str(recipe.get("model_path", EMBEDDING_MODEL_PATH)),
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Embedder #{instance_idx} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Worker pool mode - heterogeneous workers
    if worker_pool_mode and worker_type:
        log_file = LOG_DIR / f"worker-{worker_type}-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        model_path = WORKER_POOL_MODELS.get(worker_type)
        if not model_path:
            print(f"  [!] Unknown worker type: {worker_type}")
            return None

        # Per-role binary + LD_LIBRARY_PATH override (Phase 2). worker_general (gemma4
        # MTP) needs ik_llama.cpp PR #1744 binary; other workers fall back to default.
        # Lookup keyed on the primary role (e.g. "worker_general"), not worker_type.
        binary_dir, ld_paths = _runtime_requirements_for_role(registry, roles[0])
        binary_override = str(Path(binary_dir) / "llama-server") if binary_dir else None

        cmd = build_server_command(
            None,
            port,
            worker_pool_mode=True,
            worker_type=worker_type,
            binary_override=binary_override,
            numa_instance=numa_instance,  # fix: forward so gemma4 quarters get NUMA_CONFIG -t 48 (was always -t 96)
        )
        model_name = Path(model_path).stem

        print(f"  Starting worker pool [{worker_type}] on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        if binary_override:
            print(f"    Binary override: {binary_override}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        with open(log_file, "w") as log:
            # Worker pool roles map their worker_type to the canonical "worker" role for env.
            env = build_launch_env("worker", os.environ.copy())
            # When a per-role binary override is in effect (gemma4 MTP via ik_llama.cpp
            # PR #1744), strip the production-llama.cpp-tuned GGML_* env block. Those
            # flags (GGML_CCD_POOLS / GGML_CCD_WORK_DIST / GGML_BARRIER_LOCAL_BETWEEN_OPS)
            # were validated for Qwen3-Coder-30B on the production ggml fork; the
            # ik_llama.cpp gemma-mtp branch is forked at a different ggml commit and
            # leaves MTP draft tensors with no buffer assignment when these flags are
            # set, triggering "tensor buffer not set" assertion at ggml-backend.cpp:236.
            # Bench launches confirm: gemma4 MTP works with bare OMP env, no GGML_*.
            if binary_override:
                # 2026-06-26 v6 cutover: never strip GGML_IQK (gates iqk kernels)
                stripped = [
                    k for k in list(env.keys()) if k.startswith("GGML_") and k != "GGML_IQK"
                ]
                for k in stripped:
                    del env[k]
                if stripped:
                    print(f"    [binary_override] stripped GGML_* env: {stripped}")
                # 2026-05-09: KMP_BLOCKTIME=10 ms — fixes the libomp idle busy-spin.
                #
                # Background: PR #1744 uses bare `#pragma omp parallel` per
                # ggml_graph_compute() call (ggml/src/ggml.c:26739), no persistent
                # threadpool. Between dispatches, AOCC libomp's worker team stays
                # alive in a busy-wait state under OMP_WAIT_POLICY=active (95+ cores
                # spinning idle, polluting L3 / DRAM bandwidth shared with the other
                # roles — measured -40 to -69% throughput hit on frontdoor / coder /
                # ingest while gemma4 was idle).
                #
                # Tried in source: omp_pause_resource(soft) + omp_pause_resource_all(hard)
                # — verified BOTH ignored by AOCC 5.0.0 libomp (threads stayed in R
                # state, wchan=0). The OMP runtime's idle behavior isn't controllable
                # via the standard pause API on this libomp build.
                #
                # KMP_BLOCKTIME is the LLVM libomp tunable (AOCC's libomp is LLVM-
                # based). Workers busy-wait this many ms before transitioning to a
                # futex sleep. 10 ms = fast enough that MTP request dispatch finds
                # workers warm (no perceptible first-token-latency regression), short
                # enough that the multi-second gaps between requests don't waste
                # cycles. OMP_WAIT_POLICY=active stays — it controls the steady-state
                # behavior; KMP_BLOCKTIME tunes the idle transition. Full passive
                # (= KMP_BLOCKTIME=0) breaks MTP wakeup; active alone busy-spins
                # forever; active + KMP_BLOCKTIME=10 is the sweet spot.
                env["KMP_BLOCKTIME"] = "10"
            # Prepend role-specific LD_LIBRARY_PATH entries (Phase 2): ik_llama.cpp
            # PR #1744 build needs its own libllama.so / libggml.so on the resolver
            # path. Prepend so the override beats system libs without touching the
            # canonical-recipe LLVM-20 libomp path that already lives in env.
            if ld_paths:
                existing = env.get("LD_LIBRARY_PATH", "")
                merged = ":".join(ld_paths) + (f":{existing}" if existing else "")
                env["LD_LIBRARY_PATH"] = merged
                print(f"    LD_LIBRARY_PATH += {ld_paths}")
            # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
            # Fleet marker: written BEFORE Popen so the watcher can resolve
            # role→port and detect operator-initiated reloads.
            try:
                _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
            except Exception as exc:
                print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")
            proc = subprocess.Popen(
                _numa_prefix(roles[0], numa_instance) + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                **detached_stdio,
            )

        print(f"    PID: {proc.pid}")
        print("    Waiting for health...")

        # Faster timeout for smaller models (quick_check for fast workers)
        timeout = (
            int(_registry_timeout("health", "quick_check", 10)) * 6
            if worker_type == "fast"
            else _HEALTH_WORKER_SERVER
        )
        if wait_for_health(port, timeout=timeout):
            print(f"    [OK] Worker {worker_type} ready")
            # 2026-05-09: per-thread renice to nice=19 for binary_override roles
            # (ik_llama.cpp PR #1744 / gemma4 MTP). Reason: PR #1744's OMP_WAIT_POLICY=active
            # busy-loops 96 cores when idle, contaminating other-role measurements by
            # 30-69% throughput. CLI `renice -p PID` only affects the lead thread, so we
            # iterate /proc/<pid>/task/<tid> after model load completes (all OMP team
            # threads are spawned by health-check time).
            # Verified 2026-05-09: post-renice, frontdoor 4.55→7.11 t/s (+56%),
            # coder 4.02→12.34 (+207%), ingest 10.46→28.99 (+177%).
            # No sudo needed — increasing nice (lower priority) is permitted for owner.
            if binary_override:
                _renice_all_threads(proc.pid, 19)
            return ProcessInfo(
                role=f"worker_{worker_type}",
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Worker {worker_type} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Use first role's config for the server
    primary_role = roles[0]
    role_config = registry.get_role(primary_role)

    if not role_config and not dev_mode:
        print(f"  [!] Role {primary_role} not found in registry")
        return None

    log_file = LOG_DIR / f"llama-server-{port}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build command. `numa_instance` flows through so per-quarter thread counts
    # come from the role's NUMA_CONFIG entry rather than always defaulting to
    # the first instance's count.
    cmd = build_server_command(role_config, port, dev_mode, numa_instance=numa_instance)

    model_name = DEV_MODEL if dev_mode else role_config.model.name
    numa_cfg = NUMA_CONFIG.get(primary_role)
    numa_label = ""
    if numa_cfg and numa_instance < len(numa_cfg["instances"]):
        cpu_list = numa_cfg["instances"][numa_instance][0]
        numa_label = f" [NUMA {numa_instance}: cpus {cpu_list}]"

    print(f"  Starting port {port}: {model_name}{numa_label}")
    print(f"    Roles: {', '.join(roles)}")
    print(f"    Command: {' '.join(cmd[:5])}...")

    # Start process — taskset CPU-pinned per NUMA config + canonical OMP env + per-role GGML
    with open(log_file, "w") as log:
        env = build_launch_env(primary_role, os.environ.copy())
        # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
        # Fleet marker: written BEFORE Popen so the watcher can resolve
        # role→port and detect operator-initiated reloads.
        try:
            _write_llama_marker(port, roles, source=_FLEET_SRC_STACK, tmp_dir=_PATHS["tmp_dir"])
        except Exception as exc:
            print(f"    [WARN] Failed to write llama fleet marker for port {port}: {exc}")
        proc = subprocess.Popen(
            _numa_prefix(primary_role, numa_instance) + cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            **detached_stdio,
        )

    print(f"    PID: {proc.pid}")

    # Wait for health
    print("    Waiting for health...")
    if wait_for_health(port, timeout=180):
        print("    [OK] Server ready")
        return ProcessInfo(
            role=primary_role,
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path=DEV_MODEL_PATH if dev_mode else role_config.model.full_path,
            log_file=str(log_file),
        )
    else:
        print("    [FAIL] Server did not become healthy")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def _apply_orchestrator_profile(env: dict[str, str], profile: str | None) -> None:
    """Apply optional orchestrator profile env vars without overriding explicit env."""
    if not profile:
        return
    profile_vars = ORCHESTRATOR_PROFILES.get(profile)
    if not profile_vars:
        print(f"    [WARN] Unknown orchestrator profile '{profile}' (ignored)")
        return
    print(f"    Using orchestrator profile: {profile}")
    for key, value in profile_vars.items():
        env.setdefault(key, value)


PRODUCTION_FEATURE_WAVE_OVERRIDES: dict[str, bool] = {
    # Fable5 routing-truth W1: conservative wave-1 production intent.
    "specialist_routing": True,
    "model_fallback": True,
    # Wave-2 paths have been dormant for months; keep them declared-off until
    # each has its own observation window.
    "plan_review": False,
    "architect_delegation": False,
    "parallel_execution": False,
    "unified_streaming": False,
    # Weights have been absent since the 2026-05-25 memory reset.
    "routing_classifier": False,
}

LANGGRAPH_PHASE3_LIVE_ENV_VARS: tuple[str, ...] = (
    "ORCHESTRATOR_LANGGRAPH_INGEST",
    "ORCHESTRATOR_LANGGRAPH_WORKER",
    "ORCHESTRATOR_LANGGRAPH_FRONTDOOR",
    "ORCHESTRATOR_LANGGRAPH_CODER",
    "ORCHESTRATOR_LANGGRAPH_CODER_ESCALATION",
    "ORCHESTRATOR_LANGGRAPH_ARCHITECT",
)


def _production_feature_env() -> dict[str, str]:
    """Complete, attestable ORCHESTRATOR_FEATURE_* block for API workers."""
    from src.features import FEATURE_ENV_PREFIX, _FEATURE_REGISTRY, _REGISTRY_BY_NAME

    block = {
        f"{FEATURE_ENV_PREFIX}{spec.env_var}": "1" if spec.default_prod else "0"
        for spec in _FEATURE_REGISTRY
    }
    for name, enabled in PRODUCTION_FEATURE_WAVE_OVERRIDES.items():
        spec = _REGISTRY_BY_NAME[name]
        block[f"{FEATURE_ENV_PREFIX}{spec.env_var}"] = "1" if enabled else "0"
    return block


def _apply_production_feature_env(env: MutableMapping[str, str]) -> None:
    """Fill stack-managed production feature defaults without clobbering overrides."""
    for key, value in _production_feature_env().items():
        env.setdefault(key, value)


def start_orchestrator(profile: str | None = None) -> ProcessInfo | None:
    """Start the orchestrator API."""
    log_file = LOG_DIR / "orchestrator.log"

    print("  Starting orchestrator API on port 8000")
    stale_pids = _pids_on_port(8000)
    if stale_pids:
        print(f"    Clearing stale listeners on :8000 ({', '.join(str(p) for p in stale_pids)})")
        for stale_pid in stale_pids:
            kill_process(stale_pid)
        time.sleep(1)

    # Set environment — enable production feature flags
    env = os.environ.copy()
    env["HF_HOME"] = str(_PATHS["cache_dir"] / "huggingface")
    env["TMPDIR"] = str(_PATHS["tmp_dir"])
    # Feature flags: make every registry flag explicit in /proc/<pid>/environ.
    # Explicit launch-time env values are activation intent and must survive.
    _apply_production_feature_env(env)
    # 2026-05-22 Phase 5: per-CPU-region cross-process locks enabled by
    # default. Replaces the single global heavy_model.lock with
    # per-(role, atomic-region) fcntl locks so frontdoor full (0-47)
    # can run concurrently with frontdoor.q2 / q3 (48-71 / 72-95) on
    # disjoint cores. ConcurrencyAwareBackend's dispatch path uses these
    # for cross-process safe instance selection. Override with
    # ORCHESTRATOR_PER_REGION_LOCKS=0 to fall back to the legacy global
    # heavy lock if a regression surfaces.
    env.setdefault("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    # 2026-05-31: default-on cross-role physical exclusion. Adds a
    # role-agnostic cpu_region.GLOBAL.{qN}.lock layer so different roles cannot
    # decode on the same atomic CPU region.
    #
    # 2026-07-06: enable shape-aware contention by default now that
    # ConcurrencyAwareBackend dispatch threads the actual candidate_topology_idx
    # through the admission gate. Physical disjointness alone is insufficient:
    # the measured contention matrix still blocks some role pairs whose CPU
    # regions do not literally overlap.
    env.setdefault("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    env.setdefault("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", "1")
    # WP-7/J6 (2026-05-26): within-role placement rollout. These were previously
    # shell-env-only, so ANY API restart (autopilot config-apply, watcher relaunch,
    # manual) silently reverted the placement state machine to OFF — J6 ran without it
    # after a 19:11 restart until this was caught. Make them durable defaults (still
    # overridable via the env). PLACEMENT_STATE_MACHINE = WP-2 topology-safe placement;
    # REVERSE_MIGRATION = WP-4 quarter→full on load drop; URE shadow = J10 (no behavior change).
    env.setdefault("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "1")
    env.setdefault("ORCHESTRATOR_REVERSE_MIGRATION", "1")
    env.setdefault("ORCHESTRATOR_URE_UNCERTAINTY_SHADOW_LOG", "1")
    # P6.2-A2 (2026-05-21): frontdoor-specialist verifier loaded by the API
    # service when the gate flag is on. Defaults below put it in SHADOW MODE —
    # the verifier runs and logs P(success) to last_decision_meta but never
    # blocks a fast-path route. To enforce: launch with
    #   FRONTDOOR_VERIFIER_SHADOW=0 ./scripts/server/orchestrator_stack.py start
    # Both can be disabled by setting ORCHESTRATOR_FRONTDOOR_VERIFIER_GATE=0.
    # See handoffs/active/learned-routing-controller.md Phase 6 rollout.
    env.setdefault("ORCHESTRATOR_FRONTDOOR_VERIFIER_GATE", "1")
    env.setdefault("FRONTDOOR_VERIFIER_SHADOW", "1")
    env.setdefault("FRONTDOOR_VERIFIER_THRESHOLD", "0.5")
    env["ORCHESTRATOR_TOOLS"] = "1"
    env["ORCHESTRATOR_SCRIPTS"] = "1"
    # NOTE: Do NOT set ORCHESTRATOR_REPL here — it collides with
    # OrchestratorSettings.repl (REPLSettings model) in config.py.
    # The repl feature flag defaults to True in features.py already.
    env["ORCHESTRATOR_CACHING"] = "1"
    env["ORCHESTRATOR_STREAMING"] = "1"
    env["ORCHESTRATOR_MOCK_MODE"] = "0"
    env["ORCHESTRATOR_GENERATION_MONITOR"] = "1"
    env["ORCHESTRATOR_REACT_MODE"] = "1"
    env["ORCHESTRATOR_CASCADING_TOOL_POLICY"] = "1"
    env["ORCHESTRATOR_WORKER_CALL_BUDGET"] = "1"
    env["ORCHESTRATOR_TASK_TOKEN_BUDGET"] = "1"
    env.setdefault("ORCHESTRATOR_WORKER_CALL_BUDGET_CAP", "30")
    env.setdefault("ORCHESTRATOR_TASK_TOKEN_BUDGET_CAP", "200000")
    env["ORCHESTRATOR_SESSION_SCRATCHPAD"] = "1"
    env["ORCHESTRATOR_SESSION_LOG"] = "1"
    env["ORCHESTRATOR_APPROVAL_GATES"] = "1"
    env["ORCHESTRATOR_RESUME_TOKENS"] = "1"
    env["ORCHESTRATOR_SIDE_EFFECT_TRACKING"] = "1"
    env["ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT"] = "1"
    # Gate-3 tool telemetry must survive API reloads too; the authority daemon
    # already carries AUTOPILOT_TOOL_SENTINELS=1, but the orchestrator API was
    # previously restarted without it and quietly lost tool-use activation.
    env["AUTOPILOT_TOOL_SENTINELS"] = "1"
    # LangGraph Phase 3: per-node migration for live roles only.
    # The retired architect_coding role is intentionally not enabled here.
    for key in LANGGRAPH_PHASE3_LIVE_ENV_VARS:
        env[key] = "1"
    _apply_orchestrator_profile(env, profile)
    # Bound inference-lock waits by default to avoid multi-minute silent stalls
    # during iterative debugging / seeding runs. Bumped from 45s → 180s 2026-05-21
    # because GEPA-driven autopilot workloads were aborting worker_explore /
    # ingest requests after 45s while frontdoor held the exclusive lock for
    # 60-105s during reasoning-heavy spec-decode bursts. 180s rides out typical
    # spec-decode + multi-token-generation while still surfacing true deadlocks
    # within ~3 min. Tune via env override if your workload changes.
    env.setdefault("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_EXCLUSIVE_S", "180")
    env.setdefault("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_SHARED_S", "180")

    # Write the fleet-startup marker BEFORE Popen. uvicorn forks workers
    # AFTER its own startup (each worker imports the app independently),
    # and dashboard.py reads this marker at module-import time to serve
    # the same `server_started_at` across all six workers via
    # /dashboard/api/version. Without this marker, each worker would
    # carry its own `time.time()` and consumers (OrchestratorWatcher)
    # would see spurious restart signals depending on which worker the
    # load balancer routed each request to.
    try:
        marker_path = _write_orchestrator_marker(
            tmp_dir=_PATHS["tmp_dir"],
            git_sha=_repo_short_sha(),
        )
        print(f"    Fleet marker: {marker_path}")
    except Exception as exc:
        print(f"    [WARN] Failed to write orchestrator fleet marker: {exc}")

    with open(log_file, "w") as log:
        workers = int(env.get("ORCHESTRATOR_UVICORN_WORKERS", "6"))
        # 2026-06-13: bumped default from 16 to 64 (env-overridable). The
        # lower cap still allowed dashboard/SSE traffic plus AutoPilot reward
        # writes to saturate uvicorn before model-side locks could serialize
        # real inference, yielding HTTP 503 on /chat/reward during seeding.
        # This is an accept-queue headroom limit, not a model concurrency
        # limit; inference-lock/contention gates still provide serialization.
        concurrency = int(env.get("ORCHESTRATOR_UVICORN_LIMIT_CONCURRENCY", "64"))
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--workers",
                str(workers),
                "--limit-concurrency",
                str(concurrency),
            ],
            cwd=str(_PATHS["project_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print("    Waiting for health...")

    if wait_for_health(8000, timeout=60):
        print("    [OK] Orchestrator ready")
        # Durable earlyoom control-plane protection. The API master + its uvicorn
        # workers are comm=python and cannot be earlyoom --ignore'd by name (they
        # collide with runaway python evals), so set oom_score_adj=-1000 (earlyoom
        # skips exactly -1000 in both oom_score and --sort-by-rss modes). Workers
        # exist by health-check time. Best-effort; replaces the manual one-shot
        # `choom` that did not survive an API restart.
        _set_oom_protection([proc.pid, *_child_pids(proc.pid)])
        return ProcessInfo(
            role="orchestrator",
            pid=proc.pid,
            port=8000,
            started_at=datetime.now().isoformat(),
            model_path="uvicorn",
            log_file=str(log_file),
        )
    # Health probe can fail transiently (port permissions / local sandbox),
    # while the process is actually alive. Avoid killing a healthy API due to
    # a false-negative probe; only hard-fail when process already exited.
    if proc.poll() is None:
        print("    [WARN] Health probe timed out, but API process is still running")
        print(f"    Check log: {log_file}")
        _set_oom_protection([proc.pid, *_child_pids(proc.pid)])
        return ProcessInfo(
            role="orchestrator",
            pid=proc.pid,
            port=8000,
            started_at=datetime.now().isoformat(),
            model_path="uvicorn",
            log_file=str(log_file),
        )

    print("    [FAIL] Orchestrator did not start")
    print(f"    Check log: {log_file}")
    kill_process(proc.pid)
    return None


def start_document_formalizer() -> ProcessInfo | None:
    """Start the document formalizer (LightOnOCR-2) server."""
    log_file = LOG_DIR / "document_formalizer.log"
    port = 9001

    print(f"  Starting document_formalizer (LightOnOCR-2) on port {port}")

    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PATHS["project_root"]) + os.pathsep + env.get("PYTHONPATH", "")
    env["LIGHTONOCR_WORKERS"] = "8"
    env["LIGHTONOCR_THREADS"] = "12"
    env["LIGHTONOCR_MAX_TOKENS"] = "2048"
    env["LIGHTONOCR_TIMEOUT"] = "120"

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(_PATHS["project_root"] / "src/services/lightonocr_llama_server.py"),
                "--port",
                str(port),
            ],
            cwd=str(_PATHS["project_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print("    Waiting for health...")

    if wait_for_health(port, timeout=60):
        print("    [OK] Document formalizer ready")
        return ProcessInfo(
            role="document_formalizer",
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path="LightOnOCR-2-1B-bbox",
            log_file=str(log_file),
        )
    else:
        print("    [FAIL] Document formalizer did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def start_sd_server() -> ProcessInfo | None:
    """Start the sd-server diffusion inference service (stable-diffusion.cpp native).

    Replaces the ComfyUI-GGUF + PyTorch path 2026-05-07 — sd.cpp's native
    ggml backend keeps Q8_0 weights packed and uses native quantized GEMM
    kernels, skipping ComfyUI-GGUF's per-layer dequant-to-BF16 step.
    Measured ~1.74× wall-clock and ~3.43× sampler s/iter speedup at 512² /
    4 steps; expected ~2× wall-clock at production 1024² / 8 steps.
    Stack-managed per feedback_stack_managed_services. Health probe uses
    /sdapi/v1/samplers (sd-server has no dedicated /health endpoint).
    """
    log_file = LOG_DIR / "sd_server.log"
    port = 8190
    launcher = _PATHS["project_root"] / "scripts/diffusion/start_sd_server.sh"

    print(f"  Starting sd_server (ERNIE-Image-Turbo, ggml native) on port {port}")

    if not launcher.exists():
        print(f"    [FAIL] Launcher not found: {launcher}")
        return None

    env = os.environ.copy()
    env["SD_SERVER_PORT"] = str(port)

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            ["bash", str(launcher)],
            cwd=str(_PATHS["project_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print("    Waiting for health (path=/sdapi/v1/samplers, timeout=120s)...")

    if wait_for_health(port, timeout=120, path="/sdapi/v1/samplers"):
        print("    [OK] sd-server ready")
        return ProcessInfo(
            role="sd_server",
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path="ernie-image-turbo-Q8_0.gguf + ministral-3-3b + flux2-vae (sd.cpp ggml native)",
            log_file=str(log_file),
        )
    else:
        print("    [FAIL] sd-server did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def start_whisper() -> ProcessInfo | None:
    """Start the faster-whisper STT server (large-v3-turbo, int8).

    Promoted from sidecar to stack-managed 2026-05-06 per
    feedback_stack_managed_services. Reuses the existing launch script in
    epyc-inference-research; no rewrite needed.
    """
    log_file = LOG_DIR / "whisper.log"
    port = 9000
    # Whisper launcher lives in the inference-research repo (was a sidecar)
    launcher = Path("/mnt/raid0/llm/epyc-inference-research/scripts/voice/start_whisper_server.sh")

    print(f"  Starting whisper (faster-whisper large-v3-turbo) on port {port}")

    if not launcher.exists():
        print(f"    [FAIL] Launcher not found: {launcher}")
        return None

    env = os.environ.copy()
    env["WHISPER_PORT"] = str(port)

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            ["bash", str(launcher)],
            cwd=str(launcher.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print("    Waiting for health (path=/health, timeout=60s)...")

    if wait_for_health(port, timeout=60, path="/health"):
        print("    [OK] Whisper ready")
        return ProcessInfo(
            role="whisper",
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path="faster-whisper-large-v3-turbo (int8)",
            log_file=str(log_file),
        )
    else:
        print("    [FAIL] Whisper did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def start_handoff_dashboard() -> ProcessInfo | None:
    """Start the epyc-root handoff progress dashboard hub (port 8100).

    Project-wide, file/artifact-backed progress board owned by the governance
    repo (epyc-root). It is deliberately dependency-free (Python stdlib only),
    so it runs under any interpreter — the orchestrator venv is not required.
    The autopilot dashboard stays on the orchestrator (:8000/dashboard) because
    it needs live in-process state; this hub links to it and vice-versa.
    Stack-managed per feedback_stack_managed_services.
    """
    log_file = LOG_DIR / "handoff_dashboard.log"
    port = 8100
    repo = Path("/mnt/raid0/llm/epyc-root")
    server = repo / "dashboard" / "server.py"

    print(f"  Starting handoff_dashboard (epyc-root hub) on port {port}")

    if not server.exists():
        print(f"    [FAIL] hub server not found: {server}")
        return None

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "dashboard.server", "--host", "0.0.0.0", "--port", str(port)],
            cwd=str(repo),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print("    Waiting for health (path=/health, timeout=30s)...")

    if wait_for_health(port, timeout=30, path="/health"):
        print("    [OK] handoff dashboard ready")
        return ProcessInfo(
            role="handoff_dashboard",
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path="epyc-root handoff progress hub (stdlib)",
            log_file=str(log_file),
        )
    else:
        print("    [FAIL] handoff dashboard did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


# =============================================================================
# Commands
# =============================================================================


# =============================================================================
# CLI commands (cmd_start, cmd_stop, cmd_reload, cmd_status) moved to
# scripts/server/stack_commands.py (2026-05-22 refactor). main() below imports
# them lazily to avoid a module-import cycle with stack_commands, which itself
# pulls helpers from this module.
# Module-level __getattr__ exposes them as orchestrator_stack.cmd_* for
# backward compatibility (e.g. tests that call stack.cmd_reload(...) after
# `from scripts.server import orchestrator_stack as stack`).
# =============================================================================

_STACK_MANIFEST_EXPORTS = {
    "PORT_MAP",
    "ROLE_LAUNCH_META",
    "HOT_ROLES",
    "NUMA_REPLICA_PORTS",
    "HOT_SERVERS",
    "WARM_SERVERS",
    "DOCKER_SERVICES",
    "validate_model_paths",
    "validate_against_registry",
    "_build_servers_from_classification",
    "_validate_role_classification",
    "_filter_by_numa_mode",
}

_STACK_PATH_EXPORTS = {
    "_get_paths",
    "LLAMA_MATH_TOOLS",
}


def __getattr__(name: str):
    if name in ("cmd_start", "cmd_stop", "cmd_reload", "cmd_status"):
        from scripts.server import stack_commands

        return getattr(stack_commands, name)
    if name in _STACK_MANIFEST_EXPORTS:
        from scripts.server import stack_manifest

        return getattr(stack_manifest, name)
    if name in _STACK_PATH_EXPORTS:
        from scripts.server import stack_paths

        return getattr(stack_paths, name)
    raise AttributeError(f"module 'scripts.server.orchestrator_stack' has no attribute {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator stack manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the stack")
    start_parser.add_argument("--hot-only", action="store_true", help="Start HOT models only")
    start_parser.add_argument(
        "--include-warm", nargs="+", metavar="ROLE", help="Include WARM models"
    )
    start_parser.add_argument(
        "--only",
        nargs="+",
        metavar="ROLE",
        help="Start ONLY these roles (skip everything else). "
        "Searches both HOT and WARM server lists.",
    )
    start_parser.add_argument("--dev", action="store_true", help="Dev mode (single 0.5B model)")
    start_parser.add_argument(
        "--numa-mode",
        choices=["full", "quarter", "both"],
        default="full",
        help=(
            "For roles with both a full-NUMA-node instance and quarter-instance siblings "
            "(currently frontdoor + coder_escalation + worker_general — see "
            "NUMA_CONFIG[role]['full_instance_idx']), pick one mode. "
            "'full' = single full instance (max single-stream tps; recommended for single-user "
            "workloads; default for AutoPilot/eval integrity). 'quarter' = 4 concurrent quarters "
            "(max aggregate under multi-request load). 'both' = compatibility mode with all 5 — viable "
            "when the role's -t is small enough to avoid CPU oversubscription (Qwen3-Coder -t 24 "
            "and Qwen3.6-35B Q8 quarter-tuned were OK; gemma4-MTP -t 96 will hit load 420 → "
            "9 t/s with 'both', so use --numa-mode full for that role specifically). "
            "Single-instance roles (architect_general, ingest_long_context, embedders) are "
            "unaffected by this flag."
        ),
    )
    start_parser.add_argument(
        "--skip-host-prereqs",
        action="store_true",
        help="Skip host_prereq audit/apply (numa_balancing, THP, governor). NOT recommended for benchmarks.",
    )
    start_parser.add_argument(
        "--skip-stack-change-gate",
        action="store_true",
        help=(
            "Skip the canonical stack-change promotion gate before production launch. "
            "Use only for emergency diagnostics; benchmarks and AutoPilot resumes should not bypass it."
        ),
    )
    start_parser.add_argument(
        "--skip-page-cache-prewarm",
        action="store_true",
        help="Skip the [1.5] numactl --interleave=all GGUF prewarm step. "
        "NOT recommended after a cold cache / container rebuild — sequential "
        "mlock will pin all shared-GGUF pages to one NUMA node and quarters "
        "will fetch cross-socket. See handoffs/active/numa-page-cache-prewarm.md. "
        "Equivalent: set ORCHESTRATOR_SKIP_PAGE_CACHE_PREWARM=1.",
    )
    start_parser.add_argument(
        "--repair-embeddings",
        action="store_true",
        help="If [0.7] embedding health check finds orphans, run repair before launch "
        "(re-embeds via the configured parallel BGE servers, rebuilds FAISS index, ~5-15 min). "
        "Default behavior is read-only — just print warning and continue. "
        "See scripts/maintenance/repair_episodic_embeddings.py for the manual workflow.",
    )
    start_parser.add_argument(
        "--profile",
        choices=sorted(ORCHESTRATOR_PROFILES.keys()),
        help="Optional orchestrator API env profile",
    )
    start_parser.add_argument(
        "--stack-profile",
        metavar="NAME",
        help="Load stack template from stack_templates/<NAME>.yaml (DS-7). "
        "Use --validate-only to check without launching.",
    )
    start_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate stack template and exit (use with --stack-profile)",
    )
    start_parser.add_argument(
        "--migrate-to",
        metavar="NAME",
        help="Migrate running stack to stack_templates/<NAME>.yaml via full "
        "restart (DS-7 / NIB2-19). Use with --dry-run to plan only.",
    )
    start_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --migrate-to, plan the migration without stopping any servers.",
    )
    start_parser.add_argument(
        "--compile-registry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompile orchestration/model_registry.yaml from the master "
        "registry at epyc-inference-research before starting (default: on). "
        "Cache-aware: no-op if neither master nor active role set has changed. "
        "Use --no-compile-registry or set ORCHESTRATOR_REGISTRY_NO_COMPILE=1 "
        "to disable.",
    )
    start_parser.add_argument(
        "--compile-descriptors",
        action="store_true",
        help="Compile orchestration/model_descriptors.yaml from the active "
        "lean/research registries before starting. Strict by default: refuses "
        "missing load-bearing descriptor fields.",
    )
    start_parser.add_argument(
        "--allow-incomplete-descriptors",
        action="store_true",
        help="With --compile-descriptors, emit descriptor records with known_gaps "
        "instead of failing on incomplete evidence. Intended for diagnostics only.",
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop components")
    stop_parser.add_argument("--all", action="store_true", help="Stop all components")
    stop_parser.add_argument("components", nargs="*", help="Components to stop")

    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload components")
    reload_parser.add_argument("components", nargs="+", help="Components to reload")
    reload_parser.add_argument(
        "--profile",
        choices=sorted(ORCHESTRATOR_PROFILES.keys()),
        help="Optional orchestrator API env profile (used when reloading orchestrator)",
    )

    # Status command
    subparsers.add_parser("status", help="Show status")

    args = parser.parse_args()

    # Lazy import — stack_commands imports back into this module's helpers
    # (start_server, init_memrl_and_tools, the thin process/state wrappers).
    # Loading it only inside main() avoids the circular-import problem.
    from scripts.server.stack_commands import (
        cmd_start,
        cmd_stop,
        cmd_reload,
        cmd_status,
    )

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "reload":
        return cmd_reload(args)
    elif args.command == "status":
        return cmd_status(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
