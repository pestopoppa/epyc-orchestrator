"""Binary path + registry-timeout helpers used by the executor module.

Extracted from scripts/lib/executor.py during the 2026-05-22 Task-J refactor.
executor.py re-imports every name so existing callers keep working.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

try:
    from .registry import ModelRegistry, load_registry
except ImportError:
    from registry import ModelRegistry, load_registry


def _numa_prefix() -> list[str]:
    """Return numactl interleave prefix if numactl is available, else empty list."""
    if shutil.which("numactl"):
        return ["numactl", "--interleave=all"]
    return []


def _read_registry_timeout(category: str, key: str, fallback: int) -> int:
    """Read timeout from model_registry.yaml."""
    try:
        reg = load_registry()
        if reg and reg._raw:
            timeouts = reg._raw.get("runtime_defaults", {}).get("timeouts", {})
            cat_data = timeouts.get(category, {})
            return cat_data.get(key, timeouts.get("default", fallback))
    except Exception:
        pass
    return fallback


def get_binary_paths(registry: Optional["ModelRegistry"] = None) -> dict[str, str]:
    """Get binary paths from registry (single source of truth).

    Falls back to hardcoded paths only if registry is unavailable.
    """
    fallback = {
        "base_dir": "/mnt/raid0/llm/llama.cpp/build/bin",
        "completion": "llama-completion",
        "speculative": "llama-speculative",
        "lookup": "llama-lookup",
        "cli": "llama-cli",
        "server": "llama-server",
    }

    if registry is None:
        try:
            registry = load_registry()
        except Exception:
            pass

    if registry and hasattr(registry, "data"):
        binaries = registry.data.get("runtime_defaults", {}).get("binaries", {})
        if binaries:
            return binaries

    return fallback


def get_binary(name: str, registry: Optional["ModelRegistry"] = None) -> str:
    """Get full path to a specific binary.

    Args:
        name: Binary name ('completion', 'speculative', 'lookup', 'cli')
        registry: Optional registry instance

    Returns:
        Full absolute path to the binary
    """
    paths = get_binary_paths(registry)
    base_dir = paths.get("base_dir", "/mnt/raid0/llm/llama.cpp/build/bin")
    binary_name = paths.get(name, name)
    return os.path.join(base_dir, binary_name)


def validate_binaries(registry: Optional["ModelRegistry"] = None) -> dict[str, str]:
    """Validate all required binaries exist.

    Raises:
        FileNotFoundError: If any binary is missing, with clear error message.

    Returns:
        Dict mapping binary name to full path (for logging/debugging).
    """
    required = ["completion", "speculative", "lookup"]
    paths = {}
    missing = []

    for name in required:
        path = get_binary(name, registry)
        paths[name] = path
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")

    if missing:
        raise FileNotFoundError(
            f"Missing llama.cpp binaries (check registry runtime_defaults.binaries):\n"
            + "\n".join(missing)
            + f"\n\nRegistry location: /mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml"
        )

    return paths


def get_server_defaults(registry: Optional["ModelRegistry"] = None) -> dict:
    """Get server defaults from registry.

    Returns dict with: port, context_length, startup_timeout, request_timeout, parallel_slots
    """
    defaults = {
        "port": 8080,
        "context_length": 131072,  # 131K - Qwen3 native limit
        "startup_timeout": 600,
        "request_timeout": 300,
        "parallel_slots": 4,
    }

    if registry is None:
        try:
            registry = load_registry()
        except Exception:
            pass

    if registry and hasattr(registry, "data"):
        server_cfg = registry.data.get("runtime_defaults", {}).get("server_defaults", {})
        if server_cfg:
            defaults.update(server_cfg)

    return defaults
