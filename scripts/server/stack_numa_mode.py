"""Shared NUMA-mode normalization for stack readers."""

from __future__ import annotations

import os
from collections.abc import Mapping


VALID_STACK_NUMA_MODES = frozenset({"full", "quarter", "both"})
DEFAULT_STACK_NUMA_MODE = "full"
DASHBOARD_RUNTIME_FALLBACK_NUMA_MODE = "both"


def normalize_stack_numa_mode(
    value: str | None,
    *,
    default: str = DEFAULT_STACK_NUMA_MODE,
) -> str:
    """Normalize a stack NUMA mode, falling back to a validated default."""
    fallback = default if default in VALID_STACK_NUMA_MODES else DEFAULT_STACK_NUMA_MODE
    mode = (value or fallback).strip().lower()
    return mode if mode in VALID_STACK_NUMA_MODES else fallback


def env_stack_numa_mode(
    *,
    default: str = DEFAULT_STACK_NUMA_MODE,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Read and normalize ORCHESTRATOR_STACK_NUMA_MODE from an env mapping."""
    env = os.environ if environ is None else environ
    return normalize_stack_numa_mode(
        env.get("ORCHESTRATOR_STACK_NUMA_MODE"),
        default=default,
    )
