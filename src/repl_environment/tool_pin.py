"""CPU pinning for tool / REPL child processes (flag ``tool_process_cpu_pin``).

Tool subprocesses (git, rg, colgrep, sandboxed python, REPL child lanes) used to
inherit the API process affinity and ran on cores serving roles were measured on.
With the flag on, every such child is pinned via ``os.sched_setaffinity`` in
``preexec_fn`` to ``tools.cpu_set`` (env ``ORCHESTRATOR_TOOLS_CPU_SET``, default
"84-87", the shared agent core set). Flag off: ``popen_kwargs()`` is ``{}`` so
every call site is byte-identical to before.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TOOL_CPU_SET = "84-87"
_warned = False


def parse_cpus(spec: str) -> set[int]:
    """Parse a taskset-style list ('84-87,100'). Raises ValueError if malformed/empty."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < lo_i:
                raise ValueError(f"bad cpu range {part!r}")
            out.update(range(lo_i, hi_i + 1))
        else:
            out.add(int(part))
    if not out or min(out) < 0:
        raise ValueError(f"empty/invalid cpu set {spec!r}")
    return out


def tool_cpu_set_spec() -> str:
    return os.environ.get("ORCHESTRATOR_TOOLS_CPU_SET", "").strip() or DEFAULT_TOOL_CPU_SET


def tool_pin_enabled() -> bool:
    from src.features import features

    return bool(features().tool_process_cpu_pin)


def _pin_preexec(cpus: frozenset[int]) -> Callable[[], None]:
    def _pin() -> None:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError:
            pass  # a CPU outside the container cpuset must not kill the tool launch

    return _pin


def preexec() -> Callable[[], None] | None:
    """The preexec_fn pinning the child, or None when the flag is off."""
    if not tool_pin_enabled():
        return None
    try:
        cpus = parse_cpus(tool_cpu_set_spec())
    except ValueError as exc:
        logger.warning("tools.cpu_set %r invalid (%s); tool pin disabled", tool_cpu_set_spec(), exc)
        return None
    return _pin_preexec(frozenset(cpus))


def popen_kwargs() -> dict[str, Any]:
    """``**``-splat into subprocess.run/Popen: {} when off, preexec_fn when on."""
    fn = preexec()
    return {} if fn is None else {"preexec_fn": fn}


def role_cpu_overlaps(cpu_set: set[int], numa_config: dict | None = None) -> dict[str, list[int]]:
    """Map 'role[idx]' -> sorted overlapping logical CPUs, over every serving instance cpu_list."""
    if numa_config is None:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]

        numa_config = NUMA_CONFIG
    found: dict[str, list[int]] = {}
    for role, cfg in numa_config.items():
        for idx, inst in enumerate(cfg.get("instances", ())):
            cpus = parse_cpus(inst[0])
            both = sorted(cpu_set & cpus)
            if both:
                found[f"{role}[{idx}]"] = both
    return found


def check_tool_cpu_overlap(numa_config: dict | None = None, *, strict: bool = False) -> dict[str, list[int]]:
    """Startup check (no-op when the flag is off). Warns on overlap with any serving
    role's cpu_shape; raises RuntimeError instead when ``strict``. Returns the overlaps."""
    global _warned
    if not tool_pin_enabled():
        return {}
    try:
        overlaps = role_cpu_overlaps(parse_cpus(tool_cpu_set_spec()), numa_config)
    except Exception as exc:  # unreadable topology: say so, do not block startup
        logger.warning("tool_process_cpu_pin: could not check tools.cpu_set overlap: %s", exc)
        return {}
    if overlaps:
        msg = (
            f"tools.cpu_set {tool_cpu_set_spec()!r} overlaps serving role cpu_shape(s): "
            + "; ".join(f"{k} on {v}" for k, v in sorted(overlaps.items()))
        )
        if strict:
            raise RuntimeError(msg)
        if not _warned:
            logger.warning(msg)
            _warned = True
    return overlaps
