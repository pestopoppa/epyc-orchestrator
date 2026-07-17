"""In-process KV-migration counters (Prometheus-shaped; no prometheus_client dep).

`within-role-placement-state-machine.md` L134/L379 (WP-4) asks for two migration
counters — ``kv_migration_direction_total{direction=...}`` and
``kv_migration_thrash_skipped_total`` — to replace the current log/stat-only
observable evidence for the reverse-migration path.

**Surface choice (documented per the handoff, which anticipates it).** There is
NO ``prometheus_client`` in ``src/`` (verified 2026-07-17), and the plan's A3
"no stack changes" + additive-only constraints rule out introducing a new
runtime dependency. So the counters land on the existing ``src/metrics/``
telemetry surface as a minimal, thread-safe, in-process registry that renders
Prometheus text-exposition on demand. This is a drop-in shim: when
``prometheus_client`` is later adopted, swap the internals of
``record_* / snapshot() / render_prometheus()`` and the two call sites stay
unchanged —

  * direction: ``MigrationTransaction.advance()`` on the COMMITTED transition
    (``src/scheduling/migration_transaction.py``), so both the forward
    ``_migrate_kv`` and reverse ``_reverse_migrate_kv`` paths are counted once
    each with zero backend edits;
  * thrash skips: the anti-thrash guard returns in
    ``ConcurrencyAwareBackend._maybe_spawn_reverse_migration``
    (``src/backends/concurrency_aware.py``).

All record functions are best-effort and never raise — a metrics failure must
never break a migration commit or a dispatch decision.
"""

from __future__ import annotations

import threading

# Metric family names — EXACTLY as named in the handoff. Do not rename: the
# dashboard / Package-J observable-evidence contract keys off these strings.
DIRECTION_TOTAL = "kv_migration_direction_total"
THRASH_SKIPPED_TOTAL = "kv_migration_thrash_skipped_total"

# Direction label values.
FORWARD = "forward"  # full -> quarter (load rising)
REVERSE = "reverse"  # quarter -> full (load dropping; WP-4)

_lock = threading.Lock()
_direction_counts: dict[str, int] = {}
_thrash_skipped_counts: dict[str, int] = {}


def direction_for_target_quarter(target_quarter: int) -> str:
    """Map a ``MigrationTransaction.target_quarter`` to a direction label.

    Convention (``concurrency_aware.py``): ``target_quarter == -1`` means the
    destination is the full instance (a quarter->full *reverse* migration); any
    non-negative index is a full->quarter *forward* migration.
    """
    return REVERSE if target_quarter < 0 else FORWARD


def record_migration_direction(direction: str) -> None:
    """Increment ``kv_migration_direction_total{direction=...}`` by one.

    An unknown label is recorded verbatim (never raises) so a wiring bug
    degrades to an observable series rather than breaking a migration commit.
    """
    with _lock:
        _direction_counts[direction] = _direction_counts.get(direction, 0) + 1


def record_thrash_skip(reason: str = "unspecified") -> None:
    """Increment ``kv_migration_thrash_skipped_total{reason=...}`` by one.

    ``reason`` distinguishes which anti-thrash guard fired (``cooldown``,
    ``session_cap``). ``sum()`` across the ``reason`` label reproduces the bare
    ``kv_migration_thrash_skipped_total`` the handoff names.
    """
    key = reason or "unspecified"
    with _lock:
        _thrash_skipped_counts[key] = _thrash_skipped_counts.get(key, 0) + 1


def direction_total(direction: str | None = None) -> int:
    """Return the forward+reverse total, or the count for one ``direction``."""
    with _lock:
        if direction is None:
            return sum(_direction_counts.values())
        return _direction_counts.get(direction, 0)


def thrash_skipped_total(reason: str | None = None) -> int:
    """Return the total thrash skips, or the count for one ``reason``."""
    with _lock:
        if reason is None:
            return sum(_thrash_skipped_counts.values())
        return _thrash_skipped_counts.get(reason, 0)


def snapshot() -> dict[str, dict[str, int]]:
    """Return a copy of the current counter state (safe for callers to mutate)."""
    with _lock:
        return {
            DIRECTION_TOTAL: dict(_direction_counts),
            THRASH_SKIPPED_TOTAL: dict(_thrash_skipped_counts),
        }


def render_prometheus() -> str:
    """Render the counters in Prometheus text-exposition format."""
    with _lock:
        lines: list[str] = [
            f"# HELP {DIRECTION_TOTAL} KV migrations committed, by direction.",
            f"# TYPE {DIRECTION_TOTAL} counter",
        ]
        for direction in sorted(_direction_counts):
            lines.append(
                f'{DIRECTION_TOTAL}{{direction="{direction}"}} {_direction_counts[direction]}'
            )
        lines.append(
            f"# HELP {THRASH_SKIPPED_TOTAL} Reverse migrations skipped by an anti-thrash guard."
        )
        lines.append(f"# TYPE {THRASH_SKIPPED_TOTAL} counter")
        for reason in sorted(_thrash_skipped_counts):
            lines.append(
                f'{THRASH_SKIPPED_TOTAL}{{reason="{reason}"}} {_thrash_skipped_counts[reason]}'
            )
        return "\n".join(lines) + "\n"


def reset() -> None:
    """Zero all counters. Test-only — production never resets a counter."""
    with _lock:
        _direction_counts.clear()
        _thrash_skipped_counts.clear()
