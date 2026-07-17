"""Uniform freshness contract for dashboard panels.

Every dashboard data source is stamped with a ``_freshness`` envelope so the
operator UI can render one consistent staleness badge per panel and, above all,
so a silently-stale source is *never* shown as if it were current.

This is the structural guarantee that replaces the per-panel, ad-hoc mtime
handling that kept regressing (see the recurring "dashboard panel stale"
incidents in ``progress/2026-05..07``). Historically each panel invented its
own freshness signal (some had a live-dot, some an "Ns ago" string, some
nothing), and the ``topology`` / ``region_locks`` endpoints returned no source
mtime at all — so when one producer died, that panel froze quietly and the
operator only noticed by eye. A single envelope shape closes that gap.

The envelope names the three failure kinds the old code conflated:

  - ``dead``  — the source file is missing entirely (producer never ran, or the
                path drifted). A required source that is absent is ``dead``.
  - ``stale`` — the source exists but has not advanced within its stale
                threshold (producer crashed / wedged). This is the "frozen
                frontier" class that repeatedly got misread as a render bug.
  - ``aging`` — past the warn threshold but not yet stale (soft signal).
  - ``fresh`` — updated within the warn threshold.

A *transport* failure (the browser's fetch threw, or the SSE stream dropped) is
a frontend concern rendered separately from this envelope — the point of the
distinction is that "the data source stopped advancing" and "I could not reach
the server" must not look identical to the operator.

Pure module: standard library only, no imports from ``dashboard`` — so it is
safe to import from any route module and trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Staleness classes, least → most severe. Kept as plain strings so they pass
# straight through JSON to the frontend, which keys its badge CSS off them.
FRESH = "fresh"
AGING = "aging"
STALE = "stale"
DEAD = "dead"

# "Worst source wins" ordering when a panel folds several sources into one
# panel-level class.
_SEVERITY = {FRESH: 0, AGING: 1, STALE: 2, DEAD: 3}

# VALUE-consistency classes — a SEPARATE axis from the age-based staleness
# classes above. Staleness only asks "is this file's mtime advancing?"; two
# sources can BOTH be freshly written (both ``fresh``) yet hold DISAGREEING
# values, and age-classification renders both current, hiding the incoherence.
# ``divergent`` names exactly that: representations that disagree in VALUE, not
# in age. Kept as its own class set so the frontend can badge value-divergence
# independently of (and simultaneously with) a staleness badge.
COHERENT = "coherent"
DIVERGENT = "divergent"


def value_consistency(
    trial_counter: int | None,
    journal_max_trial: int | None,
    *,
    tolerance: int = 1,
) -> dict[str, Any]:
    """Cross-source VALUE-consistency check for the autopilot-state plane.

    Compares ``autopilot_state.json``'s ``trial_counter`` against the
    append-only journal's max trial id. When the journal is AHEAD of the state
    counter by more than ``tolerance`` trials, the state file is stale relative
    to the journal (a crash / rewind between saves) and the two on-disk
    representations DIVERGE — a distinct failure from age-staleness: both files
    can be freshly written yet disagree, so both read ``fresh`` under the
    age-only contract and the incoherence is otherwise invisible.

    Direction matters. The journal is appended only on metric-bearing trials, so
    ``trial_counter`` legitimately RUNS AHEAD of ``journal_max_trial`` (trials
    that produced no journal row). Only journal-ahead-of-state is flagged
    ``divergent``; the reverse is expected and stays ``coherent``. A small
    ``tolerance`` absorbs the one-trial race where the journal row for trial N
    is appended a moment before the state counter is bumped to N.

    Returns a dict whose ``class`` is ``coherent`` or ``divergent`` (plus the
    raw values, the ``trial_lag`` = journal − state, and the ``tolerance``),
    kept separate from the staleness envelope's classes so the UI can badge
    value-divergence independently of age.
    """
    lag: int | None = None
    cls = COHERENT
    reason = ""
    if trial_counter is not None and journal_max_trial is not None:
        lag = journal_max_trial - trial_counter
        if lag > tolerance:
            cls = DIVERGENT
            reason = (
                f"autopilot_state.json trial_counter ({trial_counter}) lags the "
                f"journal max trial ({journal_max_trial}) by {lag} > tolerance "
                f"{tolerance}: state file is stale relative to the append-only "
                f"journal (values disagree though both files may be fresh)."
            )
    return {
        "class": cls,
        "trial_counter": trial_counter,
        "journal_max_trial": journal_max_trial,
        "trial_lag": lag,
        "tolerance": tolerance,
        "reason": reason,
    }


@dataclass(frozen=True)
class Source:
    """One input a panel depends on, with its own age thresholds.

    ``mtime`` is epoch seconds (``float``) or ``None`` when the source is
    absent. ``warn_s``/``stale_s`` are ages, in seconds, at which the source
    becomes ``aging`` / ``stale``. An ``optional`` source that is absent is
    treated as ``fresh`` (ignored) rather than ``dead`` — for inputs that are
    legitimately empty until first use (e.g. a lock file that only appears once
    a backend first acquires a lock).
    """

    label: str
    mtime: float | None
    warn_s: float
    stale_s: float
    optional: bool = False
    # gating=True: this source drives the panel-level staleness_class. gating=False:
    # informational only — its age is reported in ``sources`` for context, but a
    # stale/old informational source does NOT flip the panel badge. Used for
    # operator-curated backing files (contention matrix) and legacy/secondary
    # inputs (prompt/repl taps) that are naturally old while the panel is live.
    gating: bool = True


def classify(
    age_s: float | None,
    warn_s: float,
    stale_s: float,
    *,
    present: bool = True,
    optional: bool = False,
) -> str:
    """Map an age (seconds) to a staleness class.

    ``present=False`` means the source file does not exist. A missing required
    source is ``dead``; a missing optional source is ``fresh`` (ignored).
    """
    if not present:
        return FRESH if optional else DEAD
    if age_s is None:
        return FRESH if optional else DEAD
    if age_s >= stale_s:
        return STALE
    if age_s >= warn_s:
        return AGING
    return FRESH


def source_status(src: Source, now: float) -> dict[str, Any]:
    """Render one :class:`Source` as a JSON-able status dict."""
    present = src.mtime is not None
    age = (now - src.mtime) if present else None
    # Guard against clock skew / a file stamped slightly in the future.
    if age is not None and age < 0:
        age = 0.0
    cls = classify(age, src.warn_s, src.stale_s, present=present, optional=src.optional)
    return {
        "label": src.label,
        "mtime": src.mtime,
        "age_s": round(age, 3) if age is not None else None,
        "class": cls,
        "warn_s": src.warn_s,
        "stale_s": src.stale_s,
        "optional": src.optional,
        "gating": src.gating,
    }


def _reason_for(st: dict[str, Any]) -> str:
    label = st["label"]
    cls = st["class"]
    if cls == DEAD:
        return f"{label}: source missing"
    if cls == STALE:
        return f"{label}: no update in {st['age_s']:.0f}s (>{st['stale_s']:.0f}s stale threshold)"
    if cls == AGING:
        return f"{label}: {st['age_s']:.0f}s old"
    return ""


def envelope(
    sources: Iterable[Source],
    *,
    now: float,
    generated_at: float | None = None,
    consistency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fold ``sources`` into one panel-level freshness envelope.

    The panel's ``staleness_class`` is the worst class across all its sources,
    and ``reason`` explains the worst offender. ``generated_at`` defaults to
    ``now`` (when the payload was built) — surfaced so the frontend can show
    "server built this Ns ago" independently of per-source staleness.

    ``now`` is a required argument (rather than defaulting to ``time.time()``)
    so callers pass a single coherent timestamp shared with the payload they
    are stamping, and so this stays deterministic under test.

    ``consistency`` (optional) is a VALUE-consistency verdict from
    :func:`value_consistency` — a distinct axis from age-staleness. When
    provided it is attached as ``value_consistency`` and its class is surfaced
    as ``consistency_class`` so a client keying only off ``staleness_class``
    still sees a value-divergence badge. Omitted for panels that have no
    cross-source value to reconcile (their envelope shape is unchanged).
    """
    statuses = [source_status(s, now) for s in sources]
    # Only GATING sources drive the panel-level class / reason / worst-age.
    # Informational sources (gating=False) are still reported in ``sources`` for
    # UI context but never flip the badge — an operator-curated matrix that is a
    # week old must not make a live, updating panel read as stale.
    worst = FRESH
    worst_reason = ""
    worst_age: float | None = None
    for st in statuses:
        if not st.get("gating", True):
            continue
        if _SEVERITY[st["class"]] > _SEVERITY[worst]:
            worst = st["class"]
            worst_reason = _reason_for(st)
        age = st["age_s"]
        if age is not None and (worst_age is None or age > worst_age):
            worst_age = age
    env: dict[str, Any] = {
        "generated_at": generated_at if generated_at is not None else now,
        "sources": statuses,
        "worst_age_s": round(worst_age, 3) if worst_age is not None else None,
        "staleness_class": worst,
        "reason": worst_reason,
    }
    if consistency is not None:
        env["value_consistency"] = consistency
        env["consistency_class"] = consistency.get("class", COHERENT)
    return env


def mtime(path: Any) -> float | None:
    """``st_mtime`` of ``path`` (epoch seconds), or ``None`` if it is absent.

    Central helper so every panel resolves source mtimes the same way — the old
    code re-defined a local ``mtime()`` in three places (dashboard.py:364, :2755
    and inline elsewhere), which is exactly how one panel's mtime handling
    drifted from another's.
    """
    try:
        return Path(path).stat().st_mtime
    except (OSError, TypeError, ValueError):
        return None


def stamp(payload: dict[str, Any], sources: Iterable[Source], *, now: float) -> dict[str, Any]:
    """Attach a ``_freshness`` envelope to ``payload`` in place and return it.

    Additive and non-breaking: existing keys are untouched, so current frontend
    code keeps working while new code reads ``payload["_freshness"]``.
    """
    payload["_freshness"] = envelope(sources, now=now, generated_at=payload.get("generated_at", now))
    return payload
