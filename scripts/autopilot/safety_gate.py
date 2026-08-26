"""Safety gate: quality floor, regression guards, rollback triggers.

Loads frozen baseline from autopilot_baseline.yaml and enforces constraints.
"""

from __future__ import annotations

import logging
import math
import os
import re
import statistics
from collections import deque, namedtuple
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.autopilot_core.tier_specs import (
    DEFAULT_FRONTIER_TIER,
    MIN_FRONTIER_EVAL_TIER,
)
from src.autopilot_core.rlvr_tiers import rlvr_reward_from_result
from src.autopilot_core.authority_consent import (
    SEQ_P0_2_BRIDGE_MODE,
    seq_p0_2_bridge_status,
)

log = logging.getLogger("autopilot.safety")


def _env_truthy(name: str) -> bool:
    """Return True when env var ``name`` is set to a truthy token."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "autopilot_baseline.yaml"
)

# Remediation shown whenever the baseline file is unreadable/unparseable. Kept as a
# single constant so the raised message and the docs stay in lockstep.
_BASELINE_REMEDIATION = (
    "Restore the file from git or recompute via "
    "`autopilot.py checkpoint --production-best`."
)


class BaselineCorruptError(RuntimeError):
    """Raised when the baseline YAML is unreadable/unparseable.

    The gate deliberately refuses to start on a silent default: a missing or
    zeroed baseline passes every trial, so falling back quietly would weaken
    regression gating exactly when the operator most needs it enforced.
    Remediation: restore the file from git or recompute via
    ``autopilot.py checkpoint --production-best``.
    """


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` (mirrors state_store.save_state).

    Write to a per-pid temp file, flush + fsync, then os.replace onto the target
    so a SIGKILL or process crash mid-write can never leave a truncated baseline
    YAML on disk (the old truncate-in-place write_text could).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)

# Hard-coded safety thresholds
# Quality is scored on a 0-3 scale (eval_tower: fraction_correct * 3.0); reliability
# is a 0-1 fraction. QUALITY_MAX guards the baseline loader: a persisted baseline.quality
# (or per-suite value) above this is physically impossible and indicates a corrupt/
# wrong-scale baseline file. Such a value silently force-reverts every trial via the
# regression gate (observed 2026-05-27→05-31: a stale baseline of 9.900 reverted ~160
# trials, including genuinely-good q=2.4 ones). Reject + fall back to defaults instead.
QUALITY_MAX = 3.0
RELIABILITY_MAX = 1.0
# T0 thresholds (10 sentinel questions — inflated scale, saturates at 3.0)
QUALITY_FLOOR_T0 = 2.0  # Average quality >= 2.0/3.0
# T1/T2 thresholds (50-500 real benchmark questions — honest signal)
QUALITY_FLOOR_T1 = 1.0  # ~33% correct minimum
REGRESSION_THRESHOLD = -0.05  # Max quality drop vs baseline (fraction of baseline)
PER_SUITE_REGRESSION = -0.1  # Max per-suite quality drop (fixed floor; see below)
# Minimum per-suite sample below which a threshold-crossing per-suite regression
# is advisory unless the suite collapses catastrophically. At n questions a single
# correct->incorrect flip moves the 0-3 score by 3/n, so sparse result/baseline
# samples can produce large apparent drops. The threshold below still detects
# those drops; SafetyGate.check decides whether the sparse signal is binding.
# (2026-06-06: at ~2 q/suite the -0.1 floor fired -1.5 "regressions" on every
# seeder trial, mass-excluding via mad_noise and deadlocking the planner/critic.)
# (2026-07-04: W8 candidate generation deadlocked again when a T1 general baseline
# with n=2 scored 3.0 and moderate 5-question candidate drops bound as AP-24
# terminal failures. Keep sparse drops visible as warnings, but require either
# adequate support or a catastrophic drop before failing the trial.)
PER_SUITE_BINDING_MIN_COUNT = 5
PER_SUITE_LOW_SUPPORT_CATASTROPHIC_DROP = 2.5
# (2026-07-16 rollback thrash, trials 1404-1433; resume-precondition in epyc-root
# handoffs/active/autopilot-continuous-optimization.md: a debugbench baseline
# measured on only n=2 scored 3.0, so a 0.0 trial read as a -3.0 "catastrophic"
# collapse and the low-support escape above still hard-failed the trial, feeding
# consecutive_failures into ~10 straight rollbacks.) A baseline sampled below
# this minimum has no resolution to certify ANY hard per-suite rollback — its
# score is quantized to multiples of 3/n, so a collapse is indistinguishable
# from a couple of unlucky draws. Threshold-crossing drops against such a
# baseline stay visible as advisory warnings (the suite is NOT dropped from
# scoring); they just never bind. Env-overridable via
# AUTOPILOT_PER_SUITE_BASELINE_MIN_N (malformed/non-positive values ignored).
PER_SUITE_BASELINE_HARD_MIN_N = 5
# tool_use sentinel suite (5 questions, REPL-mode, substring scoring) is inherently
# flaky — models invoke the tool correctly but don't reliably echo the returned secret.
# A single-question drop is -0.6 on the 0-3 scale, enough to trip the per-suite regression
# gate on essentially every config change. Keep the signal visible as an advisory warning,
# but only treat it as a hard violation when the regression is catastrophic (3+ questions
# failed, delta <= -3.0 on the 0-3 scale).
TOOL_USE_CATASTROPHIC_REGRESSION = 3.0  # magnitude; delta <= -this is hard-fail for tool_use
# B2 / SG-1 float-representation guard. per_suite_regression_threshold() returns the
# single-flip quantum (3/n) as the boundary, and its docstring states a delta must be
# "MORE negative" than the threshold to count. But `fraction*3` and `-max(...,3/n)` are
# computed separately, so a delta that is exactly one flip lands ~1e-16 to one side of the
# threshold at random — 185 (n,k) pairs cross the bare `<` purely from float rounding
# (see the artifact sweep). Comparing against `threshold - PER_SUITE_EPS` restores the
# documented intent: a delta equal to one flip is at-resolution noise, not a violation.
PER_SUITE_EPS = 1e-9
# B1 / REL-1 reliability floor. Below this non-error fraction the eval's per-question
# outcomes are untrustworthy (infra errors), so the quality-floor / regression / per-suite
# checks are computed over garbage. Env-overridable via AUTOPILOT_RELIABILITY_FLOOR (see
# _reliability_floor). A reliability-floor failure signals RETRY, not a revert.
RELIABILITY_FLOOR = 0.8
# B8 / SG-0: warn this many days before the contention matrix crosses the
# MATRIX_STALENESS_DAYS wall (in src.scheduling.contention). Past the wall the matrix goes
# STALE, _baseline_eligible fails-closed, and the baseline ratchet FREEZES.
PRE_EXPIRY_WARN_DAYS = 7
# D4 (audit MET-1): version of the ``METRIC <key>: <value>`` grep-line contract emitted by
# EvalResult.to_grep_lines. v1 = implicit / pre-2026-07-20 (conditional emission; a NaN
# printed the bare string ``nan``; NaN-gated keys were silently dropped). v2 = UNCONDITIONAL
# emission with an explicit ``null`` absence sentinel + sanitized interpolated names. The one
# blessed parser is scripts.autopilot.metric_lines.parse_metric_lines — NOT ad-hoc grep/awk.
METRIC_LINE_SCHEMA_VERSION = 2
# SG-7 (B9): minimum quality delta that counts as a REAL change against a degenerate /
# saturated MAD window (MAD == 0, i.e. every recent same-tier sample is identical). The old
# rule ("any nonzero delta is significant") let a single-question flip masquerade as a fresh
# gain. ~2 single-flip quanta: 2 * (3/n) ≈ 0.2 at n ≈ 30. The result's per-suite counts / n
# are NOT in scope inside _mad_significance, so this fixed floor stands in for 2*(3/n).
MAD_ZERO_MIN_DELTA = 0.2


def _reliability_floor() -> float:
    """Resolve the reliability floor (B1/REL-1), env-overridable via AUTOPILOT_RELIABILITY_FLOOR.

    A missing/malformed/out-of-[0,1] override is ignored (falls back to RELIABILITY_FLOOR)
    and logged, so a fat-fingered env var can never silently disarm the guard."""
    raw = os.environ.get("AUTOPILOT_RELIABILITY_FLOOR", "").strip()
    if not raw:
        return RELIABILITY_FLOOR
    try:
        val = float(raw)
    except ValueError:
        log.warning(
            "Ignoring non-numeric AUTOPILOT_RELIABILITY_FLOOR=%r; using %.2f",
            raw,
            RELIABILITY_FLOOR,
        )
        return RELIABILITY_FLOOR
    if not 0.0 <= val <= 1.0:
        log.warning(
            "Ignoring out-of-range AUTOPILOT_RELIABILITY_FLOOR=%r (need 0..1); using %.2f",
            raw,
            RELIABILITY_FLOOR,
        )
        return RELIABILITY_FLOOR
    return val


def _warn_matrix_pre_expiry() -> None:
    """Read-only heads-up (B8/SG-0): warn when the contention matrix is within
    PRE_EXPIRY_WARN_DAYS of the MATRIX_STALENESS_DAYS wall, past which it goes STALE and
    _baseline_eligible fails-closed — freezing the baseline ratchet.

    Uses ONLY read-only reads of contention's exposed DEFAULT_MATRIX_PATH + staleness
    constant; the file mtime is the exact signal matrix_status() ages against, so no
    private state of src.scheduling.contention is touched or mutated. Fail-soft: any error
    (module unimportable, path missing, stat error) is swallowed — this is an advisory
    countdown and must never affect eligibility."""
    try:
        import time as _time

        from src.scheduling.contention import (
            DEFAULT_MATRIX_PATH,
            MATRIX_STALENESS_DAYS,
        )

        path = DEFAULT_MATRIX_PATH
        if not path.exists():
            return
        age_days = (_time.time() - path.stat().st_mtime) / 86400.0
        remaining = MATRIX_STALENESS_DAYS - age_days
        if 0.0 <= remaining <= PRE_EXPIRY_WARN_DAYS:
            log.warning(
                "baseline ratchet freezes in ~%d day(s) — schedule contention-matrix "
                "re-measurement (matrix age %.1fd, staleness wall %dd).",
                math.ceil(remaining),
                age_days,
                MATRIX_STALENESS_DAYS,
            )
    except Exception:  # noqa: BLE001 — advisory only; never affect eligibility
        pass


def _drop_over_archive_max_tiers(baselines_by_tier: dict[int, float], path: Path) -> None:
    """B3 / SG-4: drop tier baselines whose quality exceeds the same-tier Pareto archive max.

    A persisted/applied tier baseline strictly above the same-tier frontier max is
    unachievable/corrupt — it force-reverts every honest trial and gate-locks the loop
    (2026-05-31). Mutates ``baselines_by_tier`` IN PLACE, logging each drop at error level.
    A tier whose archive is empty/unreadable (fresh bootstrap) is skipped so a legitimate
    first write is never blocked. Factored out of Baseline.load() so apply_state() applies
    the identical defense-in-depth guard to state-sourced tier baselines."""
    for tier, tier_quality in list(baselines_by_tier.items()):
        tier_archive_max = _pareto_frontier_best_quality(tier)
        if tier_archive_max is None:
            continue
        if tier_quality > tier_archive_max + BASELINE_ARCHIVE_TOLERANCE:
            log.error(
                "Persisted/applied T%d baseline quality %.3f in %s exceeds same-tier Pareto "
                "archive max %.3f — unachievable/corrupt; dropping tier baseline.",
                tier,
                tier_quality,
                path,
                tier_archive_max,
            )
            del baselines_by_tier[tier]


def per_suite_regression_threshold(result_n: int | None, baseline_n: int | None) -> float:
    """Resolution-aware per-suite regression threshold (a negative number).

    A suite delta must be MORE negative than this to count as a violation. The
    bound is the coarser of the two samples' single-flip quantum (3/n) and never
    tighter than the fixed PER_SUITE_REGRESSION floor. Missing/zero counts fall
    back to the fixed floor (pre-2026-06-06 behavior)."""
    quanta = [abs(PER_SUITE_REGRESSION)]
    if result_n and result_n > 0:
        quanta.append(3.0 / result_n)
    if baseline_n and baseline_n > 0:
        quanta.append(3.0 / baseline_n)
    return -max(quanta)


def _per_suite_baseline_min_n() -> int:
    """Resolve the minimum baseline sample for a BINDING per-suite regression.

    Env-overridable via AUTOPILOT_PER_SUITE_BASELINE_MIN_N. A missing/malformed/
    non-positive override is ignored (falls back to PER_SUITE_BASELINE_HARD_MIN_N)
    and logged, so a fat-fingered env var can never silently disarm the guard."""
    raw = os.environ.get("AUTOPILOT_PER_SUITE_BASELINE_MIN_N", "").strip()
    if not raw:
        return PER_SUITE_BASELINE_HARD_MIN_N
    try:
        val = int(raw)
    except ValueError:
        log.warning(
            "Ignoring non-integer AUTOPILOT_PER_SUITE_BASELINE_MIN_N=%r; using %d",
            raw,
            PER_SUITE_BASELINE_HARD_MIN_N,
        )
        return PER_SUITE_BASELINE_HARD_MIN_N
    if val < 1:
        log.warning(
            "Ignoring non-positive AUTOPILOT_PER_SUITE_BASELINE_MIN_N=%r; using %d",
            raw,
            PER_SUITE_BASELINE_HARD_MIN_N,
        )
        return PER_SUITE_BASELINE_HARD_MIN_N
    return val


def _per_suite_regression_binding(
    suite_delta: float, result_n: int | None, baseline_n: int | None
) -> bool:
    """Whether a threshold-crossing per-suite drop should fail the trial."""
    if result_n is None or baseline_n is None:
        return True
    if result_n <= 0 or baseline_n <= 0:
        return True
    if baseline_n < _per_suite_baseline_min_n():
        # 2026-07-16 thrash guard: a baseline this sparse cannot certify a hard
        # rollback, however catastrophic the apparent drop (see
        # PER_SUITE_BASELINE_HARD_MIN_N). The caller keeps the drop visible as
        # an advisory warning.
        return False
    if min(result_n, baseline_n) >= PER_SUITE_BINDING_MIN_COUNT:
        return True
    return suite_delta < -PER_SUITE_LOW_SUPPORT_CATASTROPHIC_DROP


ARCHITECT_ROUTING_CAP = 0.80  # Max fraction routed to architect-tier
MAX_CONSECUTIVE_FAILURES = 3  # Auto-rollback after this many failures
# MAD noise filter (intake-421 pi-autoresearch). Quality history depth + significance threshold.
MAD_HISTORY_DEPTH = 10
MAD_MIN_SAMPLES = 3  # Below this, skip MAD check (insufficient data → accept)
MAD_Z_THRESHOLD = 2.0  # Improvement counts as real only if > this many MADs from history median
MAD_CONSISTENCY = 1.4826  # Scaling so MAD ≈ σ under normal distribution

# Quality-history provenance (defect #4 fix). Rolling MAD-window samples are no longer bare
# floats: each carries the era + timestamp + core_id it was measured under so the MAD median
# cannot silently mix an eval-instrument boundary (E7: 79k/41-suite pool + B7 scorer). Legacy
# bare floats decode to era="" and are treated as pre-boundary priors — dropped from a
# post-boundary MAD window when an active eval_quality era is set. When no active era is set
# (the default / all pre-existing tests) the era field is inert and every sample participates,
# so the MAD math is byte-identical to the pre-provenance gate.
_QualityObs = namedtuple("_QualityObs", ["q", "ts", "era", "core_id"])


def _now_iso() -> str:
    """Current UTC timestamp as ISO8601 with a trailing Z (matches journal timestamps)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_quality_obs(entry: Any, *, default_era: str = "") -> "_QualityObs | None":
    """Normalize a persisted quality-history entry to a ``_QualityObs``.

    Accepts a bare float (legacy state → era from ``default_era``, no timestamp) or a
    provenance mapping ``{"q"/"quality", "ts"/"timestamp", "era", "core_id"}``. Returns
    None for junk (non-finite / unparseable) so a corrupt row can never poison the window.
    """
    if isinstance(entry, _QualityObs):
        return entry
    if isinstance(entry, dict):
        raw_q = entry.get("q", entry.get("quality"))
        try:
            q = float(raw_q)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(q):
            return None
        return _QualityObs(
            q=q,
            ts=str(entry.get("ts") or entry.get("timestamp") or ""),
            era=str(entry.get("era") or "") or default_era,
            core_id=str(entry.get("core_id") or ""),
        )
    try:
        q = float(entry)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(q):
        return None
    return _QualityObs(q=q, ts="", era=default_era, core_id="")
# A production-best baseline must never claim a quality the system has never actually
# achieved. Every trustworthy trial that clears the safety gate is recorded on the Pareto
# frontier, so a promotion whose quality exceeds the frontier max is a phantom/contaminated
# measurement (e.g. a T0-saturated or wrong-scale eval) that was never archived. Refusing it
# closes the exact hole that wrote a 2.900 baseline above the 2.400 archive max on 2026-05-31,
# which force-reverted every honest trial and gate-locked the loop into no-op distillation.
BASELINE_ARCHIVE_TOLERANCE = 1e-6  # float-compare slack for the archive-max guard
BASELINE_PROMOTION_REPRO_MIN = 3  # replicated cluster members before baseline ratchet
DEFAULT_BASELINE_QUALITY = 1.16  # Documented 2026-04-04 T2 calibration fallback


def _pareto_archive_for_safety_guard() -> Any | None:
    """Journal-authoritative archive view for baseline safety checks."""
    try:
        from scripts.autopilot.experiment_journal import ExperimentJournal
        from scripts.autopilot.pareto_archive import (
            ParetoArchive,
            pareto_archive_from_journal_rows,
        )

        journal = ExperimentJournal()
        rows = [asdict(entry) for entry in journal.all_entries()]
        if hasattr(journal, "supersession_events"):
            rows.extend(journal.supersession_events())
        archive = pareto_archive_from_journal_rows(
            rows,
            None,
            current_run_only=False,
        )
        return archive if archive is not None else ParetoArchive()
    except Exception as exc:  # noqa: BLE001
        log.warning("Archive-max guard: could not read Pareto frontier (%s)", exc)
        return None


def _pareto_frontier_context(tier: int | None = None) -> tuple[float, frozenset[int]] | None:
    """(best_quality, frozenset of same-tier frontier trial_ids) for the live Pareto archive, or None
    if it is empty/unreadable. Lazy import + fail-soft so a missing archive (fresh bootstrap)
    never blocks a baseline load or write; the caller treats None as "cannot verify → skip"."""
    archive = _pareto_archive_for_safety_guard()
    if archive is None:
        return None
    frontier = archive.frontier(tier=tier)
    if not frontier:
        return None
    best_q = max(e.objectives[0] for e in frontier)
    ids = frozenset(e.trial_id for e in frontier)
    return best_q, ids


def _pareto_frontier_best_quality(tier: int | None = None) -> float | None:
    """Max quality on the live same-tier Pareto frontier, or None if empty/unreadable."""
    ctx = _pareto_frontier_context(tier=tier)
    return ctx[0] if ctx is not None else None


def _normalize_float_by_tier(raw: Any, path: Path) -> dict[int, float]:
    """Normalize JSON/YAML tier keys to int and reject impossible quality values."""
    normalized: dict[int, float] = {}
    for tier, value in (raw or {}).items():
        try:
            t = int(tier)
        except (TypeError, ValueError):
            log.error("Ignoring baseline tier key %r in %s; expected integer tier", tier, path)
            continue
        q = Baseline._validate_quality(value, None, f"baselines_by_tier[{t}]", path)
        if q is not None:
            normalized[t] = q
    return normalized


def _normalize_suite_by_tier(raw: Any, path: Path) -> dict[int, dict[str, float | None]]:
    """Normalize nested per-suite baselines keyed by eval tier."""
    normalized: dict[int, dict[str, float | None]] = {}
    for tier, suites in (raw or {}).items():
        try:
            t = int(tier)
        except (TypeError, ValueError):
            log.error(
                "Ignoring per-suite baseline tier key %r in %s; expected integer tier", tier, path
            )
            continue
        normalized[t] = {
            suite: Baseline._validate_quality(
                q, None, f"per_suite_quality_by_tier[{t}][{suite}]", path
            )
            for suite, q in (suites or {}).items()
        }
    return normalized


def _normalize_counts_by_tier(raw: Any, path: Path) -> dict[int, dict[str, int]]:
    """Normalize nested per-suite question counts keyed by eval tier (ints >= 0)."""
    normalized: dict[int, dict[str, int]] = {}
    for tier, suites in (raw or {}).items():
        try:
            t = int(tier)
        except (TypeError, ValueError):
            log.error(
                "Ignoring per-suite count tier key %r in %s; expected integer tier", tier, path
            )
            continue
        tier_counts: dict[str, int] = {}
        for suite, n in (suites or {}).items():
            try:
                c = int(n)
            except (TypeError, ValueError):
                continue
            if c >= 0:
                tier_counts[suite] = c
        normalized[t] = tier_counts
    return normalized


def _normalize_tier_revisions(raw: Any, path: Path) -> dict[int, int]:
    """Normalize persisted per-tier baseline revisions (monotonic ints >= 0).

    EV-14c: the revision is the moved-reference detector — a measurement window pins
    the tier revision before it starts and compares after, so a re-score that moves
    the baseline mid-window is detected instead of silently collapsed over. A corrupt
    revision (negative / non-numeric) is dropped loudly: restoring one would pin a
    reference identity that cannot be compared.
    """
    normalized: dict[int, int] = {}
    for tier, revision in (raw or {}).items():
        try:
            t = int(tier)
        except (TypeError, ValueError):
            log.error(
                "Ignoring baseline revision tier key %r in %s; expected integer tier",
                tier,
                path,
            )
            continue
        try:
            r = int(revision)
        except (TypeError, ValueError):
            log.error(
                "Ignoring corrupt baseline revision %r for T%s in %s; expected integer",
                revision,
                t,
                path,
            )
            continue
        if r >= 0:
            normalized[t] = r
    return normalized


def _fmt_metric(value: Any, spec: str) -> str:
    """D4 / FIELD-1 (MET-1): format a numeric METRIC value for ``to_grep_lines``.

    Emits the literal ``null`` for an UNAVAILABLE value — ``None`` or a non-finite
    float (NaN / inf) — instead of the bare string ``nan`` (indistinguishable from a
    real token, and coerced to 0 by awk). A real zero still formats normally
    (``0``/``0.0000``). For any finite value the output is byte-identical to the
    pre-v2 ``format(value, spec)``, so this only ever changes the absence sentinel.
    """
    if value is None:
        return "null"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return "null"
    if not math.isfinite(fv):
        return "null"
    return format(value, spec)


def _san_metric_name(name: Any) -> str:
    """MET-2: neutralize a value interpolated into a METRIC key/value.

    Suite / species / role names are embedded directly in ``METRIC <key>: <value>``
    lines. A stray ``:`` or whitespace in the name would break ``awk -F': '`` field
    splitting (and the blessed metric_lines parser splits on the first ``': '``), so
    collapse any run of ``:``/whitespace to a single ``_``. A clean name (the common
    case, e.g. ``coder``) is returned unchanged."""
    return re.sub(r"[:\s]+", "_", str(name))


@dataclass
class SafetyVerdict:
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)  # AP-14: deficiency categories
    # LEDGER-W4: the anytime-valid sequential e-process journal block for this trial
    # (E_quality, E_rate_noninf, k, z, joint state), populated only when the default-off
    # AUTOPILOT_SEQ_VERDICT path runs and the caller supplies per-question results.
    # None preserves the legacy verdict shape for every existing caller/fixture.
    seq: dict[str, Any] | None = None
    # B1 / REL-1: True when reliability fell below the floor and the quality-floor /
    # regression / per-suite checks were SUPPRESSED (evidence untrustworthy). The trial
    # failed, but it signals RETRY, not a revert — callers use this to avoid treating an
    # infra-error trial as a quality regression, and the gate skips the consecutive-failure
    # increment for it. Defaults False so every existing verdict is behavior-identical.
    reliability_blocked: bool = False

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class EvalResult:
    """Evaluation result from EvalTower."""

    tier: int
    quality: float  # Average quality 0-3
    speed: (
        float  # Objective speed t/s: median request in serial, aggregate batch in concurrent evals.
    )
    cost: float  # Normalized cost 0-1
    reliability: float  # Fraction of non-error responses
    per_suite_quality: dict[str, float] = field(default_factory=dict)
    # Per-suite question counts for this eval (2026-06-06). Lets the per-suite
    # regression gate scale its threshold to sampling resolution (3/n per flip):
    # at ~2 questions/suite the score quantizes to {0,1.5,3} and a one-question
    # flip would otherwise trip the fixed -0.1 gate every trial. Empty ⇒ gate
    # falls back to the fixed -0.1 floor (pre-2026-06-06 behavior).
    per_suite_counts: dict[str, int] = field(default_factory=dict)
    routing_distribution: dict[str, float] = field(default_factory=dict)
    n_questions: int = 0
    # Per-question paired-design ledger vector, journaled in JSONL only. Each
    # item is compact: {qid, suite, correct, latency_ms, tools_used}.
    question_results: list[dict[str, Any]] = field(default_factory=list)
    core_id: str = ""  # Versioned paired-core identity for instrument-era tracking.
    details: dict[str, Any] = field(default_factory=dict)
    # ── INFRA-FAILED disposition rollup (2026-08-03 incident) ────────────
    # A question whose endpoint was unreachable, refused the request (HTTP
    # 4xx), timed out, or returned nothing produced NO MEASUREMENT. Such rows
    # are excluded from `quality`'s denominator, but the exclusion itself must
    # be visible: a run where 70/100 rows infra-failed and a run where the
    # model answered 70 questions wrong are different facts about the world and
    # were previously reported with the same numbers.
    #
    # quality_measured is FALSE when nothing was scored at all. `quality` is a
    # float on the Pareto/SafetyGate contract and cannot be None, so the 0.0 it
    # carries in that case is a PLACEHOLDER — consumers that treat quality as a
    # measurement MUST check this flag first. Defaults keep every existing
    # construction site behaviour-identical.
    infra_failed_count: int = 0
    scoring_failed_count: int = 0
    infra_failed_reasons: dict[str, int] = field(default_factory=dict)
    quality_measured: bool = True
    quality_unmeasured_reason: str = ""
    # HLE-4 observe-only metrics. The authoritative record schema lives in
    # src/trace/harness_schema.py; these fields only carry per-trial payloads
    # through EvalTower -> journal before any Pareto promotion is allowed.
    metric_schema_version: int = 1
    harness_metrics: dict[str, Any] = field(default_factory=dict)
    oracle_adequacy: dict[str, Any] = field(default_factory=dict)
    median_request_speed: float = 0.0  # Raw median per-request tokens/sec.
    aggregate_speed: float = 0.0  # Batch-level tokens/sec over eval wall time.
    eval_concurrency: int = 1  # Worker fan-out used by EvalTower for this batch.
    eval_wall_s: float = 0.0  # End-to-end EvalTower batch wall time.
    sum_request_elapsed_s: float = 0.0  # Sum of per-request elapsed times.
    speed_metric_mode: str = "median_request_tps"  # median_request_tps or aggregate_batch_tps.
    instruction_token_count: int = 0  # AP-16: per-request instruction overhead
    instruction_token_ratio: float = 0.0  # AP-16: instruction_tokens / total_input_tokens
    partial_count: int = 0  # Inference results with partial=True (read_timeout_partial)
    degraded_count: int = 0  # Inference results with degraded=True
    # Tool-use telemetry (2026-06-01). Trial-level rollup of per-question tool
    # invocations so the autopilot can measure — and learn to incentivize — model
    # tool use. tokens_generated/speed already credit tool-turn generation; these
    # surface the tool activity itself as an explicit, planner-visible signal.
    mean_tools_used: float = 0.0  # Mean tool invocations per (non-error) question.
    tool_use_rate: float = 0.0  # Fraction of questions that invoked >=1 tool.
    total_tool_calls: int = 0  # Total tool invocations across the trial.
    # Conditional credit, NOT a Pareto objective: marginal usefulness of tools =
    # P(correct | tools used) - P(correct | no tools). NaN until both arms have a
    # minimum sample. Used as a planner prior to steer tool-enabling experiments,
    # scored only by downstream quality/reliability movement — never optimized raw.
    tool_helpfulness: float = float("nan")
    # Per-suite marginal usefulness {suite: P(correct|tool) − P(correct|no tool)},
    # computed within-suite so cross-suite difficulty can't contaminate the signal.
    # The scalar tool_helpfulness above is the mean of these. Planner prior only.
    per_suite_tool_helpfulness: dict = field(default_factory=dict)
    # AM KV compaction telemetry (populated when compact action is used)
    avg_prompt_tokens: float = 0.0  # Average context length across results
    compaction_events: int = 0  # Number of compacted slots in this eval
    # EV-8: Diversity metrics (NIB2-42). All default to NaN ("unavailable"),
    # not zero, so SafetyGate NaN-guards cannot fire on missing signal.
    # Populated by diversity_metrics.compute_diversity() after each eval batch.
    # diversity_semantic_embedding_agreement is inference-gated: it remains
    # NaN unless an embedder is wired in at eval time.
    diversity_entropy: float = math.nan
    diversity_distinct2: float = math.nan
    diversity_self_bleu: float = math.nan
    diversity_ttr: float = math.nan
    diversity_semantic_embedding_agreement: float = math.nan
    # EV-9 / MindDR rubric telemetry. NaN means unavailable; these are emitted
    # only when a rubric scorer actually populates them.
    rubric_reasoning_trajectory: float = math.nan
    rubric_tool_calls: float = math.nan
    rubric_outline: float = math.nan
    rubric_content_stage: float = math.nan
    # EV-2: Calibration metrics (from eval-tower-verification.md)
    ece: float = 0.0  # Expected Calibration Error (10-bin). Lower = better calibrated.
    auroc: float = 0.0  # Area Under ROC Curve. Higher = better discrimination. 0 if degenerate.
    calibration_violations: int = 0  # Questions where |confidence - correctness| > 0.5
    # Branching density: fraction of reasoning steps that are divergent/exploratory.
    # From intake-378 deep-dive: high branching (>0.30) = unproductive exploration.
    # 0.0 when no <think> blocks are present in eval answers.
    branching_density: float = 0.0
    # 2026-05-23 exogenous-restart resilience (handoff Phase 4).
    # n_exogenous_recovered: questions whose initial /chat raised but the
    #   retry-after-wait succeeded. Audit-only signal — the trial is still
    #   sound.
    # n_exogenous_unrecovered: questions whose initial /chat raised, a
    #   service reload was detected, the wait+retry FAILED. Trial gets
    #   tagged bug_corrupted_by="exogenous_operator_reload" in Phase 5.
    # n_external_restart: subset of recovered+unrecovered whose marker
    #   source was != stack_commands. Surfaced for audit.
    # exogenous_question_ids: per-question audit trail.
    # exogenous_marker_log: each retry's marker_changes dict, in order.
    n_exogenous_recovered: int = 0
    n_exogenous_unrecovered: int = 0
    n_external_restart: int = 0
    exogenous_question_ids: list[str] = field(default_factory=list)
    exogenous_marker_log: list[dict] = field(default_factory=list)
    # AP-4: reviewer-calibration Pareto axes (reviewer-control-plane H8). Optional
    # quality axes carried alongside task quality/throughput. All default to NaN
    # ("unavailable this trial") so an EvalResult produced WITHOUT a reviewer in
    # the loop still leaves the objectives 4-tuple below unchanged and SafetyGate/
    # Pareto never see these. (D4: to_grep_lines now emits them UNCONDITIONALLY as
    # the literal ``null`` when NaN — the absence is explicit, not a dropped line.)
    # They become live axes only once a review_policy_trial or a
    # shadow reviewer actually populates them. reviewer_fa_rate = P(reviewer
    # approved | gate FAIL) = false-accept; reviewer_fr_rate = P(reviewer rejected
    # | gate PASS) = false-reject; ratio + per-decision latency feed H-LB.
    reviewer_fa_rate: float = math.nan
    reviewer_fr_rate: float = math.nan
    reviewer_fa_fr_ratio: float = math.nan
    review_decision_latency_ms: float = math.nan
    # SafetyGate.check() is called by several action handlers and again by the
    # main loop. Cache the first verdict on the result so one trial mutates MAD
    # history / consecutive-failure state exactly once.
    gate_verdict: SafetyVerdict | None = field(default=None, repr=False, compare=False)

    @property
    def objectives(self) -> tuple[float, float, float, float]:
        return (self.quality, self.speed, -self.cost, self.reliability)

    def to_grep_lines(self, trial_id: int = 0, species: str = "") -> str:
        """AP-13 / D4 (audit MET-1, MET-2, FIELD-1): grep-parseable METRIC lines.

        CONTRACT v2 (2026-07-20): the first line is ``METRIC schema_version: 2``.
        Every key below is emitted UNCONDITIONALLY — an unavailable / NaN value emits
        the literal ``null`` (never the string ``nan``, never a silently dropped line);
        a real zero emits ``0``/``0.0000``. This kills the old absence-vs-zero
        ambiguity where a NaN-gated key was simply omitted, so a consumer could not
        tell "not measured" from "measured zero". Interpolated names (species / suite /
        role) are sanitized so a stray ``:``/whitespace cannot break ``awk -F': '``.

        Consumers MUST parse with scripts.autopilot.metric_lines.parse_metric_lines —
        NOT ad-hoc grep/awk (that is the whole point of the versioned contract).
        """
        _fmt = _fmt_metric
        lines = [
            # D4 (MET-1): explicit contract version; v1 was the implicit, pre-null format.
            f"METRIC schema_version: {METRIC_LINE_SCHEMA_VERSION}",
            f"METRIC trial: {trial_id}",
            f"METRIC species: {_san_metric_name(species)}",
            f"METRIC tier: {self.tier}",
            f"METRIC quality: {_fmt(self.quality, '.4f')}",
            f"METRIC speed: {_fmt(self.speed, '.2f')}",
            f"METRIC speed_metric_mode: {self.speed_metric_mode}",
            f"METRIC median_request_speed: {_fmt(self.median_request_speed, '.2f')}",
            f"METRIC aggregate_speed: {_fmt(self.aggregate_speed, '.2f')}",
            f"METRIC eval_concurrency: {self.eval_concurrency}",
            f"METRIC eval_wall_s: {_fmt(self.eval_wall_s, '.2f')}",
            f"METRIC cost: {_fmt(self.cost, '.4f')}",
            f"METRIC reliability: {_fmt(self.reliability, '.4f')}",
            f"METRIC n_questions: {self.n_questions}",
        ]
        if self.core_id:
            lines.append(f"METRIC core_id: {_san_metric_name(self.core_id)}")
        for suite, q in sorted(self.per_suite_quality.items()):
            lines.append(f"METRIC suite_{_san_metric_name(suite)}: {_fmt(q, '.4f')}")
        for role, frac in sorted(self.routing_distribution.items()):
            lines.append(f"METRIC route_{_san_metric_name(role)}: {_fmt(frac, '.4f')}")
        # AP-16: Instruction token budget
        lines.append(f"METRIC instruction_tokens: {self.instruction_token_count}")
        lines.append(f"METRIC instruction_ratio: {_fmt(self.instruction_token_ratio, '.4f')}")
        # D4 (FIELD-1): degradation counts now emit UNCONDITIONALLY. A real 0 ("clean,
        # no partials") is distinct from an absent line ("never measured").
        lines.append(f"METRIC partial_count: {self.partial_count}")
        lines.append(f"METRIC degraded_count: {self.degraded_count}")
        # EV-2: Calibration metrics. ece/auroc emit ``null`` when non-finite (were
        # 'nan' / dropped); calibration_violations is an always-present count.
        lines.append(f"METRIC ece: {_fmt(self.ece, '.4f')}")
        lines.append(f"METRIC auroc: {_fmt(self.auroc, '.4f')}")
        lines.append(f"METRIC calibration_violations: {self.calibration_violations}")
        # AP-27: report-only RLVR reward view. This is deliberately log-only;
        # EvalResult.objectives, SafetyGate verdicts, Pareto archive state, and
        # journal schema remain unchanged.
        rlvr = rlvr_reward_from_result(self)
        lines.append(f"METRIC rlvr_policy: {rlvr.policy}")
        lines.append(f"METRIC rlvr_signal: {rlvr.reward_signal}")
        lines.append(f"METRIC rlvr_reward: {_fmt(rlvr.reward, '.6f')}")
        lines.append(f"METRIC rlvr_ready: {int(rlvr.ready_for_training)}")
        if rlvr.blockers:
            # Bounded, variable-length list; only meaningful when non-empty (a
            # comma-joined string value, not a numeric metric) — stays conditional.
            lines.append(f"METRIC rlvr_blockers: {','.join(rlvr.blockers)}")
        # Branching density (intake-378) + AM compaction telemetry — unconditional now.
        lines.append(f"METRIC branching_density: {_fmt(self.branching_density, '.4f')}")
        lines.append(f"METRIC avg_prompt_tokens: {_fmt(self.avg_prompt_tokens, '.0f')}")
        lines.append(f"METRIC compaction_events: {self.compaction_events}")
        # D6 (FIELD-1 partial): tool-use telemetry block. The field comments have
        # promised this since 2026-06-01 but it was never emitted; wire it in now,
        # null-gated like the rest.
        lines.append(f"METRIC mean_tools_used: {_fmt(self.mean_tools_used, '.4f')}")
        lines.append(f"METRIC tool_use_rate: {_fmt(self.tool_use_rate, '.4f')}")
        lines.append(f"METRIC total_tool_calls: {self.total_tool_calls}")
        lines.append(f"METRIC tool_helpfulness: {_fmt(self.tool_helpfulness, '.4f')}")
        # per_suite_tool_helpfulness is a dict, NOT a scalar: emit one bounded
        # ``tool_helpfulness[<suite>]`` line per POPULATED (finite) suite, and skip
        # the scalar form of the dict entirely (unavailable per-suite entries are
        # simply absent from the map rather than emitted as null — bounded).
        psth = self.per_suite_tool_helpfulness
        if isinstance(psth, dict):
            for _suite, _val in sorted(psth.items()):
                try:
                    _fv = float(_val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(_fv):
                    lines.append(
                        f"METRIC tool_helpfulness[{_san_metric_name(_suite)}]: {_fv:.4f}"
                    )
        # EV-8: Diversity metrics — unconditional; NaN → ``null``.
        for _div_key, _div_val in (
            ("diversity_entropy", self.diversity_entropy),
            ("diversity_distinct2", self.diversity_distinct2),
            ("diversity_self_bleu", self.diversity_self_bleu),
            ("diversity_ttr", self.diversity_ttr),
            ("diversity_semantic_embedding_agreement", self.diversity_semantic_embedding_agreement),
        ):
            lines.append(f"METRIC {_div_key}: {_fmt(_div_val, '.4f')}")
        # EV-9 / MindDR rubric telemetry — unconditional; NaN → ``null``.
        for _rubric_key, _rubric_val in (
            ("rubric_reasoning_trajectory", self.rubric_reasoning_trajectory),
            ("rubric_tool_calls", self.rubric_tool_calls),
            ("rubric_outline", self.rubric_outline),
            ("rubric_content_stage", self.rubric_content_stage),
        ):
            lines.append(f"METRIC {_rubric_key}: {_fmt(_rubric_val, '.4f')}")
        # AP-4: reviewer-calibration axes — unconditional; NaN → ``null`` (was dropped).
        for _rev_key, _rev_val in (
            ("reviewer_fa_rate", self.reviewer_fa_rate),
            ("reviewer_fr_rate", self.reviewer_fr_rate),
            ("reviewer_fa_fr_ratio", self.reviewer_fa_fr_ratio),
            ("review_decision_latency_ms", self.review_decision_latency_ms),
        ):
            lines.append(f"METRIC {_rev_key}: {_fmt(_rev_val, '.4f')}")
        return "\n".join(lines)


@dataclass(frozen=True)
class BaselinePin:
    """Durable identity of a baseline reference a measurement window pins against.

    EV-14c: captured BEFORE a window starts (an EV-14a band run pins the reference
    before its first repeat; ``update_baseline`` pins it at promotion entry) via
    ``Baseline.pin_tier()``. ``Baseline.pin_moved(pin)`` reports whether the tier
    reference changed since the pin — so a band measured against a reference that
    was re-scored mid-window is DETECTED as moved, never silently read as stable.
    """

    tier: int
    quality: float | None
    per_suite_quality: dict[str, float | None]
    per_suite_counts: dict[str, int]
    revision: int
    eval_quality_era: str = ""


@dataclass
class Baseline:
    quality: float = DEFAULT_BASELINE_QUALITY
    speed: float = 10.0
    cost: float = 0.5
    reliability: float = 0.9
    per_suite_quality: dict[str, float] = field(default_factory=dict)
    baselines_by_tier: dict[int, float] = field(default_factory=dict)
    per_suite_quality_by_tier: dict[int, dict[str, float | None]] = field(default_factory=dict)
    # Per-suite question counts the baseline was measured at, per tier (2026-06-06).
    # Feeds per_suite_regression_threshold so the gate knows the baseline's own
    # sampling resolution. Empty until a baseline is refreshed post-2026-06-06; the
    # gate then falls back to the result's resolution (or the fixed -0.1 floor).
    per_suite_counts_by_tier: dict[int, dict[str, int]] = field(default_factory=dict)
    frontdoor_speed: float = 10.0
    # Eval-instrument era this baseline was captured under (defect #1/#4 fix). Empty on a
    # legacy baseline (pre-provenance state) — which the gate treats as a PRE-E7 stamp, so a
    # legacy baseline vs the active E7 era trips the re-baseline hold. Set whenever a
    # promotion lands under a known active era, and by an operator reseed.
    eval_quality_era: str = ""
    # Speed-instrument era this baseline's ``frontdoor_speed`` was captured under. Exact
    # analogue of ``eval_quality_era`` on the THROUGHPUT axis (2026-08-03). Before this
    # existed the gate carried no speed provenance at all: the throughput floor
    # (0.8 * frontdoor_speed) could be derived from a pre-v8 measurement and charged
    # against a post-v8 trial with nothing recorded to detect it. Empty on a legacy
    # baseline (pre-provenance state), which the gate treats as a pre-boundary stamp, so a
    # legacy baseline vs an active autopilot_speed era trips the throughput re-baseline
    # hold. Stamped by update_baseline() whenever a promotion actually refreshes
    # frontdoor_speed, and by an operator reseed.
    autopilot_speed_era: str = ""
    # Path this baseline was loaded from. save() writes back here by default so a
    # gate constructed with a custom baseline_path (e.g. a tmp file in tests) can
    # NEVER clobber the production orchestration/autopilot_baseline.yaml. Excluded
    # from equality so two baselines with the same metrics still compare equal.
    # (2026-05-31: a test fixture's update_baseline() wrote quality=2.9 to the real
    #  baseline via the DEFAULT_BASELINE_PATH fallback, gate-locking the live loop.)
    source_path: Path | None = field(default=None, compare=False, repr=False)
    # EV-14c: per-tier monotonic revision of the tier baseline reference. Bumped by
    # update_tier() on every write that actually changes the reference identity (tier
    # quality, a per-suite entry, or a per-suite count). A measurement window
    # (e.g. an EV-14a band run) pins the reference before it starts via pin_tier()
    # and detects a moved reference afterwards via pin_moved() — a re-score can no
    # longer move the baseline silently under a running window. Persisted through
    # load/save/apply_state/to_state_dict so a restart cannot hide a move.
    tier_revisions: dict[int, int] = field(default_factory=dict)
    # Live measurement-window pins registered by pin_tier(), so update_tier can NAME
    # the windows its write invalidates. Never persisted: a pin is a per-process
    # window, and a persisted pin would outlive the process that measured against it.
    _pins: dict[str, BaselinePin] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path | None = None, state: dict[str, Any] | None = None) -> Baseline:
        path = path or DEFAULT_BASELINE_PATH
        if not path.exists():
            log.warning("No baseline file at %s, using defaults", path)
            baseline = cls(source_path=path)
            if state:
                baseline.apply_state(state, path)
            return baseline
        try:
            data = yaml.safe_load(path.read_text())
        except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
            raise BaselineCorruptError(
                f"Baseline file {path} is unreadable/unparseable ({exc}). "
                f"{_BASELINE_REMEDIATION}"
            ) from exc
        if data is None or not isinstance(data, dict):
            raise BaselineCorruptError(
                f"Baseline file {path} is an empty or non-mapping baseline file. "
                f"{_BASELINE_REMEDIATION}"
            )
        defaults = cls()
        quality = cls._validate_quality(
            data.get("quality", defaults.quality), defaults.quality, "quality", path
        )
        # Above-archive-max guard for the LOAD path (defense-in-depth). The scale guard
        # above only catches values outside [0, QUALITY_MAX] — it passes a 2.900 baseline,
        # which is within scale yet still unachievable when the Pareto frontier max is 2.400.
        # A persisted baseline strictly above the frontier max force-reverts every honest
        # trial and gate-locks the loop into no-op distillation (2026-05-31). The write-side
        # update_baseline() guard cannot help here because a corrupt file is loaded directly.
        # Fall back to the default floor and log loudly; the operator must recompute. Skipped
        # when the archive is empty/unreadable (fresh start) so bootstrap is never blocked.
        archive_max = _pareto_frontier_best_quality(DEFAULT_FRONTIER_TIER)
        if (
            quality is not None
            and archive_max is not None
            and quality > archive_max + BASELINE_ARCHIVE_TOLERANCE
        ):
            log.error(
                "Persisted baseline quality %.3f in %s exceeds Pareto archive max %.3f — "
                "unachievable/corrupt; it would gate-lock the loop. Falling back to %.3f. "
                "Recompute the baseline from a real eval (autopilot.py checkpoint --production-best).",
                quality,
                path,
                archive_max,
                defaults.quality,
            )
            quality = defaults.quality
        per_suite = {
            suite: cls._validate_quality(q, None, f"per_suite[{suite}]", path)
            for suite, q in (data.get("per_suite_quality", {}) or {}).items()
        }
        baselines_by_tier = _normalize_float_by_tier(data.get("baselines_by_tier", {}), path)
        _drop_over_archive_max_tiers(baselines_by_tier, path)
        per_suite_by_tier = _normalize_suite_by_tier(
            data.get("per_suite_quality_by_tier", {}), path
        )
        per_suite_counts_by_tier = _normalize_counts_by_tier(
            data.get("per_suite_counts_by_tier", {}), path
        )
        tier_revisions = _normalize_tier_revisions(data.get("tier_revisions", {}), path)
        reliability = data.get("reliability", defaults.reliability)
        if reliability is not None and (
            isinstance(reliability, bool)
            or not isinstance(reliability, (int, float))
            or not 0.0 <= reliability <= RELIABILITY_MAX
        ):
            log.error(
                "Corrupt baseline reliability %r in %s (valid 0..%.1f); using default %.3f",
                reliability,
                path,
                RELIABILITY_MAX,
                defaults.reliability,
            )
            reliability = defaults.reliability
        # speed/frontdoor_speed must be finite and > 0: a null/negative frontdoor_speed
        # either raises a TypeError inside check() (None * 0.8) or silently disarms the
        # throughput floor (result.speed < negative is never true). cost may be 0.
        speed = cls._validate_positive_float(
            data.get("speed", defaults.speed), defaults.speed, "speed", path
        )
        frontdoor_speed = cls._validate_positive_float(
            data.get("frontdoor_speed", defaults.frontdoor_speed),
            defaults.frontdoor_speed,
            "frontdoor_speed",
            path,
        )
        cost = cls._validate_positive_float(
            data.get("cost", defaults.cost),
            defaults.cost,
            "cost",
            path,
            allow_zero=True,
        )
        baseline = cls(
            quality=quality,
            speed=speed,
            cost=cost,
            reliability=reliability,
            per_suite_quality=per_suite,
            baselines_by_tier=baselines_by_tier,
            per_suite_quality_by_tier=per_suite_by_tier,
            per_suite_counts_by_tier=per_suite_counts_by_tier,
            tier_revisions=tier_revisions,
            frontdoor_speed=frontdoor_speed,
            source_path=path,
        )
        if state:
            baseline.apply_state(state, path)
        return baseline

    @staticmethod
    def _validate_quality(
        value: float | None, fallback: float | None, label: str, path: Path
    ) -> float | None:
        """Reject an out-of-[0, QUALITY_MAX] quality value (corrupt/wrong-scale baseline).

        Returns `fallback` when the value is impossible, after logging loudly. A None
        value (per-suite "not yet populated") passes through unchanged so the per-suite
        regression gate stays disabled for that suite.
        """
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            log.error(
                "Corrupt baseline %s=%r in %s is not a number — the regression gate "
                "cannot compare it; falling back to %s. Recompute from a real 0-3 eval.",
                label,
                value,
                path,
                fallback,
            )
            return fallback
        if not 0.0 <= value <= QUALITY_MAX:
            log.error(
                "Corrupt baseline %s=%.3f in %s exceeds valid quality scale [0, %.1f] — "
                "this would force-revert every trial via the regression gate; "
                "falling back to %s. Recompute the baseline from a real 0-3 eval.",
                label,
                value,
                path,
                QUALITY_MAX,
                fallback,
            )
            return fallback
        return value

    @staticmethod
    def _validate_positive_float(
        value: Any,
        fallback: float,
        label: str,
        path: Path,
        *,
        allow_zero: bool = False,
    ) -> float:
        """Reject a non-numeric / non-finite / non-positive baseline scalar.

        A null/None, bool, non-numeric string, NaN/inf, or <= 0 value (< 0 when
        ``allow_zero``) would later raise a TypeError inside check() or silently
        disarm a floor (e.g. a negative ``frontdoor_speed``). Log loudly and fall
        back to the safe default; mirrors the ``_validate_quality`` log style.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            log.error(
                "Corrupt baseline %s=%r in %s is not a number; using default %s.",
                label,
                value,
                path,
                fallback,
            )
            return fallback
        fval = float(value)
        if not math.isfinite(fval):
            log.error(
                "Corrupt baseline %s=%r in %s is non-finite; using default %s.",
                label,
                value,
                path,
                fallback,
            )
            return fallback
        below = fval < 0.0 if allow_zero else fval <= 0.0
        if below:
            log.error(
                "Corrupt baseline %s=%.3f in %s must be %s; using default %s.",
                label,
                fval,
                path,
                ">= 0" if allow_zero else "> 0",
                fallback,
            )
            return fallback
        return fval

    def save(self, path: Path | None = None) -> None:
        path = path or self.source_path or DEFAULT_BASELINE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "quality": self.quality,
            "speed": self.speed,
            "cost": self.cost,
            "reliability": self.reliability,
            "per_suite_quality": self.per_suite_quality,
            "baselines_by_tier": self.baselines_by_tier,
            "per_suite_quality_by_tier": self.per_suite_quality_by_tier,
            "per_suite_counts_by_tier": self.per_suite_counts_by_tier,
            "tier_revisions": self.tier_revisions,
            "frontdoor_speed": self.frontdoor_speed,
        }
        _atomic_write_text(
            path, yaml.dump(data, default_flow_style=False, allow_unicode=True)
        )

    def apply_state(self, state: dict[str, Any], path: Path | None = None) -> None:
        state_path = path or self.source_path or DEFAULT_BASELINE_PATH
        # ``baseline_state`` is the live authority while present (see
        # ``_baseline_state_for_startup_gate``). Historically this method only
        # applied tier maps and era stamps, silently retaining the frozen YAML's
        # top-level scalars. An operator reseed could therefore persist a complete
        # state payload and still start with stale speed/reliability floors. Apply
        # every field emitted by ``to_state_dict``.
        if "quality" in state:
            quality = self._validate_quality(
                state.get("quality"), self.quality, "state.quality", state_path
            )
            archive_max = _pareto_frontier_best_quality(DEFAULT_FRONTIER_TIER)
            if (
                quality is not None
                and archive_max is not None
                and quality > archive_max + BASELINE_ARCHIVE_TOLERANCE
            ):
                log.error(
                    "State baseline quality %.3f in %s exceeds Pareto archive max %.3f; "
                    "retaining %.3f.",
                    quality,
                    state_path,
                    archive_max,
                    self.quality,
                )
            elif quality is not None:
                self.quality = quality
        if "speed" in state:
            self.speed = self._validate_positive_float(
                state.get("speed"), self.speed, "state.speed", state_path
            )
        if "cost" in state:
            self.cost = self._validate_positive_float(
                state.get("cost"),
                self.cost,
                "state.cost",
                state_path,
                allow_zero=True,
            )
        if "reliability" in state:
            reliability = state.get("reliability")
            if (
                isinstance(reliability, bool)
                or not isinstance(reliability, (int, float))
                or not 0.0 <= reliability <= RELIABILITY_MAX
            ):
                log.error(
                    "Corrupt state baseline reliability %r in %s; retaining %.3f.",
                    reliability,
                    state_path,
                    self.reliability,
                )
            else:
                self.reliability = float(reliability)
        if "frontdoor_speed" in state:
            self.frontdoor_speed = self._validate_positive_float(
                state.get("frontdoor_speed"),
                self.frontdoor_speed,
                "state.frontdoor_speed",
                state_path,
            )
        if "per_suite_quality" in state:
            self.per_suite_quality = {
                suite: self._validate_quality(
                    value,
                    None,
                    f"state.per_suite_quality[{suite}]",
                    state_path,
                )
                for suite, value in (state.get("per_suite_quality") or {}).items()
            }
        self.baselines_by_tier.update(
            _normalize_float_by_tier(state.get("baselines_by_tier", {}), state_path)
        )
        # B3 / SG-4: apply the same above-archive-max guard load() uses. A state dict
        # (autopilot_state.json) can carry a corrupt/wrong-scale tier baseline just as a
        # YAML file can; without this, an over-max value applied here would gate-lock the
        # loop exactly as the 2026-05-31 file-side value did.
        _drop_over_archive_max_tiers(self.baselines_by_tier, state_path)
        self.per_suite_quality_by_tier.update(
            _normalize_suite_by_tier(state.get("per_suite_quality_by_tier", {}), state_path)
        )
        self.per_suite_counts_by_tier.update(
            _normalize_counts_by_tier(state.get("per_suite_counts_by_tier", {}), state_path)
        )
        self.tier_revisions.update(
            _normalize_tier_revisions(state.get("tier_revisions", {}), state_path)
        )
        era = state.get("eval_quality_era")
        if isinstance(era, str) and era.strip():
            self.eval_quality_era = era.strip()
        speed_era = state.get("autopilot_speed_era")
        if isinstance(speed_era, str) and speed_era.strip():
            self.autopilot_speed_era = speed_era.strip()

    def quality_for_tier(self, tier: int, *, strict: bool = False) -> float | None:
        """Return same-tier quality baseline; lenient mode falls back to legacy `quality`."""
        tier_quality = self.baselines_by_tier.get(int(tier))
        if tier_quality is not None:
            return tier_quality
        return None if strict else self.quality

    def per_suite_for_tier(self, tier: int, *, strict: bool = False) -> dict[str, float | None]:
        """Return same-tier per-suite baselines; lenient mode falls back to legacy suite values."""
        tier_suites = self.per_suite_quality_by_tier.get(int(tier))
        if tier_suites is not None:
            return tier_suites
        return {} if strict else self.per_suite_quality

    def per_suite_counts_for_tier(self, tier: int) -> dict[str, int]:
        """Return the per-suite question counts the same-tier baseline was measured at.

        Empty when the baseline predates per-suite count tracking (2026-06-06) —
        callers then fall back to the result's own resolution / the fixed floor."""
        return self.per_suite_counts_by_tier.get(int(tier), {})

    def tier_revision(self, tier: int) -> int:
        """Current revision of the T<tier> baseline reference (0 before any write)."""
        return self.tier_revisions.get(int(tier), 0)

    def pin_tier(self, tier: int, pin_id: str | None = None, *, register: bool = True) -> BaselinePin:
        """Capture the current T<tier> reference identity for a measurement window.

        EV-14c: call BEFORE the window starts (an EV-14a band run pins the reference
        before its first repeat). The returned pin records the tier revision, so any
        baseline write during the window is DETECTABLE via ``pin_moved()`` instead of
        being silently collapsed over. With ``register=True`` (the default) the pin is
        kept so ``update_tier()`` can NAME the windows its write invalidates; a
        window that IS the writer itself (``update_baseline``'s read-to-write span)
        passes ``register=False`` so its own write is not reported against it.
        """
        tier = int(tier)
        pin = BaselinePin(
            tier=tier,
            quality=self.quality_for_tier(tier, strict=True),
            per_suite_quality=dict(self.per_suite_quality_by_tier.get(tier, {})),
            per_suite_counts=dict(self.per_suite_counts_by_tier.get(tier, {})),
            revision=self.tier_revision(tier),
            eval_quality_era=self.eval_quality_era,
        )
        if register:
            key = pin_id or f"t{tier}-r{pin.revision}"
            self._pins.setdefault(key, pin)
        return pin

    def pin_moved(self, pin: BaselinePin) -> bool:
        """True when the T<tier> reference changed since ``pin`` was captured.

        A band measured against a moved reference is INVALID — never "no change".
        This is the detector that makes a silently-moved baseline impossible
        (EV-14c): callers that cannot tolerate the move refuse or re-measure.
        """
        return self.tier_revision(pin.tier) != pin.revision

    def pins_for_tier(self, tier: int) -> tuple[BaselinePin, ...]:
        """Registered measurement-window pins on one tier, oldest first."""
        tier = int(tier)
        return tuple(pin for pin in self._pins.values() if pin.tier == tier)

    @staticmethod
    def _fmt_quality(value: Any) -> str:
        """Render a quality reference for the moved-reference log line (None as absent)."""
        if value is None:
            return "absent"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    def update_tier(self, result: EvalResult) -> None:
        """Rewrite the T<tier> baseline reference — with an explicit moved-reference record.

        EV-14c (defect, not enhancement): the pre-fix write was ``dict.update``
        semantics — a re-score of the same suite silently overwrote the prior baseline
        with no record that one existed, so a measurement window (e.g. an EV-14a band
        run) pinned against the old reference could not detect that its reference had
        been moved mid-window. This is the collapse point that gates decisions. Now:

        * every write that actually changes the reference identity (tier quality, a
          per-suite entry, or a per-suite count) bumps the tier revision — the
          reference is never "same" after a move;
        * the move is logged explicitly (BASELINE MOVED: prior -> new) — never silent;
        * every REGISTERED measurement pin the write invalidates is named in that log,
          so the windows affected are visible at write time, and
          ``pin_moved(pin)`` renders the same verdict at read time.
        """
        tier = int(result.tier)
        moved: list[str] = []
        prior_quality = self.baselines_by_tier.get(tier)
        if result.quality != prior_quality:
            moved.append(
                f"tier quality {self._fmt_quality(prior_quality)} -> "
                f"{self._fmt_quality(result.quality)}"
            )
            self.baselines_by_tier[tier] = result.quality
        suite_map = self.per_suite_quality_by_tier.setdefault(tier, {})
        for suite, quality in (result.per_suite_quality or {}).items():
            if quality != suite_map.get(suite):
                moved.append(
                    f"suite {suite!r} {self._fmt_quality(suite_map.get(suite))} -> "
                    f"{self._fmt_quality(quality)}"
                )
                suite_map[suite] = quality
        result_counts = getattr(result, "per_suite_counts", None) or {}
        if result_counts:
            count_map = self.per_suite_counts_by_tier.setdefault(tier, {})
            for suite, count in result_counts.items():
                if count != count_map.get(suite):
                    moved.append(f"suite {suite!r} counts {count_map.get(suite)} -> {count}")
                    count_map[suite] = count
        if tier == DEFAULT_FRONTIER_TIER:
            self.quality = result.quality
            self.per_suite_quality.update(result.per_suite_quality)
            if result.speed > 0:
                self.frontdoor_speed = result.speed
            # B3 / MISC-1: the top-level speed/cost/reliability scalars describe the
            # DEFAULT_FRONTIER_TIER production point and feed the throughput floor; gate
            # them on the same tier check as quality/frontdoor_speed. Previously an
            # audit-only / off-frontier tier (e.g. T2) promotion clobbered them with its
            # own (differently-measured) numbers, corrupting the frontier baseline.
            self.speed = result.speed
            self.cost = result.cost
            self.reliability = result.reliability
        if not moved:
            return
        revision = self.tier_revisions.get(tier, 0) + 1
        self.tier_revisions[tier] = revision
        invalidated = self.pins_for_tier(tier)
        log.warning(
            "BASELINE MOVED T%d (%s); tier revision %d -> %d%s",
            tier,
            "; ".join(moved),
            revision - 1,
            revision,
            f"; invalidated measurement pins: {[key for key in self._pins if self._pins[key].tier == tier]}"
            if invalidated
            else "",
        )

    def to_state_dict(self) -> dict[str, Any]:
        """State payload for in-memory baseline promotions; YAML remains the seed config."""
        payload: dict[str, Any] = {
            "quality": self.quality,
            "speed": self.speed,
            "cost": self.cost,
            "reliability": self.reliability,
            "per_suite_quality": self.per_suite_quality,
            "baselines_by_tier": {
                str(tier): quality for tier, quality in sorted(self.baselines_by_tier.items())
            },
            "per_suite_quality_by_tier": {
                str(tier): suites for tier, suites in sorted(self.per_suite_quality_by_tier.items())
            },
            "per_suite_counts_by_tier": {
                str(tier): counts for tier, counts in sorted(self.per_suite_counts_by_tier.items())
            },
            "frontdoor_speed": self.frontdoor_speed,
        }
        # Only emit the revision map when non-empty — keeps a legacy (pre-EV-14c)
        # baseline's state payload byte-identical, and a missing key decodes back to
        # the no-writes-yet default (revision 0 per tier).
        if self.tier_revisions:
            payload["tier_revisions"] = {
                str(tier): revision
                for tier, revision in sorted(self.tier_revisions.items())
            }
        # Only emit the era stamp when known — keeps a legacy (unstamped) baseline's state
        # payload byte-identical, and lets a missing key decode back to the pre-E7 default.
        if self.eval_quality_era:
            payload["eval_quality_era"] = self.eval_quality_era
        # Same contract on the speed axis: only emit when known, so a legacy (unstamped)
        # baseline's state payload stays byte-identical and a missing key decodes back to
        # the pre-boundary default.
        if self.autopilot_speed_era:
            payload["autopilot_speed_era"] = self.autopilot_speed_era
        return payload


@dataclass(frozen=True)
class BaselineUpdateResult:
    updated: bool
    reason: str
    tier: int
    previous_quality: float | None
    new_quality: float
    proof: dict[str, Any] = field(default_factory=dict)
    # B8 / SG-0: non-empty ONLY when _baseline_eligible refused the write (a stale or
    # unverifiable contention matrix froze the baseline ratchet). Operator remediation is
    # to re-measure/refresh the matrix so it is certified-fresh against the live topology.
    # Empty on every eligible path (including ordinary monotonic/seq skips), so a caller
    # can distinguish "ratchet frozen, go re-measure" from "candidate simply not better".
    ineligible_reason: str = ""
    # B4 / SEQ-2: machine token naming WHY the sequential anti-ratchet refused a write.
    # ``seq_inputs_unavailable`` = the sequential path is ON but no confirmed/refuted
    # verdict could be rendered (fresh journal / missing question_results) → fail-closed.
    # ``seq_not_confirmed`` = a verdict was rendered but did not clear confirm_e. Empty on
    # every other path, so a caller can tell "accumulate journal evidence" from "candidate
    # simply not confirmed yet" from an ordinary monotonic skip. Kept distinct from
    # ineligible_reason (which is reserved for the SG-0 matrix-freeze case).
    seq_refused_reason: str = ""


class SafetyGate:
    """Enforces safety constraints on trial results."""

    def __init__(
        self,
        baseline_path: Path | None = None,
        consecutive_failures: int = 0,
        quality_history: list[float] | None = None,
        quality_history_by_tier: dict[str | int, list[float]] | None = None,
        baseline_state: dict[str, Any] | None = None,
        use_sequential: bool | None = None,
        *,
        quality_history_provenance_by_tier: dict[str | int, list[Any]] | None = None,
        eval_quality_era: str | None = None,
        autopilot_speed_era: str | None = None,
        quality_exclude_before_ts: float | None = None,
    ):
        self.baseline = Baseline.load(baseline_path, state=baseline_state)
        # LEDGER-W4 (01c §3): default-off anytime-valid sequential verdict path.
        # Flag source precedence: explicit arg > AUTOPILOT_SEQ_VERDICT env > off.
        # When off (the default), check()/update_baseline behave byte-identically to
        # the pre-W4 MAD-only gate. Deploy is evidence-gated (flip-rate >= 30% over
        # ~120 trusted vectors); flipping the env alone never changes behavior unless
        # the caller also threads per-question results into check().
        self.use_sequential = (
            _env_truthy("AUTOPILOT_SEQ_VERDICT") if use_sequential is None else bool(use_sequential)
        )
        # Active eval-instrument era fence (defect #1/#4). None => the quality axis runs
        # unfenced (single-era world / pre-existing tests) and every new-behavior branch below
        # is inert, so the gate is byte-identical to the pre-fence version. When set (from
        # active_instrument_eras.eval_quality in state), a legacy/pre-E7 baseline vs this era
        # trips the re-baseline hold, and the MAD window filters to same-era samples only.
        self._eval_quality_era = (eval_quality_era or "").strip() or None
        # Active SPEED-instrument era fence (2026-08-03). Exact analogue of the field above
        # on the throughput axis; sourced from active_instrument_eras.autopilot_speed — the
        # same era id autopilot.py already uses to fence the Pareto frontier
        # (pareto_epoch_ts / frontier_rerun_required). None => the speed axis runs unfenced
        # (single-era world / pre-existing tests) and every new branch below is inert, so
        # the gate is byte-identical to the pre-fence version.
        self._autopilot_speed_era = (autopilot_speed_era or "").strip() or None
        self._quality_exclude_before_ts = quality_exclude_before_ts
        self._rebaseline_hold_logged = False
        self._speed_rebaseline_hold_logged = False
        self._consecutive_failures = consecutive_failures
        self._quality_history_by_tier: dict[int, deque[_QualityObs]] = {}
        # Precedence: provenance (rich, authoritative) > by_tier floats > flat floats. Legacy
        # bare floats decode with era="" (pre-boundary priors); a missing provenance key means
        # old state, which is exactly the pre-E7 case.
        if quality_history_provenance_by_tier:
            for tier, history in quality_history_provenance_by_tier.items():
                self._quality_history_by_tier[int(tier)] = self._obs_deque(history or [])
        if quality_history_by_tier:
            for tier, history in quality_history_by_tier.items():
                self._quality_history_by_tier.setdefault(
                    int(tier), self._obs_deque(history or [])
                )
        if quality_history:
            # Legacy flat state had no tier label. Seed all current tiers so resumes and
            # older unit fixtures preserve their pre-migration behavior until enough
            # same-tier samples replace the migrated window.
            for tier in (0, DEFAULT_FRONTIER_TIER, 2):
                self._quality_history_by_tier.setdefault(
                    tier, self._obs_deque(quality_history)
                )
        self._last_history_tier = DEFAULT_FRONTIER_TIER

    @staticmethod
    def _obs_deque(entries: list[Any]) -> deque:
        obs = (_coerce_quality_obs(entry) for entry in entries)
        return deque((o for o in obs if o is not None), maxlen=MAD_HISTORY_DEPTH)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def quality_rebaseline_required(self) -> bool:
        """True when the resident baseline predates the active eval-instrument era.

        The fail-closed hold (defect #3): with an active ``eval_quality`` era set, a baseline
        stamped under a DIFFERENT (or no) era must not gate post-boundary results. Comparing a
        post-E7 result against a pre-E7 baseline/per-suite/MAD window would charge a
        scorer/pool change to the model. While this holds, the gate suppresses the
        baseline-comparison quality legs and refuses quality promotion until an operator
        reseeds a same-era baseline. Inert (always False) when no active era is set.
        """
        if not self._eval_quality_era:
            return False
        return (self.baseline.eval_quality_era or "") != self._eval_quality_era

    def _log_rebaseline_hold_once(self, result: EvalResult) -> None:
        """Emit the re-baseline hold at ERROR exactly once per gate instance (loud, not spammy)."""
        if self._rebaseline_hold_logged:
            return
        self._rebaseline_hold_logged = True
        log.error(
            "EVAL-INSTRUMENT RE-BASELINE HOLD — resident baseline era=%r != active eval_quality "
            "era=%r. Post-boundary results (e.g. T%s q=%.3f) will NOT gate quality promote/revert "
            "against the pre-boundary baseline/per-suite/MAD window. REMEDIATION: reseed "
            "autopilot_state.json:baseline_state from a post-boundary eval (its "
            "eval_quality_era must equal %r) so the ratchet resumes. This is the fail-closed "
            "quality fence (defect #3), analogous to the speed axis's frontier_rerun_required.",
            self.baseline.eval_quality_era or "<pre-boundary>",
            self._eval_quality_era,
            getattr(result, "tier", "?"),
            float(getattr(result, "quality", float("nan"))),
            self._eval_quality_era,
        )

    @property
    def speed_rebaseline_required(self) -> bool:
        """True when the resident baseline's frontdoor_speed predates the active speed era.

        The THROUGHPUT-axis analogue of :attr:`quality_rebaseline_required` (2026-08-03).
        With an active ``autopilot_speed`` era set, a ``frontdoor_speed`` stamped under a
        DIFFERENT (or no) era cannot attribute a throughput drop to the config-under-test:
        the floor (0.8 * frontdoor_speed) was measured on a different speed instrument
        (kernel/binary/topology), so a post-boundary trial charged against it is charged
        against a number nobody can trace. While this holds the throughput violation is
        DEMOTED to a warning — deliberately NOT hard-failed, since charging a trial against
        an unattributable floor is exactly the defect. Inert (always False) when no active
        era is set.
        """
        if not self._autopilot_speed_era:
            return False
        return (self.baseline.autopilot_speed_era or "") != self._autopilot_speed_era

    def _log_speed_rebaseline_hold_once(self, result: EvalResult) -> None:
        """Emit the throughput re-baseline hold at ERROR exactly once per gate instance."""
        if self._speed_rebaseline_hold_logged:
            return
        self._speed_rebaseline_hold_logged = True
        log.error(
            "SPEED-INSTRUMENT RE-BASELINE HOLD — resident baseline autopilot_speed era=%r != "
            "active autopilot_speed era=%r. The throughput floor %.1f t/s (80%% of baseline "
            "frontdoor_speed %.1f) was measured under a DIFFERENT speed instrument, so this "
            "trial (%.1f t/s) will NOT be charged against it — the violation is DEMOTED to a "
            "warning. REMEDIATION: reseed autopilot_state.json:baseline_state from a "
            "post-boundary eval (its autopilot_speed_era must equal %r) so the throughput "
            "floor binds again. Analogue of the quality axis's eval-instrument hold.",
            self.baseline.autopilot_speed_era or "<pre-boundary>",
            self._autopilot_speed_era,
            self.baseline.frontdoor_speed * 0.8,
            self.baseline.frontdoor_speed,
            float(getattr(result, "speed", float("nan"))),
            self._autopilot_speed_era,
        )

    @property
    def quality_history(self) -> list[float]:
        """Return the latest tier's rolling quality window (legacy state persistence)."""
        return self.quality_history_for_tier(self._last_history_tier)

    @property
    def quality_history_by_tier(self) -> dict[str, list[float]]:
        """Return rolling quality windows keyed by eval tier for state persistence."""
        return {
            str(tier): [obs.q for obs in history]
            for tier, history in sorted(self._quality_history_by_tier.items())
        }

    @property
    def quality_history_provenance_by_tier(self) -> dict[str, list[dict[str, Any]]]:
        """Return rolling quality windows WITH provenance (era/ts/core_id) for persistence.

        The authoritative persisted shape (defect #4): each sample records the era +
        timestamp + core_id it was measured under, so a resumed gate's MAD window cannot
        silently mix a pre-/post-boundary sample. The legacy float ``quality_history_by_tier``
        is still written alongside for any external float reader.
        """
        return {
            str(tier): [
                {"q": obs.q, "ts": obs.ts, "era": obs.era, "core_id": obs.core_id}
                for obs in history
            ]
            for tier, history in sorted(self._quality_history_by_tier.items())
        }

    def quality_history_for_tier(self, tier: int) -> list[float]:
        return [obs.q for obs in self._quality_history_by_tier.get(int(tier), deque())]

    def _history_for_tier(self, tier: int) -> deque:
        return self._quality_history_by_tier.setdefault(int(tier), deque(maxlen=MAD_HISTORY_DEPTH))

    def _mad_window_values(self, tier: int) -> list[float]:
        """Same-tier MAD samples, era-filtered when an eval_quality era is active.

        With an active era set, only samples measured under that same era participate —
        legacy bare floats (era="") and any other-era samples are pre-boundary priors and are
        excluded, so a post-boundary median cannot be dragged by a stale-instrument window.
        Without an active era the full window is used (pre-fence behavior).
        """
        history = self._history_for_tier(tier)
        if self._eval_quality_era:
            return [obs.q for obs in history if (obs.era or "") == self._eval_quality_era]
        return [obs.q for obs in history]

    def _mad_significance(self, new_quality: float, tier: int) -> tuple[bool, float, float, float]:
        """Decide whether ``new_quality`` is statistically significant vs history.

        Returns (is_significant, z_mad, median, mad). When history < MAD_MIN_SAMPLES,
        returns (True, NaN, NaN, NaN) — insufficient data to filter, accept at face value.
        """
        history = self._mad_window_values(tier)
        if len(history) < MAD_MIN_SAMPLES:
            return True, math.nan, math.nan, math.nan
        median_q = statistics.median(history)
        mad = statistics.median(abs(x - median_q) for x in history)
        if mad == 0:
            # SG-7 (B9): a saturated / degenerate window (every recent same-tier sample
            # identical) has MAD == 0. The old rule "any nonzero delta is significant"
            # promoted a single-question flip to a real gain. Require the delta to clear
            # TWO single-flip quanta before calling it significant. z is undefined for a
            # zero-MAD window → NaN; check() tags the resulting within-band case
            # ``mad_zero_window`` so a NaN z can no longer dodge the noise classification.
            return (abs(new_quality - median_q) > MAD_ZERO_MIN_DELTA), math.nan, median_q, 0.0
        z_mad = abs(new_quality - median_q) / (mad * MAD_CONSISTENCY)
        return z_mad > MAD_Z_THRESHOLD, z_mad, median_q, mad

    def _sequential_verdict(
        self,
        result: EvalResult,
        *,
        question_results: Any,
        task_rate: float | None,
        baseline_profile: Any,
        baseline_task_rate: float | None,
        prior_quality_obs: Any = (),
        prior_rate_obs: Any = (),
        candidate: str = "",
        core_id: str = "",
    ) -> dict[str, Any]:
        """Fold this trial into the candidate's anytime-valid e-processes (01c §3).

        Pure given its inputs: rebuilds the candidate's prior quality (and, when a
        task_rate + baseline_task_rate are supplied, rate-non-inferiority) e-process
        from ``prior_*_obs``, applies one update for THIS trial, and returns the
        ``journal_seq_block`` augmented with the JOINT state and a ``confirmed`` flag.

        Joint rule (01c §3): ``confirmed`` iff E_quality >= confirm_e AND
        E_rate_noninf >= confirm_e. ``refuted`` if EITHER axis is refuted. Otherwise
        ``accumulating``. When no rate evidence is available the verdict can never be
        ``confirmed`` (E_rate is absent) — conservative by design: a candidate cannot
        ratchet the baseline on quality alone.

        OP-1/P0.2 bridge: when the operator-owned consent file and restart env
        both enable ``SEQ_P0_2_BRIDGE_MODE``, the rate axis is recorded as
        advisory while quality remains binding. This is a temporary era-fence
        bridge, not the default measurement rule.
        """
        from src.autopilot_core.sequential_verdict import (
            DEFAULT_POLICY,
            STATE_REFUTED,
            journal_seq_block,
            quality_trial_statistic,
            rate_noninferiority_z,
            rebuild_candidate_view,
        )

        policy = DEFAULT_POLICY
        cand = candidate or "candidate"
        core = core_id or "core"
        trial_id = getattr(result, "trial_id", None)

        # Quality axis: center this trial's per-qid outcomes against the baseline
        # profile, then fold into the candidate's prior quality e-process.
        stat = quality_trial_statistic(question_results, baseline_profile)
        q_view = rebuild_candidate_view(
            candidate=cand, core_id=core, observations=prior_quality_obs, policy=policy
        )
        q_state, q_update = q_view.quality_state.update(stat.z, policy=policy, trial_id=trial_id)

        # Rate-non-inferiority axis (only when a MEASURED task_rate + positive baseline
        # exist).
        #
        # SEQ-B: the guard used to be `task_rate is not None`. `task_rate_qph_from` returns
        # 0.0 as its "wall/n unavailable" sentinel, so an unmeasurable trial was fed to
        # `rate_noninferiority_z` as a measured throughput of ZERO questions/hour =>
        # y = -1 => the clip floor z = -0.9. That fabricates the strongest possible
        # negative observation out of a missing measurement, and because `next_lambda`
        # clips a nonpositive running mean to lambda = 0, the wealth then freezes at
        # `1 + 0.1*(-0.9) = 0.91` and multiplies by EXACTLY 1.0 forever after.
        # `seq_task_rate_qph_from` now returns None for "not measured"; requiring
        # `> 0` here additionally fails closed for any legacy caller still passing the
        # 0.0 sentinel. A skipped axis leaves E_rate absent, which can never confirm —
        # the conservative outcome, and the same skip-don't-fabricate doctrine
        # `rebuild_candidate_view` applies to out-of-domain z (SEQ-3a).
        rate_state = None
        rate_update = None
        rate_axis_skip_reason = ""
        if task_rate is None or float(task_rate) <= 0.0:
            rate_axis_skip_reason = "candidate_task_rate_not_measured"
        elif baseline_task_rate is None or float(baseline_task_rate) <= 0.0:
            rate_axis_skip_reason = "incumbent_task_rate_comparator_unavailable"
        if not rate_axis_skip_reason:
            z_rate = rate_noninferiority_z(
                float(task_rate),
                float(baseline_task_rate),
                margin=policy.rate_noninferiority_margin,
            )
            rate_view = rebuild_candidate_view(
                candidate=cand,
                core_id=core,
                observations=prior_rate_obs,
                policy=policy,
                expected_axis="rate",
            )
            rate_state, rate_update = rate_view.quality_state.update(
                z_rate, policy=policy, trial_id=trial_id
            )

        e_quality = q_state.wealth
        e_rate = rate_state.wealth if rate_state is not None else None
        q_name = q_state.state_name(policy)
        rate_name = rate_state.state_name(policy) if rate_state is not None else None
        bridge = seq_p0_2_bridge_status()
        rate_axis_advisory = bool(bridge["enabled"])

        # SEQ-3 (B9): a REFUTED rate axis blocks confirmation even under the advisory
        # (P0.2 bridge) mode. Refutation is strictly STRONGER evidence than mere
        # non-confirmation: the bridge relaxes the rate axis from *required-to-confirm*
        # down to advisory (see the confirm branch below), but it must NOT let an
        # actively-refuted rate axis ratchet the baseline. Previously advisory mode
        # ignored rate refutation entirely (`not rate_axis_advisory and ...`).
        if q_name == STATE_REFUTED or rate_name == STATE_REFUTED:
            state = "refuted"
        elif e_quality >= policy.confirm_e and (
            rate_axis_advisory or (e_rate is not None and e_rate >= policy.confirm_e)
        ):
            state = "confirmed"
        else:
            state = "accumulating"

        block = journal_seq_block(
            candidate=cand,
            core_id=core,
            quality_update=q_update,
            quality_state=q_state,
            policy=policy,
            rate_noninf_update=rate_update,
        )
        block["r_eff"] = stat.r_eff
        # SEQ-B: make an OMITTED rate axis visible. Before, an absent E_rate_noninf was
        # indistinguishable in the journal from an axis that ran and produced nothing, so
        # "the gate is unreachable" could not be told apart from "the gate is unfed".
        block["rate_axis_available"] = rate_update is not None
        if rate_axis_skip_reason:
            block["rate_axis_skip_reason"] = rate_axis_skip_reason
        # SEQ-B: journal the two numbers z_rate is actually made of. Diagnosing why
        # `z_rate` sat at its clip floor required reverse-engineering the candidate rate
        # out of `eval_details.goodput_qph` and re-deriving the comparator from 120 prior
        # rows; neither input was ever recorded next to the statistic it produced.
        if task_rate is not None:
            block["task_rate_qph"] = round(float(task_rate), 6)
        if baseline_task_rate is not None:
            block["baseline_task_rate_qph"] = round(float(baseline_task_rate), 6)
        if rate_axis_advisory:
            block["rate_axis_mode"] = SEQ_P0_2_BRIDGE_MODE
            block["rate_axis_binding"] = False
            block["p0_2_bridge"] = bridge
        else:
            block["rate_axis_mode"] = "binding_joint"
            block["rate_axis_binding"] = True
        # journal_seq_block records the QUALITY-only state; override with the joint
        # verdict so consumers (categories, update_baseline gate) read one decision.
        block["state"] = state
        block["confirmed"] = state == "confirmed"
        return block

    def check(
        self,
        result: EvalResult,
        *,
        question_results: Any = None,
        task_rate: float | None = None,
        baseline_profile: Any = None,
        baseline_task_rate: float | None = None,
        prior_quality_obs: Any = None,
        prior_rate_obs: Any = None,
        candidate: str = "",
        core_id: str = "",
    ) -> SafetyVerdict:
        """Run all safety checks on an eval result.

        LEDGER-W4: when ``self.use_sequential`` is on AND the caller supplies
        ``question_results`` + ``baseline_profile``, the improvement branch uses the
        anytime-valid sequential e-process verdict (01c §3) in place of the MAD noise
        filter. All seq kwargs are keyword-only and default to ``None`` so every
        existing caller (which passes only ``result``) is unaffected — the seq path
        is inert unless both the flag and the per-question inputs are present.
        """
        seq_inputs_ready = (
            self.use_sequential and question_results is not None and bool(baseline_profile)
        )
        cached_verdict = result.gate_verdict
        record_side_effects = cached_verdict is None
        if cached_verdict is not None:
            can_upgrade_cached_seq = (
                cached_verdict.passed
                and cached_verdict.seq is None
                and seq_inputs_ready
                and not any(
                    category.startswith("throughput_host_")
                    for category in cached_verdict.categories
                )
            )
            if not can_upgrade_cached_seq:
                return cached_verdict

        violations = []
        warnings = []
        categories = []  # AP-14: track which checks failed
        seq_block: dict[str, Any] | None = None  # LEDGER-W4 journal block (default-off)

        def record_seq_shadow() -> None:
            nonlocal seq_block
            if seq_block is not None or not seq_inputs_ready:
                return
            # LEDGER-W4 (01c §3): when the default-off sequential path is active
            # and the caller threads per-question results, journal the anytime-valid
            # e-process verdict for every trusted trial, including regressions. The
            # verdict remains advisory at the safety gate: it never adds a violation,
            # and baseline promotion is still gated downstream through
            # update_baseline(seq_confirmed=...).
            seq_block = self._sequential_verdict(
                result,
                question_results=question_results,
                task_rate=task_rate,
                baseline_profile=baseline_profile,
                baseline_task_rate=baseline_task_rate,
                prior_quality_obs=prior_quality_obs or (),
                prior_rate_obs=prior_rate_obs or (),
                candidate=candidate,
                core_id=core_id,
            )
            categories.append("seq_" + seq_block["state"])
            e_rate = seq_block.get("E_rate_noninf")
            rate_axis_mode = seq_block.get("rate_axis_mode", "binding_joint")
            warnings.append(
                "Sequential verdict ({state}): E_quality={eq:.3f}, "
                "E_rate_noninf={er}, rate_axis_mode={rate_axis_mode}, "
                "k={k} (LEDGER-W4, AUTOPILOT_SEQ_VERDICT); "
                "baseline promotion {gate}.".format(
                    state=seq_block["state"],
                    eq=seq_block.get("E_quality", float("nan")),
                    er=f"{e_rate:.3f}" if e_rate is not None else "n/a",
                    rate_axis_mode=rate_axis_mode,
                    k=seq_block.get("k"),
                    gate=(
                        "permitted (confirmed)"
                        if seq_block.get("confirmed")
                        else "blocked until confirmed"
                    ),
                )
            )

        # SG-5 (B3b): fail-closed on a non-finite quality BEFORE any comparison-based
        # gate. NaN/inf silently passes every `<`/`>` check (all comparisons are False),
        # so a degenerate eval would otherwise sail through the quality floor and the
        # regression gate. This is INDEPENDENT of the reliability suppression below: a NaN
        # quality with good reliability must still fail (a degenerate measurement is not an
        # infra-error retry). Checked first so it is recorded regardless of the other legs.
        if not math.isfinite(result.quality):
            violations.append("quality is not finite — degenerate eval")
            categories.append("quality_not_finite")

        # REL-1 (B1): reliability conditioning. When the eval's non-error fraction is below
        # the floor the per-question outcomes are untrustworthy (infra errors), so the
        # quality-floor / regression / per-suite checks are computed over garbage — running
        # them would convert an infrastructure failure into a spurious quality-regression
        # REVERT. Record a violation, mark the verdict reliability_blocked, and SKIP those
        # three legs; this trial signals RETRY, not revert (see the consecutive-failure
        # guard below, which does not advance the auto-rollback counter for it). Routing /
        # throughput legs still run — they don't depend on per-question correctness.
        reliability_blocked = False
        reliability_floor = _reliability_floor()
        if math.isfinite(result.reliability) and result.reliability < reliability_floor:
            violations.append(
                f"Reliability {result.reliability:.2f} below floor {reliability_floor:.2f} — "
                "eval evidence untrustworthy (infra errors), quality checks suppressed"
            )
            categories.append("reliability_floor")
            reliability_blocked = True

        # Defect #3: eval-instrument re-baseline hold. When an eval_quality era is active and
        # the resident baseline predates it, comparing a post-boundary result against the
        # pre-boundary baseline / per-suite / MAD window would charge a scorer/pool change to
        # the model (spurious revert) or credit an easier pool (spurious promotion). SUPPRESS
        # the baseline-comparison legs (regression, per-suite, MAD) rather than compare across
        # the boundary; log loudly. The absolute quality floor still runs (era-neutral safety),
        # and update_baseline() separately refuses quality promotion until an operator reseeds a
        # same-era baseline. Inert when no active era is set.
        quality_rebaseline_hold = not reliability_blocked and self.quality_rebaseline_required
        if quality_rebaseline_hold:
            categories.append("quality_rebaseline_required")
            warnings.append(
                "Eval-instrument re-baseline hold: resident baseline era="
                f"{self.baseline.eval_quality_era or '<pre-boundary>'} != active eval_quality "
                f"era={self._eval_quality_era}; cross-era regression/per-suite/MAD gating "
                "SUPPRESSED. Reseed a same-era baseline (post-boundary eval) before quality "
                "promote/revert can resume."
            )
            self._log_rebaseline_hold_once(result)

        if not reliability_blocked:
            # 1. Quality floor (tier-aware)
            quality_floor = QUALITY_FLOOR_T0 if result.tier == 0 else QUALITY_FLOOR_T1
            if result.quality < quality_floor:
                violations.append(
                    f"Quality floor violation: {result.quality:.3f} < {quality_floor} (tier {result.tier})"
                )
                categories.append("quality_floor")

            # 2. Regression vs baseline (relative: allow 5% drop from baseline). SG-3 (B3a):
            # use the STRICT same-tier baseline — NO cross-tier legacy fallback for gating.
            # The lenient accessor would compare, e.g., a T2 result against the top-level
            # (tier-1) legacy quality, force-reverting on a difficulty gap that is not a
            # regression. When no same-tier baseline exists we SKIP the gate (log.info) and
            # let update_baseline seed the tier baseline on the next eligible promotion.
            # Under the re-baseline hold baseline_q is forced None so this cross-era compare
            # (and its MAD filter) is skipped.
            baseline_q = (
                None
                if quality_rebaseline_hold
                else self.baseline.quality_for_tier(result.tier, strict=True)
            )
            if baseline_q is None:
                if not quality_rebaseline_hold:
                    log.info(
                        "No strict same-tier T%d quality baseline — skipping regression gate "
                        "(baseline will seed on the next eligible promotion).",
                        result.tier,
                    )
            elif baseline_q > 0:
                relative_delta = (result.quality - baseline_q) / baseline_q
                if relative_delta < REGRESSION_THRESHOLD:
                    violations.append(
                        f"Quality regression: {result.quality:.3f} vs baseline {baseline_q:.3f} "
                        f"({relative_delta:+.1%}, threshold: {REGRESSION_THRESHOLD:+.0%})"
                    )
                    categories.append("regression")
                elif relative_delta < 0:
                    warnings.append(
                        f"Slight quality drop: {result.quality:.3f} vs baseline {baseline_q:.3f} "
                        f"({relative_delta:+.1%})"
                    )
                elif seq_inputs_ready:
                    record_seq_shadow()
                    # Seq evidence is already journaled above and replaces the MAD noise
                    # filter as the "is this a real improvement?" significance test.
                else:
                    # Improvement or no change — apply MAD noise filter (intake-421).
                    # Robust against outliers; only fires once history has MAD_MIN_SAMPLES.
                    # Gate never blocks; the `mad_noise` category is consumed by
                    # autopilot's classify_learning_exclusion() helper, which
                    # journals the trial but skips archive.update + AP-22 short-term
                    # memory so noise-level improvements don't poison the Pareto
                    # frontier or strategy memory.
                    is_sig, z_mad, median_q, mad = self._mad_significance(
                        result.quality, result.tier
                    )
                    if not is_sig:
                        categories.append("mad_noise")
                        # SG-7 (B9): a degenerate / saturated window yields a NaN z
                        # (MAD == 0). The old guard `not math.isnan(z_mad)` let that NaN
                        # z DODGE the mad_noise tag entirely, silently promoting a
                        # within-tolerance change as a fresh gain. Tag it distinctly so
                        # the planner still excludes it from archive/learning (it rides
                        # the same `mad_noise` exclusion path) while staying auditable.
                        if math.isnan(z_mad):
                            categories.append("mad_zero_window")
                        # Convergence-vs-corruption disambiguation (2026-05-31).
                        # `mad_noise` is correct and unchanged: this is not a NEW
                        # statistically significant improvement, so it earns no
                        # Pareto point. But "within noise" conflates two very
                        # different situations. If the established recent level
                        # (history median) is ITSELF significantly above baseline,
                        # then this result REPRODUCES an already-demonstrated
                        # above-baseline gain — a convergence/confidence signal, not
                        # corrupted or untrustworthy data. Tag it separately
                        # (`reproduction_confirmed`) so the planner's
                        # "noisy/untrustworthy instrument" summary never lumps
                        # reproductions in with kills / exogenous reloads /
                        # bug-corruptions. The MAD statistic is NOT re-anchored.
                        # SG-3: base_q is the strict same-tier baseline (== baseline_q here).
                        base_q = baseline_q
                        reproduction_confirmed = (
                            base_q is not None
                            and base_q > 0
                            and not math.isnan(median_q)
                            and mad > 0
                            and (median_q - base_q) > MAD_Z_THRESHOLD * mad * MAD_CONSISTENCY
                        )
                        if reproduction_confirmed:
                            categories.append("reproduction_confirmed")
                        convergence_note = (
                            " Reproduces an established above-baseline level "
                            "(history median {:.3f} >> baseline {:.3f}): this is a "
                            "convergence/confirmation of an existing gain, NOT "
                            "instrument noise or a corrupted trial.".format(median_q, base_q)
                            if reproduction_confirmed
                            else ""
                        )
                        warnings.append(
                            f"Improvement within noise (MAD filter): q={result.quality:.3f} "
                            f"vs history median {median_q:.3f} (MAD={mad:.4f}, z={z_mad:.2f}, "
                            f"threshold={MAD_Z_THRESHOLD}); still journaled, excluded "
                            f"from archive/learning by autopilot." + convergence_note
                        )

            # 3. Per-suite regression (resolution-aware since 2026-06-06). A per-suite
            # score is fraction_correct*3 over only the questions that suite drew; at
            # ~2 q/suite a single flip is a 1.5 swing, so a fixed -0.1 floor flagged
            # pure sampling noise as a regression on essentially every trial and
            # deadlocked the planner. The threshold is widened to the coarser of the
            # result's and baseline's single-flip quantum (3/n); counts default empty
            # ⇒ fixed -0.1 floor (unchanged behavior for pre-2026-06-06 baselines).
            # If either side has very low support, a threshold-crossing drop remains
            # visible but advisory unless it is a catastrophic 0-3 scale collapse.
            # Under the re-baseline hold the pre-boundary per-suite baselines are withheld
            # (empty), so no cross-era per-suite regression can fire.
            baseline_suites = (
                {} if quality_rebaseline_hold else self.baseline.per_suite_for_tier(result.tier)
            )
            baseline_counts = self.baseline.per_suite_counts_for_tier(result.tier)
            result_counts = getattr(result, "per_suite_counts", None) or {}
            for suite, quality in result.per_suite_quality.items():
                baseline_q = baseline_suites.get(suite)
                if baseline_q is not None:
                    suite_delta = quality - baseline_q
                    result_n = result_counts.get(suite)
                    baseline_n = baseline_counts.get(suite)
                    threshold = per_suite_regression_threshold(result_n, baseline_n)
                    # B2 / SG-1: fire only when the delta is strictly MORE negative than the
                    # single-flip quantum. A delta exactly equal to one flip is at-resolution
                    # noise; the PER_SUITE_EPS guard keeps float rounding from crossing the
                    # bare `<` (the 185 (n,k) boundary artifacts). Restores documented intent.
                    if suite_delta < threshold - PER_SUITE_EPS:
                        msg = (
                            f"Suite '{suite}' regression: {suite_delta:+.3f} "
                            f"(threshold: {threshold:+.3f}; "
                            f"n_result={result_n}, n_baseline={baseline_n})"
                        )
                        # tool_use sentinel suite is inherently flaky (substring scoring
                        # of REPL output); only catastrophic regressions (3+ questions
                        # failed, delta <= -3.0) are hard violations. Moderate drops are
                        # advisory to prevent blocking quality-positive config changes.
                        is_tool_use_advisory = (
                            suite == "tool_use" and suite_delta > -TOOL_USE_CATASTROPHIC_REGRESSION
                        )
                        if is_tool_use_advisory:
                            warnings.append(
                                f"{msg} tool_use regression treated as advisory — "
                                f"only catastrophic drops (<= -{TOOL_USE_CATASTROPHIC_REGRESSION}) "
                                "are hard violations for this suite."
                            )
                            if "tool_use_regression_advisory" not in categories:
                                categories.append("tool_use_regression_advisory")
                        elif _per_suite_regression_binding(suite_delta, result_n, baseline_n):
                            violations.append(msg)
                            if "per_suite_regression" not in categories:
                                categories.append("per_suite_regression")
                        else:
                            min_baseline_n = _per_suite_baseline_min_n()
                            if baseline_n is not None and 0 < baseline_n < min_baseline_n:
                                # 2026-07-16 thrash: tiny-n baseline (debugbench
                                # n=2 @ 3.0) — annotate the small-sample condition
                                # explicitly; the suite still counts in aggregate
                                # scoring, it just cannot trigger a HARD rollback.
                                warnings.append(
                                    f"{msg} treated as advisory (small-sample baseline): "
                                    f"baseline n={baseline_n} < {min_baseline_n} is too "
                                    "sparse to certify a hard rollback."
                                )
                            else:
                                warnings.append(
                                    f"{msg} treated as advisory because per-suite support "
                                    f"is below n={PER_SUITE_BINDING_MIN_COUNT} and the "
                                    "drop is not catastrophic."
                                )
                            if "per_suite_regression_advisory" not in categories:
                                categories.append("per_suite_regression_advisory")

        # 4. Routing diversity
        architect_frac = result.routing_distribution.get("architect", 0.0)
        if architect_frac > ARCHITECT_ROUTING_CAP:
            violations.append(
                f"Routing diversity violation: {architect_frac:.1%} architect-tier "
                f"(cap: {ARCHITECT_ROUTING_CAP:.0%})"
            )
            categories.append("routing_diversity")

        # 5. Throughput floor
        #
        # REL-1 (above) exempts the throughput leg from reliability suppression on
        # the grounds that it "does not depend on per-question correctness". That
        # reasoning holds while the eval GENERATED tokens and merely scored them
        # badly. It breaks at speed == 0: zero tokens/s is not a slow measurement,
        # it is the ABSENCE of one, and when reliability has already declared the
        # run's evidence untrustworthy, the same infra failure produced both
        # numbers. Charging it writes a fabricated regression into
        # failure_analysis, which is planner-visible evidence — trial 1459
        # (2026-08-03) carried "Throughput floor: 0.0 t/s < 10.2 t/s (80% of
        # baseline 12.7)" from a run that generated nothing at all.
        #
        # Deliberately narrow: a genuine hang with INTACT reliability still
        # violates, because there the zero is attributable to the config.
        speed_unmeasured = reliability_blocked and result.speed <= 0.0
        if speed_unmeasured:
            warnings.append(
                "Throughput not measured (0.0 t/s) on a reliability-blocked trial: "
                "the infra failure that voided the eval evidence also voided the "
                "speed sample. NOT charged as a throughput violation."
            )
            categories.append("throughput_unmeasured")
        elif result.speed < self.baseline.frontdoor_speed * 0.8:
            # 2026-05-09 / SG-9 (B9): before attributing a throughput regression to
            # the config-under-test, check whether the HOST is itself throttled (CPU
            # freq dip / page-cache fragmentation per feedback_host_throttle_check.md).
            # If so, DEMOTE the throughput violation to a warning so a transient host
            # stall doesn't force-revert a good config (the 2026-05-09 incident:
            # frontdoor measured 7.48 t/s = 1/3 of expected after 9 hours of mlocked
            # load). The gate must NOT remediate here: a bare drop_caches from an
            # arbitrary thread pins NUMA pages (feedback_drop_caches_numa_eviction.md),
            # and check() must not mutate host state — remediation belongs to the
            # host_health cadence, not this read-only gate. Detection is READ-ONLY.
            host_throttled = False
            host_triggers: list[str] = []
            try:
                from scripts.autopilot.host_health import HostHealthState

                _hh_state = HostHealthState.snapshot()
                host_throttled, host_triggers = _hh_state.is_throttled()
            except Exception as exc:  # noqa: BLE001
                # SG-9 (B9): detection must never crash the gate — but the old blanket
                # `except Exception: pass` let a broken import silently disable throttle
                # detection FOREVER, so every real host stall would have been charged to
                # the config-under-test. Surface it: the throughput violation will NOT be
                # throttle-demoted this trial, and the operator sees the breakage.
                log.warning(
                    "Host-throttle detection failed (%s); throughput violation will "
                    "NOT be throttle-demoted this trial.",
                    exc,
                )

            base_msg = (
                f"Throughput floor: {result.speed:.1f} t/s < "
                f"{self.baseline.frontdoor_speed * 0.8:.1f} t/s "
                f"(80% of baseline {self.baseline.frontdoor_speed:.1f})"
            )
            # Speed-instrument re-baseline hold (2026-08-03). Recorded BEFORE the throttle
            # branch and independently of it, so the provenance defect is always visible
            # even when a host stall also demotes the same violation (the two carry
            # different downstream meanings: exogenous_cache_flush excludes the trial from
            # the planner's trust window; this one says the FLOOR itself is unattributable).
            # Demotion, never a hard fail — charging a trial against an unattributable
            # floor is precisely the defect being fixed.
            speed_rebaseline_hold = self.speed_rebaseline_required
            if speed_rebaseline_hold:
                categories.append("throughput_rebaseline_required")
                warnings.append(
                    f"{base_msg}. Speed-instrument re-baseline hold: resident baseline "
                    f"autopilot_speed era="
                    f"{self.baseline.autopilot_speed_era or '<pre-boundary>'} != active "
                    f"autopilot_speed era={self._autopilot_speed_era}; the floor derives "
                    "from a frontdoor_speed measured on a DIFFERENT speed instrument, so "
                    "this trial cannot be charged against it. Throughput violation DEMOTED "
                    "to warning (reason=speed_rebaseline_required). Remediation: reseed "
                    "baseline_state.frontdoor_speed from a post-boundary measurement "
                    "(autopilot_speed_era must equal the active era) so the floor binds."
                )
                self._log_speed_rebaseline_hold_once(result)
            if host_throttled:
                # SG-9 (B9): explicit violation→warning DEMOTION, recorded so the pass
                # is auditable rather than an invisible skip. Tag exogenous_cache_flush
                # so the planner's trustworthiness gate excludes data taken under the
                # suspect host window (DeficiencyCategory.EXOGENOUS_CACHE_FLUSH); the
                # actual drop_caches + NUMA-interleave rewarm is performed by the
                # host_health cadence, not here.
                warnings.append(
                    f"{base_msg}. Host throttle detected ({'; '.join(host_triggers)}); "
                    f"throughput violation DEMOTED to warning (reason=throttle_demoted). "
                    f"Remediation deferred to the host_health cadence — RECOMMEND retry."
                )
                categories.append("throughput_throttle_demoted")
                categories.append("exogenous_cache_flush")
            elif not speed_rebaseline_hold:
                violations.append(base_msg)
                categories.append("throughput")
        elif result.speed < self.baseline.frontdoor_speed * 0.9:
            warnings.append(
                f"Speed marginal: {result.speed:.1f} t/s "
                f"({result.speed / self.baseline.frontdoor_speed:.0%} of baseline)"
            )

        # 6. Proxy-only improvement detection (skeptical re-questioning)
        warnings.extend(self._proxy_check(result))

        record_seq_shadow()

        passed = len(violations) == 0
        verdict = SafetyVerdict(
            passed=passed,
            violations=violations,
            warnings=warnings,
            categories=categories,
            seq=seq_block,
            reliability_blocked=reliability_blocked,
        )

        # Track consecutive failures only for the first gate pass over this
        # EvalResult. The central AutoPilot loop may re-enter check() with W4
        # per-question evidence after an action handler already cached a legacy
        # verdict; that seq-aware upgrade must not double-count failures.
        if record_side_effects:
            if reliability_blocked:
                # REL-1 (B1): an untrustworthy-evidence failure signals RETRY, not a
                # revert — it must NOT advance the auto-rollback counter (nor reset it).
                # Otherwise a run of infra-error trials would trip should_rollback() and
                # revert a config that was never actually shown to regress.
                pass
            elif not passed:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0

        # Record this trial's quality in the rolling window (after the verdict
        # so the current measurement doesn't bias its own significance test).
        # Skip if quality is nonsensical (NaN / negative). Also skip trials that
        # FAILED the gate (regression / quality-floor / throughput violations):
        # those configs are reverted, so they are NOT part of the operating-
        # quality distribution and must not inflate the MAD noise band
        # (2026-05-31: a reverted −66% prompt-mutation regression widened the
        # band and helped mask a real reproduced gain). Narrow, deliberate:
        # bug-corrupted / killed trials never reach here with a clean
        # EvalResult, so they are already excluded from the window.
        if (
            record_side_effects
            and passed
            and not math.isnan(result.quality)
            and result.quality >= 0
        ):
            # Defect #4: append WITH provenance. The era stamp is the active eval_quality era
            # (or "" when unfenced), so a resumed gate's MAD window filters to same-era samples
            # and a boundary crossing can never be silently averaged into the median.
            self._history_for_tier(result.tier).append(
                _QualityObs(
                    q=result.quality,
                    ts=_now_iso(),
                    era=self._eval_quality_era or "",
                    core_id=str(getattr(result, "core_id", "") or ""),
                )
            )
            self._last_history_tier = int(result.tier)

        result.gate_verdict = verdict
        return verdict

    def _proxy_check(self, result: EvalResult) -> list[str]:
        """Detect proxy-only improvements: quality up but concentrated in easy suites.

        Returns warnings (not violations) — these are suspicious but not blocking.
        Flags cases where overall quality improved but only 1 suite drove the gain
        while other suites declined.  (GPD "skeptical re-questioning" pattern.)
        """
        warnings: list[str] = []
        if not result.per_suite_quality or not self.baseline.per_suite_quality:
            return warnings

        improved: list[tuple[str, float]] = []
        declined: list[tuple[str, float]] = []
        for suite, q in result.per_suite_quality.items():
            bq = self.baseline.per_suite_quality.get(suite)
            if bq is None:
                continue
            delta = q - bq
            if delta > 0.05:
                improved.append((suite, delta))
            elif delta < -0.02:
                declined.append((suite, delta))

        # Flag if gains concentrated in ≤1 suite while others declined
        if improved and declined and len(improved) <= 1:
            imp_str = ", ".join(f"{s} +{d:.2f}" for s, d in improved)
            dec_str = ", ".join(f"{s} {d:+.2f}" for s, d in declined)
            warnings.append(
                f"Proxy-only improvement: gains in [{imp_str}] but declines in [{dec_str}]"
            )
        return warnings

    def should_rollback(self) -> bool:
        """True if consecutive failures exceed threshold."""
        return self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    def _baseline_eligible(self, result: EvalResult) -> tuple[bool, str, dict]:
        """(eligible, reason, proof) for a production-baseline write. Eligible iff a recognized
        speed_metric_mode is set AND the contention matrix is certified-fresh against the LIVE
        topology (matrix_status==OK with the live hash). Fail-closed: if the live topology/matrix
        cannot be determined we are NOT eligible — a baseline must never be written on unknown
        state (operator audit #3, 2026-05-27). No env override by design (hard gate)."""
        # B8 / SG-0: emit the pre-expiry countdown wherever matrix freshness is consulted so
        # the operator hears about an impending ratchet freeze BEFORE the matrix goes STALE.
        _warn_matrix_pre_expiry()
        proof: dict[str, Any] = {
            "speed_metric_mode": getattr(result, "speed_metric_mode", None),
            "eval_concurrency": getattr(result, "eval_concurrency", None),
        }
        # C5 (commit 9204d6b7): when a partition (audit shadow / tool_sentinel) is
        # excluded from the decision subset, eval_tower stamps a `_partition_filtered`
        # provenance suffix on the SAME metric definition — median request TPS, now
        # computed over the decision subset rather than the full mixed batch. It is a
        # provenance marker, NOT a new instrument, so strip the known suffix before the
        # recognized-mode allowlist check. Without this the documented-default regime
        # (audit blocks active) would refuse EVERY baseline write — an unintended total
        # freeze of the baseline ratchet (loud/fail-closed, but wrong).
        mode = proof["speed_metric_mode"]
        base_mode = (
            mode[: -len("_partition_filtered")]
            if isinstance(mode, str) and mode.endswith("_partition_filtered")
            else mode
        )
        if base_mode not in {"median_request_tps", "aggregate_batch_tps"}:
            return False, f"unrecognized speed_metric_mode={mode!r}", proof
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
            from src.scheduling.contention import (
                load_contention_matrix,
                matrix_status,
                MatrixStatus,
                topology_fingerprint_for_matrix,
            )

            matrix = load_contention_matrix()
            live = topology_fingerprint_for_matrix(NUMA_CONFIG, matrix)
            status = matrix_status(current_topology_hash=live)
            proof["topology_hash"] = live
            proof["matrix_status"] = status.value
            if status != MatrixStatus.OK:
                return False, f"matrix not certified-fresh (status={status.value})", proof
        except Exception as exc:  # noqa: BLE001
            proof["error"] = str(exc)
            return False, f"could not verify topology/matrix: {exc}", proof
        return True, "speed_metric_mode set + matrix OK against live topology", proof

    @staticmethod
    def _archive_best_quality(tier: int | None = None) -> float | None:
        """Max quality on the live same-tier Pareto frontier, or None if it cannot be read.

        Fail-soft: a missing/unreadable archive returns None (the caller skips the archive-max
        guard but the scale + eligibility gates still apply), so this can never block a
        legitimate bootstrap write on a fresh state."""
        return _pareto_frontier_best_quality(tier=tier)

    @staticmethod
    def _archive_frontier_trial_ids(tier: int | None = None) -> frozenset[int]:
        """Trial ids currently on the same-tier Pareto frontier (empty set if empty/unreadable)."""
        ctx = _pareto_frontier_context(tier=tier)
        return ctx[1] if ctx is not None else frozenset()

    @staticmethod
    def _archive_frontier_entry(
        source_trial_id: int | None, tier: int | None = None
    ) -> dict[str, Any] | None:
        """Same-tier frontier entry for source_trial_id, reduced to stable evidence fields."""
        if source_trial_id is None:
            return None
        archive = _pareto_archive_for_safety_guard()
        if archive is None:
            return None
        for entry in archive.frontier(tier=tier):
            if int(entry.trial_id) == int(source_trial_id):
                return {
                    "trial_id": int(entry.trial_id),
                    "objectives": tuple(float(x) for x in entry.objectives),
                    "n_reproductions": int(getattr(entry, "n_reproductions", 1) or 1),
                    "config_fingerprint": getattr(entry, "config_fingerprint", ""),
                }
        return None

    @staticmethod
    def _quality_quantum(result: EvalResult) -> float | None:
        """Smallest observable quality step for this eval on the 0-3 scale."""
        n = int(getattr(result, "n_questions", 0) or 0)
        if n <= 0:
            counts = getattr(result, "per_suite_counts", None) or {}
            n = sum(int(v) for v in counts.values() if int(v) > 0)
        if n <= 0 and isinstance(getattr(result, "details", None), dict):
            try:
                n = int(result.details.get("total", 0) or 0)
            except (TypeError, ValueError):
                n = 0
        return (3.0 / n) if n > 0 else None

    def update_baseline(
        self,
        result: EvalResult,
        source_trial_id: int | None = None,
        *,
        seq_confirmed: bool | None = None,
    ) -> BaselineUpdateResult:
        """Update same-tier baseline state with new production-best metrics — HARD-GATED on baseline eligibility
        (operator audit #3): refuses to write unless a recognized speed_metric_mode is set AND the
        contention matrix is certified-fresh against the live topology, so a measurement taken on a
        stale/wrong stack (or with unknown concurrent-speed semantics) can never poison the baseline.
        The baseline_eligible decision + topology/matrix proof are logged either way.

        PRECONDITION (archive-first ordering): `result` must already be admitted to the Pareto
        archive via archive.update() BEFORE promotion, and its `source_trial_id` passed here. The
        archive-max guard refuses any quality above the frontier max unless the source trial is
        actually on the frontier — this catches both phantom/contaminated measurements (never
        archived) and a caller that promotes a genuine new-best before archiving it. Promote in
        the order: archive.update(entry) → update_baseline(result, source_trial_id=entry.trial_id).

        The update is state-only: the YAML baseline remains a seed config; promotions live in
        autopilot_state.json via Baseline.to_state_dict().
        """
        tier = int(result.tier)
        # SG-3 (B3a): the monotonic gate compares against the STRICT same-tier baseline —
        # no cross-tier legacy fallback. When it is None (no prior same-tier baseline) the
        # monotonic check below is skipped and update_tier SEEDS the tier baseline.
        previous_quality = self.baseline.quality_for_tier(tier, strict=True)
        # EV-14c: pin the reference this promotion's gates compare against, so the
        # monotonic/quantum verdicts are provably rendered against the reference that
        # still exists at write time. Unregistered: this window IS the writer, and its
        # own write must not be reported against itself.
        entry_pin = self.baseline.pin_tier(tier, register=False)
        eligible, reason, proof = self._baseline_eligible(result)
        if not eligible:
            # B8 / SG-0: a frozen baseline ratchet is a loud failure, not a quiet skip —
            # nothing can promote until the operator refreshes the matrix, so log at ERROR
            # with the remediation and surface the reason distinctly via ineligible_reason.
            log.error(
                "Baseline update REFUSED — baseline_eligible=false (%s) | proof=%s | "
                "remediation: re-measure/refresh the contention matrix so it is "
                "certified-fresh against the live topology; the baseline ratchet stays "
                "FROZEN (no promotion possible) until then.",
                reason,
                proof,
            )
            return BaselineUpdateResult(
                False, reason, tier, previous_quality, result.quality, proof,
                ineligible_reason=reason,
            )
        # Defect #3: eval-instrument re-baseline hold. A promotion computed against (or that
        # would overwrite) a pre-boundary baseline crosses the eval-instrument boundary — a
        # scorer/pool change could manufacture a spurious promotion. REFUSE until an operator
        # reseeds a same-era baseline (the documented remediation; the era stamp then matches
        # and this clears). Fail-closed, loud. Inert when no active eval_quality era is set.
        if self.quality_rebaseline_required:
            reason = (
                "eval-instrument RE-BASELINE required: resident baseline era="
                f"{self.baseline.eval_quality_era or '<pre-boundary>'} != active eval_quality "
                f"era={self._eval_quality_era}; quality promotion REFUSED (defect #3, "
                "fail-closed). Remediation: reseed baseline_state from a post-boundary eval "
                "(eval_quality_era must equal the active era) before quality can ratchet."
            )
            log.error("Baseline update REFUSED — %s", reason)
            return BaselineUpdateResult(
                False, reason, tier, previous_quality, result.quality, proof,
                ineligible_reason="quality_rebaseline_required",
            )
        # LEDGER-W4 (01c §3): when the sequential path is active, a promotion requires
        # a CONFIRMED joint e-process verdict (E_quality >= confirm_e AND
        # E_rate_noninf >= confirm_e). This is the anti-ratchet: a monotonic quality
        # uptick that has not cleared the anytime-valid thresholds is journaled but
        # cannot move the baseline. Inert when the flag is off (the legacy promotion
        # path is then unchanged).
        #
        # B4 / SEQ-2 (anti-ratchet, fail-closed; operator-decided 2026-07-20): when the
        # sequential path is ON but seq_confirmed is None, the per-question inputs were
        # UNAVAILABLE (fresh journal, missing question_results) — exactly the low-evidence
        # regime the anti-ratchet exists for. The old code let None silently fall through
        # to the legacy monotonic path, ratcheting the baseline on evidence it could not
        # verify. REFUSE the write and surface seq_inputs_unavailable, restoring the
        # docstring's stated intent (a candidate cannot ratchet on quality alone).
        if self.use_sequential and seq_confirmed is None:
            reason = (
                "sequential verdict UNAVAILABLE (no per-question evidence: fresh journal "
                "or missing question_results); baseline promotion REFUSED (B4/SEQ-2 "
                "anti-ratchet, fail-closed). Remediation: accumulate paired-core journal "
                "evidence so the anytime-valid e-process can render a confirmed/refuted "
                "verdict before this candidate can ratchet the baseline."
            )
            log.warning("Baseline update REFUSED — %s", reason)
            return BaselineUpdateResult(
                False, reason, tier, previous_quality, result.quality, proof,
                seq_refused_reason="seq_inputs_unavailable",
            )
        if self.use_sequential and seq_confirmed is not None and not seq_confirmed:
            reason = (
                "sequential verdict not confirmed (E_quality/E_rate_noninf below the "
                "confirm threshold); baseline promotion blocked (LEDGER-W4)"
            )
            log.info("Baseline update skipped — %s", reason)
            return BaselineUpdateResult(
                False, reason, tier, previous_quality, result.quality, proof,
                seq_refused_reason="seq_not_confirmed",
            )
        if tier < MIN_FRONTIER_EVAL_TIER:
            reason = f"tier {tier} is audit-only and cannot update production baselines"
            log.warning("Baseline update REFUSED — %s", reason)
            return BaselineUpdateResult(
                False, reason, tier, previous_quality, result.quality, proof
            )
        if not 0.0 <= result.quality <= QUALITY_MAX:
            log.error(
                "Baseline update REFUSED — result.quality %.3f outside valid scale "
                "[0, %.1f]; refusing to persist a corrupt/wrong-scale baseline",
                result.quality,
                QUALITY_MAX,
            )
            return BaselineUpdateResult(
                False, "quality outside valid scale", tier, previous_quality, result.quality, proof
            )
        if previous_quality is not None and result.quality <= previous_quality:
            reason = (
                f"not a monotonic same-tier improvement: T{tier} q={result.quality:.3f} "
                f"<= baseline {previous_quality:.3f}"
            )
            log.info("Baseline update skipped — %s", reason)
            return BaselineUpdateResult(
                False, reason, tier, previous_quality, result.quality, proof
            )
        archive_max = self._archive_best_quality(tier)
        if archive_max is not None and result.quality > archive_max + BASELINE_ARCHIVE_TOLERANCE:
            # Above the frontier max. A genuine new-best must be archived FIRST (archive-first
            # precondition), so its source trial would already be on the frontier; if it is not,
            # this is either a phantom/contaminated measurement or a caller that skipped
            # archive.update(). Refuse either way — accepting it would force-revert every honest
            # trial and gate-lock the loop.
            on_frontier = (
                source_trial_id is not None
                and source_trial_id in self._archive_frontier_trial_ids(tier)
            )
            if not on_frontier:
                log.error(
                    "Baseline update REFUSED — result.quality %.3f exceeds Pareto archive "
                    "max %.3f for T%d and source_trial_id=%s is not on the frontier. Promote only "
                    "AFTER archive.update() admits the trial; an above-max value with no "
                    "archived source is a phantom/contaminated measurement that would "
                    "force-revert every honest trial and gate-lock the loop.",
                    result.quality,
                    archive_max,
                    tier,
                    source_trial_id,
                )
                return BaselineUpdateResult(
                    False,
                    "quality exceeds same-tier archive max",
                    tier,
                    previous_quality,
                    result.quality,
                    proof,
                )
        promotion_result = result
        if archive_max is not None:
            quantum = self._quality_quantum(result)
            if quantum is None:
                reason = "missing n_questions/per-suite counts for reproduced baseline promotion"
                log.warning("Baseline update REFUSED — %s", reason)
                return BaselineUpdateResult(
                    False, reason, tier, previous_quality, result.quality, proof
                )
            repro_entry = self._archive_frontier_entry(source_trial_id, tier)
            if repro_entry is None:
                reason = (
                    "source trial is not a same-tier frontier representative; "
                    "baseline promotions require reproduced frontier evidence"
                )
                log.warning("Baseline update REFUSED — %s", reason)
                return BaselineUpdateResult(
                    False, reason, tier, previous_quality, result.quality, proof
                )
            objectives = tuple(repro_entry.get("objectives") or ())
            if len(objectives) < 4:
                reason = "frontier representative missing objective tuple"
                log.warning("Baseline update REFUSED — %s", reason)
                return BaselineUpdateResult(
                    False, reason, tier, previous_quality, result.quality, proof
                )
            n_reproductions = int(repro_entry.get("n_reproductions", 1) or 1)
            median_quality = float(objectives[0])
            required_quality = float(previous_quality or 0.0) + quantum
            if n_reproductions < BASELINE_PROMOTION_REPRO_MIN:
                reason = (
                    f"baseline promotion needs >= {BASELINE_PROMOTION_REPRO_MIN} "
                    f"reproductions; source has {n_reproductions}"
                )
                log.info("Baseline update skipped — %s", reason)
                return BaselineUpdateResult(
                    False, reason, tier, previous_quality, result.quality, proof
                )
            if median_quality + BASELINE_ARCHIVE_TOLERANCE < required_quality:
                reason = (
                    f"reproduced median q={median_quality:.3f} does not clear baseline "
                    f"{previous_quality:.3f} by one quantum ({quantum:.4f})"
                )
                log.info("Baseline update skipped — %s", reason)
                return BaselineUpdateResult(
                    False, reason, tier, previous_quality, result.quality, proof
                )
            promotion_result = replace(
                result,
                quality=median_quality,
                speed=float(objectives[1]),
                cost=-float(objectives[2]),
                reliability=float(objectives[3]),
            )
        if previous_quality is None:
            # SG-3 (B3a): explicit seed of a tier that had no strict same-tier baseline.
            # No cross-tier legacy fallback was consulted to gate this write.
            log.info(
                "Seeding T%d baseline from q=%.3f (no prior strict same-tier baseline).",
                tier,
                promotion_result.quality,
            )
        # EV-14c: refuse when the reference moved after promotion start. Every gate
        # above compared against ``previous_quality`` / ``entry_pin``; if the tier
        # revision changed since, that reference no longer exists and the verdict is a
        # compare-to-ghost. In-process this cannot fire between the pin and the write
        # (single-threaded), but it makes the read-to-write race structurally
        # impossible rather than merely unobserved — the same fail-closed shape as
        # the archive-max guard, one layer down.
        if self.baseline.pin_moved(entry_pin):
            reason = (
                "baseline reference moved during promotion (tier revision "
                f"{entry_pin.revision} -> {self.baseline.tier_revision(tier)}): the "
                "monotonic/quantum gates compared against a reference that no longer "
                "exists. REFUSED — re-run the promotion against the current baseline "
                "(EV-14c: a silently-moved reference is impossible)."
            )
            log.warning("Baseline update REFUSED — %s", reason)
            return BaselineUpdateResult(
                False, reason, tier, previous_quality, promotion_result.quality, proof
            )
        self.baseline.update_tier(promotion_result)
        # Defect #4: stamp the era this baseline was promoted under so a future boundary can
        # detect the cross-era condition (and so a post-reseed same-era promotion keeps the
        # stamp current). Only stamps when an active era is known; never clears an existing one.
        if self._eval_quality_era:
            self.baseline.eval_quality_era = self._eval_quality_era
        # 2026-08-03: stamp the SPEED era so a reseed closes the throughput hold naturally.
        # Gated on exactly the condition under which update_tier() actually rewrites
        # frontdoor_speed (frontier tier AND a positive speed sample) — stamping an era onto
        # a frontdoor_speed that was never re-measured would itself be a provenance lie, the
        # very failure mode this field exists to prevent.
        if (
            self._autopilot_speed_era
            and int(promotion_result.tier) == DEFAULT_FRONTIER_TIER
            and promotion_result.speed > 0
        ):
            self.baseline.autopilot_speed_era = self._autopilot_speed_era
        log.info(
            "Baseline state updated — baseline_eligible=true (%s) | proof=%s | T%d q=%.3f s=%.1f",
            reason,
            proof,
            tier,
            promotion_result.quality,
            promotion_result.speed,
        )
        return BaselineUpdateResult(
            True, reason, tier, previous_quality, promotion_result.quality, proof
        )

    def reset_failures(self) -> None:
        self._consecutive_failures = 0

    @staticmethod
    def analyze_failure(result: EvalResult, verdict: SafetyVerdict) -> str:
        """Build a structured failure narrative from safety verdict and eval result.

        Pure rule-based (no LLM). Returns empty string if verdict passed.
        """
        if verdict.passed:
            return ""

        sections: list[str] = []

        # VIOLATIONS
        if verdict.violations:
            lines = ["VIOLATIONS:"]
            for v in verdict.violations:
                lines.append(f"  - {v}")
            sections.append("\n".join(lines))

        # DEGRADED SUITES (per-suite quality below floor)
        #
        # 2026-08-03 — same defect class as `throughput_unmeasured` above: ABSENT data was
        # being rendered as a measured 0.000. Two distinct problems lived here.
        #
        # (a) Absence rendered as a number. A per-suite entry that is None / NaN is "not
        #     measured", not "scored zero". `None < floor` raised TypeError (killing the
        #     whole narrative) and `nan < floor` is False (silently dropping the suite);
        #     neither says "absent". A suite that drew questions but had NONE of them
        #     scored (all errored) never reaches per_suite_quality at all, so it vanished
        #     with no trace. Both are now labelled explicitly in their own section.
        #
        # (b) Sample-size-blind rendering. Every rejection printed ~13 lines of
        #     "<suite>: 0.000 (floor: 1.0)". That is not 13 regressions: on a 50-question
        #     hybrid eval most suites draw n=1, where the 0-3 score can ONLY be 0.0 or 3.0,
        #     so a single missed question is indistinguishable from a total collapse
        #     (observed on trials 1456/1458 — every "degraded" suite had n_result=1). The
        #     gating legs already reason about this (PER_SUITE_BINDING_MIN_COUNT,
        #     PER_SUITE_BASELINE_HARD_MIN_N); this narrative did not, and it is the copy
        #     the planner actually reads. Carry n and flag low-resolution entries so a
        #     quantization artifact can never masquerade as a measured regression.
        quality_floor = QUALITY_FLOOR_T0 if result.tier == 0 else QUALITY_FLOOR_T1
        per_suite = getattr(result, "per_suite_quality", None) or {}
        suite_counts = getattr(result, "per_suite_counts", None) or {}
        details = getattr(result, "details", None) or {}
        total_counts = details.get("per_suite_total_counts")
        if not isinstance(total_counts, dict):
            total_counts = {}

        degraded: list[tuple[str, float, int | None]] = []
        unmeasured: list[str] = []
        for suite, q in per_suite.items():
            if (
                q is None
                or isinstance(q, bool)
                or not isinstance(q, (int, float))
                or not math.isfinite(float(q))
            ):
                unmeasured.append(f"{suite}: not measured (no per-suite score recorded)")
                continue
            if float(q) < quality_floor:
                raw_n = suite_counts.get(suite)
                n = (
                    int(raw_n)
                    if isinstance(raw_n, int) and not isinstance(raw_n, bool)
                    else None
                )
                degraded.append((suite, float(q), n))
        # Suites that drew questions but produced no scoreable answer at all: present in the
        # eval's per-suite TOTALS, absent from its per-suite scores. Absent, never zero.
        for suite, n_total in sorted(total_counts.items()):
            if suite in per_suite:
                continue
            unmeasured.append(f"{suite}: not measured (0 of {n_total} questions scored)")

        if degraded:
            lines = ["DEGRADED SUITES:"]
            low_res = 0
            for suite, q, n in sorted(degraded, key=lambda x: (x[1], x[0])):
                n_label = "n=unknown" if n is None else f"n={n}"
                line = f"  - {suite}: {q:.3f} (floor: {quality_floor}, {n_label})"
                if n is None or n < PER_SUITE_BINDING_MIN_COUNT:
                    low_res += 1
                    line += " [low-resolution]"
                lines.append(line)
            if low_res:
                lines.append(
                    f"  NOTE: {low_res} of {len(degraded)} entries are low-resolution "
                    f"(n < {PER_SUITE_BINDING_MIN_COUNT}, or n unknown): the 0-3 suite score "
                    "is quantized to multiples of 3/n, so a single missed question renders "
                    "as 0.000. Read these as sampling resolution, NOT as measured "
                    "regressions."
                )
            sections.append("\n".join(lines))

        if unmeasured:
            lines = ["SUITES WITHOUT A MEASURED SCORE (absent — NOT a score of 0.000):"]
            for entry in sorted(unmeasured):
                lines.append(f"  - {entry}")
            sections.append("\n".join(lines))

        # ROUTING IMBALANCE (>60% to one tier)
        for tier_name, frac in result.routing_distribution.items():
            if frac > 0.6:
                sections.append(f"ROUTING IMBALANCE:\n  - {tier_name}: {frac:.1%} of requests")

        # WARNINGS
        if verdict.warnings:
            lines = ["WARNINGS:"]
            for w in verdict.warnings:
                lines.append(f"  - {w}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
