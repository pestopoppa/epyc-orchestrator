"""Safety gate: quality floor, regression guards, rollback triggers.

Loads frozen baseline from autopilot_baseline.yaml and enforces constraints.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from src.autopilot_core.tier_specs import (
    DEFAULT_FRONTIER_TIER,
    MIN_FRONTIER_EVAL_TIER,
)

log = logging.getLogger("autopilot.safety")


def _env_truthy(name: str) -> bool:
    """Return True when env var ``name`` is set to a truthy token."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "autopilot_baseline.yaml"
)

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
# Minimum per-suite sample below which the per-suite regression gate is purely
# advisory: at n questions a single correct→incorrect flip moves the 0-3 score by
# 3/n, so the *smallest observable* regression is already 3/n. When 3/n exceeds the
# fixed PER_SUITE_REGRESSION floor the gate cannot distinguish a real regression
# from one-question sampling noise — so the threshold is widened to that quantum.
# (2026-06-06: at ~2 q/suite the -0.1 floor fired -1.5 "regressions" on every
# seeder trial, mass-excluding via mad_noise and deadlocking the planner/critic.)
def per_suite_regression_threshold(
    result_n: int | None, baseline_n: int | None
) -> float:
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
ARCHITECT_ROUTING_CAP = 0.80  # Max fraction routed to architect-tier
MAX_CONSECUTIVE_FAILURES = 3  # Auto-rollback after this many failures
# MAD noise filter (intake-421 pi-autoresearch). Quality history depth + significance threshold.
MAD_HISTORY_DEPTH = 10
MAD_MIN_SAMPLES = 3  # Below this, skip MAD check (insufficient data → accept)
MAD_Z_THRESHOLD = 2.0  # Improvement counts as real only if > this many MADs from history median
MAD_CONSISTENCY = 1.4826  # Scaling so MAD ≈ σ under normal distribution
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
            log.error("Ignoring per-suite baseline tier key %r in %s; expected integer tier", tier, path)
            continue
        normalized[t] = {
            suite: Baseline._validate_quality(q, None, f"per_suite_quality_by_tier[{t}][{suite}]", path)
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
            log.error("Ignoring per-suite count tier key %r in %s; expected integer tier", tier, path)
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

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class EvalResult:
    """Evaluation result from EvalTower."""
    tier: int
    quality: float  # Average quality 0-3
    speed: float  # Objective speed t/s: median request in serial, aggregate batch in concurrent evals.
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
    # SafetyGate.check() is called by several action handlers and again by the
    # main loop. Cache the first verdict on the result so one trial mutates MAD
    # history / consecutive-failure state exactly once.
    gate_verdict: SafetyVerdict | None = field(default=None, repr=False, compare=False)

    @property
    def objectives(self) -> tuple[float, float, float, float]:
        return (self.quality, self.speed, -self.cost, self.reliability)

    def to_grep_lines(self, trial_id: int = 0, species: str = "") -> str:
        """AP-13: Grep-parseable key: value output.

        Designed for `grep 'METRIC' autopilot.log | awk -F': '` extraction.
        """
        lines = [
            f"METRIC trial: {trial_id}",
            f"METRIC species: {species}",
            f"METRIC tier: {self.tier}",
            f"METRIC quality: {self.quality:.4f}",
            f"METRIC speed: {self.speed:.2f}",
            f"METRIC speed_metric_mode: {self.speed_metric_mode}",
            f"METRIC median_request_speed: {self.median_request_speed:.2f}",
            f"METRIC aggregate_speed: {self.aggregate_speed:.2f}",
            f"METRIC eval_concurrency: {self.eval_concurrency}",
            f"METRIC eval_wall_s: {self.eval_wall_s:.2f}",
            f"METRIC cost: {self.cost:.4f}",
            f"METRIC reliability: {self.reliability:.4f}",
            f"METRIC n_questions: {self.n_questions}",
        ]
        if self.core_id:
            lines.append(f"METRIC core_id: {self.core_id}")
        for suite, q in sorted(self.per_suite_quality.items()):
            lines.append(f"METRIC suite_{suite}: {q:.4f}")
        for role, frac in sorted(self.routing_distribution.items()):
            lines.append(f"METRIC route_{role}: {frac:.4f}")
        # AP-16: Instruction token budget
        lines.append(f"METRIC instruction_tokens: {self.instruction_token_count}")
        lines.append(f"METRIC instruction_ratio: {self.instruction_token_ratio:.4f}")
        # Degradation metrics from refactored InferenceResult
        if self.partial_count > 0:
            lines.append(f"METRIC partial_count: {self.partial_count}")
        if self.degraded_count > 0:
            lines.append(f"METRIC degraded_count: {self.degraded_count}")
        # EV-2: Calibration metrics
        lines.append(f"METRIC ece: {self.ece:.4f}")
        if self.auroc > 0:
            lines.append(f"METRIC auroc: {self.auroc:.4f}")
        if self.calibration_violations > 0:
            lines.append(f"METRIC calibration_violations: {self.calibration_violations}")
        # Branching density (intake-378)
        if self.branching_density > 0:
            lines.append(f"METRIC branching_density: {self.branching_density:.4f}")
        # AM compaction telemetry
        if self.avg_prompt_tokens > 0:
            lines.append(f"METRIC avg_prompt_tokens: {self.avg_prompt_tokens:.0f}")
        if self.compaction_events > 0:
            lines.append(f"METRIC compaction_events: {self.compaction_events}")
        # EV-8: Diversity metrics (NaN-gated — only emit when the signal was
        # actually computed; NaN means "unavailable this trial").
        for _div_key, _div_val in (
            ("diversity_entropy", self.diversity_entropy),
            ("diversity_distinct2", self.diversity_distinct2),
            ("diversity_self_bleu", self.diversity_self_bleu),
            ("diversity_ttr", self.diversity_ttr),
            ("diversity_semantic_embedding_agreement",
             self.diversity_semantic_embedding_agreement),
        ):
            if not math.isnan(_div_val):
                lines.append(f"METRIC {_div_key}: {_div_val:.4f}")
        return "\n".join(lines)


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
    # Path this baseline was loaded from. save() writes back here by default so a
    # gate constructed with a custom baseline_path (e.g. a tmp file in tests) can
    # NEVER clobber the production orchestration/autopilot_baseline.yaml. Excluded
    # from equality so two baselines with the same metrics still compare equal.
    # (2026-05-31: a test fixture's update_baseline() wrote quality=2.9 to the real
    #  baseline via the DEFAULT_BASELINE_PATH fallback, gate-locking the live loop.)
    source_path: Path | None = field(default=None, compare=False, repr=False)

    @classmethod
    def load(cls, path: Path | None = None, state: dict[str, Any] | None = None) -> Baseline:
        path = path or DEFAULT_BASELINE_PATH
        if not path.exists():
            log.warning("No baseline file at %s, using defaults", path)
            baseline = cls(source_path=path)
            if state:
                baseline.apply_state(state, path)
            return baseline
        data = yaml.safe_load(path.read_text())
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
        if (quality is not None and archive_max is not None
                and quality > archive_max + BASELINE_ARCHIVE_TOLERANCE):
            log.error(
                "Persisted baseline quality %.3f in %s exceeds Pareto archive max %.3f — "
                "unachievable/corrupt; it would gate-lock the loop. Falling back to %.3f. "
                "Recompute the baseline from a real eval (autopilot.py checkpoint --production-best).",
                quality, path, archive_max, defaults.quality,
            )
            quality = defaults.quality
        per_suite = {
            suite: cls._validate_quality(q, None, f"per_suite[{suite}]", path)
            for suite, q in (data.get("per_suite_quality", {}) or {}).items()
        }
        baselines_by_tier = _normalize_float_by_tier(data.get("baselines_by_tier", {}), path)
        for tier, tier_quality in list(baselines_by_tier.items()):
            tier_archive_max = _pareto_frontier_best_quality(tier)
            if tier_archive_max is None:
                continue
            if tier_quality > tier_archive_max + BASELINE_ARCHIVE_TOLERANCE:
                log.error(
                    "Persisted T%d baseline quality %.3f in %s exceeds same-tier Pareto "
                    "archive max %.3f — unachievable/corrupt; dropping tier baseline.",
                    tier, tier_quality, path, tier_archive_max,
                )
                del baselines_by_tier[tier]
        per_suite_by_tier = _normalize_suite_by_tier(
            data.get("per_suite_quality_by_tier", {}), path
        )
        per_suite_counts_by_tier = _normalize_counts_by_tier(
            data.get("per_suite_counts_by_tier", {}), path
        )
        reliability = data.get("reliability", defaults.reliability)
        if reliability is not None and not 0.0 <= reliability <= RELIABILITY_MAX:
            log.error(
                "Corrupt baseline reliability %.3f in %s (valid 0..%.1f); using default %.3f",
                reliability, path, RELIABILITY_MAX, defaults.reliability,
            )
            reliability = defaults.reliability
        baseline = cls(
            quality=quality,
            speed=data.get("speed", 10.0),
            cost=data.get("cost", 0.5),
            reliability=reliability,
            per_suite_quality=per_suite,
            baselines_by_tier=baselines_by_tier,
            per_suite_quality_by_tier=per_suite_by_tier,
            per_suite_counts_by_tier=per_suite_counts_by_tier,
            frontdoor_speed=data.get("frontdoor_speed", 10.0),
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
        if not 0.0 <= value <= QUALITY_MAX:
            log.error(
                "Corrupt baseline %s=%.3f in %s exceeds valid quality scale [0, %.1f] — "
                "this would force-revert every trial via the regression gate; "
                "falling back to %s. Recompute the baseline from a real 0-3 eval.",
                label, value, path, QUALITY_MAX, fallback,
            )
            return fallback
        return value

    def save(self, path: Path | None = None) -> None:
        path = path or self.source_path or DEFAULT_BASELINE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "quality": self.quality,
            "speed": self.speed,
            "cost": self.cost,
            "reliability": self.reliability,
            "per_suite_quality": self.per_suite_quality,
            "baselines_by_tier": self.baselines_by_tier,
            "per_suite_quality_by_tier": self.per_suite_quality_by_tier,
            "per_suite_counts_by_tier": self.per_suite_counts_by_tier,
            "frontdoor_speed": self.frontdoor_speed,
        }
        path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))

    def apply_state(self, state: dict[str, Any], path: Path | None = None) -> None:
        state_path = path or self.source_path or DEFAULT_BASELINE_PATH
        self.baselines_by_tier.update(
            _normalize_float_by_tier(state.get("baselines_by_tier", {}), state_path)
        )
        self.per_suite_quality_by_tier.update(
            _normalize_suite_by_tier(state.get("per_suite_quality_by_tier", {}), state_path)
        )
        self.per_suite_counts_by_tier.update(
            _normalize_counts_by_tier(state.get("per_suite_counts_by_tier", {}), state_path)
        )

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

    def update_tier(self, result: EvalResult) -> None:
        tier = int(result.tier)
        self.baselines_by_tier[tier] = result.quality
        self.per_suite_quality_by_tier.setdefault(tier, {}).update(result.per_suite_quality)
        result_counts = getattr(result, "per_suite_counts", None) or {}
        if result_counts:
            self.per_suite_counts_by_tier.setdefault(tier, {}).update(result_counts)
        if tier == DEFAULT_FRONTIER_TIER:
            self.quality = result.quality
            self.per_suite_quality.update(result.per_suite_quality)
            if result.speed > 0:
                self.frontdoor_speed = result.speed
        self.speed = result.speed
        self.cost = result.cost
        self.reliability = result.reliability

    def to_state_dict(self) -> dict[str, Any]:
        """State payload for in-memory baseline promotions; YAML remains the seed config."""
        return {
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


@dataclass(frozen=True)
class BaselineUpdateResult:
    updated: bool
    reason: str
    tier: int
    previous_quality: float | None
    new_quality: float
    proof: dict[str, Any] = field(default_factory=dict)


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
    ):
        self.baseline = Baseline.load(baseline_path, state=baseline_state)
        # LEDGER-W4 (01c §3): default-off anytime-valid sequential verdict path.
        # Flag source precedence: explicit arg > AUTOPILOT_SEQ_VERDICT env > off.
        # When off (the default), check()/update_baseline behave byte-identically to
        # the pre-W4 MAD-only gate. Deploy is evidence-gated (flip-rate >= 30% over
        # ~120 trusted vectors); flipping the env alone never changes behavior unless
        # the caller also threads per-question results into check().
        self.use_sequential = (
            _env_truthy("AUTOPILOT_SEQ_VERDICT")
            if use_sequential is None
            else bool(use_sequential)
        )
        self._consecutive_failures = consecutive_failures
        self._quality_history_by_tier: dict[int, deque[float]] = {}
        if quality_history_by_tier:
            for tier, history in quality_history_by_tier.items():
                self._quality_history_by_tier[int(tier)] = deque(
                    history or [], maxlen=MAD_HISTORY_DEPTH
                )
        if quality_history:
            # Legacy flat state had no tier label. Seed all current tiers so resumes and
            # older unit fixtures preserve their pre-migration behavior until enough
            # same-tier samples replace the migrated window.
            for tier in (0, DEFAULT_FRONTIER_TIER, 2):
                self._quality_history_by_tier.setdefault(
                    tier, deque(quality_history, maxlen=MAD_HISTORY_DEPTH)
                )
        self._last_history_tier = DEFAULT_FRONTIER_TIER

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def quality_history(self) -> list[float]:
        """Return the latest tier's rolling quality window (legacy state persistence)."""
        return self.quality_history_for_tier(self._last_history_tier)

    @property
    def quality_history_by_tier(self) -> dict[str, list[float]]:
        """Return rolling quality windows keyed by eval tier for state persistence."""
        return {
            str(tier): list(history)
            for tier, history in sorted(self._quality_history_by_tier.items())
        }

    def quality_history_for_tier(self, tier: int) -> list[float]:
        return list(self._quality_history_by_tier.get(int(tier), deque()))

    def _history_for_tier(self, tier: int) -> deque[float]:
        return self._quality_history_by_tier.setdefault(
            int(tier), deque(maxlen=MAD_HISTORY_DEPTH)
        )

    def _mad_significance(self, new_quality: float, tier: int) -> tuple[bool, float, float, float]:
        """Decide whether ``new_quality`` is statistically significant vs history.

        Returns (is_significant, z_mad, median, mad). When history < MAD_MIN_SAMPLES,
        returns (True, NaN, NaN, NaN) — insufficient data to filter, accept at face value.
        """
        history = self._history_for_tier(tier)
        if len(history) < MAD_MIN_SAMPLES:
            return True, math.nan, math.nan, math.nan
        median_q = statistics.median(history)
        mad = statistics.median(abs(x - median_q) for x in history)
        if mad == 0:
            return (new_quality != median_q), math.nan, median_q, 0.0
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
        q_state, q_update = q_view.quality_state.update(
            stat.z, policy=policy, trial_id=trial_id
        )

        # Rate-non-inferiority axis (only when a task_rate + positive baseline exist).
        rate_state = None
        rate_update = None
        if (
            task_rate is not None
            and baseline_task_rate is not None
            and baseline_task_rate > 0
        ):
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

        if q_name == STATE_REFUTED or rate_name == STATE_REFUTED:
            state = "refuted"
        elif e_quality >= policy.confirm_e and (
            e_rate is not None and e_rate >= policy.confirm_e
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
        if result.gate_verdict is not None:
            return result.gate_verdict

        violations = []
        warnings = []
        categories = []  # AP-14: track which checks failed
        seq_block: dict[str, Any] | None = None  # LEDGER-W4 journal block (default-off)

        # 1. Quality floor (tier-aware)
        quality_floor = QUALITY_FLOOR_T0 if result.tier == 0 else QUALITY_FLOOR_T1
        if result.quality < quality_floor:
            violations.append(
                f"Quality floor violation: {result.quality:.3f} < {quality_floor} (tier {result.tier})"
            )
            categories.append("quality_floor")

        # 2. Regression vs baseline (relative: allow 5% drop from baseline)
        baseline_q = self.baseline.quality_for_tier(result.tier)
        if baseline_q is not None and baseline_q > 0:
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
            elif self.use_sequential and question_results is not None and baseline_profile:
                # LEDGER-W4 (01c §3): when the default-off sequential path is active
                # and the caller threads per-question results, the anytime-valid
                # e-process verdict REPLACES the MAD noise filter as the "is this a
                # real improvement?" significance test. It appends one of
                # seq_accumulating / seq_confirmed / seq_refuted (consumed by
                # classify_learning_exclusion) and records the full E-process journal
                # block on the verdict. It never adds a violation — promotion is gated
                # downstream in update_baseline(seq_confirmed=...), so a non-confirmed
                # candidate is journaled but cannot ratchet the baseline.
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
                warnings.append(
                    "Sequential verdict ({state}): E_quality={eq:.3f}, "
                    "E_rate_noninf={er}, k={k} (LEDGER-W4, AUTOPILOT_SEQ_VERDICT); "
                    "baseline promotion {gate}.".format(
                        state=seq_block["state"],
                        eq=seq_block.get("E_quality", float("nan")),
                        er=f"{e_rate:.3f}" if e_rate is not None else "n/a",
                        k=seq_block.get("k"),
                        gate=(
                            "permitted (confirmed)"
                            if seq_block.get("confirmed")
                            else "blocked until confirmed"
                        ),
                    )
                )
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
                if not is_sig and not math.isnan(z_mad):
                    categories.append("mad_noise")
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
                    base_q = self.baseline.quality_for_tier(result.tier)
                    reproduction_confirmed = (
                        base_q is not None
                        and base_q > 0
                        and not math.isnan(median_q)
                        and mad > 0
                        and (median_q - base_q)
                        > MAD_Z_THRESHOLD * mad * MAD_CONSISTENCY
                    )
                    if reproduction_confirmed:
                        categories.append("reproduction_confirmed")
                    convergence_note = (
                        " Reproduces an established above-baseline level "
                        "(history median {:.3f} >> baseline {:.3f}): this is a "
                        "convergence/confirmation of an existing gain, NOT "
                        "instrument noise or a corrupted trial.".format(
                            median_q, base_q
                        )
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
        baseline_suites = self.baseline.per_suite_for_tier(result.tier)
        baseline_counts = self.baseline.per_suite_counts_for_tier(result.tier)
        result_counts = getattr(result, "per_suite_counts", None) or {}
        for suite, quality in result.per_suite_quality.items():
            baseline_q = baseline_suites.get(suite)
            if baseline_q is not None:
                suite_delta = quality - baseline_q
                threshold = per_suite_regression_threshold(
                    result_counts.get(suite), baseline_counts.get(suite)
                )
                if suite_delta < threshold:
                    violations.append(
                        f"Suite '{suite}' regression: {suite_delta:+.3f} "
                        f"(threshold: {threshold:+.3f}; "
                        f"n_result={result_counts.get(suite)}, "
                        f"n_baseline={baseline_counts.get(suite)})"
                    )
                    if "per_suite_regression" not in categories:
                        categories.append("per_suite_regression")

        # 4. Routing diversity
        architect_frac = result.routing_distribution.get("architect", 0.0)
        if architect_frac > ARCHITECT_ROUTING_CAP:
            violations.append(
                f"Routing diversity violation: {architect_frac:.1%} architect-tier "
                f"(cap: {ARCHITECT_ROUTING_CAP:.0%})"
            )
            categories.append("routing_diversity")

        # 5. Throughput floor
        if result.speed < self.baseline.frontdoor_speed * 0.8:
            # 2026-05-09: before attributing a throughput regression to the
            # config-under-test, check whether the host is itself throttled
            # (CPU freq dip / page-cache fragmentation per
            # feedback_host_throttle_check.md). If yes, run drop_caches and
            # tag the verdict for autopilot to retry the trial. This avoids
            # contaminating the Pareto archive with false-negative entries
            # caused by sustained mlocked load (the 2026-05-09 incident:
            # frontdoor measured 7.48 t/s = 1/3 of expected after 9 hours).
            host_throttled = False
            host_remediated = False
            host_triggers: list[str] = []
            try:
                from scripts.autopilot.host_health import (
                    HostHealthState,
                    remediate as _hh_remediate,
                    _numa_interleave_rewarm,
                )
                _hh_state = HostHealthState.snapshot()
                host_throttled, host_triggers = _hh_state.is_throttled()
                if host_throttled:
                    # In-process safety_gate path: trial already ran, so no need
                    # to pause autopilot (we ARE autopilot). Flush + rewarm so
                    # the NEXT trial starts with warm NUMA-interleaved cache;
                    # mark THIS trial as exogenous_cache_flush so the planner
                    # doesn't learn from data taken in the suspected cold-cache
                    # window (see DeficiencyCategory.EXOGENOUS_CACHE_FLUSH).
                    host_remediated = _hh_remediate()
                    if host_remediated:
                        _numa_interleave_rewarm()
            except Exception:  # noqa: BLE001 — never let host check crash gate
                pass

            base_msg = (
                f"Throughput floor: {result.speed:.1f} t/s < "
                f"{self.baseline.frontdoor_speed * 0.8:.1f} t/s "
                f"(80% of baseline {self.baseline.frontdoor_speed:.1f})"
            )
            if host_throttled and host_remediated:
                # Soft-fail: throttle was the likely cause; retry next tick.
                # 2026-05-24: tag the trial with EXOGENOUS_CACHE_FLUSH so the
                # planner's trustworthiness gate excludes it from hypothesis
                # chains (data was taken under suspect host state).
                warnings.append(
                    f"{base_msg}. Host throttle detected ({'; '.join(host_triggers)}); "
                    f"drop_caches + NUMA-interleave rewarm issued — RECOMMEND retry."
                )
                categories.append("throughput_host_throttle_retry")
                categories.append("exogenous_cache_flush")
            elif host_throttled:
                # Throttled but couldn't remediate — still flag for retry, log the gap.
                warnings.append(
                    f"{base_msg}. Host throttle detected but remediation unavailable "
                    f"({'; '.join(host_triggers)}); install per "
                    f"scripts/autopilot/host_health_install.md."
                )
                categories.append("throughput_host_throttle_no_remediate")
            else:
                violations.append(base_msg)
                categories.append("throughput")
        elif result.speed < self.baseline.frontdoor_speed * 0.9:
            warnings.append(
                f"Speed marginal: {result.speed:.1f} t/s "
                f"({result.speed / self.baseline.frontdoor_speed:.0%} of baseline)"
            )

        # 6. Proxy-only improvement detection (skeptical re-questioning)
        warnings.extend(self._proxy_check(result))

        passed = len(violations) == 0
        verdict = SafetyVerdict(
            passed=passed,
            violations=violations,
            warnings=warnings,
            categories=categories,
            seq=seq_block,
        )

        # Track consecutive failures
        if not passed:
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
        if passed and not math.isnan(result.quality) and result.quality >= 0:
            self._history_for_tier(result.tier).append(result.quality)
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
                f"Proxy-only improvement: gains in [{imp_str}] "
                f"but declines in [{dec_str}]"
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
        proof: dict[str, Any] = {"speed_metric_mode": getattr(result, "speed_metric_mode", None),
                                 "eval_concurrency": getattr(result, "eval_concurrency", None)}
        if proof["speed_metric_mode"] not in {"median_request_tps", "aggregate_batch_tps"}:
            return False, f"unrecognized speed_metric_mode={proof['speed_metric_mode']!r}", proof
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
            from src.scheduling.contention import topology_fingerprint, matrix_status, MatrixStatus
            live = topology_fingerprint(NUMA_CONFIG)
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
        previous_quality = self.baseline.quality_for_tier(tier)
        eligible, reason, proof = self._baseline_eligible(result)
        if not eligible:
            log.warning("Baseline update REFUSED — baseline_eligible=false (%s) | proof=%s",
                        reason, proof)
            return BaselineUpdateResult(False, reason, tier, previous_quality, result.quality, proof)
        # LEDGER-W4 (01c §3): when the sequential path is active, a promotion requires
        # a CONFIRMED joint e-process verdict (E_quality >= confirm_e AND
        # E_rate_noninf >= confirm_e). This is the anti-ratchet: a monotonic quality
        # uptick that has not cleared the anytime-valid thresholds is journaled but
        # cannot move the baseline. Inert when the flag is off or when the caller does
        # not pass seq_confirmed (the legacy promotion path is then unchanged).
        if self.use_sequential and seq_confirmed is not None and not seq_confirmed:
            reason = (
                "sequential verdict not confirmed (E_quality/E_rate_noninf below the "
                "confirm threshold); baseline promotion blocked (LEDGER-W4)"
            )
            log.info("Baseline update skipped — %s", reason)
            return BaselineUpdateResult(False, reason, tier, previous_quality, result.quality, proof)
        if tier < MIN_FRONTIER_EVAL_TIER:
            reason = f"tier {tier} is audit-only and cannot update production baselines"
            log.warning("Baseline update REFUSED — %s", reason)
            return BaselineUpdateResult(False, reason, tier, previous_quality, result.quality, proof)
        if not 0.0 <= result.quality <= QUALITY_MAX:
            log.error("Baseline update REFUSED — result.quality %.3f outside valid scale "
                      "[0, %.1f]; refusing to persist a corrupt/wrong-scale baseline",
                      result.quality, QUALITY_MAX)
            return BaselineUpdateResult(
                False, "quality outside valid scale", tier, previous_quality, result.quality, proof
            )
        if previous_quality is not None and result.quality <= previous_quality:
            reason = (
                f"not a monotonic same-tier improvement: T{tier} q={result.quality:.3f} "
                f"<= baseline {previous_quality:.3f}"
            )
            log.info("Baseline update skipped — %s", reason)
            return BaselineUpdateResult(False, reason, tier, previous_quality, result.quality, proof)
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
                log.error("Baseline update REFUSED — result.quality %.3f exceeds Pareto archive "
                          "max %.3f for T%d and source_trial_id=%s is not on the frontier. Promote only "
                          "AFTER archive.update() admits the trial; an above-max value with no "
                          "archived source is a phantom/contaminated measurement that would "
                          "force-revert every honest trial and gate-lock the loop.",
                          result.quality, archive_max, tier, source_trial_id)
                return BaselineUpdateResult(
                    False, "quality exceeds same-tier archive max", tier,
                    previous_quality, result.quality, proof,
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
        self.baseline.update_tier(promotion_result)
        log.info("Baseline state updated — baseline_eligible=true (%s) | proof=%s | T%d q=%.3f s=%.1f",
                 reason, proof, tier, promotion_result.quality, promotion_result.speed)
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
        quality_floor = QUALITY_FLOOR_T0 if result.tier == 0 else QUALITY_FLOOR_T1
        degraded = [
            (suite, q)
            for suite, q in result.per_suite_quality.items()
            if q < quality_floor
        ]
        if degraded:
            lines = ["DEGRADED SUITES:"]
            for suite, q in sorted(degraded, key=lambda x: x[1]):
                lines.append(f"  - {suite}: {q:.3f} (floor: {quality_floor})")
            sections.append("\n".join(lines))

        # ROUTING IMBALANCE (>60% to one tier)
        for tier_name, frac in result.routing_distribution.items():
            if frac > 0.6:
                sections.append(
                    f"ROUTING IMBALANCE:\n  - {tier_name}: {frac:.1%} of requests"
                )

        # WARNINGS
        if verdict.warnings:
            lines = ["WARNINGS:"]
            for w in verdict.warnings:
                lines.append(f"  - {w}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
