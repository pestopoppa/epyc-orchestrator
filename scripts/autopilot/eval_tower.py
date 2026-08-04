"""Tiered evaluation tower: T0 (10q/30s) → T1 (100q/5m) → T2 (500+/30m) → T3 (expert/hard workflow).

Wraps existing seeding infrastructure for orchestrator API calls and scoring.
Training set (debug suites) is kept separate from validation set (HF benchmarks).

Generation/scoring pipeline (``_eval_batch`` workers>1 path)
------------------------------------------------------------
EV-4b measurement (HE-R+ code_execution, 2026-07-22): each concurrent lane ran
generation (~1s HTTP decode) THEN client-side scoring (~11s sandbox subprocess)
INLINE, so a lane was busy ~12s while decode duty was ~4-5% — the serving fleet
idled ~82% while only ``_eval_concurrency`` (topology-capped at 4 for inference
contention) candidates executed. Scoring-bound suites therefore capped total
throughput at the *serving* fan-out even though scoring is pure client CPU on a
192-thread host.

Fix (scheduling only — verdicts/scorer semantics unchanged): the workers>1 path
splits ``_eval_question`` into ``_generate_question`` (runs on the topology-capped
generation pool, width = ``_eval_concurrency``) and ``_score_generation`` (runs on
a separate, wider SCORING pool, width = ``AUTOPILOT_EVAL_SCORING_CONCURRENCY``,
default ``min(16, os.cpu_count()//12)`` but never below the generation width). A
generation lane hands its un-scored result to the scoring pool and immediately
starts the next question. Expected speedup model for a scoring-bound suite: once
generation stops gating, wall ≈ ``n * t_score / scoring_width`` (was
``n * (t_gen + t_score) / generation_width``). Math-shaped suites (decode-bound,
math_verify scoring ~instant) are unaffected — the scoring pool idles cheap, no
mode detection. A bounded un-scored queue (2x scoring width) backpressures a fast
generator so it cannot pile unbounded memory on a slow scorer. The serial path
(workers<=1) and every direct ``_eval_question`` caller keep the pre-split,
generate-then-score-inline behavior byte-for-byte.

Env knobs (see also AUTOPILOT_EVAL_CONCURRENCY / _NO_PROGRESS_TIMEOUT_S /
_ORPHAN_DRAIN_TIMEOUT_S / _BATCH_WALL_BUDGET_S):
  * ``AUTOPILOT_EVAL_SCORING_CONCURRENCY`` — client-side scoring-pool width for
    the workers>1 pipeline. Default ON; clamped to >= the generation width.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.util
import threading
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

import eval_tower_trace_feedback
from safety_gate import EvalResult

log = logging.getLogger("autopilot.eval")

SENTINEL_PATH = Path(__file__).resolve().parent / "sentinel_questions.yaml"
# Tool-use sentinels (suite: tool_use) — impossible to pass without a real
# read_file tool call. INERT unless AUTOPILOT_TOOL_SENTINELS=1 is set, so the
# live trial set is unchanged until a deliberate cutover (see tool_sentinels.yaml).
TOOL_SENTINEL_PATH = Path(__file__).resolve().parent / "tool_sentinels.yaml"
ORCHESTRATOR_URL = "http://localhost:8000"
EVAL_T1_SPEC_N = 100
EVAL_T2_SPEC_N = 500
EVAL_T3_SPEC_N = 160
EVAL_SPEC_SEED = 42
PROMOTION_EVAL_MIN_N = 200
PROMOTION_EVAL_MAX_N = 500
PROMOTION_EVAL_DEFAULT_N = 500
PROMOTION_EVAL_SUITE_HEALTH_GLOB = "item_analytics*.json"
_EXPECTED_FREE_SCORERS = {"programmatic"}
_CORE_METADATA_KEY = "__core_metadata__"
_SPEED_ANALYTICS_MIN_TOKENS = 128
_REQUIRED_EVAL_QUESTION_FIELDS = ("prompt", "expected", "suite")
_HOST_COVARIATE_COMPACT_KEYS = (
    "min_core_mhz",
    "mean_cur_mhz",
    "base_mhz",
    "host_inflight",
    "numa_balancing",
    "cache_warm_state",
    "page_cache_mb",
    "mem_available_mb",
)
_HOST_COVARIATE_NUMERIC_KEYS = (
    "min_core_mhz",
    "mean_cur_mhz",
    "base_mhz",
    "host_inflight",
    "numa_balancing",
    "page_cache_mb",
    "mem_available_mb",
    "loadavg_1min",
    "loadavg_per_core",
)
_HOST_COVARIATE_CATEGORICAL_KEYS = ("cache_warm_state",)
# Eval-instrument identity ledger.
#
# This was a module-level dict, which meant drift was only ever detectable WITHIN one
# daemon process. The drift that actually matters happens while the daemon is DOWN: on
# 2026-08-04 the debugbench python rows were retargeted `code_execution` -> `substring`
# in the question pool, the daemon restarted, and the in-memory ledger came back empty,
# so the changed instrument was silently accepted as the same one. `core_id` cannot
# catch it either — it is `legacy_pool_seed_{seed}_n{n}`, which is identical for two
# different pools.
#
# That matters far more now that the Pareto objective is questions/HOUR: the tier mix of
# the drawn set drives wall-clock directly (T2/T3 questions are much slower than T1), so
# a pool edit moves the objective with no config change at all. The ledger is therefore
# persisted, and the realized tier mix is stamped alongside the content hash.
_DATASET_SHA_BY_CORE_ID: dict[str, str] = {}
_INSTRUMENT_LEDGER_LOCK = threading.Lock()
_INSTRUMENT_LEDGER_PATH = Path(
    os.environ.get(
        "AUTOPILOT_EVAL_INSTRUMENT_LEDGER",
        str(Path(__file__).resolve().parents[2] / "orchestration" / "eval_instrument_ledger.json"),
    )
)


def question_tier_mix(questions: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Realized difficulty-tier histogram of a drawn question set.

    The sampler stratifies by SUITE (`per_suite = n // len(suites)`); tier is never a
    sampling dimension, so the mix is a byproduct rather than a declared quantity. Under a
    questions/hour objective that byproduct is load-bearing, so it is at minimum recorded.
    """
    mix: dict[str, int] = {}
    for q in questions:
        tier = _question_pool_tier(q)
        key = "unknown" if tier is None else str(tier)
        mix[key] = mix.get(key, 0) + 1
    return dict(sorted(mix.items()))


def _read_instrument_ledger() -> dict[str, Any]:
    """Load the durable ledger; a missing file is normal, a corrupt one is NOT silent."""
    try:
        if not _INSTRUMENT_LEDGER_PATH.exists():
            return {}
        return json.loads(_INSTRUMENT_LEDGER_PATH.read_text()) or {}
    except (OSError, ValueError) as exc:
        # Never fail the eval on a bad ledger, but never pretend drift detection ran
        # either — a detector that quietly degrades to "no drift" is worse than none.
        log.error(
            "Eval instrument ledger unreadable at %s (%s) — drift detection DEGRADED "
            "for this run; a changed question pool will NOT be flagged.",
            _INSTRUMENT_LEDGER_PATH,
            exc,
        )
        return {}


def _record_instrument_identity(
    core_id: str,
    dataset_sha: str,
    tier_mix: dict[str, int],
    n_questions: int,
) -> dict[str, Any] | None:
    """Persist this core_id's instrument identity; return drift details if it changed."""
    with _INSTRUMENT_LEDGER_LOCK:
        ledger = _read_instrument_ledger()
        entry = ledger.get(core_id) if isinstance(ledger.get(core_id), dict) else None
        drift: dict[str, Any] | None = None
        # Drift on EITHER axis. `dataset_content_sha256` hashes suite/id/prompt/expected/
        # scoring_method/scoring_config — deliberately NOT `tier`. So a pure re-tiering of
        # the pool changes every tier-stratified draw and the whole mix report while
        # leaving the content hash identical. Under a questions/hour objective the mix is
        # part of the instrument's identity, so compare it too.
        content_changed = bool(entry) and entry.get("dataset_content_sha256") != dataset_sha
        mix_changed = bool(entry) and entry.get("tier_mix") != tier_mix
        if entry and (content_changed or mix_changed):
            drift = {
                "changed_content": content_changed,
                "changed_tier_mix": mix_changed,
                "core_id": core_id,
                "previous_dataset_content_sha256": entry.get("dataset_content_sha256"),
                "current_dataset_content_sha256": dataset_sha,
                "previous_tier_mix": entry.get("tier_mix"),
                "current_tier_mix": tier_mix,
                "previous_n_questions": entry.get("n_questions"),
                "current_n_questions": n_questions,
                "previous_first_seen": entry.get("first_seen"),
                "detected_across_restart": True,
            }
        ledger[core_id] = {
            "dataset_content_sha256": dataset_sha,
            "tier_mix": tier_mix,
            "n_questions": n_questions,
            "first_seen": (entry or {}).get("first_seen") or datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _INSTRUMENT_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _INSTRUMENT_LEDGER_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True))
            tmp.replace(_INSTRUMENT_LEDGER_PATH)  # atomic; concurrent evals last-writer-wins
        except OSError as exc:
            log.error("Could not persist eval instrument ledger: %s", exc)
        return drift
_EVAL_QUESTION_JSONL_SCHEMA_VERSION = 1
_DEFAULT_EVAL_ARTIFACT_ROOT = Path("/mnt/raid0/llm/tmp/eval_tower_trials")


def _eval_no_progress_timeout_s(request_timeout_s: int) -> float:
    """Max wall-clock gap between completed eval futures before failing closed."""
    raw = os.environ.get("AUTOPILOT_EVAL_NO_PROGRESS_TIMEOUT_S", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            log.warning(
                "Invalid AUTOPILOT_EVAL_NO_PROGRESS_TIMEOUT_S=%r; using default",
                raw,
            )
        else:
            return max(0.0, value)
    return max(180.0, float(request_timeout_s) + 60.0)


def _eval_orphan_drain_timeout_s(request_timeout_s: int) -> float:
    """Bounded wait for in-flight eval workers before returning to caller."""
    raw = os.environ.get("AUTOPILOT_EVAL_ORPHAN_DRAIN_TIMEOUT_S", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            log.warning(
                "Invalid AUTOPILOT_EVAL_ORPHAN_DRAIN_TIMEOUT_S=%r; using default",
                raw,
            )
        else:
            return max(0.0, value)
    return min(30.0, max(1.0, float(request_timeout_s) / 2.0))


def _eval_batch_wall_budget_s(
    *,
    n_questions: int,
    workers: int,
    request_timeout_s: int,
) -> float:
    """Max end-to-end wall time for a single EvalTower batch.

    The explicit env override exists for tests and one-off operator windows.
    The default is deliberately generous but finite so an accidental serial
    fallback cannot run forever after the fanout guard has already degraded.
    """
    raw = os.environ.get("AUTOPILOT_EVAL_BATCH_WALL_BUDGET_S", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            log.warning(
                "Invalid AUTOPILOT_EVAL_BATCH_WALL_BUDGET_S=%r; using default",
                raw,
            )
    per_wave = float(request_timeout_s) + 30.0
    waves = math.ceil(max(1, int(n_questions)) / max(1, int(workers)))
    return max(_eval_no_progress_timeout_s(request_timeout_s), min(4 * 3600.0, waves * per_wave))


def _file_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


_RESEARCH_BENCHMARK_MODULE_CACHE: dict[tuple[str, str, int], Any] = {}


def _research_root() -> Path:
    return Path(os.environ.get("EPYC_RESEARCH_ROOT", "/mnt/raid0/llm/epyc-inference-research"))


def _load_research_benchmark_module(module_name: str) -> Any:
    """Load a research benchmark module by path, independent of bare import state."""
    module_path = _research_root() / "scripts" / "benchmark" / f"{module_name}.py"
    mtime_ns = _file_mtime_ns(module_path)
    if mtime_ns is None:
        raise FileNotFoundError(f"research benchmark module not found: {module_path}")

    cache_key = (module_name, str(module_path), mtime_ns)
    cached = _RESEARCH_BENCHMARK_MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    module_hash = hashlib.sha1(f"{module_path}:{mtime_ns}".encode("utf-8")).hexdigest()[:16]
    private_name = f"_epyc_research_{module_name}_{module_hash}"
    existing = sys.modules.get(private_name)
    if existing is not None:
        _RESEARCH_BENCHMARK_MODULE_CACHE[cache_key] = existing
        return existing

    spec = importlib.util.spec_from_file_location(private_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load research benchmark module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(private_name, None)
        raise
    _RESEARCH_BENCHMARK_MODULE_CACHE[cache_key] = module
    return module


def _question_validation_errors(q: Any) -> list[str]:
    if not isinstance(q, Mapping):
        return ["row_not_object"]

    errors: list[str] = []
    prompt = q.get("prompt")
    suite = q.get("suite")
    if prompt is None or str(prompt).strip() == "":
        errors.append("missing_prompt")
    if "expected" not in q:
        errors.append("missing_expected")
    elif q.get("expected") is None:
        errors.append("null_expected")
    if suite is None or str(suite).strip() == "":
        errors.append("missing_suite")
    if not errors and not _is_scoreable_question(dict(q)):
        errors.append("unscoreable")
    return errors


def _validate_eval_question_rows(
    rows: Sequence[Any],
    *,
    source: str,
) -> tuple[list[dict], dict[str, Any]]:
    valid: list[dict] = []
    drop_reasons: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        errors = _question_validation_errors(row)
        if errors:
            for error in errors:
                drop_reasons[error] = drop_reasons.get(error, 0) + 1
            if len(examples) < 5:
                examples.append(
                    {
                        "index": idx,
                        "id": row.get("id") if isinstance(row, Mapping) else None,
                        "errors": errors,
                    }
                )
            continue
        valid.append(dict(row))

    details = {
        "source": source,
        "loaded_rows": len(rows),
        "valid_rows": len(valid),
        "dropped_rows": len(rows) - len(valid),
        "drop_reasons": drop_reasons,
        "drop_examples": examples,
        "required_fields": list(_REQUIRED_EVAL_QUESTION_FIELDS),
    }
    if drop_reasons:
        log.warning(
            "Dropped %d invalid eval question row(s) from %s: %s",
            details["dropped_rows"],
            source,
            drop_reasons,
        )
    return valid, details


def _validate_question_pool(
    raw_pool: Any,
    *,
    source: str,
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    if not isinstance(raw_pool, Mapping):
        return {}, {
            "source": source,
            "loaded_rows": 0,
            "valid_rows": 0,
            "dropped_rows": 0,
            "drop_reasons": {"pool_not_mapping": 1},
            "invalid_suites": [],
            "required_fields": list(_REQUIRED_EVAL_QUESTION_FIELDS),
        }

    pool: dict[str, list[dict]] = {}
    details: dict[str, Any] = {
        "source": source,
        "loaded_rows": 0,
        "valid_rows": 0,
        "dropped_rows": 0,
        "drop_reasons": {},
        "invalid_suites": [],
        "required_fields": list(_REQUIRED_EVAL_QUESTION_FIELDS),
    }
    for suite, rows in raw_pool.items():
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            details["invalid_suites"].append(str(suite))
            details["drop_reasons"]["suite_not_sequence"] = (
                details["drop_reasons"].get("suite_not_sequence", 0) + 1
            )
            continue
        suite_valid, suite_details = _validate_eval_question_rows(
            list(rows),
            source=f"{source}:{suite}",
        )
        details["loaded_rows"] += suite_details["loaded_rows"]
        details["valid_rows"] += suite_details["valid_rows"]
        details["dropped_rows"] += suite_details["dropped_rows"]
        for reason, count in suite_details["drop_reasons"].items():
            details["drop_reasons"][reason] = details["drop_reasons"].get(reason, 0) + count
        if suite_valid:
            pool[str(suite)] = suite_valid
    return pool, details


def _has_executable_assertion(test_code: str) -> bool:
    for line in test_code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("assert ") or stripped.startswith("assert("):
            return True
    return False


def _has_unittest_case(test_code: str) -> bool:
    return "unittest.TestCase" in test_code or "(TestCase)" in test_code


def _has_code_execution_oracle(q: dict) -> bool:
    config = q.get("scoring_config") or {}
    if not isinstance(config, dict):
        return False
    test_code = str(config.get("test_code", "") or "").strip()
    if test_code.startswith("TEST_CASES"):
        return True
    if _has_executable_assertion(test_code) or _has_unittest_case(test_code):
        return True
    expected = q.get("expected", "")
    has_expected = expected is not None and str(expected) != ""
    return bool(config.get("entry_point") and has_expected)


def _require_math_verify() -> None:
    """EV-11 hard-fail: math_verify scoring must NEVER silently degrade.

    ``debug_scorer._score_math_verify`` swallows ``ImportError`` and returns an
    ``exact_match`` result when the ``math-verify`` package is absent. On the math
    suites (whose answers are boxed/LaTeX expressions) exact_match scores almost
    everything wrong, so a missing install silently no-ops the entire suite — the
    0/1,819-question EV-11 bug. We refuse to run a ``math_verify``-scored question
    when the library cannot be imported: fail loud, never fall back.
    """
    try:
        import math_verify  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise RuntimeError(
            "EV-11: scoring_method='math_verify' requires the 'math-verify' package, "
            "which is not importable in this interpreter. Refusing to silently fall "
            "back to exact_match (that fallback no-ops the whole math suite). Install "
            "it into the eval venv with `pip install math-verify`."
        ) from exc


def _is_rubric_scored_question(q: dict) -> bool:
    scoring_method = str(q.get("scoring_method", ""))
    if scoring_method == "rubric":
        return True
    suite = str(q.get("suite", ""))
    expected_contains = q.get("expected_contains")
    return suite.startswith("deep_research") and isinstance(expected_contains, list)


def _is_scoreable_question(q: dict) -> bool:
    expected = q.get("expected", "")
    scoring_method = str(q.get("scoring_method", "exact_match"))
    if _is_rubric_scored_question(q):
        return True
    if scoring_method == "code_execution":
        return _has_code_execution_oracle(q)
    has_expected = expected is not None and str(expected) != ""
    if not has_expected and scoring_method == "substring":
        # The needle may live in scoring_config["substring"] instead of `expected`
        # (instruction_precision, agentic). debug_scorer._score_substring reads it
        # from there, so such a row IS scoreable — it was only unreachable while
        # this predicate looked at `expected` alone. Keep the two in agreement:
        # declaring a row scoreable that the scorer cannot actually grade would
        # convert a dropped row into a systematically WRONG one.
        cfg = q.get("scoring_config") or {}
        if isinstance(cfg, dict) and str(cfg.get("substring", "") or "").strip():
            return True
    return has_expected or scoring_method in _EXPECTED_FREE_SCORERS


def _sample_scoreable_questions(
    suite: str,
    suite_qs: list[dict],
    per_suite: int,
    rng: random.Random,
) -> list[dict]:
    sample_size = min(per_suite, len(suite_qs))
    sample = rng.sample(suite_qs, sample_size)
    dead = [q for q in sample if not _is_scoreable_question(q)]
    if not dead:
        return sample

    seen = {id(q) for q in sample}
    replacements = []
    for q in suite_qs:
        if id(q) in seen or not _is_scoreable_question(q):
            continue
        replacements.append(q)
        if len(replacements) == len(dead):
            break

    log.warning(
        "Excised %d structurally unscorable %s eval item(s); replacements=%d",
        len(dead),
        suite,
        len(replacements),
    )
    return [q for q in sample if _is_scoreable_question(q)] + replacements


def _sample_scoreable_eval_questions(
    pool: dict[str, list[dict]],
    n: int,
    rng: random.Random,
    *,
    exclude_qids: set[str] | None = None,
    exclude_suites: set[str] | None = None,
) -> list[dict]:
    if not pool or n <= 0:
        return []
    excluded = exclude_qids or set()
    excluded_suites = exclude_suites or set()
    filtered_pool = {
        suite: [
            q
            for q in suite_qs
            if not (_question_identity_set(q) & excluded) and _is_scoreable_question(q)
        ]
        for suite, suite_qs in pool.items()
        if suite not in excluded_suites
    }
    pool = {suite: qs for suite, qs in filtered_pool.items() if qs}
    if not pool:
        return []

    suites = list(pool.keys())
    per_suite = max(1, n // len(suites))
    questions: list[dict] = []
    seen: set[int] = set()

    for suite in suites:
        for q in _sample_scoreable_questions(suite, pool[suite], per_suite, rng):
            if id(q) in seen or not _is_scoreable_question(q):
                continue
            seen.add(id(q))
            questions.append(q)

    if len(questions) < n:
        backfill = [
            q
            for suite_qs in pool.values()
            for q in suite_qs
            if id(q) not in seen and _is_scoreable_question(q)
        ]
        rng.shuffle(backfill)
        needed = n - len(questions)
        replacements = backfill[:needed]
        for q in replacements:
            seen.add(id(q))
        questions.extend(replacements)
        log.warning(
            "Backfilled %d/%d eval question(s) from global scoreable pool",
            len(replacements),
            needed,
        )

    rng.shuffle(questions)
    return questions[:n]


def _question_pool_tier(q: dict[str, Any]) -> int | None:
    raw = q.get("tier")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# ── Declared tier mix (2026-08-04, operator: "equal thirds") ────────────────────
#
# The mixed sampler stratifies by SUITE (`per_suite = n // len(suites)`) and never by
# difficulty tier, so the realized mix was a byproduct: the seed-42 n=50 draw came out
# T1:16 / T2:22 / T3:12, and it MOVED with n (n=100 gave 39/34/27) or with any edit to
# the question pool. Under the questions/hour Pareto objective that byproduct is
# load-bearing — T2/T3 questions cost far more wall-clock than T1 — so a pool edit could
# move the objective with no config change at all.
#
# The mix is now DECLARED. Equal thirds across tiers 1/2/3, with a deterministic
# remainder rule so a given n always yields the same targets.
EVAL_TIER_MIX_POLICY = "equal_thirds_v1"
EVAL_TIER_MIX_TIERS: tuple[int, ...] = (1, 2, 3)


def declared_tier_targets(
    n: int, tiers: tuple[int, ...] = EVAL_TIER_MIX_TIERS
) -> dict[int, int]:
    """Per-tier question counts for a draw of ``n``. Equal thirds; remainder to low tiers.

    The targets always sum to exactly ``n`` — an off-by-one here would silently change the
    questions/hour denominator.
    """
    if n <= 0 or not tiers:
        return {}
    base, remainder = divmod(n, len(tiers))
    return {tier: base + (1 if i < remainder else 0) for i, tier in enumerate(tiers)}


def _sample_tier_stratified_eval_questions(
    pool: dict[str, list[dict]],
    n: int,
    rng: random.Random,
    *,
    exclude_qids: set[str] | None = None,
    exclude_suites: set[str] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Draw ``n`` questions honouring the declared tier mix; report any shortfall.

    Returns ``(questions, provenance)``. Within each tier the existing suite-stratified
    sampler still runs, so suite balance is preserved *inside* a tier.

    A tier that cannot be filled is NOT backfilled from another tier. Backfilling would
    quietly return a draw that does not match the declared mix while still reporting the
    declared policy — the exact "absence silently becomes something else" shape that has
    bitten this system repeatedly. The shortfall is recorded and the draw comes back short.
    """
    targets = declared_tier_targets(n)
    excluded: set[str] = set(exclude_qids or set())
    questions: list[dict] = []
    shortfalls: dict[str, Any] = {}

    for tier in EVAL_TIER_MIX_TIERS:
        want = targets.get(tier, 0)
        if want <= 0:
            continue
        drawn = _sample_scoreable_eval_questions_for_pool_tier(
            pool,
            tier,
            want,
            rng,
            exclude_qids=excluded,
            exclude_suites=exclude_suites,
        )
        if len(drawn) < want:
            shortfalls[str(tier)] = {"requested": want, "drawn": len(drawn)}
            log.error(
                "Tier %d could only supply %d of %d requested eval questions — the draw "
                "does NOT match the declared %s mix. Not backfilling from another tier: "
                "that would misreport the instrument.",
                tier,
                len(drawn),
                want,
                EVAL_TIER_MIX_POLICY,
            )
        for q in drawn:
            excluded |= _question_identity_set(q)
        questions.extend(drawn)

    rng.shuffle(questions)
    provenance = {
        "tier_mix_policy": EVAL_TIER_MIX_POLICY,
        "tier_mix_targets": {str(k): v for k, v in targets.items()},
        "tier_mix_shortfalls": shortfalls,
        "requested_n": int(n),
        "drawn_n": len(questions),
    }
    return questions, provenance


def _sample_scoreable_eval_questions_for_pool_tier(
    pool: dict[str, list[dict]],
    pool_tier: int,
    n: int,
    rng: random.Random,
    *,
    exclude_qids: set[str] | None = None,
    exclude_suites: set[str] | None = None,
) -> list[dict]:
    """Sample scoreable questions whose source-pool difficulty tier matches exactly.

    This keeps T3 as a first-class expert/hard workflow lane without changing the broader mixed
    T1/T2 sampler, whose caller surface is much wider.
    """
    if not pool or n <= 0:
        return []
    filtered_pool = {
        suite: [q for q in suite_qs if _question_pool_tier(q) == int(pool_tier)]
        for suite, suite_qs in pool.items()
    }
    return _sample_scoreable_eval_questions(
        filtered_pool,
        n,
        rng,
        exclude_qids=exclude_qids,
        exclude_suites=exclude_suites,
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _stable_question_qid(suite: str, prompt_text: str) -> str:
    payload = f"{suite}\x00{prompt_text}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _question_qid(q: dict[str, Any]) -> str:
    explicit = str(q.get("qid") or q.get("stable_qid") or q.get("id") or "").strip()
    if explicit:
        return explicit
    return _stable_question_qid(str(q.get("suite", "unknown")), str(q.get("prompt", "")))


def _question_identity_set(q: dict[str, Any]) -> set[str]:
    """All comparable identities for one pool question.

    Historical journals may carry only the stable prompt hash (`qid`), while
    pool rows commonly carry only their source `id`. Promotion fresh-draw
    exclusion must compare against both namespaces.
    """
    identities: set[str] = set()
    for key in ("qid", "stable_qid", "id", "question_id"):
        value = str(q.get(key) or "").strip()
        if value:
            identities.add(value)
    prompt = str(q.get("prompt") or "")
    if prompt:
        identities.add(_stable_question_qid(str(q.get("suite", "unknown")), prompt))
    return identities


def _question_identity_union(questions: Sequence[dict[str, Any]]) -> set[str]:
    identities: set[str] = set()
    for question in questions:
        identities.update(_question_identity_set(question))
    return identities


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


# ── REL-1 eval-honesty guards (2026-07-21 EV-11c circuit-open incident) ──────
#
# Guard 1 anchors to the orchestrator's in-band error format. The llm
# primitives emit failures AS answer strings of the form ``[ERROR: <detail>]``
# (src/llm_primitives/primitives.py::_call → ``return f"[ERROR: {e}]"``; the
# circuit-open detail is the RuntimeError text from
# src/llm_primitives/inference.py, ``Backend unavailable (circuit open):
# <url>``). When the breaker opens and the response is NOT run through
# ``_annotate_error`` (server-side, stages.py — which also keys on
# ``answer.startswith("[ERROR:")``), /chat returns answer="[ERROR: Backend
# unavailable (circuit open): http://localhost:8082]" with error=None. The
# eval then scored that error text as a WRONG answer (REL-1 evasion). Anchor to
# the REAL start-of-answer prefix, never a loose substring.
_INBAND_ERROR_PREFIX = "[ERROR:"


def _inband_error_text(answer: Any) -> str | None:
    """Return the in-band orchestrator error string when `answer` IS one.

    Anchored to the emitted ``[ERROR: ...]`` prefix at start-of-answer (after
    stripping leading whitespace), matching the primitives/inference emitters
    and the server-side ``_annotate_error`` convention. Returns None for a
    normal answer.
    """
    if not isinstance(answer, str):
        return None
    stripped = answer.lstrip()
    if stripped.startswith(_INBAND_ERROR_PREFIX):
        return stripped
    return None


def _forced_role_serving_mismatch(
    force_role: Any, resp: Mapping[str, Any]
) -> str | None:
    """Return the serving role when it differs from the forced role, else None.

    Guard 2: when the eval pins ``force_role`` for a role-attributed
    measurement and the orchestrator silently serves it from a DIFFERENT role
    (the 2026-07-21 circuit_open fallback ``worker_math → worker_general``), the
    number is not a measurement of the forced role. Compare ``force_role``
    against the response's ``routed_to`` (the primary role that handled the
    request), falling back to the terminal ``role_history`` entry when
    ``routed_to`` is absent. Returns None when ``force_role`` is empty or the
    serving role cannot be determined — avoiding false positives on
    partial/legacy responses.
    """
    forced = str(force_role or "").strip()
    if not forced:
        return None
    serving = str(resp.get("routed_to") or "").strip()
    if not serving:
        history = resp.get("role_history")
        if isinstance(history, (list, tuple)) and history:
            serving = str(history[-1] or "").strip()
    if not serving or serving == forced:
        return None
    return serving


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _completion_probabilities_confidence(rows: Any) -> float | None:
    """Convert llama.cpp completion probability rows into sequence confidence."""
    if not isinstance(rows, list):
        return None
    logps: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prob = _finite_float(row.get("probability", row.get("prob")))
        logprob = _finite_float(row.get("logprob"))
        if prob is None:
            candidates = row.get("probs") or row.get("top_logprobs") or []
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    prob = _finite_float(first.get("probability", first.get("prob")))
                    logprob = _finite_float(first.get("logprob"))
        if prob is None and logprob is not None:
            prob = math.exp(logprob)
        if prob is None:
            continue
        prob = min(1.0, max(1e-12, float(prob)))
        logps.append(math.log(prob))
    if not logps:
        return None
    return min(1.0, max(0.0, math.exp(sum(logps) / len(logps))))


def _upper_median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2] if values else 0.0


def _compact_host_covariates(covariates: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in _HOST_COVARIATE_COMPACT_KEYS:
        value = covariates.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            compact[key] = value
        elif isinstance(value, int):
            compact[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                compact[key] = round(value, 4)
        elif isinstance(value, str):
            compact[key] = value[:80]
    return compact


def _capture_host_timing_covariates(
    *,
    tokens_generated: int,
    elapsed_s: float,
) -> dict[str, Any]:
    try:
        from host_health import host_timing_covariates  # type: ignore

        covariates = host_timing_covariates(
            event="question_complete",
            tokens_generated=tokens_generated,
            elapsed_s=elapsed_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("host timing covariates unavailable: %s", exc)
        return {}
    return _compact_host_covariates(covariates)


def _speed_analytics_ge_128(
    results: list["QuestionResult"],
    *,
    eval_wall_s: float,
) -> dict[str, Any]:
    eligible = [
        r
        for r in results
        if not r.error and r.tokens_generated >= _SPEED_ANALYTICS_MIN_TOKENS and r.elapsed_s > 0
    ]
    short_timing_samples = sum(
        1 for r in results if not r.error and 0 < r.tokens_generated < _SPEED_ANALYTICS_MIN_TOKENS
    )
    request_speeds = [r.tokens_generated / r.elapsed_s for r in eligible]
    eligible_tokens = sum(r.tokens_generated for r in eligible)
    aggregate_tps = (
        eligible_tokens / eval_wall_s if eligible_tokens > 0 and eval_wall_s > 0 else 0.0
    )
    return {
        "speed_analytics_min_tokens": _SPEED_ANALYTICS_MIN_TOKENS,
        "speed_analytics_filter": f"tokens_generated>={_SPEED_ANALYTICS_MIN_TOKENS}",
        "speed_analytics_n_ge_128": len(eligible),
        "speed_analytics_n_lt_128": short_timing_samples,
        "speed_analytics_tokens_ge_128": eligible_tokens,
        "speed_analytics_median_request_tps_ge_128": _upper_median(request_speeds),
        "speed_analytics_aggregate_tps_ge_128": aggregate_tps,
    }


def _summarize_host_timing_covariates(
    results: list["QuestionResult"],
) -> dict[str, Any]:
    samples = [r.host_covariates for r in results if r.host_covariates]
    summary: dict[str, Any] = {"samples": len(samples)}
    if not samples:
        return summary

    for key in _HOST_COVARIATE_NUMERIC_KEYS:
        values = [
            value
            for value in (_finite_float(sample.get(key)) for sample in samples)
            if value is not None
        ]
        if values:
            summary[key] = {
                "min": min(values),
                "median": _upper_median(values),
                "max": max(values),
            }

    for key in _HOST_COVARIATE_CATEGORICAL_KEYS:
        counts: dict[str, int] = {}
        for sample in samples:
            raw = sample.get(key)
            if raw is None:
                continue
            value = str(raw)
            counts[value] = counts.get(value, 0) + 1
        if counts:
            summary[key] = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            summary[f"{key}_counts"] = dict(sorted(counts.items()))
    return summary


def _compact_question_result(r: "QuestionResult") -> dict[str, Any]:
    question_id = str(r.question_id or "").strip()
    item: dict[str, Any] = {
        "qid": r.qid or _stable_question_qid(str(r.suite), str(r.prompt)),
        "suite": r.suite,
        "partition": r.eval_partition or "core",
        "correct": bool(r.correct),
        "latency_ms": int(round(max(0.0, r.elapsed_s) * 1000)),
        "tokens_generated": int(r.tokens_generated or 0),
        "tools_used": int(r.tools_used or 0),
    }
    if question_id:
        item["question_id"] = question_id
    if r.host_covariates:
        compact_covariates = _compact_host_covariates(r.host_covariates)
        if compact_covariates:
            item["host_covariates"] = compact_covariates
    answer_hash = normalized_answer_hash(r.answer)
    if answer_hash and not r.error:
        item["answer_hash"] = answer_hash
    if r.scoring_method and r.scoring_method != "exact_match":
        item["scoring_method"] = r.scoring_method
    if r.route_used:
        item["route"] = r.route_used
    if r.tools_called:
        item["tools_called"] = list(r.tools_called[:5])
    if r.confidence_source and r.confidence_source != "binary_correctness_proxy":
        item["confidence"] = round(float(r.confidence), 6)
        item["confidence_source"] = r.confidence_source
    if r.error:
        item["error"] = True
        item["error_detail"] = str(r.error).replace("\n", " ")[:200]
        if isinstance(r.failure_provenance, dict):
            item["failure_provenance"] = dict(r.failure_provenance)
            if r.failure_provenance.get("class") == "admission_timeout":
                # Retry eligibility needs explicit negative facts. Preserve
                # both booleans even though the general compact format omits
                # false values.
                item["partial"] = bool(r.partial)
                item["degraded"] = bool(r.degraded)
    if r.partial:
        item["partial"] = True
    if r.degraded:
        item["degraded"] = True
    if r.exogenous_recovered:
        item["exogenous_recovered"] = True
    if r.exogenous_unrecovered:
        item["exogenous_unrecovered"] = True
    if r.external_restart:
        item["external_restart"] = True
    if r.retry_count:
        item["retry_count"] = int(r.retry_count)
    rubric_scores = {
        key: float(value)
        for key, value in sorted((r.rubric_scores or {}).items())
        if math.isfinite(float(value))
    }
    if rubric_scores:
        item["rubric_scores"] = rubric_scores
    if r.rubric_source:
        item["rubric_source"] = r.rubric_source
    return item


def _annotate_partition(questions: list[dict], partition: str) -> list[dict]:
    annotated = []
    for q in questions:
        item = dict(q)
        item["eval_partition"] = partition
        annotated.append(item)
    return annotated


def _audit_seed(trial_id: int, core_id: str) -> int:
    payload = f"w6-audit-v1\x00{trial_id}\x00{core_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _promotion_eval_seed(trial_id: int, n: int) -> int:
    payload = f"w8-promotion-eval-v1\x00{trial_id}\x00{n}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _promotion_eval_n(default: int = PROMOTION_EVAL_DEFAULT_N) -> int:
    raw = _env_int("AUTOPILOT_SEQ_PROMOTION_EVAL_N", default)
    return min(PROMOTION_EVAL_MAX_N, max(PROMOTION_EVAL_MIN_N, raw))


def _latest_promotion_suite_health_path() -> Path | None:
    override = os.environ.get("AUTOPILOT_SEQ_PROMOTION_SUITE_HEALTH_PATH", "").strip()
    if override:
        path = Path(override)
        return path if path.exists() else None
    reports_dir = Path(__file__).resolve().parents[2] / "orchestration" / "reports"
    candidates = sorted(
        reports_dir.glob(PROMOTION_EVAL_SUITE_HEALTH_GLOB),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _promotion_excluded_suites_from_health() -> tuple[set[str], dict[str, Any]]:
    path = _latest_promotion_suite_health_path()
    if path is None:
        return set(), {"path": None, "status": "missing", "excluded_suites": []}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return set(), {
            "path": str(path),
            "status": "unreadable",
            "error": str(exc),
            "excluded_suites": [],
        }
    windows = data.get("windows") if isinstance(data, dict) else {}
    window = {}
    if isinstance(windows, dict):
        window = windows.get("last_100_trials") or windows.get("last_7_days") or {}
    suite_rows = window.get("suite_summary") if isinstance(window, dict) else []
    excluded: set[str] = set()
    reasons: dict[str, str] = {}
    if isinstance(suite_rows, list):
        for row in suite_rows:
            if not isinstance(row, dict):
                continue
            suite = str(row.get("suite") or "").strip()
            if not suite:
                continue
            flags = set(str(flag) for flag in (row.get("flags") or []))
            verdict = str(row.get("artifact_verdict") or "")
            if verdict == "artifact" or "pinned_zero_or_broken" in flags:
                excluded.add(suite)
                reasons[suite] = verdict or ",".join(sorted(flags))
    return excluded, {
        "path": str(path),
        "status": "ok",
        "excluded_suites": sorted(excluded),
        "reasons": reasons,
    }


def _read_registry_timeout(category: str, key: str, fallback: int) -> int:
    registry_path = Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"
    try:
        data = yaml.safe_load(registry_path.read_text()) or {}
        timeouts = data.get("runtime_defaults", {}).get("timeouts", {})
        cat_data = timeouts.get(category, {})
        return int(cat_data.get(key, timeouts.get("default", fallback)))
    except Exception:
        return fallback


def _default_eval_timeout() -> int:
    """Per-question eval request timeout (seconds).

    Env override (REL-1 guard 3, 2026-07-21): ``AUTOPILOT_EVAL_REQUEST_TIMEOUT_S``
    raises the per-question budget for rebaseline-class runs whose long
    MATH-tail questions need more headroom than the registry-derived default —
    and it deliberately bypasses the 600s cap. This is the knob operators set
    so a rebaseline never *starts* with a per-call budget that would later be
    whittled below the deadline-starvation floor (see
    ``AUTOPILOT_EVAL_MIN_LLAMA_BUDGET_S`` in ``call_orchestrator_forced``).
    Unset / <=0 preserves the current behavior EXACTLY: registry frontdoor role
    timeout plus ``AUTOPILOT_EVAL_QUEUE_ALLOWANCE_S``, capped at 600.
    """
    override = _env_int("AUTOPILOT_EVAL_REQUEST_TIMEOUT_S", 0)
    if override > 0:
        return override
    role_timeout = _read_registry_timeout("roles", "frontdoor", 180)
    queue_allowance = _env_int("AUTOPILOT_EVAL_QUEUE_ALLOWANCE_S", 90)
    return min(600, max(role_timeout, role_timeout + max(0, queue_allowance)))


DEFAULT_TIMEOUT = _default_eval_timeout()


# Concurrent fan-out for sentinel/pool evaluations.
#
# Default behavior (J6/WP-7): matrix-aware topology safe-N from the bottleneck
# role (`AUTOPILOT_EVAL_BOTTLENECK_ROLE`, default "frontdoor"). The topology
# cap is the physical ceiling, and the same-role contention matrix must be
# certified fresh + ALLOW for background/eval traffic before the default rises
# above serial. Missing/stale/invalid matrix evidence fails closed to 1.
#
# Operators can still override via `AUTOPILOT_EVAL_CONCURRENCY=N`. The env
# override always wins, even over the topology/matrix cap, because some
# test/diag paths intentionally exceed it (e.g. WP-3 migration smoke tests).
#
def _live_role_ports(numa_config: dict, role: str) -> set[int]:
    instances = ((numa_config or {}).get(role) or {}).get("instances") or []
    live_ports: set[int] = set()
    for entry in instances:
        if not entry or len(entry) < 2:
            continue
        try:
            port = int(entry[1])
        except (TypeError, ValueError):
            continue
        try:
            resp = httpx.get(f"http://localhost:{port}/health", timeout=0.5)
            if resp.status_code == 200:
                live_ports.add(port)
        except Exception:
            continue
    return live_ports


def _iso_within_days(value: str, days: int) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_s = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
    return 0 <= age_s <= days * 86400


def _same_role_certification_allows_eval_fanout(role: str, *, matrix: object, numa_config: dict) -> bool:
    try:
        from src.scheduling.contention import (
            MATRIX_STALENESS_DAYS,
            PairDecision,
            TrafficClass,
            pair_policy,
            role_topology_fingerprint,
        )
    except Exception:
        return False

    get_cert = getattr(matrix, "get_same_role_certification", None)
    cert = get_cert(role) if callable(get_cert) else None
    if cert is None or getattr(cert, "verdict", "") != "allow":
        return False
    if pair_policy(role, role, TrafficClass.BACKGROUND, matrix=matrix) != PairDecision.ALLOW:
        return False
    if not _iso_within_days(str(getattr(cert, "measured_at", "")), MATRIX_STALENESS_DAYS):
        return False

    live_ports = _live_role_ports(numa_config, role)
    if not live_ports:
        return False
    certified_ports = set(getattr(cert, "live_ports", ()) or ())
    if certified_ports and certified_ports != live_ports:
        return False
    role_hash = role_topology_fingerprint(numa_config, role, live_ports=live_ports)
    return role_hash == getattr(cert, "topology_hash", "")


# Reference certified safe-N: frontdoor=3, ingest_long_context=3,
# vision_escalation=3, worker_general=1, architect_general=1, worker_vision=1.
def _same_role_matrix_allows_eval_fanout(role: str) -> bool:
    try:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
        from src.scheduling.contention import (
            MatrixStatus,
            PairDecision,
            TrafficClass,
            load_contention_matrix,
            matrix_status,
            pair_policy,
            topology_fingerprint_for_matrix,
        )

        matrix = load_contention_matrix()
        current_hash = topology_fingerprint_for_matrix(NUMA_CONFIG, matrix)
        status = matrix_status(current_topology_hash=current_hash)
        if status == MatrixStatus.OK:
            return pair_policy(role, role, TrafficClass.BACKGROUND, matrix=matrix) == PairDecision.ALLOW
        if status != MatrixStatus.STALE:
            return False
        return _same_role_certification_allows_eval_fanout(
            role,
            matrix=matrix,
            numa_config=NUMA_CONFIG,
        )
    except Exception:
        return False


def _runtime_facts_stack_numa_mode() -> str | None:
    """Return the runtime-facts stack NUMA mode ONLY when the manifest passes the
    same fail-closed contract the URL reader (read_runtime_stack_selected_servers)
    enforces: a concrete expected mode string AND a non-empty selected-server
    lineup consistent with the declared ports.

    WP-14: the launcher can leave a phantom full-era lineup behind (the real
    current shape is stack_numa_mode=None, selected_ports=[], full-era
    selected_servers). read_runtime_stack_numa_mode() alone would silently treat
    that as "not quarter" and mis-size eval fan-out onto a single quarter. Mirror
    the URL reader's rejection here and fall back to
    ORCHESTRATOR_STACK_NUMA_MODE / NUMA_CONFIG with one loud log line.
    """
    try:
        from scripts.server.runtime_facts_manifest import (
            read_runtime_stack_numa_mode,
            read_runtime_stack_selected_servers,
            runtime_facts_manifest_path,
        )
    except Exception:
        return None

    def _read(reader: Callable[..., Any]) -> Any:
        try:
            value = reader()
        except Exception:
            return None
        if value:
            return value
        try:
            return reader(state_file=None)
        except TypeError:
            return value
        except Exception:
            return None

    mode = _read(read_runtime_stack_numa_mode)
    servers = _read(read_runtime_stack_selected_servers)
    lineup_ok = isinstance(servers, list) and bool(servers)
    if isinstance(mode, str) and mode and lineup_ok:
        return mode.strip().lower()

    try:
        manifest_present = runtime_facts_manifest_path().exists()
    except Exception:
        manifest_present = False
    if manifest_present:
        log.warning(
            "runtime-facts manifest rejected (fail-closed: stack_numa_mode=%r, "
            "selected lineup %s); falling back to ORCHESTRATOR_STACK_NUMA_MODE/NUMA_CONFIG",
            mode,
            "present" if lineup_ok else "empty/inconsistent",
        )
    return None


def _live_safe_concurrency(role: str, topology_cap: int) -> int:
    """Bound eval fan-out by the currently reachable role instances.

    Static topology can say a role is safe at N>1 while the live stack is
    intentionally launched in full-only mode. In that case, concurrent evals
    pile onto one llama-server and can corrupt evidence with 5xx/timeouts.
    """
    if os.environ.get("AUTOPILOT_EVAL_REQUIRE_LIVE_FLEET", "1") == "0":
        return topology_cap
    try:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
        from src.runtime.instance_topology import cpu_list_to_regions
        from src.runtime.instance_topology import compute_max_disjoint_live_concurrency
    except Exception:
        return 1

    instances = ((NUMA_CONFIG or {}).get(role) or {}).get("instances") or []
    if not instances:
        return 1

    live_regions: list[frozenset[str]] = []
    live_ports = _live_role_ports(NUMA_CONFIG, role)
    for entry in instances:
        if not entry or len(entry) < 2:
            continue
        try:
            port = int(entry[1])
        except (TypeError, ValueError):
            continue
        if port not in live_ports:
            continue
        live_regions.append(cpu_list_to_regions(str(entry[0])))

    if not live_regions:
        return 1

    stack_numa_mode = _runtime_facts_stack_numa_mode()
    if stack_numa_mode is None:
        stack_numa_mode = os.environ.get("ORCHESTRATOR_STACK_NUMA_MODE")

    if str(stack_numa_mode or "").strip().lower() == "quarter":
        return compute_max_disjoint_live_concurrency(
            NUMA_CONFIG,
            role,
            live_ports=live_ports,
        )

    if topology_cap <= 1:
        return 1

    accepted_union: set[str] = set(live_regions[0])
    live_cap = 1
    for regions in sorted(
        live_regions[1:],
        key=lambda r: (bool(accepted_union & r), sorted(r)),
    ):
        if not regions or accepted_union & regions:
            continue
        accepted_union |= regions
        live_cap += 1
        if live_cap >= topology_cap:
            break
    return max(1, min(topology_cap, live_cap))


def _forced_roles_for_questions(questions: Sequence[Mapping[str, Any]]) -> list[str]:
    roles: list[str] = []
    for q in questions:
        role = str(q.get("force_role") or "").strip()
        if role and role not in roles:
            roles.append(role)
    return roles


def _eval_concurrency(roles: Sequence[str] | None = None) -> int:
    raw = os.environ.get("AUTOPILOT_EVAL_CONCURRENCY")
    if raw is not None:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass  # fall through to topology default

    role_candidates = [str(r).strip() for r in (roles or []) if str(r).strip()]
    if not role_candidates:
        role_candidates = [os.environ.get("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "frontdoor")]

    try:
        from src.runtime.instance_topology import max_safe_concurrency
    except Exception:
        return 1

    caps: list[int] = []
    for role in role_candidates:
        try:
            topology_cap = max(1, max_safe_concurrency(role))
            if not _same_role_matrix_allows_eval_fanout(role):
                caps.append(1)
                continue
            caps.append(_live_safe_concurrency(role, topology_cap))
        except Exception:
            caps.append(1)
    return max(1, min(caps or [1]))


def _eval_scoring_concurrency(generation_workers: int) -> int:
    """Width of the client-side SCORING pool for the ``_eval_batch`` workers>1 path.

    Scoring for suites like HE-R+ (code_execution) is pure client CPU — the scorer
    subprocesses a sandbox — so it does NOT consume the inference fan-out budget
    that caps generation at ``_eval_concurrency``. Decoupling lets scoring run
    wider than the topology-derived generation width so a scoring-bound suite stops
    idling the serving fleet.

    Default: ``min(16, os.cpu_count() // 12)``, but never below the generation
    width so the pipeline can never be narrower than the pre-split single-pool
    executor. Operators override via ``AUTOPILOT_EVAL_SCORING_CONCURRENCY`` (also
    floored at the generation width — this env raises scoring throughput, it does
    not change the inference-contention-capped generation width).
    """
    gen = max(1, int(generation_workers))
    raw = os.environ.get("AUTOPILOT_EVAL_SCORING_CONCURRENCY", "").strip()
    if raw:
        try:
            override = int(raw)
        except ValueError:
            log.warning(
                "Invalid AUTOPILOT_EVAL_SCORING_CONCURRENCY=%r; using default",
                raw,
            )
        else:
            return max(gen, override)
    cpu = os.cpu_count() or 1
    default = min(16, max(1, cpu // 12))
    return max(gen, default)


def _eval_batch_id(*, label: str, n_questions: int, started_at_s: float) -> str:
    safe_label = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(label or "eval").strip()
    ).strip("-_")
    if not safe_label:
        safe_label = "eval"
    return f"evaltower-{safe_label}-{int(started_at_s * 1000)}-{uuid.uuid4().hex[:8]}-{max(0, n_questions)}q"


def _safe_artifact_name(value: str, *, fallback: str = "eval") -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value or "").strip()
    ).strip("-_.")
    return safe or fallback


def _eval_artifact_root_from_trial_path(trial_path: Path | None) -> Path | None:
    if trial_path is None:
        return None
    path = Path(trial_path).expanduser()
    if path.suffix:
        path = path.parent
    return path / "eval_tower"


def _eval_artifact_root(trial_path: Path | None = None) -> tuple[Path, str]:
    override = os.environ.get("AUTOPILOT_EVAL_ARTIFACT_ROOT", "").strip()
    if override:
        return Path(override).expanduser(), "env:AUTOPILOT_EVAL_ARTIFACT_ROOT"
    trial_root = _eval_artifact_root_from_trial_path(trial_path)
    if trial_root is not None:
        return trial_root, "trial_path"
    return _DEFAULT_EVAL_ARTIFACT_ROOT, "default"


class _EvalQuestionJsonlWriter:
    """Durable append-only per-trial QuestionResult sidecar."""

    def __init__(
        self,
        *,
        root: Path,
        root_source: str,
        eval_batch_id: str,
        trial_id: int | None,
        label: str,
        requested_n: int,
        concurrency: int,
        path: Path | None = None,
    ) -> None:
        trial_name = f"trial_{trial_id}" if trial_id is not None else "trial_unknown"
        # EV-11c: the window-runner (verifier-mode) path has no trial id, so the
        # default trial-keyed path collapses every arm onto a single invisible
        # "trial_unknown" file. Callers on that path pass an explicit per-arm `path`
        # (e.g. <output-dir>/question_results.<arm>.jsonl) so each arm's rows are
        # durably identifiable for a targeted error requeue.
        if path is not None:
            self.path = Path(path)
        else:
            self.path = root / _safe_artifact_name(trial_name, fallback="trial_unknown") / "question_results.jsonl"
        self._eval_batch_id = eval_batch_id
        self._trial_id = trial_id
        self._label = str(label or "")
        self._requested_n = int(requested_n)
        self._concurrency = int(concurrency)
        self._root_source = root_source
        self._lock = threading.Lock()
        self._fd: int | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        self._fsync_parent_dir()

    def _fsync_parent_dir(self) -> None:
        try:
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _base_row(self, row_type: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema_version": _EVAL_QUESTION_JSONL_SCHEMA_VERSION,
            "row_type": row_type,
            "eval_batch_id": self._eval_batch_id,
            "label": self._label,
            "requested_n": self._requested_n,
            "artifact_root_source": self._root_source,
            "recovery_contract": "complete_marker_required",
        }
        if self._trial_id is not None:
            row["trial_id"] = self._trial_id
        return row

    def append_start(self) -> None:
        row = self._base_row("batch_start")
        row["concurrency"] = self._concurrency
        row["complete"] = False
        self.append_row(row)

    def append_result(
        self,
        *,
        ordinal: int,
        result: "QuestionResult",
        generated_at_s: float | None = None,
        scored_at_s: float | None = None,
    ) -> None:
        row = self._base_row("question_result")
        # Wall-clock interval so end-to-end concurrency depth and latency
        # distributions are derivable from the artifact alone (2026-07-22:
        # verifying EV-4b fan-out required /proc forensics because rows carried
        # no timing).
        #
        # Serial / single-pool path (generated_at_s is None): append runs
        # immediately on completion, so ended_at_s = append time ≈ request end —
        # behavior is byte-identical to the pre-pipeline writer.
        #
        # Pipelined path (workers>1): ``generated_at_s`` is the absolute instant
        # GENERATION finished, so ended_at_s/started_at_s/elapsed_s describe the
        # GENERATION interval, and ``scored_at_s`` (>= ended_at_s) marks when the
        # decoupled scoring pool produced the verdict — separating the two phases
        # in the artifact.
        elapsed = max(0.0, float(getattr(result, "elapsed_s", 0.0) or 0.0))
        ended_at_s = time.time() if generated_at_s is None else float(generated_at_s)
        row.update(
            {
                "ordinal": int(ordinal),
                "result": _compact_question_result(result),
                # Full raw generated answer (2026-07-22 operator directive): the
                # compact `result` above keeps only answer_hash, but the untracked
                # report sidecars must be re-scorable and resumable, and the answer
                # is the irreplaceable artifact. `prompt`/`expected` stay excluded —
                # both are reconstructable from the dataset by qid/ordinal +
                # dataset_sha256. answer_hash is retained for integrity checks.
                "answer": str(getattr(result, "answer", "") or ""),
                "complete": False,
                "ended_at_s": round(ended_at_s, 3),
                "elapsed_s": round(elapsed, 6),
                "started_at_s": round(ended_at_s - elapsed, 3) if elapsed > 0 else None,
            }
        )
        if scored_at_s is not None:
            row["scored_at_s"] = round(float(scored_at_s), 3)
        self.append_row(row)

    def append_complete(self, *, completed_n: int, elapsed_s: float) -> None:
        row = self._base_row("batch_complete")
        row.update(
            {
                "completed_n": int(completed_n),
                "elapsed_s": round(max(0.0, float(elapsed_s)), 6),
                "complete": True,
            }
        )
        self.append_row(row)

    def append_row(self, row: dict[str, Any]) -> None:
        data = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
            "utf-8"
        )
        with self._lock:
            if self._fd is None:
                raise OSError("eval question JSONL writer is closed")
            written = 0
            while written < len(data):
                written += os.write(self._fd, data[written:])
            os.fsync(self._fd)

    def close(self) -> None:
        with self._lock:
            if self._fd is None:
                return
            os.close(self._fd)
            self._fd = None


# Import seeding infrastructure
_orch_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_orch_root / "scripts" / "benchmark"))
# Repo root on path so `src.tools.eval_secret` (runtime tool-secret ground truth)
# imports from the autopilot harness, not just inside the orchestrator process.
sys.path.insert(0, str(_orch_root))

from seeding_orchestrator import call_orchestrator_forced  # noqa: E402
from seeding_scoring import score_answer_deterministic, score_answer_or_error  # noqa: E402
from rubric_scoring import (  # noqa: E402
    MINDDR_PROCESS_DIMENSIONS,
    aggregate_rubric_score,
    build_rubric_judge_prompt,
    deterministic_rubric_fallback,
)
from src.autopilot_core.instrument_era_guard import (  # noqa: E402
    designed_core_activation_guard,
)
from src.behavior_signature import normalized_answer_hash  # noqa: E402
from src.llm_primitives.stat_tests import (  # noqa: E402
    CALIBRATION_METRIC_KEYS,
    DEFAULT_WILSON_Z,
    compute_calibration_metrics as _stat_compute_calibration_metrics,
    expected_calibration_error,
    roc_auc,
    wilson_interval,
)

# Paired-significance screening for A/B arm comparisons (config/quant A/B). These
# are the LANDED clean-room stats primitives — the screening hook below only
# orchestrates them onto eval_tower's own comparison output; it does NOT
# reimplement any statistic. ``paired_stats`` lives beside this module in
# ``scripts/autopilot/``.
from paired_stats import (  # noqa: E402
    MCNEMAR_EXACT_MAX_DISCORDANT,
    PairedComparisonMismatchError,
    QuestionOutcome,
    mcnemar_from_vectors,
    mcnemar_verdict,
    require_matched_comparison,
)

DEFAULT_CORE_DIR = _orch_root / "benchmarks" / "prompts"

# Branching density: intake-378 deep-dive (arxiv:2604.01702).
# High branching (>0.30 Propose step ratio) = unproductive exploration.
import re as _re

_THINK_RE = _re.compile(r"<think>(.*?)</think>", _re.DOTALL)
_BRANCH_KEYWORDS = _re.compile(
    r"\b(?:perhaps|another\s+approach|alternatively|let\s+me\s+try|"
    r"wait[\s,]|let\s+me\s+reconsider|maybe\s+(?:I|we)\s+should|"
    r"on\s+second\s+thought|what\s+if)\b",
    _re.IGNORECASE,
)
# Approximate reasoning step boundary: sentence-ending punctuation or newlines.
_STEP_BOUNDARY = _re.compile(r"[.!?\n]")


def _compute_branching_density(answer: str) -> float:
    """Compute fraction of reasoning steps containing branching keywords.

    Returns 0.0 if no <think> blocks are present.
    """
    blocks = _THINK_RE.findall(answer)
    if not blocks:
        return 0.0
    think_text = " ".join(blocks)
    steps = [s.strip() for s in _STEP_BOUNDARY.split(think_text) if len(s.strip()) > 10]
    if not steps:
        return 0.0
    branching_steps = sum(1 for s in steps if _BRANCH_KEYWORDS.search(s))
    return branching_steps / len(steps)


@dataclass
class QuestionResult:
    question_id: str
    suite: str
    prompt: str
    expected: str
    qid: str = ""
    answer: str = ""
    correct: bool = False
    error: str | None = None
    failure_provenance: dict[str, Any] | None = None
    tokens_generated: int = 0
    elapsed_s: float = 0.0
    route_used: str = ""
    cost_tier: int = 0
    scoring_method: str = "exact_match"
    partial: bool = False  # Inference completed with partial output (read_timeout)
    degraded: bool = False  # Inference completed in degraded mode
    confidence: float = 0.0  # EV-1: Model confidence proxy (0-1). Initially float(correct); upgraded to logprobs when available.
    confidence_source: str = "binary_correctness_proxy"
    branching_density: float = 0.0  # Fraction of <think> steps with branching keywords (intake-378)
    eval_concurrency: int = 1  # Worker fan-out used for this eval batch.
    eval_wall_s: float = 0.0  # End-to-end wall time for the containing eval batch.
    # Tool telemetry (2026-06-01): captured from the /chat response so the autopilot
    # can measure — and learn to incentivize — model tool use. `tokens_generated`
    # above already SUMS every ReAct turn (repl_executor reports
    # primitives.total_tokens_generated), so tool-turn generation already contributes
    # to the throughput/speed objective; these fields make the tool activity itself
    # a recorded, planner-visible signal.
    tools_used: int = 0  # Number of tool invocations during this question.
    tools_called: list[str] = field(default_factory=list)  # Tool names, in call order.
    # 2026-05-23 exogenous-restart resilience (handoff Phase 4).
    # Populated by reading the resilient_post `_meta` dict from the /chat response.
    # exogenous_recovered: a service reload was detected and a retry inside
    #   call_orchestrator_forced succeeded — this QuestionResult's `answer`
    #   came from the retry attempt. Trial-level aggregation surfaces this
    #   as audit info only; does NOT trigger bug_corrupted tagging.
    # exogenous_unrecovered: a service reload was detected but the retry
    #   did not recover. This question has no real answer; if any question
    #   in a trial has this flag, the trial's bug_corrupted_by gets set
    #   to "exogenous_operator_reload" (handled in autopilot.py Phase 5).
    # external_restart: the restart's marker source was != stack_commands
    #   (e.g. a watchdog or manual llama-server invocation). Surfaced for
    #   audit but does not by itself flag the trial as corrupted.
    # retry_count: 0 (clean / real failure) or 1 (one resilient retry).
    exogenous_recovered: bool = False
    exogenous_unrecovered: bool = False
    external_restart: bool = False
    retry_count: int = 0
    eval_partition: str = "core"
    rubric_scores: dict[str, float] = field(default_factory=dict)
    # SCORE-08 (audit 2026-07-20): provenance of the rubric scores for this
    # question — "judge" when at least one cross-family model judge produced
    # parseable scores, "heuristic_fallback" when they fell back to the
    # deterministic structural heuristics (no roles configured / judge
    # error / unparseable), "" for non-rubric questions. Surfaced in the
    # aggregate details as rubric_source_counts so judge-scored and
    # heuristic-scored questions are not indistinguishable downstream.
    rubric_source: str = ""
    host_covariates: dict[str, Any] = field(default_factory=dict)


@dataclass
class _GenOutcome:
    """Handoff between the generation lane and the scoring pool (workers>1 path).

    Carries everything ``_score_generation`` needs to compute the verdict WITHOUT
    re-touching the network, plus ``gen_ended_at_s`` (absolute wall-clock instant
    generation finished) so the sidecar can record ended_at_s as the GENERATION
    interval while scored_at_s marks scoring completion.

    ``final_result`` short-circuits scoring for the generation-exception path: it
    is a fully-formed error ``QuestionResult`` that ``_score_generation`` returns
    unchanged (no scoring is attempted on a call that never produced an answer).
    """

    gen_ended_at_s: float
    final_result: QuestionResult | None = None
    resp: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    error: str | None = None
    tokens: int = 0
    elapsed: float = 0.0
    host_covariates: dict[str, Any] = field(default_factory=dict)
    question_id: Any = "unknown"
    suite: str = "unknown"
    prompt: str = ""
    expected: str = ""
    stable_qid: str = ""
    scoring_method: str = "exact_match"
    scoring_config: dict[str, Any] = field(default_factory=dict)
    eval_partition: str = "core"


# EV-6: Cross-family verification constraint.
# Verifier model must be from a different family than generator to avoid confirmation bias.
# See eval-tower-verification.md for research basis (confirmation bias amplifies 52%→87%).
VERIFICATION_FAMILIES = {
    "qwen": {"Qwen", "qwen", "QwQ"},
    "llama": {"Llama", "llama", "Meta-Llama"},
    "deepseek": {"DeepSeek", "deepseek"},
    "ouro": {"Ouro", "ouro", "ByteDance"},
    "mistral": {"Mistral", "mistral"},
    "gemma": {"Gemma", "gemma", "Google"},
}


def check_cross_family(generator_model: str, verifier_model: str) -> bool:
    """Ensure verifier is from a different model family than generator.

    Returns True if cross-family constraint is satisfied (safe to proceed).
    Returns True if either model family is unknown (permissive default).
    """

    def _get_family(model_name: str) -> str:
        for family, patterns in VERIFICATION_FAMILIES.items():
            if any(p.lower() in model_name.lower() for p in patterns):
                return family
        return "unknown"

    gen_family = _get_family(generator_model)
    ver_family = _get_family(verifier_model)
    return gen_family != ver_family or gen_family == "unknown"


def compute_calibration_metrics(
    confidences: Sequence[float],
    labels: Sequence[float],
) -> dict[str, float | None]:
    """Compatibility wrapper for the shared EV-4 calibration primitive."""
    return _stat_compute_calibration_metrics(confidences, labels)


def _dataset_identity_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def dataset_content_sha256(questions: Sequence[dict[str, Any]]) -> str:
    """Stable SHA-256 over an ordered question set (EV-11 reproducibility stamp).

    Hashes the ordered question identity plus the scoring oracle. Order-sensitive
    by design: the drawn arm's exact question sequence is part of its identity,
    so two arms that sampled different orderings get different digests.
    """
    h = hashlib.sha256()
    for q in questions:
        for field_name in (
            "suite",
            "id",
            "prompt",
            "expected",
            "scoring_method",
            "scoring_config",
        ):
            h.update(_dataset_identity_value(q.get(field_name, "")).encode("utf-8", "replace"))
            h.update(b"\x00")
        h.update(b"\x1e")
    return h.hexdigest()


def _stamp_eval_instrument(
    result: EvalResult,
    *,
    questions: Sequence[dict[str, Any]],
    core_id: str,
    test_profile: dict[str, Any],
) -> EvalResult:
    """Attach machine-checkable dataset/profile identity to a tier result."""
    dataset_sha = dataset_content_sha256(questions)
    tier_mix = question_tier_mix(questions)
    profile_json = json.dumps(test_profile, sort_keys=True, separators=(",", ":"), default=str)
    result.core_id = core_id
    result.details.update(
        {
            "core_id": core_id,
            "dataset_content_sha256": dataset_sha,
            "dataset_sha256": dataset_sha,
            # The tier mix is what couples this instrument to the questions/hour
            # objective: T2/T3 questions cost far more wall-clock than T1, so a mix
            # change moves the objective with no config change. Recorded per trial so a
            # rate comparison can be checked against the set it was measured on.
            "question_tier_mix": tier_mix,
            "test_profile": test_profile,
            "test_profile_json": profile_json,
        }
    )
    if core_id:
        previous = _DATASET_SHA_BY_CORE_ID.get(core_id)
        if previous and previous != dataset_sha:
            warning = {
                "core_id": core_id,
                "previous_dataset_content_sha256": previous,
                "current_dataset_content_sha256": dataset_sha,
            }
            result.details["instrument_drift_warning"] = warning
            log.warning(
                "Eval instrument drift under core_id=%s: %s -> %s",
                core_id,
                previous,
                dataset_sha,
            )
        _DATASET_SHA_BY_CORE_ID[core_id] = dataset_sha

        # Durable check — the in-memory one above cannot see across a daemon restart,
        # which is when a question-pool edit actually lands.
        durable_drift = _record_instrument_identity(
            core_id, dataset_sha, tier_mix, len(questions)
        )
        if durable_drift:
            result.details["instrument_drift_across_restart"] = durable_drift
            log.error(
                "EVAL INSTRUMENT CHANGED under an unchanged core_id=%s. content %s -> %s; "
                "tier mix %s -> %s; n %s -> %s. Trials measured before and after this point "
                "were scored on DIFFERENT question sets — quality and questions/hour are not "
                "comparable across it, and core_id alone cannot express the difference.",
                core_id,
                durable_drift.get("previous_dataset_content_sha256"),
                durable_drift.get("current_dataset_content_sha256"),
                durable_drift.get("previous_tier_mix"),
                durable_drift.get("current_tier_mix"),
                durable_drift.get("previous_n_questions"),
                durable_drift.get("current_n_questions"),
            )
    return result


def _loader_error_eval_result(
    *,
    tier: int,
    source: str,
    error: str,
    core_id: str,
    test_profile: dict[str, Any],
    loader_details: dict[str, Any] | None = None,
    extra_details: dict[str, Any] | None = None,
) -> EvalResult:
    result = EvalResult(
        tier=tier,
        quality=0,
        speed=0,
        cost=0,
        reliability=0,
        details={
            "loader_error": {
                "source": source,
                "error": error,
                "retryable_without_restart": True,
                "details": loader_details or {},
            },
            "decision_grade": False,
            **(extra_details or {}),
        },
    )
    return _stamp_eval_instrument(
        result,
        questions=[],
        core_id=core_id,
        test_profile={
            **test_profile,
            "loader_error": error,
        },
    )


# ── paired-significance screening (config/quant A/B) ─────────────────────────
#
# When eval_tower scores two (or more) arms of a config/quant A/B on the SAME
# question set under the SAME test profile, a raw accuracy delta is not a verdict:
# a 1-2pp gap on a few hundred questions is routinely inside the noise band. This
# hook screens each arm pair with the LANDED clean-room stats primitives —
#   * the exact paired McNemar sign-test over discordant (flip) pairs
#     (paired_stats.mcnemar_from_vectors), and
#   * a per-arm Wilson score interval on the shared question set
#     (stat_tests.wilson_interval)
# — after gating provenance with paired_stats.require_matched_comparison so two
# arms are only ever paired when their dataset_sha256 + test_profile match. It
# REUSES those functions verbatim and reimplements no statistic; it only shapes
# eval_tower's own per-arm outcome vectors into their inputs and collects the
# outputs. Attach the returned dict to an A/B result so downstream verdicts are
# statistically grounded instead of raw-delta.


def _arm_outcome_vector(outcomes: "Mapping[str, Any]") -> dict[str, QuestionOutcome]:
    """Coerce a ``{qid: correct}`` mapping into paired_stats QuestionOutcome form.

    Accepts bare booleans or objects exposing ``.correct``/``["correct"]`` so an
    arm can be fed either raw per-question correctness or richer result records.
    """
    vector: dict[str, QuestionOutcome] = {}
    for qid, value in outcomes.items():
        if isinstance(value, QuestionOutcome):
            vector[str(qid)] = value
            continue
        if isinstance(value, Mapping):
            correct = bool(value.get("correct"))
            suite = str(value.get("suite", ""))
        elif hasattr(value, "correct"):
            correct = bool(getattr(value, "correct"))
            suite = str(getattr(value, "suite", ""))
        else:
            correct = bool(value)
            suite = ""
        vector[str(qid)] = QuestionOutcome(qid=str(qid), suite=suite, correct=correct, trial_id=-1)
    return vector


def screen_paired_arms(
    arms: "Sequence[Mapping[str, Any]]",
    *,
    z: float = DEFAULT_WILSON_Z,
    alpha: float = 0.05,
    mcnemar_exact_max_discordant: int = MCNEMAR_EXACT_MAX_DISCORDANT,
) -> dict[str, Any]:
    """Paired-significance screen over a set of A/B arms.

    Each ``arm`` is a mapping with:
      * ``label``    — arm identity (e.g. an EV-11 ``arm`` stamp or a role name);
      * ``outcomes`` — ``{qid: correct}`` (bool, ``{"correct": ...}``, a
                       ``QuestionResult``-like object, or a ``QuestionOutcome``);
      * ``profile``  — optional ``{dataset_sha256, test_profile}`` (or a
                       :class:`ComparisonProfile`) used to gate pairing.

    For every unordered arm pair whose profiles match (per
    :func:`require_matched_comparison`) it computes the exact paired McNemar p
    over the discordant/flip pairs and a per-arm Wilson interval on the shared
    question set. Pairs whose provenance disagrees are recorded under
    ``mismatched_pairs`` rather than silently compared. Pure/deterministic — no
    inference, no I/O.

    Each matched pair carries a promoted ``verdict`` block
    (:func:`paired_stats.mcnemar_verdict`) — an explicit
    ``indistinguishable`` / ``a_better`` / ``b_better`` decision with its p-value
    (and z on the normal branch), discordant count, and method — so downstream
    consumers read a VERDICT rather than eyeballing the raw discordant counts.

    Returns a JSON-serializable dict with keys ``z``, ``alpha``,
    ``mcnemar_exact_max_discordant``, ``n_arms``, ``arms`` (per-arm accuracy +
    Wilson CI over that arm's own outcomes), ``pairs`` (the paired screen, one
    record per matched pair, each with a ``verdict`` block), and
    ``mismatched_pairs``.
    """
    prepared: list[dict[str, Any]] = []
    for arm in arms:
        label = str(arm.get("label", f"arm{len(prepared)}"))
        vector = _arm_outcome_vector(arm.get("outcomes") or {})
        prepared.append({"label": label, "vector": vector, "profile": arm.get("profile")})

    per_arm: dict[str, Any] = {}
    for arm in prepared:
        vector = arm["vector"]
        total = len(vector)
        correct = sum(1 for o in vector.values() if o.correct)
        lo, hi = wilson_interval(correct, total, z=z)
        per_arm[arm["label"]] = {
            "n": total,
            "correct": correct,
            "accuracy": (correct / total) if total else None,
            "wilson_lower": round(lo, 6),
            "wilson_upper": round(hi, 6),
        }

    pairs: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            arm_a = prepared[i]
            arm_b = prepared[j]
            # Provenance gate — only pair arms scored on the same dataset+profile.
            if (arm_a["profile"] is None) != (arm_b["profile"] is None):
                mismatched.append(
                    {
                        "arm_a": arm_a["label"],
                        "arm_b": arm_b["label"],
                        "reason": "refusing to compare one-sided paired-arm provenance",
                    }
                )
                continue
            # PAIR-1 (commit 1c655076 fix): the XOR guard above only catches the
            # one-sided case; when BOTH arms lack a profile the pair would otherwise
            # fall through to McNemar over unidentified data. paired_stats'
            # require_matched_comparison contract is explicit that missing identity on
            # either arm is itself a refusal, so refuse both-None here too.
            if arm_a["profile"] is None and arm_b["profile"] is None:
                mismatched.append(
                    {
                        "arm_a": arm_a["label"],
                        "arm_b": arm_b["label"],
                        "reason": "provenance_missing_both",
                    }
                )
                continue
            if arm_a["profile"] is not None and arm_b["profile"] is not None:
                try:
                    require_matched_comparison(arm_a["profile"], arm_b["profile"])
                except PairedComparisonMismatchError as exc:
                    mismatched.append(
                        {"arm_a": arm_a["label"], "arm_b": arm_b["label"], "reason": str(exc)}
                    )
                    continue
            result = mcnemar_from_vectors(
                arm_a["vector"], arm_b["vector"], arm_a["label"], arm_b["label"]
            )
            shared = sorted(set(arm_a["vector"]) & set(arm_b["vector"]))
            a_correct = sum(1 for qid in shared if arm_a["vector"][qid].correct)
            b_correct = sum(1 for qid in shared if arm_b["vector"][qid].correct)
            wa_lo, wa_hi = wilson_interval(a_correct, len(shared), z=z)
            wb_lo, wb_hi = wilson_interval(b_correct, len(shared), z=z)
            p = result.p_value_two_sided
            pairs.append(
                {
                    "arm_a": arm_a["label"],
                    "arm_b": arm_b["label"],
                    "shared_qids": result.shared_qids,
                    "a_correct_b_wrong": result.a_correct_b_wrong,
                    "a_wrong_b_correct": result.a_wrong_b_correct,
                    "same_correct": result.same_correct,
                    "same_wrong": result.same_wrong,
                    "mcnemar_p_two_sided": p,
                    "odds_ratio_b_over_a": result.odds_ratio_b_over_a,
                    "accuracy_a": result.accuracy_a,
                    "accuracy_b": result.accuracy_b,
                    "delta_b_minus_a": result.delta_b_minus_a,
                    # Per-arm Wilson CI on the SHARED set (the McNemar denominator).
                    "wilson_a": [round(wa_lo, 6), round(wa_hi, 6)],
                    "wilson_b": [round(wb_lo, 6), round(wb_hi, 6)],
                    # Screening signals for a grounded verdict:
                    "significant": (p < alpha),
                    "wilson_ci_overlap": not (wa_hi < wb_lo or wb_hi < wa_lo),
                    # PROMOTED gating surface: an explicit McNemar verdict
                    # (indistinguishable / a_better / b_better) with p (and z on
                    # the normal branch), discordant count, and method. This is
                    # what downstream keep/prefer decisions read.
                    "verdict": mcnemar_verdict(
                        result.a_correct_b_wrong,
                        result.a_wrong_b_correct,
                        alpha=alpha,
                        exact_max_discordant=mcnemar_exact_max_discordant,
                    ),
                }
            )

    return {
        "z": z,
        "alpha": alpha,
        "mcnemar_exact_max_discordant": mcnemar_exact_max_discordant,
        "n_arms": len(prepared),
        "arms": per_arm,
        "pairs": pairs,
        "mismatched_pairs": mismatched,
    }


def attach_role_paired_verdicts(
    per_role: "Mapping[str, Any]",
    screen: "Mapping[str, Any]",
) -> "Mapping[str, Any]":
    """Thread the paired McNemar verdict into each per-role record.

    ``screen`` is a :func:`screen_paired_arms` result and ``per_role`` maps a role
    name to its arm record (each carrying an ``arm`` label matching the labels in
    ``screen['pairs']``). For every pair a role participates in, this attaches a
    compact, role-oriented ``paired_verdicts`` entry so a downstream consumer
    reading a SINGLE role's record sees a VERDICT — normalized to that role's
    perspective (``this_better`` / ``other_better`` / ``indistinguishable``) —
    rather than only the raw discordant counts buried in the screen block.

    Mutates ``per_role`` in place (and returns it). Roles with no matched pair get
    an empty ``paired_verdicts`` list. Pure/deterministic — no inference, no I/O.
    """
    label_to_role: dict[str, str] = {
        str(rec.get("arm")): role
        for role, rec in per_role.items()
        if isinstance(rec, Mapping) and rec.get("arm")
    }
    for role, rec in per_role.items():
        if not isinstance(rec, dict):
            continue
        arm_label = str(rec.get("arm", ""))
        entries: list[dict[str, Any]] = []
        for pair in screen.get("pairs", []) or []:
            is_a = pair.get("arm_a") == arm_label
            is_b = pair.get("arm_b") == arm_label
            if not (is_a or is_b):
                continue
            verdict_block = pair.get("verdict") or {}
            raw = verdict_block.get("verdict", "indistinguishable")
            if raw == "indistinguishable":
                relative = "indistinguishable"
            elif (raw == "a_better") == is_a:
                relative = "this_better"
            else:
                relative = "other_better"
            other_label = pair.get("arm_b") if is_a else pair.get("arm_a")
            entries.append(
                {
                    "vs_arm": other_label,
                    "vs_role": label_to_role.get(str(other_label)),
                    "verdict": relative,
                    "raw_verdict": raw,
                    "method": verdict_block.get("method", "mcnemar"),
                    "approximation": verdict_block.get("approximation"),
                    "p_value": verdict_block.get("p_value"),
                    "z": verdict_block.get("z"),
                    "n_discordant": verdict_block.get("n_discordant"),
                    "significant": raw != "indistinguishable",
                }
            )
        rec["paired_verdicts"] = entries
    return per_role


def score_math_rebaseline_answers(
    questions: Sequence[dict[str, Any]],
    answers: Sequence[str],
) -> list[bool]:
    """Pure math_verify scoring of (question, model_answer) pairs — no inference.

    EV-11: hard-fails via ``_require_math_verify`` when math-verify is not
    importable rather than silently degrading to exact_match (the 0/1,819-question
    no-op). Reuses ``score_answer_deterministic`` — the exact scoring path
    ``_eval_question`` takes — so a fixture test exercises the real scorer.
    """
    if len(questions) != len(answers):
        raise ValueError(
            f"questions/answers length mismatch: {len(questions)} != {len(answers)}"
        )
    _require_math_verify()
    out: list[bool] = []
    for q, answer in zip(questions, answers):
        out.append(
            bool(
                score_answer_deterministic(
                    answer=answer,
                    expected=str(q.get("expected", "")),
                    scoring_method=str(q.get("scoring_method", "math_verify")),
                    scoring_config=q.get("scoring_config") or {},
                )
            )
        )
    return out


def _configured_rubric_judge_roles() -> list[str]:
    raw = os.environ.get("AUTOPILOT_RUBRIC_JUDGE_ROLES", "").strip()
    if not raw:
        return []
    return [role.strip() for role in raw.split(",") if role.strip()]


def _rubric_judge_timeout_s(default_timeout: int) -> int:
    raw = os.environ.get("AUTOPILOT_RUBRIC_JUDGE_TIMEOUT_S", "").strip()
    if not raw:
        return default_timeout
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("Invalid AUTOPILOT_RUBRIC_JUDGE_TIMEOUT_S=%r; using default", raw)
        return default_timeout


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if 0 <= start < end:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_rubric_judge_scores(text: str) -> dict[str, float]:
    parsed = _extract_json_object(text)
    if not parsed:
        return {}
    raw_scores = parsed.get("scores")
    if not isinstance(raw_scores, dict):
        return {}
    scores: dict[str, float] = {}
    rejected: list[str] = []
    for key, value in raw_scores.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        # SCORE-07 (audit 2026-07-20): VALIDATE the judge's declared [0,1] scale,
        # do NOT clamp. Clamping let a 0-10-scale judge answering "7" saturate to
        # a perfect 1.0 on every dimension. An out-of-range value is instead
        # REJECTED (treated as unparseable for that dimension) so the dimension
        # falls back to the deterministic heuristic path. In-range values —
        # including the exact 0.0 and 1.0 boundaries — are unchanged.
        if numeric < 0.0 or numeric > 1.0:
            rejected.append(str(key))
            continue
        scores[str(key)] = numeric
    if rejected:
        log.warning(
            "rubric judge scale drift: rejected %d out-of-[0,1] dimension score(s): %s",
            len(rejected),
            ", ".join(sorted(rejected)),
        )
    return scores


def _derive_question_confidence(
    *,
    scoring_method: str,
    correct: bool,
    probability_confidence: float | None,
    rubric_scores: Mapping[str, float] | None,
) -> tuple[float, str]:
    """Select ``(confidence, confidence_source)`` for a scored question.

    SCORE-12 (audit 2026-07-20): the former ``scoring_config.get("pass_rate",
    correct)`` read for ``code_execution`` questions was a phantom — it read a
    STATIC dataset field that no scorer writes at runtime, so a dataset carrying
    ``pass_rate: 0.9`` injected a constant fake confidence straight into the
    ECE/AUROC inputs. It is removed.

    ESC-7 extension Option A (operator-approved 2026-07-21): for
    ``code_execution`` questions the **label** is the sandbox test verdict
    (``correct``), while the **confidence** is the model's token-level
    confidence in its own generated solution — the geomean of the completion
    probabilities (``n_probs`` capture is now enabled for code_execution; see
    the call_kwargs block in ``_eval_question``). When probability rows are
    present we use that geomean and stamp ``completion_probabilities_geomean``
    (identical provenance to every other suite), so real code-confidence rows
    contribute to the ``confidence_is_real`` accounting alongside math/exact
    rows. When rows are ABSENT (drafter path, capture failure) we fall back to
    the binary correctness proxy stamped ``code_execution_binary_proxy`` — a
    proxy row correctly drops ``confidence_is_real`` to False for the whole
    batch (fail-closed), since that aggregate requires every source to be the
    real geomean.

    NOISE CAVEAT: a code_execution generation is typically long, so its geomean
    is dominated by many easy, high-probability tokens (boilerplate, syntax) and
    is a *weaker* calibration signal than a short factual answer's geomean —
    treat code-suite ECE/AUROC with that in mind.

    RUBRIC is unchanged: its ``n_probs`` stays suppressed and its confidence IS
    the rubric aggregate (``rubric_score``), which overrides any probability row.

    Precedence: a real completion-probability geomean is set first; then the
    method-specific overrides apply — code_execution keeps the geomean when
    present (else proxy), and rubric always overrides to its aggregate.
    """
    confidence = float(correct)
    source = "binary_correctness_proxy"
    if probability_confidence is not None:
        confidence = probability_confidence
        source = "completion_probabilities_geomean"
    if scoring_method == "code_execution":
        # ESC-7 Option A: keep the completion-probability geomean set above when
        # rows were captured; otherwise fall back to the binary proxy.
        if probability_confidence is None:
            confidence = float(correct)
            source = "code_execution_binary_proxy"
    elif scoring_method == "rubric" and rubric_scores:
        confidence = aggregate_rubric_score(dict(rubric_scores)).score
        source = "rubric_score"
    return confidence, source


class EvalTower:
    """Progressive evaluation: T0 → T1 → T2, with T3 expert/hard workflow eval."""

    def __init__(
        self,
        url: str = ORCHESTRATOR_URL,
        timeout: int = DEFAULT_TIMEOUT,
        sentinel_path: Path | None = None,
        on_question: "Callable[[str], None] | None" = None,
        on_progress: "Callable[[dict[str, Any]], None] | None" = None,
    ):
        self.url = url
        self.timeout = timeout
        self._sentinel_path = sentinel_path or SENTINEL_PATH
        self._sentinels: list[dict] | None = None
        self._sentinels_mtime_ns: int | None = None
        self._sentinel_load_details: dict[str, Any] = {}
        self._pool = None
        self._pool_mtime_ns: int | None = None
        self._pool_load_details: dict[str, Any] = {}
        self._core_cache: dict[str, tuple[list[dict], dict[str, Any], Path]] = {}
        self._trial_id_context: int | None = None
        self._trial_path_context: Path | None = None
        # EV-11c: when set (window-runner / verifier-mode path), _eval_batch writes
        # per-arm question rows into this directory instead of the trial-keyed root.
        self._question_artifact_dir: Path | None = None
        self.on_question = on_question
        self.on_progress = on_progress

    def set_question_artifact_dir(self, directory: str | Path | None) -> None:
        """Persist per-arm question rows under ``directory`` (window-runner path).

        Each ``_eval_batch`` arm writes ``question_results.<label>.jsonl`` here via
        the same append-only, fsync'd writer used on the trial path — so a crashed
        or partially-errored arm leaves a durable, per-question, per-arm record that
        a targeted ``--retry-errors-from`` requeue can read.
        """
        self._question_artifact_dir = Path(directory) if directory is not None else None

    def set_trial_context(
        self,
        trial_id: int | str | None,
        trial_path: str | Path | None = None,
    ) -> None:
        """Set the current AutoPilot trial id for deterministic audit sampling."""
        try:
            self._trial_id_context = int(trial_id) if trial_id is not None else None
        except (TypeError, ValueError):
            self._trial_id_context = None
        self._trial_path_context = Path(trial_path) if trial_path is not None else None

    def _resolve_trial_id(self, trial_id: int | None = None) -> int | None:
        return trial_id if trial_id is not None else self._trial_id_context

    # ── sentinel questions (T0) ──────────────────────────────────

    def _load_sentinels(self) -> list[dict]:
        mtime_ns = _file_mtime_ns(self._sentinel_path)
        if self._sentinels is not None and (
            self._sentinels_mtime_ns is None or self._sentinels_mtime_ns == mtime_ns
        ):
            return self._sentinels

        self._sentinels = None
        self._sentinels_mtime_ns = None
        if mtime_ns is None:
            log.warning("No sentinel file at %s", self._sentinel_path)
            self._sentinel_load_details = {
                "source": str(self._sentinel_path),
                "error": "missing_sentinel_file",
            }
            return []

        try:
            loaded = yaml.safe_load(self._sentinel_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load sentinel file %s: %s", self._sentinel_path, exc)
            self._sentinel_load_details = {
                "source": str(self._sentinel_path),
                "error": "sentinel_load_failed",
                "exception": str(exc),
            }
            return []
        if loaded is None:
            loaded = []
        if not isinstance(loaded, list):
            log.warning("Sentinel file %s must contain a YAML list", self._sentinel_path)
            self._sentinel_load_details = {
                "source": str(self._sentinel_path),
                "error": "sentinel_yaml_not_list",
                "loaded_type": type(loaded).__name__,
            }
            return []

        sentinels, details = _validate_eval_question_rows(
            loaded,
            source=str(self._sentinel_path),
        )
        self._sentinel_load_details = details
        if not sentinels:
            log.warning("Sentinel file %s contains no valid eval questions", self._sentinel_path)
            return []
        self._sentinels = sentinels
        self._sentinels_mtime_ns = mtime_ns
        return self._sentinels

    def _load_tool_sentinels(self) -> list[dict]:
        """Tool-use sentinels — appended to T0 only when AUTOPILOT_TOOL_SENTINELS=1.

        Returns [] (byte-identical legacy behavior) unless the env flag is set
        AND tool_sentinels.yaml exists. These questions pin force_mode="repl" and
        require a counted `get_eval_secret` tool call, moving tools_used /
        tool_helpfulness off their structural 0.

        The secret VALUES are minted at runtime by the orchestrator (never in
        source/YAML) and persisted to a tmpfs path the model can't read; here we
        inject each question's real `expected` from that ground truth, keyed by
        the name="..." in its prompt. When ground truth is unavailable the
        placeholder `expected` is left in place — it never matches a real answer,
        so the question scores INCORRECT rather than spuriously passing on "".
        """
        if os.environ.get("AUTOPILOT_TOOL_SENTINELS") != "1":
            return []
        if not TOOL_SENTINEL_PATH.exists():
            log.warning("AUTOPILOT_TOOL_SENTINELS=1 but no file at %s", TOOL_SENTINEL_PATH)
            return []
        loaded = yaml.safe_load(TOOL_SENTINEL_PATH.read_text(encoding="utf-8")) or []
        if not isinstance(loaded, list):
            log.warning("tool_sentinels: %s must contain a YAML list", TOOL_SENTINEL_PATH)
            return []
        loaded, details = _validate_eval_question_rows(
            loaded,
            source=str(TOOL_SENTINEL_PATH),
        )
        if not loaded:
            log.warning("tool_sentinels: no valid rows after schema validation: %s", details)
            return []
        try:
            from src.tools.eval_secret import load_persisted_secrets

            secrets = load_persisted_secrets()
        except Exception as e:  # noqa: BLE001
            log.warning("tool_sentinels: could not load runtime secrets: %s", e)
            secrets = {}
        if not secrets:
            log.warning(
                "tool_sentinels: no runtime eval secrets available; tool_use "
                "questions will score INCORRECT until the orchestrator (with "
                "AUTOPILOT_TOOL_SENTINELS=1) has minted them."
            )
        for q in loaded:
            m = _re.search(r'name=\\?"([A-Za-z0-9_]+)\\?"', q.get("prompt", ""))
            val = secrets.get(m.group(1).lower()) if m else None
            if val:
                q["expected"] = val  # real runtime ground truth (never on disk)
            # else: leave the non-matching placeholder from the YAML in place.
        log.info("Loaded %d tool-use sentinels (AUTOPILOT_TOOL_SENTINELS=1)", len(loaded))
        return loaded

    def _load_pool(self):
        """Load question pool for T1/T2 validation questions."""
        if self._pool is not None and self._pool_mtime_ns is None:
            return self._pool
        try:
            _research_root_path = _research_root()
            question_pool = _load_research_benchmark_module("question_pool")

            default_pool_path = _research_root_path / "benchmarks" / "prompts" / "question_pool.jsonl"
            pool_path = Path(getattr(question_pool, "POOL_FILE", default_pool_path))
            mtime_ns = _file_mtime_ns(pool_path)
            if self._pool is not None and self._pool_mtime_ns == mtime_ns:
                return self._pool

            self._pool = None
            self._pool_mtime_ns = None
            if mtime_ns is None:
                self._pool_load_details = {
                    "source": str(pool_path),
                    "error": "missing_question_pool",
                }
                log.warning("Question pool file missing: %s", pool_path)
                return {}

            raw_pool = question_pool.load_pool()
            if not raw_pool:
                self._pool_load_details = {
                    "source": str(pool_path),
                    "error": "empty_question_pool",
                }
                log.warning("Question pool is empty: %s", pool_path)
                return {}

            pool, details = _validate_question_pool(raw_pool, source=str(pool_path))
            self._pool_load_details = details
            if not pool:
                log.warning("Question pool has no valid eval questions after validation")
                return {}

            self._pool = pool
            self._pool_mtime_ns = mtime_ns
            return self._pool
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load question pool: %s", e)
            self._pool = None
            self._pool_mtime_ns = None
            self._pool_load_details = {
                "error": "question_pool_load_failed",
                "exception": str(e),
            }
            return {}

    def _core_path(self, core_id: str) -> Path:
        override = os.environ.get("AUTOPILOT_T1_CORE_PATH", "").strip()
        if override:
            return Path(override)
        return DEFAULT_CORE_DIR / f"{core_id}.jsonl"

    def _load_designed_core(self, core_id: str) -> tuple[list[dict], dict[str, Any], Path]:
        """Load a fixed, designed T1 core.

        JSONL rows may be full question records or id-only references into the
        question pool. The metadata row is optional:
        {"__core_metadata__": true, "core_id": "core_v2", ...}
        """
        path = self._core_path(core_id)
        cache_key = f"{core_id}\0{path}"
        if cache_key in self._core_cache:
            return self._core_cache[cache_key]
        if not path.exists():
            raise FileNotFoundError(f"T1 core file not found: {path}")

        items: list[tuple[str, dict[str, Any] | str]] = []
        metadata: dict[str, Any] = {}
        with open(path) as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{line_no}: core row must be a JSON object")
                if obj.get(_CORE_METADATA_KEY):
                    metadata = obj
                    continue
                if obj.get("prompt") and obj.get("suite"):
                    items.append(("question", obj))
                    continue
                qid = obj.get("id") or obj.get("question_id")
                if qid:
                    items.append(("id", str(qid)))
                    continue
                raise ValueError(
                    f"{path}:{line_no}: core row must be metadata, a full question, or an id reference"
                )

        if not items:
            raise ValueError(f"{path}: designed core contains no questions")

        lookup: dict[str, dict[str, Any]] = {}
        if any(kind == "id" for kind, _ in items):
            pool = self._load_pool()
            if not pool:
                raise ValueError(f"{path}: question pool unavailable for id references")
            for suite_qs in pool.values():
                for question in suite_qs:
                    qid = str(question.get("id", "")).strip()
                    if not qid:
                        continue
                    lookup.setdefault(qid, question)
                    suite = str(question.get("suite", "")).strip()
                    if suite:
                        lookup.setdefault(f"{suite}/{qid}", question)

        questions: list[dict] = []
        missing_ids: list[str] = []
        for kind, value in items:
            if kind == "question":
                questions.append(value)
                continue
            qid = str(value)
            bare_id = qid.split("/", 1)[1] if "/" in qid else qid
            question = lookup.get(qid) or lookup.get(bare_id)
            if question is None:
                missing_ids.append(qid)
                continue
            questions.append(question)

        if missing_ids:
            raise ValueError(
                f"{path}: {len(missing_ids)} core question id(s) not found: {missing_ids[:5]}"
            )

        unscoreable = [q.get("id", "") for q in questions if not _is_scoreable_question(q)]
        if unscoreable:
            raise ValueError(
                f"{path}: designed core contains {len(unscoreable)} unscoreable item(s): "
                f"{unscoreable[:5]}"
            )

        core_meta_id = str(metadata.get("core_id", core_id))
        if core_meta_id != core_id:
            raise ValueError(
                f"{path}: metadata core_id={core_meta_id!r} does not match requested {core_id!r}"
            )

        loaded = (questions, metadata, path)
        self._core_cache[cache_key] = loaded
        return loaded

    def _load_audit_block(
        self,
        core_questions: list[dict],
        audit_n: int,
        trial_id: int,
        core_id: str,
    ) -> tuple[list[dict], int]:
        pool = self._load_pool()
        if not pool:
            raise ValueError("question pool unavailable for W6 audit block")

        excluded_qids = {_question_qid(q) for q in core_questions}
        filtered: dict[str, list[dict]] = {}
        for suite, suite_qs in pool.items():
            keep = [
                q
                for q in suite_qs
                if _question_qid(q) not in excluded_qids and _is_scoreable_question(q)
            ]
            if keep:
                filtered[suite] = keep
        if not filtered:
            raise ValueError("question pool has no scoreable non-core W6 audit items")

        seed = _audit_seed(trial_id, core_id)
        questions = _sample_scoreable_eval_questions(
            filtered,
            audit_n,
            random.Random(seed),
        )
        return questions, seed

    def _t1_core_exclusion_qids(
        self,
        pool: dict[str, list[dict]],
        *,
        seed: int,
    ) -> tuple[set[str], dict[str, Any]]:
        configured_core_id = os.environ.get("AUTOPILOT_T1_CORE_ID", "").strip()
        configured_core_path = os.environ.get("AUTOPILOT_T1_CORE_PATH", "").strip()
        policy: dict[str, Any] = {
            "enabled": True,
            "version": "t2-nonpromotion-excludes-t1-core-v1",
        }

        if configured_core_path and not configured_core_id:
            raise ValueError("AUTOPILOT_T1_CORE_PATH requires AUTOPILOT_T1_CORE_ID")

        if configured_core_id:
            questions, _metadata, core_file = self._load_designed_core(configured_core_id)
            policy.update(
                {
                    "source": "designed_core",
                    "core_id": configured_core_id,
                    "core_path": str(core_file),
                    "actual_n": len(questions),
                }
            )
        else:
            questions = _sample_scoreable_eval_questions(
                pool,
                EVAL_T1_SPEC_N,
                random.Random(int(seed)),
            )
            policy.update(
                {
                    "source": "legacy_pool_seed",
                    "core_id": f"legacy_pool_seed_{int(seed)}_n{EVAL_T1_SPEC_N}",
                    "seed": int(seed),
                    "requested_n": EVAL_T1_SPEC_N,
                    "actual_n": len(questions),
                }
            )

        identities = _question_identity_union(questions)
        policy["excluded_t1_core_qids"] = len(identities)
        return identities, policy

    # ── single question evaluation ───────────────────────────────

    def _rubric_scores_for_answer(
        self,
        *,
        q: dict,
        answer: str,
        generator_model: str,
        tool_events: list[str],
        client: httpx.Client,
    ) -> tuple[dict[str, float], str]:
        """Return ``(rubric_scores, rubric_source)``.

        SCORE-08: ``rubric_source`` is ``"judge"`` when at least one cross-family
        model judge produced parseable scores, else ``"heuristic_fallback"`` (no
        roles configured, every judge errored, or every judge response was
        unparseable). The scores themselves are unchanged by this provenance
        stamp.
        """
        fallback = deterministic_rubric_fallback(
            answer,
            expected_contains=q.get("expected_contains") or (),
            tool_events=tool_events,
        )
        judge_roles = _configured_rubric_judge_roles()
        if not judge_roles:
            return fallback, "heuristic_fallback"

        prompt = build_rubric_judge_prompt(
            task_prompt=str(q.get("prompt", "")),
            answer=answer,
            expected_contains=q.get("expected_contains") or (),
        )
        judge_scores: list[dict[str, float]] = []
        for role in judge_roles:
            if not check_cross_family(generator_model, role):
                log.warning(
                    "Skipping rubric judge role %s; not cross-family with %s",
                    role,
                    generator_model,
                )
                continue
            resp = call_orchestrator_forced(
                prompt=prompt.prompt,
                force_role=role,
                force_mode="direct",
                url=self.url,
                timeout=_rubric_judge_timeout_s(self.timeout),
                client=client,
                allow_delegation=False,
                scoring_method="rubric_judge",
                watcher=getattr(self, "watcher", None),
            )
            if resp.get("error"):
                log.warning("rubric judge %s failed: %s", role, resp.get("error"))
                continue
            parsed = _parse_rubric_judge_scores(str(resp.get("answer", "")))
            if parsed:
                judge_scores.append(parsed)

        if not judge_scores:
            return fallback, "heuristic_fallback"
        combined = dict(fallback)
        dimensions = sorted({dim for scores in judge_scores for dim in scores})
        for dim in dimensions:
            values = [scores[dim] for scores in judge_scores if dim in scores]
            if values:
                combined[dim] = sum(values) / len(values)
        return combined, "judge"

    def _eval_question(self, q: dict, client: httpx.Client) -> QuestionResult:
        """Evaluate a single question through the orchestrator (generate then score).

        Serial / back-compat entry point: runs both phases inline so the
        single-worker (``workers <= 1``) path in ``_eval_batch`` and every direct
        caller are behaviorally identical to the pre-pipeline implementation. The
        workers>1 path in ``_eval_batch`` instead drives ``_generate_question`` and
        ``_score_generation`` on separate pools so scoring is decoupled from the
        topology-capped generation lanes (see the module docstring).
        """
        outcome = self._generate_question(q, client)
        return self._score_generation(q, outcome, client)

    def _generate_question(self, q: dict, client: httpx.Client) -> "_GenOutcome":
        """GENERATION phase: run the orchestrator request + REL-1 guards, no scoring.

        Returns a ``_GenOutcome`` carrying the response payload and timing that
        ``_score_generation`` consumes to produce the verdict. A generation-time
        exception is captured as a fully-formed error ``QuestionResult`` on
        ``final_result`` so the scoring phase can pass it through unchanged.
        """
        prompt = q.get("prompt", "")
        expected = q.get("expected", "")
        qid = q.get("id", q.get("question_id", "unknown"))
        suite = q.get("suite", "unknown")
        stable_qid = str(q.get("qid") or q.get("stable_qid") or "").strip()
        if not stable_qid:
            stable_qid = _stable_question_qid(str(suite), str(prompt))
        scoring_method = q.get("scoring_method", "exact_match")
        # SCORE-12 guard: a dataset row carrying `scoring_config: null` (or any
        # non-dict) previously raised AttributeError on the `.get()` reads below,
        # converting a scored question into an error result. Coerce to an empty
        # dict so downstream scoring/threshold/confidence reads are safe.
        scoring_config = q.get("scoring_config")
        if not isinstance(scoring_config, dict):
            scoring_config = {}
        image_path = q.get("image_path", "")
        eval_partition = str(q.get("eval_partition") or "core")

        if self.on_question:
            self.on_question(prompt)

        start = time.time()
        try:
            call_kwargs = {
                "prompt": prompt,
                # Let routing decide unless the question pins a mode/role. The
                # tool_use suite pins force_mode="repl" so the REPL CALL(...)
                # path (what production actually uses) is exercised
                # deterministically instead of being left to the router.
                # Defaults are "" → existing questions are unchanged.
                "force_role": q.get("force_role", ""),
                "force_mode": q.get("force_mode", ""),
                "url": self.url,
                "timeout": self.timeout,
                "image_path": image_path,
                "client": client,
                "watcher": getattr(self, "watcher", None),
                "request_priority": "background",
                "workload_class": "eval_batch",
            }
            eval_batch_id = str(q.get("_eval_batch_id") or "").strip()
            if eval_batch_id:
                call_kwargs["batch_id"] = eval_batch_id
            if q.get("allow_delegation") is not None:
                call_kwargs["allow_delegation"] = q["allow_delegation"]
            if "tools" in q:
                call_kwargs["tools"] = q.get("tools")
            if "tool_choice" in q:
                call_kwargs["tool_choice"] = q.get("tool_choice")
            if _is_scoreable_question(q) and not _is_rubric_scored_question(q):
                # ESC-7 extension Option A (operator-approved 2026-07-21):
                # code_execution questions now REQUEST n_probs so the model's
                # generation-logprob geomean can serve as confidence (the
                # sandbox test verdict remains the correctness LABEL). Prior to
                # this, code_execution was excluded here (e26a7cb3) and could
                # only ever carry the binary proxy. RUBRIC stays suppressed —
                # its confidence IS the rubric aggregate, not a token logprob.
                n_probs = _nonnegative_int(
                    q.get("n_probs", _env_int("AUTOPILOT_EVAL_LOGPROB_N_PROBS", 5)),
                    default=5,
                )
                if n_probs > 0:
                    call_kwargs["n_probs"] = n_probs
            prompt_root = str(q.get("_prompt_root") or "").strip()
            if prompt_root:
                call_kwargs["prompt_root"] = prompt_root
            resp = call_orchestrator_forced(**call_kwargs)
            elapsed = time.time() - start
            answer = resp.get("answer", "")
            error = resp.get("error")

            # ── Guard 1 (REL-1): in-band error surfaced as an answer ──────
            # The circuit breaker can return "[ERROR: Backend unavailable
            # (circuit open): ...]" as the answer with error=None. Convert it
            # into an ERROR row so it is EXCLUDED from the quality denominator
            # and counted against reliability — never scored as a wrong answer.
            if not error:
                _inband = _inband_error_text(answer)
                if _inband is not None:
                    error = _inband
                    log.error(
                        "REL-1 in-band error surfaced as answer "
                        "(qid=%s suite=%s force_role=%s): %s",
                        stable_qid,
                        suite,
                        q.get("force_role", "") or "<router>",
                        _inband[:200],
                    )

            # ── Guard 2 (REL-1): forced-role integrity ────────────────────
            # If this question pinned a role for a role-attributed measurement
            # but the orchestrator served it from a different role (silent
            # circuit_open fallback), reject the measurement as an ERROR row
            # rather than mis-attributing a cross-role number. Log loudly with
            # both roles.
            if not error:
                _served_by = _forced_role_serving_mismatch(
                    q.get("force_role"), resp
                )
                if _served_by is not None:
                    _forced = str(q.get("force_role") or "").strip()
                    error = (
                        f"forced_role_fallback: forced={_forced} "
                        f"served_by={_served_by}"
                    )
                    log.error(
                        "REL-1 forced-role integrity violation "
                        "(qid=%s suite=%s): forced=%s but served_by=%s — "
                        "rejecting cross-role measurement",
                        stable_qid,
                        suite,
                        _forced,
                        _served_by,
                    )

            tokens = _nonnegative_int(resp.get("tokens_generated", 0))
            host_covariates = _capture_host_timing_covariates(
                tokens_generated=tokens,
                elapsed_s=elapsed,
            )
            return _GenOutcome(
                # Generation interval end = start + elapsed (the sidecar derives
                # started_at_s = ended_at_s - elapsed_s from this).
                gen_ended_at_s=start + elapsed,
                resp=resp,
                answer=answer,
                error=error,
                tokens=tokens,
                elapsed=elapsed,
                host_covariates=host_covariates,
                question_id=qid,
                suite=suite,
                prompt=prompt,
                expected=expected,
                stable_qid=stable_qid,
                scoring_method=scoring_method,
                scoring_config=scoring_config,
                eval_partition=eval_partition,
            )
        except Exception as e:
            elapsed = time.time() - start
            host_covariates = _capture_host_timing_covariates(
                tokens_generated=0,
                elapsed_s=elapsed,
            )
            failed = QuestionResult(
                question_id=qid,
                suite=suite,
                prompt=prompt,
                expected=expected,
                qid=stable_qid,
                error=str(e),
                elapsed_s=elapsed,
                eval_partition=eval_partition,
                host_covariates=host_covariates,
            )
            return _GenOutcome(gen_ended_at_s=start + elapsed, final_result=failed)

    def _score_generation(
        self, q: dict, outcome: "_GenOutcome", client: httpx.Client
    ) -> QuestionResult:
        """SCORING phase: compute the verdict for an already-generated answer.

        Pure scheduling split from ``_generate_question`` — the scorer functions,
        per-execution timeouts, and REL-1/error classification are unchanged. A
        ``final_result`` (generation exception) is returned as-is: a question that
        never produced an answer is not scored.
        """
        if outcome.final_result is not None:
            return outcome.final_result

        resp = outcome.resp or {}
        answer = outcome.answer
        error = outcome.error
        expected = outcome.expected
        scoring_method = outcome.scoring_method
        scoring_config = outcome.scoring_config

        correct = False
        rubric_scores: dict[str, float] = {}
        rubric_source = ""
        if not error and _is_scoreable_question(q):
            if _is_rubric_scored_question(q):
                rubric_scores, rubric_source = self._rubric_scores_for_answer(
                    q=q,
                    answer=answer,
                    generator_model=str(resp.get("model") or resp.get("routed_to") or ""),
                    tool_events=list(resp.get("tools_called") or []),
                    client=client,
                )
                threshold = float((scoring_config or {}).get("rubric_pass_threshold", 0.60))
                correct = aggregate_rubric_score(rubric_scores).score >= threshold
                scoring_method = "rubric"
            else:
                if scoring_method == "math_verify":
                    # EV-11: guarantee math_verify actually runs; never let a
                    # missing library silently degrade to exact_match.
                    _require_math_verify()
                verdict, scoring_error = score_answer_or_error(
                    answer=answer,
                    expected=expected,
                    scoring_method=scoring_method,
                    scoring_config=scoring_config,
                )
                if scoring_error is not None:
                    # Generation completed successfully.  Preserve its answer,
                    # token accounting, route, and timing as durable evidence;
                    # only the verdict is unavailable.  This lets a caller
                    # replay the scorer tail without regenerating model output.
                    error = scoring_error
                else:
                    correct = bool(verdict)

        # EV-CONF: prefer model probability rows when requested/available.
        # Fall back to historical scoring-derived proxies for paths that do
        # not expose llama.cpp completion_probabilities. SCORE-12: the phantom
        # static `pass_rate` read is gone — see _derive_question_confidence.
        probability_confidence = _completion_probabilities_confidence(
            resp.get("completion_probabilities")
        )
        confidence, confidence_source = _derive_question_confidence(
            scoring_method=scoring_method,
            correct=correct,
            probability_confidence=probability_confidence,
            rubric_scores=rubric_scores,
        )

        # 2026-05-23 Phase 4 — exogenous-restart metadata propagation.
        # call_orchestrator_forced attaches the resilient_post meta dict
        # as resp["_meta"] when watcher is set. Surface the classification
        # bits onto QuestionResult so _aggregate can roll them up into
        # the trial-level EvalResult.
        meta = resp.get("_meta") or {}
        failure_provenance = (
            dict(resp["failure_provenance"])
            if error and isinstance(resp.get("failure_provenance"), dict)
            else None
        )
        provenance_role = (
            str(failure_provenance.get("role") or "")
            if failure_provenance is not None
            else ""
        )
        return QuestionResult(
            question_id=outcome.question_id,
            suite=outcome.suite,
            prompt=outcome.prompt,
            expected=expected,
            qid=outcome.stable_qid,
            answer=answer,
            correct=correct,
            error=error,
            failure_provenance=failure_provenance,
            tokens_generated=outcome.tokens,
            elapsed_s=outcome.elapsed,
            route_used=str(
                resp.get("routed_to") or resp.get("model") or provenance_role
            ),
            cost_tier=resp.get("cost_tier", 0),
            scoring_method=scoring_method,
            partial=bool(resp.get("partial", False)),
            degraded=bool(resp.get("degraded", False)),
            confidence=confidence,
            confidence_source=confidence_source,
            branching_density=_compute_branching_density(answer),
            tools_used=int(resp.get("tools_used", 0) or 0),
            tools_called=list(resp.get("tools_called") or []),
            exogenous_recovered=bool(meta.get("exogenous_recovered", False)),
            exogenous_unrecovered=bool(meta.get("exogenous_unrecovered", False)),
            external_restart=bool(meta.get("external_restart", False)),
            retry_count=int(meta.get("retry_count", 0)),
            eval_partition=outcome.eval_partition,
            rubric_scores=rubric_scores,
            rubric_source=rubric_source,
            host_covariates=outcome.host_covariates,
        )

    def _failed_question_result(
        self,
        q: dict,
        *,
        elapsed_s: float,
        error: str,
    ) -> QuestionResult:
        prompt = q.get("prompt", "")
        suite = q.get("suite", "unknown")
        stable_qid = str(q.get("qid") or q.get("stable_qid") or "").strip()
        if not stable_qid:
            stable_qid = _stable_question_qid(str(suite), str(prompt))
        host_covariates = _capture_host_timing_covariates(
            tokens_generated=0,
            elapsed_s=elapsed_s,
        )
        return QuestionResult(
            question_id=q.get("id", q.get("question_id", "unknown")),
            suite=suite,
            prompt=prompt,
            expected=q.get("expected", ""),
            qid=stable_qid,
            error=error,
            elapsed_s=elapsed_s,
            eval_partition=str(q.get("eval_partition") or "core"),
            host_covariates=host_covariates,
        )

    def _eval_batch(
        self,
        questions: list[dict],
        client: httpx.Client,
        log_every: int | None = None,
        label: str = "",
    ) -> list[QuestionResult]:
        """Evaluate a batch of questions, fanning out across N workers.

        Results are returned in the same order as `questions`. With
        AUTOPILOT_EVAL_CONCURRENCY > 1, the orchestrator's
        ConcurrencyAwareBackend spreads inflight requests across the
        full instance + idle quarter instances (frontdoor: 1 full + 4
        quarters). With concurrency=1, behavior matches the legacy
        serial loop.
        """
        n = len(questions)
        if n == 0:
            return []
        workers = min(n, _eval_concurrency(_forced_roles_for_questions(questions)))
        eval_batch_id = _eval_batch_id(
            label=label,
            n_questions=n,
            started_at_s=time.time(),
        )
        dispatch_questions = [
            {**q, "_eval_batch_id": str(q.get("_eval_batch_id") or eval_batch_id)}
            for q in questions
        ]
        results: list[QuestionResult | None] = [None] * n
        batch_start = time.time()
        writer: _EvalQuestionJsonlWriter | None = None
        artifact_root, artifact_root_source = _eval_artifact_root(self._trial_path_context)
        explicit_path: Path | None = None
        if self._question_artifact_dir is not None:
            fname = f"question_results.{_safe_artifact_name(label, fallback='arm')}.jsonl"
            explicit_path = Path(self._question_artifact_dir) / fname
            artifact_root_source = "window_output_dir"
        try:
            pending_writer = _EvalQuestionJsonlWriter(
                root=artifact_root,
                root_source=artifact_root_source,
                eval_batch_id=eval_batch_id,
                trial_id=self._resolve_trial_id(),
                label=label,
                requested_n=n,
                concurrency=workers,
                path=explicit_path,
            )
            pending_writer.append_start()
            writer = pending_writer
        except Exception as exc:  # noqa: BLE001
            if "pending_writer" in locals():
                pending_writer.close()
            log.warning("EvalTower question-result sidecar disabled: %s", exc)
            writer = None

        def _ordinal_for_pos(pos: int) -> int:
            # Resume/subset support: a question may carry an explicit `_ordinal`
            # (its position in the ORIGINAL full dataset) so a --resume-incomplete
            # run of just the remainder stamps original-dataset ordinals into the
            # sidecar — making prior+new rows mergeable by ordinal with no
            # collisions. Absent `_ordinal`, ordinal == list position (unchanged).
            try:
                return int(dispatch_questions[pos].get("_ordinal", pos))
            except (TypeError, ValueError, IndexError):
                return pos

        def append_question_result(
            pos: int,
            result: QuestionResult,
            *,
            generated_at_s: float | None = None,
            scored_at_s: float | None = None,
        ) -> None:
            if writer is None:
                return
            try:
                writer.append_result(
                    ordinal=_ordinal_for_pos(pos),
                    result=result,
                    generated_at_s=generated_at_s,
                    scored_at_s=scored_at_s,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("EvalTower question-result sidecar append failed: %s", exc)

        def append_complete_marker(completed_n: int, elapsed_s: float) -> None:
            if writer is None:
                return
            try:
                writer.append_complete(completed_n=completed_n, elapsed_s=elapsed_s)
            except Exception as exc:  # noqa: BLE001
                log.warning("EvalTower question-result sidecar complete marker failed: %s", exc)

        def mark_abandoned(idx: int, result: QuestionResult, *, drain_timeout_s: float) -> None:
            result.degraded = True
            suffix = (
                "eval_orphan_contamination: request may still be decoding "
                f"server-side after {drain_timeout_s:.1f}s drain"
            )
            result.error = f"{result.error}; {suffix}" if result.error else suffix
            results[idx] = result

        if workers <= 1:
            ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"eval-{label}-serial")
            try:
                wall_budget_s = _eval_batch_wall_budget_s(
                    n_questions=n,
                    workers=workers,
                    request_timeout_s=self.timeout,
                )
                no_progress_timeout_s = _eval_no_progress_timeout_s(self.timeout)
                drain_timeout_s = _eval_orphan_drain_timeout_s(self.timeout)
                for i, q in enumerate(dispatch_questions):
                    fut = ex.submit(self._eval_question, q, client)
                    completed, not_done = wait(
                        {fut},
                        timeout=no_progress_timeout_s or None,
                        return_when=FIRST_COMPLETED,
                    )
                    if not completed:
                        elapsed = time.time() - batch_start
                        fut.cancel()
                        log.error(
                            "%s serial eval question %d/%d made no progress for %.1fs; "
                            "failing current and remaining question(s) closed",
                            label,
                            i + 1,
                            n,
                            no_progress_timeout_s,
                        )
                        results[i] = self._failed_question_result(
                            questions[i],
                            elapsed_s=elapsed,
                            error=(
                                "eval_no_progress_timeout: serial eval question "
                                f"made no progress for {no_progress_timeout_s:.1f}s"
                            ),
                        )
                        if not_done:
                            drained, still_running = wait(not_done, timeout=drain_timeout_s)
                            if drained:
                                log.info(
                                    "%s serial eval worker drained after no-progress timeout",
                                    label,
                                )
                            if still_running:
                                mark_abandoned(i, results[i], drain_timeout_s=drain_timeout_s)
                        results[i].eval_concurrency = workers
                        append_question_result(i, results[i])
                        for j in range(i + 1, n):
                            results[j] = self._failed_question_result(
                                questions[j],
                                elapsed_s=time.time() - batch_start,
                                error="eval_cancelled_after_no_progress_timeout",
                            )
                            results[j].eval_concurrency = workers
                            append_question_result(j, results[j])
                        break

                    try:
                        results[i] = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        results[i] = self._failed_question_result(
                            questions[i],
                            elapsed_s=time.time() - batch_start,
                            error=str(exc),
                        )
                    results[i].eval_concurrency = workers
                    append_question_result(i, results[i])
                    elapsed = time.time() - batch_start
                    correct_so_far = sum(1 for r in results if r and r.correct)
                    self._emit_progress(
                        label=label,
                        completed_questions=i + 1,
                        total_questions=n,
                        correct_questions=correct_so_far,
                        concurrency=workers,
                    )
                    if log_every and (i + 1) % log_every == 0:
                        log.info(
                            "%s progress: %d/%d (%.0f%% correct)",
                            label,
                            i + 1,
                            n,
                            100 * correct_so_far / (i + 1),
                        )
                    if wall_budget_s > 0 and elapsed >= wall_budget_s and (i + 1) < n:
                        log.error(
                            "%s serial eval exceeded wall budget %.1fs after %d/%d "
                            "question(s); failing remaining question(s) closed",
                            label,
                            wall_budget_s,
                            i + 1,
                            n,
                        )
                        for j in range(i + 1, n):
                            results[j] = self._failed_question_result(
                                questions[j],
                                elapsed_s=time.time() - batch_start,
                                error=(
                                    "eval_wall_budget_timeout: serial eval exceeded "
                                    f"{wall_budget_s:.1f}s"
                                ),
                            )
                            results[j].eval_concurrency = workers
                            append_question_result(j, results[j])
                        break
                batch_wall_s = time.time() - batch_start
                out = [r for r in results if r is not None]
                for r in out:
                    r.eval_concurrency = workers
                    r.eval_wall_s = batch_wall_s
                append_complete_marker(len(out), batch_wall_s)
                return out
            finally:
                ex.shutdown(wait=False, cancel_futures=True)
                if writer is not None:
                    writer.close()

        # ── Pipelined generation + scoring (workers > 1) ─────────────────────
        # Generation lanes stay at the topology-capped `workers` width; scoring
        # runs on a separate, wider pool (`scoring_workers`) so a scoring-bound
        # suite (HE-R+ code_execution: ~11s client-CPU scoring per ~1s decode)
        # no longer idles the serving fleet. A completed generation hands its
        # un-scored result to the scoring pool and the lane immediately picks up
        # the next question. See the module docstring + _eval_scoring_concurrency.
        scoring_workers = _eval_scoring_concurrency(workers)
        # Backpressure: never admit new generation while the scoring pool already
        # holds >= 2x its width of un-scored work, so a fast generator cannot pile
        # unbounded memory on a slow scorer.
        backpressure_cap = max(2 * scoring_workers, workers)
        no_progress_timeout_s = _eval_no_progress_timeout_s(self.timeout)
        drain_timeout_s = _eval_orphan_drain_timeout_s(self.timeout)
        gen_ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"eval-{label}-gen")
        score_ex = ThreadPoolExecutor(
            max_workers=scoring_workers, thread_name_prefix=f"eval-{label}-score"
        )
        done = 0
        pending_gen: deque[tuple[int, dict]] = deque(enumerate(dispatch_questions))
        gen_future_to_idx: dict[Future, int] = {}
        score_future_to_idx: dict[Future, int] = {}
        gen_ended_at: dict[int, float] = {}

        def _idx_of(fut: Future) -> int:
            idx = gen_future_to_idx.get(fut)
            if idx is None:
                idx = score_future_to_idx.get(fut)
            return idx

        def admit_generation() -> None:
            while (
                len(gen_future_to_idx) < workers
                and pending_gen
                and len(score_future_to_idx) < backpressure_cap
            ):
                idx, gq = pending_gen.popleft()
                gen_future_to_idx[gen_ex.submit(self._generate_question, gq, client)] = idx

        def finalize_scored(idx: int, result: QuestionResult, *, generated_at_s: float | None) -> None:
            nonlocal done
            result.eval_concurrency = workers
            results[idx] = result
            append_question_result(
                idx, result, generated_at_s=generated_at_s, scored_at_s=time.time()
            )
            done += 1
            correct_so_far = sum(1 for r in results if r and r.correct)
            self._emit_progress(
                label=label,
                completed_questions=done,
                total_questions=n,
                correct_questions=correct_so_far,
                concurrency=workers,
            )
            if log_every and done % log_every == 0:
                log.info(
                    "%s progress: %d/%d (%.0f%% correct, concurrency=%d)",
                    label,
                    done,
                    n,
                    100 * correct_so_far / done,
                    workers,
                )
                self._emit_progress(
                    label=label,
                    completed_questions=done,
                    total_questions=n,
                    correct_questions=correct_so_far,
                    concurrency=workers,
                )

        try:
            while pending_gen or gen_future_to_idx or score_future_to_idx:
                admit_generation()
                pending = set(gen_future_to_idx) | set(score_future_to_idx)
                if not pending:
                    # Nothing in flight but generation still queued should be
                    # unreachable (admit fills lanes whenever the scoring pool is
                    # not saturated, and saturation implies live score futures).
                    # Break defensively rather than spin.
                    break
                completed, _ = wait(
                    pending,
                    timeout=no_progress_timeout_s or None,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    # No generation OR scoring future advanced within the window
                    # (covers a hung scorer just like a hung generation).
                    elapsed = time.time() - batch_start
                    log.error(
                        "%s no eval future (gen or score) completed for %.1fs; failing "
                        "%d in-flight + %d unstarted question(s) closed",
                        label,
                        no_progress_timeout_s,
                        len(pending),
                        len(pending_gen),
                    )
                    timed_out = set(pending)
                    for fut in timed_out:
                        idx = _idx_of(fut)
                        fut.cancel()
                        results[idx] = self._failed_question_result(
                            questions[idx],
                            elapsed_s=elapsed,
                            error=(
                                "eval_no_progress_timeout: no completed future "
                                f"for {no_progress_timeout_s:.1f}s"
                            ),
                        )
                        results[idx].eval_concurrency = workers
                    drained, still_running = wait(timed_out, timeout=drain_timeout_s)
                    if drained:
                        log.info(
                            "%s drained %d eval worker(s) after no-progress timeout",
                            label,
                            len(drained),
                        )
                    if still_running:
                        log.error(
                            "%s %d eval worker(s) still running after %.1fs drain; "
                            "marking batch contaminated by abandoned server-side request(s)",
                            label,
                            len(still_running),
                            drain_timeout_s,
                        )
                        for fut in still_running:
                            idx = _idx_of(fut)
                            if results[idx] is not None:
                                mark_abandoned(
                                    idx,
                                    results[idx],
                                    drain_timeout_s=drain_timeout_s,
                                )
                    for fut in timed_out:
                        idx = _idx_of(fut)
                        if results[idx] is not None:
                            results[idx].eval_concurrency = workers
                        append_question_result(idx, results[idx])
                    # Unstarted questions fail closed (cancelled), matching the
                    # serial path's remaining-question handling.
                    for idx, uq in pending_gen:
                        results[idx] = self._failed_question_result(
                            uq,
                            elapsed_s=time.time() - batch_start,
                            error="eval_cancelled_after_no_progress_timeout",
                        )
                        results[idx].eval_concurrency = workers
                        append_question_result(idx, results[idx])
                    pending_gen.clear()
                    gen_future_to_idx.clear()
                    score_future_to_idx.clear()
                    break

                for fut in completed:
                    if fut in gen_future_to_idx:
                        idx = gen_future_to_idx.pop(fut)
                        try:
                            outcome = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            # _generate_question captures its own exceptions; this
                            # is belt-and-braces for a truly unexpected raise.
                            gen_ended = time.time()
                            finalize_scored(
                                idx,
                                self._failed_question_result(
                                    questions[idx],
                                    elapsed_s=time.time() - batch_start,
                                    error=str(exc),
                                ),
                                generated_at_s=gen_ended,
                            )
                            continue
                        # Generation done → lane is free; hand off to scoring pool.
                        gen_ended_at[idx] = outcome.gen_ended_at_s
                        score_fut = score_ex.submit(
                            self._score_generation, dispatch_questions[idx], outcome, client
                        )
                        score_future_to_idx[score_fut] = idx
                    else:
                        idx = score_future_to_idx.pop(fut)
                        try:
                            result = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            result = self._failed_question_result(
                                questions[idx],
                                elapsed_s=time.time() - batch_start,
                                error=str(exc),
                            )
                        finalize_scored(
                            idx, result, generated_at_s=gen_ended_at.pop(idx, None)
                        )
        finally:
            gen_ex.shutdown(wait=False, cancel_futures=True)
            score_ex.shutdown(wait=False, cancel_futures=True)

        for i, q in enumerate(questions):
            if results[i] is None:
                results[i] = self._failed_question_result(
                    q,
                    elapsed_s=time.time() - batch_start,
                    error="eval_cancelled_after_no_progress_timeout",
                )
                results[i].eval_concurrency = workers
                append_question_result(i, results[i])
        if log_every and done % log_every:
            correct_so_far = sum(1 for r in results if r and r.correct)
            log.info(
                "%s progress: %d/%d (%.0f%% correct, concurrency=%d)",
                label,
                n,
                n,
                100 * correct_so_far / n,
                workers,
            )
            self._emit_progress(
                label=label,
                completed_questions=n,
                total_questions=n,
                correct_questions=correct_so_far,
                concurrency=workers,
            )
        batch_wall_s = time.time() - batch_start
        out = [r for r in results if r is not None]
        for r in out:
            r.eval_concurrency = workers
            r.eval_wall_s = batch_wall_s
        append_complete_marker(len(out), batch_wall_s)
        if writer is not None:
            writer.close()
        return out

    def _emit_progress(
        self,
        *,
        label: str,
        completed_questions: int,
        total_questions: int,
        correct_questions: int,
        concurrency: int,
    ) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(
                {
                    "label": label,
                    "completed_questions": completed_questions,
                    "total_questions": total_questions,
                    "correct_questions": correct_questions,
                    "correct_pct": 100 * correct_questions / max(1, completed_questions),
                    "concurrency": concurrency,
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("eval progress callback failed: %s", exc)

    # ── aggregate results ────────────────────────────────────────

    def _aggregate(self, results: list[QuestionResult], tier: int) -> EvalResult:
        """Aggregate individual question results into an EvalResult."""
        if not results:
            return EvalResult(tier=tier, quality=0, speed=0, cost=0, reliability=0)

        total_count = len(results)
        scored_results = [r for r in results if not r.error]
        n_scored = len(scored_results)

        # Quality: fraction correct over scored (non-error) rows, scaled to 0-3.
        # Infrastructure/scoring failures are reliability evidence, not wrong-answer
        # evidence. This matches verifier/calibration paths and keeps the two
        # denominators explicit in details.
        correct_count = sum(1 for r in scored_results if r.correct)
        quality = (correct_count / n_scored) * 3.0 if n_scored else 0.0

        # Speed: median per-request tokens/sec for non-error results. This is
        # intentionally kept stable for Pareto/backward compatibility. Concurrent
        # eval fan-out has a separate aggregate throughput metric below.
        speeds = []
        for r in results:
            if r.tokens_generated > 0 and r.elapsed_s > 0 and not r.error:
                speeds.append(r.tokens_generated / r.elapsed_s)
        median_request_speed = sorted(speeds)[len(speeds) // 2] if speeds else 0.0
        total_tokens_generated = sum(r.tokens_generated for r in results if not r.error)
        eval_wall_s = max((r.eval_wall_s for r in results), default=0.0)
        sum_request_elapsed_s = sum(r.elapsed_s for r in results if r.elapsed_s > 0)
        aggregate_speed = (
            total_tokens_generated / eval_wall_s
            if total_tokens_generated > 0 and eval_wall_s > 0
            else 0.0
        )
        speed_analytics = _speed_analytics_ge_128(results, eval_wall_s=eval_wall_s)
        host_timing_covariates = _summarize_host_timing_covariates(results)
        task_rate_qph = (total_count / (eval_wall_s / 3600.0)) if eval_wall_s > 0 else 0.0
        scored_task_rate_qph = (
            n_scored / (eval_wall_s / 3600.0) if eval_wall_s > 0 else 0.0
        )
        goodput_qph = (
            correct_count / (eval_wall_s / 3600.0) if eval_wall_s > 0 else 0.0
        )
        tokens_per_solved_task = (
            total_tokens_generated / correct_count if correct_count > 0 else 0.0
        )
        eval_concurrency = max((r.eval_concurrency for r in results), default=1)
        concurrent_eval = eval_concurrency > 1 and aggregate_speed > 0
        speed_metric_mode = "aggregate_batch_tps" if concurrent_eval else "median_request_tps"
        speed = aggregate_speed if concurrent_eval else median_request_speed

        # Cost: average cost tier normalized to 0-1 (tier 4 = 1.0)
        cost_tiers = [r.cost_tier for r in results if r.cost_tier > 0]
        cost = (sum(cost_tiers) / len(cost_tiers) / 4.0) if cost_tiers else 0.5

        # Reliability: fraction of non-error responses
        non_error = n_scored
        reliability = non_error / total_count

        # Per-suite quality
        suite_correct: dict[str, list[bool]] = {}
        suite_total_counts: dict[str, int] = {}
        for r in results:
            suite_total_counts[r.suite] = suite_total_counts.get(r.suite, 0) + 1
        for r in scored_results:
            suite_correct.setdefault(r.suite, []).append(r.correct)
        per_suite = {suite: (sum(vals) / len(vals)) * 3.0 for suite, vals in suite_correct.items()}
        # Per-suite question counts (2026-06-06). The per-suite regression gate is
        # otherwise blind to sample size: on a hybrid eval each suite draws only
        # ~2 questions, so the score is quantized to {0.0, 1.5, 3.0} and a single
        # correct→incorrect flip is a -1.5 swing — 15× the fixed -0.1 gate, tripping
        # it on essentially every trial. Carrying the count lets the gate make the
        # threshold resolution-aware (3/n single-flip quantum) instead of false-
        # positiving the seeder loop into a critic-reject deadlock.
        per_suite_counts = {suite: len(vals) for suite, vals in suite_correct.items()}
        question_results = [_compact_question_result(r) for r in results]
        partition_correct: dict[str, list[bool]] = {}
        partition_suite_correct: dict[str, dict[str, list[bool]]] = {}
        partition_total_counts: dict[str, int] = {}
        for r in results:
            partition = r.eval_partition or "core"
            partition_total_counts[partition] = partition_total_counts.get(partition, 0) + 1
        for r in scored_results:
            partition = r.eval_partition or "core"
            partition_correct.setdefault(partition, []).append(r.correct)
            partition_suite_correct.setdefault(partition, {}).setdefault(r.suite, []).append(
                r.correct
            )
        partition_quality = {
            partition: (sum(vals) / len(vals)) * 3.0
            for partition, vals in partition_correct.items()
        }
        partition_counts = {partition: len(vals) for partition, vals in partition_correct.items()}
        partition_suite_quality = {
            partition: {suite: (sum(vals) / len(vals)) * 3.0 for suite, vals in suites.items()}
            for partition, suites in partition_suite_correct.items()
        }

        # Routing distribution
        route_counts: dict[str, int] = {}
        for r in results:
            route = r.route_used or "unknown"
            # Simplify to tier
            if "architect" in route.lower():
                tier_name = "architect"
            elif "worker" in route.lower():
                tier_name = "worker"
            else:
                tier_name = "frontdoor"
            route_counts[tier_name] = route_counts.get(tier_name, 0) + 1
        total_routed = sum(route_counts.values()) or 1
        routing_dist = {k: v / total_routed for k, v in route_counts.items()}

        # EV-2: Calibration metrics (ECE, AUC, calibration violations)
        confidences = [r.confidence for r in results if not r.error]
        correctness_vals = [float(r.correct) for r in results if not r.error]
        confidence_source_counts: dict[str, int] = {}
        for r in scored_results:
            source = r.confidence_source or "unknown"
            confidence_source_counts[source] = confidence_source_counts.get(source, 0) + 1
        # EV-CONF provenance (computed once, reused in details below): confidence
        # is "real" only when EVERY scored row carried the completion-probability
        # geomean. A binary-correctness proxy or a mixed batch is fail-closed False.
        confidence_is_real = bool(confidence_source_counts) and set(
            confidence_source_counts
        ) <= {"completion_probabilities_geomean"}
        # EV-11c honesty (2026-07-22): when NO confidence data is present (every row
        # errored) the calibration metrics are UNDEFINED — emit None, never a 0.0
        # that masquerades as a real measurement (the EV-11c math re-baseline shipped
        # ECE=0.0/AUROC=0.0 placeholders that read as calibration numbers). This is a
        # REPORTING-honesty change only: ece/auroc are not SafetyGate/Pareto inputs
        # (safetygate-rlvr-provenance-audit F2), so it does NOT touch the ESC-7-reserved
        # question of whether *real* ECE re-enters gating. Decision-facing real-vs-proxy
        # gating happens in the per-role report builders (eval_math_rebaseline /
        # eval_calibration). When confidence rows ARE present the closed-top-bin ECE /
        # roc_auc computation is unchanged (a present-but-degenerate AUROC keeps 0.0).
        ece: float | None = None
        auroc: float | None = None
        cal_violations = 0
        if confidences:
            # EV-11b (operator-decided 2026-07-20): use the canonical closed-top-bin
            # ECE from stat_tests. This is a scoring-semantics change and is
            # era-labeled in details so pre/post EV-11b numbers are not mixed.
            ece = expected_calibration_error(confidences, correctness_vals, n_bins=10) or 0.0
            # Present-but-degenerate AUROC keeps its pinned 0.0 (data present, just
            # not rankable); only a fully-absent batch above leaves it None.
            auroc = 0.0
            # AUC: only meaningful with non-degenerate confidence (>2 distinct values).
            # B1/EV-consolidation (2026-07-17): compute ROC-AUC via the stdlib
            # clean-room roc_auc() in src/llm_primitives/stat_tests.py instead of
            # sklearn. stat_tests.roc_auc is the tie-averaged Mann-Whitney U
            # estimator and is numerically identical to sklearn.metrics
            # .roc_auc_score (verified to 6 d.p. on this path), so this swap is
            # behavior-preserving and removes the sklearn dependency for the
            # calibration metric. roc_auc() returns None only when a class is
            # absent — already excluded by the guard below — and `or 0.0`
            # preserves the prior ImportError/ValueError -> 0.0 convention.
            distinct_conf = len(set(round(c, 6) for c in confidences))
            if distinct_conf > 2 and len(set(correctness_vals)) > 1:
                auroc = roc_auc(confidences, correctness_vals) or 0.0
            cal_violations = sum(
                1 for c, cr in zip(confidences, correctness_vals) if abs(c - cr) > 0.5
            )

        # AP-16: Instruction token budget. This is an active-request estimate,
        # not a prompt-library size proxy.
        instruction_tokens = self._count_instruction_tokens(results)
        avg_prompt_tokens = sum(len(r.prompt) // 4 for r in results) / len(results)
        total_per_request = instruction_tokens + avg_prompt_tokens
        instruction_ratio = instruction_tokens / total_per_request if total_per_request > 0 else 0.0
        if instruction_ratio > 0.20:
            log.warning(
                "AP-16: Instruction token ratio %.1f%% exceeds 20%% threshold "
                "(%d instruction tokens per request)",
                instruction_ratio * 100,
                instruction_tokens,
            )

        # Branching density: average across questions with <think> blocks
        bd_vals = [r.branching_density for r in results if r.branching_density > 0]
        avg_branching = sum(bd_vals) / len(bd_vals) if bd_vals else 0.0

        # Tool-use telemetry (2026-06-01). Per-question tools_used summed/averaged
        # over non-error results so the planner can measure and incentivize tool
        # use. Note: tokens_generated (and hence speed) already includes the tokens
        # the model generated across every tool/ReAct turn — tool use is NOT
        # invisible to the throughput objective, only to this explicit signal.
        tool_counts = [r.tools_used for r in results if not r.error]
        total_tool_calls = sum(tool_counts)
        mean_tools_used = (total_tool_calls / len(tool_counts)) if tool_counts else 0.0
        tool_use_rate = (
            sum(1 for n in tool_counts if n > 0) / len(tool_counts) if tool_counts else 0.0
        )
        tool_name_counts: dict[str, int] = {}
        for r in results:
            for name in r.tools_called or []:
                tool_name_counts[name] = tool_name_counts.get(name, 0) + 1
        # Conditional credit — MARGINAL usefulness of tools, computed PER SUITE then
        # averaged, so a trivially-correct no-tool suite cannot anchor the baseline
        # and flip the sign. The old cross-suite delta did exactly that: easy base
        # suites at ~1.0 made the tool-required suite look harmful (−0.4 at the
        # 2026-06-04 cutover, even though every tool call succeeded). Within a suite
        # we compare like-with-like: P(correct|tool) − P(correct|no tool). The scalar
        # is the mean of per-suite deltas over suites with both arms ≥ _MIN_ARM; NaN
        # when none qualify — an honest "not measurable" beats a contaminated number.
        # Planner PRIOR, never a Pareto objective.
        _MIN_ARM = 3
        with_tools = [r for r in results if not r.error and r.tools_used > 0]
        without_tools = [r for r in results if not r.error and r.tools_used == 0]
        _by_suite: dict[str, list] = {}
        for r in results:
            if not r.error:
                _by_suite.setdefault(r.suite, []).append(r)
        per_suite_tool_helpfulness: dict[str, float] = {}
        for _suite, _rs in _by_suite.items():
            _w = [r for r in _rs if r.tools_used > 0]
            _wo = [r for r in _rs if r.tools_used == 0]
            if len(_w) >= _MIN_ARM and len(_wo) >= _MIN_ARM:
                _p_w = sum(1 for r in _w if r.correct) / len(_w)
                _p_wo = sum(1 for r in _wo if r.correct) / len(_wo)
                per_suite_tool_helpfulness[_suite] = _p_w - _p_wo
        if per_suite_tool_helpfulness:
            tool_helpfulness = sum(per_suite_tool_helpfulness.values()) / len(
                per_suite_tool_helpfulness
            )
        else:
            tool_helpfulness = float("nan")

        rubric_dimension_values: dict[str, list[float]] = {}
        for r in results:
            for dim, value in (r.rubric_scores or {}).items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    rubric_dimension_values.setdefault(dim, []).append(numeric)
        rubric_dimension_means = {
            dim: sum(values) / len(values)
            for dim, values in sorted(rubric_dimension_values.items())
            if values
        }
        rubric_process_means = {
            dim: rubric_dimension_means.get(dim, float("nan")) for dim in MINDDR_PROCESS_DIMENSIONS
        }
        # SCORE-08: judge-vs-heuristic provenance rollup. Only rubric-scored
        # questions carry a rubric_source; a run split half judge / half
        # heuristic fallback is otherwise indistinguishable downstream.
        rubric_source_counts: dict[str, int] = {}
        for r in results:
            if r.rubric_source:
                rubric_source_counts[r.rubric_source] = (
                    rubric_source_counts.get(r.rubric_source, 0) + 1
                )
        orphan_contamination_count = sum(
            1 for r in results if r.error and "eval_orphan_contamination" in r.error
        )

        return EvalResult(
            tier=tier,
            quality=quality,
            speed=speed,
            cost=cost,
            reliability=reliability,
            per_suite_quality=per_suite,
            per_suite_counts=per_suite_counts,
            routing_distribution=routing_dist,
            n_questions=total_count,
            question_results=question_results,
            details={
                "correct": correct_count,
                "total": total_count,
                "n_questions": total_count,
                "n_scored": n_scored,
                "quality_denominator": n_scored,
                "quality_denominator_semantics": "non_error_question_results",
                "scoring_errors": total_count - n_scored,
                "per_suite_counts": per_suite_counts,
                "per_suite_total_counts": suite_total_counts,
                "partition_quality": partition_quality,
                "partition_counts": partition_counts,
                "partition_total_counts": partition_total_counts,
                "partition_suite_quality": partition_suite_quality,
                "errors": sum(1 for r in results if r.error),
                "eval_contaminated_by_abandoned_requests": orphan_contamination_count > 0,
                "eval_orphan_contamination_count": orphan_contamination_count,
                "speed_semantics": "speed is the objective speed used by safety/Pareto; median_request_tps and aggregate_tps retain raw throughput components",
                "speed_metric_mode": speed_metric_mode,
                "objective_speed_tps": speed,
                "median_request_tps": median_request_speed,
                "aggregate_tps": aggregate_speed,
                **speed_analytics,
                "eval_concurrency": eval_concurrency,
                "eval_wall_s": eval_wall_s,
                "sum_request_elapsed_s": sum_request_elapsed_s,
                "tokens_generated": total_tokens_generated,
                "host_timing_covariates": host_timing_covariates,
                "task_rate_qph": task_rate_qph,
                "scored_task_rate_qph": scored_task_rate_qph,
                "goodput_qph": goodput_qph,
                "tokens_per_solved_task": tokens_per_solved_task,
                "tokens_include_tool_turns": True,
                "total_tool_calls": total_tool_calls,
                "mean_tools_used": round(mean_tools_used, 4),
                "tool_use_rate": round(tool_use_rate, 4),
                "tool_name_counts": tool_name_counts,
                "tool_helpfulness": tool_helpfulness,
                "tool_helpfulness_n_with": len(with_tools),
                "tool_helpfulness_n_without": len(without_tools),
                "per_suite_tool_helpfulness": per_suite_tool_helpfulness,
                "rubric_dimension_means": rubric_dimension_means,
                "rubric_n_questions": sum(1 for r in results if r.rubric_scores),
                "rubric_source_counts": dict(sorted(rubric_source_counts.items())),
                "ece_binning": "closed_top_bin_stat_tests",
                "ece_instrument_era": "ev11b_closed_bin_2026_07_20",
                "confidence_source_counts": dict(sorted(confidence_source_counts.items())),
                "confidence_is_real": confidence_is_real,
                # EV-11c provenance: distinguishes "None because no confidence data"
                # (calibration_confidence_present=False) from a real computed value.
                "calibration_confidence_present": bool(confidences),
            },
            mean_tools_used=mean_tools_used,
            tool_use_rate=tool_use_rate,
            total_tool_calls=total_tool_calls,
            tool_helpfulness=tool_helpfulness,
            per_suite_tool_helpfulness=per_suite_tool_helpfulness,
            median_request_speed=median_request_speed,
            aggregate_speed=aggregate_speed,
            eval_concurrency=eval_concurrency,
            eval_wall_s=eval_wall_s,
            sum_request_elapsed_s=sum_request_elapsed_s,
            speed_metric_mode=speed_metric_mode,
            instruction_token_count=instruction_tokens,
            instruction_token_ratio=instruction_ratio,
            partial_count=sum(1 for r in results if r.partial),
            degraded_count=sum(1 for r in results if r.degraded),
            avg_prompt_tokens=avg_prompt_tokens,
            ece=ece,
            auroc=auroc,
            calibration_violations=cal_violations,
            branching_density=avg_branching,
            rubric_reasoning_trajectory=rubric_process_means["reasoning_trajectory"],
            rubric_tool_calls=rubric_process_means["tool_calls"],
            rubric_outline=rubric_process_means["outline"],
            rubric_content_stage=rubric_process_means["content_stage"],
            # 2026-05-23 Phase 4 — roll up exogenous-restart counters.
            n_exogenous_recovered=sum(1 for r in results if r.exogenous_recovered),
            n_exogenous_unrecovered=sum(1 for r in results if r.exogenous_unrecovered),
            n_external_restart=sum(1 for r in results if r.external_restart),
            exogenous_question_ids=[
                r.question_id for r in results if (r.exogenous_recovered or r.exogenous_unrecovered)
            ],
        )

    def _aggregate_decision_partitions(
        self,
        results: list[QuestionResult],
        *,
        tier: int,
        excluded_partitions: set[str],
        exclusion_reasons: Mapping[str, str] | None = None,
    ) -> EvalResult:
        if not excluded_partitions:
            return self._aggregate(results, tier=tier)

        full_result = self._aggregate(results, tier=tier)
        decision_results = [
            r
            for r in results
            if (r.eval_partition or "core") not in excluded_partitions
        ]
        if len(decision_results) == len(results):
            return full_result
        if decision_results:
            result = self._aggregate(decision_results, tier=tier)
        else:
            result = EvalResult(tier=tier, quality=0, speed=0, cost=0, reliability=0)

        result.question_results = full_result.question_results
        for key in (
            "partition_quality",
            "partition_counts",
            "partition_total_counts",
            "partition_suite_quality",
        ):
            result.details[key] = full_result.details.get(key, {})

        excluded_counts: dict[str, int] = {}
        for r in results:
            partition = r.eval_partition or "core"
            if partition in excluded_partitions:
                excluded_counts[partition] = excluded_counts.get(partition, 0) + 1

        pre_filter_speed = result.speed
        pre_filter_speed_mode = result.speed_metric_mode
        if excluded_counts:
            # A filtered subset has no independent batch wall clock. Keep the
            # mixed-batch aggregate TPS as telemetry and use comparable per-request
            # median TPS for the decision objective.
            result.speed = result.median_request_speed
            result.speed_metric_mode = "median_request_tps_partition_filtered"
            result.details["speed_metric_mode"] = result.speed_metric_mode
            result.details["objective_speed_tps"] = result.speed

        result.details.update(
            {
                "decision_partition_filter": {
                    "excluded_partitions": sorted(excluded_counts),
                    "excluded_counts": excluded_counts,
                    "exclusion_reasons": dict(exclusion_reasons or {}),
                    "full_n_questions": full_result.n_questions,
                    "decision_n_questions": result.n_questions,
                    "full_batch_objective_speed_tps": full_result.speed,
                    "pre_filter_objective_speed_tps": pre_filter_speed,
                    "pre_filter_speed_metric_mode": pre_filter_speed_mode,
                    "decision_speed_metric_mode": result.speed_metric_mode,
                    "decision_subset_speed_comparable": True,
                    "decision_subset_speed_semantics": (
                        "partition-filtered decision speed uses median request TPS; "
                        "aggregate TPS over the mixed batch wall clock is retained "
                        "only as full-batch telemetry"
                    ),
                },
                "decision_excluded_partitions": sorted(excluded_counts),
                "decision_subset_speed_comparable": True,
            }
        )
        return result

    def _count_instruction_tokens(
        self,
        results: list[QuestionResult] | None = None,
    ) -> int:
        """AP-16: Estimate per-request instruction tokens from active prompts.

        Earlier AP-16 accounting summed every ``orchestration/prompts/**/*.md``
        file and treated that as request overhead. That inflated frontier rows:
        dormant templates such as debugger and compaction prompts are not loaded
        for every EvalTower request. Follow the default runtime PromptBuilder
        path instead: root scaffold plus the role prompts actually observed in
        the batch.
        """
        try:
            from src.prompt_builders.builder import PromptBuilder
            from src.roles import Role
        except Exception as exc:  # noqa: BLE001
            log.warning("AP-16 prompt accounting unavailable: %s", exc)
            return 0

        builder = PromptBuilder()
        try:
            scaffold = builder.build_root_lm_prompt(
                state="",
                original_prompt="",
                as_structured=True,
            )
            total_chars = len(scaffold.system) + len(scaffold.tools) + len(scaffold.rules)
        except Exception as exc:  # noqa: BLE001
            log.warning("AP-16 root scaffold accounting failed: %s", exc)
            total_chars = 0

        roles: set[Role] = set()
        for result in results or []:
            role = self._instruction_role_from_route(result.route_used, Role)
            if role is not None:
                roles.add(role)

        for role in roles:
            try:
                total_chars += len(builder.get_system_prompt(role))
            except Exception as exc:  # noqa: BLE001
                log.warning("AP-16 role prompt accounting failed for %s: %s", role, exc)

        return total_chars // 4

    @staticmethod
    def _instruction_role_from_route(route_used: str, role_cls: Any) -> Any | None:
        """Map EvalTower route telemetry to a concrete prompt role."""
        route = str(route_used or "").strip()
        if not route:
            return None
        direct = role_cls.from_string(route)
        if direct is not None:
            return direct
        if route == "worker":
            return role_cls.WORKER_GENERAL
        if route == "architect":
            return role_cls.ARCHITECT_GENERAL
        if route == "coder":
            return role_cls.CODER_ESCALATION
        return None

    # ── tiered evaluation ────────────────────────────────────────

    def eval_t0(self) -> EvalResult:
        """Tier 0: 10 sentinel questions, binary pass/fail, ~30s."""
        sentinels = self._load_sentinels()
        if not sentinels:
            error = "no_valid_sentinel_questions"
            log.error("No sentinel questions available for T0")
            return _loader_error_eval_result(
                tier=0,
                source="sentinel_questions",
                error=error,
                core_id="t0_sentinels_v1_n0",
                loader_details=self._sentinel_load_details,
                test_profile={
                    "version": "eval-tower-tier-profile-v1",
                    "tier": 0,
                    "source": "sentinel_questions",
                    "n_questions": 0,
                },
            )

        # T0 is a fast pass/fail GATE only — its telemetry is NOT journaled into
        # the trial record (hybrid/progressive eval journals T1/T2). Tool-use
        # sentinels therefore live in T1 AND T2 (the journaled evals), not here,
        # so get_eval_secret / tool_helpfulness telemetry actually reaches the
        # planner. (Keeping them here too would only double-run the sentinels.)
        batch = []
        for q in sentinels[:10]:
            sentinel_q = dict(q)
            suite = str(sentinel_q.get("suite", "unknown"))
            if not suite.startswith("sentinel_"):
                suite = f"sentinel_{suite}"
            sentinel_q["suite"] = suite
            batch.append(sentinel_q)
        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(batch, client, label="T0")
        for r in results:
            log.info(
                "T0 [%s/%s] %s → %s",
                r.suite,
                r.question_id,
                "PASS" if r.correct else "FAIL",
                r.error or "",
            )

        result = self._aggregate(results, tier=0)
        return _stamp_eval_instrument(
            result,
            questions=batch,
            core_id=f"t0_sentinels_v1_n{len(batch)}",
            test_profile={
                "version": "eval-tower-tier-profile-v1",
                "tier": 0,
                "source": "sentinel_questions",
                "n_questions": len(batch),
            },
        )

    def eval_t1(
        self,
        n: int = 100,
        seed: int = 42,
        trial_id: int | None = None,
    ) -> EvalResult:
        """Tier 1: 100 stratified questions from benchmark pool, ~5min."""
        configured_core_id = os.environ.get("AUTOPILOT_T1_CORE_ID", "").strip()
        configured_core_path = os.environ.get("AUTOPILOT_T1_CORE_PATH", "").strip()
        core_metadata: dict[str, Any] = {}
        core_path = ""
        core_era_guard: dict[str, Any] = {}
        core_selection = "legacy_pool_seed"
        # Empty for a designed core, which carries its own declared composition.
        tier_mix_provenance: dict[str, Any] = {}
        resolved_trial_id = self._resolve_trial_id(trial_id)
        audit_policy: dict[str, Any] = {
            "enabled": os.environ.get("AUTOPILOT_W6_AUDIT_BLOCK") == "1",
            "requested_n": max(0, _env_int("AUTOPILOT_W6_AUDIT_N", 10)),
            "every_n_trials": max(1, _env_int("AUTOPILOT_W6_AUDIT_EVERY_N_TRIALS", 1)),
            "shadow_only": os.environ.get("AUTOPILOT_W6_AUDIT_SHADOW_ONLY", "1") != "0",
        }

        if configured_core_path and not configured_core_id:
            error = "AUTOPILOT_T1_CORE_PATH requires AUTOPILOT_T1_CORE_ID"
            log.error("T1 designed core misconfigured: %s", error)
            return EvalResult(
                tier=1,
                quality=0,
                speed=0,
                cost=0,
                reliability=0,
                details={
                    "core_selection": "designed_core",
                    "core_path": configured_core_path,
                    "core_error": error,
                },
            )

        if configured_core_id:
            core_path = str(self._core_path(configured_core_id))
            core_era_guard = designed_core_activation_guard(configured_core_id)
            if not core_era_guard.get("ok"):
                error = str(core_era_guard.get("reason", "designed core is not era-authorized"))
                log.error("T1 designed core activation blocked: %s", error)
                return EvalResult(
                    tier=1,
                    quality=0,
                    speed=0,
                    cost=0,
                    reliability=0,
                    core_id=configured_core_id,
                    details={
                        "core_id": configured_core_id,
                        "core_selection": "designed_core",
                        "core_path": core_path,
                        "core_error": error,
                        "core_era_guard": core_era_guard,
                    },
                )
            try:
                questions, core_metadata, core_file = self._load_designed_core(configured_core_id)
                core_path = str(core_file)
                core_selection = "designed_core"
            except Exception as exc:  # noqa: BLE001
                log.error("T1 designed core load failed: %s", exc)
                return EvalResult(
                    tier=1,
                    quality=0,
                    speed=0,
                    cost=0,
                    reliability=0,
                    core_id=configured_core_id,
                    details={
                        "core_id": configured_core_id,
                        "core_selection": "designed_core",
                        "core_path": core_path,
                        "core_error": str(exc),
                        "core_era_guard": core_era_guard,
                    },
                )
            core_id = configured_core_id
        else:
            pool = self._load_pool()
            if not pool:
                error = "no_valid_question_pool"
                log.error("No question pool available for T1")
                return _loader_error_eval_result(
                    tier=1,
                    source="question_pool",
                    error=error,
                    core_id=f"legacy_pool_seed_{seed}_n{n}",
                    loader_details=self._pool_load_details,
                    extra_details={
                        "core_selection": core_selection,
                        "requested_n": n,
                    },
                    test_profile={
                        "version": "eval-tower-tier-profile-v1",
                        "tier": 1,
                        "core_id": f"legacy_pool_seed_{seed}_n{n}",
                        "core_selection": core_selection,
                        "seed": int(seed),
                        "requested_n": int(n),
                        "n_questions": 0,
                    },
                )
            rng = random.Random(seed)
            # Declared tier mix (operator, 2026-08-04). The core_id changes with the
            # sampler because this IS a different instrument: same seed, same n, but a
            # different question set and therefore a different achievable quality and a
            # different questions/hour. Keeping `legacy_pool_seed_...` would have let the
            # old baseline and the old frontier silently gate a draw they never measured.
            questions, tier_mix_provenance = _sample_tier_stratified_eval_questions(
                pool, n, rng
            )
            core_selection = "tier_stratified"
            core_id = f"tier_stratified_{EVAL_TIER_MIX_POLICY}_seed_{seed}_n{n}"

        audit_questions: list[dict] = []
        if audit_policy["enabled"] and audit_policy["requested_n"] > 0:
            if resolved_trial_id is None:
                error = "AUTOPILOT_W6_AUDIT_BLOCK=1 requires a trial_id"
                log.error("T1 W6 audit block misconfigured: %s", error)
                return EvalResult(
                    tier=1,
                    quality=0,
                    speed=0,
                    cost=0,
                    reliability=0,
                    core_id=core_id,
                    details={
                        "core_id": core_id,
                        "audit_policy": audit_policy,
                        "audit_error": error,
                    },
                )
            audit_policy["trial_id"] = resolved_trial_id
            if resolved_trial_id % audit_policy["every_n_trials"] == 0:
                try:
                    audit_questions, audit_seed = self._load_audit_block(
                        questions,
                        audit_policy["requested_n"],
                        resolved_trial_id,
                        core_id,
                    )
                    audit_policy.update(
                        {
                            "active": True,
                            "seed": audit_seed,
                            "actual_n": len(audit_questions),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error("T1 W6 audit block load failed: %s", exc)
                    return EvalResult(
                        tier=1,
                        quality=0,
                        speed=0,
                        cost=0,
                        reliability=0,
                        core_id=core_id,
                        details={
                            "core_id": core_id,
                            "audit_policy": audit_policy,
                            "audit_error": str(exc),
                        },
                    )
            else:
                audit_policy.update(
                    {
                        "active": False,
                        "actual_n": 0,
                        "skip_reason": "trial_not_on_audit_cadence",
                    }
                )
        else:
            audit_policy.update({"active": False, "actual_n": 0})

        # Tool-use sentinels join the JOURNALED eval (T1) so get_eval_secret /
        # tool_helpfulness telemetry reaches the trial record + planner. Inert
        # ([]) unless AUTOPILOT_TOOL_SENTINELS=1.
        base_core_questions = len(questions)
        base_audit_questions = len(audit_questions)
        tool_sentinel_questions = self._load_tool_sentinels()
        questions = (
            _annotate_partition(questions, "core")
            + _annotate_partition(audit_questions, "audit")
            + _annotate_partition(tool_sentinel_questions, "tool_sentinel")
        )

        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(questions, client, log_every=10, label="T1")

        excluded_partitions: set[str] = set()
        exclusion_reasons: dict[str, str] = {}
        if audit_policy["active"] and audit_policy["shadow_only"]:
            excluded_partitions.add("audit")
            exclusion_reasons["audit"] = "w6_audit_shadow_only"
        if tool_sentinel_questions:
            excluded_partitions.add("tool_sentinel")
            exclusion_reasons["tool_sentinel"] = "tool_secret_minting_not_decision_grade"
        result = self._aggregate_decision_partitions(
            results,
            tier=1,
            excluded_partitions=excluded_partitions,
            exclusion_reasons=exclusion_reasons,
        )
        if audit_policy["active"] and audit_policy["shadow_only"]:
            result.details.update(
                {
                    "audit_shadow_only": True,
                    "audit_shadow_total_n_questions": len(results),
                    "audit_shadow_decision_n_questions": result.n_questions,
                    "audit_shadow_excluded_partitions": ["audit"],
                }
            )
        if tool_sentinel_questions:
            result.details.update(
                {
                    "tool_sentinel_decision_excluded": True,
                    "tool_sentinel_questions": len(tool_sentinel_questions),
                }
            )
        instrument_questions = [
            q
            for q in questions
            if q.get("eval_partition", "core") not in excluded_partitions
        ]
        if excluded_partitions:
            result.details["full_batch_dataset_content_sha256"] = dataset_content_sha256(
                questions
            )
            result.details["full_batch_n_questions"] = len(questions)
        result.details.update(
            {
                "core_selection": core_selection,
                "core_path": core_path,
                "core_metadata": core_metadata,
                "core_era_guard": core_era_guard,
                "requested_n": n,
                "base_core_questions": base_core_questions,
                "base_audit_questions": base_audit_questions,
                "audit_policy": audit_policy,
                "tier_mix_provenance": tier_mix_provenance,
            }
        )
        return _stamp_eval_instrument(
            result,
            questions=instrument_questions,
            core_id=core_id,
            test_profile={
                "version": "eval-tower-tier-profile-v1",
                "tier": 1,
                "core_id": core_id,
                "core_selection": core_selection,
                "seed": int(seed),
                "requested_n": int(n),
                "n_questions": len(instrument_questions),
                "full_batch_n_questions": len(questions),
                "decision_excluded_partitions": sorted(excluded_partitions),
                "base_core_questions": base_core_questions,
                "base_audit_questions": base_audit_questions,
                "audit_policy": audit_policy,
                # Declared mix + any shortfall, so a reader can tell the instrument's
                # INTENDED composition from the one actually drawn.
                "tier_mix_provenance": tier_mix_provenance,
            },
        )

    def eval_t2(
        self,
        n: int = 500,
        seed: int = 42,
        *,
        promotion_eval: bool = False,
        trial_id: int | None = None,
        exclude_qids: set[str] | None = None,
    ) -> EvalResult:
        """Tier 2: 500+ full benchmark, ~30min."""
        pool = self._load_pool()
        if not pool:
            error = "no_valid_question_pool"
            log.error("No question pool available for T2")
            requested_n = int(n)
            return _loader_error_eval_result(
                tier=2,
                source="question_pool",
                error=error,
                core_id=f"legacy_pool_t2_seed_{seed}_n{requested_n}",
                loader_details=self._pool_load_details,
                extra_details={
                    "requested_n": requested_n,
                    "promotion_eval": bool(promotion_eval),
                },
                test_profile={
                    "version": "eval-tower-tier-profile-v1",
                    "tier": 2,
                    "core_id": f"legacy_pool_t2_seed_{seed}_n{requested_n}",
                    "seed": int(seed),
                    "requested_n": requested_n,
                    "n_questions": 0,
                    "promotion_eval": bool(promotion_eval),
                },
            )

        resolved_trial_id = self._resolve_trial_id(trial_id)
        requested_n = int(n)
        draw_seed = int(seed)
        promotion_policy: dict[str, Any] = {
            "enabled": bool(promotion_eval),
        }
        t1_core_exclusion_policy: dict[str, Any] = {
            "enabled": not bool(promotion_eval),
        }
        t1_core_exclude_qids: set[str] = set()
        effective_exclude_qids: set[str] | None = set(exclude_qids or set())
        if promotion_eval:
            effective_exclude_qids = exclude_qids
            requested_n = _promotion_eval_n()
            if resolved_trial_id is None:
                error = "promotion eval requires a trial_id"
                log.error("T2 promotion eval misconfigured: %s", error)
                return EvalResult(
                    tier=2,
                    quality=0,
                    speed=0,
                    cost=0,
                    reliability=0,
                    details={
                        "promotion_eval_policy": {
                            **promotion_policy,
                            "error": error,
                        },
                    },
                )
            draw_seed = _promotion_eval_seed(resolved_trial_id, requested_n)
            excluded_suites, suite_health = _promotion_excluded_suites_from_health()
            promotion_policy.update(
                {
                    "version": "w8-promotion-eval-v1",
                    "trial_id": resolved_trial_id,
                    "requested_n": requested_n,
                    "min_n": PROMOTION_EVAL_MIN_N,
                    "max_n": PROMOTION_EVAL_MAX_N,
                    "seed": draw_seed,
                    "recent_exclusion_qids": len(exclude_qids or set()),
                    "recency_window_days": 60,
                    "suite_health": suite_health,
                }
            )
        else:
            excluded_suites = set()
            try:
                t1_core_exclude_qids, t1_core_exclusion_policy = self._t1_core_exclusion_qids(
                    pool,
                    seed=draw_seed,
                )
                effective_exclude_qids = set(exclude_qids or set()) | t1_core_exclude_qids
                t1_core_exclusion_policy["caller_excluded_qids"] = len(exclude_qids or set())
                t1_core_exclusion_policy["total_exclude_qids"] = len(effective_exclude_qids)
            except Exception as exc:  # noqa: BLE001
                error = f"T2 T1-core exclusion unavailable: {exc}"
                log.error(error)
                return EvalResult(
                    tier=2,
                    quality=0,
                    speed=0,
                    cost=0,
                    reliability=0,
                    details={
                        "t1_core_exclusion_policy": {
                            **t1_core_exclusion_policy,
                            "error": str(exc),
                        },
                    },
                )

        rng = random.Random(seed)
        if promotion_eval:
            rng = random.Random(draw_seed)
        questions = _sample_scoreable_eval_questions(
            pool,
            requested_n,
            rng,
            exclude_qids=effective_exclude_qids,
            exclude_suites=excluded_suites if promotion_eval else None,
        )
        if not promotion_eval and not questions:
            error = "T2 drew 0 scoreable non-T1-core question(s)"
            log.error("T2 non-promotion eval failed closed: %s", error)
            return EvalResult(
                tier=2,
                quality=0,
                speed=0,
                cost=0,
                reliability=0,
                details={
                    "t1_core_exclusion_policy": {
                        **t1_core_exclusion_policy,
                        "actual_t2_core_n": 0,
                        "error": error,
                    },
                },
            )
        if promotion_eval and len(questions) < PROMOTION_EVAL_MIN_N:
            error = (
                f"promotion eval drew {len(questions)} scoreable fresh question(s); "
                f"requires >= {PROMOTION_EVAL_MIN_N}"
            )
            log.error("T2 promotion eval failed closed: %s", error)
            return EvalResult(
                tier=2,
                quality=0,
                speed=0,
                cost=0,
                reliability=0,
                details={
                    "promotion_eval_policy": {
                        **promotion_policy,
                        "actual_n": len(questions),
                        "error": error,
                    },
                },
            )
        # Tool-use sentinels also join T2 (the journaled deep eval) for the same
        # reason as T1. Inert ([]) unless AUTOPILOT_TOOL_SENTINELS=1.
        tool_sentinel_questions = self._load_tool_sentinels()
        questions = _annotate_partition(questions, "core") + _annotate_partition(
            tool_sentinel_questions, "tool_sentinel"
        )

        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(questions, client, log_every=50, label="T2")

        excluded_partitions: set[str] = set()
        exclusion_reasons: dict[str, str] = {}
        if tool_sentinel_questions:
            excluded_partitions.add("tool_sentinel")
            exclusion_reasons["tool_sentinel"] = "tool_secret_minting_not_decision_grade"
        result = self._aggregate_decision_partitions(
            results,
            tier=2,
            excluded_partitions=excluded_partitions,
            exclusion_reasons=exclusion_reasons,
        )
        if tool_sentinel_questions:
            result.details.update(
                {
                    "tool_sentinel_decision_excluded": True,
                    "tool_sentinel_questions": len(tool_sentinel_questions),
                }
            )
        instrument_questions = [
            q
            for q in questions
            if q.get("eval_partition", "core") not in excluded_partitions
        ]
        if excluded_partitions:
            result.details["full_batch_dataset_content_sha256"] = dataset_content_sha256(
                questions
            )
            result.details["full_batch_n_questions"] = len(questions)
        core_id = f"legacy_pool_t2_seed_{draw_seed}_n{requested_n}"
        if promotion_eval:
            promotion_policy["actual_n"] = len(
                [q for q in questions if q.get("eval_partition") == "core"]
            )
            result.details["promotion_eval_policy"] = promotion_policy
            core_id = f"w8_promotion_eval_v1_trial_{resolved_trial_id}_n{requested_n}"
        else:
            t1_core_exclusion_policy["actual_t2_core_n"] = len(
                [q for q in questions if q.get("eval_partition") == "core"]
            )
            result.details["t1_core_exclusion_policy"] = t1_core_exclusion_policy
        return _stamp_eval_instrument(
            result,
            questions=instrument_questions,
            core_id=core_id,
            test_profile={
                "version": "eval-tower-tier-profile-v1",
                "tier": 2,
                "core_id": core_id,
                "seed": int(draw_seed),
                "requested_n": int(requested_n),
                "n_questions": len(instrument_questions),
                "full_batch_n_questions": len(questions),
                "decision_excluded_partitions": sorted(excluded_partitions),
                "promotion_eval": bool(promotion_eval),
                "promotion_policy": promotion_policy if promotion_eval else None,
                "t1_core_exclusion_policy": t1_core_exclusion_policy
                if not promotion_eval
                else None,
            },
        )

    def eval_t3(
        self,
        n: int = EVAL_T3_SPEC_N,
        seed: int = EVAL_SPEC_SEED,
        *,
        exclude_qids: set[str] | None = None,
    ) -> EvalResult:
        """Tier 3: expert/hard workflow eval from pool rows explicitly labeled tier=3."""
        pool = self._load_pool()
        if not pool:
            log.error("No question pool available for T3")
            return EvalResult(tier=3, quality=0, speed=0, cost=0, reliability=0)

        requested_n = int(n)
        draw_seed = int(seed)
        questions = _sample_scoreable_eval_questions_for_pool_tier(
            pool,
            3,
            requested_n,
            random.Random(draw_seed),
            exclude_qids=exclude_qids,
        )
        if not questions:
            error = "question pool has no scoreable tier=3 hard eval items"
            log.error("T3 eval failed closed: %s", error)
            return EvalResult(
                tier=3,
                quality=0,
                speed=0,
                cost=0,
                reliability=0,
                details={
                    "t3_policy": {
                        "version": "t3-hard-only-v1",
                        "requested_n": requested_n,
                        "actual_n": 0,
                        "seed": draw_seed,
                        "pool_tier": 3,
                        "error": error,
                    },
                },
            )

        questions = _annotate_partition(questions, "core")

        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(questions, client, log_every=50, label="T3")

        result = self._aggregate(results, tier=3)
        result.details["t3_policy"] = {
            "version": "t3-hard-only-v1",
            "requested_n": requested_n,
            "actual_n": len(questions),
            "seed": draw_seed,
            "pool_tier": 3,
        }
        # Keep the original core_id for evidence continuity; only the human-facing
        # label changed from "hard-only" to "expert/hard workflow".
        core_id = f"t3_hard_only_v1_seed_{draw_seed}_n{requested_n}"
        return _stamp_eval_instrument(
            result,
            questions=questions,
            core_id=core_id,
            test_profile={
                "version": "eval-tower-tier-profile-v1",
                "tier": 3,
                "core_id": core_id,
                "seed": int(draw_seed),
                "requested_n": int(requested_n),
                "n_questions": len(questions),
                "pool_tier": 3,
            },
        )

    # ── verifier-mode entrypoints (EV-4 / EV-11, additive 2026-07-17) ────────
    # BUILD-evalbatch-verifier-mode. These are new ENTRYPOINTS, not new
    # generation engines: they draw a suite, then reuse the existing _eval_batch
    # dispatch + _eval_question scoring per role, and roll the per-question
    # (confidence, correctness) vectors into the pure metric helpers above.
    # Surfaced on eval_batch_serving_evaltower_window.py under the same
    # --confirm-clean-window execution gate as the tier path.

    @staticmethod
    def _normalize_roles(roles: "list[str] | str | None") -> list[str]:
        """Coerce a roles selector into a deduped ordered list; warn on unknowns.

        Defaults to the live production worker when unset. Unknown role strings
        are kept (the orchestrator is the authority on force_role) but logged.
        """
        if isinstance(roles, str):
            roles = [part.strip() for part in roles.split(",")]
        items = [str(r).strip() for r in (roles or []) if str(r).strip()]
        deduped: list[str] = []
        for role in items:
            if role not in deduped:
                deduped.append(role)
        if not deduped:
            deduped = ["worker_general"]
        try:
            from src.roles import Role

            for role in deduped:
                if Role.from_string(role) is None:
                    log.warning("verifier-mode: role %r is not a known Role", role)
        except Exception as exc:  # noqa: BLE001
            log.debug("verifier-mode role validation unavailable: %s", exc)
        return deduped

    @staticmethod
    def _with_forced_role(q: dict[str, Any], role: str) -> dict[str, Any]:
        forced = dict(q)
        forced["force_role"] = str(role)
        return forced

    def _load_dataset_adapter(self, suite: str):
        """Return a research-repo dataset adapter INSTANCE for a named suite.

        Loads the canonical research registry by file path so a stale bare
        ``dataset_adapters`` module cannot win through ``sys.modules`` order.
        """
        dataset_adapters = _load_research_benchmark_module("dataset_adapters")
        get_adapter = getattr(dataset_adapters, "get_adapter")

        adapter = get_adapter(suite)
        if adapter is None:
            raise ValueError(f"no dataset adapter registered for suite {suite!r}")
        return adapter

    @staticmethod
    def _filter_questions_by_split(
        questions: list[dict[str, Any]],
        split: str | None,
    ) -> list[dict[str, Any]]:
        """Filter adapter questions to a named split/subset.

        For scoring_verifiers the subset is carried in ``metadata.subset`` and the
        id prefix (``sv_<subset>_...``); for math it is the id prefix
        (``gsm8k_...`` / ``math500_...``). ``None`` / ``all`` / ``*`` keep everything.
        """
        split_l = str(split or "").strip().lower()
        if not split_l or split_l in ("all", "*"):
            return list(questions)
        needle = EvalTower._normalize_split_label(split_l)

        def _matches(q: dict[str, Any]) -> bool:
            meta = q.get("metadata") or {}
            subset = EvalTower._normalize_split_label(str(meta.get("subset", "")))
            qid = EvalTower._normalize_split_label(str(q.get("id", "")))
            if subset:
                return needle == subset
            qid_split = qid
            if qid.startswith("sv_"):
                tail = qid[3:]
                parts = tail.rsplit("_", 1)
                qid_split = parts[0] if len(parts) == 2 and parts[1].isdigit() else tail
            else:
                parts = qid.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    qid_split = parts[0]
            return qid_split == needle

        return [q for q in questions if _matches(q)]

    @staticmethod
    def _normalize_split_label(value: str) -> str:
        normalized = str(value or "").strip().lower().replace("+", "_plus")
        normalized = _re.sub(r"[^a-z0-9]+", "_", normalized)
        return _re.sub(r"_+", "_", normalized).strip("_")

    def _load_verifier_suite_questions(
        self,
        suite: str,
        split: str | None,
        *,
        n: int | None,
        seed: int,
        full: bool,
    ) -> list[dict[str, Any]]:
        adapter = self._load_dataset_adapter(suite)
        questions = adapter.extract_all()
        questions = self._filter_questions_by_split(questions, split)
        if not full and n is not None and len(questions) > int(n):
            rng = random.Random(int(seed))
            questions = rng.sample(questions, int(n))
        for q in questions:
            q.setdefault("suite", suite)
        return questions

    def eval_calibration(
        self,
        suite: str,
        split: str | None = None,
        roles: "list[str] | str | None" = None,
        seed: int = EVAL_SPEC_SEED,
        *,
        n: int | None = None,
        full: bool = False,
    ) -> dict[str, Any]:
        """EV-4 calibration baseline: per-role calibration metrics over a suite/split.

        Runs the suite over the split for each role (reusing ``_eval_batch``), then
        computes the six per-role calibration metrics (ECE, AUROC, Top-1, Bottom-1,
        Spearman rho, MAE) via ``compute_calibration_metrics``. Returns a plain,
        JSON-serializable dict — the window runner embeds it in its report.

        Execution here calls the orchestrator (generation); the caller (window
        runner) gates that behind ``--confirm-clean-window``.
        """
        roles = self._normalize_roles(roles)
        questions = self._load_verifier_suite_questions(
            suite, split, n=n, seed=int(seed), full=full
        )
        if not questions:
            raise ValueError(
                f"calibration suite {suite!r} split {split!r} yielded 0 questions"
            )
        dataset_sha256 = dataset_content_sha256(questions)
        per_role: dict[str, Any] = {}
        with httpx.Client(timeout=self.timeout) as client:
            for role in roles:
                role_qs = [self._with_forced_role(q, role) for q in questions]
                results = self._eval_batch(role_qs, client, log_every=25, label=f"cal-{role}")
                scored = [r for r in results if not r.error]
                confidences = [r.confidence for r in scored]
                labels = [float(r.correct) for r in scored]
                metrics = compute_calibration_metrics(confidences, labels)
                correct = sum(1 for r in scored if r.correct)
                # EV-11c: the six EV-4 calibration metrics are meaningful ONLY over
                # real (completion-probability geomean) confidence. On a binary/mixed
                # batch, ECE collapses to a fake ~0.0; emit None + provenance rather
                # than a placeholder. confidence_is_real is fail-closed (whole-batch).
                cal_source_counts: dict[str, int] = {}
                for r in scored:
                    src = r.confidence_source or "unknown"
                    cal_source_counts[src] = cal_source_counts.get(src, 0) + 1
                cal_real = bool(cal_source_counts) and set(
                    cal_source_counts
                ) <= {"completion_probabilities_geomean"}
                reliability = (len(scored) / len(results)) if results else 0.0
                per_role[role] = {
                    "role": role,
                    "n_questions": len(results),
                    "n_scored": len(scored),
                    "reliability": reliability,
                    "accuracy": (correct / len(scored)) if scored else None,
                    **{
                        key: (metrics.get(key) if cal_real else None)
                        for key in CALIBRATION_METRIC_KEYS
                    },
                    "confidence_is_real": cal_real,
                    "confidence_source_counts": dict(cal_source_counts),
                }
        return {
            "mode": "calibration",
            "suite": suite,
            "split": split,
            "seed": int(seed),
            "roles": roles,
            "n_questions": len(questions),
            "dataset_sha256": dataset_sha256,
            "metric_keys": list(CALIBRATION_METRIC_KEYS),
            "per_role": per_role,
        }

    def eval_math_rebaseline(
        self,
        full: bool = True,
        scoring: str = "math_verify",
        roles: "list[str] | str | None" = None,
        seed: int = EVAL_SPEC_SEED,
        production_sampling: bool = True,
        *,
        n: int | None = None,
    ) -> dict[str, Any]:
        """EV-11 math re-baseline: GSM8K(1,319)+MATH-500(500)=1,819/arm under math_verify.

        Per role, runs the math suite (full=1,819 or a subsample of ``n``) through
        the existing dispatch/scoring with ``scoring_method`` forced to ``scoring``,
        recording ``dataset_sha256`` and a ``test_profile``/arm stamp per role.
        Hard-fails up front when ``scoring == "math_verify"`` and math-verify is not
        importable (reuses the landed ``_require_math_verify`` guard). Returns a
        plain JSON-serializable dict.
        """
        if str(scoring) == "math_verify":
            _require_math_verify()
        roles = self._normalize_roles(roles)
        adapter = self._load_dataset_adapter("math")
        if full or n is None:
            questions = adapter.extract_all()
        else:
            questions = adapter.sample(n=int(n), seed=int(seed))
        for q in questions:
            q.setdefault("suite", "math")
            if scoring:
                q["scoring_method"] = str(scoring)
        if not questions:
            raise ValueError(
                "math re-baseline drew 0 questions — GSM8K/MATH-500 datasets "
                "unavailable (MODEL-DOWNLOAD/data required)"
            )
        dataset_sha256 = dataset_content_sha256(questions)
        n_gsm8k = sum(1 for q in questions if str(q.get("id", "")).startswith("gsm8k"))
        n_math500 = sum(1 for q in questions if str(q.get("id", "")).startswith("math500"))
        if n_math500 <= 0:
            raise ValueError(
                "math re-baseline has n_math500=0; MATH-500 coverage is required "
                "before EV-11 can be decision-grade; repair adapter/pool source "
                "accounting and rebuild under the operator era label"
            )
        test_profile = {
            "version": "ev11-math-rebaseline-v1",
            "scoring": str(scoring),
            "seed": int(seed),
            "production_sampling": bool(production_sampling),
            # Per feedback_production_sampling_seed_not_temp0: sampling-sensitive
            # (math_verify/spec-dec) arms record production temp + seed42 intent.
            "sampling_profile": "production_temp_seed42"
            if production_sampling
            else "greedy_temp0",
            "n_questions": len(questions),
            "n_gsm8k": n_gsm8k,
            "n_math500": n_math500,
            "dataset_sha256": dataset_sha256,
        }
        per_role: dict[str, Any] = {}
        # Provenance stamp shared by every arm of this re-baseline: all arms drew
        # the identical question set (dataset_sha256) under the identical
        # test_profile, so require_matched_comparison inside screen_paired_arms
        # will pair them. A canonical JSON string pins the profile identity.
        profile_stamp = {
            "dataset_sha256": dataset_sha256,
            "test_profile": json.dumps(test_profile, sort_keys=True, default=str),
        }
        arm_screens: list[dict[str, Any]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for role in roles:
                role_qs = [self._with_forced_role(q, role) for q in questions]
                results = self._eval_batch(role_qs, client, log_every=100, label=f"ev11-{role}")
                agg = self._aggregate(results, tier=2)
                scored = [r for r in results if not r.error]
                correct = sum(1 for r in scored if r.correct)
                arm_label = f"ev11-math-rebaseline::{role}::seed{int(seed)}::{dataset_sha256[:12]}"
                # EV-11c: ECE/AUROC are decision-grade ONLY when every scored row
                # carried real (completion-probability geomean) confidence. The math
                # dispatch requests n_probs via _eval_question, but if the backend
                # returns no completion_probabilities the batch falls back to the
                # binary-correctness proxy — which makes ECE trivially ~0 and AUROC
                # degenerate. On a binary/mixed batch we emit None + provenance rather
                # than a 0.0 placeholder that masqueraded as a measurement (EV-11c).
                cal_real = bool(agg.details.get("confidence_is_real"))
                per_role[role] = {
                    "role": role,
                    "arm": arm_label,
                    "n_questions": len(results),
                    "n_scored": len(scored),
                    "correct": correct,
                    "accuracy": (correct / len(scored)) if scored else None,
                    "quality": agg.quality,
                    "reliability": agg.reliability,
                    "ece": agg.ece if cal_real else None,
                    "auroc": agg.auroc if cal_real else None,
                    "confidence_is_real": cal_real,
                    "confidence_source_counts": dict(
                        agg.details.get("confidence_source_counts") or {}
                    ),
                    "test_profile": test_profile,
                }
                # Per-question correctness vector for the paired screen. Keyed by
                # the stable qid so the same question aligns across arms (only the
                # forced role differs). Errored questions are dropped, matching the
                # accuracy denominator above.
                arm_screens.append(
                    {
                        "label": arm_label,
                        "profile": profile_stamp,
                        "outcomes": {r.qid: r.correct for r in scored if r.qid},
                    }
                )
        # Paired-significance screen over the arms: exact/normal McNemar p on the
        # flip pairs + per-arm Wilson CIs, gated on matched dataset+profile. Each
        # matched pair now carries a promoted ``verdict`` block. Empty ``pairs``
        # when fewer than two arms were scored.
        paired_significance = screen_paired_arms(arm_screens)
        # Thread the paired verdict into each per-role record so a downstream
        # consumer reading a single role sees a VERDICT, not raw discordant counts.
        attach_role_paired_verdicts(per_role, paired_significance)
        return {
            "mode": "math_rebaseline",
            "suite": "math",
            "scoring": str(scoring),
            "full": bool(full),
            "seed": int(seed),
            "roles": roles,
            "production_sampling": bool(production_sampling),
            "dataset_sha256": dataset_sha256,
            "test_profile": test_profile,
            "per_role": per_role,
            "paired_significance": paired_significance,
        }

    def eval_question_subset(
        self,
        *,
        suite: str,
        question_ids: "Sequence[str]",
        roles: "list[str] | str | None",
        scoring: str | None = None,
        seed: int = EVAL_SPEC_SEED,
        production_sampling: bool = True,
    ) -> dict[str, Any]:
        """Rerun a SPECIFIC subset of a suite's questions (by dataset id or stable
        qid) per role — the targeted-requeue primitive behind
        ``--retry-errors-from``.

        Unlike ``eval_math_rebaseline`` this imposes NO full-coverage guard: the
        error set to requeue may legitimately be, e.g., all-GSM8K with zero
        MATH-500, so the n_math500>0 guard must NOT apply. Matches question dicts
        against ``id`` / ``qid`` / ``question_id`` so it works regardless of which
        identity the prior per-arm file recorded. Returns a per_role dict with the
        same provenance fields as ``eval_math_rebaseline`` (ece/auroc gated on
        ``confidence_is_real``; None on a binary/mixed batch) PLUS a compact
        per-question ``rows`` list the merge step reconciles against the prior run.
        """
        roles = self._normalize_roles(roles)
        want = {str(x).strip() for x in question_ids if str(x).strip()}
        if not want:
            raise ValueError("eval_question_subset requires a non-empty question_ids set")
        adapter = self._load_dataset_adapter(suite)
        all_questions = adapter.extract_all()

        def _identity(q: dict[str, Any]) -> set[str]:
            return {
                str(q.get("id", "")).strip(),
                str(q.get("qid", "")).strip(),
                str(q.get("stable_qid", "")).strip(),
                str(q.get("question_id", "")).strip(),
            } - {""}

        questions = [q for q in all_questions if _identity(q) & want]
        for q in questions:
            q.setdefault("suite", suite)
            if scoring:
                q["scoring_method"] = str(scoring)
        if not questions:
            raise ValueError(
                "eval_question_subset matched 0 questions for the requested ids "
                f"(suite={suite!r}, requested={len(want)}) — dataset/id drift or the "
                "prior run used a different suite"
            )
        dataset_sha256 = dataset_content_sha256(questions)
        per_role: dict[str, Any] = {}
        with httpx.Client(timeout=self.timeout) as client:
            for role in roles:
                role_qs = [self._with_forced_role(q, role) for q in questions]
                results = self._eval_batch(
                    role_qs, client, log_every=100, label=f"retry-{role}"
                )
                agg = self._aggregate(results, tier=2)
                scored = [r for r in results if not r.error]
                correct = sum(1 for r in scored if r.correct)
                cal_real = bool(agg.details.get("confidence_is_real"))
                per_role[role] = {
                    "role": role,
                    "n_questions": len(results),
                    "n_scored": len(scored),
                    "correct": correct,
                    "accuracy": (correct / len(scored)) if scored else None,
                    "reliability": agg.reliability,
                    "ece": agg.ece if cal_real else None,
                    "auroc": agg.auroc if cal_real else None,
                    "confidence_is_real": cal_real,
                    "confidence_source_counts": dict(
                        agg.details.get("confidence_source_counts") or {}
                    ),
                    "rows": [_compact_question_result(r) for r in results],
                }
        return {
            "mode": "question_subset",
            "suite": suite,
            "scoring": str(scoring) if scoring else None,
            "seed": int(seed),
            "roles": roles,
            "production_sampling": bool(production_sampling),
            "n_questions": len(questions),
            "requested_ids": sorted(want),
            "dataset_sha256": dataset_sha256,
            "per_role": per_role,
        }

    def eval_resume_incomplete(
        self,
        *,
        suite: str,
        split: str | None,
        roles: "list[str] | str | None",
        seed: int = EVAL_SPEC_SEED,
        scoring: str | None = None,
        n: int | None = None,
        full: bool = False,
        completed_ids: "Sequence[str]" = (),
        expected_dataset_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Run ONLY the not-yet-completed remainder of a prior suite/split draw —
        the primitive behind the window runner's ``--resume-incomplete-from``.

        Reconstructs the SAME ordered question set the prior run drew
        (``_load_verifier_suite_questions`` with the identical suite/split/seed/n/
        full), then REFUSES (raises ``ValueError``) if its ``dataset_content_sha256``
        does not match ``expected_dataset_sha256`` — dataset drift must not be
        silently resumed. Drops every question whose identity is in ``completed_ids``
        (ANY prior verdict, including error rows — reruning those is
        ``--retry-errors-from``'s job), stamps each survivor with its ORIGINAL
        ordinal (``_ordinal``) so the resumed sidecar rows carry original-dataset
        ordinals that merge cleanly with the prior run, and runs the remainder per
        role via ``_eval_batch``.
        """
        roles = self._normalize_roles(roles)
        questions = self._load_verifier_suite_questions(
            suite, split, n=n, seed=int(seed), full=full
        )
        if not questions:
            raise ValueError(
                f"resume suite {suite!r} split {split!r} yielded 0 questions"
            )
        dataset_sha256 = dataset_content_sha256(questions)
        if expected_dataset_sha256 and dataset_sha256 != str(expected_dataset_sha256):
            raise ValueError(
                "resume dataset mismatch: reconstructed dataset_sha256="
                f"{dataset_sha256} != prior {expected_dataset_sha256} "
                f"(suite={suite!r} split={split!r} seed={seed} full={full} n={n}) — "
                "refusing to resume against a drifted dataset"
            )
        completed = {str(x).strip() for x in completed_ids if str(x).strip()}

        def _identity(q: dict[str, Any]) -> set[str]:
            return {
                str(q.get("id", "")).strip(),
                str(q.get("qid", "")).strip(),
                str(q.get("stable_qid", "")).strip(),
                str(q.get("question_id", "")).strip(),
            } - {""}

        remainder = [(i, q) for i, q in enumerate(questions) if not (_identity(q) & completed)]
        if scoring:
            for _i, q in remainder:
                q["scoring_method"] = str(scoring)

        per_role: dict[str, Any] = {}
        with httpx.Client(timeout=self.timeout) as client:
            for role in roles:
                role_qs = [
                    {**self._with_forced_role(q, role), "_ordinal": int(i)}
                    for i, q in remainder
                ]
                results = (
                    self._eval_batch(role_qs, client, log_every=100, label=f"resume-{role}")
                    if role_qs
                    else []
                )
                scored = [r for r in results if not r.error]
                correct = sum(1 for r in scored if r.correct)
                agg = self._aggregate(results, tier=2) if results else None
                cal_real = bool(agg.details.get("confidence_is_real")) if agg else False
                per_role[role] = {
                    "role": role,
                    "n_questions": len(results),
                    "n_scored": len(scored),
                    "correct": correct,
                    "accuracy": (correct / len(scored)) if scored else None,
                    "reliability": agg.reliability if agg else 0.0,
                    "ece": (agg.ece if cal_real else None) if agg else None,
                    "auroc": (agg.auroc if cal_real else None) if agg else None,
                    "confidence_is_real": cal_real,
                    "confidence_source_counts": (
                        dict(agg.details.get("confidence_source_counts") or {}) if agg else {}
                    ),
                    "rows": [_compact_question_result(r) for r in results],
                }
        return {
            "mode": "resume_incomplete",
            "suite": suite,
            "split": split,
            "scoring": str(scoring) if scoring else None,
            "seed": int(seed),
            "full": bool(full),
            "roles": roles,
            "n_total": len(questions),
            "resume_completed_n": len(questions) - len(remainder),
            "resumed_n": len(remainder),
            "resumed_ordinals": [i for i, _q in remainder],
            "dataset_sha256": dataset_sha256,
            "per_role": per_role,
        }

    def evaluate(
        self,
        tier: int = 0,
        n: int | None = None,
        seed: int = 42,
        trial_id: int | None = None,
        promotion_eval: bool = False,
        exclude_qids: set[str] | None = None,
    ) -> EvalResult:
        """Run the production eval spec for a tier.

        ``n`` and ``seed`` are accepted for backward-compatible callers but are
        intentionally ignored here: planner-selected deep_eval actions must not
        choose their own question quantum. Calibration utilities that need an
        explicit sample size call ``eval_t1``/``eval_t2`` directly.
        """
        if tier == 0:
            return self.eval_t0()
        elif tier == 1:
            if trial_id is None:
                return self.eval_t1(n=EVAL_T1_SPEC_N, seed=EVAL_SPEC_SEED)
            return self.eval_t1(
                n=EVAL_T1_SPEC_N,
                seed=EVAL_SPEC_SEED,
                trial_id=trial_id,
            )
        elif tier == 2:
            if promotion_eval or trial_id is not None or exclude_qids:
                return self.eval_t2(
                    n=EVAL_T2_SPEC_N,
                    seed=EVAL_SPEC_SEED,
                    promotion_eval=promotion_eval,
                    trial_id=trial_id,
                    exclude_qids=exclude_qids,
                )
            return self.eval_t2(n=EVAL_T2_SPEC_N, seed=EVAL_SPEC_SEED)
        elif tier == 3:
            return self.eval_t3(
                n=EVAL_T3_SPEC_N,
                seed=EVAL_SPEC_SEED,
                exclude_qids=exclude_qids,
            )
        else:
            raise ValueError(f"Unknown eval tier: {tier}")

    # ── trace capture ──────────────────────────────────────────

    TAP_PATH = Path("/mnt/raid0/llm/tmp/inference_tap.log")

    def capture_recent_traces(self, n_lines: int = 50) -> str:
        """Read the last n_lines from inference_tap.log for PromptForge feedback.

        Returns raw trace text (ROLE/PROMPT/RESPONSE sections) that shows
        how the orchestrator actually handled recent requests.  Empty string
        if the tap file doesn't exist or is unreadable.
        """
        return eval_tower_trace_feedback.capture_recent_traces(
            self.TAP_PATH,
            n_lines,
            logger=log,
        )

    @staticmethod
    def _trim_trace_text(trace_text: Any, max_chars: int) -> str:
        return eval_tower_trace_feedback.trim_trace_text(trace_text, max_chars)

    @staticmethod
    def _trace_ir_steps(
        trace_text: str, *, max_steps: int = 12, preview_chars: int = 240
    ) -> list[dict[str, Any]]:
        return eval_tower_trace_feedback.trace_ir_steps(
            trace_text,
            max_steps=max_steps,
            preview_chars=preview_chars,
        )

    @classmethod
    def build_critic_trace_ir(
        cls,
        *,
        trace_bank: list[dict[str, Any]] | None = None,
        raw_trace_text: str = "",
        trial_id: int | None = None,
        failure_summary: str = "",
        k_success: int = 2,
        k_failure: int = 2,
        max_trace_chars: int = 1600,
    ) -> dict[str, Any]:
        return eval_tower_trace_feedback.build_critic_trace_ir(
            trace_bank=trace_bank,
            raw_trace_text=raw_trace_text,
            trial_id=trial_id,
            failure_summary=failure_summary,
            k_success=k_success,
            k_failure=k_failure,
            max_trace_chars=max_trace_chars,
        )

    @staticmethod
    def format_critic_trace_ir(trace_ir: dict[str, Any] | None) -> str:
        return eval_tower_trace_feedback.format_critic_trace_ir(trace_ir)

    @classmethod
    def update_contrastive_trace_bank(
        cls,
        trace_bank: list[dict[str, Any]] | None,
        *,
        trace_text: str,
        outcome: str,
        trial_id: int | None = None,
        species: str = "",
        action_type: str = "",
        reason: str = "",
        max_examples_per_outcome: int = 8,
        max_trace_chars: int = 1600,
    ) -> list[dict[str, Any]]:
        return eval_tower_trace_feedback.update_contrastive_trace_bank(
            trace_bank,
            trace_text=trace_text,
            outcome=outcome,
            trial_id=trial_id,
            species=species,
            action_type=action_type,
            reason=reason,
            max_examples_per_outcome=max_examples_per_outcome,
            max_trace_chars=max_trace_chars,
        )

    def capture_contrastive_traces(
        self,
        *,
        k_success: int = 2,
        k_failure: int = 2,
        trace_bank: list[dict[str, Any]] | None = None,
    ) -> str:
        return eval_tower_trace_feedback.format_contrastive_traces(
            k_success=k_success,
            k_failure=k_failure,
            trace_bank=trace_bank,
        )

    def hybrid_eval(
        self,
        seed: int = 42,
        t1_n: int = 50,
        trial_id: int | None = None,
    ) -> EvalResult:
        """Hybrid evaluation: T1 as the real gate; legacy T0 prefilter is opt-in.

        Fable5 instrument review found the T0 sentinel slice can hide harder
        sentinels behind ``[:10]`` and produce bad fast rejects. Default to the
        journaled T1 instrument; operators can temporarily restore the old
        prefilter with AUTOPILOT_HYBRID_T0_GATE=1.
        """
        if os.environ.get("AUTOPILOT_HYBRID_T0_GATE") != "1":
            log.info("Hybrid eval: T0 gate disabled, running T1 (%d questions)...", t1_n)
            if trial_id is None:
                t1 = self.eval_t1(n=t1_n, seed=seed)
            else:
                t1 = self.eval_t1(n=t1_n, seed=seed, trial_id=trial_id)
            log.info("Hybrid eval: T1 result q=%.3f r=%.2f", t1.quality, t1.reliability)
            return t1

        t0 = self.eval_t0()
        if t0.quality < 2.5:
            log.info("Hybrid eval: T0 failed (q=%.3f), fast-reject", t0.quality)
            return t0

        log.info("Hybrid eval: T0 passed (q=%.3f), running T1 (%d questions)...", t0.quality, t1_n)
        if trial_id is None:
            t1 = self.eval_t1(n=t1_n, seed=seed)
        else:
            t1 = self.eval_t1(n=t1_n, seed=seed, trial_id=trial_id)
        log.info("Hybrid eval: T1 result q=%.3f r=%.2f", t1.quality, t1.reliability)
        return t1

    def progressive_eval(self, seed: int = 42) -> tuple[EvalResult, int]:
        """Progressive evaluation: T0 → T1 if passed → T2 if Pareto candidate.

        Returns (result, max_tier_reached).
        """
        t0 = self.eval_t0()
        if t0.quality < 1.5:  # T0 binary gate
            log.warning("T0 failed (quality=%.3f), skipping T1/T2", t0.quality)
            return t0, 0

        t1 = self.eval_t1(seed=seed)
        return t1, 1
