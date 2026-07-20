"""Tiered evaluation tower: T0 (10q/30s) → T1 (100q/5m) → T2 (500+/30m) → T3 (expert/hard workflow).

Wraps existing seeding infrastructure for orchestrator API calls and scoring.
Training set (debug suites) is kept separate from validation set (HF benchmarks).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

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


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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
    if r.error:
        item["error"] = True
        item["error_detail"] = str(r.error).replace("\n", " ")[:200]
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

    try:
        from scripts.server.runtime_facts_manifest import read_runtime_stack_numa_mode

        stack_numa_mode = read_runtime_stack_numa_mode()
        if not stack_numa_mode:
            try:
                stack_numa_mode = read_runtime_stack_numa_mode(state_file=None)
            except TypeError:
                pass
    except Exception:
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


def _eval_batch_id(*, label: str, n_questions: int, started_at_s: float) -> str:
    safe_label = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(label or "eval").strip()
    ).strip("-_")
    if not safe_label:
        safe_label = "eval"
    return f"evaltower-{safe_label}-{int(started_at_s * 1000)}-{max(0, n_questions)}q"


# Import seeding infrastructure
import sys

_orch_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_orch_root / "scripts" / "benchmark"))
# Repo root on path so `src.tools.eval_secret` (runtime tool-secret ground truth)
# imports from the autopilot harness, not just inside the orchestrator process.
sys.path.insert(0, str(_orch_root))

from seeding_orchestrator import call_orchestrator_forced  # noqa: E402
from seeding_scoring import score_answer_deterministic  # noqa: E402
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
    DEFAULT_WILSON_Z,
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
    PairedComparisonMismatchError,
    QuestionOutcome,
    mcnemar_from_vectors,
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
    tokens_generated: int = 0
    elapsed_s: float = 0.0
    route_used: str = ""
    cost_tier: int = 0
    scoring_method: str = "exact_match"
    partial: bool = False  # Inference completed with partial output (read_timeout)
    degraded: bool = False  # Inference completed in degraded mode
    confidence: float = 0.0  # EV-1: Model confidence proxy (0-1). Initially float(correct); upgraded to logprobs when available.
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
    host_covariates: dict[str, Any] = field(default_factory=dict)


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


# ── EV-4 / EV-11 verifier-mode metric primitives (additive, 2026-07-17) ──────
# BUILD-evalbatch-verifier-mode. These are PURE, inference-free helpers.
# EvalTower.eval_calibration (EV-4) and EvalTower.eval_math_rebaseline (EV-11)
# below reuse the existing _eval_batch dispatch/scoring for GENERATION, then feed
# the resulting (confidence, correctness) vectors through these functions. They
# are unit-tested on synthetic vectors WITHOUT any inference. ECE + AUROC delegate
# to the clean-room src/llm_primitives/stat_tests implementations (already the
# consolidated impls the tower uses); the remaining four calibration metrics are
# computed here because stat_tests does not carry them.

# The six per-role EV-4 calibration metrics, in report order.
CALIBRATION_METRIC_KEYS = (
    "ece",
    "auroc",
    "top1_accuracy",
    "bottom1_accuracy",
    "spearman_rho",
    "mae",
)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based tie-averaged ranks (same convention as stat_tests.roc_auc)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation (tie-averaged).

    None when undefined: fewer than 2 paired points, a length mismatch, or a
    constant vector on either axis (rank variance is 0).
    """
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _cohort_accuracy(
    confidences: Sequence[float],
    labels: Sequence[float],
    *,
    pick_max: bool,
) -> float | None:
    """Accuracy of the max- (or min-) confidence cohort.

    Ties are averaged: when several items share the extreme confidence the metric
    is the mean label over that whole cohort, so with distinct confidences this
    reduces to exactly the single top-1 / bottom-1 item. A discriminative
    confidence signal shows top1_accuracy high and bottom1_accuracy low.
    """
    if not confidences:
        return None
    target = max(confidences) if pick_max else min(confidences)
    cohort = [lab for conf, lab in zip(confidences, labels) if conf == target]
    if not cohort:
        return None
    return sum(cohort) / len(cohort)


def compute_calibration_metrics(
    confidences: Sequence[float],
    labels: Sequence[float],
) -> dict[str, float | None]:
    """EV-4 calibration metrics from paired (confidence, correctness) vectors.

    ``confidences`` — model confidence proxy in [0, 1]
    (EvalTower ``QuestionResult.confidence``).
    ``labels`` — ground-truth correctness (0/1 or float in [0, 1]).

    Returns a dict carrying every key in ``CALIBRATION_METRIC_KEYS`` plus ``n``.
    Every metric is None-safe (returns None where undefined — empty input, a
    single class / <3 distinct confidences for AUROC, a constant vector for
    Spearman). No inference is performed.

    Definitions:
      * ece             — Expected Calibration Error (10-bin) via stat_tests.
      * auroc           — ROC-AUC (tie-averaged Mann-Whitney U) via stat_tests.
      * top1_accuracy   — accuracy of the most-confident cohort (see
                          ``_cohort_accuracy``).
      * bottom1_accuracy— accuracy of the least-confident cohort.
      * spearman_rho    — rank correlation of confidence vs correctness.
      * mae             — mean |confidence - label| (sample-level calibration error).
    """
    conf = [float(c) for c in confidences]
    lab = [float(y) for y in labels]
    if len(conf) != len(lab):
        raise ValueError(
            f"confidences/labels length mismatch: {len(conf)} != {len(lab)}"
        )
    n = len(conf)
    ece = expected_calibration_error(conf, lab, n_bins=10) if n else None
    # AUROC is only meaningful with both classes present AND >2 distinct
    # confidences — the same guard EvalTower._aggregate applies. roc_auc() itself
    # already returns None when a single class is present.
    auroc: float | None = None
    if n and len({round(c, 6) for c in conf}) > 2 and len({round(y) for y in lab}) > 1:
        auroc = roc_auc(conf, lab)
    mae = (sum(abs(c - y) for c, y in zip(conf, lab)) / n) if n else None
    return {
        "n": n,
        "ece": ece,
        "auroc": auroc,
        "top1_accuracy": _cohort_accuracy(conf, lab, pick_max=True),
        "bottom1_accuracy": _cohort_accuracy(conf, lab, pick_max=False),
        "spearman_rho": _spearman_rho(conf, lab),
        "mae": mae,
    }


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

    Returns a JSON-serializable dict with keys ``z``, ``alpha``, ``n_arms``,
    ``arms`` (per-arm accuracy + Wilson CI over that arm's own outcomes),
    ``pairs`` (the paired screen, one record per matched pair), and
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
                }
            )

    return {
        "z": z,
        "alpha": alpha,
        "n_arms": len(prepared),
        "arms": per_arm,
        "pairs": pairs,
        "mismatched_pairs": mismatched,
    }


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
    for key, value in raw_scores.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            scores[str(key)] = min(max(numeric, 0.0), 1.0)
    return scores


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
        self._pool = None
        self._core_cache: dict[str, tuple[list[dict], dict[str, Any], Path]] = {}
        self._trial_id_context: int | None = None
        self.on_question = on_question
        self.on_progress = on_progress

    def set_trial_context(self, trial_id: int | str | None) -> None:
        """Set the current AutoPilot trial id for deterministic audit sampling."""
        try:
            self._trial_id_context = int(trial_id) if trial_id is not None else None
        except (TypeError, ValueError):
            self._trial_id_context = None

    def _resolve_trial_id(self, trial_id: int | None = None) -> int | None:
        return trial_id if trial_id is not None else self._trial_id_context

    # ── sentinel questions (T0) ──────────────────────────────────

    def _load_sentinels(self) -> list[dict]:
        if self._sentinels is not None:
            return self._sentinels
        if not self._sentinel_path.exists():
            log.warning("No sentinel file at %s", self._sentinel_path)
            self._sentinels = []
            return self._sentinels
        self._sentinels = yaml.safe_load(self._sentinel_path.read_text()) or []
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
        loaded = yaml.safe_load(TOOL_SENTINEL_PATH.read_text()) or []
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
        if self._pool is not None:
            return self._pool
        try:
            _research_root = Path("/mnt/raid0/llm/epyc-inference-research")
            sys.path.insert(0, str(_research_root / "scripts" / "benchmark"))
            from question_pool import load_pool

            self._pool = load_pool()
        except Exception as e:
            log.warning("Could not load question pool: %s", e)
            self._pool = {}
        return self._pool

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

    # ── single question evaluation ───────────────────────────────

    def _rubric_scores_for_answer(
        self,
        *,
        q: dict,
        answer: str,
        generator_model: str,
        tool_events: list[str],
        client: httpx.Client,
    ) -> dict[str, float]:
        fallback = deterministic_rubric_fallback(
            answer,
            expected_contains=q.get("expected_contains") or (),
            tool_events=tool_events,
        )
        judge_roles = _configured_rubric_judge_roles()
        if not judge_roles:
            return fallback

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
            return fallback
        combined = dict(fallback)
        dimensions = sorted({dim for scores in judge_scores for dim in scores})
        for dim in dimensions:
            values = [scores[dim] for scores in judge_scores if dim in scores]
            if values:
                combined[dim] = sum(values) / len(values)
        return combined

    def _eval_question(self, q: dict, client: httpx.Client) -> QuestionResult:
        """Evaluate a single question through the orchestrator."""
        prompt = q.get("prompt", "")
        expected = q.get("expected", "")
        qid = q.get("id", q.get("question_id", "unknown"))
        suite = q.get("suite", "unknown")
        stable_qid = str(q.get("qid") or q.get("stable_qid") or "").strip()
        if not stable_qid:
            stable_qid = _stable_question_qid(str(suite), str(prompt))
        scoring_method = q.get("scoring_method", "exact_match")
        scoring_config = q.get("scoring_config", {})
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
            if "tools" in q:
                call_kwargs["tools"] = q.get("tools")
            if "tool_choice" in q:
                call_kwargs["tool_choice"] = q.get("tool_choice")
            prompt_root = str(q.get("_prompt_root") or "").strip()
            if prompt_root:
                call_kwargs["prompt_root"] = prompt_root
            resp = call_orchestrator_forced(**call_kwargs)
            elapsed = time.time() - start
            answer = resp.get("answer", "")
            error = resp.get("error")
            tokens = _nonnegative_int(resp.get("tokens_generated", 0))
            host_covariates = _capture_host_timing_covariates(
                tokens_generated=tokens,
                elapsed_s=elapsed,
            )

            correct = False
            rubric_scores: dict[str, float] = {}
            if not error and _is_scoreable_question(q):
                if _is_rubric_scored_question(q):
                    rubric_scores = self._rubric_scores_for_answer(
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
                    correct = score_answer_deterministic(
                        answer=answer,
                        expected=expected,
                        scoring_method=scoring_method,
                        scoring_config=scoring_config,
                    )

            # EV-1: Confidence proxy. Binary for now (correct=1.0, incorrect=0.0).
            # When logprob passthrough lands, replace with model output confidence.
            # For code_execution, scoring_config may contain a pass_rate (0-1).
            confidence = float(correct)
            if scoring_method == "code_execution":
                confidence = float(scoring_config.get("pass_rate", correct))
            elif scoring_method == "rubric" and rubric_scores:
                confidence = aggregate_rubric_score(rubric_scores).score

            # 2026-05-23 Phase 4 — exogenous-restart metadata propagation.
            # call_orchestrator_forced attaches the resilient_post meta dict
            # as resp["_meta"] when watcher is set. Surface the classification
            # bits onto QuestionResult so _aggregate can roll them up into
            # the trial-level EvalResult.
            meta = resp.get("_meta") or {}
            return QuestionResult(
                question_id=qid,
                suite=suite,
                prompt=prompt,
                expected=expected,
                qid=stable_qid,
                answer=answer,
                correct=correct,
                error=error,
                tokens_generated=tokens,
                elapsed_s=elapsed,
                route_used=str(resp.get("routed_to") or resp.get("model") or ""),
                cost_tier=resp.get("cost_tier", 0),
                scoring_method=scoring_method,
                partial=bool(resp.get("partial", False)),
                degraded=bool(resp.get("degraded", False)),
                confidence=confidence,
                branching_density=_compute_branching_density(answer),
                tools_used=int(resp.get("tools_used", 0) or 0),
                tools_called=list(resp.get("tools_called") or []),
                exogenous_recovered=bool(meta.get("exogenous_recovered", False)),
                exogenous_unrecovered=bool(meta.get("exogenous_unrecovered", False)),
                external_restart=bool(meta.get("external_restart", False)),
                retry_count=int(meta.get("retry_count", 0)),
                eval_partition=eval_partition,
                rubric_scores=rubric_scores,
                host_covariates=host_covariates,
            )
        except Exception as e:
            elapsed = time.time() - start
            host_covariates = _capture_host_timing_covariates(
                tokens_generated=0,
                elapsed_s=elapsed,
            )
            return QuestionResult(
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
        if workers <= 1:
            wall_budget_s = _eval_batch_wall_budget_s(
                n_questions=n,
                workers=workers,
                request_timeout_s=self.timeout,
            )
            for i, q in enumerate(dispatch_questions):
                results[i] = self._eval_question(q, client)
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
                    break
            batch_wall_s = time.time() - batch_start
            out = [r for r in results if r is not None]
            for r in out:
                r.eval_concurrency = workers
                r.eval_wall_s = batch_wall_s
            return out

        ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"eval-{label}")
        done = 0
        try:
            future_to_idx = {
                ex.submit(self._eval_question, q, client): i
                for i, q in enumerate(dispatch_questions)
            }
            pending = set(future_to_idx)
            no_progress_timeout_s = _eval_no_progress_timeout_s(self.timeout)
            while pending:
                completed, pending = wait(
                    pending,
                    timeout=no_progress_timeout_s or None,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    elapsed = time.time() - batch_start
                    log.error(
                        "%s no eval future completed for %.1fs; failing %d "
                        "remaining question(s) closed",
                        label,
                        no_progress_timeout_s,
                        len(pending),
                    )
                    for fut in pending:
                        idx = future_to_idx[fut]
                        fut.cancel()
                        results[idx] = self._failed_question_result(
                            questions[idx],
                            elapsed_s=elapsed,
                            error=(
                                "eval_no_progress_timeout: no completed future "
                                f"for {no_progress_timeout_s:.1f}s"
                            ),
                        )
                    break

                for fut in completed:
                    idx = future_to_idx[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        results[idx] = self._failed_question_result(
                            questions[idx],
                            elapsed_s=time.time() - batch_start,
                            error=str(exc),
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
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        for i, q in enumerate(questions):
            if results[i] is None:
                results[i] = self._failed_question_result(
                    q,
                    elapsed_s=time.time() - batch_start,
                    error="eval_cancelled_after_no_progress_timeout",
                )
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
        ece = 0.0
        auroc = 0.0
        cal_violations = 0
        if confidences:
            # EV-11b (operator-decided 2026-07-20): use the canonical closed-top-bin
            # ECE from stat_tests. This is a scoring-semantics change and is
            # era-labeled in details so pre/post EV-11b numbers are not mixed.
            ece = expected_calibration_error(confidences, correctness_vals, n_bins=10) or 0.0
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
                "ece_binning": "closed_top_bin_stat_tests",
                "ece_instrument_era": "ev11b_closed_bin_2026_07_20",
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
            log.error("No sentinel questions available for T0")
            return EvalResult(tier=0, quality=0, speed=0, cost=0, reliability=0)

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

        return self._aggregate(results, tier=0)

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
                log.error("No question pool available for T1")
                return EvalResult(tier=1, quality=0, speed=0, cost=0, reliability=0)
            rng = random.Random(seed)
            questions = _sample_scoreable_eval_questions(pool, n, rng)
            core_id = f"legacy_pool_seed_{seed}_n{n}"

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
        questions = (
            _annotate_partition(questions, "core")
            + _annotate_partition(audit_questions, "audit")
            + _annotate_partition(self._load_tool_sentinels(), "tool_sentinel")
        )

        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(questions, client, log_every=10, label="T1")

        full_result = self._aggregate(results, tier=1)
        # W6 audit rows are an overfit/generalization signal. Keep them in the
        # per-question ledger, but keep decision metrics paired-core-only by default.
        if audit_policy["active"] and audit_policy["shadow_only"]:
            decision_results = [r for r in results if (r.eval_partition or "core") != "audit"]
            result = self._aggregate(decision_results, tier=1)
            result.question_results = full_result.question_results
            for key in (
                "partition_quality",
                "partition_counts",
                "partition_suite_quality",
            ):
                result.details[key] = full_result.details.get(key, {})
            result.details.update(
                {
                    "audit_shadow_only": True,
                    "audit_shadow_total_n_questions": full_result.n_questions,
                    "audit_shadow_decision_n_questions": result.n_questions,
                    "audit_shadow_excluded_partitions": ["audit"],
                }
            )
        else:
            result = full_result
        result.core_id = core_id
        result.details.update(
            {
                "core_id": core_id,
                "core_selection": core_selection,
                "core_path": core_path,
                "core_metadata": core_metadata,
                "core_era_guard": core_era_guard,
                "requested_n": n,
                "base_core_questions": base_core_questions,
                "base_audit_questions": base_audit_questions,
                "audit_policy": audit_policy,
            }
        )
        return result

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
            log.error("No question pool available for T2")
            return EvalResult(tier=2, quality=0, speed=0, cost=0, reliability=0)

        resolved_trial_id = self._resolve_trial_id(trial_id)
        requested_n = int(n)
        draw_seed = int(seed)
        promotion_policy: dict[str, Any] = {
            "enabled": bool(promotion_eval),
        }
        if promotion_eval:
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

        rng = random.Random(seed)
        if promotion_eval:
            rng = random.Random(draw_seed)
        questions = _sample_scoreable_eval_questions(
            pool,
            requested_n,
            rng,
            exclude_qids=exclude_qids if promotion_eval else None,
            exclude_suites=excluded_suites if promotion_eval else None,
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
        questions = _annotate_partition(questions, "core") + _annotate_partition(
            self._load_tool_sentinels(), "tool_sentinel"
        )

        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(questions, client, log_every=50, label="T2")

        result = self._aggregate(results, tier=2)
        if promotion_eval:
            promotion_policy["actual_n"] = len(
                [q for q in questions if q.get("eval_partition") == "core"]
            )
            result.details["promotion_eval_policy"] = promotion_policy
            result.core_id = f"w8_promotion_eval_v1_trial_{resolved_trial_id}_n{requested_n}"
        return result

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
        result.core_id = f"t3_hard_only_v1_seed_{draw_seed}_n{requested_n}"
        return result

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

        Reuses the same research ``scripts/benchmark`` sys.path insertion as
        ``_load_pool`` so the canonical suite→adapter registry (``get_adapter``)
        is the single source of truth for what each suite contains.
        """
        research_bench = str(
            Path("/mnt/raid0/llm/epyc-inference-research") / "scripts" / "benchmark"
        )
        if research_bench not in sys.path:
            sys.path.insert(0, research_bench)
        from dataset_adapters import get_adapter  # research suite registry

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
            if qid.startswith("sv_"):
                tail = qid[3:]
                parts = tail.rsplit("_", 1)
                qid_subset = parts[0] if len(parts) == 2 and parts[1].isdigit() else tail
                return qid_subset == needle
            return (
                qid == needle
                or qid.startswith(f"{needle}_")
            )

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
                per_role[role] = {
                    "role": role,
                    "n_questions": len(results),
                    "n_scored": len(scored),
                    "accuracy": (correct / len(scored)) if scored else None,
                    **{key: metrics.get(key) for key in CALIBRATION_METRIC_KEYS},
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
                per_role[role] = {
                    "role": role,
                    "arm": arm_label,
                    "n_questions": len(results),
                    "n_scored": len(scored),
                    "correct": correct,
                    "accuracy": (correct / len(scored)) if scored else None,
                    "quality": agg.quality,
                    "reliability": agg.reliability,
                    "ece": agg.ece,
                    "auroc": agg.auroc,
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
            # Paired-significance screen over the arms: exact McNemar p on the
            # flip pairs + per-arm Wilson CIs, gated on matched dataset+profile.
            # Empty ``pairs`` when fewer than two arms were scored.
            "paired_significance": screen_paired_arms(arm_screens),
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
        try:
            if not self.TAP_PATH.exists():
                return ""
            with open(self.TAP_PATH, "rb") as f:
                # Seek to approximate tail position
                f.seek(0, 2)  # EOF
                size = f.tell()
                # Read last ~8KB (generous for n_lines)
                read_bytes = min(size, n_lines * 160)
                f.seek(max(0, size - read_bytes))
                tail = f.read().decode("utf-8", errors="replace")
            lines = tail.splitlines()
            return "\n".join(lines[-n_lines:])
        except Exception as e:
            log.warning("Could not capture traces: %s", e)
            return ""

    @staticmethod
    def _trim_trace_text(trace_text: Any, max_chars: int) -> str:
        text = str(trace_text or "").strip()
        if not text or max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return "[trace truncated]\n" + text[-max_chars:]

    @staticmethod
    def _trace_ir_steps(
        trace_text: str, *, max_steps: int = 12, preview_chars: int = 240
    ) -> list[dict[str, Any]]:
        """Convert a tap tail into compact ROLE/PROMPT/RESPONSE steps."""
        text = str(trace_text or "").strip()
        if not text:
            return []

        sections: list[tuple[str, str]] = []
        current_kind = "trace"
        current_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            upper = line.upper()
            if upper.startswith("ROLE"):
                if current_lines:
                    sections.append((current_kind, "\n".join(current_lines).strip()))
                sections.append(("role", line.strip()))
                current_kind = "trace"
                current_lines = []
            elif upper in {"PROMPT:", "PROMPT"}:
                if current_lines:
                    sections.append((current_kind, "\n".join(current_lines).strip()))
                current_kind = "prompt"
                current_lines = []
            elif upper in {"RESPONSE:", "RESPONSE"}:
                if current_lines:
                    sections.append((current_kind, "\n".join(current_lines).strip()))
                current_kind = "response"
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sections.append((current_kind, "\n".join(current_lines).strip()))

        steps: list[dict[str, Any]] = []
        for kind, content in sections:
            cleaned = content.strip()
            if not cleaned:
                continue
            steps.append(
                {
                    "step_id": f"s{len(steps) + 1}",
                    "kind": kind,
                    "line_count": len(cleaned.splitlines()),
                    "content_hash": hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12],
                    "content_preview": cleaned[:preview_chars],
                }
            )
            if len(steps) >= max_steps:
                break
        return steps

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
        """Build a deterministic, observe-only trace IR for critic/prompt context.

        This is a structured companion to the legacy formatted trace text. It is
        intentionally not consumed by any score, safety, or acceptance gate.
        """

        def selected(outcome: str, limit: int) -> list[dict[str, Any]]:
            if limit <= 0:
                return []
            matches = [
                item
                for item in trace_bank or []
                if isinstance(item, dict)
                and str(item.get("outcome") or "").lower() == outcome
                and str(item.get("trace") or "").strip()
            ]
            return matches[-limit:]

        examples: list[dict[str, Any]] = []
        for outcome, limit in (("success", int(k_success)), ("failure", int(k_failure))):
            for raw in selected(outcome, limit):
                trace = cls._trim_trace_text(raw.get("trace", ""), max_trace_chars)
                if not trace:
                    continue
                examples.append(
                    {
                        "outcome": outcome,
                        "trial_id": raw.get("trial_id"),
                        "species": str(raw.get("species") or ""),
                        "action_type": str(raw.get("action_type") or ""),
                        "reason": str(raw.get("reason") or "")[:500],
                        "trace_hash": str(
                            raw.get("trace_hash")
                            or hashlib.sha256(trace.encode("utf-8")).hexdigest()[:12]
                        ),
                        "steps": cls._trace_ir_steps(trace),
                    }
                )

        raw_tail = ""
        if not examples:
            raw_tail = cls._trim_trace_text(raw_trace_text, max_trace_chars)
            if raw_tail:
                examples.append(
                    {
                        "outcome": "unlabeled",
                        "trial_id": trial_id,
                        "species": "",
                        "action_type": "",
                        "reason": "raw_recent_trace_fallback",
                        "trace_hash": hashlib.sha256(raw_tail.encode("utf-8")).hexdigest()[:12],
                        "steps": cls._trace_ir_steps(raw_tail),
                    }
                )

        return {
            "schema_version": "harness_trace_ir.v1",
            "observe_only": True,
            "acceptance_effect": "none_observe_only",
            "trial_id": trial_id,
            "failure_summary": str(failure_summary or "")[:500],
            "source": "contrastive_trace_bank"
            if trace_bank and examples and not raw_tail
            else "raw_recent_traces",
            "trace_examples": examples,
        }

    @staticmethod
    def format_critic_trace_ir(trace_ir: dict[str, Any] | None) -> str:
        """Render critic trace IR as a prompt-safe JSON block."""
        if not isinstance(trace_ir, dict) or not trace_ir.get("trace_examples"):
            return ""
        return (
            "## Harness Trace IR (MH-11 observe-only)\n"
            "This structured trace evidence is diagnostic context only; it is not "
            "an acceptance score or quality gate.\n"
            "```json\n"
            f"{json.dumps(trace_ir, sort_keys=True, indent=2)}\n"
            "```"
        )

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
        """Append one labeled trace example and cap the in-state contrastive bank.

        The raw tap file has no success/failure label, so callers add labels only
        after the trial verdict is known. The returned bank is JSON-serializable
        for autopilot_state.json.
        """
        normalized_outcome = str(outcome or "").strip().lower()
        if normalized_outcome not in {"success", "failure"}:
            return list(trace_bank or [])
        trace = cls._trim_trace_text(trace_text, max_trace_chars)
        if not trace:
            return list(trace_bank or [])

        normalized: list[dict[str, Any]] = []
        for raw in trace_bank or []:
            if not isinstance(raw, dict):
                continue
            raw_outcome = str(raw.get("outcome") or "").strip().lower()
            if raw_outcome not in {"success", "failure"}:
                continue
            raw_trace = cls._trim_trace_text(raw.get("trace", ""), max_trace_chars)
            if not raw_trace:
                continue
            normalized.append(
                {
                    "outcome": raw_outcome,
                    "trial_id": raw.get("trial_id"),
                    "species": str(raw.get("species") or ""),
                    "action_type": str(raw.get("action_type") or ""),
                    "reason": str(raw.get("reason") or ""),
                    "trace": raw_trace,
                    "trace_hash": str(
                        raw.get("trace_hash")
                        or hashlib.sha256(raw_trace.encode("utf-8")).hexdigest()[:12]
                    ),
                }
            )

        trace_hash = hashlib.sha256(trace.encode("utf-8")).hexdigest()[:12]
        normalized = [
            item
            for item in normalized
            if not (
                item.get("outcome") == normalized_outcome
                and item.get("trial_id") == trial_id
                and item.get("trace_hash") == trace_hash
            )
        ]
        normalized.append(
            {
                "outcome": normalized_outcome,
                "trial_id": trial_id,
                "species": str(species or ""),
                "action_type": str(action_type or ""),
                "reason": str(reason or ""),
                "trace": trace,
                "trace_hash": trace_hash,
            }
        )

        capped: list[dict[str, Any]] = []
        for bucket in ("success", "failure"):
            capped.extend(
                [item for item in normalized if item.get("outcome") == bucket][
                    -max_examples_per_outcome:
                ]
            )
        return capped

    def capture_contrastive_traces(
        self,
        *,
        k_success: int = 2,
        k_failure: int = 2,
        trace_bank: list[dict[str, Any]] | None = None,
    ) -> str:
        """Format labeled success/failure trace examples for PromptForge.

        This intentionally reads only a caller-maintained trace bank. Raw tap
        tails are still available through capture_recent_traces() as the fallback
        when labeled examples are not available yet.
        """
        if not trace_bank:
            return ""

        def selected(outcome: str, limit: int) -> list[dict[str, Any]]:
            if limit <= 0:
                return []
            matches = [
                item
                for item in trace_bank
                if isinstance(item, dict)
                and str(item.get("outcome") or "").lower() == outcome
                and str(item.get("trace") or "").strip()
            ]
            return matches[-limit:]

        success_examples = selected("success", int(k_success))
        failure_examples = selected("failure", int(k_failure))
        if not success_examples and not failure_examples:
            return ""

        def append_entry(lines: list[str], idx: int, entry: dict[str, Any]) -> None:
            trial = entry.get("trial_id")
            label_parts = []
            if trial is not None:
                label_parts.append(f"trial #{trial}")
            species = str(entry.get("species") or "").strip()
            action_type = str(entry.get("action_type") or "").strip()
            if species or action_type:
                label_parts.append("/".join(part for part in (species, action_type) if part))
            label = ", ".join(label_parts) or "unlabeled trial"
            lines.append(f"[{idx}] {label}")
            reason = str(entry.get("reason") or "").strip()
            if reason:
                lines.append(f"Reason: {reason}")
            lines.append("Trace:")
            lines.append(str(entry.get("trace") or "").strip())

        lines: list[str] = ["## Contrastive Execution Traces"]
        if success_examples:
            lines.append("### Success Examples")
            for idx, entry in enumerate(success_examples, start=1):
                append_entry(lines, idx, entry)
        if failure_examples:
            lines.append("### Failure Examples")
            for idx, entry in enumerate(failure_examples, start=1):
                append_entry(lines, idx, entry)
        return "\n".join(lines)

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
