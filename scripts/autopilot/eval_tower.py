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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
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
            q for q in suite_qs if _question_qid(q) not in excluded and _is_scoreable_question(q)
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


def _compact_question_result(r: "QuestionResult") -> dict[str, Any]:
    item: dict[str, Any] = {
        "qid": r.qid or _stable_question_qid(str(r.suite), str(r.prompt)),
        "suite": r.suite,
        "partition": r.eval_partition or "core",
        "correct": bool(r.correct),
        "latency_ms": int(round(max(0.0, r.elapsed_s) * 1000)),
        "tools_used": int(r.tools_used or 0),
    }
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
        if matrix_status(current_topology_hash=current_hash) != MatrixStatus.OK:
            return False
        return pair_policy(role, role, TrafficClass.BACKGROUND, matrix=matrix) == PairDecision.ALLOW
    except Exception:
        return False


def _live_safe_concurrency(role: str, topology_cap: int) -> int:
    """Bound eval fan-out by the currently reachable role instances.

    Static topology can say a role is safe at N>1 while the live stack is
    intentionally launched in full-only mode. In that case, concurrent evals
    pile onto one llama-server and can corrupt evidence with 5xx/timeouts.
    """
    if topology_cap <= 1:
        return 1
    if os.environ.get("AUTOPILOT_EVAL_REQUIRE_LIVE_FLEET", "1") == "0":
        return topology_cap
    try:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
        from src.runtime.instance_topology import cpu_list_to_regions
    except Exception:
        return 1

    instances = ((NUMA_CONFIG or {}).get(role) or {}).get("instances") or []
    if not instances:
        return 1

    live_regions: list[frozenset[str]] = []
    for entry in instances:
        if not entry or len(entry) < 2:
            continue
        try:
            port = int(entry[1])
        except (TypeError, ValueError):
            continue
        try:
            resp = httpx.get(f"http://localhost:{port}/health", timeout=0.5)
            if resp.status_code != 200:
                continue
        except Exception:
            continue
        live_regions.append(cpu_list_to_regions(str(entry[0])))

    if not live_regions:
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


def _eval_concurrency() -> int:
    raw = os.environ.get("AUTOPILOT_EVAL_CONCURRENCY")
    if raw is not None:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass  # fall through to topology default
    bottleneck = os.environ.get("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "frontdoor")
    try:
        from src.runtime.instance_topology import max_safe_concurrency

        topology_cap = max(1, max_safe_concurrency(bottleneck))
        if topology_cap <= 1:
            return 1
        if not _same_role_matrix_allows_eval_fanout(bottleneck):
            return 1
        return _live_safe_concurrency(bottleneck, topology_cap)
    except Exception:
        return 1


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
            tokens = resp.get("tokens_generated", 0)

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
            )
        except Exception as e:
            elapsed = time.time() - start
            return QuestionResult(
                question_id=qid,
                suite=suite,
                prompt=prompt,
                expected=expected,
                qid=stable_qid,
                error=str(e),
                elapsed_s=elapsed,
                eval_partition=eval_partition,
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
        return QuestionResult(
            question_id=q.get("id", q.get("question_id", "unknown")),
            suite=suite,
            prompt=prompt,
            expected=q.get("expected", ""),
            qid=stable_qid,
            error=error,
            elapsed_s=elapsed_s,
            eval_partition=str(q.get("eval_partition") or "core"),
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
        workers = min(n, _eval_concurrency())
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
            for i, q in enumerate(dispatch_questions):
                results[i] = self._eval_question(q, client)
                if log_every and (i + 1) % log_every == 0:
                    correct_so_far = sum(1 for r in results if r and r.correct)
                    log.info(
                        "%s progress: %d/%d (%.0f%% correct)",
                        label,
                        i + 1,
                        n,
                        100 * correct_so_far / (i + 1),
                    )
                    self._emit_progress(
                        label=label,
                        completed_questions=i + 1,
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
                    if log_every and done % log_every == 0:
                        correct_so_far = sum(1 for r in results if r and r.correct)
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

        # Quality: fraction correct scaled to 0-3
        correct_count = sum(1 for r in results if r.correct)
        quality = (correct_count / len(results)) * 3.0

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
        task_rate_qph = (len(results) / (eval_wall_s / 3600.0)) if eval_wall_s > 0 else 0.0
        goodput_qph = (quality / 3.0) * task_rate_qph
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
        non_error = sum(1 for r in results if not r.error)
        reliability = non_error / len(results)

        # Per-suite quality
        suite_correct: dict[str, list[bool]] = {}
        for r in results:
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
        for r in results:
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
            n_bins = 10
            for i in range(n_bins):
                lo = i / n_bins
                hi = (i + 1) / n_bins
                mask = [lo <= c < hi for c in confidences]
                bin_count = sum(mask)
                if bin_count > 0:
                    bin_acc = sum(cr for cr, m in zip(correctness_vals, mask) if m) / bin_count
                    bin_conf = sum(c for c, m in zip(confidences, mask) if m) / bin_count
                    ece += (bin_count / len(confidences)) * abs(bin_acc - bin_conf)
            # AUC: only meaningful with non-degenerate confidence (>2 distinct values)
            distinct_conf = len(set(round(c, 6) for c in confidences))
            if distinct_conf > 2 and len(set(correctness_vals)) > 1:
                try:
                    from sklearn.metrics import roc_auc_score

                    auroc = roc_auc_score(correctness_vals, confidences)
                except (ImportError, ValueError):
                    auroc = 0.0
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
            n_questions=len(results),
            question_results=question_results,
            details={
                "correct": correct_count,
                "total": len(results),
                "per_suite_counts": per_suite_counts,
                "partition_quality": partition_quality,
                "partition_counts": partition_counts,
                "partition_suite_quality": partition_suite_quality,
                "errors": sum(1 for r in results if r.error),
                "speed_semantics": "speed is the objective speed used by safety/Pareto; median_request_tps and aggregate_tps retain raw throughput components",
                "speed_metric_mode": speed_metric_mode,
                "objective_speed_tps": speed,
                "median_request_tps": median_request_speed,
                "aggregate_tps": aggregate_speed,
                "eval_concurrency": eval_concurrency,
                "eval_wall_s": eval_wall_s,
                "sum_request_elapsed_s": sum_request_elapsed_s,
                "tokens_generated": total_tokens_generated,
                "task_rate_qph": task_rate_qph,
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
