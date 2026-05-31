"""Tiered evaluation tower: T0 (10q/30s) → T1 (100q/5m) → T2 (500+/30m).

Wraps existing seeding infrastructure for orchestrator API calls and scoring.
Training set (debug suites) is kept separate from validation set (HF benchmarks).
"""

from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from safety_gate import EvalResult

log = logging.getLogger("autopilot.eval")

SENTINEL_PATH = Path(__file__).resolve().parent / "sentinel_questions.yaml"
ORCHESTRATOR_URL = "http://localhost:8000"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _read_registry_timeout(category: str, key: str, fallback: int) -> int:
    registry_path = (
        Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"
    )
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
# Default behavior (WP-1): topology-derived safe-N from the bottleneck role
# (`AUTOPILOT_EVAL_BOTTLENECK_ROLE`, default "frontdoor"). For the current
# stack, that's 3 (full + q3 + q2 disjoint) — the largest fan-out that lands
# every request on a cpuset disjoint from all the others under
# ConcurrencyAwareBackend's full-first dispatcher. Roles whose full instance
# covers all 0-95 (worker_general, architect_general) cap the default at 1.
#
# Operators can still override via `AUTOPILOT_EVAL_CONCURRENCY=N`. The env
# override always wins, even over the topology cap, because some test/diag
# paths intentionally exceed it (e.g. WP-3 migration smoke tests).
#
# Reference topology safe-N: frontdoor=3, ingest_long_context=3,
# vision_escalation=3, worker_general=1, architect_general=1, worker_vision=1.
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
        return max(1, max_safe_concurrency(bottleneck))
    except Exception:
        return 1

# Import seeding infrastructure
import sys

_orch_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_orch_root / "scripts" / "benchmark"))

from seeding_orchestrator import call_orchestrator_forced  # noqa: E402
from seeding_scoring import score_answer_deterministic  # noqa: E402

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


class EvalTower:
    """Progressive evaluation: T0 → T1 → T2."""

    def __init__(
        self,
        url: str = ORCHESTRATOR_URL,
        timeout: int = DEFAULT_TIMEOUT,
        sentinel_path: Path | None = None,
        on_question: "Callable[[str], None] | None" = None,
    ):
        self.url = url
        self.timeout = timeout
        self._sentinel_path = sentinel_path or SENTINEL_PATH
        self._sentinels: list[dict] | None = None
        self._pool = None
        self.on_question = on_question

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

    # ── single question evaluation ───────────────────────────────

    def _eval_question(
        self, q: dict, client: httpx.Client
    ) -> QuestionResult:
        """Evaluate a single question through the orchestrator."""
        prompt = q.get("prompt", "")
        expected = q.get("expected", "")
        qid = q.get("id", q.get("question_id", "unknown"))
        suite = q.get("suite", "unknown")
        scoring_method = q.get("scoring_method", "exact_match")
        scoring_config = q.get("scoring_config", {})
        image_path = q.get("image_path", "")

        if self.on_question:
            self.on_question(prompt)

        start = time.time()
        try:
            resp = call_orchestrator_forced(
                prompt=prompt,
                force_role="",  # Let routing decide
                force_mode="",
                url=self.url,
                timeout=self.timeout,
                image_path=image_path,
                client=client,
                watcher=getattr(self, "watcher", None),
            )
            elapsed = time.time() - start
            answer = resp.get("answer", "")
            error = resp.get("error")
            tokens = resp.get("tokens_generated", 0)

            correct = False
            if not error and expected:
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
                answer=answer,
                correct=correct,
                error=error,
                tokens_generated=tokens,
                elapsed_s=elapsed,
                route_used=resp.get("model", ""),
                cost_tier=resp.get("cost_tier", 0),
                scoring_method=scoring_method,
                partial=bool(resp.get("partial", False)),
                degraded=bool(resp.get("degraded", False)),
                confidence=confidence,
                branching_density=_compute_branching_density(answer),
                exogenous_recovered=bool(meta.get("exogenous_recovered", False)),
                exogenous_unrecovered=bool(meta.get("exogenous_unrecovered", False)),
                external_restart=bool(meta.get("external_restart", False)),
                retry_count=int(meta.get("retry_count", 0)),
            )
        except Exception as e:
            elapsed = time.time() - start
            return QuestionResult(
                question_id=qid,
                suite=suite,
                prompt=prompt,
                expected=expected,
                error=str(e),
                elapsed_s=elapsed,
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
        results: list[QuestionResult | None] = [None] * n
        batch_start = time.time()
        if workers <= 1:
            for i, q in enumerate(questions):
                results[i] = self._eval_question(q, client)
                if log_every and (i + 1) % log_every == 0:
                    correct_so_far = sum(1 for r in results if r and r.correct)
                    log.info(
                        "%s progress: %d/%d (%.0f%% correct)",
                        label, i + 1, n, 100 * correct_so_far / (i + 1),
                    )
            batch_wall_s = time.time() - batch_start
            out = [r for r in results if r is not None]
            for r in out:
                r.eval_concurrency = workers
                r.eval_wall_s = batch_wall_s
            return out

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"eval-{label}") as ex:
            future_to_idx = {
                ex.submit(self._eval_question, q, client): i
                for i, q in enumerate(questions)
            }
            done = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                done += 1
                if log_every and done % log_every == 0:
                    correct_so_far = sum(1 for r in results if r and r.correct)
                    log.info(
                        "%s progress: %d/%d (%.0f%% correct, concurrency=%d)",
                        label, done, n, 100 * correct_so_far / done, workers,
                    )
        batch_wall_s = time.time() - batch_start
        out = [r for r in results if r is not None]
        for r in out:
            r.eval_concurrency = workers
            r.eval_wall_s = batch_wall_s
        return out

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
        eval_concurrency = max((r.eval_concurrency for r in results), default=1)
        concurrent_eval = eval_concurrency > 1 and aggregate_speed > 0
        speed_metric_mode = (
            "aggregate_batch_tps" if concurrent_eval else "median_request_tps"
        )
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
        per_suite = {
            suite: (sum(vals) / len(vals)) * 3.0
            for suite, vals in suite_correct.items()
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

        # AP-16: Instruction token budget
        instruction_tokens = self._count_instruction_tokens()
        avg_prompt_tokens = sum(len(r.prompt) // 4 for r in results) / len(results)
        total_per_request = instruction_tokens + avg_prompt_tokens
        instruction_ratio = (
            instruction_tokens / total_per_request if total_per_request > 0 else 0.0
        )
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

        return EvalResult(
            tier=tier,
            quality=quality,
            speed=speed,
            cost=cost,
            reliability=reliability,
            per_suite_quality=per_suite,
            routing_distribution=routing_dist,
            n_questions=len(results),
            details={
                "correct": correct_count,
                "total": len(results),
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
            },
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
            # 2026-05-23 Phase 4 — roll up exogenous-restart counters.
            n_exogenous_recovered=sum(1 for r in results if r.exogenous_recovered),
            n_exogenous_unrecovered=sum(1 for r in results if r.exogenous_unrecovered),
            n_external_restart=sum(1 for r in results if r.external_restart),
            exogenous_question_ids=[
                r.question_id for r in results
                if (r.exogenous_recovered or r.exogenous_unrecovered)
            ],
        )

    def _count_instruction_tokens(self) -> int:
        """AP-16: Count approximate instruction tokens from .md prompt templates.

        Scans orchestration/prompts/*.md for system prompt templates loaded on
        each request. Uses ~4 chars/token heuristic (typical for English text
        with Qwen/Llama tokenizers).
        """
        prompts_dir = _orch_root / "orchestration" / "prompts"
        total_chars = 0
        if prompts_dir.exists():
            for md in prompts_dir.rglob("*.md"):
                total_chars += md.stat().st_size
        return total_chars // 4

    # ── tiered evaluation ────────────────────────────────────────

    def eval_t0(self) -> EvalResult:
        """Tier 0: 10 sentinel questions, binary pass/fail, ~30s."""
        sentinels = self._load_sentinels()
        if not sentinels:
            log.error("No sentinel questions available for T0")
            return EvalResult(tier=0, quality=0, speed=0, cost=0, reliability=0)

        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(sentinels[:10], client, label="T0")
        for r in results:
            log.info(
                "T0 [%s/%s] %s → %s",
                r.suite, r.question_id,
                "PASS" if r.correct else "FAIL",
                r.error or "",
            )

        return self._aggregate(results, tier=0)

    def eval_t1(self, n: int = 100, seed: int = 42) -> EvalResult:
        """Tier 1: 100 stratified questions from benchmark pool, ~5min."""
        pool = self._load_pool()
        if not pool:
            log.error("No question pool available for T1")
            return EvalResult(tier=1, quality=0, speed=0, cost=0, reliability=0)

        # Stratified sampling: equal questions per suite
        suites = list(pool.keys())
        per_suite = max(1, n // len(suites))
        rng = random.Random(seed)
        questions = []
        for suite in suites:
            suite_qs = pool[suite]
            sample = rng.sample(suite_qs, min(per_suite, len(suite_qs)))
            questions.extend(sample)
        rng.shuffle(questions)
        questions = questions[:n]

        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(questions, client, log_every=10, label="T1")

        return self._aggregate(results, tier=1)

    def eval_t2(self, n: int = 500, seed: int = 42) -> EvalResult:
        """Tier 2: 500+ full benchmark, ~30min."""
        pool = self._load_pool()
        if not pool:
            log.error("No question pool available for T2")
            return EvalResult(tier=2, quality=0, speed=0, cost=0, reliability=0)

        suites = list(pool.keys())
        per_suite = max(1, n // len(suites))
        rng = random.Random(seed)
        questions = []
        for suite in suites:
            suite_qs = pool[suite]
            sample = rng.sample(suite_qs, min(per_suite, len(suite_qs)))
            questions.extend(sample)
        rng.shuffle(questions)
        questions = questions[:n]

        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(questions, client, log_every=50, label="T2")

        return self._aggregate(results, tier=2)

    def evaluate(
        self, tier: int = 0, n: int | None = None, seed: int = 42
    ) -> EvalResult:
        """Run evaluation at specified tier."""
        if tier == 0:
            return self.eval_t0()
        elif tier == 1:
            return self.eval_t1(n=n or 100, seed=seed)
        elif tier == 2:
            return self.eval_t2(n=n or 500, seed=seed)
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

    def hybrid_eval(self, seed: int = 42, t1_n: int = 50) -> EvalResult:
        """Hybrid evaluation: T0 as fast pre-filter, T1 as real gate.

        - If T0 fails (quality < 2.5), reject immediately without T1 cost.
        - If T0 passes, run T1 (50 questions, ~2-3min) for real signal.
        - Returns T0 result on fast-reject, T1 result otherwise.
        """
        t0 = self.eval_t0()
        if t0.quality < 2.5:
            log.info("Hybrid eval: T0 failed (q=%.3f), fast-reject", t0.quality)
            return t0

        log.info("Hybrid eval: T0 passed (q=%.3f), running T1 (%d questions)...",
                 t0.quality, t1_n)
        t1 = self.eval_t1(n=t1_n, seed=seed)
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
