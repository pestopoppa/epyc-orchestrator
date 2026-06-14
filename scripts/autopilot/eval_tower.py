"""Tiered evaluation tower: T0 (10q/30s) → T1 (100q/5m) → T2 (500+/30m).

Wraps existing seeding infrastructure for orchestrator API calls and scoring.
Training set (debug suites) is kept separate from validation set (HF benchmarks).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
_EXPECTED_FREE_SCORERS = {"programmatic"}
_CORE_METADATA_KEY = "__core_metadata__"


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


def _is_scoreable_question(q: dict) -> bool:
    expected = q.get("expected", "")
    scoring_method = str(q.get("scoring_method", "exact_match"))
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
) -> list[dict]:
    if not pool or n <= 0:
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
# Repo root on path so `src.tools.eval_secret` (runtime tool-secret ground truth)
# imports from the autopilot harness, not just inside the orchestrator process.
sys.path.insert(0, str(_orch_root))

from seeding_orchestrator import call_orchestrator_forced  # noqa: E402
from seeding_scoring import score_answer_deterministic  # noqa: E402

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
        self._core_cache: dict[str, tuple[list[dict], dict[str, Any], Path]] = {}
        self._trial_id_context: int | None = None
        self.on_question = on_question

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

    def _eval_question(
        self, q: dict, client: httpx.Client
    ) -> QuestionResult:
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
            resp = call_orchestrator_forced(
                prompt=prompt,
                # Let routing decide unless the question pins a mode/role. The
                # tool_use suite pins force_mode="repl" so the REPL CALL(...)
                # path (what production actually uses) is exercised
                # deterministically instead of being left to the router.
                # Defaults are "" → existing questions are unchanged.
                force_role=q.get("force_role", ""),
                force_mode=q.get("force_mode", ""),
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
            if not error and _is_scoreable_question(q):
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
        task_rate_qph = (len(results) / (eval_wall_s / 3600.0)) if eval_wall_s > 0 else 0.0
        goodput_qph = (quality / 3.0) * task_rate_qph
        tokens_per_solved_task = (
            total_tokens_generated / correct_count if correct_count > 0 else 0.0
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
        # Per-suite question counts (2026-06-06). The per-suite regression gate is
        # otherwise blind to sample size: on a hybrid eval each suite draws only
        # ~2 questions, so the score is quantized to {0.0, 1.5, 3.0} and a single
        # correct→incorrect flip is a -1.5 swing — 15× the fixed -0.1 gate, tripping
        # it on essentially every trial. Carrying the count lets the gate make the
        # threshold resolution-aware (3/n single-flip quantum) instead of false-
        # positiving the seeder loop into a critic-reject deadlock.
        per_suite_counts = {suite: len(vals) for suite, vals in suite_correct.items()}
        question_results = [
            {
                "qid": r.qid or _stable_question_qid(str(r.suite), str(r.prompt)),
                "suite": r.suite,
                "partition": r.eval_partition or "core",
                "correct": bool(r.correct),
                "latency_ms": int(round(max(0.0, r.elapsed_s) * 1000)),
                "tools_used": int(r.tools_used or 0),
            }
            for r in results
        ]
        partition_correct: dict[str, list[bool]] = {}
        partition_suite_correct: dict[str, dict[str, list[bool]]] = {}
        for r in results:
            partition = r.eval_partition or "core"
            partition_correct.setdefault(partition, []).append(r.correct)
            partition_suite_correct.setdefault(partition, {}).setdefault(
                r.suite, []
            ).append(r.correct)
        partition_quality = {
            partition: (sum(vals) / len(vals)) * 3.0
            for partition, vals in partition_correct.items()
        }
        partition_counts = {
            partition: len(vals) for partition, vals in partition_correct.items()
        }
        partition_suite_quality = {
            partition: {
                suite: (sum(vals) / len(vals)) * 3.0
                for suite, vals in suites.items()
            }
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
            for name in (r.tools_called or []):
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
            tool_helpfulness = (
                sum(per_suite_tool_helpfulness.values()) / len(per_suite_tool_helpfulness)
            )
        else:
            tool_helpfulness = float("nan")

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
                r.suite, r.question_id,
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
        core_selection = "legacy_pool_seed"
        resolved_trial_id = self._resolve_trial_id(trial_id)
        audit_policy: dict[str, Any] = {
            "enabled": os.environ.get("AUTOPILOT_W6_AUDIT_BLOCK") == "1",
            "requested_n": max(0, _env_int("AUTOPILOT_W6_AUDIT_N", 10)),
            "every_n_trials": max(
                1, _env_int("AUTOPILOT_W6_AUDIT_EVERY_N_TRIALS", 1)
            ),
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

        result = self._aggregate(results, tier=1)
        result.core_id = core_id
        result.details.update(
            {
                "core_id": core_id,
                "core_selection": core_selection,
                "core_path": core_path,
                "core_metadata": core_metadata,
                "requested_n": n,
                "base_core_questions": base_core_questions,
                "base_audit_questions": base_audit_questions,
                "audit_policy": audit_policy,
            }
        )
        return result

    def eval_t2(self, n: int = 500, seed: int = 42) -> EvalResult:
        """Tier 2: 500+ full benchmark, ~30min."""
        pool = self._load_pool()
        if not pool:
            log.error("No question pool available for T2")
            return EvalResult(tier=2, quality=0, speed=0, cost=0, reliability=0)

        rng = random.Random(seed)
        questions = _sample_scoreable_eval_questions(pool, n, rng)
        # Tool-use sentinels also join T2 (the journaled deep eval) for the same
        # reason as T1. Inert ([]) unless AUTOPILOT_TOOL_SENTINELS=1.
        questions = (
            _annotate_partition(questions, "core")
            + _annotate_partition(self._load_tool_sentinels(), "tool_sentinel")
        )

        with httpx.Client(timeout=self.timeout) as client:
            results = self._eval_batch(questions, client, log_every=50, label="T2")

        return self._aggregate(results, tier=2)

    def evaluate(
        self,
        tier: int = 0,
        n: int | None = None,
        seed: int = 42,
        trial_id: int | None = None,
    ) -> EvalResult:
        """Run evaluation at specified tier."""
        if tier == 0:
            return self.eval_t0()
        elif tier == 1:
            if trial_id is None:
                return self.eval_t1(n=n or 100, seed=seed)
            return self.eval_t1(n=n or 100, seed=seed, trial_id=trial_id)
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

        log.info("Hybrid eval: T0 passed (q=%.3f), running T1 (%d questions)...",
                 t0.quality, t1_n)
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
