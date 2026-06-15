"""
QScorer: Async Q-value update agent for episodic memory.

Runs periodically (or on-demand) to:
1. Read progress logs for completed tasks
2. Compute rewards from outcomes
3. Update Q-values in the episodic store
4. Optionally run Claude-as-Judge for graded rewards

This implements the async scoring path from the MemRL architecture,
keeping Q-value computation off the critical inference path.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# DAR-2: Contrastive Q-value updates (decision-aware-routing.md).
# When enabled, routing Q-updates include a contrastive adjustment that
# sharpens decision boundaries between alternative models. The adjustment
# is additive to the reward signal, capped at ±0.1, and zero when the
# ranking is already correct with sufficient margin.
CONTRASTIVE_Q_UPDATES = os.environ.get("CONTRASTIVE_Q_UPDATES", "1") == "1"

# DAR-3: SPO+ with exploration. Replaces the contrastive adjustment (DAR-2)
# with a convex surrogate loss that drives the selected model's Q-value toward
# the ranking that matches true costs. Epsilon-greedy exploration (default 10%)
# is configured in retriever.py. The SPO+ adjustment is computed here when
# counterfactual data (alternative Q-values) is available.
SPO_PLUS_ENABLED = os.environ.get("SPO_PLUS_ENABLED", "0") == "1"
SPO_PLUS_MARGIN = float(os.environ.get("SPO_PLUS_MARGIN", "0.05"))

from .embedder import TaskEmbedder
from .episodic_store import EpisodicStore
from .progress_logger import EventType, ProgressEntry, ProgressLogger, ProgressReader
from .staged_scorer import StagedQScorer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"
)
DEFAULT_MODEL_DESCRIPTOR_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "model_descriptors.yaml"
)
DEFAULT_STACK_PRIORS_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "derived" / "stack_priors.yaml"
)

FALLBACK_BASELINE_TPS_BY_ROLE: Dict[str, float] = {
    "frontdoor": 12.7,
    "coder_escalation": 10.8,
    "architect_general": 4.3,
    "ingest_long_context": 12.0,
    "worker_explore": 50.0,
    "worker_general": 50.0,
    "worker_math": 50.0,
    "toolrunner": 50.0,
    "worker_vision": 15.28,
    "vision_escalation": 27.6,
}

BASELINE_QUALITY_BY_ROLE: Dict[str, float] = {
    "frontdoor": 0.895,
    "coder_escalation": 0.915,
    "architect_general": 0.94,
    "worker_explore": 0.745,
    "worker_math": 0.85,
    "worker_vision": 0.81,
}

FALLBACK_MEMORY_COST_BY_ROLE: Dict[str, float] = {
    "frontdoor": 1.0,
    "coder_escalation": 1.0,
    "architect_general": 1.0,
    "ingest_long_context": 1.0,
    "worker_explore": 1.0,
    "worker_general": 1.0,
    "worker_math": 1.0,
    "toolrunner": 1.0,
    "worker_vision": 1.0,
    "vision_escalation": 1.0,
}

SERVER_MODE_TPS_ROLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "frontdoor": ("frontdoor",),
    "coder_escalation": ("coder_escalation",),
    "architect_general": ("architect_general",),
    "ingest_long_context": ("ingest_long_context",),
    # The production worker server is shared by these scorer roles.
    "worker": ("worker_explore", "worker_general", "worker_math", "toolrunner"),
}

ROLE_PERFORMANCE_TPS_FALLBACKS: Dict[str, str] = {
    "worker_vision": "worker_vision",
    "vision_escalation": "vision_escalation",
}

REGISTRY_MEMORY_ROLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "frontdoor": ("frontdoor",),
    "coder_escalation": ("coder_escalation",),
    "architect_general": ("architect_general",),
    "ingest_long_context": ("ingest_long_context",),
    "worker": ("worker_explore", "worker_general", "worker_math", "toolrunner"),
}

STACK_PRIOR_SCORER_ROLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    # worker_explore remains a q_scorer action label but shares the live worker
    # serving/cost priors compiled under worker_general.
    "worker_general": ("worker_explore",),
}
PRIOR_SOURCE_STACK_PRIORS = "stack_priors"
PRIOR_SOURCE_MODEL_DESCRIPTORS = "model_descriptors"
PRIOR_SOURCE_REGISTRY = "registry"
PRIOR_SOURCE_DEGRADED_FALLBACK = "degraded_fallback"


@dataclass(frozen=True)
class QScorerPriors:
    """Descriptor-aware scorer priors with registry-backed fallbacks."""

    baseline_tps_by_role: Dict[str, float]
    baseline_quality_by_role: Dict[str, float]
    memory_cost_by_role: Dict[str, float]
    baseline_tps_source_by_role: Dict[str, str] = field(default_factory=dict)
    baseline_quality_source_by_role: Dict[str, str] = field(default_factory=dict)
    memory_cost_source_by_role: Dict[str, str] = field(default_factory=dict)
    degraded_reason: str | None = None

    @property
    def uses_degraded_fallback(self) -> bool:
        """Whether any scorer prior value came from a degraded fallback table."""
        sources = (
            *self.baseline_tps_source_by_role.values(),
            *self.baseline_quality_source_by_role.values(),
            *self.memory_cost_source_by_role.values(),
        )
        return any(source == PRIOR_SOURCE_DEGRADED_FALLBACK for source in sources)


def _fallback_q_scorer_priors(*, degraded_reason: str | None = None) -> QScorerPriors:
    return QScorerPriors(
        baseline_tps_by_role=dict(FALLBACK_BASELINE_TPS_BY_ROLE),
        baseline_quality_by_role=dict(BASELINE_QUALITY_BY_ROLE),
        memory_cost_by_role=dict(FALLBACK_MEMORY_COST_BY_ROLE),
        baseline_tps_source_by_role={
            role: PRIOR_SOURCE_DEGRADED_FALLBACK
            for role in FALLBACK_BASELINE_TPS_BY_ROLE
        },
        baseline_quality_source_by_role={
            role: PRIOR_SOURCE_DEGRADED_FALLBACK for role in BASELINE_QUALITY_BY_ROLE
        },
        memory_cost_source_by_role={
            role: PRIOR_SOURCE_DEGRADED_FALLBACK
            for role in FALLBACK_MEMORY_COST_BY_ROLE
        },
        degraded_reason=degraded_reason,
    )


def _valid_quality(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if 0.0 <= parsed <= 1.0 else None


def _coerce_tps(value: Any) -> float | None:
    """Return a numeric t/s value from registry scalars or lower-bound ranges."""
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed > 0 else None
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            parsed = float(match.group(0))
            return parsed if parsed > 0 else None
    return None


def _load_model_descriptors(descriptor_path: Path) -> list[dict[str, Any]]:
    try:
        import yaml

        data = yaml.safe_load(descriptor_path.read_text()) or {}
    except Exception as exc:
        logger.warning("Using registry q_scorer priors; descriptor load failed: %s", exc)
        return []

    models = data.get("models", [])
    if not isinstance(models, list):
        logger.warning(
            "Using registry q_scorer priors; descriptor models field is %s",
            type(models).__name__,
        )
        return []
    return [model for model in models if isinstance(model, dict)]


def _descriptor_has_role_server_conflict(descriptor: dict[str, Any]) -> bool:
    serving = descriptor.get("serving", {})
    if isinstance(serving, dict) and serving.get("numa_policy") == "unresolved_role_server_conflict":
        return True

    gaps = descriptor.get("known_gaps", [])
    if not isinstance(gaps, list):
        return False
    return any("role-server conflict" in str(gap).lower() for gap in gaps)


def _descriptor_speed_tps(descriptor: dict[str, Any]) -> float | None:
    speed = descriptor.get("speed", {})
    if not isinstance(speed, dict):
        return None

    candidates = [
        _coerce_tps(speed.get("solo_96t_tps")),
        _coerce_tps(speed.get("quarter_48t_tps")),
        _coerce_tps(speed.get("prefill_tps")),
        _coerce_tps(speed.get("generation_tps_range")),
    ]
    positive = [candidate for candidate in candidates if candidate is not None]
    return max(positive) if positive else None


def _descriptor_overall_quality(descriptor: dict[str, Any]) -> float | None:
    quality = descriptor.get("quality", {})
    if not isinstance(quality, dict):
        return None
    suite_vector = quality.get("suite_vector", {})
    if not isinstance(suite_vector, dict):
        return None
    return _valid_quality(suite_vector.get("overall"))


def stack_prior_q_scorer_priors_by_role(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> QScorerPriors:
    """Return live q_scorer priors from the generated stack-priors contract.

    Local fallback tables remain available for explicit degraded/offline mode.
    Live scoring should prefer this generated contract so model swaps, role
    retirements, HOT/WARM tier changes, and shared-server bindings are data-only.
    """
    fallback_priors = _fallback_q_scorer_priors()
    tps_by_role = dict(fallback_priors.baseline_tps_by_role)
    quality_by_role = dict(fallback_priors.baseline_quality_by_role)
    memory_by_role = dict(fallback_priors.memory_cost_by_role)
    tps_sources = dict(fallback_priors.baseline_tps_source_by_role)
    quality_sources = dict(fallback_priors.baseline_quality_source_by_role)
    memory_sources = dict(fallback_priors.memory_cost_source_by_role)

    try:
        data = _load_valid_stack_priors(stack_priors_path)
    except Exception as exc:
        logger.warning("Using fallback q_scorer priors; stack-priors load failed: %s", exc)
        return _fallback_q_scorer_priors(
            degraded_reason=f"stack-priors load failed: {exc}"
        )

    live_role_records = _live_stack_q_scorer_role_records(data)
    if live_role_records is None:
        return _fallback_q_scorer_priors(
            degraded_reason="stack-priors roles section is not a mapping"
        )

    for role, record in live_role_records.items():
        priors = record.get("priors", {})
        if not isinstance(priors, dict):
            continue
        target_roles = (role, *STACK_PRIOR_SCORER_ROLE_ALIASES.get(role, ()))
        tps = _coerce_tps(priors.get("throughput_tps"))
        if tps is not None:
            for target_role in target_roles:
                tps_by_role[target_role] = tps
                tps_sources[target_role] = PRIOR_SOURCE_STACK_PRIORS
        quality = _valid_quality(priors.get("quality_overall"))
        if quality is not None:
            for target_role in target_roles:
                quality_by_role[target_role] = quality
                quality_sources[target_role] = PRIOR_SOURCE_STACK_PRIORS
        memory_cost = priors.get("memory_cost")
        if isinstance(memory_cost, (int, float)) and float(memory_cost) > 0:
            for target_role in target_roles:
                memory_by_role[target_role] = float(memory_cost)
                memory_sources[target_role] = PRIOR_SOURCE_STACK_PRIORS

    return QScorerPriors(
        baseline_tps_by_role=tps_by_role,
        baseline_quality_by_role=quality_by_role,
        memory_cost_by_role=memory_by_role,
        baseline_tps_source_by_role=tps_sources,
        baseline_quality_source_by_role=quality_sources,
        memory_cost_source_by_role=memory_sources,
    )


def _load_valid_stack_priors(stack_priors_path: Path) -> dict[str, Any]:
    from src.registry.stack_priors import (
        load_stack_priors_artifact,
        validate_stack_priors_contract,
    )

    data = load_stack_priors_artifact(stack_priors_path)
    if data is None:
        raise ValueError(f"stack-priors artifact unavailable or malformed: {stack_priors_path}")
    contract_errors = validate_stack_priors_contract(data)
    if contract_errors:
        raise ValueError("; ".join(contract_errors[:3]))
    return data


def _live_stack_q_scorer_role_records(
    stack_priors: dict[str, Any],
) -> dict[str, dict[str, Any]] | None:
    roles = stack_priors.get("roles", {})
    if not isinstance(roles, dict):
        return None
    live_roles: dict[str, dict[str, Any]] = {}
    for role, record in roles.items():
        if not isinstance(role, str) or not isinstance(record, dict):
            continue
        if record.get("deployment_status") != "live_stack":
            continue
        live_roles[role] = record
    return live_roles


def _live_stack_q_scorer_roles(stack_priors: dict[str, Any]) -> set[str]:
    live_role_records = _live_stack_q_scorer_role_records(stack_priors)
    if live_role_records is None:
        return set()
    live_roles: set[str] = set()
    for role in live_role_records:
        live_roles.add(role)
        live_roles.update(STACK_PRIOR_SCORER_ROLE_ALIASES.get(role, ()))
    return live_roles


def validate_live_q_scorer_prior_sources(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> list[str]:
    """Return promotion-blocking errors for live q_scorer degraded priors.

    Degraded fallback tables remain valid for offline/replay maintenance. They
    must not silently satisfy live-stack promotion when a generated stack-prior
    artifact is present and structurally valid.
    """
    try:
        stack_priors = _load_valid_stack_priors(stack_priors_path)
    except Exception as exc:
        return [f"q_scorer stack-priors validation failed: {exc}"]

    priors = stack_prior_q_scorer_priors_by_role(stack_priors_path)
    errors: list[str] = []
    if priors.degraded_reason:
        errors.append(f"q_scorer priors are degraded: {priors.degraded_reason}")

    source_maps = (
        ("throughput", priors.baseline_tps_source_by_role),
        ("quality", priors.baseline_quality_source_by_role),
        ("memory_cost", priors.memory_cost_source_by_role),
    )
    for role in sorted(_live_stack_q_scorer_roles(stack_priors)):
        for prior_name, source_by_role in source_maps:
            source = source_by_role.get(role, "<missing>")
            if source != PRIOR_SOURCE_STACK_PRIORS:
                errors.append(
                    f"live q_scorer role {role!r} uses {prior_name} source "
                    f"{source}; expected {PRIOR_SOURCE_STACK_PRIORS}"
                )
    return errors


def descriptor_q_scorer_priors_by_role(
    descriptor_path: Path = DEFAULT_MODEL_DESCRIPTOR_PATH,
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
    stack_priors_path: Path | None = None,
) -> QScorerPriors:
    """Return scorer priors over existing roles, skipping ambiguous descriptors.

    Generated stack priors are the live default contract. Descriptors are used
    only as a gap-fill overlay for roles already present in the live/default
    priors. Role-server conflict records are intentionally ignored, so q_scorer
    does not learn from descriptor rows that disagree with production bindings.
    """
    if stack_priors_path is None and (
        descriptor_path == DEFAULT_MODEL_DESCRIPTOR_PATH
        and registry_path == DEFAULT_MODEL_REGISTRY_PATH
    ):
        stack_priors_path = DEFAULT_STACK_PRIORS_PATH

    if stack_priors_path is not None:
        return stack_prior_q_scorer_priors_by_role(stack_priors_path)

    tps_by_role = registry_baseline_tps_by_role(registry_path)
    quality_by_role = dict(BASELINE_QUALITY_BY_ROLE)
    memory_by_role = registry_memory_cost_by_role(registry_path)
    tps_sources = {role: PRIOR_SOURCE_REGISTRY for role in tps_by_role}
    quality_sources = {
        role: PRIOR_SOURCE_DEGRADED_FALLBACK for role in quality_by_role
    }
    memory_sources = {role: PRIOR_SOURCE_REGISTRY for role in memory_by_role}

    known_tps_roles = set(tps_by_role)
    known_quality_roles = set(quality_by_role)
    for descriptor in _load_model_descriptors(descriptor_path):
        if _descriptor_has_role_server_conflict(descriptor):
            continue

        role_bindings = descriptor.get("role_bindings", {})
        if not isinstance(role_bindings, dict):
            continue
        roles = role_bindings.get("roles", [])
        if not isinstance(roles, list):
            continue

        tps = _descriptor_speed_tps(descriptor)
        quality = _descriptor_overall_quality(descriptor)
        for role in roles:
            if not isinstance(role, str):
                continue
            if tps is not None and role in known_tps_roles:
                tps_by_role[role] = tps
                tps_sources[role] = PRIOR_SOURCE_MODEL_DESCRIPTORS
            if quality is not None and role in known_quality_roles:
                quality_by_role[role] = quality
                quality_sources[role] = PRIOR_SOURCE_MODEL_DESCRIPTORS

    return QScorerPriors(
        baseline_tps_by_role=tps_by_role,
        baseline_quality_by_role=quality_by_role,
        memory_cost_by_role=memory_by_role,
        baseline_tps_source_by_role=tps_sources,
        baseline_quality_source_by_role=quality_sources,
        memory_cost_source_by_role=memory_sources,
    )


def _residency_memory_cost(residency: Any) -> float | None:
    if not isinstance(residency, str):
        return None
    normalized = residency.strip().lower()
    if normalized == "hot":
        return 1.0
    if normalized == "warm":
        return 2.0
    if normalized == "cold":
        return 3.0
    return None


def _performance_tps(role_record: dict[str, Any]) -> float | None:
    perf = role_record.get("performance", {})
    if not isinstance(perf, dict):
        return None
    return _coerce_tps(perf.get("optimized_tps")) or _coerce_tps(perf.get("baseline_tps"))


def registry_baseline_tps_by_role(
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> Dict[str, float]:
    """Load QScorer t/s baselines from the lean model registry.

    The scorer keeps a fallback table so tests, replay tools, and degraded
    maintenance scripts can run without the registry. Live roles prefer the
    deployment `server_mode.*.throughput` values; vision-only roles use
    `roles.*.performance.optimized_tps` because they are not normal text
    server-mode entries.
    """
    baselines = dict(FALLBACK_BASELINE_TPS_BY_ROLE)
    try:
        import yaml

        data = yaml.safe_load(registry_path.read_text()) or {}
    except Exception as exc:
        logger.warning("Using fallback q_scorer TPS baselines; registry load failed: %s", exc)
        return baselines

    server_mode = data.get("server_mode", {})
    if isinstance(server_mode, dict):
        for server_key, target_roles in SERVER_MODE_TPS_ROLE_ALIASES.items():
            record = server_mode.get(server_key, {})
            if not isinstance(record, dict):
                continue
            tps = _coerce_tps(record.get("throughput"))
            if tps is None:
                continue
            for role in target_roles:
                baselines[role] = tps

    roles = data.get("roles", {})
    if isinstance(roles, dict):
        for target_role, registry_role in ROLE_PERFORMANCE_TPS_FALLBACKS.items():
            record = roles.get(registry_role, {})
            if not isinstance(record, dict):
                continue
            tps = _performance_tps(record)
            if tps is not None:
                baselines[target_role] = tps

    return baselines


def registry_memory_cost_by_role(
    registry_path: Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> Dict[str, float]:
    """Load per-role memory-residency costs from the live registry.

    Values are tier costs, not raw model-size costs. HOT roles normalize to
    1.0, so they do not trigger the warm-tier reward penalty. Retired roles
    absent from live server/role records are intentionally not synthesized.
    """
    costs = dict(FALLBACK_MEMORY_COST_BY_ROLE)
    try:
        import yaml

        data = yaml.safe_load(registry_path.read_text()) or {}
    except Exception as exc:
        logger.warning("Using fallback q_scorer memory costs; registry load failed: %s", exc)
        return costs

    roles = data.get("roles", {})
    if isinstance(roles, dict):
        known_roles = set(costs)
        for role, record in roles.items():
            if role not in known_roles or not isinstance(record, dict):
                continue
            memory = record.get("memory", {})
            if not isinstance(memory, dict):
                continue
            cost = _residency_memory_cost(memory.get("residency"))
            if cost is not None:
                costs[role] = cost

    server_mode = data.get("server_mode", {})
    if isinstance(server_mode, dict):
        for server_key, target_roles in REGISTRY_MEMORY_ROLE_ALIASES.items():
            record = server_mode.get(server_key, {})
            if not isinstance(record, dict):
                continue
            cost = _residency_memory_cost(record.get("tier"))
            if cost is None:
                continue
            for role in target_roles:
                costs[role] = cost

    return costs


@dataclass
class ScoringConfig:
    """Configuration for Q-scoring."""

    # Learning rate for Q-value updates
    learning_rate: float = 0.1

    # Reward values
    success_reward: float = 1.0
    failure_reward: float = -0.5
    partial_reward: float = 0.3

    # Temporal decay: Q-values decay toward neutral (0.5) over time.
    # decay_rate ^ days_elapsed is applied before each TD update.
    temporal_decay_rate: float = 0.99

    # Claude-as-Judge settings (optional)
    use_claude_judge: bool = False
    judge_model_path: Optional[Path] = None
    judge_binary: Optional[Path] = None

    # Cost-aware reward (xRouter-style correctness-gated cost penalty).
    # reward_final = quality_reward - lambda * max(0, cost_ratio - 1.0)
    # where cost_ratio = actual_elapsed / expected_elapsed.
    # Only applied when answer is correct (incorrect = 0.0, no cost term).
    cost_penalty_lambda: float = 0.15

    # Per-role optimized tokens/second from generated stack priors at config
    # construction time, with fallback tables for degraded/offline scripts. Used
    # to normalize cost:
    # expected_elapsed = tokens_generated / baseline_tps.
    baseline_tps_by_role: Dict[str, float] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().baseline_tps_by_role
    )
    baseline_tps_source_by_role: Dict[str, str] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().baseline_tps_source_by_role
    )

    # Per-role quality baselines from generated stack priors, with legacy
    # benchmark fallbacks for roles whose structured quality prior is still a gap.
    # Used for quality-gap penalty: penalize using expensive model when cheap suffices.
    baseline_quality_by_role: Dict[str, float] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().baseline_quality_by_role
    )
    baseline_quality_source_by_role: Dict[str, str] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().baseline_quality_source_by_role
    )

    # Per-role memory residency cost. HOT roles normalize to 1.0 and incur no
    # warm-tier penalty; non-HOT roles load from the registry when present.
    memory_cost_by_role: Dict[str, float] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().memory_cost_by_role
    )
    memory_cost_source_by_role: Dict[str, str] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().memory_cost_source_by_role
    )
    prior_degraded_reason: Optional[str] = field(
        default_factory=lambda: descriptor_q_scorer_priors_by_role().degraded_reason
    )

    # Multi-dimensional cost weights (tunable).
    # cost_lambda_latency is the existing cost_penalty_lambda.
    cost_lambda_quality_gap: float = 0.10  # Penalize using higher-quality model than needed
    cost_lambda_memory: float = 0.05       # Penalize non-HOT residency when HOT sufficient

    # Delegation/teacher attribution shaping.
    delegation_misattribution_penalty: float = 0.10
    specialist_credit_bonus: float = 0.05
    teacher_regret_penalty: float = 0.20
    teacher_speedup_bonus: float = 0.05

    # Scoring frequency
    min_score_interval_seconds: int = 300  # 5 minutes

    # Batch size for processing
    batch_size: int = 50


class QScorer:
    """
    Async Q-value scoring agent.

    Workflow:
    1. Read progress logs for completed tasks
    2. For each task:
       a. Find associated memory entries
       b. Compute reward from outcome
       c. Update Q-values
    3. Log scoring events
    """

    def __init__(
        self,
        store: EpisodicStore,
        embedder: TaskEmbedder,
        logger: ProgressLogger,
        reader: ProgressReader,
        config: Optional[ScoringConfig] = None,
        staged_scorer: Optional[StagedQScorer] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.logger = logger
        self.reader = reader
        self.config = config or ScoringConfig()
        self.staged_scorer = staged_scorer
        self._last_score_time: Optional[datetime] = None

    def score_pending_tasks(self) -> Dict[str, Any]:
        """
        Score all pending tasks from progress logs.

        Returns:
            Summary of scoring results
        """
        # Check minimum interval
        now = datetime.now(timezone.utc)
        if self._last_score_time:
            elapsed = (now - self._last_score_time).total_seconds()
            if elapsed < self.config.min_score_interval_seconds:
                return {
                    "skipped": True,
                    "reason": f"Too soon ({elapsed:.0f}s < {self.config.min_score_interval_seconds}s)",
                }

        # Find unscored tasks
        unscored_task_ids = self.reader.get_unscored_tasks()

        if not unscored_task_ids:
            return {"tasks_processed": 0, "message": "No pending tasks to score"}

        # Process in batches
        results = {
            "tasks_processed": 0,
            "memories_updated": 0,
            "memories_created": 0,
            "errors": [],
        }

        for task_id in unscored_task_ids[: self.config.batch_size]:
            try:
                task_result = self._score_task(task_id)
                results["tasks_processed"] += 1
                results["memories_updated"] += task_result.get("memories_updated", 0)
                results["memories_created"] += task_result.get("memories_created", 0)
            except Exception as e:
                results["errors"].append({"task_id": task_id, "error": str(e)})

        self._last_score_time = now
        self.logger.flush()

        return results

    def _score_task(self, task_id: str) -> Dict[str, Any]:
        """
        Score a single task.

        Args:
            task_id: Task ID to score

        Returns:
            Scoring results for this task
        """
        # Get task trajectory
        trajectory = self.reader.get_task_trajectory(task_id)

        if not trajectory:
            return {"error": "No trajectory found"}

        # Extract key events
        task_started = None
        routing_decision = None
        task_outcome = None
        gate_results = []
        escalations = []
        plan_reviews = []

        for entry in trajectory:
            if entry.event_type == EventType.TASK_STARTED:
                task_started = entry
            elif entry.event_type == EventType.ROUTING_DECISION:
                routing_decision = entry
            elif entry.event_type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
                task_outcome = entry
            elif entry.event_type in (EventType.GATE_PASSED, EventType.GATE_FAILED):
                gate_results.append(entry)
            elif entry.event_type == EventType.ESCALATION_TRIGGERED:
                escalations.append(entry)
            elif entry.event_type == EventType.PLAN_REVIEWED:
                plan_reviews.append(entry)

        if not task_outcome:
            return {"error": "Task not completed yet"}

        # Compute reward (pass completion data as optional cost/telemetry metrics)
        reward = self._compute_reward(
            task_outcome,
            gate_results,
            escalations,
            plan_reviews,
            cost_metrics=(task_outcome.data if task_outcome and task_outcome.data else None),
        )
        # Delegation credit assignment to avoid over-crediting envelope roles.
        reward = self._apply_delegation_credit(reward, routing_decision, task_outcome)

        # Apply staged reward shaping if enabled (explore early, exploit later)
        if self.staged_scorer is not None:
            task_type = ""
            if task_started and task_started.data:
                task_type = task_started.data.get("task_type", "")
            action_str = ""
            if routing_decision and routing_decision.data:
                routing = routing_decision.data.get("routing", [])
                action_str = ",".join(routing) if isinstance(routing, list) else str(routing)
            if action_str and task_type:
                reward = self.staged_scorer.compute_staged_reward(
                    reward, action_str, task_type, self.store,
                )

        # DAR-2/DAR-3: Reward adjustment (feature-flagged).
        # SPO_PLUS_ENABLED (DAR-3) supersedes CONTRASTIVE_Q_UPDATES (DAR-2) when both on.
        ranking_adj = 0.0
        ranking_source = "none"
        if routing_decision and task_started:
            if SPO_PLUS_ENABLED:
                ranking_adj = self._compute_spo_plus_adjustment(
                    task_started, routing_decision, reward,
                    margin=SPO_PLUS_MARGIN,
                )
                ranking_source = "spo_plus"
            elif CONTRASTIVE_Q_UPDATES:
                ranking_adj = self._compute_contrastive_adjustment(
                    task_started, routing_decision, reward,
                )
                ranking_source = "contrastive"

            if abs(ranking_adj) > 0.001:
                logger.info(
                    "DAR %s: adj=%.4f reward=%.3f→%.3f task=%s",
                    ranking_source, ranking_adj, reward, reward + ranking_adj, task_id,
                )

        reward_for_update = max(-1.0, min(1.0, reward + ranking_adj))

        result = {
            "memories_updated": 0,
            "memories_created": 0,
            "reward": reward,
            "contrastive_adj": ranking_adj,
            "ranking_source": ranking_source,
        }

        # Update or create routing memory (uses contrastive-adjusted reward)
        if routing_decision:
            memory_result = self._update_routing_memory(
                task_id,
                task_started,
                routing_decision,
                reward_for_update,
            )
            result.update(memory_result)

        # Update escalation memories (use base reward, not contrastive-adjusted)
        for escalation in escalations:
            esc_result = self._update_escalation_memory(task_id, escalation, reward)
            result["memories_updated"] += esc_result.get("memories_updated", 0)
            result["memories_created"] += esc_result.get("memories_created", 0)

        return result

    def _apply_delegation_credit(
        self,
        reward: float,
        routing_decision: Optional[ProgressEntry],
        task_outcome: ProgressEntry,
    ) -> float:
        """Adjust reward for delegation lineage attribution.

        Penalize envelope over-credit (architect routed but specialist produced final answer).
        Slightly bonus direct specialist attribution when selected specialist finished task.
        """
        if routing_decision is None:
            return reward
        routing = routing_decision.data.get("routing", [])
        if isinstance(routing, str):
            routed_roles = [r.strip() for r in routing.split(",") if r.strip()]
        elif isinstance(routing, list):
            routed_roles = [str(r) for r in routing]
        else:
            routed_roles = []
        final_role = str(task_outcome.data.get("final_answer_role", "") or "")
        if not routed_roles or not final_role:
            return reward

        routed_architect = any(r.startswith("architect_") for r in routed_roles)
        final_is_architect = final_role.startswith("architect_")

        if routed_architect and not final_is_architect:
            reward -= self.config.delegation_misattribution_penalty
        elif final_role in routed_roles and not final_is_architect:
            reward += self.config.specialist_credit_bonus

        return max(-1.0, min(1.0, reward))

    def _compute_reward(
        self,
        task_outcome: ProgressEntry,
        gate_results: List[ProgressEntry],
        escalations: List[ProgressEntry],
        plan_reviews: List[ProgressEntry] | None = None,
        cost_metrics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute reward — delegates to q_reward.compute_reward (Task-G refactor)."""
        return _compute_reward_impl(
            task_outcome, gate_results, escalations, plan_reviews, cost_metrics,
            config=self.config,
        )

    def _compute_contrastive_adjustment(
        self,
        task_started: ProgressEntry,
        routing_decision: ProgressEntry,
        reward: float,
        margin: float = 0.05,
        max_adj: float = 0.1,
    ) -> float:
        """DAR-2: Compute contrastive ranking adjustment for Q-value update.

        Sharpens decision boundaries between alternative models:
        - Success: if selected model's Q is below alternatives, boost reward
          so the TD update pushes its Q above competitors.
        - Failure: if selected model's Q is above alternatives, penalize more
          so the TD update pushes its Q below competitors.

        The adjustment is zero when:
        - The ranking is already correct with sufficient margin
        - No alternative memories with learned Q-values exist
        - The feature flag CONTRASTIVE_Q_UPDATES is off (checked by caller)

        Bounded to [-max_adj, +max_adj] to prevent runaway drift.
        With α=0.1 and max_adj=0.1, the maximum extra Q-shift per update
        is 0.01 — negligible individually, significant cumulatively.
        """
        task_context = task_started.data if task_started and task_started.data else {}
        if not task_context:
            return 0.0

        # Generate embedding for this task
        try:
            embedding = self.embedder.embed_task_ir(task_context)
        except Exception:
            return 0.0

        # Retrieve similar routing memories
        candidates = self.store.retrieve_by_similarity(
            embedding, k=10, action_type="routing",
        )
        if not candidates:
            return 0.0

        # Identify the selected action
        routing = routing_decision.data.get("routing", [])
        selected_action = routing[0] if isinstance(routing, list) and routing else str(routing)

        # Get selected memory's current Q-value
        selected_q = 0.5
        memory_id = routing_decision.memory_id
        if memory_id:
            mem = self.store.get_by_id(memory_id)
            if mem:
                selected_q = mem.q_value

        # Collect alternative Q-values (different actions, skip unlearned defaults)
        alt_q_values = []
        for c in candidates:
            if c.action != selected_action and abs(c.q_value - 0.5) > 0.001:
                alt_q_values.append(c.q_value)

        if not alt_q_values:
            return 0.0

        if reward > 0:
            # Success: push selected Q above the best alternative
            max_alt_q = max(alt_q_values)
            gap = max_alt_q + margin - selected_q
            if gap > 0:
                return min(max_adj, margin * gap)
        else:
            # Failure: push selected Q below the worst alternative
            min_alt_q = min(alt_q_values)
            gap = selected_q + margin - min_alt_q
            if gap > 0:
                return max(-max_adj, -margin * gap)

        return 0.0

    def _compute_spo_plus_adjustment(
        self,
        task_started: ProgressEntry,
        routing_decision: ProgressEntry,
        reward: float,
        margin: float = 0.05,
        max_adj: float = 0.15,
    ) -> float:
        """DAR-3: SPO+ convex surrogate adjustment for Q-value update.

        The SPO+ loss drives the predicted cost ranking toward the true cost
        ranking. Unlike DAR-2's contrastive adjustment which only compares
        selected vs best/worst alternative, SPO+ considers all alternatives:

            L_SPO+ = sum_j max(0, 2*q_hat[j] - q_true[j]) - q_hat[i*] + q_true[i*]

        where q_hat = predicted Q-values, q_true = true Q-values (estimated
        from reward), and i* = true optimal model.

        The adjustment is bounded to [-max_adj, +max_adj] to prevent
        runaway drift. When no counterfactual data is available, returns 0.0.

        Args:
            task_started: Task context entry.
            routing_decision: Routing decision entry.
            reward: Observed reward for the selected model.
            margin: Minimum gap for adjustment activation.
            max_adj: Maximum absolute adjustment.

        Returns:
            Reward adjustment in [-max_adj, +max_adj].
        """
        task_context = task_started.data if task_started and task_started.data else {}
        if not task_context:
            return 0.0

        try:
            embedding = self.embedder.embed_task_ir(task_context)
        except Exception:
            return 0.0

        candidates = self.store.retrieve_by_similarity(
            embedding, k=10, action_type="routing",
        )
        if not candidates:
            return 0.0

        # Identify selected action and its Q-value
        routing = routing_decision.data.get("routing", [])
        selected_action = routing[0] if isinstance(routing, list) and routing else str(routing)

        selected_q = 0.5
        memory_id = routing_decision.memory_id
        if memory_id:
            mem = self.store.get_by_id(memory_id)
            if mem:
                selected_q = mem.q_value

        # Collect all alternative actions with learned Q-values
        alt_actions: dict[str, float] = {}
        for c in candidates:
            if c.action != selected_action and abs(c.q_value - 0.5) > 0.001:
                # Keep the best Q-value per distinct action
                if c.action not in alt_actions or c.q_value > alt_actions[c.action]:
                    alt_actions[c.action] = c.q_value

        if not alt_actions:
            return 0.0

        # True Q-value estimate for the selected model (from observed reward)
        # Map reward to [0,1] range like initial_q does
        true_q_selected = 0.5 + (reward * 0.5)

        # SPO+ surrogate: penalize if predicted ranking disagrees with true ranking
        # For each alternative, compute the surrogate loss term
        spo_sum = 0.0
        for alt_action, alt_q_hat in alt_actions.items():
            # We don't have the true Q for alternatives (counterfactual),
            # so we use their current Q-values as the "predicted" costs
            # and the selected model's observed reward as the ground truth.
            # SPO+ term: max(0, 2*q_hat_alt - true_q_alt_estimate)
            # Since we lack true_q_alt, we use the conservative estimate:
            # If selected succeeded, alternatives would have done no better than their Q
            # If selected failed, alternatives might have done better
            spo_term = max(0.0, 2.0 * alt_q_hat - alt_q_hat)  # = alt_q_hat when positive
            spo_sum += spo_term

        # Selected model's contribution: -q_hat_selected + true_q_selected
        spo_loss = spo_sum - selected_q + true_q_selected

        # Convert loss to adjustment: positive loss → boost selected, negative → penalize
        if abs(spo_loss) < margin:
            return 0.0

        adjustment = margin * spo_loss
        return max(-max_adj, min(max_adj, adjustment))

    def _update_routing_memory(
        self,
        task_id: str,
        task_started: Optional[ProgressEntry],
        routing_decision: ProgressEntry,
        reward: float,
    ) -> Dict[str, Any]:
        """Update or create routing memory."""
        result = {"memories_updated": 0, "memories_created": 0}

        # Check if memory already exists
        memory_id = routing_decision.memory_id

        if memory_id:
            # Update existing memory
            memory = self.store.get_by_id(memory_id)
            if memory:
                old_q = memory.q_value
                new_q = self.store.update_q_value(
                    memory_id, reward, self.config.learning_rate,
                    temporal_decay_rate=self.config.temporal_decay_rate,
                )
                self.logger.log_memory_update(memory_id, old_q, new_q, reward, task_id)
                result["memories_updated"] = 1
        else:
            # Create new memory from this routing decision
            if task_started and task_started.data:
                task_context = {
                    "task_type": task_started.data.get("task_type"),
                    "objective": task_started.data.get("objective"),
                    "priority": task_started.data.get("priority"),
                }

                # Generate embedding for task context
                embedding = self.embedder.embed_task_ir(task_context)

                # Store new memory
                routing = routing_decision.data.get("routing", [])
                action = ",".join(routing) if isinstance(routing, list) else str(routing)

                # Initial Q-value based on first observation
                initial_q = 0.5 + (reward * 0.5)  # Map reward to [0, 1]

                memory_id = self.store.store(
                    embedding=embedding,
                    action=action,
                    action_type="routing",
                    context=task_context,
                    outcome="success" if reward > 0 else "failure",
                    initial_q=initial_q,
                )

                self.logger.log(
                    ProgressEntry(
                        event_type=EventType.MEMORY_STORED,
                        task_id=task_id,
                        memory_id=memory_id,
                        data={"action_type": "routing", "initial_q": initial_q},
                    )
                )
                result["memories_created"] = 1

        return result

    def _update_escalation_memory(
        self,
        task_id: str,
        escalation: ProgressEntry,
        reward: float,
    ) -> Dict[str, Any]:
        """Update or create escalation memory."""
        result = {"memories_updated": 0, "memories_created": 0}

        memory_id = escalation.memory_id

        if memory_id:
            # Update existing memory
            memory = self.store.get_by_id(memory_id)
            if memory:
                old_q = memory.q_value
                new_q = self.store.update_q_value(
                    memory_id, reward, self.config.learning_rate,
                    temporal_decay_rate=self.config.temporal_decay_rate,
                )
                self.logger.log_memory_update(memory_id, old_q, new_q, reward, task_id)
                result["memories_updated"] = 1
        else:
            # Create new escalation memory
            failure_context = {
                "from_tier": escalation.data.get("from_tier"),
                "to_tier": escalation.data.get("to_tier"),
                "reason": escalation.data.get("reason"),
            }

            embedding = self.embedder.embed_failure_context(failure_context)
            action = f"escalate:{escalation.data.get('from_tier')}->{escalation.data.get('to_tier')}"

            initial_q = 0.5 + (reward * 0.5)

            memory_id = self.store.store(
                embedding=embedding,
                action=action,
                action_type="escalation",
                context=failure_context,
                outcome="success" if reward > 0 else "failure",
                initial_q=initial_q,
            )

            self.logger.log(
                ProgressEntry(
                    event_type=EventType.MEMORY_STORED,
                    task_id=task_id,
                    memory_id=memory_id,
                    data={"action_type": "escalation", "initial_q": initial_q},
                )
            )
            result["memories_created"] = 1

        return result

    def score_external_result(
        self,
        task_description: str,
        action: str,
        reward: float,
        context: Dict[str, Any] | None = None,
        embedding: List[float] | None = None,
    ) -> Dict[str, Any]:
        """Score an externally-evaluated result.

        Accepts pre-computed rewards from external scoring (e.g., the MemRL
        learning loop or debug scorer). Bypasses progress log reader and
        directly creates/updates episodic memory.

        Args:
            task_description: Description of the task.
            action: The action taken (e.g., "frontdoor:direct").
            reward: Pre-computed reward (-1.0 to 1.0).
            context: Additional context to store with the memory.
            embedding: Precomputed embedding for task_description (avoids re-embedding).

        Returns:
            Dict with memories_created and memories_updated counts.
        """
        result = {"memories_updated": 0, "memories_created": 0}
        context = context or {}

        # Clamp reward to valid range
        reward = max(-1.0, min(1.0, reward))

        # Use precomputed embedding or compute it
        if embedding is not None:
            emb_array = np.array(embedding, dtype=np.float32)
        else:
            task_ir = {
                "task_type": context.get("task_type", "chat"),
                "objective": task_description[:200],
            }
            emb_array = self.embedder.embed_task_ir(task_ir)

        # Search for existing similar memory with same action
        # Note: retrieve_by_similarity returns memories sorted by similarity
        similar = self.store.retrieve_by_similarity(
            query_embedding=emb_array,
            k=5,
            action_type="routing",
        )
        # Filter to high-similarity matches (similarity_score >= 0.85)
        similar = [m for m in similar if m.similarity_score >= 0.85]

        # Update existing memory if action matches closely
        updated = False
        for mem in similar:
            if mem.action == action or (
                hasattr(mem, "action") and mem.action.startswith(action.split(":")[0])
            ):
                old_q = mem.q_value
                new_q = self.store.update_q_value(
                    mem.id, reward, self.config.learning_rate,
                    temporal_decay_rate=self.config.temporal_decay_rate,
                )
                self.logger.log_memory_update(
                    mem.id, old_q, new_q, reward, "external"
                )
                result["memories_updated"] += 1
                updated = True
                break

        # Create new memory if no similar one found
        if not updated:
            initial_q = 0.5 + (reward * 0.5)
            context["task_description"] = task_description
            context["source"] = "external"

            memory_id = self.store.store(
                embedding=emb_array,
                action=action,
                action_type="routing",
                context=context,
                outcome="success" if reward > 0 else "failure",
                initial_q=initial_q,
            )

            self.logger.log(
                ProgressEntry(
                    event_type=EventType.MEMORY_STORED,
                    task_id="external",
                    memory_id=memory_id,
                    data={
                        "action_type": "routing",
                        "initial_q": initial_q,
                        "source": "external_score",
                    },
                )
            )
            result["memories_created"] = 1

        return result




# ClaudeAsJudge moved to q_judge.py (2026-05-22 Task-G refactor).
# Re-exported so existing imports keep working.
from .q_judge import ClaudeAsJudge  # noqa: E402,F401

# Pure reward computation moved to q_reward.py (2026-05-22 Task-G refactor).
from .q_reward import compute_reward as _compute_reward_impl  # noqa: E402
