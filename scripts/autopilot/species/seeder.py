"""Species 0 — Seeder: 3-way evaluation + Q-value reward injection.

Wraps the seed_specialist_routing.py pipeline as a callable species,
monitoring memory accumulation and Q-value convergence.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("autopilot.seeder")

_orch_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_orch_root / "scripts" / "benchmark"))

from seeding_eval import ThreeWayResult, evaluate_question_3way  # noqa: E402
from seeding_injection import _inject_3way_rewards_http  # noqa: E402

# Import sample_unseen_questions from the main seeding script
from seed_specialist_routing import sample_unseen_questions  # noqa: E402

DEFAULT_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120

# Convergence thresholds
TD_ERROR_EPSILON = 0.05  # Below this, Q-values are "converged"
CONVERGENCE_WINDOW = 5  # N consecutive batches with |TD| < epsilon


@dataclass
class SeederBatchResult:
    n_questions: int = 0
    n_correct: int = 0
    n_errors: int = 0
    rewards_injected: int = 0
    avg_td_error: float = 0.0
    memory_count: int = 0
    elapsed_s: float = 0.0
    per_action_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    results: list[ThreeWayResult] = field(default_factory=list)


class Seeder:
    """Species 0: 3-way seeding with Q-value convergence monitoring."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        timeout: int = DEFAULT_TIMEOUT,
        batch_size: int = 10,
        suites: list[str] | None = None,
        dry_run: bool = False,
    ):
        self.url = url
        self.timeout = timeout
        self.batch_size = batch_size
        self.suites = suites or [
            "coder", "thinking", "math", "general", "simpleqa",
            "hotpotqa", "agentic", "instruction_precision",
        ]
        self.dry_run = dry_run
        self._seen: set[str] = set()
        self._td_errors: list[tuple[int, float]] = []  # (batch_num, avg_td_error)
        self._batch_count = 0
        self._consecutive_converged = 0

    @property
    def td_errors(self) -> list[tuple[int, float]]:
        return list(self._td_errors)

    @property
    def is_converged(self) -> bool:
        return self._consecutive_converged >= CONVERGENCE_WINDOW

    # ── main entry point ─────────────────────────────────────────

    def run_batch(
        self,
        n_questions: int | None = None,
        suites: list[str] | None = None,
        seed: int | None = None,
    ) -> SeederBatchResult:
        """Run a batch of 3-way evaluations and inject rewards."""
        import httpx

        n = n_questions or self.batch_size
        suites = suites or self.suites
        seed = seed if seed is not None else int(time.time()) % 10000

        # Sample questions
        per_suite = max(1, n // len(suites))
        questions = sample_unseen_questions(
            suites=suites,
            sample_per_suite=per_suite,
            seen=self._seen,
            seed=seed,
            use_pool=True,
        )
        if not questions:
            log.warning("No unseen questions available")
            return SeederBatchResult()

        questions = questions[:n]
        log.info("Seeding batch %d: %d questions across %s", self._batch_count, len(questions), suites)

        start = time.time()
        batch_result = SeederBatchResult(n_questions=len(questions))
        td_errors_batch = []

        with httpx.Client(timeout=self.timeout) as client:
            for i, q in enumerate(questions):
                try:
                    role_results, rewards, metadata = evaluate_question_3way(
                        prompt_info=q,
                        url=self.url,
                        timeout=self.timeout,
                        client=client,
                        dry_run=self.dry_run,
                    )

                    # Track per-action stats
                    for action, reward in rewards.items():
                        if action not in batch_result.per_action_stats:
                            batch_result.per_action_stats[action] = {
                                "total": 0, "correct": 0
                            }
                        batch_result.per_action_stats[action]["total"] += 1
                        if reward > 0.5:
                            batch_result.per_action_stats[action]["correct"] += 1
                            batch_result.n_correct += 1

                    # Inject rewards
                    if not self.dry_run:
                        qid = q.get("id", q.get("question_id", f"q_{i}"))
                        suite = q.get("suite", "unknown")
                        injected = _inject_3way_rewards_http(
                            prompt=q.get("prompt", ""),
                            suite=suite,
                            question_id=qid,
                            rewards=rewards,
                            metadata=metadata,
                            url=self.url,
                            client=client,
                        )
                        batch_result.rewards_injected += injected

                    # Track TD error from metadata
                    td = metadata.get("avg_td_error", 0.0)
                    if td > 0:
                        td_errors_batch.append(td)

                    # Mark as seen
                    qid = q.get("id", q.get("question_id", ""))
                    if qid:
                        self._seen.add(qid)

                    # Build ThreeWayResult for logging
                    result = ThreeWayResult(
                        suite=q.get("suite", "unknown"),
                        question_id=qid,
                        prompt=q.get("prompt", ""),
                        expected=q.get("expected", ""),
                        role_results=role_results,
                        rewards=rewards,
                        metadata=metadata,
                        rewards_injected=batch_result.rewards_injected,
                    )
                    batch_result.results.append(result)

                except Exception as e:
                    log.error("Error on question %d: %s", i, e)
                    batch_result.n_errors += 1

                if (i + 1) % 5 == 0:
                    log.info("  Seeding progress: %d/%d", i + 1, len(questions))

        batch_result.elapsed_s = time.time() - start

        # TD error tracking
        if td_errors_batch:
            avg_td = sum(td_errors_batch) / len(td_errors_batch)
        else:
            avg_td = 0.0
        batch_result.avg_td_error = avg_td
        self._td_errors.append((self._batch_count, avg_td))

        # Convergence tracking
        if avg_td < TD_ERROR_EPSILON:
            self._consecutive_converged += 1
        else:
            self._consecutive_converged = 0

        # Memory count
        batch_result.memory_count = self._get_memory_count()

        self._batch_count += 1
        log.info(
            "Seeder batch %d done: %d/%d correct, %d rewards, "
            "TD=%.4f, converged=%d/%d, memories=%d",
            self._batch_count - 1,
            batch_result.n_correct,
            batch_result.n_questions,
            batch_result.rewards_injected,
            avg_td,
            self._consecutive_converged,
            CONVERGENCE_WINDOW,
            batch_result.memory_count,
        )
        return batch_result

    # ── memory monitoring ────────────────────────────────────────

    def _get_memory_count(self) -> int:
        """Get routing memory count from episodic store."""
        try:
            sys.path.insert(
                0, str(_orch_root / "orchestration" / "repl_memory")
            )
            from episodic_store import EpisodicStore
            store = EpisodicStore()
            count = store.count("routing")
            store.close()
            return count
        except Exception as e:
            log.debug("Could not get memory count: %s", e)
            return 0

    def get_memory_count(self) -> int:
        return self._get_memory_count()

    def convergence_status(self) -> dict[str, Any]:
        """Status summary for controller consumption."""
        return {
            "batch_count": self._batch_count,
            "is_converged": self.is_converged,
            "consecutive_converged": self._consecutive_converged,
            "convergence_threshold": CONVERGENCE_WINDOW,
            "last_td_error": self._td_errors[-1][1] if self._td_errors else None,
            "td_epsilon": TD_ERROR_EPSILON,
            "memory_count": self._get_memory_count(),
        }
