"""GEPA integration for PromptForge — evolutionary prompt optimization (AP-19/20).

Bridges GEPA's evolutionary optimizer to the autopilot eval tower, allowing
prompt optimization through the full orchestrator pipeline rather than
isolated LLM calls. Replaces Claude-CLI-based mutation with GEPA's
reflective-mutation + Pareto-selection loop.

Usage within autopilot:
    optimizer = GEPAPromptOptimizer(tower, forge)
    result = optimizer.run(target_file="frontdoor.md", max_evals=50)
    if result:
        mutation = result.to_prompt_mutation()
        # Apply through standard PromptForge safety gates
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("autopilot.gepa")


@dataclass
class GEPAOptResult:
    """Result of a GEPA optimization run."""
    target_file: str
    original_content: str
    best_content: str
    best_score: float
    baseline_score: float
    n_evals: int
    elapsed_s: float
    improvement: float = 0.0
    objective_scores: dict[str, float] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return self.improvement > 0.0

    def to_prompt_mutation(self):
        """Convert to PromptMutation for safety gate compatibility."""
        from .prompt_forge import PromptMutation
        return PromptMutation(
            file=self.target_file,
            mutation_type="gepa",
            description=(
                f"GEPA optimization: {self.baseline_score:.3f} → "
                f"{self.best_score:.3f} ({self.improvement:+.3f}) "
                f"over {self.n_evals} evals in {self.elapsed_s:.0f}s"
            ),
            original_content=self.original_content,
            mutated_content=self.best_content,
        )


class OrchestratorGEPAAdapter:
    """GEPAAdapter that evaluates candidate prompts through the orchestrator API.

    Each evaluation:
    1. Writes candidate prompt text to the target .md file
    2. Runs sentinel questions through the orchestrator API
    3. Scores responses deterministically
    4. Returns scores + execution traces for GEPA's reflective mutation
    """

    def __init__(
        self,
        eval_tower,
        prompt_forge,
        target_file: str = "frontdoor.md",
        component_name: str = "prompt",
    ):
        self.tower = eval_tower
        self.forge = prompt_forge
        self.target_file = target_file
        self.component_name = component_name

    def evaluate(self, batch, candidate, capture_traces=False):
        """Evaluate a candidate prompt on a batch of sentinel questions."""
        from gepa.core.adapter import EvaluationBatch

        # Write candidate prompt to disk (orchestrator reads from disk)
        prompt_text = candidate[self.component_name]
        self.forge.write_prompt(self.target_file, prompt_text)

        scores: list[float] = []
        outputs: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] | None = [] if capture_traces else None

        with httpx.Client(timeout=self.tower.timeout) as client:
            for q in batch:
                r = self.tower._eval_question(q, client)
                score = 1.0 if r.correct else 0.0
                scores.append(score)
                outputs.append({
                    "answer": r.answer,
                    "route": r.route_used,
                    "correct": r.correct,
                })
                if traces is not None:
                    traces.append({
                        "question": q.get("prompt", ""),
                        "expected": q.get("expected", ""),
                        "suite": q.get("suite", "unknown"),
                        "answer": r.answer,
                        "route": r.route_used,
                        "correct": r.correct,
                        "error": r.error,
                        "elapsed_s": r.elapsed_s,
                    })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """Build per-component feedback for GEPA's reflective mutation."""
        dataset: dict[str, list[dict[str, Any]]] = {}
        if eval_batch.trajectories:
            items = []
            for trace, score in zip(eval_batch.trajectories, eval_batch.scores):
                feedback = (
                    f"{'CORRECT' if score > 0 else 'WRONG'}. "
                    f"Route: {trace.get('route', 'unknown')}. "
                    f"Suite: {trace.get('suite', 'unknown')}. "
                )
                if score == 0:
                    feedback += (
                        f"Expected: {trace.get('expected', '?')}, "
                        f"Got: {trace.get('answer', '(empty)')}"
                    )
                    if trace.get("error"):
                        feedback += f". Error: {trace['error']}"

                items.append({
                    "input": trace.get("question", ""),
                    "expected_output": trace.get("expected", ""),
                    "actual_output": trace.get("answer", ""),
                    "score": score,
                    "feedback": feedback,
                })
            for comp in components_to_update:
                dataset[comp] = items
        return dataset

    def propose_new_texts(self, *args, **kwargs):
        """Not used — GEPA handles proposal internally via reflective mutation."""
        raise NotImplementedError("Use GEPA's built-in proposer")


class GEPAPromptOptimizer:
    """High-level optimizer that runs GEPA on a target prompt file.

    Manages the lifecycle: load current prompt → run GEPA → return best
    candidate as a GEPAOptResult (convertible to PromptMutation).
    """

    def __init__(
        self,
        eval_tower,
        prompt_forge,
        reflection_lm: str = "openai/local",
        reflection_lm_url: str = "http://localhost:8082/v1",
    ):
        self.tower = eval_tower
        self.forge = prompt_forge
        self.reflection_lm = reflection_lm
        self.reflection_lm_url = reflection_lm_url

    def run(
        self,
        target_file: str = "frontdoor.md",
        max_evals: int = 50,
        component_name: str = "prompt",
    ) -> GEPAOptResult | None:
        """Run GEPA optimization on a target prompt file.

        Args:
            target_file: Prompt .md file to optimize (relative to prompts_dir).
            max_evals: Maximum number of evaluation calls GEPA can make.
            component_name: Name for the text component in GEPA's candidate dict.

        Returns:
            GEPAOptResult if optimization completed, None on error.
        """
        try:
            import gepa
            from gepa.utils.stop_condition import MaxMetricCallsStopper
        except ImportError:
            log.error("GEPA not installed — pip install gepa")
            return None

        original_content = self.forge.read_prompt(target_file)
        seed_candidate = {component_name: original_content}

        # Load sentinel questions as training data
        sentinels = self.tower._load_sentinels()
        if not sentinels:
            log.error("No sentinel questions available for GEPA optimization")
            return None

        adapter = OrchestratorGEPAAdapter(
            eval_tower=self.tower,
            prompt_forge=self.forge,
            target_file=target_file,
            component_name=component_name,
        )

        # Evaluate baseline first
        log.info("GEPA: evaluating baseline for %s (%d sentinels)", target_file, len(sentinels))
        baseline_batch = adapter.evaluate(sentinels, seed_candidate)
        baseline_score = sum(baseline_batch.scores) / len(baseline_batch.scores) if baseline_batch.scores else 0.0
        log.info("GEPA: baseline score = %.3f", baseline_score)

        start = time.time()
        try:
            # Configure reflection LM via litellm (GEPA uses litellm internally)
            import os
            os.environ.setdefault("OPENAI_API_KEY", "not-needed")

            result = gepa.optimize(
                seed_candidate=seed_candidate,
                trainset=sentinels,
                adapter=adapter,
                reflection_lm=self.reflection_lm,
                max_metric_calls=max_evals,
                stop_callbacks=MaxMetricCallsStopper(max_evals),
                candidate_selection_strategy="pareto",
                frontier_type="instance",
                skip_perfect_score=True,
                display_progress_bar=False,
                raise_on_exception=False,
            )
            elapsed = time.time() - start

        except Exception as e:
            elapsed = time.time() - start
            log.error("GEPA optimization failed after %.0fs: %s", elapsed, e)
            # Restore original prompt
            self.forge.write_prompt(target_file, original_content)
            return None

        # Restore original prompt (GEPA may have left a candidate in place)
        self.forge.write_prompt(target_file, original_content)

        if result is None:
            log.warning("GEPA returned None result")
            return None

        # Extract best candidate
        best_candidate = result.best_candidate
        best_content = best_candidate.get(component_name, original_content)
        best_score = result.best_score if hasattr(result, 'best_score') else 0.0

        # If best_score isn't available, evaluate the best candidate
        if best_score == 0.0 and best_content != original_content:
            eval_batch = adapter.evaluate(sentinels, best_candidate)
            best_score = sum(eval_batch.scores) / len(eval_batch.scores) if eval_batch.scores else 0.0

        improvement = best_score - baseline_score
        n_evals = max_evals  # GEPA doesn't expose actual eval count directly

        log.info(
            "GEPA optimization complete: %.3f → %.3f (%+.3f) in %.0fs (%d evals)",
            baseline_score, best_score, improvement, elapsed, n_evals,
        )

        return GEPAOptResult(
            target_file=target_file,
            original_content=original_content,
            best_content=best_content,
            best_score=best_score,
            baseline_score=baseline_score,
            n_evals=n_evals,
            elapsed_s=elapsed,
            improvement=improvement,
        )
