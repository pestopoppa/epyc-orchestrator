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

import contextlib
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("autopilot.gepa")
DEFAULT_SCRATCH_BASE = Path(__file__).resolve().parents[3] / "tmp" / "gepa_prompt_roots"


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
    1. Copies the prompt tree into a scratch prompt root
    2. Writes candidate prompt text only inside that scratch root
    3. Runs sentinel questions through the orchestrator API with that root
       attached to each eval request
    4. Scores responses deterministically
    5. Returns scores + execution traces for GEPA's reflective mutation
    """

    def __init__(
        self,
        eval_tower,
        prompt_forge,
        target_file: str = "frontdoor.md",
        component_name: str = "prompt",
        scratch_base: str | Path | None = None,
    ):
        self.tower = eval_tower
        self.forge = prompt_forge
        self.target_file = target_file
        self.component_name = component_name
        self.scratch_base = Path(scratch_base) if scratch_base is not None else DEFAULT_SCRATCH_BASE

    @contextlib.contextmanager
    def _candidate_prompt_root(self, prompt_text: str):
        source_root = Path(self.forge.prompts_dir).resolve(strict=True)
        self.scratch_base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gepa-", dir=self.scratch_base) as temp_dir:
            scratch_root = Path(temp_dir).resolve(strict=True)
            shutil.copytree(
                source_root,
                scratch_root,
                dirs_exist_ok=True,
                symlinks=False,
                ignore_dangling_symlinks=True,
            )
            from .prompt_forge import PromptForge

            scratch_forge = PromptForge(
                prompts_dir=scratch_root,
                timeout=getattr(self.forge, "timeout", 300),
                auto_commit=False,
            )
            scratch_forge.write_prompt(self.target_file, prompt_text)
            yield scratch_root

    def evaluate(self, batch, candidate, capture_traces=False):
        """Evaluate a candidate prompt on a batch of sentinel questions."""
        from gepa.core.adapter import EvaluationBatch

        prompt_text = candidate[self.component_name]

        scores: list[float] = []
        outputs: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] | None = [] if capture_traces else None

        with self._candidate_prompt_root(prompt_text) as prompt_root:
            # Fan-out across frontdoor quarters via EvalTower._eval_batch
            # (AUTOPILOT_EVAL_CONCURRENCY, default 4). Results preserve input
            # order, so the zip with `batch` for trace building is correct.
            isolated_batch = [
                {**q, "_prompt_root": str(prompt_root)}
                for q in batch
            ]
            with httpx.Client(timeout=self.tower.timeout) as client:
                results = self.tower._eval_batch(isolated_batch, client, label="GEPA")

            for q, r in zip(isolated_batch, results):
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

    # GEPA's reflective mutation dispatches on PRESENCE, not None-ness:
    # `if self.adapter.propose_new_texts is not None` (reflective_mutation.py:66).
    # A bound method is never None, so defining this as a method that raises made
    # every reflection step fail before any LM call — GEPA's built-in proposer was
    # never reached. Declare it None (matching GEPAAdapter's protocol default at
    # gepa/core/adapter.py:180) so the built-in proposer runs. Do NOT delete the
    # attribute: this class does not inherit the protocol, so a missing attribute
    # would turn the same check into an AttributeError.
    propose_new_texts = None


class GEPAPromptOptimizer:
    """High-level optimizer that runs GEPA on a target prompt file.

    Manages the lifecycle: load current prompt → run GEPA → return best
    candidate as a GEPAOptResult (convertible to PromptMutation).
    """

    def __init__(
        self,
        eval_tower,
        prompt_forge,
        reflection_lm: str | None = None,
        reflection_lm_url: str | None = None,
    ):
        import os

        self.tower = eval_tower
        self.forge = prompt_forge
        # Env-overridable so the reflection endpoint can follow the live stack without
        # touching call sites (prompt_forge constructs this with defaults).
        self.reflection_lm = reflection_lm or os.environ.get(
            "ORCHESTRATOR_GEPA_REFLECTION_LM", "openai/local"
        )
        self.reflection_lm_url = reflection_lm_url or os.environ.get(
            "ORCHESTRATOR_GEPA_REFLECTION_URL", "http://localhost:8082/v1"
        )

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
            # Configure reflection LM via litellm (GEPA uses litellm internally).
            # Pass a CALLABLE, not the model-id string: GEPA's str path builds its
            # own wrapper that calls litellm WITHOUT api_base (gepa/api.py:242-244),
            # so a bare model id always resolves to api.openai.com and can never
            # reach our local server. The callable is where reflection_lm_url
            # actually becomes load-bearing.
            import os
            import litellm
            os.environ.setdefault("OPENAI_API_KEY", "not-needed")

            reflection_model = self.reflection_lm
            reflection_url = self.reflection_lm_url

            def _reflection_lm(prompt: str) -> str:
                completion = litellm.completion(
                    model=reflection_model,
                    messages=[{"role": "user", "content": prompt}],
                    api_base=reflection_url,
                )
                return completion.choices[0].message.content

            result = gepa.optimize(
                seed_candidate=seed_candidate,
                trainset=sentinels,
                adapter=adapter,
                reflection_lm=_reflection_lm,
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
            return None

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

        # A run that spends the full eval budget and returns the seed unchanged is a
        # no-op, not a null result. That is what the NotImplementedError proposer bug
        # produced for months: 633s and 50 evals per invocation, logged at INFO as a
        # 0.718 -> 0.000 "completion". Surface it loudly so it cannot hide again.
        if best_content == original_content:
            log.error(
                "GEPA produced NO mutation for %s — seed returned unchanged after %.0fs (%d evals). "
                "The reflective proposer emitted nothing; check the reflection LM at %s and the "
                "preceding log for 'Reflective mutation did not propose a new candidate'.",
                target_file, elapsed, n_evals, self.reflection_lm_url,
            )
            return None

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
