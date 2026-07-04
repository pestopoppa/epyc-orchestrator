#!/usr/bin/env python3
"""Corpus-Augmented Prompt Stuffing — Claude-as-Judge Quality Gate.

Runs the same code generation prompts with and without corpus injection,
then uses Claude to judge whether corpus injection degrades output quality.

Usage:
    python scripts/benchmark/corpus_quality_gate.py --models coder_escalation worker_general
    python scripts/benchmark/corpus_quality_gate.py --models coder_escalation worker_general --preflight-only
    python scripts/benchmark/corpus_quality_gate.py --models coder_escalation --dry-run
    python scripts/benchmark/corpus_quality_gate.py --models coder_escalation worker_general --results-only
    python scripts/benchmark/corpus_quality_gate.py --models coder_escalation --mode rag --confirm-clean-window

Modes:
  speed (default): Inject snippets silently in ## Reference Code (Phase 2A)
  rag: Inject snippets with explicit RAG instruction (Phase 2B-Quality)

Quality gate:
  speed mode: PASS if average quality delta >= -0.5 (must not degrade)
  rag mode: PASS if average quality delta > 0 (must IMPROVE)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_primary_port,
    stack_prior_serving,
)
from scripts.server.stack_manifest import HOT_ROLES, PORT_MAP
STACK_PRIORS_PATH = PROJECT_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
CORPUS_AB_MODEL_ROLES = ("coder_escalation", "worker_general")
FALLBACK_MODEL_ROLES = (*CORPUS_AB_MODEL_ROLES, "frontdoor", "architect_general")


def _preferred_fallback_model_roles() -> tuple[str, ...]:
    """Return the preferred fallback model order when live models are absent."""
    return tuple(
        role
        for role in FALLBACK_MODEL_ROLES
        if role in HOT_ROLES and isinstance(PORT_MAP.get(role), int)
    )


def _load_live_models(path: Path = STACK_PRIORS_PATH) -> dict[str, dict]:
    roles = live_stack_role_records(path)

    models: dict[str, dict] = {}
    for role, record in roles.items():
        if not isinstance(record, dict):
            continue
        serving = stack_prior_serving(record)
        port = stack_prior_primary_port(serving)
        if port is None:
            continue
        models[str(role)] = {
            "port": port,
            "name": str(record.get("display_name") or record.get("model_id") or role),
            "role": str(record.get("role") or role),
        }
    return models


def _fallback_models() -> dict[str, dict]:
    """Return degraded model choices from the manifest, without copied model names."""
    models: dict[str, dict] = {}
    for role in _preferred_fallback_model_roles():
        port = PORT_MAP.get(role)
        models[role] = {
            "port": port,
            "name": f"{role} (manifest fallback)",
            "role": role,
        }
    return models


def _default_model_keys(models: dict[str, dict]) -> list[str]:
    preferred = [role for role in CORPUS_AB_MODEL_ROLES if role in models]
    if preferred:
        return preferred
    fallback = [role for role in _preferred_fallback_model_roles() if role in models]
    return fallback or list(models.keys())


def _model_role(model_key: str, models: dict[str, dict] | None = None) -> str:
    model_info = (models or MODELS).get(model_key, {})
    return str(model_info.get("role") or model_key)


def _role_corpus_retrieval_metadata(
    model_keys: list[str],
    *,
    models: dict[str, dict] | None = None,
    registry_loader_cls: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Report production role corpus flags for a benchmark role set.

    The benchmark corpus arm intentionally forces prompt injection to test
    candidate roles such as ``worker_general`` without flipping production
    registry flags. Record both facts so A/B artifacts cannot be mistaken for
    live enablement evidence.
    """
    selected_models = models or MODELS
    try:
        if registry_loader_cls is None:
            from src.registry.registry_loader import RegistryLoader

            registry_loader_cls = RegistryLoader
        registry = registry_loader_cls(validate_paths=False)
        runtime_cfg = registry.get_corpus_config()
    except Exception as exc:
        return {
            model_key: {
                "role": _model_role(model_key, selected_models),
                "production_runtime_enabled": None,
                "production_role_enabled": None,
                "benchmark_forces_prompt_injection": True,
                "status": "registry_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            for model_key in model_keys
        }

    metadata: dict[str, dict[str, Any]] = {}
    for model_key in model_keys:
        role = _model_role(model_key, selected_models)
        try:
            role_cfg = registry.get_role(role)
            role_enabled = bool(role_cfg.acceleration.corpus_retrieval)
            status = "ok"
            error = ""
        except Exception as exc:
            role_enabled = None
            status = "role_error"
            error = f"{type(exc).__name__}: {exc}"
        metadata[model_key] = {
            "role": role,
            "production_runtime_enabled": bool(runtime_cfg.get("enabled", False)),
            "production_role_enabled": role_enabled,
            "benchmark_forces_prompt_injection": True,
            "status": status,
            "error": error,
        }
    return metadata


MODELS = _load_live_models() or _fallback_models()

# Code generation prompts — novel tasks where corpus could help or hurt
PROMPTS = [
    {
        "id": "async_retry",
        "prompt": "Write a Python async HTTP client with retry logic, exponential backoff, and circuit breaker pattern. Include type hints and a usage example.",
        "language": "python",
    },
    {
        "id": "bst_iterator",
        "prompt": "Implement a binary search tree in Python with an in-order iterator that uses O(h) memory where h is the height of the tree. Include insert, search, delete, and the iterator protocol (__iter__, __next__).",
        "language": "python",
    },
    {
        "id": "lru_cache",
        "prompt": "Write a thread-safe LRU cache in Python using a doubly-linked list and a dictionary. Support get, put, and resize operations. Include proper locking and a decorator version.",
        "language": "python",
    },
    {
        "id": "json_parser",
        "prompt": "Write a recursive descent JSON parser in Python from scratch (no json module). Handle strings (with escapes), numbers (int and float), booleans, null, arrays, and objects. Return native Python types.",
        "language": "python",
    },
    {
        "id": "rate_limiter",
        "prompt": "Implement a token bucket rate limiter in Python that supports per-key limits, burst capacity, and automatic refill. Make it work both synchronously and with asyncio.",
        "language": "python",
    },
    {
        "id": "graph_shortest",
        "prompt": "Write Dijkstra's algorithm and A* search in Python. Support weighted directed graphs with an adjacency list representation. Include a priority queue implementation and path reconstruction.",
        "language": "python",
    },
]

# Claude-as-Judge prompt template
JUDGE_PROMPT = """You are evaluating code generation quality. You will see two code outputs for the same prompt — Output A and Output B. One was generated with corpus-augmented prompt stuffing (injected reference code snippets), the other without.

You do NOT know which is which. Judge each output independently on these criteria:

1. **Correctness** (1-10): Does the code work? Are there bugs?
2. **Completeness** (1-10): Does it address all requirements in the prompt?
3. **Code quality** (1-10): Clean style, good naming, proper error handling, type hints?
4. **Originality** (1-10): Does it feel like a thoughtful solution vs. copied boilerplate?

Return your scores in this exact JSON format (no other text):
{{"a_correctness": N, "a_completeness": N, "a_quality": N, "a_originality": N, "b_correctness": N, "b_completeness": N, "b_quality": N, "b_originality": N, "notes": "brief comparison"}}

## Task Prompt
{task_prompt}

## Output A
```python
{output_a}
```

## Output B
```python
{output_b}
```

Return ONLY the JSON object, no other text."""


@dataclass
class GenerationResult:
    model: str
    prompt_id: str
    corpus_enabled: bool
    output: str
    speed_tps: float
    tokens_generated: int
    draft_n: int = 0
    draft_accepted: int = 0
    wall_time: float = 0.0
    corpus_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusPromptBuild:
    prompt: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    prompt_id: str
    model: str
    baseline_score: float  # avg of 4 criteria
    corpus_score: float
    delta: float
    raw_scores: dict = field(default_factory=dict)


def generate(port: int, prompt: str, max_tokens: int = 1024) -> dict:
    """Send generation request to llama-server."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # Deterministic for fair comparison
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=600)
    wall = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    timings = data.get("timings", {})

    return {
        "output": content,
        "tokens": usage.get("completion_tokens", len(content.split())),
        "speed": timings.get("predicted_per_second", 0),
        "draft_n": timings.get("draft_n", 0),
        "draft_accepted": timings.get("draft_n_accepted", 0),
        "wall_time": wall,
    }


def _same_retriever_config(current: Any, desired: Any) -> bool:
    """Return True when the singleton already has the requested corpus config."""
    return (
        bool(getattr(current, "enabled", False)) == bool(getattr(desired, "enabled", False))
        and str(getattr(current, "index_path", "")) == str(getattr(desired, "index_path", ""))
        and int(getattr(current, "max_snippets", 0) or 0)
        == int(getattr(desired, "max_snippets", 0) or 0)
        and int(getattr(current, "max_chars", 0) or 0)
        == int(getattr(desired, "max_chars", 0) or 0)
        and float(getattr(current, "min_score", 0.0) or 0.0)
        == float(getattr(desired, "min_score", 0.0) or 0.0)
        and bool(getattr(current, "rag_enabled", False))
        == bool(getattr(desired, "rag_enabled", False))
        and int(getattr(current, "rag_max_snippets", 0) or 0)
        == int(getattr(desired, "rag_max_snippets", 0) or 0)
        and int(getattr(current, "rag_max_chars", 0) or 0)
        == int(getattr(desired, "rag_max_chars", 0) or 0)
        and float(getattr(current, "rag_min_score", 0.0) or 0.0)
        == float(getattr(desired, "rag_min_score", 0.0) or 0.0)
    )


def _configured_retriever(retriever_cls: Any, config: Any) -> Any:
    """Reuse the corpus singleton when possible so the harness matches production."""
    retriever = retriever_cls.get_instance(config)
    if not _same_retriever_config(retriever.config, config):
        retriever_cls.reset_instance()
        retriever = retriever_cls.get_instance(config)
    else:
        retriever.config = config
    return retriever


def _snippet_sources(snippets: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "file": str(getattr(snippet, "file", "") or ""),
            "start_line": int(getattr(snippet, "start_line", 0) or 0),
            "score": float(getattr(snippet, "score", 0.0) or 0.0),
            "hash": str(getattr(snippet, "hash", "") or ""),
        }
        for snippet in snippets
    ]


def _corpus_diagnostics_payload(
    *,
    mode: str,
    query: str,
    snippets: list[Any],
    retriever: Any,
    prompt: str,
    built_prompt: str,
) -> dict[str, Any]:
    diagnostics = getattr(retriever, "last_diagnostics", None)
    diag_payload = asdict(diagnostics) if diagnostics is not None else {}
    return {
        "mode": mode,
        "query": query,
        "injected": built_prompt != prompt,
        "prompt_chars_added": max(len(built_prompt) - len(prompt), 0),
        "snippets_returned": len(snippets),
        "snippet_sources": _snippet_sources(snippets),
        "loaded": bool(diag_payload.get("loaded", False)),
        "format": str(diag_payload.get("format", "") or ""),
        "query_ngrams": int(diag_payload.get("query_ngrams", 0) or 0),
        "candidates_found": int(diag_payload.get("candidates_found", 0) or 0),
        "retrieval_latency_ms": round(
            float(diag_payload.get("elapsed_ms", 0.0) or 0.0),
            3,
        ),
        "failure_reason": str(diag_payload.get("failure_reason", "") or ""),
        "failure_detail": str(diag_payload.get("failure_detail", "") or ""),
        "shards_queried": int(diag_payload.get("shards_queried", 0) or 0),
        "shards_failed": int(diag_payload.get("shards_failed", 0) or 0),
        "shards_unavailable": int(diag_payload.get("shards_unavailable", 0) or 0),
    }


def build_corpus_prompt_with_diagnostics(
    prompt: str,
    corpus_config: dict,
    mode: str = "speed",
) -> CorpusPromptBuild:
    """Build a corpus prompt and return lookup diagnostics for the A/B artifact."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.services.corpus_retrieval import CorpusConfig, CorpusRetriever, extract_code_query

    if mode == "rag":
        config = CorpusConfig(
            enabled=True,
            index_path=corpus_config.get("index_path", "/mnt/raid0/llm/cache/corpus/mvp_index"),
            max_snippets=corpus_config.get("max_snippets", 3),
            max_chars=corpus_config.get("max_chars", 3000),
            min_score=corpus_config.get("min_score", 0.5),
            rag_enabled=True,
            rag_max_snippets=corpus_config.get("rag_max_snippets", 5),
            rag_max_chars=corpus_config.get("rag_max_chars", 5000),
            rag_min_score=corpus_config.get("rag_min_score", 0.3),
        )
        retriever = _configured_retriever(CorpusRetriever, config)
        query = extract_code_query(prompt)
        snippets = retriever.retrieve_for_rag(query)
        log.info("    RAG retrieval: query=%r -> %d snippets", query[:60], len(snippets))
        built_prompt = retriever.format_for_rag(snippets, prompt) if snippets else prompt
        return CorpusPromptBuild(
            prompt=built_prompt,
            diagnostics=_corpus_diagnostics_payload(
                mode=mode,
                query=query,
                snippets=snippets,
                retriever=retriever,
                prompt=prompt,
                built_prompt=built_prompt,
            ),
        )

    config = CorpusConfig(
        enabled=True,
        index_path=corpus_config.get("index_path", "/mnt/raid0/llm/cache/corpus/mvp_index"),
        max_snippets=corpus_config.get("max_snippets", 3),
        max_chars=corpus_config.get("max_chars", 3000),
        min_score=corpus_config.get("min_score", 0.5),
    )
    retriever = _configured_retriever(CorpusRetriever, config)
    query = extract_code_query(prompt)
    snippets = retriever.retrieve(query)
    corpus_ctx = retriever.format_for_prompt(snippets)
    built_prompt = f"{corpus_ctx}\n\n{prompt}" if corpus_ctx else prompt
    return CorpusPromptBuild(
        prompt=built_prompt,
        diagnostics=_corpus_diagnostics_payload(
            mode=mode,
            query=query,
            snippets=snippets,
            retriever=retriever,
            prompt=prompt,
            built_prompt=built_prompt,
        ),
    )


def build_corpus_prompt(prompt: str, corpus_config: dict, mode: str = "speed") -> str:
    """Build prompt with corpus context injected and discard benchmark diagnostics."""
    return build_corpus_prompt_with_diagnostics(prompt, corpus_config, mode=mode).prompt


def warmup(port: int) -> None:
    """Send a short warmup request to prime the KV cache and JIT paths."""
    log.info("  Warming up port %d...", port)
    try:
        generate(port, "Say hello.", max_tokens=5)
        log.info("  Warmup done.")
    except Exception as e:
        log.warning("  Warmup failed (non-fatal): %s", e)


def _active_autopilot() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "scripts/autopilot/autopilot.py start"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _live_generation_refusal(args: argparse.Namespace) -> tuple[int, str] | None:
    """Return a refusal for production-port generation outside a clean window."""
    if args.preflight_only or args.dry_run or args.results_only:
        return None
    if not args.confirm_clean_window:
        return (
            2,
            "refusing to run live generation: pass --confirm-clean-window for "
            "corpus-on/off production-port A/B",
        )
    if _active_autopilot() and not args.allow_active_autopilot:
        return (
            75,
            "refusing to run live generation: AutoPilot appears active; stop it "
            "or pass --allow-active-autopilot for non-claim-grade live-load telemetry",
        )
    return None


def _run_single_pair(
    model_key: str, port: int, prompt_info: dict, corpus_config: dict, mode: str,
) -> tuple[GenerationResult, GenerationResult]:
    """Run baseline + corpus generation for a single prompt."""
    p = prompt_info
    log.info("  [%s] %s — baseline...", model_key, p["id"])

    result_b = generate(port, p["prompt"])
    baseline = GenerationResult(
        model=model_key,
        prompt_id=p["id"],
        corpus_enabled=False,
        output=result_b["output"],
        speed_tps=result_b["speed"],
        tokens_generated=result_b["tokens"],
        draft_n=result_b["draft_n"],
        draft_accepted=result_b["draft_accepted"],
        wall_time=result_b["wall_time"],
    )

    log.info("  [%s] %s — with corpus (%s)...", model_key, p["id"], mode)

    corpus_build = build_corpus_prompt_with_diagnostics(
        p["prompt"],
        corpus_config,
        mode=mode,
    )
    result_c = generate(port, corpus_build.prompt)
    corpus = GenerationResult(
        model=model_key,
        prompt_id=p["id"],
        corpus_enabled=True,
        output=result_c["output"],
        speed_tps=result_c["speed"],
        tokens_generated=result_c["tokens"],
        draft_n=result_c["draft_n"],
        draft_accepted=result_c["draft_accepted"],
        wall_time=result_c["wall_time"],
        corpus_diagnostics=corpus_build.diagnostics,
    )

    log.info(
        "    [%s] %s done: baseline=%.1f t/s, corpus=%.1f t/s, "
        "snippets=%s, lookup_ms=%.1f",
        model_key,
        p["id"],
        baseline.speed_tps,
        corpus.speed_tps,
        corpus.corpus_diagnostics.get("snippets_returned", 0),
        corpus.corpus_diagnostics.get("retrieval_latency_ms", 0.0),
    )
    return baseline, corpus


def run_generation_pairs(
    model_key: str,
    corpus_config: dict,
    dry_run: bool = False,
    mode: str = "speed",
    prompts: list[dict[str, Any]] | None = None,
) -> list[tuple[GenerationResult, GenerationResult]]:
    """Run all prompts with and without corpus for a model.

    Each prompt pair (baseline + corpus) runs sequentially for fair comparison.
    Different prompt pairs can run in parallel when the server has multiple slots.
    """
    cfg = MODELS[model_key]
    port = cfg["port"]
    selected_prompts = list(prompts or PROMPTS)

    if dry_run:
        return [
            (
                GenerationResult(model_key, p["id"], False, "# dry run", 0, 0),
                GenerationResult(model_key, p["id"], True, "# dry run", 0, 0),
            )
            for p in selected_prompts
        ]

    # Run prompt pairs sequentially — each pair does baseline then corpus
    # to ensure fair comparison. Pairs themselves are sequential since
    # each pair uses 2 requests and the server has limited slots.
    pairs = []
    for p in selected_prompts:
        pair = _run_single_pair(model_key, port, p, corpus_config, mode)
        pairs.append(pair)

    return pairs


def _selected_prompts(max_prompts: int | None) -> list[dict[str, Any]]:
    if max_prompts is None:
        return list(PROMPTS)
    if max_prompts <= 0:
        return []
    return list(PROMPTS[:max_prompts])


def run_corpus_preflight(
    corpus_config: dict[str, Any],
    *,
    mode: str,
    prompts: list[dict[str, Any]],
    model_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Build corpus prompts without inference and summarize injection readiness."""
    records: list[dict[str, Any]] = []
    for prompt_info in prompts:
        build = build_corpus_prompt_with_diagnostics(
            prompt_info["prompt"],
            corpus_config,
            mode=mode,
        )
        diagnostics = build.diagnostics
        records.append(
            {
                "prompt_id": prompt_info["id"],
                "language": prompt_info.get("language", ""),
                "prompt_chars": len(prompt_info["prompt"]),
                "built_prompt_chars": len(build.prompt),
                "injected": bool(diagnostics.get("injected", False)),
                "corpus": diagnostics,
            }
        )

    injected_count = sum(1 for record in records if record["injected"])
    failure_count = sum(
        1
        for record in records
        if record["corpus"].get("failure_reason")
        or int(record["corpus"].get("shards_failed", 0) or 0) > 0
    )
    ready = bool(records) and injected_count == len(records) and failure_count == 0
    return {
        "schema_version": "corpus_quality_preflight.v1",
        "mode": mode,
        "index_path": corpus_config.get("index_path", ""),
        "prompt_count": len(records),
        "injected_count": injected_count,
        "failure_count": failure_count,
        "ready_for_ab": ready,
        "selected_models": list(model_keys or []),
        "production_role_corpus_retrieval": _role_corpus_retrieval_metadata(
            list(model_keys or [])
        ),
        "benchmark_forces_prompt_injection": True,
        "records": records,
    }


def judge_pair(
    prompt_id: str,
    task_prompt: str,
    baseline_output: str,
    corpus_output: str,
) -> JudgeResult | None:
    """Use Claude to judge output quality. Randomizes A/B assignment."""
    import random

    # Randomize which is A vs B to avoid position bias
    corpus_is_a = random.random() < 0.5

    if corpus_is_a:
        output_a, output_b = corpus_output, baseline_output
    else:
        output_a, output_b = baseline_output, corpus_output

    judge_input = JUDGE_PROMPT.format(
        task_prompt=task_prompt,
        output_a=output_a,
        output_b=output_b,
    )

    try:
        result = subprocess.run(
            ["claude", "-p", judge_input, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if result.returncode != 0:
            log.warning("Claude judge failed for %s: %s", prompt_id, result.stderr[:200])
            return None

        # Parse JSON from response
        response = result.stdout.strip()
        # Try to extract JSON if wrapped in markdown
        if "```" in response:
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)

        scores = json.loads(response)

        # Map back: which was baseline vs corpus?
        if corpus_is_a:
            corpus_avg = (scores["a_correctness"] + scores["a_completeness"] + scores["a_quality"] + scores["a_originality"]) / 4
            baseline_avg = (scores["b_correctness"] + scores["b_completeness"] + scores["b_quality"] + scores["b_originality"]) / 4
        else:
            baseline_avg = (scores["a_correctness"] + scores["a_completeness"] + scores["a_quality"] + scores["a_originality"]) / 4
            corpus_avg = (scores["b_correctness"] + scores["b_completeness"] + scores["b_quality"] + scores["b_originality"]) / 4

        return JudgeResult(
            prompt_id=prompt_id,
            model="",
            baseline_score=baseline_avg,
            corpus_score=corpus_avg,
            delta=corpus_avg - baseline_avg,
            raw_scores=scores,
        )
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("Failed to parse judge response for %s: %s", prompt_id, e)
        return None
    except subprocess.TimeoutExpired:
        log.warning("Claude judge timed out for %s", prompt_id)
        return None


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Corpus quality gate")
    parser.add_argument(
        "--models",
        nargs="+",
        default=_default_model_keys(MODELS),
        choices=list(MODELS.keys()),
    )
    parser.add_argument("--index-path", default="/mnt/raid0/llm/cache/corpus/v3_sharded")
    parser.add_argument("--mode", choices=["speed", "rag"], default="speed",
                        help="speed: silent injection (2A), rag: quality RAG instruction (2B)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Build corpus prompts and write injection diagnostics without inference.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Run generation and write results without invoking Claude-as-Judge.",
    )
    parser.add_argument(
        "--confirm-clean-window",
        action="store_true",
        help="Required for live generation against production ports.",
    )
    parser.add_argument(
        "--allow-active-autopilot",
        action="store_true",
        help="Override the default refusal when AutoPilot is running.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        help="Limit prompt pairs for shakedown runs before a full A/B.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum n-gram overlap for speed-mode retrieval candidate evaluation.",
    )
    parser.add_argument(
        "--rag-min-score",
        type=float,
        default=0.3,
        help="Minimum n-gram overlap for RAG-mode retrieval candidate evaluation.",
    )
    parser.add_argument("--results-only", help="Path to existing results JSON to re-judge")
    parser.add_argument("--output", default="/mnt/raid0/llm/tmp/corpus_quality_gate.json")
    args = parser.parse_args()
    selected_prompts = _selected_prompts(args.max_prompts)

    # RAG mode uses more snippets and lower threshold for diverse examples
    if args.mode == "rag":
        corpus_config = {
            "index_path": args.index_path,
            "max_snippets": 3,
            "max_chars": 3000,
            "min_score": args.min_score,
            "rag_max_snippets": 5,
            "rag_max_chars": 5000,
            "rag_min_score": args.rag_min_score,
        }
    else:
        corpus_config = {
            "index_path": args.index_path,
            "max_snippets": 3,
            "max_chars": 3000,
            "min_score": args.min_score,
        }

    if args.preflight_only:
        preflight = run_corpus_preflight(
            corpus_config,
            mode=args.mode,
            prompts=selected_prompts,
            model_keys=list(args.models),
        )
        _write_json(args.output, preflight)
        log.info(
            "Preflight saved to %s: injected=%d/%d, failures=%d, ready_for_ab=%s",
            args.output,
            preflight["injected_count"],
            preflight["prompt_count"],
            preflight["failure_count"],
            preflight["ready_for_ab"],
        )
        sys.exit(0 if preflight["ready_for_ab"] else 1)

    refusal = _live_generation_refusal(args)
    if refusal is not None:
        status, message = refusal
        print(message, file=sys.stderr)
        sys.exit(status)

    # Gate threshold: speed mode tolerates slight degradation, RAG must improve
    gate_threshold = 0.0 if args.mode == "rag" else -0.5

    all_results = {
        "_metadata": {
            "schema_version": "corpus_quality_gate.v2",
            "mode": args.mode,
            "index_path": args.index_path,
            "selected_models": list(args.models),
            "production_role_corpus_retrieval": _role_corpus_retrieval_metadata(
                list(args.models)
            ),
            "benchmark_forces_prompt_injection": True,
            "clean_window_confirmed": bool(args.confirm_clean_window),
            "active_autopilot_override": bool(args.allow_active_autopilot),
        }
    }

    if args.results_only:
        with open(args.results_only) as f:
            all_results = json.load(f)
    else:
        for model_key in args.models:
            log.info("=== Generating for %s (%s) [mode=%s] ===", model_key, MODELS[model_key]["name"], args.mode)
            cfg = MODELS[model_key]
            port = cfg["port"]
            model_results: list[dict] = []

            if not args.dry_run:
                warmup(port)

            if args.dry_run:
                for p in selected_prompts:
                    model_results.append({
                        "prompt_id": p["id"],
                        "baseline": {"output": "# dry run", "speed": 0, "tokens": 0, "draft_n": 0, "draft_accepted": 0, "wall_time": 0},
                        "corpus": {"output": "# dry run", "speed": 0, "tokens": 0, "draft_n": 0, "draft_accepted": 0, "wall_time": 0},
                    })
            else:
                for p in selected_prompts:
                    baseline, corpus = _run_single_pair(model_key, port, p, corpus_config, args.mode)
                    model_results.append({
                        "prompt_id": baseline.prompt_id,
                        "baseline": {
                            "output": baseline.output,
                            "speed": baseline.speed_tps,
                            "tokens": baseline.tokens_generated,
                            "draft_n": baseline.draft_n,
                            "draft_accepted": baseline.draft_accepted,
                            "wall_time": baseline.wall_time,
                        },
                        "corpus": {
                            "output": corpus.output,
                            "speed": corpus.speed_tps,
                            "tokens": corpus.tokens_generated,
                            "draft_n": corpus.draft_n,
                            "draft_accepted": corpus.draft_accepted,
                            "wall_time": corpus.wall_time,
                            "retrieval": corpus.corpus_diagnostics,
                        },
                    })
                    # Write after each prompt pair so partial results are reviewable
                    all_results[model_key] = model_results
                    _write_json(args.output, all_results)
                    log.info(
                        "  Incremental results written (%d/%d prompts)",
                        len(model_results),
                        len(selected_prompts),
                    )

            all_results[model_key] = model_results

        log.info("Generation results saved to %s", args.output)

    if args.dry_run:
        log.info("[DRY RUN] Would judge %d pairs per model", len(selected_prompts))
        return

    if args.skip_judge:
        log.info("[SKIP JUDGE] Generation results saved to %s", args.output)
        return

    # Judge phase
    log.info("\n=== Claude-as-Judge Quality Scoring ===")
    all_judge_results = {}
    gate_pass = True

    for model_key in args.models:
        results = all_results.get(model_key, [])
        judge_results = []

        for r in results:
            prompt_text = next((p["prompt"] for p in PROMPTS if p["id"] == r["prompt_id"]), "")
            log.info("  Judging %s / %s...", model_key, r["prompt_id"])

            jr = judge_pair(
                r["prompt_id"],
                prompt_text,
                r["baseline"]["output"],
                r["corpus"]["output"],
            )
            if jr:
                jr.model = model_key
                judge_results.append(jr)
                log.info(
                    "    baseline=%.1f  corpus=%.1f  delta=%+.1f  %s",
                    jr.baseline_score, jr.corpus_score, jr.delta,
                    "PASS" if jr.delta >= gate_threshold else "FAIL",
                )

        if judge_results:
            avg_delta = sum(j.delta for j in judge_results) / len(judge_results)
            avg_baseline = sum(j.baseline_score for j in judge_results) / len(judge_results)
            avg_corpus = sum(j.corpus_score for j in judge_results) / len(judge_results)

            model_pass = avg_delta >= gate_threshold
            if not model_pass:
                gate_pass = False

            log.info(
                "\n  %s SUMMARY: baseline=%.2f  corpus=%.2f  delta=%+.2f  %s",
                model_key.upper(),
                avg_baseline,
                avg_corpus,
                avg_delta,
                "GATE PASS" if model_pass else "GATE FAIL",
            )

            all_judge_results[model_key] = {
                "avg_baseline": avg_baseline,
                "avg_corpus": avg_corpus,
                "avg_delta": avg_delta,
                "gate_pass": model_pass,
                "per_prompt": [
                    {
                        "prompt_id": j.prompt_id,
                        "baseline": j.baseline_score,
                        "corpus": j.corpus_score,
                        "delta": j.delta,
                        "raw": j.raw_scores,
                    }
                    for j in judge_results
                ],
            }

    # Save judge results
    judge_output = args.output.replace(".json", "_judge.json")
    with open(judge_output, "w") as f:
        json.dump(all_judge_results, f, indent=2)
    log.info("\nJudge results saved to %s", judge_output)

    # Final verdict
    log.info("\n" + "=" * 60)
    if gate_pass:
        if args.mode == "rag":
            log.info("QUALITY GATE: PASS — RAG injection improves quality (delta > 0)")
        else:
            log.info("QUALITY GATE: PASS — corpus injection does not degrade quality")
    else:
        if args.mode == "rag":
            log.info("QUALITY GATE: FAIL — RAG injection does not improve quality (need delta > 0)")
        else:
            log.info("QUALITY GATE: FAIL — corpus injection degrades quality beyond -0.5 threshold")
    log.info("=" * 60)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
