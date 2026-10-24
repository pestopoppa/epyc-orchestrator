"""
Hierarchical Local-Agent Orchestration.

A multi-tier agent system for CPU-optimized LLM inference on AMD EPYC hardware.

Packages:
    api: FastAPI application, routes, request/response models, structured logging
    backends: LLM inference backends (llama.cpp HTTP, completion, speculative)
    classifiers: Intent and complexity classification for task routing
    db: SQLite models for episodic memory, skills, and session storage
    graph: Pydantic-graph orchestration nodes (triage, plan, execute, review)
    llm_primitives: Low-level LLM call wrappers with retry and streaming
    metrics: Inference telemetry and performance counters
    models: Pydantic models for document processing pipeline
    pipeline_monitor: Real-time pipeline health and anomaly detection
    proactive_delegation: Complexity-driven task delegation to specialist agents
    prompt_builders: Role-specific prompt assembly (frontdoor, coder, architect)
    repl_environment: Sandboxed Python REPL with AST security and tool access
    services: Document processing, caching, corpus retrieval services
    session: Session persistence, checkpointing, and document caching
    tools: Callable tool implementations (code, data, file, web, canvas)
    vision: Image/video analysis with CLIP embeddings and batch processing
"""

# --- Compat-alias bootstrap (TD-21; A3-16/A4-08) -----------------------------
#
# Each key below is a backward-compatible top-level name for a module that now
# lives in a subpackage. Shim files that did `sys.modules[__name__] = _real`
# from inside their own module exec are unsafe under concurrent first import:
# a concurrent importer's fast path can read the alias's
# `__spec__._initializing` off the wrong (already fully loaded) target spec and
# skip CPython's own module lock, intermittently observing a half-swapped
# placeholder (`ImportError: cannot import name 'inference_lock' from
# 'src.inference_lock'`, ~1% under 24-way concurrency; see
# tests/unit/test_inference_lock_concurrent_import.py). Those shims also
# renamed the target module to the alias.
#
# Fix: resolve each alias through a MetaPathFinder, installed here ahead of the
# standard path finder, whose Loader's create_module() returns the target
# module object itself. importlib inserts that FINAL object into
# sys.modules[alias] exactly once, so there is no placeholder-then-swap window,
# and `sys.modules[alias] is sys.modules[target]` keeps
# `monkeypatch.setattr("<alias>.X", ...)` visible to the target. exec_module()
# restores the target's own __spec__, which module_from_spec() overwrites.
#
# This only teaches the import system how to resolve these dotted names; no
# target is imported until its alias is.
import sys as _sys
from importlib.abc import Loader as _Loader, MetaPathFinder as _MetaPathFinder
from importlib.util import spec_from_loader as _spec_from_loader

_COMPAT_ALIASES = {
    "src.concurrency": "src.runtime.concurrency",
    "src.delegation_cache": "src.orchestration.delegation_cache",
    "src.delegation_reports": "src.orchestration.delegation_reports",
    "src.dispatcher": "src.orchestration.dispatcher",
    "src.escalation": "src.orchestration.escalation",
    "src.executor": "src.runtime.executor",
    "src.inference_lock": "src.runtime.inference_lock",
    "src.inference_tap": "src.runtime.inference_tap",
    "src.llm_cache": "src.inference.llm_cache",
    "src.model_server": "src.inference.model_server",
    "src.prefix_cache": "src.inference.prefix_cache",
    "src.registry_loader": "src.registry.registry_loader",
    "src.script_registry": "src.registry.script_registry",
    "src.task_ir": "src.orchestration.task_ir",
    "src.tool_loader": "src.registry.tool_loader",
    "src.tool_registry": "src.registry.tool_registry",
}


class _CompatAliasLoader(_Loader):
    """``create_module()`` hands back the already-imported target module."""

    def create_module(self, spec):
        import importlib

        module = importlib.import_module(_COMPAT_ALIASES[spec.name])
        self._target_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Target module is already fully initialized; only undo the spec swap.
        module.__spec__ = self._target_spec


class _CompatAliasFinder(_MetaPathFinder):
    """Resolves only the names in ``_COMPAT_ALIASES``; defers everything else."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in _COMPAT_ALIASES:
            return None
        return _spec_from_loader(fullname, _CompatAliasLoader())


if not any(isinstance(_finder, _CompatAliasFinder) for _finder in _sys.meta_path):
    _sys.meta_path.insert(0, _CompatAliasFinder())
