"""AGG Phase 0: corpus store, index registry, deterministic fact layer, typed planner.

Plan of record: ``aggregation-plan-fable.md`` rev 2 (2026-11-10), section (b.2):
one parse (``DocumentPreprocessResult``), one chunk table, N indexes over it, one
planner. Everything here is **default-off**: nothing in the request path imports
this package, and every public entry point that would change behaviour checks
:func:`agg_layer_enabled` first. With the flag off, today's flow is unchanged.

Flag: ``ORCHESTRATOR_FEATURE_AGG_LAYER=1`` (preferred) or the legacy
``ORCHESTRATOR_AGG_LAYER=1``. Default False.

Layers:
    L0 chunks       ``store.CorpusStore`` (content-hashed per doc and chunk)
    L1 lexical      ``registry.IndexRegistry`` builder ``l1_bm25`` (reuses ``BM25Index``, one document per stored
                    chunk) or ``l1_passage_bm25`` (the ``context.search`` engine at 800-character passage
                    granularity over the corpus text, built once per corpus content hash)
    L2 dense        not built in Phase 0 here (no embedding); registry key reserved
    L3 facts-det    ``facts_det`` (parsed tables + regex spans, provenance kept)
    L4 facts-llm    ``facts_llm`` (grammar-constrained, injected backend; never
                    calls a model on its own)
"""

from __future__ import annotations

from src.env_parsing import env_bool

FLAG_ENV = "ORCHESTRATOR_FEATURE_AGG_LAYER"
LEGACY_FLAG_ENV = "ORCHESTRATOR_AGG_LAYER"


def agg_layer_enabled() -> bool:
    """True only when the AGG layer flag is explicitly set. Default off."""
    return env_bool(FLAG_ENV, False) or env_bool(LEGACY_FLAG_ENV, False)


class AggLayerDisabled(RuntimeError):
    """Raised when a behaviour-changing AGG entry point is used with the flag off."""


def require_enabled() -> None:
    if not agg_layer_enabled():
        raise AggLayerDisabled(f"AGG layer is off; set {FLAG_ENV}=1 to enable")
