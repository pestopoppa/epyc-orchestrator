"""AGG pipeline version flag (AGG-1.12): ``agg_pipeline.v1`` (the default) or ``agg_pipeline.v2``.

One switch, three behaviours that were each measured separately and are adopted together:

=================  ==========================================  ================================================
stage              ``agg_pipeline.v1`` (Phase 0 / AGG-1.1)      ``agg_pipeline.v2`` (AGG-1.12)
=================  ==========================================  ================================================
L3 period          legacy attribution (``period_version=None``)  ``l3_period.v2`` (default components)
L4 extraction      ``facts_llm`` row stage, one call per         ``facts_typed`` (``l4_typed.v1``): union windows,
                   admitted window and field                     ONE typed multi-field call per distinct window
canonical          ``canonical.v1``                              ``canonical.v2`` with the attributable-net-income
                                                                 guard (the thousands-unit guard is NOT adopted:
                                                                 measured 0 fixes, 0 breaks)
=================  ==========================================  ================================================

Selection: ``pipeline_spec(version)`` with an explicit version, else the environment variable
``ORCHESTRATOR_AGG_PIPELINE`` (``agg_pipeline.v1`` | ``agg_pipeline.v2``), else ``agg_pipeline.v1``. An unknown value is
refused, never silently mapped to the default. ``v1`` calls every helper below with EXACTLY the arguments the Phase 0 /
AGG-1.1 code used (no keyword at all), so those runs reproduce byte for byte; the tests assert it.

The per-document fiscal-year and unit-label fixes (``facts_llm.unit_override``, ``declared_scale``) are not part of the
version: they are in ``facts_llm`` itself and are on for both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.aggregation import facts_det
from src.aggregation.store import CANONICAL_RULE, CANONICAL_RULE_V2, CorpusStore

PIPELINE_V1 = "agg_pipeline.v1"
PIPELINE_V2 = "agg_pipeline.v2"
PIPELINES = (PIPELINE_V1, PIPELINE_V2)
FLAG_ENV = "ORCHESTRATOR_AGG_PIPELINE"
DEFAULT = PIPELINE_V1

L4_ROWS_V1 = "l4_rows.v1"            # facts_llm.LazyFieldBackfill
L4_TYPED_V1 = "l4_typed.v1"          # facts_typed.TypedBackfill
V2_CANONICAL_GUARDS = ("attributable_net_income",)


@dataclass(frozen=True)
class PipelineSpec:
    version: str
    l3_period_version: str | None
    l4_stage: str
    canonical_rule: str
    canonical_guards: tuple[str, ...] | None

    # ------------------------------------------------------------------ L3
    def l3_kwargs(self, fiscal_years: Mapping[str, str] | None) -> dict[str, Any]:
        """Keyword arguments of ``facts_det.extract_corpus_facts``: EMPTY for v1 (the legacy call)."""
        if self.l3_period_version is None:
            return {}
        return {"period_version": self.l3_period_version, "fiscal_years": dict(fiscal_years or {})}

    def extract_l3(self, store: CorpusStore, corpus_id: str,
                   fiscal_years: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
        """The L3 facts of a corpus under this pipeline version (no model). ``fiscal_years`` maps ``doc_id`` to the
        filing's fiscal year (the corpus manifest); v2 needs it for the first-column attribution of a table with no header."""
        return facts_det.extract_corpus_facts(store, corpus_id, **self.l3_kwargs(fiscal_years))

    # ------------------------------------------------------------ canonical
    def canonical_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``CorpusStore.refresh_canonical`` / ``select_canonical``: EMPTY for v1."""
        if self.canonical_rule == CANONICAL_RULE:
            return {}
        return {"rule": self.canonical_rule, "guards": self.canonical_guards}

    def refresh_canonical(self, store: CorpusStore, corpus_id: str | None = None) -> dict[str, Any]:
        return store.refresh_canonical(corpus_id, **self.canonical_kwargs())

    def as_record(self) -> dict[str, Any]:
        return {"pipeline": self.version, "l3_period": self.l3_period_version or "legacy", "l4_stage": self.l4_stage,
                "canonical_rule": self.canonical_rule, "canonical_guards": list(self.canonical_guards or [])}


_SPECS = {
    PIPELINE_V1: PipelineSpec(PIPELINE_V1, None, L4_ROWS_V1, CANONICAL_RULE, None),
    PIPELINE_V2: PipelineSpec(PIPELINE_V2, facts_det.L3_PERIOD_V2, L4_TYPED_V1, CANONICAL_RULE_V2, V2_CANONICAL_GUARDS),
}


def pipeline_spec(version: str | None = None, *, environ: Mapping[str, str] | None = None) -> PipelineSpec:
    """The spec of ``version`` (explicit wins), else ``ORCHESTRATOR_AGG_PIPELINE``, else ``agg_pipeline.v1``."""
    if version is None:
        env = os.environ if environ is None else environ
        version = (env.get(FLAG_ENV) or "").strip() or DEFAULT
    if version not in _SPECS:
        raise ValueError(f"unknown AGG pipeline version {version!r}; known: {PIPELINES}")
    return _SPECS[version]


def l4_stage_for(spec: PipelineSpec) -> str:
    """The L4 stage module a caller runs for this version: ``facts_llm`` (v1) or ``facts_typed`` (v2)."""
    return "src.aggregation.facts_typed" if spec.l4_stage == L4_TYPED_V1 else "src.aggregation.facts_llm"


def versions() -> Sequence[str]:
    return PIPELINES
