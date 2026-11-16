"""AGG pipeline version flag (AGG-1.12): ``agg_pipeline.v3`` (the default since 2026-11-16), ``agg_pipeline.v1`` or ``agg_pipeline.v2``.

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

``agg_pipeline.v3`` (AGG-1.16, 2026-11-16; the DEFAULT since the operator decision "Bundle v3, verify" of 2026-11-16, AGG-1.16b held-out 6/14 vs v2 5/14, 0 regressions): ``agg_pipeline.v2`` plus the two AGG-1.15 fixes, bundled:
``keep_slot.harvest.v1`` (``facts_typed.keep_slot_harvest``; ``fill_doc(harvest=True)`` via ``PipelineSpec.l4_kwargs()``: rows of a slot the
document did not plan are kept from any called window, zero extra calls) and ``canonical.v3`` (``store.CANONICAL_RULE_V3`` = v2's
attributable-net-income guard plus the ``long_term_debt`` label guard). Everything else is v2's. ``v1`` and ``v2`` are unchanged byte
for byte (their ``l4_kwargs()`` is EMPTY and their ``as_record()`` has no new key); Pin ``ORCHESTRATOR_AGG_PIPELINE=agg_pipeline.v1`` (or ``v2``) to reproduce those pipelines.

Selection: ``pipeline_spec(version)`` with an explicit version, else the environment variable
``ORCHESTRATOR_AGG_PIPELINE`` (``agg_pipeline.v1`` | ``agg_pipeline.v2`` | ``agg_pipeline.v3``), else ``agg_pipeline.v3``. An unknown value is
refused, never silently mapped to the default. ``v1`` calls every helper below with EXACTLY the arguments the Phase 0 /
AGG-1.1 code used (no keyword at all), so those runs reproduce byte for byte; the tests assert it.

Orthogonal opt-in (AGG-1.13 amendment 1, default OFF): ``fy_end`` (``facts_det.fy_end_months``; ``extract_l3(..., fy_end=...)``,
``IndexRegistry(fy_end=...)``) labels a DATE column header by the filing's fiscal year (Ulta's "January 28, 2023" first column is fiscal 2022)
under EITHER version; with it unset every helper makes exactly the call above.

The per-document fiscal-year and unit-label fixes (``facts_llm.unit_override``, ``declared_scale``) are not part of the
version: they are in ``facts_llm`` itself and are on for both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.aggregation import facts_det
from src.aggregation.facts_typed import KEEP_SLOT_HARVEST
from src.aggregation.store import CANONICAL_RULE, CANONICAL_RULE_V2, CANONICAL_RULE_V3, CorpusStore

PIPELINE_V1 = "agg_pipeline.v1"
PIPELINE_V2 = "agg_pipeline.v2"
PIPELINE_V3 = "agg_pipeline.v3"
PIPELINES = (PIPELINE_V1, PIPELINE_V2, PIPELINE_V3)
FLAG_ENV = "ORCHESTRATOR_AGG_PIPELINE"
DEFAULT = PIPELINE_V3

L4_ROWS_V1 = "l4_rows.v1"            # facts_llm.LazyFieldBackfill
L4_TYPED_V1 = "l4_typed.v1"          # facts_typed.TypedBackfill
V2_CANONICAL_GUARDS = ("attributable_net_income",)
KEEP_SLOT_V1 = "keep_slot.v1"        # facts_typed.keep_slot (v1 has no typed stage; v2's gate)


@dataclass(frozen=True)
class PipelineSpec:
    version: str
    l3_period_version: str | None
    l4_stage: str
    canonical_rule: str
    canonical_guards: tuple[str, ...] | None
    l4_slot_gate: str = KEEP_SLOT_V1     # AGG-1.16: ``KEEP_SLOT_HARVEST`` for v3 only

    # ------------------------------------------------------------------ L3
    def l3_kwargs(self, fiscal_years: Mapping[str, str] | None, fy_end: Any = None) -> dict[str, Any]:
        """Keyword arguments of ``facts_det.extract_corpus_facts``: EMPTY for v1 (the legacy call). ``fy_end`` (AGG-1.13 amendment 1,
        opt-in, default None = off) adds the fiscal-year-end column labelling to EITHER version: v1 then gets ``fy_end`` and ``fiscal_years``
        (still ``period_version=None``), v2 gets ``fy_end`` on top of its own keywords."""
        months = facts_det.fy_end_months(fy_end) if fy_end is not None else None
        out: dict[str, Any] = {}
        if self.l3_period_version is not None:
            out = {"period_version": self.l3_period_version, "fiscal_years": dict(fiscal_years or {})}
        if months:
            out["fy_end"] = months
            out.setdefault("fiscal_years", dict(fiscal_years or {}))
        return out

    def extract_l3(self, store: CorpusStore, corpus_id: str,
                   fiscal_years: Mapping[str, str] | None = None, fy_end: Any = None) -> list[dict[str, Any]]:
        """The L3 facts of a corpus under this pipeline version (no model). ``fiscal_years`` maps ``doc_id`` to the
        filing's fiscal year (the corpus manifest); v2 needs it for the first-column attribution of a table with no header, and
        ``fy_end`` (either version) for the fiscal-year-end labelling of date column headers."""
        return facts_det.extract_corpus_facts(store, corpus_id, **self.l3_kwargs(fiscal_years, fy_end))

    # ------------------------------------------------------------------ L4
    def l4_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``facts_typed.TypedBackfill.fill_doc``: EMPTY for v1 and v2, ``{"harvest": True}`` for v3."""
        return {"harvest": True} if self.l4_slot_gate == KEEP_SLOT_HARVEST else {}

    # ------------------------------------------------------------ canonical
    def canonical_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``CorpusStore.refresh_canonical`` / ``select_canonical``: EMPTY for v1."""
        if self.canonical_rule == CANONICAL_RULE:
            return {}
        return {"rule": self.canonical_rule, "guards": self.canonical_guards}

    def refresh_canonical(self, store: CorpusStore, corpus_id: str | None = None) -> dict[str, Any]:
        return store.refresh_canonical(corpus_id, **self.canonical_kwargs())

    def as_record(self) -> dict[str, Any]:
        out = {"pipeline": self.version, "l3_period": self.l3_period_version or "legacy", "l4_stage": self.l4_stage,
               "canonical_rule": self.canonical_rule, "canonical_guards": list(self.canonical_guards or [])}
        if self.l4_slot_gate != KEEP_SLOT_V1:     # v1 / v2 records keep their exact keys
            out["l4_slot_gate"] = self.l4_slot_gate
        return out


_SPECS = {
    PIPELINE_V1: PipelineSpec(PIPELINE_V1, None, L4_ROWS_V1, CANONICAL_RULE, None),
    PIPELINE_V2: PipelineSpec(PIPELINE_V2, facts_det.L3_PERIOD_V2, L4_TYPED_V1, CANONICAL_RULE_V2, V2_CANONICAL_GUARDS),
    PIPELINE_V3: PipelineSpec(PIPELINE_V3, facts_det.L3_PERIOD_V2, L4_TYPED_V1, CANONICAL_RULE_V3, V2_CANONICAL_GUARDS,
                              KEEP_SLOT_HARVEST),
}


def pipeline_spec(version: str | None = None, *, environ: Mapping[str, str] | None = None) -> PipelineSpec:
    """The spec of ``version`` (explicit wins), else ``ORCHESTRATOR_AGG_PIPELINE``, else ``agg_pipeline.v3``."""
    if version is None:
        env = os.environ if environ is None else environ
        version = (env.get(FLAG_ENV) or "").strip() or DEFAULT
    if version not in _SPECS:
        raise ValueError(f"unknown AGG pipeline version {version!r}; known: {PIPELINES}")
    return _SPECS[version]


def l4_stage_for(spec: PipelineSpec) -> str:
    """The L4 stage module a caller runs for this version: ``facts_llm`` (v1) or ``facts_typed`` (v2, v3)."""
    return "src.aggregation.facts_typed" if spec.l4_stage == L4_TYPED_V1 else "src.aggregation.facts_llm"


def versions() -> Sequence[str]:
    return PIPELINES
