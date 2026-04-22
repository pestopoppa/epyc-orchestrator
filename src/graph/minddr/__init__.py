"""MindDR deep-research subgraph (NIB2-45 Phase 1).

Three-agent research pipeline — Planning → DeepSearch fan-out → Report
— gated by the ``deep_research_mode`` feature flag.

Phase 1 is prompt-only and training-free. Phase 2 four-stage RL training
(SFT → Search-RL/GSPO/GRPO → Report-RL/DAPO → preference alignment) is
GPU-gated and out of scope here.
"""

from src.graph.minddr.state import (
    MindDRState,
    MindDRDeps,
    MindDRResult,
    SubQuestion,
    SubReport,
    EvidenceTag,
)
from src.graph.minddr.nodes import (
    PlanningNode,
    DeepSearchFanOutNode,
    ReportSynthesisNode,
)
from src.graph.minddr.parsing import (
    parse_planning_output,
    parse_sub_report,
    EVIDENCE_TAGS,
)
from src.graph.minddr.graph import (
    load_minddr_prompts,
    minddr_graph,
    run_minddr,
)

__all__ = [
    "MindDRState", "MindDRDeps", "MindDRResult",
    "SubQuestion", "SubReport", "EvidenceTag",
    "PlanningNode", "DeepSearchFanOutNode", "ReportSynthesisNode",
    "parse_planning_output", "parse_sub_report",
    "EVIDENCE_TAGS",
    "load_minddr_prompts", "minddr_graph", "run_minddr",
]
