"""Dataset adapter registry — suite → adapter-class mapping.

Extracted from dataset_adapters.py during the 2026-05-22 Task-A refactor.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseAdapter
from .coding import (
    BigCodeBenchAdapter,
    CRUXEvalAdapter,
    CoderAdapter,
    DebugBenchAdapter,
    LiveCodeBenchAdapter,
    USACOAdapter,
)
from .general import (
    GPQAAdapter,
    HotpotQAAdapter,
    MMLUAdapter,
    SimpleQAAdapter,
)
from .math_adapter import MathAdapter
from .reasoning import IFEvalAdapter, ThinkingAdapter
from .vision_agentic import GaiaAdapter, VLAdapter


# Suites that have dataset adapters (vs YAML-only)
ADAPTER_SUITES = {
    "general", "math", "coder", "thinking", "instruction_precision", "vl",
    "gaia", "cruxeval", "bigcodebench",
    # Phase 1 hard benchmarks (mode-advantage signal)
    "gpqa", "simpleqa", "hotpotqa", "livecodebench",
    # Phase 2 hard benchmarks
    "debugbench", "usaco",
    # Phase 3: physics reasoning (mapped to adapters in get_adapter if available)
    "physics", "physreason",
}

# Suites that stay YAML-based (no public dataset or intentionally synthetic)
YAML_ONLY_SUITES = {
    "agentic", "long_context", "mode_advantage", "mode_advantage_hard", "skill_transfer",
}

# Suite → adapter class (single source of truth)
ADAPTER_CLASSES: dict[str, type[BaseAdapter]] = {
    "general": MMLUAdapter,
    "math": MathAdapter,
    "coder": CoderAdapter,
    "thinking": ThinkingAdapter,
    "instruction_precision": IFEvalAdapter,
    "vl": VLAdapter,
    "gaia": GaiaAdapter,
    "cruxeval": CRUXEvalAdapter,
    "bigcodebench": BigCodeBenchAdapter,
    # Phase 1 hard benchmarks
    "gpqa": GPQAAdapter,
    "simpleqa": SimpleQAAdapter,
    "hotpotqa": HotpotQAAdapter,
    "livecodebench": LiveCodeBenchAdapter,
    # Phase 2 hard benchmarks
    "debugbench": DebugBenchAdapter,
    "usaco": USACOAdapter,
}


def get_adapter(suite: str) -> Optional[BaseAdapter]:
    """Get the dataset adapter for a suite, or None if YAML-only / unmapped."""
    cls = ADAPTER_CLASSES.get(suite)
    if cls is None:
        return None
    return cls()
