#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility facade for the dataset adapter package.

The 1761-line monolith was split into scripts/benchmark/dataset_adapter_modules/
during the 2026-05-22 Task-A refactor. This module re-exports every public name
the old monolith exposed (adapter classes, ADAPTER_SUITES, YAML_ONLY_SUITES,
get_adapter) so existing imports keep working:

    from dataset_adapters import get_adapter, ADAPTER_SUITES, BaseAdapter
    from scripts.benchmark.dataset_adapters import MMLUAdapter

Supported suites and their data sources (unchanged from the monolith):
  - general:              MMLU (cais/mmlu, 14,042 questions)
  - math:                 GSM8K (gsm8k, 1,319) + MATH-500 (HuggingFaceH4/MATH-500, 500)
  - coder:                HumanEval (openai_humaneval, 164) + MBPP (mbpp, 500)
  - thinking:             ARC-Challenge (allenai/ai2_arc, 1,172) + HellaSwag (Rowan/hellaswag, 10,042)
  - instruction_precision: IFEval (google/IFEval, 541)
  - vl:                   OCRBench + ChartQA (via extract_vl_debug_suite.py, 3,500)
  - agentic:              No public dataset (stays YAML-based)
  - long_context:         Synthetic (stays YAML-based)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))


from scripts.benchmark.dataset_adapter_modules.base import BaseAdapter
from scripts.benchmark.dataset_adapter_modules.coding import (
    BigCodeBenchAdapter,
    CoderAdapter,
    CRUXEvalAdapter,
    DebugBenchAdapter,
    LiveCodeBenchAdapter,
    USACOAdapter,
)
from scripts.benchmark.dataset_adapter_modules.general import (
    GPQAAdapter,
    HotpotQAAdapter,
    MMLUAdapter,
    SimpleQAAdapter,
)
from scripts.benchmark.dataset_adapter_modules.math_adapter import MathAdapter
from scripts.benchmark.dataset_adapter_modules.reasoning import (
    IFEvalAdapter,
    ThinkingAdapter,
)
from scripts.benchmark.dataset_adapter_modules.registry import (
    ADAPTER_CLASSES,
    ADAPTER_SUITES,
    YAML_ONLY_SUITES,
    get_adapter,
)
from scripts.benchmark.dataset_adapter_modules.vision_agentic import (
    GaiaAdapter,
    VLAdapter,
)


__all__ = [
    "ADAPTER_CLASSES",
    "ADAPTER_SUITES",
    "YAML_ONLY_SUITES",
    "BaseAdapter",
    "BigCodeBenchAdapter",
    "CoderAdapter",
    "CRUXEvalAdapter",
    "DebugBenchAdapter",
    "GaiaAdapter",
    "GPQAAdapter",
    "HotpotQAAdapter",
    "IFEvalAdapter",
    "LiveCodeBenchAdapter",
    "MMLUAdapter",
    "MathAdapter",
    "SimpleQAAdapter",
    "ThinkingAdapter",
    "USACOAdapter",
    "VLAdapter",
    "get_adapter",
]
