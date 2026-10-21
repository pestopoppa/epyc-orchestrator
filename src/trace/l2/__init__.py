"""L2 mechanical failure-taxonomy scan (HIL-12 spec, HIL-13 implementation, HIL-14 calibration).

Read-only, zero inference. Declared inputs only: status, exception, category and count fields
plus the structured per-step fields of ``model.L2Step``. No decision-to-provenance edge exists in
the recorded traces (intake-2315#record), so every deviation is a *first visible deviation*, a
first-event label, never a counterfactual root cause.
"""
from src.trace.l2.model import L2Step, L2Trace  # noqa: F401
from src.trace.l2.detectors import (  # noqa: F401
    CATEGORIES, JUDGE_ONLY, Detection, DetectorConfig, detect,
)
