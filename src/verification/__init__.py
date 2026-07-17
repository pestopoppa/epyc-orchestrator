"""Execution-free verification signals (EV-12).

Public API for the Dockerless patch-verdict verifier. See
``patch_verifier`` for the check list, the
``orchestration/verification_report.schema.json`` alignment, and the eval-tower
hook contract consumed by Wave-2 B1.
"""

from __future__ import annotations

from src.verification.patch_verifier import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    Certificate,
    Check,
    PatchParseError,
    VerdictResult,
    parse_unified_diff,
    verify_patch,
)

__all__ = [
    "verify_patch",
    "VerdictResult",
    "Check",
    "Certificate",
    "PatchParseError",
    "parse_unified_diff",
    "PASS",
    "FAIL",
    "INCONCLUSIVE",
]
