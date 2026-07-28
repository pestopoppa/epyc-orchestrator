#!/usr/bin/env python3
"""Proposed minimal human-owned applier amendment: accept protocol v5."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


APPLIER = Path("/mnt/raid0/llm/epyc-root/artifacts/operator/apply_e8_quality_baseline_state.py")
spec = importlib.util.spec_from_file_location("e8_v5_base_applier", APPLIER)
if spec is None or spec.loader is None:
    raise SystemExit("ERROR: cannot import canonical E8 state applier")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.EXPECTED_PROTOCOL = "e8_quality_full_pool_tier_baseline.v5"


if __name__ == "__main__":
    raise SystemExit(module.main())
