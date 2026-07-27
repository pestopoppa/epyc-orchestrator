#!/usr/bin/env python3
"""Create an immutable terminal E8 repair source without changing staging."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


MODULE = Path(__file__).with_name("repair_e8_quality_baseline_multirow.py")
spec = importlib.util.spec_from_file_location("e8_multirow_repair", MODULE)
assert spec is not None and spec.loader is not None
repair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = repair
spec.loader.exec_module(repair)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repair.terminalize_source(args.staging_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
