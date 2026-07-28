#!/usr/bin/env python3
"""Read-only validator for the human-amended E8 final-c1 retry namespace."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


RUNNER_PATH = Path(__file__).with_name("final_c1_retry.py")


def _runner() -> Any:
    spec = importlib.util.spec_from_file_location("e8_final_c1_validator_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import final-c1 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(output_dir: Path, *, require_complete: bool = False) -> dict[str, Any]:
    return _runner().validate_output(output_dir.resolve(strict=True), require_complete=require_complete)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.output_dir, require_complete=args.require_complete), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
