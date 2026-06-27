#!/usr/bin/env python3
"""Compile capability_registry.yaml into generated coordination surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.registry.capability_registry import (  # noqa: E402
    build_action_availability_section,
    build_index_a_by_table,
    load_capability_registry,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the capability registry into generated markdown."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to capability_registry.yaml; defaults to the repo registry.",
    )
    parser.add_argument(
        "--target",
        choices=("action-availability", "index-a-by"),
        default="action-availability",
        help="Generated surface to emit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Omit to print to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    capabilities = load_capability_registry(args.registry)
    if args.target == "action-availability":
        rendered = build_action_availability_section(capabilities)
    else:
        rendered = build_index_a_by_table(capabilities)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
