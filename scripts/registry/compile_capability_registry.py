#!/usr/bin/env python3
"""Compile capability_registry.yaml into generated coordination surfaces."""

from __future__ import annotations

import argparse
import difflib
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


def _block_markers(target: str) -> tuple[str, str]:
    return (
        f"<!-- capability-registry:{target}:start -->",
        f"<!-- capability-registry:{target}:end -->",
    )


def _render_marked_block(target: str, rendered: str) -> str:
    start, end = _block_markers(target)
    return f"{start}\n{rendered.rstrip()}\n{end}"


def _replace_marked_block(text: str, *, target: str, rendered: str) -> str:
    start, end = _block_markers(target)
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count != 1 or end_count != 1:
        raise ValueError(
            f"expected exactly one {start!r} and one {end!r}; "
            f"found {start_count} and {end_count}"
        )
    start_index = text.index(start)
    end_index = text.index(end)
    if start_index > end_index:
        raise ValueError(f"marker {start!r} appears after {end!r}")

    return (
        text[:start_index]
        + _render_marked_block(target, rendered)
        + text[end_index + len(end) :]
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
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Omit to print to stdout.",
    )
    output_group.add_argument(
        "--replace-block",
        type=Path,
        default=None,
        help="Replace the marked capability-registry block in this markdown file.",
    )
    output_group.add_argument(
        "--check-block",
        type=Path,
        default=None,
        help="Check that this markdown file's marked block matches generated output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    capabilities = load_capability_registry(args.registry)
    if args.target == "action-availability":
        rendered = build_action_availability_section(capabilities)
    else:
        rendered = build_index_a_by_table(capabilities)

    if args.replace_block is not None:
        try:
            existing = args.replace_block.read_text(encoding="utf-8")
            updated = _replace_marked_block(
                existing, target=args.target, rendered=rendered
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        args.replace_block.write_text(updated, encoding="utf-8")
    elif args.check_block is not None:
        try:
            existing = args.check_block.read_text(encoding="utf-8")
            expected = _replace_marked_block(
                existing, target=args.target, rendered=rendered
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if existing != expected:
            diff = difflib.unified_diff(
                existing.splitlines(),
                expected.splitlines(),
                fromfile=str(args.check_block),
                tofile=f"generated:{args.target}",
                lineterm="",
            )
            print("\n".join(diff), file=sys.stderr)
            return 1
    elif args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
