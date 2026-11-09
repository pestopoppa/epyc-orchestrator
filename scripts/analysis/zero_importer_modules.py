#!/usr/bin/env python3
"""List `src/` modules with zero production importers (autopilot-program RTG-02 box).

The prose claim "ten fully-built, zero-production-importer modules" was never enumerated and
its count depended on the method. This script is the deterministic enumeration: it reads the
WORKING TREE (no per-file `git show`), parses every Python file once, and applies two
independent methods:

* dotted -- AST-resolved imports (absolute and relative, ``from pkg import mod`` counted as
  ``pkg.mod``) plus string constants that look like a dotted ``src.*`` module path (dynamic
  ``importlib.import_module`` / ``python -m`` references);
* stem   -- the module's last name component appears anywhere in another file's import
  statements (over-matches common words; catches bare ``import foo`` from script directories).

A module is a CANDIDATE when either method finds no production importer. The intersection
(both methods) is the defensible floor, the union the ceiling. Declared entry points
(``[project.scripts]`` in pyproject.toml plus ``--entry-point``, default ``src.mcp_server``) are
excluded and listed separately: zero importers is correct for them. Every remaining candidate
gets a ``wire-or-retire`` column that is deliberately left for a per-module read; nothing here
deletes or decides.

Production importers = other files under ``src/``, ``scripts/`` and ``orchestration/``; tests
are counted separately. Read-only; no network, no inference.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
PROD_DIRS = ("src", "scripts", "orchestration")
TEST_DIRS = ("tests",)
DEFAULT_ENTRY_POINTS = ("src.mcp_server",)
_DOTTED_STRING = re.compile(r"^src(\.[A-Za-z_]\w*)+$")


@dataclass
class FileFacts:
    path: Path
    rel: str
    dotted: set[str] = field(default_factory=set)  # module names this file imports
    tokens: set[str] = field(default_factory=set)  # every name component in its imports
    has_main: bool = False


def module_name_for(rel: str) -> str:
    parts = Path(rel).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(rel: str) -> list[str]:
    # package containing the module (for an __init__.py, the package it defines)
    return list(Path(rel).with_suffix("").parts)[:-1]


def _resolve_from(rel: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module or ""
    package = _package_of(rel)
    if node.level - 1 > len(package):
        return None
    base = package[: len(package) - (node.level - 1)]
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def parse_file(path: Path, rel: str) -> FileFacts | None:
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None
    facts = FileFacts(path=path, rel=rel)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                facts.dotted.add(alias.name)
                facts.tokens.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(rel, node)
            if node.module:
                facts.tokens.update(node.module.split("."))
            if base is None:
                continue
            if base:
                facts.dotted.add(base)
            for alias in node.names:
                if alias.name != "*":
                    facts.dotted.add(f"{base}.{alias.name}" if base else alias.name)
                    facts.tokens.add(alias.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _DOTTED_STRING.match(node.value):
                facts.dotted.add(node.value)
                facts.tokens.add(node.value.rsplit(".", 1)[-1])
        elif isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                facts.has_main = True
    return facts


def load_tree(root: Path, dirs: tuple[str, ...]) -> list[FileFacts]:
    out: list[FileFacts] = []
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in {"__pycache__", ".venv", "node_modules"} for part in path.parts):
                continue
            rel = path.relative_to(root).as_posix()
            facts = parse_file(path, rel)
            if facts is not None:
                out.append(facts)
    return out


LAUNCH_SUFFIXES = {".sh", ".yaml", ".yml", ".toml", ".service", ".cfg", ".ini"}
LAUNCH_DIRS = ("scripts", "orchestration", "deploy", ".github")
MAX_LAUNCH_FILE_BYTES = 2_000_000


def load_launch_texts(root: Path) -> list[tuple[str, str]]:
    """Non-Python files that can launch a module (`python -m src.x`, systemd units, CI)."""
    out: list[tuple[str, str]] = []
    paths = [p for p in root.iterdir() if p.is_file()] if root.is_dir() else []
    for d in LAUNCH_DIRS:
        base = root / d
        if base.is_dir():
            paths.extend(p for p in base.rglob("*") if p.is_file())
    for path in sorted(paths):
        if path.suffix not in LAUNCH_SUFFIXES and path.name != "Makefile":
            continue
        try:
            if path.stat().st_size > MAX_LAUNCH_FILE_BYTES:
                continue
            out.append((path.relative_to(root).as_posix(), path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return out


def declared_entry_points(root: Path, extra: tuple[str, ...]) -> dict[str, str]:
    """module -> reason, from pyproject [project.scripts] plus explicit extras."""
    entries: dict[str, str] = {m: "explicit entry point" for m in extra}
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        for name, target in (data.get("project", {}).get("scripts", {}) or {}).items():
            module = str(target).split(":", 1)[0].strip()
            if module:
                entries[module] = f"pyproject console script `{name}`"
    return entries


def build_report(root: Path, entry_points: tuple[str, ...] = DEFAULT_ENTRY_POINTS) -> dict:
    prod = load_tree(root, PROD_DIRS)
    tests = load_tree(root, TEST_DIRS)
    entries = declared_entry_points(root, entry_points)
    launch_texts = load_launch_texts(root)

    modules = [
        f for f in prod
        if f.rel.startswith("src/") and Path(f.rel).name not in {"__init__.py", "__main__.py"}
    ]
    rows: list[dict] = []
    excluded_entries: list[dict] = []
    for mod in modules:
        name = module_name_for(mod.rel)
        stem = name.rsplit(".", 1)[-1]
        ast_prod = sorted(f.rel for f in prod if f.rel != mod.rel and name in f.dotted)
        stem_prod = sorted(f.rel for f in prod if f.rel != mod.rel and stem in f.tokens)
        ast_tests = sorted(f.rel for f in tests if name in f.dotted)
        stem_tests = sorted(f.rel for f in tests if stem in f.tokens)
        zero_ast, zero_stem = not ast_prod, not stem_prod
        if not (zero_ast or zero_stem):
            continue
        row = {
            "module": name,
            "path": mod.rel,
            "zero_by_dotted": zero_ast,
            "zero_by_stem": zero_stem,
            "method": "both" if zero_ast and zero_stem else ("dotted-only" if zero_ast else "stem-only"),
            "test_importers_dotted": len(ast_tests),
            "test_importers_stem": len(stem_tests),
            "has_main_guard": mod.has_main,
        }
        if name in entries:
            row["entry_point_reason"] = entries[name]
            excluded_entries.append(row)
            continue
        launch_refs = sorted(rel for rel, text in launch_texts if name in text or mod.rel in text)
        row["non_python_launch_refs"] = launch_refs[:5]
        row["non_python_launch_ref_count"] = len(launch_refs)
        row["wire_or_retire"] = (
            "per-module read required: referenced by a non-Python launcher, confirm it is a CLI/server"
            if launch_refs
            else "per-module read required (wire it, or retire it and correct its docstring)"
        )
        rows.append(row)
    both = [r for r in rows if r["method"] == "both"]
    return {
        "schema_version": 1,
        "root": str(root),
        "modules_scanned": len(modules),
        "prod_files_parsed": len(prod),
        "test_files_parsed": len(tests),
        "candidates": rows,
        "counts": {
            "candidates_union": len(rows),
            "candidates_floor_both_methods": len(both),
            "excluded_entry_points": len(excluded_entries),
        },
        "excluded_entry_points": excluded_entries,
        "method_note": (
            "dotted = AST-resolved import / dotted string constant; stem = last name appears in another "
            "file's import statements (over-matches common words). Floor = zero by both; union = zero by either."
        ),
    }


def render_markdown(report: dict) -> str:
    c = report["counts"]
    lines = [
        "# Zero-production-importer modules",
        "",
        f"Scanned {report['modules_scanned']} `src/` modules against {report['prod_files_parsed']} "
        f"production and {report['test_files_parsed']} test files.",
        f"Candidates: **{c['candidates_union']}** (union), **{c['candidates_floor_both_methods']}** "
        f"zero by both methods (floor); {c['excluded_entry_points']} declared entry point(s) excluded.",
        "",
        "| module | method | test importers (dotted/stem) | main guard | launcher refs | wire-or-retire |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["candidates"]:
        lines.append(
            f"| `{r['module']}` | {r['method']} | {r['test_importers_dotted']}/{r['test_importers_stem']} "
            f"| {'yes' if r['has_main_guard'] else 'no'} | {r['non_python_launch_ref_count']} "
            f"| {r['wire_or_retire']} |"
        )
    if report["excluded_entry_points"]:
        lines += ["", "Excluded entry points (zero importers is correct):", ""]
        for r in report["excluded_entry_points"]:
            lines.append(f"- `{r['module']}`: {r['entry_point_reason']}")
    lines += ["", report["method_note"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--entry-point", action="append", default=None,
                    help="extra module to treat as an entry point (default: src.mcp_server)")
    ap.add_argument("--format", choices=("md", "json"), default="md")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    entry = tuple(args.entry_point) if args.entry_point else DEFAULT_ENTRY_POINTS
    report = build_report(args.root.resolve(), entry)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.format == "json" else render_markdown(report)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
