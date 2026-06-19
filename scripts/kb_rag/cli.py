"""KB-RAG CLI.

Subcommands:
  build    — full or incremental rebuild of the index
  eval     — K7 retrieval-quality sweep over curated evidence cases
  query    — top-K MaxSim retrieval
  update   — re-encode a specified list of files (incremental)
  stats    — index summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when invoked directly.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.retrieval.kb_rag import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    CorpusConfig,
    build_index,
    query,
    remove_files,
    stats,
    update_files,
)

DEFAULT_CONFIG = _REPO / "config" / "kb_rag_config.yaml"
DEFAULT_EVAL_CASES = _REPO / "scripts" / "kb_rag" / "k7_seed_cases.json"
DEFAULT_MANIFEST_ROOT = Path("/workspace")


def _resolve_manifest_paths(sources: list, *, manifest_root: Path) -> list[str]:
    """Resolve manifest source rows to absolute paths."""
    root = manifest_root.expanduser().resolve()
    paths: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        raw_path = source.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        paths.append(str(path.expanduser().resolve()))
    return paths


def _paths_from_source_manifest(
    manifest_path: Path,
    *,
    manifest_root: Path = DEFAULT_MANIFEST_ROOT,
) -> tuple[list[str], list[str]]:
    """Extract updateable and removable file paths from a source manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"manifest sources must be a list: {manifest_path}")

    removed = manifest.get("removed_sources", [])
    if not isinstance(removed, list):
        removed = []
    return (
        _resolve_manifest_paths(sources, manifest_root=manifest_root),
        _resolve_manifest_paths(removed, manifest_root=manifest_root),
    )


def _cmd_build(args: argparse.Namespace) -> int:
    cfg = CorpusConfig.from_yaml(args.config or DEFAULT_CONFIG)
    result = build_index(cfg, index_dir=args.index_dir or DEFAULT_INDEX_DIR, force=args.force)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _cmd_update(args: argparse.Namespace) -> int:
    cfg = CorpusConfig.from_yaml(args.config or DEFAULT_CONFIG)
    paths = list(args.files or [])
    removed_paths: list[str] = []
    manifest_paths_count = 0
    if args.manifest:
        try:
            manifest_paths, removed_paths = _paths_from_source_manifest(
                Path(args.manifest).expanduser(),
                manifest_root=Path(args.manifest_root),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"manifest update input failed: {exc}", file=sys.stderr)
            return 1
        manifest_paths_count = len(manifest_paths)
        paths.extend(manifest_paths)
    if not paths and not removed_paths:
        print(
            "usage: update (--files file1.md [file2.md ...] | --manifest manifest.json)",
            file=sys.stderr,
        )
        return 1
    removed_result = None
    if removed_paths:
        removed_result = remove_files(
            removed_paths,
            index_dir=args.index_dir or DEFAULT_INDEX_DIR,
        )
        if not removed_result.get("ok"):
            print(json.dumps(removed_result, indent=2))
            return 1
    if paths:
        result = update_files(paths, cfg, index_dir=args.index_dir or DEFAULT_INDEX_DIR)
    else:
        result = {"ok": True, "files_processed": 0, "chunks_encoded": 0}
    if args.manifest:
        result["manifest"] = str(Path(args.manifest).expanduser())
        result["manifest_paths"] = manifest_paths_count
        result["manifest_removed_paths"] = len(removed_paths)
        if removed_result is not None:
            result["manifest_removed_result"] = removed_result
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _cmd_query(args: argparse.Namespace) -> int:
    rows = query(args.text, top_k=args.top_k, index_dir=args.index_dir or DEFAULT_INDEX_DIR)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        if not rows:
            print("(no results)")
            return 0
        for r in rows:
            crumb = " > ".join(r["heading_path"]) if r["heading_path"] else "(no headings)"
            print(f"\n[{r['score']:.4f}] {r['file']}:{r['line_range'][0]}-{r['line_range'][1]}")
            print(f"  {crumb}")
            print(f"  {r['snippet'][:200]}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from scripts.kb_rag.eval_k7 import run_eval

    summary = run_eval(
        cases_path=args.cases or DEFAULT_EVAL_CASES,
        index_dir=args.index_dir or DEFAULT_INDEX_DIR,
        output_dir=args.output_dir,
        configs=args.configs,
        cutoffs=args.cutoffs,
        top_k=args.top_k,
        limit_cases=args.limit_cases,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("ok") else 1


def _cmd_stats(args: argparse.Namespace) -> int:
    s = stats(index_dir=args.index_dir or DEFAULT_INDEX_DIR)
    print(json.dumps(s, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kb_rag", description="Internal KB-RAG CLI")
    p.add_argument("--index-dir", help=f"index directory (default: {DEFAULT_INDEX_DIR})")
    p.add_argument("--config", help=f"corpus config YAML (default: {DEFAULT_CONFIG})")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="(re)build index")
    pb.add_argument("--force", action="store_true", help="re-encode unchanged chunks too")
    pb.set_defaults(func=_cmd_build)

    pu = sub.add_parser("update", help="incremental re-encode of changed files")
    pu.add_argument("--files", nargs="+", help="files to refresh")
    pu.add_argument(
        "--manifest",
        help=(
            "project-wiki source manifest to refresh; use a "
            "--changed-since-manifest output for incremental updates"
        ),
    )
    pu.add_argument(
        "--manifest-root",
        default=str(DEFAULT_MANIFEST_ROOT),
        help=(
            "root for manifest-relative paths "
            f"(default: {DEFAULT_MANIFEST_ROOT})"
        ),
    )
    pu.set_defaults(func=_cmd_update)

    pq = sub.add_parser("query", help="top-K MaxSim retrieval")
    pq.add_argument("text", help="query text")
    pq.add_argument("--top-k", type=int, default=8)
    pq.add_argument("--json", action="store_true")
    pq.set_defaults(func=_cmd_query)

    pe = sub.add_parser("eval", help="K7 retrieval-quality sweep")
    pe.add_argument("--cases", help=f"case JSON file (default: {DEFAULT_EVAL_CASES})")
    pe.add_argument("--output-dir", help="directory for summary.json + rows.jsonl")
    pe.add_argument(
        "--configs",
        default="default",
        help="comma list of configs or default/all",
    )
    pe.add_argument("--cutoffs", default="3,5,10", help="comma-separated recall cutoffs")
    pe.add_argument("--top-k", type=int, help="query depth; defaults to max cutoff")
    pe.add_argument("--limit-cases", type=int, help="run only the first N cases")
    pe.set_defaults(func=_cmd_eval)

    ps = sub.add_parser("stats", help="index summary")
    ps.set_defaults(func=_cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
