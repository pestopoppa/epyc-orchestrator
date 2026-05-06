"""KB-RAG CLI.

Subcommands:
  build    — full or incremental rebuild of the index
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
    stats,
    update_files,
)

DEFAULT_CONFIG = _REPO / "config" / "kb_rag_config.yaml"


def _cmd_build(args: argparse.Namespace) -> int:
    cfg = CorpusConfig.from_yaml(args.config or DEFAULT_CONFIG)
    result = build_index(cfg, index_dir=args.index_dir or DEFAULT_INDEX_DIR, force=args.force)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _cmd_update(args: argparse.Namespace) -> int:
    cfg = CorpusConfig.from_yaml(args.config or DEFAULT_CONFIG)
    paths = args.files or []
    if not paths:
        print("usage: update --files file1.md [file2.md ...]", file=sys.stderr)
        return 1
    result = update_files(paths, cfg, index_dir=args.index_dir or DEFAULT_INDEX_DIR)
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
    pu.set_defaults(func=_cmd_update)

    pq = sub.add_parser("query", help="top-K MaxSim retrieval")
    pq.add_argument("text", help="query text")
    pq.add_argument("--top-k", type=int, default=8)
    pq.add_argument("--json", action="store_true")
    pq.set_defaults(func=_cmd_query)

    ps = sub.add_parser("stats", help="index summary")
    ps.set_defaults(func=_cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
