"""GitNexus <-> ColBERT KB federation query.

Joins EPYC's two *disjoint* knowledge indexes (the gap named in
``handoffs/active/internal-kb-rag.md:21`` — "GitNexus indexes code only;
markdown is invisible to the code-intelligence pipeline"):

  * ColBERT KB (markdown docs)  -> ``src.retrieval.kb_rag`` (imported READ-ONLY)
  * GitNexus code graph         -> the ``gitnexus`` CLI (subprocess, JSON output)

Two bidirectional joins:

  1. ``symbol_to_kb(symbol)`` — code symbol / file path -> ranked handoff / wiki /
     research / progress chunks that discuss it. Uses ``gitnexus context`` to pull
     the symbol's identity + call-neighbours, folds their names into a text query,
     and runs it through ``kb_rag.query()``. Answers "where did we discuss
     symbol / benchmark X?".

  2. ``doc_to_code(text)`` — a KB chunk or free-form doc phrase -> matching code
     symbols with ``file:line``. Extracts code-like identifiers from the text,
     resolves each via ``gitnexus context``, and also runs the whole phrase
     through ``gitnexus query`` (execution-flow / concept search) as a fuzzy
     fallback.

This module is **purely additive**: it imports ``kb_rag`` / ``colbert_encoder``
read-only and never mutates them or the on-disk index. It performs no inference
beyond the CPU ColBERT *query* encoder that ``kb_rag.query`` already uses.

Environment notes handled gracefully:
  * ``onnxruntime`` may be missing from the active venv. The encoder deps are
    discovered in three tiers (see ``ensure_encoder_importable``): a normal
    import first; then any dirs in ``FEDERATION_ORT_SITE_PACKAGES``
    (``:``-separated); then a bounded, last-resort search of *discovered*
    interpreter / sibling-venv site-packages — no absolute path is hardcoded as
    the mechanism. If it still can't load, ``symbol_to_kb`` degrades to a
    well-formed report with a reason instead of fabricating hits.
  * A given repo's GitNexus index may be unreadable (e.g. a corrupt LadybugDB
    WAL that segfaults the CLI). Per-repo failures are caught, recorded in
    ``notes``, and the other repos are still queried.

CLI:
    python -m src.retrieval.federation --symbol kb_rag.query
    python -m src.retrieval.federation --doc "iqk IQ-quant enablement"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Repos with a GitNexus index, in the order we probe them for a symbol.
DEFAULT_REPOS: tuple[str, ...] = (
    "epyc-orchestrator",
    "epyc-inference-research",
    "epyc-root",
)

# Where `gitnexus` is invoked from (it is cwd-independent, but be explicit).
_GITNEXUS_CWD = os.environ.get("FEDERATION_GITNEXUS_CWD", "/mnt/raid0/llm/epyc-root")
_GITNEXUS_BIN = os.environ.get("FEDERATION_GITNEXUS_BIN", "gitnexus")
_GITNEXUS_TIMEOUT = float(os.environ.get("FEDERATION_GITNEXUS_TIMEOUT", "90"))

# onnxruntime (needed by the CPU ColBERT encoder) may be absent from the active
# venv. It is located lazily in ``ensure_encoder_importable`` — no site-packages
# path is hardcoded here; discovery is env-driven then filesystem-derived.

# Make ``src...`` imports resolve when this module is run as a loose script.
_ORCH_ROOT = Path(__file__).resolve().parents[2]
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

# READ-ONLY import of the KB — never mutated by this module.
from src.retrieval import colbert_encoder, kb_rag  # noqa: E402


# --------------------------------------------------------------------------- #
# Encoder dependency bootstrap (onnxruntime may be missing from the venv)
# --------------------------------------------------------------------------- #

def _ort_fallback_site_packages() -> list[str]:
    """Tier-2 fallback: operator-provided ``FEDERATION_ORT_SITE_PACKAGES`` dirs."""
    raw = os.environ.get("FEDERATION_ORT_SITE_PACKAGES", "")
    return [p for p in raw.split(os.pathsep) if p]


def _discover_ort_site_packages() -> list[str]:
    """Tier-3 last-resort: *discover* site-packages dirs that hold onnxruntime.

    Only reached when onnxruntime is neither importable in the active
    interpreter nor found via ``FEDERATION_ORT_SITE_PACKAGES``. Searches a small,
    bounded set of candidate roots that are themselves discovered at call time
    (an activated virtualenv, this interpreter's own prefixes, and sibling venvs
    beside the repo) for a ``.../site-packages/onnxruntime`` directory — so no
    single absolute path is baked in as the mechanism.
    """
    import glob

    roots: list[Path] = []
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        roots.append(Path(venv))
    for pfx in (sys.prefix, sys.base_prefix, sys.exec_prefix):
        if pfx:
            roots.append(Path(pfx))
    # Sibling venvs — the common layout is a shared venv beside the repos
    # (e.g. <llm-root>/venv next to epyc-orchestrator).
    for base in (_ORCH_ROOT, _ORCH_ROOT.parent):
        for name in ("venv", ".venv", "env", ".env"):
            roots.append(base / name)

    seen: set[str] = set()
    found: list[str] = []
    for root in roots:
        try:
            if not root or not root.is_dir():
                continue
        except OSError:
            continue
        for pat in ("lib/python*/site-packages", "lib64/python*/site-packages",
                    "Lib/site-packages", "site-packages"):
            for sp in glob.glob(str(root / pat)):
                if sp in seen:
                    continue
                seen.add(sp)
                if (Path(sp) / "onnxruntime").is_dir():
                    found.append(sp)
    return found


def _import_ort_via(paths: Iterable[str]) -> bool:
    """Append any ``paths`` that hold onnxruntime to ``sys.path`` then re-import.

    Appends (never prepends) so it cannot shadow already-imported packages
    (numpy / tokenizers stay as-is). Returns True iff onnxruntime imports after.
    """
    added = False
    for sp in paths:
        if sp and (Path(sp) / "onnxruntime").is_dir() and sp not in sys.path:
            sys.path.append(sp)
            added = True
    if not added:
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_encoder_importable() -> bool:
    """Best-effort: make ``onnxruntime`` importable without installing anything.

    Three tiers, cheapest first: (1) a normal import from the active
    interpreter; (2) dirs from ``FEDERATION_ORT_SITE_PACKAGES``; (3) a bounded
    last-resort filesystem search of discovered site-packages. Returns True as
    soon as onnxruntime imports, False if all tiers fail.
    """
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        pass
    if _import_ort_via(_ort_fallback_site_packages()):
        return True
    if _import_ort_via(_discover_ort_site_packages()):
        return True
    return False


def encoder_status() -> dict[str, Any]:
    """Report whether the CPU ColBERT encoder + KB index are usable."""
    ort_ok = ensure_encoder_importable()
    model_on_disk = colbert_encoder.is_available()
    loaded = colbert_encoder.ensure_loaded() if (ort_ok and model_on_disk) else False
    index_dir = Path(kb_rag.DEFAULT_INDEX_DIR)
    index_present = (index_dir / "catalog.sqlite").exists()
    return {
        "onnxruntime_importable": ort_ok,
        "model_on_disk": model_on_disk,
        "encoder_loaded": loaded,
        "kb_index_present": index_present,
        "kb_index_dir": str(index_dir),
        "kb_queryable": bool(loaded and index_present),
    }


# --------------------------------------------------------------------------- #
# GitNexus CLI wrapper
# --------------------------------------------------------------------------- #

def _run_gitnexus(
    subcommand: str,
    positional: str | None,
    *,
    repo: str | None = None,
    extra: Iterable[str] = (),
) -> tuple[dict[str, Any] | None, str | None]:
    """Run a ``gitnexus`` subcommand and parse its JSON stdout.

    Returns ``(data, error)``. ``data`` is the parsed JSON (or None on any
    failure), ``error`` is a short human string when ``data`` is None. All
    stderr (WAL-recovery notices, GPU-discovery warnings) is discarded.
    A non-zero exit (e.g. 139 = segfault from a corrupt index) yields a clean
    error string rather than an exception.
    """
    cmd = [_GITNEXUS_BIN, subcommand]
    if positional is not None:
        cmd.append(positional)
    if repo:
        cmd += ["--repo", repo]
    cmd += list(extra)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_GITNEXUS_TIMEOUT,
            cwd=_GITNEXUS_CWD,
        )
    except FileNotFoundError:
        return None, f"'{_GITNEXUS_BIN}' not found on PATH"
    except subprocess.TimeoutExpired:
        return None, f"gitnexus {subcommand} timed out (repo={repo})"
    except Exception as e:  # noqa: BLE001 — defensive; caller degrades.
        return None, f"gitnexus {subcommand} raised {type(e).__name__}: {e}"

    if proc.returncode != 0:
        # Python reports a signal death as a negative returncode (SIGSEGV -> -11);
        # a shell would surface it as 128+signo (139). Treat both as a likely
        # corrupt/unreadable index so callers can surface a clear reason.
        if proc.returncode < 0 or proc.returncode == 139:
            signo = -proc.returncode if proc.returncode < 0 else proc.returncode - 128
            return None, (
                f"gitnexus {subcommand} died on signal {signo} "
                f"(repo={repo}; corrupt/unreadable index?)"
            )
        return None, f"gitnexus {subcommand} exit {proc.returncode} (repo={repo})"

    data = _parse_json_lenient(proc.stdout)
    if data is None:
        return None, f"gitnexus {subcommand} produced no JSON (repo={repo})"
    return data, None


def _parse_json_lenient(text: str) -> dict[str, Any] | None:
    """Parse JSON, tolerating a leading noise line before the first ``{``."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start > 0:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Symbol resolution helpers
# --------------------------------------------------------------------------- #

_CODE_EXTS = (".py", ".sh", ".cpp", ".c", ".h", ".hpp", ".cu", ".js", ".ts",
              ".mjs", ".yaml", ".yml", ".toml", ".go", ".rs")


def _looks_like_path(token: str) -> bool:
    return token.endswith(_CODE_EXTS) or "/" in token


def _split_symbol(symbol: str) -> tuple[str, str | None]:
    """Split a ``module.symbol`` / ``pkg.mod.func`` form into (name, file_hint).

    ``kb_rag.query`` -> ("query", "kb_rag"). A dotted *file* path
    (``foo.py``) is returned as (name=whole, file_hint=whole)."""
    if _looks_like_path(symbol):
        return symbol, symbol
    if "." in symbol:
        head, _, tail = symbol.rpartition(".")
        return tail, head
    return symbol, None


def _neighbor_names(ctx: dict[str, Any]) -> list[str]:
    """Collect distinct caller/callee names from a `gitnexus context` result."""
    names: list[str] = []
    seen: set[str] = set()
    for direction in ("incoming", "outgoing"):
        edges = ctx.get(direction) or {}
        if not isinstance(edges, dict):
            continue
        for edge_list in edges.values():
            if not isinstance(edge_list, list):
                continue
            for e in edge_list:
                nm = (e or {}).get("name")
                if nm and nm not in seen:
                    seen.add(nm)
                    names.append(nm)
    return names


def resolve_symbol(
    symbol: str,
    repos: Iterable[str] = DEFAULT_REPOS,
    *,
    file_hint: str | None = None,
) -> dict[str, Any]:
    """Resolve ``symbol`` to its GitNexus identity + neighbours across repos.

    Returns a dict:
        {resolved: {...}|None, neighbors: [...], repo: str|None,
         candidates: [...], notes: [...]}
    ``resolved`` is None when no repo could resolve it (degraded mode).
    """
    name, hint_from_dotted = _split_symbol(symbol)
    file_hint = file_hint or (hint_from_dotted if hint_from_dotted else None)

    notes: list[str] = []
    # Ambiguous candidates only get surfaced as a last resort (when no repo
    # yields an exact resolution), and only when a file hint was given — an
    # unhinted common name like "query" matches unrelated symbols everywhere.
    fallback_candidates: list[dict[str, Any]] = []
    for repo in repos:
        extra: list[str] = []
        if file_hint and _looks_like_path(file_hint):
            extra = ["--file", file_hint]
        data, err = _run_gitnexus("context", name, repo=repo, extra=extra)
        if data is None:
            notes.append(f"{repo}: {err}")
            continue

        status = data.get("status")
        if status == "found" and isinstance(data.get("symbol"), dict):
            sym = data["symbol"]
            return {
                "resolved": _symbol_record(sym, repo),
                "neighbors": _neighbor_names(data),
                "repo": repo,
                "candidates": [],
                "notes": notes,
            }
        if status == "ambiguous":
            cands = data.get("candidates") or []
            # Prefer a candidate whose file matches the dotted-module hint.
            chosen = _pick_candidate(cands, file_hint)
            if chosen is not None:
                # Re-resolve precisely by uid for full neighbour data.
                data2, _ = _run_gitnexus(
                    "context", None, repo=repo, extra=["--uid", chosen["uid"]]
                )
                neighbors = _neighbor_names(data2) if data2 else []
                return {
                    "resolved": _symbol_record(chosen, repo),
                    "neighbors": neighbors,
                    "repo": repo,
                    "candidates": [_symbol_record(c, repo) for c in cands[:8]],
                    "notes": notes,
                }
            notes.append(
                f"{repo}: ambiguous ({len(cands)} candidates, none matched hint "
                f"{file_hint!r})"
            )
            if file_hint:
                fallback_candidates.extend(_symbol_record(c, repo) for c in cands[:8])
            # keep probing other repos for an exact hit
            continue
        # error/not-found -> try next repo
        if "error" in data:
            notes.append(f"{repo}: {data['error']}")

    return {"resolved": None, "neighbors": [], "repo": None,
            "candidates": fallback_candidates[:8], "notes": notes}


def _symbol_record(sym: dict[str, Any], repo: str) -> dict[str, Any]:
    return {
        "repo": repo,
        "uid": sym.get("uid"),
        "name": sym.get("name"),
        "kind": sym.get("kind"),
        "file": sym.get("filePath"),
        "line_start": sym.get("startLine"),
        "line_end": sym.get("endLine"),
    }


def _pick_candidate(
    candidates: list[dict[str, Any]], file_hint: str | None
) -> dict[str, Any] | None:
    if not candidates:
        return None
    if file_hint:
        needle = Path(file_hint).stem.lower()
        for c in candidates:
            fp = (c.get("filePath") or "").lower()
            if needle and needle in fp:
                return c
    return None


# --------------------------------------------------------------------------- #
# Direction 1: symbol / file  ->  KB doc chunks
# --------------------------------------------------------------------------- #

def symbol_to_kb(
    symbol: str,
    top_k: int = 8,
    repos: Iterable[str] = DEFAULT_REPOS,
    *,
    file_hint: str | None = None,
    max_neighbors: int = 6,
) -> dict[str, Any]:
    """Given a code symbol / file, return ranked KB chunks discussing it.

    Folds the symbol's GitNexus identity (name, file basename, call-neighbours)
    into a text query and runs it through ``kb_rag.query()`` (read-only). If
    GitNexus cannot resolve the symbol, degrades to a name + file-token query.
    """
    repos = tuple(repos)
    res = resolve_symbol(symbol, repos, file_hint=file_hint)
    notes = list(res["notes"])
    degraded = res["resolved"] is None

    query_terms: list[str] = []
    resolved = res["resolved"]
    if resolved:
        if resolved.get("name"):
            query_terms.append(resolved["name"])
        if resolved.get("file"):
            query_terms.append(Path(resolved["file"]).stem.replace("_", " "))
        query_terms.extend(res["neighbors"][:max_neighbors])
    else:
        # Degraded: build a query straight from the requested symbol / file.
        # Split on '.', '_', '/' and drop code extensions so a dotted symbol
        # like "kb_rag.query" -> "kb rag query" (not just the bare "query").
        stem = symbol
        for ext in _CODE_EXTS:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        query_terms.extend(re.split(r"[._/]+", stem))
        notes.append("degraded: no GitNexus resolution; query built from raw symbol")

    query_text = " ".join(dict.fromkeys(t for t in query_terms if t)).strip()

    enc = encoder_status()
    kb_hits: list[dict[str, Any]] = []
    if not enc["kb_queryable"]:
        notes.append(
            "KB not queryable: "
            + ", ".join(
                k for k in ("onnxruntime_importable", "model_on_disk",
                            "kb_index_present") if not enc[k]
            )
            + " (no live ColBERT results)"
        )
    elif not query_text:
        notes.append("empty query text; nothing to search")
    else:
        raw = kb_rag.query(query_text, top_k=top_k)  # READ-ONLY
        kb_hits = [_kb_hit(h) for h in raw]

    return {
        "direction": "symbol_to_kb",
        "symbol": symbol,
        "resolved": resolved,
        "neighbors": res["neighbors"],
        "candidates": res["candidates"],
        "degraded": degraded,
        "query_text": query_text,
        "kb_hits": kb_hits,
        "encoder_status": enc,
        "notes": notes,
    }


def _kb_hit(h: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": h.get("file"),
        "heading": " > ".join(h.get("heading_path") or []),
        "line_range": list(h.get("line_range") or []),
        "score": h.get("score"),
        "snippet": (h.get("snippet") or "").replace("\n", " ")[:200],
    }


# --------------------------------------------------------------------------- #
# Direction 2: doc phrase / chunk  ->  code symbols
# --------------------------------------------------------------------------- #

# Dotted identifiers (a.b / foo.py), snake_case, CamelCase, or func().
_IDENT_RE = re.compile(
    r"""
    (?P<dotted>[A-Za-z_][\w/]*\.[A-Za-z_][\w./]*)   # module.symbol / file.py
    | (?P<snake>[A-Za-z_]*[a-z0-9]_[A-Za-z0-9_]+)   # snake_case (has an underscore)
    | (?P<camel>[A-Za-z]+[a-z]+[A-Z][A-Za-z0-9]+)   # CamelCase
    | (?P<call>[A-Za-z_][A-Za-z0-9_]{2,})(?=\s*\()  # something(
    """,
    re.VERBOSE,
)

# Very common English / prose tokens that are never worth resolving as symbols.
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "code", "doc", "docs",
    "handoff", "wiki", "research", "enablement", "support", "quant", "quants",
    "value", "values", "which", "where", "when", "into", "over", "under",
}


def extract_identifiers(text: str, limit: int = 12) -> list[str]:
    """Pull code-like candidate identifiers out of free-form doc text."""
    found: list[str] = []
    seen: set[str] = set()
    for m in _IDENT_RE.finditer(text or ""):
        tok = next(g for g in m.groups() if g is not None)
        tok = tok.strip(".")
        low = tok.lower()
        if low in _STOPWORDS or len(tok) < 3:
            continue
        # Keep dotted / underscored / mixed-case; drop bland lowercase words.
        is_codeish = (
            "." in tok or "_" in tok or "/" in tok
            or any(c.isupper() for c in tok[1:])
        )
        if not is_codeish:
            continue
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
        if len(found) >= limit:
            break
    return found


def doc_to_code(
    text: str,
    top_k: int = 8,
    repos: Iterable[str] = DEFAULT_REPOS,
    *,
    concept_search: bool = True,
) -> dict[str, Any]:
    """Given a doc chunk / phrase, return matching code symbols (file:line).

    Two resolution strategies, merged + de-duplicated:
      * exact — each extracted identifier via ``gitnexus context``
      * concept — the whole phrase via ``gitnexus query`` (execution-flow search)
    """
    repos = tuple(repos)
    identifiers = extract_identifiers(text)
    notes: list[str] = []
    hits: list[dict[str, Any]] = []
    seen_uids: set[str] = set()

    def _add(rec: dict[str, Any], via: str, identifier: str | None) -> None:
        uid = rec.get("uid")
        key = uid or f"{rec.get('file')}:{rec.get('line_start')}:{rec.get('name')}"
        if key in seen_uids:
            return
        seen_uids.add(key)
        hits.append({**rec, "via": via, "matched": identifier})

    # --- exact identifier resolution ---
    for ident in identifiers:
        r = resolve_symbol(ident, repos)
        if r["resolved"]:
            _add(r["resolved"], "context", ident)
        else:
            for c in r["candidates"][:3]:
                _add(c, "context-ambiguous", ident)
        notes.extend(n for n in r["notes"] if "corrupt/unreadable" in n)  # surface crashes

    # --- concept / execution-flow search over the whole phrase ---
    if concept_search and text.strip():
        for repo in repos:
            data, err = _run_gitnexus(
                "query", text.strip(), repo=repo,
                extra=["--limit", str(max(3, top_k // 2))],
            )
            if data is None:
                if err and "corrupt/unreadable" in err:
                    notes.append(f"{repo}: {err}")
                continue
            for d in (data.get("definitions") or []):
                _add(
                    {
                        "repo": repo,
                        "uid": d.get("id"),
                        "name": d.get("name"),
                        "kind": (d.get("id", "").split(":", 1)[0] or None),
                        "file": d.get("filePath"),
                        "line_start": d.get("startLine"),
                        "line_end": d.get("endLine"),
                    },
                    "query", None,
                )

    if not hits:
        notes.append("no code symbols matched (identifiers may live in an "
                     "un-indexed repo, e.g. llama.cpp)")

    # De-dup notes, keep order.
    notes = list(dict.fromkeys(notes))
    return {
        "direction": "doc_to_code",
        "input": text,
        "identifiers": identifiers,
        "code_hits": hits[:top_k],
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Pretty printing + CLI
# --------------------------------------------------------------------------- #

def _format_symbol_to_kb(r: dict[str, Any]) -> str:
    out = [f"symbol->KB: {r['symbol']}"]
    res = r["resolved"]
    if res:
        out.append(f"  resolved: {res['kind']} {res['name']}  "
                   f"{res['file']}:{res['line_start']}  [{res['repo']}]")
        if r["neighbors"]:
            out.append(f"  neighbours: {', '.join(r['neighbors'][:8])}")
    else:
        out.append("  resolved: <none> (degraded)")
        if r["candidates"]:
            out.append(f"  candidates: "
                       + "; ".join(f"{c['name']} ({c['file']})"
                                   for c in r['candidates'][:4]))
    out.append(f"  query_text: {r['query_text']!r}")
    out.append(f"  KB hits ({len(r['kb_hits'])}):")
    for h in r["kb_hits"]:
        fname = Path(h["file"]).name if h["file"] else "?"
        lr = h["line_range"]
        out.append(f"    [{h['score']}] {fname}  ({lr[0]}-{lr[1]})  {h['heading']}")
    for n in r["notes"]:
        out.append(f"  note: {n}")
    return "\n".join(out)


def _format_doc_to_code(r: dict[str, Any]) -> str:
    out = [f"doc->code: {r['input']!r}"]
    out.append(f"  identifiers: {', '.join(r['identifiers']) or '<none>'}")
    out.append(f"  code hits ({len(r['code_hits'])}):")
    for h in r["code_hits"]:
        loc = f"{h['file']}:{h['line_start']}" if h["file"] else "?"
        out.append(f"    {h.get('name')}  [{h.get('kind')}]  {loc}  "
                   f"({h['via']}{'/'+h['matched'] if h.get('matched') else ''}) [{h.get('repo')}]")
    for n in r["notes"]:
        out.append(f"  note: {n}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="federation",
        description="GitNexus <-> ColBERT KB federation query (bidirectional).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--symbol", metavar="SYM",
                      help="code symbol or file -> KB doc chunks "
                           "(e.g. kb_rag.query, src/retrieval/kb_rag.py)")
    mode.add_argument("--doc", metavar="TEXT",
                      help="doc phrase / KB chunk -> matching code symbols")
    parser.add_argument("--repo", action="append", dest="repos",
                        help="restrict to repo(s); repeatable "
                             "(default: all indexed repos)")
    parser.add_argument("--file-hint", help="disambiguate --symbol by file path")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    args = parser.parse_args(argv)

    repos = tuple(args.repos) if args.repos else DEFAULT_REPOS

    if args.symbol:
        result = symbol_to_kb(args.symbol, top_k=args.top_k, repos=repos,
                              file_hint=args.file_hint)
        text = _format_symbol_to_kb(result)
    else:
        result = doc_to_code(args.doc, top_k=args.top_k, repos=repos)
        text = _format_doc_to_code(result)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
