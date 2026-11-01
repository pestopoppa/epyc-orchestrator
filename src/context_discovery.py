"""DCP-2/DCP-3: candidate discovery + codemap producer for budget-bounded pre-assembly.

Per `delegation-context-preassembly.md` (intake-605). Turns a sub-task description into a
budget-bounded `ContextBundle` (the *assemble* side):

  discover (DCP-2)  →  cost (read survivors, DCP-2 pass 2)  →  pack (context_assembly.pack_to_budget)

Read-only integration. Backends are INJECTED so the core is pure + unit-testable:
- `code_search_fn(query, limit) -> list[DiscoveredHit | dict]` — ranked code hits. The orchestrator
  wires this from its ColGREP mixin via `parse_colgrep_json(self._code_search(q, limit=k))`.
- `file_reader_fn(path) -> str` — reads a file body (orchestrator wires the workspace reader).

DCP-3 codemap (`build_python_codemap`) is a self-contained `ast`-based signature extractor — no
GitNexus runtime dependency (GitNexus is a dev-time CLI/MCP tool, not importable here). It yields
signature-only skeletons (classes/functions, no bodies) — the "CodeMaps as a separate budget
class" idea, dependency-free.

The thin live wiring (pass the orchestrator's ColGREP + workspace reader, attach at the
dispatcher) is DCP-4 — a reviewed hook, since it touches the live delegation path.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass, field

from src.context_assembly import (
    Candidate,
    BundleEntry,
    ContextBundle,
    BudgetBands,
    LineRange,
    InclusionMode,
    SourceKind,
    merge_line_ranges,
    conservative_char_estimator,
    pack_to_budget,
    default_exclusion_reason,
)


@dataclass
class DiscoveredHit:
    path: str
    line_ranges: list[LineRange] = field(default_factory=list)
    score: float = 0.0


def _finite_score_or_zero(value) -> float:
    """Convert one score to a finite float, preserving the existing neutral fallback."""
    try:
        score = float(value)
    except (OverflowError, TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


# ─── DCP-2 pass 1: discovery (cheap metadata, no body reads) ─────────────────────


def parse_colgrep_json(payload: str | list) -> list[DiscoveredHit]:
    """Parse ColGREP `--json` output (or an already-decoded list) into DiscoveredHits.

    Tolerant of field-name variants (path/file, start_line/start, score/relevance) and of the
    orchestrator's wrapped string output. Bad items are skipped, not fatal.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload) if payload.strip() else []
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):  # some wrappers nest under "results"
        payload = payload.get("results") or payload.get("matches") or []
    hits: list[DiscoveredHit] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("file") or item.get("filepath")
        if not path:
            continue
        start = item.get("start_line") or item.get("start") or item.get("line")
        end = item.get("end_line") or item.get("end") or start
        ranges: list[LineRange] = []
        if isinstance(start, int) and isinstance(end, int) and start >= 1 and end >= start:
            ranges = [LineRange(start, end)]
        score = item.get("score", item.get("relevance", item.get("rank_score", 0.0)))
        hits.append(
            DiscoveredHit(path=path, line_ranges=ranges, score=_finite_score_or_zero(score))
        )
    return hits


def discover_candidates(
    query: str,
    *,
    code_search_fn,
    limit: int = 20,
    max_files: int = 8,
    exclude_by_policy: bool = True,
) -> list[DiscoveredHit]:
    """DCP-2 pass 1: rank candidate files by code-search score, merge per-file line ranges.

    Groups multiple hits in the same file into one DiscoveredHit (max score, merged ranges),
    drops policy-excluded paths (binaries/vendored/secrets — capped before any body read), and
    returns the top `max_files` by score. No file bodies are read here (cheap pass).
    """
    raw = code_search_fn(query, limit)
    hits = raw if raw and isinstance(raw[0], DiscoveredHit) else parse_colgrep_json(raw)

    by_path: dict[str, DiscoveredHit] = {}
    for h in hits:
        if exclude_by_policy and default_exclusion_reason(h.path) is not None:
            continue
        score = _finite_score_or_zero(h.score)
        cur = by_path.get(h.path)
        if cur is None:
            by_path[h.path] = DiscoveredHit(
                path=h.path, line_ranges=list(h.line_ranges), score=score
            )
        else:
            cur.line_ranges.extend(h.line_ranges)
            cur.score = max(cur.score, score)
    for h in by_path.values():
        h.line_ranges = merge_line_ranges(h.line_ranges)
    ranked = sorted(by_path.values(), key=lambda h: (-h.score, h.path))
    return ranked[:max_files]


# ─── DCP-3: codemap producer (ast-based signature skeleton, no GitNexus runtime dep) ──


def build_python_codemap(source: str, *, max_doc_chars: int = 80) -> str | None:
    """Signature-only skeleton of a Python source (classes/functions + first docstring line).

    Returns None on syntax error (caller falls back to slices/full). No bodies → token-cheap
    architectural context. This is the dependency-free CodeMaps analog (DCP-3).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    out: list[str] = []

    def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str) -> str:
        kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{indent}{kw} {node.name}({args}){ret}: ..."

    def _doc(node, indent: str) -> None:
        d = ast.get_docstring(node)
        if d:
            first = d.strip().splitlines()[0][:max_doc_chars]
            out.append(f"{indent}# {first}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_sig(node, ""))
            _doc(node, "    ")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            out.append(f"class {node.name}({bases}):" if bases else f"class {node.name}:")
            _doc(node, "    ")
            members = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not members:
                out.append("    ...")
            for m in members:
                out.append(_sig(m, "    "))
    return "\n".join(out) if out else None


def _python_edit_segments(source: str, hit_ranges: list[LineRange]):
    """Return hit AST segments, imports, and enclosing class headers with source locations.

    The returned rendered text is made only from verbatim source lines. ``None`` means that
    Python parsing or range resolution failed, so callers must not present a line fragment as
    AST-backed evidence.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None

    lines = source.splitlines()
    if not lines or not hit_ranges:
        return None

    def span(node):
        decorators = getattr(node, "decorator_list", ())
        start = min([node.lineno, *(d.lineno for d in decorators)])
        return start, getattr(node, "end_lineno", None) or node.lineno

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    # Module imports are shared setup context. Local imports in unrelated functions are not.
    imports = [span(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    selected: list[tuple[int, int]] = []
    enclosing_classes: set[ast.ClassDef] = set()
    statements = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.stmt) and not isinstance(node, ast.ClassDef)
    ]
    functions = [
        node for node in statements
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for hit in hit_ranges:
        if hit.start < 1 or hit.end > len(lines) or hit.start > hit.end:
            return None
        for line_no in range(hit.start, hit.end + 1):
            function_hits = [node for node in functions if span(node)[0] <= line_no <= span(node)[1]]
            if function_hits:
                # Keep the innermost enclosing function/method, including decorators and body.
                node = min(function_hits, key=lambda item: span(item)[1] - span(item)[0])
            else:
                # Module/class-level edits retain the smallest complete Python statement.
                hit_nodes = [node for node in statements if span(node)[0] <= line_no <= span(node)[1]]
                if not hit_nodes:
                    return None
                node = min(hit_nodes, key=lambda item: span(item)[1] - span(item)[0])
            selected.append(span(node))
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.ClassDef):
                    enclosing_classes.add(parent)
                parent = parents.get(parent)

    class_headers = []
    for node in enclosing_classes:
        body_start = min(
            (span(child)[0] for child in node.body),
            default=node.end_lineno or node.lineno,
        )
        header_start, _ = span(node)
        class_headers.append((header_start, max(header_start, body_start - 1)))
    segments = merge_line_ranges(
        [LineRange(a, b) for a, b in (*selected, *imports, *class_headers) if a >= 1 and b >= a]
    )
    if not segments:
        return None

    return _render_planned_ranges(source, segments), segments


def _render_planned_ranges(source: str, ranges: list[LineRange]) -> str:
    """Render finalized source ranges without treating them as fresh AST hit locations."""
    lines = source.splitlines(keepends=True)
    marker_newline = "\r\n" if "\r\n" in source else "\n"
    pieces: list[str] = []
    cursor = 1
    for segment in ranges:
        if segment.start > cursor:
            pieces.append(f"# ... {segment.start - cursor} lines omitted{marker_newline}")
        pieces.append(f"# Lines {segment.start}-{segment.end}{marker_newline}")
        source_text = "".join(lines[segment.start - 1:segment.end])
        pieces.append(source_text)
        if source_text and not source_text.endswith(("\n", "\r")):
            pieces.append(marker_newline)
        cursor = segment.end + 1
    if cursor <= len(lines):
        pieces.append(f"# ... {len(lines) - cursor + 1} lines omitted{marker_newline}")
    return "".join(pieces)


def _rendered_block(path: str, mode: str, text: str) -> str:
    return f"### {path} ({mode})\n{text}"


# ─── DCP-2 pass 2: cost survivors → Candidates ───────────────────────────────────


def cost_candidates(
    hits: list[DiscoveredHit],
    *,
    file_reader_fn,
    token_estimator=conservative_char_estimator,
    codemap_fn=build_python_codemap,
) -> list[Candidate]:
    """DCP-2 pass 2: read each survivor and compute per-mode token costs → Candidates.

    cost_full is the whole file; Python hit slices cost the exact AST-expanded rendering,
    including imports, class headers, line labels, and omission markers. Non-Python or
    unparseable hit files are FULL-only and use the whole-file cost. `priority` = hit score.
    Files the reader can't read are omitted here so assemble_delegation_bundle can record them
    as missing evidence in the returned bundle manifest.
    """
    out: list[Candidate] = []
    for h in hits:
        try:
            body = file_reader_fn(h.path)
        except Exception:
            continue
        if body is None:
            continue
        content_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
        cost_full = token_estimator(
            _rendered_block(h.path, InclusionMode.FULL, body) + "\n\n"
        )
        render_ranges_verbatim = False

        if h.line_ranges:
            edit_segments = (
                _python_edit_segments(body, h.line_ranges) if h.path.endswith(".py") else None
            )
            if edit_segments is not None:
                sliced, planned_ranges = edit_segments
                cost_slices = token_estimator(
                    _rendered_block(h.path, InclusionMode.SLICES, sliced) + "\n\n"
                )
                ranges = planned_ranges
                render_ranges_verbatim = True
                desired = InclusionMode.SLICES
            else:
                # Non-Python and unparseable files may enter only as a complete file.
                cost_slices = cost_full
                ranges = list(h.line_ranges)
                desired = InclusionMode.FULL
        else:
            cost_slices = cost_full
            ranges = []
            desired = InclusionMode.FULL

        codemap = codemap_fn(body) if h.path.endswith(".py") else None
        cost_codemap = (
            token_estimator(
                _rendered_block(h.path, InclusionMode.CODEMAP_ONLY, codemap) + "\n\n"
            )
            if codemap
            else cost_full
        )

        out.append(
            Candidate(
                path=h.path,
                priority=h.score,
                cost_full=cost_full,
                cost_slices=cost_slices,
                cost_codemap=cost_codemap,
                desired_mode=desired,
                line_ranges=ranges,
                content_sha256=content_sha256,
                source=SourceKind.COLGREP,
                render_ranges_verbatim=render_ranges_verbatim,
            )
        )
    return out


def assemble_delegation_bundle(
    query: str,
    budget: int,
    *,
    code_search_fn,
    file_reader_fn,
    bands: BudgetBands | None = None,
    bundle_id: str | None = None,
    repo_sha: str | None = None,
    limit: int = 20,
    max_files: int = 8,
) -> ContextBundle:
    """End-to-end DCP-2: discover → cost → pack into a budget-bounded ContextBundle.

    Pure given the injected `code_search_fn` + `file_reader_fn` (so it is fully unit-testable).
    The live wiring that supplies the orchestrator's ColGREP + workspace reader and attaches the
    bundle at delegation is DCP-4 (a reviewed hook).
    """
    hits = discover_candidates(
        query, code_search_fn=code_search_fn, limit=limit, max_files=max_files
    )
    candidates = cost_candidates(hits, file_reader_fn=file_reader_fn)
    bundle = pack_to_budget(candidates, budget, bands=bands, bundle_id=bundle_id, repo_sha=repo_sha)
    represented = {candidate.path for candidate in candidates}
    present = {entry.path for entry in bundle.entries}
    for hit in hits:
        if hit.path not in represented and hit.path not in present:
            bundle.entries.append(
                BundleEntry(
                    path=hit.path,
                    mode=InclusionMode.EXCLUDED,
                    source=SourceKind.COLGREP,
                    line_ranges=list(hit.line_ranges),
                    reason_downgraded_or_excluded="missing evidence: source file could not be read",
                )
            )
            present.add(hit.path)
    return bundle


# ─── DCP-4: render a packed bundle to prompt text ────────────────────────────────


def render_bundle(
    bundle: ContextBundle,
    *,
    file_reader_fn,
    codemap_fn=build_python_codemap,
) -> str:
    """Render a packed ContextBundle's *included* entries to prompt text (DCP-4).

    The bundle is a budget-bounded *plan* (which file at which inclusion mode); this reads
    each included entry's body via `file_reader_fn` and materializes it per mode: FULL = whole
    file, SLICES = AST edit segments and source context for ColGREP entries (ordinary source
    entries retain line-slice behavior), CODEMAP_ONLY = the ast signature skeleton. Entries
    whose body can't be read or rendered are marked EXCLUDED with a missing-evidence reason.
    Pure given the injected reader.
    """
    blocks: list[str] = []
    for e in bundle.included():
        try:
            body = file_reader_fn(e.path)
        except Exception:
            body = None
        if body is None:
            e.mode = InclusionMode.EXCLUDED
            e.reason_downgraded_or_excluded = (
                "missing evidence: source file could not be read during rendering"
            )
            continue
        if e.content_sha256 is not None:
            rendered_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if rendered_sha256 != e.content_sha256:
                raise ValueError(
                    f"content changed after context planning for {e.path!r}: "
                    f"expected {e.content_sha256}, got {rendered_sha256}"
                )
        if not body:
            e.mode = InclusionMode.EXCLUDED
            e.reason_downgraded_or_excluded = (
                "missing evidence: source file is empty during rendering"
            )
            continue
        if e.mode == InclusionMode.FULL:
            text = body
        elif e.mode == InclusionMode.SLICES:
            if e.source == SourceKind.COLGREP:
                lines = body.splitlines(keepends=True)
                canonical_ranges = merge_line_ranges(e.line_ranges)
                if (
                    not e.render_ranges_verbatim
                    or e.content_sha256 is None
                    or not e.line_ranges
                    or e.line_ranges != canonical_ranges
                    or any(
                        item.start < 1 or item.end < item.start or item.end > len(lines)
                        for item in e.line_ranges
                    )
                ):
                    e.mode = InclusionMode.EXCLUDED
                    e.reason_downgraded_or_excluded = (
                        "missing evidence: ColGREP slices lack a valid pinned AST range plan"
                    )
                    continue
                text = _render_planned_ranges(body, e.line_ranges)
            elif e.line_ranges:
                lines = body.splitlines()
                text = "\n".join(
                    "\n".join(lines[r.start - 1:r.end]) for r in e.line_ranges
                )
            else:
                text = ""
        elif e.mode == InclusionMode.CODEMAP_ONLY:
            text = (codemap_fn(body) if e.path.endswith(".py") else None) or ""
        else:
            text = ""
        if not text.strip():
            e.mode = InclusionMode.EXCLUDED
            e.reason_downgraded_or_excluded = (
                "missing evidence: selected render mode produced no content"
            )
            continue
        blocks.append(f"### {e.path} ({e.mode})\n{text}")
    return "\n\n".join(blocks)
