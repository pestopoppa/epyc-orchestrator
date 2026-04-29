"""Code and document search via NextPLAID multi-vector retrieval.

Provides mixin with: code_search, doc_search.

These tools complement recall() (episodic memory) by searching actual source
code and documentation using token-level ColBERT matching — finding specific
function names, class definitions, and code patterns rather than past routing
decisions.

Phase 5 architecture: two NextPLAID containers with specialized models.
  :8088  nextplaid-code   LateOn-Code (130M, 128-dim, INT8)   → code index (AST-chunked)
  :8089  nextplaid-docs   answerai-colbert-small-v1-onnx      → docs index

Degrades gracefully: if docs container down, falls back to code container.

code_search() routes through the ColGREP CLI binary
(github.com/lightonai/next-plaid) by default. ColGREP is the same ColBERT
family with hybrid FTS5 keyword fusion + tree-sitter chunking, runs as a
single Rust binary, and falls back to CPU on hosts without CUDA. To opt
back into NextPLAID for code_search, set REPL_COLGREP=0 (also accepts
"false"/"off"). doc_search() always uses NextPLAID (ColGREP is code-focused).

Default flipped 2026-04-29 after a 14-query paired A/B showed colgrep
top-1 = 10/14 (71%) vs NextPLAID top-1 = 2/14 (14%) on canonical
production-code queries. See handoffs/active/repl-turn-efficiency.md S7.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any


logger = logging.getLogger(__name__)

CODE_SEARCH_URL = "http://localhost:8088"
DOC_SEARCH_URL = "http://localhost:8089"
VALID_INDICES = frozenset({"code", "docs"})

# ColGREP CLI integration (default ON; set REPL_COLGREP=0 to use NextPLAID).
# Subprocess-per-query: every call pays full ONNX runtime + ColBERT model
# load (~770 ms p50, ~2.3 s worst-case). Acceptable for human-paced REPL;
# if soak telemetry shows the hit is real for high-frequency tool loops,
# see handoffs/active/repl-turn-efficiency.md "S7: Cold-start daemon options"
# for the two evaluated paths (homegrown sidecar vs upstream next-plaid SDK CLI).
COLGREP_BIN = "/mnt/raid0/llm/UTILS/bin/colgrep"
COLGREP_DEFAULT_PATH = "/mnt/raid0/llm/epyc-orchestrator/src"
COLGREP_TIMEOUT_S = 10


def _colgrep_enabled() -> bool:
    """ColGREP is the default code_search engine. Explicit opt-out via REPL_COLGREP=0."""
    return os.environ.get("REPL_COLGREP", "1").lower() not in ("0", "false", "off")


class _CodeSearchMixin:
    """Mixin providing multi-vector code/doc search tools.

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig
        artifacts: dict
        _exploration_calls: int
        _exploration_log: ExplorationLog
        _research_context: ResearchContext
        _last_research_node: str | None
    """

    _code_client: Any = None  # Lazy-loaded, :8088
    _docs_client: Any = None  # Lazy-loaded, :8089

    def _init_nextplaid_client(self, url: str) -> Any:
        """Create and health-check a NextPLAID client. Returns None if unavailable."""
        try:
            from next_plaid_client import NextPlaidClient

            client = NextPlaidClient(url)
            health = client.health()
            if health.status != "healthy":
                logger.warning("NextPLAID at %s unhealthy: %s", url, health.status)
                return None
            return client
        except ImportError:
            logger.debug("next-plaid-client not installed")
            return None
        except Exception as e:
            logger.debug("NextPLAID at %s unavailable: %s", url, e)
            return None

    def _get_nextplaid_client(self, index: str = "code") -> Any:
        """Return the appropriate NextPLAID client for the given index.

        Routing:
            code → :8088 (_code_client)
            docs → :8089 (_docs_client), falls back to :8088 if unavailable
        """
        if index == "code":
            if self._code_client is None:
                self._code_client = self._init_nextplaid_client(CODE_SEARCH_URL)
            return self._code_client

        # docs index → try dedicated docs container first
        if self._docs_client is None:
            self._docs_client = self._init_nextplaid_client(DOC_SEARCH_URL)
        if self._docs_client is not None:
            return self._docs_client

        # Fallback: docs container down → use code container (lower quality but functional)
        logger.info("Docs container (:8089) unavailable, falling back to code container (:8088)")
        if self._code_client is None:
            self._code_client = self._init_nextplaid_client(CODE_SEARCH_URL)
        return self._code_client

    def _code_search(self, query: str, limit: int = 5) -> str:
        """Search project source code for relevant passages.

        Uses multi-vector (ColBERT) retrieval with token-level matching.
        Finds specific function names, class definitions, and code patterns —
        not just semantic similarity.

        Unlike recall() which searches episodic memories (past routing decisions),
        code_search() finds actual source code in the project.

        Args:
            query: Natural language or code pattern to search for.
                   e.g., "escalation policy implementation",
                         "def embed_task_ir", "FAISS index configuration"
            limit: Maximum results to return (default 5).

        Returns:
            JSON with matching code passages, file paths, and line ranges.
        """
        if _colgrep_enabled():
            return self._colgrep_search(query, limit=limit)
        return self._nextplaid_search(query, index="code", limit=limit)

    def _doc_search(self, query: str, limit: int = 5) -> str:
        """Search project documentation for relevant sections.

        Searches markdown docs, handoffs, model registry, and config files.
        For source code, use code_search() instead.

        Args:
            query: What to look for in documentation.
            limit: Maximum results (default 5).

        Returns:
            JSON with matching doc passages and metadata.
        """
        return self._nextplaid_search(query, index="docs", limit=limit)

    def _nextplaid_search(self, query: str, index: str, limit: int) -> str:
        """Internal: execute search against a NextPLAID index."""
        lock = getattr(self, "_state_lock", None)
        if lock:
            with lock:
                self._exploration_calls += 1
        else:
            self._exploration_calls += 1

        if index not in VALID_INDICES:
            output = json.dumps(
                {"results": [], "error": f"Invalid index '{index}'. Valid: {sorted(VALID_INDICES)}"}
            )
            return self._maybe_wrap_tool_output(output)

        client = self._get_nextplaid_client(index)
        if client is None:
            output = json.dumps(
                {"results": [], "error": "NextPLAID not available"}
            )
            return self._maybe_wrap_tool_output(output)

        try:
            from next_plaid_client.models import SearchParams

            params = SearchParams(top_k=min(limit, 20))
            result = client.search_with_encoding(index, queries=[query], params=params)

            results = []
            if result.results:
                qr = result.results[0]  # Single query → first QueryResult
                for doc_id, score, meta in zip(qr.document_ids, qr.scores, qr.metadata):
                    if len(results) >= limit:
                        break
                    entry = {
                        "file": meta.get("file", "unknown"),
                        "lines": f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
                        "score": round(float(score), 3),
                    }
                    # Include AST metadata when available (Phase 5)
                    unit_name = meta.get("unit_name")
                    if unit_name:
                        entry["unit"] = f"{meta.get('unit_type', '')}:{unit_name}"
                        sig = meta.get("signature", "")
                        if sig:
                            entry["signature"] = sig[:100]
                    results.append(entry)

            # Frecency boost for search results (feature-flagged)
            import os as _os

            if results and _os.environ.get("REPL_FRECENCY", "").lower() in ("1", "true", "on"):
                try:
                    from src.repl_environment.file_recency import FrecencyStore

                    _frecency = getattr(self, "_frecency_store", None)
                    if _frecency is None:
                        _frecency = FrecencyStore()
                        self._frecency_store = _frecency
                    for r in results:
                        boost = _frecency.get_score(r["file"])
                        r["score"] = round(r["score"] * (1 + 0.3 * boost), 3)
                    results.sort(key=lambda r: r["score"], reverse=True)
                except Exception:
                    logger.debug("Frecency boost failed", exc_info=True)

            response = {"results": results, "index": index, "query": query}

            self._exploration_log.add_event(
                "code_search" if index == "code" else "doc_search",
                {"query": query, "index": index},
                response,
            )

            # Track in research context
            tool_name = "code_search" if index == "code" else "doc_search"
            node_id = self._research_context.add(
                tool=tool_name,
                query=query[:100],
                content=json.dumps(results[:3]),
                parent_id=self._last_research_node,
            )
            self._last_research_node = node_id

            output = json.dumps(response, indent=2)
            return self._maybe_wrap_tool_output(output)

        except Exception as e:
            logger.warning("NextPLAID search failed: %s", e)
            output = json.dumps({"results": [], "error": str(e)})
            return self._maybe_wrap_tool_output(output)

    def _colgrep_search(self, query: str, limit: int) -> str:
        """Internal: execute code_search via the ColGREP CLI binary.

        Subprocess-per-query (no daemon mode upstream). Falls back to
        NextPLAID on missing binary, timeout, or non-zero exit so callers
        always get a valid response shape.
        """
        lock = getattr(self, "_state_lock", None)
        if lock:
            with lock:
                self._exploration_calls += 1
        else:
            self._exploration_calls += 1

        bin_path = os.environ.get("REPL_COLGREP_BIN", COLGREP_BIN)
        proj_path = os.environ.get("REPL_COLGREP_PATH", COLGREP_DEFAULT_PATH)
        if not shutil.which(bin_path) and not os.path.isfile(bin_path):
            logger.warning("ColGREP binary not found at %s, falling back to NextPLAID", bin_path)
            return self._nextplaid_search(query, index="code", limit=limit)

        env = {**os.environ, "NEXT_PLAID_FORCE_CPU": "1"}
        # alpha=0.95 weights semantic ColBERT over FTS5 keyword. Default 0.75
        # over-ranks __init__.py re-exports for symbol queries in this corpus
        # (validated 2026-04-29: alpha 0.95 recovers correct top-1 on
        # FinalSignal/ASTSecurityVisitor/create_repl_environment).
        alpha = os.environ.get("REPL_COLGREP_ALPHA", "0.95")
        cmd = [
            bin_path, "search", query,
            "-k", str(min(limit, 20)),
            "--alpha", alpha,
            "--json",
            proj_path,
        ]
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True,
                timeout=COLGREP_TIMEOUT_S, check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("ColGREP timed out after %ds, falling back to NextPLAID", COLGREP_TIMEOUT_S)
            return self._nextplaid_search(query, index="code", limit=limit)
        except OSError as e:
            logger.warning("ColGREP subprocess failed: %s, falling back to NextPLAID", e)
            return self._nextplaid_search(query, index="code", limit=limit)

        if proc.returncode != 0:
            logger.warning("ColGREP exit %d: %s", proc.returncode, proc.stderr[:500])
            return self._nextplaid_search(query, index="code", limit=limit)

        try:
            raw = json.loads(proc.stdout) if proc.stdout.strip() else []
        except json.JSONDecodeError as e:
            logger.warning("ColGREP JSON parse failed: %s", e)
            return self._nextplaid_search(query, index="code", limit=limit)

        results = []
        for item in raw[:limit]:
            unit = item.get("unit", {}) if isinstance(item, dict) else {}
            file_path = unit.get("file", "unknown")
            try:
                rel = os.path.relpath(file_path, proj_path)
            except ValueError:
                rel = file_path
            entry = {
                "file": rel,
                "lines": f"{unit.get('line', '?')}-{unit.get('end_line', '?')}",
                "score": round(float(item.get("score", 0.0)), 3),
            }
            unit_name = unit.get("name")
            unit_type = unit.get("unit_type")
            if unit_name and unit_type and unit_type != "rawcode":
                entry["unit"] = f"{unit_type}:{unit_name}"
                sig = unit.get("signature") or ""
                if sig:
                    entry["signature"] = sig[:100]
            results.append(entry)

        # Frecency boost (same flag as NextPLAID path)
        if results and os.environ.get("REPL_FRECENCY", "").lower() in ("1", "true", "on"):
            try:
                from src.repl_environment.file_recency import FrecencyStore

                _frecency = getattr(self, "_frecency_store", None)
                if _frecency is None:
                    _frecency = FrecencyStore()
                    self._frecency_store = _frecency
                for r in results:
                    boost = _frecency.get_score(r["file"])
                    r["score"] = round(r["score"] * (1 + 0.3 * boost), 3)
                results.sort(key=lambda r: r["score"], reverse=True)
            except Exception:
                logger.debug("Frecency boost failed", exc_info=True)

        response = {"results": results, "index": "code", "query": query, "engine": "colgrep"}
        self._exploration_log.add_event(
            "code_search", {"query": query, "index": "code", "engine": "colgrep"}, response,
        )
        node_id = self._research_context.add(
            tool="code_search",
            query=query[:100],
            content=json.dumps(results[:3]),
            parent_id=self._last_research_node,
        )
        self._last_research_node = node_id

        output = json.dumps(response, indent=2)
        return self._maybe_wrap_tool_output(output)
