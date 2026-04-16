"""Combined REPL operations to reduce turn count.

Mixin providing batched multi-tool patterns based on autopilot log analysis:
- batch_web_search: N sequential web searches in 1 turn
- search_and_verify: web_search + wikipedia verification in 1 turn
- peek_grep: file read + regex search in 1 turn

Feature-gated by REPL_COMBINED_OPS env var (default off).
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_FEATURE_FLAG_VALUES = {"1", "true", "on"}
_MAX_QUERIES = 5


def _feature_enabled() -> bool:
    """Check whether combined ops are enabled via env var."""
    return os.environ.get("REPL_COMBINED_OPS", "").strip().lower() in _FEATURE_FLAG_VALUES


class _CombinedOpsMixin:
    """Mixin providing combined REPL operations to reduce turn count.

    Feature flag: REPL_COMBINED_OPS env var (1/true/on to enable, default off).

    These operations batch common multi-tool patterns into single calls,
    based on autopilot log analysis showing repeated sequential web_search
    calls as the dominant REPL pattern (94.8% of tool calls).

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        tool_registry: Any | None — ToolRegistry for TOOL() invocations
        role: str — current agent role
        _validate_file_path: Callable[[str], tuple[bool, str | None]] — path validation
    """

    def _web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web for a query and return results.

        Args:
            query: Search query string.
            max_results: Maximum number of results (default 5).

        Returns:
            Search results as formatted text.
        """
        if self.tool_registry is None:
            return "[ERROR: No tool registry configured — cannot invoke web_search]"
        try:
            result = self.tool_registry.invoke(
                "web_search",
                self.role,
                caller_type="direct",
                query=query,
                max_results=max_results,
            )
            import json as _json
            if isinstance(result, (dict, list)):
                return _json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as exc:
            return f"[ERROR: web_search failed: {exc}]"

    def _batch_web_search(self, queries: list[str], max_results: int = 3) -> str:
        """Run multiple web searches and return consolidated results.

        Addresses the dominant pattern: repeated web_search -> web_search calls
        (5727 bigrams in autopilot logs). Reduces N sequential turns to 1 turn.

        Args:
            queries: List of search queries (max 5).
            max_results: Results per query (default 3).

        Returns:
            Consolidated results in JSON or TOON format.
        """
        if not _feature_enabled():
            return "[ERROR: Combined ops disabled. Set REPL_COMBINED_OPS=1 to enable.]"

        self._exploration_calls += 1

        if not queries:
            return "[ERROR: No queries provided]"

        # Cap queries to prevent abuse
        if len(queries) > _MAX_QUERIES:
            queries = queries[:_MAX_QUERIES]
            capped = True
        else:
            capped = False

        if self.tool_registry is None:
            return "[ERROR: No tool registry configured — cannot invoke web_search]"

        results = {}
        for query in queries:
            try:
                result = self.tool_registry.invoke(
                    "web_search",
                    self.role,
                    caller_type="direct",
                    query=query,
                    max_results=max_results,
                )
                results[query] = result
            except Exception as e:
                results[query] = f"[ERROR: {type(e).__name__}: {e}]"

        self._exploration_log.add_event(
            "batch_web_search",
            {"queries": queries, "max_results": max_results, "capped": capped},
            results,
        )

        # Format output
        if getattr(self.config, "use_toon_encoding", False):
            lines = [f"=== batch_web_search ({len(queries)} queries) ==="]
            if capped:
                lines.append(f"[WARNING: Capped to {_MAX_QUERIES} queries]")
            for query, result in results.items():
                lines.append(f'--- "{query}" ---')
                if isinstance(result, (list, dict)):
                    lines.append(json.dumps(result, indent=2, default=str))
                else:
                    lines.append(str(result))
            return "\n".join(lines)
        else:
            output = {
                "operation": "batch_web_search",
                "query_count": len(queries),
                "capped": capped,
                "results": {q: r for q, r in results.items()},
            }
            return json.dumps(output, indent=2, default=str)

    def _search_and_verify(self, query: str, max_results: int = 3) -> str:
        """Web search followed by Wikipedia verification in one turn.

        Addresses: web_search -> search_wikipedia pattern (171 bigrams in logs).

        Args:
            query: Search query.
            max_results: Web results to return (default 3).

        Returns:
            Combined web + wikipedia results.
        """
        if not _feature_enabled():
            return "[ERROR: Combined ops disabled. Set REPL_COMBINED_OPS=1 to enable.]"

        self._exploration_calls += 1

        if self.tool_registry is None:
            return "[ERROR: No tool registry configured — cannot invoke search tools]"

        web_result = None
        wiki_result = None

        # Step 1: web search
        try:
            web_result = self.tool_registry.invoke(
                "web_search",
                self.role,
                caller_type="direct",
                query=query,
                max_results=max_results,
            )
        except Exception as e:
            web_result = f"[ERROR: {type(e).__name__}: {e}]"

        # Step 2: wikipedia verification
        try:
            wiki_result = self.tool_registry.invoke(
                "search_wikipedia",
                self.role,
                caller_type="direct",
                query=query,
            )
        except Exception as e:
            wiki_result = f"[ERROR: {type(e).__name__}: {e}]"

        self._exploration_log.add_event(
            "search_and_verify",
            {"query": query, "max_results": max_results},
            {"web": web_result, "wiki": wiki_result},
        )

        # Format output
        if getattr(self.config, "use_toon_encoding", False):
            lines = [
                f'=== search_and_verify: "{query}" ===',
                "## Web Results",
            ]
            if isinstance(web_result, (list, dict)):
                lines.append(json.dumps(web_result, indent=2, default=str))
            else:
                lines.append(str(web_result))
            lines.append("## Wikipedia")
            if isinstance(wiki_result, (list, dict)):
                lines.append(json.dumps(wiki_result, indent=2, default=str))
            else:
                lines.append(str(wiki_result))
            return "\n".join(lines)
        else:
            output = {
                "operation": "search_and_verify",
                "query": query,
                "web_results": web_result,
                "wikipedia": wiki_result,
            }
            return json.dumps(output, indent=2, default=str)

    def _peek_grep(self, path: str, pattern: str, context_lines: int = 3) -> str:
        """Read a file and grep it in a single call.

        Preemptive: file exploration tools aren't used yet in autopilot,
        but this is the most natural combined file operation for when they are.

        Args:
            path: File path to read.
            pattern: Regex pattern to search.
            context_lines: Lines of context around matches (default 3).

        Returns:
            Matched lines with surrounding context.
        """
        if not _feature_enabled():
            return "[ERROR: Combined ops disabled. Set REPL_COMBINED_OPS=1 to enable.]"

        self._exploration_calls += 1

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        # Read file
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except FileNotFoundError:
            return f"[ERROR: File not found: {path}]"
        except IsADirectoryError:
            return f"[ERROR: Path is a directory: {path}]"
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

        # Apply regex
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return f"[ERROR: Invalid regex pattern: {e}]"

        lines = content.splitlines()
        matches = []

        for i, line in enumerate(lines):
            if compiled.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context_block = []
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    context_block.append(f"{marker} {j + 1:4d} | {lines[j]}")
                matches.append("\n".join(context_block))

        self._exploration_log.add_event(
            "peek_grep",
            {"path": path, "pattern": pattern, "context_lines": context_lines},
            {"match_count": len(matches)},
        )

        if not matches:
            return f'[No matches for pattern "{pattern}" in {path}]'

        # Format output
        if getattr(self.config, "use_toon_encoding", False):
            header = f'=== peek_grep: "{pattern}" in {path} ({len(matches)} matches) ==='
            return header + "\n" + "\n---\n".join(matches)
        else:
            output = {
                "operation": "peek_grep",
                "path": path,
                "pattern": pattern,
                "match_count": len(matches),
                "matches": matches,
            }
            return json.dumps(output, indent=2, default=str)
