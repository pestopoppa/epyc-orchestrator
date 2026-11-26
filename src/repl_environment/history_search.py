"""Bounded, session-owned search over the verbatim history artifact."""
from __future__ import annotations

import copy
import json
import time

import regex

LIMIT = 6000
TIMEOUT = 10.0


def search(env, pattern, context_lines=0, offset=0, count=False, page_size=10):
    if not isinstance(pattern, str) or len(pattern) > 1024:
        return '{"error":"pattern must be a string of at most 1024 characters"}'
    if type(context_lines) is not int or not 0 <= context_lines <= 3 or type(offset) is not int or offset < 0 or type(count) is not bool or type(page_size) is not int or not 1 <= page_size <= 30:
        return '{"error":"invalid context_lines, offset or count"}'
    started = time.monotonic()
    try:
        compiled = regex.compile(pattern, regex.IGNORECASE)
    except regex.error as exc:
        return json.dumps({"error": f"invalid regex: {str(exc)[:300]}"})
    try:
        text = env._read_registered_artifact(env._history_search_path, -1, 0)
        lines = text.splitlines()
        hits, counts, source = [], {}, 'header'
        for i, line in enumerate(lines):
            if line in ('Code:', 'Output:', 'Error:'):
                source = line[:-1].lower()
            remaining = TIMEOUT - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError
            if compiled.search(line, timeout=remaining):
                counts[source] = counts.get(source, 0) + 1
                if not count:
                    hits.append((i, source))
        if count:
            return json.dumps({'count': sum(counts.values()), 'per_source': counts}, ensure_ascii=True)
        records = []
        for index, (line_no, source) in enumerate(hits[offset:offset + page_size], start=offset):
            a, b = max(0, line_no-context_lines), min(len(lines), line_no+context_lines+1)
            record = {'line': line_no + 1, 'source': source,
                      'context': [s[:300] for s in lines[a:b]], 'context_start': a + 1}
            candidate = records + [record]
            next_offset = index + 1 if index + 1 < len(hits) else None
            result = {'matches': candidate, 'next_offset': next_offset}
            if len(json.dumps(result, ensure_ascii=False)) > LIMIT:
                if records:
                    break
                # JSON escaping can expand control characters sixfold; shorten an
                # oversized first match so pagination always makes progress.
                record['context'] = [s[:80] for s in record['context']]
                candidate = [record]
                result['matches'] = candidate

            records = candidate
            if len(records) == page_size:
                break
        next_offset = offset + len(records) if offset + len(records) < len(hits) else None
        return json.dumps({'matches': records, 'next_offset': next_offset}, ensure_ascii=False)
    except TimeoutError:
        return '{"error":"search timed out after 10 seconds"}'
    except (OSError, UnicodeError, ValueError, PermissionError) as exc:
        return json.dumps({'error': f'history unavailable: {type(exc).__name__}'})


def install(env, state):
    """Install once, only for the owning REPL after a real boundary."""
    from src.repl_environment.trace_navigation_tools import enabled
    if not enabled() or not state.history_log_path or getattr(env, 'task_id', None) != state.task_id:
        return
    if not env._register_session_artifact(state.history_log_path, state.task_id):
        raise PermissionError('verbatim history registration failed')
    env._history_search_path = state.history_log_path
    if 'grep_history' in env._globals:
        return
    if env.tool_registry is not None:
        from src.registry.tool_registry import Tool, ToolCategory, ToolRegistry
        if isinstance(env.tool_registry, ToolRegistry):
            local = copy.copy(env.tool_registry)
            local._tools = env.tool_registry._tools.copy()
            local.register_tool(Tool(name='grep_history', description='Search this session verbatim log, oldest first; offset is a match index.',
                category=ToolCategory.DATA, parameters={'pattern': {'type': 'string', 'required': True},
                'context_lines': {'type': 'integer'}, 'offset': {'type': 'integer'}, 'count': {'type': 'boolean'}, 'page_size': {'type': 'integer'}},
                handler=env._grep_history, side_effects=[], allowed_callers=['direct', 'chain']), update=True)
            env.tool_registry = local
    globals_dict = {'grep_history': env._grep_history}
    env._count_tool_calls(globals_dict)
    env._globals.update(globals_dict)
    env._builtin_global_keys = env._builtin_global_keys | {'grep_history'}
